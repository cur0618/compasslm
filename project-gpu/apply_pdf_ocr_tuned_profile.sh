#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 summary.json [runtime.env] [--write]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

SUMMARY_JSON="$1"
TARGET_ENV="${2:-${COMPASSLM_HOME}/project-gpu/runtime.env}"
WRITE_MODE="${3:-}"

if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "[ERROR] summary.json not found: ${SUMMARY_JSON}" >&2
  exit 2
fi
if [[ ! -f "${TARGET_ENV}" ]]; then
  echo "[ERROR] runtime env not found: ${TARGET_ENV}" >&2
  exit 2
fi

SUMMARY_JSON="${SUMMARY_JSON}" TARGET_ENV="${TARGET_ENV}" WRITE_MODE="${WRITE_MODE}" python3 - <<'PY'
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

summary_path = Path(os.environ["SUMMARY_JSON"]).resolve()
target_env = Path(os.environ["TARGET_ENV"]).resolve()
write_mode = os.environ.get("WRITE_MODE", "")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
cases = summary.get("target_met_cases") or []
if not isinstance(cases, list) or not cases:
    print("[ERROR] No target_met_cases found. Run the matrix until at least one case reaches target_achieved=true.", file=sys.stderr)
    raise SystemExit(3)


def _num(row: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _positive_metric(row: Dict[str, Any], key: str, default: float = 10**9) -> float:
    value = _num(row, key, default)
    return value if value > 0 else default


def _ram_metric(row: Dict[str, Any]) -> float:
    value = _positive_metric(row, "peak_process_tree_rss_mb")
    if value < 10**9:
        return value
    return _positive_metric(row, "peak_process_rss_mb")


winner = sorted(
    [case for case in cases if isinstance(case, dict)],
    key=lambda row: (
        _positive_metric(row, "peak_gpu_memory_used_mb"),
        _ram_metric(row),
        _num(row, "elapsed_seconds", 10**9),
        -_num(row, "pages_per_minute", 0),
    ),
)[0]
env = winner.get("env") if isinstance(winner.get("env"), dict) else {}

updates = {
    "PDF_OCR_OPTIMIZATION_PROFILE": env.get("PDF_OCR_OPTIMIZATION_PROFILE", "v100_32gb_fast"),
    "PDF_OCR_USE_INTERNAL_QUEUES": env.get("PDF_OCR_USE_INTERNAL_QUEUES", "1"),
    "PDF_OCR_GPU_PROCESS_ISOLATION": env.get("PDF_OCR_GPU_PROCESS_ISOLATION", "1"),
    "PDF_OCR_MAX_PIXELS": env.get("PDF_OCR_MAX_PIXELS", ""),
    "PDF_OCR_MAX_NEW_TOKENS": env.get("PDF_OCR_MAX_NEW_TOKENS", ""),
    "PDF_OCR_VL_REC_MAX_CONCURRENCY": env.get("PDF_OCR_VL_REC_MAX_CONCURRENCY", "1"),
}
updates = {key: str(value).strip() for key, value in updates.items() if str(value).strip()}

print(json.dumps(
    {
        "selected_case": winner.get("name", ""),
        "elapsed_seconds": winner.get("elapsed_seconds", 0),
        "pages_per_minute": winner.get("pages_per_minute", 0),
        "peak_gpu_memory_used_mb": winner.get("peak_gpu_memory_used_mb", 0),
        "peak_process_rss_mb": winner.get("peak_process_rss_mb", 0),
        "peak_process_tree_rss_mb": winner.get("peak_process_tree_rss_mb", winner.get("peak_process_rss_mb", 0)),
        "updates": updates,
        "target_env": str(target_env),
        "write": write_mode == "--write",
    },
    ensure_ascii=False,
    indent=2,
))

if write_mode != "--write":
    print("[INFO] Dry run only. Re-run with --write to update runtime.env.", file=sys.stderr)
    raise SystemExit(0)

lines = target_env.read_text(encoding="utf-8").splitlines()
seen = set()
next_lines: List[str] = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        next_lines.append(line)
        continue
    key = stripped.split("=", 1)[0].strip()
    if key in updates:
        next_lines.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        next_lines.append(line)

missing = [key for key in updates if key not in seen]
if missing:
    next_lines.append("")
    next_lines.append("# Applied from V100 OCR benchmark summary")
    for key in missing:
        next_lines.append(f"{key}={updates[key]}")

backup_path = target_env.with_suffix(target_env.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
shutil.copy2(target_env, backup_path)
target_env.write_text("\n".join(next_lines) + "\n", encoding="utf-8")
print(f"[INFO] Updated {target_env}")
print(f"[INFO] Backup {backup_path}")
PY
