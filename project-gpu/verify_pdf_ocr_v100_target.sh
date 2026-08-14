#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 benchmark.json|summary.json [max-peak-gpu-mb] [max-process-rss-mb]" >&2
  exit 2
fi

RESULT_JSON="$1"
MAX_PEAK_GPU_MB="${2:-0}"
MAX_PROCESS_RSS_MB="${3:-0}"

if [[ ! -f "${RESULT_JSON}" ]]; then
  echo "[ERROR] Result JSON not found: ${RESULT_JSON}" >&2
  exit 2
fi

RESULT_JSON="${RESULT_JSON}" MAX_PEAK_GPU_MB="${MAX_PEAK_GPU_MB}" MAX_PROCESS_RSS_MB="${MAX_PROCESS_RSS_MB}" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


path = Path(os.environ["RESULT_JSON"]).resolve()
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"[ERROR] Could not read JSON: {exc}", file=sys.stderr)
    raise SystemExit(2)

try:
    max_peak_gpu_mb = int(float(os.environ.get("MAX_PEAK_GPU_MB", "0") or 0))
except Exception:
    max_peak_gpu_mb = 0
try:
    max_process_rss_mb = int(float(os.environ.get("MAX_PROCESS_RSS_MB", "0") or 0))
except Exception:
    max_process_rss_mb = 0


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _case_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(data.get("target_met_cases"), list) or isinstance(data.get("cases"), list):
        rows = data.get("target_met_cases") or []
        if not rows:
            rows = [row for row in data.get("cases", []) if isinstance(row, dict) and row.get("target_met")]
        return [row for row in rows if isinstance(row, dict)]

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        row = dict(metrics)
        row["name"] = path.name
        row["status"] = data.get("status", "")
        row["target_met"] = bool(metrics.get("target_met"))
        row["elapsed_seconds"] = _num(metrics, "elapsed_seconds")
        row["ocr_pages"] = int(_num(metrics, "ocr_pages"))
        row["target_pages"] = int(_num(metrics, "target_pages", 200))
        row["target_seconds"] = _num(metrics, "target_seconds", 300.0)
        row["peak_gpu_memory_used_mb"] = int(_num(metrics, "peak_gpu_memory_used_mb"))
        row["peak_process_rss_mb"] = int(_num(metrics, "peak_process_rss_mb"))
        row["peak_process_tree_rss_mb"] = int(_num(metrics, "peak_process_tree_rss_mb", row["peak_process_rss_mb"]))
        row["pages_per_minute"] = _num(metrics, "pages_per_minute")
        return [row]
    return []


rows = _case_rows(payload if isinstance(payload, dict) else {})
valid: List[Dict[str, Any]] = []
failures: List[str] = []

for row in rows:
    name = str(row.get("name", path.name) or path.name)
    target_pages = int(_num(row, "target_pages", 200))
    target_seconds = _num(row, "target_seconds", 300.0)
    ocr_pages = int(_num(row, "ocr_pages"))
    elapsed_seconds = _num(row, "elapsed_seconds")
    peak_gpu = int(_num(row, "peak_gpu_memory_used_mb"))
    peak_rss = int(_num(row, "peak_process_rss_mb"))
    peak_tree_rss = int(_num(row, "peak_process_tree_rss_mb", peak_rss))
    effective_peak_rss = peak_tree_rss if peak_tree_rss > 0 else peak_rss
    target_met = bool(row.get("target_met"))
    status = str(row.get("status", "ok") or "ok")

    case_errors = []
    if status not in {"ok", ""}:
        case_errors.append(f"status={status}")
    if not target_met:
        case_errors.append("target_met=false")
    if ocr_pages < target_pages:
        case_errors.append(f"ocr_pages {ocr_pages} < target_pages {target_pages}")
    if elapsed_seconds <= 0:
        case_errors.append("elapsed_seconds missing")
    if target_seconds <= 0 or elapsed_seconds > target_seconds:
        case_errors.append(f"elapsed_seconds {elapsed_seconds} > target_seconds {target_seconds}")
    if max_peak_gpu_mb > 0 and peak_gpu <= 0:
        case_errors.append("peak_gpu_memory_used_mb missing")
    if max_peak_gpu_mb > 0 and peak_gpu > max_peak_gpu_mb:
        case_errors.append(f"peak_gpu_memory_used_mb {peak_gpu} > limit {max_peak_gpu_mb}")
    if max_process_rss_mb > 0 and effective_peak_rss <= 0:
        case_errors.append("peak_process_rss_mb missing")
    if max_process_rss_mb > 0 and effective_peak_rss > max_process_rss_mb:
        case_errors.append(f"peak_process_rss_mb {effective_peak_rss} > limit {max_process_rss_mb}")

    if case_errors:
        failures.append(f"{name}: " + "; ".join(case_errors))
    else:
        valid.append(row)

summary_target_failed = isinstance(payload, dict) and "target_achieved" in payload and not payload.get("target_achieved")
if summary_target_failed:
    failures.append("summary target_achieved=false")

if summary_target_failed or not valid:
    print(json.dumps({"status": "fail", "result_json": str(path), "failures": failures or ["no target-met case found"]}, ensure_ascii=False, indent=2))
    raise SystemExit(3)

winner = sorted(
    valid,
    key=lambda row: (
        int(_num(row, "peak_gpu_memory_used_mb", 10**9)),
        int(_num(row, "peak_process_tree_rss_mb", _num(row, "peak_process_rss_mb", 10**9))),
        _num(row, "elapsed_seconds", 10**9),
        -_num(row, "pages_per_minute", 0),
    ),
)[0]
print(json.dumps(
    {
        "status": "ok",
        "result_json": str(path),
        "selected_case": winner.get("name", path.name),
        "ocr_pages": int(_num(winner, "ocr_pages")),
        "elapsed_seconds": _num(winner, "elapsed_seconds"),
        "pages_per_minute": _num(winner, "pages_per_minute"),
        "peak_gpu_memory_used_mb": int(_num(winner, "peak_gpu_memory_used_mb")),
        "peak_process_rss_mb": int(_num(winner, "peak_process_rss_mb")),
        "peak_process_tree_rss_mb": int(_num(winner, "peak_process_tree_rss_mb", _num(winner, "peak_process_rss_mb"))),
        "max_peak_gpu_mb": max_peak_gpu_mb,
        "max_process_rss_mb": max_process_rss_mb,
    },
    ensure_ascii=False,
    indent=2,
))
PY
