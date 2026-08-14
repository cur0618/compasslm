#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/200-page.pdf [max-peak-gpu-mb] [max-process-rss-mb] [--write]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

PDF_PATH="$1"
shift
MAX_PEAK_GPU_MB="0"
MAX_PROCESS_RSS_MB="0"
WRITE_MODE=""

if [[ ! -f "${PDF_PATH}" ]]; then
  echo "[ERROR] PDF not found: ${PDF_PATH}" >&2
  exit 2
fi

numeric_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--write" ]]; then
    WRITE_MODE="--write"
  elif [[ "${arg}" =~ ^[0-9]+$ ]]; then
    numeric_args+=("${arg}")
  else
    echo "[ERROR] Unknown argument: ${arg}" >&2
    exit 2
  fi
done
if (( ${#numeric_args[@]} > 2 )); then
  echo "[ERROR] Too many numeric limits. Expected [max-peak-gpu-mb] [max-process-rss-mb]." >&2
  exit 2
fi
if (( ${#numeric_args[@]} >= 1 )); then
  MAX_PEAK_GPU_MB="${numeric_args[0]}"
fi
if (( ${#numeric_args[@]} >= 2 )); then
  MAX_PROCESS_RSS_MB="${numeric_args[1]}"
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${COMPASSLM_HOME}/logs/ocr-benchmarks/${RUN_ID}_v100_acceptance"
MATRIX_DIR="${OUTPUT_DIR}/matrix"
SUMMARY_JSON="${MATRIX_DIR}/summary.json"
ACCEPTANCE_REPORT="${OUTPUT_DIR}/acceptance_report.json"

mkdir -p "${OUTPUT_DIR}"

echo "[ACCEPTANCE] PDF=${PDF_PATH}"
echo "[ACCEPTANCE] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ACCEPTANCE] MAX_PEAK_GPU_MB=${MAX_PEAK_GPU_MB}"
echo "[ACCEPTANCE] MAX_PROCESS_RSS_MB=${MAX_PROCESS_RSS_MB}"
echo "[ACCEPTANCE] WRITE=${WRITE_MODE:-0}"

echo "[ACCEPTANCE] 1/4 preflight"
bash "${SCRIPT_DIR}/preflight_pdf_ocr_v100.sh"

echo "[ACCEPTANCE] 2/4 matrix"
bash "${SCRIPT_DIR}/tune_pdf_ocr_v100_matrix.sh" "${PDF_PATH}" "${MATRIX_DIR}"

echo "[ACCEPTANCE] 3/4 verify"
bash "${SCRIPT_DIR}/verify_pdf_ocr_v100_target.sh" "${SUMMARY_JSON}" "${MAX_PEAK_GPU_MB}" "${MAX_PROCESS_RSS_MB}"

echo "[ACCEPTANCE] 4/4 apply dry-run"
bash "${SCRIPT_DIR}/apply_pdf_ocr_tuned_profile.sh" "${SUMMARY_JSON}"

if [[ "${WRITE_MODE}" == "--write" ]]; then
  echo "[ACCEPTANCE] 4/4 apply write"
  bash "${SCRIPT_DIR}/apply_pdf_ocr_tuned_profile.sh" "${SUMMARY_JSON}" "${SCRIPT_DIR}/runtime.env" --write
  echo "[ACCEPTANCE] runtime.env updated. Restart backend with: ${SCRIPT_DIR}/run_backend_api.sh"
fi

ACCEPTANCE_REPORT="${ACCEPTANCE_REPORT}" SUMMARY_JSON="${SUMMARY_JSON}" PDF_PATH="${PDF_PATH}" MAX_PEAK_GPU_MB="${MAX_PEAK_GPU_MB}" MAX_PROCESS_RSS_MB="${MAX_PROCESS_RSS_MB}" WRITE_MODE="${WRITE_MODE}" python3 - <<'PY'
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _positive(row: Dict[str, Any], key: str, default: float = 10**9) -> float:
    value = _num(row, key, default)
    return value if value > 0 else default


def _ram(row: Dict[str, Any]) -> float:
    tree = _positive(row, "peak_process_tree_rss_mb")
    if tree < 10**9:
        return tree
    return _positive(row, "peak_process_rss_mb")


summary_path = Path(os.environ["SUMMARY_JSON"]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
cases = [row for row in summary.get("target_met_cases", []) if isinstance(row, dict)]
max_peak_gpu_mb = int(float(os.environ.get("MAX_PEAK_GPU_MB", "0") or 0))
max_process_rss_mb = int(float(os.environ.get("MAX_PROCESS_RSS_MB", "0") or 0))
selected = None
if cases:
    selected = sorted(
        cases,
        key=lambda row: (
            _positive(row, "peak_gpu_memory_used_mb"),
            _ram(row),
            _num(row, "elapsed_seconds", 10**9),
            -_num(row, "pages_per_minute", 0),
        ),
    )[0]
selected_peak_gpu = _num(selected or {}, "peak_gpu_memory_used_mb", 0)
selected_peak_rss = _num(
    selected or {},
    "peak_process_tree_rss_mb",
    _num(selected or {}, "peak_process_rss_mb", 0),
)

report = {
    "status": "pass",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pdf_path": str(Path(os.environ["PDF_PATH"]).resolve()),
    "summary_json": str(summary_path),
    "target_pages": int(_num(selected or {}, "target_pages", 200)),
    "target_seconds": _num(selected or {}, "target_seconds", 300),
    "max_peak_gpu_mb": max_peak_gpu_mb,
    "max_process_rss_mb": max_process_rss_mb,
    "runtime_env_written": os.environ.get("WRITE_MODE") == "--write",
    "target_achieved": bool(summary.get("target_achieved")),
    "selected_case": selected,
    "checklist": {
        "target_achieved_true": bool(summary.get("target_achieved")),
        "target_met_case_exists": bool(cases),
        "ocr_pages_ge_target": bool(selected and int(_num(selected, "ocr_pages")) >= int(_num(selected, "target_pages", 200))),
        "elapsed_seconds_le_target": bool(selected and _num(selected, "elapsed_seconds", 10**9) <= _num(selected, "target_seconds", 300)),
        "peak_gpu_metric_present": selected_peak_gpu > 0,
        "process_rss_metric_present": selected_peak_rss > 0,
        "peak_gpu_limit_checked": max_peak_gpu_mb > 0,
        "process_rss_limit_checked": max_process_rss_mb > 0,
        "peak_gpu_le_limit": bool(max_peak_gpu_mb <= 0 or (selected_peak_gpu > 0 and selected_peak_gpu <= max_peak_gpu_mb)),
        "process_rss_le_limit": bool(max_process_rss_mb <= 0 or (selected_peak_rss > 0 and selected_peak_rss <= max_process_rss_mb)),
    },
}
Path(os.environ["ACCEPTANCE_REPORT"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[ACCEPTANCE] report={os.environ['ACCEPTANCE_REPORT']}")
PY

echo "[ACCEPTANCE] PASS summary=${SUMMARY_JSON} report=${ACCEPTANCE_REPORT}"
