#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/200-page.pdf [output-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

PDF_PATH="$1"
if [[ ! -f "${PDF_PATH}" ]]; then
  echo "[ERROR] PDF not found: ${PDF_PATH}" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${2:-${COMPASSLM_HOME}/logs/ocr-benchmarks/${RUN_ID}_v100_matrix}"
mkdir -p "${OUTPUT_DIR}"

SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
SUMMARY_JSONL="${OUTPUT_DIR}/summary.jsonl"
: > "${SUMMARY_JSONL}"

run_case() {
  local name="$1"
  local max_pixels="$2"
  local max_tokens="$3"
  local use_chart="$4"
  local use_image_block="$5"
  local use_queues="$6"
  local out_json="${OUTPUT_DIR}/${name}.json"
  local exit_code=0

  echo "[MATRIX] ${name}: max_pixels=${max_pixels} max_tokens=${max_tokens} chart=${use_chart} image_block=${use_image_block} queues=${use_queues}"
  set +e
  PDF_OCR_OPTIMIZATION_PROFILE=v100_32gb_fast \
  PDF_OCR_MAX_PIXELS="${max_pixels}" \
  PDF_OCR_MAX_NEW_TOKENS="${max_tokens}" \
  PDF_OCR_USE_CHART_RECOGNITION="${use_chart}" \
  PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK="${use_image_block}" \
  PDF_OCR_USE_INTERNAL_QUEUES="${use_queues}" \
  PDF_OCR_GPU_PROCESS_ISOLATION=1 \
    bash "${SCRIPT_DIR}/benchmark_pdf_ocr_v100.sh" "${PDF_PATH}" "${out_json}"
  exit_code=$?
  set -e

  CASE_NAME="${name}" CASE_EXIT_CODE="${exit_code}" CASE_JSON="${out_json}" python3 - <<'PY' >> "${SUMMARY_JSONL}"
import json
import os
from pathlib import Path

path = Path(os.environ["CASE_JSON"])
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = {"status": "error", "error": f"json_read_fail: {exc}", "metrics": {}}
metrics = payload.get("metrics") if isinstance(payload, dict) else {}
if not isinstance(metrics, dict):
    metrics = {}
env = payload.get("env") if isinstance(payload, dict) else {}
if not isinstance(env, dict):
    env = {}
print(json.dumps({
    "name": os.environ["CASE_NAME"],
    "exit_code": int(os.environ["CASE_EXIT_CODE"]),
    "status": payload.get("status", "missing") if isinstance(payload, dict) else "missing",
    "error": payload.get("error", "") if isinstance(payload, dict) else "",
    "target_met": bool(metrics.get("target_met", False)),
    "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0) or 0.0),
    "ocr_pages": int(metrics.get("ocr_pages", 0) or 0),
    "pages_per_minute": float(metrics.get("pages_per_minute", 0.0) or 0.0),
    "peak_gpu_memory_used_mb": int(metrics.get("peak_gpu_memory_used_mb", 0) or 0),
    "peak_process_rss_mb": int(metrics.get("peak_process_rss_mb", 0) or 0),
    "peak_process_tree_rss_mb": int(metrics.get("peak_process_tree_rss_mb", metrics.get("peak_process_rss_mb", 0)) or 0),
    "ocr_gpu_fallback_used": bool(metrics.get("ocr_gpu_fallback_used", False)),
    "ocr_gpu_failure_reason": str(metrics.get("ocr_gpu_failure_reason", "") or ""),
    "env": env,
    "path": str(path),
}, ensure_ascii=False))
PY
}

# Conservative-to-aggressive order. The first row is the runtime default.
run_case "default_fast" "589824" "512" "0" "0" "1"
run_case "lower_pixels" "393216" "512" "0" "0" "1"
run_case "lower_tokens" "589824" "384" "0" "0" "1"
run_case "low_pixels_low_tokens" "393216" "384" "0" "0" "1"
run_case "minimal_vlm_budget" "262144" "320" "0" "0" "1"

SUMMARY_JSONL="${SUMMARY_JSONL}" SUMMARY_JSON="${SUMMARY_JSON}" python3 - <<'PY'
import json
import os
from pathlib import Path

rows = []
for line in Path(os.environ["SUMMARY_JSONL"]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rows.append(json.loads(line))

successful = [row for row in rows if row.get("status") == "ok" and int(row.get("ocr_pages", 0) or 0) > 0]
target_met = [row for row in successful if row.get("target_met")]

def positive_metric(row, key, default=10**9):
    try:
        value = float(row.get(key, default) or default)
    except Exception:
        return default
    return value if value > 0 else default

def ram_metric(row):
    value = positive_metric(row, "peak_process_tree_rss_mb")
    if value < 10**9:
        return value
    return positive_metric(row, "peak_process_rss_mb")

fastest = None
if successful:
    fastest = sorted(successful, key=lambda row: (float(row.get("elapsed_seconds", 10**9) or 10**9), positive_metric(row, "peak_gpu_memory_used_mb"), ram_metric(row)))[0]
lowest_vram = None
if successful:
    lowest_vram = sorted(successful, key=lambda row: (positive_metric(row, "peak_gpu_memory_used_mb"), ram_metric(row), float(row.get("elapsed_seconds", 10**9) or 10**9)))[0]
lowest_ram = None
if successful:
    lowest_ram = sorted(successful, key=lambda row: (ram_metric(row), positive_metric(row, "peak_gpu_memory_used_mb"), float(row.get("elapsed_seconds", 10**9) or 10**9)))[0]

summary = {
    "target_achieved": bool(target_met),
    "target_met_cases": target_met,
    "fastest_case": fastest,
    "lowest_vram_case": lowest_vram,
    "lowest_ram_case": lowest_ram,
    "cases": rows,
}
Path(os.environ["SUMMARY_JSON"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "target_achieved": summary["target_achieved"],
    "fastest_case": fastest,
    "lowest_vram_case": lowest_vram,
    "lowest_ram_case": lowest_ram,
    "summary_json": os.environ["SUMMARY_JSON"],
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if target_met else 3)
PY
