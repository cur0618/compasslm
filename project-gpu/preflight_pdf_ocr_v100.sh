#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"

compass_load_env_file "${MAIN_BACKEND_HOME}/.env.auto"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
compass_load_env_file "${MAIN_BACKEND_HOME}/.env"

export PDF_OCR_OPTIMIZATION_PROFILE="${PDF_OCR_OPTIMIZATION_PROFILE:-v100_32gb_fast}"
export PDF_OCR_DEVICE="${PDF_OCR_DEVICE:-gpu:0}"
export PDF_OCR_TARGET_PAGES="${PDF_OCR_TARGET_PAGES:-200}"
export PDF_OCR_TARGET_SECONDS="${PDF_OCR_TARGET_SECONDS:-300}"

VENV_PY="${MAIN_BACKEND_HOME}/compassvenv/bin/python"
GPU_INDEX="${PDF_OCR_DEVICE##*:}"
if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  GPU_INDEX="0"
fi

status=0
check_ok() {
  echo "[OK]   $*"
}
check_warn() {
  echo "[WARN] $*"
}
check_fail() {
  echo "[FAIL] $*"
  status=1
}

echo "=== V100 PDF OCR preflight ==="
echo "COMPASSLM_HOME=${COMPASSLM_HOME}"
echo "MAIN_BACKEND_HOME=${MAIN_BACKEND_HOME}"
echo "PDF_OCR_DEVICE=${PDF_OCR_DEVICE}"
echo "PDF_OCR_OPTIMIZATION_PROFILE=${PDF_OCR_OPTIMIZATION_PROFILE}"
echo "PDF_OCR_TARGET_PAGES=${PDF_OCR_TARGET_PAGES}"
echo "PDF_OCR_TARGET_SECONDS=${PDF_OCR_TARGET_SECONDS}"
echo

if [[ -x "${VENV_PY}" ]]; then
  check_ok "Backend Python: ${VENV_PY}"
else
  check_fail "Backend Python not executable: ${VENV_PY}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -n "${gpu_line}" ]]; then
    IFS=',' read -r gpu_name gpu_total gpu_free <<< "${gpu_line}"
    gpu_name="$(echo "${gpu_name}" | xargs)"
    gpu_total="$(echo "${gpu_total}" | xargs)"
    gpu_free="$(echo "${gpu_free}" | xargs)"
    echo "GPU name=${gpu_name} total_mb=${gpu_total} free_mb=${gpu_free}"
    if [[ "${gpu_name}" == *"V100"* ]]; then
      check_ok "GPU model is V100"
    else
      check_warn "GPU model is not V100: ${gpu_name}"
    fi
    if [[ "${gpu_total}" =~ ^[0-9]+$ ]] && (( gpu_total >= 30000 )); then
      check_ok "GPU memory is near 32GB"
    else
      check_fail "GPU total memory is below V100 32GB expectation: ${gpu_total} MB"
    fi
    if [[ "${gpu_free}" =~ ^[0-9]+$ ]] && (( gpu_free >= 12000 )); then
      check_ok "GPU free memory is enough for fast OCR trial"
    else
      check_warn "GPU free memory is low for benchmark: ${gpu_free} MB"
    fi
  else
    check_fail "nvidia-smi could not read GPU ${GPU_INDEX}"
  fi
else
  check_fail "nvidia-smi not found"
fi

for key in \
  PDF_OCR_MODEL_NAME \
  PDF_OCR_VL_MODEL_DIR \
  PDF_OCR_LAYOUT_MODEL_DIR \
  PDF_OCR_DOC_ORIENTATION_MODEL_DIR \
  PDF_OCR_DOC_UNWARP_MODEL_DIR
do
  value="${!key:-}"
  expanded="$(eval "printf '%s' \"${value}\"")"
  if [[ -n "${expanded}" && -d "${expanded}" ]]; then
    check_ok "${key}: ${expanded}"
  else
    check_fail "${key} missing directory: ${value}"
  fi
done

if [[ -x "${VENV_PY}" ]]; then
  "${VENV_PY}" - <<'PY' || status=1
import importlib
import json
import os
import sys

modules = ["paddle", "paddleocr", "paddlex", "fitz"]
results = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        results[name] = {
            "ok": True,
            "version": getattr(module, "__version__", ""),
        }
    except Exception as exc:
        results[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

try:
    from paddleocr import PaddleOCRVL  # noqa: F401
    results["PaddleOCRVL"] = {"ok": True}
except Exception as exc:
    results["PaddleOCRVL"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(results, ensure_ascii=False, indent=2))
if not all(item.get("ok") for item in results.values()):
    sys.exit(1)
PY
fi

if [[ "${PDF_OCR_OPTIMIZATION_PROFILE}" == "v100_32gb_fast" ]]; then
  check_ok "V100 fast profile enabled"
else
  check_warn "V100 fast profile is not enabled"
fi

if [[ "${PDF_OCR_TARGET_PAGES}" == "200" && "${PDF_OCR_TARGET_SECONDS}" == "300" ]]; then
  check_ok "Target is 200 pages / 300 seconds"
else
  check_warn "Target differs from 200 pages / 300 seconds"
fi

exit "${status}"
