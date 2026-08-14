#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/pdf [output-json]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"
compass_init_runtime_state

LLM_WAS_RUNNING=0
LLM_PID="$(compass_state_value LLM_PID || true)"
LLM_LOG="${COMPASSLM_LOGS_DIR:-${COMPASSLM_HOME}/logs}/llm-after-ocr.log"

restore_services() {
  "${SCRIPT_DIR}/paddleocr-hps/manage_hps.sh" stop >/dev/null 2>&1 || true
  if [[ "${LLM_WAS_RUNNING}" == "1" ]]; then
    mkdir -p "$(dirname "${LLM_LOG}")"
    nohup "${SCRIPT_DIR}/run_llm_server.sh" >>"${LLM_LOG}" 2>&1 &
  fi
}
trap restore_services EXIT INT TERM

case "${PDF_OCR_PAUSE_LLM_DURING_JOB:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ "${LLM_PID}" =~ ^[0-9]+$ ]] && kill -0 "${LLM_PID}" 2>/dev/null; then
      LLM_WAS_RUNNING=1
      kill "${LLM_PID}"
      for _ in $(seq 1 30); do
        kill -0 "${LLM_PID}" 2>/dev/null || break
        sleep 1
      done
    fi
    ;;
esac

"${SCRIPT_DIR}/paddleocr-hps/manage_hps.sh" start
export PDF_OCR_BACKEND=hps
export PDF_OCR_HPS_FALLBACK_TO_LOCAL=0
export PDF_OCR_EXEC_BATCH_PAGES="${PDF_OCR_EXEC_BATCH_PAGES:-3}"
export PDF_OCR_PARALLEL_MAX_WORKERS=1
export PDF_OCR_PERSISTENT_WORKERS=1
"${SCRIPT_DIR}/benchmark_pdf_ocr_h100.sh" "$@"
