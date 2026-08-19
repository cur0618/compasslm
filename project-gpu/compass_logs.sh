#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export COMPASS_SKIP_MODEL_RESOLUTION=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"

SESSION_NAME="${COMPASS_TMUX_SESSION:-compasslm}"
SERVICE="all"
FOLLOW=0
LINES="${COMPASS_LOG_TAIL_LINES:-160}"
PANE=0

usage() {
  cat <<'EOF'
Usage: project-gpu/compass_logs.sh [SERVICE] [--service SERVICE] [-f|--follow] [--lines N] [--pane]

SERVICE can be all, assets, embedding, llm, backend, monitor, or summary.
By default this tails logs/runtime/latest_path.txt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --service)
      SERVICE="${2:-}"
      shift 2
      ;;
    -f|--follow)
      FOLLOW=1
      shift
      ;;
    --lines)
      LINES="${2:-160}"
      shift 2
      ;;
    --pane)
      PANE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      SERVICE="$1"
      shift
      ;;
  esac
done

latest_file="${COMPASSLM_HOME}/logs/runtime/latest_path.txt"
LOG_DIR="${COMPASS_RUNTIME_LOG_DIR:-}"
if [[ -z "${LOG_DIR}" && -f "${latest_file}" ]]; then
  LOG_DIR="$(head -n 1 "${latest_file}")"
fi
if [[ -z "${LOG_DIR}" || ! -d "${LOG_DIR}" ]]; then
  echo "[ERROR] Runtime log directory not found. Run project-gpu/compass_up.sh first or set COMPASS_RUNTIME_LOG_DIR." >&2
  exit 3
fi

if [[ "${PANE}" == "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not available." >&2
    exit 4
  fi
  if [[ "${SERVICE}" == "all" ]]; then
    tmux list-windows -t "${SESSION_NAME}" 2>/dev/null || true
  else
    tmux capture-pane -t "${SESSION_NAME}:${SERVICE}" -p -S -"${LINES}"
  fi
  exit 0
fi

log_file_for_service() {
  case "$1" in
    assets) echo "${LOG_DIR}/assets.log" ;;
    embedding) echo "${LOG_DIR}/embedding.log" ;;
    llm) echo "${LOG_DIR}/llm.log" ;;
    backend) echo "${LOG_DIR}/backend.log" ;;
    monitor) echo "${LOG_DIR}/monitor.log" ;;
    summary) echo "${LOG_DIR}/startup_summary.json" ;;
    *) return 2 ;;
  esac
}

show_one() {
  local service="$1"
  local path
  path="$(log_file_for_service "${service}")"
  echo "===== ${service}: ${path} ====="
  if [[ ! -f "${path}" ]]; then
    echo "[missing]"
    return 0
  fi
  if [[ "${FOLLOW}" == "1" ]]; then
    tail -n "${LINES}" -f "${path}"
  else
    tail -n "${LINES}" "${path}"
  fi
}

case "${SERVICE}" in
  all)
    for service in summary assets embedding llm backend monitor; do
      show_one "${service}"
    done
    ;;
  assets|embedding|llm|backend|monitor|summary)
    show_one "${SERVICE}"
    ;;
  *)
    echo "[ERROR] Unknown service: ${SERVICE}" >&2
    usage >&2
    exit 2
    ;;
esac
