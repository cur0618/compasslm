#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export COMPASS_SKIP_MODEL_RESOLUTION=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"
compass_load_runtime_service_env_files

SESSION_NAME="${COMPASS_TMUX_SESSION:-compasslm}"
FORCE=0

usage() {
  cat <<'EOF'
Usage: project-gpu/compass_down.sh [--session NAME] [--force]

Stops CompassLM services in reverse startup order:
  backend -> llm -> embedding

Without --force, leftover candidate processes are only reported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

compass_init_runtime_state

pid_for_service() {
  local service="$1"
  local port_key url_key pid_key
  read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"
  compass_state_value "${pid_key}" || true
}

echo "[DOWN] Stopping CompassLM services session=${SESSION_NAME}"
backend_pid="$(pid_for_service backend)"
llm_pid="$(pid_for_service llm)"
embedding_pid="$(pid_for_service embedding)"

compass_shutdown_child backend "${backend_pid:-}" compass_down INT || true
compass_shutdown_child llm "${llm_pid:-}" compass_down INT || true
compass_shutdown_child embedding "${embedding_pid:-}" compass_down INT || true

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[DOWN] Killing tmux session=${SESSION_NAME}"
  tmux kill-session -t "${SESSION_NAME}" || true
fi

find_orphans() {
  ps -efww 2>/dev/null \
    | grep -F "${COMPASSLM_HOME}" \
    | grep -E 'uvicorn|llama-server|run_.*server|run_backend_api|paddle|ocr' \
    | grep -v grep || true
}

orphans="$(find_orphans)"
if [[ -n "${orphans}" ]]; then
  echo "[DOWN][WARN] Remaining CompassLM-related candidate processes:"
  echo "${orphans}"
  if [[ "${FORCE}" == "1" ]]; then
    echo "[DOWN] --force enabled; terminating remaining candidates under ${COMPASSLM_HOME}"
    while read -r _user pid _rest; do
      if [[ "${pid:-}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
      fi
    done <<<"${orphans}"
    sleep 2
    while read -r _user pid _rest; do
      if [[ "${pid:-}" =~ ^[0-9]+$ ]] && compass_process_is_alive "${pid}"; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    done <<<"${orphans}"
  else
    echo "[DOWN][INFO] Re-run with --force only if these are stale CompassLM processes."
  fi
else
  echo "[DOWN] No remaining CompassLM-related candidate processes found."
fi

echo "[DOWN] Done."
