#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"
compass_load_runtime_service_env_files

SESSION_NAME="${COMPASS_TMUX_SESSION:-compasslm}"
RESTART=0
SKIP_ASSETS=0

usage() {
  cat <<'EOF'
Usage: project-gpu/compass_up.sh [--session NAME] [--restart] [--skip-assets]

Starts CompassLM in a tmux session:
  embedding -> llm -> backend -> monitor
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --restart)
      RESTART=1
      shift
      ;;
    --skip-assets)
      SKIP_ASSETS=1
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

if [[ -z "${SESSION_NAME}" ]]; then
  echo "[ERROR] tmux session name must not be empty." >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is required for integrated CompassLM startup." >&2
  echo "        Install example: sudo apt-get update && sudo apt-get install -y tmux" >&2
  echo "        Conda example:   conda install -c conda-forge tmux" >&2
  exit 20
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${COMPASS_RUNTIME_LOG_ROOT:-${COMPASSLM_HOME}/logs/runtime}"
LOG_DIR="${COMPASS_RUNTIME_LOG_DIR:-${LOG_ROOT}/${timestamp}}"
mkdir -p "${LOG_DIR}"
printf '%s\n' "${LOG_DIR}" >"${LOG_ROOT}/latest_path.txt"
export COMPASS_RUNTIME_LOG_DIR="${LOG_DIR}"

write_summary() {
  local status="$1"
  local failure_stage="${2:-}"
  local started_at="${3:-${timestamp}}"
  python3 - "$status" "$failure_stage" "$started_at" "$LOG_DIR" "$SESSION_NAME" <<'PY'
import json
import os
import sys
import time

status, failure_stage, started_at, log_dir, session = sys.argv[1:6]
state_file = os.environ.get("COMPASS_PORT_STATE_FILE", "")
state = {}
if state_file and os.path.exists(state_file):
    with open(state_file, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                state[key] = value

def proxy(port: str) -> str:
    if not port:
        return ""
    base = os.environ.get("COMPASS_JUPYTER_PROXY_BASE", "").rstrip("/")
    if base:
        return f"{base}/{port}/"
    public = os.environ.get("JUPYTER_PUBLIC_BASE_URL", "https://<host>:8000").rstrip("/")
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").rstrip("/")
    if prefix:
        return f"{public}{prefix}/proxy/{port}/"
    user = os.environ.get("JUPYTERHUB_USER", "<user>")
    return f"{public}/user/{user}/proxy/{port}/"

payload = {
    "status": status,
    "failure_stage": failure_stage,
    "session": session,
    "started_at": started_at,
    "completed_at_unix": int(time.time()),
    "log_dir": log_dir,
    "services": {
        "embedding": {
            "port": state.get("EMBED_PORT_SELECTED", ""),
            "pid": state.get("EMBED_PID", ""),
            "local_url": state.get("EMBEDDING_API_URL_SELECTED", ""),
            "jupyter_proxy_url": proxy(state.get("EMBED_PORT_SELECTED", "")),
        },
        "llm": {
            "port": state.get("LLM_PORT_SELECTED", ""),
            "pid": state.get("LLM_PID", ""),
            "local_url": state.get("LLM_API_URL_SELECTED", ""),
            "jupyter_proxy_url": proxy(state.get("LLM_PORT_SELECTED", "")),
        },
        "backend": {
            "port": state.get("API_PORT_SELECTED", ""),
            "pid": state.get("API_PID", ""),
            "local_url": state.get("COMPASSLM_BASE_URL_SELECTED", ""),
            "jupyter_proxy_url": proxy(state.get("API_PORT_SELECTED", "")),
        },
    },
}
path = os.path.join(log_dir, "startup_summary.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
PY
}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${RESTART}" == "1" ]]; then
    "${SCRIPT_DIR}/compass_down.sh" --session "${SESSION_NAME}" || true
  else
    echo "[ERROR] tmux session already exists: ${SESSION_NAME}" >&2
    echo "        Use --restart or project-gpu/compass_down.sh --session ${SESSION_NAME}" >&2
    exit 21
  fi
fi

if ! python3 "${COMPASSLM_HOME}/scripts/prune_runtime_logs.py" \
  --root "${LOG_ROOT}" \
  --retention-days "${COMPASS_RUNTIME_LOG_RETENTION_DAYS:-14}" \
  --max-dirs "${COMPASS_RUNTIME_LOG_MAX_DIRS:-30}" \
  --active-dir "${LOG_DIR}"; then
  echo "[WARN] Runtime log retention cleanup failed; startup will continue." >&2
fi

compass_init_runtime_state

if [[ "${SKIP_ASSETS}" != "1" ]]; then
  echo "[UP] Checking GPU assets..."
  if ! "${SCRIPT_DIR}/check_gpu_assets.sh" 2>&1 | tee "${LOG_DIR}/assets.log"; then
    write_summary "failed" "check_gpu_assets" "${timestamp}"
    exit 30
  fi
  if grep -q '^\[MISS\]' "${LOG_DIR}/assets.log"; then
    echo "[ERROR] Asset check found MISS entries. See ${LOG_DIR}/assets.log" >&2
    write_summary "failed" "check_gpu_assets_miss" "${timestamp}"
    exit 31
  fi
fi

tmux new-session -d -s "${SESSION_NAME}" -n embedding \
  "cd '${COMPASSLM_HOME}' && COMPASS_RUNTIME_LOG_DIR='${LOG_DIR}' '${SCRIPT_DIR}/run_embedding_server.sh' 2>&1 | tee -a '${LOG_DIR}/embedding.log'"
echo "[UP] Started embedding pane. Waiting for readiness..."
echo "[UP] If this waits, inspect logs with: project-gpu/compass_logs.sh embedding -f"
if ! EMBEDDING_URL="$(compass_require_live_embedding_url)"; then
  write_summary "failed" "embedding_ready" "${timestamp}"
  echo "[ERROR] Embedding server did not become ready. See ${LOG_DIR}/embedding.log" >&2
  tail -n 80 "${LOG_DIR}/embedding.log" >&2 || true
  exit 40
fi

tmux new-window -t "${SESSION_NAME}" -n llm \
  "cd '${COMPASSLM_HOME}' && COMPASS_RUNTIME_LOG_DIR='${LOG_DIR}' '${SCRIPT_DIR}/run_llm_server.sh' 2>&1 | tee -a '${LOG_DIR}/llm.log'"
echo "[UP] Started LLM pane. Waiting for readiness..."
echo "[UP] If this waits, inspect logs with: project-gpu/compass_logs.sh llm -f"
if ! LLM_URL="$(compass_require_live_llm_url)"; then
  write_summary "failed" "llm_ready" "${timestamp}"
  echo "[ERROR] LLM server did not become ready. See ${LOG_DIR}/llm.log" >&2
  tail -n 80 "${LOG_DIR}/llm.log" >&2 || true
  exit 50
fi

tmux new-window -t "${SESSION_NAME}" -n backend \
  "cd '${COMPASSLM_HOME}' && COMPASS_RUNTIME_LOG_DIR='${LOG_DIR}' '${SCRIPT_DIR}/run_backend_api.sh' 2>&1 | tee -a '${LOG_DIR}/backend.log'"
echo "[UP] Started backend pane. Waiting for readiness..."
echo "[UP] If this waits, inspect logs with: project-gpu/compass_logs.sh backend -f"
if ! BACKEND_URL="$(compass_require_live_backend_url)"; then
  write_summary "failed" "backend_ready" "${timestamp}"
  echo "[ERROR] Backend API did not become ready. See ${LOG_DIR}/backend.log" >&2
  tail -n 80 "${LOG_DIR}/backend.log" >&2 || true
  exit 60
fi

tmux new-window -t "${SESSION_NAME}" -n monitor \
  "cd '${COMPASSLM_HOME}' && while true; do clear; date; COMPASS_RUNTIME_LOG_DIR='${LOG_DIR}' '${SCRIPT_DIR}/compass_status.sh' --session '${SESSION_NAME}'; sleep 10; done 2>&1 | tee -a '${LOG_DIR}/monitor.log'"

write_summary "ok" "" "${timestamp}"

backend_port="$(compass_state_value API_PORT_SELECTED || true)"
llm_port="$(compass_state_value LLM_PORT_SELECTED || true)"
embed_port="$(compass_state_value EMBED_PORT_SELECTED || true)"

cat <<EOF
[UP] CompassLM is ready.
  tmux session: ${SESSION_NAME}
  logs: ${LOG_DIR}
  embedding: ${EMBEDDING_URL}
  llm:       ${LLM_URL}
  backend:   ${BACKEND_URL}

Jupyter proxy URLs:
  backend:   $(compass_jupyter_proxy_url "${backend_port}")
  llm UI:    $(compass_jupyter_proxy_url "${llm_port}")
  embedding: $(compass_jupyter_proxy_url "${embed_port}")

Attach:
  tmux attach -t ${SESSION_NAME}
EOF
