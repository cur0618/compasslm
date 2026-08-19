#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export COMPASS_SKIP_MODEL_RESOLUTION=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"
compass_load_runtime_service_env_files

SESSION_NAME="${COMPASS_TMUX_SESSION:-compasslm}"
JSON_OUTPUT=0

usage() {
  cat <<'EOF'
Usage: project-gpu/compass_status.sh [--session NAME] [--json]

Shows CompassLM tmux session, ports.env state, process liveness, health probes,
local URLs, and Jupyter proxy URLs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=1
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

service_status() {
  local service="$1"
  local port_key url_key pid_key port url pid process_state health_state proxy_url
  read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"
  port="$(compass_state_value "${port_key}" || true)"
  url="$(compass_state_value "${url_key}" || true)"
  pid="$(compass_state_value "${pid_key}" || true)"
  proxy_url="$(compass_jupyter_proxy_url "${port}")"

  process_state="missing"
  if compass_process_is_alive "${pid:-}"; then
    process_state="alive"
  elif [[ -n "${pid:-}" ]]; then
    process_state="dead"
  fi

  health_state="unknown"
  if [[ "${process_state}" == "alive" && -n "${url}" ]]; then
    case "${service}" in
      embedding)
        if compass_probe_embedding_api "${url}" >/dev/null 2>&1; then health_state="ok"; else health_state="fail"; fi
        ;;
      llm)
        if compass_probe_llm_api "${url}" >/dev/null 2>&1; then health_state="ok"; else health_state="fail"; fi
        ;;
      backend)
        if compass_probe_backend_api "${url}" >/dev/null 2>&1; then health_state="ok"; else health_state="fail"; fi
        ;;
    esac
  fi

  printf '%s|%s|%s|%s|%s|%s|%s\n' "${service}" "${pid:-}" "${port:-}" "${process_state}" "${health_state}" "${url:-}" "${proxy_url:-}"
}

tmux_state="missing"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux_state="alive"
fi

rows="$(
  service_status embedding
  service_status llm
  service_status backend
)"

if [[ "${JSON_OUTPUT}" == "1" ]]; then
  STATUS_PYTHON_BIN="${COMPASS_STATUS_PYTHON_BIN:-}"
  if [[ -z "${STATUS_PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      STATUS_PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
      STATUS_PYTHON_BIN="python"
    else
      echo "[ERROR] --json requires python3 or python. Set COMPASS_STATUS_PYTHON_BIN=/path/to/python." >&2
      exit 5
    fi
  fi
  COMPASS_STATUS_ROWS="${rows}" "${STATUS_PYTHON_BIN}" - "${SESSION_NAME}" "${tmux_state}" "${COMPASS_PORT_STATE_FILE}" <<'PY'
import json
import os
import sys

session, tmux_state, state_file = sys.argv[1:4]
services = {}
for line in os.environ.get("COMPASS_STATUS_ROWS", "").splitlines():
    if not line.strip():
        continue
    service, pid, port, process_state, health_state, url, proxy_url = line.split("|", 6)
    services[service] = {
        "pid": pid,
        "port": port,
        "process": process_state,
        "health": health_state,
        "local_url": url,
        "jupyter_proxy_url": proxy_url,
    }
print(json.dumps({
    "session": session,
    "tmux": tmux_state,
    "state_file": state_file,
    "services": services,
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "[STATUS] session=${SESSION_NAME} tmux=${tmux_state}"
echo "[STATUS] state_file=${COMPASS_PORT_STATE_FILE}"
printf '%-10s %-8s %-6s %-8s %-7s %-38s %s\n' "service" "pid" "port" "process" "health" "local_url" "jupyter_proxy_url"
while IFS='|' read -r service pid port process_state health_state url proxy_url; do
  printf '%-10s %-8s %-6s %-8s %-7s %-38s %s\n' \
    "${service}" "${pid:-"-"}" "${port:-"-"}" "${process_state}" "${health_state}" "${url:-"-"}" "${proxy_url:-"-"}"
done <<<"${rows}"
