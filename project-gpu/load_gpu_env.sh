#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${COMPASSLM_HOME:-}" ]]; then
  COMPASSLM_HOME="$(cd "${COMPASSLM_HOME}" && pwd)"
else
  COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

export COMPASSLM_HOME
export PROJECT_GPU_HOME="${COMPASSLM_HOME}/project-gpu"
export EMBEDDING_SERVER_HOME="${PROJECT_GPU_HOME}/embedding-gpu-server"
export MAIN_BACKEND_HOME="${PROJECT_GPU_HOME}/main-backend"

compass_init_runtime_state() {
  if [[ -n "${COMPASS_PORT_STATE_FILE:-}" && -n "${COMPASS_PORT_LOCK_FILE:-}" ]]; then
    return 0
  fi

  local state_root project_hash uid_value
  uid_value="$(id -u)"
  if command -v sha256sum >/dev/null 2>&1; then
    project_hash="$(printf '%s' "${COMPASSLM_HOME}" | sha256sum | cut -c1-12)"
  else
    project_hash="$(printf '%s' "${COMPASSLM_HOME}" | cksum | awk '{print $1}')"
  fi
  state_root="${COMPASS_RUNTIME_STATE_DIR:-${XDG_RUNTIME_DIR:-/tmp}/compasslm-${uid_value}-${project_hash}}"
  mkdir -p "${state_root}"
  chmod 700 "${state_root}" 2>/dev/null || true
  export COMPASS_RUNTIME_STATE_DIR="${state_root}"
  export COMPASS_PORT_STATE_FILE="${state_root}/ports.env"
  export COMPASS_PORT_LOCK_FILE="${state_root}/ports.lock"
}

compass_port_is_available() {
  local port="$1"
  python3 -c 'import socket,sys
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()' "${port}" >/dev/null 2>&1
}

compass_port_is_reserved() {
  local wanted_port="$1"
  local service port_key url_key pid_key reserved_port reserved_pid
  for service in embedding llm backend; do
    read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"
    reserved_port="$(compass_state_value "${port_key}" || true)"
    reserved_pid="$(compass_state_value "${pid_key}" || true)"
    if [[ "${reserved_port}" == "${wanted_port}" && "${reserved_pid}" =~ ^[0-9]+$ ]] \
      && kill -0 "${reserved_pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

compass_select_available_port() {
  local service="$1"
  local start_port="$2"
  local range_start="${COMPASS_PORT_RANGE_START:-8000}"
  local range_end="${COMPASS_PORT_RANGE_END:-8099}"
  local candidate raw_candidates

  if [[ ! "${range_start}" =~ ^[0-9]+$ || ! "${range_end}" =~ ^[0-9]+$ || ! "${start_port}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid port range for ${service}: start=${start_port} range=${range_start}-${range_end}" >&2
    return 2
  fi
  if (( range_start < 1 || range_end > 65535 || range_start > range_end )); then
    echo "[ERROR] Invalid port range for ${service}: ${range_start}-${range_end}" >&2
    return 2
  fi

  if [[ -n "${COMPASS_PORT_CANDIDATES:-}" ]]; then
    raw_candidates="${COMPASS_PORT_CANDIDATES//,/ }"
  else
    if (( start_port < range_start || start_port > range_end )); then
      start_port="${range_start}"
    fi
    raw_candidates="$(seq "${start_port}" "${range_end}")"
    if (( start_port > range_start )); then
      raw_candidates+=" $(seq "${range_start}" "$((start_port - 1))")"
    fi
  fi

  for candidate in ${raw_candidates}; do
    [[ "${candidate}" =~ ^[0-9]+$ ]] || continue
    if compass_port_is_available "${candidate}" && ! compass_port_is_reserved "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "[ERROR] No available port for ${service} in ${range_start}-${range_end}." >&2
  return 3
}

compass_select_service_port() {
  local service="$1"
  local start_port="$2"
  local configured_port="${3:-${start_port}}"
  local auto_port="${COMPASS_AUTO_PORT:-1}"
  local selected lock_fd
  compass_init_runtime_state

  case "${auto_port,,}" in
    0|false|no|off)
      if ! compass_port_is_available "${configured_port}"; then
        echo "[ERROR] Fixed port ${configured_port} for ${service} is already in use." >&2
        return 4
      fi
      compass_reserve_service_port "${service}" "${configured_port}" "$$"
      echo "[PORT] service=${service} requested=${configured_port} selected=${configured_port} mode=fixed" >&2
      echo "${configured_port}"
      return 0
      ;;
  esac

  exec {lock_fd}>"${COMPASS_PORT_LOCK_FILE}"
  flock "${lock_fd}"
  selected="$(compass_select_available_port "${service}" "${start_port}")"
  _compass_reserve_service_port_unlocked "${service}" "${selected}" "$$"
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
  echo "[PORT] service=${service} requested=${start_port} selected=${selected} mode=auto" >&2
  echo "${selected}"
}

compass_service_state_keys() {
  case "$1" in
    embedding) echo "EMBED_PORT_SELECTED EMBEDDING_API_URL_SELECTED EMBED_PID" ;;
    llm) echo "LLM_PORT_SELECTED LLM_API_URL_SELECTED LLM_PID" ;;
    backend) echo "API_PORT_SELECTED COMPASSLM_BASE_URL_SELECTED API_PID" ;;
    *) return 2 ;;
  esac
}

_compass_reserve_service_port_unlocked() {
  local service="$1"
  local port="$2"
  local pid="${3:-$$}"
  local port_key url_key pid_key tmp_file
  compass_init_runtime_state
  read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"
  tmp_file="${COMPASS_PORT_STATE_FILE}.tmp.$$"
  if [[ -f "${COMPASS_PORT_STATE_FILE}" ]]; then
    grep -Ev "^(${port_key}|${url_key}|${pid_key})=" "${COMPASS_PORT_STATE_FILE}" >"${tmp_file}" || true
  else
    : >"${tmp_file}"
  fi
  printf '%s=%s\n%s=%s\n' "${port_key}" "${port}" "${pid_key}" "${pid}" >>"${tmp_file}"
  chmod 600 "${tmp_file}"
  mv -f "${tmp_file}" "${COMPASS_PORT_STATE_FILE}"
}

compass_reserve_service_port() {
  local service="$1"
  local port="$2"
  local pid="${3:-$$}"
  local lock_fd
  compass_init_runtime_state
  exec {lock_fd}>"${COMPASS_PORT_LOCK_FILE}"
  flock "${lock_fd}"
  _compass_reserve_service_port_unlocked "${service}" "${port}" "${pid}"
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

compass_write_service_state() {
  local service="$1"
  local port="$2"
  local url="$3"
  local pid="${4:-$$}"
  local port_key url_key pid_key lock_fd tmp_file
  compass_init_runtime_state
  read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"

  exec {lock_fd}>"${COMPASS_PORT_LOCK_FILE}"
  flock "${lock_fd}"
  tmp_file="${COMPASS_PORT_STATE_FILE}.tmp.$$"
  if [[ -f "${COMPASS_PORT_STATE_FILE}" ]]; then
    grep -Ev "^(${port_key}|${url_key}|${pid_key})=" "${COMPASS_PORT_STATE_FILE}" >"${tmp_file}" || true
  else
    : >"${tmp_file}"
  fi
  {
    printf '%s=%s\n' "${port_key}" "${port}"
    printf '%s=%s\n' "${url_key}" "${url}"
    printf '%s=%s\n' "${pid_key}" "${pid}"
  } >>"${tmp_file}"
  chmod 600 "${tmp_file}"
  mv -f "${tmp_file}" "${COMPASS_PORT_STATE_FILE}"
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
  echo "[PORT] state_file=${COMPASS_PORT_STATE_FILE}" >&2
}

compass_state_value() {
  local key="$1"
  compass_init_runtime_state
  [[ -f "${COMPASS_PORT_STATE_FILE}" ]] || return 1
  awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); value=$0} END {if (value != "") print value}' "${COMPASS_PORT_STATE_FILE}"
}

compass_load_live_service_url() {
  local service="$1"
  local port_key url_key pid_key port url pid
  read -r port_key url_key pid_key <<<"$(compass_service_state_keys "${service}")"
  port="$(compass_state_value "${port_key}" || true)"
  url="$(compass_state_value "${url_key}" || true)"
  pid="$(compass_state_value "${pid_key}" || true)"
  [[ "${port}" =~ ^[0-9]+$ && -n "${url}" && "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  if compass_port_is_available "${port}"; then
    return 1
  fi
  echo "${url}"
}

compass_probe_embedding_api() {
  local base_url="$1"
  local api_key="${EMBEDDING_API_KEY:-}"
  local timeout_seconds="${EMBEDDING_PROBE_TIMEOUT_SECONDS:-10}"
  python3 - "${base_url%/}" "${api_key}" "${timeout_seconds}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url, api_key, timeout_raw = sys.argv[1:4]
timeout = max(1.0, float(timeout_raw))

try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as response:
        health = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or health.get("status") != "ok":
        print("health_failed", file=sys.stderr)
        raise SystemExit(23)

    payload = json.dumps({
        "texts": ["embedding readiness probe"],
        "task": "passage",
        "index_name": "large",
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/embed", data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    vectors = result.get("vectors")
    if response.status != 200 or not isinstance(vectors, list) or len(vectors) != 1 or not vectors[0]:
        print("embed_probe_failed", file=sys.stderr)
        raise SystemExit(25)
except urllib.error.HTTPError as exc:
    if exc.code == 401:
        print("unauthorized", file=sys.stderr)
        raise SystemExit(24)
    print(f"embed_probe_failed http_status={exc.code}", file=sys.stderr)
    raise SystemExit(25)
except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"port_unreachable detail={type(exc).__name__}", file=sys.stderr)
    raise SystemExit(22)
PY
}

compass_require_live_embedding_url() {
  local auto_port="${COMPASS_AUTO_PORT:-1}"
  local url port pid probe_error
  case "${auto_port,,}" in
    0|false|no|off)
      url="${EMBEDDING_API_URL:-}"
      if [[ -z "${url}" ]]; then
        echo "[ERROR] embedding_ready_check=failed reason=state_missing mode=fixed EMBEDDING_API_URL is empty" >&2
        return 31
      fi
      if ! probe_error="$(compass_probe_embedding_api "${url}" 2>&1)"; then
        echo "[ERROR] embedding_ready_check=failed reason=${probe_error:-embed_probe_failed} mode=fixed url=${url}" >&2
        return 32
      fi
      echo "${url%/}"
      return 0
      ;;
  esac

  compass_init_runtime_state
  port="$(compass_state_value EMBED_PORT_SELECTED || true)"
  url="$(compass_state_value EMBEDDING_API_URL_SELECTED || true)"
  pid="$(compass_state_value EMBED_PID || true)"
  if [[ -z "${port}" || -z "${url}" || -z "${pid}" ]]; then
    echo "[ERROR] embedding_ready_check=failed reason=state_missing mode=auto state_file=${COMPASS_PORT_STATE_FILE}" >&2
    return 33
  fi
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[ERROR] embedding_ready_check=failed reason=process_dead mode=auto pid=${pid} url=${url}" >&2
    return 34
  fi
  if ! probe_error="$(compass_probe_embedding_api "${url}" 2>&1)"; then
    echo "[ERROR] embedding_ready_check=failed reason=${probe_error:-embed_probe_failed} mode=auto pid=${pid} url=${url}" >&2
    return 35
  fi
  echo "${url%/}"
}

compass_wait_for_embedding_ready() {
  local url="$1"
  local pid="$2"
  local timeout_seconds="${3:-${EMBEDDING_READY_TIMEOUT_SECONDS:-600}}"
  local started now
  started="$(date +%s)"
  while true; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] embedding_ready_check=failed reason=process_dead pid=${pid} url=${url}" >&2
      return 41
    fi
    if compass_probe_embedding_api "${url}" >/dev/null 2>&1; then
      echo "[READY] embedding url=${url} pid=${pid}" >&2
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= timeout_seconds )); then
      echo "[ERROR] embedding_ready_check=failed reason=embed_probe_failed timeout_seconds=${timeout_seconds} pid=${pid} url=${url}" >&2
      return 42
    fi
    sleep 1
  done
}

compass_probe_llm_api() {
  local api_url="$1"
  local api_key="${LLM_API_KEY:-}"
  local timeout_seconds="${LLM_PROBE_TIMEOUT_SECONDS:-10}"
  python3 - "${api_url}" "${api_key}" "${timeout_seconds}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

api_url, api_key, timeout_raw = sys.argv[1:4]
timeout = max(1.0, float(timeout_raw))
base_url = api_url.split("/v1/", 1)[0].rstrip("/")
headers = {"Accept": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

try:
    health_request = urllib.request.Request(f"{base_url}/health", headers=headers)
    with urllib.request.urlopen(health_request, timeout=timeout) as response:
        if response.status != 200:
            print("health_failed", file=sys.stderr)
            raise SystemExit(43)
    models_request = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
    with urllib.request.urlopen(models_request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or not isinstance(payload.get("data"), list):
        print("model_list_failed", file=sys.stderr)
        raise SystemExit(45)
except urllib.error.HTTPError as exc:
    if exc.code == 401:
        print("unauthorized", file=sys.stderr)
        raise SystemExit(44)
    print(f"model_list_failed http_status={exc.code}", file=sys.stderr)
    raise SystemExit(45)
except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"port_unreachable detail={type(exc).__name__}", file=sys.stderr)
    raise SystemExit(42)
PY
}

compass_require_live_llm_url() {
  local auto_port="${COMPASS_AUTO_PORT:-1}"
  local url port pid probe_error
  case "${auto_port,,}" in
    0|false|no|off)
      url="${LLM_API_URL:-}"
      if [[ -z "${url}" ]]; then
        echo "[ERROR] llm_ready_check=failed reason=state_missing mode=fixed LLM_API_URL is empty" >&2
        return 51
      fi
      if ! probe_error="$(compass_probe_llm_api "${url}" 2>&1)"; then
        echo "[ERROR] llm_ready_check=failed reason=${probe_error:-model_list_failed} mode=fixed url=${url}" >&2
        return 52
      fi
      echo "${url}"
      return 0
      ;;
  esac

  compass_init_runtime_state
  port="$(compass_state_value LLM_PORT_SELECTED || true)"
  url="$(compass_state_value LLM_API_URL_SELECTED || true)"
  pid="$(compass_state_value LLM_PID || true)"
  if [[ -z "${port}" || -z "${url}" || -z "${pid}" ]]; then
    echo "[ERROR] llm_ready_check=failed reason=state_missing mode=auto state_file=${COMPASS_PORT_STATE_FILE}" >&2
    return 53
  fi
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[ERROR] llm_ready_check=failed reason=process_dead mode=auto pid=${pid} url=${url}" >&2
    return 54
  fi
  if ! probe_error="$(compass_probe_llm_api "${url}" 2>&1)"; then
    echo "[ERROR] llm_ready_check=failed reason=${probe_error:-model_list_failed} mode=auto pid=${pid} url=${url}" >&2
    return 55
  fi
  echo "${url}"
}

compass_wait_for_llm_ready() {
  local url="$1"
  local pid="$2"
  local timeout_seconds="${3:-${LLM_READY_TIMEOUT_SECONDS:-600}}"
  local started now
  started="$(date +%s)"
  while true; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] llm_ready_check=failed reason=process_dead pid=${pid} url=${url}" >&2
      return 61
    fi
    if compass_probe_llm_api "${url}" >/dev/null 2>&1; then
      echo "[READY] llm url=${url} pid=${pid}" >&2
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= timeout_seconds )); then
      echo "[ERROR] llm_ready_check=failed reason=model_list_failed timeout_seconds=${timeout_seconds} pid=${pid} url=${url}" >&2
      return 62
    fi
    sleep 1
  done
}

compass_probe_backend_api() {
  local base_url="$1"
  local timeout_seconds="${BACKEND_PROBE_TIMEOUT_SECONDS:-10}"
  python3 - "${base_url%/}" "${timeout_seconds}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url, timeout_raw = sys.argv[1:3]
timeout = max(1.0, float(timeout_raw))
try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or payload.get("status") != "ok" or payload.get("service") != "compasslm-backend":
        print("health_failed", file=sys.stderr)
        raise SystemExit(73)
except urllib.error.HTTPError as exc:
    print(f"health_failed http_status={exc.code}", file=sys.stderr)
    raise SystemExit(73)
except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"port_unreachable detail={type(exc).__name__}", file=sys.stderr)
    raise SystemExit(72)
PY
}

compass_require_live_backend_url() {
  local url port pid probe_error
  compass_init_runtime_state
  port="$(compass_state_value API_PORT_SELECTED || true)"
  url="$(compass_state_value COMPASSLM_BASE_URL_SELECTED || true)"
  pid="$(compass_state_value API_PID || true)"
  if [[ -z "${port}" || -z "${url}" || -z "${pid}" ]]; then
    echo "[ERROR] backend_ready_check=failed reason=state_missing state_file=${COMPASS_PORT_STATE_FILE}" >&2
    return 81
  fi
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[ERROR] backend_ready_check=failed reason=process_dead pid=${pid} url=${url}" >&2
    return 82
  fi
  if ! probe_error="$(compass_probe_backend_api "${url}" 2>&1)"; then
    echo "[ERROR] backend_ready_check=failed reason=${probe_error:-health_failed} pid=${pid} url=${url}" >&2
    return 83
  fi
  echo "${url%/}"
}

compass_wait_for_backend_ready() {
  local url="$1"
  local pid="$2"
  local timeout_seconds="${3:-${BACKEND_READY_TIMEOUT_SECONDS:-120}}"
  local started now
  started="$(date +%s)"
  while true; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] backend_ready_check=failed reason=process_dead pid=${pid} url=${url}" >&2
      return 91
    fi
    if compass_probe_backend_api "${url}" >/dev/null 2>&1; then
      echo "[READY] backend url=${url} pid=${pid}" >&2
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= timeout_seconds )); then
      echo "[ERROR] backend_ready_check=failed reason=health_failed timeout_seconds=${timeout_seconds} pid=${pid} url=${url}" >&2
      return 92
    fi
    sleep 1
  done
}

compass_load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

compass_detect_embedding_model_path() {
  local model_root="${EMBEDDING_SERVER_HOME}/models"
  local alt_root="${EMBEDDING_SERVER_HOME}/multilingual-e5-large"
  local preferred_dirs=(
    "${model_root}/qwen3-embedding-0.6b"
    "${model_root}/Qwen3-Embedding-0.6B"
    "${model_root}/Qwen/Qwen3-Embedding-0.6B"
    "${model_root}/Qwen__Qwen3-Embedding-0.6B"
    "${model_root}/kure-v1"
    "${model_root}/KURE-v1"
    "${model_root}/nlpai-lab/kure-v1"
    "${model_root}/nlpai-lab/KURE-v1"
    "${model_root}/nlpai-lab__kure-v1"
    "${model_root}/nlpai-lab__KURE-v1"
    "${model_root}/multilingual-e5-large"
    "${model_root}/multilingual-e5-base"
    "${model_root}/qwen3-embedding-4b"
    "${model_root}/Qwen3-Embedding-4B"
    "${model_root}/Qwen/Qwen3-Embedding-4B"
    "${model_root}/Qwen__Qwen3-Embedding-4B"
    "${model_root}/jina-embeddings-v5-text-small"
    "${model_root}/jinaai/jina-embeddings-v5-text-small"
    "${model_root}/jinaai__jina-embeddings-v5-text-small"
  )
  local candidate=""

  # Prefer already-valid model directories first.
  for candidate in "${preferred_dirs[@]}"; do
    if compass_model_marker_exists "${candidate}"; then
      echo "${candidate}"
      return
    fi
  done

  # If preferred paths exist but are incomplete, resolver will retry under parent.
  for candidate in "${preferred_dirs[@]}"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done

  if [[ -d "${model_root}" ]]; then
    echo "${model_root}"
    return
  fi
  if [[ -d "${alt_root}" ]]; then
    echo "${alt_root}"
    return
  fi
  echo "${model_root}"
}

compass_model_marker_exists() {
  local model_dir="$1"
  local has_st_metadata=""
  local has_config=""
  local has_weights=""

  [[ -d "${model_dir}" ]] || return 1

  # Sentence-transformers metadata strongly indicates the model root.
  has_st_metadata="$(find "${model_dir}" -maxdepth 1 -type f \( -name modules.json -o -name config_sentence_transformers.json -o -name sentence_bert_config.json \) | head -n 1 || true)"
  if [[ -n "${has_st_metadata}" ]]; then
    return 0
  fi

  # Generic HF layout fallback: config + weight file in same directory.
  has_config="$(find "${model_dir}" -maxdepth 1 -type f -name config.json | head -n 1 || true)"
  has_weights="$(find "${model_dir}" -maxdepth 1 -type f \( -name model.safetensors -o -name '*.safetensors' -o -name pytorch_model.bin -o -name 'pytorch_model-*.bin' \) | head -n 1 || true)"
  [[ -n "${has_config}" && -n "${has_weights}" ]]
}

compass_find_first_model_dir_under() {
  local root_dir="$1"
  local search_depth="${2:-4}"
  local child=""

  [[ -d "${root_dir}" ]] || return 1
  while IFS= read -r child; do
    if compass_model_marker_exists "${child}"; then
      echo "${child}"
      return 0
    fi
  done < <(find "${root_dir}" -mindepth 0 -maxdepth "${search_depth}" -type d | sort)
  return 1
}

compass_resolve_embedding_model_path() {
  local requested="${1:-}"
  local model_root preferred child candidate parent_dir
  local child_base
  local priority_dirs=()
  local priority_patterns=(
    "*qwen*3*embedding*0.6b*"
    "*qwen*3*embedding*0-6b*"
    "*kure-v1*"
    "*kure_v1*"
    "*nlpai-lab*kure-v1*"
    "*multilingual-e5-large*"
    "*multilingual-e5-base*"
    "*qwen*3*embedding*4b*"
    "*jina*embeddings-v5-text-small*"
  )
  local pattern lower_path

  if [[ -z "${requested}" ]]; then
    requested="$(compass_detect_embedding_model_path)"
  fi
  requested="${requested/#\~/${HOME}}"

  if [[ ! -d "${requested}" ]]; then
    echo "${requested}"
    return
  fi

  # If path itself already points to a model folder, keep it.
  if compass_model_marker_exists "${requested}"; then
    echo "${requested}"
    return
  fi

  model_root="${requested}"
  priority_dirs=(
    "${model_root}/qwen3-embedding-0.6b"
    "${model_root}/Qwen3-Embedding-0.6B"
    "${model_root}/Qwen/Qwen3-Embedding-0.6B"
    "${model_root}/Qwen__Qwen3-Embedding-0.6B"
    "${model_root}/kure-v1"
    "${model_root}/KURE-v1"
    "${model_root}/nlpai-lab/kure-v1"
    "${model_root}/nlpai-lab/KURE-v1"
    "${model_root}/nlpai-lab__kure-v1"
    "${model_root}/nlpai-lab__KURE-v1"
    "${model_root}/multilingual-e5-large"
    "${model_root}/multilingual-e5-base"
    "${model_root}/qwen3-embedding-4b"
    "${model_root}/Qwen3-Embedding-4B"
    "${model_root}/Qwen/Qwen3-Embedding-4B"
    "${model_root}/Qwen__Qwen3-Embedding-4B"
    "${model_root}/jina-embeddings-v5-text-small"
    "${model_root}/jinaai/jina-embeddings-v5-text-small"
    "${model_root}/jinaai__jina-embeddings-v5-text-small"
  )
  for preferred in "${priority_dirs[@]}"; do
    candidate="$(compass_find_first_model_dir_under "${preferred}" 4 || true)"
    if [[ -n "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done

  # Prefer KURE-v1 directories under model_root first (case-insensitive).
  while IFS= read -r child; do
    child_base="$(basename "${child}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${child_base}" == "kure-v1" || "${child_base}" == "kure_v1" ]]; then
      candidate="$(compass_find_first_model_dir_under "${child}" 4 || true)"
      if [[ -n "${candidate}" ]]; then
        echo "${candidate}"
        return
      fi
    fi
  done < <(find "${model_root}" -mindepth 1 -maxdepth 6 -type d | sort)

  # Prefer model directories by name pattern under 6-level depth.
  while IFS= read -r child; do
    lower_path="$(echo "${child}" | tr '[:upper:]' '[:lower:]')"
    for pattern in "${priority_patterns[@]}"; do
      if [[ "${lower_path}" == ${pattern} ]]; then
        candidate="$(compass_find_first_model_dir_under "${child}" 4 || true)"
        if [[ -n "${candidate}" ]]; then
          echo "${candidate}"
          return
        fi
      fi
    done
  done < <(find "${model_root}" -mindepth 1 -maxdepth 6 -type d | sort)

  # Fallback: first valid model directory under model_root.
  candidate="$(compass_find_first_model_dir_under "${model_root}" 6 || true)"
  if [[ -n "${candidate}" ]]; then
    echo "${candidate}"
    return
  fi

  # If a fixed subdir (ex: .../models/kure-v1) is empty, retry from parent.
  parent_dir="$(dirname "${requested}")"
  if [[ "${requested}" != "${parent_dir}" && -d "${parent_dir}" && "$(basename "${requested}")" != "models" ]]; then
    candidate="$(compass_resolve_embedding_model_path "${parent_dir}")"
    if [[ -n "${candidate}" && "${candidate}" != "${parent_dir}" ]]; then
      echo "${candidate}"
      return
    fi
  fi

  echo "${requested}"
}

compass_rebase_project_path() {
  local raw="${1:-}"
  local expanded normalized suffix

  if [[ -z "${raw}" ]]; then
    echo ""
    return
  fi

  expanded="${raw/#\~/${HOME}}"

  if [[ "${expanded}" == project-gpu/* ]]; then
    echo "${COMPASSLM_HOME}/${expanded}"
    return
  fi

  normalized="${expanded//\\//}"
  if [[ "${normalized}" == *"/project-gpu/"* ]]; then
    suffix="${normalized#*"/project-gpu/"}"
    echo "${COMPASSLM_HOME}/project-gpu/${suffix}"
    return
  fi
  if [[ "${normalized}" == *"/compasslm/"* ]]; then
    suffix="${normalized#*"/compasslm/"}"
    echo "${COMPASSLM_HOME}/${suffix}"
    return
  fi

  echo "${expanded}"
}

compass_prefer_existing_or_rebased_path() {
  local raw="${1:-}"
  local expanded rebased

  if [[ -z "${raw}" ]]; then
    echo ""
    return
  fi

  expanded="${raw/#\~/${HOME}}"
  if [[ -e "${expanded}" || -d "${expanded}" ]]; then
    echo "${expanded}"
    return
  fi

  rebased="$(compass_rebase_project_path "${expanded}")"
  if [[ "${rebased}" != "${expanded}" ]]; then
    echo "${rebased}"
    return
  fi

  echo "${expanded}"
}

compass_is_wsl() {
  [[ -r /proc/version ]] && grep -qiE "(microsoft|wsl)" /proc/version
}

compass_allow_windows_exe_runtime() {
  local os_name
  os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "${os_name}" in
    mingw*|msys*|cygwin*)
      return 0
      ;;
  esac
  if compass_is_wsl && command -v cmd.exe >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

compass_detect_llm_model_path() {
  local models_dir="${LLM_MODELS_DIR:-${MAIN_BACKEND_HOME}/models/llm}"
  local detected=""
  local pattern=""
  local preferred_patterns=(
    'qwen*3.5*9b*.gguf'
    'qwen*3.5*9b*q4*k*m*.gguf'
    'qwen*3.5*9b*instruct*q4*k*m*.gguf'
    'qwen*3*9b*q4*k*m*.gguf'
    'qwen*3*9b*instruct*q4*k*m*.gguf'
    'qwen*3*32b*q4*k*m*.gguf'
    'qwen*3*32b*instruct*q4*k*m*.gguf'
    'qwen*2.5*14b*instruct*q4*k*m*.gguf'
    'qwen*14b*instruct*q4*k*m*.gguf'
    'gemma*3n*e4b*q4*k*m*.gguf'
  )

  if [[ -d "${models_dir}" ]]; then
    for pattern in "${preferred_patterns[@]}"; do
      detected="$(find "${models_dir}" -maxdepth 5 -type f -iname "${pattern}" 2>/dev/null | sort | head -n 1 || true)"
      if [[ -n "${detected}" ]]; then
        echo "${detected}"
        return
      fi
    done

    detected="$(find "${models_dir}" -maxdepth 5 -type f -iname '*.gguf' 2>/dev/null | sort | head -n 1 || true)"
    if [[ -n "${detected}" ]]; then
      echo "${detected}"
      return
    fi
  fi

  echo "${models_dir}/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf"
}

compass_detect_llm_runtime() {
  local candidates=("${MAIN_BACKEND_HOME}/runtime/llama-server" "${COMPASSLM_HOME}/runtime/llama-server")
  local candidate

  if compass_allow_windows_exe_runtime; then
    candidates+=("${MAIN_BACKEND_HOME}/runtime/llama-server.exe" "${COMPASSLM_HOME}/runtime/llama-server.exe")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done

  if command -v llama-server >/dev/null 2>&1; then
    command -v llama-server
    return
  fi
  if compass_allow_windows_exe_runtime && command -v llama-server.exe >/dev/null 2>&1; then
    command -v llama-server.exe
    return
  fi

  echo "${MAIN_BACKEND_HOME}/runtime/llama-server"
}

EMBEDDING_MODEL_LARGE_PATH_RAW="${EMBEDDING_MODEL_LARGE_PATH:-$(compass_detect_embedding_model_path)}"
EMBEDDING_MODEL_LARGE_PATH_RAW="$(compass_prefer_existing_or_rebased_path "${EMBEDDING_MODEL_LARGE_PATH_RAW}")"
export EMBEDDING_MODEL_LARGE_PATH="$(compass_resolve_embedding_model_path "${EMBEDDING_MODEL_LARGE_PATH_RAW}")"

LLM_MODELS_DIR_RAW="${LLM_MODELS_DIR:-${MAIN_BACKEND_HOME}/models/llm}"
LLM_MODELS_DIR_RAW="$(compass_prefer_existing_or_rebased_path "${LLM_MODELS_DIR_RAW}")"
export LLM_MODELS_DIR="${LLM_MODELS_DIR_RAW}"

LLM_RUNTIME_RAW="${LLM_RUNTIME:-$(compass_detect_llm_runtime)}"
LLM_RUNTIME_RAW="$(compass_prefer_existing_or_rebased_path "${LLM_RUNTIME_RAW}")"
if [[ ! -e "${LLM_RUNTIME_RAW}" ]]; then
  LLM_RUNTIME_RAW="$(compass_detect_llm_runtime)"
fi
export LLM_RUNTIME="${LLM_RUNTIME_RAW}"

LLM_MODEL_PATH_RAW="${LLM_MODEL_PATH:-$(compass_detect_llm_model_path)}"
LLM_MODEL_PATH_RAW="$(compass_prefer_existing_or_rebased_path "${LLM_MODEL_PATH_RAW}")"
if [[ ! -f "${LLM_MODEL_PATH_RAW}" ]]; then
  LLM_MODEL_PATH_RAW="$(compass_detect_llm_model_path)"
fi
export LLM_MODEL_PATH="${LLM_MODEL_PATH_RAW}"

if [[ "${1:-}" == "--print" ]]; then
  echo "COMPASSLM_HOME=${COMPASSLM_HOME}"
  echo "PROJECT_GPU_HOME=${PROJECT_GPU_HOME}"
  echo "EMBEDDING_SERVER_HOME=${EMBEDDING_SERVER_HOME}"
  echo "MAIN_BACKEND_HOME=${MAIN_BACKEND_HOME}"
  echo "LLM_MODELS_DIR=${LLM_MODELS_DIR}"
  echo "EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_MODEL_LARGE_PATH}"
  echo "LLM_MODEL_PATH=${LLM_MODEL_PATH}"
  echo "LLM_RUNTIME=${LLM_RUNTIME}"
fi
