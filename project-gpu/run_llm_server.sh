#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

PORT_OVERRIDE_SET=0
AUTO_PORT_OVERRIDE_SET=0
[[ -n "${LLM_PORT+x}" ]] && PORT_OVERRIDE_SET=1 && LLM_PORT_OVERRIDE="${LLM_PORT}"
[[ -n "${COMPASS_AUTO_PORT+x}" ]] && AUTO_PORT_OVERRIDE_SET=1 && AUTO_PORT_OVERRIDE="${COMPASS_AUTO_PORT}"

compass_load_env_file "${MAIN_BACKEND_HOME}/.env.auto"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
compass_load_env_file "${MAIN_BACKEND_HOME}/.env"

[[ "${PORT_OVERRIDE_SET}" == "1" ]] && LLM_PORT="${LLM_PORT_OVERRIDE}"
[[ "${AUTO_PORT_OVERRIDE_SET}" == "1" ]] && COMPASS_AUTO_PORT="${AUTO_PORT_OVERRIDE}"

LLM_PORT_START="${LLM_PORT_START:-${LLM_PORT:-8003}}"
LLM_PORT="$(compass_select_service_port llm "${LLM_PORT_START}" "${LLM_PORT:-${LLM_PORT_START}}")"
LLM_API_URL_SELECTED="http://127.0.0.1:${LLM_PORT}/v1/chat/completions"
LLM_HOST="${LLM_HOST:-0.0.0.0}"
LLM_API_PREFIX="${LLM_API_PREFIX:-}"
LLM_WEBUI="${LLM_WEBUI:-1}"
LLM_CONTEXT_LIMIT="${LLM_CONTEXT_LIMIT:-131072}"
LLM_CTX_SIZE="${LLM_CTX_SIZE:-${LLM_CONTEXT_LIMIT}}"
LLM_RUNTIME="${LLM_RUNTIME:-$(compass_detect_llm_runtime)}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-$(compass_detect_llm_model_path)}"

if [[ -n "${LLM_RUNTIME:-}" ]] && [[ ! -e "${LLM_RUNTIME/#\~/${HOME}}" ]]; then
  echo "[WARN] LLM_RUNTIME not found, fallback to local project runtime: ${LLM_RUNTIME}"
  LLM_RUNTIME="$(compass_detect_llm_runtime)"
fi

if [[ -n "${LLM_MODELS_DIR:-}" ]] && [[ ! -d "${LLM_MODELS_DIR/#\~/${HOME}}" ]]; then
  echo "[WARN] LLM_MODELS_DIR not found, fallback to local project models dir: ${LLM_MODELS_DIR}"
  LLM_MODELS_DIR="${MAIN_BACKEND_HOME}/models/llm"
fi

if [[ -n "${LLM_MODEL_PATH:-}" ]] && [[ ! -f "${LLM_MODEL_PATH/#\~/${HOME}}" ]]; then
  echo "[WARN] LLM_MODEL_PATH not found, fallback to local project model: ${LLM_MODEL_PATH}"
  LLM_MODEL_PATH="$(compass_detect_llm_model_path)"
fi

LLM_RUNTIME_DIR="${MAIN_BACKEND_HOME}/runtime"
if [[ "${LLM_RUNTIME}" == */* ]]; then
  LLM_RUNTIME_DIR="$(cd "$(dirname "${LLM_RUNTIME}")" && pwd)"
fi
BUNDLED_LIB_DIR="${LLM_RUNTIME_DIR}/lib"
OPENSSL_LIB_DIR="${LLM_RUNTIME_DIR}/openssl-lib"

latest_lib_basename() {
  local pattern="$1"
  find "${LLM_RUNTIME_DIR}" -maxdepth 1 -type f -name "${pattern}" | sort | tail -n 1 | xargs -r basename
}

repair_runtime_link() {
  local link_name="$1"
  local target_name="$2"
  local link_path="${LLM_RUNTIME_DIR}/${link_name}"
  local target_path="${LLM_RUNTIME_DIR}/${target_name}"

  if [[ ! -f "${target_path}" ]]; then
    return 1
  fi
  if [[ ! -s "${target_path}" ]]; then
    echo "[ERROR] Runtime library is empty/corrupted: ${target_path}" >&2
    return 1
  fi

  if [[ -L "${link_path}" ]]; then
    local cur_target
    cur_target="$(readlink "${link_path}")"
    if [[ "${cur_target}" == "${target_name}" ]]; then
      return 0
    fi
  fi

  rm -f "${link_path}"
  ln -s "${target_name}" "${link_path}"
  return 0
}

repair_runtime_links() {
  local ggml_ver llm_ver mtmd_ver base_ver

  ggml_ver="$(latest_lib_basename 'libggml.so.0.*')"
  llm_ver="$(latest_lib_basename 'libllama.so.0.0.*')"
  mtmd_ver="$(latest_lib_basename 'libmtmd.so.0.0.*')"
  base_ver="$(latest_lib_basename 'libggml-base.so.0.*')"

  [[ -n "${ggml_ver}" ]] && repair_runtime_link "libggml.so.0" "${ggml_ver}" || true
  [[ -n "${ggml_ver}" ]] && repair_runtime_link "libggml.so" "libggml.so.0" || true
  [[ -n "${base_ver}" ]] && repair_runtime_link "libggml-base.so.0" "${base_ver}" || true
  [[ -n "${base_ver}" ]] && repair_runtime_link "libggml-base.so" "libggml-base.so.0" || true
  [[ -n "${llm_ver}" ]] && repair_runtime_link "libllama.so.0" "${llm_ver}" || true
  [[ -n "${llm_ver}" ]] && repair_runtime_link "libllama.so" "libllama.so.0" || true
  [[ -n "${mtmd_ver}" ]] && repair_runtime_link "libmtmd.so.0" "${mtmd_ver}" || true
  [[ -n "${mtmd_ver}" ]] && repair_runtime_link "libmtmd.so" "libmtmd.so.0" || true
}

repair_runtime_links

prepare_openssl_bundle() {
  local lib_name src dst
  [[ -d "${BUNDLED_LIB_DIR}" ]] || return 0
  mkdir -p "${OPENSSL_LIB_DIR}"
  for lib_name in libssl.so.3 libcrypto.so.3; do
    src="${BUNDLED_LIB_DIR}/${lib_name}"
    dst="${OPENSSL_LIB_DIR}/${lib_name}"
    if [[ -f "${src}" && -s "${src}" ]]; then
      cp -f "${src}" "${dst}"
      chmod 755 "${dst}"
    fi
  done
}

prepare_openssl_bundle

sanitize_ld_library_path() {
  local raw="${1:-}"
  local cleaned=""
  local part
  local old_ifs="${IFS}"
  IFS=':'
  for part in ${raw}; do
    [[ -z "${part}" ]] && continue
    [[ "${part}" == "${BUNDLED_LIB_DIR}" ]] && continue
    if [[ -z "${cleaned}" ]]; then
      cleaned="${part}"
    else
      cleaned="${cleaned}:${part}"
    fi
  done
  IFS="${old_ifs}"
  echo "${cleaned}"
}

lib_path_prefix="${LLM_RUNTIME_DIR}"
if [[ -f "${OPENSSL_LIB_DIR}/libssl.so.3" && -f "${OPENSSL_LIB_DIR}/libcrypto.so.3" ]]; then
  lib_path_prefix="${OPENSSL_LIB_DIR}:${lib_path_prefix}"
elif [[ "${USE_FULL_BUNDLED_LIB_DIR:-0}" == "1" && -d "${BUNDLED_LIB_DIR}" ]]; then
  # Fallback only when explicitly requested; runtime/lib may also contain stale toolchain libs.
  lib_path_prefix="${BUNDLED_LIB_DIR}:${lib_path_prefix}"
fi

sanitized_existing_ld="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"
if [[ -n "${sanitized_existing_ld}" ]]; then
  export LD_LIBRARY_PATH="${lib_path_prefix}:${sanitized_existing_ld}"
else
  export LD_LIBRARY_PATH="${lib_path_prefix}"
fi

if [[ "${LLM_RUNTIME}" == *.exe ]] && ! compass_allow_windows_exe_runtime; then
  echo "[ERROR] Windows runtime (.exe) is not supported on this Linux host: ${LLM_RUNTIME}" >&2
  echo "        Place a Linux llama-server binary in ${MAIN_BACKEND_HOME}/runtime or ${COMPASSLM_HOME}/runtime." >&2
  exit 1
fi

if [[ ! -e "${LLM_RUNTIME}" ]]; then
  echo "[ERROR] llama-server runtime not found: ${LLM_RUNTIME}" >&2
  echo "        Put llama-server binary in ${MAIN_BACKEND_HOME}/runtime or ${COMPASSLM_HOME}/runtime." >&2
  exit 1
fi

if [[ ! -x "${LLM_RUNTIME}" ]]; then
  echo "[ERROR] llama-server is not executable: ${LLM_RUNTIME}" >&2
  echo "        Run: chmod +x \"${LLM_RUNTIME}\"" >&2
  exit 1
fi

if [[ ! -f "${LLM_MODEL_PATH}" ]]; then
  echo "[ERROR] LLM model file not found: ${LLM_MODEL_PATH}" >&2
  echo "        Search dir: ${LLM_MODELS_DIR}" >&2
  echo "        Example filename: qwen3.5-9b-q4_k_m.gguf or qwen2.5-14b-instruct-q4_k_m.gguf (Linux is case-sensitive)." >&2
  echo "        If this PC intentionally removed large files, copy model/runtime on target GPU server first." >&2
  exit 1
fi

if [[ "${LLM_CTX_SIZE}" != "${LLM_CONTEXT_LIMIT}" ]]; then
  echo "[WARN] LLM_CTX_SIZE (${LLM_CTX_SIZE}) != LLM_CONTEXT_LIMIT (${LLM_CONTEXT_LIMIT})." >&2
  echo "       Use the same value for llama.cpp and backend prompt budgeting during long-context validation." >&2
fi

echo "[INFO] LLM_RUNTIME=${LLM_RUNTIME}"
echo "[INFO] LLM_MODEL_PATH=${LLM_MODEL_PATH}"
echo "[INFO] LLM_HOST=${LLM_HOST} LLM_PORT=${LLM_PORT} LLM_CTX_SIZE=${LLM_CTX_SIZE} LLM_CONTEXT_LIMIT=${LLM_CONTEXT_LIMIT}"
echo "[PORT] LLM_API_URL=${LLM_API_URL_SELECTED}"
echo "[INFO] LLM_API_PREFIX=${LLM_API_PREFIX:-<none>} LLM_WEBUI=${LLM_WEBUI}"

cd "${MAIN_BACKEND_HOME}"
llama_args=(
  -m "${LLM_MODEL_PATH}"
  --host "${LLM_HOST}"
  --port "${LLM_PORT}"
  --ctx-size "${LLM_CTX_SIZE}"
)

if [[ -n "${LLM_API_PREFIX}" ]]; then
  llama_args+=(--api-prefix "${LLM_API_PREFIX}")
fi

case "${LLM_WEBUI,,}" in
  0|false|no|off)
    llama_args+=(--no-webui)
    ;;
  *)
    llama_args+=(--webui)
    ;;
esac

if [[ -n "${LLM_API_KEY:-}" ]]; then
  llama_args+=(--api-key "${LLM_API_KEY}")
fi

"${LLM_RUNTIME}" "${llama_args[@]}" &
LLM_SERVER_PID="$!"
trap 'kill "${LLM_SERVER_PID}" 2>/dev/null || true' INT TERM EXIT
if ! compass_wait_for_llm_ready "${LLM_API_URL_SELECTED}" "${LLM_SERVER_PID}"; then
  kill "${LLM_SERVER_PID}" 2>/dev/null || true
  wait "${LLM_SERVER_PID}" 2>/dev/null || true
  exit 1
fi
compass_write_service_state llm "${LLM_PORT}" "${LLM_API_URL_SELECTED}" "${LLM_SERVER_PID}"
wait "${LLM_SERVER_PID}"
