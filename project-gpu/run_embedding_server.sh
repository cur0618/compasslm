#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

compass_load_service_env_files embedding

EMBED_HOST="${EMBED_HOST:-0.0.0.0}"
EMBED_PORT_START="${EMBED_PORT_START:-${EMBED_PORT:-8002}}"
EMBED_PORT="$(compass_select_service_port embedding "${EMBED_PORT_START}" "${EMBED_PORT:-${EMBED_PORT_START}}")"
EMBEDDING_API_URL_SELECTED="http://127.0.0.1:${EMBED_PORT}"
VENV_PATH="${EMBEDDING_SERVER_HOME}/compassvenv"
requested_model_path="${EMBEDDING_MODEL_LARGE_PATH:-}"

if [[ -n "${EMBEDDING_MODEL_LARGE_PATH:-}" ]]; then
  # If .env has a stale absolute path from another machine, fall back to local project path.
  model_path_expanded="${EMBEDDING_MODEL_LARGE_PATH/#\~/${HOME}}"
  if [[ ! -e "${model_path_expanded}" ]]; then
    echo "[WARN] EMBEDDING_MODEL_LARGE_PATH not found, fallback to local project path: ${EMBEDDING_MODEL_LARGE_PATH}"
    unset EMBEDDING_MODEL_LARGE_PATH
  fi
fi
EMBEDDING_MODEL_LARGE_PATH="$(compass_resolve_embedding_model_path "${EMBEDDING_MODEL_LARGE_PATH:-}")"
if [[ -n "${requested_model_path}" && "${requested_model_path}" != "${EMBEDDING_MODEL_LARGE_PATH}" ]]; then
  echo "[INFO] EMBEDDING_MODEL_LARGE_PATH auto-resolved: ${requested_model_path} -> ${EMBEDDING_MODEL_LARGE_PATH}"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Embedding venv not found: ${VENV_PATH}" >&2
  echo "Run: ${PROJECT_GPU_HOME}/setup_gpu_track.sh" >&2
  exit 1
fi

if [[ "${EMBEDDING_MODEL_LARGE_PATH}" == /* || "${EMBEDDING_MODEL_LARGE_PATH}" == ./* || "${EMBEDDING_MODEL_LARGE_PATH}" == ../* || "${EMBEDDING_MODEL_LARGE_PATH}" == ~* ]]; then
  if [[ ! -e "${EMBEDDING_MODEL_LARGE_PATH}" ]]; then
    echo "[ERROR] Embedding model path not found: ${EMBEDDING_MODEL_LARGE_PATH}" >&2
    echo "        Copy embedding model under ${EMBEDDING_SERVER_HOME} and update EMBEDDING_MODEL_LARGE_PATH." >&2
    exit 1
  fi

  if [[ -d "${EMBEDDING_MODEL_LARGE_PATH}" ]] && ! compass_model_marker_exists "${EMBEDDING_MODEL_LARGE_PATH}"; then
    echo "[ERROR] No embedding model files found under: ${EMBEDDING_MODEL_LARGE_PATH}" >&2
    echo "        Expect sentence-transformers files (modules.json/config.json/model.safetensors)." >&2
    echo "        If large files were removed on this PC, copy them on the target GPU server before run." >&2
    exit 1
  fi
fi

echo "[INFO] EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_MODEL_LARGE_PATH}"
echo "[INFO] EMBED_HOST=${EMBED_HOST} EMBED_PORT=${EMBED_PORT}"
echo "[INFO] EMBED_MODEL_DTYPE=${EMBED_MODEL_DTYPE:-bf16}"
echo "[INFO] EMBED_MAX_QUERY_TOKENS=${EMBED_MAX_QUERY_TOKENS:-384}"
echo "[INFO] EMBED_MAX_PASSAGE_TOKENS=${EMBED_MAX_PASSAGE_TOKENS:-768}"
echo "[INFO] EMBED_MAX_BATCH_TOKENS=${EMBED_MAX_BATCH_TOKENS:-8192}"
echo "[INFO] EMBED_LENGTH_BUCKETING=${EMBED_LENGTH_BUCKETING:-1}"
echo "[PORT] EMBEDDING_API_URL=${EMBEDDING_API_URL_SELECTED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${EMBEDDING_SERVER_HOME}"
source "${VENV_PATH}/bin/activate"

python -m uvicorn src.main:app --host "${EMBED_HOST}" --port "${EMBED_PORT}" &
EMBEDDING_SERVER_PID="$!"
EMBEDDING_SHUTDOWN_REQUESTED=0

compass_embedding_shutdown() {
  local exit_code="${1:-0}"
  local reason="${2:-script_exit}"
  local signal_name="${3:-TERM}"
  if [[ "${EMBEDDING_SHUTDOWN_REQUESTED}" == "1" ]]; then
    exit "${exit_code}"
  fi
  EMBEDDING_SHUTDOWN_REQUESTED=1
  trap - INT TERM EXIT
  compass_shutdown_child embedding "${EMBEDDING_SERVER_PID:-}" "${reason}" "${signal_name}"
  exit "${exit_code}"
}

trap 'compass_embedding_shutdown 130 ctrl_c INT' INT
trap 'compass_embedding_shutdown 143 termination TERM' TERM
trap 'compass_embedding_shutdown 0 script_exit TERM' EXIT
if ! compass_wait_for_embedding_ready "${EMBEDDING_API_URL_SELECTED}" "${EMBEDDING_SERVER_PID}"; then
  compass_shutdown_child embedding "${EMBEDDING_SERVER_PID}" startup_failed TERM
  exit 1
fi
compass_write_service_state embedding "${EMBED_PORT}" "${EMBEDDING_API_URL_SELECTED}" "${EMBEDDING_SERVER_PID}"
set +e
wait "${EMBEDDING_SERVER_PID}"
EMBEDDING_STATUS="$?"
set -e
trap - EXIT
exit "${EMBEDDING_STATUS}"
