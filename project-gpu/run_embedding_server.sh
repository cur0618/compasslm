#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

PORT_OVERRIDE_SET=0
AUTO_PORT_OVERRIDE_SET=0
[[ -n "${EMBED_PORT+x}" ]] && PORT_OVERRIDE_SET=1 && EMBED_PORT_OVERRIDE="${EMBED_PORT}"
[[ -n "${COMPASS_AUTO_PORT+x}" ]] && AUTO_PORT_OVERRIDE_SET=1 && AUTO_PORT_OVERRIDE="${COMPASS_AUTO_PORT}"

compass_load_env_file "${EMBEDDING_SERVER_HOME}/.env.auto"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
compass_load_env_file "${EMBEDDING_SERVER_HOME}/.env"

[[ "${PORT_OVERRIDE_SET}" == "1" ]] && EMBED_PORT="${EMBED_PORT_OVERRIDE}"
[[ "${AUTO_PORT_OVERRIDE_SET}" == "1" ]] && COMPASS_AUTO_PORT="${AUTO_PORT_OVERRIDE}"

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
echo "[PORT] EMBEDDING_API_URL=${EMBEDDING_API_URL_SELECTED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${EMBEDDING_SERVER_HOME}"
source "${VENV_PATH}/bin/activate"

python -m uvicorn src.main:app --host "${EMBED_HOST}" --port "${EMBED_PORT}" &
EMBEDDING_SERVER_PID="$!"
trap 'kill "${EMBEDDING_SERVER_PID}" 2>/dev/null || true' INT TERM EXIT
if ! compass_wait_for_embedding_ready "${EMBEDDING_API_URL_SELECTED}" "${EMBEDDING_SERVER_PID}"; then
  kill "${EMBEDDING_SERVER_PID}" 2>/dev/null || true
  wait "${EMBEDDING_SERVER_PID}" 2>/dev/null || true
  exit 1
fi
compass_write_service_state embedding "${EMBED_PORT}" "${EMBEDDING_API_URL_SELECTED}" "${EMBEDDING_SERVER_PID}"
wait "${EMBEDDING_SERVER_PID}"
