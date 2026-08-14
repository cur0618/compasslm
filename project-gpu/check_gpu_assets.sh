#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

compass_load_env_file "${EMBEDDING_SERVER_HOME}/.env.auto"
compass_load_env_file "${MAIN_BACKEND_HOME}/.env.auto"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
compass_load_env_file "${EMBEDDING_SERVER_HOME}/.env"
compass_load_env_file "${MAIN_BACKEND_HOME}/.env"

LLM_RUNTIME="${LLM_RUNTIME:-$(compass_detect_llm_runtime)}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-$(compass_detect_llm_model_path)}"
OFFLINE_DIR="${OFFLINE_PACKAGE_DIR:-${EMBEDDING_SERVER_HOME}/offline_packages/cp311-linux_x86_64}"
PREFERRED_PADDLE_RUNTIME="${BACKEND_PADDLE_RUNTIME_KIND:-}"
if [[ -z "${PREFERRED_PADDLE_RUNTIME}" ]]; then
  case "${PDF_OCR_DEVICE:-cpu}" in
    gpu*|cuda*)
      PREFERRED_PADDLE_RUNTIME="gpu"
      ;;
    *)
      PREFERRED_PADDLE_RUNTIME="cpu"
      ;;
  esac
fi
PREFERRED_PADDLE_CUDA_TRACK="${BACKEND_PADDLE_CUDA_TRACK:-${PADDLE_GPU_CUDA_TRACK:-cu126}}"

if [[ -n "${EMBEDDING_MODEL_LARGE_PATH:-}" ]]; then
  model_path_expanded="${EMBEDDING_MODEL_LARGE_PATH/#\~/${HOME}}"
  if [[ ! -e "${model_path_expanded}" ]]; then
    echo "[WARN] EMBEDDING_MODEL_LARGE_PATH not found, fallback to local project path: ${EMBEDDING_MODEL_LARGE_PATH}"
    unset EMBEDDING_MODEL_LARGE_PATH
  fi
fi
EMBEDDING_MODEL_LARGE_PATH="$(compass_resolve_embedding_model_path "${EMBEDDING_MODEL_LARGE_PATH:-}")"

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

ok_or_missing() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    echo "[OK]   ${label}: ${path}"
  else
    echo "[MISS] ${label}: ${path}"
  fi
}

ok_or_missing_glob() {
  local pattern="$1"
  local label="$2"
  local match=""

  match="$(compgen -G "${pattern}" | head -n 1 || true)"
  if [[ -n "${match}" ]]; then
    echo "[OK]   ${label}: ${match}"
  else
    echo "[MISS] ${label}: ${pattern}"
  fi
}

ok_or_missing_any_glob() {
  local label="$1"
  shift
  local pattern=""
  for pattern in "$@"; do
    local match=""
    match="$(compgen -G "${pattern}" | head -n 1 || true)"
    if [[ -n "${match}" ]]; then
      echo "[OK]   ${label}: ${match}"
      return 0
    fi
  done
  echo "[MISS] ${label}: $*"
}

embedding_model_ready() {
  local model_path="$1"
  if [[ ! -e "${model_path}" ]]; then
    return 1
  fi

  if [[ ! -d "${model_path}" ]]; then
    return 0
  fi

  compass_model_marker_exists "${model_path}"
}

runtime_ready() {
  local runtime_path="$1"
  local runtime_dir lib_path
  local require_cuda="${LLM_REQUIRE_CUDA_RUNTIME:-1}"
  if [[ ! -e "${runtime_path}" ]]; then
    return 1
  fi
  if [[ "${runtime_path}" == *.exe ]] && ! compass_allow_windows_exe_runtime; then
    return 2
  fi
  if [[ ! -x "${runtime_path}" ]]; then
    return 3
  fi
  runtime_dir="$(cd "$(dirname "${runtime_path}")" && pwd)"
  for lib_path in "${runtime_dir}/libggml.so.0" "${runtime_dir}/libggml-base.so.0" "${runtime_dir}/libllama.so.0" "${runtime_dir}/libmtmd.so.0"; do
    if [[ -L "${lib_path}" && ! -e "${lib_path}" ]]; then
      return 4
    fi
    if [[ -e "${lib_path}" && ! -s "${lib_path}" ]]; then
      return 4
    fi
  done
  if [[ "${require_cuda}" == "1" ]]; then
    lib_path="${runtime_dir}/libggml-cuda.so"
    if [[ ! -e "${lib_path}" ]]; then
      return 5
    fi
    if [[ ! -s "${lib_path}" ]]; then
      return 5
    fi
  fi
  return 0
}

echo "=== project-gpu asset check ==="
ok_or_missing "${EMBEDDING_SERVER_HOME}/compassvenv/bin/python" "Embedding venv python"
ok_or_missing "${MAIN_BACKEND_HOME}/compassvenv/bin/python" "Backend venv python"
ok_or_missing "${MAIN_BACKEND_HOME}/runtime/ccache" "llama.cpp build ccache"
if embedding_model_ready "${EMBEDDING_MODEL_LARGE_PATH}"; then
  echo "[OK]   Embedding model files: ${EMBEDDING_MODEL_LARGE_PATH}"
else
  echo "[MISS] Embedding model files: ${EMBEDDING_MODEL_LARGE_PATH}"
fi
if runtime_ready "${LLM_RUNTIME}"; then
  echo "[OK]   llama-server runtime: ${LLM_RUNTIME}"
  if [[ "${LLM_RUNTIME}" == *.exe ]] && compass_is_wsl; then
    echo "[WARN] llama-server runtime is Windows .exe on WSL; Linux GPU host needs native llama-server binary."
  fi
else
  runtime_code=$?
  if [[ "${runtime_code}" == "2" ]]; then
    echo "[MISS] llama-server runtime: ${LLM_RUNTIME} (Windows .exe cannot run on this Linux host)"
  elif [[ "${runtime_code}" == "3" ]]; then
    echo "[MISS] llama-server runtime (not executable): ${LLM_RUNTIME}"
  elif [[ "${runtime_code}" == "4" ]]; then
    echo "[MISS] llama-server runtime libraries look broken (empty or broken symlink): ${LLM_RUNTIME}"
  elif [[ "${runtime_code}" == "5" ]]; then
    echo "[MISS] llama-server CUDA backend library missing: $(dirname "${LLM_RUNTIME}")/libggml-cuda.so"
    echo "       Rebuild with: project-gpu/build_llama_runtime_offline.sh --cuda on --cuda-arch 90 --keep-build"
  else
    echo "[MISS] llama-server runtime: ${LLM_RUNTIME}"
  fi
fi
ok_or_missing "${LLM_MODEL_PATH}" "LLM model file"
echo
echo "Offline bundle:"
ok_or_missing "${OFFLINE_DIR}" "Offline package dir"
if [[ -d "${OFFLINE_DIR}" ]]; then
  ok_or_missing_glob "${OFFLINE_DIR}/requirements.embedding.txt" "Offline embedding manifest"
  ok_or_missing_glob "${OFFLINE_DIR}/requirements.backend.txt" "Offline backend manifest"
  ok_or_missing_glob "${OFFLINE_DIR}/requirements.merged.txt" "Offline merged manifest"
  ok_or_missing_glob "${OFFLINE_DIR}/pydantic_ai_slim-*.whl" "PydanticAI wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/pydantic_graph-*.whl" "Pydantic graph wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/openai-*.whl" "OpenAI client wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/paddleocr-*.whl" "PaddleOCR wheel"
  if [[ "${PREFERRED_PADDLE_RUNTIME}" == "gpu" ]]; then
    ok_or_missing_any_glob "PaddlePaddle GPU runtime" "${OFFLINE_DIR}/paddlepaddle_gpu-*.whl" "${OFFLINE_DIR}/paddlepaddle-gpu-*.whl"
    if [[ "${PREFERRED_PADDLE_CUDA_TRACK}" == "cu126" || "${PREFERRED_PADDLE_CUDA_TRACK}" == cu12* ]]; then
      ok_or_missing_glob "${OFFLINE_DIR}/paddle_cuda_track.txt" "Paddle CUDA track marker"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_nvrtc_cu12-12.6.77-*.whl" "Paddle CUDA 12 nvrtc wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_runtime_cu12-12.6.77-*.whl" "Paddle CUDA 12 runtime wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cudnn_cu12-9.5.1.17-*.whl" "Paddle CUDA 12 cudnn wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cublas_cu12-12.6.4.1-*.whl" "Paddle CUDA 12 cublas wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cufft_cu12-11.3.0.4-*.whl" "Paddle CUDA 12 cufft wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_curand_cu12-10.3.7.77-*.whl" "Paddle CUDA 12 curand wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusolver_cu12-11.7.1.2-*.whl" "Paddle CUDA 12 cusolver wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusparse_cu12-12.5.4.2-*.whl" "Paddle CUDA 12 cusparse wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusparselt_cu12-0.6.3-*.whl" "Paddle CUDA 12 cusparselt wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nccl_cu12-2.25.1-*.whl" "Paddle CUDA 12 nccl wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nvtx_cu12-12.6.77-*.whl" "Paddle CUDA 12 nvtx wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nvjitlink_cu12-12.6.85-*.whl" "Paddle CUDA 12 nvjitlink wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cufile_cu12-1.11.1.6-*.whl" "Paddle CUDA 12 cufile wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_cccl_cu12-12.6.77-*.whl" "Paddle CUDA 12 cccl wheel"
    else
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_nvrtc_cu11-11.8.89-*.whl" "Paddle CUDA 11 nvrtc wheel"
      ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusolver_cu11-11.4.1.48-*.whl" "Paddle CUDA 11 cusolver wheel"
    fi
  else
    ok_or_missing_any_glob "PaddlePaddle runtime" "${OFFLINE_DIR}/paddlepaddle_gpu-*.whl" "${OFFLINE_DIR}/paddlepaddle-gpu-*.whl" "${OFFLINE_DIR}/paddlepaddle-*.whl"
  fi
  ok_or_missing_any_glob "PyMuPDF wheel" "${OFFLINE_DIR}/PyMuPDF-*.whl" "${OFFLINE_DIR}/pymupdf-*.whl"
  ok_or_missing "${OFFLINE_DIR}/pymupdf_sitepkg/fitz/__init__.py" "PyMuPDF extracted fitz fallback"
  ok_or_missing "${OFFLINE_DIR}/pymupdf_sitepkg/pymupdf/_mupdf.so" "PyMuPDF extracted native fallback"
  ok_or_missing "${OFFLINE_DIR}/README_PyMuPDF_fallback.txt" "PyMuPDF fallback note"
  ok_or_missing_glob "${OFFLINE_DIR}/python_hwpx-*.whl" "python-hwpx HWPX parser wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/torch-*.whl" "PyTorch wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_nvrtc_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvrtc wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_runtime_cu12-12.4.127-*.whl" "PyTorch CUDA 12 runtime wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cuda_cupti_cu12-12.4.127-*.whl" "PyTorch CUDA 12 cupti wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cudnn_cu12-9.1.0.70-*.whl" "PyTorch CUDA 12 cudnn wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cublas_cu12-12.4.5.8-*.whl" "PyTorch CUDA 12 cublas wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cufft_cu12-11.2.1.3-*.whl" "PyTorch CUDA 12 cufft wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_curand_cu12-10.3.5.147-*.whl" "PyTorch CUDA 12 curand wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusolver_cu12-11.6.1.9-*.whl" "PyTorch CUDA 12 cusolver wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusparse_cu12-12.3.1.170-*.whl" "PyTorch CUDA 12 cusparse wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_cusparselt_cu12-0.6.2-*.whl" "PyTorch CUDA 12 cusparselt wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nccl_cu12-2.21.5-*.whl" "PyTorch CUDA 12 nccl wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nvtx_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvtx wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/nvidia_nvjitlink_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvjitlink wheel"
  ok_or_missing_glob "${OFFLINE_DIR}/hnswlib-*.tar.gz" "hnswlib source package"
fi

echo
echo "Resolved:"
echo "  COMPASSLM_HOME=${COMPASSLM_HOME}"
echo "  EMBEDDING_SERVER_HOME=${EMBEDDING_SERVER_HOME}"
echo "  MAIN_BACKEND_HOME=${MAIN_BACKEND_HOME}"
echo "  EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_MODEL_LARGE_PATH}"
echo "  LLM_RUNTIME=${LLM_RUNTIME}"
echo "  LLM_MODEL_PATH=${LLM_MODEL_PATH}"
echo "  OFFLINE_DIR=${OFFLINE_DIR}"
echo "  BACKEND_PADDLE_CUDA_TRACK=${PREFERRED_PADDLE_CUDA_TRACK}"
