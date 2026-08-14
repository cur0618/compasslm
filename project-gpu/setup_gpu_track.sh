#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RECREATE=0
OFFLINE_EMBED=0
OFFLINE_BACKEND=0
OFFLINE_DIR=""
SKIP_BACKEND=0
PYMUPDF_FALLBACK_DIR_NAME="pymupdf_sitepkg"

usage() {
  cat <<'EOF'
Usage:
  ./project-gpu/setup_gpu_track.sh [options]

Options:
  --python <cmd>      Python command (default: python3)
  --offline-embed     Install embedding server deps from offline_packages/cp311-linux_x86_64
  --offline-backend   Install main-backend deps from offline package dir
  --offline-dir <dir> Offline package dir override (default: embedding-gpu-server/offline_packages/cp311-linux_x86_64)
  --recreate          Recreate venvs
  --skip-backend      Skip main-backend venv install
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --offline-embed)
      OFFLINE_EMBED=1
      shift
      ;;
    --offline-backend)
      OFFLINE_BACKEND=1
      shift
      ;;
    --offline-dir)
      OFFLINE_DIR="${2:-}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-backend)
      SKIP_BACKEND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python command not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${OFFLINE_EMBED}" == "1" || "${OFFLINE_BACKEND}" == "1" ]]; then
  OFFLINE_DIR="${OFFLINE_DIR:-${EMBEDDING_SERVER_HOME}/offline_packages/cp311-linux_x86_64}"
fi

create_or_update_venv() {
  local venv_path="$1"
  if [[ "${RECREATE}" == "1" && -d "${venv_path}" ]]; then
    rm -rf "${venv_path}"
  fi
  if [[ ! -d "${venv_path}" ]]; then
    "${PYTHON_BIN}" -m venv "${venv_path}"
  fi
  "${venv_path}/bin/python" -m ensurepip --upgrade
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${venv_path}/bin/python" -m pip install --upgrade pip
}

require_offline_artifact() {
  local dir="$1"
  local pattern="$2"
  local name="$3"
  if ! compgen -G "${dir}/${pattern}" > /dev/null; then
    echo "[ERROR] Missing offline package (${name}): ${pattern}" >&2
    echo "        Rebuild bundle: ${EMBEDDING_SERVER_HOME}/prepare_offline_packages_linux.sh" >&2
    exit 1
  fi
}

require_any_offline_artifact() {
  local dir="$1"
  local name="$2"
  shift 2
  local pattern=""
  for pattern in "$@"; do
    if compgen -G "${dir}/${pattern}" > /dev/null; then
      return 0
    fi
  done
  echo "[ERROR] Missing offline package (${name}): one of $*" >&2
  echo "        Rebuild bundle or place the matching wheel into ${dir}" >&2
  exit 1
}

resolve_offline_requirements_file() {
  local default_file="$1"
  local offline_dir="$2"
  local manifest_name="$3"

  if [[ -n "${offline_dir}" && -f "${offline_dir}/${manifest_name}" ]]; then
    printf '%s\n' "${offline_dir}/${manifest_name}"
    return 0
  fi
  printf '%s\n' "${default_file}"
}

resolve_backend_paddle_runtime_kind() {
  local preferred_runtime="${BACKEND_PADDLE_RUNTIME_KIND:-}"
  if [[ -n "${preferred_runtime}" ]]; then
    printf '%s\n' "${preferred_runtime}"
    return 0
  fi
  case "${PDF_OCR_DEVICE:-cpu}" in
    gpu*|cuda*)
      printf '%s\n' "gpu"
      ;;
    *)
      printf '%s\n' "cpu"
      ;;
  esac
}

resolve_backend_paddle_cuda_track() {
  local explicit_track="${BACKEND_PADDLE_CUDA_TRACK:-${PADDLE_GPU_CUDA_TRACK:-}}"
  if [[ -n "${explicit_track}" ]]; then
    printf '%s\n' "${explicit_track}"
    return 0
  fi
  if [[ "$(resolve_backend_paddle_runtime_kind)" == "gpu" ]]; then
    printf '%s\n' "cu126"
    return 0
  fi
  printf '%s\n' ""
}

select_backend_paddle_wheel() {
  local dir="$1"
  local preferred_runtime=""
  local gpu_wheel=""
  local cpu_wheel=""

  preferred_runtime="$(resolve_backend_paddle_runtime_kind)"

  gpu_wheel="$(find "${dir}" -maxdepth 1 -type f \( -name 'paddlepaddle_gpu-*.whl' -o -name 'paddlepaddle-gpu-*.whl' \) | sort | tail -n 1)"
  cpu_wheel="$(find "${dir}" -maxdepth 1 -type f -name 'paddlepaddle-*.whl' | sort | tail -n 1)"

  if [[ "${preferred_runtime}" == "cpu" ]]; then
    if [[ -n "${cpu_wheel}" ]]; then
      printf '%s\n' "${cpu_wheel}"
      return 0
    fi
    if [[ -n "${gpu_wheel}" ]]; then
      printf '%s\n' "${gpu_wheel}"
      return 0
    fi
  else
    if [[ -n "${gpu_wheel}" ]]; then
      printf '%s\n' "${gpu_wheel}"
      return 0
    fi
  fi

  return 1
}

verify_backend_offline_bundle() {
  local dir="$1"
  local preferred_runtime=""
  local paddle_cuda_track=""
  preferred_runtime="$(resolve_backend_paddle_runtime_kind)"
  paddle_cuda_track="$(resolve_backend_paddle_cuda_track)"
  require_offline_artifact "${dir}" "requirements.embedding.txt" "Embedding requirement manifest"
  require_offline_artifact "${dir}" "requirements.backend.txt" "Backend requirement manifest"
  require_offline_artifact "${dir}" "requirements.merged.txt" "Merged requirement manifest"
  if [[ "${preferred_runtime}" == "gpu" && -f "${dir}/paddle_cuda_track.txt" ]]; then
    local bundled_track=""
    bundled_track="$(tr -d '[:space:]' < "${dir}/paddle_cuda_track.txt")"
    if [[ -n "${paddle_cuda_track}" && "${bundled_track}" != "${paddle_cuda_track}" ]]; then
      echo "[ERROR] Paddle CUDA track mismatch: expected ${paddle_cuda_track}, bundle has ${bundled_track}" >&2
      echo "        Rebuild the offline bundle with PADDLE_GPU_CUDA_TRACK=${paddle_cuda_track}." >&2
      exit 1
    fi
  fi
  if [[ "${preferred_runtime}" == "gpu" ]]; then
    require_any_offline_artifact "${dir}" "PaddlePaddle GPU runtime" "paddlepaddle_gpu-*.whl" "paddlepaddle-gpu-*.whl"
  else
    require_any_offline_artifact "${dir}" "PaddlePaddle runtime" "paddlepaddle_gpu-*.whl" "paddlepaddle-gpu-*.whl" "paddlepaddle-*.whl"
  fi
  require_offline_artifact "${dir}" "pydantic_ai_slim-*.whl" "PydanticAI runtime"
  require_offline_artifact "${dir}" "pydantic_graph-*.whl" "PydanticAI graph runtime"
  require_offline_artifact "${dir}" "genai_prices-*.whl" "PydanticAI pricing metadata"
  require_offline_artifact "${dir}" "griffelib-*.whl" "PydanticAI schema helper"
  require_offline_artifact "${dir}" "opentelemetry_api-*.whl" "PydanticAI telemetry shim"
  require_offline_artifact "${dir}" "distro-*.whl" "OpenAI runtime helper"
  require_offline_artifact "${dir}" "importlib_metadata-*.whl" "OpenTelemetry importlib backport"
  require_offline_artifact "${dir}" "logfire_api-*.whl" "Pydantic graph telemetry API"
  require_offline_artifact "${dir}" "zipp-*.whl" "importlib-metadata dependency"
  require_offline_artifact "${dir}" "paddleocr-*.whl" "PDF OCR"
  require_any_offline_artifact "${dir}" "PyMuPDF text extractor" "PyMuPDF-*.whl" "pymupdf-*.whl"
  require_offline_artifact "${dir}" "paddlex-*.whl" "PaddleX OCR core"
  require_offline_artifact "${dir}" "einops-*.whl" "PaddleOCR-VL tensor ops"
  require_offline_artifact "${dir}" "sentencepiece-*.whl" "PaddleOCR-VL tokenizer"
  require_offline_artifact "${dir}" "tiktoken-*.whl" "PaddleOCR-VL tokenizer"
  require_offline_artifact "${dir}" "ftfy-*.whl" "PaddleOCR-VL text cleanup"
  require_offline_artifact "${dir}" "premailer-*.whl" "PaddleOCR-VL html parser"
  require_offline_artifact "${dir}" "lxml-*.whl" "PaddleOCR-VL html parser"
  require_offline_artifact "${dir}" "opencv_contrib_python-*.whl" "OpenCV OCR dependency"
  require_offline_artifact "${dir}" "pypdfium2-*.whl" "PDF parser dependency"
  require_offline_artifact "${dir}/pymupdf_sitepkg" "fitz/__init__.py" "PyMuPDF extracted fitz package"
  require_offline_artifact "${dir}/pymupdf_sitepkg" "pymupdf/_mupdf.so" "PyMuPDF extracted native module"
  require_offline_artifact "${dir}" "hnswlib-*.tar.gz" "HNSW source package"
  require_offline_artifact "${dir}" "pyinstaller-*.whl" "PyInstaller bundler"
  verify_torch_cuda12_offline_bundle "${dir}"
  if [[ "${preferred_runtime}" == "gpu" ]]; then
    case "${paddle_cuda_track}" in
      cu126|cu12*)
        verify_paddle_cuda12_offline_bundle "${dir}"
        ;;
      cu118|cu11*|"")
        verify_paddle_cuda11_offline_bundle "${dir}"
        ;;
      *)
        echo "[ERROR] Unsupported BACKEND_PADDLE_CUDA_TRACK=${paddle_cuda_track}" >&2
        echo "        Supported values: cu126, cu118." >&2
        exit 1
        ;;
    esac
  fi
}

verify_embedding_offline_bundle() {
  local dir="$1"
  require_offline_artifact "${dir}" "requirements.embedding.txt" "Embedding requirement manifest"
  require_offline_artifact "${dir}" "torch-*.whl" "PyTorch runtime"
  require_offline_artifact "${dir}" "sentence_transformers-*.whl" "SentenceTransformers runtime"
  require_offline_artifact "${dir}" "transformers-*.whl" "Transformers runtime"
  require_offline_artifact "${dir}" "tokenizers-*.whl" "Transformers tokenizer runtime"
  require_offline_artifact "${dir}" "safetensors-*.whl" "SafeTensors runtime"
  require_offline_artifact "${dir}" "jupyter_server_proxy-*.whl" "Jupyter server proxy"
  require_offline_artifact "${dir}" "simpervisor-*.whl" "Jupyter supervisor helper"
  verify_torch_cuda12_offline_bundle "${dir}"
}

verify_torch_cuda12_offline_bundle() {
  local dir="$1"
  require_offline_artifact "${dir}" "nvidia_cuda_nvrtc_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvrtc runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_runtime_cu12-12.4.127-*.whl" "PyTorch CUDA 12 runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_cupti_cu12-12.4.127-*.whl" "PyTorch CUDA 12 cupti runtime"
  require_offline_artifact "${dir}" "nvidia_cudnn_cu12-9.1.0.70-*.whl" "PyTorch CUDA 12 cudnn runtime"
  require_offline_artifact "${dir}" "nvidia_cublas_cu12-12.4.5.8-*.whl" "PyTorch CUDA 12 cublas runtime"
  require_offline_artifact "${dir}" "nvidia_cufft_cu12-11.2.1.3-*.whl" "PyTorch CUDA 12 cufft runtime"
  require_offline_artifact "${dir}" "nvidia_curand_cu12-10.3.5.147-*.whl" "PyTorch CUDA 12 curand runtime"
  require_offline_artifact "${dir}" "nvidia_cusolver_cu12-11.6.1.9-*.whl" "PyTorch CUDA 12 cusolver runtime"
  require_offline_artifact "${dir}" "nvidia_cusparse_cu12-12.3.1.170-*.whl" "PyTorch CUDA 12 cusparse runtime"
  require_offline_artifact "${dir}" "nvidia_cusparselt_cu12-0.6.2-*.whl" "PyTorch CUDA 12 cusparselt runtime"
  require_offline_artifact "${dir}" "nvidia_nccl_cu12-2.21.5-*.whl" "PyTorch CUDA 12 nccl runtime"
  require_offline_artifact "${dir}" "nvidia_nvtx_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvtx runtime"
  require_offline_artifact "${dir}" "nvidia_nvjitlink_cu12-12.4.127-*.whl" "PyTorch CUDA 12 nvjitlink runtime"
}

verify_paddle_cuda11_offline_bundle() {
  local dir="$1"
  require_offline_artifact "${dir}" "nvidia_cuda_runtime_cu11-11.8.89-*.whl" "Paddle CUDA 11 runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_cupti_cu11-11.8.87-*.whl" "Paddle CUDA 11 cupti runtime"
  require_offline_artifact "${dir}" "nvidia_cudnn_cu11-8.9.6.50-*.whl" "Paddle CUDA 11 cudnn runtime"
  require_offline_artifact "${dir}" "nvidia_cublas_cu11-11.11.3.6-*.whl" "Paddle CUDA 11 cublas runtime"
  require_offline_artifact "${dir}" "nvidia_cufft_cu11-10.9.0.58-*.whl" "Paddle CUDA 11 cufft runtime"
  require_offline_artifact "${dir}" "nvidia_curand_cu11-10.3.0.86-*.whl" "Paddle CUDA 11 curand runtime"
  require_offline_artifact "${dir}" "nvidia_cusolver_cu11-11.4.1.48-*.whl" "Paddle CUDA 11 cusolver runtime"
  require_offline_artifact "${dir}" "nvidia_cusparse_cu11-11.7.5.86-*.whl" "Paddle CUDA 11 cusparse runtime"
  require_offline_artifact "${dir}" "nvidia_nccl_cu11-2.19.3-*.whl" "Paddle CUDA 11 nccl runtime"
  require_offline_artifact "${dir}" "nvidia_nvtx_cu11-11.8.86-*.whl" "Paddle CUDA 11 nvtx runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_nvrtc_cu11-11.8.89-*.whl" "Paddle CUDA 11 nvrtc runtime"
  require_offline_artifact "${dir}" "protobuf-*.whl" "Paddle protobuf dependency"
  require_offline_artifact "${dir}" "opt_einsum-*.whl" "Paddle opt_einsum dependency"
}

verify_paddle_cuda12_offline_bundle() {
  local dir="$1"
  require_offline_artifact "${dir}" "nvidia_cuda_runtime_cu12-12.6.77-*.whl" "Paddle CUDA 12 runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_cupti_cu12-12.6.80-*.whl" "Paddle CUDA 12 cupti runtime"
  require_offline_artifact "${dir}" "nvidia_cudnn_cu12-9.5.1.17-*.whl" "Paddle CUDA 12 cudnn runtime"
  require_offline_artifact "${dir}" "nvidia_cublas_cu12-12.6.4.1-*.whl" "Paddle CUDA 12 cublas runtime"
  require_offline_artifact "${dir}" "nvidia_cufft_cu12-11.3.0.4-*.whl" "Paddle CUDA 12 cufft runtime"
  require_offline_artifact "${dir}" "nvidia_curand_cu12-10.3.7.77-*.whl" "Paddle CUDA 12 curand runtime"
  require_offline_artifact "${dir}" "nvidia_cusolver_cu12-11.7.1.2-*.whl" "Paddle CUDA 12 cusolver runtime"
  require_offline_artifact "${dir}" "nvidia_cusparse_cu12-12.5.4.2-*.whl" "Paddle CUDA 12 cusparse runtime"
  require_offline_artifact "${dir}" "nvidia_cusparselt_cu12-0.6.3-*.whl" "Paddle CUDA 12 cusparselt runtime"
  require_offline_artifact "${dir}" "nvidia_nccl_cu12-2.25.1-*.whl" "Paddle CUDA 12 nccl runtime"
  require_offline_artifact "${dir}" "nvidia_nvtx_cu12-12.6.77-*.whl" "Paddle CUDA 12 nvtx runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_nvrtc_cu12-12.6.77-*.whl" "Paddle CUDA 12 nvrtc runtime"
  require_offline_artifact "${dir}" "nvidia_nvjitlink_cu12-12.6.85-*.whl" "Paddle CUDA 12 nvjitlink runtime"
  require_offline_artifact "${dir}" "nvidia_cufile_cu12-1.11.1.6-*.whl" "Paddle CUDA 12 cufile runtime"
  require_offline_artifact "${dir}" "nvidia_cuda_cccl_cu12-12.6.77-*.whl" "Paddle CUDA 12 cccl runtime"
  require_offline_artifact "${dir}" "protobuf-*.whl" "Paddle protobuf dependency"
  require_offline_artifact "${dir}" "opt_einsum-*.whl" "Paddle opt_einsum dependency"
}

resolve_python_site_packages_dir() {
  local python_bin="$1"
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${python_bin}" - <<'PY'
import sysconfig

paths = sysconfig.get_paths()
print(paths.get("purelib") or paths.get("platlib") or "")
PY
}

pymupdf_import_smoke_test() {
  local python_bin="$1"
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${python_bin}" - <<'PY' >/dev/null 2>&1
import fitz  # noqa: F401
import pymupdf  # noqa: F401
PY
}

restore_backend_pymupdf_sitepkg_fallback() {
  local python_bin="$1"
  local offline_dir="$2"
  local fallback_dir="${offline_dir}/${PYMUPDF_FALLBACK_DIR_NAME}"
  local site_packages=""
  local dist_info_dir=""

  if [[ ! -d "${fallback_dir}/fitz" || ! -d "${fallback_dir}/pymupdf" ]]; then
    return 1
  fi

  site_packages="$(resolve_python_site_packages_dir "${python_bin}")"
  if [[ -z "${site_packages}" || ! -d "${site_packages}" ]]; then
    echo "[WARN] Could not resolve backend site-packages dir for PyMuPDF fallback." >&2
    return 1
  fi

  echo "[WARN] PyMuPDF import failed after pip install. Restoring extracted fallback package set into ${site_packages}" >&2
  rm -rf "${site_packages}/fitz" "${site_packages}/pymupdf"
  find "${site_packages}" -maxdepth 1 -type d \( -name 'pymupdf-*.dist-info' -o -name 'PyMuPDF-*.dist-info' \) -exec rm -rf {} +
  cp -a "${fallback_dir}/fitz" "${site_packages}/"
  cp -a "${fallback_dir}/pymupdf" "${site_packages}/"
  dist_info_dir="$(find "${fallback_dir}" -maxdepth 1 -type d \( -name 'pymupdf-*.dist-info' -o -name 'PyMuPDF-*.dist-info' \) | sort | head -n 1)"
  if [[ -n "${dist_info_dir}" ]]; then
    cp -a "${dist_info_dir}" "${site_packages}/"
  fi
  return 0
}

ensure_backend_pymupdf_importable() {
  local python_bin="$1"
  local offline_dir="$2"
  if pymupdf_import_smoke_test "${python_bin}"; then
    return 0
  fi
  if [[ -n "${offline_dir}" ]]; then
    restore_backend_pymupdf_sitepkg_fallback "${python_bin}" "${offline_dir}" || true
  fi
  pymupdf_import_smoke_test "${python_bin}"
}

install_backend_offline_paddle_runtime() {
  local python_bin="$1"
  local dir="$2"
  local wheel_path=""
  wheel_path="$(select_backend_paddle_wheel "${dir}")" || {
    if [[ "$(resolve_backend_paddle_runtime_kind)" == "gpu" ]]; then
      echo "[ERROR] PaddlePaddle GPU runtime wheel not found in ${dir}" >&2
      echo "        Add a matching paddlepaddle-gpu wheel first." >&2
    else
      echo "[ERROR] PaddlePaddle runtime wheel not found in ${dir}" >&2
      echo "        Add a matching paddlepaddle-gpu or paddlepaddle wheel first." >&2
    fi
    exit 1
  }
  echo "[INFO] Installing PaddlePaddle runtime wheel: $(basename "${wheel_path}")"
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${python_bin}" -m pip install --force-reinstall --no-index --find-links="${dir}" "${wheel_path}"
}

install_backend_online_paddle_runtime() {
  local python_bin="$1"
  local paddle_runtime_version="${PADDLE_RUNTIME_VERSION:-3.3.0}"
  local paddle_runtime_kind=""
  local paddle_cuda_track=""
  local default_gpu_whl_url=""
  local paddle_gpu_whl_url=""

  paddle_runtime_kind="$(resolve_backend_paddle_runtime_kind)"
  paddle_cuda_track="$(resolve_backend_paddle_cuda_track)"

  case "${paddle_runtime_kind}" in
    gpu)
      if ! env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
        "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
        echo "[ERROR] online PaddlePaddle GPU wheel requires Python 3.11 (cp311); selected: $("${python_bin}" --version 2>&1)" >&2
        exit 1
      fi
      default_gpu_whl_url="https://paddle-whl.bj.bcebos.com/stable/${paddle_cuda_track}/paddlepaddle-gpu/paddlepaddle_gpu-${paddle_runtime_version}-cp311-cp311-linux_x86_64.whl"
      paddle_gpu_whl_url="${PADDLE_GPU_WHL_URL:-${default_gpu_whl_url}}"
      echo "[INFO] Installing online PaddlePaddle GPU runtime (${paddle_runtime_version}, ${paddle_cuda_track})"
      env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
        "${python_bin}" -m pip install "${paddle_gpu_whl_url}"
      ;;
    cpu)
      echo "[INFO] Installing online PaddlePaddle CPU runtime (${paddle_runtime_version})"
      env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
        "${python_bin}" -m pip install "paddlepaddle==${paddle_runtime_version}"
      ;;
    *)
      echo "[ERROR] Unsupported BACKEND_PADDLE_RUNTIME_KIND=${paddle_runtime_kind}; expected gpu or cpu." >&2
      exit 1
      ;;
  esac
}

verify_backend_ocr_runtime_imports() {
  local python_bin="$1"
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}" \
    "${python_bin}" - <<'PY'
import importlib
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

required_modules = {
    "paddle": "paddle",
    "paddleocr": "paddleocr",
    "paddlex": "paddlex",
    "openai": "openai",
    "pydantic_ai": "pydantic_ai",
    "pydantic_graph": "pydantic_graph",
    "genai_prices": "genai_prices",
    "griffe": "griffelib",
    "opentelemetry": "opentelemetry-api",
    "einops": "einops",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "ftfy": "ftfy",
    "premailer": "premailer",
    "lxml": "lxml",
    "bs4": "beautifulsoup4",
    "fitz": "PyMuPDF",
    "hwpx": "python-hwpx",
}

failures = []
for import_name, label in required_modules.items():
    try:
        importlib.import_module(import_name)
    except Exception as exc:
        failures.append(f"{label}:{type(exc).__name__}:{exc}")

try:
    from paddleocr import PaddleOCRVL  # noqa: F401
except Exception as exc:
    failures.append(f"PaddleOCRVL:{type(exc).__name__}:{exc}")

try:
    import fitz  # noqa: F401
except Exception as exc:
    failures.append(f"PyMuPDF:{type(exc).__name__}:{exc}")

try:
    from hwpx import TextExtractor  # noqa: F401
except Exception as exc:
    failures.append(f"python-hwpx/TextExtractor:{type(exc).__name__}:{exc}")

if failures:
    print("; ".join(failures), file=sys.stderr)
    raise SystemExit(1)
PY
}

write_env_files() {
  cat > "${EMBEDDING_SERVER_HOME}/.env.auto" <<EOF
EMBED_HOST=127.0.0.1
EMBEDDING_API_KEY=replace-with-strong-secret
EMBEDDING_DISABLE_AUTH=0
EMBEDDING_MODEL_LARGE_PATH=\${EMBEDDING_SERVER_HOME}/models/Qwen/Qwen3-Embedding-0.6B
# Alternative example (larger Qwen):
# EMBEDDING_MODEL_LARGE_PATH=\${EMBEDDING_SERVER_HOME}/models/qwen3-embedding-4b
# Alternative example (Jina):
# EMBEDDING_MODEL_LARGE_PATH=\${EMBEDDING_SERVER_HOME}/models/jinaai/jina-embeddings-v5-text-small
EMBEDDING_DEFAULT_INDEX=large
EMBEDDING_API_LARGE_ALIAS=large
EMBEDDING_TASK_PREFIX_MODE=auto
EMBEDDING_QWEN_QUERY_INSTRUCTION="Given a user question, retrieve relevant passages that answer the question"
EMBEDDING_MODEL_DEVICE=cuda
EMBED_BATCH_SIZE=16
EMBED_MIN_BATCH_SIZE=4
EMBED_NORMALIZE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

  cat > "${MAIN_BACKEND_HOME}/.env.auto" <<EOF
LLM_API_URL=http://127.0.0.1:8003/v1/chat/completions
LLM_PORT=8003
LLM_HOST=127.0.0.1
# Keep llama.cpp ctx-size and backend context limit aligned.
# Default long-context baseline: keep both values aligned.
LLM_CTX_SIZE=131072
LLM_CONTEXT_LIMIT=131072
LLM_MODELS_DIR=\${MAIN_BACKEND_HOME}/models/llm
LLM_MODEL_PATH=\${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf
LLM_MODEL_NAME=qwen3.5-9b-q4_k_m
LLM_API_KEY=replace-with-strong-secret
LLM_PROBE_TIMEOUT_SECONDS=10
LLM_READY_TIMEOUT_SECONDS=600
LLM_RUNTIME=\${MAIN_BACKEND_HOME}/runtime/llama-server
LLM_QUALITY_RETRY_ENABLED=1
LLM_QUALITY_MAX_RETRY=2
LLM_MIN_ANSWER_LINE_CHARS=40
RAG_LLM_QUERY_ANALYZE_ENABLED=1
PYDANTIC_AI_REQUEST_LIMIT=6
PYDANTIC_AI_TOOL_CALLS_LIMIT=8
PYDANTIC_AI_ENABLE_RETRIEVAL_TOOL=1
PYDANTIC_AI_TOOL_TIMEOUT_SECONDS=20
PYDANTIC_AI_MAX_CONCURRENCY=1
EMBEDDING_PROVIDER=api
EMBEDDING_API_URL=http://127.0.0.1:8002
EMBEDDING_API_KEY=replace-with-strong-secret
EMBEDDING_TIMEOUT=60
EMBEDDING_API_BATCH_SIZE=16
EMBEDDING_PROBE_TIMEOUT_SECONDS=10
EMBEDDING_READY_TIMEOUT_SECONDS=600
BACKEND_PROBE_TIMEOUT_SECONDS=10
BACKEND_READY_TIMEOUT_SECONDS=120
EMBEDDING_API_LARGE_ALIAS=large
RAG_TOP_K=24
RAG_TOP_K_OVERVIEW=36
RAG_LLM_RERANK_CANDIDATES=72
RAG_LLM_RERANK_KEEP=20
RAG_LLM_HELPER_MAX_TOKENS=220
RAG_LLM_HELPER_TIMEOUT=45
RAG_CONTEXT_MAX_CHARS=5600
RAG_CONTEXT_PER_RESULT_MAX_CHARS=700
RAG_DIVERSIFY_ENABLED=1
RAG_MAX_PER_SECTION=2
RAG_MAX_PER_FILE=6
RAG_GROUNDING_GATE_ENABLED=1
RAG_GROUNDING_TOP1_MIN=0.33
RAG_GROUNDING_COVERAGE_MIN=0.22
RAG_GROUNDING_SOFTEN_ENABLED=1
RAG_GROUNDING_TOP1_SOFT_MIN=0.26
RAG_GROUNDING_COVERAGE_SOFT_MIN=0.12
RAG_GROUNDING_MIN_KEYWORD_HITS=1
RAG_GROUNDING_CONFLICT_CHECK_ENABLED=1
RAG_ANSWER_COVERAGE_TOP_K=8
RAG_SEARCH_CANDIDATES=120
RAG_LEXICAL_WEIGHT=0.48
RAG_HYBRID_FTS_WEIGHT=0.18
RAG_SQLITE_DENSE_ENABLED=1
RAG_HNSW_ENABLED=0
RAG_INDEX_INCLUDE_RAW_WITH_NORMALIZED=1
RAG_ENGINE_MAX_LOADED_KBS=1
RAG_ENGINE_IDLE_TTL_SECONDS=900
RAG_CONCEPT_LINKS_ENABLED=1
RAG_CONCEPT_MAX_TERMS_PER_CHUNK=6
RAG_CONCEPT_MAX_NGRAM=2
RAG_CONCEPT_SIMILARITY_THRESHOLD=0.84
RAG_CONCEPT_QUERY_LIMIT=24
RAG_CONCEPT_CHUNK_EXPAND_LIMIT=64
RAG_CONCEPT_SCORE_WEIGHT=0.22
RAG_NORMALIZED_SCORE_PENALTY=0.04
RAG_CODE_MATCH_BOOST=0.12
RAG_CODE_HINT_BOOST_RATIO=0.45
RAG_EXACT_KEYWORD_BOOST=0.07
RAG_RECENCY_BOOST=0.06
RAG_RECENCY_HALF_LIFE_DAYS=45
RAG_LITERAL_MATCH_BOOST=0.08
RAG_MAX_NORMALIZED_RESULTS=2
RAG_TRACE_LOG_ENABLED=1
RAG_ROLE_ROUTING_ENABLED=1
RAG_ROLE_ROUTING_STRICT=0
RAG_TRACE_TOP_N=8
RAG_NORMALIZED_TARGET_TOKENS=360
RAG_NORMALIZED_MAX_TOKENS=460
RAG_NORMALIZED_CONFLICT_LIMIT=3
TXT_SPLIT_ENABLED=1
TXT_SPLIT_TRIGGER_LINES=120
TXT_SPLIT_TARGET_TOKENS=2200
TXT_SPLIT_MIN_TOKENS=1000
TXT_SPLIT_MAX_TOKENS=2800
TXT_CHUNK_TARGET_TOKENS=640
TXT_CHUNK_MIN_TOKENS=420
TXT_CHUNK_MAX_TOKENS=900
TXT_CHUNK_OVERLAP_RATIO=0.25
PDF_CHUNK_TARGET_TOKENS=640
PDF_CHUNK_MIN_TOKENS=420
PDF_CHUNK_MAX_TOKENS=900
PDF_PARSE_MODE=ocr_first
PDF_TEXT_EXTRACTOR=pymupdf
PDF_TEXT_MIN_CHARS=12
PDF_TEXT_MIN_NONSPACE_RATIO=0.20
PDF_OCR_MODEL_NAME=\${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL
PDF_OCR_MAX_PAGES=400
PDF_OCR_TARGET_PAGES=200
PDF_OCR_TARGET_SECONDS=300
PDF_OCR_BACKEND=local
PDF_OCR_HPS_URL=http://127.0.0.1:8080
PDF_OCR_HPS_ENDPOINT=/layout-parsing
PDF_OCR_HPS_READY_TIMEOUT_SECONDS=600
PDF_OCR_HPS_REQUEST_TIMEOUT_SECONDS=600
PDF_OCR_HPS_CHUNK_PAGES=16
PDF_OCR_HPS_MAX_CONCURRENCY=4
PDF_OCR_HPS_FALLBACK_TO_LOCAL=1
PDF_OCR_PAUSE_LLM_DURING_JOB=1
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK=0
PDF_OCR_OPTIMIZATION_PROFILE=h100_96gb_fast
PDF_OCR_USE_INTERNAL_QUEUES=1
PDF_OCR_GPU_PROCESS_ISOLATION=1
PDF_OCR_WARMUP_ON_STARTUP=1
PDF_OCR_PERSISTENT_WORKER=1
PDF_OCR_PERSISTENT_WORKERS=1
# H100 96GB profile: use CUDA 12.6 Paddle runtime and keep optional recognizers off first.
PDF_OCR_USE_CHART_RECOGNITION=0
PDF_OCR_USE_SEAL_RECOGNITION=0
PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK=0
PDF_OCR_MAX_NEW_TOKENS=768
PDF_OCR_MIN_PIXELS=3136
PDF_OCR_MAX_PIXELS=786432
PDF_OCR_LAYOUT_SHAPE_MODE=rect
PDF_OCR_VL_REC_MAX_CONCURRENCY=1
# PDF_OCR_ENGINE=paddle
# PDF_OCR_VLM_EXTRA_ARGS_JSON={"ocr_max_pixels":262144,"table_max_pixels":786432}
PDF_UPLOAD_OCR_ENABLED=1
PDF_LAZY_OCR_CACHE_ENABLED=0
PDF_ANSWER_PATH_LAZY_OCR_ENABLED=0
PDF_BACKGROUND_OCR_ENABLED=0
PDF_BACKGROUND_OCR_WORKER_COUNT=1
PDF_OCR_DEVICE=gpu:0
BACKEND_PADDLE_RUNTIME_KIND=gpu
BACKEND_PADDLE_CUDA_TRACK=cu126
PDF_OCR_STRICT_GPU_COMPAT=1
# H100 upload path should fail fast on GPU OCR errors instead of retrying the full PDF on CPU.
PDF_OCR_GPU_FALLBACK_TO_CPU=0
PDF_OCR_GPU_BUDGET_GB=40
PDF_OCR_GPU_INITIAL_MEMORY_MB=2048
PDF_OCR_GPU_REALLOCATE_MEMORY_MB=1024
PDF_OCR_GPU_ALLOCATOR_STRATEGY=auto_growth
PDF_OCR_PARALLEL_MAX_WORKERS=1
PDF_OCR_PARALLEL_MIN_PAGES=1
# H100 tuning candidates after confirming model reuse and no GPU fallback:
# PDF_OCR_EXEC_BATCH_PAGES=3 means one worker handles three-page batches.
# Increase workers only for an explicit server benchmark after the single-worker baseline passes.
# PDF_OCR_VL_REC_MAX_CONCURRENCY=2
PDF_OCR_GPU_BATCH_TIMEOUT_SECONDS=420
PDF_OCR_GPU_SINGLE_BATCH_TIMEOUT_SECONDS=240
PDF_OCR_EXEC_BATCH_PAGES=3
PDF_OCR_PROGRESS_HEARTBEAT_SECONDS=5
PDF_OCR_VL_MODEL_DIR=\${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL
PDF_OCR_LAYOUT_MODEL_DIR=\${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3
PDF_OCR_DOC_ORIENTATION_MODEL_DIR=\${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori
PDF_OCR_DOC_UNWARP_MODEL_DIR=\${MAIN_BACKEND_HOME}/models/ocr/UVDoc
API_HOST=127.0.0.1
EOF

  cat > "${PROJECT_GPU_HOME}/runtime.env.example" <<'EOF'
# Non-hidden runtime config for environments where .env upload is restricted.
# Copy to project-gpu/runtime.env and edit values as needed.

COMPASS_AUTO_PORT=1
COMPASS_PORT_RANGE_START=8000
COMPASS_PORT_RANGE_END=8099
EMBED_PORT_START=8002
LLM_PORT_START=8003
API_PORT_START=8004

EMBED_HOST=127.0.0.1
EMBEDDING_API_KEY=replace-with-strong-secret
EMBEDDING_DISABLE_AUTH=0
EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_SERVER_HOME}/models/Qwen/Qwen3-Embedding-0.6B
# Alternative example (larger Qwen):
# EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_SERVER_HOME}/models/qwen3-embedding-4b
# Alternative example (Jina):
# EMBEDDING_MODEL_LARGE_PATH=${EMBEDDING_SERVER_HOME}/models/jinaai/jina-embeddings-v5-text-small
EMBEDDING_DEFAULT_INDEX=large
EMBEDDING_API_LARGE_ALIAS=large
EMBEDDING_TASK_PREFIX_MODE=auto
EMBEDDING_QWEN_QUERY_INSTRUCTION="Given a user question, retrieve relevant passages that answer the question"
EMBEDDING_MODEL_DEVICE=cuda
EMBED_BATCH_SIZE=16
EMBED_MIN_BATCH_SIZE=4
EMBED_NORMALIZE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LLM_API_URL=http://127.0.0.1:8003/v1/chat/completions
LLM_PORT=8003
LLM_HOST=127.0.0.1
LLM_WEBUI=1
# Jupyter server-proxy strips /user/<user>/proxy/<port> before forwarding to llama-server.
# Keep this empty for the normal Jupyter URL and open the Web UI with a trailing slash:
#   https://<host>:8000/user/<user>/proxy/8003/
LLM_API_PREFIX=
# Keep llama.cpp ctx-size and backend context limit aligned.
# Default long-context baseline: keep both values aligned.
LLM_CTX_SIZE=131072
LLM_CONTEXT_LIMIT=131072
LLM_MODELS_DIR=${MAIN_BACKEND_HOME}/models/llm
LLM_MODEL_PATH=${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf
LLM_MODEL_NAME=qwen3.5-9b-q4_k_m
LLM_API_KEY=replace-with-strong-secret
LLM_PROBE_TIMEOUT_SECONDS=10
LLM_READY_TIMEOUT_SECONDS=600
LLM_RUNTIME=${MAIN_BACKEND_HOME}/runtime/llama-server
LLM_QUALITY_RETRY_ENABLED=1
LLM_QUALITY_MAX_RETRY=2
LLM_MIN_ANSWER_LINE_CHARS=40
RAG_LLM_QUERY_ANALYZE_ENABLED=1
PYDANTIC_AI_REQUEST_LIMIT=6
PYDANTIC_AI_TOOL_CALLS_LIMIT=8
PYDANTIC_AI_ENABLE_RETRIEVAL_TOOL=1
PYDANTIC_AI_TOOL_TIMEOUT_SECONDS=20
PYDANTIC_AI_MAX_CONCURRENCY=1
PYDANTIC_AI_PROVIDER_KIND=openai_compatible
PYDANTIC_AI_PROVIDER_LABEL=local-openai-compatible
PYDANTIC_AI_HISTORY_STRATEGY=compact_text
PYDANTIC_AI_COMPACT_HISTORY_TURNS=8
PYDANTIC_AI_COMPACT_HISTORY_CHARS=1400
PYDANTIC_AI_ENABLE_PHASE_EVENTS=1

EMBEDDING_PROVIDER=api
EMBEDDING_API_URL=http://127.0.0.1:8002
EMBEDDING_TIMEOUT=60
EMBEDDING_API_BATCH_SIZE=16
EMBEDDING_PROBE_TIMEOUT_SECONDS=10
EMBEDDING_READY_TIMEOUT_SECONDS=600
BACKEND_PROBE_TIMEOUT_SECONDS=10
BACKEND_READY_TIMEOUT_SECONDS=120
RAG_TOP_K=24
RAG_TOP_K_OVERVIEW=36
RAG_LLM_RERANK_CANDIDATES=72
RAG_LLM_RERANK_KEEP=20
RAG_LLM_HELPER_MAX_TOKENS=220
RAG_LLM_HELPER_TIMEOUT=45
# Long-context validation example: raise this with LLM_CTX_SIZE/LLM_CONTEXT_LIMIT.
RAG_CONTEXT_MAX_CHARS=5600
RAG_CONTEXT_PER_RESULT_MAX_CHARS=700
RAG_DIVERSIFY_ENABLED=1
RAG_MAX_PER_SECTION=2
RAG_MAX_PER_FILE=6
RAG_GROUNDING_GATE_ENABLED=1
RAG_GROUNDING_TOP1_MIN=0.33
RAG_GROUNDING_COVERAGE_MIN=0.22
RAG_GROUNDING_SOFTEN_ENABLED=1
RAG_GROUNDING_TOP1_SOFT_MIN=0.26
RAG_GROUNDING_COVERAGE_SOFT_MIN=0.12
RAG_GROUNDING_MIN_KEYWORD_HITS=1
RAG_GROUNDING_CONFLICT_CHECK_ENABLED=1
RAG_ANSWER_COVERAGE_TOP_K=8
RAG_SEARCH_CANDIDATES=120
RAG_LEXICAL_WEIGHT=0.48
RAG_HYBRID_FTS_WEIGHT=0.18
RAG_SQLITE_DENSE_ENABLED=1
RAG_HNSW_ENABLED=0
RAG_INDEX_INCLUDE_RAW_WITH_NORMALIZED=1
RAG_ENGINE_MAX_LOADED_KBS=1
RAG_ENGINE_IDLE_TTL_SECONDS=900
RAG_CONCEPT_LINKS_ENABLED=1
RAG_CONCEPT_MAX_TERMS_PER_CHUNK=6
RAG_CONCEPT_MAX_NGRAM=2
RAG_CONCEPT_SIMILARITY_THRESHOLD=0.84
RAG_CONCEPT_QUERY_LIMIT=24
RAG_CONCEPT_CHUNK_EXPAND_LIMIT=64
RAG_CONCEPT_SCORE_WEIGHT=0.22
RAG_NORMALIZED_SCORE_PENALTY=0.04
RAG_CODE_MATCH_BOOST=0.12
RAG_CODE_HINT_BOOST_RATIO=0.45
RAG_EXACT_KEYWORD_BOOST=0.07
RAG_RECENCY_BOOST=0.06
RAG_RECENCY_HALF_LIFE_DAYS=45
RAG_LITERAL_MATCH_BOOST=0.08
RAG_MAX_NORMALIZED_RESULTS=2
RAG_TRACE_LOG_ENABLED=1
RAG_ROLE_ROUTING_ENABLED=1
RAG_ROLE_ROUTING_STRICT=0
RAG_TRACE_TOP_N=8
WIKI_ANSWER_MEMORY_ENABLED=1
WIKI_ANSWER_MEMORY_FASTPATH_ENABLED=1
WIKI_ANSWER_MEMORY_FASTPATH_MIN_SCORE=0.82
WIKI_ACTIVE_RETRIEVAL_ENABLED=0
WIKI_PAGE_WORKFLOW_ENABLED=0
ONTOLOGY_RAG_ENABLED=1
ONTOLOGY_LLM_EXTRACTION_ENABLED=1
ONTOLOGY_MAX_HOPS=2
ONTOLOGY_MIN_FACT_CONFIDENCE=0.62
ONTOLOGY_WIKI_CONFIDENCE_BOOST=0.08
# HWPX/XLSX 구조 RAG v2는 운영 평가 전까지 기본 OFF입니다.
STRUCTURE_RAG_V2_ENABLED=0
HWPX_STRUCTURE_RAG_V2_ENABLED=0
XLSX_STRUCTURE_RAG_V2_ENABLED=0
DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED=1
DOCUMENT_UPLOAD_ONTOLOGY_LLM_JOB_ENABLED=0
REPORTED_ANSWER_ONTOLOGY_RECHECK_ENABLED=1
ONTOLOGY_REPORTED_MAX_CHUNKS=8
STRUCTURE_RAG_PARENT_RESULT_LIMIT=1
RAG_NORMALIZED_TARGET_TOKENS=360
RAG_NORMALIZED_MAX_TOKENS=460
RAG_NORMALIZED_CONFLICT_LIMIT=3
TXT_SPLIT_ENABLED=1
TXT_SPLIT_TRIGGER_LINES=120
TXT_SPLIT_TARGET_TOKENS=2200
TXT_SPLIT_MIN_TOKENS=1000
TXT_SPLIT_MAX_TOKENS=2800
TXT_CHUNK_TARGET_TOKENS=640
TXT_CHUNK_MIN_TOKENS=420
TXT_CHUNK_MAX_TOKENS=900
TXT_CHUNK_OVERLAP_RATIO=0.25
PDF_CHUNK_TARGET_TOKENS=640
PDF_CHUNK_MIN_TOKENS=420
PDF_CHUNK_MAX_TOKENS=900
HWPX_EXTRACT_ENABLED=1
HWPX_INCLUDE_TABLES=1
HWPX_INCLUDE_FOOTNOTES=0
HWPX_INCLUDE_ENDNOTES=0
HWPX_INCLUDE_HEADER_FOOTER=0
HWPX_CHUNK_TARGET_TOKENS=220
HWPX_CHUNK_MIN_TOKENS=80
HWPX_CHUNK_MAX_TOKENS=320
HWPX_CHUNK_OVERLAP_RATIO=0.12
PDF_PARSE_MODE=ocr_first
PDF_TEXT_EXTRACTOR=pymupdf
# PDF_TEXT_EXTRACTOR=disabled
PDF_TEXT_MIN_CHARS=12
PDF_TEXT_MIN_NONSPACE_RATIO=0.20
PDF_OCR_MODEL_NAME=${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL
PDF_OCR_MAX_PAGES=400
PDF_OCR_TARGET_PAGES=200
PDF_OCR_TARGET_SECONDS=300
PDF_OCR_BACKEND=local
PDF_OCR_HPS_URL=http://127.0.0.1:8080
PDF_OCR_HPS_ENDPOINT=/layout-parsing
PDF_OCR_HPS_READY_TIMEOUT_SECONDS=600
PDF_OCR_HPS_REQUEST_TIMEOUT_SECONDS=600
PDF_OCR_HPS_CHUNK_PAGES=16
PDF_OCR_HPS_MAX_CONCURRENCY=4
PDF_OCR_HPS_FALLBACK_TO_LOCAL=1
PDF_OCR_PAUSE_LLM_DURING_JOB=1
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK=0
PDF_OCR_OPTIMIZATION_PROFILE=h100_96gb_fast
PDF_OCR_USE_INTERNAL_QUEUES=1
PDF_OCR_GPU_PROCESS_ISOLATION=1
PDF_OCR_WARMUP_ON_STARTUP=1
PDF_OCR_PERSISTENT_WORKER=1
PDF_OCR_PERSISTENT_WORKERS=1
# H100 96GB profile: use CUDA 12.6 Paddle runtime and keep optional recognizers off first.
PDF_OCR_USE_CHART_RECOGNITION=0
PDF_OCR_USE_SEAL_RECOGNITION=0
PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK=0
PDF_OCR_MAX_NEW_TOKENS=768
PDF_OCR_MIN_PIXELS=3136
PDF_OCR_MAX_PIXELS=786432
PDF_OCR_LAYOUT_SHAPE_MODE=rect
PDF_OCR_VL_REC_MAX_CONCURRENCY=1
# PDF_OCR_ENGINE=paddle
# PDF_OCR_VLM_EXTRA_ARGS_JSON={"ocr_max_pixels":262144,"table_max_pixels":786432}
PDF_UPLOAD_OCR_ENABLED=1
PDF_LAZY_OCR_CACHE_ENABLED=0
PDF_ANSWER_PATH_LAZY_OCR_ENABLED=0
PDF_BACKGROUND_OCR_ENABLED=0
PDF_BACKGROUND_OCR_WORKER_COUNT=1
PDF_OCR_DEVICE=gpu:0
BACKEND_PADDLE_RUNTIME_KIND=gpu
BACKEND_PADDLE_CUDA_TRACK=cu126
PDF_OCR_STRICT_GPU_COMPAT=1
# H100 upload path should fail fast on GPU OCR errors instead of retrying the full PDF on CPU.
PDF_OCR_GPU_FALLBACK_TO_CPU=0
PDF_OCR_GPU_BUDGET_GB=40
PDF_OCR_GPU_INITIAL_MEMORY_MB=2048
PDF_OCR_GPU_REALLOCATE_MEMORY_MB=1024
PDF_OCR_GPU_ALLOCATOR_STRATEGY=auto_growth
PDF_OCR_PARALLEL_MAX_WORKERS=1
PDF_OCR_PARALLEL_MIN_PAGES=1
# H100 tuning candidates after confirming model reuse and no GPU fallback:
# PDF_OCR_EXEC_BATCH_PAGES=3 means one worker handles three-page batches.
# Increase workers only for an explicit server benchmark after the single-worker baseline passes.
# PDF_OCR_VL_REC_MAX_CONCURRENCY=2
PDF_OCR_GPU_BATCH_TIMEOUT_SECONDS=420
PDF_OCR_GPU_SINGLE_BATCH_TIMEOUT_SECONDS=240
PDF_OCR_EXEC_BATCH_PAGES=3
PDF_OCR_PROGRESS_HEARTBEAT_SECONDS=5
PDF_OCR_VL_MODEL_DIR=${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL
PDF_OCR_LAYOUT_MODEL_DIR=${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3
PDF_OCR_DOC_ORIENTATION_MODEL_DIR=${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori
PDF_OCR_DOC_UNWARP_MODEL_DIR=${MAIN_BACKEND_HOME}/models/ocr/UVDoc

API_HOST=127.0.0.1
API_PORT=8004
API_RELOAD=0
# UPLOAD_WORKER_COUNT=1
# UPLOAD_FAST_WORKER_COUNT=1
EOF

  if [[ ! -f "${PROJECT_GPU_HOME}/runtime.env" ]]; then
    cp "${PROJECT_GPU_HOME}/runtime.env.example" "${PROJECT_GPU_HOME}/runtime.env"
  fi
}

echo "[INFO] COMPASSLM_HOME=${COMPASSLM_HOME}"
echo "[INFO] EMBEDDING_SERVER_HOME=${EMBEDDING_SERVER_HOME}"
echo "[INFO] MAIN_BACKEND_HOME=${MAIN_BACKEND_HOME}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
if [[ -n "${OFFLINE_DIR}" ]]; then
  echo "[INFO] OFFLINE_DIR=${OFFLINE_DIR}"
fi

EMBED_VENV="${EMBEDDING_SERVER_HOME}/compassvenv"
BACKEND_VENV="${MAIN_BACKEND_HOME}/compassvenv"
EMBED_REQ="${EMBEDDING_SERVER_HOME}/requirements.txt"
BACKEND_REQ="${MAIN_BACKEND_HOME}/requirements.txt"

echo "[INFO] Preparing embedding server venv..."
create_or_update_venv "${EMBED_VENV}"
if [[ "${OFFLINE_EMBED}" == "1" ]]; then
  if [[ ! -d "${OFFLINE_DIR}" ]]; then
    echo "[ERROR] Offline package dir not found: ${OFFLINE_DIR}" >&2
    exit 1
  fi
  "${EMBED_VENV}/bin/python" - <<'PY' >/dev/null 2>&1 || {
import sys
raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) == (3, 11) else 1)
PY
    echo "[ERROR] --offline-embed requires Python 3.11 venv." >&2
    exit 1
  }
  verify_embedding_offline_bundle "${OFFLINE_DIR}"
  EMBED_REQ="$(resolve_offline_requirements_file "${EMBED_REQ}" "${OFFLINE_DIR}" "requirements.embedding.txt")"
  echo "[INFO] Offline embedding requirements: ${EMBED_REQ}"
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${EMBED_VENV}/bin/python" -m pip install --no-index --find-links="${OFFLINE_DIR}" -r "${EMBED_REQ}"
else
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${EMBED_VENV}/bin/python" -m pip install -r "${EMBED_REQ}"
fi

if [[ "${SKIP_BACKEND}" != "1" ]]; then
  if [[ ! -f "${BACKEND_REQ}" ]]; then
    echo "[ERROR] Main-backend requirements not found: ${BACKEND_REQ}" >&2
    exit 1
  fi
  echo "[INFO] Preparing main-backend venv..."
  create_or_update_venv "${BACKEND_VENV}"
  if [[ "${OFFLINE_BACKEND}" == "1" ]]; then
    if [[ ! -d "${OFFLINE_DIR}" ]]; then
      echo "[ERROR] Offline package dir not found: ${OFFLINE_DIR}" >&2
      exit 1
    fi
    "${BACKEND_VENV}/bin/python" - <<'PY' >/dev/null 2>&1 || {
import sys
raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) == (3, 11) else 1)
PY
      echo "[ERROR] --offline-backend requires Python 3.11 venv (cp311 package set)." >&2
      exit 1
    }
    verify_backend_offline_bundle "${OFFLINE_DIR}"
    install_backend_offline_paddle_runtime "${BACKEND_VENV}/bin/python" "${OFFLINE_DIR}"
    BACKEND_REQ="$(resolve_offline_requirements_file "${BACKEND_REQ}" "${OFFLINE_DIR}" "requirements.backend.txt")"
    echo "[INFO] Offline backend requirements: ${BACKEND_REQ}"
    env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
      "${BACKEND_VENV}/bin/python" -m pip install --no-index --find-links="${OFFLINE_DIR}" -r "${BACKEND_REQ}"
    echo "[INFO] Re-applying PaddlePaddle runtime after backend requirements..."
    install_backend_offline_paddle_runtime "${BACKEND_VENV}/bin/python" "${OFFLINE_DIR}"
  else
    env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
      "${BACKEND_VENV}/bin/python" -m pip install -r "${BACKEND_REQ}"
    install_backend_online_paddle_runtime "${BACKEND_VENV}/bin/python"
  fi
  echo "[INFO] Verifying main-backend OCR runtime imports..."
  ensure_backend_pymupdf_importable "${BACKEND_VENV}/bin/python" "${OFFLINE_DIR:-}" || {
    echo "[ERROR] PyMuPDF import check failed in ${BACKEND_VENV}" >&2
    echo "        Confirm ${OFFLINE_DIR:-<offline_dir>}/${PYMUPDF_FALLBACK_DIR_NAME} or PyMuPDF wheel contents were copied with the bundle." >&2
    exit 1
  }
  verify_backend_ocr_runtime_imports "${BACKEND_VENV}/bin/python" || {
    echo "[ERROR] Main-backend OCR runtime import check failed in ${BACKEND_VENV}" >&2
    echo "        Check missing module names above before rerunning backend." >&2
    exit 1
  }
fi

write_env_files

echo
echo "[OK] Setup complete."
echo "Next:"
echo "  1) ${PROJECT_GPU_HOME}/check_gpu_assets.sh"
echo "  2) source ${PROJECT_GPU_HOME}/load_gpu_env.sh"
echo "  3) ${PROJECT_GPU_HOME}/run_embedding_server.sh"
echo "  4) ${PROJECT_GPU_HOME}/run_llm_server.sh"
echo "  5) ${PROJECT_GPU_HOME}/run_backend_api.sh"
