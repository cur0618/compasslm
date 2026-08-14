#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBED_REQ_FILE="${ROOT_DIR}/requirements.txt"
BACKEND_REQ_FILE="${ROOT_DIR}/../main-backend/requirements.txt"
PYINSTALLER_PIN="${PYINSTALLER_PIN:-pyinstaller==6.19.0}"
TARGET_TAG="cp311-linux_x86_64"
TARGET_DIR="${ROOT_DIR}/offline_packages/${TARGET_TAG}"
HNSWLIB_VERSION="${HNSWLIB_VERSION:-0.8.0}"
HNSWLIB_STAGING_DIR="${ROOT_DIR}/offline_packages/_staging_sources"
HNSWLIB_STAGED_SOURCE="${HNSWLIB_STAGING_DIR}/hnswlib-${HNSWLIB_VERSION}.tar.gz"
PYMUPDF_FALLBACK_DIR="${TARGET_DIR}/pymupdf_sitepkg"
PADDLE_RUNTIME_VERSION="${PADDLE_RUNTIME_VERSION:-3.3.0}"
PADDLE_GPU_CUDA_TRACK="${PADDLE_GPU_CUDA_TRACK:-cu126}"
PADDLE_GPU_WHL_FILE="${PADDLE_GPU_WHL_FILE:-paddlepaddle_gpu-${PADDLE_RUNTIME_VERSION}-cp311-cp311-linux_x86_64.whl}"
PADDLE_GPU_WHL_URL="${PADDLE_GPU_WHL_URL:-https://paddle-whl.bj.bcebos.com/stable/${PADDLE_GPU_CUDA_TRACK}/paddlepaddle-gpu/${PADDLE_GPU_WHL_FILE}}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

retry() {
  local max_attempts="$1"
  shift
  local attempt=1
  until "$@"; do
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      return 1
    fi
    echo "[WARN] Command failed (attempt ${attempt}/${max_attempts}). Retrying..."
    attempt=$((attempt + 1))
    sleep 3
  done
}

if [[ ! -f "${EMBED_REQ_FILE}" ]]; then
  echo "[ERROR] embedding requirements.txt not found: ${EMBED_REQ_FILE}" >&2
  exit 1
fi

if [[ ! -f "${BACKEND_REQ_FILE}" ]]; then
  echo "[ERROR] main-backend requirements.txt not found: ${BACKEND_REQ_FILE}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 command not found." >&2
  exit 1
fi

echo "[INFO] Preparing ${TARGET_DIR}"
if [[ -d "${TARGET_DIR}" && "${FORCE_CLEAN}" != "1" ]]; then
  echo "[ERROR] Refusing to delete existing offline bundle: ${TARGET_DIR}" >&2
  echo "        This script rebuilds the bundle from the network and would remove the current files." >&2
  echo "        Move the directory aside or rerun with FORCE_CLEAN=1 only on an online preparation machine." >&2
  exit 1
fi
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

MERGED_REQ="$(mktemp)"
cleanup() {
  rm -f "${MERGED_REQ}"
}
trap cleanup EXIT

download_hnswlib_source() {
  local meta_file
  local source_url

  if [[ -f "${HNSWLIB_STAGED_SOURCE}" ]]; then
    echo "[INFO] Using staged hnswlib source: ${HNSWLIB_STAGED_SOURCE}"
    cp "${HNSWLIB_STAGED_SOURCE}" "${TARGET_DIR}/hnswlib-${HNSWLIB_VERSION}.tar.gz"
    return 0
  fi

  meta_file="$(mktemp)"
  retry 4 curl -fsSL "https://pypi.org/pypi/hnswlib/${HNSWLIB_VERSION}/json" -o "${meta_file}"
  source_url="$(python3 - "${meta_file}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

urls = payload.get("urls", [])
for item in urls:
    if item.get("packagetype") == "sdist":
        print(item["url"])
        raise SystemExit(0)

raise SystemExit("sdist url not found")
PY
)"
  rm -f "${meta_file}"

  retry 4 curl -fLo "${TARGET_DIR}/hnswlib-${HNSWLIB_VERSION}.tar.gz" "${source_url}"
}

extract_pymupdf_fallback_tree() {
  local wheel_path=""
  wheel_path="$(find "${TARGET_DIR}" -maxdepth 1 -type f \( -name 'PyMuPDF-*.whl' -o -name 'pymupdf-*.whl' \) | sort | tail -n 1)"
  if [[ -z "${wheel_path}" ]]; then
    echo "[ERROR] Cannot build PyMuPDF fallback tree because no PyMuPDF wheel exists in ${TARGET_DIR}" >&2
    exit 1
  fi

  rm -rf "${PYMUPDF_FALLBACK_DIR}"
  mkdir -p "${PYMUPDF_FALLBACK_DIR}"
  python3 - "${wheel_path}" "${PYMUPDF_FALLBACK_DIR}" <<'PY'
import pathlib
import sys
import zipfile

wheel_path = pathlib.Path(sys.argv[1])
target_dir = pathlib.Path(sys.argv[2])
keep_prefixes = ("fitz/", "pymupdf/")
keep_dist_info_suffix = ".dist-info/"

with zipfile.ZipFile(wheel_path) as zf:
    for member in zf.infolist():
        name = member.filename
        if member.is_dir():
            if name.startswith(keep_prefixes) or name.endswith(keep_dist_info_suffix):
                zf.extract(member, target_dir)
            continue
        if name.startswith(keep_prefixes) or "/RECORD" in name or name.endswith("dist-info/METADATA") or name.endswith("dist-info/WHEEL") or name.endswith("dist-info/entry_points.txt") or name.endswith("dist-info/COPYING") or name.endswith("dist-info/README.md"):
            zf.extract(member, target_dir)
PY

  cat > "${TARGET_DIR}/README_PyMuPDF_fallback.txt" <<'EOF'
PyMuPDF offline fallback bundle

- Primary install path: pip installs PyMuPDF wheel from this folder.
- Fallback path: if backend venv still reports `No module named fitz`, use `pymupdf_sitepkg/`.
- Included files:
  - `pymupdf_sitepkg/fitz/`
  - `pymupdf_sitepkg/pymupdf/`
  - `pymupdf_sitepkg/pymupdf-*.dist-info/`

These directories are extracted from the exact PyMuPDF wheel in this bundle and can be copied into
backend `site-packages` as a manual fallback. `setup_gpu_track.sh --offline-backend` now tries this
fallback automatically before failing the OCR runtime import check.
EOF
}

download_paddle_gpu_runtime() {
  local target_path="${TARGET_DIR}/${PADDLE_GPU_WHL_FILE}"
  local -a paddle_dep_specs=()
  echo "[INFO] Downloading PaddlePaddle GPU runtime (${PADDLE_GPU_CUDA_TRACK}): ${PADDLE_GPU_WHL_FILE}"
  retry 4 curl -fL "${PADDLE_GPU_WHL_URL}" -o "${target_path}"
  echo "${PADDLE_GPU_CUDA_TRACK}" > "${TARGET_DIR}/paddle_cuda_track.txt"

  if [[ "${PADDLE_GPU_CUDA_TRACK}" == "cu126" || "${PADDLE_GPU_CUDA_TRACK}" == "cu12" ]]; then
    paddle_dep_specs=(
      "nvidia-cuda-runtime-cu12==12.6.77"
      "nvidia-cuda-cupti-cu12==12.6.80"
      "nvidia-cudnn-cu12==9.5.1.17"
      "nvidia-cublas-cu12==12.6.4.1"
      "nvidia-cufft-cu12==11.3.0.4"
      "nvidia-curand-cu12==10.3.7.77"
      "nvidia-cusolver-cu12==11.7.1.2"
      "nvidia-cusparse-cu12==12.5.4.2"
      "nvidia-cusparselt-cu12==0.6.3"
      "nvidia-nccl-cu12==2.25.1"
      "nvidia-nvtx-cu12==12.6.77"
      "nvidia-cuda-nvrtc-cu12==12.6.77"
      "nvidia-nvjitlink-cu12==12.6.85"
      "nvidia-cufile-cu12==1.11.1.6"
      "nvidia-cuda-cccl-cu12==12.6.77"
      "protobuf>=3.20.2"
      "opt_einsum==3.3.0"
    )
  else
    paddle_dep_specs=("${target_path}")
  fi

  echo "[INFO] Downloading PaddlePaddle GPU runtime dependencies (${PADDLE_GPU_CUDA_TRACK})..."
  retry 4 python3 -m pip download \
    --dest "${TARGET_DIR}" \
    --only-binary=:all: \
    --no-deps \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --abi cp311 \
    "${paddle_dep_specs[@]}"
}

# Merge embedding/backend requirements for a single offline bundle.
# hnswlib is downloaded separately as source because cp311 manylinux wheel is not guaranteed.
{
  cat "${EMBED_REQ_FILE}"
  cat "${BACKEND_REQ_FILE}"
} | awk '
  {
    line=$0
    sub(/[ \t]*#.*/, "", line)
    gsub(/^[ \t]+|[ \t]+$/, "", line)
    if (line == "") next
    key=tolower(line)
    if (key ~ /^-r[ \t]/) next
    if (key ~ /^hnswlib([<=>!~ ].*)?$/) next
    if (!(key in seen)) {
      seen[key]=1
      print line
    }
  }
' > "${MERGED_REQ}"

echo "[INFO] Downloading Linux cp311 wheels for embedding + main-backend requirements..."
retry 4 python3 -m pip download \
  --dest "${TARGET_DIR}" \
  --requirement "${MERGED_REQ}" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.11 \
  --abi cp311

download_paddle_gpu_runtime

echo "[INFO] Downloading source package for hnswlib (no universal cp311 wheel)..."
download_hnswlib_source

echo "[INFO] Downloading bootstrap/build tools..."
retry 4 python3 -m pip download \
  --dest "${TARGET_DIR}" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.11 \
  --abi cp311 \
  pip setuptools wheel pybind11 overrides==7.7.0 "${PYINSTALLER_PIN}"

TORCH_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'torch-*.whl' | wc -l | tr -d ' ')"
TRANSFORMERS_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'transformers-*.whl' | wc -l | tr -d ' ')"
TOKENIZERS_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'tokenizers-*.whl' | wc -l | tr -d ' ')"
SAFETENSORS_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'safetensors-*.whl' | wc -l | tr -d ' ')"
OCR_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'paddleocr-*.whl' | wc -l | tr -d ' ')"
PADDLE_GPU_RUNTIME_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f \( -name 'paddlepaddle_gpu-*.whl' -o -name 'paddlepaddle-gpu-*.whl' \) | wc -l | tr -d ' ')"
PYMUPDF_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f \( -name 'PyMuPDF-*.whl' -o -name 'pymupdf-*.whl' \) | wc -l | tr -d ' ')"
HNSW_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'hnswlib-*.tar.gz' | wc -l | tr -d ' ')"
PYINSTALLER_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'pyinstaller-*.whl' | wc -l | tr -d ' ')"
PYDANTIC_AI_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'pydantic_ai_slim-*.whl' | wc -l | tr -d ' ')"
PYDANTIC_GRAPH_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'pydantic_graph-*.whl' | wc -l | tr -d ' ')"
GENAI_PRICES_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'genai_prices-*.whl' | wc -l | tr -d ' ')"
GRIFFELIB_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'griffelib-*.whl' | wc -l | tr -d ' ')"
OTEL_API_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'opentelemetry_api-*.whl' | wc -l | tr -d ' ')"
DISTRO_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'distro-*.whl' | wc -l | tr -d ' ')"
IMPORTLIB_METADATA_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'importlib_metadata-*.whl' | wc -l | tr -d ' ')"
LOGFIRE_API_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'logfire_api-*.whl' | wc -l | tr -d ' ')"
ZIPP_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'zipp-*.whl' | wc -l | tr -d ' ')"
EINOPS_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'einops-*.whl' | wc -l | tr -d ' ')"
SENTENCEPIECE_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'sentencepiece-*.whl' | wc -l | tr -d ' ')"
TIKTOKEN_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'tiktoken-*.whl' | wc -l | tr -d ' ')"
FTFY_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'ftfy-*.whl' | wc -l | tr -d ' ')"
PREMAILER_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'premailer-*.whl' | wc -l | tr -d ' ')"
LXML_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'lxml-*.whl' | wc -l | tr -d ' ')"
PYTHON_HWPX_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f -name 'python_hwpx-*.whl' | wc -l | tr -d ' ')"

if [[ "${TORCH_COUNT}" == "0" ]]; then
  echo "[ERROR] torch wheel was not downloaded into ${TARGET_DIR}" >&2
  exit 1
fi

if [[ "${TRANSFORMERS_COUNT}" == "0" || "${TOKENIZERS_COUNT}" == "0" || "${SAFETENSORS_COUNT}" == "0" ]]; then
  echo "[ERROR] Qwen embedding offline dependencies are incomplete in ${TARGET_DIR}" >&2
  echo "        Required wheels: transformers, tokenizers, safetensors" >&2
  exit 1
fi

if [[ "${OCR_COUNT}" == "0" ]]; then
  echo "[ERROR] paddleocr wheel was not downloaded into ${TARGET_DIR}" >&2
  exit 1
fi

if [[ "${PADDLE_GPU_RUNTIME_COUNT}" == "0" ]]; then
  echo "[ERROR] PaddlePaddle GPU runtime wheel was not downloaded into ${TARGET_DIR}" >&2
  echo "        Expected ${PADDLE_GPU_WHL_FILE} from ${PADDLE_GPU_WHL_URL}" >&2
  exit 1
fi

if [[ "${PYMUPDF_COUNT}" == "0" ]]; then
  echo "[ERROR] PyMuPDF wheel was not downloaded into ${TARGET_DIR}" >&2
  exit 1
fi

extract_pymupdf_fallback_tree

if [[ ! -f "${PYMUPDF_FALLBACK_DIR}/fitz/__init__.py" || ! -f "${PYMUPDF_FALLBACK_DIR}/pymupdf/_mupdf.so" || ! -f "${PYMUPDF_FALLBACK_DIR}/pymupdf/libmupdf.so.26.1" ]]; then
  echo "[ERROR] PyMuPDF fallback tree is incomplete in ${PYMUPDF_FALLBACK_DIR}" >&2
  exit 1
fi

if [[ "${HNSW_COUNT}" == "0" ]]; then
  echo "[ERROR] hnswlib source package was not downloaded into ${TARGET_DIR}" >&2
  exit 1
fi

if [[ "${PYINSTALLER_COUNT}" == "0" ]]; then
  echo "[ERROR] pyinstaller wheel was not downloaded into ${TARGET_DIR}" >&2
  exit 1
fi

if [[ "${PYDANTIC_AI_COUNT}" == "0" || "${PYDANTIC_GRAPH_COUNT}" == "0" || "${GENAI_PRICES_COUNT}" == "0" || "${GRIFFELIB_COUNT}" == "0" || "${OTEL_API_COUNT}" == "0" || "${DISTRO_COUNT}" == "0" || "${IMPORTLIB_METADATA_COUNT}" == "0" || "${LOGFIRE_API_COUNT}" == "0" || "${ZIPP_COUNT}" == "0" ]]; then
  echo "[ERROR] PydanticAI offline dependencies are incomplete in ${TARGET_DIR}" >&2
  echo "        Required wheels: pydantic_ai_slim, pydantic_graph, genai_prices, griffelib, opentelemetry_api, distro, importlib_metadata, logfire_api, zipp" >&2
  exit 1
fi

if [[ "${EINOPS_COUNT}" == "0" || "${SENTENCEPIECE_COUNT}" == "0" || "${TIKTOKEN_COUNT}" == "0" || "${FTFY_COUNT}" == "0" || "${PREMAILER_COUNT}" == "0" || "${LXML_COUNT}" == "0" ]]; then
  echo "[ERROR] PaddleOCR-VL doc-parser offline dependencies are incomplete in ${TARGET_DIR}" >&2
  echo "        Required wheels: einops, sentencepiece, tiktoken, ftfy, premailer, lxml" >&2
  exit 1
fi

if [[ "${PYTHON_HWPX_COUNT}" == "0" ]]; then
  echo "[ERROR] HWPX parser offline dependency is incomplete in ${TARGET_DIR}" >&2
  echo "        Required wheel: python_hwpx" >&2
  exit 1
fi

cp "${EMBED_REQ_FILE}" "${TARGET_DIR}/requirements.embedding.txt"
cp "${BACKEND_REQ_FILE}" "${TARGET_DIR}/requirements.backend.txt"
cp "${MERGED_REQ}" "${TARGET_DIR}/requirements.merged.txt"

PAYLOAD_COUNT="$(find "${TARGET_DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')"

echo "[OK] Prepared ${TARGET_TAG}: ${PAYLOAD_COUNT} payload files"
echo "[OK] Manifests: requirements.embedding.txt, requirements.backend.txt, requirements.merged.txt"
