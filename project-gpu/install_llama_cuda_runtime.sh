#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

API_BASE_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases"
MODE="auto"
TAG="latest"
CUDA_VERSION=""
ASSET_NAME=""
RUNTIME_DIR="${MAIN_BACKEND_HOME}/runtime"
BUILD_ROOT="${COMPASSLM_HOME}/.cache/llama-cuda-build"
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
LLAMA_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES:-90}"
SET_ENV=1
KEEP_BUILD=0
FORCE=0
CCACHE_BIN="${CCACHE_BIN:-}"

RESOLVED_TAG=""
RELEASE_JSON_FILE=""

usage() {
  cat <<'EOF'
Usage:
  ./project-gpu/install_llama_cuda_runtime.sh [options]

Description:
  Install/replace Linux llama-server runtime with CUDA build.
  Default mode is:
    1) try official release CUDA binary
    2) fallback to local source build (GGML_CUDA=ON)

Options:
  --mode <auto|release|build>  install mode (default: auto)
  --tag <release-tag|latest>   llama.cpp tag (default: latest)
  --cuda <12.4|13.1|...>       preferred CUDA version for release asset selection
  --asset <asset-name>         exact release asset name to download
  --runtime-dir <path>         runtime install dir (default: project-gpu/main-backend/runtime)
  --build-root <path>          source build workspace (default: .cache/llama-cuda-build)
  --jobs <n>                   build parallel jobs (default: nproc)
  --cuda-arch <arch>           CUDA architectures for source builds (default: 90 for H100)
  --ccache-bin <path>          path to ccache binary for source builds (default: auto-detect)
  --no-set-env                 do not update main-backend/.env LLM_RUNTIME
  --keep-build                 keep build directory after source build
  --force                      overwrite runtime dir directly (without backup)
  -h, --help                   show help

Examples:
  ./project-gpu/install_llama_cuda_runtime.sh
  ./project-gpu/install_llama_cuda_runtime.sh --mode release --cuda 12.4
  ./project-gpu/install_llama_cuda_runtime.sh --mode build --tag b8095 --jobs 16 --cuda-arch 90
EOF
}

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

resolve_ccache_bin() {
  if [[ -n "${CCACHE_BIN}" ]]; then
    [[ -x "${CCACHE_BIN}" ]] || die "--ccache-bin is not executable: ${CCACHE_BIN}"
    return
  fi
  if [[ -x "${MAIN_BACKEND_HOME}/runtime/ccache" ]]; then
    CCACHE_BIN="${MAIN_BACKEND_HOME}/runtime/ccache"
  elif [[ -x "${MAIN_BACKEND_HOME}/compassvenv/bin/ccache" ]]; then
    CCACHE_BIN="${MAIN_BACKEND_HOME}/compassvenv/bin/ccache"
  elif command -v ccache >/dev/null 2>&1; then
    CCACHE_BIN="$(command -v ccache)"
  fi
}

cleanup_release_json() {
  if [[ -n "${RELEASE_JSON_FILE}" && -f "${RELEASE_JSON_FILE}" ]]; then
    rm -f "${RELEASE_JSON_FILE}"
  fi
}

trap cleanup_release_json EXIT

detect_cuda_version() {
  if [[ -n "${CUDA_VERSION}" ]]; then
    return
  fi
  local detected=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 || true)"
  fi
  if [[ -z "${detected}" ]] && command -v nvcc >/dev/null 2>&1; then
    detected="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 || true)"
  fi
  if [[ -n "${detected}" ]]; then
    CUDA_VERSION="${detected}"
    log "Detected CUDA version: ${CUDA_VERSION}"
  else
    warn "CUDA version auto-detect failed. Release asset match will use generic CUDA patterns."
  fi
}

fetch_release_json() {
  require_cmd curl
  require_cmd python3
  RELEASE_JSON_FILE="$(mktemp)"
  local url=""
  if [[ "${TAG}" == "latest" ]]; then
    url="${API_BASE_URL}/latest"
  else
    url="${API_BASE_URL}/tags/${TAG}"
  fi
  log "Fetching release metadata: ${url}"
  if ! curl -fsSL "${url}" > "${RELEASE_JSON_FILE}"; then
    warn "Failed to fetch release metadata from GitHub."
    rm -f "${RELEASE_JSON_FILE}"
    RELEASE_JSON_FILE=""
    RESOLVED_TAG=""
    return 1
  fi
  if ! RESOLVED_TAG="$(python3 - "${RELEASE_JSON_FILE}" <<'PY'
import json,sys
path=sys.argv[1]
with open(path,"r",encoding="utf-8") as f:
    data=json.load(f)
print((data.get("tag_name") or "").strip())
PY
  )"; then
    warn "Failed to parse release metadata JSON."
    rm -f "${RELEASE_JSON_FILE}"
    RELEASE_JSON_FILE=""
    RESOLVED_TAG=""
    return 1
  fi
  if [[ -z "${RESOLVED_TAG}" ]]; then
    warn "Failed to resolve release tag from GitHub response."
    rm -f "${RELEASE_JSON_FILE}"
    RELEASE_JSON_FILE=""
    return 1
  fi
  log "Resolved release tag: ${RESOLVED_TAG}"
  return 0
}

pick_release_asset() {
  python3 - "${RELEASE_JSON_FILE}" "${ASSET_NAME}" "${CUDA_VERSION}" <<'PY'
import json,re,sys

path,asset_override,cuda = sys.argv[1:4]
with open(path,"r",encoding="utf-8") as f:
    rel=json.load(f)
assets = rel.get("assets", [])

def emit(item):
    print(item.get("name",""))
    print(item.get("browser_download_url",""))

if asset_override:
    for item in assets:
        if item.get("name","") == asset_override:
            emit(item)
            raise SystemExit(0)
    raise SystemExit(1)

patterns = []
if cuda:
    esc = re.escape(cuda)
    patterns.extend([
        rf"^llama-.*-bin-ubuntu-cuda-{esc}-x64\.tar\.gz$",
        rf"^llama-.*-bin-linux-cuda-{esc}-x64\.tar\.gz$",
        rf"^llama-.*-bin-ubuntu-cuda-{esc}.*x64.*\.tar\.gz$",
        rf"^llama-.*-bin-linux-cuda-{esc}.*x64.*\.tar\.gz$",
    ])
patterns.extend([
    r"^llama-.*-bin-ubuntu-cuda.*x64.*\.tar\.gz$",
    r"^llama-.*-bin-linux-cuda.*x64.*\.tar\.gz$",
    r"^llama-.*cuda.*ubuntu.*x64.*\.tar\.gz$",
    r"^llama-.*cuda.*linux.*x64.*\.tar\.gz$",
])

for pat in patterns:
    rx = re.compile(pat)
    for item in assets:
        name = item.get("name","")
        if "win" in name.lower():
            continue
        if rx.match(name):
            emit(item)
            raise SystemExit(0)

raise SystemExit(1)
PY
}

backup_or_prepare_runtime_dir() {
  if [[ -d "${RUNTIME_DIR}" && -n "$(ls -A "${RUNTIME_DIR}" 2>/dev/null || true)" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      rm -rf "${RUNTIME_DIR}"
      mkdir -p "${RUNTIME_DIR}"
      return
    fi
    local backup="${RUNTIME_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
    mv "${RUNTIME_DIR}" "${backup}"
    log "Existing runtime moved to backup: ${backup}"
  fi
  mkdir -p "${RUNTIME_DIR}"
}

verify_installed_runtime() {
  local runtime_bin="${RUNTIME_DIR}/llama-server"
  local cuda_lib="${RUNTIME_DIR}/libggml-cuda.so"
  [[ -x "${runtime_bin}" ]] || die "Installed runtime missing executable: ${runtime_bin}"

  if ! file "${runtime_bin}" | grep -q "ELF 64-bit"; then
    die "Installed runtime is not Linux ELF binary: ${runtime_bin}"
  fi

  if [[ -s "${cuda_lib}" ]]; then
    log "CUDA backend library detected: ${cuda_lib}"
    if command -v cuobjdump >/dev/null 2>&1; then
      if cuobjdump --list-elf "${cuda_lib}" 2>/dev/null | grep -Eq "sm_90|compute_90"; then
        log "H100 sm_90/compute_90 CUDA code detected in libggml-cuda.so."
      else
        die "libggml-cuda.so does not appear to contain H100 sm_90/compute_90 CUDA code. Rebuild with --mode build --cuda-arch 90 on the H100 CUDA toolkit host."
      fi
    else
      warn "cuobjdump not found; verified CUDA backend file exists but could not inspect sm_90 code."
    fi
  else
    die "Installed runtime is missing CUDA backend library: ${cuda_lib}"
  fi
}

install_from_release_asset() {
  local asset_name="$1"
  local asset_url="$2"
  [[ -n "${asset_name}" && -n "${asset_url}" ]] || die "Invalid release asset metadata."

  require_cmd curl
  require_cmd tar

  local tmp_root
  tmp_root="$(mktemp -d)"
  local archive_path="${tmp_root}/${asset_name}"
  local extract_dir="${tmp_root}/extract"
  mkdir -p "${extract_dir}"

  log "Downloading release asset: ${asset_name}"
  curl -fL -o "${archive_path}" "${asset_url}"
  tar -xzf "${archive_path}" -C "${extract_dir}"

  local release_bin
  release_bin="$(find "${extract_dir}" -type f -name llama-server -perm -u+x | sort | head -n 1 || true)"
  [[ -n "${release_bin}" ]] || die "llama-server not found in downloaded asset: ${asset_name}"

  local release_dir
  release_dir="$(dirname "${release_bin}")"
  backup_or_prepare_runtime_dir
  cp -a "${release_dir}/." "${RUNTIME_DIR}/"
  chmod +x "${RUNTIME_DIR}/llama-server"
  log "Installed runtime from release asset into: ${RUNTIME_DIR}"
}

install_from_source_build() {
  require_cmd git
  require_cmd cmake
  resolve_ccache_bin

  local repo_dir="${BUILD_ROOT}/llama.cpp"
  local build_dir="${repo_dir}/build-cuda"
  local -a cmake_args

  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${repo_dir}/.git" ]]; then
    log "Cloning llama.cpp source..."
    git clone https://github.com/ggml-org/llama.cpp "${repo_dir}"
  fi

  log "Updating llama.cpp source..."
  git -C "${repo_dir}" fetch --tags --force

  local checkout_ref="${RESOLVED_TAG:-${TAG}}"
  if [[ -n "${checkout_ref}" && "${checkout_ref}" != "latest" ]]; then
    log "Checking out ${checkout_ref}"
    git -C "${repo_dir}" checkout --detach "${checkout_ref}"
  else
    log "Checking out latest main branch"
    git -C "${repo_dir}" checkout main
    git -C "${repo_dir}" pull --ff-only
  fi

  log "Configuring CUDA build..."
  log "Using CMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCHITECTURES}"
  cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    -DGGML_NATIVE=ON
    -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES}"
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
  )
  if [[ -n "${CCACHE_BIN}" ]]; then
    log "Using ccache: ${CCACHE_BIN}"
    cmake_args+=(
      "-DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_BIN}"
      "-DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_BIN}"
    )
  else
    log "Using ccache: disabled"
  fi
  cmake -S "${repo_dir}" -B "${build_dir}" "${cmake_args[@]}"

  log "Building llama-server target..."
  cmake --build "${build_dir}" --config Release --parallel "${JOBS}" --target llama-server

  local built_bin
  built_bin="$(find "${build_dir}" -type f -name llama-server -perm -u+x | sort | head -n 1 || true)"
  [[ -n "${built_bin}" ]] || die "CUDA build succeeded but llama-server binary was not found."

  local built_dir
  built_dir="$(dirname "${built_bin}")"
  backup_or_prepare_runtime_dir
  cp -a "${built_dir}/." "${RUNTIME_DIR}/"
  chmod +x "${RUNTIME_DIR}/llama-server"
  log "Installed runtime from source build into: ${RUNTIME_DIR}"

  if [[ "${KEEP_BUILD}" != "1" ]]; then
    rm -rf "${build_dir}"
    log "Cleaned build directory: ${build_dir}"
  fi
}

update_llm_runtime_env() {
  local env_file="${MAIN_BACKEND_HOME}/.env"
  local runtime_entry='${MAIN_BACKEND_HOME}/runtime/llama-server'

  if [[ ! -f "${env_file}" ]]; then
    printf "LLM_RUNTIME=%s\n" "${runtime_entry}" > "${env_file}"
    log "Created ${env_file} with LLM_RUNTIME."
    return
  fi

  if grep -q "^LLM_RUNTIME=" "${env_file}"; then
    sed -i "s|^LLM_RUNTIME=.*|LLM_RUNTIME=${runtime_entry}|" "${env_file}"
  else
    printf "\nLLM_RUNTIME=%s\n" "${runtime_entry}" >> "${env_file}"
  fi
  log "Updated ${env_file}: LLM_RUNTIME=${runtime_entry}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --cuda)
      CUDA_VERSION="${2:-}"
      shift 2
      ;;
    --asset)
      ASSET_NAME="${2:-}"
      shift 2
      ;;
    --runtime-dir)
      RUNTIME_DIR="${2:-}"
      shift 2
      ;;
    --build-root)
      BUILD_ROOT="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --cuda-arch)
      LLAMA_CUDA_ARCHITECTURES="${2:-}"
      shift 2
      ;;
    --ccache-bin)
      CCACHE_BIN="${2:-}"
      shift 2
      ;;
    --no-set-env)
      SET_ENV=0
      shift
      ;;
    --keep-build)
      KEEP_BUILD=1
      shift
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
      die "Unknown argument: $1"
      ;;
  esac
done

case "${MODE}" in
  auto|release|build)
    ;;
  *)
    die "Invalid --mode: ${MODE}. Use auto|release|build."
    ;;
esac

[[ "${JOBS}" =~ ^[0-9]+$ ]] || die "--jobs must be a positive integer."
RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"
BUILD_ROOT="$(realpath -m "${BUILD_ROOT}")"

if [[ "$(uname -s | tr '[:upper:]' '[:lower:]')" != "linux" ]]; then
  die "This script is for Linux runtime installation only."
fi

log "COMPASSLM_HOME=${COMPASSLM_HOME}"
log "MAIN_BACKEND_HOME=${MAIN_BACKEND_HOME}"
log "RUNTIME_DIR=${RUNTIME_DIR}"
log "MODE=${MODE} TAG=${TAG}"
resolve_ccache_bin
log "CCACHE_BIN=${CCACHE_BIN:-<disabled>}"

detect_cuda_version

installed=0
if [[ "${MODE}" == "auto" || "${MODE}" == "release" ]]; then
  if fetch_release_json; then
    if asset_meta="$(pick_release_asset 2>/dev/null)"; then
      asset_name="$(echo "${asset_meta}" | sed -n '1p')"
      asset_url="$(echo "${asset_meta}" | sed -n '2p')"
      install_from_release_asset "${asset_name}" "${asset_url}"
      installed=1
    else
      if [[ -n "${ASSET_NAME}" ]]; then
        die "Requested asset not found in release ${RESOLVED_TAG}: ${ASSET_NAME}"
      fi
      if [[ "${MODE}" == "release" ]]; then
        die "No Linux CUDA release asset found for tag ${RESOLVED_TAG}."
      fi
      warn "No matching Linux CUDA release asset found. Falling back to source build."
    fi
  else
    if [[ "${MODE}" == "release" ]]; then
      die "Release metadata fetch failed in --mode release."
    fi
    warn "Release lookup skipped/failed. Falling back to source build."
  fi
fi

if [[ "${installed}" == "0" ]]; then
  install_from_source_build
fi

verify_installed_runtime

if [[ "${SET_ENV}" == "1" ]]; then
  update_llm_runtime_env
fi

echo
echo "[OK] CUDA runtime install completed."
echo "Check:"
echo "  ${PROJECT_GPU_HOME}/check_gpu_assets.sh"
echo "Then run:"
echo "  ${PROJECT_GPU_HOME}/run_llm_server.sh"
