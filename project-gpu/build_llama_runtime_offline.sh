#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_gpu_env.sh"

SRC_TAR="${SRC_TAR:-${PROJECT_GPU_HOME}/offline_assets/llama.cpp-b9060-src.tar.gz}"
SRC_DIR="${SRC_DIR:-}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_GPU_HOME}/.build/llama-offline}"
JOBS="${JOBS:-$(nproc)}"
CUDA="${CUDA:-on}" # on|off|auto
LLAMA_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES:-90}"
KEEP_BUILD="${KEEP_BUILD:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_GPU_HOME}/.build/logs}"
CMAKE_BIN="${CMAKE_BIN:-}"
CCACHE_BIN="${CCACHE_BIN:-}"
SELECTED_CC="${CC:-}"
SELECTED_CXX="${CXX:-}"
NEED_STDCXXFS=0
STDCXXFS_LIB="${STDCXXFS_LIB:-}"
STDCXXFS_LINK_TOKEN=""
STDCXXFS_SEARCH_DIR="${STDCXXFS_SEARCH_DIR:-${PROJECT_GPU_HOME}/offline_assets/toolchain}"

usage() {
  cat <<'EOF'
Usage:
  project-gpu/build_llama_runtime_offline.sh [options]

Options:
  --src-tar <path>   Path to llama.cpp source tarball (default: project-gpu/offline_assets/llama.cpp-b9060-src.tar.gz)
  --src-dir <path>   Path to extracted llama.cpp source directory (overrides --src-tar)
  --build-dir <path> Build directory (default: project-gpu/.build/llama-offline)
  --jobs <N>         Parallel build jobs (default: nproc)
  --cuda <on|off|auto>  Build with CUDA (default: on; auto may fall back to CPU)
  --cuda-arch <arch> CUDA architectures for llama.cpp CUDA builds (default: 90 for H100)
  --stdcxxfs-lib <path> Path to libstdc++fs.a/.so for GCC8 filesystem link
  --ccache-bin <path>   Path to ccache binary (default: auto-detect)
  --keep-build       Keep build directory
  -h, --help         Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-tar)
      SRC_TAR="${2:-}"
      shift 2
      ;;
    --src-dir)
      SRC_DIR="${2:-}"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --cuda)
      CUDA="${2:-}"
      shift 2
      ;;
    --cuda-arch)
      LLAMA_CUDA_ARCHITECTURES="${2:-}"
      shift 2
      ;;
    --stdcxxfs-lib)
      STDCXXFS_LIB="${2:-}"
      shift 2
      ;;
    --ccache-bin)
      CCACHE_BIN="${2:-}"
      shift 2
      ;;
    --keep-build)
      KEEP_BUILD=1
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

if [[ -z "${CMAKE_BIN}" ]]; then
  if [[ -x "${PROJECT_GPU_HOME}/tools/cmake/bin/cmake" ]]; then
    CMAKE_BIN="${PROJECT_GPU_HOME}/tools/cmake/bin/cmake"
  elif command -v cmake >/dev/null 2>&1; then
    CMAKE_BIN="$(command -v cmake)"
  fi
fi
if [[ -z "${CMAKE_BIN}" || ! -x "${CMAKE_BIN}" ]]; then
  echo "[ERROR] cmake is required. Put portable cmake at ${PROJECT_GPU_HOME}/tools/cmake/bin/cmake or set CMAKE_BIN." >&2
  exit 1
fi

if [[ -z "${SELECTED_CXX}" ]]; then
  for cand in g++-14 g++-13 g++-12 g++-11 g++-10 g++-9 g++-8 g++; do
    if command -v "${cand}" >/dev/null 2>&1; then
      SELECTED_CXX="$(command -v "${cand}")"
      break
    fi
  done
fi
if [[ -z "${SELECTED_CXX}" ]]; then
  echo "[ERROR] g++ is required for build." >&2
  exit 1
fi

if [[ -z "${SELECTED_CC}" ]]; then
  cxx_base="$(basename "${SELECTED_CXX}")"
  cc_cand="${cxx_base/g++/gcc}"
  if command -v "${cc_cand}" >/dev/null 2>&1; then
    SELECTED_CC="$(command -v "${cc_cand}")"
  elif command -v gcc >/dev/null 2>&1; then
    SELECTED_CC="$(command -v gcc)"
  fi
fi
if [[ -z "${SELECTED_CC}" ]]; then
  echo "[ERROR] gcc is required for build." >&2
  exit 1
fi

if [[ -n "${CCACHE_BIN}" && ! -x "${CCACHE_BIN}" ]]; then
  echo "[ERROR] --ccache-bin is not executable: ${CCACHE_BIN}" >&2
  exit 1
fi
if [[ -z "${CCACHE_BIN}" ]]; then
  if [[ -x "${MAIN_BACKEND_HOME}/runtime/ccache" ]]; then
    CCACHE_BIN="${MAIN_BACKEND_HOME}/runtime/ccache"
  elif [[ -x "${MAIN_BACKEND_HOME}/compassvenv/bin/ccache" ]]; then
    CCACHE_BIN="${MAIN_BACKEND_HOME}/compassvenv/bin/ccache"
  elif command -v ccache >/dev/null 2>&1; then
    CCACHE_BIN="$(command -v ccache)"
  fi
fi

cxx_major="$("${SELECTED_CXX}" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
if [[ "${cxx_major}" =~ ^[0-9]+$ ]] && (( cxx_major < 9 )); then
  NEED_STDCXXFS=1
fi

resolve_stdcxxfs_link_token() {
  local cand=""
  local resolved=""

  if [[ -n "${STDCXXFS_LIB}" ]]; then
    if [[ ! -f "${STDCXXFS_LIB}" ]]; then
      echo "[ERROR] --stdcxxfs-lib file not found: ${STDCXXFS_LIB}" >&2
      return 1
    fi
    resolved="$(cd "$(dirname "${STDCXXFS_LIB}")" && pwd)/$(basename "${STDCXXFS_LIB}")"
    STDCXXFS_LINK_TOKEN="${resolved}"
    return 0
  fi

  for cand in \
    "${STDCXXFS_SEARCH_DIR}/libstdc++fs.a" \
    "${STDCXXFS_SEARCH_DIR}/libstdc++fs.so" \
    "${PROJECT_GPU_HOME}/offline_assets/libstdc++fs.a" \
    "${PROJECT_GPU_HOME}/offline_assets/libstdc++fs.so"; do
    if [[ -f "${cand}" ]]; then
      resolved="$(cd "$(dirname "${cand}")" && pwd)/$(basename "${cand}")"
      STDCXXFS_LINK_TOKEN="${resolved}"
      return 0
    fi
  done

  cand="$("${SELECTED_CXX}" -print-file-name=libstdc++fs.a 2>/dev/null || true)"
  if [[ -n "${cand}" && "${cand}" != "libstdc++fs.a" && -f "${cand}" ]]; then
    STDCXXFS_LINK_TOKEN="${cand}"
    return 0
  fi

  cand="$("${SELECTED_CXX}" -print-file-name=libstdc++fs.so 2>/dev/null || true)"
  if [[ -n "${cand}" && "${cand}" != "libstdc++fs.so" && -f "${cand}" ]]; then
    STDCXXFS_LINK_TOKEN="${cand}"
    return 0
  fi

  STDCXXFS_LINK_TOKEN="-lstdc++fs"
  return 0
}

if [[ "${NEED_STDCXXFS}" == "1" ]]; then
  resolve_stdcxxfs_link_token
fi

RUNTIME_DIR="${MAIN_BACKEND_HOME}/runtime"
SRC_WORK_DIR=""

if [[ -n "${SRC_DIR}" ]]; then
  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[ERROR] Source dir not found: ${SRC_DIR}" >&2
    exit 1
  fi
  SRC_WORK_DIR="$(cd "${SRC_DIR}" && pwd)"
else
  if [[ ! -f "${SRC_TAR}" ]]; then
    echo "[ERROR] Source tar not found: ${SRC_TAR}" >&2
    exit 1
  fi
  EXTRACT_ROOT="${PROJECT_GPU_HOME}/.build/src-extract"
  rm -rf "${EXTRACT_ROOT}"
  mkdir -p "${EXTRACT_ROOT}"
  tar -xf "${SRC_TAR}" -C "${EXTRACT_ROOT}"
  SRC_WORK_DIR="$(find "${EXTRACT_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  if [[ -z "${SRC_WORK_DIR}" ]]; then
    echo "[ERROR] Could not locate extracted source root." >&2
    exit 1
  fi
fi

if [[ ! -f "${SRC_WORK_DIR}/CMakeLists.txt" ]]; then
  echo "[ERROR] Not a valid llama.cpp source dir: ${SRC_WORK_DIR}" >&2
  exit 1
fi

GGML_CUDA_FLAG=OFF
case "${CUDA}" in
  on)
    GGML_CUDA_FLAG=ON
    ;;
  off)
    GGML_CUDA_FLAG=OFF
    ;;
  auto)
    if command -v nvcc >/dev/null 2>&1; then
      GGML_CUDA_FLAG=ON
    fi
    ;;
  *)
    echo "[ERROR] --cuda must be auto|on|off" >&2
    exit 1
    ;;
esac

echo "[INFO] SRC_WORK_DIR=${SRC_WORK_DIR}"
echo "[INFO] BUILD_DIR=${BUILD_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] RUNTIME_DIR=${RUNTIME_DIR}"
echo "[INFO] Requested CUDA mode=${CUDA}"
echo "[INFO] Initial GGML_CUDA=${GGML_CUDA_FLAG}"
echo "[INFO] LLAMA_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCHITECTURES}"
echo "[INFO] CMAKE_BIN=${CMAKE_BIN}"
"${CMAKE_BIN}" --version | head -n 1 || true
echo "[INFO] CC=${SELECTED_CC}"
echo "[INFO] CXX=${SELECTED_CXX}"
echo "[INFO] CCACHE_BIN=${CCACHE_BIN:-<disabled>}"
"${SELECTED_CC}" --version | head -n 1 || true
"${SELECTED_CXX}" --version | head -n 1 || true
echo "[INFO] NEED_STDCXXFS=${NEED_STDCXXFS}"
if [[ "${NEED_STDCXXFS}" == "1" ]]; then
  echo "[INFO] STDCXXFS_SEARCH_DIR=${STDCXXFS_SEARCH_DIR}"
  echo "[INFO] STDCXXFS_LINK_TOKEN=${STDCXXFS_LINK_TOKEN}"
fi
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -n 1; fi

mkdir -p "${LOG_DIR}"
CONFIG_LOG="${LOG_DIR}/llama_offline_configure.log"
BUILD_LOG="${LOG_DIR}/llama_offline_build.log"

configure_and_build() {
  local cuda_flag="$1"
  local -a cmake_args
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"

  cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA="${cuda_flag}"
    -DGGML_NATIVE=ON
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DCMAKE_CXX_STANDARD=17
  )
  if [[ "${cuda_flag}" == "ON" ]]; then
    cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES}")
  fi
  if [[ "${NEED_STDCXXFS}" == "1" ]]; then
    cmake_args+=(
      "-DCMAKE_EXE_LINKER_FLAGS=${STDCXXFS_LINK_TOKEN}"
      "-DCMAKE_SHARED_LINKER_FLAGS=${STDCXXFS_LINK_TOKEN}"
      "-DCMAKE_MODULE_LINKER_FLAGS=${STDCXXFS_LINK_TOKEN}"
      "-DCMAKE_CXX_STANDARD_LIBRARIES=${STDCXXFS_LINK_TOKEN}"
    )
  fi
  if [[ -n "${CCACHE_BIN}" ]]; then
    cmake_args+=(
      "-DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_BIN}"
      "-DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_BIN}"
    )
  fi

  echo "[INFO] Configuring (GGML_CUDA=${cuda_flag}) ..."
  if ! CC="${SELECTED_CC}" CXX="${SELECTED_CXX}" \
    "${CMAKE_BIN}" -S "${SRC_WORK_DIR}" -B "${BUILD_DIR}" \
    "${cmake_args[@]}" >"${CONFIG_LOG}" 2>&1; then
    echo "[ERROR] CMake configure failed (GGML_CUDA=${cuda_flag})." >&2
    echo "[INFO] Configure log: ${CONFIG_LOG}" >&2
    tail -n 80 "${CONFIG_LOG}" >&2 || true
    return 1
  fi

  echo "[INFO] Building llama-server (GGML_CUDA=${cuda_flag}) ..."
  if ! "${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${JOBS}" --target llama-server >"${BUILD_LOG}" 2>&1; then
    echo "[ERROR] Build failed (GGML_CUDA=${cuda_flag})." >&2
    echo "[INFO] Build log: ${BUILD_LOG}" >&2
    tail -n 80 "${BUILD_LOG}" >&2 || true
    return 2
  fi

  return 0
}

verify_cuda_runtime_bundle() {
  local runtime_dir="$1"
  local cuda_lib="${runtime_dir}/libggml-cuda.so"
  local cuobjdump_out="${LOG_DIR}/llama_offline_cuobjdump_list_elf.log"
  if [[ ! -s "${cuda_lib}" ]]; then
    echo "[ERROR] CUDA build did not install ${cuda_lib}." >&2
    echo "        This runtime will fail on the GPU server or run without CUDA." >&2
    echo "        Check ${CONFIG_LOG} and ${BUILD_LOG}; look for GGML_CUDA=ON and CMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCHITECTURES}." >&2
    return 1
  fi
  if command -v cuobjdump >/dev/null 2>&1; then
    if ! cuobjdump --list-elf "${cuda_lib}" >"${cuobjdump_out}" 2>&1; then
      echo "[ERROR] cuobjdump failed while inspecting ${cuda_lib}." >&2
      echo "        cuobjdump log: ${cuobjdump_out}" >&2
      tail -n 80 "${cuobjdump_out}" >&2 || true
      return 1
    fi
    if ! grep -Eq "(sm|compute|lto)_90|arch[ =:]+90" "${cuobjdump_out}"; then
      echo "[ERROR] ${cuda_lib} does not appear to contain H100 sm_90/compute_90 CUDA code." >&2
      echo "        Rebuild with --cuda on --cuda-arch 90 on the H100 CUDA toolkit host." >&2
      echo "        cuobjdump log: ${cuobjdump_out}" >&2
      echo "        configure log: ${CONFIG_LOG}" >&2
      echo "        build log: ${BUILD_LOG}" >&2
      echo "[INFO] CUDA architecture evidence from configure log:" >&2
      grep -En "CMAKE_CUDA_ARCHITECTURES|GGML_CUDA|CUDA Toolkit|CUDA compiler" "${CONFIG_LOG}" >&2 || true
      echo "[INFO] CUDA architecture evidence from build log:" >&2
      grep -En "arch=compute_90|code=sm_90|compute_90|sm_90|lto_90|CMAKE_CUDA_ARCHITECTURES" "${BUILD_LOG}" >&2 || true
      echo "[INFO] cuobjdump output preview:" >&2
      head -n 80 "${cuobjdump_out}" >&2 || true
      return 1
    fi
  else
    echo "[WARN] cuobjdump not found; verified CUDA backend file exists but could not inspect sm_90 code."
  fi
  return 0
}

BUILD_MODE_USED=""
if [[ "${CUDA}" == "auto" && "${GGML_CUDA_FLAG}" == "ON" ]]; then
  if configure_and_build ON; then
    BUILD_MODE_USED="cuda"
  else
    echo "[WARN] CUDA auto build failed. Falling back to CPU build (GGML_CUDA=OFF)." >&2
    configure_and_build OFF
    BUILD_MODE_USED="cpu-fallback"
  fi
elif [[ "${GGML_CUDA_FLAG}" == "ON" ]]; then
  configure_and_build ON
  BUILD_MODE_USED="cuda"
else
  configure_and_build OFF
  BUILD_MODE_USED="cpu"
fi

echo "[INFO] Build mode used: ${BUILD_MODE_USED}"

BUILT_BIN="$(find "${BUILD_DIR}" -type f -name llama-server -perm -u+x | sort | head -n 1 || true)"
if [[ -z "${BUILT_BIN}" ]]; then
  echo "[ERROR] Build succeeded but llama-server binary not found." >&2
  exit 1
fi

BUILT_DIR="$(dirname "${BUILT_BIN}")"
BACKUP_DIR="${MAIN_BACKEND_HOME}/runtime.backup.$(date +%Y%m%d_%H%M%S)"
if [[ -d "${RUNTIME_DIR}" ]]; then
  mv "${RUNTIME_DIR}" "${BACKUP_DIR}"
  echo "[INFO] Backup runtime: ${BACKUP_DIR}"
fi

mkdir -p "${RUNTIME_DIR}"
cp -a "${BUILT_DIR}/." "${RUNTIME_DIR}/"
chmod +x "${RUNTIME_DIR}/llama-server"

if [[ "${BUILD_MODE_USED}" == "cuda" ]]; then
  verify_cuda_runtime_bundle "${RUNTIME_DIR}"
fi

echo "[OK] Installed new runtime into: ${RUNTIME_DIR}"
"${RUNTIME_DIR}/llama-server" --version || true

if [[ "${KEEP_BUILD}" != "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi
