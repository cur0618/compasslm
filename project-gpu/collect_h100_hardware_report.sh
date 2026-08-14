#!/usr/bin/env bash
set -u

# RHEL command-not-found/PackageKit hooks may try to access broken yum repos when
# an optional command is missing. Restrict this probe to commands that already
# exist and never invoke package installation or repository metadata refreshes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_FILE="${1:-${PWD}/h100_hardware_report_$(date +%Y%m%d_%H%M%S).txt}"

run_section() {
  local title="$1"
  shift
  echo
  echo "=== ${title} ==="
  "$@" 2>&1 || echo "[WARN] Command failed or is unavailable: $*"
}

run_if_available() {
  local title="$1"
  local command_name="$2"
  shift 2
  echo
  echo "=== ${title} ==="
  if [[ -x "$(type -P "${command_name}" 2>/dev/null || true)" ]]; then
    "$@" 2>&1 || echo "[WARN] Command failed: $*"
  else
    echo "[MISS] ${command_name} is not installed"
  fi
}

collect_report() {
  echo "CompassLM H100 hardware report"
  echo "Generated: $(date 2>/dev/null || true)"
  echo "Host: $(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)"
  echo "COMPASSLM_HOME=${COMPASSLM_HOME}"

  run_section "OS" sh -c 'cat /etc/os-release 2>/dev/null || uname -a'
  run_section "KERNEL AND ARCHITECTURE" uname -a
  run_section "CPU" sh -c 'sed -n "1,40p" /proc/cpuinfo 2>/dev/null || true'
  run_section "MEMORY" sh -c 'sed -n "1,20p" /proc/meminfo 2>/dev/null || true'
  run_section "DISK" df -h

  run_if_available "NVIDIA SMI" nvidia-smi nvidia-smi
  run_if_available "GPU SUMMARY" nvidia-smi nvidia-smi \
    --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total,memory.free \
    --format=csv
  run_if_available "CUDA TOOLKIT" nvcc nvcc --version

  run_if_available "DOCKER VERSION" docker docker version
  run_if_available "DOCKER COMPOSE" docker docker compose version
  run_if_available "DOCKER INFO" docker docker info

  run_if_available "NVIDIA CONTAINER TOOLKIT VERSION" nvidia-container-cli \
    nvidia-container-cli --version

  run_if_available "PYTHON" python3 python3 --version

  echo
  echo "=== OCR MODEL DIRECTORIES ==="
  for model_root in \
    "${COMPASSLM_HOME}/project-gpu/main-backend/models/ocr" \
    "${COMPASSLM_HOME}/models/ocr"
  do
    if [[ -d "${model_root}" ]]; then
      echo "[FOUND] ${model_root}"
      find "${model_root}" -maxdepth 2 -type f \
        -printf '%P | %s bytes\n' 2>/dev/null | sort | head -n 200
    else
      echo "[MISS] ${model_root}"
    fi
  done

  echo
  echo "=== EXISTING HPS ASSETS ==="
  hps_root="${COMPASSLM_HOME}/project-gpu/paddleocr-hps"
  if [[ -d "${hps_root}" ]]; then
    find "${hps_root}" -maxdepth 4 -type f \
      -printf '%P | %s bytes\n' 2>/dev/null | sort | head -n 300
  else
    echo "[MISS] ${hps_root}"
  fi
}

mkdir -p "$(dirname "${OUTPUT_FILE}")"
collect_report | tee "${OUTPUT_FILE}"
echo
echo "[READY] Report saved: ${OUTPUT_FILE}"
