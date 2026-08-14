#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${ROOT_DIR}/offline_packages/cp311-linux_x86_64"
REQ_FILE_DEFAULT="${ROOT_DIR}/requirements.txt"
REQ_FILE_BUNDLE="${PKG_DIR}/requirements.embedding.txt"
REQ_FILE="${REQ_FILE_DEFAULT}"
VENV_DIR="${ROOT_DIR}/compassvenv"
PYTHON_CMD="${PYTHON_CMD:-}"
JUPYTER_PYTHON="${JUPYTER_PYTHON:-python3}"
RECREATE=0
JUPYTER_USER_PROXY=0
JUPYTER_ONLY=0

resolve_requirements_file() {
  if [[ -f "${REQ_FILE_BUNDLE}" ]]; then
    printf '%s\n' "${REQ_FILE_BUNDLE}"
    return 0
  fi
  printf '%s\n' "${REQ_FILE_DEFAULT}"
}

usage() {
  cat <<'EOF'
Usage:
  ./install_offline_linux.sh [options]

Options:
  --python <cmd>           Python command override (default: python3.11, fallback: python3)
  --recreate               Remove existing compassvenv and create it again
  --jupyter-user-proxy     Install jupyter-server-proxy into Jupyter Python user-site and enable it
  --jupyter-python <cmd>   Jupyter Python command/path (default: python3)
  --jupyter-only           Skip compassvenv install and run only Jupyter proxy step
  -h, --help               Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_CMD="${2:-}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --jupyter-user-proxy)
      JUPYTER_USER_PROXY=1
      shift
      ;;
    --jupyter-python)
      JUPYTER_PYTHON="${2:-}"
      shift 2
      ;;
    --jupyter-only)
      JUPYTER_ONLY=1
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

if [[ ! -d "${PKG_DIR}" ]]; then
  echo "[ERROR] Offline package directory not found: ${PKG_DIR}" >&2
  echo "Run ./prepare_offline_packages_linux.sh on an online machine first." >&2
  exit 1
fi

REQ_FILE="$(resolve_requirements_file)"

ensure_py311() {
  local python_cmd="$1"
  local label="$2"
  if ! "${python_cmd}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) == (3, 11) else 1)
PY
  then
    echo "[ERROR] ${label} must be Python 3.11 for cp311-linux_x86_64 package set." >&2
    echo "Detected interpreter: $("${python_cmd}" --version 2>&1)" >&2
    exit 1
  fi
}

install_jupyter_user_proxy() {
  if ! command -v "${JUPYTER_PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Jupyter Python command not found: ${JUPYTER_PYTHON}" >&2
    exit 1
  fi

  ensure_py311 "${JUPYTER_PYTHON}" "Jupyter Python"
  echo "[INFO] Jupyter Python: $("${JUPYTER_PYTHON}" --version 2>&1)"
  echo "[INFO] Installing jupyter-server-proxy to user-site..."
  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null \
    "${JUPYTER_PYTHON}" -m pip install --user --no-index --find-links="${PKG_DIR}" \
      jupyter-server-proxy==4.4.0 simpervisor==1.0.0 overrides==7.7.0

  "${JUPYTER_PYTHON}" - <<'PY'
import site
import jupyter_server_proxy
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
print("USER_SITE", site.getusersitepackages())
print("jupyter_server_proxy", jupyter_server_proxy.__file__)
PY

  echo "[INFO] Enabling server extension for user..."
  if ! "${JUPYTER_PYTHON}" -m jupyter server extension enable jupyter_server_proxy --user; then
    echo "[WARN] Extension validation failed. Writing user config fallback."
    mkdir -p "${HOME}/.jupyter/jupyter_server_config.d"
    cat > "${HOME}/.jupyter/jupyter_server_config.d/jupyter_server_proxy.json" <<'JSON'
{"ServerApp":{"jpserver_extensions":{"jupyter_server_proxy": true}}}
JSON
  fi
  "${JUPYTER_PYTHON}" -m jupyter server extension list | sed -n '/jupyter_server_proxy/,+5p' || true
  echo "[OK] Jupyter user-site proxy setup completed. Restart Jupyter user server."
}

if [[ "${JUPYTER_ONLY}" != "1" ]]; then
  if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[ERROR] requirements file not found: ${REQ_FILE}" >&2
    exit 1
  fi

  if [[ -z "${PYTHON_CMD}" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
      PYTHON_CMD="python3.11"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_CMD="python3"
    else
      echo "[ERROR] python3.11 (or python3) command not found." >&2
      exit 1
    fi
  fi

  ensure_py311 "${PYTHON_CMD}" "Python"
  if ! "${PYTHON_CMD}" -m venv --help >/dev/null 2>&1; then
    echo "[ERROR] python3.11-venv is missing. Install it first (ex: sudo apt install python3.11-venv)." >&2
    exit 1
  fi

  echo "[INFO] Python: $("${PYTHON_CMD}" --version 2>&1)"
  echo "[INFO] Package dir: ${PKG_DIR}"
  echo "[INFO] Requirements: ${REQ_FILE}"
  if [[ "${RECREATE}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "[INFO] Recreating venv: ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
  fi
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[INFO] Creating venv: ${VENV_DIR}"
    "${PYTHON_CMD}" -m venv "${VENV_DIR}"
  fi

  "${VENV_DIR}/bin/python" -m ensurepip --upgrade
  if [[ -n "${PIP_USER:-}" ]]; then
    echo "[WARN] PIP_USER is set in environment (${PIP_USER}). Ignoring it for venv install."
  fi
  if [[ -n "${PYTHONUSERBASE:-}" ]]; then
    echo "[WARN] PYTHONUSERBASE is set in environment (${PYTHONUSERBASE}). Ignoring it for venv install."
  fi

  env -u PIP_USER -u PYTHONUSERBASE PIP_CONFIG_FILE=/dev/null PIP_USER=0 \
    "${VENV_DIR}/bin/python" -m pip install --no-user --no-index --find-links="${PKG_DIR}" -r "${REQ_FILE}"

  echo "[INFO] Verifying imports..."
  "${VENV_DIR}/bin/python" - <<'PY'
import fastapi
import sentence_transformers
import torch
import transformers
import uvicorn
import jupyter_server_proxy
import simpervisor
from importlib.metadata import version

print("fastapi", fastapi.__version__)
print("sentence_transformers", sentence_transformers.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("uvicorn", uvicorn.__version__)
print("jupyter-server-proxy", version("jupyter-server-proxy"))
print("simpervisor", version("simpervisor"))
PY

  echo "[OK] Offline install and import verification completed."
else
  echo "[INFO] --jupyter-only set: skipping embedding compassvenv install."
fi

if [[ "${JUPYTER_USER_PROXY}" == "1" ]]; then
  install_jupyter_user_proxy
fi
