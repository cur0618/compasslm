#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_DEPLOY_DIR="${PADDLEOCR_HPS_DEPLOY_DIR:-${SCRIPT_DIR}/runtime}"
HPS_ENV_FILE="${PADDLEOCR_HPS_ENV_FILE:-${SCRIPT_DIR}/hps.env}"
HPS_URL="${PDF_OCR_HPS_URL:-http://127.0.0.1:8080}"
READY_TIMEOUT="${PDF_OCR_HPS_READY_TIMEOUT_SECONDS:-600}"

compose() {
  [[ -d "${HPS_DEPLOY_DIR}" ]] || {
    echo "[ERROR] HPS deploy directory not found: ${HPS_DEPLOY_DIR}" >&2
    exit 2
  }
  local env_args=()
  [[ -f "${HPS_ENV_FILE}" ]] && env_args=(--env-file "${HPS_ENV_FILE}")
  docker compose "${env_args[@]}" -f "${HPS_DEPLOY_DIR}/docker-compose.yml" "$@"
}

ready() {
  curl -fsS "${HPS_URL%/}/health/ready" >/dev/null
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  until ready; do
    (( SECONDS < deadline )) || {
      echo "[ERROR] PaddleOCR HPS readiness timeout: ${HPS_URL%/}/health/ready" >&2
      return 1
    }
    sleep 2
  done
  echo "[READY] ${HPS_URL%/}/health/ready"
}

case "${1:-}" in
  start)
    compose up -d
    wait_ready
    ;;
  stop)
    compose down
    ;;
  ready)
    ready
    ;;
  wait-ready)
    wait_ready
    ;;
  logs)
    shift
    compose logs "$@"
    ;;
  *)
    echo "Usage: $0 {start|stop|ready|wait-ready|logs}" >&2
    exit 2
    ;;
esac
