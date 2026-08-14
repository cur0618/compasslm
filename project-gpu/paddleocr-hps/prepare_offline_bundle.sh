#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_DEPLOY_DIR="${PADDLEOCR_HPS_DEPLOY_DIR:-${SCRIPT_DIR}/runtime}"
HPS_ENV_FILE="${PADDLEOCR_HPS_ENV_FILE:-${SCRIPT_DIR}/hps.env}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/offline-bundle}"

[[ -f "${HPS_DEPLOY_DIR}/docker-compose.yml" ]] || {
  echo "[ERROR] Missing ${HPS_DEPLOY_DIR}/docker-compose.yml" >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}/deploy"
cp -a "${HPS_DEPLOY_DIR}/." "${OUTPUT_DIR}/deploy/"
[[ -f "${HPS_ENV_FILE}" ]] && cp "${HPS_ENV_FILE}" "${OUTPUT_DIR}/hps.env"

env_args=()
[[ -f "${HPS_ENV_FILE}" ]] && env_args=(--env-file "${HPS_ENV_FILE}")
mapfile -t images < <(
  docker compose "${env_args[@]}" -f "${HPS_DEPLOY_DIR}/docker-compose.yml" config --images | sort -u
)
(( ${#images[@]} > 0 )) || {
  echo "[ERROR] No Docker images found in the HPS compose file." >&2
  exit 3
}

printf '%s\n' "${images[@]}" >"${OUTPUT_DIR}/images.txt"
docker save -o "${OUTPUT_DIR}/paddleocr-hps-images.tar" "${images[@]}"
(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[READY] Offline HPS bundle: ${OUTPUT_DIR}"
