#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/pdf [output-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_PATH="$1"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/../logs/ocr-benchmarks/hps-matrix-$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

run_case() {
  local chunk_pages="$1"
  local concurrency="$2"
  local label="chunk${chunk_pages}-concurrency${concurrency}"
  echo "[MATRIX] ${label}"
  local status=0
  set +e
  PDF_OCR_HPS_CHUNK_PAGES="${chunk_pages}" \
  PDF_OCR_HPS_MAX_CONCURRENCY="${concurrency}" \
    "${SCRIPT_DIR}/run_pdf_ocr_hps_benchmark.sh" \
      "${PDF_PATH}" "${OUTPUT_DIR}/${label}.json"
  status=$?
  set -e
  if [[ "${status}" != "0" && "${status}" != "3" ]]; then
    return "${status}"
  fi
}

# First tune service concurrency while holding the client chunk size fixed.
for concurrency in 1 2 4 8; do
  run_case 16 "${concurrency}"
done

# Then tune chunk size using the selected concurrency (default 4).
selected_concurrency="${PDF_OCR_HPS_MATRIX_SELECTED_CONCURRENCY:-4}"
for chunk_pages in 8 32; do
  run_case "${chunk_pages}" "${selected_concurrency}"
done

echo "[READY] Matrix results: ${OUTPUT_DIR}"
