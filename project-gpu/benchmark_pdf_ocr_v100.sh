#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/200-page.pdf [output-json]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

__BENCH_OVERRIDE_PDF_OCR_OPTIMIZATION_PROFILE="${PDF_OCR_OPTIMIZATION_PROFILE:-}"
__BENCH_OVERRIDE_PDF_PARSE_MODE="${PDF_PARSE_MODE:-}"
__BENCH_OVERRIDE_PDF_UPLOAD_OCR_ENABLED="${PDF_UPLOAD_OCR_ENABLED:-}"
__BENCH_OVERRIDE_PDF_BACKGROUND_OCR_ENABLED="${PDF_BACKGROUND_OCR_ENABLED:-}"
__BENCH_OVERRIDE_PDF_OCR_DEVICE="${PDF_OCR_DEVICE:-}"
__BENCH_OVERRIDE_PDF_OCR_MAX_PAGES="${PDF_OCR_MAX_PAGES:-}"
__BENCH_OVERRIDE_PDF_OCR_TARGET_PAGES="${PDF_OCR_TARGET_PAGES:-}"
__BENCH_OVERRIDE_PDF_OCR_TARGET_SECONDS="${PDF_OCR_TARGET_SECONDS:-}"
__BENCH_OVERRIDE_PDF_OCR_USE_INTERNAL_QUEUES="${PDF_OCR_USE_INTERNAL_QUEUES:-}"
__BENCH_OVERRIDE_PDF_OCR_GPU_PROCESS_ISOLATION="${PDF_OCR_GPU_PROCESS_ISOLATION:-}"
__BENCH_OVERRIDE_PDF_OCR_USE_CHART_RECOGNITION="${PDF_OCR_USE_CHART_RECOGNITION:-}"
__BENCH_OVERRIDE_PDF_OCR_USE_SEAL_RECOGNITION="${PDF_OCR_USE_SEAL_RECOGNITION:-}"
__BENCH_OVERRIDE_PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK="${PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK:-}"
__BENCH_OVERRIDE_PDF_OCR_MAX_NEW_TOKENS="${PDF_OCR_MAX_NEW_TOKENS:-}"
__BENCH_OVERRIDE_PDF_OCR_MIN_PIXELS="${PDF_OCR_MIN_PIXELS:-}"
__BENCH_OVERRIDE_PDF_OCR_MAX_PIXELS="${PDF_OCR_MAX_PIXELS:-}"
__BENCH_OVERRIDE_PDF_OCR_LAYOUT_SHAPE_MODE="${PDF_OCR_LAYOUT_SHAPE_MODE:-}"
__BENCH_OVERRIDE_PDF_OCR_VL_REC_MAX_CONCURRENCY="${PDF_OCR_VL_REC_MAX_CONCURRENCY:-}"
__BENCH_OVERRIDE_PDF_OCR_BACKEND="${PDF_OCR_BACKEND:-}"
__BENCH_OVERRIDE_PDF_OCR_HPS_CHUNK_PAGES="${PDF_OCR_HPS_CHUNK_PAGES:-}"
__BENCH_OVERRIDE_PDF_OCR_HPS_MAX_CONCURRENCY="${PDF_OCR_HPS_MAX_CONCURRENCY:-}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"

compass_load_env_file "${MAIN_BACKEND_HOME}/.env.auto"
compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
compass_load_env_file "${MAIN_BACKEND_HOME}/.env"

bench_restore_override() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    export "${name}=${value}"
  fi
}

bench_restore_override PDF_OCR_OPTIMIZATION_PROFILE "${__BENCH_OVERRIDE_PDF_OCR_OPTIMIZATION_PROFILE}"
bench_restore_override PDF_PARSE_MODE "${__BENCH_OVERRIDE_PDF_PARSE_MODE}"
bench_restore_override PDF_UPLOAD_OCR_ENABLED "${__BENCH_OVERRIDE_PDF_UPLOAD_OCR_ENABLED}"
bench_restore_override PDF_BACKGROUND_OCR_ENABLED "${__BENCH_OVERRIDE_PDF_BACKGROUND_OCR_ENABLED}"
bench_restore_override PDF_OCR_DEVICE "${__BENCH_OVERRIDE_PDF_OCR_DEVICE}"
bench_restore_override PDF_OCR_MAX_PAGES "${__BENCH_OVERRIDE_PDF_OCR_MAX_PAGES}"
bench_restore_override PDF_OCR_TARGET_PAGES "${__BENCH_OVERRIDE_PDF_OCR_TARGET_PAGES}"
bench_restore_override PDF_OCR_TARGET_SECONDS "${__BENCH_OVERRIDE_PDF_OCR_TARGET_SECONDS}"
bench_restore_override PDF_OCR_USE_INTERNAL_QUEUES "${__BENCH_OVERRIDE_PDF_OCR_USE_INTERNAL_QUEUES}"
bench_restore_override PDF_OCR_GPU_PROCESS_ISOLATION "${__BENCH_OVERRIDE_PDF_OCR_GPU_PROCESS_ISOLATION}"
bench_restore_override PDF_OCR_USE_CHART_RECOGNITION "${__BENCH_OVERRIDE_PDF_OCR_USE_CHART_RECOGNITION}"
bench_restore_override PDF_OCR_USE_SEAL_RECOGNITION "${__BENCH_OVERRIDE_PDF_OCR_USE_SEAL_RECOGNITION}"
bench_restore_override PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK "${__BENCH_OVERRIDE_PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK}"
bench_restore_override PDF_OCR_MAX_NEW_TOKENS "${__BENCH_OVERRIDE_PDF_OCR_MAX_NEW_TOKENS}"
bench_restore_override PDF_OCR_MIN_PIXELS "${__BENCH_OVERRIDE_PDF_OCR_MIN_PIXELS}"
bench_restore_override PDF_OCR_MAX_PIXELS "${__BENCH_OVERRIDE_PDF_OCR_MAX_PIXELS}"
bench_restore_override PDF_OCR_LAYOUT_SHAPE_MODE "${__BENCH_OVERRIDE_PDF_OCR_LAYOUT_SHAPE_MODE}"
bench_restore_override PDF_OCR_VL_REC_MAX_CONCURRENCY "${__BENCH_OVERRIDE_PDF_OCR_VL_REC_MAX_CONCURRENCY}"
bench_restore_override PDF_OCR_BACKEND "${__BENCH_OVERRIDE_PDF_OCR_BACKEND}"
bench_restore_override PDF_OCR_HPS_CHUNK_PAGES "${__BENCH_OVERRIDE_PDF_OCR_HPS_CHUNK_PAGES}"
bench_restore_override PDF_OCR_HPS_MAX_CONCURRENCY "${__BENCH_OVERRIDE_PDF_OCR_HPS_MAX_CONCURRENCY}"

PDF_PATH="$1"
if [[ ! -f "${PDF_PATH}" ]]; then
  echo "[ERROR] PDF not found: ${PDF_PATH}" >&2
  exit 2
fi

export PDF_OCR_OPTIMIZATION_PROFILE="${PDF_OCR_OPTIMIZATION_PROFILE:-v100_32gb_fast}"
export PDF_PARSE_MODE="${PDF_PARSE_MODE:-ocr_first}"
export PDF_UPLOAD_OCR_ENABLED="${PDF_UPLOAD_OCR_ENABLED:-1}"
export PDF_BACKGROUND_OCR_ENABLED="${PDF_BACKGROUND_OCR_ENABLED:-0}"
export PDF_OCR_DEVICE="${PDF_OCR_DEVICE:-gpu:0}"
export PDF_OCR_MAX_PAGES="${PDF_OCR_MAX_PAGES:-400}"
export PDF_OCR_TARGET_PAGES="${PDF_OCR_TARGET_PAGES:-200}"
export PDF_OCR_TARGET_SECONDS="${PDF_OCR_TARGET_SECONDS:-300}"
export PDF_OCR_USE_INTERNAL_QUEUES="${PDF_OCR_USE_INTERNAL_QUEUES:-1}"
export PDF_OCR_GPU_PROCESS_ISOLATION="${PDF_OCR_GPU_PROCESS_ISOLATION:-1}"
export PDF_OCR_USE_CHART_RECOGNITION="${PDF_OCR_USE_CHART_RECOGNITION:-0}"
export PDF_OCR_USE_SEAL_RECOGNITION="${PDF_OCR_USE_SEAL_RECOGNITION:-0}"
export PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK="${PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK:-0}"
export PDF_OCR_MAX_NEW_TOKENS="${PDF_OCR_MAX_NEW_TOKENS:-512}"
export PDF_OCR_MIN_PIXELS="${PDF_OCR_MIN_PIXELS:-3136}"
export PDF_OCR_MAX_PIXELS="${PDF_OCR_MAX_PIXELS:-589824}"
export PDF_OCR_LAYOUT_SHAPE_MODE="${PDF_OCR_LAYOUT_SHAPE_MODE:-rect}"
export PDF_OCR_VL_REC_MAX_CONCURRENCY="${PDF_OCR_VL_REC_MAX_CONCURRENCY:-1}"

LOG_DIR="${COMPASSLM_LOGS_DIR:-${COMPASSLM_HOME}/logs}/ocr-benchmarks"
mkdir -p "${LOG_DIR}"
BENCH_ID="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${2:-${LOG_DIR}/${BENCH_ID}_v100_ocr_benchmark.json}"
VENV_PATH="${MAIN_BACKEND_HOME}/compassvenv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Main-backend venv not found: ${VENV_PATH}" >&2
  echo "Run: ${PROJECT_GPU_HOME}/setup_gpu_track.sh --offline-backend" >&2
  exit 1
fi

echo "[INFO] PDF=${PDF_PATH}"
echo "[INFO] OUTPUT_JSON=${OUTPUT_JSON}"
echo "[INFO] PDF_OCR_OPTIMIZATION_PROFILE=${PDF_OCR_OPTIMIZATION_PROFILE}"
echo "[INFO] PDF_OCR_DEVICE=${PDF_OCR_DEVICE}"
echo "[INFO] PDF_OCR_MAX_PAGES=${PDF_OCR_MAX_PAGES}"
echo "[INFO] PDF_OCR_TARGET_SECONDS=${PDF_OCR_TARGET_SECONDS}"
echo "[INFO] PDF_OCR_MAX_PIXELS=${PDF_OCR_MAX_PIXELS}"
echo "[INFO] PDF_OCR_MAX_NEW_TOKENS=${PDF_OCR_MAX_NEW_TOKENS}"

cd "${COMPASSLM_HOME}"
source "${VENV_PATH}/bin/activate"

BENCH_PDF_PATH="${PDF_PATH}" BENCH_OUTPUT_JSON="${OUTPUT_JSON}" python3 - <<'PY'
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from src.pdf_ocr import extract_pdf_pages


pdf_path = Path(os.environ["BENCH_PDF_PATH"]).resolve()
output_path = Path(os.environ["BENCH_OUTPUT_JSON"]).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)

peak = {
    "gpu_memory_used_mb": 0,
    "gpu_memory_free_mb": 0,
    "gpu_memory_total_mb": 0,
    "gpu_samples": 0,
    "process_rss_mb": 0,
    "process_tree_rss_mb": 0,
    "process_rss_samples": 0,
}
gpu_sample_rows: List[Dict[str, Any]] = []
process_sample_rows: List[Dict[str, Any]] = []
stop_event = threading.Event()


def _gpu_index(device: str) -> int:
    raw = (device or "").strip().lower()
    for sep in (":", "gpu", "cuda"):
        if sep in raw:
            tail = raw.rsplit(sep, 1)[-1]
            if tail.isdigit():
                return max(0, int(tail))
    return 0


def _read_process_rss_mb() -> int:
    return _read_pid_rss_mb(os.getpid())


def _read_pid_rss_mb(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return max(0, int(int(parts[1]) / 1024))
    except Exception:
        return 0
    return 0


def _read_process_tree_rss_mb() -> int:
    root_pid = os.getpid()
    rss_total = 0
    children_by_ppid: Dict[int, List[int]] = {}
    try:
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                raw = stat_path.read_text(encoding="utf-8")
                after_name = raw.rsplit(") ", 1)[1]
                parts = after_name.split()
                if len(parts) >= 2:
                    pid = int(stat_path.parent.name)
                    ppid = int(parts[1])
                    children_by_ppid.setdefault(ppid, []).append(pid)
            except Exception:
                continue
    except Exception:
        return _read_process_rss_mb()

    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss_total += _read_pid_rss_mb(pid)
        stack.extend(children_by_ppid.get(pid, []))
    return rss_total


def _sample_resources() -> None:
    gpu_index = _gpu_index(os.environ.get("PDF_OCR_DEVICE", "gpu:0"))
    while not stop_event.wait(1.0):
        rss_mb = _read_process_rss_mb()
        if rss_mb > 0:
            peak["process_rss_samples"] = int(peak["process_rss_samples"]) + 1
            peak["process_rss_mb"] = max(int(peak["process_rss_mb"]), rss_mb)
            peak["process_tree_rss_mb"] = max(int(peak["process_tree_rss_mb"]), _read_process_tree_rss_mb())
            if len(process_sample_rows) < 360:
                process_sample_rows.append(
                    {
                        "t": round(time.monotonic(), 3),
                        "rss_mb": rss_mb,
                        "tree_rss_mb": int(peak["process_tree_rss_mb"]),
                    }
                )
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except Exception:
            continue
        if proc.returncode != 0:
            continue
        line = next((item.strip() for item in proc.stdout.splitlines() if item.strip()), "")
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            used, free, total = [int(float(part)) for part in parts[:3]]
        except ValueError:
            continue
        peak["gpu_samples"] = int(peak["gpu_samples"]) + 1
        if used >= int(peak["gpu_memory_used_mb"]):
            peak["gpu_memory_used_mb"] = used
            peak["gpu_memory_free_mb"] = free
            peak["gpu_memory_total_mb"] = total
        if len(gpu_sample_rows) < 360:
            gpu_sample_rows.append(
                {
                    "t": round(time.monotonic(), 3),
                    "used_mb": used,
                    "free_mb": free,
                    "total_mb": total,
                }
            )


progress_events: List[Dict[str, Any]] = []


def _progress(percent: int, message: str, stage: str, **meta: Any) -> None:
    event = {
        "t": round(time.monotonic(), 3),
        "percent": int(percent),
        "stage": str(stage or ""),
        "message": str(message or ""),
    }
    event.update(meta)
    progress_events.append(event)
    print(f"[BENCH][{percent:03d}] {stage}: {message}", flush=True)


sampler = threading.Thread(target=_sample_resources, name="ocr-benchmark-resource-sampler", daemon=True)
started = time.monotonic()
started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
sampler.start()
status = "ok"
error = ""
result: Dict[str, Any] = {}
try:
    result = extract_pdf_pages(str(pdf_path), progress_callback=_progress, force_upload_ocr=True)
except Exception as exc:
    status = "error"
    error = f"{type(exc).__name__}: {exc}"
finally:
    elapsed = time.monotonic() - started
    stop_event.set()
    sampler.join(timeout=2.0)

ocr_pages = int(result.get("ocr_pages", 0) or 0)
target_pages = int(result.get("ocr_target_pages", os.environ.get("PDF_OCR_TARGET_PAGES", 200)) or 0)
target_seconds = float(result.get("ocr_target_seconds", os.environ.get("PDF_OCR_TARGET_SECONDS", 300)) or 0)
pages_per_minute = float(result.get("ocr_pages_per_minute", 0.0) or 0.0)
if pages_per_minute <= 0 and elapsed > 0:
    pages_per_minute = round(ocr_pages / elapsed * 60.0, 3)
target_met = bool(result.get("ocr_target_met", False))
if target_pages > 0 and target_seconds > 0:
    target_met = ocr_pages >= target_pages and elapsed <= target_seconds

payload = {
    "status": status,
    "error": error,
    "started_at": started_wall,
    "pdf_path": str(pdf_path),
    "env": {
        "PDF_OCR_OPTIMIZATION_PROFILE": os.environ.get("PDF_OCR_OPTIMIZATION_PROFILE", ""),
        "PDF_PARSE_MODE": os.environ.get("PDF_PARSE_MODE", ""),
        "PDF_OCR_DEVICE": os.environ.get("PDF_OCR_DEVICE", ""),
        "PDF_OCR_MAX_PAGES": os.environ.get("PDF_OCR_MAX_PAGES", ""),
        "PDF_OCR_TARGET_PAGES": os.environ.get("PDF_OCR_TARGET_PAGES", ""),
        "PDF_OCR_TARGET_SECONDS": os.environ.get("PDF_OCR_TARGET_SECONDS", ""),
        "PDF_OCR_USE_INTERNAL_QUEUES": os.environ.get("PDF_OCR_USE_INTERNAL_QUEUES", ""),
        "PDF_OCR_GPU_PROCESS_ISOLATION": os.environ.get("PDF_OCR_GPU_PROCESS_ISOLATION", ""),
        "PDF_OCR_MAX_PIXELS": os.environ.get("PDF_OCR_MAX_PIXELS", ""),
        "PDF_OCR_MAX_NEW_TOKENS": os.environ.get("PDF_OCR_MAX_NEW_TOKENS", ""),
        "PDF_OCR_VL_REC_MAX_CONCURRENCY": os.environ.get("PDF_OCR_VL_REC_MAX_CONCURRENCY", ""),
        "PDF_OCR_BACKEND": os.environ.get("PDF_OCR_BACKEND", "local"),
        "PDF_OCR_HPS_CHUNK_PAGES": os.environ.get("PDF_OCR_HPS_CHUNK_PAGES", ""),
        "PDF_OCR_HPS_MAX_CONCURRENCY": os.environ.get("PDF_OCR_HPS_MAX_CONCURRENCY", ""),
    },
    "metrics": {
        "elapsed_seconds": round(elapsed, 3),
        "total_pages": int(result.get("total_pages", 0) or 0),
        "ocr_pages": ocr_pages,
        "text_pages": int(result.get("text_pages", 0) or 0),
        "failed_pages": int(result.get("failed_pages", 0) or 0),
        "attempted_ocr_pages": int(result.get("attempted_ocr_pages", 0) or 0),
        "pages_per_minute": pages_per_minute,
        "target_pages": target_pages,
        "target_seconds": target_seconds,
        "target_met": target_met,
        "ocr_device_effective": result.get("ocr_device_effective", ""),
        "ocr_gpu_fallback_used": bool(result.get("ocr_gpu_fallback_used", False)),
        "ocr_gpu_failure_reason": result.get("ocr_gpu_failure_reason", ""),
        "ocr_backend": result.get("ocr_backend", ""),
        "ocr_backend_effective": result.get("ocr_backend_effective", ""),
        "ocr_backend_fallback_used": bool(result.get("ocr_backend_fallback_used", False)),
        "ocr_subset_build_seconds": float(result.get("ocr_subset_build_seconds", 0.0) or 0.0),
        "ocr_model_load_seconds": float(result.get("ocr_model_load_seconds", 0.0) or 0.0),
        "ocr_predict_seconds": float(result.get("ocr_predict_seconds", 0.0) or 0.0),
        "ocr_output_materialize_seconds": float(result.get("ocr_output_materialize_seconds", 0.0) or 0.0),
        "ocr_payload_convert_seconds": float(result.get("ocr_payload_convert_seconds", 0.0) or 0.0),
        "ocr_text_merge_seconds": float(result.get("ocr_text_merge_seconds", 0.0) or 0.0),
        "ocr_merge_seconds": float(result.get("ocr_merge_seconds", 0.0) or 0.0),
        "ocr_batch_wall_seconds_mean": float(result.get("ocr_batch_wall_seconds_mean", 0.0) or 0.0),
        "ocr_batch_wall_seconds_p50": float(result.get("ocr_batch_wall_seconds_p50", 0.0) or 0.0),
        "ocr_batch_wall_seconds_p95": float(result.get("ocr_batch_wall_seconds_p95", 0.0) or 0.0),
        "ocr_batch_wall_seconds_max": float(result.get("ocr_batch_wall_seconds_max", 0.0) or 0.0),
        "peak_gpu_memory_used_mb": int(peak["gpu_memory_used_mb"]),
        "peak_gpu_memory_total_mb": int(peak["gpu_memory_total_mb"]),
        "gpu_memory_samples": int(peak["gpu_samples"]),
        "peak_process_rss_mb": int(peak["process_rss_mb"]),
        "peak_process_tree_rss_mb": int(peak["process_tree_rss_mb"]),
        "process_rss_samples": int(peak["process_rss_samples"]),
    },
    "warnings": result.get("warnings", []),
    "progress_tail": progress_events[-40:],
    "gpu_samples_tail": gpu_sample_rows[-40:],
    "process_rss_samples_tail": process_sample_rows[-40:],
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), flush=True)
print(f"[INFO] Benchmark result: {output_path}", flush=True)
if status != "ok":
    raise SystemExit(1)
if not target_met:
    raise SystemExit(3)
PY
