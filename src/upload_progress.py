from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


_STAGE_BOUNDS: Dict[str, Tuple[int, int]] = {
    "queued": (0, 4),
    "preparing": (5, 9),
    "prepare_kb": (10, 16),
    "inspect_file": (12, 17),
    "check_cache": (18, 27),
    "cache_hit": (36, 45),
    "read_text": (30, 41),
    "split_text": (42, 49),
    "chunk_text": (54, 61),
    "read_sheet": (30, 47),
    "chunk_sheet": (48, 61),
    "inspect_pdf": (28, 33),
    "extract_pdf_text": (34, 41),
    "fallback_pdf_ocr": (38, 45),
    "load_pdf_ocr_model": (42, 48),
    "run_pdf_ocr": (49, 76),
    "merge_pdf_ocr": (77, 84),
    "chunk_pdf": (85, 92),
    "prepare_pdf_chunks": (85, 92),
    "store_cache": (93, 94),
    "store_ocr_cache": (93, 94),
    "persist_meta": (95, 96),
    "store_chunks": (96, 98),
    "refresh_index": (99, 99),
    "embed_chunks": (99, 99),
    "sync_derived": (99, 99),
    "done": (100, 100),
}
_OCR_STALL_STAGES = {"load_pdf_ocr_model", "run_pdf_ocr", "fallback_pdf_ocr"}
_OCR_COMPLETED_STAGES = {
    "prepare_pdf_chunks",
    "store_ocr_cache",
    "store_cache",
    "store_chunks",
    "refresh_index",
    "embed_chunks",
    "sync_derived",
    "done",
}
_UPLOAD_INDEX_STAGES = {"store_chunks", "refresh_index", "sync_derived"}
_UPLOAD_EMBEDDING_STAGES = {"embed_chunks"}
_UPLOAD_COMMIT_STAGES = _UPLOAD_INDEX_STAGES | _UPLOAD_EMBEDDING_STAGES


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def update_upload_phase_state(state: Dict[str, Any], stage: str, *, now_ts: int) -> Dict[str, Any]:
    next_state = dict(state or {})
    normalized_stage = str(stage or "processing").strip().lower() or "processing"
    next_state["stage"] = normalized_stage
    next_state.setdefault("ocr_completed", False)
    next_state.setdefault("index_completed", False)
    next_state.setdefault("embedding_started_at", 0)
    next_state.setdefault("embedding_completed_at", 0)

    if normalized_stage in _OCR_COMPLETED_STAGES:
        next_state["ocr_completed"] = True
    if normalized_stage in _UPLOAD_COMMIT_STAGES and not _safe_int(next_state.get("embedding_started_at", 0)):
        next_state["embedding_started_at"] = int(now_ts)
    if normalized_stage == "done":
        next_state["ocr_completed"] = True
        next_state["index_completed"] = True
        if not _safe_int(next_state.get("embedding_started_at", 0)):
            next_state["embedding_started_at"] = int(now_ts)
        next_state["embedding_completed_at"] = int(now_ts)
    return next_state


def upload_failure_default_for_stage(stage: str, *, ocr_completed: bool) -> str:
    normalized_stage = str(stage or "").strip().lower()
    if not bool(ocr_completed):
        return "upload_ingest_fail"
    if normalized_stage in _UPLOAD_EMBEDDING_STAGES:
        return "upload_embedding_fail"
    if normalized_stage in _UPLOAD_INDEX_STAGES:
        return "upload_index_fail"
    return "upload_ingest_fail"


def _count_ratio(stage: str, payload: Dict[str, Any]) -> Optional[float]:
    if stage not in {"extract_pdf_text", "run_pdf_ocr", "fallback_pdf_ocr", "merge_pdf_ocr", "chunk_pdf", "prepare_pdf_chunks", "store_chunks"}:
        return None
    current_page = max(0, _safe_int(payload.get("current_page", 0)))
    total_pages = max(0, _safe_int(payload.get("total_pages", 0)))
    if total_pages <= 0 or current_page <= 0:
        return None
    return max(0.0, min(1.0, current_page / float(total_pages)))


def _stage_elapsed_seconds(payload: Dict[str, Any], now_ts: int) -> int:
    stage_started_at = max(0, _safe_int(payload.get("last_progress_at", 0)))
    if stage_started_at <= 0:
        stage_started_at = max(0, _safe_int(payload.get("processing_started_at", 0)))
    if stage_started_at <= 0:
        return 0
    return max(0, int(now_ts) - stage_started_at)


def _expected_stage_seconds(stage: str, payload: Dict[str, Any]) -> float:
    total_pages = max(1, _safe_int(payload.get("total_pages", 0), default=1))
    if stage == "extract_pdf_text":
        return min(45.0, 3.0 + (total_pages * 0.12))
    if stage == "load_pdf_ocr_model":
        return 18.0
    if stage == "run_pdf_ocr":
        return min(240.0, 12.0 + (total_pages * 3.2))
    if stage == "fallback_pdf_ocr":
        return 12.0
    if stage == "store_cache":
        return 8.0
    if stage == "persist_meta":
        return 6.0
    if stage == "refresh_index":
        return 24.0
    return 12.0


def _queue_elapsed_seconds(payload: Dict[str, Any], now_ts: int) -> int:
    queued_at = max(0, _safe_int(payload.get("created_at", 0)))
    if queued_at <= 0:
        queued_at = max(0, _safe_int(payload.get("updated_at", 0)))
    if queued_at <= 0:
        return 0
    return max(0, int(now_ts) - queued_at)


def _processing_stall_timeout_seconds(
    stage: str,
    payload: Dict[str, Any],
    processing_timeout_seconds: int,
) -> int:
    expected_seconds = float(_expected_stage_seconds(stage, payload))
    dynamic_timeout = int(round(expected_seconds * 3.0))
    if stage == "load_pdf_ocr_model":
        return max(int(processing_timeout_seconds), dynamic_timeout, 600)
    if stage in {"run_pdf_ocr", "fallback_pdf_ocr"}:
        return max(int(processing_timeout_seconds), dynamic_timeout, 720)
    return max(int(processing_timeout_seconds), dynamic_timeout)


def _last_progress_signal_age_seconds(stage: str, payload: Dict[str, Any], now_ts: int) -> int:
    signal_at = max(0, _safe_int(payload.get("last_progress_at", 0)))
    if stage in _OCR_STALL_STAGES:
        signal_at = max(
            signal_at,
            _safe_int(payload.get("ocr_heartbeat_at", 0)),
            _safe_int(payload.get("ocr_last_batch_completed_at", 0)),
        )
    if signal_at <= 0:
        signal_at = max(0, _safe_int(payload.get("processing_started_at", 0)))
    if signal_at <= 0:
        signal_at = max(0, _safe_int(payload.get("created_at", 0)))
    if signal_at <= 0:
        return 0
    return max(0, int(now_ts) - signal_at)


def _elapsed_ratio(stage: str, payload: Dict[str, Any], now_ts: int) -> float:
    expected_seconds = max(1.0, float(_expected_stage_seconds(stage, payload)))
    elapsed_seconds = float(_stage_elapsed_seconds(payload, now_ts))
    linear_ratio = max(0.0, min(1.0, elapsed_seconds / expected_seconds))
    # Ease-out so long-running stages move early instead of appearing frozen.
    return linear_ratio ** 0.82


def estimate_display_progress_percent(payload: Dict[str, Any], now_ts: int) -> int:
    status = str(payload.get("status", "") or "").strip().lower()
    stored_percent = _clamp(_safe_int(payload.get("progress_percent", 0)), 0, 100)
    if status in {"success", "error", "not_found"}:
        return stored_percent

    stage = str(payload.get("progress_stage", "") or "").strip().lower()
    bounds = _STAGE_BOUNDS.get(stage)
    if not bounds:
        return stored_percent

    lower, upper = bounds
    if upper <= lower:
        return max(stored_percent, upper)

    ratio = _count_ratio(stage, payload)
    if ratio is None:
        ratio = _elapsed_ratio(stage, payload, now_ts)

    estimated = lower + int(round((upper - lower) * ratio))
    return _clamp(max(stored_percent, estimated), 0, 99)


def estimate_background_ocr_progress_percent(payload: Dict[str, Any]) -> int:
    status = str(payload.get("status", "") or "").strip().lower()
    stored_percent = _clamp(_safe_int(payload.get("progress_percent", 0)), 0, 100)
    if status in {"success", "error", "skipped", "not_found"}:
        return stored_percent

    target_pages = _safe_int(payload.get("ocr_target_pages", 0))
    completed_pages = _safe_int(payload.get("ocr_completed_pages", 0))
    if target_pages <= 0:
        return _clamp(stored_percent, 0, 99)

    ratio = max(0.0, min(1.0, completed_pages / float(max(1, target_pages))))
    lower, upper = 5, 99
    return _clamp(lower + int(round((upper - lower) * ratio)), 0, 99)


def build_upload_stall_state(
    payload: Dict[str, Any],
    *,
    now_ts: int,
    processing_timeout_seconds: int = 480,
    queue_timeout_seconds: int = 180,
) -> Optional[Dict[str, Any]]:
    status = str(payload.get("status", "") or "").strip().lower()
    if status in {"success", "error", "not_found"}:
        return None

    stage = str(payload.get("progress_stage", "") or "").strip().lower()
    if status == "queued":
        if int(queue_timeout_seconds) <= 0:
            return None
        queue_age_seconds = _queue_elapsed_seconds(payload, now_ts)
        if queue_age_seconds < max(60, int(queue_timeout_seconds)):
            return None
        return {
            "status": "error",
            "progress_stage": "error",
            "failure_code": "upload_job_stalled",
            "stall_seconds": queue_age_seconds,
            "stall_timeout_seconds": max(60, int(queue_timeout_seconds)),
            "message": (
                f"업로드 대기 상태가 {queue_age_seconds}초 이상 이어져 작업을 중단했습니다. "
                "server upload worker 상태를 확인해 주세요."
            ),
        }

    if status != "processing":
        return None

    progress_age_seconds = _last_progress_signal_age_seconds(stage, payload, now_ts)
    stall_timeout = _processing_stall_timeout_seconds(
        stage,
        payload,
        max(180, int(processing_timeout_seconds)),
    )
    if progress_age_seconds < stall_timeout:
        return None

    if stage in _OCR_STALL_STAGES:
        attempted_device = str(
            payload.get("ocr_device_effective", "") or payload.get("ocr_device_attempted", "") or ""
        ).strip().lower()
        if attempted_device.startswith("gpu") or attempted_device.startswith("cuda"):
            message = (
                f"PDF OCR 진행 신호가 {progress_age_seconds}초 동안 멈춰 작업을 중단했습니다. "
                "backend 로그를 확인하고 GPU OCR이 멈춘 경우 CPU fallback 사용 여부와 "
                "`PDF_OCR_GPU_BUDGET_GB`, `PDF_OCR_GPU_INITIAL_MEMORY_MB`, `PDF_OCR_MAX_PAGES`를 함께 점검해 주세요."
            )
        else:
            message = (
                f"PDF OCR 진행 신호가 {progress_age_seconds}초 동안 멈춰 작업을 중단했습니다. "
                "backend 로그를 확인하고 `PDF_OCR_MAX_PAGES`를 낮추거나 OCR 전용 GPU 분리를 검토해 주세요."
            )
    else:
        message = (
            f"문서 정리 진행 신호가 {progress_age_seconds}초 동안 멈춰 작업을 중단했습니다. "
            "backend 로그를 확인해 주세요."
        )

    return {
        "status": "error",
        "progress_stage": "error",
        "failure_code": "upload_job_stalled",
        "stall_seconds": progress_age_seconds,
        "stall_timeout_seconds": stall_timeout,
        "message": message,
        "ocr_stall_detected": stage in _OCR_STALL_STAGES,
        "ocr_last_batch_completed_at": _safe_int(payload.get("ocr_last_batch_completed_at", 0)),
    }
