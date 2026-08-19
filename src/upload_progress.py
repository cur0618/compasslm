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
    "release_pdf_ocr_worker": (85, 86),
    "chunk_pdf": (85, 92),
    "prepare_pdf_chunks": (85, 92),
    "store_cache": (93, 94),
    "store_ocr_cache": (93, 94),
    "persist_meta": (95, 96),
    "store_chunks": (96, 98),
    "refresh_index": (99, 99),
    "embed_chunks": (99, 99),
    "embed_concepts": (99, 99),
    "sync_derived_concept_embedding": (99, 99),
    "sync_derived": (99, 99),
    "done": (100, 100),
}

_OCR_STALL_STAGES = {
    "load_pdf_ocr_model",
    "run_pdf_ocr",
    "fallback_pdf_ocr",
    "merge_pdf_ocr",
    "release_pdf_ocr_worker",
}

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
_UPLOAD_EMBEDDING_STAGES = {"embed_chunks", "embed_concepts", "sync_derived_concept_embedding"}
_UPLOAD_COMMIT_STAGES = _UPLOAD_INDEX_STAGES | _UPLOAD_EMBEDDING_STAGES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _max_metric(payload: Dict[str, Any], *keys: str) -> int:
    values = [_safe_int(payload.get(key, 0)) for key in keys]
    return max([0, *values])


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def update_upload_phase_state(
    state: Dict[str, Any],
    stage: str,
    *,
    now_ts: int,
    progress_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    next_state = dict(state or {})
    normalized_stage = str(stage or "processing").strip().lower() or "processing"
    normalized_meta = dict(progress_meta or {})
    effective_phase_name = str(
        normalized_meta.get("phase_name_effective", normalized_stage) or normalized_stage
    ).strip().lower() or normalized_stage
    previous_phase_name = str(
        next_state.get("phase_name_effective", next_state.get("stage", "")) or ""
    ).strip().lower()

    next_state["stage"] = normalized_stage
    next_state.setdefault("ocr_completed", False)
    next_state.setdefault("index_completed", False)
    next_state.setdefault("embedding_started_at", 0)
    next_state.setdefault("embedding_completed_at", 0)
    next_state.setdefault("phase_started_at", 0)
    next_state.setdefault("phase_last_heartbeat_at", 0)
    next_state.setdefault("phase_elapsed_seconds", 0)
    next_state.setdefault("phase_name_effective", normalized_stage)
    next_state.setdefault("phase_rows_total", 0)
    next_state.setdefault("phase_rows_done", 0)
    next_state.setdefault("phase_chunks_total", 0)
    next_state.setdefault("phase_chunks_done", 0)
    next_state.setdefault("embed_batch", 0)
    next_state.setdefault("embed_batches", 0)
    next_state.setdefault("embed_rows_done", 0)
    next_state.setdefault("embed_rows_total", 0)
    next_state.setdefault("embed_input_tokens_total", 0)
    next_state.setdefault("embed_input_tokens_done", 0)
    next_state.setdefault("embed_input_tokens_p95", 0)
    next_state.setdefault("embed_input_tokens_max", 0)
    next_state.setdefault("embed_truncated_rows", 0)
    next_state.setdefault("embed_effective_batch_tokens", 0)

    phase_changed = previous_phase_name != effective_phase_name
    if phase_changed:
        next_state["phase_started_at"] = int(now_ts)
        next_state["phase_rows_total"] = 0
        next_state["phase_rows_done"] = 0
        next_state["phase_chunks_total"] = 0
        next_state["phase_chunks_done"] = 0
        next_state["embed_batch"] = 0
        next_state["embed_batches"] = 0
        next_state["embed_rows_done"] = 0
        next_state["embed_rows_total"] = 0
        next_state["embed_input_tokens_total"] = 0
        next_state["embed_input_tokens_done"] = 0
        next_state["embed_input_tokens_p95"] = 0
        next_state["embed_input_tokens_max"] = 0
        next_state["embed_truncated_rows"] = 0
        next_state["embed_effective_batch_tokens"] = 0
    elif not _safe_int(next_state.get("phase_started_at", 0)):
        next_state["phase_started_at"] = int(now_ts)

    next_state["phase_last_heartbeat_at"] = int(now_ts)
    next_state["phase_elapsed_seconds"] = max(
        0,
        int(now_ts) - _safe_int(next_state.get("phase_started_at", now_ts), default=now_ts),
    )
    next_state["phase_name_effective"] = effective_phase_name

    for key in (
        "phase_rows_total",
        "phase_rows_done",
        "phase_chunks_total",
        "phase_chunks_done",
        "embed_batch",
        "embed_batches",
        "embed_rows_done",
        "embed_rows_total",
        "embed_input_tokens_total",
        "embed_input_tokens_done",
        "embed_input_tokens_p95",
        "embed_input_tokens_max",
        "embed_truncated_rows",
        "embed_effective_batch_tokens",
    ):
        if key in normalized_meta and normalized_meta.get(key) is not None:
            next_state[key] = max(0, _safe_int(normalized_meta.get(key, 0)))

    if normalized_stage in _OCR_COMPLETED_STAGES:
        next_state["ocr_completed"] = True
    if (
        normalized_stage in _UPLOAD_EMBEDDING_STAGES
        or effective_phase_name in _UPLOAD_EMBEDDING_STAGES
        or "embed" in effective_phase_name
    ) and not _safe_int(next_state.get("embedding_started_at", 0)):
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
    if stage in {
        "extract_pdf_text",
        "run_pdf_ocr",
        "fallback_pdf_ocr",
        "merge_pdf_ocr",
        "chunk_pdf",
        "prepare_pdf_chunks",
    }:
        current_page = max(0, _safe_int(payload.get("current_page", 0)))
        total_pages = max(0, _safe_int(payload.get("total_pages", 0)))
        if total_pages > 0 and current_page > 0:
            return max(0.0, min(1.0, current_page / float(total_pages)))

    if stage == "embed_chunks":
        embedded_input_done = max(0, _safe_int(payload.get("embed_input_tokens_done", 0)))
        embedded_input_total = max(0, _safe_int(payload.get("embed_input_tokens_total", 0)))
        if embedded_input_total > 0:
            return max(0.0, min(1.0, embedded_input_done / float(embedded_input_total)))
        rows_done = max(0, _safe_int(payload.get("embed_rows_done", 0)))
        rows_total = max(0, _safe_int(payload.get("embed_rows_total", 0)))
        if rows_total > 0:
            return max(0.0, min(1.0, rows_done / float(rows_total)))

    rows_done = max(0, _safe_int(payload.get("phase_rows_done", 0)))
    rows_total = max(0, _safe_int(payload.get("phase_rows_total", 0)))
    if rows_total > 0:
        return max(0.0, min(1.0, rows_done / float(rows_total)))

    chunks_done = max(0, _safe_int(payload.get("phase_chunks_done", 0)))
    chunks_total = max(0, _safe_int(payload.get("phase_chunks_total", 0)))
    if chunks_total > 0:
        return max(0.0, min(1.0, chunks_done / float(chunks_total)))

    return None


def _stage_elapsed_seconds(payload: Dict[str, Any], now_ts: int) -> int:
    stage_started_at = max(0, _safe_int(payload.get("phase_started_at", 0)))
    if stage_started_at <= 0:
        stage_started_at = max(0, _safe_int(payload.get("last_progress_at", 0)))
    if stage_started_at <= 0:
        stage_started_at = max(0, _safe_int(payload.get("processing_started_at", 0)))
    if stage_started_at <= 0:
        return 0
    return max(0, int(now_ts) - stage_started_at)


def _expected_stage_seconds(stage: str, payload: Dict[str, Any]) -> float:
    total_pages = max(1, _safe_int(payload.get("total_pages", 0), default=1))
    phase_rows_total = _max_metric(payload, "phase_rows_total")
    phase_chunks_total = _max_metric(payload, "phase_chunks_total")
    embed_rows_total = _max_metric(payload, "embed_rows_total", "phase_rows_total")
    indexable_chunks = _max_metric(payload, "indexable_chunks", "normalized_chunks", "chunks", "phase_chunks_total")
    changed_chunks = _max_metric(payload, "phase_chunks_total", "chunks")
    deleted_chunks = _max_metric(payload, "replaced_chunks")
    if stage == "extract_pdf_text":
        return min(45.0, 3.0 + (total_pages * 0.12))
    if stage == "load_pdf_ocr_model":
        return 18.0
    if stage == "run_pdf_ocr":
        return min(240.0, 12.0 + (total_pages * 3.2))
    if stage == "fallback_pdf_ocr":
        return 12.0
    if stage == "merge_pdf_ocr":
        return min(300.0, 8.0 + (total_pages * 1.2))
    if stage == "release_pdf_ocr_worker":
        return 30.0
    if stage in {"store_cache", "store_ocr_cache"}:
        return min(180.0, 8.0 + (max(total_pages, phase_rows_total) * 0.2))
    if stage == "persist_meta":
        return max(6.0, 8.0 + (phase_rows_total * 0.04))
    if stage == "store_chunks":
        return max(12.0, 10.0 + (phase_rows_total * 0.08))
    if stage == "refresh_index":
        return max(24.0, 24.0 + (changed_chunks * 0.35) + (deleted_chunks * 0.08))
    if stage == "embed_chunks":
        return max(36.0, 24.0 + (embed_rows_total * 0.40))
    if stage == "sync_derived":
        return max(45.0, 30.0 + (indexable_chunks * 0.45) + (phase_chunks_total * 0.10))
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
    if stage in {"merge_pdf_ocr", "release_pdf_ocr_worker"}:
        return max(int(processing_timeout_seconds), dynamic_timeout, 600)
    if stage == "refresh_index":
        return max(int(processing_timeout_seconds), dynamic_timeout, 540)
    if stage == "embed_chunks":
        return max(int(processing_timeout_seconds), dynamic_timeout, 540)
    if stage == "sync_derived":
        return max(int(processing_timeout_seconds), dynamic_timeout, 600)
    return max(int(processing_timeout_seconds), dynamic_timeout)


def _last_progress_signal_age_seconds(stage: str, payload: Dict[str, Any], now_ts: int) -> int:
    signal_at = max(
        0,
        _safe_int(payload.get("phase_last_heartbeat_at", 0)),
        _safe_int(payload.get("last_progress_at", 0)),
    )
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


def _stage_label(stage: str, payload: Dict[str, Any]) -> str:
    effective_name = str(payload.get("phase_name_effective", "") or stage or "processing").strip().lower()
    labels = {
        "run_pdf_ocr": "PDF OCR",
        "fallback_pdf_ocr": "PDF OCR fallback",
        "merge_pdf_ocr": "PDF OCR merge",
        "release_pdf_ocr_worker": "PDF OCR worker release",
        "refresh_index": "search index refresh",
        "embed_chunks": "embedding/vector storage",
        "embed_concepts": "concept embedding",
        "sync_derived_concept_embedding": "derived search sync (concept embedding)",
        "sync_derived": "derived search sync",
        "sync_derived_rebuild_index": "derived search sync (rebuild_index)",
        "sync_derived_concept_links": "derived search sync (concept_links)",
        "sync_derived_ontology_facts": "derived search sync (ontology_facts)",
        "sync_derived_compile_wiki": "derived search sync (compile_wiki)",
    }
    return labels.get(effective_name, labels.get(stage, effective_name or stage or "processing"))


def _progress_snapshot_fields(stage: str, payload: Dict[str, Any]) -> Dict[str, int]:
    if stage in _OCR_STALL_STAGES:
        pages_total = _max_metric(payload, "ocr_target_pages", "total_pages", "pdf_total_pages")
        pages_done = _max_metric(payload, "ocr_completed_pages", "current_page")
        return {"pages_done": pages_done, "pages_total": pages_total}

    rows_total = _max_metric(payload, "embed_rows_total", "phase_rows_total")
    rows_done = _max_metric(payload, "embed_rows_done", "phase_rows_done")
    chunks_total = _max_metric(payload, "phase_chunks_total", "chunks", "normalized_chunks", "indexable_chunks")
    chunks_done = _max_metric(payload, "phase_chunks_done", "embed_rows_done")
    return {
        "rows_done": rows_done,
        "rows_total": rows_total,
        "chunks_done": chunks_done,
        "chunks_total": chunks_total,
    }


def _stall_message(stage: str, payload: Dict[str, Any], progress_age_seconds: int, phase_elapsed_seconds: int) -> str:
    label = _stage_label(stage, payload)
    phase_name = str(payload.get("phase_name_effective", "") or stage or "processing").strip().lower() or stage
    last_heartbeat_at = _safe_int(payload.get("phase_last_heartbeat_at", payload.get("last_progress_at", 0)))
    stats = _progress_snapshot_fields(stage, payload)
    detail_parts = [
        f"stage={phase_name}",
        f"progress_age_seconds={int(progress_age_seconds)}",
        f"phase_elapsed_seconds={int(phase_elapsed_seconds)}",
        f"last_heartbeat_at={int(last_heartbeat_at)}",
    ]

    if stage in _OCR_STALL_STAGES:
        if stats.get("pages_total", 0) > 0:
            detail_parts.append(f"pages={int(stats['pages_done'])}/{int(stats['pages_total'])}")
        return (
            f"{label} 진행 신호가 {progress_age_seconds}초 동안 멈춰 작업을 중단했습니다. "
            f"backend 로그를 확인해 주세요. ({', '.join(detail_parts)})"
        )

    if stage == "refresh_index":
        prefix = "검색 인덱스 갱신 단계가 멈춰 작업을 중단했습니다."
    elif stage == "embed_chunks":
        prefix = "임베딩/벡터 저장 단계가 멈춰 작업을 중단했습니다."
    elif stage == "sync_derived":
        prefix = "파생 검색 구조 동기화 단계가 멈춰 작업을 중단했습니다."
    else:
        prefix = "문서 정리 진행 신호가 멈춰 작업을 중단했습니다."

    if stats.get("rows_total", 0) > 0:
        detail_parts.append(f"rows={int(stats['rows_done'])}/{int(stats['rows_total'])}")
    if stats.get("chunks_total", 0) > 0:
        detail_parts.append(f"chunks={int(stats['chunks_done'])}/{int(stats['chunks_total'])}")
    return f"{prefix} backend 로그를 확인해 주세요. ({', '.join(detail_parts)})"


def _elapsed_ratio(stage: str, payload: Dict[str, Any], now_ts: int) -> float:
    expected_seconds = max(1.0, float(_expected_stage_seconds(stage, payload)))
    elapsed_seconds = float(_stage_elapsed_seconds(payload, now_ts))
    linear_ratio = max(0.0, min(1.0, elapsed_seconds / expected_seconds))
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

    phase_elapsed_seconds = _stage_elapsed_seconds(payload, now_ts)
    stats = _progress_snapshot_fields(stage, payload)
    return {
        "status": "error",
        "progress_stage": "error",
        "failure_code": "upload_job_stalled",
        "stalled_stage": stage,
        "stall_seconds": progress_age_seconds,
        "stall_timeout_seconds": stall_timeout,
        "message": _stall_message(stage, payload, progress_age_seconds, phase_elapsed_seconds),
        "ocr_stall_detected": stage in _OCR_STALL_STAGES,
        "ocr_last_batch_completed_at": _safe_int(payload.get("ocr_last_batch_completed_at", 0)),
        "phase_elapsed_seconds": phase_elapsed_seconds,
        "phase_last_heartbeat_at": _safe_int(
            payload.get("phase_last_heartbeat_at", payload.get("last_progress_at", 0))
        ),
        **stats,
    }
