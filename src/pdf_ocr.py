import concurrent.futures
from contextlib import contextmanager
import base64
import gc
import inspect
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.table_facts import build_flat_table_row_fact_text


_TEXT_KEYS = {
    "markdown",
    "text",
    "rec_text",
    "rec_texts",
    "ocr_text",
    "ocr_texts",
    "content",
    "contents",
    "full_text",
    "plain_text",
    "plaintext",
    "html",
    "latex",
}
_SKIP_RECURSIVE_KEYS = {
    "bbox",
    "boxes",
    "points",
    "polygon",
    "polygons",
    "score",
    "scores",
    "confidence",
    "confidences",
}
_PAGE_KEYS = ("page_no", "page", "page_id", "page_idx", "page_index")

_OCR_LOCK = threading.Lock()
_OCR_MODEL = None
_OCR_MODEL_NAME = ""
_OCR_MODEL_DEVICE = ""
_LAST_OCR_RUNTIME_INFO: Dict[str, Any] = {}
_LAST_OCR_SERIAL_TIMING_INFO: Dict[str, Any] = {}
_PERSISTENT_OCR_LOCK = threading.Lock()
_PERSISTENT_OCR_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None
_PERSISTENT_OCR_WORKERS = 0
_PERSISTENT_OCR_DEVICE = ""
_PERSISTENT_OCR_MODEL_NAME = ""
_PERSISTENT_OCR_READY_INFO: Dict[str, Any] = {}
_PERSISTENT_OCR_WARMUP_ERROR = ""
_OCR_DEFAULT_MODEL_NAME = "PaddleOCR-VL-1.5"
_OCR_FALLBACK_MODEL_NAME = "PaddleOCR-VL"
_OCR_VL_MODEL_BY_VERSION = {
    "v1.5": "PaddleOCR-VL-1.5-0.9B",
    "v1": "PaddleOCR-VL-0.9B",
}
_OCR_LAYOUT_MODEL_BY_VERSION = {
    "v1.5": "PP-DocLayoutV3",
    "v1": "PP-DocLayoutV2",
}
_OCR_DOC_ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori"
_OCR_DOC_UNWARP_MODEL = "UVDoc"

_OCR_NAME_ALIASES = {
    "paddleocr-vl-1.5": ("PaddleOCR-VL-1.5", "PaddleOCR-VL"),
    "paddleocr-vl": ("PaddleOCR-VL", "PaddleOCR-VL-1.5"),
}
_PDF_PARSE_MODES = {"hybrid", "ocr_first", "ocr_only", "text_only"}
_PDF_TEXT_EXTRACTORS = {"pymupdf", "disabled"}
_TABLE_HEADER_HINTS = (
    "단가",
    "금액",
    "조사항목",
    "항목",
    "구분",
    "비고",
    "답례품",
    "품목",
    "조사명",
    "수량",
    "비율",
    "건수",
    "작물명",
    "조사 시기",
    "조사시기",
    "보고 기일",
    "보고기일",
    "지급 기준월",
    "지급기준월",
    "지급 단가",
    "지급단가",
    "지급 단위",
    "지급단위",
    "조사 방법",
    "조사방법",
    "마감 시기",
    "마감시기",
    "지급대상월",
)
_TABLE_VALUE_UNIT_HINTS = ("천원", "만원", "억원", "원", "%", "퍼센트", "명", "건", "개", "호", "회")
_TABLE_VALUE_PATTERN = re.compile(
    r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(?:\(\s*(천원|만원|억원|원|퍼센트|%)\s*\)|(천원|만원|억원|원|퍼센트|%|명|건|개|호|회))"
)
_TABLE_NUMERIC_TOKEN_PATTERN = re.compile(
    r"^\d[\d,]*(?:\.\d+)?(?:년|월|일|천원|만원|억원|원|퍼센트|%|명|건|개|호|회)?$"
)
_TABLE_HEADER_UNIT_TOKENS = {
    "(천원)",
    "(만원)",
    "(억원)",
    "(원)",
    "(퍼센트)",
    "(%)",
}

_GIB = 1024 ** 3
_OCR_ASSUMED_TOTAL_GPU_MEMORY_GB = 32.0
_OCR_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 15.0


class PdfOCRExecutionError(RuntimeError):
    def __init__(self, message: str, *, runtime_info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.runtime_info = dict(runtime_info or {})


def _normalize_space(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", (text or "").strip())


def _normalize_block(text: str) -> str:
    lines = []
    for raw in (text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _to_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else 0
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if not m:
            return 0
        iv = int(m.group(0))
        return iv if iv > 0 else 0
    return 0


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_gpu_device(device: str) -> bool:
    normalized = (device or "").strip().lower()
    return normalized.startswith("gpu") or normalized.startswith("cuda")


def _gpu_budget_gb(default: float = 9.0) -> float:
    raw = os.getenv("PDF_OCR_GPU_BUDGET_GB")
    if raw is None:
        raw = os.getenv("PDF_OCR_GPU_MEMORY_LIMIT_GB")
    return max(1.0, _to_float(raw if raw is not None else str(default), default))


def _configure_paddle_gpu_memory_env(device: str) -> Dict[str, str]:
    if not _is_gpu_device(device):
        return {}

    allocator_strategy = (os.getenv("PDF_OCR_GPU_ALLOCATOR_STRATEGY", "auto_growth") or "auto_growth").strip()
    budget_gb = _gpu_budget_gb()
    initial_memory_mb = max(256, _to_positive_int(os.getenv("PDF_OCR_GPU_INITIAL_MEMORY_MB", "512")))
    reallocate_memory_mb = max(0, _to_positive_int(os.getenv("PDF_OCR_GPU_REALLOCATE_MEMORY_MB", "512")))
    assumed_total_gb = max(budget_gb, _to_float(os.getenv("PDF_OCR_GPU_ASSUMED_TOTAL_MEMORY_GB", "32"), 32.0))
    fraction = max(0.05, min(0.95, budget_gb / assumed_total_gb))

    updates = {
        "FLAGS_allocator_strategy": allocator_strategy,
        "FLAGS_initial_gpu_memory_in_mb": str(initial_memory_mb),
        "FLAGS_reallocate_gpu_memory_in_mb": str(reallocate_memory_mb),
        "FLAGS_fraction_of_gpu_memory_to_use": f"{fraction:.4f}",
    }
    for key, value in updates.items():
        os.environ[key] = value
    return updates


def _ocr_failure_reason(error: BaseException) -> str:
    text = str(error or "").strip().lower()
    if not text:
        return ""
    if (
        "resourceexhaustederror" in text
        or "out of memory error on gpu" in text
        or "cuda out of memory" in text
        or ("memoryerror" in text and "gpu" in text)
    ):
        return "gpu_oom"
    if (
        "cannot init gpu" in text
        or "failed to initialize gpu" in text
        or "gpu is not available" in text
        or "cuda driver version is insufficient" in text
        or "gpu device" in text and "not found" in text
        or "cannot set device" in text
    ):
        return "gpu_init_fail"
    if "no kernel image is available for execution on the device" in text:
        return "gpu_kernel_incompatible"
    return ""


def _ocr_failure_needs_cpu_fallback(error: BaseException, *, device: Optional[str] = None) -> bool:
    resolved_device = (device or os.getenv("PDF_OCR_DEVICE", "") or "").strip()
    if resolved_device and not _is_gpu_device(resolved_device):
        return False
    if not _env_enabled("PDF_OCR_GPU_FALLBACK_TO_CPU", False):
        return False
    reason = _ocr_failure_reason(error)
    if reason == "gpu_kernel_incompatible":
        return False
    return bool(reason)


def _reset_cached_ocr_model() -> None:
    global _OCR_MODEL, _OCR_MODEL_NAME, _OCR_MODEL_DEVICE
    with _OCR_LOCK:
        _OCR_MODEL = None
        _OCR_MODEL_NAME = ""
        _OCR_MODEL_DEVICE = ""
    gc.collect()


def _clear_paddle_gpu_cache() -> bool:
    try:
        import paddle
    except Exception:
        return False
    try:
        cuda_module = getattr(getattr(paddle, "device", None), "cuda", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
            return True
    except Exception:
        return False
    return False


def release_cached_ocr_model(*, device: Optional[str] = None, force: bool = False) -> bool:
    resolved_device = (device or _OCR_MODEL_DEVICE or os.getenv("PDF_OCR_DEVICE", "") or "").strip()
    should_release = force or (
        _is_gpu_device(resolved_device)
        and _env_enabled("PDF_OCR_RELEASE_GPU_MODEL_AFTER_RUN", True)
    )
    if not should_release:
        return False
    _reset_cached_ocr_model()
    _clear_paddle_gpu_cache()
    return True


def _set_last_ocr_runtime_info(info: Optional[Dict[str, Any]]) -> None:
    global _LAST_OCR_RUNTIME_INFO
    _LAST_OCR_RUNTIME_INFO = dict(info or {})


def _peek_last_ocr_runtime_info() -> Dict[str, Any]:
    return dict(_LAST_OCR_RUNTIME_INFO)


def _set_last_ocr_serial_timing_info(info: Optional[Dict[str, Any]]) -> None:
    global _LAST_OCR_SERIAL_TIMING_INFO
    _LAST_OCR_SERIAL_TIMING_INFO = dict(info or {})


def _peek_last_ocr_serial_timing_info() -> Dict[str, Any]:
    return dict(_LAST_OCR_SERIAL_TIMING_INFO)


def _merge_ocr_timing_info(target: Dict[str, Any], source: Optional[Dict[str, Any]]) -> None:
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for key in (
        "ocr_subset_build_seconds",
        "ocr_model_load_seconds",
        "ocr_predict_seconds",
        "ocr_output_materialize_seconds",
        "ocr_payload_convert_seconds",
        "ocr_text_merge_seconds",
        "ocr_merge_seconds",
    ):
        if key not in source:
            continue
        target[key] = round(float(target.get(key, 0.0) or 0.0) + float(source.get(key, 0.0) or 0.0), 3)
    if "ocr_batch_count" in source:
        target["ocr_batch_count"] = int(target.get("ocr_batch_count", 0) or 0) + int(source.get("ocr_batch_count", 0) or 0)
    batch_wall_seconds = source.get("ocr_batch_wall_seconds")
    if isinstance(batch_wall_seconds, list):
        target.setdefault("ocr_batch_wall_seconds", []).extend(
            float(value) for value in batch_wall_seconds if float(value) >= 0
        )


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _summarize_batch_wall_seconds(values: List[float]) -> Dict[str, float]:
    samples = [float(value) for value in values if float(value) >= 0]
    if not samples:
        return {
            "ocr_batch_wall_seconds_mean": 0.0,
            "ocr_batch_wall_seconds_p50": 0.0,
            "ocr_batch_wall_seconds_p95": 0.0,
            "ocr_batch_wall_seconds_max": 0.0,
        }
    return {
        "ocr_batch_wall_seconds_mean": round(sum(samples) / len(samples), 3),
        "ocr_batch_wall_seconds_p50": round(_percentile(samples, 0.50), 3),
        "ocr_batch_wall_seconds_p95": round(_percentile(samples, 0.95), 3),
        "ocr_batch_wall_seconds_max": round(max(samples), 3),
    }


def _emit_progress(
    progress_callback: Optional[Callable[..., None]],
    percent: int,
    message: str,
    stage: str,
    **progress_meta: Any,
) -> None:
    if progress_callback is None:
        return
    payload = {
        key: value
        for key, value in progress_meta.items()
        if value is not None
    }
    if payload:
        try:
            progress_callback(percent, message, stage, **payload)
            return
        except TypeError:
            pass
    progress_callback(percent, message, stage)


@contextmanager
def _ocr_progress_heartbeat(
    progress_callback: Optional[Callable[[int, str, str], None]],
    *,
    percent: int,
    message: str,
    stage: str,
    interval_seconds: Optional[float] = None,
    progress_meta: Optional[Dict[str, Any]] = None,
):
    if progress_callback is None:
        yield
        return

    resolved_interval = max(
        0.01,
        _to_float(
            interval_seconds if interval_seconds is not None else _OCR_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
            _OCR_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
        ),
    )
    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(resolved_interval):
            try:
                _emit_progress(
                    progress_callback,
                    percent,
                    message,
                    stage,
                    **dict(progress_meta or {}),
                )
            except Exception:
                break

    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        name=f"pdf-ocr-heartbeat-{stage or 'processing'}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=min(1.0, resolved_interval))


def _call_serial_ocr(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]],
    *,
    model_name: str,
    max_pages: int,
    min_text_chars: int,
    device: str,
    release_after_run_override: Optional[bool] = None,
    progress_meta: Optional[Dict[str, Any]] = None,
    original_pages_label: str = "",
) -> List[Dict[str, Any]]:
    primary_kwargs: Dict[str, Any] = {
        "progress_callback": progress_callback,
        "model_name": model_name,
        "max_pages": max_pages,
        "min_text_chars": min_text_chars,
        "device_override": device,
        "release_after_run_override": release_after_run_override,
        "progress_meta": progress_meta,
    }
    if str(original_pages_label or "").strip():
        primary_kwargs["original_pages_label"] = original_pages_label
    try:
        return _extract_pdf_pages_with_paddleocr_vl_serial(pdf_path, **primary_kwargs)
    except TypeError:
        try:
            return _extract_pdf_pages_with_paddleocr_vl_serial(
                pdf_path,
                progress_callback=progress_callback,
                model_name=model_name,
                max_pages=max_pages,
                min_text_chars=min_text_chars,
                device_override=device,
                original_pages_label=original_pages_label,
            )
        except TypeError:
            return _extract_pdf_pages_with_paddleocr_vl_serial(
                pdf_path,
                progress_callback=progress_callback,
                model_name=model_name,
                max_pages=max_pages,
                min_text_chars=min_text_chars,
            )


def _ocr_retry_mode_for(device: str, max_workers: int) -> str:
    if _is_gpu_device(device):
        return "parallel_gpu" if max(1, int(max_workers or 1)) > 1 else "single_gpu"
    return "cpu"


def _resolve_ocr_batch_timeout_seconds(
    tasks: List[Dict[str, Any]],
    *,
    device: str,
    max_workers: int,
) -> Optional[float]:
    if not _is_gpu_device(device):
        return None
    env_name = (
        "PDF_OCR_GPU_SINGLE_BATCH_TIMEOUT_SECONDS"
        if max(1, int(max_workers or 1)) <= 1
        else "PDF_OCR_GPU_BATCH_TIMEOUT_SECONDS"
    )
    default_seconds = 600.0 if max(1, int(max_workers or 1)) <= 1 else 420.0
    configured_seconds = _to_float(os.getenv(env_name, str(default_seconds)), default_seconds)
    largest_batch_pages = max(
        (len(list(task.get("page_numbers", []) or [])) for task in list(tasks or [])),
        default=1,
    )
    dynamic_floor = 120.0 + (
        largest_batch_pages * (120.0 if max(1, int(max_workers or 1)) <= 1 else 90.0)
    )
    return max(configured_seconds, dynamic_floor)


def _terminate_process_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    processes = getattr(executor, "_processes", None)
    if isinstance(processes, dict):
        for process in list(processes.values()):
            try:
                if process is not None and process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in list(processes.values()):
            try:
                if process is not None:
                    process.join(timeout=1.0)
            except Exception:
                pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _persistent_ocr_worker_enabled(device: str) -> bool:
    return (
        _is_gpu_device(device)
        and _env_enabled("PDF_OCR_GPU_PROCESS_ISOLATION", bool(_profile_default("PDF_OCR_GPU_PROCESS_ISOLATION")))
        and _env_enabled("PDF_OCR_PERSISTENT_WORKER", True)
    )


def _persistent_ocr_worker_count(requested_workers: int = 1) -> int:
    configured = max(1, int(os.getenv("PDF_OCR_PERSISTENT_WORKERS", "1") or "1"))
    return max(1, min(max(1, int(requested_workers or 1)), configured))


def _clear_persistent_ocr_ready_info(*, clear_error: bool = False) -> None:
    global _PERSISTENT_OCR_READY_INFO, _PERSISTENT_OCR_WARMUP_ERROR
    _PERSISTENT_OCR_READY_INFO = {}
    if clear_error:
        _PERSISTENT_OCR_WARMUP_ERROR = ""


def shutdown_persistent_ocr_worker(*, clear_error: bool = True) -> None:
    global _PERSISTENT_OCR_EXECUTOR, _PERSISTENT_OCR_WORKERS, _PERSISTENT_OCR_DEVICE, _PERSISTENT_OCR_MODEL_NAME
    executor = None
    with _PERSISTENT_OCR_LOCK:
        executor = _PERSISTENT_OCR_EXECUTOR
        _PERSISTENT_OCR_EXECUTOR = None
        _PERSISTENT_OCR_WORKERS = 0
        _PERSISTENT_OCR_DEVICE = ""
        _PERSISTENT_OCR_MODEL_NAME = ""
        _clear_persistent_ocr_ready_info(clear_error=clear_error)
    if executor is not None:
        _terminate_process_pool(executor)


def _ensure_persistent_ocr_worker(
    *,
    model_name: str,
    device: str,
    requested_workers: int = 1,
) -> concurrent.futures.ProcessPoolExecutor:
    global _PERSISTENT_OCR_EXECUTOR, _PERSISTENT_OCR_WORKERS, _PERSISTENT_OCR_DEVICE, _PERSISTENT_OCR_MODEL_NAME
    resolved_workers = _persistent_ocr_worker_count(requested_workers)
    resolved_device = (device or os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    resolved_model_name = (model_name or os.getenv("PDF_OCR_MODEL_NAME", _OCR_DEFAULT_MODEL_NAME) or _OCR_DEFAULT_MODEL_NAME).strip()
    stale_executor = None
    with _PERSISTENT_OCR_LOCK:
        if (
            _PERSISTENT_OCR_EXECUTOR is not None
            and _PERSISTENT_OCR_WORKERS == resolved_workers
            and _PERSISTENT_OCR_DEVICE == resolved_device
            and _PERSISTENT_OCR_MODEL_NAME == resolved_model_name
        ):
            return _PERSISTENT_OCR_EXECUTOR
        stale_executor = _PERSISTENT_OCR_EXECUTOR
        _PERSISTENT_OCR_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        _PERSISTENT_OCR_WORKERS = resolved_workers
        _PERSISTENT_OCR_DEVICE = resolved_device
        _PERSISTENT_OCR_MODEL_NAME = resolved_model_name
        _clear_persistent_ocr_ready_info()
        executor = _PERSISTENT_OCR_EXECUTOR
    if stale_executor is not None:
        try:
            stale_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            _terminate_process_pool(stale_executor)
    return executor


def _persistent_ocr_warmup_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    device = str(task.get("device", "cpu") or "cpu").strip()
    model_name = str(task.get("model_name", "") or _OCR_DEFAULT_MODEL_NAME).strip()
    if device:
        os.environ["PDF_OCR_DEVICE"] = device
    started_at = time.monotonic()
    _load_ocr_model(model_name=model_name, device=device)
    return {
        "status": "ready",
        "device": device,
        "model_name": model_name,
        "model_load_seconds": round(time.monotonic() - started_at, 3),
        "worker_pid": os.getpid(),
    }


def warmup_persistent_ocr_worker(
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    global _PERSISTENT_OCR_READY_INFO, _PERSISTENT_OCR_WARMUP_ERROR
    resolved_device = (device or os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    resolved_model_name = (model_name or os.getenv("PDF_OCR_MODEL_NAME", _OCR_DEFAULT_MODEL_NAME) or _OCR_DEFAULT_MODEL_NAME).strip()
    if (os.getenv("PDF_OCR_BACKEND", "local") or "local").strip().lower() == "hps":
        return {
            "status": "skipped",
            "device": resolved_device,
            "model_name": resolved_model_name,
            "reason": "hps_backend_selected",
        }
    requested_workers = max(1, int(os.getenv("PDF_OCR_PERSISTENT_WORKERS", "1") or "1"))
    resolved_workers = _persistent_ocr_worker_count(requested_workers)
    if not _persistent_ocr_worker_enabled(resolved_device):
        return {
            "status": "skipped",
            "device": resolved_device,
            "model_name": resolved_model_name,
            "reason": "persistent_gpu_worker_disabled",
        }
    with _PERSISTENT_OCR_LOCK:
        cached = dict(_PERSISTENT_OCR_READY_INFO)
    if (
        cached.get("status") == "ready"
        and cached.get("device") == resolved_device
        and cached.get("model_name") == resolved_model_name
        and int(cached.get("worker_count", 0) or 0) == resolved_workers
    ):
        return cached
    executor = _ensure_persistent_ocr_worker(
        model_name=resolved_model_name,
        device=resolved_device,
        requested_workers=resolved_workers,
    )
    timeout = timeout_seconds
    if timeout is None:
        timeout = _to_float(os.getenv("PDF_OCR_WARMUP_TIMEOUT_SECONDS", "900"), 900.0)
    futures = [
        executor.submit(
            _persistent_ocr_warmup_worker,
            {
                "model_name": resolved_model_name,
                "device": resolved_device,
            },
        )
        for _ in range(resolved_workers)
    ]
    try:
        deadline = time.monotonic() + max(1.0, float(timeout or 900.0))
        worker_results = [
            dict(future.result(timeout=max(0.05, deadline - time.monotonic())))
            for future in futures
        ]
        worker_pids = sorted(
            {
                int(item.get("worker_pid", 0) or 0)
                for item in worker_results
                if int(item.get("worker_pid", 0) or 0) > 0
            }
        )
        if len(worker_pids) != resolved_workers:
            raise RuntimeError(
                "Persistent GPU OCR warmup did not reach all distinct worker processes: "
                f"expected={resolved_workers} actual={len(worker_pids)} pids={worker_pids}"
            )
    except Exception as exc:
        with _PERSISTENT_OCR_LOCK:
            _PERSISTENT_OCR_WARMUP_ERROR = str(exc)
        shutdown_persistent_ocr_worker(clear_error=False)
        raise
    result = dict(worker_results[0])
    result["worker_count"] = len(worker_results)
    result["worker_pids"] = worker_pids
    result["model_load_seconds"] = round(
        max(float(item.get("model_load_seconds", 0.0) or 0.0) for item in worker_results),
        3,
    )
    with _PERSISTENT_OCR_LOCK:
        _PERSISTENT_OCR_READY_INFO = dict(result)
        _PERSISTENT_OCR_WARMUP_ERROR = ""
    return result


def _persistent_ocr_warmup_error() -> str:
    with _PERSISTENT_OCR_LOCK:
        return str(_PERSISTENT_OCR_WARMUP_ERROR or "")


def _execute_ocr_subset_tasks(
    tasks: List[Dict[str, Any]],
    *,
    model_name: str,
    min_text_chars: int,
    device: str,
    max_workers: int,
    stage: str,
    progress_callback: Optional[Callable[..., None]] = None,
    progress_percent_fn: Optional[Callable[[], int]] = None,
    progress_meta_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    batch_timeout_seconds: Optional[float] = None,
    on_batch_completed: Optional[
        Callable[[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    if not tasks:
        return {
            "completed": [],
            "remaining_tasks": [],
            "failure_reason": "",
        }

    resolved_workers = max(1, min(max(1, int(max_workers or 1)), len(tasks)))
    use_process_pool = (
        resolved_workers > 1
        or (batch_timeout_seconds or 0) > 0
        or (
            _is_gpu_device(device)
            and _env_enabled(
                "PDF_OCR_GPU_PROCESS_ISOLATION",
                bool(_profile_default("PDF_OCR_GPU_PROCESS_ISOLATION")),
            )
        )
    )
    heartbeat_interval_seconds = _resolve_ocr_progress_heartbeat_interval_seconds()
    message = (
        "GPU OCR이 실패해 CPU fallback으로 다시 시도하는 중입니다."
        if str(stage or "").strip().lower() == "fallback_pdf_ocr"
        else "PDF OCR을 실행하는 중입니다."
    )

    if not use_process_pool:
        completed: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        timing_info: Dict[str, Any] = {}
        for index, task in enumerate(list(tasks or [])):
            batch_started_at = time.monotonic()
            try:
                with _ocr_progress_heartbeat(
                    progress_callback,
                    percent=int(progress_percent_fn() if callable(progress_percent_fn) else 0),
                    message=message,
                    stage=stage,
                    interval_seconds=heartbeat_interval_seconds,
                    progress_meta=dict(progress_meta_fn() if callable(progress_meta_fn) else {}),
        ):
                    _set_last_ocr_serial_timing_info({})
                    subset_results = _call_serial_ocr(
                        str(task.get("pdf_path", "") or ""),
                        progress_callback=None,
                        model_name=model_name,
                        max_pages=len(list(task.get("page_numbers", []) or [])),
                        min_text_chars=min_text_chars,
                        device=device,
                        release_after_run_override=False,
                        original_pages_label=str(task.get("original_pages_label", "") or ""),
                    )
                    _merge_ocr_timing_info(timing_info, _peek_last_ocr_serial_timing_info())
                    batch_runtime_info = _peek_last_ocr_serial_timing_info()
                    batch_runtime_info["ocr_batch_wall_seconds"] = [
                        round(time.monotonic() - batch_started_at, 3)
                    ]
                    _merge_ocr_timing_info(timing_info, {"ocr_batch_wall_seconds": batch_runtime_info["ocr_batch_wall_seconds"]})
            except Exception as exc:
                return {
                    "completed": completed,
                    "remaining_tasks": list(tasks[index:]),
                    "failure_reason": _ocr_failure_reason(exc),
                    "error": exc,
                    "runtime_info": timing_info,
                }
            completed.append((task, subset_results))
            if callable(on_batch_completed):
                on_batch_completed(task, subset_results, batch_runtime_info)
        return {
            "completed": completed,
            "remaining_tasks": [],
            "failure_reason": "",
            "runtime_info": timing_info,
        }

    pending: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
    completed: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    timing_info: Dict[str, Any] = {}
    persistent_executor = _persistent_ocr_worker_enabled(device)
    warmup_error = _persistent_ocr_warmup_error() if persistent_executor else ""
    if warmup_error and not _env_enabled("PDF_OCR_GPU_FALLBACK_TO_CPU", False):
        error = PdfOCRExecutionError(f"Persistent GPU OCR warmup failed: {warmup_error}")
        return {
            "completed": [],
            "remaining_tasks": list(tasks),
            "failure_reason": "gpu_init_fail",
            "runtime_info": timing_info,
            "error": error,
        }
    if persistent_executor:
        executor = _ensure_persistent_ocr_worker(
            model_name=model_name,
            device=device,
            requested_workers=resolved_workers,
        )
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    task_queue = list(tasks or [])
    next_task_index = 0
    failure_reason = ""
    failure_error: Optional[BaseException] = None
    remaining_tasks: List[Dict[str, Any]] = []

    def _submit_task(task: Dict[str, Any]) -> None:
        nonlocal next_task_index
        task_pages = list(task.get("page_numbers", []) or [])
        first_page = int(task_pages[0]) if task_pages else 0
        last_page = int(task_pages[-1]) if task_pages else 0
        submit_meta = dict(progress_meta_fn() if callable(progress_meta_fn) else {})
        total_pages = int(submit_meta.get("total_pages", submit_meta.get("pdf_total_pages", 0)) or 0)
        if first_page > 0:
            submit_meta["current_page"] = first_page
            submit_meta["ocr_current_page"] = first_page
        if total_pages > 0:
            submit_meta["total_pages"] = total_pages
            submit_meta["pdf_total_pages"] = total_pages
        if progress_callback is not None and first_page > 0:
            page_label = f"{first_page}/{total_pages}페이지" if total_pages > 0 else f"{first_page}페이지"
            _emit_progress(
                progress_callback,
                percent=int(progress_percent_fn() if callable(progress_percent_fn) else 0),
                message=f"PDF OCR을 실행하는 중입니다. {page_label}",
                stage=stage,
                **submit_meta,
            )
        print(
            "[PDF_OCR][BATCH_SUBMIT] "
            f"device={device or 'cpu'} stage={stage} "
            f"batch={int(task.get('batch_index', 0) or 0)} "
            f"original_pages={first_page}-{last_page} "
            f"subset_page_count={len(task_pages)} "
            f"timeout_seconds={float(batch_timeout_seconds or 0):.1f}",
            flush=True,
        )
        future = executor.submit(
                _parallel_ocr_subset_worker,
                {
                    "pdf_path": task["pdf_path"],
                    "model_name": model_name,
                    "subset_page_count": len(task_pages),
                    "min_text_chars": min_text_chars,
                    "device": device,
                    "original_pages_label": f"{first_page}-{last_page}" if first_page and last_page else "",
                },
            )
        pending[future] = {
            "task": task,
            "submitted_at": time.monotonic(),
        }
        next_task_index += 1

    def _fill_worker_slots() -> None:
        while len(pending) < resolved_workers and next_task_index < len(task_queue):
            _submit_task(task_queue[next_task_index])

    try:
        _fill_worker_slots()
        while pending:
            wait_budget = None
            if batch_timeout_seconds and float(batch_timeout_seconds) > 0:
                now = time.monotonic()
                expired = [
                    (future, item)
                    for future, item in pending.items()
                    if now - float(item.get("submitted_at", now)) >= float(batch_timeout_seconds)
                ]
                if expired:
                    failure_reason = "gpu_timeout" if _is_gpu_device(device) else "ocr_timeout"
                    expired_task = expired[0][1]["task"]
                    expired_pages = list(expired_task.get("page_numbers", []) or [])
                    first_page = int(expired_pages[0]) if expired_pages else 0
                    last_page = int(expired_pages[-1]) if expired_pages else 0
                    elapsed_seconds = now - float(expired[0][1].get("submitted_at", now))
                    print(
                        "[PDF_OCR][BATCH_TIMEOUT] "
                        f"device={device or 'cpu'} stage={stage} "
                        f"batch={int(expired_task.get('batch_index', 0) or 0)} "
                        f"original_pages={first_page}-{last_page} "
                        f"elapsed_seconds={elapsed_seconds:.3f} "
                        f"timeout_seconds={float(batch_timeout_seconds):.3f}",
                        flush=True,
                    )
                    remaining_tasks = [
                        *[item["task"] for item in pending.values()],
                        *task_queue[next_task_index:],
                    ]
                    break
                wait_budget = min(
                    max(
                        0.0,
                        float(batch_timeout_seconds)
                        - (now - float(item.get("submitted_at", now))),
                    )
                    for item in pending.values()
                )
            wait_timeout = 5.0
            if wait_budget is not None:
                wait_timeout = max(0.05, min(wait_timeout, wait_budget))
            with _ocr_progress_heartbeat(
                progress_callback,
                percent=int(progress_percent_fn() if callable(progress_percent_fn) else 0),
                message=message,
                stage=stage,
                interval_seconds=heartbeat_interval_seconds,
                progress_meta=dict(progress_meta_fn() if callable(progress_meta_fn) else {}),
            ):
                done, _ = concurrent.futures.wait(
                    set(pending.keys()),
                    timeout=wait_timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            if not done:
                continue
            for future in done:
                pending_item = pending.pop(future)
                task = pending_item["task"]
                try:
                    future_result = future.result()
                except Exception as exc:
                    failure_reason = _ocr_failure_reason(exc)
                    if not failure_reason and _is_gpu_device(device):
                        failure_reason = "gpu_fail"
                    failure_error = exc
                    remaining_tasks = [
                        task,
                        *[item["task"] for item in pending.values()],
                        *task_queue[next_task_index:],
                    ]
                    break
                if isinstance(future_result, dict):
                    subset_results = list(future_result.get("pages", []) or [])
                    batch_runtime_info = dict(future_result.get("runtime_info", {}) or {})
                    _merge_ocr_timing_info(timing_info, batch_runtime_info)
                else:
                    subset_results = list(future_result or [])
                    batch_runtime_info = {}
                batch_wall_seconds = round(
                    time.monotonic() - float(pending_item.get("submitted_at", time.monotonic())),
                    3,
                )
                batch_runtime_info["ocr_batch_wall_seconds"] = [batch_wall_seconds]
                _merge_ocr_timing_info(timing_info, {"ocr_batch_wall_seconds": [batch_wall_seconds]})
                completed.append((task, subset_results))
                if callable(on_batch_completed):
                    on_batch_completed(task, subset_results, batch_runtime_info)
            if failure_reason:
                break
            _fill_worker_slots()
    finally:
        if persistent_executor:
            if failure_reason:
                print(
                    "[PDF_OCR][POOL_RESET] "
                    f"device={device or 'cpu'} stage={stage} reason={failure_reason} "
                    f"unfinished_batches={len(remaining_tasks)}",
                    flush=True,
                )
                shutdown_persistent_ocr_worker()
        elif failure_reason:
            print(
                "[PDF_OCR][POOL_RESET] "
                f"device={device or 'cpu'} stage={stage} reason={failure_reason} "
                f"unfinished_batches={len(remaining_tasks)}",
                flush=True,
            )
            _terminate_process_pool(executor)
        else:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                _terminate_process_pool(executor)

    result: Dict[str, Any] = {
        "completed": completed,
        "remaining_tasks": remaining_tasks,
        "failure_reason": failure_reason,
        "runtime_info": timing_info,
    }
    if failure_error is not None:
        result["error"] = failure_error
    return result


def _available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return max(0, int(parts[1])) * 1024
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(0, int(pages) * int(page_size))
    except Exception:
        return 0


def _gpu_device_index(device: str) -> int:
    normalized = (device or "").strip().lower()
    match = re.search(r":(\d+)$", normalized)
    if match:
        return max(0, int(match.group(1)))
    match = re.search(r"(?:gpu|cuda)(\d+)$", normalized)
    if match:
        return max(0, int(match.group(1)))
    return 0


def _available_gpu_memory_bytes(device: str) -> int:
    if not _is_gpu_device(device):
        return 0

    gpu_index = _gpu_device_index(device)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if completed.returncode == 0:
            first_line = next(
                (line.strip() for line in str(completed.stdout or "").splitlines() if line.strip()),
                "",
            )
            parts = [part.strip() for part in first_line.split(",") if part.strip()]
            free_mib = _to_positive_int(parts[0]) if parts else 0
            if free_mib > 0:
                free_bytes = free_mib * 1024 * 1024
                return min(free_bytes, int(_gpu_budget_gb() * _GIB))
    except Exception:
        pass
    return 0


def _recommended_parallel_ocr_workers(
    device: str,
    candidate_page_count: int,
    cpu_count: Optional[int] = None,
    available_bytes: Optional[int] = None,
) -> int:
    normalized_device = (device or "cpu").strip().lower()
    gpu_device = _is_gpu_device(normalized_device)
    max_workers = max(1, int(os.getenv("PDF_OCR_PARALLEL_MAX_WORKERS", "1")))
    min_pages = max(1, int(os.getenv("PDF_OCR_PARALLEL_MIN_PAGES", "24")))
    cpu_per_worker = max(2, int(os.getenv("PDF_OCR_PARALLEL_CPU_PER_WORKER", "4")))
    if gpu_device:
        mem_per_worker_gb = max(
            2.0,
            _to_float(
                os.getenv(
                    "PDF_OCR_PARALLEL_GPU_MEM_GB_PER_WORKER",
                    os.getenv("PDF_OCR_PARALLEL_MEM_GB_PER_WORKER", "4"),
                ),
                4.0,
            ),
        )
        reserve_gb = max(
            1.0,
            _to_float(
                os.getenv(
                    "PDF_OCR_PARALLEL_GPU_MEM_RESERVE_GB",
                    os.getenv("PDF_OCR_PARALLEL_MEM_RESERVE_GB", "2"),
                ),
                2.0,
            ),
        )
    else:
        mem_per_worker_gb = max(6.0, _to_float(os.getenv("PDF_OCR_PARALLEL_MEM_GB_PER_WORKER", "12"), 12.0))
        reserve_gb = max(4.0, _to_float(os.getenv("PDF_OCR_PARALLEL_MEM_RESERVE_GB", "8"), 8.0))

    safe_pages = max(0, int(candidate_page_count or 0))
    if max_workers < 2 or safe_pages < min_pages:
        return 1

    safe_cpu_count = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    cpu_limited_workers = min(max_workers, safe_cpu_count // cpu_per_worker)
    if cpu_limited_workers < 2:
        return 1

    safe_available_bytes = max(
        0,
        int(
            available_bytes
            if available_bytes is not None
            else (_available_gpu_memory_bytes(device) if gpu_device else _available_memory_bytes())
        ),
    )
    usable_bytes = max(0, safe_available_bytes - int(reserve_gb * _GIB))
    mem_limited_workers = usable_bytes // int(mem_per_worker_gb * _GIB) if usable_bytes > 0 else 0
    if mem_limited_workers < 2:
        return 1

    page_limited_workers = max(1, safe_pages // min_pages)
    workers = min(max_workers, cpu_limited_workers, mem_limited_workers, page_limited_workers)
    return max(1, int(workers))


def _split_parallel_ocr_page_batches(page_numbers: List[int], worker_count: int) -> List[List[int]]:
    ordered_pages = [int(page_no) for page_no in page_numbers if int(page_no) > 0]
    if not ordered_pages:
        return []
    if worker_count <= 1 or len(ordered_pages) <= 1:
        return [ordered_pages]
    batch_size = max(1, math.ceil(len(ordered_pages) / max(1, int(worker_count))))
    return [ordered_pages[idx : idx + batch_size] for idx in range(0, len(ordered_pages), batch_size)]


def _split_ocr_progress_page_batches(page_numbers: List[int], batch_size: int) -> List[List[int]]:
    ordered_pages = [int(page_no) for page_no in page_numbers if int(page_no) > 0]
    if not ordered_pages:
        return []
    resolved_batch_size = max(1, int(batch_size or 1))
    return [ordered_pages[idx : idx + resolved_batch_size] for idx in range(0, len(ordered_pages), resolved_batch_size)]


def _resolve_ocr_exec_batch_pages() -> int:
    raw = os.getenv("PDF_OCR_EXEC_BATCH_PAGES")
    if raw is None or not str(raw).strip():
        raw = os.getenv("PDF_OCR_PROGRESS_BATCH_PAGES", "16")
    return max(1, int(raw or "4"))


def _resolve_ocr_progress_heartbeat_interval_seconds() -> float:
    return max(
        0.5,
        _to_float(
            os.getenv("PDF_OCR_PROGRESS_HEARTBEAT_SECONDS", str(_OCR_PROGRESS_HEARTBEAT_INTERVAL_SECONDS)),
            _OCR_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
        ),
    )


def _run_pdf_ocr_progress_percent(current_page: int, total_pages: int) -> int:
    lower, upper = 49, 76
    safe_total = max(1, int(total_pages or 1))
    safe_current = max(0, min(int(current_page or 0), safe_total))
    ratio = max(0.0, min(1.0, safe_current / float(safe_total)))
    return max(lower, min(upper, lower + int(round((upper - lower) * ratio))))


def _extract_pdf_pages_with_paddleocr_vl_selected_pages(
    pdf_path: str,
    *,
    page_numbers: List[int],
    progress_callback: Optional[Callable[..., None]] = None,
    model_name: str,
    min_text_chars: int,
    device: str,
    total_document_pages: Optional[int] = None,
    completed_pages_base: int = 0,
    runtime_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    ordered_pages = sorted({int(page_no) for page_no in page_numbers if int(page_no) > 0})
    if not ordered_pages:
        return []

    safe_total_document_pages = max(
        len(ordered_pages),
        int(total_document_pages or 0) or _count_pdf_pages(pdf_path),
    )
    safe_completed_base = max(0, min(int(completed_pages_base or 0), safe_total_document_pages))
    ocr_target_pages = len(ordered_pages)
    worker_count = _recommended_parallel_ocr_workers(
        device=device,
        candidate_page_count=ocr_target_pages,
    )
    exec_batch_pages = _resolve_ocr_exec_batch_pages()
    page_batches = _split_ocr_progress_page_batches(ordered_pages, exec_batch_pages)
    if not page_batches:
        return []

    merged_pages: List[Dict[str, Any]] = []
    temp_paths: List[str] = []
    completed_ocr_pages = 0
    ocr_last_batch_completed_at = 0
    runtime_state = runtime_info if isinstance(runtime_info, dict) else {}
    runtime_state.setdefault("ocr_device_attempted", device)
    runtime_state.setdefault("ocr_device_effective", device)
    runtime_state.setdefault("ocr_gpu_fallback_used", False)
    runtime_state.setdefault("ocr_gpu_failure_reason", "")
    runtime_state.setdefault("ocr_retry_mode", _ocr_retry_mode_for(device, worker_count))
    runtime_state.setdefault("ocr_retry_reason", "")

    def _progress_meta() -> Dict[str, Any]:
        current_page = max(0, min(safe_total_document_pages, safe_completed_base + completed_ocr_pages))
        payload = {
            "current_page": current_page,
            "total_pages": safe_total_document_pages,
            "pdf_total_pages": safe_total_document_pages,
            "ocr_target_pages": ocr_target_pages,
            "ocr_completed_pages": completed_ocr_pages,
            "ocr_device_effective": str(runtime_state.get("ocr_device_effective", device) or device),
            "ocr_gpu_fallback_used": bool(runtime_state.get("ocr_gpu_fallback_used", False)),
            "ocr_gpu_failure_reason": str(runtime_state.get("ocr_gpu_failure_reason", "") or ""),
            "ocr_retry_mode": str(runtime_state.get("ocr_retry_mode", "") or ""),
            "ocr_retry_reason": str(runtime_state.get("ocr_retry_reason", "") or ""),
        }
        if ocr_last_batch_completed_at > 0:
            payload["ocr_last_batch_completed_at"] = int(ocr_last_batch_completed_at)
        return payload

    def _emit_run_progress(*, stage: str = "run_pdf_ocr", message: str = "PDF OCR을 실행하는 중입니다.") -> None:
        meta = _progress_meta()
        _emit_progress(
            progress_callback,
            _run_pdf_ocr_progress_percent(int(meta.get("current_page", 0) or 0), safe_total_document_pages),
            message,
            stage,
            **meta,
        )

    def _apply_completed_batches(
        completed_items: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
        *,
        mode: str,
        resolved_device: str,
        stage: str,
        message: str,
        total_batches: int,
    ) -> None:
        nonlocal completed_ocr_pages, ocr_last_batch_completed_at
        for task, subset_results in list(completed_items or []):
            subset_page_map = dict(task.get("subset_page_map", {}) or {})
            for page in list(subset_results or []):
                subset_page_no = max(1, int(page.get("page_no", 0) or 0))
                original_page_no = subset_page_map.get(subset_page_no, subset_page_no)
                merged_pages.append(
                    {
                        "page_no": int(original_page_no),
                        "text": str(page.get("text", "") or ""),
                    }
                )
            completed_ocr_pages += len(list(task.get("page_numbers", []) or []))
            ocr_last_batch_completed_at = int(time.time())
            _emit_run_progress(stage=stage, message=message)
            first_page = int(task["page_numbers"][0]) if list(task.get("page_numbers", []) or []) else 0
            last_page = int(task["page_numbers"][-1]) if list(task.get("page_numbers", []) or []) else 0
            print(
                f"[PDF_OCR][BATCH] device={resolved_device or 'cpu'} mode={mode} "
                f"batch={int(task.get('batch_index', 0) or 0)}/{total_batches} pages={first_page}-{last_page} "
                f"ocr_completed_pages={completed_ocr_pages}/{ocr_target_pages} "
                f"pdf_completed_pages={safe_completed_base + completed_ocr_pages}/{safe_total_document_pages}"
            )

    def _emit_retry_progress(mode: str, reason: str) -> None:
        if mode == "single_gpu":
            _emit_run_progress(
                stage="run_pdf_ocr",
                message="GPU OCR 메모리 부족 또는 지연으로 작업 수를 줄여 다시 시도하는 중입니다.",
            )
            return
        if mode == "cpu":
            _emit_run_progress(
                stage="fallback_pdf_ocr",
                message="GPU OCR이 계속 실패해 CPU로 다시 시도하는 중입니다.",
            )

    def _next_retry_plan(current_mode: str, failure_reason: str, remaining: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not remaining:
            return None
        normalized_reason = str(failure_reason or "").strip().lower()
        if current_mode == "parallel_gpu":
            if normalized_reason in {"gpu_oom", "gpu_timeout"}:
                return {
                    "mode": "single_gpu",
                    "device": device,
                    "max_workers": 1,
                    "stage": "run_pdf_ocr",
                    "message": "PDF OCR을 실행하는 중입니다.",
                }
            if normalized_reason in {"gpu_init_fail", "gpu_fail"} and _env_enabled("PDF_OCR_GPU_FALLBACK_TO_CPU", False):
                return {
                    "mode": "cpu",
                    "device": "cpu",
                    "max_workers": 1,
                    "stage": "fallback_pdf_ocr",
                    "message": "GPU OCR이 계속 실패해 CPU로 다시 시도하는 중입니다.",
                }
            return None
        if current_mode == "single_gpu" and normalized_reason in {"gpu_oom", "gpu_timeout", "gpu_init_fail", "gpu_fail"}:
            if _env_enabled("PDF_OCR_GPU_FALLBACK_TO_CPU", False):
                return {
                    "mode": "cpu",
                    "device": "cpu",
                    "max_workers": 1,
                    "stage": "fallback_pdf_ocr",
                    "message": "GPU OCR이 계속 실패해 CPU로 다시 시도하는 중입니다.",
                }
        return None

    print(
        f"[PDF_OCR][START] device={device or 'cpu'} total_pages={safe_total_document_pages} "
        f"ocr_page_limit={max(ordered_pages) if ordered_pages else 0} "
        f"ocr_target_pages={ocr_target_pages} worker_count={max(1, int(worker_count))} "
        f"exec_batch_pages={exec_batch_pages} batch_count={len(page_batches)}"
    )

    try:
        subset_build_started_at = time.monotonic()
        tasks: List[Dict[str, Any]] = []
        for batch_index, batch in enumerate(page_batches, start=1):
            subset_pdf_path, subset_page_map = _build_pdf_subset_for_pages(pdf_path, batch)
            temp_paths.append(subset_pdf_path)
            original_pages_label = f"{int(batch[0])}-{int(batch[-1])}" if batch else ""
            tasks.append(
                {
                    "batch_index": batch_index,
                    "page_numbers": list(batch),
                    "pdf_path": subset_pdf_path,
                    "subset_page_map": subset_page_map,
                    "original_pages_label": original_pages_label,
                }
            )
        runtime_state["ocr_subset_build_seconds"] = round(time.monotonic() - subset_build_started_at, 3)
        runtime_state["ocr_batch_count"] = len(tasks)

        _emit_run_progress()
        remaining_tasks = list(tasks)
        current_device = device
        current_workers = min(max(1, int(worker_count or 1)), len(remaining_tasks))
        current_mode = _ocr_retry_mode_for(current_device, current_workers)
        runtime_state["ocr_retry_mode"] = current_mode

        while remaining_tasks:
            current_stage = "fallback_pdf_ocr" if current_mode == "cpu" else "run_pdf_ocr"
            current_message = (
                "GPU OCR이 계속 실패해 CPU로 다시 시도하는 중입니다."
                if current_mode == "cpu"
                else "PDF OCR을 실행하는 중입니다."
            )
            current_timeout = None
            if current_mode in {"parallel_gpu", "single_gpu"}:
                current_timeout = _resolve_ocr_batch_timeout_seconds(
                    remaining_tasks,
                    device=current_device,
                    max_workers=current_workers,
                )

            def _on_batch_completed(
                task: Dict[str, Any],
                subset_results: List[Dict[str, Any]],
                _batch_runtime_info: Dict[str, Any],
            ) -> None:
                _apply_completed_batches(
                    [(task, subset_results)],
                    mode=current_mode,
                    resolved_device=current_device,
                    stage=current_stage,
                    message=current_message,
                    total_batches=len(tasks),
                )

            batch_result = _execute_ocr_subset_tasks(
                remaining_tasks,
                model_name=model_name,
                min_text_chars=min_text_chars,
                device=current_device,
                max_workers=current_workers,
                stage=current_stage,
                progress_callback=progress_callback,
                progress_percent_fn=lambda: _run_pdf_ocr_progress_percent(
                    safe_completed_base + completed_ocr_pages,
                    safe_total_document_pages,
                ),
                progress_meta_fn=_progress_meta,
                batch_timeout_seconds=current_timeout,
                on_batch_completed=_on_batch_completed,
            )
            _merge_ocr_timing_info(runtime_state, batch_result.get("runtime_info"))

            remaining_tasks = list(batch_result.get("remaining_tasks", []) or [])
            if not remaining_tasks:
                break

            failure_reason = str(batch_result.get("failure_reason", "") or "").strip().lower()
            runtime_state["ocr_retry_reason"] = failure_reason
            if failure_reason.startswith("gpu"):
                runtime_state["ocr_gpu_failure_reason"] = failure_reason
            next_plan = _next_retry_plan(current_mode, failure_reason, remaining_tasks)
            if not next_plan:
                error = batch_result.get("error")
                if isinstance(error, BaseException):
                    raise error
                if failure_reason == "gpu_timeout":
                    raise RuntimeError("GPU OCR batch timed out before any additional pages completed.")
                raise RuntimeError("PaddleOCR-VL subset execution failed.")

            current_mode = str(next_plan["mode"] or current_mode)
            current_device = str(next_plan["device"] or current_device)
            current_workers = max(1, int(next_plan["max_workers"] or 1))
            runtime_state["ocr_retry_mode"] = current_mode
            runtime_state["ocr_device_effective"] = current_device
            if current_mode == "cpu":
                runtime_state["ocr_gpu_fallback_used"] = True
            _emit_retry_progress(current_mode, failure_reason)

        merged_pages.sort(key=lambda item: int(item.get("page_no", 0) or 0))
        runtime_state.update(
            _summarize_batch_wall_seconds(
                list(runtime_state.get("ocr_batch_wall_seconds", []) or [])
            )
        )
        return merged_pages
    finally:
        release_cached_ocr_model(device=device)
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def _to_plain_data(value: Any, depth: int = 0) -> Any:
    if depth > 7:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _to_plain_data(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v, depth + 1) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_plain_data(to_dict(), depth + 1)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_plain_data(tolist(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            raw = {
                str(k): v
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
            if raw:
                return _to_plain_data(raw, depth + 1)
        except Exception:
            pass
    return str(value)


def _collect_text_fragments(node: Any, out: List[str]):
    if node is None:
        return
    if isinstance(node, str):
        text = _normalize_block(node)
        if text:
            out.append(text)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            key_lower = str(key).strip().lower()
            if key_lower in _SKIP_RECURSIVE_KEYS:
                continue
            if key_lower in _TEXT_KEYS:
                if isinstance(value, str):
                    text = _normalize_block(value)
                    if text:
                        out.append(text)
                    continue
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            text = _normalize_block(item)
                            if text:
                                out.append(text)
                        else:
                            _collect_text_fragments(item, out)
                    continue
            _collect_text_fragments(value, out)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_text_fragments(item, out)


def _bounded_page_candidate(value: Any, *, max_page: int) -> int:
    page = _to_positive_int(value)
    if page <= 0:
        return 0
    if max_page > 0 and page > max_page:
        return 0
    return page


def _extract_page_no(node: Any, fallback: int, *, max_page: int = 0) -> int:
    safe_fallback = max(1, int(fallback or 1))
    if max_page > 0 and safe_fallback > max_page:
        safe_fallback = max(1, int(max_page))
    if isinstance(node, dict):
        for key in _PAGE_KEYS:
            if key in node:
                page = _bounded_page_candidate(node.get(key), max_page=max_page)
                if page > 0:
                    return page
        for key in ("path", "input_path", "image_path", "img_path", "file", "filename"):
            value = node.get(key)
            if isinstance(value, str):
                lowered = value.lower()
                m = re.search(r"(?:^|[/_.-])(?:page|p)[_.-](\d{1,4})(?:\D|$)", lowered)
                if m:
                    page = _bounded_page_candidate(m.group(1), max_page=max_page)
                    if page > 0:
                        return page
        return safe_fallback
    return safe_fallback


def _dedupe_texts(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for text in values:
        normalized = _normalize_block(text)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _append_unique(values: List[str], value: str):
    candidate = (value or "").strip()
    if not candidate:
        return
    if candidate not in values:
        values.append(candidate)


def _normalize_pdf_parse_mode(raw_value: str) -> str:
    lowered = (raw_value or "").strip().lower()
    if lowered in _PDF_PARSE_MODES:
        return lowered
    if lowered in {"ocr", "paddleocr"}:
        return "ocr_only"
    if lowered in {"paddleocr_first", "paddleocr-first", "ocr-first"}:
        return "ocr_first"
    if lowered in {"text", "pymupdf"}:
        return "text_only"
    return "ocr_first"


def _normalize_pdf_text_extractor(raw_value: str) -> str:
    lowered = (raw_value or "").strip().lower()
    if lowered in {"", "auto", "pymupdf", "fitz", "text"}:
        return "pymupdf"
    if lowered in {"disabled", "disable", "none", "off", "ocr", "ocr_only"}:
        return "disabled"
    if lowered in _PDF_TEXT_EXTRACTORS:
        return lowered
    return "pymupdf"


def _count_nonspace_chars(text: str) -> int:
    return sum(1 for ch in (text or "") if not ch.isspace())


def _text_nonspace_ratio(text: str) -> float:
    raw = text or ""
    if not raw:
        return 0.0
    return _count_nonspace_chars(raw) / max(1, len(raw))


def _page_has_meaningful_text(text: str, min_chars: int, min_nonspace_ratio: float) -> bool:
    normalized = _normalize_block(text)
    if len(normalized) < max(1, int(min_chars)):
        return False
    return _text_nonspace_ratio(normalized) >= max(0.0, float(min_nonspace_ratio))


def _extract_table_value_hints(line: str) -> List[str]:
    compact_hints: List[str] = []
    for match in _TABLE_VALUE_PATTERN.finditer(line or ""):
        value = (match.group(1) or "").replace(",", "").strip()
        unit = (match.group(2) or match.group(3) or "").strip()
        if not value or not unit:
            continue
        hint = f"{value}{unit}"
        if hint not in compact_hints:
            compact_hints.append(hint)
    return compact_hints


def _is_table_header_like_line(line: str) -> bool:
    normalized = _normalize_space(line)
    if not normalized:
        return False
    header_hits = sum(1 for hint in _TABLE_HEADER_HINTS if hint in normalized)
    digit_count = len(re.findall(r"\d", normalized))
    return header_hits >= 1 and digit_count <= 1


def _is_table_value_like_line(line: str) -> bool:
    normalized = _normalize_space(line)
    if not normalized:
        return False
    if "|" in normalized or "\t" in line:
        return True
    numeric_hits = len(re.findall(r"(?<![\d.])\d[\d,]*(?:\.\d+)?", normalized))
    has_unit = any(unit in normalized for unit in _TABLE_VALUE_UNIT_HINTS)
    if numeric_hits >= 2:
        return True
    return bool(numeric_hits >= 1 and has_unit and len(normalized.split()) <= 12)


def _is_table_numeric_token(token: str) -> bool:
    normalized = (token or "").strip().strip(",")
    if not normalized:
        return False
    return bool(_TABLE_NUMERIC_TOKEN_PATTERN.fullmatch(normalized))


def _is_table_header_token(token: str) -> bool:
    normalized = _normalize_space(token)
    if not normalized:
        return False
    if normalized in _TABLE_HEADER_UNIT_TOKENS:
        return True
    return any(hint in normalized for hint in _TABLE_HEADER_HINTS)


def _looks_like_table_row_label(tokens: List[str]) -> bool:
    if not tokens:
        return False
    compact_tokens = [token for token in tokens if token]
    if not compact_tokens:
        return False
    if len(compact_tokens) > 8:
        return False
    has_word = any(re.search(r"[A-Za-z가-힣]", token) for token in compact_tokens)
    if not has_word:
        return False
    header_hits = sum(1 for token in compact_tokens if _is_table_header_token(token))
    return header_hits < len(compact_tokens)


def _split_flattened_header_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    last_header_idx = -1
    for idx, token in enumerate(tokens):
        if _is_table_header_token(token):
            last_header_idx = idx
    if 0 <= last_header_idx < len(tokens) - 1:
        return tokens[: last_header_idx + 1], tokens[last_header_idx + 1 :]
    return [], tokens


def _build_flattened_table_row_hints(lines: List[str]) -> List[str]:
    hints: List[str] = []
    for line in lines:
        tokens = [token for token in _normalize_space(line).split() if token]
        if len(tokens) < 8:
            continue
        header_tokens: List[str] = []
        cursor = 0
        found_rows = 0
        while cursor < len(tokens):
            start = cursor
            label_tokens: List[str] = []
            while cursor < len(tokens) and not _is_table_numeric_token(tokens[cursor]):
                label_tokens.append(tokens[cursor])
                cursor += 1
            numeric_tokens: List[str] = []
            while cursor < len(tokens) and _is_table_numeric_token(tokens[cursor]):
                numeric_tokens.append(tokens[cursor])
                cursor += 1
            if not found_rows and label_tokens:
                split_header, split_label = _split_flattened_header_tokens(label_tokens)
                if split_header:
                    header_tokens = split_header
                    label_tokens = split_label
            if _looks_like_table_row_label(label_tokens) and len(numeric_tokens) >= 2:
                row_text = " ".join(label_tokens + numeric_tokens[:12]).strip()
                if row_text:
                    header_text = " ".join(header_tokens).strip()
                    if header_text:
                        hint = f"표행요약: {header_text} | {row_text}"
                    else:
                        hint = f"표행요약: {row_text}"
                    if hint not in hints:
                        hints.append(hint)
                    if header_text:
                        fact_hint = build_flat_table_row_fact_text(header_text, row_text)
                        if fact_hint and fact_hint not in hints:
                            hints.append(fact_hint)
                    found_rows += 1
                continue
            if found_rows > 0:
                continue
            cursor = start + 1
    return hints


def _build_table_hint_lines(text: str) -> List[str]:
    normalized_lines = [_normalize_space(line) for line in (text or "").splitlines()]
    lines = [line for line in normalized_lines if line]
    if not lines:
        return []

    hints: List[str] = []
    current_header = ""
    for line in lines:
        if _is_table_header_like_line(line):
            current_header = line
            header_hint = f"표헤더: {line}"
            if header_hint not in hints:
                hints.append(header_hint)
            continue
        if not _is_table_value_like_line(line):
            current_header = ""
            continue
        row_hint = f"표행: {line}"
        if row_hint not in hints:
            hints.append(row_hint)
            if current_header:
                combined_hint = f"표행: {current_header} | {line}"
                if combined_hint not in hints:
                    hints.append(combined_hint)
                summary_hint = f"표행요약: {current_header} | {line}"
                if summary_hint not in hints:
                    hints.append(summary_hint)
                fact_hint = build_flat_table_row_fact_text(current_header, line)
                if fact_hint and fact_hint not in hints:
                    hints.append(fact_hint)
        for compact_value in _extract_table_value_hints(line):
            value_hint = f"표값: {compact_value}"
            if value_hint not in hints:
                hints.append(value_hint)
    for hint in _build_flattened_table_row_hints(lines):
        if hint not in hints:
            hints.append(hint)
    return hints


def _should_ocr_table_like_text_page(text: str) -> bool:
    table_hints = _build_table_hint_lines(text)
    if not table_hints:
        return False
    lines = [_normalize_space(line) for line in (text or "").splitlines()]
    normalized_lines = [line for line in lines if line]
    numeric_token_count = sum(
        1
        for token in re.findall(r"\S+", " ".join(normalized_lines))
        if _is_table_numeric_token(token)
    )
    if any(hint.startswith("표행요약:") for hint in table_hints):
        return numeric_token_count >= 4 or len(normalized_lines) >= 3
    if len(normalized_lines) <= 2 and numeric_token_count >= 3:
        return True
    return numeric_token_count >= 10


def _upload_ocr_enabled() -> bool:
    return _env_enabled("PDF_UPLOAD_OCR_ENABLED", False)


def _is_pymupdf_parser(parser: str) -> bool:
    normalized = (parser or "").strip().lower()
    return normalized.startswith("pymupdf")


def _dedupe_hint_lines(lines: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for raw_line in lines:
        normalized = _normalize_space(raw_line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_lazy_ocr_hints(
    *,
    page_no: int,
    table_like: bool,
    table_hints: Optional[List[str]] = None,
) -> List[str]:
    reason_hint = (
        f"OCR후보: 표형 페이지 | 페이지 {page_no} | 숫자 표 단가 금액 합계 지급 기준월 OCR 필요"
        if table_like
        else f"OCR후보: 텍스트 부족 페이지 | 페이지 {page_no} | 이미지 스캔 표 숫자 단가 금액 확인"
    )
    hints = [reason_hint]
    if table_like:
        hints.append(f"OCR후보: 표형 페이지 | 페이지 {page_no} | 표 원문 재확인 필요")
    else:
        hints.append(f"OCR후보: 텍스트 부족 페이지 | 페이지 {page_no} | 원문 확인 필요")
    for table_hint in list(table_hints or []):
        if table_hint.startswith(("표행:", "표행요약:", "표값:")):
            hints.append(table_hint)
    return _dedupe_hint_lines(hints)


def _extract_pdf_pages_with_pymupdf(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> Dict[str, Any]:
    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF 로더를 불러오지 못했습니다. `PyMuPDF` 설치를 확인해 주세요."
        ) from e

    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            if progress_callback is not None:
                progress_callback(
                    34,
                    f"PDF 텍스트를 빠르게 추출하는 중입니다. (0/{max(1, total_pages)})",
                    "extract_pdf_text",
                )
            pages: List[Dict[str, Any]] = []
            for page_index in range(total_pages):
                page = doc.load_page(page_index) if hasattr(doc, "load_page") else list(doc)[page_index]
                raw_text = ""
                get_text = getattr(page, "get_text", None)
                if callable(get_text):
                    try:
                        raw_text = get_text("text")
                    except TypeError:
                        raw_text = get_text()
                page_text = _normalize_block(str(raw_text or ""))
                pages.append(
                    {
                        "page_no": page_index + 1,
                        "text": page_text,
                        "parser": "pymupdf_text",
                    }
                )
                if progress_callback is not None and total_pages > 0:
                    current_page = page_index + 1
                    if (
                        current_page == 1
                        or current_page == total_pages
                        or current_page % max(1, total_pages // 8) == 0
                    ):
                        progress_callback(
                            34 + int(round((current_page / total_pages) * 7)),
                            f"PDF 텍스트를 빠르게 추출하는 중입니다. ({current_page}/{total_pages})",
                            "extract_pdf_text",
                        )
    except Exception as e:
        raise RuntimeError(f"PyMuPDF PDF 추출 실패: {e}") from e

    return {"pages": pages, "total_pages": total_pages}


def _build_pdf_subset_for_pages(
    pdf_path: str,
    page_numbers: List[int],
) -> Tuple[str, Dict[int, int]]:
    try:
        import fitz
    except Exception as e:
        raise RuntimeError("PyMuPDF subset 생성에 필요한 fitz import에 실패했습니다.") from e

    unique_pages = sorted({int(page_no) for page_no in page_numbers if int(page_no) > 0})
    if not unique_pages:
        raise RuntimeError("subset PDF 대상 페이지가 비어 있습니다.")

    source_doc = fitz.open(pdf_path)
    subset_doc = fitz.open()
    temp_path = ""
    try:
        subset_page_map: Dict[int, int] = {}
        for subset_page_no, original_page_no in enumerate(unique_pages, start=1):
            subset_doc.insert_pdf(
                source_doc,
                from_page=original_page_no - 1,
                to_page=original_page_no - 1,
            )
            subset_page_map[subset_page_no] = original_page_no

        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        subset_doc.save(temp_path)
        return temp_path, subset_page_map
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"OCR subset PDF 생성 실패: {e}") from e
    finally:
        close_subset = getattr(subset_doc, "close", None)
        if callable(close_subset):
            close_subset()
        close_source = getattr(source_doc, "close", None)
        if callable(close_source):
            close_source()


def _count_pdf_pages(pdf_path: str) -> int:
    try:
        import fitz
    except Exception:
        return 0

    try:
        with fitz.open(pdf_path) as doc:
            return max(0, int(len(doc)))
    except Exception:
        return 0


def _build_pdf_result(
    *,
    total_pages: int,
    pages: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    attempted_ocr_pages: int = 0,
    ocr_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ordered_pages: List[Dict[str, Any]] = []
    seen_pages = set()
    text_pages = 0
    ocr_pages = 0
    for page in sorted(pages, key=lambda item: int(item.get("page_no", 0) or 0)):
        page_no = max(1, int(page.get("page_no", 0) or 0))
        if page_no in seen_pages:
            continue
        seen_pages.add(page_no)
        parser = str(page.get("parser", "") or "").strip() or "paddleocr_vl"
        text = _normalize_block(str(page.get("text", "") or ""))
        lazy_ocr_hints = _dedupe_hint_lines(list(page.get("lazy_ocr_hints", []) or []))
        if not text and not lazy_ocr_hints:
            continue
        table_hints = _build_table_hint_lines(text)
        normalized_page = {
            "page_no": page_no,
            "text": text,
            "parser": parser,
            "table_like": bool(table_hints),
            "table_hints": table_hints,
        }
        if lazy_ocr_hints:
            normalized_page["lazy_ocr_hints"] = lazy_ocr_hints
        ordered_pages.append(normalized_page)
        if _is_pymupdf_parser(parser):
            text_pages += 1
        else:
            ocr_pages += 1

    safe_total_pages = max(len(ordered_pages), int(total_pages or 0))
    warning_list = [str(item).strip() for item in (warnings or []) if str(item).strip()]
    failed_pages = max(0, safe_total_pages - len(ordered_pages))
    parser = "pymupdf_text"
    if ocr_pages and text_pages:
        parser = "hybrid_pdf"
    elif ocr_pages:
        parser = "paddleocr_vl"
    elif warning_list or attempted_ocr_pages > 0 or failed_pages > 0:
        parser = "hybrid_pdf"

    runtime_payload = dict(ocr_runtime or {})
    return {
        "parser": parser,
        "pages": ordered_pages,
        "total_pages": safe_total_pages,
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": failed_pages,
        "attempted_ocr_pages": max(0, int(attempted_ocr_pages)),
        "warnings": warning_list,
        "ocr_device_attempted": str(runtime_payload.get("ocr_device_attempted", "") or "").strip(),
        "ocr_device_effective": str(runtime_payload.get("ocr_device_effective", "") or "").strip(),
        "ocr_gpu_fallback_used": bool(runtime_payload.get("ocr_gpu_fallback_used", False)),
        "ocr_gpu_failure_reason": str(runtime_payload.get("ocr_gpu_failure_reason", "") or "").strip(),
        "ocr_elapsed_seconds": runtime_payload.get("ocr_elapsed_seconds"),
        "ocr_pages_processed": runtime_payload.get("ocr_pages_processed"),
        "ocr_pages_per_minute": runtime_payload.get("ocr_pages_per_minute"),
        "ocr_target_pages": runtime_payload.get("ocr_target_pages"),
        "ocr_target_seconds": runtime_payload.get("ocr_target_seconds"),
        "ocr_target_met": runtime_payload.get("ocr_target_met"),
        "ocr_subset_build_seconds": runtime_payload.get("ocr_subset_build_seconds"),
        "ocr_model_load_seconds": runtime_payload.get("ocr_model_load_seconds"),
        "ocr_predict_seconds": runtime_payload.get("ocr_predict_seconds"),
        "ocr_output_materialize_seconds": runtime_payload.get("ocr_output_materialize_seconds"),
        "ocr_payload_convert_seconds": runtime_payload.get("ocr_payload_convert_seconds"),
        "ocr_text_merge_seconds": runtime_payload.get("ocr_text_merge_seconds"),
        "ocr_merge_seconds": runtime_payload.get("ocr_merge_seconds"),
        "ocr_batch_count": runtime_payload.get("ocr_batch_count"),
        "ocr_backend": runtime_payload.get("ocr_backend"),
        "ocr_backend_attempted": runtime_payload.get("ocr_backend_attempted"),
        "ocr_backend_effective": runtime_payload.get("ocr_backend_effective"),
        "ocr_backend_fallback_used": runtime_payload.get("ocr_backend_fallback_used"),
        "ocr_hps_chunk_pages": runtime_payload.get("ocr_hps_chunk_pages"),
        "ocr_hps_max_concurrency": runtime_payload.get("ocr_hps_max_concurrency"),
        "ocr_batch_wall_seconds_mean": runtime_payload.get("ocr_batch_wall_seconds_mean"),
        "ocr_batch_wall_seconds_p50": runtime_payload.get("ocr_batch_wall_seconds_p50"),
        "ocr_batch_wall_seconds_p95": runtime_payload.get("ocr_batch_wall_seconds_p95"),
        "ocr_batch_wall_seconds_max": runtime_payload.get("ocr_batch_wall_seconds_max"),
    }


def _selected_ocr_first_page_numbers(pdf_path: str) -> Tuple[int, Optional[List[int]], bool]:
    total_pages = _count_pdf_pages(pdf_path)
    max_pages = max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "400")))
    if total_pages <= 0:
        return 0, None, False
    selected_count = min(max_pages, total_pages)
    return total_pages, list(range(1, selected_count + 1)), total_pages > selected_count


def _extract_ocr_first_pymupdf_pages(
    pdf_path: str,
    *,
    min_text_chars: int,
    min_nonspace_ratio: float,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    allowed_page_numbers: Optional[set[int]] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    text_result = _extract_pdf_pages_with_pymupdf(
        pdf_path,
        progress_callback=progress_callback,
    )
    total_pages = max(0, int(text_result.get("total_pages", 0) or 0))
    pages: List[Dict[str, Any]] = []
    allowed = set(allowed_page_numbers or set())
    for page in list(text_result.get("pages", []) or []):
        page_no = max(1, int(page.get("page_no", 0) or 0))
        if allowed and page_no not in allowed:
            continue
        page_text = _normalize_block(str(page.get("text", "") or ""))
        if not _page_has_meaningful_text(page_text, min_text_chars, min_nonspace_ratio):
            continue
        pages.append(
            {
                "page_no": page_no,
                "text": page_text,
                "parser": "pymupdf_text",
            }
        )
    return total_pages, pages


def _is_likely_path(raw: str) -> bool:
    value = (raw or "").strip()
    if not value:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    if value.startswith("$") or "${" in value:
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    return os.path.exists(os.path.expanduser(os.path.expandvars(value)))


def _resolve_main_backend_home() -> str:
    from_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project-gpu", "main-backend"))
    env_home = (os.getenv("MAIN_BACKEND_HOME", "") or "").strip()
    if env_home:
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(env_home)))
        return expanded
    return from_file


def _candidate_local_ocr_dirs() -> List[str]:
    main_backend_home = _resolve_main_backend_home()
    ocr_root = os.path.join(main_backend_home, "models", "ocr")
    return [
        os.path.join(ocr_root, "PaddleOCR-VL-1.5"),
        os.path.join(ocr_root, "PaddleOCR-VL"),
    ]


def _allow_online_model_fallback() -> bool:
    return _env_enabled("PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK", False)


def _resolve_model_candidates(raw_model_name: str) -> List[str]:
    raw = (raw_model_name or "").strip()
    candidates: List[str] = []
    allow_online_fallback = _allow_online_model_fallback()

    # 1) Explicit user value first (including env-expanded path).
    if raw:
        expanded = os.path.expanduser(os.path.expandvars(raw))
        if _is_likely_path(expanded) or allow_online_fallback:
            _append_unique(candidates, expanded)
        if expanded != raw and allow_online_fallback:
            _append_unique(candidates, raw)

    # 2) Known aliases to absorb naming differences.
    if allow_online_fallback and raw and not _is_likely_path(raw):
        lowered = raw.lower()
        for alias in _OCR_NAME_ALIASES.get(lowered, ()):
            _append_unique(candidates, alias)

    # 3) Prefer local offline snapshots when available.
    for local_dir in _candidate_local_ocr_dirs():
        if os.path.isdir(local_dir):
            _append_unique(candidates, local_dir)

    # 4) Canonical online model names are opt-in only for closed-network installs.
    if allow_online_fallback:
        _append_unique(candidates, _OCR_DEFAULT_MODEL_NAME)
        _append_unique(candidates, _OCR_FALLBACK_MODEL_NAME)

    return candidates


def _safe_signature_params(callable_obj: Any) -> Dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return {}


def _read_text_prefix(path: str, max_chars: int = 8192) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except OSError:
        return ""


def _looks_like_paddlex_model_dir(model_dir: str, expected_model_name: str = "") -> bool:
    if not os.path.isdir(model_dir):
        return False
    config_path = os.path.join(model_dir, "inference.yml")
    if not os.path.isfile(config_path):
        return False
    if not expected_model_name:
        return True
    payload = _read_text_prefix(config_path)
    return not payload or expected_model_name in payload


def _infer_pipeline_version(raw_value: str) -> str:
    raw = (raw_value or "").strip()
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if os.path.isdir(expanded):
        payload = _read_text_prefix(os.path.join(expanded, "inference.yml")).lower()
        if "paddleocr-vl-1.5" in payload:
            return "v1.5"
        if "paddleocr-vl-0.9b" in payload or "paddleocr-vl" in payload:
            return "v1"

    lowered = raw.lower()
    if "1.5" in lowered:
        return "v1.5"
    return "v1"


def _append_unique_dir(values: List[str], path: str):
    candidate = os.path.abspath(os.path.expanduser(os.path.expandvars(path or "")))
    if candidate and os.path.isdir(candidate) and candidate not in values:
        values.append(candidate)


def _resolve_env_model_dir(env_name: str) -> Optional[str]:
    raw = (os.getenv(env_name, "") or "").strip()
    if not raw:
        return None
    candidate = os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
    if os.path.isdir(candidate):
        return candidate
    return None


def _build_search_roots(local_source: str) -> List[str]:
    roots: List[str] = []
    main_backend_home = _resolve_main_backend_home()
    ocr_root = os.path.join(main_backend_home, "models", "ocr")
    for path in (
        local_source,
        os.path.dirname(local_source),
        ocr_root,
    ):
        _append_unique_dir(roots, path)
    return roots


def _find_model_dir_by_name(search_roots: List[str], model_name: str) -> Optional[str]:
    if not model_name:
        return None
    target = model_name.strip().lower()
    for root in search_roots:
        if os.path.basename(root).strip().lower() == target and _looks_like_paddlex_model_dir(root, model_name):
            return root
        candidate = os.path.join(root, model_name)
        if _looks_like_paddlex_model_dir(candidate, model_name):
            return candidate
    return None


def _resolve_local_pipeline_overrides(local_source: str, pipeline_version: str) -> Dict[str, Any]:
    source = os.path.abspath(os.path.expanduser(os.path.expandvars(local_source)))
    search_roots = _build_search_roots(source)
    vl_model_name = _OCR_VL_MODEL_BY_VERSION.get(pipeline_version, _OCR_VL_MODEL_BY_VERSION["v1.5"])
    layout_model_name = _OCR_LAYOUT_MODEL_BY_VERSION.get(
        pipeline_version,
        _OCR_LAYOUT_MODEL_BY_VERSION["v1.5"],
    )

    overrides: Dict[str, Any] = {}
    explicit_vl_model_dir = _resolve_env_model_dir("PDF_OCR_VL_MODEL_DIR")
    if explicit_vl_model_dir:
        overrides["vl_rec_model_dir"] = explicit_vl_model_dir
    elif _looks_like_paddlex_model_dir(source, vl_model_name):
        overrides["vl_rec_model_dir"] = source
    else:
        vl_model_dir = _find_model_dir_by_name(search_roots, vl_model_name)
        if vl_model_dir:
            overrides["vl_rec_model_dir"] = vl_model_dir

    layout_model_dir = _resolve_env_model_dir("PDF_OCR_LAYOUT_MODEL_DIR")
    if not layout_model_dir:
        layout_model_dir = _find_model_dir_by_name(search_roots, layout_model_name)
    if layout_model_dir:
        overrides["layout_detection_model_dir"] = layout_model_dir

    doc_orientation_dir = _resolve_env_model_dir("PDF_OCR_DOC_ORIENTATION_MODEL_DIR")
    if not doc_orientation_dir:
        doc_orientation_dir = _find_model_dir_by_name(search_roots, _OCR_DOC_ORIENTATION_MODEL)
    if doc_orientation_dir:
        overrides["doc_orientation_classify_model_dir"] = doc_orientation_dir

    doc_unwarp_dir = _resolve_env_model_dir("PDF_OCR_DOC_UNWARP_MODEL_DIR")
    if not doc_unwarp_dir:
        doc_unwarp_dir = _find_model_dir_by_name(search_roots, _OCR_DOC_UNWARP_MODEL)
    if doc_unwarp_dir:
        overrides["doc_unwarping_model_dir"] = doc_unwarp_dir

    return overrides


def _optional_env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return None


def _optional_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


def _optional_env_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _optional_env_json_dict(name: str) -> Optional[Dict[str, Any]]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _ocr_optimization_profile() -> str:
    return (os.getenv("PDF_OCR_OPTIMIZATION_PROFILE", "") or "").strip().lower()


def _v100_fast_profile_enabled() -> bool:
    return _ocr_optimization_profile() in {"v100_32gb_fast", "v100-fast", "fast_v100"}


def _h100_fast_profile_enabled() -> bool:
    return _ocr_optimization_profile() in {"h100_96gb_fast", "h100-fast", "fast_h100"}


def _profile_default(name: str) -> Any:
    if _h100_fast_profile_enabled():
        defaults: Dict[str, Any] = {
            "PDF_OCR_USE_INTERNAL_QUEUES": True,
            "PDF_OCR_GPU_PROCESS_ISOLATION": True,
            "PDF_OCR_USE_CHART_RECOGNITION": False,
            "PDF_OCR_USE_SEAL_RECOGNITION": False,
            "PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK": False,
            "PDF_OCR_MAX_NEW_TOKENS": 768,
            "PDF_OCR_MIN_PIXELS": 3136,
            "PDF_OCR_MAX_PIXELS": 786432,
            "PDF_OCR_LAYOUT_SHAPE_MODE": "rect",
            "PDF_OCR_VL_REC_MAX_CONCURRENCY": 1,
        }
        return defaults.get(name)
    if not _v100_fast_profile_enabled():
        return None
    defaults: Dict[str, Any] = {
        "PDF_OCR_USE_INTERNAL_QUEUES": True,
        "PDF_OCR_GPU_PROCESS_ISOLATION": True,
        "PDF_OCR_USE_CHART_RECOGNITION": False,
        "PDF_OCR_USE_SEAL_RECOGNITION": False,
        "PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK": False,
        "PDF_OCR_MAX_NEW_TOKENS": 512,
        "PDF_OCR_MIN_PIXELS": 3136,
        "PDF_OCR_MAX_PIXELS": 589824,
        "PDF_OCR_LAYOUT_SHAPE_MODE": "rect",
        "PDF_OCR_VL_REC_MAX_CONCURRENCY": 1,
    }
    return defaults.get(name)


def _env_or_profile_bool(name: str) -> Optional[bool]:
    value = _optional_env_bool(name)
    if value is not None:
        return value
    profile_value = _profile_default(name)
    return bool(profile_value) if isinstance(profile_value, bool) else None


def _env_or_profile_int(name: str) -> Optional[int]:
    value = _optional_env_int(name)
    if value is not None:
        return value
    profile_value = _profile_default(name)
    return int(profile_value) if isinstance(profile_value, int) and not isinstance(profile_value, bool) else None


def _env_or_profile_str(name: str) -> Optional[str]:
    value = _optional_env_str(name)
    if value is not None:
        return value
    profile_value = _profile_default(name)
    return str(profile_value) if isinstance(profile_value, str) and profile_value else None


def _apply_signature_safe_ocr_tuning_kwargs(kwargs: Dict[str, Any], params: Dict[str, inspect.Parameter]) -> None:
    for key, env_name in {
        "use_chart_recognition": "PDF_OCR_USE_CHART_RECOGNITION",
        "use_seal_recognition": "PDF_OCR_USE_SEAL_RECOGNITION",
        "use_ocr_for_image_block": "PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK",
    }.items():
        if key in params:
            value = _env_or_profile_bool(env_name)
            if value is not None:
                kwargs[key] = value

    for key, env_name in {
        "max_new_tokens": "PDF_OCR_MAX_NEW_TOKENS",
        "min_pixels": "PDF_OCR_MIN_PIXELS",
        "max_pixels": "PDF_OCR_MAX_PIXELS",
        "vl_rec_max_concurrency": "PDF_OCR_VL_REC_MAX_CONCURRENCY",
    }.items():
        if key in params:
            value = _env_or_profile_int(env_name)
            if value is not None:
                kwargs[key] = value

    for key, env_name in {
        "layout_shape_mode": "PDF_OCR_LAYOUT_SHAPE_MODE",
        "engine": "PDF_OCR_ENGINE",
    }.items():
        if key in params:
            value = _env_or_profile_str(env_name)
            if value is not None:
                kwargs[key] = value

    if "vlm_extra_args" in params:
        value = _optional_env_json_dict("PDF_OCR_VLM_EXTRA_ARGS_JSON")
        if value is not None:
            kwargs["vlm_extra_args"] = value


def _build_paddleocrvl_kwargs(
    paddleocrvl_cls: Any,
    candidate: str,
    requested_model_name: str,
    device: str,
) -> Dict[str, Any]:
    params = _safe_signature_params(paddleocrvl_cls)
    # Older PaddleOCR builds accepted `model_name` directly.
    if "model_name" in params:
        kwargs: Dict[str, Any] = {"model_name": candidate}
        if device:
            kwargs["device"] = device
        return kwargs

    pipeline_version = _infer_pipeline_version(candidate or requested_model_name)
    kwargs = {"pipeline_version": pipeline_version}

    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(candidate)))
    if os.path.isdir(expanded):
        for key, value in _resolve_local_pipeline_overrides(expanded, pipeline_version).items():
            if key in params and value:
                kwargs[key] = value

    if "use_queues" in params:
        queue_value = _env_or_profile_bool("PDF_OCR_USE_INTERNAL_QUEUES")
        kwargs["use_queues"] = bool(queue_value) if queue_value is not None else False

    _apply_signature_safe_ocr_tuning_kwargs(kwargs, params)

    if device:
        kwargs["device"] = device
    return kwargs


def _prepare_local_ocr_import_paths(candidate: str, kwargs: Dict[str, Any]) -> None:
    candidate_path = os.path.abspath(os.path.expanduser(os.path.expandvars(candidate or "")))
    import_paths: List[str] = []
    for path in (candidate_path, os.path.dirname(candidate_path)):
        _append_unique_dir(import_paths, path)
    for key in (
        "vl_rec_model_dir",
        "layout_detection_model_dir",
        "doc_orientation_classify_model_dir",
        "doc_unwarping_model_dir",
    ):
        _append_unique_dir(import_paths, kwargs.get(key, ""))
        _append_unique_dir(import_paths, os.path.dirname(str(kwargs.get(key, "") or "")))

    # Local HF/PaddleOCR-VL bundles may ship custom python modules alongside weights.
    # Putting the bundle directories on sys.path helps older PaddleOCR/PaddleX loaders
    # import files such as configuration_paddleocr_vl.py in offline deployments.
    for path in reversed(import_paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _build_ocr_runtime_hint(module_names: List[str]) -> str:
    normalized = sorted({(name or "").strip() for name in module_names if (name or "").strip()})
    if not normalized:
        return (
            "실제 누락 모듈명은 오류 목록을 확인해 주세요. "
            f"현재 backend python={sys.executable}. "
        )

    joined = ", ".join(normalized)
    common_runtime_modules = {
        "paddle",
        "openai",
        "einops",
        "sentencepiece",
        "tiktoken",
        "ftfy",
        "premailer",
        "lxml",
        "bs4",
        "bidi",
        "regex",
        "sklearn",
        "scipy",
    }
    local_bundle_modules = {
        "configuration_paddleocr_vl",
        "modeling_paddleocr_vl",
        "processing_paddleocr_vl",
        "image_processing_paddleocr_vl",
    }

    if any(name == "paddle" for name in normalized):
        return (
            f"실제 누락 모듈={joined}. "
            f"현재 backend python={sys.executable}. "
            "backend venv에 PaddlePaddle 런타임(`paddlepaddle-gpu` 또는 `paddlepaddle`)이 아직 없습니다. "
        )
    if any(name in common_runtime_modules for name in normalized):
        return (
            f"실제 누락 모듈={joined}. "
            f"현재 backend python={sys.executable}. "
            "backend venv가 다른 경로를 보고 있거나, 오프라인 설치가 부분 성공 상태일 수 있습니다. "
        )
    if any(name in local_bundle_modules for name in normalized):
        return (
            f"실제 누락 모듈={joined}. "
            f"현재 backend python={sys.executable}. "
            "PaddleOCR-VL 로컬 모델 폴더 import 경로 또는 모델 파일 구성을 먼저 확인해 주세요. "
        )
    return (
        f"실제 누락 모듈={joined}. "
        f"현재 backend python={sys.executable}. "
    )


def _build_ocr_error_hint(errors: List[str]) -> str:
    joined = " ".join((item or "") for item in errors)
    if "fused_rms_norm_ext" in joined:
        return (
            "현재 backend venv의 PaddlePaddle 런타임이 PaddleOCR-VL이 기대하는 연산을 제공하지 않습니다. "
            "`paddlepaddle-gpu`를 3.3.0 계열로 올려 다시 설치해 주세요. "
            "CUDA 12.2 서버라면 Paddle 공식 설치 선택지에는 11.8 또는 12.6이 있으므로, "
            "보수적으로는 cu118 wheel 세트를 쓰는 편이 안전합니다. "
        )
    lowered = joined.lower()
    if (
        "resourceexhaustederror" in lowered
        or "out of memory error on gpu" in lowered
        or "cuda out of memory" in lowered
        or ("memoryerror" in lowered and "gpu" in lowered)
    ):
        return (
            "현재 실패의 1차 원인은 로컬 모델 누락이 아니라 GPU 메모리 부족입니다. "
            "PaddleOCR-VL 로딩 중 GPU 여유 메모리가 부족해 backend OCR이 시작 단계에서 중단됐습니다. "
            "backend OCR을 다른 GPU로 분리하거나 `PDF_OCR_DEVICE=cpu`로 내려서 재시도해 주세요. "
            "같은 GPU에서 임베딩 서버나 LLM 서버를 함께 돌리는 중이면 분리가 필요합니다. "
        )
    if "No model source is available" in joined:
        return (
            "폐쇄망 서버에서 PaddleX가 온라인 다운로드로 fallback 했습니다. "
            "로컬 OCR 모델 경로와 보조 모델(레이아웃/문서 회전/언워프) 디렉터리 구성을 다시 확인해 주세요. "
        )
    return ""


def _load_ocr_model(model_name: str, device: Optional[str] = None):
    global _OCR_MODEL, _OCR_MODEL_NAME, _OCR_MODEL_DEVICE
    candidates = _resolve_model_candidates(model_name)
    resolved_device = (device or os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    _configure_paddle_gpu_memory_env(resolved_device)
    with _OCR_LOCK:
        if _OCR_MODEL is not None and _OCR_MODEL_NAME in candidates and _OCR_MODEL_DEVICE == resolved_device:
            return _OCR_MODEL
        if _OCR_MODEL is not None and _OCR_MODEL_DEVICE != resolved_device:
            _OCR_MODEL = None
            _OCR_MODEL_NAME = ""
            _OCR_MODEL_DEVICE = ""
            gc.collect()
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK", "0")
        if not candidates:
            raise RuntimeError(
                "PaddleOCR-VL 로컬 모델 후보를 찾지 못했습니다. "
                "`PDF_OCR_MODEL_NAME` 또는 `PDF_OCR_VL_MODEL_DIR`가 폐쇄망 서버의 로컬 모델 폴더를 가리키는지 확인해 주세요. "
                "온라인 다운로드 fallback이 필요할 때만 `PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK=1`로 명시합니다."
            )
        try:
            from paddleocr import PaddleOCRVL
        except Exception as e:
            raise RuntimeError(
                "PaddleOCR-VL 로더를 불러오지 못했습니다. "
                "`paddleocr` 및 PaddlePaddle 설치를 확인해 주세요."
            ) from e
        errors: List[str] = []
        missing_modules: List[str] = []
        for candidate in candidates:
            try:
                kwargs = _build_paddleocrvl_kwargs(
                    paddleocrvl_cls=PaddleOCRVL,
                    candidate=candidate,
                    requested_model_name=model_name,
                    device=resolved_device,
                )
                _prepare_local_ocr_import_paths(candidate, kwargs)
                _OCR_MODEL = PaddleOCRVL(**kwargs)
                _OCR_MODEL_NAME = candidate
                _OCR_MODEL_DEVICE = resolved_device
                return _OCR_MODEL
            except ModuleNotFoundError as e:
                missing_name = getattr(e, "name", "") or str(e)
                if missing_name:
                    missing_modules.append(str(missing_name))
                errors.append(f"{candidate}: ModuleNotFoundError({missing_name}): {e}")
            except Exception as e:
                errors.append(f"{candidate}: {type(e).__name__}: {e}")

        raise RuntimeError(
            "PaddleOCR-VL 모델 로딩 실패. "
            f"{_build_ocr_runtime_hint(missing_modules)}"
            f"{_build_ocr_error_hint(errors)}"
            "로컬 모델 파일이 있어도 backend venv에는 "
            "`paddleocr[doc-parser]==3.4.0`과 PaddlePaddle 런타임이 함께 필요할 수 있습니다. "
            f"시도한 후보={candidates}. "
            f"오류={'; '.join(errors[:5])}"
        )


def _predict_pdf(model: Any, pdf_path: str, max_pages: int):
    kwargs: Dict[str, Any] = {"input": pdf_path}
    if max_pages > 0:
        kwargs["page_num"] = int(max_pages)
    try:
        return model.predict(**kwargs)
    except TypeError:
        # Compatibility fallback for older/newer signatures.
        try:
            return model.predict(input=pdf_path)
        except TypeError:
            return model.predict(pdf_path)


def _materialize_ocr_output(raw_output: Any) -> Tuple[List[Any], float]:
    started_at = time.monotonic()
    if raw_output is None:
        items: List[Any] = []
    elif isinstance(raw_output, list):
        items = raw_output
    elif isinstance(raw_output, tuple):
        items = list(raw_output)
    elif hasattr(raw_output, "__iter__"):
        items = list(raw_output)
    else:
        items = [raw_output]
    return items, round(time.monotonic() - started_at, 3)


def _hps_page_text(item: Dict[str, Any]) -> str:
    markdown = item.get("markdown")
    if isinstance(markdown, dict):
        for key in ("text", "markdown", "content"):
            value = markdown.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    for key in ("markdownText", "text", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    pruned = item.get("prunedResult")
    if isinstance(pruned, dict):
        fragments: List[str] = []
        _collect_text_fragments(pruned, fragments)
        return "\n".join(_dedupe_texts(fragments)).strip()
    return ""


def _normalize_hps_layout_response(payload: Dict[str, Any], *, min_text_chars: int) -> List[Dict[str, Any]]:
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        result = payload if isinstance(payload, dict) else {}
    raw_pages = result.get("layoutParsingResults", [])
    if not isinstance(raw_pages, list):
        raise RuntimeError("HPS response is missing result.layoutParsingResults")
    pages: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_pages, start=1):
        if not isinstance(item, dict):
            continue
        raw_page_no = item.get("pageNo", item.get("page_no", item.get("pageIndex", index)))
        page_no = _to_positive_int(raw_page_no) or index
        text = _hps_page_text(item)
        if len(text) >= max(1, int(min_text_chars or 1)):
            pages.append({"page_no": page_no, "text": text})
    return sorted(pages, key=lambda page: int(page.get("page_no", 0) or 0))


def _hps_request_json(pdf_path: str, *, timeout_seconds: float) -> Dict[str, Any]:
    base_url = (os.getenv("PDF_OCR_HPS_URL", "http://127.0.0.1:8080") or "").strip().rstrip("/")
    endpoint = (os.getenv("PDF_OCR_HPS_ENDPOINT", "/layout-parsing") or "/layout-parsing").strip()
    url = f"{base_url}/{endpoint.lstrip('/')}"
    with open(pdf_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    body = json.dumps({"file": encoded, "fileType": 0}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = (os.getenv("PDF_OCR_HPS_API_KEY", "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib_request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=max(1.0, timeout_seconds)) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"PaddleOCR HPS request failed: {exc}") from exc


def _extract_pdf_pages_with_paddleocr_vl_serial(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    model_name: Optional[str] = None,
    max_pages: Optional[int] = None,
    min_text_chars: Optional[int] = None,
    device_override: Optional[str] = None,
    release_after_run_override: Optional[bool] = None,
    progress_meta: Optional[Dict[str, Any]] = None,
    original_pages_label: str = "",
) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    resolved_model_name = (
        model_name
        or os.getenv("PDF_OCR_MODEL_NAME", _OCR_DEFAULT_MODEL_NAME)
        or _OCR_DEFAULT_MODEL_NAME
    ).strip()
    resolved_max_pages = max(1, int(max_pages if max_pages is not None else os.getenv("PDF_OCR_MAX_PAGES", "400")))
    resolved_min_text_chars = max(
        1,
        int(min_text_chars if min_text_chars is not None else os.getenv("PDF_OCR_MIN_TEXT_CHARS", "4")),
    )
    _set_last_ocr_serial_timing_info({})
    release_after_run = _is_gpu_device(device_override or os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu")
    load_message = "PDF OCR 모델을 준비하는 중입니다."
    run_message = "PDF OCR을 실행하는 중입니다. 페이지 수에 따라 오래 걸릴 수 있습니다."

    if progress_callback is not None:
        _emit_progress(
            progress_callback,
            34,
            load_message,
            "load_pdf_ocr_model",
            **dict(progress_meta or {}),
        )
    resolved_device = (device_override or os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    release_after_run = _is_gpu_device(resolved_device) and _env_enabled(
        "PDF_OCR_RELEASE_GPU_MODEL_AFTER_RUN",
        True,
    )
    if release_after_run_override is not None:
        release_after_run = bool(release_after_run_override)
    timing_info: Dict[str, Any] = {}
    with _ocr_progress_heartbeat(
        progress_callback,
        percent=34,
        message=load_message,
        stage="load_pdf_ocr_model",
        interval_seconds=_resolve_ocr_progress_heartbeat_interval_seconds(),
        progress_meta=progress_meta,
    ):
        try:
            load_started_at = time.monotonic()
            print(
                "[PDF_OCR][SERIAL_LOAD] "
                f"device={resolved_device} model={resolved_model_name} pdf={os.path.basename(pdf_path)}",
                flush=True,
            )
            model = _load_ocr_model(model_name=resolved_model_name, device=resolved_device)
            timing_info["ocr_model_load_seconds"] = round(time.monotonic() - load_started_at, 3)
        except TypeError:
            load_started_at = time.monotonic()
            print(
                "[PDF_OCR][SERIAL_LOAD] "
                f"device={resolved_device} model={resolved_model_name} pdf={os.path.basename(pdf_path)}",
                flush=True,
            )
            model = _load_ocr_model(model_name=resolved_model_name)
            timing_info["ocr_model_load_seconds"] = round(time.monotonic() - load_started_at, 3)
    raw_output = None
    try:
        if progress_callback is not None:
            _emit_progress(
                progress_callback,
                42,
                run_message,
                "run_pdf_ocr",
                **dict(progress_meta or {}),
            )
        with _ocr_progress_heartbeat(
            progress_callback,
            percent=42,
            message=run_message,
            stage="run_pdf_ocr",
            interval_seconds=_resolve_ocr_progress_heartbeat_interval_seconds(),
            progress_meta=progress_meta,
        ):
            predict_started_at = time.monotonic()
            print(
                "[PDF_OCR][SERIAL_PREDICT_START] "
                f"device={resolved_device} pdf={os.path.basename(pdf_path)} "
                f"paddle_page_num={resolved_max_pages} "
                f"original_pages={str(original_pages_label or '').strip() or '-'}",
                flush=True,
            )
            raw_output = _predict_pdf(model=model, pdf_path=pdf_path, max_pages=resolved_max_pages)
            timing_info["ocr_predict_seconds"] = round(time.monotonic() - predict_started_at, 3)
            raw_output, materialize_seconds = _materialize_ocr_output(raw_output)
            timing_info["ocr_output_materialize_seconds"] = materialize_seconds
            print(
                "[PDF_OCR][SERIAL_PREDICT_DONE] "
                f"device={resolved_device} pdf={os.path.basename(pdf_path)} "
                f"predict_seconds={timing_info['ocr_predict_seconds']:.3f} "
                f"materialize_seconds={materialize_seconds:.3f}",
                flush=True,
            )
    except Exception as e:
        _set_last_ocr_serial_timing_info(timing_info)
        raise RuntimeError(f"PaddleOCR-VL PDF 인식 실패: {e}") from e

    if progress_callback is not None:
        _emit_progress(
            progress_callback,
            55,
            "OCR 결과를 페이지별로 정리하는 중입니다.",
            "merge_pdf_ocr",
            **dict(progress_meta or {}),
        )
    merged_by_page: Dict[int, List[str]] = {}
    fallback_page = 1
    merge_started_at = time.monotonic()

    page_iter = raw_output
    total_items = max(1, len(raw_output))
    payload_convert_seconds = 0.0

    try:
        try:
            for item_index, item in enumerate(page_iter, start=1):
                if progress_callback is not None:
                    if total_items > 0:
                        if item_index == 1 or item_index == total_items or item_index % max(1, total_items // 3) == 0:
                            merge_percent = 55 + int(round((item_index / total_items) * 3))
                            _emit_progress(
                                progress_callback,
                                merge_percent,
                                f"OCR 결과를 페이지별로 정리하는 중입니다. ({item_index}/{total_items})",
                                "merge_pdf_ocr",
                                **dict(progress_meta or {}),
                            )
                    elif item_index == 1 or item_index % 4 == 0:
                        _emit_progress(
                            progress_callback,
                            56,
                            f"OCR 결과를 페이지별로 정리하는 중입니다. ({item_index}개 처리)",
                            "merge_pdf_ocr",
                            **dict(progress_meta or {}),
                        )

                convert_started_at = time.monotonic()
                payload = _to_plain_data(item)
                payload_convert_seconds += time.monotonic() - convert_started_at
                page_no = _extract_page_no(payload, fallback=fallback_page, max_page=resolved_max_pages)
                fallback_page = max(fallback_page + 1, page_no + 1)

                fragments: List[str] = []
                _collect_text_fragments(payload, fragments)
                deduped = _dedupe_texts(fragments)
                if not deduped:
                    continue
                text = "\n".join(deduped).strip()
                if len(text) < resolved_min_text_chars:
                    continue
                merged_by_page.setdefault(page_no, []).append(text)
        finally:
            del raw_output
            gc.collect()

        pages: List[Dict[str, Any]] = []
        for page_no in sorted(merged_by_page.keys()):
            page_text = "\n".join(_dedupe_texts(merged_by_page[page_no])).strip()
            if len(page_text) < resolved_min_text_chars:
                continue
            pages.append({"page_no": int(page_no), "text": page_text})
        timing_info["ocr_merge_seconds"] = round(time.monotonic() - merge_started_at, 3)
        timing_info["ocr_payload_convert_seconds"] = round(payload_convert_seconds, 3)
        timing_info["ocr_text_merge_seconds"] = round(
            max(0.0, float(timing_info["ocr_merge_seconds"]) - payload_convert_seconds),
            3,
        )
        _set_last_ocr_serial_timing_info(timing_info)

        if pages:
            return pages

        raise RuntimeError("PaddleOCR-VL 결과에서 추출 가능한 텍스트를 찾지 못했습니다.")
    finally:
        if release_after_run:
            release_cached_ocr_model(device=resolved_device)


def _parallel_ocr_subset_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    device = str(task.get("device", "cpu") or "cpu").strip()
    if device:
        os.environ["PDF_OCR_DEVICE"] = device
    subset_page_count = int(task.get("subset_page_count", task.get("max_pages", 1)) or 1)
    print(
        "[PDF_OCR][WORKER_START] "
        f"device={device or 'cpu'} pdf={os.path.basename(str(task.get('pdf_path', '') or ''))} "
        f"subset_page_count={subset_page_count} "
        f"paddle_page_num={subset_page_count} "
        f"original_pages={str(task.get('original_pages_label', '') or '').strip() or '-'}",
        flush=True,
    )
    _set_last_ocr_serial_timing_info({})
    pages = _call_serial_ocr(
        str(task.get("pdf_path", "") or ""),
        progress_callback=None,
        model_name=str(task.get("model_name", "") or _OCR_DEFAULT_MODEL_NAME),
        max_pages=subset_page_count,
        min_text_chars=int(task.get("min_text_chars", 4) or 4),
        device=device,
        release_after_run_override=False,
        original_pages_label=str(task.get("original_pages_label", "") or ""),
    )
    return {
        "pages": pages,
        "runtime_info": _peek_last_ocr_serial_timing_info(),
    }

def _extract_pdf_pages_with_paddleocr_vl_once(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    model_name: str,
    max_pages: int,
    min_text_chars: int,
    device: str,
    runtime_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    total_pages = _count_pdf_pages(pdf_path)
    candidate_page_count = min(max_pages, total_pages) if total_pages > 0 else 0
    if candidate_page_count <= 0:
        return _call_serial_ocr(
            pdf_path,
            progress_callback=progress_callback,
            model_name=model_name,
            max_pages=max_pages,
            min_text_chars=min_text_chars,
            device=device,
        )
    if candidate_page_count == 1:
        return _call_serial_ocr(
            pdf_path,
            progress_callback=progress_callback,
            model_name=model_name,
            max_pages=1,
            min_text_chars=min_text_chars,
            device=device,
        )
    return _extract_pdf_pages_with_paddleocr_vl_selected_pages(
        pdf_path,
        page_numbers=list(range(1, candidate_page_count + 1)),
        progress_callback=progress_callback,
        model_name=model_name,
        min_text_chars=min_text_chars,
        device=device,
        total_document_pages=total_pages,
        completed_pages_base=0,
        runtime_info=runtime_info,
    )


def _finalize_ocr_runtime_info(
    runtime_info: Dict[str, Any],
    *,
    started_at: float,
    pages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    elapsed_seconds = max(0.0, time.monotonic() - float(started_at))
    pages_processed = len(list(pages or []))
    pages_per_minute = (pages_processed / elapsed_seconds * 60.0) if elapsed_seconds > 0 else 0.0
    target_pages = max(0, int(os.getenv("PDF_OCR_TARGET_PAGES", os.getenv("PDF_OCR_MAX_PAGES", "200")) or "0"))
    target_seconds = max(0.0, _to_float(os.getenv("PDF_OCR_TARGET_SECONDS", "300"), 300.0))
    runtime_info["ocr_elapsed_seconds"] = round(elapsed_seconds, 3)
    runtime_info["ocr_pages_processed"] = pages_processed
    runtime_info["ocr_pages_per_minute"] = round(pages_per_minute, 3)
    runtime_info["ocr_target_pages"] = target_pages
    runtime_info["ocr_target_seconds"] = target_seconds
    runtime_info["ocr_target_met"] = bool(
        target_pages > 0
        and target_seconds > 0
        and pages_processed >= target_pages
        and elapsed_seconds <= target_seconds
    )
    print(
        "[PDF_OCR][RUNTIME] "
        f"elapsed_seconds={runtime_info['ocr_elapsed_seconds']} "
        f"pages_processed={runtime_info['ocr_pages_processed']} "
        f"pages_per_minute={runtime_info['ocr_pages_per_minute']} "
        f"batch_count={int(runtime_info.get('ocr_batch_count', 0) or 0)} "
        f"subset_build_seconds={float(runtime_info.get('ocr_subset_build_seconds', 0.0) or 0.0):.3f} "
        f"model_load_seconds={float(runtime_info.get('ocr_model_load_seconds', 0.0) or 0.0):.3f} "
        f"predict_seconds={float(runtime_info.get('ocr_predict_seconds', 0.0) or 0.0):.3f} "
        f"materialize_seconds={float(runtime_info.get('ocr_output_materialize_seconds', 0.0) or 0.0):.3f} "
        f"payload_convert_seconds={float(runtime_info.get('ocr_payload_convert_seconds', 0.0) or 0.0):.3f} "
        f"text_merge_seconds={float(runtime_info.get('ocr_text_merge_seconds', 0.0) or 0.0):.3f} "
        f"merge_seconds={float(runtime_info.get('ocr_merge_seconds', 0.0) or 0.0):.3f} "
        f"backend={str(runtime_info.get('ocr_backend_effective', runtime_info.get('ocr_backend', 'local')) or 'local')} "
        f"batch_p50_seconds={float(runtime_info.get('ocr_batch_wall_seconds_p50', 0.0) or 0.0):.3f} "
        f"batch_p95_seconds={float(runtime_info.get('ocr_batch_wall_seconds_p95', 0.0) or 0.0):.3f} "
        f"target_met={bool(runtime_info.get('ocr_target_met', False))}"
    )


def _execute_local_paddleocr_vl_with_runtime(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    page_numbers: Optional[List[int]] = None,
    total_document_pages: Optional[int] = None,
    completed_pages_base: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model_name = (
        os.getenv("PDF_OCR_MODEL_NAME", _OCR_DEFAULT_MODEL_NAME) or _OCR_DEFAULT_MODEL_NAME
    ).strip()
    max_pages = max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "400")))
    min_text_chars = max(1, int(os.getenv("PDF_OCR_MIN_TEXT_CHARS", "4")))
    requested_device = (os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    runtime_info: Dict[str, Any] = {
        "ocr_device_attempted": requested_device,
        "ocr_device_effective": requested_device,
        "ocr_gpu_fallback_used": False,
        "ocr_gpu_failure_reason": "",
    }
    original_device_env = os.environ.get("PDF_OCR_DEVICE")
    started_at = time.monotonic()

    try:
        os.environ["PDF_OCR_DEVICE"] = requested_device
        if page_numbers is not None:
            pages = _extract_pdf_pages_with_paddleocr_vl_selected_pages(
                pdf_path,
                page_numbers=list(page_numbers),
                progress_callback=progress_callback,
                model_name=model_name,
                min_text_chars=min_text_chars,
                device=requested_device,
                total_document_pages=total_document_pages,
                completed_pages_base=completed_pages_base,
                runtime_info=runtime_info,
            )
        else:
            pages = _extract_pdf_pages_with_paddleocr_vl_once(
                pdf_path,
                progress_callback=progress_callback,
                model_name=model_name,
                max_pages=max_pages,
                min_text_chars=min_text_chars,
                device=requested_device,
                runtime_info=runtime_info,
            )
        _finalize_ocr_runtime_info(runtime_info, started_at=started_at, pages=pages)
        return pages, runtime_info
    except Exception as exc:
        if not str(runtime_info.get("ocr_gpu_failure_reason", "") or "").strip():
            runtime_info["ocr_gpu_failure_reason"] = _ocr_failure_reason(exc)
        if (
            not bool(runtime_info.get("ocr_gpu_fallback_used", False))
            and _ocr_failure_needs_cpu_fallback(exc, device=requested_device)
        ):
            _emit_progress(
                progress_callback,
                40,
                "GPU OCR이 실패해 CPU fallback으로 다시 시도하는 중입니다.",
                "fallback_pdf_ocr",
                current_page=max(0, int(completed_pages_base or 0)),
                total_pages=max(0, int(total_document_pages or 0)),
                pdf_total_pages=max(0, int(total_document_pages or 0)),
                ocr_target_pages=max(0, len(list(page_numbers or []))),
                ocr_completed_pages=0,
            )
            _reset_cached_ocr_model()
            try:
                os.environ["PDF_OCR_DEVICE"] = "cpu"
                if page_numbers is not None:
                    fallback_pages = _extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        pdf_path,
                        page_numbers=list(page_numbers),
                        progress_callback=progress_callback,
                        model_name=model_name,
                        min_text_chars=min_text_chars,
                        device="cpu",
                        total_document_pages=total_document_pages,
                        completed_pages_base=completed_pages_base,
                        runtime_info=runtime_info,
                    )
                else:
                    fallback_pages = _extract_pdf_pages_with_paddleocr_vl_once(
                        pdf_path,
                        progress_callback=progress_callback,
                        model_name=model_name,
                        max_pages=max_pages,
                        min_text_chars=min_text_chars,
                        device="cpu",
                        runtime_info=runtime_info,
                    )
            except Exception as fallback_exc:
                runtime_info["ocr_device_effective"] = "cpu"
                runtime_info["ocr_gpu_fallback_used"] = True
                raise PdfOCRExecutionError(
                    f"{exc} GPU OCR 실패 후 CPU fallback도 실패했습니다: {fallback_exc}",
                    runtime_info=runtime_info,
                ) from fallback_exc
            runtime_info["ocr_device_effective"] = "cpu"
            runtime_info["ocr_gpu_fallback_used"] = True
            _finalize_ocr_runtime_info(runtime_info, started_at=started_at, pages=fallback_pages)
            return fallback_pages, runtime_info
        _finalize_ocr_runtime_info(runtime_info, started_at=started_at, pages=[])
        raise PdfOCRExecutionError(str(exc), runtime_info=runtime_info) from exc
    finally:
        if original_device_env is None:
            os.environ.pop("PDF_OCR_DEVICE", None)
        else:
            os.environ["PDF_OCR_DEVICE"] = original_device_env


def _execute_hps_ocr_with_runtime(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    page_numbers: Optional[List[int]] = None,
    total_document_pages: Optional[int] = None,
    completed_pages_base: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    max_pages = max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "400") or "400"))
    min_text_chars = max(1, int(os.getenv("PDF_OCR_MIN_TEXT_CHARS", "4") or "4"))
    chunk_pages = max(1, int(os.getenv("PDF_OCR_HPS_CHUNK_PAGES", "16") or "16"))
    max_concurrency = max(1, int(os.getenv("PDF_OCR_HPS_MAX_CONCURRENCY", "4") or "4"))
    request_timeout = max(
        1.0,
        _to_float(os.getenv("PDF_OCR_HPS_REQUEST_TIMEOUT_SECONDS", "600"), 600.0),
    )
    total_pages = int(total_document_pages or 0) or _count_pdf_pages(pdf_path)
    selected_pages = sorted(
        {
            int(page_no)
            for page_no in (
                page_numbers
                if page_numbers is not None
                else range(1, min(max_pages, total_pages or max_pages) + 1)
            )
            if int(page_no) > 0 and int(page_no) <= max_pages
        }
    )
    if not selected_pages:
        return [], {"ocr_backend": "hps", "ocr_pages_processed": 0}

    started_at = time.monotonic()
    subset_build_seconds = 0.0
    tasks: List[Tuple[str, Dict[int, int]]] = []
    try:
        for offset in range(0, len(selected_pages), chunk_pages):
            chunk = selected_pages[offset : offset + chunk_pages]
            subset_started_at = time.monotonic()
            subset_path, page_map = _build_pdf_subset_for_pages(pdf_path, chunk)
            subset_build_seconds += time.monotonic() - subset_started_at
            tasks.append((subset_path, page_map))

        _emit_progress(
            progress_callback,
            42,
            "PaddleOCR HPS 서비스에서 PDF OCR을 실행하는 중입니다.",
            "run_pdf_ocr_hps",
            current_page=max(0, int(completed_pages_base or 0)),
            total_pages=max(0, int(total_pages or 0)),
            ocr_target_pages=len(selected_pages),
            ocr_completed_pages=0,
        )
        request_started_at = time.monotonic()
        responses: List[Tuple[Dict[int, int], Dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrency, len(tasks))) as executor:
            futures = {
                executor.submit(_hps_request_json, subset_path, timeout_seconds=request_timeout): {
                    "page_map": page_map,
                    "submitted_at": time.monotonic(),
                }
                for subset_path, page_map in tasks
            }
            batch_wall_seconds: List[float] = []
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                future_meta = futures[future]
                responses.append((dict(future_meta["page_map"]), future.result()))
                batch_wall_seconds.append(
                    round(time.monotonic() - float(future_meta["submitted_at"]), 3)
                )
                _emit_progress(
                    progress_callback,
                    42 + int(round((completed_count / len(tasks)) * 13)),
                    f"PaddleOCR HPS 결과를 수신하는 중입니다. ({completed_count}/{len(tasks)})",
                    "run_pdf_ocr_hps",
                    current_page=max(0, int(completed_pages_base or 0)),
                    total_pages=max(0, int(total_pages or 0)),
                    ocr_target_pages=len(selected_pages),
                    ocr_completed_pages=min(len(selected_pages), completed_count * chunk_pages),
                )
        request_seconds = time.monotonic() - request_started_at

        normalize_started_at = time.monotonic()
        pages: List[Dict[str, Any]] = []
        for page_map, response in responses:
            for page in _normalize_hps_layout_response(response, min_text_chars=min_text_chars):
                subset_page_no = int(page.get("page_no", 0) or 0)
                original_page_no = int(page_map.get(subset_page_no, subset_page_no) or subset_page_no)
                pages.append({"page_no": original_page_no, "text": str(page.get("text", "") or "")})
        pages.sort(key=lambda page: int(page.get("page_no", 0) or 0))
        runtime_info: Dict[str, Any] = {
            "ocr_backend": "hps",
            "ocr_backend_attempted": "hps",
            "ocr_backend_effective": "hps",
            "ocr_backend_fallback_used": False,
            "ocr_hps_url": (os.getenv("PDF_OCR_HPS_URL", "http://127.0.0.1:8080") or "").strip(),
            "ocr_hps_chunk_pages": chunk_pages,
            "ocr_hps_max_concurrency": max_concurrency,
            "ocr_batch_count": len(tasks),
            "ocr_batch_wall_seconds": batch_wall_seconds,
            "ocr_subset_build_seconds": round(subset_build_seconds, 3),
            "ocr_predict_seconds": round(request_seconds, 3),
            "ocr_output_materialize_seconds": 0.0,
            "ocr_payload_convert_seconds": round(time.monotonic() - normalize_started_at, 3),
            "ocr_text_merge_seconds": 0.0,
            "ocr_merge_seconds": round(time.monotonic() - normalize_started_at, 3),
            "ocr_device_attempted": "hps",
            "ocr_device_effective": "hps",
            "ocr_gpu_fallback_used": False,
            "ocr_gpu_failure_reason": "",
        }
        runtime_info.update(_summarize_batch_wall_seconds(batch_wall_seconds))
        _finalize_ocr_runtime_info(runtime_info, started_at=started_at, pages=pages)
        if not pages:
            raise RuntimeError("PaddleOCR HPS 결과에서 추출 가능한 텍스트를 찾지 못했습니다.")
        return pages, runtime_info
    finally:
        for subset_path, _page_map in tasks:
            try:
                os.remove(subset_path)
            except OSError:
                pass


def _execute_paddleocr_vl_with_runtime(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    page_numbers: Optional[List[int]] = None,
    total_document_pages: Optional[int] = None,
    completed_pages_base: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    backend = (os.getenv("PDF_OCR_BACKEND", "local") or "local").strip().lower()
    kwargs = {
        "progress_callback": progress_callback,
        "page_numbers": page_numbers,
        "total_document_pages": total_document_pages,
        "completed_pages_base": completed_pages_base,
    }
    if backend != "hps":
        pages, runtime_info = _execute_local_paddleocr_vl_with_runtime(pdf_path, **kwargs)
        runtime_info.setdefault("ocr_backend", "local")
        runtime_info.setdefault("ocr_backend_attempted", "local")
        runtime_info.setdefault("ocr_backend_effective", "local")
        runtime_info.setdefault("ocr_backend_fallback_used", False)
        return pages, runtime_info
    try:
        return _execute_hps_ocr_with_runtime(pdf_path, **kwargs)
    except Exception as exc:
        if not _env_enabled("PDF_OCR_HPS_FALLBACK_TO_LOCAL", True):
            raise
        print(f"[PDF_OCR][HPS_FALLBACK] error={type(exc).__name__}: {exc}", flush=True)
        pages, runtime_info = _execute_local_paddleocr_vl_with_runtime(pdf_path, **kwargs)
        runtime_info["ocr_backend"] = "local"
        runtime_info["ocr_backend_attempted"] = "hps"
        runtime_info["ocr_backend_effective"] = "local"
        runtime_info["ocr_backend_fallback_used"] = True
        runtime_info["ocr_hps_error"] = f"{type(exc).__name__}: {exc}"
        return pages, runtime_info


def extract_pdf_pages_with_paddleocr_vl(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    page_numbers: Optional[List[int]] = None,
    total_document_pages: Optional[int] = None,
    completed_pages_base: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run PaddleOCR-VL on a PDF file and return ordered page texts.

    Returns:
      [
        {"page_no": 1, "text": "..."},
        {"page_no": 2, "text": "..."},
      ]
    """
    pages, _runtime_info = _execute_paddleocr_vl_with_runtime(
        pdf_path,
        progress_callback=progress_callback,
        page_numbers=page_numbers,
        total_document_pages=total_document_pages,
        completed_pages_base=completed_pages_base,
    )
    _set_last_ocr_runtime_info(_runtime_info)
    return pages


def extract_pdf_pages(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    *,
    force_upload_ocr: bool = False,
) -> Dict[str, Any]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    parse_mode = _normalize_pdf_parse_mode(os.getenv("PDF_PARSE_MODE", "ocr_first"))
    text_extractor = _normalize_pdf_text_extractor(os.getenv("PDF_TEXT_EXTRACTOR", "pymupdf"))
    min_text_chars = max(1, int(os.getenv("PDF_TEXT_MIN_CHARS", os.getenv("PDF_OCR_MIN_TEXT_CHARS", "4"))))
    min_nonspace_ratio = max(0.0, min(1.0, _to_float(os.getenv("PDF_TEXT_MIN_NONSPACE_RATIO", "0.20"), 0.20)))

    if parse_mode == "text_only" and text_extractor == "disabled":
        raise RuntimeError("PDF_PARSE_MODE=text_only 인데 PDF_TEXT_EXTRACTOR=disabled 로 설정돼 있습니다.")

    if parse_mode == "ocr_first":
        ocr_total_pages, selected_page_numbers, has_pages_over_limit = _selected_ocr_first_page_numbers(pdf_path)
        attempted_ocr_pages = len(selected_page_numbers or [])
        warnings: List[str] = []
        _set_last_ocr_runtime_info({})
        try:
            ocr_pages_raw = extract_pdf_pages_with_paddleocr_vl(
                pdf_path,
                progress_callback=progress_callback,
                page_numbers=selected_page_numbers,
                total_document_pages=ocr_total_pages or None,
                completed_pages_base=0,
            )
            ocr_runtime = _peek_last_ocr_runtime_info()
        except Exception as exc:
            if (
                _ocr_failure_reason(exc) == "gpu_kernel_incompatible"
                and _env_enabled("PDF_OCR_STRICT_GPU_COMPAT", True)
            ):
                raise
            warnings.append("ocr_first_failed_fallback_to_pymupdf")
            try:
                text_total_pages, text_pages = _extract_ocr_first_pymupdf_pages(
                    pdf_path,
                    min_text_chars=min_text_chars,
                    min_nonspace_ratio=min_nonspace_ratio,
                    progress_callback=progress_callback,
                )
            except Exception:
                raise exc
            result = _build_pdf_result(
                total_pages=text_total_pages or ocr_total_pages,
                pages=text_pages,
                warnings=warnings,
                attempted_ocr_pages=attempted_ocr_pages,
                ocr_runtime=None,
            )
            if result["pages"]:
                return result
            raise exc

        wrapped_ocr_pages = [
            {
                "page_no": max(1, int(page.get("page_no", 0) or 0)),
                "text": _normalize_block(str(page.get("text", "") or "")),
                "parser": "paddleocr_vl",
            }
            for page in list(ocr_pages_raw or [])
        ]
        result_pages: List[Dict[str, Any]] = list(wrapped_ocr_pages)

        if not result_pages:
            warnings.append("ocr_first_empty_fallback_to_pymupdf")
            text_total_pages, text_pages = _extract_ocr_first_pymupdf_pages(
                pdf_path,
                min_text_chars=min_text_chars,
                min_nonspace_ratio=min_nonspace_ratio,
                progress_callback=progress_callback,
            )
            result = _build_pdf_result(
                total_pages=text_total_pages or ocr_total_pages,
                pages=text_pages,
                warnings=warnings,
                attempted_ocr_pages=attempted_ocr_pages,
                ocr_runtime=ocr_runtime,
            )
            if result["pages"]:
                return result
            raise RuntimeError("PaddleOCR-first 결과와 PyMuPDF fallback 모두에서 추출 가능한 텍스트를 찾지 못했습니다.")

        if has_pages_over_limit and text_extractor != "disabled":
            over_limit_pages = set(range(attempted_ocr_pages + 1, max(ocr_total_pages, attempted_ocr_pages) + 1))
            try:
                text_total_pages, text_pages = _extract_ocr_first_pymupdf_pages(
                    pdf_path,
                    min_text_chars=min_text_chars,
                    min_nonspace_ratio=min_nonspace_ratio,
                    progress_callback=progress_callback,
                    allowed_page_numbers=over_limit_pages,
                )
                result_pages.extend(text_pages)
                warnings.append("ocr_first_pymupdf_for_pages_over_ocr_limit")
                ocr_total_pages = max(ocr_total_pages, text_total_pages)
            except Exception:
                warnings.append("ocr_first_pymupdf_over_limit_failed")

        return _build_pdf_result(
            total_pages=max(len(result_pages), int(ocr_total_pages or 0)),
            pages=result_pages,
            warnings=warnings,
            attempted_ocr_pages=attempted_ocr_pages,
            ocr_runtime=ocr_runtime,
        )

    if parse_mode == "ocr_only":
        ocr_total_pages = _count_pdf_pages(pdf_path)
        selected_page_numbers = list(
            range(
                1,
                max(1, min(max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "400"))), max(1, ocr_total_pages))) + 1,
            )
        ) if ocr_total_pages > 0 else None
        _set_last_ocr_runtime_info({})
        ocr_pages = extract_pdf_pages_with_paddleocr_vl(
            pdf_path,
            progress_callback=progress_callback,
            page_numbers=selected_page_numbers,
            total_document_pages=ocr_total_pages or None,
            completed_pages_base=0,
        )
        ocr_runtime = _peek_last_ocr_runtime_info()
        wrapped_pages = [
            {
                "page_no": max(1, int(page.get("page_no", 0) or 0)),
                "text": _normalize_block(str(page.get("text", "") or "")),
                "parser": "paddleocr_vl",
            }
            for page in ocr_pages
        ]
        return _build_pdf_result(
            total_pages=max(len(wrapped_pages), int(ocr_total_pages or 0)),
            pages=wrapped_pages,
            ocr_runtime=ocr_runtime,
        )

    if text_extractor == "disabled":
        ocr_total_pages = _count_pdf_pages(pdf_path)
        selected_page_numbers = list(
            range(
                1,
                max(1, min(max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "400"))), max(1, ocr_total_pages))) + 1,
            )
        ) if ocr_total_pages > 0 else None
        _set_last_ocr_runtime_info({})
        ocr_pages = extract_pdf_pages_with_paddleocr_vl(
            pdf_path,
            progress_callback=progress_callback,
            page_numbers=selected_page_numbers,
            total_document_pages=ocr_total_pages or None,
            completed_pages_base=0,
        )
        ocr_runtime = _peek_last_ocr_runtime_info()
        wrapped_pages = [
            {
                "page_no": max(1, int(page.get("page_no", 0) or 0)),
                "text": _normalize_block(str(page.get("text", "") or "")),
                "parser": "paddleocr_vl",
            }
            for page in ocr_pages
        ]
        return _build_pdf_result(
            total_pages=max(len(wrapped_pages), int(ocr_total_pages or 0)),
            pages=wrapped_pages,
            ocr_runtime=ocr_runtime,
        )

    text_pages_raw: List[Dict[str, Any]] = []
    total_pages = 0
    warnings: List[str] = []
    pymupdf_failed = False
    try:
        text_result = _extract_pdf_pages_with_pymupdf(
            pdf_path,
            progress_callback=progress_callback,
        )
        text_pages_raw = list(text_result.get("pages", []) or [])
        total_pages = max(0, int(text_result.get("total_pages", 0) or 0))
    except Exception:
        pymupdf_failed = True
        warnings.append("pymupdf_text_unavailable")
        if parse_mode == "text_only":
            raise

    text_pages: Dict[int, Dict[str, Any]] = {}
    for page in text_pages_raw:
        page_no = max(1, int(page.get("page_no", 0) or 0))
        page_text = _normalize_block(str(page.get("text", "") or ""))
        if not _page_has_meaningful_text(page_text, min_text_chars, min_nonspace_ratio):
            continue
        text_pages[page_no] = {
            "page_no": page_no,
            "text": page_text,
            "parser": "pymupdf_text",
        }

    if parse_mode == "text_only":
        result = _build_pdf_result(total_pages=total_pages, pages=list(text_pages.values()))
        if result["pages"]:
            return result
        raise RuntimeError("PyMuPDF 결과에서 추출 가능한 텍스트를 찾지 못했습니다.")

    missing_pages: List[int] = []
    if total_pages > 0:
        for page_no in range(1, total_pages + 1):
            if page_no not in text_pages:
                missing_pages.append(page_no)
    elif not text_pages and pymupdf_failed:
        missing_pages.append(1)

    table_ocr_pages = sorted(
        page_no
        for page_no, page in text_pages.items()
        if _should_ocr_table_like_text_page(str(page.get("text", "") or ""))
    )
    ocr_candidate_pages = sorted({*missing_pages, *table_ocr_pages})

    if not ocr_candidate_pages and text_pages:
        return _build_pdf_result(total_pages=total_pages, pages=list(text_pages.values()))

    if ocr_candidate_pages and not (force_upload_ocr or _upload_ocr_enabled()):
        warnings.append("lazy_ocr_deferred")
        deferred_pages: Dict[int, Dict[str, Any]] = dict(text_pages)
        for page_no in ocr_candidate_pages:
            existing_page = deferred_pages.get(page_no)
            if existing_page is not None:
                lazy_hints = _build_lazy_ocr_hints(
                    page_no=page_no,
                    table_like=True,
                    table_hints=_build_table_hint_lines(str(existing_page.get("text", "") or "")),
                )
                existing_page["lazy_ocr_hints"] = _dedupe_hint_lines(
                    list(existing_page.get("lazy_ocr_hints", []) or []) + lazy_hints
                )
                continue
            deferred_pages[page_no] = {
                "page_no": page_no,
                "text": "",
                "parser": "pymupdf_hint",
                "lazy_ocr_hints": _build_lazy_ocr_hints(
                    page_no=page_no,
                    table_like=False,
                ),
            }
        result = _build_pdf_result(
            total_pages=total_pages or max(ocr_candidate_pages),
            pages=list(deferred_pages.values()),
            warnings=warnings,
            attempted_ocr_pages=len(ocr_candidate_pages),
            ocr_runtime=None,
        )
        if result["pages"]:
            return result
        raise RuntimeError("PDF 결과에서 추출 가능한 텍스트를 찾지 못했습니다.")

    attempted_ocr_pages = len(ocr_candidate_pages)
    try:
        if progress_callback is not None:
            _emit_progress(
                progress_callback,
                38,
                "텍스트가 부족한 PDF 페이지를 OCR로 보완하는 중입니다.",
                "fallback_pdf_ocr",
                current_page=len(text_pages),
                total_pages=max(0, total_pages),
                pdf_total_pages=max(0, total_pages),
                ocr_target_pages=attempted_ocr_pages,
                ocr_completed_pages=0,
            )
        _set_last_ocr_runtime_info({})
        ocr_pages_raw = extract_pdf_pages_with_paddleocr_vl(
            pdf_path,
            progress_callback=progress_callback,
            page_numbers=ocr_candidate_pages,
            total_document_pages=total_pages or None,
            completed_pages_base=len(text_pages),
        )
        ocr_runtime = _peek_last_ocr_runtime_info()
    except Exception:
        if text_pages:
            warnings.append("ocr_fallback_failed")
            return _build_pdf_result(
                total_pages=total_pages,
                pages=list(text_pages.values()),
                warnings=warnings,
                attempted_ocr_pages=attempted_ocr_pages,
                ocr_runtime=None,
            )
        raise
    finally:
        gc.collect()

    merged_pages: Dict[int, Dict[str, Any]] = dict(text_pages)
    max_ocr_page_no = 0
    for page in ocr_pages_raw:
        page_no = max(1, int(page.get("page_no", 0) or 0))
        max_ocr_page_no = max(max_ocr_page_no, page_no)
        if total_pages > 0 and page_no not in ocr_candidate_pages and page_no in merged_pages:
            continue
        page_text = _normalize_block(str(page.get("text", "") or ""))
        if not _page_has_meaningful_text(page_text, min_text_chars, min_nonspace_ratio):
            continue
        merged_pages[page_no] = {
            "page_no": page_no,
            "text": page_text,
            "parser": "paddleocr_vl",
        }

    safe_total_pages = total_pages
    if safe_total_pages <= 0:
        safe_total_pages = max(max_ocr_page_no, len(merged_pages))

    result = _build_pdf_result(
        total_pages=safe_total_pages,
        pages=list(merged_pages.values()),
        warnings=warnings,
        attempted_ocr_pages=attempted_ocr_pages,
        ocr_runtime=ocr_runtime,
    )
    if result["pages"]:
        del ocr_pages_raw
        gc.collect()
        return result
    raise RuntimeError("PDF 결과에서 추출 가능한 텍스트를 찾지 못했습니다.")
