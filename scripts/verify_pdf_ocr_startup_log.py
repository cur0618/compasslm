#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_WORKER_COUNT_RE = re.compile(r"\bworker_count=(\d+)\b")
_WORKER_PIDS_RE = re.compile(r"\bworker_pids=([^\s]+)")


def verify_pdf_ocr_startup_log(log_path: Path, *, expected_workers: int) -> Dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    errors: List[str] = []

    start_indexes = [index for index, line in enumerate(lines) if "[PDF_OCR][START]" in line]
    if not start_indexes:
        return {
            "ok": False,
            "errors": ["ocr_start_missing"],
            "warmup_worker_count": 0,
            "initial_submit_count": 0,
        }
    start_index = start_indexes[-1]

    warmup_lines = [line for line in lines[:start_index] if "[PDF_OCR][WARMUP]" in line]
    warmup_worker_count = 0
    warmup_worker_pids: List[int] = []
    if not warmup_lines or "status=ready" not in warmup_lines[-1]:
        errors.append("warmup_ready_missing")
    else:
        match = _WORKER_COUNT_RE.search(warmup_lines[-1])
        if match is None:
            errors.append("warmup_worker_count_missing")
        else:
            warmup_worker_count = int(match.group(1))
            if warmup_worker_count != expected_workers:
                errors.append("warmup_worker_count_mismatch")
        pid_match = _WORKER_PIDS_RE.search(warmup_lines[-1])
        if pid_match is None or pid_match.group(1) == "-":
            errors.append("warmup_worker_pids_missing")
        else:
            try:
                parsed_pids = [int(value) for value in pid_match.group(1).split(",")]
            except ValueError:
                parsed_pids = []
            warmup_worker_pids = sorted(set(pid for pid in parsed_pids if pid > 0))
            if len(warmup_worker_pids) != expected_workers:
                errors.append("warmup_worker_pids_mismatch")

    first_batch_index = next(
        (
            index
            for index in range(start_index + 1, len(lines))
            if "[PDF_OCR][BATCH]" in lines[index]
        ),
        None,
    )
    scan_end = first_batch_index if first_batch_index is not None else len(lines)
    initial_submit_count = sum(
        1 for line in lines[start_index + 1 : scan_end] if "[PDF_OCR][BATCH_SUBMIT]" in line
    )
    if initial_submit_count > expected_workers:
        errors.append("initial_submit_window_exceeded")

    expected_mode = "single_gpu" if expected_workers == 1 else "parallel_gpu"
    if first_batch_index is None or f"mode={expected_mode}" not in lines[first_batch_index]:
        errors.append("first_gpu_batch_missing")

    pre_first_batch = lines[start_index + 1 : scan_end]
    if any("Creating model:" in line for line in pre_first_batch):
        errors.append("model_reloaded_after_ocr_start")
    if any("gpu_timeout" in line for line in pre_first_batch):
        errors.append("gpu_timeout_before_first_batch")

    return {
        "ok": not errors,
        "errors": errors,
        "warmup_worker_count": warmup_worker_count,
        "warmup_worker_pids": warmup_worker_pids,
        "expected_workers": expected_workers,
        "initial_submit_count": initial_submit_count,
        "first_batch_line": (first_batch_index + 1) if first_batch_index is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that PDF OCR starts with fully warmed workers and bounded batch submission."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--expected-workers", type=int, default=1)
    args = parser.parse_args()

    result = verify_pdf_ocr_startup_log(
        args.log_path,
        expected_workers=max(1, int(args.expected_workers)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
