#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 acceptance_report.json" >&2
  exit 2
fi

REPORT_JSON="$1"
if [[ ! -f "${REPORT_JSON}" ]]; then
  echo "[ERROR] Acceptance report not found: ${REPORT_JSON}" >&2
  exit 2
fi

REPORT_JSON="${REPORT_JSON}" python3 - <<'PY'
import json
import os
from pathlib import Path


path = Path(os.environ["REPORT_JSON"]).resolve()
report = json.loads(path.read_text(encoding="utf-8"))
checklist = report.get("checklist") if isinstance(report, dict) else {}
if not isinstance(checklist, dict):
    checklist = {}

required_true = [
    "target_achieved_true",
    "target_met_case_exists",
    "ocr_pages_ge_target",
    "elapsed_seconds_le_target",
    "peak_gpu_metric_present",
    "process_rss_metric_present",
    "peak_gpu_le_limit",
    "process_rss_le_limit",
]

failures = []
if report.get("status") != "pass":
    failures.append(f"status={report.get('status')!r}")
for key in required_true:
    if checklist.get(key) is not True:
        failures.append(f"checklist.{key}={checklist.get(key)!r}")

selected = report.get("selected_case")
if not isinstance(selected, dict):
    failures.append("selected_case missing")
else:
    target_pages = int(float(selected.get("target_pages", report.get("target_pages", 200)) or 0))
    target_seconds = float(selected.get("target_seconds", report.get("target_seconds", 300)) or 0)
    ocr_pages = int(float(selected.get("ocr_pages", 0) or 0))
    elapsed = float(selected.get("elapsed_seconds", 0) or 0)
    if target_pages < 200:
        failures.append(f"target_pages {target_pages} < 200")
    if target_seconds > 300:
        failures.append(f"target_seconds {target_seconds} > 300")
    if ocr_pages < target_pages:
        failures.append(f"ocr_pages {ocr_pages} < target_pages {target_pages}")
    if elapsed <= 0 or elapsed > target_seconds:
        failures.append(f"elapsed_seconds {elapsed} > target_seconds {target_seconds}")

summary_json = str(report.get("summary_json", "") or "")
if not summary_json:
    failures.append("summary_json missing")

if failures:
    print(json.dumps({"status": "fail", "report_json": str(path), "failures": failures}, ensure_ascii=False, indent=2))
    raise SystemExit(3)

print(json.dumps({
    "status": "ok",
    "report_json": str(path),
    "summary_json": summary_json,
    "selected_case": selected.get("name", ""),
    "ocr_pages": int(float(selected.get("ocr_pages", 0) or 0)),
    "elapsed_seconds": float(selected.get("elapsed_seconds", 0) or 0),
    "peak_gpu_memory_used_mb": int(float(selected.get("peak_gpu_memory_used_mb", 0) or 0)),
    "peak_process_tree_rss_mb": int(float(selected.get("peak_process_tree_rss_mb", selected.get("peak_process_rss_mb", 0)) or 0)),
    "runtime_env_written": bool(report.get("runtime_env_written", False)),
}, ensure_ascii=False, indent=2))
PY
