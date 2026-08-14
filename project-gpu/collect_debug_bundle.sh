#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASSLM_HOME="${COMPASSLM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_gpu_env.sh"

if [[ "${COMPASSLM_DEBUG_SKIP_ENV_LOAD:-0}" != "1" ]]; then
  compass_load_env_file "${MAIN_BACKEND_HOME}/.env.auto"
  compass_load_env_file "${PROJECT_GPU_HOME}/runtime.env"
  compass_load_env_file "${MAIN_BACKEND_HOME}/.env"
fi

export API_HOST="${API_HOST:-127.0.0.1}"
export API_PORT="${API_PORT:-8004}"
export COMPASSLM_LOGS_DIR="${COMPASSLM_LOGS_DIR:-${COMPASSLM_HOME}/logs}"
export COMPASSLM_APP_DB_PATH="${COMPASSLM_APP_DB_PATH:-${COMPASSLM_HOME}/data/app.sqlite}"
export KB_DATA_DIR="${KB_DATA_DIR:-${COMPASSLM_HOME}/data/kb}"
if [[ -z "${COMPASSLM_DEBUG_API_BASE_URL:-}" ]]; then
  if ! COMPASSLM_DEBUG_API_BASE_URL="$(compass_require_live_backend_url)"; then
    echo "[ERROR] Debug bundle could not resolve a ready backend URL." >&2
    exit 1
  fi
fi
export COMPASSLM_DEBUG_API_BASE_URL
export COMPASSLM_DEBUG_AUTH_TOKEN="${COMPASSLM_DEBUG_AUTH_TOKEN:-}"
export COMPASSLM_DEBUG_AUTH_COOKIE="${COMPASSLM_DEBUG_AUTH_COOKIE:-}"
export COMPASSLM_DEBUG_AUTH_COOKIE_NAME="${COMPASSLM_DEBUG_AUTH_COOKIE_NAME:-compass_auth_session}"
export COMPASSLM_DEBUG_COOKIE_HEADER="${COMPASSLM_DEBUG_COOKIE_HEADER:-}"
export COMPASSLM_BACKEND_LOG_PATH="${COMPASSLM_BACKEND_LOG_PATH:-}"

python3 - <<'PY'
import json
import os
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request


PROJECT_ROOT = Path(os.environ["COMPASSLM_HOME"]).resolve()
LOGS_DIR = Path(os.environ["COMPASSLM_LOGS_DIR"]).resolve()
APP_DB_PATH = Path(os.environ["COMPASSLM_APP_DB_PATH"]).resolve()
KB_DATA_DIR = Path(os.environ["KB_DATA_DIR"]).resolve()
API_BASE_URL = os.environ["COMPASSLM_DEBUG_API_BASE_URL"].rstrip("/")
DEBUG_AUTH_TOKEN = os.environ.get("COMPASSLM_DEBUG_AUTH_TOKEN", "").strip()
DEBUG_AUTH_COOKIE = os.environ.get("COMPASSLM_DEBUG_AUTH_COOKIE", "").strip()
DEBUG_AUTH_COOKIE_NAME = os.environ.get("COMPASSLM_DEBUG_AUTH_COOKIE_NAME", "compass_auth_session").strip() or "compass_auth_session"
DEBUG_COOKIE_HEADER = os.environ.get("COMPASSLM_DEBUG_COOKIE_HEADER", "").strip()
CAPTURE_ID = os.environ.get("COMPASSLM_DEBUG_CAPTURE_NOW") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
CAPTURE_DIR = LOGS_DIR / "debug-captures"
CAPTURE_PATH = CAPTURE_DIR / f"{CAPTURE_ID}.json"
RECENT_CHAT_MESSAGE_LIMIT = 50
RECENT_AGENT_RUN_LIMIT = 50
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def truncate_text(value: Any, limit: int = 180) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[:1] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def sqlite_rows(db_path: Path, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return rows


def sqlite_value(db_path: Path, query: str, params: Tuple[Any, ...] = ()) -> Optional[Any]:
    rows = sqlite_rows(db_path, query, params)
    if not rows:
        return None
    first = rows[0]
    if not first:
        return None
    return next(iter(first.values()))


def read_jsonl_tail(path: Path, limit: int = 20) -> List[Any]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [{"_error": f"log_read_fail: {exc}", "_path": str(path)}]
    entries: List[Any] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            entries.append({"_raw": line})
    return entries


def read_text_tail(path: Path, limit: int = 120) -> List[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception as exc:
        return [f"log_read_fail: {exc}"]


def fetch_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    if DEBUG_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {DEBUG_AUTH_TOKEN}"
    if DEBUG_COOKIE_HEADER:
        headers["Cookie"] = DEBUG_COOKIE_HEADER
    elif DEBUG_AUTH_COOKIE:
        headers["Cookie"] = f"{DEBUG_AUTH_COOKIE_NAME}={DEBUG_AUTH_COOKIE}"
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload: Any
            try:
                payload = json.loads(body)
            except Exception:
                payload = body
            return {
                "ok": True,
                "url": url,
                "status_code": getattr(resp, "status", 200),
                "data": payload,
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = body
        return {
            "ok": False,
            "url": url,
            "status_code": exc.code,
            "error": str(exc),
            "data": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": str(exc),
        }


def decode_fields(rows: List[Dict[str, Any]], json_fields: List[str]) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field in json_fields:
            if field in item:
                item[field] = parse_maybe_json(item[field])
        decoded.append(item)
    return decoded


def coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def unix_to_iso(value: Any) -> str:
    ts = coerce_int(value)
    if ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()
    except Exception:
        return ""


def kb_meta_latest_activity(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    retrieval_latest = coerce_int(sqlite_value(db_path, "SELECT MAX(created_at) FROM retrieval_logs"))
    answer_latest = coerce_int(sqlite_value(db_path, "SELECT MAX(created_at) FROM answer_logs"))
    return max(retrieval_latest, answer_latest)


def build_kb_snapshot(kb_name: str) -> Dict[str, Any]:
    kb_db_path = KB_DATA_DIR / kb_name / "meta.sqlite"
    latest_session_id = sqlite_value(
        APP_DB_PATH,
        "SELECT session_id FROM agent_runs WHERE kb_name = ? ORDER BY created_at DESC, run_id DESC LIMIT 1",
        (kb_name,),
    )
    if not isinstance(latest_session_id, str) or not latest_session_id.strip():
        latest_session_id = sqlite_value(
            APP_DB_PATH,
            "SELECT session_id FROM chat_messages WHERE kb_name = ? ORDER BY created_at DESC, message_id DESC LIMIT 1",
            (kb_name,),
        )

    chat_messages_query = """
    SELECT message_id, session_id, kb_name, role, text, created_at
    FROM chat_messages
    WHERE kb_name = ?
    """
    chat_message_params: List[Any] = [kb_name]
    if isinstance(latest_session_id, str) and latest_session_id.strip():
        chat_messages_query += " AND session_id = ?"
        chat_message_params.append(latest_session_id)
    chat_messages_query += " ORDER BY created_at DESC, message_id DESC LIMIT ?"
    chat_message_params.append(RECENT_CHAT_MESSAGE_LIMIT)
    recent_chat_messages = sqlite_rows(APP_DB_PATH, chat_messages_query, tuple(chat_message_params))
    recent_chat_messages.reverse()

    recent_agent_runs = decode_fields(
        sqlite_rows(
            APP_DB_PATH,
            """
            SELECT run_id, query_id, session_id, kb_name, user_message, answer_text, metadata_json,
                   response_quality_issue, usage_json, request_count, tool_call_count, input_tokens,
                   output_tokens, context_chars, created_at
            FROM agent_runs
            WHERE kb_name = ?
            ORDER BY created_at DESC, run_id DESC
            LIMIT ?
            """,
            (kb_name, RECENT_AGENT_RUN_LIMIT),
        ),
        ["metadata_json", "usage_json"],
    )

    latest_query_id = ""
    for row in recent_agent_runs:
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id.strip():
            latest_query_id = query_id.strip()
            break

    if latest_query_id:
        retrieval_query = """
        SELECT query_id, user_id, query_text, topk_ids_json, meta_json, created_at
        FROM retrieval_logs
        WHERE query_id = ?
        ORDER BY created_at DESC, log_id DESC
        LIMIT ?
        """
        retrieval_params = (latest_query_id, 6)
        answer_query = """
        SELECT query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at
        FROM answer_logs
        WHERE query_id = ?
        ORDER BY created_at DESC, log_id DESC
        LIMIT ?
        """
        answer_params = (latest_query_id, 6)
    else:
        retrieval_query = """
        SELECT query_id, user_id, query_text, topk_ids_json, meta_json, created_at
        FROM retrieval_logs
        ORDER BY created_at DESC, log_id DESC
        LIMIT ?
        """
        retrieval_params = (6,)
        answer_query = """
        SELECT query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at
        FROM answer_logs
        ORDER BY created_at DESC, log_id DESC
        LIMIT ?
        """
        answer_params = (6,)

    recent_retrieval_logs = decode_fields(
        sqlite_rows(kb_db_path, retrieval_query, retrieval_params),
        ["topk_ids_json", "meta_json"],
    )
    recent_answer_logs = decode_fields(
        sqlite_rows(kb_db_path, answer_query, answer_params),
        ["citations_json", "answer_meta_json"],
    )

    latest_activity_at = max(
        coerce_int(sqlite_value(APP_DB_PATH, "SELECT MAX(created_at) FROM chat_messages WHERE kb_name = ?", (kb_name,))),
        coerce_int(sqlite_value(APP_DB_PATH, "SELECT MAX(created_at) FROM agent_runs WHERE kb_name = ?", (kb_name,))),
        kb_meta_latest_activity(kb_db_path),
    )

    ops_url = API_BASE_URL + "/ops/failure-patterns?" + parse.urlencode({"kb_name": kb_name, "limit": 120})
    files_url = API_BASE_URL + f"/kbs/{parse.quote(kb_name, safe='')}/files"
    api_results = {
        "ops_failure_patterns": fetch_json(ops_url),
        "kb_files": fetch_json(files_url),
    }

    latest_user_message = next(
        (truncate_text(row.get("user_message")) for row in recent_agent_runs if row.get("user_message")),
        next(
            (truncate_text(row.get("text")) for row in reversed(recent_chat_messages) if row.get("role") == "user"),
            "",
        ),
    )

    return {
        "kb_name": kb_name,
        "kb_db_path": str(kb_db_path),
        "latest_session_id": latest_session_id or "",
        "latest_query_id": latest_query_id,
        "latest_activity_at_unix": latest_activity_at,
        "latest_activity_at_utc": unix_to_iso(latest_activity_at),
        "latest_user_message": latest_user_message,
        "recent_chat_messages": recent_chat_messages,
        "recent_agent_runs": recent_agent_runs,
        "recent_retrieval_logs": recent_retrieval_logs,
        "recent_answer_logs": recent_answer_logs,
        "api_results": api_results,
    }


warnings: List[str] = []

requested_kb_name = (os.environ.get("COMPASSLM_DEBUG_KB_NAME") or "").strip()
candidate_kb_names: List[str] = []
if requested_kb_name:
    candidate_kb_names.append(requested_kb_name)
else:
    for row in sqlite_rows(
        APP_DB_PATH,
        """
        SELECT kb_name, MAX(created_at) AS latest_created_at
        FROM (
            SELECT kb_name, created_at FROM chat_messages
            UNION ALL
            SELECT kb_name, created_at FROM agent_runs
        )
        GROUP BY kb_name
        ORDER BY latest_created_at DESC, kb_name ASC
        """,
    ):
        kb_name = str(row.get("kb_name") or "").strip()
        if kb_name and kb_name not in candidate_kb_names:
            candidate_kb_names.append(kb_name)
    if KB_DATA_DIR.exists():
        for child in sorted(KB_DATA_DIR.iterdir()):
            if child.is_dir() and child.name not in candidate_kb_names:
                candidate_kb_names.append(child.name)

if not candidate_kb_names:
    candidate_kb_names = ["default"]

if not APP_DB_PATH.exists():
    warnings.append(f"app_db_missing:{APP_DB_PATH}")

recent_kb_snapshots = [build_kb_snapshot(kb_name) for kb_name in candidate_kb_names]
recent_kb_snapshots.sort(
    key=lambda item: (
        -coerce_int(item.get("latest_activity_at_unix")),
        str(item.get("kb_name") or ""),
    )
)

primary_snapshot = recent_kb_snapshots[0]
resolved_kb_name = str(primary_snapshot.get("kb_name") or "default")
latest_session_id = str(primary_snapshot.get("latest_session_id") or "")
latest_query_id = str(primary_snapshot.get("latest_query_id") or "")
kb_db_path = Path(str(primary_snapshot.get("kb_db_path") or (KB_DATA_DIR / resolved_kb_name / "meta.sqlite")))
recent_chat_messages = list(primary_snapshot.get("recent_chat_messages") or [])
recent_agent_runs = list(primary_snapshot.get("recent_agent_runs") or [])
recent_retrieval_logs = list(primary_snapshot.get("recent_retrieval_logs") or [])
recent_answer_logs = list(primary_snapshot.get("recent_answer_logs") or [])

admin_feedback_entries = read_jsonl_tail(LOGS_DIR / "admin_feedback.jsonl", limit=10)
rag_trace_entries = read_jsonl_tail(LOGS_DIR / "rag_trace.jsonl", limit=10)

backend_log_candidates = [
    Path(os.environ["COMPASSLM_BACKEND_LOG_PATH"]) if os.environ.get("COMPASSLM_BACKEND_LOG_PATH") else None,
    LOGS_DIR / "backend.log",
    LOGS_DIR / "backend_api.log",
    LOGS_DIR / "run_backend_api.log",
    LOGS_DIR / "uvicorn.log",
    PROJECT_ROOT / "project-gpu" / "main-backend" / "logs" / "backend.log",
    PROJECT_ROOT / "project-gpu" / "main-backend" / "logs" / "backend_api.log",
    PROJECT_ROOT / "project-gpu" / "main-backend" / "logs" / "uvicorn.log",
]
backend_log_path = next((path for path in backend_log_candidates if path is not None and path.exists()), None)
backend_log_tail = read_text_tail(backend_log_path, limit=120) if backend_log_path else []
if backend_log_path is None:
    warnings.append("backend_log_not_found")
if not kb_db_path.exists():
    warnings.append(f"kb_db_missing:{kb_db_path}")

api_results = dict(primary_snapshot.get("api_results") or {})
api_auth_configured = bool(DEBUG_AUTH_TOKEN or DEBUG_AUTH_COOKIE or DEBUG_COOKIE_HEADER)
api_auth_missing = (not api_auth_configured) and any(
    int(result.get("status_code", 0) or 0) == 401
    for result in api_results.values()
    if isinstance(result, dict)
)
if api_auth_missing:
    warnings.append("api_auth_missing")

startup_snapshot = {
    "project_root": str(PROJECT_ROOT),
    "logs_dir": str(LOGS_DIR),
    "app_db_path": str(APP_DB_PATH),
    "kb_data_dir": str(KB_DATA_DIR),
    "kb_db_path": str(kb_db_path),
    "api_base_url": API_BASE_URL,
    "api_auth_token_configured": bool(DEBUG_AUTH_TOKEN),
    "api_auth_cookie_configured": bool(DEBUG_AUTH_COOKIE or DEBUG_COOKIE_HEADER),
    "api_host": os.environ.get("API_HOST", ""),
    "api_port": os.environ.get("API_PORT", ""),
    "embedding_api_url": os.environ.get("EMBEDDING_API_URL", ""),
    "llm_api_url": os.environ.get("LLM_API_URL", ""),
    "llm_ctx_size": os.environ.get("LLM_CTX_SIZE", ""),
    "llm_context_limit": os.environ.get("LLM_CONTEXT_LIMIT", ""),
    "pdf_parse_mode": os.environ.get("PDF_PARSE_MODE", "hybrid"),
    "pdf_text_extractor": os.environ.get("PDF_TEXT_EXTRACTOR", "pymupdf"),
    "pdf_ocr_model_name": os.environ.get("PDF_OCR_MODEL_NAME", ""),
    "pdf_ocr_device": os.environ.get("PDF_OCR_DEVICE", "cpu"),
    "pdf_ocr_allow_online_model_fallback": os.environ.get("PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK", ""),
    "pdf_ocr_use_internal_queues": os.environ.get("PDF_OCR_USE_INTERNAL_QUEUES", ""),
    "pdf_ocr_vl_model_dir": os.environ.get("PDF_OCR_VL_MODEL_DIR", ""),
    "pdf_ocr_layout_model_dir": os.environ.get("PDF_OCR_LAYOUT_MODEL_DIR", ""),
    "pydantic_ai_provider_kind": os.environ.get("PYDANTIC_AI_PROVIDER_KIND", "openai_compatible"),
    "pydantic_ai_provider_label": os.environ.get("PYDANTIC_AI_PROVIDER_LABEL", ""),
    "pydantic_ai_history_strategy": os.environ.get("PYDANTIC_AI_HISTORY_STRATEGY", "compact_text"),
    "paddle_pdx_disable_model_source_check": os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", ""),
    "env_files": [
        {"path": str(Path(os.environ["MAIN_BACKEND_HOME"]) / ".env.auto"), "exists": (Path(os.environ["MAIN_BACKEND_HOME"]) / ".env.auto").exists()},
        {"path": str(Path(os.environ["PROJECT_GPU_HOME"]) / "runtime.env"), "exists": (Path(os.environ["PROJECT_GPU_HOME"]) / "runtime.env").exists()},
        {"path": str(Path(os.environ["MAIN_BACKEND_HOME"]) / ".env"), "exists": (Path(os.environ["MAIN_BACKEND_HOME"]) / ".env").exists()},
    ],
}

summary = {
    "kb_count": len(recent_kb_snapshots),
    "kb_names_by_recent_activity": [str(item.get("kb_name") or "") for item in recent_kb_snapshots],
    "kb_last_questions": {
        str(item.get("kb_name") or ""): str(item.get("latest_user_message") or "")
        for item in recent_kb_snapshots
        if str(item.get("kb_name") or "")
    },
    "recent_chat_message_count": len(recent_chat_messages),
    "recent_agent_run_count": len(recent_agent_runs),
    "recent_retrieval_log_count": len(recent_retrieval_logs),
    "recent_answer_log_count": len(recent_answer_logs),
    "admin_feedback_tail_count": len(admin_feedback_entries),
    "rag_trace_tail_count": len(rag_trace_entries),
    "latest_session_id": latest_session_id or "",
    "latest_query_id": latest_query_id,
    "latest_user_message": str(primary_snapshot.get("latest_user_message") or ""),
    "ops_failure_patterns_ok": bool(api_results["ops_failure_patterns"].get("ok")),
    "api_auth_missing": api_auth_missing,
}

bundle = {
    "bundle_version": 1,
    "capture_id": CAPTURE_ID,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "resolved_kb_name": resolved_kb_name,
    "latest_session_id": latest_session_id or "",
    "latest_query_id": latest_query_id,
    "startup_snapshot": startup_snapshot,
    "api_results": api_results,
    "recent_kb_snapshots": recent_kb_snapshots,
    "kb_snapshots_by_name": {
        str(item.get("kb_name") or ""): item
        for item in recent_kb_snapshots
        if str(item.get("kb_name") or "")
    },
    "recent_chat_messages": recent_chat_messages,
    "recent_agent_runs": recent_agent_runs,
    "recent_retrieval_logs": recent_retrieval_logs,
    "recent_answer_logs": recent_answer_logs,
    "log_tails": {
        "admin_feedback": admin_feedback_entries,
        "rag_trace": rag_trace_entries,
        "backend_log_path": str(backend_log_path) if backend_log_path else "",
        "backend_log_tail": backend_log_tail,
    },
    "summary": summary,
    "warnings": warnings,
    "host": {
        "hostname": socket.gethostname(),
    },
}

CAPTURE_PATH.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[DEBUG_BUNDLE] saved={CAPTURE_PATH}")
print(
    "[DEBUG_BUNDLE] primary_kb="
    f"{resolved_kb_name} "
    f"kb_names={','.join(summary['kb_names_by_recent_activity'])} "
    f"latest_question={summary['latest_user_message']}"
)
PY
