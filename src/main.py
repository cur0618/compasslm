import asyncio
import contextvars
import hashlib
import os
import queue
import json
import re
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Mapping
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from src.citation_labels import canonicalize_doc_citations, render_answer_with_bottom_citations
from src.answer_validation import (
    contains_disallowed_markdown,
    extract_numeric_signatures,
    has_grounded_numeric_answer,
    is_grounded_abstention_text,
    is_numeric_evidence_query,
    is_weak_ocr_hint_text,
    should_treat_abstention_as_quality_issue,
    summarize_evidence_strength,
    should_allow_general_knowledge_fallback as should_allow_general_knowledge_fallback_by_policy,
    should_auto_prefetch_numeric_evidence,
)
from src.chat_policy import (
    build_scope_narrowing_response,
    explain_scope_nudge_reason,
    should_prompt_for_narrower_summary,
)
from src.chat_store import is_failed_history_answer_text
from src.auth_store import AuthStore
from src.conversation_mode import (
    is_live_info_request,
    resolve_conversation_mode,
    should_force_followup_rewrite,
    summarize_recent_conversation_state,
)
from src.compass_ai import (
    AgentRunState,
    ChatStore,
    CompassAgentDeps,
    FollowupAnalysis,
    HelperRunDiagnostics,
    PydanticAIService,
    QuestionAnalysis,
    RetrievalSnapshot,
    RetrievedDocRecord,
    ToolEventRecord,
    build_phase_event,
    compact_chat_history_rows,
    phase_events_to_dicts,
    run_usage_to_dict,
    trim_preview,
)
from src.kb_engine_registry import KBEngineRegistry
from src.ontology_store import OntologyStore
from src.persistence_retention import rotate_file_if_oversize
from src.query_orchestration import decide_rerank_usage, run_parallel_helper_tasks
from src.query_rewrite import resolve_effective_query, should_attempt_followup_rewrite
from src.rag import RAGEngine, list_kbs, get_kb_files, rename_kb_dir, delete_kb_dir, delete_file_from_kb
from src.rerank_budget import trim_rerank_candidate_lines_to_budget
from src.retrieval_quality import apply_critical_term_gate, rerank_results_for_grounded_answer, select_auto_prefetch_documents
from src.table_facts import parse_table_fact_line, table_fact_matches_query
from src.upload_helpers import (
    build_stored_upload_name,
    is_hwpx_signature,
    is_pdf_signature,
    is_zip_signature,
    safe_upload_filename,
    validate_upload_meta,
)
from src.upload_progress import (
    build_upload_stall_state,
    estimate_background_ocr_progress_percent,
    estimate_display_progress_percent,
    update_upload_phase_state,
    upload_failure_default_for_stage,
)
from src.upload_job_store import UploadJobStore
from src.wiki_memory_store import WikiMemoryStore
from src.wiki_page_builder import WikiPageBuilder
from src.wiki_store import WikiStore

app = FastAPI()
_request_user_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "compasslm_request_user_id",
    default="",
)

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = Path(os.getenv("COMPASSLM_HOME", str(APP_FILE.parent.parent))).resolve()
KB_DATA_DIR = Path(os.getenv("KB_DATA_DIR", str(PROJECT_ROOT / "data" / "kb"))).resolve()
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", str(APP_FILE.parent / "templates"))).resolve()
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(APP_FILE.parent / "static"))).resolve()
LOGS_DIR = Path(os.getenv("COMPASSLM_LOGS_DIR", str(PROJECT_ROOT / "logs"))).resolve()
APP_DB_PATH = Path(os.getenv("COMPASSLM_APP_DB_PATH", str(PROJECT_ROOT / "data" / "app.sqlite"))).resolve()
ADMIN_FEEDBACK_LOG_PATH = Path(
    os.getenv("ADMIN_FEEDBACK_LOG_PATH", str(LOGS_DIR / "admin_feedback.jsonl"))
).resolve()
OPERATIONAL_JSONL_MAX_BYTES = max(
    1024 * 1024,
    int(os.getenv("OPERATIONAL_JSONL_MAX_BYTES", str(20 * 1024 * 1024))),
)
OPERATIONAL_JSONL_BACKUP_COUNT = max(
    1,
    int(os.getenv("OPERATIONAL_JSONL_BACKUP_COUNT", "5")),
)

# Configuration
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8003/v1/chat/completions")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1800"))
LLM_CONTEXT_LIMIT = int(os.getenv("LLM_CONTEXT_LIMIT", "131072"))
LLM_CONTEXT_SAFETY_MARGIN = int(os.getenv("LLM_CONTEXT_SAFETY_MARGIN", "120"))
LLM_PROMPT_OVERHEAD_TOKENS = int(os.getenv("LLM_PROMPT_OVERHEAD_TOKENS", "280"))
LLM_MIN_RESPONSE_TOKENS = int(os.getenv("LLM_MIN_RESPONSE_TOKENS", "160"))
RAG_CHARS_PER_TOKEN_EST = float(os.getenv("RAG_CHARS_PER_TOKEN_EST", "2.2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_QUALITY_RETRY_ENABLED = _env_bool("LLM_QUALITY_RETRY_ENABLED", True)
LLM_QUALITY_MAX_RETRY = max(0, int(os.getenv("LLM_QUALITY_MAX_RETRY", "2")))
LLM_MIN_ANSWER_LINE_CHARS = max(16, int(os.getenv("LLM_MIN_ANSWER_LINE_CHARS", "40")))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "24"))
RAG_TOP_K_OVERVIEW = int(os.getenv("RAG_TOP_K_OVERVIEW", "36"))
RAG_CONTEXT_MAX_CHARS = max(2000, int(os.getenv("RAG_CONTEXT_MAX_CHARS", "5600")))
RAG_CONTEXT_PER_RESULT_MAX_CHARS = max(240, int(os.getenv("RAG_CONTEXT_PER_RESULT_MAX_CHARS", "700")))
RAG_CONTEXT_RETRY_MIN_CHARS = int(os.getenv("RAG_CONTEXT_RETRY_MIN_CHARS", "700"))
RAG_OVERVIEW_EXTRA_CHARS = int(os.getenv("RAG_OVERVIEW_EXTRA_CHARS", "2400"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
RAG_LLM_QUERY_EXPAND_ENABLED = _env_bool("RAG_LLM_QUERY_EXPAND_ENABLED", True)
RAG_LLM_QUERY_ANALYZE_ENABLED = _env_bool("RAG_LLM_QUERY_ANALYZE_ENABLED", True)
RAG_LLM_RERANK_ENABLED = _env_bool("RAG_LLM_RERANK_ENABLED", True)
RAG_LLM_RERANK_CANDIDATES = int(os.getenv("RAG_LLM_RERANK_CANDIDATES", "72"))
RAG_LLM_RERANK_KEEP = int(os.getenv("RAG_LLM_RERANK_KEEP", "20"))
RAG_LLM_HELPER_MAX_TOKENS = int(os.getenv("RAG_LLM_HELPER_MAX_TOKENS", "220"))
RAG_LLM_HELPER_TIMEOUT = int(os.getenv("RAG_LLM_HELPER_TIMEOUT", "45"))
RAG_NUMBER_REF_STRICT = _env_bool("RAG_NUMBER_REF_STRICT", True)
RAG_DIVERSIFY_ENABLED = _env_bool("RAG_DIVERSIFY_ENABLED", True)
RAG_MAX_PER_SECTION = max(1, int(os.getenv("RAG_MAX_PER_SECTION", "2")))
RAG_MAX_PER_FILE = max(1, int(os.getenv("RAG_MAX_PER_FILE", "6")))
RAG_GROUNDING_GATE_ENABLED = _env_bool("RAG_GROUNDING_GATE_ENABLED", True)
RAG_GROUNDING_TOP1_MIN = float(os.getenv("RAG_GROUNDING_TOP1_MIN", "0.33"))
RAG_GROUNDING_COVERAGE_MIN = float(os.getenv("RAG_GROUNDING_COVERAGE_MIN", "0.22"))
RAG_GROUNDING_SOFTEN_ENABLED = _env_bool("RAG_GROUNDING_SOFTEN_ENABLED", True)
RAG_GROUNDING_TOP1_SOFT_MIN = float(os.getenv("RAG_GROUNDING_TOP1_SOFT_MIN", "0.26"))
RAG_GROUNDING_COVERAGE_SOFT_MIN = float(os.getenv("RAG_GROUNDING_COVERAGE_SOFT_MIN", "0.12"))
RAG_GROUNDING_MIN_KEYWORD_HITS = max(1, int(os.getenv("RAG_GROUNDING_MIN_KEYWORD_HITS", "1")))
RAG_GROUNDING_CONFLICT_CHECK_ENABLED = _env_bool("RAG_GROUNDING_CONFLICT_CHECK_ENABLED", True)
RAG_ANSWER_COVERAGE_TOP_K = int(os.getenv("RAG_ANSWER_COVERAGE_TOP_K", "8"))
RAG_MAX_NORMALIZED_RESULTS = max(0, int(os.getenv("RAG_MAX_NORMALIZED_RESULTS", "2")))
RAG_CRITICAL_TERM_GATE_ENABLED = _env_bool("RAG_CRITICAL_TERM_GATE_ENABLED", True)
RAG_NORMALIZED_RAW_BACKING_REQUIRED = _env_bool("RAG_NORMALIZED_RAW_BACKING_REQUIRED", True)
RAG_TRACE_LOG_ENABLED = _env_bool("RAG_TRACE_LOG_ENABLED", True)
RAG_TRACE_TOP_N = max(1, int(os.getenv("RAG_TRACE_TOP_N", "8")))
RAG_TRACE_LOG_PATH = Path(os.getenv("RAG_TRACE_LOG_PATH", str(LOGS_DIR / "rag_trace.jsonl"))).resolve()
RAG_ROLE_ROUTING_ENABLED = _env_bool("RAG_ROLE_ROUTING_ENABLED", True)
RAG_ROLE_ROUTING_STRICT = _env_bool("RAG_ROLE_ROUTING_STRICT", False)
WIKI_ANSWER_MEMORY_ENABLED = _env_bool("WIKI_ANSWER_MEMORY_ENABLED", True)
WIKI_ANSWER_MEMORY_FASTPATH_ENABLED = _env_bool("WIKI_ANSWER_MEMORY_FASTPATH_ENABLED", True)
WIKI_ANSWER_MEMORY_FASTPATH_MIN_SCORE = float(os.getenv("WIKI_ANSWER_MEMORY_FASTPATH_MIN_SCORE", "0.82"))
WIKI_ACTIVE_RETRIEVAL_ENABLED = _env_bool("WIKI_ACTIVE_RETRIEVAL_ENABLED", False)
WIKI_PAGE_WORKFLOW_ENABLED = _env_bool("WIKI_PAGE_WORKFLOW_ENABLED", False)
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "compass_session_id")
AUTH_SESSION_COOKIE_NAME = os.getenv("AUTH_SESSION_COOKIE_NAME", "compass_auth_session")
AUTH_SESSION_TTL_SECONDS = max(300, int(os.getenv("AUTH_SESSION_TTL_SECONDS", "86400")))
AUTH_BOOTSTRAP_ADMIN_LOGIN = os.getenv("AUTH_BOOTSTRAP_ADMIN_LOGIN", "admin")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "200"))
CHAT_AGENT_RUN_LIMIT = max(1, int(os.getenv("CHAT_AGENT_RUN_LIMIT", "10")))
UPLOAD_MAX_BYTES = max(1024 * 1024, int(os.getenv("UPLOAD_MAX_BYTES", str(40 * 1024 * 1024))))
UPLOAD_ASYNC_ENABLED = _env_bool("UPLOAD_ASYNC_ENABLED", True)
UPLOAD_QUEUE_MAXSIZE = max(4, int(os.getenv("UPLOAD_QUEUE_MAXSIZE", "200")))
UPLOAD_WORKER_COUNT = max(1, int(os.getenv("UPLOAD_WORKER_COUNT", "1")))
UPLOAD_FAST_WORKER_COUNT = max(1, int(os.getenv("UPLOAD_FAST_WORKER_COUNT", "1")))
UPLOAD_JOB_RETENTION_SECONDS = max(300, int(os.getenv("UPLOAD_JOB_RETENTION_SECONDS", "86400")))
UPLOAD_JOB_RETENTION_MAX_ROWS = max(100, int(os.getenv("UPLOAD_JOB_RETENTION_MAX_ROWS", "5000")))
UPLOAD_JOB_PRUNE_INTERVAL_SECONDS = max(60, int(os.getenv("UPLOAD_JOB_PRUNE_INTERVAL_SECONDS", "3600")))
UPLOAD_JOB_PRUNE_BATCH_SIZE = max(100, int(os.getenv("UPLOAD_JOB_PRUNE_BATCH_SIZE", "1000")))
AUTH_SESSION_PRUNE_LIMIT = max(100, int(os.getenv("AUTH_SESSION_PRUNE_LIMIT", "1000")))
UPLOAD_JOB_STALL_TIMEOUT_SECONDS = max(180, int(os.getenv("UPLOAD_JOB_STALL_TIMEOUT_SECONDS", "480")))
UPLOAD_QUEUE_STALL_TIMEOUT_SECONDS = max(0, int(os.getenv("UPLOAD_QUEUE_STALL_TIMEOUT_SECONDS", "0")))
UPLOAD_JOB_LONG_POLL_MAX_WAIT_SECONDS = max(
    5.0,
    float(os.getenv("UPLOAD_JOB_LONG_POLL_MAX_WAIT_SECONDS", "45")),
)
UPLOAD_JOB_RECOVERY_LIMIT = max(0, int(os.getenv("UPLOAD_JOB_RECOVERY_LIMIT", "100")))
PDF_BACKGROUND_OCR_ENABLED = _env_bool("PDF_BACKGROUND_OCR_ENABLED", False)
PDF_BACKGROUND_OCR_QUEUE_MAXSIZE = max(4, int(os.getenv("PDF_BACKGROUND_OCR_QUEUE_MAXSIZE", "100")))
PDF_BACKGROUND_OCR_WORKER_COUNT = max(1, int(os.getenv("PDF_BACKGROUND_OCR_WORKER_COUNT", "1")))
PDF_BACKGROUND_OCR_JOB_RETENTION_SECONDS = max(300, int(os.getenv("PDF_BACKGROUND_OCR_JOB_RETENTION_SECONDS", "86400")))
ONTOLOGY_REBUILD_QUEUE_MAXSIZE = max(2, int(os.getenv("ONTOLOGY_REBUILD_QUEUE_MAXSIZE", "20")))
ONTOLOGY_REBUILD_WORKER_COUNT = max(1, int(os.getenv("ONTOLOGY_REBUILD_WORKER_COUNT", "1")))
ONTOLOGY_REBUILD_JOB_RETENTION_SECONDS = max(300, int(os.getenv("ONTOLOGY_REBUILD_JOB_RETENTION_SECONDS", "86400")))
DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED = _env_bool("DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED", True)
DOCUMENT_UPLOAD_ONTOLOGY_LLM_JOB_ENABLED = _env_bool("DOCUMENT_UPLOAD_ONTOLOGY_LLM_JOB_ENABLED", False)
REPORTED_ANSWER_ONTOLOGY_RECHECK_ENABLED = _env_bool("REPORTED_ANSWER_ONTOLOGY_RECHECK_ENABLED", True)
ONTOLOGY_REPORTED_MAX_CHUNKS = max(1, min(8, int(os.getenv("ONTOLOGY_REPORTED_MAX_CHUNKS", "8"))))
_UPLOAD_OCR_PROGRESS_STAGES = {
    "load_pdf_ocr_model",
    "run_pdf_ocr",
    "fallback_pdf_ocr",
    "merge_pdf_ocr",
    "release_pdf_ocr_worker",
}
RAG_ENGINE_MAX_LOADED_KBS = max(1, int(os.getenv("RAG_ENGINE_MAX_LOADED_KBS", "1")))
RAG_ENGINE_IDLE_TTL_SECONDS = max(60, int(os.getenv("RAG_ENGINE_IDLE_TTL_SECONDS", "900")))
UPLOAD_CHUNK_BYTES = 1024 * 1024
ADMIN_FEEDBACK_MAX_TEXT_CHARS = max(400, int(os.getenv("ADMIN_FEEDBACK_MAX_TEXT_CHARS", "16000")))
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".xlsx", ".pdf", ".hwpx"}
FAST_LANE_UPLOAD_EXTENSIONS = {".txt", ".xlsx", ".hwpx"}
ALLOWED_MIME_BY_EXT = {
    ".txt": {"", "text/plain", "application/octet-stream"},
    ".xlsx": {
        "",
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
        "application/x-zip-compressed",
    },
    ".pdf": {
        "",
        "application/pdf",
        "application/octet-stream",
    },
    ".hwpx": {
        "",
        "application/octet-stream",
        "application/zip",
        "application/x-zip-compressed",
        "application/hwp+zip",
        "application/haansofthwpx",
        "application/x-hwp",
        "application/vnd.hancom.hwp",
        "application/vnd.hancom.hwpx",
        "application/vnd.hancom.hwp+zip",
    },
}
DOC_ROLE_GUIDE = "guide"
DOC_ROLE_CASEBOOK = "casebook"
DOC_ROLE_UNKNOWN = "unknown"
ALLOWED_UPLOAD_DOC_ROLES = {DOC_ROLE_GUIDE, DOC_ROLE_CASEBOOK}
DOC_ROLE_ALIASES = {
    "guide": DOC_ROLE_GUIDE,
    "guideline": DOC_ROLE_GUIDE,
    "manual": DOC_ROLE_GUIDE,
    "지침": DOC_ROLE_GUIDE,
    "지침서": DOC_ROLE_GUIDE,
    "규정": DOC_ROLE_GUIDE,
    "rule": DOC_ROLE_GUIDE,
    "case": DOC_ROLE_CASEBOOK,
    "cases": DOC_ROLE_CASEBOOK,
    "casebook": DOC_ROLE_CASEBOOK,
    "qa": DOC_ROLE_CASEBOOK,
    "q&a": DOC_ROLE_CASEBOOK,
    "faq": DOC_ROLE_CASEBOOK,
    "질답": DOC_ROLE_CASEBOOK,
    "사례": DOC_ROLE_CASEBOOK,
    "사례집": DOC_ROLE_CASEBOOK,
}
GUIDE_QUERY_HINTS = (
    "지침",
    "규정",
    "원칙",
    "기준",
    "정의",
    "절차",
    "작성요령",
    "매뉴얼",
    "원문 규정",
)
CASEBOOK_QUERY_HINTS = (
    "사례",
    "사례집",
    "질답",
    "q&a",
    "qa",
    "faq",
    "예시",
    "예를 들어",
    "이럴때",
    "이럴 때",
    "이런 경우",
    "실제 케이스",
    "케이스",
    "case",
)

OVERVIEW_QUERY_KEYWORDS = (
    "전체 흐름",
    "흐름",
    "전체 과정",
    "과정",
    "프로세스",
    "요약",
    "개요",
    "처음부터",
    "단계별",
    "전체",
    "중요한 내용",
    "중요 내용",
    "핵심 내용",
    "핵심",
    "요점",
    "핵심 정리",
)
SUMMARY_QUERY_HINTS = (
    "요약",
    "정리",
    "핵심",
    "중요한 내용",
    "중요 내용",
    "무슨 내용",
    "무슨내용",
    "어떤 내용",
    "어떤내용",
    "뭐라고",
    "무엇을 말",
)

NUMBER_REF_PATTERN = re.compile(r"(?<!\d)(\d{1,3})\s*번")
DOC_LABEL_PATTERN = re.compile(r"\[DOC\s+(\d+)\]")
ASCII_WORD_PATTERN = re.compile(r"[0-9A-Za-z]+")
KOREAN_CHAR_PATTERN = re.compile(r"[가-힣]")
SYMBOL_PATTERN = re.compile(r"[^\s0-9A-Za-z가-힣]")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
GROUNDING_STOP_TOKENS = {
    "전체",
    "내용",
    "질문",
    "문의",
    "요약",
    "설명",
    "알려줘",
    "해주세요",
    "해줘",
    "무엇",
    "뭐",
    "어떤",
}
CODE_QUERY_HINTS = (
    "부호",
    "코드",
    "분류",
    "항목",
    "몇번",
    "몇 번",
    "코드값",
)
HIGH_RISK_FALLBACK_HINTS = (
    "법",
    "법률",
    "세금",
    "세무",
    "의료",
    "병원",
    "치료",
    "약",
    "복용",
    "투자",
    "주식",
    "대출",
    "계약",
    "판결",
    "판례",
    "규정",
    "정책",
    "안전",
    "보안",
    "개인정보",
    "보험",
)


def _classify_failure_code(error: Any, *, default: str = "runtime_error") -> str:
    if isinstance(error, BaseException):
        exc_name = type(error).__name__.lower()
        raw_text = str(error)
    else:
        exc_name = ""
        raw_text = str(error or "")

    text = raw_text.strip().lower()
    if not text:
        return default

    if "tool_calls_limit" in text or "tool calls limit" in text:
        return "answer_tool_budget_exhausted"
    if "파일 형식이 올바르지" in text or "공간 이름 형식이 올바르지" in text:
        return "upload_validation_fail"
    if ("파일 크기" in text and "초과" in text) or "too large" in text:
        return "upload_size_limit"
    if "queue" in text and "upload" in text:
        return "upload_queue_full"
    if "paddleocr-vl 로더를 불러오지 못했습니다" in text or (
        "paddleocr" in text and ("module" in text or "설치" in text)
    ):
        return "ocr_runtime_missing"
    if "paddleocr-vl 모델 로딩 실패" in text:
        return "ocr_model_load_fail"
    if "paddleocr-vl pdf 인식 실패" in text or ("pdf" in text and "ocr" in text):
        return "ocr_inference_fail"
    if "pymupdf 로더를 불러오지 못했습니다" in text:
        return "pdf_text_runtime_missing"
    if "pymupdf pdf 추출 실패" in text:
        return "pdf_text_extract_fail"
    if "pdf 결과에서 추출 가능한 텍스트를 찾지 못했습니다" in text:
        return "pdf_extract_empty"
    if "embedding_provider=api requires embedding_api_url" in text:
        return "embedding_config_fail"
    if "failed to initialize remote embedding provider" in text:
        return "embedding_provider_fail"
    if "could not load embedding model" in text:
        return "embedding_model_load_fail"
    if "llama-server runtime" in text or "llm model file not found" in text:
        return "llm_runtime_fail"
    if "context" in text and any(token in text for token in ("token", "maximum", "available", "size")):
        return "context_overflow"
    if exc_name == "modulenotfounderror":
        return "runtime_import_missing"
    if exc_name == "filenotfounderror":
        return "file_not_found"
    if exc_name == "permissionerror":
        return "file_permission_fail"
    return default


def _error_json_response(
    *,
    status_code: int,
    message: str,
    failure_code: str,
    filename: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    content: Dict[str, Any] = {
        "status": "error",
        "message": message,
        "failure_code": failure_code,
    }
    if filename:
        content["filename"] = filename
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


def _helper_diag_payload(diag: Optional[HelperRunDiagnostics]) -> Dict[str, Any]:
    if diag is None:
        return {}
    payload: Dict[str, Any] = {
        "helper_status": (diag.status or "").strip() or "ok",
        "fallback_used": bool(diag.fallback_used),
    }
    if getattr(diag, "helper_degraded", False):
        payload["helper_degraded"] = True
    if getattr(diag, "deterministic_parallel_used", False):
        payload["deterministic_parallel_used"] = True
    if getattr(diag, "helper_wait_skipped", False):
        payload["helper_wait_skipped"] = True
    if diag.failure_code:
        payload["failure_code"] = diag.failure_code
    if diag.error_type:
        payload["error_type"] = diag.error_type
    return payload


def _exception_to_helper_diag(error: BaseException, failure_code: str) -> HelperRunDiagnostics:
    return HelperRunDiagnostics(
        status="error",
        failure_code=failure_code,
        error_type=type(error).__name__,
        error_detail=str(error or "").strip(),
        fallback_used=True,
    )


async def _read_request_json_object(
    request: Request,
    *,
    failure_code: str,
) -> tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    try:
        data = await request.json()
    except Exception:
        return None, _error_json_response(
            status_code=400,
            message="JSON 형식 요청이 필요합니다.",
            failure_code=failure_code,
        )
    if not isinstance(data, dict):
        return None, _error_json_response(
            status_code=400,
            message="JSON 형식 요청이 필요합니다.",
            failure_code=failure_code,
        )
    return data, None


def _startup_config_summary() -> Dict[str, Any]:
    llm_ctx_size = int(os.getenv("LLM_CTX_SIZE", str(LLM_CONTEXT_LIMIT)) or LLM_CONTEXT_LIMIT)
    return {
        "python": sys.executable,
        "kb_data_dir": str(KB_DATA_DIR),
        "app_db_path": str(APP_DB_PATH),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local").strip().lower() or "local",
        "embedding_api_url": os.getenv("EMBEDDING_API_URL", "").strip() or "(local)",
        "llm_api_url": LLM_API_URL,
        "llm_ctx_size": llm_ctx_size,
        "llm_context_limit": int(LLM_CONTEXT_LIMIT),
        "llm_context_mismatch": bool(llm_ctx_size != int(LLM_CONTEXT_LIMIT)),
        "llm_max_tokens": int(LLM_MAX_TOKENS),
        "ocr_model_name": (os.getenv("PDF_OCR_MODEL_NAME", "") or "").strip() or "(default)",
        "ocr_device": (os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip() or "cpu",
        "pydantic_ai_provider_kind": ai_service.provider_kind,
        "pydantic_ai_provider_label": ai_service.provider_label,
        "pydantic_ai_history_strategy": ai_service.settings.history_strategy,
        "pydantic_ai_enable_retrieval_tool": bool(ai_service.settings.enable_retrieval_tool),
        "rag_engine_max_loaded_kbs": int(RAG_ENGINE_MAX_LOADED_KBS),
        "rag_engine_idle_ttl_seconds": int(RAG_ENGINE_IDLE_TTL_SECONDS),
        "rag_concept_links_enabled": bool(_env_bool("RAG_CONCEPT_LINKS_ENABLED", True)),
        "rag_concept_chunk_expand_limit": int(os.getenv("RAG_CONCEPT_CHUNK_EXPAND_LIMIT", "64") or 64),
    }


def _print_startup_config_summary():
    print("[STARTUP] Backend config summary:")
    for key, value in _startup_config_summary().items():
        print(f"[STARTUP] {key}={value}")

GREETING_TOKENS = {
    "안녕",
    "안녕하세요",
    "안녕하십니까",
    "반가워",
    "반가워요",
    "반갑습니다",
    "좋은아침",
    "좋은저녁",
    "좋은밤",
    "hi",
    "hello",
    "hey",
    "goodmorning",
    "goodevening",
}

GREETING_POLITE_TOKENS = {
    "요",
    "ㅎㅎ",
    "ㅋㅋ",
    "네",
    "넵",
    "예",
    "헬로",
}

IDENTITY_PATTERNS = (
    r"넌\s*누구",
    r"너는\s*누구",
    r"너\s*누구",
    r"누구야",
    r"누구냐",
    r"정체가?\s*뭐",
    r"무슨\s*모델",
    r"what\s+are\s+you",
    r"who\s+are\s+you",
)
IDENTITY_DOC_REQUEST_HINTS = (
    "알려줘",
    "설명해",
    "요약해",
    "분석해",
    "정리해",
    "문서",
    "파일",
    "규정",
    "절차",
    "업로드",
    "근거",
)

HISTORY_ENABLE_HINTS = (
    "대화기록저장해줘",
    "대화기록남겨줘",
    "대화기록켜줘",
    "대화기록on",
    "대화내역저장해줘",
    "대화내역남겨줘",
    "기록저장해줘",
    "기록남겨줘",
    "기록켜줘",
    "이대화기억해줘",
)

HISTORY_DISABLE_HINTS = (
    "대화기록끄기",
    "대화기록꺼줘",
    "대화기록off",
    "기록저장하지마",
    "기록남기지마",
    "기록중지",
    "기억하지마",
)

HISTORY_CLEAR_HINTS = (
    "대화기록지워줘",
    "대화내역삭제",
    "대화내역지워줘",
    "기록삭제해줘",
    "기록초기화",
    "대화초기화",
)


def _normalize_doc_role(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return DOC_ROLE_UNKNOWN
    return DOC_ROLE_ALIASES.get(raw, DOC_ROLE_UNKNOWN)


def _doc_role_label(doc_role: str) -> str:
    role = _normalize_doc_role(doc_role)
    if role == DOC_ROLE_GUIDE:
        return "지침서"
    if role == DOC_ROLE_CASEBOOK:
        return "사례집(Q&A)"
    return "미분류"


def _infer_doc_role_from_filename(filename: str, ext: str) -> str:
    lowered = (filename or "").strip().lower()
    if any(tok in lowered for tok in ("사례", "질답", "faq", "q&a", "qa", "case")):
        return DOC_ROLE_CASEBOOK
    if any(tok in lowered for tok in ("지침", "규정", "manual", "guide")):
        return DOC_ROLE_GUIDE
    if ext == ".xlsx":
        return DOC_ROLE_CASEBOOK
    if ext == ".txt":
        return DOC_ROLE_GUIDE
    if ext == ".pdf":
        return DOC_ROLE_GUIDE
    return DOC_ROLE_UNKNOWN


def _resolve_upload_doc_role(requested: str, filename: str, ext: str) -> str:
    raw = (requested or "").strip().lower()
    if raw in {"", "auto", "자동"}:
        return _infer_doc_role_from_filename(filename=filename, ext=ext)
    normalized = _normalize_doc_role(raw)
    if normalized in ALLOWED_UPLOAD_DOC_ROLES:
        return normalized
    raise ValueError("문서 유형은 guide(지침서) 또는 casebook(사례집)만 허용됩니다.")


def _classify_query_doc_intent(query: str) -> str:
    text = (query or "").strip().lower()
    compact = re.sub(r"\s+", "", text)
    guide_score = 0
    case_score = 0

    for hint in GUIDE_QUERY_HINTS:
        hint_norm = hint.lower()
        if hint_norm in text or hint_norm.replace(" ", "") in compact:
            guide_score += 1
    for hint in CASEBOOK_QUERY_HINTS:
        hint_norm = hint.lower()
        if hint_norm in text or hint_norm.replace(" ", "") in compact:
            case_score += 1

    if re.search(r"했는데|인데|라면|이면|할 때|경우", text):
        case_score += 1
    if re.search(r"정의|원칙|기준|규정", text):
        guide_score += 1

    if case_score >= 2 and guide_score == 0:
        return DOC_ROLE_CASEBOOK
    if guide_score >= 2 and case_score == 0:
        return DOC_ROLE_GUIDE
    if case_score >= guide_score + 2:
        return DOC_ROLE_CASEBOOK
    if guide_score >= case_score + 2:
        return DOC_ROLE_GUIDE
    return "mixed"


def _build_role_search_plan(query_intent: str) -> List[Optional[List[str]]]:
    if not RAG_ROLE_ROUTING_ENABLED:
        return [None]
    if query_intent == DOC_ROLE_CASEBOOK:
        if RAG_ROLE_ROUTING_STRICT:
            return [[DOC_ROLE_CASEBOOK]]
        return [[DOC_ROLE_CASEBOOK], None]
    if query_intent == DOC_ROLE_GUIDE:
        if RAG_ROLE_ROUTING_STRICT:
            return [[DOC_ROLE_GUIDE]]
        return [[DOC_ROLE_GUIDE], None]
    return [None]


def _role_filter_label(doc_roles: Optional[List[str]]) -> str:
    if not doc_roles:
        return "all"
    normalized = [r for r in (_normalize_doc_role(x) for x in doc_roles) if r in ALLOWED_UPLOAD_DOC_ROLES]
    if not normalized:
        return "all"
    return ",".join(dict.fromkeys(normalized))


def _no_evidence_response(reason: str = "") -> str:
    _ = reason
    unknown = "올려주신 문서에서 질문에 직접 대응하는 근거를 찾지 못했습니다."
    return (
        "죄송하지만, 문서 근거가 부족해 단정적으로 안내드리기 어렵습니다. "
        f"{unknown} 확인 가능한 원문이 있으면 그 범위 안에서 다시 정리해 드리겠습니다."
    )


def _estimate_tokens(text: str) -> int:
    """
    Heuristic token estimator for mixed Korean/English text.
    Used only for context-budgeting (not exact tokenization).
    """
    if not text:
        return 0
    ascii_words = len(ASCII_WORD_PATTERN.findall(text))
    ko_chars = len(KOREAN_CHAR_PATTERN.findall(text))
    symbols = len(SYMBOL_PATTERN.findall(text))
    # Heuristic tuned for mixed Korean/English prompts.
    est = (ascii_words * 1.1) + (ko_chars * 0.65) + (symbols * 0.3)
    return max(1, int(est))


def _normalize_tokens(text: str) -> List[str]:
    raw = TOKEN_PATTERN.findall(text.lower())
    tokens: List[str] = []
    for t in raw:
        t2 = t.replace(" ", "")
        if t2:
            tokens.append(t2)
    return tokens


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _primary_answer_span(text: str) -> str:
    for line in (text or "").splitlines():
        cleaned = line.strip(" -*\t")
        if cleaned:
            return cleaned
    return ""


def _is_grounded_abstention(text: str) -> bool:
    return is_grounded_abstention_text(text)


def _contains_outside_document_claim(text: str) -> bool:
    compact = _compact_text(text)
    outside_markers = (
        "문서밖참고",
        "문서밖정보",
        "문서밖",
        "외부지식",
        "관련기관에문의",
        "기관에문의",
    )
    return any(marker in compact for marker in outside_markers)


def _sanitize_outside_document_claims(answer_text: str) -> str:
    raw = canonicalize_doc_citations(answer_text or "").strip()
    if not raw or not _contains_outside_document_claim(raw):
        return raw
    kept_blocks: List[str] = []
    for block in re.split(r"\n{2,}", raw):
        cleaned = block.strip()
        if not cleaned:
            continue
        if _contains_outside_document_claim(cleaned):
            continue
        kept_blocks.append(cleaned)
    sanitized = "\n\n".join(kept_blocks).strip()
    return sanitized or raw


def _query_requires_code_hint(query: str) -> bool:
    compact = _compact_text(query)
    return any(h in compact for h in CODE_QUERY_HINTS)


def _has_code_like_token(text: str) -> bool:
    return bool(re.search(r"\(\d{2,4}\)|\b\d{2,4}\b", text or ""))


def _is_question_echo_answer(query: str, answer_line: str) -> bool:
    q = _normalize_space(query)
    a = _normalize_space(answer_line)
    if not q or not a:
        return False
    if a.startswith(q):
        tail = a[len(q) :].strip(" .。!?？！:：-–—")
        return len(tail) <= 8
    if len(q) >= 14 and q[: min(40, len(q))] in a[: max(60, min(len(a), len(q) + 20))]:
        return len(a) <= len(q) + 12

    q_tokens = {t for t in _normalize_tokens(q) if len(t) >= 2 and t not in GROUNDING_STOP_TOKENS}
    a_tokens = {t for t in _normalize_tokens(a) if len(t) >= 2 and t not in GROUNDING_STOP_TOKENS}
    if not q_tokens or not a_tokens:
        return False

    shared = q_tokens & a_tokens
    overlap = len(shared) / max(1, min(len(q_tokens), len(a_tokens)))
    novel_tokens = len(a_tokens - q_tokens)
    return overlap >= 0.8 and novel_tokens <= 2


def _response_quality_issue(query: str, response_text: str, metrics: Dict[str, Any]) -> str:
    raw = canonicalize_doc_citations((response_text or "").strip())
    if not raw:
        return "empty"

    if _contains_outside_document_claim(raw):
        return "outside_document_claim"

    if _is_grounded_abstention(raw):
        if should_treat_abstention_as_quality_issue(query, metrics):
            return "abstained_with_evidence"
        return ""

    if "[DOC" not in raw:
        return "missing_doc_citation"
    lead = _primary_answer_span(raw)
    if _is_question_echo_answer(query, lead or raw):
        return "question_echo"
    if len(raw) < LLM_MIN_ANSWER_LINE_CHARS:
        return "too_short"
    if _query_requires_code_hint(query) and not _has_code_like_token(raw):
        return "missing_code"
    return ""


def _quality_retry_hint(issue: str) -> str:
    if issue == "missing_doc_citation":
        return "핵심 판단마다 [DOC i]를 직접 붙이고, 필요하면 마지막에 짧게 근거 위치를 정리해라."
    if issue == "question_echo":
        return "질문 문장을 반복하지 말고 결론부터 작성하고, 조건/분기를 포함해라."
    if issue == "too_short":
        return "답이 너무 짧다. 결론과 핵심 근거를 2~4문장 정도로 보강해라."
    if issue == "missing_code":
        return "질문이 코드/부호/분류를 묻는다. 문서에 있는 코드(숫자)를 반드시 포함해라."
    if issue == "abstained_with_evidence":
        return "CONTEXT에 근거가 있다. 근거 부족으로 회피하지 말고 확인 가능한 범위를 답해라."
    if issue == "outside_document_claim":
        return "문서 밖 참고 정보, 일반적 조언, 기관 문의 같은 외부 추정을 빼고 현재 문서에서 확인되는 사실과 확인되지 않는 부분만 답해라."
    if issue == "empty":
        return "응답이 비었다. 한국어로 결론과 근거를 다시 작성해라."
    return "근거 연결과 답변 품질을 개선해 다시 작성해라."


def _is_pure_greeting(text: str) -> bool:
    tokens = _normalize_tokens(text)
    if not tokens:
        return False
    if len(tokens) > 8:
        return False
    for tok in tokens:
        compact = tok.replace("!", "").replace("?", "")
        if compact in GREETING_TOKENS or compact in GREETING_POLITE_TOKENS:
            continue
        if compact.startswith("안녕") or compact.startswith("반가"):
            continue
        return False
    return True


def _is_greeting_with_extra_intent(text: str) -> bool:
    tokens = _normalize_tokens(text)
    if not tokens:
        return False
    has_greeting = False
    has_non_greeting = False
    for tok in tokens:
        compact = tok.replace("!", "").replace("?", "")
        is_greeting = (
            compact in GREETING_TOKENS
            or compact in GREETING_POLITE_TOKENS
            or compact.startswith("안녕")
            or compact.startswith("반가")
        )
        if is_greeting:
            has_greeting = True
        else:
            has_non_greeting = True
    if not has_greeting:
        return False
    if "?" in text or "해줘" in text or "알려" in text or "설명" in text:
        return True
    return has_non_greeting


def _is_identity_question(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw or len(raw) > 80:
        return False
    return any(re.search(pat, raw) for pat in IDENTITY_PATTERNS)


def _is_identity_with_extra_intent(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not _is_identity_question(raw):
        return False
    # Mixed intent if identity question also asks for document/task handling.
    if any(hint in raw for hint in IDENTITY_DOC_REQUEST_HINTS):
        # Allow plain identity phrasings such as "너 누구야 알려줘".
        if raw.replace(" ", "") in {"너누구야알려줘", "넌누구야알려줘", "너는누구야알려줘"}:
            return False
        return True
    # Multiple question marks / conjunctions often indicate combined intent.
    return ("그리고" in raw) or (raw.count("?") >= 2)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _contains_any_compact(text: str, hints: Any) -> bool:
    compact = _compact_text(text)
    return any(h in compact for h in hints)


def _allow_general_knowledge_fallback(text: str, *, kb_has_docs: bool) -> bool:
    return should_allow_general_knowledge_fallback_by_policy(text, kb_has_docs=kb_has_docs)


def _is_summary_request(text: str) -> bool:
    return _contains_any_compact(text, SUMMARY_QUERY_HINTS)


def _is_history_enable_request(text: str) -> bool:
    return _contains_any_compact(text, HISTORY_ENABLE_HINTS)


def _is_history_disable_request(text: str) -> bool:
    return _contains_any_compact(text, HISTORY_DISABLE_HINTS)


def _is_history_clear_request(text: str) -> bool:
    return _contains_any_compact(text, HISTORY_CLEAR_HINTS)


def _identity_text() -> str:
    return (
        "안녕하세요. 저는 CompassLM입니다. 사용자가 올린 TXT/XLSX 문서를 읽고, "
        "질문과 가장 관련 있는 내용을 찾아 안내드리는 문서 도우미입니다. "
        "문서에 없는 내용은 추측하지 않고, 확인되는 범위에서만 친절하게 설명해 드리겠습니다."
    )


async def _greeting_text(user_message: str) -> tuple[str, HelperRunDiagnostics]:
    return await ai_service.greeting_text_diagnostic(
        user_message,
        timeout=min(LLM_REQUEST_TIMEOUT, 20),
    )


async def _casual_chat_text(
    user_message: str,
    *,
    recent_history: str = "",
) -> tuple[str, HelperRunDiagnostics]:
    return await ai_service.casual_chat_diagnostic(
        user_message,
        recent_history=recent_history,
        timeout=min(LLM_REQUEST_TIMEOUT, 25),
    )


def _append_upload_nudge(answer_text: str) -> str:
    base = (answer_text or "").strip()
    nudge = "업무 관련 문서를 올려 주시면 그 기준으로 더 정확하게 도와드릴 수 있습니다."
    if not base:
        return nudge
    if nudge in base:
        return base
    return f"{base}\n\n{nudge}"


def _live_info_limit_response(user_message: str) -> str:
    subject = (user_message or "").strip() or "실시간 정보"
    return (
        "죄송하지만, "
        f"[{subject}]처럼 실시간으로 확인이 필요한 내용은 "
        "이 환경에서 인터넷을 직접 조회할 수 없어 정확하게 안내드리기 어렵습니다. "
        "업무 관련 문서를 올려 주시거나, 일반적인 설명이 필요하시면 그 범위에서 도와드리겠습니다."
    )


async def _llm_followup_rewrite(
    user_message: str,
    recent_history: str,
) -> tuple[FollowupAnalysis, HelperRunDiagnostics]:
    if not (recent_history or "").strip():
        return FollowupAnalysis(), HelperRunDiagnostics(status="disabled")

    return await ai_service.followup_rewrite_diagnostic(
        user_message,
        recent_history=recent_history,
        max_tokens=min(180, RAG_LLM_HELPER_MAX_TOKENS),
        timeout=min(LLM_REQUEST_TIMEOUT, RAG_LLM_HELPER_TIMEOUT),
    )


def _single_chunk_stream(text: str):
    yield text


def _matches_number_anchor(text: str, n: int) -> bool:
    if not text:
        return False
    pat = [
        rf"(^|\n)\s*#{0,6}\s*{n}\s*[\.\)]",
        rf"(^|\n)\s*{n}\s*[\.\)]",
        rf"\b{n}\s*번\b",
        rf"제\s*{n}\s*(장|절|항|조)",
        rf"\b{n}\s*단계\b",
        rf"\bresult\s*{n}\b",
        rf"\[{n}\]",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pat)


def _apply_number_reference_guard(
    results: List[Dict[str, Any]],
    number_refs: List[int],
    keep_n: int,
) -> tuple[List[Dict[str, Any]], bool]:
    """
    Re-rank / filter results to honor numbered references in user question.
    Returns (guarded_results, has_anchor_match).
    """
    if not number_refs:
        return results[:keep_n], True

    anchored: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for r in results:
        txt = r.get("text", "") or ""
        if any(_matches_number_anchor(txt, n) for n in number_refs):
            anchored.append(r)
        else:
            others.append(r)

    if not anchored:
        return results[:keep_n], False

    if RAG_NUMBER_REF_STRICT:
        return anchored[:keep_n], True
    return (anchored + others)[:keep_n], True


def _expand_query_with_number_refs(query: str) -> str:
    """
    Expand numeric references (e.g., '1번') to improve retrieval for numbered sections.
    """
    refs = NUMBER_REF_PATTERN.findall(query)
    if not refs:
        return query

    extra_terms: List[str] = []
    for n in refs[:3]:
        extra_terms.extend(
            [
                f"{n}번",
                f"{n}.",
                f"{n} 항목",
                f"{n} 단계",
                f"제{n}항",
                f"section {n}",
                f"result {n}",
            ]
        )

    # Keep insertion order while removing duplicates.
    deduped = list(dict.fromkeys(extra_terms))
    return f"{query} {' '.join(deduped)}"


def _dedupe_text_items(values: List[str], limit: int = 8) -> List[str]:
    items: List[str] = []
    for raw in values:
        cleaned = " ".join((raw or "").strip().split())
        if not cleaned:
            continue
        if cleaned in items:
            continue
        items.append(cleaned)
        if len(items) >= max(1, limit):
            break
    return items


def _result_evidence_fingerprint(row: Dict[str, Any]) -> str:
    source_path = (row.get("source_path", "") or "").strip()
    source_ref = (row.get("source_ref", "") or row.get("source_display", "") or "").strip()
    page = str(row.get("page_no", "") or row.get("page", "") or "").strip()
    section = (row.get("section", "") or "").strip()
    if not page:
        match = re.search(r"PDF page\s+(\d+)", section or source_ref, re.IGNORECASE)
        if match:
            page = match.group(1)
    return "|".join(
        [
            source_path or source_ref,
            page,
            section,
            (row.get("sheet", "") or "").strip(),
            str(int(row.get("row", 0) or 0)),
            str(int(row.get("row_end", 0) or 0)),
            str(int(row.get("line_start", 0) or 0)),
            str(int(row.get("line_end", 0) or 0)),
            trim_preview((row.get("text", "") or "").strip(), 120),
        ]
    )


def _result_dedupe_key(row: Dict[str, Any]) -> str:
    fingerprint = _result_evidence_fingerprint(row)
    if fingerprint.strip("|"):
        return f"evidence:{fingerprint}"
    chunk_id = int(row.get("id", 0) or 0)
    if chunk_id > 0:
        return f"id:{chunk_id}"
    return "empty"


def _merge_search_candidates(candidate_groups: List[tuple[str, List[Dict[str, Any]]]], limit: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for query_variant, rows in candidate_groups:
        for row in rows:
            key = _result_dedupe_key(row)
            existing = merged.get(key)
            if existing is None:
                item = dict(row)
                item["search_variants"] = [query_variant]
                item["variant_hits"] = 1
                merged[key] = item
                continue

            variants = list(existing.get("search_variants", []))
            if query_variant not in variants:
                variants.append(query_variant)
            best_score = float(existing.get("score", 0.0) or 0.0)
            row_score = float(row.get("score", 0.0) or 0.0)
            replacement = dict(existing)
            if row_score > best_score:
                replacement.update(dict(row))
            replacement["search_variants"] = variants
            replacement["variant_hits"] = len(variants)
            merged[key] = replacement

    rows = list(merged.values())
    rows.sort(
        key=lambda x: (
            int(x.get("variant_hits", 0) or 0),
            float(x.get("score", 0.0) or 0.0),
            int(x.get("uploaded_at", x.get("source_updated_at", 0)) or 0),
            int(x.get("id", 0) or 0),
        ),
        reverse=True,
    )
    return rows[: max(1, int(limit))]


def _collect_search_candidates(
    rag: RAGEngine,
    search_variants: List[str],
    *,
    top_k: int,
    index_name: str,
    doc_roles: Optional[List[str]],
) -> List[Dict[str, Any]]:
    variants = _dedupe_text_items(search_variants, limit=6)
    if not variants:
        return []

    per_query_k = max(8, min(int(top_k), max(8, int(top_k) // max(1, len(variants)))))
    candidate_groups: List[tuple[str, List[Dict[str, Any]]]] = []
    for variant in variants:
        rows = rag.search(
            variant,
            top_k=per_query_k,
            index_name=index_name,
            doc_roles=doc_roles,
        )
        if rows:
            candidate_groups.append((variant, rows))

    if not candidate_groups:
        return []
    return _merge_search_candidates(candidate_groups, limit=top_k)


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text or "")
    except Exception:
        return {}


async def _llm_analyze_question(user_message: str) -> tuple[QuestionAnalysis, HelperRunDiagnostics]:
    if not RAG_LLM_QUERY_ANALYZE_ENABLED:
        return QuestionAnalysis(), HelperRunDiagnostics(status="disabled")

    return await ai_service.analyze_question_diagnostic(
        user_message,
        max_tokens=min(200, RAG_LLM_HELPER_MAX_TOKENS),
        timeout=min(LLM_REQUEST_TIMEOUT, RAG_LLM_HELPER_TIMEOUT),
    )


async def _llm_expand_query(user_message: str) -> tuple[str, HelperRunDiagnostics]:
    if not RAG_LLM_QUERY_EXPAND_ENABLED:
        return "", HelperRunDiagnostics(status="disabled")

    expanded, diag = await ai_service.expand_query_diagnostic(
        user_message,
        max_tokens=RAG_LLM_HELPER_MAX_TOKENS,
        timeout=min(LLM_REQUEST_TIMEOUT, RAG_LLM_HELPER_TIMEOUT),
    )
    if not expanded:
        return "", diag
    return expanded.replace("\n", " ").strip(), diag


async def _llm_rerank_results(
    user_message: str,
    candidates: List[Dict[str, Any]],
    keep_n: int,
) -> tuple[List[Dict[str, Any]], HelperRunDiagnostics, Dict[str, Any]]:
    if not RAG_LLM_RERANK_ENABLED or not candidates:
        return candidates[:keep_n], HelperRunDiagnostics(status="disabled"), {
            "original_count": len(candidates),
            "selected_count": min(len(candidates), keep_n),
            "trimmed_count": 0,
            "line_char_cap": 0,
        }

    lines = []
    for i, c in enumerate(candidates, start=1):
        snippet = (c.get("text", "") or "").replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        uploaded_at = int(c.get("uploaded_at", c.get("source_updated_at", 0)) or 0)
        uploaded_label = _format_timestamp(uploaded_at)
        is_normalized = int(c.get("is_normalized", 0) or 0)
        location = ""
        if c.get("sheet"):
            location = f"{c.get('sheet')} Row {c.get('row', 0)}"
        elif c.get("line_start"):
            location = f"Lines {c.get('line_start', 0)}-{c.get('line_end', 0)}"
        lines.append(
            f"{i}) file={c.get('source_display', c.get('source_path', ''))} | "
            f"loc={location or '-'} | uploaded={uploaded_label} | "
            f"score={float(c.get('score', 0.0)):.3f} | norm={is_normalized} | "
            f"code={int(c.get('code_match', 0) or 0)} | text={snippet}"
        )

    trimmed_lines, budget_meta = trim_rerank_candidate_lines_to_budget(
        user_message=user_message,
        candidate_lines=lines,
        keep_n=keep_n,
        llm_context_limit=LLM_CONTEXT_LIMIT,
        helper_max_tokens=min(120, RAG_LLM_HELPER_MAX_TOKENS),
        prompt_overhead_tokens=LLM_PROMPT_OVERHEAD_TOKENS,
        safety_margin=LLM_CONTEXT_SAFETY_MARGIN,
    )
    if not trimmed_lines:
        return candidates[:keep_n], HelperRunDiagnostics(
            status="disabled",
            failure_code="rerank_budget_exhausted",
            error_detail="rerank prompt budget exhausted before any candidate could fit",
        ), budget_meta

    content, diag = await ai_service.rerank_candidates_diagnostic(
        user_message=user_message,
        candidate_lines=trimmed_lines,
        max_tokens=min(120, RAG_LLM_HELPER_MAX_TOKENS),
        timeout=min(LLM_REQUEST_TIMEOUT, RAG_LLM_HELPER_TIMEOUT),
    )
    if not content:
        return candidates[:keep_n], HelperRunDiagnostics(
            status="fallback",
            failure_code=diag.failure_code or "rerank_deterministic_fallback",
            error_type=diag.error_type,
            error_detail="LLM rerank failed; kept retrieval order.",
            fallback_used=True,
        ), budget_meta

    picks = re.findall(r"\d+", content)
    order: List[int] = []
    seen = set()
    selected_count = min(len(trimmed_lines), len(candidates))
    for p in picks:
        idx = int(p)
        if 1 <= idx <= selected_count and idx not in seen:
            order.append(idx - 1)
            seen.add(idx)

    reranked = [candidates[i] for i in order]
    selected_index_set = set(order)
    for i in range(selected_count):
        if i not in selected_index_set:
            reranked.append(candidates[i])
    if selected_count < len(candidates):
        reranked.extend(candidates[selected_count:])
    return reranked[:keep_n], diag, budget_meta



def _build_source_hints(
    rag: Optional[RAGEngine],
    results: List[Dict[str, Any]],
    limit: int = 8,
    start_index: int = 1,
) -> str:
    if not rag or not results:
        return ""
    lines: List[str] = []
    for i, r in enumerate(results[: max(1, limit)], start=1):
        source_label = rag.format_source_ref(r)
        lines.append(f"[DOC {start_index + i - 1}] {source_label}")
    return "\n".join(lines)


def _max_doc_number(text: str) -> int:
    matches = DOC_LABEL_PATTERN.findall(text or "")
    if not matches:
        return 0
    try:
        return max(int(v) for v in matches)
    except Exception:
        return 0


def _renumber_doc_labels(text: str, start_index: int) -> str:
    if start_index <= 1:
        return text or ""

    def _replace(match: re.Match[str]) -> str:
        return f"[DOC {start_index + int(match.group(1)) - 1}]"

    return DOC_LABEL_PATTERN.sub(_replace, text or "")


def _prefer_source_chunks(results: List[Dict[str, Any]], keep_n: int, max_normalized: int) -> List[Dict[str, Any]]:
    if not results:
        return []
    raw = [r for r in results if int(r.get("is_normalized", 0) or 0) == 0]
    normalized = [r for r in results if int(r.get("is_normalized", 0) or 0) == 1]

    if max_normalized <= 0:
        preferred = list(raw)
    else:
        preferred = list(raw) + normalized[:max_normalized]

    if not preferred:
        preferred = list(results)

    seen_ids = {r.get("id") for r in preferred}
    if len(preferred) < keep_n:
        for r in results:
            rid = r.get("id")
            if rid in seen_ids:
                continue
            preferred.append(r)
            seen_ids.add(rid)
            if len(preferred) >= keep_n:
                break

    return preferred[:keep_n]


def _build_retrieval_meta_block(
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    query_doc_intent: str = "mixed",
    retrieval_role_filter: Optional[List[str]] = None,
) -> str:
    top1 = float(metrics.get("top1", 0.0))
    coverage = float(metrics.get("coverage", 0.0))
    unique_sources = int(metrics.get("unique_sources", 0) or 0)
    has_conflict = bool(metrics.get("has_conflict", False))
    latest_uploaded = int(metrics.get("latest_uploaded_at", 0) or 0)
    doc_roles = metrics.get("doc_roles", [])
    doc_roles_txt = ", ".join(doc_roles) if isinstance(doc_roles, list) and doc_roles else "-"
    lines = [
        "RETRIEVAL_META:",
        f"- top1={top1:.3f}",
        f"- coverage={coverage:.3f}",
        f"- unique_sources={unique_sources}",
        f"- query_doc_intent={query_doc_intent}",
        f"- retrieval_role_filter={_role_filter_label(retrieval_role_filter)}",
        f"- evidence_doc_roles={doc_roles_txt}",
        f"- has_conflict={'yes' if has_conflict else 'no'}",
        f"- latest_upload={_format_timestamp(latest_uploaded)}",
        f"- result_count={len(results)}",
    ]
    return "\n".join(lines) + "\n"


def _build_trace_results(rag: Optional[RAGEngine], results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results[: max(1, limit)]:
        text = (r.get("text", "") or "").replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220] + "..."
        rows.append(
            {
                "source": rag.format_source_ref(r) if rag else (r.get("source_path", "") or ""),
                "score": round(float(r.get("score", 0.0) or 0.0), 6),
                "concept_score": round(float(r.get("concept_score", 0.0) or 0.0), 6),
                "semantic_score": round(float(r.get("semantic_score", 0.0) or 0.0), 6),
                "lexical_score": round(float(r.get("lexical_score", 0.0) or 0.0), 6),
                "literal_score": round(float(r.get("literal_score", 0.0) or 0.0), 6),
                "recency_score": round(float(r.get("recency_score", 0.0) or 0.0), 6),
                "is_normalized": int(r.get("is_normalized", 0) or 0),
                "doc_role": _normalize_doc_role(str(r.get("doc_role", ""))),
                "uploaded_at": int(r.get("uploaded_at", r.get("source_updated_at", 0)) or 0),
                "matched_concepts": list(r.get("matched_concepts", []) or [])[:4],
                "snippet": text,
            }
        )
    return rows


def _summarize_matched_concepts(results: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    concepts: List[str] = []
    for row in results:
        for concept in list(row.get("matched_concepts", []) or []):
            text = str(concept or "").strip()
            if not text or text in concepts:
                continue
            concepts.append(text)
            if len(concepts) >= max(1, limit):
                return concepts
    return concepts


def _trim_to_max_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n...[truncated]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def _merge_context_with_cap(base: str, extra: str, max_chars: int) -> str:
    if not extra:
        return _trim_to_max_chars(base, max_chars)
    merged = f"{base}\n{extra}" if base else extra
    return _trim_to_max_chars(merged, max_chars)


def _adaptive_per_result_context_cap(max_chars: int, result_count: int) -> int:
    if max_chars <= 0:
        return 0
    if result_count <= 0:
        return min(max_chars, RAG_CONTEXT_PER_RESULT_MAX_CHARS)
    divisor = max(2, min(8, result_count + 1))
    adaptive = max(400, max_chars // divisor)
    return min(max_chars, max(RAG_CONTEXT_PER_RESULT_MAX_CHARS, adaptive))


def _apply_diversity_filter(
    results: List[Dict[str, Any]],
    keep_n: int,
    max_per_section: int,
    max_per_file: int,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    filtered: List[Dict[str, Any]] = []
    section_counter: Counter = Counter()
    file_counter: Counter = Counter()

    for res in results:
        source = (res.get("source_path", "") or "").strip()
        section = (res.get("section", "") or res.get("sheet", "") or "").strip().lower()
        section_key = (source, section or "_")
        file_key = source or "_"

        if section_counter[section_key] >= max_per_section:
            continue
        if file_counter[file_key] >= max_per_file:
            continue

        filtered.append(res)
        section_counter[section_key] += 1
        file_counter[file_key] += 1
        if len(filtered) >= keep_n:
            return filtered

    if len(filtered) < keep_n:
        seen_ids = {r.get("id") for r in filtered}
        for res in results:
            rid = res.get("id")
            if rid in seen_ids:
                continue
            filtered.append(res)
            seen_ids.add(rid)
            if len(filtered) >= keep_n:
                break

    return filtered[:keep_n]


def _has_explicit_evidence(query: str, results: List[Dict[str, Any]]) -> bool:
    if not results:
        return False
    query_tokens = [t for t in _normalize_tokens(query) if len(t) >= 2]
    if not query_tokens:
        return True
    keywords = [t for t in query_tokens if t not in GROUNDING_STOP_TOKENS]
    if not keywords:
        return True
    keyword_set = set(keywords)
    joined = "\n".join((r.get("text", "") or "").lower() for r in results[:8])
    hit = sum(1 for k in keyword_set if k in joined)
    required = 1 if len(keyword_set) <= 4 else 2
    return hit >= required


def _has_conflicting_signals(query: str, results: List[Dict[str, Any]]) -> bool:
    if not RAG_GROUNDING_CONFLICT_CHECK_ENABLED:
        return False
    if len(results) < 2:
        return False
    neg_markers = ("아니다", "않", "불가", "금지", "없음", "없다")
    pos_markers = ("가능", "허용", "된다", "할 수", "있다")
    query_tokens = {t for t in _normalize_tokens(query) if len(t) >= 2 and t not in GROUNDING_STOP_TOKENS}

    sample = results[:6]
    for i in range(len(sample)):
        a = sample[i]
        text_a = (a.get("text", "") or "").lower()
        source_a = (a.get("source_path", "") or "").strip()
        if not text_a:
            continue
        for j in range(i + 1, len(sample)):
            b = sample[j]
            source_b = (b.get("source_path", "") or "").strip()
            if source_a and source_b and source_a == source_b:
                continue
            text_b = (b.get("text", "") or "").lower()
            if not text_b:
                continue
            tokens_a = {t for t in _normalize_tokens(text_a) if len(t) >= 2}
            tokens_b = {t for t in _normalize_tokens(text_b) if len(t) >= 2}
            if not tokens_a or not tokens_b:
                continue
            if query_tokens:
                if not ((tokens_a & query_tokens) and (tokens_b & query_tokens)):
                    continue
            shared = tokens_a & tokens_b
            if query_tokens:
                shared = shared & query_tokens
            if not shared:
                continue
            a_neg = any(m in text_a for m in neg_markers)
            a_pos = any(m in text_a for m in pos_markers)
            b_neg = any(m in text_b for m in neg_markers)
            b_pos = any(m in text_b for m in pos_markers)
            if (a_neg and b_pos) or (a_pos and b_neg):
                return True
    return False


def _has_numeric_grounding(query: str, results: List[Dict[str, Any]]) -> bool:
    if not is_numeric_evidence_query(query):
        return True
    evidence_text = "\n".join((row.get("text", "") or "") for row in results[:8])
    return bool(extract_numeric_signatures(evidence_text))


def _passes_grounding_gate(query: str, metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
    if not RAG_GROUNDING_GATE_ENABLED:
        return True
    if not results:
        return False

    top1 = float(metrics.get("top1", 0.0))
    coverage = float(metrics.get("coverage", 0.0))
    keyword_hits = int(metrics.get("keyword_hits", 0) or 0)
    keyword_total = int(metrics.get("keyword_total", 0) or 0)
    explicit_evidence = _has_explicit_evidence(query, results)
    has_conflict = _has_conflicting_signals(query, results)
    metrics["has_conflict"] = has_conflict

    if not explicit_evidence:
        return False
    if not _has_numeric_grounding(query, results):
        return False

    # Keep conflict cases when there is enough evidence so the model can explain
    # "충돌 + 최신 근거 우선" instead of hard-abstaining.
    if has_conflict:
        if top1 < RAG_GROUNDING_TOP1_SOFT_MIN:
            return False
        if coverage < RAG_GROUNDING_COVERAGE_SOFT_MIN:
            return False
        return True

    strict_pass = (top1 >= RAG_GROUNDING_TOP1_MIN) and (coverage >= RAG_GROUNDING_COVERAGE_MIN)
    if strict_pass:
        return True

    if not RAG_GROUNDING_SOFTEN_ENABLED:
        return False
    if top1 < RAG_GROUNDING_TOP1_SOFT_MIN:
        return False
    if coverage < RAG_GROUNDING_COVERAGE_SOFT_MIN:
        return False
    if keyword_total > 0 and keyword_hits < min(keyword_total, RAG_GROUNDING_MIN_KEYWORD_HITS):
        return False

    return True



# RAG Instances Cache
rag_registry = KBEngineRegistry(
    max_loaded_kbs=RAG_ENGINE_MAX_LOADED_KBS,
    idle_ttl_seconds=RAG_ENGINE_IDLE_TTL_SECONDS,
)
upload_jobs: Dict[str, Dict[str, Any]] = {}
upload_jobs_lock = threading.Lock()
upload_jobs_condition = threading.Condition(upload_jobs_lock)
_upload_job_store_last_prune_at = 0
ocr_jobs: Dict[str, Dict[str, Any]] = {}
ocr_jobs_lock = threading.Lock()
ocr_jobs_condition = threading.Condition(ocr_jobs_lock)
ontology_rebuild_jobs: Dict[str, Dict[str, Any]] = {}
ontology_rebuild_jobs_lock = threading.Lock()
ontology_rebuild_jobs_condition = threading.Condition(ontology_rebuild_jobs_lock)
reported_answer_ontology_lock = threading.Lock()
upload_pdf_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=UPLOAD_QUEUE_MAXSIZE)
upload_fast_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=UPLOAD_QUEUE_MAXSIZE)
ocr_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=PDF_BACKGROUND_OCR_QUEUE_MAXSIZE)
ontology_rebuild_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=ONTOLOGY_REBUILD_QUEUE_MAXSIZE)
upload_workers: List[threading.Thread] = []
ocr_workers: List[threading.Thread] = []
ontology_rebuild_workers: List[threading.Thread] = []
upload_workers_started = False
upload_workers_lock = threading.Lock()
recovered_upload_job_ids: set[str] = set()
upload_shutdown_event = threading.Event()
ocr_shutdown_event = threading.Event()
ontology_rebuild_shutdown_event = threading.Event()
admin_feedback_log_lock = threading.Lock()
rag_trace_log_lock = threading.Lock()
auth_store = AuthStore(str(APP_DB_PATH))
upload_job_store = UploadJobStore(str(APP_DB_PATH))
chat_store = ChatStore(
    str(APP_DB_PATH),
    history_limit=CHAT_HISTORY_LIMIT,
    agent_run_limit=CHAT_AGENT_RUN_LIMIT,
)
ai_service = PydanticAIService()


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _resolve_session_id(request: Request) -> str:
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid and re.fullmatch(r"[0-9a-f]{32}", sid):
        return sid
    return _new_session_id()


def _ensure_chat_session(session_id: str) -> Dict[str, Any]:
    return chat_store.ensure_session(session_id)


def _set_history_enabled(session_id: str, enabled: bool):
    chat_store.set_history_enabled(session_id, enabled)


def _is_history_enabled(session_id: str) -> bool:
    return chat_store.is_history_enabled(session_id)


def _clear_chat_history(session_id: str, kb_name: Optional[str] = None, *, user_id: str = ""):
    chat_store.clear_history(session_id, kb_name=kb_name, user_id=user_id)


def _append_chat_message(session_id: str, kb_name: str, role: str, text: str, *, user_id: str = ""):
    if not text:
        return
    if not _is_history_enabled(session_id):
        return
    chat_store.append_chat_message(session_id, kb_name, role, text, user_id=user_id)


def _get_chat_history(session_id: str, kb_name: str, *, user_id: str = "") -> List[Dict[str, str]]:
    return chat_store.get_chat_history(session_id, kb_name, user_id=user_id)


def _load_agent_message_history(session_id: str, kb_name: str, *, user_id: str = ""):
    return chat_store.load_agent_message_history(session_id, kb_name, user_id=user_id)


def _build_compact_history_block(session_id: str, kb_name: str, *, user_id: str = "") -> str:
    rows = [
        row
        for row in _get_chat_history(session_id, kb_name, user_id=user_id)
        if not (str(row.get("role", "")).lower() == "assistant" and is_failed_history_answer_text(str(row.get("text", ""))))
    ]
    return compact_chat_history_rows(
        rows,
        turn_limit=ai_service.settings.compact_history_turn_limit,
        char_limit=ai_service.settings.compact_history_char_limit,
    )


def _append_agent_run_history(
    session_id: str,
    kb_name: str,
    query_id: str,
    user_message: str,
    answer_text: str,
    new_messages_json: bytes,
    metadata: Optional[Dict[str, Any]] = None,
    response_quality_issue: str = "",
    usage: Optional[Dict[str, Any]] = None,
    context_chars: int = 0,
    user_id: str = "",
):
    if not _is_history_enabled(session_id):
        return
    chat_store.append_agent_run(
        session_id=session_id,
        kb_name=kb_name,
        query_id=query_id,
        user_message=user_message,
        answer_text=answer_text,
        new_messages_json=new_messages_json,
        metadata=metadata,
        response_quality_issue=response_quality_issue,
        usage=usage,
        context_chars=context_chars,
        user_id=user_id,
    )


def _attach_session_cookie(response, session_id: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": str(user.get("user_id", "") or ""),
        "login_id": str(user.get("login_id", "") or ""),
        "display_name": str(user.get("display_name", "") or ""),
        "role": str(user.get("role", "user") or "user"),
    }


def _current_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    return auth_store.get_user_by_session(request.cookies.get(AUTH_SESSION_COOKIE_NAME, ""))


def _require_current_user(request: Request) -> Optional[Dict[str, Any]]:
    return _current_user_from_request(request)


def _auth_required_response() -> JSONResponse:
    return _error_json_response(
        status_code=401,
        message="로그인이 필요합니다.",
        failure_code="auth_required",
    )


def _admin_required_response() -> JSONResponse:
    return _error_json_response(
        status_code=403,
        message="관리자 권한이 필요합니다.",
        failure_code="admin_required",
    )


def _sync_legacy_kbs_for_admin(user: Dict[str, Any]):
    if str(user.get("role", "") or "") != "admin":
        return
    auth_store.ensure_legacy_kbs_for_admin(str(user.get("user_id", "") or ""), list_kbs(data_dir=str(KB_DATA_DIR)))


def _sync_legacy_kbs_for_admin_best_effort(user: Dict[str, Any]):
    try:
        _sync_legacy_kbs_for_admin(user)
    except Exception as e:
        print(f"[AUTH][WARN] legacy_kb_admin_registration=failed error={e}", file=sys.stderr)


def _resolve_user_kb(user: Dict[str, Any], display_name: str, *, create_if_missing: bool = False) -> Optional[Dict[str, Any]]:
    user_id = str(user.get("user_id", "") or "")
    name = (display_name or "default").strip() or "default"
    if str(user.get("role", "") or "") == "admin":
        _sync_legacy_kbs_for_admin_best_effort(user)
    record = auth_store.get_kb(user_id, name)
    if record is None and create_if_missing:
        record = auth_store.create_kb(user_id, name)
        ensure_kb_directory(str(record["internal_kb_id"]))
    return record


def _public_kb_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    display_name = str(row.get("display_name", "") or "")
    return {
        "name": display_name,
        "display_name": display_name,
        "kb_id": str(row.get("kb_id", "") or ""),
        "internal_kb_id": str(row.get("internal_kb_id", "") or display_name),
    }


def _truncate_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def _format_timestamp(ts: int) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


# TEMP: 관리자 QA 평가/로그 기능. 배포 시 이 함수들과 /feedback 엔드포인트를 주석 처리 가능.
def _append_admin_feedback_log(entry: Dict[str, Any]):
    ADMIN_FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with admin_feedback_log_lock:
        rotate_file_if_oversize(
            ADMIN_FEEDBACK_LOG_PATH,
            max_bytes=OPERATIONAL_JSONL_MAX_BYTES,
            backup_count=OPERATIONAL_JSONL_BACKUP_COUNT,
        )
        with ADMIN_FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _append_rag_trace_log(entry: Dict[str, Any]):
    if not RAG_TRACE_LOG_ENABLED:
        return
    RAG_TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with rag_trace_log_lock:
        rotate_file_if_oversize(
            RAG_TRACE_LOG_PATH,
            max_bytes=OPERATIONAL_JSONL_MAX_BYTES,
            backup_count=OPERATIONAL_JSONL_BACKUP_COUNT,
        )
        with RAG_TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_recent_admin_feedback(limit: int = 120, kb_name: Optional[str] = None) -> List[Dict[str, Any]]:
    entries, _ = _load_recent_admin_feedback_with_warnings(limit=limit, kb_name=kb_name)
    return entries


def _load_recent_admin_feedback_with_warnings(
    limit: int = 120,
    kb_name: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], int]:
    if not ADMIN_FEEDBACK_LOG_PATH.exists():
        return [], 0

    with admin_feedback_log_lock:
        try:
            lines = ADMIN_FEEDBACK_LOG_PATH.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(
                f"[ADMIN_FEEDBACK][WARN] log_read_fail path={ADMIN_FEEDBACK_LOG_PATH} error={e}",
                file=sys.stderr,
            )
            return [], 1

    entries: List[Dict[str, Any]] = []
    warning_count = 0
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            warning_count += 1
            continue
        if kb_name and (payload.get("kb_name", "default") or "default") != kb_name:
            continue
        entries.append(payload)
        if len(entries) >= max(1, int(limit)):
            break
    if warning_count:
        print(
            f"[ADMIN_FEEDBACK][WARN] parse_failures={warning_count} path={ADMIN_FEEDBACK_LOG_PATH}",
            file=sys.stderr,
        )
    return entries, warning_count


async def _build_ops_failure_review(kb_name: str, limit: int = 120) -> Dict[str, Any]:
    rag = get_rag(kb_name)
    if not rag:
        return {
            "status": "error",
            "kb_name": kb_name,
            "error": "knowledge_base_not_found",
            "message": "지정한 KB를 찾지 못했습니다.",
            "failure_code": "knowledge_base_not_found",
        }

    agent_runs = chat_store.get_recent_agent_runs(kb_name=kb_name, limit=limit)
    answer_logs = rag.get_recent_answer_logs(limit=limit)
    retrieval_logs = rag.get_recent_retrieval_logs(limit=limit)
    feedback_logs, feedback_warning_count = _load_recent_admin_feedback_with_warnings(limit=limit, kb_name=kb_name)

    quality_counter: Counter = Counter()
    failure_code_counter: Counter = Counter()
    question_intent_counter: Counter = Counter()
    tool_name_counter: Counter = Counter()
    history_strategy_counter: Counter = Counter()
    phase_status_counter: Counter = Counter()
    tool_gap_count = 0
    outline_gap_count = 0
    citation_gap_count = 0
    incorrect_feedback_count = 0
    sample_failures: List[str] = []
    counted_failure_events: set[tuple[str, str]] = set()

    for row in agent_runs:
        metadata = _safe_json_loads(str(row.get("metadata_json", "") or ""))
        query_id = str(row.get("query_id", "") or "").strip()
        issue = (row.get("response_quality_issue", "") or "").strip()
        if issue:
            quality_counter[issue] += 1
        analysis = metadata.get("question_analysis", {}) if isinstance(metadata, dict) else {}
        question_intent = (
            metadata.get("question_intent", "")
            or (analysis.get("intent_type", "") if isinstance(analysis, dict) else "")
            or "mixed"
        )
        question_intent_counter[str(question_intent)] += 1
        history_strategy_counter[str(metadata.get("history_strategy", "") or "unknown")] += 1
        failure_code = str(metadata.get("failure_code", "") or "").strip()
        if failure_code:
            failure_key = (query_id or f"agent_run:{int(row.get('run_id', 0) or 0)}", failure_code)
            if failure_key not in counted_failure_events:
                counted_failure_events.add(failure_key)
                failure_code_counter[failure_code] += 1
        tool_events = metadata.get("tool_events", []) if isinstance(metadata, dict) else []
        if isinstance(tool_events, list):
            seen_this_run = set()
            for event in tool_events:
                tool_name = str((event or {}).get("tool_name", "")).strip()
                if not tool_name:
                    continue
                tool_name_counter[tool_name] += 1
                seen_this_run.add(tool_name)
            if int(row.get("tool_call_count", 0) or 0) <= 0 and not seen_this_run:
                tool_gap_count += 1
            if (
                isinstance(analysis, dict)
                and bool(analysis.get("use_source_outline", False))
                and not ({"get_source_outline", "get_source_overview", "open_document"} & seen_this_run)
            ):
                outline_gap_count += 1
        phase_events = metadata.get("phase_events", []) if isinstance(metadata, dict) else []
        if isinstance(phase_events, list):
            for event in phase_events:
                phase = str((event or {}).get("phase", "")).strip() or "unknown"
                status = str((event or {}).get("status", "")).strip() or "ok"
                if status not in {"", "ok"}:
                    phase_status_counter[f"{phase}:{status}"] += 1

        if issue and len(sample_failures) < 6:
            sample_failures.append(
                f"issue={issue} | intent={question_intent} | q={trim_preview(str(row.get('user_message', '') or ''), 110)}"
            )

    for row in answer_logs:
        meta = _safe_json_loads(str(row.get("answer_meta_json", "") or ""))
        citations = _safe_json_loads(str(row.get("citations_json", "") or "[]"))
        answer_text = str(row.get("answer_text", "") or "")
        query_id = str(row.get("query_id", "") or "").strip()
        if not citations and not _is_grounded_abstention(answer_text):
            citation_gap_count += 1
        issue = ""
        if isinstance(meta, dict):
            issue = str(meta.get("response_quality_issue", "") or "").strip()
        if issue:
            quality_counter[issue] += 1
        if isinstance(meta, dict):
            failure_code = str(meta.get("failure_code", "") or "").strip()
            if failure_code:
                failure_key = (query_id or f"answer_log:{int(row.get('created_at', 0) or 0)}", failure_code)
                if failure_key not in counted_failure_events:
                    counted_failure_events.add(failure_key)
                    failure_code_counter[failure_code] += 1

    for entry in feedback_logs:
        if not bool(entry.get("is_correct", False)):
            incorrect_feedback_count += 1
            if len(sample_failures) < 6:
                sample_failures.append(
                    "feedback=X | q="
                    + trim_preview(str(entry.get("question", "") or ""), 80)
                    + " | expected="
                    + trim_preview(str(entry.get("expected_answer", "") or ""), 80)
                )

    search_variant_counter: Counter = Counter()
    for row in retrieval_logs:
        meta = _safe_json_loads(str(row.get("meta_json", "") or ""))
        if not isinstance(meta, dict):
            continue
        for variant in meta.get("search_variants", [])[:6]:
            search_variant_counter[str(variant)] += 1

    report_lines = [
        f"KB={kb_name}",
        f"agent_runs={len(agent_runs)}",
        f"answer_logs={len(answer_logs)}",
        f"retrieval_logs={len(retrieval_logs)}",
        f"feedback_logs={len(feedback_logs)}",
        f"incorrect_feedback_count={incorrect_feedback_count}",
        "response_quality_issue_counts="
        + ", ".join(f"{k}:{v}" for k, v in quality_counter.most_common(8))
        if quality_counter
        else "response_quality_issue_counts=-",
        "question_intent_counts="
        + ", ".join(f"{k}:{v}" for k, v in question_intent_counter.most_common(8))
        if question_intent_counter
        else "question_intent_counts=-",
        "failure_code_counts="
        + ", ".join(f"{k}:{v}" for k, v in failure_code_counter.most_common(12))
        if failure_code_counter
        else "failure_code_counts=-",
        "tool_name_counts="
        + ", ".join(f"{k}:{v}" for k, v in tool_name_counter.most_common(10))
        if tool_name_counter
        else "tool_name_counts=-",
        "history_strategy_counts="
        + ", ".join(f"{k}:{v}" for k, v in history_strategy_counter.most_common(4))
        if history_strategy_counter
        else "history_strategy_counts=-",
        "phase_status_counts="
        + ", ".join(f"{k}:{v}" for k, v in phase_status_counter.most_common(12))
        if phase_status_counter
        else "phase_status_counts=-",
        "top_search_variants="
        + ", ".join(f"{k}:{v}" for k, v in search_variant_counter.most_common(8))
        if search_variant_counter
        else "top_search_variants=-",
        f"tool_gap_count={tool_gap_count}",
        f"outline_gap_count={outline_gap_count}",
        f"citation_gap_count={citation_gap_count}",
        "samples=" + " || ".join(sample_failures[:6]) if sample_failures else "samples=-",
    ]

    ops_review, ops_review_diag = await ai_service.review_operations_diagnostic(
        "\n".join(report_lines),
        max_tokens=min(320, RAG_LLM_HELPER_MAX_TOKENS + 80),
        timeout=min(LLM_REQUEST_TIMEOUT, RAG_LLM_HELPER_TIMEOUT),
    )
    if ops_review_diag.status == "error":
        print(
            f"[OPS_REVIEW][WARN] kb={kb_name} failure_code={ops_review_diag.failure_code or '-'} "
            f"error={ops_review_diag.error_detail or '-'}",
            file=sys.stderr,
        )

    response_quality_issue_counts = dict(quality_counter.most_common(12))
    question_intent_counts = dict(question_intent_counter.most_common(12))
    failure_code_counts = dict(failure_code_counter.most_common(12))
    tool_name_counts = dict(tool_name_counter.most_common(12))
    history_strategy_counts = dict(history_strategy_counter.most_common(8))
    phase_status_counts = dict(phase_status_counter.most_common(12))
    search_variant_counts = dict(search_variant_counter.most_common(12))

    return {
        "status": "success",
        "kb_name": kb_name,
        "window_limit": int(limit),
        "counts": {
            "agent_runs": len(agent_runs),
            "answer_logs": len(answer_logs),
            "retrieval_logs": len(retrieval_logs),
            "feedback_logs": len(feedback_logs),
            "feedback_load_warnings": int(feedback_warning_count),
            "incorrect_feedback": incorrect_feedback_count,
            "tool_gap_runs": tool_gap_count,
            "outline_gap_runs": outline_gap_count,
            "citation_gap_answers": citation_gap_count,
        },
        "response_quality_issue_counts": response_quality_issue_counts,
        "question_intent_counts": question_intent_counts,
        "failure_code_counts": failure_code_counts,
        "tool_name_counts": tool_name_counts,
        "history_strategy_counts": history_strategy_counts,
        "phase_status_counts": phase_status_counts,
        "search_variant_counts": search_variant_counts,
        "tool_gap_count": tool_gap_count,
        "outline_gap_count": outline_gap_count,
        "citation_gap_count": citation_gap_count,
        "quality_issues": response_quality_issue_counts,
        "question_intents": question_intent_counts,
        "tool_usage": tool_name_counts,
        "search_variants": search_variant_counts,
        "samples": sample_failures[:6],
        "feedback_load_warning_count": int(feedback_warning_count),
        "ops_review": ops_review.model_dump(),
        "ops_review_status": ops_review_diag.status,
        "ops_review_failure_code": ops_review_diag.failure_code,
        "ops_review_error": ops_review_diag.error_detail,
    }


def _safe_upload_filename(name: str) -> str:
    return safe_upload_filename(name)


def _build_stored_upload_name(original_name: str) -> str:
    return build_stored_upload_name(original_name)


def _validate_upload_meta(upload_file: UploadFile) -> str:
    return validate_upload_meta(
        filename=upload_file.filename or "",
        content_type=upload_file.content_type or "",
        allowed_extensions=ALLOWED_UPLOAD_EXTENSIONS,
        allowed_mime_by_ext=ALLOWED_MIME_BY_EXT,
    )


def _save_upload_stream(upload_file: UploadFile, dest_path: str, max_bytes: int) -> int:
    total = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"파일이 너무 큽니다. 최대 {max_bytes} 바이트까지 가능합니다.")
            buffer.write(chunk)
    return total


def _is_zip_signature(path: str) -> bool:
    return is_zip_signature(path)


def _is_pdf_signature(path: str) -> bool:
    return is_pdf_signature(path)


def _is_hwpx_signature(path: str) -> bool:
    return is_hwpx_signature(path)


def _cleanup_upload_jobs_locked(now_ts: Optional[int] = None):
    global _upload_job_store_last_prune_at
    now = int(now_ts or time.time())
    _coerce_stalled_upload_jobs_locked(now)
    expire_before = now - UPLOAD_JOB_RETENTION_SECONDS
    remove_ids: List[str] = []
    for job_id, job in upload_jobs.items():
        status = (job.get("status", "") or "").lower()
        updated_at = int(job.get("updated_at", 0) or 0)
        if status in {"success", "error"} and updated_at < expire_before:
            remove_ids.append(job_id)
    for job_id in remove_ids:
        upload_jobs.pop(job_id, None)
    if now - int(_upload_job_store_last_prune_at or 0) >= UPLOAD_JOB_PRUNE_INTERVAL_SECONDS:
        try:
            upload_job_store.prune_terminal_jobs(
                expire_before=expire_before,
                max_terminal_rows=UPLOAD_JOB_RETENTION_MAX_ROWS,
                batch_size=UPLOAD_JOB_PRUNE_BATCH_SIZE,
            )
            _upload_job_store_last_prune_at = now
        except Exception as exc:
            print(f"[UPLOAD][WARN] persisted_job_prune_failed error={exc}", file=sys.stderr)


def _coerce_stalled_upload_jobs_locked(now_ts: Optional[int] = None):
    now = int(now_ts or time.time())
    has_public_change = False
    for job in upload_jobs.values():
        previous_snapshot = _upload_job_public_version_snapshot(job)
        stalled = build_upload_stall_state(
            job,
            now_ts=now,
            processing_timeout_seconds=UPLOAD_JOB_STALL_TIMEOUT_SECONDS,
            queue_timeout_seconds=UPLOAD_QUEUE_STALL_TIMEOUT_SECONDS,
        )
        if not stalled:
            continue
        job["status"] = stalled["status"]
        job["progress_stage"] = stalled["progress_stage"]
        job["failure_code"] = stalled["failure_code"]
        job["message"] = stalled["message"]
        job["stall_seconds"] = int(stalled.get("stall_seconds", 0) or 0)
        job["stall_timeout_seconds"] = int(stalled.get("stall_timeout_seconds", 0) or 0)
        job["stalled_stage"] = str(stalled.get("stalled_stage", "") or "").strip().lower()
        job["ocr_stall_detected"] = bool(stalled.get("ocr_stall_detected", False))
        job["phase_elapsed_seconds"] = int(stalled.get("phase_elapsed_seconds", 0) or 0)
        job["phase_last_heartbeat_at"] = int(stalled.get("phase_last_heartbeat_at", 0) or 0)
        for key in ("pages_done", "pages_total", "rows_done", "rows_total", "chunks_done", "chunks_total"):
            if key in stalled:
                job[key] = int(stalled.get(key, 0) or 0)
        job["updated_at"] = now
        if not int(job.get("completed_at", 0) or 0):
            job["completed_at"] = now
        if _bump_upload_job_version_locked(job, previous_snapshot):
            has_public_change = True
    if has_public_change:
        upload_jobs_condition.notify_all()


def _normalize_upload_progress_percent(value: Any, fallback: int = 0) -> int:
    try:
        percent = int(float(value))
    except (TypeError, ValueError):
        percent = int(fallback or 0)
    return max(0, min(100, percent))


def _extract_progress_page_stats(message: str) -> Dict[str, int]:
    text = str(message or "")
    match = re.search(r"\((\d+)\s*/\s*(\d+)\)", text)
    if not match:
        return {}
    current_page = max(0, int(match.group(1)))
    total_pages = max(current_page, int(match.group(2)))
    return {
        "current_page": current_page,
        "total_pages": total_pages,
    }


def _upload_job_int_tuple(value: Any) -> tuple[int, ...]:
    result: List[int] = []
    for item in list(value or []):
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            result.append(parsed)
    return tuple(sorted(set(result)))


def _upload_job_public_version_snapshot(job: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(job.get("status", "") or "").strip().lower(),
        str(job.get("message", "") or ""),
        int(job.get("progress_percent", 0) or 0),
        str(job.get("progress_stage", "") or "").strip().lower(),
        bool(job.get("used_cache", False)),
        int(job.get("chunks", 0) or 0),
        int(job.get("replaced_chunks", 0) or 0),
        int(job.get("normalized_chunks", 0) or 0),
        int(job.get("current_page", 0) or 0),
        int(job.get("total_pages", 0) or 0),
        int(job.get("pdf_total_pages", 0) or 0),
        int(job.get("ocr_target_pages", 0) or 0),
        int(job.get("ocr_completed_pages", 0) or 0),
        str(job.get("pdf_parser", "") or "").strip(),
        int(job.get("pdf_text_pages", 0) or 0),
        int(job.get("pdf_ocr_pages", 0) or 0),
        int(job.get("pdf_failed_pages", 0) or 0),
        tuple(str(item or "") for item in list(job.get("pdf_warnings", []) or [])),
        str(job.get("ocr_device_attempted", "") or "").strip(),
        str(job.get("ocr_device_effective", "") or "").strip(),
        bool(job.get("ocr_gpu_fallback_used", False)),
        str(job.get("ocr_gpu_failure_reason", "") or "").strip(),
        float(job.get("ocr_elapsed_seconds", 0.0) or 0.0),
        int(job.get("ocr_pages_processed", 0) or 0),
        float(job.get("ocr_pages_per_minute", 0.0) or 0.0),
        int(job.get("ocr_pages_attempted", 0) or 0),
        int(job.get("ocr_pages_emitted", 0) or 0),
        int(job.get("ocr_pages_skipped_empty", 0) or 0),
        int(job.get("ocr_pages_skipped_short_text", 0) or 0),
        float(job.get("ocr_attempted_pages_per_minute", 0.0) or 0.0),
        float(job.get("ocr_emitted_pages_per_minute", 0.0) or 0.0),
        bool(job.get("ocr_worker_released", False)),
        float(job.get("ocr_worker_release_seconds", 0.0) or 0.0),
        _upload_job_int_tuple(job.get("ocr_worker_pids", [])),
        bool(job.get("ocr_worker_shutdown_confirmed", True)),
        _upload_job_int_tuple(job.get("ocr_worker_alive_after_shutdown", [])),
        float(job.get("ocr_duration_seconds", 0.0) or 0.0),
        float(job.get("persist_duration_seconds", 0.0) or 0.0),
        float(job.get("embedding_duration_seconds", 0.0) or 0.0),
        float(job.get("index_duration_seconds", 0.0) or 0.0),
        float(job.get("derived_sync_duration_seconds", 0.0) or 0.0),
        int(job.get("phase_started_at", 0) or 0),
        int(job.get("phase_last_heartbeat_at", 0) or 0),
        int(job.get("phase_elapsed_seconds", 0) or 0),
        str(job.get("phase_name_effective", "") or "").strip().lower(),
        int(job.get("phase_rows_total", 0) or 0),
        int(job.get("phase_rows_done", 0) or 0),
        int(job.get("phase_chunks_total", 0) or 0),
        int(job.get("phase_chunks_done", 0) or 0),
        int(job.get("embed_batch", 0) or 0),
        int(job.get("embed_batches", 0) or 0),
        int(job.get("embed_rows_done", 0) or 0),
        int(job.get("embed_rows_total", 0) or 0),
        int(job.get("embed_input_tokens_total", 0) or 0),
        int(job.get("embed_input_tokens_done", 0) or 0),
        int(job.get("embed_input_tokens_p95", 0) or 0),
        int(job.get("embed_input_tokens_max", 0) or 0),
        int(job.get("embed_truncated_rows", 0) or 0),
        int(job.get("embed_effective_batch_tokens", 0) or 0),
        int(job.get("ocr_fast_pages", 0) or 0),
        int(job.get("ocr_vl_pages", 0) or 0),
        float(job.get("ocr_fast_seconds", 0.0) or 0.0),
        float(job.get("ocr_vl_seconds", 0.0) or 0.0),
        float(job.get("ocr_fast_avg_score", 0.0) or 0.0),
        float(job.get("ocr_fast_pair_ratio", 0.0) or 0.0),
        float(job.get("ocr_fast_orphan_ratio", 0.0) or 0.0),
        bool(job.get("ocr_high_quality_requested", False)),
        int(job.get("ocr_target_pages_goal", 0) or 0),
        float(job.get("ocr_target_seconds", 0.0) or 0.0),
        bool(job.get("ocr_target_met", False)),
        int(job.get("ocr_last_batch_completed_at", 0) or 0),
        str(job.get("ocr_retry_mode", "") or "").strip(),
        str(job.get("ocr_retry_reason", "") or "").strip(),
        bool(job.get("ocr_stall_detected", False)),
        str(job.get("failure_code", "") or "").strip(),
        str(job.get("document_role", "") or "").strip(),
    )


def _bump_upload_job_version_locked(job: Dict[str, Any], previous_snapshot: tuple[Any, ...]) -> bool:
    next_snapshot = _upload_job_public_version_snapshot(job)
    if next_snapshot == previous_snapshot:
        return False
    job["version"] = max(1, int(job.get("version", 0) or 0) + 1)
    return True


def _normalize_upload_job_version(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _clamp_upload_job_wait_seconds(value: Optional[float]) -> float:
    try:
        seconds = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(UPLOAD_JOB_LONG_POLL_MAX_WAIT_SECONDS, seconds))


def _wait_for_upload_job_change(job_id: str, since_version: int, wait_seconds: float) -> str:
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    with upload_jobs_condition:
        while True:
            _coerce_stalled_upload_jobs_locked()
            job = upload_jobs.get(job_id)
            if not job:
                return "missing"
            if _normalize_upload_job_version(job.get("version", 0)) != int(since_version):
                return "changed"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            upload_jobs_condition.wait(timeout=remaining)


def _create_upload_job(
    kb_name: str,
    original_filename: str,
    stored_filename: str,
    stored_path: str,
    document_role: str,
    user_id: str = "",
) -> Dict[str, Any]:
    now = int(time.time())
    default_ocr_device = (os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip()
    job = {
        "job_id": uuid.uuid4().hex,
        "kb_name": kb_name,
        "user_id": user_id or "",
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "stored_path": stored_path,
        "document_role": _normalize_doc_role(document_role),
        "status": "queued",
        "message": "업로드가 접수되었습니다. 잠시만 기다려 주세요.",
        "progress_percent": 0,
        "progress_stage": "queued",
        "used_cache": False,
        "chunks": 0,
        "replaced_chunks": 0,
        "normalized_chunks": 0,
        "current_page": 0,
        "total_pages": 0,
        "pdf_total_pages": 0,
        "ocr_target_pages": 0,
        "ocr_completed_pages": 0,
        "pdf_parser": "",
        "pdf_text_pages": 0,
        "pdf_ocr_pages": 0,
        "pdf_failed_pages": 0,
        "pdf_warnings": [],
        "ocr_device_attempted": default_ocr_device,
        "ocr_device_effective": default_ocr_device,
        "ocr_gpu_fallback_used": False,
        "ocr_gpu_failure_reason": "",
        "ocr_elapsed_seconds": 0.0,
        "ocr_pages_processed": 0,
        "ocr_pages_per_minute": 0.0,
        "ocr_pages_attempted": 0,
        "ocr_pages_emitted": 0,
        "ocr_pages_skipped_empty": 0,
        "ocr_pages_skipped_short_text": 0,
        "ocr_attempted_pages_per_minute": 0.0,
        "ocr_emitted_pages_per_minute": 0.0,
        "ocr_worker_released": False,
        "ocr_worker_release_seconds": 0.0,
        "ocr_worker_pids": [],
        "ocr_worker_shutdown_confirmed": True,
        "ocr_worker_alive_after_shutdown": [],
        "ocr_duration_seconds": 0.0,
        "persist_duration_seconds": 0.0,
        "embedding_duration_seconds": 0.0,
        "index_duration_seconds": 0.0,
        "derived_sync_duration_seconds": 0.0,
        "phase_started_at": 0,
        "phase_last_heartbeat_at": 0,
        "phase_elapsed_seconds": 0,
        "phase_name_effective": "",
        "phase_rows_total": 0,
        "phase_rows_done": 0,
        "phase_chunks_total": 0,
        "phase_chunks_done": 0,
        "embed_batch": 0,
        "embed_batches": 0,
        "embed_rows_done": 0,
        "embed_rows_total": 0,
        "embed_input_tokens_total": 0,
        "embed_input_tokens_done": 0,
        "embed_input_tokens_p95": 0,
        "embed_input_tokens_max": 0,
        "embed_truncated_rows": 0,
        "embed_effective_batch_tokens": 0,
        "ocr_subset_build_seconds": 0.0,
        "ocr_model_load_seconds": 0.0,
        "ocr_predict_seconds": 0.0,
        "ocr_output_materialize_seconds": 0.0,
        "ocr_payload_convert_seconds": 0.0,
        "ocr_fragment_collect_seconds": 0.0,
        "ocr_page_dedupe_seconds": 0.0,
        "ocr_page_join_seconds": 0.0,
        "ocr_text_merge_seconds": 0.0,
        "ocr_merge_seconds": 0.0,
        "ocr_batch_count": 0,
        "ocr_backend": (os.getenv("PDF_OCR_BACKEND", "ppocr_fast_v1") or "ppocr_fast_v1").strip(),
        "ocr_backend_attempted": (os.getenv("PDF_OCR_BACKEND", "ppocr_fast_v1") or "ppocr_fast_v1").strip(),
        "ocr_backend_effective": (os.getenv("PDF_OCR_BACKEND", "ppocr_fast_v1") or "ppocr_fast_v1").strip(),
        "ocr_backend_fallback_used": False,
        "ocr_fast_pages": 0,
        "ocr_vl_pages": 0,
        "ocr_fast_seconds": 0.0,
        "ocr_vl_seconds": 0.0,
        "ocr_fast_avg_score": 0.0,
        "ocr_fast_pair_ratio": 0.0,
        "ocr_fast_orphan_ratio": 0.0,
        "ocr_high_quality_requested": False,
        "ocr_target_pages_goal": int(os.getenv("PDF_OCR_TARGET_PAGES", os.getenv("PDF_OCR_MAX_PAGES", "200")) or 0),
        "ocr_target_seconds": float(os.getenv("PDF_OCR_TARGET_SECONDS", "300") or 0),
        "ocr_target_met": False,
        "ocr_last_batch_completed_at": 0,
        "ocr_retry_mode": "",
        "ocr_retry_reason": "",
        "ocr_stall_detected": False,
        "stalled_stage": "",
        "ocr_heartbeat_at": 0,
        "failure_code": "",
        "queued_at": now,
        "processing_started_at": 0,
        "completed_at": 0,
        "last_progress_at": now,
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }
    with upload_jobs_condition:
        _cleanup_upload_jobs_locked(now)
        upload_jobs[job["job_id"]] = job
        try:
            upload_job_store.save_job(job)
        except Exception as e:
            print(f"[UPLOAD][WARN] persist_create_failed job_id={job['job_id']} error={e}", file=sys.stderr)
        return dict(job)


def _update_upload_job(job_id: str, **updates: Any):
    with upload_jobs_condition:
        job = upload_jobs.get(job_id)
        if not job:
            return
        previous_snapshot = _upload_job_public_version_snapshot(job)
        job.update(updates)
        job["updated_at"] = int(time.time())
        status = str(job.get("status", "") or "").strip().lower()
        progress_stage = str(job.get("progress_stage", "") or "").strip().lower()
        if status == "processing" and not int(job.get("processing_started_at", 0) or 0):
            job["processing_started_at"] = job["updated_at"]
        if status in {"processing", "success", "error"}:
            job["last_progress_at"] = job["updated_at"]
        if status == "processing" and progress_stage in _UPLOAD_OCR_PROGRESS_STAGES:
            job["ocr_heartbeat_at"] = job["updated_at"]
        if status in {"success", "error"} and not int(job.get("completed_at", 0) or 0):
            job["completed_at"] = job["updated_at"]
        _cleanup_upload_jobs_locked(job["updated_at"])
        if _bump_upload_job_version_locked(job, previous_snapshot):
            upload_jobs_condition.notify_all()
        persisted_job = dict(job)
    try:
        upload_job_store.save_job(persisted_job)
    except Exception as e:
        print(f"[UPLOAD][WARN] persist_update_failed job_id={job_id} error={e}", file=sys.stderr)


def _get_upload_job(job_id: str) -> Optional[Dict[str, Any]]:
    with upload_jobs_lock:
        _coerce_stalled_upload_jobs_locked()
        job = upload_jobs.get(job_id)
        if not job:
            payload = None
        else:
            payload = dict(job)
    if payload is None:
        payload = upload_job_store.get_job(job_id)
        if not payload:
            return None
    now = int(time.time())
    created_at = int(payload.get("created_at", 0) or 0)
    processing_started_at = int(payload.get("processing_started_at", 0) or 0)
    completed_at = int(payload.get("completed_at", 0) or 0)
    last_progress_at = int(payload.get("last_progress_at", 0) or created_at or now)
    reference_end = completed_at or now
    payload["elapsed_seconds"] = max(0, reference_end - created_at) if created_at else 0
    payload["queue_wait_seconds"] = (
        max(0, processing_started_at - created_at) if created_at and processing_started_at else 0
    )
    payload["processing_elapsed_seconds"] = (
        max(0, reference_end - processing_started_at) if processing_started_at else 0
    )
    payload["last_progress_age_seconds"] = max(0, now - last_progress_at)
    ocr_heartbeat_at = int(payload.get("ocr_heartbeat_at", 0) or 0)
    payload["ocr_heartbeat_age_seconds"] = max(0, now - ocr_heartbeat_at) if ocr_heartbeat_at else 0
    ocr_last_batch_completed_at = int(payload.get("ocr_last_batch_completed_at", 0) or 0)
    payload["ocr_last_batch_age_seconds"] = (
        max(0, now - ocr_last_batch_completed_at) if ocr_last_batch_completed_at else 0
    )
    phase_started_at = int(payload.get("phase_started_at", 0) or 0)
    payload["phase_elapsed_seconds"] = max(0, now - phase_started_at) if phase_started_at else int(
        payload.get("phase_elapsed_seconds", 0) or 0
    )
    phase_last_heartbeat_at = int(payload.get("phase_last_heartbeat_at", 0) or 0)
    payload["phase_last_heartbeat_age_seconds"] = (
        max(0, now - phase_last_heartbeat_at) if phase_last_heartbeat_at else 0
    )
    payload["raw_progress_percent"] = int(payload.get("progress_percent", 0) or 0)
    payload["progress_percent"] = estimate_display_progress_percent(payload, now_ts=now)
    return payload


def _list_active_upload_jobs(exclude_kb_name: str = "", *, user_id: str = "") -> List[Dict[str, Any]]:
    excluded = (exclude_kb_name or "").strip()
    now = int(time.time())
    active_jobs: List[Dict[str, Any]] = []
    with upload_jobs_lock:
        _cleanup_upload_jobs_locked(now)
        for job in upload_jobs.values():
            status = str(job.get("status", "") or "").strip().lower()
            if status not in {"queued", "processing"}:
                continue
            if user_id and str(job.get("user_id", "") or "") != user_id:
                continue
            kb_name = (str(job.get("kb_name", "default") or "default").strip() or "default")
            if excluded and kb_name == excluded:
                continue
            active_jobs.append(
                {
                    "job_id": str(job.get("job_id", "") or "").strip(),
                    "kb_name": kb_name,
                    "user_id": str(job.get("user_id", "") or "").strip(),
                    "status": status,
                    "progress_stage": str(job.get("progress_stage", "") or "").strip(),
                    "progress_percent": int(job.get("progress_percent", 0) or 0),
                    "original_filename": str(job.get("original_filename", "") or "").strip(),
                    "updated_at": int(job.get("updated_at", 0) or 0),
                }
            )
    active_jobs.sort(
        key=lambda row: (
            0 if row.get("status") == "processing" else 1,
            -int(row.get("updated_at", 0) or 0),
        )
    )
    return active_jobs


def _find_cross_kb_upload_blocker(target_kb_name: str) -> Optional[Dict[str, Any]]:
    user_id = _request_user_id_context.get("")
    blockers = _list_active_upload_jobs(exclude_kb_name=target_kb_name, user_id=user_id)
    if not blockers:
        return None
    return dict(blockers[0])


def _render_cross_kb_upload_busy_text(target_kb_name: str, blocker: Dict[str, Any]) -> str:
    blocker_kb = (str(blocker.get("kb_name", "") or "").strip() or "다른 공간")
    filename = (str(blocker.get("original_filename", "") or "").strip())
    detail = f" 진행 중 공간은 '{blocker_kb}'입니다."
    if filename:
        detail = f" 진행 중 공간은 '{blocker_kb}'이고, 파일은 '{filename}'입니다."
    return (
        f"지금은 다른 공간에서 문서 업로드가 진행 중이라 '{target_kb_name}' 공간 문서 검색을 잠시 미룹니다."
        f"{detail} 업로드가 끝난 뒤 다시 질문해 주세요."
    )


def _cleanup_ocr_jobs_locked(now_ts: Optional[int] = None):
    now = int(now_ts or time.time())
    expire_before = now - PDF_BACKGROUND_OCR_JOB_RETENTION_SECONDS
    remove_ids: List[str] = []
    for job_id, job in ocr_jobs.items():
        status = str(job.get("status", "") or "").strip().lower()
        updated_at = int(job.get("updated_at", 0) or 0)
        if status in {"success", "error", "skipped"} and updated_at < expire_before:
            remove_ids.append(job_id)
    for job_id in remove_ids:
        ocr_jobs.pop(job_id, None)


def _ocr_job_public_version_snapshot(job: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(job.get("status", "") or "").strip().lower(),
        str(job.get("message", "") or ""),
        int(job.get("progress_percent", 0) or 0),
        str(job.get("progress_stage", "") or "").strip().lower(),
        int(job.get("current_page", 0) or 0),
        int(job.get("total_pages", 0) or 0),
        int(job.get("ocr_target_pages", 0) or 0),
        int(job.get("ocr_completed_pages", 0) or 0),
        str(job.get("failure_code", "") or "").strip(),
        int(job.get("completed_at", 0) or 0),
    )


def _bump_ocr_job_version_locked(job: Dict[str, Any], previous_snapshot: tuple[Any, ...]) -> bool:
    if _ocr_job_public_version_snapshot(job) == previous_snapshot:
        return False
    job["version"] = max(1, int(job.get("version", 0) or 0) + 1)
    return True


def _create_ocr_job(
    *,
    kb_name: str,
    original_filename: str,
    stored_filename: str,
    stored_path: str,
    document_role: str,
    upload_job_id: str,
    user_id: str = "",
    total_pages: int = 0,
    ocr_target_pages: int = 0,
) -> Dict[str, Any]:
    now = int(time.time())
    job = {
        "job_id": uuid.uuid4().hex,
        "kb_name": kb_name,
        "user_id": user_id or "",
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "stored_path": stored_path,
        "document_role": _normalize_doc_role(document_role),
        "upload_job_id": upload_job_id,
        "status": "queued",
        "message": "OCR 보강 작업이 접수되었습니다.",
        "progress_percent": 0,
        "progress_stage": "queued",
        "current_page": 0,
        "total_pages": max(0, int(total_pages or 0)),
        "ocr_target_pages": max(0, int(ocr_target_pages or 0)),
        "ocr_completed_pages": 0,
        "failure_code": "",
        "queued_at": now,
        "processing_started_at": 0,
        "completed_at": 0,
        "last_progress_at": now,
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }
    with ocr_jobs_condition:
        _cleanup_ocr_jobs_locked(now)
        ocr_jobs[job["job_id"]] = job
        return dict(job)


def _update_ocr_job(job_id: str, **updates: Any):
    with ocr_jobs_condition:
        job = ocr_jobs.get(job_id)
        if not job:
            return
        previous_snapshot = _ocr_job_public_version_snapshot(job)
        job.update(updates)
        job["updated_at"] = int(time.time())
        status = str(job.get("status", "") or "").strip().lower()
        if status == "processing" and not int(job.get("processing_started_at", 0) or 0):
            job["processing_started_at"] = job["updated_at"]
        if status in {"processing", "success", "error", "skipped"}:
            job["last_progress_at"] = job["updated_at"]
        if status in {"success", "error", "skipped"} and not int(job.get("completed_at", 0) or 0):
            job["completed_at"] = job["updated_at"]
        _cleanup_ocr_jobs_locked(job["updated_at"])
        if _bump_ocr_job_version_locked(job, previous_snapshot):
            ocr_jobs_condition.notify_all()


def _get_ocr_job(job_id: str) -> Optional[Dict[str, Any]]:
    with ocr_jobs_lock:
        _cleanup_ocr_jobs_locked()
        job = ocr_jobs.get(job_id)
        if not job:
            return None
        payload = dict(job)
    now = int(time.time())
    created_at = int(payload.get("created_at", 0) or 0)
    completed_at = int(payload.get("completed_at", 0) or 0)
    reference_end = completed_at or now
    payload["elapsed_seconds"] = max(0, reference_end - created_at) if created_at else 0
    return payload


def _list_ocr_jobs(kb_name: str = "", include_terminal: bool = False, user_id: str = "") -> List[Dict[str, Any]]:
    normalized_kb = (str(kb_name or "").strip() or "default") if kb_name else ""
    with ocr_jobs_lock:
        _cleanup_ocr_jobs_locked()
        rows = []
        for job in ocr_jobs.values():
            status = str(job.get("status", "") or "").strip().lower()
            job_kb = str(job.get("kb_name", "default") or "default").strip() or "default"
            if normalized_kb and job_kb != normalized_kb:
                continue
            if user_id and str(job.get("user_id", "") or "") != user_id:
                continue
            if not include_terminal and status in {"success", "error", "skipped"}:
                continue
            rows.append(dict(job))
    rows.sort(
        key=lambda row: (
            0 if str(row.get("status", "")).lower() == "processing" else 1,
            -int(row.get("updated_at", 0) or 0),
        )
    )
    return rows


def _wait_for_ocr_job_change(job_id: str, since_version: int, wait_seconds: float) -> str:
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    with ocr_jobs_condition:
        while True:
            _cleanup_ocr_jobs_locked()
            job = ocr_jobs.get(job_id)
            if not job:
                return "missing"
            if _normalize_upload_job_version(job.get("version", 0)) != int(since_version):
                return "changed"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            ocr_jobs_condition.wait(timeout=remaining)


def _cleanup_ontology_rebuild_jobs_locked(now_ts: Optional[int] = None):
    now = int(now_ts or time.time())
    expire_before = now - ONTOLOGY_REBUILD_JOB_RETENTION_SECONDS
    remove_ids: List[str] = []
    for job_id, job in ontology_rebuild_jobs.items():
        status = str(job.get("status", "") or "").strip().lower()
        updated_at = int(job.get("updated_at", 0) or 0)
        if status in {"success", "error", "cancelled"} and updated_at < expire_before:
            remove_ids.append(job_id)
    for job_id in remove_ids:
        ontology_rebuild_jobs.pop(job_id, None)


def _ontology_rebuild_job_public_version_snapshot(job: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(job.get("status", "") or "").strip().lower(),
        str(job.get("message", "") or ""),
        int(job.get("progress_percent", 0) or 0),
        int(job.get("chunks_total", 0) or 0),
        int(job.get("chunks_processed", 0) or 0),
        int(job.get("ontology_extraction_errors", 0) or 0),
        str(job.get("failure_code", "") or "").strip(),
        bool(job.get("cancel_requested", False)),
        int(job.get("completed_at", 0) or 0),
    )


def _bump_ontology_rebuild_job_version_locked(job: Dict[str, Any], previous_snapshot: tuple[Any, ...]) -> bool:
    if _ontology_rebuild_job_public_version_snapshot(job) == previous_snapshot:
        return False
    job["version"] = max(1, int(job.get("version", 0) or 0) + 1)
    return True


def _create_ontology_rebuild_job(
    *,
    kb_name: str,
    internal_kb_id: str,
    user_id: str,
    include_llm: bool,
    retry_of_job_id: str = "",
    chunk_ids: Optional[List[int]] = None,
    trigger: str = "manual_rebuild",
    source_type: str = "",
    source_path: str = "",
    parser_signature: str = "",
    query_id: str = "",
    saved_answer_id: int = 0,
    llm_fact_status: str = "",
) -> Dict[str, Any]:
    now = int(time.time())
    job = {
        "job_id": uuid.uuid4().hex,
        "kb_name": str(kb_name or "default"),
        "internal_kb_id": str(internal_kb_id or kb_name or "default"),
        "user_id": str(user_id or ""),
        "include_llm": bool(include_llm),
        "retry_of_job_id": str(retry_of_job_id or ""),
        "chunk_ids": [int(value) for value in (chunk_ids or []) if int(value) > 0],
        "trigger": str(trigger or "manual_rebuild"),
        "source_type": str(source_type or ""),
        "source_path": str(source_path or ""),
        "parser_signature": str(parser_signature or ""),
        "query_id": str(query_id or ""),
        "saved_answer_id": int(saved_answer_id or 0),
        "llm_fact_status": str(llm_fact_status or ""),
        "status": "queued",
        "message": "Ontology rebuild 작업이 접수되었습니다.",
        "progress_percent": 0,
        "chunks_total": 0,
        "chunks_processed": 0,
        "ontology_facts_added": 0,
        "ontology_facts_deleted": 0,
        "ontology_extraction_errors": 0,
        "ontology_extraction_disabled": False,
        "ontology_extraction_disabled_reason": "",
        "failure_code": "",
        "cancel_requested": False,
        "queued_at": now,
        "processing_started_at": 0,
        "completed_at": 0,
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }
    with ontology_rebuild_jobs_condition:
        _cleanup_ontology_rebuild_jobs_locked(now)
        ontology_rebuild_jobs[job["job_id"]] = job
        return dict(job)


def _update_ontology_rebuild_job(job_id: str, **updates: Any):
    with ontology_rebuild_jobs_condition:
        job = ontology_rebuild_jobs.get(job_id)
        if not job:
            return
        previous_snapshot = _ontology_rebuild_job_public_version_snapshot(job)
        job.update(updates)
        job["updated_at"] = int(time.time())
        status = str(job.get("status", "") or "").strip().lower()
        if status == "processing" and not int(job.get("processing_started_at", 0) or 0):
            job["processing_started_at"] = job["updated_at"]
        if status in {"success", "error", "cancelled"} and not int(job.get("completed_at", 0) or 0):
            job["completed_at"] = job["updated_at"]
        _cleanup_ontology_rebuild_jobs_locked(job["updated_at"])
        if _bump_ontology_rebuild_job_version_locked(job, previous_snapshot):
            ontology_rebuild_jobs_condition.notify_all()


def _get_ontology_rebuild_job(job_id: str) -> Optional[Dict[str, Any]]:
    with ontology_rebuild_jobs_lock:
        _cleanup_ontology_rebuild_jobs_locked()
        job = ontology_rebuild_jobs.get(str(job_id or ""))
        if not job:
            return None
        payload = dict(job)
    now = int(time.time())
    created_at = int(payload.get("created_at", 0) or 0)
    processing_started_at = int(payload.get("processing_started_at", 0) or 0)
    completed_at = int(payload.get("completed_at", 0) or 0)
    reference_end = completed_at or now
    payload["elapsed_seconds"] = max(0, reference_end - created_at) if created_at else 0
    payload["processing_elapsed_seconds"] = (
        max(0, reference_end - processing_started_at) if processing_started_at else 0
    )
    return payload


def _list_ontology_rebuild_jobs(kb_name: str = "", user_id: str = "", include_terminal: bool = True) -> List[Dict[str, Any]]:
    safe_kb = str(kb_name or "").strip()
    safe_user_id = str(user_id or "").strip()
    with ontology_rebuild_jobs_lock:
        _cleanup_ontology_rebuild_jobs_locked()
        rows = []
        for job in ontology_rebuild_jobs.values():
            status = str(job.get("status", "") or "").strip().lower()
            if safe_kb and str(job.get("kb_name", "") or "") != safe_kb:
                continue
            if safe_user_id and str(job.get("user_id", "") or "") != safe_user_id:
                continue
            if not include_terminal and status in {"success", "error", "cancelled"}:
                continue
            rows.append(dict(job))
    rows.sort(
        key=lambda row: (
            0 if str(row.get("status", "")).lower() == "processing" else 1,
            -int(row.get("updated_at", 0) or 0),
        )
    )
    return rows


def _find_reported_answer_ontology_job(
    *,
    kb_name: str,
    user_id: str,
    query_id: str,
) -> Optional[Dict[str, Any]]:
    safe_kb = str(kb_name or "").strip()
    safe_user = str(user_id or "").strip()
    safe_query = str(query_id or "").strip()
    if not safe_kb or not safe_query:
        return None
    with ontology_rebuild_jobs_lock:
        _cleanup_ontology_rebuild_jobs_locked()
        matches = [
            dict(job)
            for job in ontology_rebuild_jobs.values()
            if str(job.get("kb_name", "") or "") == safe_kb
            and str(job.get("user_id", "") or "") == safe_user
            and str(job.get("trigger", "") or "") == "reported_answer"
            and str(job.get("query_id", "") or "") == safe_query
        ]
    if not matches:
        return None
    matches.sort(key=lambda item: int(item.get("created_at", 0) or 0), reverse=True)
    return matches[0]


def _request_cancel_ontology_rebuild_job(job_id: str) -> Optional[Dict[str, Any]]:
    with ontology_rebuild_jobs_condition:
        job = ontology_rebuild_jobs.get(str(job_id or ""))
        if not job:
            return None
        status = str(job.get("status", "") or "").strip().lower()
        if status in {"success", "error", "cancelled"}:
            return dict(job)
        previous_snapshot = _ontology_rebuild_job_public_version_snapshot(job)
        job["cancel_requested"] = True
        job["message"] = "Ontology rebuild 취소가 요청되었습니다."
        job["updated_at"] = int(time.time())
        if status == "queued":
            job["status"] = "cancelled"
            job["completed_at"] = job["updated_at"]
            job["message"] = "Ontology rebuild가 시작 전에 취소되었습니다."
        if _bump_ontology_rebuild_job_version_locked(job, previous_snapshot):
            ontology_rebuild_jobs_condition.notify_all()
        return dict(job)


def _enqueue_ontology_rebuild_job(
    *,
    kb_name: str,
    internal_kb_id: str,
    user_id: str,
    include_llm: bool,
    retry_of_job_id: str = "",
    chunk_ids: Optional[List[int]] = None,
    trigger: str = "manual_rebuild",
    source_type: str = "",
    source_path: str = "",
    parser_signature: str = "",
    query_id: str = "",
    saved_answer_id: int = 0,
    llm_fact_status: str = "",
) -> Dict[str, Any]:
    job = _create_ontology_rebuild_job(
        kb_name=kb_name,
        internal_kb_id=internal_kb_id,
        user_id=user_id,
        include_llm=include_llm,
        retry_of_job_id=retry_of_job_id,
        chunk_ids=chunk_ids,
        trigger=trigger,
        source_type=source_type,
        source_path=source_path,
        parser_signature=parser_signature,
        query_id=query_id,
        saved_answer_id=saved_answer_id,
        llm_fact_status=llm_fact_status,
    )
    try:
        ontology_rebuild_queue.put_nowait(
            {
                "job_id": job["job_id"],
                "kb_name": kb_name,
                "internal_kb_id": internal_kb_id,
                "user_id": user_id,
                "include_llm": include_llm,
                "chunk_ids": list(job.get("chunk_ids", []) or []),
                "trigger": str(job.get("trigger", "") or ""),
                "source_type": str(job.get("source_type", "") or ""),
                "source_path": str(job.get("source_path", "") or ""),
                "parser_signature": str(job.get("parser_signature", "") or ""),
                "query_id": str(job.get("query_id", "") or ""),
                "saved_answer_id": int(job.get("saved_answer_id", 0) or 0),
                "llm_fact_status": str(job.get("llm_fact_status", "") or ""),
            }
        )
    except queue.Full:
        _update_ontology_rebuild_job(
            job["job_id"],
            status="error",
            message="Ontology rebuild 대기열이 가득 차 작업을 시작하지 못했습니다.",
            failure_code="ontology_rebuild_queue_full",
        )
    return _get_ontology_rebuild_job(job["job_id"]) or job


def _run_ontology_rebuild_job(task: Dict[str, Any]):
    job_id = str(task.get("job_id", "") or "")
    internal_kb_id = str(task.get("internal_kb_id", "") or task.get("kb_name", "default") or "default")
    include_llm = bool(task.get("include_llm", False))
    llm_fact_status = str(task.get("llm_fact_status", "") or "")
    requested_chunk_ids = [int(value) for value in (task.get("chunk_ids", []) or []) if int(value) > 0]
    if not job_id:
        return
    current = _get_ontology_rebuild_job(job_id)
    if current and bool(current.get("cancel_requested", False)):
        _update_ontology_rebuild_job(job_id, status="cancelled", message="Ontology rebuild가 시작 전에 취소되었습니다.")
        return
    _update_ontology_rebuild_job(
        job_id,
        status="processing",
        message="Ontology rebuild 대상 chunk를 확인하는 중입니다.",
        progress_percent=2,
    )
    lease_context = None
    try:
        lease_context = rag_registry.lease(internal_kb_id, _create_rag_engine)
        rag = lease_context.__enter__()
        conn = rag._connect_db()
        if requested_chunk_ids:
            placeholders = ",".join("?" for _ in requested_chunk_ids)
            rows = conn.execute(
                f"SELECT id FROM chunks WHERE COALESCE(is_normalized, 0) = 0 AND COALESCE(is_derived, 0) = 0 AND id IN ({placeholders}) ORDER BY id ASC",
                tuple(requested_chunk_ids),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id FROM chunks WHERE COALESCE(is_normalized, 0) = 0 ORDER BY id ASC").fetchall()
        conn.close()
        chunk_ids = [int(row[0]) for row in rows]
        total = len(chunk_ids)
        _update_ontology_rebuild_job(
            job_id,
            chunks_total=total,
            message=f"Ontology rebuild 대상 chunk {total}개를 처리합니다.",
            progress_percent=5 if total else 100,
        )
        aggregate = {
            "ontology_facts_added": 0,
            "ontology_facts_deleted": 0,
            "ontology_extraction_errors": 0,
            "ontology_extraction_disabled": False,
            "ontology_extraction_disabled_reason": "",
        }
        if not chunk_ids:
            _update_ontology_rebuild_job(
                job_id,
                status="success",
                message="Ontology rebuild 대상 chunk가 없습니다.",
                progress_percent=100,
                **aggregate,
            )
            rag.query_cache.clear()
            return
        for index, chunk_id in enumerate(chunk_ids, start=1):
            current = _get_ontology_rebuild_job(job_id)
            if current and bool(current.get("cancel_requested", False)):
                _update_ontology_rebuild_job(
                    job_id,
                    status="cancelled",
                    message="Ontology rebuild가 취소되었습니다.",
                    chunks_processed=max(0, index - 1),
                    progress_percent=int(((index - 1) / max(1, total)) * 100),
                    **aggregate,
                )
                rag.query_cache.clear()
                return
            try:
                summary = rag._sync_ontology_facts(
                    changed_chunk_ids=[chunk_id],
                    deleted_chunk_ids=[chunk_id],
                    include_llm=include_llm,
                    llm_fact_status=llm_fact_status,
                )
            except Exception as exc:
                aggregate["ontology_extraction_errors"] += 1
                print(f"[ONTOLOGY][REBUILD][WARN] job_id={job_id} chunk_id={chunk_id} error={exc}", file=sys.stderr)
                summary = {}
            aggregate["ontology_facts_added"] += int(summary.get("ontology_facts_added", 0) or 0)
            aggregate["ontology_facts_deleted"] += int(summary.get("ontology_facts_deleted", 0) or 0)
            aggregate["ontology_extraction_errors"] += int(summary.get("ontology_extraction_errors", 0) or 0)
            if bool(summary.get("ontology_extraction_disabled", False)):
                aggregate["ontology_extraction_disabled"] = True
                aggregate["ontology_extraction_disabled_reason"] = str(
                    summary.get("ontology_extraction_disabled_reason", "") or "ONTOLOGY_LLM_EXTRACTION_ENABLED=0"
                )
            _update_ontology_rebuild_job(
                job_id,
                status="processing",
                message=f"Ontology rebuild 진행 중입니다. {index}/{total} chunks",
                chunks_processed=index,
                progress_percent=max(5, min(99, int((index / max(1, total)) * 100))),
                **aggregate,
            )
        rag.query_cache.clear()
        _update_ontology_rebuild_job(
            job_id,
            status="success",
            message="Ontology rebuild가 끝났습니다.",
            chunks_processed=total,
            progress_percent=100,
            **aggregate,
        )
    except Exception as exc:
        failure_code = _classify_failure_code(exc, default="ontology_rebuild_fail")
        print(f"[ONTOLOGY][REBUILD][ERROR] job_id={job_id} failure_code={failure_code} error={exc}", file=sys.stderr)
        _update_ontology_rebuild_job(
            job_id,
            status="error",
            message=str(exc),
            failure_code=failure_code,
        )
    finally:
        if lease_context is not None:
            lease_context.__exit__(None, None, None)


def _enqueue_document_upload_ontology_job(
    *,
    kb_name: str,
    user_id: str,
    stored_path: str,
    ingest_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED or not DOCUMENT_UPLOAD_ONTOLOGY_LLM_JOB_ENABLED:
        return None
    source_type = Path(str(stored_path or "")).suffix.lower().lstrip(".")
    if source_type not in {"hwpx", "xlsx"}:
        return None
    chunk_ids = [int(value) for value in list((ingest_result or {}).get("ontology_chunk_ids", []) or []) if int(value) > 0]
    if not chunk_ids:
        return None
    _ensure_upload_workers()
    return _enqueue_ontology_rebuild_job(
        kb_name=kb_name,
        internal_kb_id=kb_name,
        user_id=user_id,
        include_llm=True,
        chunk_ids=list((ingest_result or {}).get("ontology_chunk_ids", []) or []),
        trigger="document_upload",
        source_type=source_type,
        source_path=str((ingest_result or {}).get("source_path", "") or Path(stored_path).name),
        parser_signature=str((ingest_result or {}).get("parser_signature", "") or ""),
    )


def _enqueue_reported_answer_ontology_job(
    *,
    kb_name: str,
    internal_kb_id: str,
    user_id: str,
    query_id: str,
    saved_answer_id: int,
    chunk_ids: List[int],
) -> Optional[Dict[str, Any]]:
    if not REPORTED_ANSWER_ONTOLOGY_RECHECK_ENABLED:
        return None
    bounded_chunk_ids = [int(value) for value in list(chunk_ids or []) if int(value) > 0]
    bounded_chunk_ids = list(dict.fromkeys(bounded_chunk_ids))[:ONTOLOGY_REPORTED_MAX_CHUNKS]
    if not bounded_chunk_ids:
        return None
    with reported_answer_ontology_lock:
        existing = _find_reported_answer_ontology_job(
            kb_name=kb_name,
            user_id=user_id,
            query_id=query_id,
        )
        if existing is not None:
            return existing
        _ensure_upload_workers()
        return _enqueue_ontology_rebuild_job(
            kb_name=kb_name,
            internal_kb_id=internal_kb_id,
            user_id=user_id,
            include_llm=True,
            chunk_ids=bounded_chunk_ids,
            trigger="reported_answer",
            query_id=query_id,
            saved_answer_id=int(saved_answer_id or 0),
            llm_fact_status="needs_review",
        )


def _enqueue_background_ocr_after_upload(
    *,
    upload_job_id: str,
    kb_name: str,
    stored_path: str,
    original_filename: str,
    stored_filename: str,
    document_role: str,
    ingest_result: Dict[str, Any],
    user_id: str = "",
) -> Optional[Dict[str, Any]]:
    if not PDF_BACKGROUND_OCR_ENABLED:
        return None
    if not stored_path.lower().endswith(".pdf"):
        return None
    target_pages = int((ingest_result or {}).get("pdf_attempted_ocr_pages", 0) or 0)
    warnings = set(str(item or "") for item in list((ingest_result or {}).get("pdf_warnings", []) or []))
    if target_pages <= 0 and "lazy_ocr_deferred" not in warnings:
        return None
    job = _create_ocr_job(
        kb_name=kb_name,
        original_filename=original_filename,
        stored_filename=stored_filename,
        stored_path=stored_path,
        document_role=document_role,
        upload_job_id=upload_job_id,
        user_id=user_id,
        total_pages=int((ingest_result or {}).get("pdf_total_pages", 0) or 0),
        ocr_target_pages=target_pages,
    )
    try:
        ocr_queue.put_nowait(
            {
                "job_id": job["job_id"],
                "kb_name": kb_name,
                "user_id": user_id or "",
                "stored_path": stored_path,
                "original_filename": original_filename,
                "document_role": document_role,
            }
        )
    except queue.Full:
        _update_ocr_job(
            job["job_id"],
            status="error",
            progress_stage="error",
            progress_percent=0,
            message="OCR 보강 대기열이 가득 차 작업을 시작하지 못했습니다.",
            failure_code="ocr_queue_full",
        )
    return job


def _ingest_upload_job(task: Dict[str, Any]):
    job_id = task.get("job_id", "")
    kb_name = task.get("kb_name", "default")
    stored_path = task.get("stored_path", "")
    original_filename = task.get("original_filename", "")
    stored_filename = task.get("stored_filename", "")
    document_role = _normalize_doc_role(str(task.get("document_role", "")))
    pdf_ocr_mode = str(task.get("pdf_ocr_mode", "") or "").strip().lower().replace("-", "_")
    progress_state = {
        "percent": 0,
        "last_ocr_log_key": None,
        "stage": "queued",
        "ocr_completed": False,
        "index_completed": False,
        "embedding_started_at": 0,
        "embedding_completed_at": 0,
    }

    def update_progress(percent: int, message: str, stage: str = "processing", **extra_progress_updates: Any):
        normalized = _normalize_upload_progress_percent(
            percent,
            fallback=progress_state["percent"],
        )
        if normalized < progress_state["percent"]:
            normalized = progress_state["percent"]
        progress_state["percent"] = normalized
        progress_state.update(
            update_upload_phase_state(
                progress_state,
                stage or "processing",
                now_ts=int(time.time()),
                progress_meta=extra_progress_updates,
            )
        )
        progress_updates = _extract_progress_page_stats(message)
        for key, value in dict(extra_progress_updates or {}).items():
            if value is None:
                continue
            progress_updates[key] = value
        _update_upload_job(
            job_id,
            status="processing",
            message=message,
            progress_percent=normalized,
            progress_stage=(stage or "processing"),
            ocr_completed=bool(progress_state["ocr_completed"]),
            index_completed=bool(progress_state["index_completed"]),
            embedding_started_at=int(progress_state["embedding_started_at"] or 0),
            embedding_completed_at=int(progress_state["embedding_completed_at"] or 0),
            phase_started_at=int(progress_state.get("phase_started_at", 0) or 0),
            phase_last_heartbeat_at=int(progress_state.get("phase_last_heartbeat_at", 0) or 0),
            phase_elapsed_seconds=int(progress_state.get("phase_elapsed_seconds", 0) or 0),
            phase_name_effective=str(progress_state.get("phase_name_effective", "") or ""),
            phase_rows_total=int(progress_state.get("phase_rows_total", 0) or 0),
            phase_rows_done=int(progress_state.get("phase_rows_done", 0) or 0),
            phase_chunks_total=int(progress_state.get("phase_chunks_total", 0) or 0),
            phase_chunks_done=int(progress_state.get("phase_chunks_done", 0) or 0),
            embed_batch=int(progress_state.get("embed_batch", 0) or 0),
            embed_batches=int(progress_state.get("embed_batches", 0) or 0),
            embed_rows_done=int(progress_state.get("embed_rows_done", 0) or 0),
            embed_rows_total=int(progress_state.get("embed_rows_total", 0) or 0),
            embed_input_tokens_total=int(progress_state.get("embed_input_tokens_total", 0) or 0),
            embed_input_tokens_done=int(progress_state.get("embed_input_tokens_done", 0) or 0),
            embed_input_tokens_p95=int(progress_state.get("embed_input_tokens_p95", 0) or 0),
            embed_input_tokens_max=int(progress_state.get("embed_input_tokens_max", 0) or 0),
            embed_truncated_rows=int(progress_state.get("embed_truncated_rows", 0) or 0),
            embed_effective_batch_tokens=int(progress_state.get("embed_effective_batch_tokens", 0) or 0),
            **progress_updates,
        )
        if (stage or "").strip().lower() in {"load_pdf_ocr_model", "run_pdf_ocr", "fallback_pdf_ocr", "merge_pdf_ocr"}:
            log_key = (
                str(stage or "").strip().lower(),
                int(progress_updates.get("current_page", 0) or 0),
                int(progress_updates.get("total_pages", 0) or 0),
                int(progress_updates.get("ocr_completed_pages", 0) or 0),
                int(progress_updates.get("ocr_target_pages", 0) or 0),
            )
            if log_key != progress_state["last_ocr_log_key"]:
                progress_state["last_ocr_log_key"] = log_key
                print(
                    f"[UPLOAD][OCR_PROGRESS] kb={kb_name} file={original_filename} "
                    f"stage={(stage or 'processing').strip().lower()} "
                    f"pdf_pages={int(progress_updates.get('current_page', 0) or 0)}/{int(progress_updates.get('total_pages', 0) or 0)} "
                    f"ocr_pages={int(progress_updates.get('ocr_completed_pages', 0) or 0)}/{int(progress_updates.get('ocr_target_pages', 0) or 0)} "
                    f"retry_mode={str(progress_updates.get('ocr_retry_mode', '') or '').strip() or '-'} "
                    f"retry_reason={str(progress_updates.get('ocr_retry_reason', '') or '').strip() or '-'} "
                    f"percent={normalized} message={message}"
                )

    update_progress(5, "업로드한 문서를 확인하는 중입니다.", "preparing")

    try:
        update_progress(10, "지식베이스를 준비하는 중입니다.", "prepare_kb")
        with rag_registry.lease(kb_name, _create_rag_engine) as rag:
            ingest_result = rag.ingest_file(
                stored_path,
                original_filename=original_filename,
                document_role=document_role,
                progress_callback=update_progress,
                pdf_ocr_mode=pdf_ocr_mode,
            )
        used_cache = bool((ingest_result or {}).get("used_cache", False))
        chunks = int((ingest_result or {}).get("chunks", 0))
        replaced = int((ingest_result or {}).get("replaced_chunks", 0))
        normalized_chunks = int((ingest_result or {}).get("normalized_chunks", chunks))
        ontology_job = _enqueue_document_upload_ontology_job(
            kb_name=kb_name,
            user_id=str(task.get("user_id", "") or ""),
            stored_path=stored_path,
            ingest_result=ingest_result or {},
        )
        message = "문서 정리가 끝났습니다."
        completed_at = int(time.time())
        progress_state.update(update_upload_phase_state(progress_state, "done", now_ts=completed_at))
        _update_upload_job(
            job_id,
            status="success",
            message=message,
            used_cache=used_cache,
            chunks=chunks,
            replaced_chunks=replaced,
            normalized_chunks=normalized_chunks,
            ontology_job_id=str((ontology_job or {}).get("job_id", "") or ""),
            ontology_status=str((ontology_job or {}).get("status", "") or ""),
            document_role=document_role,
            progress_percent=100,
            progress_stage="done",
            ocr_completed=True,
            index_completed=True,
            embedding_started_at=int(progress_state["embedding_started_at"] or completed_at),
            embedding_completed_at=completed_at,
            pdf_parser=str((ingest_result or {}).get("pdf_parser", "") or ""),
            total_pages=int((ingest_result or {}).get("pdf_total_pages", 0) or 0),
            pdf_total_pages=int((ingest_result or {}).get("pdf_total_pages", 0) or 0),
            pdf_text_pages=int((ingest_result or {}).get("pdf_text_pages", 0) or 0),
            pdf_ocr_pages=int((ingest_result or {}).get("pdf_ocr_pages", 0) or 0),
            ocr_target_pages=int((ingest_result or {}).get("pdf_attempted_ocr_pages", 0) or 0),
            ocr_completed_pages=int((ingest_result or {}).get("pdf_ocr_pages", 0) or 0),
            pdf_failed_pages=int((ingest_result or {}).get("pdf_failed_pages", 0) or 0),
            pdf_warnings=list((ingest_result or {}).get("pdf_warnings", []) or []),
            ocr_device_attempted=str((ingest_result or {}).get("ocr_device_attempted", "") or "").strip()
            or str(os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip(),
            ocr_device_effective=str((ingest_result or {}).get("ocr_device_effective", "") or "").strip()
            or str((ingest_result or {}).get("ocr_device_attempted", "") or "").strip()
            or str(os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip(),
            ocr_gpu_fallback_used=bool((ingest_result or {}).get("ocr_gpu_fallback_used", False)),
            ocr_gpu_failure_reason=str((ingest_result or {}).get("ocr_gpu_failure_reason", "") or "").strip(),
            ocr_elapsed_seconds=float((ingest_result or {}).get("ocr_elapsed_seconds", 0.0) or 0.0),
            ocr_pages_processed=int((ingest_result or {}).get("ocr_pages_processed", 0) or 0),
            ocr_pages_per_minute=float((ingest_result or {}).get("ocr_pages_per_minute", 0.0) or 0.0),
            ocr_pages_attempted=int((ingest_result or {}).get("ocr_pages_attempted", 0) or 0),
            ocr_pages_emitted=int((ingest_result or {}).get("ocr_pages_emitted", 0) or 0),
            ocr_pages_skipped_empty=int((ingest_result or {}).get("ocr_pages_skipped_empty", 0) or 0),
            ocr_pages_skipped_short_text=int((ingest_result or {}).get("ocr_pages_skipped_short_text", 0) or 0),
            ocr_attempted_pages_per_minute=float((ingest_result or {}).get("ocr_attempted_pages_per_minute", 0.0) or 0.0),
            ocr_emitted_pages_per_minute=float((ingest_result or {}).get("ocr_emitted_pages_per_minute", 0.0) or 0.0),
            ocr_worker_released=bool((ingest_result or {}).get("ocr_worker_released", False)),
            ocr_worker_release_seconds=float((ingest_result or {}).get("ocr_worker_release_seconds", 0.0) or 0.0),
            ocr_worker_pids=list((ingest_result or {}).get("ocr_worker_pids", []) or []),
            ocr_worker_shutdown_confirmed=bool((ingest_result or {}).get("ocr_worker_shutdown_confirmed", True)),
            ocr_worker_alive_after_shutdown=list((ingest_result or {}).get("ocr_worker_alive_after_shutdown", []) or []),
            ocr_duration_seconds=float((ingest_result or {}).get("ocr_duration_seconds", 0.0) or 0.0),
            persist_duration_seconds=float((ingest_result or {}).get("persist_duration_seconds", 0.0) or 0.0),
            embedding_duration_seconds=float((ingest_result or {}).get("embedding_duration_seconds", 0.0) or 0.0),
            index_duration_seconds=float((ingest_result or {}).get("index_duration_seconds", 0.0) or 0.0),
            derived_sync_duration_seconds=float(
                (ingest_result or {}).get("derived_sync_duration_seconds", 0.0) or 0.0
            ),
            ocr_subset_build_seconds=float((ingest_result or {}).get("ocr_subset_build_seconds", 0.0) or 0.0),
            ocr_model_load_seconds=float((ingest_result or {}).get("ocr_model_load_seconds", 0.0) or 0.0),
            ocr_predict_seconds=float((ingest_result or {}).get("ocr_predict_seconds", 0.0) or 0.0),
            ocr_output_materialize_seconds=float((ingest_result or {}).get("ocr_output_materialize_seconds", 0.0) or 0.0),
            ocr_payload_convert_seconds=float((ingest_result or {}).get("ocr_payload_convert_seconds", 0.0) or 0.0),
            ocr_fragment_collect_seconds=float((ingest_result or {}).get("ocr_fragment_collect_seconds", 0.0) or 0.0),
            ocr_page_dedupe_seconds=float((ingest_result or {}).get("ocr_page_dedupe_seconds", 0.0) or 0.0),
            ocr_page_join_seconds=float((ingest_result or {}).get("ocr_page_join_seconds", 0.0) or 0.0),
            ocr_text_merge_seconds=float((ingest_result or {}).get("ocr_text_merge_seconds", 0.0) or 0.0),
            ocr_merge_seconds=float((ingest_result or {}).get("ocr_merge_seconds", 0.0) or 0.0),
            ocr_batch_count=int((ingest_result or {}).get("ocr_batch_count", 0) or 0),
            ocr_backend=str((ingest_result or {}).get("ocr_backend", "") or "").strip(),
            ocr_backend_attempted=str((ingest_result or {}).get("ocr_backend_attempted", "") or "").strip(),
            ocr_backend_effective=str((ingest_result or {}).get("ocr_backend_effective", "") or "").strip(),
            ocr_backend_fallback_used=bool((ingest_result or {}).get("ocr_backend_fallback_used", False)),
            ocr_fast_pages=int((ingest_result or {}).get("ocr_fast_pages", 0) or 0),
            ocr_vl_pages=int((ingest_result or {}).get("ocr_vl_pages", 0) or 0),
            ocr_fast_seconds=float((ingest_result or {}).get("ocr_fast_seconds", 0.0) or 0.0),
            ocr_vl_seconds=float((ingest_result or {}).get("ocr_vl_seconds", 0.0) or 0.0),
            ocr_fast_avg_score=float((ingest_result or {}).get("ocr_fast_avg_score", 0.0) or 0.0),
            ocr_fast_pair_ratio=float((ingest_result or {}).get("ocr_fast_pair_ratio", 0.0) or 0.0),
            ocr_fast_orphan_ratio=float((ingest_result or {}).get("ocr_fast_orphan_ratio", 0.0) or 0.0),
            ocr_high_quality_requested=bool((ingest_result or {}).get("ocr_high_quality_requested", False)),
            ocr_target_pages_goal=int((ingest_result or {}).get("ocr_target_pages", 0) or 0),
            ocr_target_seconds=float((ingest_result or {}).get("ocr_target_seconds", 0.0) or 0.0),
            ocr_target_met=bool((ingest_result or {}).get("ocr_target_met", False)),
            ocr_stall_detected=False,
            current_page=int((ingest_result or {}).get("pdf_total_pages", 0) or 0),
            failure_code="",
        )
        _enqueue_background_ocr_after_upload(
            upload_job_id=job_id,
            kb_name=kb_name,
            stored_path=stored_path,
            original_filename=original_filename,
            stored_filename=stored_filename,
            document_role=document_role,
            ingest_result=ingest_result or {},
            user_id=str(task.get("user_id", "") or ""),
        )
    except Exception as e:
        failure_default = upload_failure_default_for_stage(
            str(progress_state.get("stage", "") or ""),
            ocr_completed=bool(progress_state.get("ocr_completed", False)),
        )
        failure_code = _classify_failure_code(e, default=failure_default)
        runtime_info = dict(getattr(e, "runtime_info", {}) or {})
        print(
            f"[UPLOAD][ERROR] kb={kb_name} file={original_filename} failure_code={failure_code} error={e}",
            file=sys.stderr,
        )
        _update_upload_job(
            job_id,
            status="error",
            message=str(e),
            progress_percent=progress_state["percent"],
            progress_stage="error",
            failure_code=failure_code,
            ocr_completed=bool(progress_state.get("ocr_completed", False)),
            index_completed=False,
            embedding_started_at=int(progress_state.get("embedding_started_at", 0) or 0),
            embedding_completed_at=int(progress_state.get("embedding_completed_at", 0) or 0),
            ocr_device_attempted=str(runtime_info.get("ocr_device_attempted", "") or "").strip()
            or str(os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip(),
            ocr_device_effective=str(runtime_info.get("ocr_device_effective", "") or "").strip()
            or str(runtime_info.get("ocr_device_attempted", "") or "").strip()
            or str(os.getenv("PDF_OCR_DEVICE", "cpu") or "cpu").strip(),
            ocr_gpu_fallback_used=bool(runtime_info.get("ocr_gpu_fallback_used", False)),
            ocr_gpu_failure_reason=str(runtime_info.get("ocr_gpu_failure_reason", "") or "").strip(),
            ocr_elapsed_seconds=float(runtime_info.get("ocr_elapsed_seconds", 0.0) or 0.0),
            ocr_pages_processed=int(runtime_info.get("ocr_pages_processed", 0) or 0),
            ocr_pages_per_minute=float(runtime_info.get("ocr_pages_per_minute", 0.0) or 0.0),
            ocr_pages_attempted=int(runtime_info.get("ocr_pages_attempted", 0) or 0),
            ocr_pages_emitted=int(runtime_info.get("ocr_pages_emitted", 0) or 0),
            ocr_pages_skipped_empty=int(runtime_info.get("ocr_pages_skipped_empty", 0) or 0),
            ocr_pages_skipped_short_text=int(runtime_info.get("ocr_pages_skipped_short_text", 0) or 0),
            ocr_attempted_pages_per_minute=float(runtime_info.get("ocr_attempted_pages_per_minute", 0.0) or 0.0),
            ocr_emitted_pages_per_minute=float(runtime_info.get("ocr_emitted_pages_per_minute", 0.0) or 0.0),
            ocr_worker_released=bool(runtime_info.get("ocr_worker_released", False)),
            ocr_worker_release_seconds=float(runtime_info.get("ocr_worker_release_seconds", 0.0) or 0.0),
            ocr_worker_pids=list(runtime_info.get("ocr_worker_pids", []) or []),
            ocr_worker_shutdown_confirmed=bool(runtime_info.get("ocr_worker_shutdown_confirmed", True)),
            ocr_worker_alive_after_shutdown=list(runtime_info.get("ocr_worker_alive_after_shutdown", []) or []),
            ocr_duration_seconds=float(runtime_info.get("ocr_duration_seconds", 0.0) or 0.0),
            persist_duration_seconds=float(runtime_info.get("persist_duration_seconds", 0.0) or 0.0),
            embedding_duration_seconds=float(runtime_info.get("embedding_duration_seconds", 0.0) or 0.0),
            index_duration_seconds=float(runtime_info.get("index_duration_seconds", 0.0) or 0.0),
            derived_sync_duration_seconds=float(runtime_info.get("derived_sync_duration_seconds", 0.0) or 0.0),
            ocr_subset_build_seconds=float(runtime_info.get("ocr_subset_build_seconds", 0.0) or 0.0),
            ocr_model_load_seconds=float(runtime_info.get("ocr_model_load_seconds", 0.0) or 0.0),
            ocr_predict_seconds=float(runtime_info.get("ocr_predict_seconds", 0.0) or 0.0),
            ocr_output_materialize_seconds=float(runtime_info.get("ocr_output_materialize_seconds", 0.0) or 0.0),
            ocr_payload_convert_seconds=float(runtime_info.get("ocr_payload_convert_seconds", 0.0) or 0.0),
            ocr_fragment_collect_seconds=float(runtime_info.get("ocr_fragment_collect_seconds", 0.0) or 0.0),
            ocr_page_dedupe_seconds=float(runtime_info.get("ocr_page_dedupe_seconds", 0.0) or 0.0),
            ocr_page_join_seconds=float(runtime_info.get("ocr_page_join_seconds", 0.0) or 0.0),
            ocr_text_merge_seconds=float(runtime_info.get("ocr_text_merge_seconds", 0.0) or 0.0),
            ocr_merge_seconds=float(runtime_info.get("ocr_merge_seconds", 0.0) or 0.0),
            ocr_batch_count=int(runtime_info.get("ocr_batch_count", 0) or 0),
            ocr_backend=str(runtime_info.get("ocr_backend", "") or "").strip(),
            ocr_backend_attempted=str(runtime_info.get("ocr_backend_attempted", "") or "").strip(),
            ocr_backend_effective=str(runtime_info.get("ocr_backend_effective", "") or "").strip(),
            ocr_backend_fallback_used=bool(runtime_info.get("ocr_backend_fallback_used", False)),
            ocr_fast_pages=int(runtime_info.get("ocr_fast_pages", 0) or 0),
            ocr_vl_pages=int(runtime_info.get("ocr_vl_pages", 0) or 0),
            ocr_fast_seconds=float(runtime_info.get("ocr_fast_seconds", 0.0) or 0.0),
            ocr_vl_seconds=float(runtime_info.get("ocr_vl_seconds", 0.0) or 0.0),
            ocr_fast_avg_score=float(runtime_info.get("ocr_fast_avg_score", 0.0) or 0.0),
            ocr_fast_pair_ratio=float(runtime_info.get("ocr_fast_pair_ratio", 0.0) or 0.0),
            ocr_fast_orphan_ratio=float(runtime_info.get("ocr_fast_orphan_ratio", 0.0) or 0.0),
            ocr_high_quality_requested=bool(runtime_info.get("ocr_high_quality_requested", False)),
            ocr_target_pages_goal=int(runtime_info.get("ocr_target_pages", 0) or 0),
            ocr_target_seconds=float(runtime_info.get("ocr_target_seconds", 0.0) or 0.0),
            ocr_target_met=bool(runtime_info.get("ocr_target_met", False)),
        )


def _run_background_ocr_job(task: Dict[str, Any]):
    job_id = str(task.get("job_id", "") or "")
    kb_name = str(task.get("kb_name", "default") or "default").strip() or "default"
    stored_path = str(task.get("stored_path", "") or "").strip()
    original_filename = str(task.get("original_filename", "") or "").strip()
    document_role = _normalize_doc_role(str(task.get("document_role", "") or ""))
    progress_state = {"percent": 0}

    def update_progress(percent: int, message: str, stage: str = "processing", **extra_progress_updates: Any):
        normalized = _normalize_upload_progress_percent(
            percent,
            fallback=progress_state["percent"],
        )
        updates = _extract_progress_page_stats(message)
        updates.update({key: value for key, value in dict(extra_progress_updates or {}).items() if value is not None})
        normalized = estimate_background_ocr_progress_percent(
            {
                "status": "processing",
                "progress_stage": stage or "processing",
                "progress_percent": normalized,
                **updates,
            }
        )
        if normalized < progress_state["percent"]:
            normalized = progress_state["percent"]
        progress_state["percent"] = normalized
        _update_ocr_job(
            job_id,
            status="processing",
            message=message,
            progress_percent=normalized,
            progress_stage=stage or "processing",
            **updates,
        )

    if not job_id:
        return
    update_progress(5, "OCR 보강을 준비하는 중입니다.", "preparing")
    try:
        with rag_registry.lease(kb_name, _create_rag_engine) as rag:
            result = rag.ingest_file(
                stored_path,
                original_filename=original_filename,
                document_role=document_role,
                progress_callback=update_progress,
                force_pdf_ocr=True,
            )
        status = str((result or {}).get("status", "") or "").strip().lower()
        if status == "empty":
            _update_ocr_job(
                job_id,
                status="skipped",
                message="OCR 보강 결과에서 추가로 색인할 텍스트를 찾지 못했습니다.",
                progress_percent=100,
                progress_stage="done",
                failure_code="ocr_empty",
            )
            return
        _update_ocr_job(
            job_id,
            status="success",
            message="OCR 보강이 끝났고 검색 인덱스에 반영되었습니다.",
            progress_percent=100,
            progress_stage="done",
            current_page=int((result or {}).get("pdf_total_pages", 0) or 0),
            total_pages=int((result or {}).get("pdf_total_pages", 0) or 0),
            ocr_target_pages=int((result or {}).get("pdf_attempted_ocr_pages", 0) or 0),
            ocr_completed_pages=int((result or {}).get("pdf_ocr_pages", 0) or 0),
            failure_code="",
        )
    except Exception as e:
        failure_code = _classify_failure_code(e, default="background_ocr_fail")
        print(
            f"[BACKGROUND_OCR][ERROR] kb={kb_name} file={original_filename} failure_code={failure_code} error={e}",
            file=sys.stderr,
        )
        _update_ocr_job(
            job_id,
            status="error",
            message=str(e),
            progress_percent=progress_state["percent"],
            progress_stage="error",
            failure_code=failure_code,
        )


def _upload_lane_for_extension(ext: str) -> str:
    normalized_ext = str(ext or "").strip().lower()
    if normalized_ext in FAST_LANE_UPLOAD_EXTENSIONS:
        return "fast"
    return "pdf"


def _upload_queue_for_extension(ext: str) -> "queue.Queue[Optional[Dict[str, Any]]]":
    if _upload_lane_for_extension(ext) == "fast":
        return upload_fast_queue
    return upload_pdf_queue


def _next_upload_job_version(job: Dict[str, Any]) -> int:
    return max(1, int(job.get("version", 1) or 1) + 1)


def _mark_persisted_upload_recovery_error(job: Dict[str, Any], *, failure_code: str, message: str):
    now = int(time.time())
    job_id = str(job.get("job_id", "") or "").strip()
    if not job_id:
        return
    upload_job_store.update_job(
        job_id,
        {
            "status": "error",
            "progress_stage": "error",
            "failure_code": failure_code,
            "message": message,
            "completed_at": now,
            "updated_at": now,
            "version": _next_upload_job_version(job),
        },
    )


def _recover_persisted_upload_jobs_on_startup(limit: int = UPLOAD_JOB_RECOVERY_LIMIT) -> Dict[str, int]:
    stats = {"requeued": 0, "missing_file": 0, "queue_full": 0, "skipped": 0}
    if int(limit or 0) <= 0:
        return stats

    for job in upload_job_store.list_incomplete_jobs(limit=int(limit or 0)):
        job_id = str(job.get("job_id", "") or "").strip()
        if not job_id or job_id in recovered_upload_job_ids:
            stats["skipped"] += 1
            continue

        stored_path = str(job.get("stored_path", "") or "").strip()
        if not stored_path or not os.path.exists(stored_path):
            _mark_persisted_upload_recovery_error(
                job,
                failure_code="upload_requeue_missing_file",
                message="서버 재시작 후 업로드 파일을 찾지 못해 작업을 다시 시작하지 못했습니다.",
            )
            recovered_upload_job_ids.add(job_id)
            stats["missing_file"] += 1
            continue

        now = int(time.time())
        restored_job = dict(job)
        restored_job.update(
            {
                "status": "queued",
                "message": "서버 재시작 후 업로드 작업을 다시 대기열에 올렸습니다.",
                "progress_percent": 0,
                "progress_stage": "queued",
                "failure_code": "",
                "queued_at": now,
                "processing_started_at": 0,
                "completed_at": 0,
                "last_progress_at": now,
                "updated_at": now,
                "version": _next_upload_job_version(job),
            }
        )
        task = {
            "job_id": job_id,
            "kb_name": str(restored_job.get("kb_name", "default") or "default"),
            "user_id": str(restored_job.get("user_id", "") or ""),
            "original_filename": str(restored_job.get("original_filename", "") or ""),
            "stored_filename": str(restored_job.get("stored_filename", "") or ""),
            "stored_path": stored_path,
            "document_role": str(restored_job.get("document_role", "") or ""),
            "upload_lane": _upload_lane_for_extension(os.path.splitext(stored_path)[1].lower()),
        }
        selected_queue = _upload_queue_for_extension(os.path.splitext(stored_path)[1].lower())
        with upload_jobs_condition:
            if job_id in upload_jobs:
                stats["skipped"] += 1
                continue
            upload_jobs[job_id] = restored_job
            try:
                upload_job_store.save_job(restored_job)
            except Exception as e:
                print(f"[UPLOAD][WARN] persist_requeue_failed job_id={job_id} error={e}", file=sys.stderr)
        try:
            selected_queue.put_nowait(task)
        except queue.Full:
            with upload_jobs_condition:
                upload_jobs.pop(job_id, None)
            _mark_persisted_upload_recovery_error(
                restored_job,
                failure_code="upload_requeue_full",
                message="서버 재시작 후 업로드 대기열이 가득 차 작업을 다시 시작하지 못했습니다.",
            )
            recovered_upload_job_ids.add(job_id)
            stats["queue_full"] += 1
            continue

        recovered_upload_job_ids.add(job_id)
        stats["requeued"] += 1
    if any(stats.values()):
        print(
            "[UPLOAD][RECOVERY] "
            f"requeued={stats['requeued']} missing_file={stats['missing_file']} "
            f"queue_full={stats['queue_full']} skipped={stats['skipped']}"
        )
    return stats


def _upload_worker_loop(work_queue: "queue.Queue[Optional[Dict[str, Any]]]", lane_name: str):
    while not upload_shutdown_event.is_set():
        try:
            task = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            work_queue.task_done()
            break

        try:
            _ingest_upload_job(task)
        finally:
            work_queue.task_done()


def _ocr_worker_loop():
    while not ocr_shutdown_event.is_set():
        try:
            task = ocr_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            ocr_queue.task_done()
            break

        try:
            _run_background_ocr_job(task)
        finally:
            ocr_queue.task_done()


def _ontology_rebuild_worker_loop():
    while not ontology_rebuild_shutdown_event.is_set():
        try:
            task = ontology_rebuild_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            ontology_rebuild_queue.task_done()
            break

        try:
            _run_ontology_rebuild_job(task)
        finally:
            ontology_rebuild_queue.task_done()


def _ensure_upload_workers():
    global upload_workers_started
    with upload_workers_lock:
        if upload_workers_started:
            return
        upload_shutdown_event.clear()
        ocr_shutdown_event.clear()
        ontology_rebuild_shutdown_event.clear()
        for idx in range(UPLOAD_WORKER_COUNT):
            t = threading.Thread(
                target=_upload_worker_loop,
                args=(upload_pdf_queue, "pdf"),
                name=f"upload-pdf-worker-{idx + 1}",
                daemon=True,
            )
            t.start()
            upload_workers.append(t)
        for idx in range(UPLOAD_FAST_WORKER_COUNT):
            t = threading.Thread(
                target=_upload_worker_loop,
                args=(upload_fast_queue, "fast"),
                name=f"upload-fast-worker-{idx + 1}",
                daemon=True,
            )
            t.start()
            upload_workers.append(t)
        for idx in range(PDF_BACKGROUND_OCR_WORKER_COUNT):
            t = threading.Thread(
                target=_ocr_worker_loop,
                name=f"background-ocr-worker-{idx + 1}",
                daemon=True,
            )
            t.start()
            ocr_workers.append(t)
        for idx in range(ONTOLOGY_REBUILD_WORKER_COUNT):
            t = threading.Thread(
                target=_ontology_rebuild_worker_loop,
                name=f"ontology-rebuild-worker-{idx + 1}",
                daemon=True,
            )
            t.start()
            ontology_rebuild_workers.append(t)
        upload_workers_started = True

def _create_rag_engine(kb_id: str) -> RAGEngine:
    return RAGEngine(kb_id=kb_id, data_dir=str(KB_DATA_DIR))


def get_rag(kb_id: str) -> Optional[RAGEngine]:
    """Get or load RAG engine for a specific KB."""
    existing = rag_registry.get(kb_id)
    if existing is not None:
        return existing

    if kb_id in list_kbs(data_dir=str(KB_DATA_DIR)):
        try:
            print(f"Loading RAG for {kb_id}...")
            return rag_registry.get_or_create(
                kb_id,
                _create_rag_engine,
            )
        except Exception as e:
            failure_code = _classify_failure_code(e, default="rag_load_fail")
            print(f"Error loading RAG for {kb_id}: failure_code={failure_code} error={e}", file=sys.stderr)
            return None
    return None

def create_rag(kb_id: str) -> RAGEngine:
    """Create a new RAG engine (and directory)."""
    try:
        print(f"Creating RAG for {kb_id}...")
        return rag_registry.get_or_create(
            kb_id,
            _create_rag_engine,
        )
    except Exception as e:
        failure_code = _classify_failure_code(e, default="rag_create_fail")
        print(f"Error creating RAG for {kb_id}: failure_code={failure_code} error={e}", file=sys.stderr)
        raise e


def _get_wiki_store_for_rag(rag: RAGEngine) -> WikiStore:
    return WikiStore(str(rag.db_path))


def _get_wiki_memory_store_for_rag(rag: RAGEngine) -> WikiMemoryStore:
    return WikiMemoryStore(str(rag.db_path))


def _get_wiki_page_builder_for_rag(rag: RAGEngine) -> WikiPageBuilder:
    return WikiPageBuilder(_get_wiki_store_for_rag(rag))


def _get_ontology_store_for_rag(rag: RAGEngine) -> OntologyStore:
    return OntologyStore(str(rag.db_path), kb_id=rag.kb_id)


def _feature_disabled_response(feature_name: str) -> JSONResponse:
    return _error_json_response(
        status_code=404,
        message=f"{feature_name} 기능은 현재 비활성화되어 있습니다.",
        failure_code="feature_disabled",
    )


def _wiki_space_payload(kb_record: Mapping[str, Any], wiki_store: WikiStore) -> Dict[str, Any]:
    display_name = str(kb_record.get("display_name", "") or kb_record.get("name", "") or "").strip()
    internal_id = str(kb_record.get("internal_kb_id", "") or display_name or "default").strip()
    return {
        "space": {
            "display_name": display_name or internal_id,
            "space_id": internal_id,
            "scope": "kb",
            "isolation": "wiki content is scoped to this guide space",
        },
        "space_summary": wiki_store.space_summary(),
    }


def ensure_kb_directory(kb_id: str) -> Path:
    kb_dir = KB_DATA_DIR / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "uploads").mkdir(parents=True, exist_ok=True)
    return kb_dir

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _warmup_pdf_ocr_worker_on_startup() -> None:
    if not _env_bool("PDF_OCR_WARMUP_ON_STARTUP", True):
        return
    try:
        from src.pdf_ocr import warmup_persistent_ocr_worker

        info = warmup_persistent_ocr_worker(
            model_name=(os.getenv("PDF_OCR_MODEL_NAME", "") or "").strip() or None,
        )
        print(
            "[PDF_OCR][WARMUP] "
            f"status={info.get('status', '')} "
            f"backend={info.get('backend', '')} "
            f"execution_scope={info.get('execution_scope', 'worker_process')} "
            f"device={info.get('device', '')} "
            f"worker_count={int(info.get('worker_count', 0) or 0)} "
            f"worker_pids={','.join(str(pid) for pid in list(info.get('worker_pids', []) or [])) or '-'} "
            f"model_load_seconds={float(info.get('model_load_seconds', 0.0) or 0.0):.3f}"
        )
    except Exception as e:
        print(f"[PDF_OCR][WARMUP] status=error error={e}", file=sys.stderr)


@app.on_event("startup")
async def startup_event():
    _print_startup_config_summary()
    try:
        auth_store.prune_expired_sessions(limit=AUTH_SESSION_PRUNE_LIMIT)
    except Exception as exc:
        print(f"[AUTH][WARN] expired_session_prune_failed error={exc}", file=sys.stderr)
    _warmup_pdf_ocr_worker_on_startup()
    _ensure_upload_workers()
    _recover_persisted_upload_jobs_on_startup()
    # Pre-load 'default' if it exists, or create it?
    # Let's not force create 'default' unless user wants.
    # But for backward compatibility/initial run, maybe create 'default'.
    if not list_kbs(data_dir=str(KB_DATA_DIR)):
        try:
            create_rag("default")
            print("[STARTUP] default_kb_bootstrap=ok")
        except Exception as e:
            failure_code = _classify_failure_code(e, default="default_kb_init_fail")
            print(
                f"[STARTUP][WARN] default_kb_bootstrap=failed failure_code={failure_code} error={e}",
                file=sys.stderr,
            )
    admin_user = auth_store.get_user_by_login(AUTH_BOOTSTRAP_ADMIN_LOGIN)
    if admin_user and str(admin_user.get("role", "") or "") == "admin":
        try:
            _sync_legacy_kbs_for_admin(admin_user)
        except Exception as e:
            print(f"[STARTUP][WARN] legacy_kb_admin_registration=failed error={e}", file=sys.stderr)


@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] service=backend status=begin", flush=True)
    upload_shutdown_event.set()
    ocr_shutdown_event.set()
    ontology_rebuild_shutdown_event.set()
    try:
        from src.pdf_ocr import shutdown_persistent_ocr_worker

        shutdown_persistent_ocr_worker()
    except Exception as e:
        print(f"[PDF_OCR][WARMUP] shutdown=error error={e}", file=sys.stderr)
    for _ in range(max(1, UPLOAD_WORKER_COUNT)):
        try:
            upload_pdf_queue.put_nowait(None)
        except queue.Full:
            break
    for _ in range(max(1, UPLOAD_FAST_WORKER_COUNT)):
        try:
            upload_fast_queue.put_nowait(None)
        except queue.Full:
            break
    for _ in range(max(1, PDF_BACKGROUND_OCR_WORKER_COUNT)):
        try:
            ocr_queue.put_nowait(None)
        except queue.Full:
            break
    for _ in range(max(1, ONTOLOGY_REBUILD_WORKER_COUNT)):
        try:
            ontology_rebuild_queue.put_nowait(None)
        except queue.Full:
            break
    print("[SHUTDOWN] service=backend status=complete", flush=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "service": "compasslm-backend"}


@app.post("/auth/login")
async def login_endpoint(request: Request):
    data, error_response = await _read_request_json_object(
        request,
        failure_code="auth_validation_fail",
    )
    if error_response is not None:
        return error_response
    login_id = str(data.get("login_id", "") or "").strip()
    password = str(data.get("password", "") or "")
    user = auth_store.authenticate(login_id, password)
    if not user:
        return _error_json_response(
            status_code=401,
            message="아이디 또는 비밀번호가 올바르지 않습니다.",
            failure_code="auth_invalid_credentials",
        )
    token = auth_store.create_session(
        str(user["user_id"]),
        ttl_seconds=AUTH_SESSION_TTL_SECONDS,
        ip_address=str(request.client.host if request.client else ""),
        user_agent=str(request.headers.get("user-agent", "") or ""),
    )
    _sync_legacy_kbs_for_admin_best_effort(user)
    response = JSONResponse(content={"status": "success", "user": _public_user(user)})
    response.set_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=AUTH_SESSION_TTL_SECONDS,
    )
    return response


@app.post("/auth/register")
async def register_endpoint(request: Request):
    data, error_response = await _read_request_json_object(
        request,
        failure_code="auth_register_invalid_json",
    )
    if error_response is not None:
        return error_response
    login_id = str(data.get("login_id", "") or "").strip()
    display_name = str(data.get("display_name", "") or "").strip()
    password = str(data.get("password", "") or "")
    password_confirm = str(data.get("password_confirm", "") or "")
    if not login_id:
        return _error_json_response(
            status_code=400,
            message="아이디를 입력해 주세요.",
            failure_code="auth_login_id_required",
        )
    if not password:
        return _error_json_response(
            status_code=400,
            message="비밀번호를 입력해 주세요.",
            failure_code="auth_password_required",
        )
    if password != password_confirm:
        return _error_json_response(
            status_code=400,
            message="비밀번호가 일치하지 않습니다.",
            failure_code="auth_password_mismatch",
        )
    try:
        user = auth_store.create_user(
            login_id,
            password,
            display_name=display_name or login_id,
            role="user",
        )
    except ValueError as exc:
        if "login_id already exists" in str(exc):
            return _error_json_response(
                status_code=409,
                message="이미 사용 중인 아이디입니다.",
                failure_code="auth_login_id_exists",
            )
        return _error_json_response(
            status_code=400,
            message="계정을 만들 수 없습니다.",
            failure_code="auth_register_invalid_input",
        )
    user.pop("password_hash", None)
    token = auth_store.create_session(
        str(user["user_id"]),
        ttl_seconds=AUTH_SESSION_TTL_SECONDS,
        ip_address=str(request.client.host if request.client else ""),
        user_agent=str(request.headers.get("user-agent", "") or ""),
    )
    response = JSONResponse(content={"status": "success", "user": _public_user(user)})
    response.set_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=AUTH_SESSION_TTL_SECONDS,
    )
    return response


@app.post("/auth/logout")
async def logout_endpoint(request: Request):
    token = request.cookies.get(AUTH_SESSION_COOKIE_NAME, "")
    if token:
        auth_store.revoke_session(token)
    response = JSONResponse(content={"status": "success"})
    response.delete_cookie(key=AUTH_SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/auth/me")
async def me_endpoint(request: Request):
    user = _current_user_from_request(request)
    if not user:
        return _auth_required_response()
    _sync_legacy_kbs_for_admin_best_effort(user)
    return {"user": _public_user(user)}


@app.get("/kbs")
async def get_kbs_endpoint(request: Request):
    """List all Knowledge Bases."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    _sync_legacy_kbs_for_admin_best_effort(user)
    rows = auth_store.list_kbs(str(user["user_id"]))
    return {
        "kbs": [row["display_name"] for row in rows],
        "kb_records": [_public_kb_record(row) for row in rows],
    }


@app.get("/chat/history")
async def get_chat_history_endpoint(request: Request, kb_name: str = "default"):
    """Return SQLite-backed chat history for the current browser session."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _auth_required_response() if not user else _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    session_id = _resolve_session_id(request)
    session = _ensure_chat_session(session_id)
    payload = {
        "history_enabled": bool(session.get("history_enabled", False)),
        "messages": _get_chat_history(
            session_id,
            str(kb_record["internal_kb_id"]),
            user_id=str(user["user_id"]),
        ),
    }
    response = JSONResponse(content=payload)
    return _attach_session_cookie(response, session_id)

# TEMP: 관리자 QA 평가/로그 기능 엔드포인트.
@app.post("/feedback")
async def submit_feedback_endpoint(request: Request):
    """Store admin answer-quality feedback as JSONL logs."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    session_id = _resolve_session_id(request)
    data, error_response = await _read_request_json_object(
        request,
        failure_code="feedback_validation_fail",
    )
    if error_response is not None:
        return _attach_session_cookie(
            error_response,
            session_id,
        )

    question = _truncate_text(data.get("question", ""), ADMIN_FEEDBACK_MAX_TEXT_CHARS)
    answer = _truncate_text(data.get("answer", ""), ADMIN_FEEDBACK_MAX_TEXT_CHARS)
    expected_answer = _truncate_text(data.get("expected_answer", ""), ADMIN_FEEDBACK_MAX_TEXT_CHARS)
    kb_name = _truncate_text(data.get("kb_name", "default"), 120) or "default"
    is_correct = data.get("is_correct", None)

    if not isinstance(is_correct, bool):
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="is_correct는 true/false여야 합니다.",
                failure_code="feedback_validation_fail",
            ),
            session_id,
        )
    if not question:
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="question은 필수입니다.",
                failure_code="feedback_validation_fail",
            ),
            session_id,
        )
    if not answer:
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="answer는 필수입니다.",
                failure_code="feedback_validation_fail",
            ),
            session_id,
        )
    if (not is_correct) and (not expected_answer):
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="X 평가일 때 expected_answer가 필요합니다.",
                failure_code="feedback_validation_fail",
            ),
            session_id,
        )

    now = datetime.now(timezone.utc)
    log_entry = {
        "timestamp_utc": now.isoformat(),
        "timestamp_unix": int(now.timestamp()),
        "session_id": session_id,
        "kb_name": kb_name,
        "is_correct": is_correct,
        "question": question,
        "answer": answer,
        "expected_answer": expected_answer if (not is_correct) else "",
    }
    try:
        _append_admin_feedback_log(log_entry)
    except Exception as e:
        return _attach_session_cookie(
            _error_json_response(
                status_code=500,
                message=f"피드백 로그 저장 실패: {str(e)}",
                failure_code=_classify_failure_code(e, default="feedback_log_fail"),
            ),
            session_id,
        )

    return _attach_session_cookie(
        JSONResponse(
            content={
                "status": "success",
                "log_file": str(ADMIN_FEEDBACK_LOG_PATH),
                "timestamp_utc": log_entry["timestamp_utc"],
            }
        ),
        session_id,
    )


@app.get("/ops/failure-patterns")
async def get_ops_failure_patterns_endpoint(request: Request, kb_name: str = "default", limit: int = 120):
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    session_id = _resolve_session_id(request)
    review = await _build_ops_failure_review(
        kb_name=str(kb_record["internal_kb_id"]),
        limit=max(20, min(int(limit or 120), 300)),
    )
    return _attach_session_cookie(JSONResponse(content=review), session_id)


@app.get("/ops/wiki-lint")
async def get_ops_wiki_lint_endpoint(request: Request, kb_name: str = "default"):
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    session_id = _resolve_session_id(request)
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    findings = wiki_store.run_lint()
    memory_findings = wiki_memory_store.run_lint()
    return _attach_session_cookie(
        JSONResponse(
            content={
                "status": "success",
                "kb_name": kb_name,
                "finding_count": len(findings) + len(memory_findings),
                "findings": findings,
                "answer_memory_findings": memory_findings,
            }
        ),
        session_id,
    )


@app.get("/kbs/{kb_name}/files")
async def get_kb_files_endpoint(kb_name: str, request: Request):
    """List files in a specific KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    files = get_kb_files(str(kb_record["internal_kb_id"]), data_dir=str(KB_DATA_DIR))
    return {"files": files}


@app.get("/kbs/{kb_name}/ontology/entities")
async def list_kb_ontology_entities_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    """List ontology entities extracted for a KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    ontology_store = _get_ontology_store_for_rag(rag)
    safe_status = _truncate_text(status, 40)
    return {
        "entities": ontology_store.list_entities(
            status=safe_status,
            limit=max(1, min(int(limit or 100), 500)),
        )
    }


@app.get("/kbs/{kb_name}/ontology/facts")
async def list_kb_ontology_facts_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    """List ontology subject-predicate-object facts extracted for a KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    ontology_store = _get_ontology_store_for_rag(rag)
    safe_status = _truncate_text(status, 40)
    return {
        "facts": ontology_store.list_facts(
            status=safe_status,
            limit=max(1, min(int(limit or 100), 500)),
        )
    }


@app.get("/kbs/{kb_name}/ontology/search")
async def search_kb_ontology_endpoint(
    kb_name: str,
    request: Request,
    query: str,
    limit: int = 20,
    max_hops: int = 2,
):
    """Search ontology facts directly for operational verification."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    matches = _get_ontology_store_for_rag(rag).search_facts(
        query=_truncate_text(query, 1000),
        limit=max(1, min(int(limit or 20), 100)),
        max_hops=max(1, min(int(max_hops or 2), 2)),
        experiment_mode="operational_smoke",
    )
    return {"matches": matches}


@app.get("/kbs/{kb_name}/ontology/overview")
async def get_kb_ontology_overview_endpoint(kb_name: str, request: Request):
    """Summarize ontology graph size, status, and relation distribution."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    return {"overview": _get_ontology_store_for_rag(rag).overview()}


@app.post("/kbs/{kb_name}/ontology/rebuild")
async def rebuild_kb_ontology_endpoint(kb_name: str, request: Request, include_llm: bool = False):
    """Rebuild ontology facts from existing raw chunks."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    if include_llm:
        _ensure_upload_workers()
        job = _enqueue_ontology_rebuild_job(
            kb_name=str(kb_name or "default"),
            internal_kb_id=str(kb_record["internal_kb_id"]),
            user_id=str(user.get("user_id", "") or ""),
            include_llm=True,
        )
        return {
            "status": "accepted" if str(job.get("status", "")) != "error" else "error",
            "queued": str(job.get("status", "")) != "error",
            "ontology_rebuild_job_id": str(job.get("job_id", "")),
            "job": job,
        }
    else:
        summary = _get_ontology_store_for_rag(rag).rebuild_from_chunks()
    rag.query_cache.clear()
    return {"status": "success", "summary": summary}


@app.get("/kbs/{kb_name}/ontology/rebuild/jobs")
async def list_kb_ontology_rebuild_jobs_endpoint(kb_name: str, request: Request, include_terminal: bool = True):
    """List ontology rebuild jobs for this KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    return {
        "jobs": _list_ontology_rebuild_jobs(
            kb_name=str(kb_name or "default"),
            user_id=str(user.get("user_id", "") or ""),
            include_terminal=bool(include_terminal),
        )
    }


@app.get("/kbs/{kb_name}/ontology/rebuild/jobs/{job_id}")
async def get_kb_ontology_rebuild_job_endpoint(kb_name: str, job_id: str, request: Request):
    """Return one ontology rebuild job status."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    job = _get_ontology_rebuild_job(job_id)
    if not job or str(job.get("kb_name", "") or "") != str(kb_name or "default") or str(job.get("user_id", "") or "") != str(user.get("user_id", "") or ""):
        return _error_json_response(status_code=404, message="ontology rebuild job을 찾지 못했습니다.", failure_code="ontology_rebuild_job_not_found")
    return {"job": job}


@app.post("/kbs/{kb_name}/ontology/rebuild/jobs/{job_id}/cancel")
async def cancel_kb_ontology_rebuild_job_endpoint(kb_name: str, job_id: str, request: Request):
    """Request cancellation for an ontology rebuild job."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    existing = _get_ontology_rebuild_job(job_id)
    if not existing or str(existing.get("kb_name", "") or "") != str(kb_name or "default") or str(existing.get("user_id", "") or "") != str(user.get("user_id", "") or ""):
        return _error_json_response(status_code=404, message="ontology rebuild job을 찾지 못했습니다.", failure_code="ontology_rebuild_job_not_found")
    job = _request_cancel_ontology_rebuild_job(job_id)
    return {"status": "success", "job": job}


@app.post("/kbs/{kb_name}/ontology/rebuild/jobs/{job_id}/retry")
async def retry_kb_ontology_rebuild_job_endpoint(kb_name: str, job_id: str, request: Request):
    """Queue a retry for an ontology rebuild job."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    existing = _get_ontology_rebuild_job(job_id)
    if not existing or str(existing.get("kb_name", "") or "") != str(kb_name or "default") or str(existing.get("user_id", "") or "") != str(user.get("user_id", "") or ""):
        return _error_json_response(status_code=404, message="ontology rebuild job을 찾지 못했습니다.", failure_code="ontology_rebuild_job_not_found")
    _ensure_upload_workers()
    retry_job = _enqueue_ontology_rebuild_job(
        kb_name=str(kb_name or "default"),
        internal_kb_id=str(kb_record["internal_kb_id"]),
        user_id=str(user.get("user_id", "") or ""),
        include_llm=bool(existing.get("include_llm", False)),
        retry_of_job_id=str(job_id or ""),
        chunk_ids=list(existing.get("chunk_ids", []) or []),
        trigger=str(existing.get("trigger", "") or "manual_rebuild"),
        source_type=str(existing.get("source_type", "") or ""),
        source_path=str(existing.get("source_path", "") or ""),
        parser_signature=str(existing.get("parser_signature", "") or ""),
        query_id=str(existing.get("query_id", "") or ""),
        saved_answer_id=int(existing.get("saved_answer_id", 0) or 0),
        llm_fact_status=str(existing.get("llm_fact_status", "") or ""),
    )
    return {
        "status": "accepted" if str(retry_job.get("status", "")) != "error" else "error",
        "queued": str(retry_job.get("status", "")) != "error",
        "ontology_rebuild_job_id": str(retry_job.get("job_id", "")),
        "job": retry_job,
    }


@app.get("/kbs/{kb_name}/ontology/facts/{fact_id}")
async def get_kb_ontology_fact_detail_endpoint(kb_name: str, fact_id: int, request: Request):
    """Return one ontology fact with source/evidence records."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    fact = _get_ontology_store_for_rag(rag).get_fact_detail(int(fact_id))
    if fact is None:
        return _error_json_response(status_code=404, message="ontology fact를 찾지 못했습니다.", failure_code="ontology_fact_not_found")
    return {"fact": fact}


@app.post("/kbs/{kb_name}/ontology/facts/{fact_id}/publish")
async def publish_kb_ontology_fact_endpoint(kb_name: str, fact_id: int, request: Request):
    """Promote an ontology fact after admin review."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    fact = _get_ontology_store_for_rag(rag).update_fact_status(int(fact_id), "published")
    rag.query_cache.clear()
    if fact is None:
        return _error_json_response(status_code=404, message="ontology fact를 찾지 못했습니다.", failure_code="ontology_fact_not_found")
    return {"status": "success", "fact": fact}


@app.post("/kbs/{kb_name}/ontology/facts/{fact_id}/archive")
async def archive_kb_ontology_fact_endpoint(kb_name: str, fact_id: int, request: Request):
    """Archive an ontology fact after admin review."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    fact = _get_ontology_store_for_rag(rag).update_fact_status(int(fact_id), "archived")
    rag.query_cache.clear()
    if fact is None:
        return _error_json_response(status_code=404, message="ontology fact를 찾지 못했습니다.", failure_code="ontology_fact_not_found")
    return {"status": "success", "fact": fact}


@app.post("/kbs/{kb_name}/ontology/facts/{fact_id}/needs-review")
async def needs_review_kb_ontology_fact_endpoint(kb_name: str, fact_id: int, request: Request):
    """Mark an ontology fact as needing admin review."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    fact = _get_ontology_store_for_rag(rag).update_fact_status(int(fact_id), "needs_review")
    rag.query_cache.clear()
    if fact is None:
        return _error_json_response(status_code=404, message="ontology fact를 찾지 못했습니다.", failure_code="ontology_fact_not_found")
    return {"status": "success", "fact": fact}


@app.get("/kbs/{kb_name}/wiki")
async def get_kb_wiki_endpoint(kb_name: str, request: Request):
    """List compiled wiki pages in a specific KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    return {"pages": wiki_store.list_pages(), **_wiki_space_payload(kb_record, wiki_store)}


@app.get("/kbs/{kb_name}/wiki/pages/{slug:path}")
async def get_kb_wiki_page_endpoint(kb_name: str, slug: str, request: Request):
    """Fetch one compiled wiki page by slug."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    normalized_slug = (slug or "").strip("/")
    for page in wiki_store.list_pages():
        if str(page.get("slug", "") or "") == normalized_slug:
            return {"page": page}
    return _error_json_response(
        status_code=404,
        message="wiki 페이지를 찾지 못했습니다.",
        failure_code="wiki_page_not_found",
    )


@app.get("/kbs/{kb_name}/wiki/export")
async def export_kb_wiki_endpoint(kb_name: str, request: Request):
    """Export compiled wiki pages as markdown."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    files = wiki_store.export_markdown(
        space_name=str(kb_record.get("display_name", "") or kb_name),
        space_id=str(kb_record["internal_kb_id"]),
    )
    files.update(wiki_memory_store.export_markdown())
    return {"files": files, **_wiki_space_payload(kb_record, wiki_store)}


@app.post("/kbs/{kb_name}/wiki/save-answer")
async def save_kb_wiki_answer_endpoint(kb_name: str, request: Request):
    """Save a user-facing answer as a draft wiki claim page."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    data, error_response = await _read_request_json_object(
        request,
        failure_code="wiki_save_answer_invalid_request",
    )
    if error_response is not None:
        return error_response
    assert data is not None
    question = _truncate_text(data.get("question", ""), ADMIN_FEEDBACK_MAX_TEXT_CHARS)
    answer_text = _truncate_text(data.get("answer_text", data.get("answer", "")), ADMIN_FEEDBACK_MAX_TEXT_CHARS)
    if not answer_text.strip():
        return _error_json_response(
            status_code=400,
            message="저장할 답변이 비어 있습니다.",
            failure_code="wiki_save_answer_empty",
        )
    raw_citations = data.get("citations", [])
    citations = [dict(item) for item in raw_citations if isinstance(item, dict)] if isinstance(raw_citations, list) else []
    query_id = _truncate_text(data.get("query_id", ""), 120) or uuid.uuid4().hex
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    page = wiki_store.save_answer_page(
        query_id=query_id,
        question=question,
        answer_text=answer_text,
        citations=citations,
    )
    return {"status": "success", "page": page}


@app.get("/kbs/{kb_name}/wiki/answers")
async def list_kb_wiki_answers_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    """List user-approved wiki answer memory rows for a KB."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    safe_status = _truncate_text(status, 40)
    if safe_status and safe_status not in {"published", "needs_review", "reported", "archived"}:
        return _error_json_response(
            status_code=400,
            message="지원하지 않는 wiki answer 상태입니다.",
            failure_code="wiki_answer_status_invalid",
        )
    return {"answers": wiki_memory_store.list_saved_answers(limit=max(1, min(int(limit or 100), 500)), status=safe_status)}


@app.get("/kbs/{kb_name}/wiki/quality")
async def get_kb_wiki_quality_endpoint(kb_name: str, request: Request):
    """Summarize answer memory review state and quality flags."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"quality": wiki_memory_store.quality_summary()}


@app.get("/kbs/{kb_name}/wiki/page-candidates")
async def list_kb_wiki_page_candidates_endpoint(kb_name: str, request: Request, limit: int = 100):
    """List deterministic source-backed wiki page candidates."""
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"candidates": wiki_memory_store.build_wiki_page_candidates(limit=max(1, min(int(limit or 100), 500)))}


@app.get("/kbs/{kb_name}/wiki/overview")
async def get_kb_wiki_overview_endpoint(kb_name: str, request: Request):
    """Summarize compiled wiki pages, answer memory, and review queue counts."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_store = _get_wiki_store_for_rag(rag)
    pages = wiki_store.list_pages()
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag) if WIKI_ANSWER_MEMORY_ENABLED else None
    quality = wiki_memory_store.quality_summary() if wiki_memory_store is not None else {}
    candidates = wiki_memory_store.build_wiki_page_candidates(limit=100) if (wiki_memory_store is not None and WIKI_PAGE_WORKFLOW_ENABLED) else []
    space_payload = _wiki_space_payload(kb_record, wiki_store)
    return {
        "overview": {
            **space_payload,
            "page_count": len(pages),
            "candidate_count": len(candidates),
            "answer_memory": quality,
            "page_status_counts": dict(Counter(str(page.get("status", "") or "draft") for page in pages)),
        }
    }


@app.post("/kbs/{kb_name}/wiki/build-pages")
async def build_kb_wiki_pages_endpoint(kb_name: str, request: Request):
    """Build draft/needs_review wiki pages from deterministic candidates."""
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    builder = _get_wiki_page_builder_for_rag(rag)
    candidates = wiki_memory_store.build_wiki_page_candidates(limit=200)
    space_id = str(kb_record["internal_kb_id"])
    for candidate in candidates:
        provenance = dict(candidate.get("provenance", {}) or {})
        provenance["space_id"] = space_id
        provenance["space_display_name"] = str(kb_record.get("display_name", "") or kb_name)
        candidate["provenance"] = provenance
    pages = builder.build_pages(candidates)
    return {"status": "success", "pages": pages, "built_count": len(pages)}


@app.post("/kbs/{kb_name}/wiki/pages/{slug:path}/publish")
async def publish_kb_wiki_page_endpoint(kb_name: str, slug: str, request: Request):
    """Promote one source-backed wiki page for hint use."""
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    page = _get_wiki_store_for_rag(rag).update_page_status(slug, "published")
    if page is None:
        return _error_json_response(status_code=404, message="wiki 페이지를 찾지 못했습니다.", failure_code="wiki_page_not_found")
    return {"status": "success", "page": page}


@app.post("/kbs/{kb_name}/wiki/pages/{slug:path}/archive")
async def archive_kb_wiki_page_endpoint(kb_name: str, slug: str, request: Request):
    """Archive one wiki page so it no longer participates in hint boost."""
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    page = _get_wiki_store_for_rag(rag).update_page_status(slug, "archived")
    if page is None:
        return _error_json_response(status_code=404, message="wiki 페이지를 찾지 못했습니다.", failure_code="wiki_page_not_found")
    return {"status": "success", "page": page}


@app.get("/kbs/{kb_name}/wiki/lint")
async def list_kb_wiki_lint_endpoint(kb_name: str, request: Request, status: str = "open", limit: int = 100):
    """List KB-scoped wiki memory lint findings."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"findings": wiki_memory_store.list_lint_findings(status=_truncate_text(status, 40), limit=max(1, min(int(limit or 100), 500)))}


@app.post("/kbs/{kb_name}/wiki/lint/{finding_id}/resolve")
async def resolve_kb_wiki_lint_endpoint(kb_name: str, finding_id: int, request: Request):
    """Resolve one KB-scoped wiki memory lint finding."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    finding = wiki_memory_store.resolve_lint_finding(int(finding_id))
    if finding is None:
        return _error_json_response(
            status_code=404,
            message="wiki lint finding을 찾지 못했습니다.",
            failure_code="wiki_lint_finding_not_found",
        )
    return {"status": "success", "finding": finding}


@app.get("/kbs/{kb_name}/wiki/conflicts")
async def list_kb_wiki_conflicts_endpoint(kb_name: str, request: Request, status: str = "open", limit: int = 100):
    """List KB-scoped wiki answer memory conflicts."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"conflicts": wiki_memory_store.list_conflicts(status=_truncate_text(status, 40), limit=max(1, min(int(limit or 100), 500)))}


@app.post("/kbs/{kb_name}/wiki/conflicts/{conflict_id}/resolve")
async def resolve_kb_wiki_conflict_endpoint(kb_name: str, conflict_id: int, request: Request):
    """Resolve one KB-scoped wiki answer memory conflict."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    conflict = wiki_memory_store.resolve_conflict(int(conflict_id))
    if conflict is None:
        return _error_json_response(
            status_code=404,
            message="wiki conflict를 찾지 못했습니다.",
            failure_code="wiki_conflict_not_found",
        )
    return {"status": "success", "conflict": conflict}


@app.get("/kbs/{kb_name}/wiki/answers/{saved_answer_id}")
async def get_kb_wiki_answer_endpoint(kb_name: str, saved_answer_id: int, request: Request):
    """Fetch one saved wiki answer memory entry."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    answer = wiki_memory_store.get_saved_answer(int(saved_answer_id))
    if answer is None:
        return _error_json_response(
            status_code=404,
            message="저장된 wiki answer를 찾지 못했습니다.",
            failure_code="wiki_answer_not_found",
        )
    return {"answer": answer}


@app.patch("/kbs/{kb_name}/wiki/answers/{saved_answer_id}")
async def update_kb_wiki_answer_endpoint(kb_name: str, saved_answer_id: int, request: Request):
    """Update review status for a saved wiki answer memory entry."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    data, error_response = await _read_request_json_object(
        request,
        failure_code="wiki_answer_update_invalid_request",
    )
    if error_response is not None:
        return error_response
    assert data is not None
    status = _truncate_text(data.get("status", ""), 40)
    if status not in {"published", "needs_review", "reported", "archived"}:
        return _error_json_response(
            status_code=400,
            message="지원하지 않는 wiki answer 상태입니다.",
            failure_code="wiki_answer_status_invalid",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    answer = wiki_memory_store.update_saved_answer_status(int(saved_answer_id), status)
    if answer is None:
        return _error_json_response(
            status_code=404,
            message="저장된 wiki answer를 찾지 못했습니다.",
            failure_code="wiki_answer_not_found",
        )
    return {"status": "success", "answer": answer}


@app.post("/kbs/{kb_name}/wiki/answers/{saved_answer_id}/compile")
async def compile_kb_wiki_answer_endpoint(kb_name: str, saved_answer_id: int, request: Request):
    """Compile one saved answer into deterministic claim/concept/procedure/table-rule candidates."""
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    try:
        compiled = wiki_memory_store.compile_saved_answer(int(saved_answer_id))
    except LookupError:
        return _error_json_response(
            status_code=404,
            message="저장된 wiki answer를 찾지 못했습니다.",
            failure_code="wiki_answer_not_found",
        )
    return {"status": "success", "compiled": compiled}


@app.get("/kbs/{kb_name}/wiki/concepts")
async def list_kb_wiki_concepts_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"concepts": wiki_memory_store.list_concepts(status=_truncate_text(status, 40), limit=max(1, min(int(limit or 100), 500)))}


@app.get("/kbs/{kb_name}/wiki/procedures")
async def list_kb_wiki_procedures_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"procedures": wiki_memory_store.list_procedures(status=_truncate_text(status, 40), limit=max(1, min(int(limit or 100), 500)))}


@app.get("/kbs/{kb_name}/wiki/table-rules")
async def list_kb_wiki_table_rules_endpoint(kb_name: str, request: Request, status: str = "", limit: int = 100):
    if not WIKI_PAGE_WORKFLOW_ENABLED:
        return _feature_disabled_response("Wiki page workflow")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(status_code=404, message="지정한 공간을 찾지 못했습니다.", failure_code="knowledge_base_not_found")
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    return {"table_rules": wiki_memory_store.list_table_rules(status=_truncate_text(status, 40), limit=max(1, min(int(limit or 100), 500)))}


@app.post("/kbs/{kb_name}/wiki/answers/{saved_answer_id}/lint")
async def lint_kb_wiki_answer_endpoint(kb_name: str, saved_answer_id: int, request: Request):
    """Re-run answer memory lint and return findings for one saved answer."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    answer = wiki_memory_store.get_saved_answer(int(saved_answer_id))
    if answer is None:
        return _error_json_response(
            status_code=404,
            message="저장된 wiki answer를 찾지 못했습니다.",
            failure_code="wiki_answer_not_found",
        )
    findings = [
        finding
        for finding in wiki_memory_store.run_lint()
        if int(finding.get("target_id", 0) or 0) == int(saved_answer_id)
        or (
            str(finding.get("target_type", "") or "") == "answer_source"
            and any(
                int(source.get("answer_source_id", 0) or 0) == int(finding.get("target_id", 0) or 0)
                for source in answer.get("sources", [])
            )
        )
    ]
    return {"status": "success", "findings": findings}


@app.delete("/kbs/{kb_name}/wiki/answers/{saved_answer_id}")
async def delete_kb_wiki_answer_endpoint(kb_name: str, saved_answer_id: int, request: Request):
    """Delete a saved wiki answer memory entry."""
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    if str(user.get("role", "") or "") != "admin":
        return _admin_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    if not wiki_memory_store.delete_saved_answer(int(saved_answer_id)):
        return _error_json_response(
            status_code=404,
            message="저장된 wiki answer를 찾지 못했습니다.",
            failure_code="wiki_answer_not_found",
        )
    return {"status": "success"}


async def _save_answer_memory_feedback(
    *,
    kb_name: str,
    query_id: str,
    request: Request,
    feedback_type: str,
):
    if not WIKI_ANSWER_MEMORY_ENABLED:
        return _feature_disabled_response("Wiki answer memory")
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    rag = get_rag(str(kb_record["internal_kb_id"])) or create_rag(str(kb_record["internal_kb_id"]))
    wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
    try:
        saved = wiki_memory_store.save_answer_from_query_id(
            query_id=_truncate_text(query_id, 120),
            user_id=str(user.get("user_id", "") or ""),
            feedback_type=feedback_type,
        )
    except LookupError:
        return _error_json_response(
            status_code=404,
            message="저장할 답변 로그를 찾지 못했습니다.",
            failure_code="wiki_answer_log_not_found",
        )
    except ValueError:
        return _error_json_response(
            status_code=400,
            message="query_id가 필요합니다.",
            failure_code="wiki_answer_query_id_missing",
        )
    payload: Dict[str, Any] = {"status": "success", "answer": saved}
    if feedback_type == "report_citation_issue":
        ontology_job = None
        try:
            ontology_job = _enqueue_reported_answer_ontology_job(
                kb_name=str(kb_name or ""),
                internal_kb_id=str(kb_record["internal_kb_id"]),
                user_id=str(user.get("user_id", "") or ""),
                query_id=_truncate_text(query_id, 120),
                saved_answer_id=int(saved.get("saved_answer_id", 0) or 0),
                chunk_ids=list(saved.get("ontology_review_chunk_ids", []) or []),
            )
        except Exception as exc:
            print(
                f"[ONTOLOGY][REPORTED_ANSWER][WARN] kb={kb_name} query_id={query_id} error={exc}",
                file=sys.stderr,
            )
        payload["ontology_job_id"] = str((ontology_job or {}).get("job_id", "") or "")
        payload["ontology_status"] = str((ontology_job or {}).get("status", "") or "")
        payload["ontology_chunk_count"] = len(list(saved.get("ontology_review_chunk_ids", []) or []))
    return payload


@app.post("/kbs/{kb_name}/answers/{query_id}/save-to-wiki")
async def save_answer_to_wiki_memory_endpoint(kb_name: str, query_id: str, request: Request):
    """Promote the logged answer for query_id into citation-backed wiki answer memory."""
    return await _save_answer_memory_feedback(
        kb_name=kb_name,
        query_id=query_id,
        request=request,
        feedback_type="save_to_wiki",
    )


@app.post("/kbs/{kb_name}/answers/{query_id}/thumbs-up")
async def thumbs_up_answer_to_wiki_memory_endpoint(kb_name: str, query_id: str, request: Request):
    """Treat thumbs-up as positive feedback and save the answer to wiki memory."""
    return await _save_answer_memory_feedback(
        kb_name=kb_name,
        query_id=query_id,
        request=request,
        feedback_type="thumbs_up",
    )


@app.post("/kbs/{kb_name}/answers/{query_id}/report-citation-issue")
async def report_answer_citation_issue_endpoint(kb_name: str, query_id: str, request: Request):
    """Record that the logged answer needs citation review."""
    return await _save_answer_memory_feedback(
        kb_name=kb_name,
        query_id=query_id,
        request=request,
        feedback_type="report_citation_issue",
    )


@app.delete("/kbs/{kb_name}")
async def delete_kb_endpoint(kb_name: str, request: Request):
    """Delete a Knowledge Base."""
    if kb_name == "default":
        return _error_json_response(
            status_code=400,
            message="기본 공간은 삭제할 수 없습니다.",
            failure_code="kb_validation_fail",
        )
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    internal_kb_id = str(kb_record["internal_kb_id"])
    try:
        delete_kb_dir(internal_kb_id, data_dir=str(KB_DATA_DIR))
        auth_store.soft_delete_kb(str(user["user_id"]), kb_name)
        rag_registry.remove(internal_kb_id)
        return {"status": "success", "name": kb_name, "internal_kb_id": internal_kb_id}
    except Exception as e:
        return _error_json_response(
            status_code=500,
            message=str(e) or "공간 삭제 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="kb_delete_fail"),
        )

@app.put("/kbs/{kb_name}")
async def rename_kb_endpoint(kb_name: str, request: Request):
    """Rename a Knowledge Base."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    data, error_response = await _read_request_json_object(
        request,
        failure_code="kb_validation_fail",
    )
    if error_response is not None:
        return error_response
    new_name = data.get("new_name")
    if not isinstance(new_name, str):
        return _error_json_response(
            status_code=400,
            message="새 이름은 문자열이어야 합니다.",
            failure_code="kb_validation_fail",
        )
    new_name = new_name.strip()
    if not new_name:
        return _error_json_response(
            status_code=400,
            message="새 이름을 입력해 주세요.",
            failure_code="kb_validation_fail",
        )
    if ".." in new_name or "/" in new_name or "\\" in new_name:
        return _error_json_response(
            status_code=400,
            message="이름 형식이 올바르지 않습니다.",
            failure_code="kb_validation_fail",
        )

    try:
        renamed_record = auth_store.rename_kb(str(user["user_id"]), kb_name, new_name)
        if not renamed_record:
            return _error_json_response(
                status_code=404,
                message="지정한 공간을 찾지 못했습니다.",
                failure_code="knowledge_base_not_found",
            )
        return {"status": "success", "new_name": new_name, "kb_record": _public_kb_record(renamed_record)}
    except ValueError as e:
        message = str(e) or "공간 이름 변경 중 문제가 생겼습니다."
        if "already exists" in message:
            return _error_json_response(
                status_code=409,
                message="이미 같은 이름의 공간이 있습니다.",
                failure_code="kb_name_already_exists",
            )
        return _error_json_response(
            status_code=400,
            message="이름 형식이 올바르지 않습니다.",
            failure_code="kb_validation_fail",
        )
    except Exception as e:
        return _error_json_response(
            status_code=500,
            message=str(e) or "공간 이름 변경 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="kb_rename_fail"),
        )

@app.delete("/kbs/{kb_name}/files/{file_key}")
async def delete_file_endpoint(kb_name: str, file_key: str, request: Request):
    """Delete a file from a KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
            filename=file_key,
        )
    internal_kb_id = str(kb_record["internal_kb_id"])
    try:
        delete_file_from_kb(internal_kb_id, file_key, data_dir=str(KB_DATA_DIR))
        rag_registry.remove(internal_kb_id)
        return {"status": "success"}
    except Exception as e:
        return _error_json_response(
            status_code=500,
            message=str(e) or "파일 삭제 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="kb_file_delete_fail"),
            filename=file_key,
        )

@app.post("/kbs")
async def create_kb_endpoint(request: Request):
    """Create a new Knowledge Base."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    data, error_response = await _read_request_json_object(
        request,
        failure_code="kb_validation_fail",
    )
    if error_response is not None:
        return error_response
    kb_name = data.get("name")
    if not isinstance(kb_name, str):
        return _error_json_response(
            status_code=400,
            message="공간 이름은 문자열이어야 합니다.",
            failure_code="kb_validation_fail",
        )
    kb_name = kb_name.strip()
    if not kb_name:
        return _error_json_response(
            status_code=400,
            message="공간 이름을 입력해 주세요.",
            failure_code="kb_validation_fail",
        )

    # Simple validation for filenames
    if ".." in kb_name or "/" in kb_name or "\\" in kb_name:
        return _error_json_response(
            status_code=400,
            message="공간 이름 형식이 올바르지 않습니다.",
            failure_code="kb_validation_fail",
        )

    try:
        kb_record = auth_store.create_kb(str(user["user_id"]), kb_name)
        kb_dir = ensure_kb_directory(str(kb_record["internal_kb_id"]))
        kb_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "success",
            "name": kb_name,
            "kb_id": str(kb_record["kb_id"]),
            "internal_kb_id": str(kb_record["internal_kb_id"]),
            "kb_record": _public_kb_record(kb_record),
        }
    except Exception as e:
        return _error_json_response(
            status_code=500,
            message=str(e) or "공간 생성 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="kb_init_fail"),
        )

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    kb_name: str = Form("default"),
    document_role: str = Form(""),
    sync: str = Form(""),
    pdf_ocr_mode: str = Form(""),
):
    """Handle file upload to a specific KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_name = (kb_name or "default").strip()
    if not kb_name or ".." in kb_name or "/" in kb_name or "\\" in kb_name:
        return _error_json_response(
            status_code=400,
            message="공간 이름 형식이 올바르지 않습니다.",
            failure_code="upload_validation_fail",
        )
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    internal_kb_id = str(kb_record["internal_kb_id"])

    try:
        original_name = _validate_upload_meta(file)
    except Exception as e:
        return _error_json_response(
            status_code=400,
            message=str(e) or "파일 확인 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="upload_validation_fail"),
            filename=file.filename or "",
        )
    ext = os.path.splitext(original_name)[1].lower()
    try:
        resolved_doc_role = _resolve_upload_doc_role(document_role, filename=original_name, ext=ext)
    except ValueError as e:
        return _error_json_response(
            status_code=400,
            message=str(e),
            failure_code="upload_validation_fail",
            filename=original_name,
        )

    upload_dir = os.path.join(str(KB_DATA_DIR), internal_kb_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    stored_filename = _build_stored_upload_name(original_name)
    stored_path = os.path.join(upload_dir, stored_filename)

    try:
        saved_bytes = _save_upload_stream(file, stored_path, UPLOAD_MAX_BYTES)
    except Exception as e:
        if os.path.exists(stored_path):
            os.remove(stored_path)
        return _error_json_response(
            status_code=400,
            message=str(e) or "파일 저장 중 문제가 생겼습니다.",
            failure_code=_classify_failure_code(e, default="upload_save_fail"),
            filename=file.filename or "",
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    if ext == ".xlsx" and not _is_zip_signature(stored_path):
        os.remove(stored_path)
        return _error_json_response(
            status_code=400,
            message="엑셀 파일 형식이 올바르지 않습니다.",
            failure_code="upload_signature_fail",
            filename=original_name,
        )
    if ext == ".pdf" and not _is_pdf_signature(stored_path):
        os.remove(stored_path)
        return _error_json_response(
            status_code=400,
            message="PDF 파일 형식이 올바르지 않습니다.",
            failure_code="upload_signature_fail",
            filename=original_name,
        )
    if ext == ".hwpx" and not _is_hwpx_signature(stored_path):
        os.remove(stored_path)
        return _error_json_response(
            status_code=400,
            message="한글 HWPX 파일 형식이 올바르지 않습니다.",
            failure_code="upload_signature_fail",
            filename=original_name,
        )

    force_sync = str(sync or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    normalized_pdf_ocr_mode = str(pdf_ocr_mode or "").strip().lower().replace("-", "_")
    if normalized_pdf_ocr_mode not in {"", "fast", "high_quality", "highquality"}:
        os.remove(stored_path)
        return _error_json_response(
            status_code=400,
            message="pdf_ocr_mode must be fast or high_quality.",
            failure_code="upload_validation_fail",
            filename=original_name,
        )
    if normalized_pdf_ocr_mode == "highquality":
        normalized_pdf_ocr_mode = "high_quality"
    run_async = UPLOAD_ASYNC_ENABLED and not force_sync

    if run_async:
        _ensure_upload_workers()
        upload_lane = _upload_lane_for_extension(ext)
        selected_queue = _upload_queue_for_extension(ext)
        job = _create_upload_job(
            kb_name=internal_kb_id,
            original_filename=original_name,
            stored_filename=stored_filename,
            stored_path=stored_path,
            document_role=resolved_doc_role,
            user_id=str(user["user_id"]),
        )
        task = {
            "job_id": job["job_id"],
            "kb_name": internal_kb_id,
            "user_id": str(user["user_id"]),
            "original_filename": original_name,
            "stored_filename": stored_filename,
            "stored_path": stored_path,
            "saved_bytes": int(saved_bytes),
            "document_role": resolved_doc_role,
            "upload_lane": upload_lane,
            "pdf_ocr_mode": normalized_pdf_ocr_mode,
        }
        try:
            selected_queue.put_nowait(task)
        except queue.Full:
            with upload_jobs_lock:
                upload_jobs.pop(job["job_id"], None)
            os.remove(stored_path)
            return _error_json_response(
                status_code=503,
                message="현재 업로드 요청이 많습니다. 잠시 후 다시 시도해 주세요.",
                failure_code="upload_queue_full",
                filename=original_name,
            )

        return {
            "filename": original_name,
            "stored_filename": stored_filename,
            "status": "success",
            "queued": True,
            "job_id": job["job_id"],
            "version": int(job.get("version", 1) or 1),
            "message": "파일이 올라갔습니다. 문서 정리를 시작했습니다.",
            "size_bytes": int(saved_bytes),
            "document_role": resolved_doc_role,
            "document_role_label": _doc_role_label(resolved_doc_role),
            "pdf_ocr_mode": normalized_pdf_ocr_mode,
        }

    try:
        with rag_registry.lease(internal_kb_id, _create_rag_engine) as rag:
            ingest_result = rag.ingest_file(
                stored_path,
                original_filename=original_name,
                document_role=resolved_doc_role,
                pdf_ocr_mode=normalized_pdf_ocr_mode,
            )
        used_cache = bool((ingest_result or {}).get("used_cache", False))
        replaced = int((ingest_result or {}).get("replaced_chunks", 0))
        chunks = int((ingest_result or {}).get("chunks", 0))
        normalized_chunks = int((ingest_result or {}).get("normalized_chunks", chunks))

        if used_cache:
            msg = "이미 읽은 파일이라 빠르게 처리했습니다."
        else:
            msg = "문서 정리가 끝났습니다."
        if replaced > 0:
            msg += f" 같은 파일의 이전 내용 {replaced}개는 새 내용으로 바꿨습니다."

        _ensure_upload_workers()
        ontology_job = _enqueue_document_upload_ontology_job(
            kb_name=internal_kb_id,
            user_id=str(user["user_id"]),
            stored_path=stored_path,
            ingest_result=ingest_result or {},
        )
        ocr_job = _enqueue_background_ocr_after_upload(
            upload_job_id="",
            kb_name=internal_kb_id,
            stored_path=stored_path,
            original_filename=original_name,
            stored_filename=stored_filename,
            document_role=resolved_doc_role,
            ingest_result=ingest_result or {},
            user_id=str(user["user_id"]),
        )

        return {
            "filename": original_name,
            "stored_filename": stored_filename,
            "status": "success",
            "queued": False,
            "ocr_job_id": str((ocr_job or {}).get("job_id", "") or ""),
            "ontology_job_id": str((ontology_job or {}).get("job_id", "") or ""),
            "ontology_status": str((ontology_job or {}).get("status", "") or ""),
            "message": msg,
            "used_cache": used_cache,
            "chunks": chunks,
            "replaced_chunks": replaced,
            "normalized_chunks": normalized_chunks,
            "size_bytes": int(saved_bytes),
            "document_role": resolved_doc_role,
            "document_role_label": _doc_role_label(resolved_doc_role),
            "ocr_worker_released": bool((ingest_result or {}).get("ocr_worker_released", False)),
            "ocr_worker_release_seconds": float((ingest_result or {}).get("ocr_worker_release_seconds", 0.0) or 0.0),
            "ocr_worker_pids": list((ingest_result or {}).get("ocr_worker_pids", []) or []),
            "ocr_worker_shutdown_confirmed": bool((ingest_result or {}).get("ocr_worker_shutdown_confirmed", True)),
            "ocr_worker_alive_after_shutdown": list((ingest_result or {}).get("ocr_worker_alive_after_shutdown", []) or []),
            "ocr_duration_seconds": float((ingest_result or {}).get("ocr_duration_seconds", 0.0) or 0.0),
            "persist_duration_seconds": float((ingest_result or {}).get("persist_duration_seconds", 0.0) or 0.0),
            "embedding_duration_seconds": float((ingest_result or {}).get("embedding_duration_seconds", 0.0) or 0.0),
            "index_duration_seconds": float((ingest_result or {}).get("index_duration_seconds", 0.0) or 0.0),
            "derived_sync_duration_seconds": float(
                (ingest_result or {}).get("derived_sync_duration_seconds", 0.0) or 0.0
            ),
            "ocr_pages_attempted": int((ingest_result or {}).get("ocr_pages_attempted", 0) or 0),
            "ocr_pages_emitted": int((ingest_result or {}).get("ocr_pages_emitted", 0) or 0),
            "ocr_pages_skipped_empty": int((ingest_result or {}).get("ocr_pages_skipped_empty", 0) or 0),
            "ocr_pages_skipped_short_text": int((ingest_result or {}).get("ocr_pages_skipped_short_text", 0) or 0),
            "ocr_backend": str((ingest_result or {}).get("ocr_backend", "") or "").strip(),
            "ocr_backend_attempted": str((ingest_result or {}).get("ocr_backend_attempted", "") or "").strip(),
            "ocr_backend_effective": str((ingest_result or {}).get("ocr_backend_effective", "") or "").strip(),
            "ocr_backend_fallback_used": bool((ingest_result or {}).get("ocr_backend_fallback_used", False)),
            "ocr_fast_pages": int((ingest_result or {}).get("ocr_fast_pages", 0) or 0),
            "ocr_vl_pages": int((ingest_result or {}).get("ocr_vl_pages", 0) or 0),
            "ocr_fast_seconds": float((ingest_result or {}).get("ocr_fast_seconds", 0.0) or 0.0),
            "ocr_vl_seconds": float((ingest_result or {}).get("ocr_vl_seconds", 0.0) or 0.0),
            "ocr_fast_avg_score": float((ingest_result or {}).get("ocr_fast_avg_score", 0.0) or 0.0),
            "ocr_fast_pair_ratio": float((ingest_result or {}).get("ocr_fast_pair_ratio", 0.0) or 0.0),
            "ocr_fast_orphan_ratio": float((ingest_result or {}).get("ocr_fast_orphan_ratio", 0.0) or 0.0),
            "ocr_high_quality_requested": bool((ingest_result or {}).get("ocr_high_quality_requested", False)),
        }
    except Exception as e:
        return {
            "filename": original_name,
            "stored_filename": stored_filename,
            "status": "error",
            "queued": False,
            "message": str(e) or "문서 처리 중 문제가 생겼습니다.",
            "failure_code": _classify_failure_code(e, default="upload_ingest_fail"),
            "size_bytes": int(saved_bytes),
            "document_role": resolved_doc_role,
            "document_role_label": _doc_role_label(resolved_doc_role),
        }


@app.get("/upload/jobs/{job_id}")
async def get_upload_job_endpoint(
    request: Request,
    job_id: str,
    since_version: Optional[int] = None,
    wait_seconds: Optional[float] = None,
):
    """Get asynchronous upload job status."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    payload = _get_upload_job(job_id)
    if not payload:
        return _error_json_response(
            status_code=404,
            message="작업 정보를 찾지 못했습니다.",
            failure_code="upload_job_not_found",
        )
    if str(payload.get("user_id", "") or "") != str(user["user_id"]) and str(user.get("role", "") or "") != "admin":
        return _error_json_response(
            status_code=404,
            message="작업 정보를 찾지 못했습니다.",
            failure_code="upload_job_not_found",
        )
    normalized_since_version = _normalize_upload_job_version(since_version)
    normalized_wait_seconds = _clamp_upload_job_wait_seconds(wait_seconds)
    current_version = _normalize_upload_job_version(payload.get("version", 0))
    if normalized_since_version <= 0 or normalized_wait_seconds <= 0 or current_version != normalized_since_version:
        return payload
    wait_state = await asyncio.to_thread(
        _wait_for_upload_job_change,
        job_id,
        normalized_since_version,
        normalized_wait_seconds,
    )
    if wait_state == "timeout":
        return Response(status_code=204)
    payload = _get_upload_job(job_id)
    if not payload:
        return _error_json_response(
            status_code=404,
            message="작업 정보를 찾지 못했습니다.",
            failure_code="upload_job_not_found",
        )
    if str(payload.get("user_id", "") or "") != str(user["user_id"]) and str(user.get("role", "") or "") != "admin":
        return _error_json_response(
            status_code=404,
            message="작업 정보를 찾지 못했습니다.",
            failure_code="upload_job_not_found",
        )
    return payload


@app.get("/ocr/jobs")
async def list_ocr_jobs_endpoint(request: Request, kb_name: str = "default", include_terminal: bool = False):
    """List background OCR enrichment jobs for the current KB."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    kb_record = _resolve_user_kb(user, kb_name)
    if not kb_record:
        return _error_json_response(
            status_code=404,
            message="지정한 공간을 찾지 못했습니다.",
            failure_code="knowledge_base_not_found",
        )
    return {
        "jobs": [
            _get_ocr_job(str(job.get("job_id", "") or "")) or job
            for job in _list_ocr_jobs(
                kb_name=str(kb_record["internal_kb_id"]),
                include_terminal=include_terminal,
                user_id=str(user["user_id"]),
            )
        ]
    }


@app.get("/ocr/jobs/{job_id}")
async def get_ocr_job_endpoint(
    request: Request,
    job_id: str,
    since_version: Optional[int] = None,
    wait_seconds: Optional[float] = None,
):
    """Get background OCR enrichment job status."""
    user = _require_current_user(request)
    if not user:
        return _auth_required_response()
    payload = _get_ocr_job(job_id)
    if not payload:
        return _error_json_response(
            status_code=404,
            message="OCR 작업 정보를 찾지 못했습니다.",
            failure_code="ocr_job_not_found",
        )
    if str(payload.get("user_id", "") or "") != str(user["user_id"]) and str(user.get("role", "") or "") != "admin":
        return _error_json_response(
            status_code=404,
            message="OCR 작업 정보를 찾지 못했습니다.",
            failure_code="ocr_job_not_found",
        )
    normalized_since_version = _normalize_upload_job_version(since_version)
    normalized_wait_seconds = _clamp_upload_job_wait_seconds(wait_seconds)
    current_version = _normalize_upload_job_version(payload.get("version", 0))
    if normalized_since_version <= 0 or normalized_wait_seconds <= 0 or current_version != normalized_since_version:
        return payload
    wait_state = await asyncio.to_thread(
        _wait_for_ocr_job_change,
        job_id,
        normalized_since_version,
        normalized_wait_seconds,
    )
    if wait_state == "timeout":
        return Response(status_code=204)
    payload = _get_ocr_job(job_id)
    if not payload:
        return _error_json_response(
            status_code=404,
            message="OCR 작업 정보를 찾지 못했습니다.",
            failure_code="ocr_job_not_found",
        )
    if str(payload.get("user_id", "") or "") != str(user["user_id"]) and str(user.get("role", "") or "") != "admin":
        return _error_json_response(
            status_code=404,
            message="OCR 작업 정보를 찾지 못했습니다.",
            failure_code="ocr_job_not_found",
        )
    return payload


@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint."""
    session_id = _resolve_session_id(request)
    _ensure_chat_session(session_id)
    user = _require_current_user(request)
    if not user:
        return _attach_session_cookie(_auth_required_response(), session_id)
    user_id = str(user["user_id"])
    _request_user_id_context.set(user_id)
    data, error_response = await _read_request_json_object(
        request,
        failure_code="chat_validation_fail",
    )
    if error_response is not None:
        return _attach_session_cookie(error_response, session_id)
    raw_user_message = data.get("message", "")
    if not isinstance(raw_user_message, str):
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="message는 문자열이어야 합니다.",
                failure_code="chat_validation_fail",
            ),
            session_id,
        )
    raw_kb_name = data.get("kb_name", "default")
    if raw_kb_name is None:
        raw_kb_name = "default"
    if not isinstance(raw_kb_name, str):
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="kb_name은 문자열이어야 합니다.",
                failure_code="chat_validation_fail",
            ),
            session_id,
        )
    user_message = raw_user_message.strip()
    display_kb_name = (raw_kb_name or "default").strip() or "default"
    kb_record = _resolve_user_kb(user, display_kb_name)
    if not kb_record:
        return _attach_session_cookie(
            _error_json_response(
                status_code=404,
                message="지정한 공간을 찾지 못했습니다.",
                failure_code="knowledge_base_not_found",
            ),
            session_id,
        )
    kb_name = str(kb_record["internal_kb_id"])
    query_id = uuid.uuid4().hex
    rag_lease_state: Dict[str, Any] = {
        "context": None,
        "transferred_to_response": False,
    }

    def _release_rag_lease() -> None:
        lease_context = rag_lease_state.pop("context", None)
        if lease_context is not None:
            lease_context.__exit__(None, None, None)

    def _stream_response(text: str, headers: Optional[Dict[str, str]] = None):
        response = StreamingResponse(_single_chunk_stream(text), media_type="text/event-stream")
        if (
            rag_lease_state.get("context") is not None
            and not bool(rag_lease_state.get("transferred_to_response", False))
        ):
            rag_lease_state["transferred_to_response"] = True
            response.background = BackgroundTask(_release_rag_lease)
        response.headers["X-Query-Id"] = query_id
        for key, value in (headers or {}).items():
            if value:
                response.headers[key] = value
        return _attach_session_cookie(response, session_id)

    if not user_message:
        return _attach_session_cookie(
            _error_json_response(
                status_code=400,
                message="질문을 입력해 주세요.",
                failure_code="chat_validation_fail",
            ),
            session_id,
        )

    # Chat history controls: default ON, with explicit commands still allowed.
    if _is_history_disable_request(user_message):
        _set_history_enabled(session_id, False)
        _clear_chat_history(session_id, user_id=user_id)
        return _stream_response(
            "대화 저장을 껐고, 지금까지 저장된 대화도 모두 지웠습니다.",
            headers={"X-Conversation-Mode": "history_control"},
        )

    if _is_history_clear_request(user_message):
        _clear_chat_history(session_id, kb_name=kb_name, user_id=user_id)
        return _stream_response(
            f"'{display_kb_name}' 공간의 대화 내용을 지웠습니다.",
            headers={"X-Conversation-Mode": "history_control"},
        )

    if _is_history_enable_request(user_message):
        _set_history_enabled(session_id, True)
        ack = (
            "지금부터 대화를 저장합니다. "
            "새로고침하거나 서버를 다시 켜도 같은 브라우저 세션이면 이어집니다."
        )
        _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
        _append_chat_message(session_id, kb_name, "assistant", ack, user_id=user_id)
        return _stream_response(ack, headers={"X-Conversation-Mode": "history_control"})

    history_enabled = _is_history_enabled(session_id)
    rag: Optional[RAGEngine] = None
    kb_files = get_kb_files(kb_name, data_dir=str(KB_DATA_DIR))
    kb_has_docs = bool(kb_files)
    kb_file_count = len(kb_files)
    raw_history_rows = _get_chat_history(session_id, kb_name, user_id=user_id) if history_enabled else []
    history_rows = [
        row
        for row in raw_history_rows
        if not (str(row.get("role", "")).lower() == "assistant" and is_failed_history_answer_text(str(row.get("text", ""))))
    ]
    history_failed_turns_dropped = max(0, len(raw_history_rows) - len(history_rows))
    compact_history_success_turns = sum(1 for row in history_rows if str(row.get("role", "")).lower() == "assistant")
    recent_mode_state = (
        summarize_recent_conversation_state(
            chat_store.get_recent_agent_runs(
                session_id=session_id,
                kb_name=kb_name,
                user_id=user_id,
                limit=CHAT_AGENT_RUN_LIMIT,
            )
        )
        if history_enabled
        else None
    )
    rewrite_history_block = (
        compact_chat_history_rows(
            history_rows,
            turn_limit=ai_service.settings.compact_history_turn_limit,
            char_limit=ai_service.settings.compact_history_char_limit,
        )
        if history_rows
        else ""
    )
    effective_user_message = user_message
    followup_analysis = FollowupAnalysis()
    followup_diag = HelperRunDiagnostics(status="disabled")
    if rewrite_history_block and (
        should_attempt_followup_rewrite(user_message)
        or should_force_followup_rewrite(
            user_message,
            last_active_mode=(recent_mode_state.last_active_mode if recent_mode_state else None),
        )
    ):
        followup_analysis, followup_diag = await _llm_followup_rewrite(user_message, rewrite_history_block)
        effective_user_message = resolve_effective_query(user_message, followup_analysis)
        if followup_diag.status == "error":
            print(
                f"[FOLLOWUP_REWRITE][WARN] query_id={query_id} failure_code={followup_diag.failure_code or '-'} "
                f"error={followup_diag.error_detail or '-'}",
                file=sys.stderr,
            )

    # Fast-path for identity questions (no embedding/RAG).
    if _is_identity_question(user_message) and not _is_identity_with_extra_intent(user_message):
        identity_text = _identity_text()
        if history_enabled:
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                identity_text,
                b"[]",
                metadata={
                    "session_id": session_id,
                    "kb_name": kb_name,
                    "query_id": query_id,
                    "provider_kind": ai_service.provider_kind,
                    "provider_label": ai_service.provider_label,
                    "model_name": ai_service.model_name,
                    "history_strategy": ai_service.settings.history_strategy,
                    "conversation_mode": "identity",
                    "mode_resolution_reason": "identity_fast_path",
                    "mode_anchor_run_id": 0,
                    "casual_turn_streak": 0,
                    "upload_nudge_applied": False,
                },
                usage={},
                context_chars=0,
                user_id=user_id,
            )
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", identity_text, user_id=user_id)
        return _stream_response(identity_text, headers={"X-Conversation-Mode": "identity"})

    # Fast-path for greeting: skip embedding/RAG and answer with lightweight LLM response.
    if _is_pure_greeting(user_message) and not _is_greeting_with_extra_intent(user_message):
        greeting_text, greeting_diag = await _greeting_text(user_message)
        greeting_streak = (recent_mode_state.casual_turn_streak if recent_mode_state else 0) + 1
        greeting_nudged = bool(history_enabled and greeting_streak >= 4)
        if greeting_nudged:
            greeting_text = _append_upload_nudge(greeting_text)
        if greeting_diag.status == "error":
            print(
                f"[GREETING][WARN] query_id={query_id} failure_code={greeting_diag.failure_code or '-'} "
                f"error={greeting_diag.error_detail or '-'}",
                file=sys.stderr,
            )
        if history_enabled:
            greeting_phase_event = build_phase_event(
                "greeting",
                "failed" if greeting_diag.status == "error" else "completed",
                status=greeting_diag.status,
                detail=greeting_diag.error_detail,
                payload=_helper_diag_payload(greeting_diag),
            )
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                greeting_text,
                b"[]",
                metadata={
                    "session_id": session_id,
                    "kb_name": kb_name,
                    "query_id": query_id,
                    "provider_kind": ai_service.provider_kind,
                    "provider_label": ai_service.provider_label,
                    "model_name": ai_service.model_name,
                    "history_strategy": ai_service.settings.history_strategy,
                    "failure_code": greeting_diag.failure_code,
                    "conversation_mode": "casual_chat",
                    "mode_resolution_reason": "greeting_fast_path",
                    "mode_anchor_run_id": int(recent_mode_state.mode_anchor_run_id if recent_mode_state else 0),
                    "casual_turn_streak": int(greeting_streak if history_enabled else 0),
                    "upload_nudge_applied": bool(greeting_nudged),
                    "phase_events": phase_events_to_dicts([greeting_phase_event], limit=4),
                },
                response_quality_issue="greeting_helper_fail" if greeting_diag.status == "error" else "",
                usage={},
                context_chars=0,
                user_id=user_id,
            )
        if history_enabled:
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", greeting_text, user_id=user_id)
        return _stream_response(greeting_text, headers={"X-Conversation-Mode": "casual_chat"})

    mode_decision = resolve_conversation_mode(
        user_message,
        kb_has_docs=kb_has_docs,
        last_active_mode=(recent_mode_state.last_active_mode if recent_mode_state else None),
        followup_type=followup_analysis.followup_type,
        is_small_talk=bool(followup_analysis.is_small_talk),
    )
    conversation_mode = mode_decision.mode
    mode_resolution_reason = mode_decision.reason
    mode_anchor_run_id = (
        int(recent_mode_state.mode_anchor_run_id)
        if recent_mode_state and mode_resolution_reason == "inherited_from_last_assistant_mode"
        else 0
    )
    casual_turn_streak = 0
    upload_nudge_applied = False

    if conversation_mode == "casual_chat":
        casual_turn_streak = (recent_mode_state.casual_turn_streak if (history_enabled and recent_mode_state) else 0) + 1
        live_info_request = bool(is_live_info_request(user_message))
        upload_nudge_applied = bool(history_enabled and casual_turn_streak >= 4 and not live_info_request)
        if live_info_request:
            casual_text = _live_info_limit_response(user_message)
            casual_diag = HelperRunDiagnostics(status="disabled")
        else:
            casual_text, casual_diag = await _casual_chat_text(
                user_message,
                recent_history=rewrite_history_block if history_enabled else "",
            )
        if upload_nudge_applied:
            casual_text = _append_upload_nudge(casual_text)
        if casual_diag.status == "error":
            print(
                f"[CASUAL_CHAT][WARN] query_id={query_id} failure_code={casual_diag.failure_code or '-'} "
                f"error={casual_diag.error_detail or '-'}",
                file=sys.stderr,
            )
        if history_enabled:
            casual_phase_event = build_phase_event(
                "conversation_mode",
                "resolved",
                status="info",
                payload={
                    "conversation_mode": conversation_mode,
                    "mode_resolution_reason": mode_resolution_reason,
                    "mode_anchor_run_id": int(mode_anchor_run_id or 0),
                    "casual_turn_streak": int(casual_turn_streak),
                    "upload_nudge_applied": bool(upload_nudge_applied),
                },
            )
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                casual_text,
                b"[]",
                metadata={
                    "session_id": session_id,
                    "kb_name": kb_name,
                    "query_id": query_id,
                    "provider_kind": ai_service.provider_kind,
                    "provider_label": ai_service.provider_label,
                    "model_name": ai_service.model_name,
                    "history_strategy": ai_service.settings.history_strategy,
                    "effective_user_message": effective_user_message,
                    "followup_analysis": {
                        "followup_type": followup_analysis.followup_type,
                        "rewritten_query": followup_analysis.rewritten_query,
                        "should_use_history": bool(followup_analysis.should_use_history),
                        "is_small_talk": bool(followup_analysis.is_small_talk),
                    },
                    "conversation_mode": conversation_mode,
                    "mode_resolution_reason": mode_resolution_reason,
                    "mode_anchor_run_id": int(mode_anchor_run_id or 0),
                    "casual_turn_streak": int(casual_turn_streak),
                    "upload_nudge_applied": bool(upload_nudge_applied),
                    "phase_events": phase_events_to_dicts([casual_phase_event], limit=4),
                },
                response_quality_issue="casual_chat_helper_fail" if casual_diag.status == "error" else "",
                usage={},
                context_chars=0,
                user_id=user_id,
            )
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", casual_text, user_id=user_id)
        return _stream_response(casual_text, headers={"X-Conversation-Mode": "casual_chat"})

    def _sql_citations_from_results(rows: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for r in rows[: max(1, limit)]:
            citations.append(
                {
                    "chunk_id": int(r.get("id", 0) or 0),
                    "source_path": (r.get("source_path", "") or "").strip(),
                    "source_display": (r.get("source_display", r.get("source_path", "")) or "").strip(),
                    "source_type": (r.get("source_type", "") or "").strip(),
                    "doc_role": _normalize_doc_role(str(r.get("doc_role", ""))),
                    "sheet": (r.get("sheet", "") or "").strip(),
                    "row": int(r.get("row", 0) or 0),
                    "row_end": int(r.get("row_end", r.get("row", 0)) or 0),
                    "line_start": int(r.get("line_start", 0) or 0),
                    "line_end": int(r.get("line_end", 0) or 0),
                    "section": (r.get("section", "") or "").strip(),
                    "uploaded_at": int(r.get("uploaded_at", r.get("source_updated_at", 0)) or 0),
                    "score": float(r.get("score", 0.0) or 0.0),
                }
            )
        return citations

    def _latest_output_validator_name() -> str:
        for event in reversed(run_state.phase_events):
            if event.phase != "output_validation":
                continue
            validator = str((event.payload or {}).get("validator", "") or "").strip()
            if validator:
                return validator
        return ""

    def _validation_event_summary() -> Dict[str, Any]:
        validator_counts: Dict[str, int] = {}
        repair_validators = {"answer_format_repair", "citation_repair", "answer_sanitizer"}
        retry_validators = {"citation", "tool_recheck", "numeric_recheck", "outline_recheck", "quality_checker"}
        repair_applied = False
        retry_counts: Dict[str, int] = {}
        final_issue = ""
        for event in run_state.phase_events:
            if event.phase != "output_validation":
                continue
            payload = event.payload or {}
            validator = str(payload.get("validator", "") or "").strip() or "unknown"
            validator_counts[validator] = validator_counts.get(validator, 0) + 1
            if validator in repair_validators:
                repair_applied = True
            if validator in retry_validators:
                retry_counts[validator] = retry_counts.get(validator, 0) + 1
                final_issue = str(event.detail or "").strip()
        return {
            "validator_counts": validator_counts,
            "validator_retry_counts": retry_counts,
            "validator_repair_applied": bool(repair_applied),
            "final_validator_issue": final_issue,
        }

    def _build_answer_log_metadata(
        *,
        response_quality_issue: str = "",
        failure_code: str = "",
        usage: Optional[Dict[str, Any]] = None,
        context_chars: int = 0,
        prompt_tokens: int = 0,
        max_tokens: int = 0,
        answer_text: str = "",
        attempt_index: int = 0,
    ) -> Dict[str, Any]:
        evidence_summary = _doc_evidence_summary()
        validation_summary = _validation_event_summary()
        return {
            "session_id": session_id,
            "kb_name": kb_name,
            "query_id": query_id,
            "original_user_message": user_message,
            "effective_user_message": effective_user_message,
            "provider_kind": ai_service.provider_kind,
            "provider_label": ai_service.provider_label,
            "model_name": ai_service.model_name,
            "search_query": run_state.latest_search_query or search_query,
            "search_variants": search_variants[:6],
            "query_doc_intent": query_doc_intent,
            "question_intent": question_analysis.intent_type,
            "history_strategy": history_strategy,
            "followup_analysis": {
                "followup_type": followup_analysis.followup_type,
                "rewritten_query": followup_analysis.rewritten_query,
                "should_use_history": bool(followup_analysis.should_use_history),
                "is_small_talk": bool(followup_analysis.is_small_talk),
            },
            "question_analysis": {
                "intent_type": question_analysis.intent_type,
                "search_queries": question_analysis.search_queries[:4],
                "answer_focus": question_analysis.answer_focus[:4],
                "literal_first": bool(question_analysis.literal_first),
                "prefer_recent_sources": bool(question_analysis.prefer_recent_sources),
                "use_source_outline": bool(question_analysis.use_source_outline),
                "require_tool_evidence": bool(question_analysis.require_tool_evidence),
                "numeric_evidence_required": bool(question_analysis.numeric_evidence_required),
            },
            "retrieval_role_filter": run_state.latest_role_filter or _role_filter_label(retrieval_role_filter),
            "number_refs": number_refs[:8],
            "overview_mode": bool(overview_mode),
            "history_enabled": bool(history_enabled),
            "history_failed_turns_dropped": int(history_failed_turns_dropped),
            "compact_history_success_turns": int(compact_history_success_turns),
            "conversation_mode": conversation_mode,
            "mode_resolution_reason": mode_resolution_reason,
            "mode_anchor_run_id": int(mode_anchor_run_id or 0),
            "casual_turn_streak": int(casual_turn_streak or 0),
            "upload_nudge_applied": bool(upload_nudge_applied),
            "scope_nudge_applied": False,
            "scope_nudge_reason": "",
            "kb_file_count": int(kb_file_count),
            "attempt_index": max(0, int(attempt_index or 0)),
            "result_count": int(run_state.latest_result_count or len(results)),
            "context_chars": max(0, int(context_chars or 0)),
            "prompt_tokens_est": max(0, int(prompt_tokens or 0)),
            "max_tokens": max(0, int(max_tokens or 0)),
            "answer_chars": len((answer_text or "").strip()),
            "answer_memory_fastpath_used": False,
            "answer_memory_match_score": 0.0,
            "answer_memory_saved_answer_id": 0,
            "answer_memory_citation_recheck_status": "",
            "answer_memory_fallback_reason": "",
            "response_quality_issue": (response_quality_issue or "").strip(),
            "failure_code": (failure_code or "").strip(),
            "last_validator": _latest_output_validator_name(),
            "validator_counts": validation_summary["validator_counts"],
            "validator_retry_counts": validation_summary["validator_retry_counts"],
            "validator_repair_applied": validation_summary["validator_repair_applied"],
            "final_validator_issue": validation_summary["final_validator_issue"],
            "auto_prefetch_doc_count": len(last_auto_prefetch_doc_nos),
            "auto_prefetch_doc_nos": list(last_auto_prefetch_doc_nos[:4]),
            "weak_evidence_only": bool(evidence_summary.get("weak_evidence_only", False)),
            "weak_evidence_count": int(evidence_summary.get("weak_evidence_count", 0) or 0),
            "strong_evidence_count": int(evidence_summary.get("strong_evidence_count", 0) or 0),
            "retrieval_metrics": dict(run_state.latest_metrics or retrieval_metrics or {}),
            "usage": usage or {},
            "source_hints_preview": trim_preview(_list_current_sources_text(), 800),
            "retrieval_meta_preview": trim_preview(retrieval_meta_block, 800),
            "compact_history_preview": trim_preview(compact_history_block, 800),
            "tool_events": [
                {
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                    "result_preview": event.result_preview,
                    "created_at_iso": event.created_at_iso,
                }
                for event in run_state.tool_events[-12:]
            ],
            "phase_events": phase_events_to_dicts(run_state.phase_events, limit=32),
            "doc_catalog": _doc_catalog_metadata(limit=min(12, RAG_TRACE_TOP_N)),
            "top_results": _build_trace_results(rag, results, limit=min(4, RAG_TRACE_TOP_N)),
        }

    def _log_answer_to_sql(answer_text: str, metadata: Optional[Dict[str, Any]] = None):
        if (not rag) or (not answer_text.strip()):
            return
        prompt_seed = (
            f"QUESTION:{user_message}\n"
            f"RETRIEVAL_META:{retrieval_meta_block}\n"
            f"SOURCE_HINTS:{source_hint}\n"
        )
        prompt_hash = hashlib.sha256(prompt_seed.encode("utf-8")).hexdigest()
        model_label = ai_service.model_name
        citations = _sql_citations_from_state(limit=RAG_TRACE_TOP_N) if run_state.docs else _sql_citations_from_results(results, limit=RAG_TRACE_TOP_N)
        try:
            rag.log_answer(
                query_id=query_id,
                llm_model=model_label,
                prompt_hash=prompt_hash,
                answer_text=answer_text,
                citations=citations,
                metadata=metadata,
            )
        except Exception:
            pass

    context_str = ""
    results: List[Dict[str, Any]] = []
    has_number_anchor_match = True
    overview_mode = any(k in effective_user_message for k in OVERVIEW_QUERY_KEYWORDS)
    number_refs = [int(x) for x in NUMBER_REF_PATTERN.findall(effective_user_message)]
    retrieval_metrics: Dict[str, Any] = {}
    source_hint = ""
    retrieval_meta_block = ""
    search_query = effective_user_message
    search_variants: List[str] = []
    query_doc_intent = _classify_query_doc_intent(effective_user_message)
    question_analysis = QuestionAnalysis()
    retrieval_role_filter: Optional[List[str]] = None
    bootstrap_notes: List[str] = []
    run_state = AgentRunState()
    last_auto_prefetch_doc_nos: List[int] = []

    def _evidence_rows(records: Optional[List[RetrievedDocRecord]] = None) -> List[Dict[str, Any]]:
        target_records = records or list(run_state.docs.values())
        rows: List[Dict[str, Any]] = []
        for record in target_records:
            rows.append(dict((record.metadata.get("raw_row", {}) or {})))
        return rows

    def _doc_evidence_summary(records: Optional[List[RetrievedDocRecord]] = None) -> Dict[str, Any]:
        return dict(summarize_evidence_strength(_evidence_rows(records)))

    def _seeded_evidence_sufficient_for_answer() -> bool:
        if not rag or not run_state.docs:
            return False
        if overview_mode or question_analysis.use_source_outline or question_analysis.numeric_evidence_required:
            return False
        evidence_summary = _doc_evidence_summary()
        metrics = run_state.latest_metrics or retrieval_metrics or {}
        return (
            int(evidence_summary.get("strong_evidence_count", 0) or 0) > 0
            and float(metrics.get("coverage", 0.0) or 0.0) >= 0.6
            and float(metrics.get("top1", 0.0) or 0.0) >= 0.5
        )

    def _render_user_visible_answer(answer_text: str) -> str:
        raw_answer = (answer_text or "").strip()
        if not raw_answer or not run_state.docs:
            return raw_answer
        return render_answer_with_bottom_citations(raw_answer, run_state.docs)

    def _repair_missing_answer_citations(answer_text: str) -> str:
        raw_answer = canonicalize_doc_citations(answer_text or "").strip()
        if not raw_answer or not run_state.docs:
            return raw_answer
        if _is_grounded_abstention(raw_answer) or DOC_LABEL_PATTERN.search(raw_answer):
            return raw_answer
        evidence_summary = _doc_evidence_summary()
        if bool(evidence_summary.get("weak_evidence_only", False)):
            return raw_answer
        if int(evidence_summary.get("strong_evidence_count", 0) or 0) <= 0:
            return raw_answer

        preferred_doc_numbers: List[int] = []
        for doc_no in list(last_auto_prefetch_doc_nos):
            if doc_no in run_state.docs and doc_no not in preferred_doc_numbers:
                preferred_doc_numbers.append(int(doc_no))
        for doc_no in sorted(run_state.docs.keys()):
            if doc_no not in preferred_doc_numbers:
                preferred_doc_numbers.append(int(doc_no))
            if len(preferred_doc_numbers) >= 2:
                break
        if not preferred_doc_numbers:
            return raw_answer
        citation_refs = ", ".join(f"[DOC {doc_no}]" for doc_no in preferred_doc_numbers[:2])
        return f"{raw_answer}\n\n근거: {citation_refs}"

    def _append_phase_event(
        phase: str,
        event_name: str,
        *,
        status: str = "ok",
        detail: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ):
        if not ai_service.settings.enable_phase_events:
            return
        run_state.phase_events.append(
            build_phase_event(
                phase,
                event_name,
                status=status,
                detail=detail,
                payload=payload,
            )
        )

    _append_phase_event(
        "request",
        "received",
        payload={
            "kb_name": kb_name,
            "history_enabled": bool(history_enabled),
            "kb_has_docs": bool(kb_has_docs),
            "rag_enabled": bool(rag),
        },
    )
    if rewrite_history_block:
        followup_status = "rewritten" if effective_user_message != user_message else "inspected"
        if followup_diag.status == "disabled":
            followup_status = "skipped"
        _append_phase_event(
            "followup",
            followup_status,
            status="info" if followup_diag.status in {"disabled", "ok"} else followup_diag.status,
            detail=followup_diag.error_detail,
            payload={
                "followup_type": followup_analysis.followup_type,
                "should_use_history": bool(followup_analysis.should_use_history),
                "is_small_talk": bool(followup_analysis.is_small_talk),
                "rewritten": bool(effective_user_message != user_message),
                "effective_query": effective_user_message,
                **_helper_diag_payload(followup_diag),
            },
        )

    def _register_answer_memory_rows(rows: List[Dict[str, Any]]) -> None:
        for raw in rows:
            raw_row = dict(raw)
            chunk_id = int(raw_row.get("id", 0) or 0)
            if chunk_id <= 0 or chunk_id in run_state.doc_numbers_by_chunk_id:
                continue
            doc_no = max(run_state.docs.keys(), default=0) + 1
            record = RetrievedDocRecord(
                doc_no=doc_no,
                chunk_id=chunk_id,
                source_path=(raw_row.get("source_path", "") or "").strip(),
                source_ref=rag.format_source_ref(raw_row) if rag else (raw_row.get("source_path", "") or "").strip(),
                text=(raw_row.get("text", "") or "").strip(),
                source_type=(raw_row.get("source_type", "") or "").strip(),
                section=(raw_row.get("section", "") or "").strip(),
                sheet=(raw_row.get("sheet", "") or "").strip(),
                row=int(raw_row.get("row", 0) or 0),
                row_end=int(raw_row.get("row_end", raw_row.get("row", 0)) or 0),
                page_no=int(raw_row.get("page_no", 0) or 0),
                line_start=int(raw_row.get("line_start", 0) or 0),
                line_end=int(raw_row.get("line_end", raw_row.get("line_start", 0)) or 0),
                uploaded_at=int(raw_row.get("uploaded_at", raw_row.get("source_updated_at", 0)) or 0),
                score=float(raw_row.get("score", 0.0) or 0.0),
                metadata={"raw_row": raw_row, "answer_memory_fastpath": True},
            )
            run_state.docs[doc_no] = record
            run_state.doc_numbers_by_chunk_id[chunk_id] = doc_no

    def _try_answer_memory_fastpath() -> Optional[Dict[str, Any]]:
        if (
            not WIKI_ANSWER_MEMORY_ENABLED
            or not WIKI_ANSWER_MEMORY_FASTPATH_ENABLED
            or not rag
            or conversation_mode != "document_qa"
            or not kb_has_docs
        ):
            return None
        try:
            memory_store = _get_wiki_memory_store_for_rag(rag)
            matches = memory_store.search_memory(effective_user_message, limit=1)
            if not matches:
                _append_phase_event(
                    "answer_memory_fastpath",
                    "miss",
                    status="info",
                    payload={"reason": "no_published_match"},
                )
                return None
            match = dict(matches[0])
            score = float(match.get("memory_score", 0.0) or 0.0)
            saved_answer_id = int(match.get("saved_answer_id", 0) or 0)
            if score < WIKI_ANSWER_MEMORY_FASTPATH_MIN_SCORE or saved_answer_id <= 0:
                _append_phase_event(
                    "answer_memory_fastpath",
                    "miss",
                    status="info",
                    payload={
                        "reason": "score_below_threshold",
                        "score": score,
                        "threshold": float(WIKI_ANSWER_MEMORY_FASTPATH_MIN_SCORE),
                    },
                )
                return None
            detail = memory_store.get_saved_answer(saved_answer_id)
            if not detail or str(detail.get("status", "") or "") != "published":
                _append_phase_event(
                    "answer_memory_fastpath",
                    "miss",
                    status="info",
                    payload={"reason": "saved_answer_not_published", "saved_answer_id": saved_answer_id},
                )
                return None
            sources = [
                dict(source)
                for source in list(detail.get("sources", []) or [])
                if isinstance(source, dict) and str(source.get("status", "") or "") == "active"
            ]
            chunk_ids = [
                int(source.get("chunk_id", 0) or 0)
                for source in sources
                if int(source.get("chunk_id", 0) or 0) > 0
            ]
            if not chunk_ids:
                _append_phase_event(
                    "answer_memory_fastpath",
                    "fallback",
                    status="warning",
                    payload={"reason": "no_chunk_citations", "saved_answer_id": saved_answer_id},
                )
                return None
            rows = rag._load_candidate_rows(list(dict.fromkeys(chunk_ids)))  # Recheck original source chunks.
            row_ids = {int(row.get("id", 0) or 0) for row in rows}
            if not row_ids or not set(chunk_ids).issubset(row_ids):
                _append_phase_event(
                    "answer_memory_fastpath",
                    "fallback",
                    status="warning",
                    payload={
                        "reason": "citation_recheck_failed",
                        "saved_answer_id": saved_answer_id,
                        "requested_chunk_ids": chunk_ids[:8],
                        "found_chunk_ids": sorted(row_ids)[:8],
                    },
                )
                return None
            _register_answer_memory_rows(rows)
            run_state.latest_search_query = effective_user_message
            run_state.latest_result_count = len(rows)
            run_state.latest_metrics = {
                "answer_memory_score": score,
                "unique_sources": len({str(row.get("source_path", "") or "") for row in rows}),
            }
            run_state.latest_role_filter = "answer_memory_fastpath"
            answer_text = str(detail.get("answer_text", "") or detail.get("answer_summary", "") or "").strip()
            if not answer_text:
                _append_phase_event(
                    "answer_memory_fastpath",
                    "fallback",
                    status="warning",
                    payload={"reason": "empty_saved_answer", "saved_answer_id": saved_answer_id},
                )
                return None
            _append_phase_event(
                "answer_memory_fastpath",
                "hit",
                status="ok",
                payload={
                    "saved_answer_id": saved_answer_id,
                    "score": score,
                    "source_count": len(sources),
                    "chunk_count": len(rows),
                    "citation_recheck_status": "ok",
                },
            )
            return {
                "saved_answer_id": saved_answer_id,
                "score": score,
                "answer_text": answer_text,
                "chunk_count": len(rows),
            }
        except Exception as e:
            _append_phase_event(
                "answer_memory_fastpath",
                "failed",
                status="warning",
                detail=str(e),
                payload={"failure_code": _classify_failure_code(e, default="answer_memory_fastpath_fail")},
            )
            return None
    _append_phase_event(
        "conversation_mode",
        "resolved",
        status="info",
        payload={
            "conversation_mode": conversation_mode,
            "mode_resolution_reason": mode_resolution_reason,
            "mode_anchor_run_id": int(mode_anchor_run_id or 0),
            "casual_turn_streak": int(casual_turn_streak or 0),
            "upload_nudge_applied": bool(upload_nudge_applied),
        },
    )

    if conversation_mode == "document_qa" and kb_has_docs:
        cross_kb_upload_blocker = _find_cross_kb_upload_blocker(kb_name)
        if cross_kb_upload_blocker:
            busy_text = _render_cross_kb_upload_busy_text(kb_name, cross_kb_upload_blocker)
            _append_phase_event(
                "upload_guard",
                "blocked",
                status="info",
                detail="cross_kb_document_qa_blocked_during_upload",
                payload={
                    "target_kb_name": kb_name,
                    "active_upload_kb_name": cross_kb_upload_blocker.get("kb_name", ""),
                    "active_upload_status": cross_kb_upload_blocker.get("status", ""),
                    "active_upload_stage": cross_kb_upload_blocker.get("progress_stage", ""),
                    "active_upload_percent": int(cross_kb_upload_blocker.get("progress_percent", 0) or 0),
                    "active_upload_filename": cross_kb_upload_blocker.get("original_filename", ""),
                },
            )
            if history_enabled:
                _append_agent_run_history(
                    session_id,
                    kb_name,
                    query_id,
                    user_message,
                    busy_text,
                    b"[]",
                    metadata={
                        "session_id": session_id,
                        "kb_name": kb_name,
                        "query_id": query_id,
                        "provider_kind": ai_service.provider_kind,
                        "provider_label": ai_service.provider_label,
                        "model_name": ai_service.model_name,
                        "history_strategy": ai_service.settings.history_strategy,
                        "effective_user_message": effective_user_message,
                        "conversation_mode": conversation_mode,
                        "mode_resolution_reason": mode_resolution_reason,
                        "mode_anchor_run_id": int(mode_anchor_run_id or 0),
                        "casual_turn_streak": int(casual_turn_streak or 0),
                        "upload_nudge_applied": bool(upload_nudge_applied),
                        "kb_file_count": int(kb_file_count),
                        "busy_upload_blocker": dict(cross_kb_upload_blocker),
                        "phase_events": phase_events_to_dicts(run_state.phase_events, limit=12),
                    },
                    response_quality_issue="other_kb_upload_busy",
                    usage={},
                    context_chars=0,
                    user_id=user_id,
                )
                _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
                _append_chat_message(session_id, kb_name, "assistant", busy_text, user_id=user_id)
            return _stream_response(
                busy_text,
                headers={"X-Conversation-Mode": conversation_mode},
            )

    if conversation_mode == "document_qa" and kb_has_docs and rag is None:
        try:
            lease_context = rag_registry.lease(kb_name, _create_rag_engine)
            rag = lease_context.__enter__()
            rag_lease_state["context"] = lease_context
        except Exception:
            rag = None
        if rag is None:
            _append_phase_event(
                "rag_load",
                "failed",
                status="error",
                detail="문서 검색 엔진을 불러오지 못했습니다.",
                payload={"kb_name": kb_name},
            )
        else:
            _append_phase_event(
                "rag_load",
                "completed",
                status="info",
                payload={"kb_name": kb_name},
            )

    fastpath = _try_answer_memory_fastpath()
    if fastpath is not None:
        fastpath_text = str(fastpath.get("answer_text", "") or "").strip()
        rendered_fastpath = _render_user_visible_answer(fastpath_text)
        fastpath_metadata = _build_answer_log_metadata(
            usage={"requests": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0},
            context_chars=0,
            prompt_tokens=0,
            max_tokens=0,
            answer_text=fastpath_text,
            attempt_index=0,
        )
        fastpath_metadata.update(
            {
                "answer_memory_fastpath_used": True,
                "answer_memory_saved_answer_id": int(fastpath.get("saved_answer_id", 0) or 0),
                "answer_memory_match_score": float(fastpath.get("score", 0.0) or 0.0),
                "answer_memory_citation_recheck_status": "ok",
                "answer_memory_fallback_reason": "",
            }
        )
        _log_answer_to_sql(fastpath_text, metadata=fastpath_metadata)
        if history_enabled:
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                rendered_fastpath,
                b"",
                metadata=fastpath_metadata,
                response_quality_issue="",
                usage=fastpath_metadata.get("usage", {}),
                context_chars=0,
                user_id=user_id,
            )
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", rendered_fastpath, user_id=user_id)
        return _stream_response(
            rendered_fastpath,
            headers={
                "X-Conversation-Mode": conversation_mode,
                "X-Answer-Memory-Fastpath": "1",
            },
        )

    if rag:
        try:
            analysis_started_at = time.perf_counter()
            analysis_result, expand_result = await run_parallel_helper_tasks(
                effective_user_message,
                _llm_analyze_question,
                _llm_expand_query,
            )
            if isinstance(analysis_result, BaseException):
                question_analysis = QuestionAnalysis()
                analysis_diag = _exception_to_helper_diag(
                    analysis_result,
                    "question_analysis_parallel_fail",
                )
            else:
                question_analysis, analysis_diag = analysis_result

            if isinstance(expand_result, BaseException):
                llm_expanded = ""
                expand_diag = _exception_to_helper_diag(
                    expand_result,
                    "query_expand_parallel_fail",
                )
            else:
                llm_expanded, expand_diag = expand_result
            if analysis_diag.status == "error":
                print(
                    f"[QUESTION_ANALYSIS][WARN] query_id={query_id} status={analysis_diag.status} "
                    f"failure_code={analysis_diag.failure_code or '-'} error={analysis_diag.error_detail or '-'}",
                    file=sys.stderr,
                )
            if overview_mode or len((effective_user_message or "").strip()) >= 90:
                question_analysis = question_analysis.model_copy(
                    update={"use_source_outline": True, "require_tool_evidence": True}
                )
            elif kb_has_docs:
                question_analysis = question_analysis.model_copy(
                    update={"require_tool_evidence": True}
                )
            if kb_has_docs and is_numeric_evidence_query(effective_user_message):
                question_analysis = question_analysis.model_copy(
                    update={"require_tool_evidence": True, "numeric_evidence_required": True}
                )
            elif question_analysis.numeric_evidence_required and not is_numeric_evidence_query(effective_user_message):
                question_analysis = question_analysis.model_copy(
                    update={"numeric_evidence_required": False}
                )

            search_terms = [_expand_query_with_number_refs(effective_user_message)]
            if expand_diag.status == "error":
                print(
                    f"[QUERY_EXPAND][WARN] query_id={query_id} status=error "
                    f"failure_code={expand_diag.failure_code or '-'} error={expand_diag.error_detail or '-'}",
                    file=sys.stderr,
                )
            if llm_expanded:
                search_terms.append(llm_expanded)
            if question_analysis.search_queries:
                search_terms.extend(_expand_query_with_number_refs(q) for q in question_analysis.search_queries[:4])
                bootstrap_notes.append(
                    "질문 의도를 분석해 검색 구문을 보강했다: "
                    + ", ".join(question_analysis.search_queries[:3])
                )
            search_variants = _dedupe_text_items(search_terms, limit=6)
            search_query = search_variants[0] if search_variants else effective_user_message
            if WIKI_ACTIVE_RETRIEVAL_ENABLED:
                try:
                    wiki_boost_targets = _get_wiki_memory_store_for_rag(rag).retrieval_boost_targets()
                    rag.set_wiki_memory_boost_targets(wiki_boost_targets)
                    _append_phase_event(
                        "wiki_boost",
                        "applied" if any(wiki_boost_targets.get(key) for key in ("chunks", "table_cells", "sources")) else "empty",
                        status="info",
                        payload={
                            "chunk_targets": len(wiki_boost_targets.get("chunks", {}) or {}),
                            "table_cell_targets": len(wiki_boost_targets.get("table_cells", {}) or {}),
                            "source_targets": len(wiki_boost_targets.get("sources", {}) or {}),
                            "scope": "current_kb_only",
                        },
                    )
                except Exception as e:
                    rag.set_wiki_memory_boost_targets({})
                    _append_phase_event(
                        "wiki_boost",
                        "failed",
                        status="error",
                        detail=str(e),
                        payload={"scope": "current_kb_only"},
                    )
            else:
                rag.set_wiki_memory_boost_targets({})
                _append_phase_event(
                    "wiki_active_retrieval",
                    "disabled",
                    status="info",
                    payload={"wiki_boost": False, "wiki_page_hints": False},
                )
            _append_phase_event(
                "question_analysis",
                "skipped" if analysis_diag.status == "disabled" else ("failed" if analysis_diag.status == "error" else "completed"),
                status="info" if analysis_diag.status == "disabled" else analysis_diag.status,
                detail=analysis_diag.error_detail,
                payload={
                    "intent_type": question_analysis.intent_type,
                    "search_query_count": len(question_analysis.search_queries),
                    "answer_focus_count": len(question_analysis.answer_focus),
                    "use_source_outline": bool(question_analysis.use_source_outline),
                    "require_tool_evidence": bool(question_analysis.require_tool_evidence),
                    "analysis_parallelized": True,
                    "latency_ms": int((time.perf_counter() - analysis_started_at) * 1000),
                    **_helper_diag_payload(analysis_diag),
                },
            )
            _append_phase_event(
                "query_expand",
                "skipped" if expand_diag.status == "disabled" else ("failed" if expand_diag.status == "error" else "completed"),
                status="info" if expand_diag.status == "disabled" else expand_diag.status,
                detail=expand_diag.error_detail,
                payload={
                    "variant_count": len(search_variants),
                    "search_query": search_query,
                    "expanded_query_present": bool(llm_expanded),
                    "expand_parallelized": True,
                    **_helper_diag_payload(expand_diag),
                },
            )

            top_k = RAG_TOP_K_OVERVIEW if overview_mode else RAG_TOP_K
            retrieve_k = max(top_k, RAG_LLM_RERANK_CANDIDATES)
            keep_n = max(top_k, RAG_LLM_RERANK_KEEP)
            gate_passed = False
            pre_gate_results: List[Dict[str, Any]] = []
            critical_meta: Dict[str, Any] = {}
            role_search_plan = _build_role_search_plan(query_doc_intent)
            for role_filter in role_search_plan:
                candidates = _collect_search_candidates(
                    rag,
                    search_variants,
                    top_k=retrieve_k,
                    index_name="large",
                    doc_roles=role_filter,
                )
                should_rerank, rerank_reason = decide_rerank_usage(
                    effective_user_message,
                    candidate_count=len(candidates),
                )
                if not RAG_LLM_RERANK_ENABLED:
                    should_rerank = False
                    rerank_reason = "rerank_disabled"
                if should_rerank:
                    reranked, rerank_diag, rerank_budget_meta = await _llm_rerank_results(
                        effective_user_message,
                        candidates,
                        keep_n=keep_n,
                    )
                else:
                    reranked = candidates[:keep_n]
                    rerank_diag = HelperRunDiagnostics(status="disabled")
                    rerank_budget_meta = {
                        "original_count": len(candidates),
                        "selected_count": min(len(candidates), keep_n),
                        "trimmed_count": 0,
                        "line_char_cap": 0,
                    }
                if rerank_diag.status == "error":
                    print(
                        f"[RERANK][WARN] query_id={query_id} status=error "
                        f"failure_code={rerank_diag.failure_code or '-'} error={rerank_diag.error_detail or '-'}",
                        file=sys.stderr,
                    )
                _append_phase_event(
                    "rerank",
                    "skipped" if rerank_diag.status == "disabled" else ("failed" if rerank_diag.status == "error" else "completed"),
                    status="info" if rerank_diag.status == "disabled" else rerank_diag.status,
                    detail=rerank_diag.error_detail,
                    payload={
                        "candidate_count": len(candidates),
                        "returned_count": len(reranked),
                        "keep_n": keep_n,
                        "rerank_skipped_reason": rerank_reason if not should_rerank else "",
                        "rerank_used_reason": rerank_reason if should_rerank else "",
                        **rerank_budget_meta,
                        **_helper_diag_payload(rerank_diag),
                    },
                )
                attempt_results, attempt_number_anchor_match = _apply_number_reference_guard(
                    results=reranked,
                    number_refs=number_refs,
                    keep_n=keep_n,
                )
                attempt_results = _prefer_source_chunks(
                    attempt_results,
                    keep_n=keep_n,
                    max_normalized=RAG_MAX_NORMALIZED_RESULTS,
                )
                attempt_results = rerank_results_for_grounded_answer(
                    effective_user_message,
                    attempt_results,
                )[:keep_n]
                attempt_results, critical_meta = apply_critical_term_gate(
                    effective_user_message,
                    attempt_results,
                    enabled=RAG_CRITICAL_TERM_GATE_ENABLED,
                    require_raw_backing_for_normalized=RAG_NORMALIZED_RAW_BACKING_REQUIRED,
                )

                if RAG_DIVERSIFY_ENABLED and attempt_results:
                    attempt_results = _apply_diversity_filter(
                        attempt_results,
                        keep_n=keep_n,
                        max_per_section=RAG_MAX_PER_SECTION,
                        max_per_file=RAG_MAX_PER_FILE,
                    )

                attempt_pre_gate = list(attempt_results)
                attempt_metrics = rag.evaluate_answerability(
                    effective_user_message,
                    attempt_pre_gate,
                    coverage_top_k=RAG_ANSWER_COVERAGE_TOP_K,
                )
                attempt_gate_passed = _passes_grounding_gate(effective_user_message, attempt_metrics, attempt_pre_gate)

                retrieval_role_filter = role_filter
                has_number_anchor_match = attempt_number_anchor_match
                pre_gate_results = attempt_pre_gate
                retrieval_metrics = attempt_metrics
                gate_passed = attempt_gate_passed
                results = attempt_results if attempt_gate_passed else []

                if attempt_gate_passed:
                    break

            retrieval_meta_block = _build_retrieval_meta_block(
                retrieval_metrics,
                pre_gate_results,
                query_doc_intent=query_doc_intent,
                retrieval_role_filter=retrieval_role_filter,
            )
            matched_concepts = _summarize_matched_concepts(pre_gate_results)

            if not results:
                recent_limit = 12 if (question_analysis.prefer_recent_sources or _is_summary_request(effective_user_message)) else min(8, max(4, RAG_TOP_K // 2))
                seed_candidates = rag.get_recent_chunks(
                    limit=recent_limit,
                    doc_roles=retrieval_role_filter,
                )
                if (not seed_candidates) and retrieval_role_filter:
                    seed_candidates = rag.get_recent_chunks(
                        limit=recent_limit,
                        doc_roles=None,
                    )
                if seed_candidates:
                    seed_candidates, seed_critical_meta = apply_critical_term_gate(
                        effective_user_message,
                        seed_candidates,
                        enabled=RAG_CRITICAL_TERM_GATE_ENABLED,
                        require_raw_backing_for_normalized=RAG_NORMALIZED_RAW_BACKING_REQUIRED,
                    )
                    if seed_critical_meta:
                        critical_meta = seed_critical_meta
                if seed_candidates:
                    results = _prefer_source_chunks(
                        seed_candidates,
                        keep_n=min(8, max(4, RAG_TOP_K // 2)),
                        max_normalized=RAG_MAX_NORMALIZED_RESULTS,
                    )
                    if RAG_DIVERSIFY_ENABLED and results:
                        results = _apply_diversity_filter(
                            results,
                            keep_n=min(8, max(4, RAG_TOP_K // 2)),
                            max_per_section=RAG_MAX_PER_SECTION,
                            max_per_file=RAG_MAX_PER_FILE,
                        )
                    retrieval_metrics = rag.evaluate_answerability(
                        effective_user_message,
                        results,
                        coverage_top_k=RAG_ANSWER_COVERAGE_TOP_K,
                    )
                    retrieval_meta_block = _build_retrieval_meta_block(
                        retrieval_metrics,
                        results,
                        query_doc_intent=query_doc_intent,
                        retrieval_role_filter=retrieval_role_filter,
                    )
                    bootstrap_notes.append("질문 검색 매칭이 약해 최근 업로드 원문을 시드 컨텍스트로 사용했다.")
                    if question_analysis.prefer_recent_sources:
                        bootstrap_notes.append("질문 의도 분석 결과 최신 업로드 문서를 우선 참고하도록 조정했다.")
            matched_concepts = _summarize_matched_concepts(pre_gate_results or results)

            context_per_result_max = _adaptive_per_result_context_cap(
                RAG_CONTEXT_MAX_CHARS,
                len(results),
            )
            context_str = rag.get_context_string(
                results,
                query=effective_user_message,
                max_chars=RAG_CONTEXT_MAX_CHARS,
                per_result_max_chars=context_per_result_max,
                focus_relevant=True,
                top1_score=float(retrieval_metrics.get("top1", 0.0)),
            )
            if overview_mode and results:
                top_source = results[0].get("source_path", "")
                overview_ctx = rag.get_source_overview_context(
                    top_source,
                    max_chars=min(RAG_OVERVIEW_EXTRA_CHARS, max(400, RAG_CONTEXT_MAX_CHARS // 2)),
                )
                if overview_ctx:
                    context_str = _merge_context_with_cap(context_str, overview_ctx, RAG_CONTEXT_MAX_CHARS)
            else:
                context_str = _trim_to_max_chars(context_str, RAG_CONTEXT_MAX_CHARS)

            source_hint = _build_source_hints(rag, results, limit=8)
            _append_rag_trace_log(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "kb_name": kb_name,
                    "session_id": session_id,
                    "query": user_message,
                    "effective_query": effective_user_message,
                    "search_query": search_query,
                    "search_variants": search_variants[:6],
                    "query_doc_intent": query_doc_intent,
                    "question_intent": question_analysis.intent_type,
                    "retrieval_role_filter": _role_filter_label(retrieval_role_filter),
                    "overview_mode": bool(overview_mode),
                    "number_refs": number_refs[:8],
                    "gate_passed": bool(gate_passed),
                    "concept_graph_used": bool(matched_concepts),
                    "matched_concepts": matched_concepts,
                    "metrics": {
                        "top1": float(retrieval_metrics.get("top1", 0.0)),
                        "top2": float(retrieval_metrics.get("top2", 0.0)),
                        "margin": float(retrieval_metrics.get("margin", 0.0)),
                        "coverage": float(retrieval_metrics.get("coverage", 0.0)),
                        "keyword_hits": int(retrieval_metrics.get("keyword_hits", 0) or 0),
                        "keyword_total": int(retrieval_metrics.get("keyword_total", 0) or 0),
                        "unique_sources": int(retrieval_metrics.get("unique_sources", 0) or 0),
                        "latest_uploaded_at": int(retrieval_metrics.get("latest_uploaded_at", 0) or 0),
                        "doc_roles": retrieval_metrics.get("doc_roles", []),
                        "has_conflict": bool(retrieval_metrics.get("has_conflict", False)),
                        "critical_terms": critical_meta.get("critical_terms", []),
                        "critical_term_hits": critical_meta.get("critical_term_hits", {}),
                        "critical_term_gate_passed": bool(critical_meta.get("critical_term_gate_passed", True)),
                        "literal_title_hit_count": int(critical_meta.get("literal_title_hit_count", 0) or 0),
                        "normalized_only_blocked": bool(critical_meta.get("normalized_only_blocked", False)),
                        "related_terms_suggestions": critical_meta.get("related_terms_suggestions", []),
                    },
                    "result_count_before_gate": len(pre_gate_results),
                    "result_count_after_gate": len(results),
                    "top_results": _build_trace_results(rag, pre_gate_results, limit=RAG_TRACE_TOP_N),
                }
            )
            try:
                rag.log_retrieval(
                    query_id=query_id,
                    user_id=session_id,
                    query_text=effective_user_message,
                    topk_ids=[
                        int(r.get("id", 0) or 0)
                        for r in pre_gate_results[: max(top_k, RAG_TRACE_TOP_N)]
                        if int(r.get("id", 0) or 0) > 0
                    ],
                    meta={
                            "kb_name": kb_name,
                            "original_user_message": user_message,
                            "effective_user_message": effective_user_message,
                            "search_query": search_query,
                        "search_variants": search_variants[:6],
                        "query_doc_intent": query_doc_intent,
                        "question_intent": question_analysis.intent_type,
                        "retrieval_role_filter": _role_filter_label(retrieval_role_filter),
                        "overview_mode": bool(overview_mode),
                        "number_refs": number_refs[:8],
                        "gate_passed": bool(gate_passed),
                        "concept_graph_used": bool(matched_concepts),
                        "matched_concepts": matched_concepts,
                        "metrics": retrieval_metrics,
                        "critical_terms": critical_meta.get("critical_terms", []),
                        "critical_term_hits": critical_meta.get("critical_term_hits", {}),
                        "critical_term_gate_passed": bool(critical_meta.get("critical_term_gate_passed", True)),
                        "literal_title_hit_count": int(critical_meta.get("literal_title_hit_count", 0) or 0),
                        "normalized_only_blocked": bool(critical_meta.get("normalized_only_blocked", False)),
                        "related_terms_suggestions": critical_meta.get("related_terms_suggestions", []),
                    },
                )
            except Exception:
                pass
            _append_phase_event(
                "retrieval",
                "completed",
                payload={
                    "search_query": search_query,
                    "variant_count": len(search_variants),
                    "result_count_before_gate": len(pre_gate_results),
                    "result_count_after_gate": len(results),
                    "grounding_gate_passed": bool(gate_passed),
                    "concept_graph_used": bool(matched_concepts),
                    "matched_concepts": matched_concepts,
                    "retrieval_role_filter": _role_filter_label(retrieval_role_filter),
                    "top1": float(retrieval_metrics.get("top1", 0.0)),
                    "coverage": float(retrieval_metrics.get("coverage", 0.0)),
                    "critical_terms": critical_meta.get("critical_terms", []),
                    "critical_term_hits": critical_meta.get("critical_term_hits", {}),
                    "critical_term_gate_passed": bool(critical_meta.get("critical_term_gate_passed", True)),
                    "literal_title_hit_count": int(critical_meta.get("literal_title_hit_count", 0) or 0),
                    "normalized_only_blocked": bool(critical_meta.get("normalized_only_blocked", False)),
                    "related_terms_suggestions": critical_meta.get("related_terms_suggestions", []),
                },
            )
        except Exception as e:
            failure_code = _classify_failure_code(e, default="retrieval_fail")
            print(
                f"[RETRIEVAL][ERROR] kb={kb_name} query_id={query_id} failure_code={failure_code} error={e}",
                file=sys.stderr,
            )
            context_str = f"(문서 검색 중 문제가 발생했습니다: {e})"
            retrieval_meta_block = ""
            _append_phase_event(
                "retrieval",
                "failed",
                status="error",
                detail=str(e),
                payload={"failure_code": failure_code},
            )
    else:
        context_str = f"(System: Knowledge Base '{kb_name}' not found. Answering without context.)"
        _append_phase_event(
            "retrieval",
            "skipped",
            status="info",
            detail="knowledge_base_not_found",
        )

    if not rag:
        missing_kb_text = _no_evidence_response("자료 공간을 찾지 못했다")
        if history_enabled:
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                missing_kb_text,
                b"[]",
                metadata={
                    "session_id": session_id,
                    "kb_name": kb_name,
                    "query_id": query_id,
                    "provider_kind": ai_service.provider_kind,
                    "provider_label": ai_service.provider_label,
                    "model_name": ai_service.model_name,
                    "history_strategy": ai_service.settings.history_strategy,
                    "effective_user_message": effective_user_message,
                    "conversation_mode": conversation_mode,
                    "mode_resolution_reason": mode_resolution_reason,
                    "mode_anchor_run_id": int(mode_anchor_run_id or 0),
                    "casual_turn_streak": int(casual_turn_streak or 0),
                    "upload_nudge_applied": bool(upload_nudge_applied),
                    "phase_events": phase_events_to_dicts(run_state.phase_events, limit=12),
                },
                response_quality_issue="knowledge_base_not_found",
                usage={},
                context_chars=0,
                user_id=user_id,
            )
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", missing_kb_text, user_id=user_id)
        return _stream_response(
            missing_kb_text,
            headers={"X-Conversation-Mode": conversation_mode},
        )

    scope_nudge_applied = False
    scope_nudge_reason = ""
    scope_nudge_response = ""
    scope_nudge_rows = pre_gate_results or results
    if should_prompt_for_narrower_summary(
        effective_user_message,
        metrics=retrieval_metrics,
        results=scope_nudge_rows,
        kb_file_count=kb_file_count,
        overview_mode=bool(overview_mode),
    ):
        scope_nudge_applied = True
        scope_nudge_reason = explain_scope_nudge_reason(
            retrieval_metrics,
            scope_nudge_rows,
            kb_file_count=kb_file_count,
            overview_mode=bool(overview_mode),
        )
        scope_nudge_response = build_scope_narrowing_response(
            effective_user_message,
            results=scope_nudge_rows,
        )
        _append_phase_event(
            "scope_nudge",
            "applied",
            status="info",
            payload={
                "scope_nudge_applied": True,
                "scope_nudge_reason": scope_nudge_reason,
                "kb_file_count": int(kb_file_count),
                "result_count_before_gate": len(pre_gate_results),
                "result_count_after_gate": len(results),
                "unique_sources": int(retrieval_metrics.get("unique_sources", 0) or 0),
                "top1": float(retrieval_metrics.get("top1", 0.0) or 0.0),
                "coverage": float(retrieval_metrics.get("coverage", 0.0) or 0.0),
            },
        )
        if history_enabled:
            _append_agent_run_history(
                session_id,
                kb_name,
                query_id,
                user_message,
                scope_nudge_response,
                b"[]",
                metadata={
                    "session_id": session_id,
                    "kb_name": kb_name,
                    "query_id": query_id,
                    "provider_kind": ai_service.provider_kind,
                    "provider_label": ai_service.provider_label,
                    "model_name": ai_service.model_name,
                    "history_strategy": ai_service.settings.history_strategy,
                    "effective_user_message": effective_user_message,
                    "conversation_mode": conversation_mode,
                    "mode_resolution_reason": mode_resolution_reason,
                    "mode_anchor_run_id": int(mode_anchor_run_id or 0),
                    "casual_turn_streak": int(casual_turn_streak or 0),
                    "upload_nudge_applied": bool(upload_nudge_applied),
                    "scope_nudge_applied": True,
                    "scope_nudge_reason": scope_nudge_reason,
                    "kb_file_count": int(kb_file_count),
                    "retrieval_metrics": dict(retrieval_metrics or {}),
                    "phase_events": phase_events_to_dicts(run_state.phase_events, limit=12),
                },
                response_quality_issue="scope_nudge_applied",
                usage={},
                context_chars=len(context_str),
                user_id=user_id,
            )
            _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
            _append_chat_message(session_id, kb_name, "assistant", scope_nudge_response, user_id=user_id)
        return _stream_response(
            scope_nudge_response,
            headers={"X-Conversation-Mode": conversation_mode},
        )

    if rag and (not results):
        bootstrap_notes.append("초기 검색에서는 질문과 직접 맞는 근거를 충분히 찾지 못했다.")
    if rag and number_refs and not has_number_anchor_match:
        bootstrap_notes.append("초기 검색에서는 질문에서 지정한 번호 근거를 바로 찾지 못했다.")

    ref_hint = ""
    if number_refs:
        refs_txt = ", ".join(str(n) for n in number_refs[:5])
        ref_hint = f"\n\nREFERENCE_HINT:\n사용자 질문에서 언급한 번호: {refs_txt}\n"

    wiki_memory_block = ""
    wiki_page_hint_block = ""
    wiki_memory_matches: List[Dict[str, Any]] = []
    if rag and WIKI_ACTIVE_RETRIEVAL_ENABLED:
        try:
            wiki_memory_store = _get_wiki_memory_store_for_rag(rag)
            wiki_memory_matches = wiki_memory_store.search_memory(effective_user_message, limit=3)
            wiki_page_targets = wiki_memory_store.retrieval_boost_targets()
            wiki_page_hints = _get_wiki_store_for_rag(rag).published_page_hints(limit=5)
            if any(wiki_page_targets.get(key) for key in ("chunks", "table_cells", "sources")):
                hint_lines = [
                    "\n\nWIKI_PAGE_HINTS:",
                    "아래 published wiki page는 현재 지침서 공간 안에서만 누적된 검색 참고 신호다.",
                    "wiki page 자체를 최종 citation으로 직접 쓰지 마라. 연결된 원본 source/chunk/table citation을 도구로 다시 확인하라.",
                    f"published_page_source_targets=chunks:{len(wiki_page_targets.get('chunks', {}))}, "
                    f"table_cells:{len(wiki_page_targets.get('table_cells', {}))}, "
                    f"sources:{len(wiki_page_targets.get('sources', {}))}",
                ]
                for hint in wiki_page_hints:
                    source_refs = [
                        str(source.get("source_ref", "") or source.get("source_path", "") or "").strip()
                        for source in list(hint.get("sources", []) or [])[:3]
                        if isinstance(source, dict)
                    ]
                    hint_lines.append(
                        "- "
                        f"title={trim_preview(str(hint.get('title', '') or ''), 80)} | "
                        f"type={hint.get('page_type', '')} | "
                        f"summary={trim_preview(str(hint.get('summary', '') or ''), 180)} | "
                        f"source_refs={trim_preview('; '.join(ref for ref in source_refs if ref), 180)}"
                    )
                wiki_page_hint_block = (
                    "\n".join(hint_lines) + "\n"
                )
            if wiki_memory_matches:
                memory_lines = [
                    "WIKI_ANSWER_MEMORY:",
                    "아래는 사용자가 과거에 저장한 답변 기억이다. question_rewrite_hint와 근거 후보로만 사용하라.",
                    "원본 문서 citation을 재확인하기 전에는 saved answer 내용을 최종 답변으로 확정하지 마라.",
                    "saved answer 자체를 근거로 직접 인용하지 말고, SOURCE_HINTS 또는 도구로 확인 가능한 원본 chunk/table citation만 최종 근거로 사용하라.",
                    f"retrieval_boost_targets=chunks:{len(wiki_page_targets.get('chunks', {}))}, table_cells:{len(wiki_page_targets.get('table_cells', {}))}, sources:{len(wiki_page_targets.get('sources', {}))}",
                ]
                for item in wiki_memory_matches:
                    memory_lines.append(
                        "- "
                        f"question_rewrite_hint={trim_preview(str(item.get('question_text', '') or ''), 120)} | "
                        f"summary={trim_preview(str(item.get('answer_summary', '') or ''), 220)} | "
                        f"status={item.get('status', '')} | "
                        f"source_count={int(item.get('source_count', 0) or 0)} | "
                        f"quality_flags={trim_preview(str(item.get('quality_flags_json', '[]') or '[]'), 120)}"
                    )
                wiki_memory_block = "\n\n" + "\n".join(memory_lines) + "\n"
                _append_phase_event(
                    "wiki_memory",
                    "matched",
                    status="info",
                    payload={
                        "match_count": len(wiki_memory_matches),
                        "saved_answer_ids": [
                            int(item.get("saved_answer_id", 0) or 0)
                            for item in wiki_memory_matches
                        ],
                    },
                )
        except Exception as e:
            _append_phase_event(
                "wiki_memory",
                "failed",
                status="warning",
                detail=str(e),
                payload={"failure_code": _classify_failure_code(e, default="wiki_memory_search_fail")},
            )
    elif rag:
        _append_phase_event(
            "wiki_memory",
            "active_retrieval_disabled",
            status="info",
            payload={"prompt_hints": False, "retrieval_boost": False},
        )

    source_hint_block = f"\n\nSOURCE_HINTS:\n{source_hint}\n" if source_hint else "\n\nSOURCE_HINTS:\n(없음)\n"
    retrieval_meta_prompt_block = (
        f"\n\n{retrieval_meta_block}"
        if retrieval_meta_block
        else "\n\nRETRIEVAL_META:\n(없음)\n"
    )
    if wiki_memory_block:
        retrieval_meta_prompt_block += wiki_memory_block
    if wiki_page_hint_block:
        retrieval_meta_prompt_block += wiki_page_hint_block
    if bootstrap_notes:
        retrieval_meta_prompt_block += "\nBOOTSTRAP_NOTES:\n- " + "\n- ".join(bootstrap_notes) + "\n"

    def _append_tool_event(tool_name: str, arguments: Dict[str, Any], result_text: str):
        run_state.tool_events.append(
            ToolEventRecord(
                tool_name=tool_name,
                arguments=dict(arguments or {}),
                result_preview=trim_preview(result_text, 1200),
                created_at_iso=datetime.now(timezone.utc).isoformat(),
            )
        )
        _append_phase_event(
            "tool_call",
            tool_name,
            payload={
                "tool_name": tool_name,
                "arguments": dict(arguments or {}),
                "result_preview": trim_preview(result_text, 320),
            },
        )

    def _register_doc_rows(rows: List[Dict[str, Any]]) -> List[RetrievedDocRecord]:
        registered: List[RetrievedDocRecord] = []
        for row in rows:
            raw_row = dict(row)
            chunk_id = int(raw_row.get("id", 0) or 0)
            existing_doc_no = run_state.doc_numbers_by_chunk_id.get(chunk_id) if chunk_id > 0 else None
            if existing_doc_no:
                record = run_state.docs.get(existing_doc_no)
                if record is not None:
                    registered.append(record)
                    continue

            doc_no = max(run_state.docs.keys(), default=0) + 1
            record = RetrievedDocRecord(
                doc_no=doc_no,
                chunk_id=chunk_id,
                source_path=(raw_row.get("source_path", "") or "").strip(),
                source_ref=rag.format_source_ref(raw_row) if rag else (raw_row.get("source_path", "") or "").strip(),
                text=(raw_row.get("text", "") or "").strip(),
                source_type=(raw_row.get("source_type", "") or "").strip(),
                section=(raw_row.get("section", "") or "").strip(),
                sheet=(raw_row.get("sheet", "") or "").strip(),
                row=int(raw_row.get("row", 0) or 0),
                row_end=int(raw_row.get("row_end", raw_row.get("row", 0)) or 0),
                page_no=int(raw_row.get("page_no", 0) or 0),
                line_start=int(raw_row.get("line_start", 0) or 0),
                line_end=int(raw_row.get("line_end", raw_row.get("line_start", 0)) or 0),
                uploaded_at=int(raw_row.get("uploaded_at", raw_row.get("source_updated_at", 0)) or 0),
                score=float(raw_row.get("score", 0.0) or 0.0),
                metadata={"raw_row": raw_row},
            )
            run_state.docs[doc_no] = record
            if chunk_id > 0:
                run_state.doc_numbers_by_chunk_id[chunk_id] = doc_no
            registered.append(record)
        return registered

    def _seed_run_state(rows: List[Dict[str, Any]], metrics: Dict[str, Any], search_value: str):
        _register_doc_rows(rows)
        run_state.latest_metrics = dict(metrics or {})
        run_state.latest_search_query = (search_value or "").strip()
        run_state.latest_role_filter = _role_filter_label(retrieval_role_filter)
        run_state.latest_result_count = len(rows)

    def _list_current_sources_text() -> str:
        if not run_state.docs:
            return "(아직 없음)"
        lines: List[str] = []
        for doc_no in sorted(run_state.docs):
            record = run_state.docs[doc_no]
            lines.append(f"[DOC {doc_no}] {record.source_ref}")
        return "\n".join(lines)

    def _sql_citations_from_state(limit: int = 12) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for doc_no in sorted(run_state.docs)[: max(1, limit)]:
            record = run_state.docs[doc_no]
            citations.append(
                {
                    "doc_no": int(doc_no),
                    "chunk_id": int(record.chunk_id or 0),
                    "source_path": record.source_path,
                    "source_display": record.metadata.get("raw_row", {}).get("source_display", record.source_path),
                    "source_type": record.source_type,
                    "section": record.section,
                    "page_no": int(record.page_no or 0),
                    "sheet": record.sheet,
                    "row": int(record.row or 0),
                    "row_end": int(record.row_end or 0),
                    "line_start": int(record.line_start or 0),
                    "line_end": int(record.line_end or 0),
                    "uploaded_at": int(record.uploaded_at or 0),
                    "score": float(record.score or 0.0),
                    "source_ref": record.source_ref,
                }
            )
        return citations

    def _doc_catalog_metadata(limit: int = 12) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc_no in sorted(run_state.docs)[: max(1, limit)]:
            record = run_state.docs[doc_no]
            rows.append(
                {
                    "doc_no": int(doc_no),
                    "source_ref": record.source_ref,
                    "section": record.section,
                    "page_no": int(record.page_no or 0),
                    "sheet": record.sheet,
                    "row": int(record.row or 0),
                    "row_end": int(record.row_end or 0),
                    "line_start": int(record.line_start or 0),
                    "line_end": int(record.line_end or 0),
                    "score": float(record.score or 0.0),
                }
            )
        return rows

    def _normalize_tool_doc_role(doc_role_hint: str) -> Optional[List[str]]:
        hint = (doc_role_hint or "auto").strip().lower()
        if hint in {"", "auto", "mixed"}:
            return None
        if hint in {"all", "any"}:
            return None
        normalized = _normalize_doc_role(hint)
        if normalized in ALLOWED_UPLOAD_DOC_ROLES:
            return [normalized]
        return None

    if results:
        _seed_run_state(results, retrieval_metrics, search_query)
    history_strategy = ai_service.settings.history_strategy
    compact_history_block = rewrite_history_block if history_enabled else ""
    if history_enabled and history_strategy == "pydantic_messages":
        agent_history = _load_agent_message_history(session_id, kb_name, user_id=user_id)
        compact_history_block = ""
    else:
        agent_history = []
    _append_phase_event(
        "history",
        "loaded",
        status="info",
        payload={
            "strategy": history_strategy,
            "message_history_count": len(agent_history),
            "compact_history_chars": len(compact_history_block),
            "history_failed_turns_dropped": int(history_failed_turns_dropped),
            "compact_history_success_turns": int(compact_history_success_turns),
        },
    )
    agent_instruction_token_budget = int(os.getenv("PYDANTIC_AI_PROMPT_BASE_TOKENS", "1100"))

    def _context_char_budget() -> int:
        base_tokens = (
            agent_instruction_token_budget
            + _estimate_tokens(effective_user_message)
            + _estimate_tokens(ref_hint)
            + _estimate_tokens(source_hint_block)
            + _estimate_tokens(retrieval_meta_prompt_block)
            + LLM_PROMPT_OVERHEAD_TOKENS
            + LLM_MIN_RESPONSE_TOKENS
            + LLM_CONTEXT_SAFETY_MARGIN
        )
        remaining = max(120, LLM_CONTEXT_LIMIT - base_tokens)
        return max(320, int(remaining * RAG_CHARS_PER_TOKEN_EST))

    def _build_answer_runtime_instructions() -> str:
        lines = [
            f"오늘 날짜: {datetime.now(timezone.utc).date().isoformat()}",
            f"kb_name={kb_name}",
            "이전 대화가 있어도 현재 CONTEXT와 RETRIEVAL_META를 우선하라.",
            "답변은 자연스러운 한국어로 쓰고, 확신도 숫자나 정형 라벨을 억지로 만들지 마라.",
            "답변은 항상 공손하고 친절한 존댓말로 작성하라.",
            "답변에 마크다운 제목, 불릿, 번호 목록, 굵게, 백틱을 쓰지 말고 일반 문장과 줄바꿈만 사용하라.",
            "강조가 필요하면 괄호나 대괄호만 사용하라.",
        ]
        lines.extend(
            [
                "첫 문단은 결론 또는 처리 방향을 1~2문장으로 짧게 쓰고, 다음 내용과 빈 줄로 구분하라.",
                "비교, 정의, 처리 기준이 여러 개면 '기준: 내용', '처리: 내용'처럼 짧은 라벨 줄을 사용해 읽기 쉽게 정리하라.",
                "근거는 마지막에 '근거:' 한 줄로 따로 두고 [DOC i]를 붙여라.",
            ]
        )
        lines.extend(
            [
                "\ubb38\uc11c\uc5d0 \uba85\uc2dc, \ud56d\ubaa9\uc5d0 \uba85\uc2dc, \uc6d0\ubb38\uc744 \ud655\uc778, PDF\uc5d0\uc11c \ud655\uc778 \uac19\uc740 \ud45c\ud604\uc73c\ub85c \uc0ac\uc6a9\uc790\ub97c \ub2e4\uc2dc \uc790\ub8cc\ub85c \ub3cc\ub824\ubcf4\ub0b4\uc9c0 \ub9d0\ub77c.",
                "\ud544\uc694\ud558\uba74 '\ucc98\ub9ac: ...', '\uae30\uc900: ...', '\uc8fc\uc758: ...'\ucc98\ub7fc \uc9e7\uc740 \ub77c\ubca8 \uc904\ub85c \ubc14\ub85c \uc815\ub9ac\ud558\ub77c.",
            ]
        )
        if not _allow_general_knowledge_fallback(effective_user_message, kb_has_docs=kb_has_docs):
            lines.append(
                "문서 밖 참고 정보, 일반적 조언, 기관 문의 권고를 덧붙이지 말고 현재 문서에서 확인되는 사실과 확인되지 않는 부분만 답하라."
            )
        if compact_history_block:
            if question_analysis.numeric_evidence_required:
                lines.append(
                    "이전 대화의 숫자/금액 답변은 현재 문서 근거가 아니므로 그대로 따르지 말고 참고 히스토리로만 보라."
                )
            lines.append(compact_history_block)
        if question_analysis.intent_type and question_analysis.intent_type != "mixed":
            lines.append(f"사용자 질문 의도 분류: {question_analysis.intent_type}")
        if question_analysis.answer_focus:
            lines.append(
                "답변에서 우선 다룰 포인트: " + "; ".join(question_analysis.answer_focus[:3])
            )
        if question_analysis.require_tool_evidence:
            lines.append("답변 전에 tool로 근거를 다시 확인한 뒤 작성하라.")
        if question_analysis.use_source_outline:
            lines.append(
                "이 질문은 긴 문서 탐색 또는 구조 파악이 중요할 수 있다. 답변 전에 get_source_outline 또는 get_source_overview로 범위를 먼저 좁혀라."
            )
        if question_analysis.numeric_evidence_required:
            lines.append(
                "이 질문은 숫자/단가/금액/비율처럼 단위와 값이 중요하다. 문서에 적힌 숫자와 단위를 그대로 다시 확인하고, 추측 변환이나 보간을 하지 마라."
            )
            lines.append(
                "숫자 답변은 open_document 또는 outline/overview로 표나 본문 원문을 다시 확인한 뒤 작성하라."
            )
            lines.append(
                "표의미 또는 표행요약 근거가 있으면 질문 대상 항목과 같은 행의 값만 답하라. 다른 조사명, 다른 구분, 다른 행의 금액을 섞어 적용하지 마라."
            )
        if _allow_general_knowledge_fallback(effective_user_message, kb_has_docs=kb_has_docs):
            lines.append(
                "문서에 답이 없더라도 일상적이고 저위험인 질문이면, 먼저 문서에 없는 정보라고 밝힌 뒤 "
                "'참고로 일반적으로는 ...' 형식의 짧은 보충 설명을 덧붙여도 된다."
            )
        else:
            lines.append("이 질문은 일반상식으로 확장하지 말고 문서 근거 범위 안에서만 답하라.")
        if _is_summary_request(effective_user_message):
            lines.append(
                "질문이 요약/정리 요청이면 문서가 짧더라도 실제로 적힌 사실을 빠짐없이 먼저 정리하라. "
                "한 줄 문서라면 그 한 줄의 핵심 사실을 바로 결론으로 써라."
            )
        if question_analysis.literal_first or (len(run_state.docs) <= 2 and len((context_str or "").strip()) <= 900):
            lines.append(
                "현재 업로드 소스가 매우 짧을 수 있다. 문서에 실제로 적힌 표현과 그 한계를 먼저 짚고, "
                "해석이나 보충 설명은 그 다음에 분리해서 적어라."
            )
        if question_analysis.intent_type == "identity":
            lines.append(
                "정체성 질문이면 문서에 해당 정보가 있는지 먼저 말하고, 문서 밖 서비스 설명은 별도 참고 정보로 분리하라."
            )
        return "\n".join(lines)

    async def event_stream():
        nonlocal last_auto_prefetch_doc_nos
        ctx_max = RAG_CONTEXT_MAX_CHARS
        attempt = 0
        history_saved = False
        last_candidate = ""
        last_agent_messages_json = b""
        last_log_metadata: Dict[str, Any] = {}
        last_usage: Dict[str, Any] = {}
        last_context_chars = 0

        def _save_history_if_needed(
            assistant_text: str,
            new_messages_json: bytes = b"",
            *,
            metadata: Optional[Dict[str, Any]] = None,
            response_quality_issue: str = "",
            usage: Optional[Dict[str, Any]] = None,
            context_chars: int = 0,
        ):
            nonlocal history_saved
            if history_enabled and (not history_saved) and assistant_text.strip():
                _append_chat_message(session_id, kb_name, "user", user_message, user_id=user_id)
                _append_chat_message(session_id, kb_name, "assistant", assistant_text.strip(), user_id=user_id)
                _append_agent_run_history(
                    session_id=session_id,
                    kb_name=kb_name,
                    query_id=query_id,
                    user_message=user_message,
                    answer_text=assistant_text.strip(),
                    new_messages_json=new_messages_json,
                    metadata=metadata,
                    response_quality_issue=response_quality_issue,
                    usage=usage,
                    context_chars=context_chars,
                    user_id=user_id,
                )
                history_saved = True

        try:
            while attempt < 6:
                attempt += 1
                last_auto_prefetch_doc_nos = []

                if rag:
                    adaptive_max = min(ctx_max, _context_char_budget())
                    per_result_cap = _adaptive_per_result_context_cap(
                        adaptive_max,
                        len(results),
                    )
                    context_for_attempt = rag.get_context_string(
                        results,
                        query=effective_user_message,
                        max_chars=adaptive_max,
                        per_result_max_chars=per_result_cap,
                        focus_relevant=True,
                        top1_score=float(retrieval_metrics.get("top1", 0.0)),
                    )
                    if not context_for_attempt.strip():
                        context_for_attempt = rag.get_context_string(
                            results,
                            query="",
                            max_chars=adaptive_max,
                            per_result_max_chars=per_result_cap,
                            focus_relevant=False,
                        )

                    if overview_mode and results:
                        top_source = results[0].get("source_path", "")
                        overview_ctx = rag.get_source_overview_context(
                            top_source,
                            max_chars=min(RAG_OVERVIEW_EXTRA_CHARS, max(400, adaptive_max // 2)),
                        )
                        if overview_ctx:
                            context_for_attempt = _merge_context_with_cap(
                                context_for_attempt,
                                overview_ctx,
                                adaptive_max,
                            )
                    else:
                        context_for_attempt = _trim_to_max_chars(context_for_attempt, adaptive_max)
                else:
                    context_for_attempt = context_str

                prompt_tokens = (
                    agent_instruction_token_budget
                    + _estimate_tokens(effective_user_message)
                    + _estimate_tokens(ref_hint)
                    + _estimate_tokens(source_hint_block)
                    + _estimate_tokens(retrieval_meta_prompt_block)
                    + _estimate_tokens(context_for_attempt)
                    + LLM_PROMPT_OVERHEAD_TOKENS
                )
                dynamic_max_tokens = min(
                    LLM_MAX_TOKENS,
                    max(32, LLM_CONTEXT_LIMIT - prompt_tokens - LLM_CONTEXT_SAFETY_MARGIN),
                )
                if rag and dynamic_max_tokens < LLM_MIN_RESPONSE_TOKENS and ctx_max > RAG_CONTEXT_RETRY_MIN_CHARS:
                    _append_phase_event(
                        "answer_attempt",
                        "context_retry",
                        status="retry",
                        detail="dynamic_max_tokens_below_min_response",
                        payload={
                            "attempt": attempt,
                            "ctx_max": ctx_max,
                            "per_result_max": per_result_cap,
                            "dynamic_max_tokens": dynamic_max_tokens,
                        },
                    )
                    ctx_max = max(RAG_CONTEXT_RETRY_MIN_CHARS, int(ctx_max * 0.75))
                    continue

                def _search_tool(search_term: str, top_k: int, doc_role_hint: str) -> str:
                    if not rag:
                        return ""

                    query_for_tool = (search_term or "").strip() or effective_user_message
                    keep_n = min(12, max(3, int(top_k or 8)))
                    candidate_k = max(keep_n, min(RAG_LLM_RERANK_CANDIDATES, keep_n * 4))
                    tool_search_variants = _dedupe_text_items(
                        [query_for_tool, _expand_query_with_number_refs(query_for_tool)],
                        limit=4,
                    )
                    local_number_refs = [int(x) for x in NUMBER_REF_PATTERN.findall(query_for_tool)]

                    explicit_role_filter = _normalize_tool_doc_role(doc_role_hint)
                    if (doc_role_hint or "").strip().lower() in {"all", "any"}:
                        role_plan = [None]
                    elif explicit_role_filter is not None:
                        role_plan = [explicit_role_filter]
                    else:
                        tool_query_intent = _classify_query_doc_intent(query_for_tool)
                        role_plan = _build_role_search_plan(tool_query_intent)

                    best_rows: List[Dict[str, Any]] = []
                    best_metrics: Dict[str, Any] = {}
                    best_role_filter: Optional[List[str]] = None
                    tool_critical_meta: Dict[str, Any] = {}
                    gate_passed = False
                    for role_filter in role_plan:
                        candidates = _collect_search_candidates(
                            rag,
                            tool_search_variants,
                            top_k=candidate_k,
                            index_name="large",
                            doc_roles=role_filter,
                        )
                        guarded, anchor_match = _apply_number_reference_guard(
                            results=candidates,
                            number_refs=local_number_refs,
                            keep_n=keep_n,
                        )
                        guarded = _prefer_source_chunks(
                            guarded,
                            keep_n=keep_n,
                            max_normalized=RAG_MAX_NORMALIZED_RESULTS,
                        )
                        guarded, candidate_critical_meta = apply_critical_term_gate(
                            query_for_tool,
                            guarded,
                            enabled=RAG_CRITICAL_TERM_GATE_ENABLED,
                            require_raw_backing_for_normalized=RAG_NORMALIZED_RAW_BACKING_REQUIRED,
                        )
                        if candidate_critical_meta:
                            tool_critical_meta = candidate_critical_meta
                        if RAG_DIVERSIFY_ENABLED and guarded:
                            guarded = _apply_diversity_filter(
                                guarded,
                                keep_n=keep_n,
                                max_per_section=RAG_MAX_PER_SECTION,
                                max_per_file=RAG_MAX_PER_FILE,
                            )

                        metrics = rag.evaluate_answerability(
                            query_for_tool,
                            guarded,
                            coverage_top_k=RAG_ANSWER_COVERAGE_TOP_K,
                        )
                        selection_score = float(metrics.get("top1", 0.0)) + (float(metrics.get("coverage", 0.0)) * 0.35)
                        best_score = float(best_metrics.get("top1", 0.0)) + (float(best_metrics.get("coverage", 0.0)) * 0.35)
                        if guarded and (not best_rows or selection_score > best_score):
                            best_rows = list(guarded)
                            best_metrics = metrics
                            best_role_filter = role_filter
                        if guarded and _passes_grounding_gate(query_for_tool, metrics, guarded):
                            best_rows = list(guarded)
                            best_metrics = metrics
                            best_role_filter = role_filter
                            gate_passed = True
                            break

                    run_state.latest_search_query = query_for_tool
                    run_state.latest_role_filter = _role_filter_label(best_role_filter)
                    run_state.latest_metrics = dict(best_metrics or {})
                    run_state.latest_result_count = len(best_rows)

                    if best_rows:
                        registered_docs = _register_doc_rows(best_rows)
                    else:
                        registered_docs = []

                    trace_rows = best_rows[: max(keep_n, RAG_TRACE_TOP_N)]
                    _append_rag_trace_log(
                        {
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "kb_name": kb_name,
                            "session_id": session_id,
                            "query": user_message,
                            "effective_query": effective_user_message,
                            "search_query": query_for_tool,
                            "search_variants": tool_search_variants[:4],
                            "query_doc_intent": _classify_query_doc_intent(query_for_tool),
                            "retrieval_role_filter": _role_filter_label(best_role_filter),
                            "overview_mode": bool(overview_mode),
                            "number_refs": local_number_refs[:8],
                            "gate_passed": bool(gate_passed),
                            "metrics": dict(best_metrics or {}),
                            "critical_terms": tool_critical_meta.get("critical_terms", []),
                            "critical_term_hits": tool_critical_meta.get("critical_term_hits", {}),
                            "critical_term_gate_passed": bool(tool_critical_meta.get("critical_term_gate_passed", True)),
                            "literal_title_hit_count": int(tool_critical_meta.get("literal_title_hit_count", 0) or 0),
                            "normalized_only_blocked": bool(tool_critical_meta.get("normalized_only_blocked", False)),
                            "related_terms_suggestions": tool_critical_meta.get("related_terms_suggestions", []),
                            "result_count_after_gate": len(best_rows),
                            "top_results": _build_trace_results(rag, trace_rows, limit=RAG_TRACE_TOP_N),
                            "tool_origin": "search_knowledge_base",
                        }
                    )
                    try:
                        rag.log_retrieval(
                            query_id=query_id,
                            user_id=session_id,
                            query_text=query_for_tool,
                            topk_ids=[
                                int(r.get("id", 0) or 0)
                                for r in trace_rows
                                if int(r.get("id", 0) or 0) > 0
                            ],
                            meta={
                                "kb_name": kb_name,
                                "search_query": query_for_tool,
                                "search_variants": tool_search_variants[:4],
                                "query_doc_intent": _classify_query_doc_intent(query_for_tool),
                                "retrieval_role_filter": _role_filter_label(best_role_filter),
                                "overview_mode": bool(overview_mode),
                                "number_refs": local_number_refs[:8],
                                "gate_passed": bool(gate_passed),
                                "metrics": best_metrics,
                                "critical_terms": tool_critical_meta.get("critical_terms", []),
                                "critical_term_hits": tool_critical_meta.get("critical_term_hits", {}),
                                "critical_term_gate_passed": bool(tool_critical_meta.get("critical_term_gate_passed", True)),
                                "literal_title_hit_count": int(tool_critical_meta.get("literal_title_hit_count", 0) or 0),
                                "normalized_only_blocked": bool(tool_critical_meta.get("normalized_only_blocked", False)),
                                "related_terms_suggestions": tool_critical_meta.get("related_terms_suggestions", []),
                                "tool_origin": "search_knowledge_base",
                            },
                        )
                    except Exception:
                        pass

                    if not registered_docs:
                        payload = (
                            "SEARCH_RESULT:\n"
                            f"- search_query={query_for_tool}\n"
                            f"- retrieval_role_filter={_role_filter_label(best_role_filter)}\n"
                            "- result_count=0\n"
                            "- note=현재 검색어로는 문서 근거를 찾지 못했다. 더 구체적인 키워드나 코드/번호/페이지 힌트를 넣어 다시 검색하라."
                        )
                        _append_tool_event(
                            "search_knowledge_base",
                            {"search_query": query_for_tool, "top_k": keep_n, "doc_role": doc_role_hint},
                            payload,
                        )
                        return payload

                    source_lines = [
                        (
                            f"[DOC {record.doc_no}] {record.source_ref}\n"
                            f"snippet={trim_preview(record.text, 340)}"
                        )
                        for record in registered_docs
                    ]
                    payload = (
                        "SEARCH_RESULT:\n"
                        f"- search_query={query_for_tool}\n"
                        f"- retrieval_role_filter={_role_filter_label(best_role_filter)}\n"
                        f"- result_count={len(registered_docs)}\n"
                        f"- top1={float(best_metrics.get('top1', 0.0)):.3f}\n"
                        f"- coverage={float(best_metrics.get('coverage', 0.0)):.3f}\n"
                        f"- has_conflict={'yes' if bool(best_metrics.get('has_conflict', False)) else 'no'}\n\n"
                        + "\n\n".join(source_lines)
                    )
                    _append_tool_event(
                        "search_knowledge_base",
                        {"search_query": query_for_tool, "top_k": keep_n, "doc_role": doc_role_hint},
                        payload,
                    )
                    return payload

                def _open_document_payload(doc_no: int, max_chars: int) -> tuple[str, bool]:
                    if not rag:
                        return "", True
                    record = run_state.docs.get(int(doc_no))
                    if record is None:
                        return "", True
                    raw_row = dict(record.metadata.get("raw_row", {}))
                    lazy_ocr_text = rag.get_lazy_pdf_page_text_for_row(raw_row)
                    detail_row = dict(raw_row)
                    if lazy_ocr_text:
                        detail_row["text"] = lazy_ocr_text
                    detail = rag.get_context_string(
                        [detail_row],
                        query=effective_user_message,
                        max_chars=max_chars,
                        per_result_max_chars=max_chars,
                        focus_relevant=True,
                        top1_score=float(record.score or 0.0),
                    )
                    if not detail.strip() and lazy_ocr_text:
                        detail = rag.get_context_string(
                            [raw_row],
                            query=effective_user_message,
                            max_chars=max_chars,
                            per_result_max_chars=max_chars,
                            focus_relevant=True,
                            top1_score=float(record.score or 0.0),
                        )
                    if not detail.strip():
                        detail = rag.get_context_string(
                            [detail_row],
                            query="",
                            max_chars=max_chars,
                            per_result_max_chars=max_chars,
                            focus_relevant=False,
                        )
                    if not detail.strip() and lazy_ocr_text:
                        detail = rag.get_context_string(
                            [raw_row],
                            query="",
                            max_chars=max_chars,
                            per_result_max_chars=max_chars,
                            focus_relevant=False,
                        )
                    detail = _renumber_doc_labels(detail, record.doc_no).strip()
                    if not detail:
                        detail = f"[DOC {record.doc_no}]\nfile={record.source_path}\ntext={trim_preview(record.text, max_chars)}"
                    evidence_text = (lazy_ocr_text or detail or record.text or "").strip()
                    detail_is_weak = is_weak_ocr_hint_text(evidence_text)
                    payload = f"DOCUMENT_DETAIL:\n[DOC {record.doc_no}] {record.source_ref}\n\n{detail}"
                    _append_tool_event(
                        "open_document",
                        {
                            "doc_no": int(doc_no),
                            "max_chars": int(max_chars),
                            "lazy_ocr": bool(lazy_ocr_text),
                            "evidence_strength": "weak" if detail_is_weak else "strong",
                        },
                        payload,
                    )
                    return payload, detail_is_weak

                def _open_document_tool(doc_no: int, max_chars: int) -> str:
                    payload, _ = _open_document_payload(doc_no, max_chars)
                    return payload

                def _source_overview_tool(doc_no: int, max_chars: int) -> str:
                    if not rag:
                        return ""
                    record = run_state.docs.get(int(doc_no))
                    if record is None:
                        return ""
                    overview_text = rag.get_source_overview_context(
                        record.source_path,
                        max_chars=max_chars,
                    )
                    if not overview_text.strip():
                        return ""
                    payload = f"SOURCE_OVERVIEW:\n[DOC {record.doc_no}] {record.source_ref}\n\n{overview_text.strip()}"
                    _append_tool_event(
                        "get_source_overview",
                        {"doc_no": int(doc_no), "max_chars": int(max_chars)},
                        payload,
                    )
                    return payload

                def _source_outline_tool(doc_no: int, max_chars: int) -> str:
                    if not rag:
                        return ""
                    record = run_state.docs.get(int(doc_no))
                    if record is None:
                        return ""
                    outline_text = rag.get_source_outline_context(
                        record.source_path,
                        max_chars=max_chars,
                    )
                    if not outline_text.strip():
                        return ""
                    payload = f"SOURCE_OUTLINE:\n[DOC {record.doc_no}] {record.source_ref}\n\n{outline_text.strip()}"
                    _append_tool_event(
                        "get_source_outline",
                        {"doc_no": int(doc_no), "max_chars": int(max_chars)},
                        payload,
                    )
                    return payload

                def _list_sources_tool() -> str:
                    payload = _list_current_sources_text()
                    _append_tool_event("list_current_sources", {}, payload)
                    return payload

                tool_event_baseline = len(run_state.tool_events)
                if should_auto_prefetch_numeric_evidence(
                    require_tool_evidence=bool(question_analysis.require_tool_evidence),
                    allow_retrieval_tool=bool(rag and ai_service.settings.enable_retrieval_tool),
                    docs_available=len(run_state.docs),
                    metrics=run_state.latest_metrics or retrieval_metrics,
                    new_tool_event_count=0,
                    numeric_evidence_required=bool(question_analysis.numeric_evidence_required),
                ):
                    evidence_summary = _doc_evidence_summary()
                    def _fact_match_rank(record: RetrievedDocRecord) -> int:
                        raw_text = record.text or ""
                        for line in raw_text.splitlines():
                            fact = parse_table_fact_line(line)
                            if fact and fact.get("kind") == "table_row" and table_fact_matches_query(fact, effective_user_message):
                                return 1
                        return 0

                    ranked_docs = select_auto_prefetch_documents(
                        effective_user_message,
                        sorted(
                            run_state.docs.values(),
                            key=lambda record: (
                                int(bool((record.metadata.get("raw_row", {}) or {}).get("weak_ocr_hint", False))),
                                -_fact_match_rank(record),
                                -float((record.metadata.get("raw_row", {}) or {}).get("numeric_table_boost", 0.0) or 0.0),
                                -int(bool("표의미: kind=table_row" in (record.text or ""))),
                                -float(record.score or 0.0),
                                -int(record.uploaded_at or 0),
                                int(record.doc_no or 0),
                            ),
                        ),
                        limit=2,
                    )
                    if ranked_docs:
                        auto_open_max_chars = min(2200, max(900, adaptive_max // 2))
                        candidate_limit = 4 if bool(evidence_summary.get("weak_evidence_only", False)) else 2
                        strong_prefetch_count = 0
                        for auto_doc in ranked_docs[:candidate_limit]:
                            auto_detail, detail_is_weak = _open_document_payload(int(auto_doc.doc_no), auto_open_max_chars)
                            if not auto_detail.strip():
                                continue
                            if detail_is_weak:
                                continue
                            last_auto_prefetch_doc_nos.append(int(auto_doc.doc_no))
                            context_for_attempt = _merge_context_with_cap(
                                context_for_attempt,
                                auto_detail,
                                adaptive_max,
                            )
                            strong_prefetch_count += 1
                            if strong_prefetch_count >= 2:
                                break
                    if last_auto_prefetch_doc_nos:
                        primary_doc_no = int(last_auto_prefetch_doc_nos[0])
                        _append_phase_event(
                            "answer_attempt",
                            "auto_prefetch",
                            status="info",
                            payload={
                                "attempt": attempt,
                                "doc_no": primary_doc_no,
                                "doc_nos": list(last_auto_prefetch_doc_nos[:2]),
                                "doc_count": len(last_auto_prefetch_doc_nos),
                                "auto_prefetch_doc_count": len(last_auto_prefetch_doc_nos),
                                "max_chars": auto_open_max_chars,
                                "reason": "numeric_table_question",
                            },
                        )

                def _citation_issue(answer_text: str) -> str:
                    raw_answer = canonicalize_doc_citations(answer_text or "").strip()
                    if not raw_answer:
                        return ""
                    evidence_summary = _doc_evidence_summary()
                    cited_doc_numbers = sorted({int(v) for v in DOC_LABEL_PATTERN.findall(raw_answer)})
                    if _is_grounded_abstention(raw_answer):
                        invalid = [n for n in cited_doc_numbers if n not in run_state.docs]
                        if invalid:
                            return (
                                f"존재하지 않는 인용 {', '.join(f'[DOC {n}]' for n in invalid)}를 사용했다. "
                                f"현재 근거는 {_list_current_sources_text()} 뿐이다."
                            )
                        return ""
                    if not run_state.docs:
                        return "먼저 search_knowledge_base로 문서 근거를 확보한 뒤 답하라."
                    if not cited_doc_numbers:
                        if bool(evidence_summary.get("weak_evidence_only", False)):
                            return (
                                "현재 확보한 문서는 OCR 후보 힌트뿐이라 사실을 단정하면 안 된다. "
                                "문서에 실제 문장이나 숫자가 확인되지 않으면 "
                                "'현재 열람한 문서에서는 확인되지 않는다'라고 답하고 바깥 지식은 쓰지 마라."
                            )
                        return f"답변에 [DOC n] 또는 [[CITATION:n|...]] 인용이 없다. 현재 사용 가능한 근거: {_list_current_sources_text()}"
                    invalid = [n for n in cited_doc_numbers if n not in run_state.docs]
                    if invalid:
                        return (
                            f"존재하지 않는 인용 {', '.join(f'[DOC {n}]' for n in invalid)}를 사용했다. "
                            f"현재 근거는 {_list_current_sources_text()} 뿐이다."
                        )
                    if contains_disallowed_markdown(raw_answer):
                        return (
                            "답변은 마크다운 없이 일반 문장과 줄바꿈으로만 작성하라. "
                            "강조는 괄호나 대괄호로 표현하라."
                        )
                    if question_analysis.numeric_evidence_required:
                        if bool(evidence_summary.get("weak_evidence_only", False)):
                            return (
                                "현재 확보한 문서는 OCR 후보 힌트뿐이라 숫자/단가를 확정할 수 없다. "
                                "실제 값과 단위가 보이지 않으면 '현재 열람한 문서에서는 확인되지 않는다'라고 답하고 "
                                "추정이나 외부 지식을 쓰지 마라."
                            )
                        evidence_texts = [context_for_attempt]
                        evidence_texts.extend(record.text for record in run_state.docs.values())
                        if not has_grounded_numeric_answer(
                            query=effective_user_message,
                            answer_text=raw_answer,
                            evidence_texts=evidence_texts,
                        ):
                            return (
                                "숫자/단가 답변의 값 또는 단위가 현재 문서 근거와 일치하지 않는다. "
                                "open_document 또는 outline/overview로 숫자와 단위를 다시 확인하라."
                            )
                    return ""

                retrieval_snapshot = RetrievalSnapshot(
                    context=context_for_attempt,
                    retrieval_meta=(retrieval_meta_block or "RETRIEVAL_META:\n(없음)\n").strip(),
                    source_hint=_list_current_sources_text(),
                    reference_hint=(f"사용자 질문에서 언급한 번호: {', '.join(str(n) for n in number_refs[:5])}" if number_refs else ""),
                    query_doc_intent=query_doc_intent,
                    retrieval_role_filter=run_state.latest_role_filter or _role_filter_label(retrieval_role_filter),
                    search_query=run_state.latest_search_query or search_query,
                    metrics=run_state.latest_metrics or retrieval_metrics,
                    overview_mode=overview_mode,
                )
                seeded_evidence_sufficient = _seeded_evidence_sufficient_for_answer()
                allow_answer_retrieval_tool = bool(
                    rag
                    and ai_service.settings.enable_retrieval_tool
                    and not seeded_evidence_sufficient
                )
                if seeded_evidence_sufficient:
                    _append_phase_event(
                        "answer_attempt",
                        "retrieval_tools_disabled",
                        status="info",
                        detail="seeded_evidence_sufficient",
                        payload={
                            "attempt": attempt,
                            "docs_available": len(run_state.docs),
                            "strong_evidence_count": int(_doc_evidence_summary().get("strong_evidence_count", 0) or 0),
                            "metrics_top1": float((run_state.latest_metrics or retrieval_metrics or {}).get("top1", 0.0) or 0.0),
                            "metrics_coverage": float((run_state.latest_metrics or retrieval_metrics or {}).get("coverage", 0.0) or 0.0),
                            "numeric_evidence_required": bool(question_analysis.numeric_evidence_required),
                            "use_source_outline": bool(question_analysis.use_source_outline),
                        },
                    )
                deps = CompassAgentDeps(
                    kb_name=kb_name,
                    query_id=query_id,
                    user_message=effective_user_message,
                    retrieval=retrieval_snapshot,
                    runtime_date_iso=datetime.now(timezone.utc).date().isoformat(),
                    question_analysis=question_analysis,
                    run_state=run_state,
                    quality_checker=_response_quality_issue if LLM_QUALITY_RETRY_ENABLED else None,
                    quality_hint_builder=_quality_retry_hint if LLM_QUALITY_RETRY_ENABLED else None,
                    search_tool=_search_tool,
                    open_document_tool=_open_document_tool,
                    source_overview_tool=_source_overview_tool,
                    source_outline_tool=_source_outline_tool,
                    list_sources_tool=_list_sources_tool,
                    citation_validator=_citation_issue,
                    citation_repairer=_repair_missing_answer_citations,
                    answer_sanitizer=_sanitize_outside_document_claims,
                    require_tool_evidence=bool(question_analysis.require_tool_evidence),
                    tool_event_baseline=tool_event_baseline,
                    allow_retrieval_tool=allow_answer_retrieval_tool,
                )
                _append_phase_event(
                    "answer_attempt",
                    "started",
                    status="info",
                    payload={
                        "attempt": attempt,
                        "context_chars": len(context_for_attempt),
                        "prompt_tokens_est": prompt_tokens,
                        "dynamic_max_tokens": dynamic_max_tokens,
                        "docs_available": len(run_state.docs),
                        "seeded_evidence_sufficient": bool(seeded_evidence_sufficient),
                        "allow_retrieval_tool": bool(allow_answer_retrieval_tool),
                    },
                )
                try:
                    run_result = await ai_service.answer_question(
                        deps,
                        message_history=agent_history or None,
                        max_tokens=dynamic_max_tokens,
                        temperature=LLM_TEMPERATURE,
                        top_p=LLM_TOP_P,
                        timeout=LLM_REQUEST_TIMEOUT,
                        runtime_instructions=_build_answer_runtime_instructions(),
                        run_metadata={
                            "session_id": session_id,
                            "search_query": search_query,
                            "query_doc_intent": query_doc_intent,
                            "question_intent": question_analysis.intent_type,
                            "retrieval_role_filter": _role_filter_label(retrieval_role_filter),
                            "overview_mode": bool(overview_mode),
                        },
                    )
                except Exception as e:
                    error_body = str(e)
                    failure_code = _classify_failure_code(e, default="answer_run_fail")
                    lowered = error_body.lower()
                    context_error = (
                        "context" in lowered
                        and ("size" in lowered or "token" in lowered or "maximum" in lowered or "available" in lowered)
                    )
                    if rag and context_error and ctx_max > RAG_CONTEXT_RETRY_MIN_CHARS:
                        _append_phase_event(
                            "answer_attempt",
                            "failed",
                            status="retry",
                            detail=error_body,
                            payload={"attempt": attempt, "reason": "context_window", "failure_code": "context_overflow"},
                        )
                        ctx_max = max(RAG_CONTEXT_RETRY_MIN_CHARS, ctx_max // 2)
                        continue
                    _append_phase_event(
                        "answer_attempt",
                        "failed",
                        status="error",
                        detail=error_body,
                        payload={"attempt": attempt, "failure_code": failure_code},
                    )
                    raise

                candidate = (run_result.output or "").strip()
                last_candidate = candidate
                rendered_candidate = _render_user_visible_answer(candidate)
                last_context_chars = len(context_for_attempt)
                try:
                    last_agent_messages_json = run_result.new_messages_json()
                except Exception:
                    last_agent_messages_json = b""
                last_usage = run_usage_to_dict(run_result.usage())
                _append_phase_event(
                    "answer_attempt",
                    "completed",
                    status="ok" if candidate else "empty",
                    payload={
                        "attempt": attempt,
                        "answer_chars": len(candidate),
                        "tool_calls": int(last_usage.get("tool_calls", 0) or 0),
                        "requests": int(last_usage.get("requests", 0) or 0),
                        "input_tokens": int(last_usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(last_usage.get("output_tokens", 0) or 0),
                    },
                )

                if not candidate:
                    candidate = _no_evidence_response("응답이 비어 있어 재생성에 실패했다")
                    last_candidate = candidate
                    last_log_metadata = _build_answer_log_metadata(
                        response_quality_issue="empty_after_run",
                        failure_code="answer_empty",
                        usage=last_usage,
                        context_chars=last_context_chars,
                        prompt_tokens=prompt_tokens,
                        max_tokens=dynamic_max_tokens,
                        answer_text=candidate,
                        attempt_index=attempt,
                    )
                else:
                    last_log_metadata = _build_answer_log_metadata(
                        usage=last_usage,
                        context_chars=last_context_chars,
                        prompt_tokens=prompt_tokens,
                        max_tokens=dynamic_max_tokens,
                        answer_text=candidate,
                        attempt_index=attempt,
                    )

                _log_answer_to_sql(candidate, metadata=last_log_metadata)
                _save_history_if_needed(
                    rendered_candidate,
                    last_agent_messages_json,
                    metadata=last_log_metadata,
                    response_quality_issue=str(last_log_metadata.get("response_quality_issue", "")),
                    usage=last_usage,
                    context_chars=last_context_chars,
                )
                yield rendered_candidate
                return
        except Exception as e:
            failure_code = _classify_failure_code(e, default="answer_stream_fail")
            _append_phase_event(
                "response_stream",
                "failed",
                status="error",
                detail=str(e),
                payload={"attempt": attempt, "failure_code": failure_code},
            )
            if last_candidate.strip():
                if not last_log_metadata:
                    last_log_metadata = _build_answer_log_metadata(
                        response_quality_issue="exception_after_candidate",
                        failure_code=failure_code,
                        usage=last_usage,
                        context_chars=last_context_chars,
                        answer_text=last_candidate,
                        attempt_index=attempt,
                    )
                _log_answer_to_sql(last_candidate, metadata=last_log_metadata)
                rendered_last_candidate = _render_user_visible_answer(last_candidate)
                _save_history_if_needed(
                    rendered_last_candidate,
                    last_agent_messages_json,
                    metadata=last_log_metadata,
                    response_quality_issue=str(last_log_metadata.get("response_quality_issue", "")),
                    usage=last_usage,
                    context_chars=last_context_chars,
                )
                yield rendered_last_candidate
                return
            fallback_text = _no_evidence_response("응답 생성 중 문제가 생겼다")
            fallback_metadata = _build_answer_log_metadata(
                response_quality_issue="run_exception",
                failure_code=failure_code,
                answer_text=fallback_text,
                attempt_index=attempt,
            )
            _log_answer_to_sql(fallback_text, metadata=fallback_metadata)
            _save_history_if_needed(
                fallback_text,
                metadata=fallback_metadata,
                response_quality_issue="run_exception",
            )
            yield fallback_text

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    if (
        rag_lease_state.get("context") is not None
        and not bool(rag_lease_state.get("transferred_to_response", False))
    ):
        rag_lease_state["transferred_to_response"] = True
        response.background = BackgroundTask(_release_rag_lease)
    response.headers["X-Conversation-Mode"] = conversation_mode
    response.headers["X-Query-Id"] = query_id
    return _attach_session_cookie(response, session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
