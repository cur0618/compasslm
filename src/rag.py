import hashlib
import heapq
import json
import math
import os
import pickle
import re
import requests
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer_validation import is_numeric_evidence_query, is_weak_ocr_hint_text
from src.document_structure import build_embedding_text, chunk_structure_records, normalize_structure_record
from src.hwpx_loader import load_hwpx_records
from src.persistence_retention import prune_timestamped_rows
from src.rag_text_markers import (
    ANSWER_COLUMN_MARKERS,
    EMBEDDING_DIMENSION_PROBE_TEXT,
    LABEL_ANSWER_PRIORITY,
    LABEL_BODY,
    LABEL_LATEST,
    LABEL_LATEST_UPLOAD_REFLECTED,
    LABEL_LATEST_UPLOAD_REFLECTED_AT,
    LABEL_LINE,
    LABEL_LOCATION,
    LABEL_NO_LOCATION,
    LABEL_QUESTION,
    LABEL_ROW,
    LABEL_SOURCE,
    LABEL_SOURCE_SUMMARY,
    LABEL_UPLOAD,
    NORMALIZED_BUNDLE_LABEL,
    QUESTION_COLUMN_MARKERS,
    TABLE_HINT_MARKERS,
    TABLE_ROW_SUMMARY_MARKER,
    TABLE_SEMANTIC_ROW_MARKER,
    normalized_bundle_header,
    normalized_bundle_section,
)
try:
    from src.pdf_ocr import (
        extract_pdf_pages,
        extract_pdf_pages_with_paddleocr_vl,
        release_cached_ocr_model,
        shutdown_persistent_ocr_worker,
    )
except ImportError:
    from src.pdf_ocr import extract_pdf_pages, extract_pdf_pages_with_paddleocr_vl, release_cached_ocr_model

    def shutdown_persistent_ocr_worker(*, clear_error: bool = True) -> None:
        return None
from src.table_alias import alias_match_boost
from src.ontology_store import OntologyStore
from src.utils import chunk_txt_items, chunk_xlsx_rows, load_txt, load_xlsx
from src.wiki_store import WikiStore


RAG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.getenv("COMPASSLM_HOME", os.path.join(RAG_FILE_DIR, "..")))
DEFAULT_KB_DATA_DIR = os.path.abspath(os.getenv("KB_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "kb")))
DEFAULT_EMBEDDING_MODEL_LARGE_PATH = os.getenv(
    "EMBEDDING_MODEL_LARGE_PATH",
    os.path.join(PROJECT_ROOT, "models", "embed", "Qwen", "Qwen3-Embedding-0.6B"),
)
DEFAULT_QWEN_QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QWEN_QUERY_INSTRUCTION",
    "Given a user question, retrieve relevant passages that answer the question",
)
DOC_ROLE_GUIDE = "guide"
DOC_ROLE_CASEBOOK = "casebook"
DOC_ROLE_UNKNOWN = "unknown"
_HANGUL_TERM_CLASS = r"0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3"
_OVERLAP_TERM_RE = re.compile(rf"[{_HANGUL_TERM_CLASS}]+")
_SLUG_DISALLOWED_RE = re.compile(r"[^0-9a-z\u3131-\u318e\uac00-\ud7a3]+")
_QUOTED_LITERAL_RE = re.compile(
    r'["\'\u2018\u2019\u201c\u201d]([^"\'\u2018\u2019\u201c\u201d]{2,80})["\'\u2018\u2019\u201c\u201d]'
)
_MIXED_LITERAL_RE = re.compile(
    rf"[{_HANGUL_TERM_CLASS}][{_HANGUL_TERM_CLASS}_\-/]{{2,40}}"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。\u3002\uff01\uff1f])\s+")
DOC_ROLE_ALIASES = {
    "guide": DOC_ROLE_GUIDE,
    "guideline": DOC_ROLE_GUIDE,
    "guidelines": DOC_ROLE_GUIDE,
    "manual": DOC_ROLE_GUIDE,
    "instruction": DOC_ROLE_GUIDE,
    "rule": DOC_ROLE_GUIDE,
    "rules": DOC_ROLE_GUIDE,
    "casebook": DOC_ROLE_CASEBOOK,
    "case": DOC_ROLE_CASEBOOK,
    "cases": DOC_ROLE_CASEBOOK,
    "qa": DOC_ROLE_CASEBOOK,
    "q&a": DOC_ROLE_CASEBOOK,
    "faq": DOC_ROLE_CASEBOOK,
}

def _is_pymupdf_page_parser(parser: str) -> bool:
    return (parser or "").strip().lower().startswith("pymupdf")


def _survey_alias_match_boost(query: str, text: str) -> float:
    return alias_match_boost(query, text)


def _summarize_pdf_chunk_pages(
    items: List[Dict[str, Any]],
    fallback_parser: str = "paddleocr_vl",
) -> Dict[str, Any]:
    page_parsers: Dict[int, str] = {}
    for item in items:
        try:
            page_no = int(item.get("page_no", 0) or 0)
        except (TypeError, ValueError):
            page_no = 0
        if page_no <= 0:
            section = str(item.get("section", "") or "")
            match = re.search(r"PDF page\s+(\d+)", section, flags=re.IGNORECASE)
            if match:
                page_no = int(match.group(1))
        if page_no <= 0:
            continue
        parser = str(item.get("page_parser", "") or item.get("parser_name", "") or "").strip()
        if not parser:
            parser = fallback_parser
        page_parsers.setdefault(page_no, parser)

    text_pages = sum(1 for parser in page_parsers.values() if _is_pymupdf_page_parser(parser))
    ocr_pages = sum(1 for parser in page_parsers.values() if not _is_pymupdf_page_parser(parser))
    parser_name = fallback_parser
    if text_pages and ocr_pages:
        parser_name = "hybrid_pdf"
    elif text_pages:
        parser_name = "pymupdf_text"
    elif ocr_pages:
        parser_name = fallback_parser

    return _normalize_pdf_ingest_stats(
        {
            "pdf_parser": parser_name,
            "pdf_total_pages": len(page_parsers),
            "pdf_text_pages": text_pages,
            "pdf_ocr_pages": ocr_pages,
            "pdf_attempted_ocr_pages": ocr_pages,
            "pdf_failed_pages": 0,
            "pdf_warnings": [],
            "ocr_device_attempted": "",
            "ocr_device_effective": "",
            "ocr_gpu_fallback_used": False,
            "ocr_gpu_failure_reason": "",
            "ocr_elapsed_seconds": None,
            "ocr_pages_processed": None,
            "ocr_pages_per_minute": None,
            "ocr_pages_attempted": None,
            "ocr_pages_emitted": None,
            "ocr_pages_skipped_empty": None,
            "ocr_pages_skipped_short_text": None,
            "ocr_attempted_pages_per_minute": None,
            "ocr_emitted_pages_per_minute": None,
            "ocr_worker_released": False,
            "ocr_worker_release_seconds": 0.0,
            "ocr_worker_pids": [],
            "ocr_worker_shutdown_confirmed": True,
            "ocr_worker_alive_after_shutdown": [],
            "ocr_target_pages": None,
            "ocr_target_seconds": None,
            "ocr_target_met": None,
            "ocr_subset_build_seconds": None,
            "ocr_model_load_seconds": None,
            "ocr_predict_seconds": None,
            "ocr_output_materialize_seconds": None,
            "ocr_payload_convert_seconds": None,
            "ocr_fragment_collect_seconds": None,
            "ocr_page_dedupe_seconds": None,
            "ocr_page_join_seconds": None,
            "ocr_text_merge_seconds": None,
            "ocr_merge_seconds": None,
            "ocr_batch_count": None,
            "ocr_backend": "",
            "ocr_backend_attempted": "",
            "ocr_backend_effective": "",
            "ocr_backend_fallback_used": False,
            "ocr_fast_pages": 0,
            "ocr_vl_pages": 0,
            "ocr_fast_seconds": 0.0,
            "ocr_vl_seconds": 0.0,
            "ocr_fast_avg_score": 0.0,
            "ocr_fast_pair_ratio": 0.0,
            "ocr_fast_orphan_ratio": 0.0,
            "ocr_high_quality_requested": False,
        },
        fallback_parser=fallback_parser,
    )


def _normalize_pdf_ingest_stats(
    payload: Optional[Dict[str, Any]],
    fallback_parser: str = "paddleocr_vl",
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    text_pages = max(0, int(payload.get("pdf_text_pages", 0) or 0))
    ocr_pages = max(0, int(payload.get("pdf_ocr_pages", 0) or 0))
    parser_name = str(payload.get("pdf_parser", "") or "").strip()
    if not parser_name:
        if text_pages and ocr_pages:
            parser_name = "hybrid_pdf"
        elif text_pages:
            parser_name = "pymupdf_text"
        elif ocr_pages:
            parser_name = fallback_parser
    warnings = [str(item).strip() for item in (payload.get("pdf_warnings", []) or []) if str(item).strip()]
    def _int_list(value: Any) -> List[int]:
        result: List[int] = []
        for item in list(value or []):
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                result.append(parsed)
        return sorted(set(result))

    return {
        "pdf_parser": parser_name or fallback_parser,
        "pdf_total_pages": max(0, int(payload.get("pdf_total_pages", 0) or 0)),
        "pdf_text_pages": text_pages,
        "pdf_ocr_pages": ocr_pages,
        "pdf_attempted_ocr_pages": max(0, int(payload.get("pdf_attempted_ocr_pages", payload.get("attempted_ocr_pages", 0)) or 0)),
        "pdf_failed_pages": max(0, int(payload.get("pdf_failed_pages", 0) or 0)),
        "pdf_warnings": warnings,
        "ocr_device_attempted": str(payload.get("ocr_device_attempted", "") or "").strip(),
        "ocr_device_effective": str(payload.get("ocr_device_effective", "") or "").strip(),
        "ocr_gpu_fallback_used": bool(payload.get("ocr_gpu_fallback_used", False)),
        "ocr_gpu_failure_reason": str(payload.get("ocr_gpu_failure_reason", "") or "").strip(),
        "ocr_elapsed_seconds": payload.get("ocr_elapsed_seconds"),
        "ocr_pages_processed": payload.get("ocr_pages_processed"),
        "ocr_pages_per_minute": payload.get("ocr_pages_per_minute"),
        "ocr_pages_attempted": payload.get("ocr_pages_attempted"),
        "ocr_pages_emitted": payload.get("ocr_pages_emitted"),
        "ocr_pages_skipped_empty": payload.get("ocr_pages_skipped_empty"),
        "ocr_pages_skipped_short_text": payload.get("ocr_pages_skipped_short_text"),
        "ocr_attempted_pages_per_minute": payload.get("ocr_attempted_pages_per_minute"),
        "ocr_emitted_pages_per_minute": payload.get("ocr_emitted_pages_per_minute"),
        "ocr_worker_released": bool(payload.get("ocr_worker_released", False)),
        "ocr_worker_release_seconds": float(payload.get("ocr_worker_release_seconds", 0.0) or 0.0),
        "ocr_worker_pids": _int_list(payload.get("ocr_worker_pids", [])),
        "ocr_worker_shutdown_confirmed": bool(payload.get("ocr_worker_shutdown_confirmed", True)),
        "ocr_worker_alive_after_shutdown": _int_list(payload.get("ocr_worker_alive_after_shutdown", [])),
        "ocr_target_pages": payload.get("ocr_target_pages"),
        "ocr_target_seconds": payload.get("ocr_target_seconds"),
        "ocr_target_met": payload.get("ocr_target_met"),
        "ocr_subset_build_seconds": payload.get("ocr_subset_build_seconds"),
        "ocr_model_load_seconds": payload.get("ocr_model_load_seconds"),
        "ocr_predict_seconds": payload.get("ocr_predict_seconds"),
        "ocr_output_materialize_seconds": payload.get("ocr_output_materialize_seconds"),
        "ocr_payload_convert_seconds": payload.get("ocr_payload_convert_seconds"),
        "ocr_fragment_collect_seconds": payload.get("ocr_fragment_collect_seconds"),
        "ocr_page_dedupe_seconds": payload.get("ocr_page_dedupe_seconds"),
        "ocr_page_join_seconds": payload.get("ocr_page_join_seconds"),
        "ocr_text_merge_seconds": payload.get("ocr_text_merge_seconds"),
        "ocr_merge_seconds": payload.get("ocr_merge_seconds"),
        "ocr_batch_count": payload.get("ocr_batch_count"),
        "ocr_backend": str(payload.get("ocr_backend", "") or "").strip(),
        "ocr_backend_attempted": str(payload.get("ocr_backend_attempted", "") or "").strip(),
        "ocr_backend_effective": str(payload.get("ocr_backend_effective", "") or "").strip(),
        "ocr_backend_fallback_used": bool(payload.get("ocr_backend_fallback_used", False)),
        "ocr_fast_pages": int(payload.get("ocr_fast_pages", 0) or 0),
        "ocr_vl_pages": int(payload.get("ocr_vl_pages", 0) or 0),
        "ocr_fast_seconds": float(payload.get("ocr_fast_seconds", 0.0) or 0.0),
        "ocr_vl_seconds": float(payload.get("ocr_vl_seconds", 0.0) or 0.0),
        "ocr_fast_avg_score": float(payload.get("ocr_fast_avg_score", 0.0) or 0.0),
        "ocr_fast_pair_ratio": float(payload.get("ocr_fast_pair_ratio", 0.0) or 0.0),
        "ocr_fast_orphan_ratio": float(payload.get("ocr_fast_orphan_ratio", 0.0) or 0.0),
        "ocr_high_quality_requested": bool(payload.get("ocr_high_quality_requested", False)),
        "chars_before_compaction": int(payload.get("chars_before_compaction", 0) or 0),
        "chars_after_compaction": int(payload.get("chars_after_compaction", 0) or 0),
        "lines_before_dedupe": int(payload.get("lines_before_dedupe", 0) or 0),
        "lines_after_dedupe": int(payload.get("lines_after_dedupe", 0) or 0),
        "hints_dropped": int(payload.get("hints_dropped", 0) or 0),
        "chunks_created": int(payload.get("chunks_created", 0) or 0),
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


ONTOLOGY_RAG_ENABLED = _env_bool("ONTOLOGY_RAG_ENABLED", True)
ONTOLOGY_LLM_EXTRACTION_ENABLED = _env_bool("ONTOLOGY_LLM_EXTRACTION_ENABLED", True)
ONTOLOGY_MAX_HOPS = max(1, min(2, int(os.getenv("ONTOLOGY_MAX_HOPS", "2"))))
ONTOLOGY_MIN_FACT_CONFIDENCE = float(os.getenv("ONTOLOGY_MIN_FACT_CONFIDENCE", "0.62"))
ONTOLOGY_WIKI_CONFIDENCE_BOOST = float(os.getenv("ONTOLOGY_WIKI_CONFIDENCE_BOOST", "0.08"))
ONTOLOGY_LLM_API_URL = os.getenv("ONTOLOGY_LLM_API_URL", os.getenv("LLM_API_URL", "http://127.0.0.1:8003/v1/chat/completions"))
ONTOLOGY_LLM_API_KEY = os.getenv("ONTOLOGY_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
ONTOLOGY_LLM_MODEL_NAME = os.getenv("ONTOLOGY_LLM_MODEL_NAME", os.getenv("LLM_MODEL_NAME", "local-model"))
ONTOLOGY_LLM_TIMEOUT_SECONDS = max(1, int(os.getenv("ONTOLOGY_LLM_TIMEOUT_SECONDS", "25")))
DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED = _env_bool("DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED", True)


def _looks_like_local_path(value: str) -> bool:
    if not value:
        return False
    v = value.strip()
    if v.startswith(("/", "./", "../", "~", ".\\")):
        return True
    if len(v) >= 2 and v[1] == ":":
        return True
    return os.path.exists(os.path.expanduser(v))


def _embedding_model_load_kwargs(model_ref: str) -> Dict[str, Any]:
    lowered = (model_ref or "").strip().lower()
    if "jina" in lowered and "embeddings-v5" in lowered:
        return {"trust_remote_code": True}
    return {}


def _embedding_model_requirements_hint(model_ref: str) -> str:
    lowered = (model_ref or "").strip().lower()
    if "jina" in lowered and "embeddings-v5" in lowered:
        return " Jina v5 계열은 transformers>=4.57.0, torch>=2.8.0, peft>=0.15.2 환경이 필요할 수 있습니다."
    return ""


def list_kbs(data_dir: str = DEFAULT_KB_DATA_DIR) -> List[str]:
    """List all available Knowledge Bases."""
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


def _kb_meta_db_path(kb_id: str, data_dir: str) -> str:
    return os.path.join(data_dir, kb_id, "meta.sqlite")


def _connect_meta_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _source_child_prefix(source_path: str) -> str:
    return f"{source_path}::part:"


def get_kb_files(kb_id: str, data_dir: str = DEFAULT_KB_DATA_DIR) -> List[Dict[str, Any]]:
    """List KB files with stable identifiers and display names."""
    db_path = _kb_meta_db_path(kb_id, data_dir)
    if os.path.exists(db_path):
        conn = _connect_meta_db(db_path)
        try:
            if _table_exists(conn, "files"):
                joins = ""
                select_original = "'' AS source_original_filename"
                if _table_exists(conn, "source_uploads"):
                    joins = "LEFT JOIN source_uploads s ON s.source_path = f.source_path"
                    select_original = "COALESCE(NULLIF(s.original_filename, ''), '') AS source_original_filename"
                rows = conn.execute(
                    f"""
                    SELECT
                        f.file_id,
                        f.source_path,
                        f.orig_name,
                        f.stored_path,
                        f.uploaded_at,
                        {select_original}
                    FROM files f
                    {joins}
                    ORDER BY COALESCE(f.uploaded_at, 0) DESC, COALESCE(f.source_path, '') ASC
                    """
                ).fetchall()
                files: List[Dict[str, Any]] = []
                for row in rows:
                    stored_path = str(row["stored_path"] or "")
                    stored_name = os.path.basename(stored_path) if stored_path else str(row["source_path"] or "")
                    display_name = (
                        str(row["orig_name"] or "").strip()
                        or str(row["source_original_filename"] or "").strip()
                        or str(row["source_path"] or "").strip()
                        or stored_name
                    )
                    files.append(
                        {
                            "file_id": str(row["file_id"] or stored_name),
                            "display_name": display_name,
                            "stored_name": stored_name,
                            "source_path": str(row["source_path"] or ""),
                            "uploaded_at": int(row["uploaded_at"] or 0),
                            "delete_key": str(row["file_id"] or stored_name),
                        }
                    )
                if files:
                    return files
        finally:
            conn.close()

    path = os.path.join(data_dir, kb_id, "uploads")
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(path)):
        out.append(
            {
                "file_id": name,
                "display_name": name,
                "stored_name": name,
                "source_path": name,
                "uploaded_at": 0,
                "delete_key": name,
            }
        )
    return out


def rename_kb_dir(old_id: str, new_id: str, data_dir: str = DEFAULT_KB_DATA_DIR):
    """Rename a KB directory."""
    old_path = os.path.join(data_dir, old_id)
    new_path = os.path.join(data_dir, new_id)
    if os.path.exists(new_path):
        raise ValueError(f"KB {new_id} already exists")
    os.rename(old_path, new_path)


def delete_kb_dir(kb_id: str, data_dir: str = DEFAULT_KB_DATA_DIR):
    """Delete a KB directory."""
    path = os.path.join(data_dir, kb_id)
    if os.path.exists(path):
        shutil.rmtree(path)


def _sql_source_match(column: str, source_paths: List[str]) -> Tuple[str, List[str]]:
    clauses: List[str] = []
    params: List[str] = []
    for source_path in sorted({str(value or "").strip() for value in source_paths if str(value or "").strip()}):
        escaped_prefix = _source_child_prefix(source_path).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append(f"({column} = ? OR {column} LIKE ? ESCAPE '\\')")
        params.extend((source_path, f"{escaped_prefix}%"))
    return (" OR ".join(clauses) or "0"), params


def _delete_rows_for_ids(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    values: List[int],
) -> None:
    if not values or not _table_exists(conn, table):
        return
    placeholders = ",".join("?" for _ in values)
    conn.execute(
        f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
        tuple(int(value) for value in values),
    )


def _delete_source_records(
    conn: sqlite3.Connection,
    source_paths: List[str],
    *,
    delete_normalized_groups: bool = False,
) -> Dict[str, Any]:
    sources = sorted({str(value or "").strip() for value in source_paths if str(value or "").strip()})
    result: Dict[str, Any] = {
        "chunk_ids": [],
        "cache_paths": [],
        "stored_paths": [],
        "source_types": [],
    }
    if not sources:
        return result

    source_types: List[str] = []
    chunk_ids: List[int] = []
    if _table_exists(conn, "chunks"):
        chunk_where, chunk_params = _sql_source_match("source_path", sources)
        chunk_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        source_type_sql = "source_type" if "source_type" in chunk_columns else "'' AS source_type"
        rows = conn.execute(
            f"SELECT id, {source_type_sql} FROM chunks WHERE {chunk_where}",
            tuple(chunk_params),
        ).fetchall()
        chunk_ids = [int(row[0]) for row in rows if int(row[0] or 0) > 0]
        source_types = sorted({str(row[1] or "").strip().lower() for row in rows if str(row[1] or "").strip()})

        if delete_normalized_groups and "is_normalized" in chunk_columns:
            normalized_groups = {
                "txt" if value in {"txt", "pdf"} else value
                for value in source_types
                if value in {"txt", "pdf", "xlsx"}
            }
            if normalized_groups and "normalized_group" in chunk_columns:
                placeholders = ",".join("?" for _ in normalized_groups)
                normalized_rows = conn.execute(
                    f"""
                    SELECT id FROM chunks
                    WHERE COALESCE(is_normalized, 0) = 1
                      AND normalized_group IN ({placeholders})
                    """,
                    tuple(sorted(normalized_groups)),
                ).fetchall()
                chunk_ids.extend(int(row[0]) for row in normalized_rows if int(row[0] or 0) > 0)
    chunk_ids = sorted(set(chunk_ids))
    result["chunk_ids"] = chunk_ids
    result["source_types"] = source_types

    concept_ids: List[int] = []
    if chunk_ids and _table_exists(conn, "chunk_concept_edges"):
        placeholders = ",".join("?" for _ in chunk_ids)
        concept_ids = [
            int(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT concept_id FROM chunk_concept_edges WHERE chunk_pk IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
            if int(row[0] or 0) > 0
        ]
        _delete_rows_for_ids(conn, "chunk_concept_edges", "chunk_pk", chunk_ids)

    fact_ids: List[int] = []
    if _table_exists(conn, "ontology_fact_sources"):
        fact_conditions: List[str] = []
        fact_params: List[Any] = []
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            fact_conditions.append(f"chunk_id IN ({placeholders})")
            fact_params.extend(chunk_ids)
        fact_source_where, fact_source_params = _sql_source_match("source_path", sources)
        fact_conditions.append(f"({fact_source_where})")
        fact_params.extend(fact_source_params)
        where_sql = " OR ".join(fact_conditions)
        fact_ids = [
            int(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT fact_id FROM ontology_fact_sources WHERE {where_sql}",
                tuple(fact_params),
            ).fetchall()
            if int(row[0] or 0) > 0
        ]
        conn.execute(
            f"DELETE FROM ontology_fact_sources WHERE {where_sql}",
            tuple(fact_params),
        )

    wiki_page_ids: List[int] = []
    if _table_exists(conn, "wiki_page_sources"):
        wiki_conditions: List[str] = []
        wiki_params: List[Any] = []
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            wiki_conditions.append(f"chunk_id IN ({placeholders})")
            wiki_params.extend(chunk_ids)
        wiki_source_where, wiki_source_params = _sql_source_match("source_path", sources)
        wiki_conditions.append(f"({wiki_source_where})")
        wiki_params.extend(wiki_source_params)
        wiki_where = " OR ".join(wiki_conditions)
        wiki_page_ids = [
            int(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT page_id FROM wiki_page_sources WHERE {wiki_where}",
                tuple(wiki_params),
            ).fetchall()
            if int(row[0] or 0) > 0
        ]
        conn.execute(
            f"DELETE FROM wiki_page_sources WHERE {wiki_where}",
            tuple(wiki_params),
        )

    _delete_rows_for_ids(conn, "chunk_vec", "chunk_pk", chunk_ids)
    if chunk_ids and _table_exists(conn, "chunk_fts"):
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(
            f"DELETE FROM chunk_fts WHERE rowid IN ({placeholders})",
            tuple(chunk_ids),
        )
    _delete_rows_for_ids(conn, "chunks", "id", chunk_ids)

    if concept_ids and _table_exists(conn, "concept_nodes"):
        placeholders = ",".join("?" for _ in concept_ids)
        conn.execute(
            f"""
            DELETE FROM concept_nodes
            WHERE concept_id IN ({placeholders})
              AND NOT EXISTS (
                  SELECT 1 FROM chunk_concept_edges edge
                  WHERE edge.concept_id = concept_nodes.concept_id
              )
            """,
            tuple(concept_ids),
        )

    if fact_ids and _table_exists(conn, "ontology_facts"):
        placeholders = ",".join("?" for _ in fact_ids)
        orphan_fact_ids = [
            int(row[0])
            for row in conn.execute(
                f"""
                SELECT fact_id FROM ontology_facts
                WHERE fact_id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM ontology_fact_sources source
                      WHERE source.fact_id = ontology_facts.fact_id
                  )
                """,
                tuple(fact_ids),
            ).fetchall()
        ]
        _delete_rows_for_ids(conn, "ontology_fact_feedback", "fact_id", orphan_fact_ids)
        _delete_rows_for_ids(conn, "ontology_fact_history", "fact_id", orphan_fact_ids)
        _delete_rows_for_ids(conn, "ontology_facts", "fact_id", orphan_fact_ids)

    if wiki_page_ids and _table_exists(conn, "wiki_pages"):
        placeholders = ",".join("?" for _ in wiki_page_ids)
        orphan_source_page_ids = [
            int(row[0])
            for row in conn.execute(
                f"""
                SELECT page_id FROM wiki_pages
                WHERE page_id IN ({placeholders})
                  AND page_type = 'source'
                  AND NOT EXISTS (
                      SELECT 1 FROM wiki_page_sources source
                      WHERE source.page_id = wiki_pages.page_id
                  )
                """,
                tuple(wiki_page_ids),
            ).fetchall()
        ]
        _delete_rows_for_ids(conn, "wiki_links", "from_page_id", orphan_source_page_ids)
        _delete_rows_for_ids(conn, "wiki_claims", "page_id", orphan_source_page_ids)
        _delete_rows_for_ids(conn, "wiki_update_log", "page_id", orphan_source_page_ids)
        _delete_rows_for_ids(conn, "wiki_lint_findings", "page_id", orphan_source_page_ids)
        _delete_rows_for_ids(conn, "wiki_pages", "page_id", orphan_source_page_ids)

    if _table_exists(conn, "documents"):
        doc_where, doc_params = _sql_source_match("source_path", sources)
        doc_ids = [
            str(row[0])
            for row in conn.execute(
                f"SELECT doc_id FROM documents WHERE {doc_where}",
                tuple(doc_params),
            ).fetchall()
            if str(row[0] or "")
        ]
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            if _table_exists(conn, "doc_blocks"):
                conn.execute(f"DELETE FROM doc_blocks WHERE doc_id IN ({placeholders})", tuple(doc_ids))
            if _table_exists(conn, "doc_table_cells"):
                conn.execute(f"DELETE FROM doc_table_cells WHERE doc_id IN ({placeholders})", tuple(doc_ids))
        conn.execute(f"DELETE FROM documents WHERE {doc_where}", tuple(doc_params))

    if _table_exists(conn, "source_uploads"):
        upload_where, upload_params = _sql_source_match("source_path", sources)
        conn.execute(f"DELETE FROM source_uploads WHERE {upload_where}", tuple(upload_params))

    if _table_exists(conn, "file_cache"):
        cache_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(file_cache)").fetchall()}
        cache_path_columns = [
            value
            for value in (
                "items_cache_path",
                "embeddings_cache_path",
                "emb_small_cache_path",
                "emb_large_cache_path",
            )
            if value in cache_columns
        ]
        cache_where, cache_params = _sql_source_match("source_path", sources)
        if cache_path_columns:
            cache_rows = conn.execute(
                f"SELECT {', '.join(cache_path_columns)} FROM file_cache WHERE {cache_where}",
                tuple(cache_params),
            ).fetchall()
            result["cache_paths"] = sorted(
                {
                    str(value or "").strip()
                    for row in cache_rows
                    for value in row
                    if str(value or "").strip()
                }
            )
        conn.execute(f"DELETE FROM file_cache WHERE {cache_where}", tuple(cache_params))

    if _table_exists(conn, "files"):
        file_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(files)").fetchall()}
        file_where, file_params = _sql_source_match("source_path", sources)
        if "stored_path" in file_columns:
            result["stored_paths"] = sorted(
                {
                    str(row[0] or "").strip()
                    for row in conn.execute(
                        f"SELECT stored_path FROM files WHERE {file_where}",
                        tuple(file_params),
                    ).fetchall()
                    if str(row[0] or "").strip()
                }
            )
        conn.execute(f"DELETE FROM files WHERE {file_where}", tuple(file_params))
    return result


def _remove_managed_file(path: str, allowed_roots: List[str]) -> bool:
    candidate = str(path or "").strip()
    if not candidate:
        return False
    resolved = os.path.realpath(os.path.abspath(candidate))
    allowed = False
    for root in allowed_roots:
        resolved_root = os.path.realpath(os.path.abspath(root))
        try:
            if os.path.commonpath([resolved, resolved_root]) == resolved_root:
                allowed = True
                break
        except ValueError:
            continue
    if not allowed or not os.path.isfile(resolved):
        return False
    os.remove(resolved)
    return True


def delete_file_from_kb(kb_id: str, file_key: str, data_dir: str = DEFAULT_KB_DATA_DIR):
    """Delete a file from KB uploads using file_id or stored filename."""
    uploads_dir = os.path.join(data_dir, kb_id, "uploads")
    cache_dir = os.path.join(data_dir, kb_id, "cache")
    db_path = _kb_meta_db_path(kb_id, data_dir)
    stored_path = os.path.join(uploads_dir, file_key)
    source_path = ""

    if os.path.exists(db_path):
        conn = _connect_meta_db(db_path)
        try:
            if _table_exists(conn, "files"):
                row = conn.execute(
                    """
                    SELECT file_id, source_path, stored_path
                    FROM files
                    WHERE file_id = ? OR source_path = ?
                    """,
                    (file_key, file_key),
                ).fetchone()
                if row is None:
                    rows = conn.execute(
                        "SELECT file_id, source_path, stored_path FROM files"
                    ).fetchall()
                    for candidate in rows:
                        candidate_stored_path = str(candidate["stored_path"] or "")
                        if os.path.basename(candidate_stored_path) == file_key:
                            row = candidate
                            break
                if row is not None:
                    source_path = str(row["source_path"] or "")
                    candidate_stored_path = str(row["stored_path"] or "")
                    if candidate_stored_path:
                        stored_path = candidate_stored_path

            if source_path:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    cleanup = _delete_source_records(
                        conn,
                        [source_path],
                        delete_normalized_groups=True,
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                for candidate in [
                    *list(cleanup.get("stored_paths", []) or []),
                    stored_path,
                ]:
                    _remove_managed_file(candidate, [uploads_dir])
                for candidate in list(cleanup.get("cache_paths", []) or []):
                    _remove_managed_file(candidate, [cache_dir])
                return
        finally:
            conn.close()

    _remove_managed_file(stored_path, [uploads_dir])


class RAGEngine:
    def __init__(
        self,
        kb_id: str = "default",
        model_large_path: Optional[str] = None,
        data_dir: str = DEFAULT_KB_DATA_DIR,
    ):
        self.kb_id = kb_id
        data_dir_resolved = os.path.abspath(os.path.expanduser(data_dir))
        self.data_dir = os.path.join(data_dir_resolved, kb_id)
        self.index_dir = os.path.join(self.data_dir, "index")
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.db_path = os.path.join(self.data_dir, "meta.sqlite")
        self._engine_lock = threading.RLock()

        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.derived_txt_dir = os.path.join(self.data_dir, "derived_txt")
        os.makedirs(self.derived_txt_dir, exist_ok=True)

        # Chunking / search configuration
        self.txt_target_tokens = int(os.getenv("TXT_CHUNK_TARGET_TOKENS", "640"))
        self.txt_min_tokens = int(os.getenv("TXT_CHUNK_MIN_TOKENS", "420"))
        self.txt_max_tokens = int(os.getenv("TXT_CHUNK_MAX_TOKENS", "900"))
        self.txt_overlap_ratio = float(os.getenv("TXT_CHUNK_OVERLAP_RATIO", "0.25"))
        self.txt_split_enabled = _env_bool("TXT_SPLIT_ENABLED", True)
        self.txt_split_trigger_lines = max(40, int(os.getenv("TXT_SPLIT_TRIGGER_LINES", "120")))
        self.txt_split_target_tokens = int(os.getenv("TXT_SPLIT_TARGET_TOKENS", "2200"))
        self.txt_split_min_tokens = int(os.getenv("TXT_SPLIT_MIN_TOKENS", "1000"))
        self.txt_split_max_tokens = int(os.getenv("TXT_SPLIT_MAX_TOKENS", "2800"))
        self.pdf_target_tokens = int(os.getenv("PDF_CHUNK_TARGET_TOKENS", str(self.txt_target_tokens)))
        self.pdf_min_tokens = int(os.getenv("PDF_CHUNK_MIN_TOKENS", str(self.txt_min_tokens)))
        self.pdf_max_tokens = int(os.getenv("PDF_CHUNK_MAX_TOKENS", str(self.txt_max_tokens)))
        self.pdf_target_tokens_real = int(os.getenv("PDF_CHUNK_TARGET_TOKENS_REAL", "480"))
        self.pdf_max_tokens_real = int(os.getenv("PDF_CHUNK_MAX_TOKENS_REAL", "640"))
        self.txt_target_tokens_real = int(os.getenv("TXT_CHUNK_TARGET_TOKENS_REAL", str(self.txt_target_tokens)))
        self.txt_max_tokens_real = int(os.getenv("TXT_CHUNK_MAX_TOKENS_REAL", str(self.txt_max_tokens)))
        self.hwpx_target_tokens_real = int(os.getenv("HWPX_CHUNK_TARGET_TOKENS_REAL", "220"))
        self.hwpx_max_tokens_real = int(os.getenv("HWPX_CHUNK_MAX_TOKENS_REAL", "320"))
        self.embedding_compaction_enabled = _env_bool("EMBEDDING_COMPACTION_ENABLED", True)
        self.embedding_compaction_min_line_chars = max(1, int(os.getenv("EMBEDDING_COMPACTION_MIN_LINE_CHARS", "3")))
        self.embedding_compaction_max_hint_lines = max(0, int(os.getenv("EMBEDDING_COMPACTION_MAX_HINT_LINES", "2")))
        self.embedding_compaction_max_chunk_chars = max(400, int(os.getenv("EMBEDDING_COMPACTION_MAX_CHUNK_CHARS", "1800")))
        self.hwpx_extract_enabled = _env_bool("HWPX_EXTRACT_ENABLED", True)
        self.hwpx_include_tables = _env_bool("HWPX_INCLUDE_TABLES", True)
        self.hwpx_target_tokens = int(os.getenv("HWPX_CHUNK_TARGET_TOKENS", "220"))
        self.hwpx_min_tokens = int(os.getenv("HWPX_CHUNK_MIN_TOKENS", "80"))
        self.hwpx_max_tokens = int(os.getenv("HWPX_CHUNK_MAX_TOKENS", "320"))
        self.hwpx_overlap_ratio = float(os.getenv("HWPX_CHUNK_OVERLAP_RATIO", "0.12"))
        self.structure_rag_v2_enabled = _env_bool("STRUCTURE_RAG_V2_ENABLED", False)
        self.hwpx_structure_rag_v2_enabled = _env_bool("HWPX_STRUCTURE_RAG_V2_ENABLED", False)
        self.xlsx_structure_rag_v2_enabled = _env_bool("XLSX_STRUCTURE_RAG_V2_ENABLED", False)
        self.structure_rag_parent_result_limit = max(1, int(os.getenv("STRUCTURE_RAG_PARENT_RESULT_LIMIT", "1")))

        self.xlsx_group_min_rows = int(os.getenv("XLSX_CHUNK_MIN_ROWS", "1"))
        self.xlsx_group_max_rows = int(os.getenv("XLSX_CHUNK_MAX_ROWS", "1"))
        self.xlsx_overlap_rows = int(os.getenv("XLSX_CHUNK_OVERLAP_ROWS", "0"))
        self.xlsx_target_tokens = int(os.getenv("XLSX_CHUNK_TARGET_TOKENS", "1000"))
        self.xlsx_max_tokens = int(os.getenv("XLSX_CHUNK_MAX_TOKENS", "1200"))

        self.search_ef = int(os.getenv("RAG_SEARCH_EF", "640"))
        self.search_candidates = int(os.getenv("RAG_SEARCH_CANDIDATES", "120"))
        self.lexical_weight = min(1.0, max(0.0, float(os.getenv("RAG_LEXICAL_WEIGHT", "0.48"))))
        self.hybrid_fts_weight = min(1.0, max(0.0, float(os.getenv("RAG_HYBRID_FTS_WEIGHT", "0.18"))))
        self.sqlite_dense_enabled = _env_bool("RAG_SQLITE_DENSE_ENABLED", True)
        self.hnsw_enabled = _env_bool("RAG_HNSW_ENABLED", False)
        self.index_include_raw_with_normalized = _env_bool("RAG_INDEX_INCLUDE_RAW_WITH_NORMALIZED", True)
        self.normalized_score_penalty = max(0.0, float(os.getenv("RAG_NORMALIZED_SCORE_PENALTY", "0.04")))
        self.code_match_boost = max(0.0, float(os.getenv("RAG_CODE_MATCH_BOOST", "0.12")))
        self.code_hint_boost_ratio = min(
            1.0,
            max(0.0, float(os.getenv("RAG_CODE_HINT_BOOST_RATIO", "0.45"))),
        )
        self.exact_keyword_boost = max(0.0, float(os.getenv("RAG_EXACT_KEYWORD_BOOST", "0.07")))
        self.recency_boost = max(0.0, float(os.getenv("RAG_RECENCY_BOOST", "0.06")))
        self.recency_half_life_days = max(1.0, float(os.getenv("RAG_RECENCY_HALF_LIFE_DAYS", "45")))
        self.literal_match_boost = max(0.0, float(os.getenv("RAG_LITERAL_MATCH_BOOST", "0.08")))
        self.embed_batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", "16")))
        self.index_m = int(os.getenv("RAG_INDEX_M", "40"))
        self.index_ef_construction = int(os.getenv("RAG_INDEX_EF_CONSTRUCTION", "600"))
        self.index_max_elements = int(os.getenv("RAG_INDEX_MAX_ELEMENTS", "150000"))
        self.query_cache_limit = int(os.getenv("RAG_QUERY_CACHE_LIMIT", "300"))
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.wiki_memory_boost_targets: Dict[str, Dict[Any, int]] = {"chunks": {}, "table_cells": {}, "sources": {}}
        self.wiki_memory_boost_weight = max(0.0, float(os.getenv("RAG_WIKI_MEMORY_BOOST_WEIGHT", "0.025")))
        self._last_concept_search_meta: Dict[str, Any] = {}
        self.normalized_txt_source = "__normalized_txt__"
        self.normalized_xlsx_source = "__normalized_xlsx__"
        self.normalized_conflict_limit = max(2, int(os.getenv("RAG_NORMALIZED_CONFLICT_LIMIT", "3")))
        self.normalized_target_tokens = int(os.getenv("RAG_NORMALIZED_TARGET_TOKENS", "360"))
        self.normalized_max_tokens = int(os.getenv("RAG_NORMALIZED_MAX_TOKENS", "460"))
        self.concept_links_enabled = _env_bool("RAG_CONCEPT_LINKS_ENABLED", True)
        self.concept_max_terms_per_chunk = max(2, int(os.getenv("RAG_CONCEPT_MAX_TERMS_PER_CHUNK", "6")))
        self.concept_max_ngram = max(1, min(3, int(os.getenv("RAG_CONCEPT_MAX_NGRAM", "2"))))
        self.concept_similarity_threshold = min(
            0.995,
            max(0.50, float(os.getenv("RAG_CONCEPT_SIMILARITY_THRESHOLD", "0.84"))),
        )
        self.concept_query_limit = max(8, int(os.getenv("RAG_CONCEPT_QUERY_LIMIT", "24")))
        self.concept_chunk_expand_limit = max(8, int(os.getenv("RAG_CONCEPT_CHUNK_EXPAND_LIMIT", "64")))
        self.concept_score_weight = max(0.0, float(os.getenv("RAG_CONCEPT_SCORE_WEIGHT", "0.22")))
        self.ontology_rag_enabled = ONTOLOGY_RAG_ENABLED
        self.ontology_max_hops = ONTOLOGY_MAX_HOPS
        self.ontology_min_fact_confidence = ONTOLOGY_MIN_FACT_CONFIDENCE
        self.ontology_score_weight = max(0.0, float(os.getenv("ONTOLOGY_SCORE_WEIGHT", "0.26")))
        self._last_ontology_search_meta: Dict[str, Any] = {}

        self.xlsx_merged_cell_policy = os.getenv("XLSX_MERGED_CELL_POLICY", "expand")
        self.xlsx_comment_policy = os.getenv("XLSX_COMMENT_POLICY", "footnote")
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "").strip().rstrip("/")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        self.embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))
        self.embedding_api_batch_size = max(1, int(os.getenv("EMBEDDING_API_BATCH_SIZE", "16")))
        self.embedding_max_batch_tokens = max(256, int(os.getenv("EMBED_MAX_BATCH_TOKENS", "8192")))
        self.embedding_api_large_alias = os.getenv("EMBEDDING_API_LARGE_ALIAS", "large").strip() or "large"
        self.embedding_task_prefix_mode_raw = (
            os.getenv("EMBEDDING_TASK_PREFIX_MODE", "auto").strip().lower() or "auto"
        )
        self.embedding_qwen_query_instruction = (
            os.getenv("EMBEDDING_QWEN_QUERY_INSTRUCTION", DEFAULT_QWEN_QUERY_INSTRUCTION).strip()
            or DEFAULT_QWEN_QUERY_INSTRUCTION
        )
        self.embedding_model_ref = (model_large_path or os.getenv("EMBEDDING_MODEL_LARGE_PATH", "") or "").strip()
        self.sqlite_timeout_seconds = max(5.0, float(os.getenv("RAG_SQLITE_TIMEOUT_SECONDS", "60")))
        self.sqlite_busy_timeout_ms = max(
            5000,
            int(
                os.getenv(
                    "RAG_SQLITE_BUSY_TIMEOUT_MS",
                    str(int(self.sqlite_timeout_seconds * 1000)),
                )
            ),
        )
        self.sqlite_journal_mode = (os.getenv("RAG_SQLITE_JOURNAL_MODE", "WAL") or "WAL").strip().upper()
        self.log_retention_days = max(1, int(os.getenv("RAG_LOG_RETENTION_DAYS", "30")))
        self.log_retention_max_rows = max(100, int(os.getenv("RAG_LOG_RETENTION_MAX_ROWS", "50000")))
        self.log_prune_interval_seconds = max(
            60,
            int(os.getenv("RAG_LOG_PRUNE_INTERVAL_SECONDS", "3600")),
        )
        self.log_prune_batch_size = max(100, int(os.getenv("RAG_LOG_PRUNE_BATCH_SIZE", "2000")))
        self._last_log_prune_at = 0
        self.embedding_task_prefix_mode = self._resolve_task_prefix_mode(
            model_ref=self.embedding_model_ref,
            requested_mode=self.embedding_task_prefix_mode_raw,
        )

        self._init_db()
        self._prune_operational_logs_if_due(force=True)

        self.model_large: Optional[SentenceTransformer] = None

        if self._is_api_provider():
            if not self.embedding_api_url:
                raise RuntimeError("EMBEDDING_PROVIDER=api requires EMBEDDING_API_URL.")
            try:
                self.dim_large = self._infer_remote_embedding_dim(index_name="large")
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize remote embedding provider. "
                    f"url={self.embedding_api_url}. "
                    "Check embedding server status, API key, and model files."
                ) from e
            print(
                "RAGEngine: embedding provider=api "
                f"url={self.embedding_api_url} dim={self.dim_large}"
            )
        else:
            self.model_large = self._load_embedding_model(
                model_path=model_large_path,
                env_name="EMBEDDING_MODEL_LARGE_PATH",
                default_path=DEFAULT_EMBEDDING_MODEL_LARGE_PATH,
                hf_fallback="Qwen/Qwen3-Embedding-0.6B",
                tag="large",
            )
            if not self.embedding_model_ref:
                self.embedding_model_ref = "Qwen/Qwen3-Embedding-0.6B"
            self.embedding_task_prefix_mode = self._resolve_task_prefix_mode(
                model_ref=self.embedding_model_ref,
                requested_mode=self.embedding_task_prefix_mode_raw,
            )

            self.dim_large = self._infer_embedding_dim(self.model_large)
            print(f"RAGEngine: embedding dim={self.dim_large}")

        print(f"RAGEngine: embedding task_prefix_mode={self.embedding_task_prefix_mode}")

        self.index_large_path = os.path.join(self.index_dir, "index.large.bin")
        self.index_large: Optional[hnswlib.Index] = None

        chunk_count = self._count_indexable_chunks()
        self._ensure_sqlite_search_artifacts(chunk_count=chunk_count)
        if self.hnsw_enabled:
            self.index_large = hnswlib.Index(space="cosine", dim=self.dim_large)
            self._init_or_load_indexes(chunk_count=chunk_count)

    def set_wiki_memory_boost_targets(self, targets: Optional[Dict[str, Dict[Any, int]]]) -> None:
        safe_targets = targets if isinstance(targets, dict) else {}
        self.wiki_memory_boost_targets = {
            "chunks": dict(safe_targets.get("chunks", {}) or {}),
            "table_cells": dict(safe_targets.get("table_cells", {}) or {}),
            "sources": dict(safe_targets.get("sources", {}) or {}),
        }
        self.query_cache.clear()

    def _load_embedding_model(
        self,
        model_path: Optional[str],
        env_name: str,
        default_path: str,
        hf_fallback: str,
        tag: str,
    ) -> SentenceTransformer:
        requested = (model_path or os.getenv(env_name, default_path) or default_path).strip()

        if _looks_like_local_path(requested):
            local_path = os.path.expanduser(requested)
            if not os.path.isabs(local_path) and not (len(local_path) >= 2 and local_path[1] == ":"):
                local_path = os.path.abspath(os.path.join(PROJECT_ROOT, local_path))
            if os.path.isdir(local_path) and os.listdir(local_path):
                try:
                    print(f"RAGEngine: Loading {tag} embedding model from {os.path.abspath(local_path)}")
                    return SentenceTransformer(local_path, **_embedding_model_load_kwargs(local_path))
                except Exception as e:
                    print(
                        f"RAGEngine Warning: Failed local {tag} model '{local_path}': {e}"
                        f"{_embedding_model_requirements_hint(local_path)}"
                    )
            else:
                print(f"RAGEngine Warning: Local {tag} model path is missing/empty: {local_path}")
        else:
            try:
                print(f"RAGEngine: Attempting configured embedding model id for {tag}: {requested}")
                return SentenceTransformer(requested, **_embedding_model_load_kwargs(requested))
            except Exception as e:
                print(
                    f"RAGEngine Warning: Failed configured model id for {tag} '{requested}': {e}"
                    f"{_embedding_model_requirements_hint(requested)}"
                )

        try:
            print(f"RAGEngine: Attempting HuggingFace fallback for {tag}: {hf_fallback}")
            return SentenceTransformer(hf_fallback, **_embedding_model_load_kwargs(hf_fallback))
        except Exception as e:
            print(
                f"RAGEngine Warning: Failed HF fallback for {tag}: {e}"
                f"{_embedding_model_requirements_hint(hf_fallback)}"
            )

        raise RuntimeError(f"Could not load embedding model ({tag}).")

    def _infer_embedding_dim(self, model: SentenceTransformer) -> int:
        probe_text = self._apply_task_prefix(EMBEDDING_DIMENSION_PROBE_TEXT, task="passage")
        probe = self._encode_texts_local(model, [probe_text], task="passage")
        arr = np.asarray(probe)
        if arr.ndim == 1:
            return int(arr.shape[0])
        return int(arr.shape[-1])

    def _is_api_provider(self) -> bool:
        return self.embedding_provider in {"api", "remote"}

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.sqlite_timeout_seconds)
        try:
            conn.execute(f"PRAGMA journal_mode={self.sqlite_journal_mode}")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
        return conn

    def _resolve_task_prefix_mode(self, model_ref: str, requested_mode: str) -> str:
        mode = (requested_mode or "auto").strip().lower()
        if mode in {"none", "e5", "qwen", "jina_v5"}:
            return mode
        if mode != "auto":
            print(f"RAGEngine Warning: unsupported EMBEDDING_TASK_PREFIX_MODE='{requested_mode}', fallback to auto")
        lowered = (model_ref or "").lower()
        if "jina" in lowered and "embeddings-v5" in lowered:
            return "jina_v5"
        if "qwen3-embedding" in lowered or ("qwen" in lowered and "embedding" in lowered):
            return "qwen"
        if "e5" in lowered:
            return "e5"
        return "none"

    def _apply_task_prefix(self, text: str, task: str) -> str:
        value = (text or "").strip()
        if self.embedding_task_prefix_mode == "qwen" and task == "query":
            return f"Instruct: {self.embedding_qwen_query_instruction}\nQuery: {value}"
        if self.embedding_task_prefix_mode == "e5":
            return f"{task}: {value}"
        return value

    def _prompt_name_aliases_for_task(self, task: str) -> List[str]:
        if task == "query":
            return ["query", "question", "qry"]
        return ["document", "doc", "passage", "text"]

    def _extract_model_prompt_keys(self, model: SentenceTransformer) -> List[str]:
        prompts = getattr(model, "prompts", None)
        if isinstance(prompts, dict):
            return [str(key) for key in prompts.keys()]
        return []

    def _resolve_prompt_name_candidates(self, model: SentenceTransformer, task: str) -> List[str]:
        aliases = self._prompt_name_aliases_for_task(task)
        configured_keys = self._extract_model_prompt_keys(model)
        if not configured_keys:
            return aliases

        lowered = {str(key).strip().lower(): str(key) for key in configured_keys}
        candidates: List[str] = []
        for alias in aliases:
            actual = lowered.get(alias)
            if actual and actual not in candidates:
                candidates.append(actual)

        if candidates:
            return candidates
        return configured_keys if task == "query" else configured_keys[::-1]

    def _encode_attempt_options(self, model: SentenceTransformer, task: str) -> List[Dict[str, Any]]:
        mode = self.embedding_task_prefix_mode
        if mode == "jina_v5":
            prompt_candidates = self._resolve_prompt_name_candidates(model, task)
            attempts: List[Dict[str, Any]] = [
                {"task": "retrieval", "prompt_name": prompt_name}
                for prompt_name in prompt_candidates
            ]
            attempts.append({"task": "retrieval"})
            attempts.extend({"prompt_name": prompt_name} for prompt_name in prompt_candidates)
            attempts.append({})
            return attempts
        if mode == "qwen":
            return [{"prompt_name": prompt_name} for prompt_name in self._resolve_prompt_name_candidates(model, task)] + [{}]
        return [{}]

    def _normalize_index_name(self, index_name: str) -> str:
        raw = (index_name or "large").strip().lower()
        if raw in {"large", "b", "secondary", "index_b", "default"}:
            return "large"
        if raw in {"small", "a", "primary", "index_a"}:
            return "large"
        return "large"

    def _remote_index_alias(self, index_name: str) -> str:
        return self.embedding_api_large_alias

    def _api_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.embedding_api_key}"
        return headers

    def _infer_remote_embedding_dim(self, index_name: str) -> int:
        probe = self._encode_texts_api(
            texts=[EMBEDDING_DIMENSION_PROBE_TEXT],
            task="passage",
            index_name=index_name,
            expected_dim=None,
        )
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError(f"Unexpected embedding probe shape for {index_name}: {probe.shape}")
        return int(probe.shape[1])

    def _encode_texts_api(
        self,
        texts: List[str],
        task: str,
        index_name: str,
        expected_dim: Optional[int],
    ) -> np.ndarray:
        if not texts:
            dim = int(expected_dim or 0)
            return np.empty((0, dim), dtype=np.float32)

        if task not in {"query", "passage"}:
            raise ValueError(f"Unsupported embedding task: {task}")

        url = f"{self.embedding_api_url}/embed"
        merged: List[np.ndarray] = []
        alias = self._remote_index_alias(index_name)

        for start in range(0, len(texts), self.embedding_api_batch_size):
            batch = texts[start : start + self.embedding_api_batch_size]
            payload = {"texts": batch, "task": task, "index_name": alias}
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._api_headers(),
                    timeout=self.embedding_timeout,
                )
            except requests.RequestException as e:
                raise RuntimeError(f"Embedding API request failed: {e}") from e

            if resp.status_code == 401:
                raise RuntimeError("Embedding API unauthorized (check EMBEDDING_API_KEY).")
            if resp.status_code >= 400:
                body = resp.text[:300]
                raise RuntimeError(f"Embedding API error {resp.status_code}: {body}")

            data = resp.json()
            vectors = data.get("vectors")
            if not isinstance(vectors, list):
                raise RuntimeError("Embedding API response missing 'vectors' list.")

            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim != 2:
                raise RuntimeError(f"Embedding API returned invalid shape: {arr.shape}")
            if arr.shape[0] != len(batch):
                raise RuntimeError(
                    "Embedding API vector count mismatch: "
                    f"expected {len(batch)}, got {arr.shape[0]}"
                )
            if expected_dim is not None and arr.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch for index '{index_name}': "
                    f"expected {expected_dim}, got {arr.shape[1]}"
                )
            merged.append(arr)

        if not merged:
            dim = int(expected_dim or 0)
            return np.empty((0, dim), dtype=np.float32)

        return np.vstack(merged).astype(np.float32, copy=False)

    def _encode_texts_local(
        self,
        model: Optional[SentenceTransformer],
        texts: List[str],
        task: str,
    ) -> np.ndarray:
        if model is None:
            raise RuntimeError("Local embedding model is not initialized.")
        kwargs: Dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "batch_size": self.embed_batch_size,
            "show_progress_bar": False,
        }
        last_error: Optional[Exception] = None
        for extra_kwargs in self._encode_attempt_options(model, task):
            attempt_kwargs = dict(kwargs)
            attempt_kwargs.update(extra_kwargs)
            try:
                return model.encode(texts, **attempt_kwargs)
            except TypeError as e:
                last_error = e
                continue
            except ValueError as e:
                if extra_kwargs and ("Prompt name" in str(e) or "task" in str(e).lower()):
                    last_error = e
                    continue
                raise

        if last_error is not None:
            raise last_error
        return model.encode(texts, **kwargs)

    def _encode_texts(self, index_name: str, texts: List[str], task: str) -> np.ndarray:
        index_key = self._normalize_index_name(index_name)
        if task not in {"query", "passage"}:
            raise ValueError(f"Unsupported embedding task: {task}")

        if self._is_api_provider():
            expected_dim = self.dim_large
            return self._encode_texts_api(
                texts=texts,
                task=task,
                index_name=index_key,
                expected_dim=expected_dim,
            )

        model = self.model_large
        prefixed = [self._apply_task_prefix(t, task=task) for t in texts]
        return self._encode_texts_local(model, prefixed, task=task)

    def _model_tokenizer(self, model: Optional[SentenceTransformer]):
        if model is None:
            return None
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        first_module = None
        try:
            modules = getattr(model, "_modules", None)
            if modules:
                first_module = next(iter(modules.values()))
        except Exception:
            first_module = None
        tokenizer = getattr(first_module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        first_module_getter = getattr(model, "_first_module", None)
        if callable(first_module_getter):
            try:
                first_module = first_module_getter()
            except Exception:
                first_module = None
            tokenizer = getattr(first_module, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        return None

    def _tokenize_lengths_api(self, texts: List[str], task: str, index_name: str = "large") -> List[int]:
        url = f"{self.embedding_api_url}/tokenize_lengths"
        alias = self._remote_index_alias(index_name)
        try:
            resp = requests.post(
                url,
                json={"texts": texts, "task": task, "index_name": alias},
                headers=self._api_headers(),
                timeout=self.embedding_timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Embedding tokenize_lengths request failed: {e}") from e
        if resp.status_code == 401:
            raise RuntimeError("Embedding API unauthorized (check EMBEDDING_API_KEY).")
        if resp.status_code >= 400:
            raise RuntimeError(f"Embedding tokenize_lengths error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        lengths = data.get("lengths")
        if not isinstance(lengths, list):
            raise RuntimeError("Embedding tokenize_lengths response missing 'lengths'.")
        return [max(1, int(value or 0)) for value in lengths]

    def _tokenize_lengths_local(
        self,
        texts: List[str],
        task: str,
        index_name: str = "large",
    ) -> List[int]:
        index_key = self._normalize_index_name(index_name)
        if index_key != "large":
            raise ValueError(f"Unsupported embedding index: {index_name}")
        model = self.model_large
        tokenizer = self._model_tokenizer(model)
        if model is None or tokenizer is None:
            return [max(1, len(_OVERLAP_TERM_RE.findall(str(text or "")))) for text in texts]
        prefixed = [self._apply_task_prefix((text or ""), task=task) for text in texts]
        encoded = tokenizer(prefixed, add_special_tokens=True, truncation=False, padding=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if not isinstance(input_ids, list):
            return [max(1, len(_OVERLAP_TERM_RE.findall(str(text or "")))) for text in texts]
        return [max(1, int(len(ids or []))) for ids in input_ids]

    def _measure_text_token_lengths(
        self,
        texts: List[str],
        *,
        task: str,
        index_name: str = "large",
    ) -> List[int]:
        if not texts:
            return []
        try:
            if self._is_api_provider():
                return self._tokenize_lengths_api(texts, task=task, index_name=index_name)
            return self._tokenize_lengths_local(texts, task=task, index_name=index_name)
        except Exception:
            return [max(1, len(_OVERLAP_TERM_RE.findall(str(text or "")))) for text in texts]

    def _compact_pdf_embedding_payload(
        self,
        page_text: str,
        table_hints: List[str],
        lazy_ocr_hints: List[str],
    ) -> Tuple[List[str], Dict[str, int]]:
        min_line_chars = max(1, int(getattr(self, "embedding_compaction_min_line_chars", 3) or 3))
        max_hint_lines = max(0, int(getattr(self, "embedding_compaction_max_hint_lines", 2) or 2))
        max_chunk_chars = max(400, int(getattr(self, "embedding_compaction_max_chunk_chars", 1800) or 1800))
        raw_lines = [
            re.sub(r"\s+", " ", (line or "").strip())
            for line in str(page_text or "").splitlines()
        ]
        hint_lines = [
            re.sub(r"\s+", " ", (hint or "").strip())
            for hint in [*list(table_hints or []), *list(lazy_ocr_hints or [])]
        ]
        chars_before = sum(len(line) for line in raw_lines) + sum(len(line) for line in hint_lines)
        lines_before = len([line for line in raw_lines if line]) + len([line for line in hint_lines if line])
        seen: "OrderedDict[str, None]" = OrderedDict()
        compact_lines: List[str] = []
        max_line_chars = max(120, min(max_chunk_chars, 480))

        def _line_segments(value: str) -> List[str]:
            normalized = re.sub(r"\s+", " ", (value or "").strip())
            if len(normalized) <= max_line_chars:
                return [normalized]
            sentence_parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
            if not sentence_parts:
                sentence_parts = [normalized]
            chunks: List[str] = []
            current = ""
            for part in sentence_parts:
                candidate = part if not current else f"{current} {part}"
                if current and len(candidate) > max_line_chars:
                    chunks.append(current)
                    current = part
                else:
                    current = candidate
            if current:
                chunks.append(current)
            split_chunks: List[str] = []
            for chunk in chunks or [normalized]:
                if len(chunk) <= max_line_chars:
                    split_chunks.append(chunk)
                    continue
                for start in range(0, len(chunk), max_line_chars):
                    split_chunks.append(chunk[start : start + max_line_chars].strip())
            return [chunk for chunk in split_chunks if chunk]

        def _keep_line(value: str, *, is_hint: bool = False) -> None:
            normalized = re.sub(r"\s+", " ", (value or "").strip())
            if not normalized:
                return
            for segment in _line_segments(normalized):
                if len(segment) < min_line_chars:
                    continue
                lowered = segment.lower()
                if lowered in seen:
                    continue
                if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", lowered) or re.fullmatch(r"\d+\s*/\s*\d+", lowered):
                    continue
                seen[lowered] = None
                compact_lines.append(segment)

        for raw_line in raw_lines:
            _keep_line(raw_line, is_hint=False)
        kept_hints = 0
        for hint_line in hint_lines:
            if kept_hints >= max_hint_lines:
                break
            before_len = len(compact_lines)
            _keep_line(hint_line, is_hint=True)
            if len(compact_lines) > before_len:
                kept_hints += 1

        chars_after = sum(len(line) for line in compact_lines)
        return compact_lines, {
            "chars_before_compaction": chars_before,
            "chars_after_compaction": chars_after,
            "lines_before_dedupe": lines_before,
            "lines_after_dedupe": len(compact_lines),
            "hints_dropped": max(0, len([line for line in hint_lines if line]) - kept_hints),
        }

    def _split_lines_for_real_token_budget(
        self,
        lines: List[Dict[str, Any]],
        *,
        target_tokens: int,
        min_tokens: int,
        max_tokens: int,
        task: str = "passage",
    ) -> List[Dict[str, Any]]:
        if not lines:
            return []
        measured_lengths = self._measure_text_token_lengths(
            [str(item.get("text", "") or "") for item in lines],
            task=task,
            index_name="large",
        )
        return chunk_txt_items(
            lines,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_ratio=self.txt_overlap_ratio,
            measured_lengths=measured_lengths,
        )

    def _token_budget_batches(
        self,
        rows: List[sqlite3.Row],
    ) -> List[List[sqlite3.Row]]:
        if not rows:
            return []
        texts = [str(row["embedding_text"] or row["text"] or "") for row in rows]
        lengths = self._measure_text_token_lengths(texts, task="passage", index_name="large")
        max_rows = self.embedding_api_batch_size if self.embedding_provider == "api" else self.embed_batch_size
        batches: List[List[sqlite3.Row]] = []
        current_batch: List[sqlite3.Row] = []
        current_tokens = 0
        for row, token_len in zip(rows, lengths):
            bounded = max(1, int(token_len or 0))
            if current_batch and (len(current_batch) >= max_rows or current_tokens + bounded > self.embedding_max_batch_tokens):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(row)
            current_tokens += bounded
        if current_batch:
            batches.append(current_batch)
        return batches

    def _init_db(self):
        conn = self._connect_db()
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                kb_id TEXT,
                source_path TEXT,
                source_type TEXT,
                doc_role TEXT,
                sheet TEXT,
                row INTEGER,
                row_end INTEGER,
                line_start INTEGER,
                line_end INTEGER,
                section TEXT,
                doc_version TEXT,
                text TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS file_cache
            (
                source_path TEXT PRIMARY KEY,
                file_hash TEXT,
                parser_sig TEXT,
                items_cache_path TEXT,
                embeddings_cache_path TEXT,
                emb_small_cache_path TEXT,
                emb_large_cache_path TEXT,
                item_count INTEGER,
                updated_at INTEGER
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS source_uploads
            (
                source_path TEXT PRIMARY KEY,
                source_type TEXT,
                doc_role TEXT,
                file_hash TEXT,
                doc_version TEXT,
                uploaded_at INTEGER,
                original_filename TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS files
            (
                file_id TEXT PRIMARY KEY,
                source_path TEXT UNIQUE,
                sha256 TEXT,
                mime TEXT,
                orig_name TEXT,
                size INTEGER,
                stored_path TEXT,
                uploaded_at INTEGER,
                uploader_id TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents
            (
                doc_id TEXT PRIMARY KEY,
                file_id TEXT,
                source_path TEXT UNIQUE,
                doc_type TEXT,
                parser_name TEXT,
                parser_version TEXT,
                created_at INTEGER
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_blocks
            (
                block_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                page INTEGER,
                block_type TEXT,
                reading_order INTEGER,
                bbox_json TEXT,
                text TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_table_cells
            (
                cell_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                page INTEGER,
                table_id TEXT,
                r INTEGER,
                c INTEGER,
                bbox_json TEXT,
                text TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_vec
            (
                chunk_pk INTEGER,
                index_name TEXT,
                dim INTEGER,
                embedding BLOB,
                updated_at INTEGER,
                PRIMARY KEY (chunk_pk, index_name)
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_logs
            (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                user_id TEXT,
                query_text TEXT,
                topk_ids_json TEXT,
                meta_json TEXT,
                created_at INTEGER
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_nodes
            (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_key TEXT UNIQUE,
                display_text TEXT,
                dim INTEGER,
                embedding BLOB,
                updated_at INTEGER
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_concept_edges
            (
                chunk_pk INTEGER,
                concept_id INTEGER,
                weight REAL,
                updated_at INTEGER,
                PRIMARY KEY (chunk_pk, concept_id)
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_logs
            (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                llm_model TEXT,
                prompt_hash TEXT,
                answer_text TEXT,
                citations_json TEXT,
                answer_meta_json TEXT,
                created_at INTEGER
            )
            """
        )

        self._ensure_column(conn, "chunks", "chunk_id", "TEXT")
        self._ensure_column(conn, "chunks", "source_type", "TEXT")
        self._ensure_column(conn, "chunks", "doc_role", "TEXT")
        self._ensure_column(conn, "chunks", "row_end", "INTEGER")
        self._ensure_column(conn, "chunks", "page_no", "INTEGER")
        self._ensure_column(conn, "chunks", "section", "TEXT")
        self._ensure_column(conn, "chunks", "doc_version", "TEXT")
        self._ensure_column(conn, "chunks", "is_normalized", "INTEGER")
        self._ensure_column(conn, "chunks", "normalized_group", "TEXT")
        self._ensure_column(conn, "chunks", "source_updated_at", "INTEGER")
        self._ensure_column(conn, "chunks", "embedding_text", "TEXT")
        self._ensure_column(conn, "chunks", "chunk_kind", "TEXT")
        self._ensure_column(conn, "chunks", "heading_path_json", "TEXT")
        self._ensure_column(conn, "chunks", "parent_chunk_key", "TEXT")
        self._ensure_column(conn, "chunks", "structure_path", "TEXT")
        self._ensure_column(conn, "chunks", "table_id", "TEXT")
        self._ensure_column(conn, "chunks", "row_no", "INTEGER")
        self._ensure_column(conn, "chunks", "cell_no", "INTEGER")
        self._ensure_column(conn, "chunks", "is_derived", "INTEGER DEFAULT 0")

        self._ensure_column(conn, "file_cache", "embeddings_cache_path", "TEXT")
        self._ensure_column(conn, "file_cache", "emb_small_cache_path", "TEXT")
        self._ensure_column(conn, "file_cache", "emb_large_cache_path", "TEXT")
        self._ensure_column(conn, "source_uploads", "doc_role", "TEXT")
        self._ensure_column(conn, "source_uploads", "original_filename", "TEXT")
        self._ensure_column(conn, "documents", "source_path", "TEXT")
        self._ensure_column(conn, "retrieval_logs", "meta_json", "TEXT")
        self._ensure_column(conn, "answer_logs", "answer_meta_json", "TEXT")

        # Create indexes after schema migration so legacy DBs do not fail on missing columns.
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_normalized ON chunks(is_normalized, source_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_role ON chunks(doc_role)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_updated_at ON chunks(source_updated_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_source_uploads_type_time ON source_uploads(source_type, uploaded_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_source_uploads_role_time ON source_uploads(doc_role, uploaded_at)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_files_source_path ON files(source_path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_documents_file ON documents(file_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_doc_blocks_doc_order ON doc_blocks(doc_id, page, reading_order)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_doc_cells_doc_order ON doc_table_cells(doc_id, page, table_id, r, c)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunk_vec_index ON chunk_vec(index_name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_concept_nodes_key ON concept_nodes(normalized_key)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunk_concept_edges_concept ON chunk_concept_edges(concept_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_logs_query ON retrieval_logs(query_id, created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_answer_logs_query ON answer_logs(query_id, created_at)")

        self.fts_available = True
        try:
            c.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                USING fts5(text, tokenize='unicode61')
                """
            )
        except sqlite3.OperationalError:
            # FTS5 can be disabled depending on SQLite build.
            self.fts_available = False

        conn.commit()
        conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str):
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in c.fetchall()}
        if column not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _safe_json_dump(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return "{}"

    def _count_rows(self, table_name: str) -> int:
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = int(c.fetchone()[0])
        conn.close()
        return count

    def _count_chunk_vectors(self, index_name: str = "large") -> int:
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunk_vec WHERE index_name = ?", (index_name,))
        count = int(c.fetchone()[0])
        conn.close()
        return count

    def _count_fts_rows(self) -> int:
        if not getattr(self, "fts_available", False):
            return 0
        conn = self._connect_db()
        c = conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM chunk_fts")
            count = int(c.fetchone()[0])
        except sqlite3.OperationalError:
            count = 0
        conn.close()
        return count

    def _use_normalized_only_for_index(self) -> bool:
        return (not self.index_include_raw_with_normalized) and self._has_normalized_chunks()

    def _load_indexable_rows(self) -> List[sqlite3.Row]:
        use_normalized = self._use_normalized_only_for_index()
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if self.index_include_raw_with_normalized:
            c.execute("SELECT id, text, embedding_text FROM chunks ORDER BY id ASC")
        elif use_normalized:
            c.execute(
                """
                SELECT id, text, embedding_text
                FROM chunks
                WHERE COALESCE(is_normalized, 0) = 1
                ORDER BY id ASC
                """
            )
        else:
            c.execute("SELECT id, text, embedding_text FROM chunks ORDER BY id ASC")
        rows = c.fetchall()
        conn.close()
        return rows

    def _load_indexable_rows_by_ids(self, chunk_ids: List[int]) -> List[sqlite3.Row]:
        if not chunk_ids:
            return []
        use_normalized = self._use_normalized_only_for_index()
        placeholders = ",".join("?" for _ in chunk_ids)
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if self.index_include_raw_with_normalized:
            c.execute(
                f"SELECT id, text, embedding_text FROM chunks WHERE id IN ({placeholders}) ORDER BY id ASC",
                tuple(int(chunk_id) for chunk_id in chunk_ids),
            )
        elif use_normalized:
            c.execute(
                f"""
                SELECT id, text, embedding_text
                FROM chunks
                WHERE COALESCE(is_normalized, 0) = 1
                  AND id IN ({placeholders})
                ORDER BY id ASC
                """,
                tuple(int(chunk_id) for chunk_id in chunk_ids),
            )
        else:
            c.execute(
                f"SELECT id, text, embedding_text FROM chunks WHERE id IN ({placeholders}) ORDER BY id ASC",
                tuple(int(chunk_id) for chunk_id in chunk_ids),
            )
        rows = c.fetchall()
        conn.close()
        return rows

    def _delete_search_artifacts_for_chunk_ids(self, chunk_ids: List[int], index_name: str = "large"):
        if not chunk_ids:
            return
        ordered_ids = [int(chunk_id) for chunk_id in chunk_ids if int(chunk_id) > 0]
        if not ordered_ids:
            return
        placeholders = ",".join("?" for _ in ordered_ids)
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            f"DELETE FROM chunk_vec WHERE index_name = ? AND chunk_pk IN ({placeholders})",
            (index_name, *ordered_ids),
        )
        if getattr(self, "fts_available", False):
            try:
                c.execute(
                    f"DELETE FROM chunk_fts WHERE rowid IN ({placeholders})",
                    tuple(ordered_ids),
                )
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()

    def _upsert_chunk_vectors_for_rows(
        self,
        rows: List[sqlite3.Row],
        index_name: str = "large",
        *,
        progress_callback: Optional[Callable[..., None]] = None,
        log_context: Optional[Dict[str, Any]] = None,
    ):
        if (not self.sqlite_dense_enabled) or (not rows):
            return
        total_rows = len(rows)
        texts_all = [str(row["embedding_text"] or row["text"] or "") for row in rows]
        measured_lengths = self._measure_text_token_lengths(texts_all, task="passage", index_name=index_name)
        max_rows = self.embedding_api_batch_size if self.embedding_provider == "api" else self.embed_batch_size
        batches: List[Tuple[List[sqlite3.Row], List[int]]] = []
        current_rows: List[sqlite3.Row] = []
        current_lengths: List[int] = []
        current_tokens = 0
        for row, token_len in zip(rows, measured_lengths):
            bounded_len = max(1, int(token_len or 0))
            if current_rows and (len(current_rows) >= max_rows or current_tokens + bounded_len > self.embedding_max_batch_tokens):
                batches.append((current_rows, current_lengths))
                current_rows = []
                current_lengths = []
                current_tokens = 0
            current_rows.append(row)
            current_lengths.append(bounded_len)
            current_tokens += bounded_len
        if current_rows:
            batches.append((current_rows, current_lengths))
        total_batches = max(1, len(batches))
        source_path = str((log_context or {}).get("source_path", "") or "")
        job_id = str((log_context or {}).get("job_id", "") or "")
        rows_done = 0
        tokens_done = 0
        total_input_tokens = int(sum(measured_lengths))
        measured_lengths_sorted = sorted(int(value) for value in measured_lengths)
        p95_index = max(0, min(len(measured_lengths_sorted) - 1, math.ceil(len(measured_lengths_sorted) * 0.95) - 1))
        conn = self._connect_db()
        c = conn.cursor()
        try:
            for batch_index, (batch_rows, batch_lengths) in enumerate(batches, start=1):
                texts = [row["embedding_text"] or row["text"] or "" for row in batch_rows]
                batch_tokens = int(sum(batch_lengths))
                batch_started = time.perf_counter()
                if callable(progress_callback):
                    try:
                        progress_callback(
                            batch_index,
                            total_batches,
                            rows_done,
                            embed_input_tokens_total=total_input_tokens,
                            embed_input_tokens_done=tokens_done,
                            embed_input_tokens_p95=int(measured_lengths_sorted[p95_index] if measured_lengths_sorted else 0),
                            embed_input_tokens_max=int(max(measured_lengths) if measured_lengths else 0),
                            embed_truncated_rows=0,
                            embed_effective_batch_tokens=batch_tokens,
                        )
                    except TypeError:
                        progress_callback(batch_index, total_batches, rows_done)
                print(
                    "[UPLOAD][PHASE] stage=embed_chunks status=batch_start "
                    f"job_id={job_id or '-'} source_path={source_path or '-'} "
                    f"batch={batch_index}/{total_batches} rows={len(batch_rows)} rows_done={rows_done}/{total_rows} "
                    f"batch_input_tokens={batch_tokens} batch_max_tokens={self.embedding_max_batch_tokens}",
                    flush=True,
                )
                embeddings = self._encode_texts(index_name=index_name, texts=texts, task="passage")
                now_ts = int(time.time())
                payload = []
                for row, vec in zip(batch_rows, embeddings):
                    arr = np.asarray(vec, dtype=np.float32)
                    payload.append(
                        (
                            int(row["id"]),
                            index_name,
                            int(arr.shape[0]),
                            sqlite3.Binary(arr.tobytes()),
                            now_ts,
                        )
                    )
                c.executemany(
                    """
                    INSERT INTO chunk_vec (chunk_pk, index_name, dim, embedding, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_pk, index_name) DO UPDATE SET
                        dim = excluded.dim,
                        embedding = excluded.embedding,
                        updated_at = excluded.updated_at
                    """,
                    payload,
                )
                conn.commit()
                rows_done += len(batch_rows)
                tokens_done += batch_tokens
                batch_seconds = max(0.0, time.perf_counter() - batch_started)
                print(
                    "[UPLOAD][PHASE] stage=embed_chunks status=batch_done "
                    f"job_id={job_id or '-'} source_path={source_path or '-'} "
                    f"batch={batch_index}/{total_batches} embed_batch_rows={len(batch_rows)} "
                    f"rows_done={rows_done}/{total_rows} embed_batch_seconds={batch_seconds:.3f} "
                    f"embed_batch_tokens={batch_tokens} embed_tokens_done={tokens_done}/{total_input_tokens}",
                    flush=True,
                )
                if callable(progress_callback):
                    try:
                        progress_callback(
                            batch_index,
                            total_batches,
                            rows_done,
                            embed_input_tokens_total=total_input_tokens,
                            embed_input_tokens_done=tokens_done,
                            embed_input_tokens_p95=int(measured_lengths_sorted[p95_index] if measured_lengths_sorted else 0),
                            embed_input_tokens_max=int(max(measured_lengths) if measured_lengths else 0),
                            embed_truncated_rows=0,
                            embed_effective_batch_tokens=batch_tokens,
                        )
                    except TypeError:
                        progress_callback(batch_index, total_batches, rows_done)
        finally:
            conn.close()

    def _upsert_fts_rows_for_rows(self, rows: List[sqlite3.Row]):
        if (not getattr(self, "fts_available", False)) or (not rows):
            return
        conn = self._connect_db()
        c = conn.cursor()
        try:
            c.executemany(
                "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                [
                    (
                        int(row["id"]),
                        "\n".join(
                            part
                            for part in (row["text"] or "", row["embedding_text"] or "")
                            if part
                        ),
                    )
                    for row in rows
                ],
            )
            conn.commit()
        except sqlite3.IntegrityError:
            for row in rows:
                c.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["id"]),))
                c.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                    (
                        int(row["id"]),
                        "\n".join(
                            part
                            for part in (row["text"] or "", row["embedding_text"] or "")
                            if part
                        ),
                    ),
                )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def _sync_sqlite_search_artifacts(
        self,
        *,
        changed_chunk_ids: Optional[List[int]] = None,
        deleted_chunk_ids: Optional[List[int]] = None,
        index_name: str = "large",
        vector_progress_callback: Optional[Callable[[int, int, int], None]] = None,
        vector_log_context: Optional[Dict[str, Any]] = None,
    ):
        deleted_ids = [int(chunk_id) for chunk_id in (deleted_chunk_ids or []) if int(chunk_id) > 0]
        changed_ids = [int(chunk_id) for chunk_id in (changed_chunk_ids or []) if int(chunk_id) > 0]
        if deleted_ids:
            self._delete_search_artifacts_for_chunk_ids(deleted_ids, index_name=index_name)
        if not changed_ids:
            return
        rows = self._load_indexable_rows_by_ids(changed_ids)
        indexed_ids = {int(row["id"]) for row in rows}
        skipped_ids = [chunk_id for chunk_id in changed_ids if chunk_id not in indexed_ids]
        if skipped_ids:
            self._delete_search_artifacts_for_chunk_ids(skipped_ids, index_name=index_name)
        if rows:
            self._upsert_chunk_vectors_for_rows(
                rows,
                index_name=index_name,
                progress_callback=vector_progress_callback,
                log_context=vector_log_context,
            )
            self._upsert_fts_rows_for_rows(rows)

    def _normalize_concept_key(self, text: str) -> str:
        value = re.sub(r"\s+", " ", (text or "").strip().lower())
        return value

    def _concept_stopwords(self) -> set[str]:
        return {
            "guide",
            "manual",
            "summary",
            "question",
            "answer",
            "document",
            "file",
            "upload",
            "content",
            "info",
        }

    def _extract_concept_terms(self, text: str) -> List[str]:
        max_terms_per_chunk = max(2, int(getattr(self, "concept_max_terms_per_chunk", 6) or 6))
        tokens = self._tokenize_for_overlap(text)
        if not tokens:
            return []
        stopwords = self._concept_stopwords()
        filtered: List[str] = []
        for token in tokens:
            normalized = self._normalize_concept_key(token)
            if len(normalized) < 2:
                continue
            if normalized in stopwords:
                continue
            filtered.append(normalized)
        if not filtered:
            return []

        unigram_scores: Dict[str, float] = {}
        ngram_scores: Dict[str, float] = {}
        for idx, token in enumerate(filtered):
            base_score = 1.0 + (0.35 if idx < 8 else 0.0) + (0.15 if len(token) >= 4 else 0.0)
            unigram_scores[token] = unigram_scores.get(token, 0.0) + base_score

        max_ngram = max(1, int(getattr(self, "concept_max_ngram", 2) or 2))
        for ngram_size in range(2, max_ngram + 1):
            for start in range(0, len(filtered) - ngram_size + 1):
                phrase_tokens = filtered[start : start + ngram_size]
                phrase = " ".join(phrase_tokens).strip()
                if len(phrase) < 4:
                    continue
                phrase_score = 1.25 + (0.25 if start < 6 else 0.0) + (0.10 * (ngram_size - 1))
                ngram_scores[phrase] = ngram_scores.get(phrase, 0.0) + phrase_score

        ordered_unigrams = sorted(
            unigram_scores.items(),
            key=lambda item: (float(item[1]), len(item[0]), item[0]),
            reverse=True,
        )
        ordered_ngrams = sorted(
            ngram_scores.items(),
            key=lambda item: (float(item[1]), len(item[0]), item[0]),
            reverse=True,
        )
        out: List[str] = []
        seen = set()
        leading_terms = []
        for token in filtered[: min(3, len(filtered))]:
            if token not in leading_terms:
                leading_terms.append(token)
        for term in leading_terms:
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
            if len(out) >= max_terms_per_chunk:
                return out
        preferred_unigrams = max(1, min(max_terms_per_chunk, max(2, max_terms_per_chunk // 2)))
        for term, _score in ordered_unigrams:
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
            if len(out) >= preferred_unigrams:
                break
        for term, _score in ordered_ngrams + ordered_unigrams:
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
            if len(out) >= max_terms_per_chunk:
                break
        return out

    def _load_raw_chunk_rows_by_ids(self, chunk_ids: List[int]) -> List[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            f"""
            SELECT id, text
            FROM chunks
            WHERE COALESCE(is_normalized, 0) = 0
              AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            tuple(int(chunk_id) for chunk_id in chunk_ids),
        )
        rows = c.fetchall()
        conn.close()
        return rows

    def _load_all_concept_nodes(self) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT concept_id, normalized_key, display_text, dim, embedding
            FROM concept_nodes
            ORDER BY concept_id ASC
            """
        )
        rows: List[Dict[str, Any]] = []
        for row in c.fetchall():
            blob = row["embedding"]
            vector = np.frombuffer(blob, dtype=np.float32) if blob is not None else np.empty((0,), dtype=np.float32)
            rows.append(
                {
                    "concept_id": int(row["concept_id"]),
                    "normalized_key": str(row["normalized_key"] or ""),
                    "display_text": str(row["display_text"] or ""),
                    "dim": int(row["dim"] or 0),
                    "vector": vector,
                }
            )
        conn.close()
        return rows

    def _resolve_concept_records_for_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        if not terms:
            return {}
        similarity_threshold = float(getattr(self, "concept_similarity_threshold", 0.84) or 0.84)
        normalized_terms = [self._normalize_concept_key(term) for term in terms if self._normalize_concept_key(term)]
        if not normalized_terms:
            return {}

        existing_nodes = self._load_all_concept_nodes()
        exact_map = {row["normalized_key"]: row for row in existing_nodes if row["normalized_key"]}
        out: Dict[str, Dict[str, Any]] = {}
        pending_terms: List[str] = []
        for term in normalized_terms:
            exact = exact_map.get(term)
            if exact is not None:
                out[term] = {
                    "concept_id": int(exact["concept_id"]),
                    "display_text": str(exact["display_text"] or term),
                    "score": 1.0,
                }
            else:
                pending_terms.append(term)
        if not pending_terms:
            return out

        encoded = self._encode_texts(index_name="large", texts=pending_terms, task="passage")
        conn = self._connect_db()
        c = conn.cursor()
        now_ts = int(time.time())
        try:
            for term, vec in zip(pending_terms, encoded):
                arr = np.asarray(vec, dtype=np.float32)
                best_match: Optional[Dict[str, Any]] = None
                best_score = 0.0
                for node in existing_nodes:
                    node_vec = np.asarray(node.get("vector", np.empty((0,), dtype=np.float32)), dtype=np.float32)
                    if node_vec.shape[0] != arr.shape[0] or node_vec.shape[0] <= 0:
                        continue
                    score = float(np.dot(arr, node_vec))
                    if score > best_score:
                        best_score = score
                        best_match = node
                if best_match is not None and best_score >= similarity_threshold:
                    out[term] = {
                        "concept_id": int(best_match["concept_id"]),
                        "display_text": str(best_match["display_text"] or term),
                        "score": best_score,
                    }
                    continue

                c.execute(
                    """
                    INSERT INTO concept_nodes (normalized_key, display_text, dim, embedding, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        term,
                        term,
                        int(arr.shape[0]),
                        sqlite3.Binary(arr.tobytes()),
                        now_ts,
                    ),
                )
                concept_id = int(c.lastrowid)
                node_payload = {
                    "concept_id": concept_id,
                    "normalized_key": term,
                    "display_text": term,
                    "vector": arr,
                }
                existing_nodes.append(node_payload)
                exact_map[term] = node_payload
                out[term] = {
                    "concept_id": concept_id,
                    "display_text": term,
                    "score": 1.0,
                }
            conn.commit()
        finally:
            conn.close()
        return out

    def _prune_orphan_concepts(self):
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            DELETE FROM concept_nodes
            WHERE concept_id NOT IN (
                SELECT DISTINCT concept_id
                FROM chunk_concept_edges
            )
            """
        )
        conn.commit()
        conn.close()

    def _sync_concept_links(
        self,
        *,
        changed_chunk_ids: Optional[List[int]] = None,
        deleted_chunk_ids: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        if not bool(getattr(self, "concept_links_enabled", True)):
            return {
                "concept_nodes_added": 0,
                "chunk_concept_edges_added": 0,
            }
        changed_ids = [int(chunk_id) for chunk_id in (changed_chunk_ids or []) if int(chunk_id) > 0]
        deleted_ids = [int(chunk_id) for chunk_id in (deleted_chunk_ids or []) if int(chunk_id) > 0]
        concept_nodes_before = self._count_rows("concept_nodes")
        edge_count_before = self._count_rows("chunk_concept_edges")
        if deleted_ids:
            conn = self._connect_db()
            c = conn.cursor()
            placeholders = ",".join("?" for _ in deleted_ids)
            c.execute(
                f"DELETE FROM chunk_concept_edges WHERE chunk_pk IN ({placeholders})",
                tuple(deleted_ids),
            )
            conn.commit()
            conn.close()

        if changed_ids:
            conn = self._connect_db()
            c = conn.cursor()
            placeholders = ",".join("?" for _ in changed_ids)
            c.execute(
                f"DELETE FROM chunk_concept_edges WHERE chunk_pk IN ({placeholders})",
                tuple(changed_ids),
            )
            conn.commit()
            conn.close()

            rows = self._load_raw_chunk_rows_by_ids(changed_ids)
            term_map: Dict[int, List[str]] = {}
            all_terms: List[str] = []
            for row in rows:
                chunk_id = int(row["id"])
                terms = self._extract_concept_terms(str(row["text"] or ""))
                if not terms:
                    continue
                term_map[chunk_id] = terms
                all_terms.extend(terms)

            resolved_terms = self._resolve_concept_records_for_terms(all_terms)
            payload: List[Tuple[int, int, float, int]] = []
            now_ts = int(time.time())
            for chunk_id, terms in term_map.items():
                for rank, term in enumerate(terms, start=1):
                    concept = resolved_terms.get(term)
                    if concept is None:
                        continue
                    weight = max(0.15, 1.0 - ((rank - 1) * 0.12))
                    payload.append(
                        (
                            int(chunk_id),
                            int(concept["concept_id"]),
                            float(weight),
                            now_ts,
                        )
                    )
            if payload:
                conn = self._connect_db()
                c = conn.cursor()
                c.executemany(
                    """
                    INSERT INTO chunk_concept_edges (chunk_pk, concept_id, weight, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chunk_pk, concept_id) DO UPDATE SET
                        weight = excluded.weight,
                        updated_at = excluded.updated_at
                    """,
                    payload,
                )
                conn.commit()
                conn.close()

        self._prune_orphan_concepts()
        concept_nodes_after = self._count_rows("concept_nodes")
        edge_count = self._count_rows("chunk_concept_edges")
        return {
            "concept_nodes_added": max(0, concept_nodes_after - concept_nodes_before),
            "chunk_concept_edges_added": max(0, edge_count - edge_count_before),
        }

    def _search_concept_candidates(self, query: str, candidate_limit: int) -> Dict[int, float]:
        self._last_concept_search_meta = {
            "query_terms": [],
            "matched_concepts": [],
            "chunk_labels": {},
        }
        if not bool(getattr(self, "concept_links_enabled", True)):
            return {}
        concept_max_terms = max(2, int(getattr(self, "concept_max_terms_per_chunk", 6) or 6))
        concept_query_limit = max(8, int(getattr(self, "concept_query_limit", 24) or 24))
        similarity_threshold = float(getattr(self, "concept_similarity_threshold", 0.84) or 0.84)

        query_terms = self._extract_concept_terms(query)
        if not query_terms:
            query_terms = self._query_keywords(query)[:concept_max_terms]
        query_terms = [self._normalize_concept_key(term) for term in query_terms if self._normalize_concept_key(term)]
        if not query_terms:
            return {}

        concept_rows = self._load_all_concept_nodes()
        if not concept_rows:
            return {}
        exact_map = {row["normalized_key"]: row for row in concept_rows if row["normalized_key"]}
        matched_scores: Dict[int, float] = {}
        matched_labels: Dict[int, str] = {}
        pending_terms: List[str] = []
        for term in query_terms:
            exact = exact_map.get(term)
            if exact is not None:
                concept_id = int(exact["concept_id"])
                matched_scores[concept_id] = max(matched_scores.get(concept_id, 0.0), 1.0)
                matched_labels[concept_id] = str(exact["display_text"] or term)
            else:
                pending_terms.append(term)

        if pending_terms:
            encoded = self._encode_texts(index_name="large", texts=pending_terms, task="query")
            for term, vec in zip(pending_terms, encoded):
                arr = np.asarray(vec, dtype=np.float32)
                best_row: Optional[Dict[str, Any]] = None
                best_score = 0.0
                for row in concept_rows:
                    row_vec = np.asarray(row.get("vector", np.empty((0,), dtype=np.float32)), dtype=np.float32)
                    if row_vec.shape[0] != arr.shape[0] or row_vec.shape[0] <= 0:
                        continue
                    score = float(np.dot(arr, row_vec))
                    if score > best_score:
                        best_score = score
                        best_row = row
                if best_row is not None and best_score >= similarity_threshold:
                    concept_id = int(best_row["concept_id"])
                    matched_scores[concept_id] = max(matched_scores.get(concept_id, 0.0), best_score)
                    matched_labels[concept_id] = str(best_row["display_text"] or term)

        if not matched_scores:
            return {}

        ordered_concepts = sorted(
            matched_scores.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:concept_query_limit]
        concept_ids = [int(concept_id) for concept_id, _score in ordered_concepts]
        placeholders = ",".join("?" for _ in concept_ids)
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            f"""
            SELECT chunk_pk, concept_id, weight
            FROM chunk_concept_edges
            WHERE concept_id IN ({placeholders})
            """,
            tuple(concept_ids),
        )
        chunk_scores: Dict[int, float] = {}
        chunk_labels: Dict[int, List[str]] = {}
        for row in c.fetchall():
            chunk_pk = int(row["chunk_pk"])
            concept_id = int(row["concept_id"])
            score = float(matched_scores.get(concept_id, 0.0)) * float(row["weight"] or 0.0)
            if score > chunk_scores.get(chunk_pk, 0.0):
                chunk_scores[chunk_pk] = score
            chunk_labels.setdefault(chunk_pk, [])
            label = matched_labels.get(concept_id, "")
            if label and label not in chunk_labels[chunk_pk]:
                chunk_labels[chunk_pk].append(label)
        conn.close()

        ranked = sorted(chunk_scores.items(), key=lambda item: float(item[1]), reverse=True)[: max(8, int(candidate_limit))]
        self._last_concept_search_meta = {
            "query_terms": query_terms[:concept_max_terms],
            "matched_concepts": [matched_labels.get(concept_id, str(concept_id)) for concept_id in concept_ids],
            "chunk_labels": {chunk_id: chunk_labels.get(chunk_id, [])[:4] for chunk_id, _score in ranked},
        }
        return {int(chunk_id): float(score) for chunk_id, score in ranked}

    def _sync_ontology_facts(
        self,
        *,
        changed_chunk_ids: Optional[List[int]] = None,
        deleted_chunk_ids: Optional[List[int]] = None,
        include_llm: bool = False,
        llm_fact_status: str = "",
    ) -> Dict[str, int]:
        if not bool(getattr(self, "ontology_rag_enabled", True)):
            return {
                "ontology_facts_added": 0,
                "ontology_facts_deleted": 0,
            }
        changed_ids = [int(chunk_id) for chunk_id in (changed_chunk_ids or []) if int(chunk_id) > 0]
        llm_payloads_by_chunk: Dict[int, Any] = {}
        errors = 0
        extraction_disabled = bool(include_llm and not ONTOLOGY_LLM_EXTRACTION_ENABLED)
        if include_llm and ONTOLOGY_LLM_EXTRACTION_ENABLED and changed_ids:
            conn = self._connect_db()
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in changed_ids)
            rows = conn.execute(
                f"SELECT id, text, source_path, section FROM chunks WHERE id IN ({placeholders})",
                tuple(changed_ids),
            ).fetchall()
            conn.close()
            for row in rows:
                try:
                    llm_payloads_by_chunk[int(row["id"])] = self._extract_limited_llm_ontology_payload(
                        str(row["text"] or ""),
                        {
                            "source_path": str(row["source_path"] or ""),
                            "section": str(row["section"] or ""),
                        },
                    )
                except Exception:
                    errors += 1
        summary = OntologyStore(str(self.db_path), kb_id=self.kb_id).sync_facts_for_chunks(
            changed_chunk_ids=changed_ids,
            deleted_chunk_ids=[int(chunk_id) for chunk_id in (deleted_chunk_ids or []) if int(chunk_id) > 0],
            llm_payloads_by_chunk=llm_payloads_by_chunk,
            min_confidence=float(getattr(self, "ontology_min_fact_confidence", ONTOLOGY_MIN_FACT_CONFIDENCE)),
            llm_fact_status=llm_fact_status,
        )
        if errors:
            summary["ontology_extraction_errors"] = errors
        if extraction_disabled:
            summary["ontology_extraction_disabled"] = True
            summary["ontology_extraction_disabled_reason"] = "ONTOLOGY_LLM_EXTRACTION_ENABLED=0"
        return summary

    def _extract_limited_llm_ontology_payload(self, chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        text = str(chunk_text or "").strip()
        if not text:
            return []
        clipped = text[:3500]
        prompt = (
            "Extract only grounded subject-predicate-object facts from the Korean document chunk. "
            "Return JSON array only. Each item must have subject, predicate, object, confidence, evidence_quote. "
            "The evidence_quote must be an exact substring from the chunk. Do not infer facts without a quote.\n\n"
            f"metadata: {json.dumps(metadata or {}, ensure_ascii=False)}\n"
            f"chunk:\n{clipped}"
        )
        headers = {"Content-Type": "application/json"}
        if ONTOLOGY_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {ONTOLOGY_LLM_API_KEY}"
        response = requests.post(
            ONTOLOGY_LLM_API_URL,
            headers=headers,
            json={
                "model": ONTOLOGY_LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=ONTOLOGY_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            return []
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = str(message.get("content", first.get("text", "")) or "").strip()
        try:
            parsed = json.loads(content)
        except Exception:
            match = re.search(r"\[[\s\S]*\]", content)
            if not match:
                raise ValueError("ontology LLM response did not contain JSON array")
            parsed = json.loads(match.group(0))
        return parsed

    def _search_ontology_candidates(self, query: str, candidate_limit: int) -> Dict[int, float]:
        self._last_ontology_search_meta = {
            "matched_facts": [],
            "chunk_labels": {},
            "chunk_meta": {},
            "ontology_query_rewrite": query,
        }
        if not bool(getattr(self, "ontology_rag_enabled", True)) or not getattr(self, "db_path", None):
            return {}
        matches = OntologyStore(str(self.db_path), kb_id=self.kb_id).search_facts(
            query=query,
            limit=max(8, int(candidate_limit or 8)),
            max_hops=int(getattr(self, "ontology_max_hops", 2) or 2),
            min_confidence=ONTOLOGY_MIN_FACT_CONFIDENCE,
            allowed_extraction_methods=getattr(self, "ontology_allowed_extraction_methods", None),
            experiment_mode=str(getattr(self, "ontology_experiment_mode", "runtime") or "runtime"),
        )
        chunk_scores: Dict[int, float] = {}
        chunk_labels: Dict[int, List[str]] = {}
        chunk_meta: Dict[int, Dict[str, Any]] = {}
        for match in matches:
            chunk_id = int(match.get("chunk_id", 0) or 0)
            if chunk_id <= 0:
                continue
            score = float(match.get("score", 0.0) or 0.0)
            chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)
            label = str(match.get("label", "") or "")
            if label:
                chunk_labels.setdefault(chunk_id, [])
                if label not in chunk_labels[chunk_id]:
                    chunk_labels[chunk_id].append(label)
            meta = chunk_meta.setdefault(chunk_id, {
                "ontology_query_rewrite": str(match.get("ontology_query_rewrite", "") or ""),
                "ontology_hop_count": int(match.get("ontology_hop_count", 0) or 0),
                "ontology_candidate_reason": str(match.get("ontology_candidate_reason", "") or ""),
            })
            hop_count = int(match.get("ontology_hop_count", 0) or 0)
            current_hop = int(meta.get("ontology_hop_count", 0) or 0)
            if hop_count > 0 and (current_hop == 0 or hop_count < current_hop):
                meta["ontology_hop_count"] = hop_count
                meta["ontology_candidate_reason"] = str(match.get("ontology_candidate_reason", "") or "")
        self._last_ontology_search_meta = {
            "matched_facts": [str(match.get("label", "") or "") for match in matches[:8]],
            "chunk_labels": {chunk_id: labels[:4] for chunk_id, labels in chunk_labels.items()},
            "chunk_meta": chunk_meta,
            "ontology_query_rewrite": str(matches[0].get("ontology_query_rewrite", "") or query) if matches else query,
            "match_count": len(matches),
        }
        return chunk_scores

    def _rebuild_chunk_vectors_from_db(self, index_name: str = "large"):
        if not self.sqlite_dense_enabled:
            return

        rows = self._load_indexable_rows()
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("DELETE FROM chunk_vec WHERE index_name = ?", (index_name,))
        if not rows:
            conn.commit()
            conn.close()
            return

        texts = [r["embedding_text"] or r["text"] or "" for r in rows]
        labels = [int(r["id"]) for r in rows]
        embeddings = self._encode_texts(index_name=index_name, texts=texts, task="passage")
        now_ts = int(time.time())

        payload = []
        for row_id, vec in zip(labels, embeddings):
            arr = np.asarray(vec, dtype=np.float32)
            payload.append(
                (
                    int(row_id),
                    index_name,
                    int(arr.shape[0]),
                    sqlite3.Binary(arr.tobytes()),
                    now_ts,
                )
            )

        c.executemany(
            """
            INSERT INTO chunk_vec (chunk_pk, index_name, dim, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
        conn.close()

    def _rebuild_fts_index_from_db(self):
        if not getattr(self, "fts_available", False):
            return
        rows = self._load_indexable_rows()
        conn = self._connect_db()
        c = conn.cursor()
        try:
            c.execute("DELETE FROM chunk_fts")
            if rows:
                payload = [
                    (
                        int(r["id"]),
                        "\n".join(
                            part
                            for part in (r["text"] or "", r["embedding_text"] or "")
                            if part
                        ),
                    )
                    for r in rows
                ]
                c.executemany("INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)", payload)
            conn.commit()
        finally:
            conn.close()

    def _ensure_sqlite_search_artifacts(self, chunk_count: int):
        if self.sqlite_dense_enabled:
            vec_count = self._count_chunk_vectors(index_name="large")
            if vec_count != chunk_count:
                print("RAGEngine: rebuilding SQLite dense vectors...")
                self._rebuild_chunk_vectors_from_db(index_name="large")
        if getattr(self, "fts_available", False):
            fts_count = self._count_fts_rows()
            if fts_count != chunk_count:
                print("RAGEngine: rebuilding SQLite FTS index...")
                self._rebuild_fts_index_from_db()

    def _upsert_file_record(
        self,
        source_path: str,
        file_hash: str,
        file_path: str,
        source_label: str,
        source_type: str,
    ) -> str:
        file_id = uuid.uuid4().hex
        mime = (
            "application/pdf"
            if source_type == "pdf"
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if source_type == "xlsx"
            else "application/hwp+zip"
            if source_type == "hwpx"
            else "text/plain"
        )
        size = 0
        try:
            size = int(os.path.getsize(file_path))
        except Exception:
            size = 0
        now_ts = int(time.time())

        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT file_id FROM files WHERE source_path = ?", (source_path,))
        row = c.fetchone()
        if row and row[0]:
            file_id = str(row[0])

        c.execute(
            """
            INSERT INTO files
                (file_id, source_path, sha256, mime, orig_name, size, stored_path, uploaded_at, uploader_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                sha256 = excluded.sha256,
                mime = excluded.mime,
                orig_name = excluded.orig_name,
                size = excluded.size,
                stored_path = excluded.stored_path,
                uploaded_at = excluded.uploaded_at
            """,
            (
                file_id,
                source_path,
                file_hash,
                mime,
                source_label,
                size,
                file_path,
                now_ts,
                "",
            ),
        )
        conn.commit()
        conn.close()
        return file_id

    def _upsert_document_record(
        self,
        file_id: str,
        source_path: str,
        source_type: str,
        parser_name: str,
        parser_version: str,
    ) -> str:
        doc_id = uuid.uuid4().hex
        now_ts = int(time.time())
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT doc_id FROM documents WHERE source_path = ?", (source_path,))
        row = c.fetchone()
        if row and row[0]:
            doc_id = str(row[0])
        c.execute(
            """
            INSERT INTO documents
                (doc_id, file_id, source_path, doc_type, parser_name, parser_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_id = excluded.file_id,
                doc_type = excluded.doc_type,
                parser_name = excluded.parser_name,
                parser_version = excluded.parser_version,
                created_at = excluded.created_at
            """,
            (
                doc_id,
                file_id,
                source_path,
                source_type,
                parser_name,
                parser_version,
                now_ts,
            ),
        )
        conn.commit()
        conn.close()
        return doc_id

    def _replace_canonical_rows(
        self,
        doc_id: str,
        source_type: str,
        items: List[Dict[str, Any]],
        row_items: Optional[List[Dict[str, Any]]] = None,
    ):
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("DELETE FROM doc_blocks WHERE doc_id = ?", (doc_id,))
        c.execute("DELETE FROM doc_table_cells WHERE doc_id = ?", (doc_id,))

        blocks_payload = []
        for idx, item in enumerate(items, start=1):
            section = (item.get("section", "") or "").strip()
            page = 1
            if source_type == "pdf":
                m = re.search(r"PDF page\s+(\d+)", section, flags=re.IGNORECASE)
                if m:
                    page = max(1, int(m.group(1)))
            elif source_type == "xlsx":
                page = 1
            bbox = {
                "sheet": item.get("sheet", "") or "",
                "row": int(item.get("row", 0) or 0),
                "row_end": int(item.get("row_end", item.get("row", 0)) or 0),
                "line_start": int(item.get("line_start", 0) or 0),
                "line_end": int(item.get("line_end", 0) or 0),
                "section": section,
            }
            if source_type == "pdf":
                block_type = "pdf_text" if _is_pymupdf_page_parser(str(item.get("page_parser", "") or "").strip()) else "pdf_ocr"
            elif source_type == "hwpx":
                block_type = "hwpx_paragraph"
            else:
                block_type = "sheet_row" if source_type == "xlsx" else "text"
            blocks_payload.append(
                (
                    doc_id,
                    int(page),
                    block_type,
                    int(idx),
                    self._safe_json_dump(bbox),
                    item.get("text", "") or "",
                )
            )

        if blocks_payload:
            c.executemany(
                """
                INSERT INTO doc_blocks
                    (doc_id, page, block_type, reading_order, bbox_json, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                blocks_payload,
            )

        if source_type == "xlsx" and row_items:
            cell_payload = []
            for row in row_items:
                sheet = (row.get("sheet", "") or "").strip() or "Sheet1"
                row_no = int(row.get("row", 0) or 0)
                row_text = row.get("row_text", "") or ""
                parsed_row_no, kv = self._extract_xlsx_row_kv(row_text)
                effective_row = parsed_row_no if parsed_row_no and parsed_row_no > 0 else row_no
                col_idx = 0
                for key, val in kv.items():
                    col_idx += 1
                    cell_payload.append(
                        (
                            doc_id,
                            1,
                            sheet,
                            int(effective_row or 0),
                            int(col_idx),
                            self._safe_json_dump({"sheet": sheet, "row": effective_row, "col_key": key}),
                            val,
                        )
                    )
            if cell_payload:
                c.executemany(
                    """
                    INSERT INTO doc_table_cells
                        (doc_id, page, table_id, r, c, bbox_json, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    cell_payload,
                )

        conn.commit()
        conn.close()

    def _count_chunks(self) -> int:
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunks")
        count = int(c.fetchone()[0])
        conn.close()
        return count

    def _has_normalized_chunks(self) -> bool:
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT 1 FROM chunks WHERE COALESCE(is_normalized, 0) = 1 LIMIT 1")
        row = c.fetchone()
        conn.close()
        return bool(row)

    def _count_indexable_chunks(self) -> int:
        conn = self._connect_db()
        c = conn.cursor()
        if self.index_include_raw_with_normalized:
            c.execute("SELECT COUNT(*) FROM chunks")
            total = int(c.fetchone()[0])
            conn.close()
            return total

        c.execute("SELECT COUNT(*) FROM chunks WHERE COALESCE(is_normalized, 0) = 1")
        normalized_count = int(c.fetchone()[0])
        if normalized_count > 0:
            conn.close()
            return normalized_count
        c.execute("SELECT COUNT(*) FROM chunks")
        raw_count = int(c.fetchone()[0])
        conn.close()
        return raw_count

    def _count_normalized_chunks(self) -> int:
        conn = self._connect_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunks WHERE COALESCE(is_normalized, 0) = 1")
        count = int(c.fetchone()[0])
        conn.close()
        return count

    def _index_capacity(self, chunk_count: int) -> int:
        minimum = max(1024, chunk_count + 1024)
        return max(self.index_max_elements, minimum)

    def _create_empty_index(self, index: hnswlib.Index, chunk_count: int):
        max_elements = self._index_capacity(chunk_count)
        index.init_index(
            max_elements=max_elements,
            ef_construction=self.index_ef_construction,
            M=self.index_m,
        )
        index.set_ef(self.search_ef)

    def _init_or_load_indexes(self, chunk_count: int):
        loaded_large = self._try_load_index(self.index_large, self.index_large_path)

        if loaded_large:
            self.index_large.set_ef(self.search_ef)
            if self.index_large.element_count == chunk_count:
                return
            print("RAGEngine: Index/document count mismatch, rebuilding index...")
            self._rebuild_index_from_db(chunk_count=chunk_count)
            return

        # Missing index case: recreate index.
        self.index_large = hnswlib.Index(space="cosine", dim=self.dim_large)
        self._create_empty_index(self.index_large, chunk_count)

        if chunk_count > 0:
            print("RAGEngine: Building index from SQLite chunks...")
            self._rebuild_index_from_db(chunk_count=chunk_count)
        else:
            self._persist_index_atomically(self.index_large)

    def _try_load_index(self, index: hnswlib.Index, index_path: str) -> bool:
        if not os.path.exists(index_path):
            return False
        try:
            index.load_index(index_path)
            return True
        except Exception as e:
            print(f"RAGEngine Warning: failed to load index {index_path}: {e}")
            return False

    def _persist_index_atomically(self, index: hnswlib.Index):
        temp_path = f"{self.index_large_path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
        backup_path = f"{self.index_large_path}.bak"
        index.save_index(temp_path)
        try:
            if os.path.exists(self.index_large_path):
                try:
                    shutil.copy2(self.index_large_path, backup_path)
                except Exception:
                    pass
            os.replace(temp_path, self.index_large_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _rebuild_index_from_db(self, chunk_count: int):
        if not self.hnsw_enabled:
            return
        new_index = hnswlib.Index(space="cosine", dim=self.dim_large)
        self._create_empty_index(new_index, chunk_count)

        rows = self._load_indexable_rows()

        if not rows:
            self._persist_index_atomically(new_index)
            self.index_large = new_index
            self.index_large.set_ef(self.search_ef)
            return

        labels = np.asarray([int(r["id"]) for r in rows], dtype=np.int64)
        texts = [r["embedding_text"] or r["text"] or "" for r in rows]

        emb_large = self._encode_texts(index_name="large", texts=texts, task="passage")

        new_index.add_items(emb_large, labels)
        self._persist_index_atomically(new_index)
        self.index_large = new_index
        self.index_large.set_ef(self.search_ef)

    def _parser_signature(self) -> str:
        if self._is_api_provider():
            provider_sig = (
                f"api:{self.embedding_api_url}"
                f":{self.embedding_api_large_alias}"
            )
        else:
            large_sig = os.getenv("EMBEDDING_MODEL_LARGE_PATH", DEFAULT_EMBEDDING_MODEL_LARGE_PATH)
            provider_sig = f"local:{large_sig}"
        return (
            "v20-ppocr-fast-v1"
            f"|txt={self.txt_target_tokens}:{self.txt_min_tokens}:{self.txt_max_tokens}:{self.txt_overlap_ratio}"
            f"|txtreal={self.txt_target_tokens_real}:{self.txt_max_tokens_real}"
            f"|txtsplit={int(self.txt_split_enabled)}:{self.txt_split_trigger_lines}:{self.txt_split_target_tokens}:{self.txt_split_min_tokens}:{self.txt_split_max_tokens}"
            f"|pdf={self.pdf_target_tokens}:{self.pdf_min_tokens}:{self.pdf_max_tokens}"
            f"|pdfreal={self.pdf_target_tokens_real}:{self.pdf_max_tokens_real}:{int(self.embedding_compaction_enabled)}:{self.embedding_compaction_min_line_chars}:{self.embedding_compaction_max_hint_lines}:{self.embedding_compaction_max_chunk_chars}"
            f"|pdfocr={os.getenv('PDF_OCR_BACKEND', 'ppocr_fast_v1')}:{os.getenv('PDF_OCR_MODEL_NAME', 'PaddleOCR-VL-1.5')}:{os.getenv('PDF_OCR_MAX_PAGES', '240')}:{os.getenv('PDF_OCR_FAST_LANG', 'korean')}:{os.getenv('PDF_OCR_FAST_VL_FALLBACK', '0')}"
            f"|pdfparse={os.getenv('PDF_PARSE_MODE', 'hybrid')}:{os.getenv('PDF_TEXT_EXTRACTOR', 'pymupdf')}:{os.getenv('PDF_TEXT_MIN_CHARS', os.getenv('PDF_OCR_MIN_TEXT_CHARS', '4'))}:{os.getenv('PDF_TEXT_MIN_NONSPACE_RATIO', '0.20')}:{int(_env_bool('PDF_UPLOAD_OCR_ENABLED', False))}:lazyhintv1"
            f"|xlsx={self.xlsx_group_min_rows}:{self.xlsx_group_max_rows}:{self.xlsx_overlap_rows}:{self.xlsx_target_tokens}:{self.xlsx_max_tokens}"
            f"|hwpx={int(self.hwpx_extract_enabled)}:{int(self.hwpx_include_tables)}:{self.hwpx_target_tokens}:{self.hwpx_min_tokens}:{self.hwpx_max_tokens}:{self.hwpx_overlap_ratio}:{self.hwpx_target_tokens_real}:{self.hwpx_max_tokens_real}:python-hwpx-2.9-line-v4-generic-alias"
            f"|structurev2={int(self.structure_rag_v2_enabled)}:{int(self.hwpx_structure_rag_v2_enabled)}:{int(self.xlsx_structure_rag_v2_enabled)}"
            f"|policy={self.xlsx_merged_cell_policy}:{self.xlsx_comment_policy}"
            f"|xlsxguard={os.getenv('XLSX_MAX_SHEETS', '20')}:{os.getenv('XLSX_MAX_ROWS', '50000')}:{os.getenv('XLSX_MAX_COLS', '200')}:{os.getenv('XLSX_PARSE_TIMEOUT_SEC', '25')}"
            f"|embed={provider_sig}"
            f"|embedprefix={self.embedding_task_prefix_mode}:{hashlib.sha1(self.embedding_qwen_query_instruction.encode('utf-8')).hexdigest()[:8]}"
            f"|embedbudget={self.embedding_max_batch_tokens}:{os.getenv('EMBED_MAX_QUERY_TOKENS', '384')}:{os.getenv('EMBED_MAX_PASSAGE_TOKENS', '768')}"
            f"|dim={self.dim_large}"
            f"|indexmix={int(self.index_include_raw_with_normalized)}:{self.normalized_score_penalty}:{self.code_match_boost}:{self.code_hint_boost_ratio}:{self.exact_keyword_boost}"
            f"|hybrid={self.hybrid_fts_weight}:{int(self.sqlite_dense_enabled)}:{int(self.hnsw_enabled)}"
        )

    def _compute_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                hasher.update(block)
        return hasher.hexdigest()

    def _source_name(self, file_path: str, original_filename: Optional[str] = None) -> str:
        candidate = str(original_filename or "").strip()
        if candidate:
            candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not candidate:
            candidate = os.path.basename(file_path)
        candidate = "".join(char for char in candidate if ord(char) >= 32).strip()
        candidate = candidate.replace("::part:", "_part_")
        return candidate or os.path.basename(file_path)

    def _source_identity_key(self, filename: str) -> str:
        basename = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        return unicodedata.normalize("NFKC", basename).casefold()

    def _find_previous_source_records(
        self,
        source_path: str,
        original_filename: str,
    ) -> List[Dict[str, str]]:
        identity_key = self._source_identity_key(original_filename or source_path)
        if not identity_key:
            return []
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        records: Dict[str, Dict[str, str]] = {}
        try:
            if _table_exists(conn, "files"):
                for row in conn.execute(
                    "SELECT source_path, orig_name, stored_path FROM files"
                ).fetchall():
                    row_source = str(row["source_path"] or "").strip()
                    row_original = str(row["orig_name"] or row_source).strip()
                    if row_source and self._source_identity_key(row_original) == identity_key:
                        records[row_source] = {
                            "source_path": row_source,
                            "stored_path": str(row["stored_path"] or "").strip(),
                        }
            if _table_exists(conn, "source_uploads"):
                for row in conn.execute(
                    "SELECT source_path, original_filename FROM source_uploads"
                ).fetchall():
                    row_source = str(row["source_path"] or "").strip()
                    row_original = str(row["original_filename"] or row_source).strip()
                    if row_source and self._source_identity_key(row_original) == identity_key:
                        records.setdefault(
                            row_source,
                            {"source_path": row_source, "stored_path": ""},
                        )
        finally:
            conn.close()
        return [records[key] for key in sorted(records)]

    def _source_child_prefix(self, source_path: str) -> str:
        return f"{source_path}::part:"

    def _slugify_title(self, text: str, max_len: int = 48) -> str:
        raw = re.sub(r"\s+", " ", (text or "").strip().lower())
        slug = _SLUG_DISALLOWED_RE.sub("-", raw).strip("-")
        if not slug:
            return "part"
        return slug[:max_len]

    def _split_txt_lines_for_ingest(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not lines:
            return []

        docs: List[Dict[str, Any]] = []
        bucket: List[Dict[str, Any]] = []
        used_tokens = 0
        first_section = ""

        def flush_bucket():
            nonlocal bucket, used_tokens, first_section
            if not bucket:
                return
            title = first_section or bucket[0].get("text", "") or "part"
            docs.append({"title": title, "lines": list(bucket), "tokens": used_tokens})
            bucket = []
            used_tokens = 0
            first_section = ""

        for item in lines:
            line_text = item.get("text", "") or ""
            line_tokens = self._estimate_tokens_for_pack(line_text)
            is_section = bool(item.get("is_section", False))

            should_cut = False
            if bucket:
                if used_tokens + line_tokens > self.txt_split_max_tokens:
                    should_cut = True
                elif is_section and used_tokens >= self.txt_split_target_tokens and used_tokens >= self.txt_split_min_tokens:
                    should_cut = True

            if should_cut:
                flush_bucket()

            bucket.append(item)
            used_tokens += line_tokens
            if is_section and not first_section:
                first_section = line_text

        flush_bucket()

        if len(docs) >= 2:
            last = docs[-1]
            if int(last.get("tokens", 0) or 0) < max(180, int(self.txt_split_min_tokens * 0.35)):
                merged_lines = docs[-2]["lines"] + last["lines"]
                merged_tokens = int(docs[-2].get("tokens", 0) or 0) + int(last.get("tokens", 0) or 0)
                docs[-2] = {
                    "title": docs[-2].get("title", "") or last.get("title", "") or "part",
                    "lines": merged_lines,
                    "tokens": merged_tokens,
                }
                docs.pop()

        return docs

    def _materialize_split_txt_docs(
        self,
        source_path: str,
        source_label: str,
        docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []
        source_base = self._sanitize_for_cache(os.path.splitext(source_path)[0] or source_path)
        split_dir = os.path.join(self.derived_txt_dir, source_base)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

        parts: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            title = (doc.get("title", "") or f"part-{idx}").strip()
            slug = self._slugify_title(title)
            part_name = f"{idx:04d}_{slug}.txt"
            part_path = os.path.join(split_dir, part_name)
            part_source = f"{source_path}::part:{idx:04d}:{slug}"
            part_label = f"{source_label} [{idx:03d}]"
            content = "\n".join((ln.get("text", "") or "") for ln in doc.get("lines", []))
            with open(part_path, "w", encoding="utf-8") as f:
                f.write(content)
            parts.append(
                {
                    "source_path": part_source,
                    "source_label": part_label,
                    "file_path": part_path,
                    "lines": list(doc.get("lines", [])),
                }
            )
        return parts

    def _format_timestamp(self, ts: int) -> str:
        if not ts:
            return "-"
        try:
            return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "-"

    def _normalize_text_key(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        return cleaned

    def _normalize_doc_role(self, value: Optional[str]) -> str:
        raw = (value or "").strip().lower()
        if not raw:
            return DOC_ROLE_UNKNOWN
        normalized = DOC_ROLE_ALIASES.get(raw, "")
        return normalized or DOC_ROLE_UNKNOWN

    def _infer_doc_role(self, source_type: str, source_name: str = "") -> str:
        hint = (source_name or "").strip().lower()
        if any(k in hint for k in ("faq", "q&a", "qa", "case", "casebook")):
            return DOC_ROLE_CASEBOOK
        if any(k in hint for k in ("manual", "guide", "guideline", "rule")):
            return DOC_ROLE_GUIDE
        if source_type == "xlsx":
            return DOC_ROLE_CASEBOOK
        if source_type in {"txt", "pdf", "hwpx"}:
            return DOC_ROLE_GUIDE
        return DOC_ROLE_UNKNOWN

    def _normalize_doc_roles_filter(self, doc_roles: Optional[List[str]]) -> List[str]:
        if not doc_roles:
            return []
        normalized: List[str] = []
        for raw in doc_roles:
            role = self._normalize_doc_role(raw)
            if role == DOC_ROLE_UNKNOWN:
                continue
            if role not in normalized:
                normalized.append(role)
        return normalized

    def _normalized_source_for_group_role(self, group: str, doc_role: str) -> str:
        g = (group or "data").strip().lower() or "data"
        role = self._normalize_doc_role(doc_role)
        return f"__normalized_{g}_{role}__"

    def _estimate_tokens_for_pack(self, text: str) -> int:
        if not text:
            return 0
        tokens = re.findall(r"[0-9A-Za-z_]+", text)
        return max(1, int(len(tokens) * 1.05))

    def _upsert_source_upload_meta(
        self,
        source_path: str,
        source_type: str,
        doc_role: str,
        file_hash: str,
        doc_version: str,
        uploaded_at: int,
        original_filename: str,
    ):
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO source_uploads
                (source_path, source_type, doc_role, file_hash, doc_version, uploaded_at, original_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_type = excluded.source_type,
                doc_role = excluded.doc_role,
                file_hash = excluded.file_hash,
                doc_version = excluded.doc_version,
                uploaded_at = excluded.uploaded_at,
                original_filename = excluded.original_filename
            """,
            (
                source_path,
                source_type,
                self._normalize_doc_role(doc_role),
                file_hash,
                doc_version,
                int(uploaded_at),
                original_filename,
            ),
        )
        conn.commit()
        conn.close()

    def _fetch_raw_chunks_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT
                c.*,
                COALESCE(s.uploaded_at, c.source_updated_at, 0) AS uploaded_at,
                COALESCE(NULLIF(s.doc_role, ''), NULLIF(c.doc_role, ''), ?) AS doc_role,
                COALESCE(NULLIF(s.original_filename, ''), c.source_path) AS source_display
            FROM chunks c
            LEFT JOIN source_uploads s
                ON c.source_path = s.source_path
            WHERE c.source_type = ?
              AND COALESCE(c.is_normalized, 0) = 0
            ORDER BY COALESCE(s.uploaded_at, c.source_updated_at, 0) DESC, c.id DESC
            """,
            (self._infer_doc_role(source_type), source_type),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_recent_chunks(
        self,
        limit: int = 12,
        doc_roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        role_filters = self._normalize_doc_roles_filter(doc_roles)
        params: List[Any] = [self._infer_doc_role("txt"), max(1, int(limit))]
        role_sql = ""
        if role_filters:
            placeholders = ", ".join("?" for _ in role_filters)
            role_sql = f" AND COALESCE(NULLIF(s.doc_role, ''), NULLIF(c.doc_role, ''), ?) IN ({placeholders})"
            params = [
                self._infer_doc_role("txt"),
                self._infer_doc_role("txt"),
                *role_filters,
                max(1, int(limit)),
            ]

        c.execute(
            f"""
            SELECT
                c.*,
                COALESCE(s.uploaded_at, c.source_updated_at, 0) AS uploaded_at,
                COALESCE(NULLIF(s.doc_role, ''), NULLIF(c.doc_role, ''), ?) AS doc_role,
                COALESCE(NULLIF(s.original_filename, ''), c.source_path) AS source_display
            FROM chunks c
            LEFT JOIN source_uploads s
                ON c.source_path = s.source_path
            WHERE COALESCE(c.is_normalized, 0) = 0
              {role_sql}
            ORDER BY COALESCE(s.uploaded_at, c.source_updated_at, 0) DESC, c.id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def _build_txt_location_label(self, row: Dict[str, Any]) -> str:
        line_start = int(row.get("line_start", 0) or 0)
        line_end = int(row.get("line_end", line_start) or line_start)
        section = (row.get("section", "") or "").strip()
        if line_start > 0:
            return f"{LABEL_LINE} {line_start}-{line_end}"
        if section:
            return section
        return LABEL_NO_LOCATION

    def _extract_xlsx_row_kv(self, line: str) -> Tuple[Optional[int], Dict[str, str]]:
        m = re.match(r"^Row\s+(\d+)\s*:\s*(.*)$", (line or "").strip(), flags=re.IGNORECASE)
        if not m:
            return None, {}
        row_no = int(m.group(1))
        body = m.group(2).strip()
        kv: Dict[str, str] = {}
        if not body:
            return row_no, kv
        for part in body.split("|"):
            token = part.strip()
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            k = key.strip()
            v = val.strip()
            if k:
                kv[k] = v
        return row_no, kv

    def _extract_xlsx_qa_records(
        self,
        text: str,
        default_sheet: str,
        default_row: int,
    ) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return []

        current_sheet = default_sheet or ""
        records: List[Dict[str, Any]] = []

        for line in lines:
            if line.startswith("[Sheet:") and line.endswith("]"):
                current_sheet = line[len("[Sheet:") : -1].strip()
                continue
            row_no, kv = self._extract_xlsx_row_kv(line)
            if row_no is None:
                continue
            if not kv:
                continue

            question = ""
            answer = ""
            for k, v in kv.items():
                norm = re.sub(r"\s+", "", k).lower()
                if (not question) and any(m in norm for m in QUESTION_COLUMN_MARKERS):
                    question = v
                if (not answer) and any(m in norm for m in ANSWER_COLUMN_MARKERS):
                    answer = v

            if not question:
                # Heuristic fallback for loosely structured rows.
                values = [v for v in kv.values() if v]
                if values:
                    question = values[0]
            if not answer:
                values = [v for v in kv.values() if v]
                if len(values) >= 2:
                    answer = values[1]
                elif values:
                    answer = values[0]

            if not question and not answer:
                continue

            records.append(
                {
                    "question": (question or "").strip(),
                    "answer": (answer or "").strip(),
                    "sheet": current_sheet or default_sheet or "-",
                    "row": int(row_no or default_row or 0),
                }
            )

        if records:
            return records

        # Last fallback: preserve full text to avoid dropping weakly formatted rows.
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        return [
            {
                "question": compact[:220],
                "answer": compact[:320],
                "sheet": default_sheet or "-",
                "row": int(default_row or 0),
            }
        ]

    def _pack_normalized_entries(
        self,
        entries: List[Dict[str, Any]],
        normalized_source: str,
        normalized_group: str,
        normalized_doc_role: str,
    ) -> List[Dict[str, Any]]:
        if not entries:
            return []

        entries = sorted(
            entries,
            key=lambda x: int(x.get("uploaded_at", 0) or 0),
            reverse=True,
        )

        chunks: List[Dict[str, Any]] = []
        idx = 0
        chunk_no = 1

        while idx < len(entries):
            used_tokens = 0
            parts: List[str] = []
            max_uploaded = 0
            source_order: List[str] = []

            while idx < len(entries):
                item = entries[idx]
                item_text = (item.get("text", "") or "").strip()
                if not item_text:
                    idx += 1
                    continue

                add_tokens = self._estimate_tokens_for_pack(item_text)
                if parts and used_tokens + add_tokens > self.normalized_max_tokens:
                    break

                parts.append(item_text)
                used_tokens += add_tokens

                uploaded = int(item.get("uploaded_at", 0) or 0)
                if uploaded > max_uploaded:
                    max_uploaded = uploaded

                src = (item.get("source_label", "") or item.get("source_path", "") or "").strip()
                if src and src not in source_order:
                    source_order.append(src)

                idx += 1
                if used_tokens >= self.normalized_target_tokens:
                    break

            if not parts:
                continue

            source_preview = ", ".join(source_order[:3]) if source_order else "-"
            if len(source_order) > 3:
                source_preview += f" +{len(source_order) - 3} more"

            header = (
                f"{normalized_bundle_header(normalized_group)}\n"
                f"[{LABEL_LATEST_UPLOAD_REFLECTED_AT}] {self._format_timestamp(max_uploaded)}\n"
                f"[{LABEL_SOURCE_SUMMARY}] {source_preview}\n"
            )
            body = "\n\n---\n\n".join(parts)
            text = f"{header}\n{body}"

            item_row = chunk_no if normalized_group == "xlsx" else 0
            item_line = chunk_no if normalized_group in {"txt", "pdf"} else 0
            chunks.append(
                {
                    "text": text,
                    "source_path": normalized_source,
                    "source_type": normalized_group,
                    "doc_role": self._normalize_doc_role(normalized_doc_role),
                    "sheet": "normalized" if normalized_group == "xlsx" else "",
                    "row": item_row,
                    "row_end": item_row,
                    "line_start": item_line,
                    "line_end": item_line,
                    "section": normalized_bundle_section(normalized_group),
                    "normalized_group": normalized_group,
                    "source_updated_at": int(max_uploaded),
                }
            )
            chunk_no += 1

        return chunks

    def _build_normalized_txt_chunks(self) -> List[Dict[str, Any]]:
        rows = self._fetch_raw_chunks_by_type("txt")
        rows.extend(self._fetch_raw_chunks_by_type("pdf"))
        if not rows:
            return []

        dedup_by_role: Dict[str, set] = {}
        entries_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            text = (row.get("text", "") or "").strip()
            if not text:
                continue
            row_source_type = (row.get("source_type", "") or "").strip().lower() or "txt"
            row_role = self._normalize_doc_role(row.get("doc_role", ""))
            if row_role == DOC_ROLE_UNKNOWN:
                row_role = self._infer_doc_role(
                    row_source_type,
                    row.get("source_display", row.get("source_path", "")),
                )
            key = self._normalize_text_key(text)
            if not key:
                continue
            if row_role not in dedup_by_role:
                dedup_by_role[row_role] = set()
            if key in dedup_by_role[row_role]:
                continue
            dedup_by_role[row_role].add(key)

            uploaded_at = int(row.get("uploaded_at", 0) or 0)
            source = (row.get("source_path", "") or "").strip()
            source_label = (row.get("source_display", "") or source).strip()
            location = self._build_txt_location_label(row)

            entry_text = (
                f"[{LABEL_SOURCE}] {source_label} | {LABEL_UPLOAD} {self._format_timestamp(uploaded_at)}\n"
                f"[{LABEL_LOCATION}] {location}\n"
                f"{text}"
            )
            entries_by_role.setdefault(row_role, []).append(
                {
                    "text": entry_text,
                    "uploaded_at": uploaded_at,
                    "source_path": source,
                    "source_label": source_label,
                }
            )

        packed: List[Dict[str, Any]] = []
        for role, entries in entries_by_role.items():
            packed.extend(
                self._pack_normalized_entries(
                    entries=entries,
                    normalized_source=self._normalized_source_for_group_role("txt", role),
                    normalized_group="txt",
                    normalized_doc_role=role,
                )
            )
        return packed

    def _build_normalized_xlsx_chunks(self) -> List[Dict[str, Any]]:
        rows = self._fetch_raw_chunks_by_type("xlsx")
        if not rows:
            return []

        qa_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for row in rows:
            uploaded_at = int(row.get("uploaded_at", 0) or 0)
            source = (row.get("source_path", "") or "").strip()
            source_label = (row.get("source_display", "") or source).strip()
            default_sheet = (row.get("sheet", "") or "").strip()
            default_row = int(row.get("row", 0) or 0)
            row_role = self._normalize_doc_role(row.get("doc_role", ""))
            if row_role == DOC_ROLE_UNKNOWN:
                row_role = self._infer_doc_role("xlsx", source_label or source)

            qa_records = self._extract_xlsx_qa_records(
                text=row.get("text", "") or "",
                default_sheet=default_sheet,
                default_row=default_row,
            )
            for qa in qa_records:
                question = (qa.get("question", "") or "").strip()
                answer = (qa.get("answer", "") or "").strip()
                if not question:
                    continue

                q_key = self._normalize_text_key(question)
                if not q_key:
                    continue
                grouped_key = f"{row_role}::{q_key}"

                if grouped_key not in qa_map:
                    qa_map[grouped_key] = {
                        "doc_role": row_role,
                        "question": question,
                        "answers": [],
                    }

                answer_key = self._normalize_text_key(answer)
                answers: List[Dict[str, Any]] = qa_map[grouped_key]["answers"]
                if any(a.get("answer_key") == answer_key for a in answers):
                    continue

                row_no = int(qa.get("row", 0) or 0)
                sheet = (qa.get("sheet", default_sheet) or default_sheet or "-").strip()
                location = f"{sheet} {LABEL_ROW} {row_no}" if row_no > 0 else f"{sheet}"
                answers.append(
                    {
                        "answer": answer,
                        "answer_key": answer_key,
                        "source_path": source,
                        "source_label": source_label,
                        "uploaded_at": uploaded_at,
                        "location": location,
                    }
                )

        entries: List[Dict[str, Any]] = []
        for payload in qa_map.values():
            question = payload.get("question", "")
            answers = payload.get("answers", [])
            payload_role = self._normalize_doc_role(payload.get("doc_role", ""))
            if not question or not answers:
                continue

            answers = sorted(
                answers,
                key=lambda x: int(x.get("uploaded_at", 0) or 0),
                reverse=True,
            )

            lines = [f"[{LABEL_QUESTION}] {question}", f"[{LABEL_ANSWER_PRIORITY}]"]
            for idx, ans in enumerate(answers[: self.normalized_conflict_limit], start=1):
                label = LABEL_LATEST if idx == 1 else f"이전{idx - 1}"
                answer_text = (ans.get("answer", "") or "").strip() or "(답변 공백)"
                lines.append(f"{idx}. ({label}) {answer_text}")
                lines.append(
                    "   "
                    f"{LABEL_SOURCE}={ans.get('source_label', ans.get('source_path', '-'))} | "
                    f"{LABEL_LOCATION}={ans.get('location', '-')} | "
                    f"{LABEL_UPLOAD}={self._format_timestamp(int(ans.get('uploaded_at', 0) or 0))}"
                )
            if len(answers) > self.normalized_conflict_limit:
                lines.append(f"... additional answers: {len(answers) - self.normalized_conflict_limit}")

            latest = answers[0]
            entries.append(
                {
                    "text": "\n".join(lines),
                    "doc_role": payload_role,
                    "uploaded_at": int(latest.get("uploaded_at", 0) or 0),
                    "source_path": latest.get("source_path", "") or "",
                    "source_label": latest.get("source_label", "") or latest.get("source_path", ""),
                }
            )

        entries_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            role = self._normalize_doc_role(entry.get("doc_role", ""))
            entries_by_role.setdefault(role, []).append(entry)

        packed: List[Dict[str, Any]] = []
        for role, role_entries in entries_by_role.items():
            packed.extend(
                self._pack_normalized_entries(
                    entries=role_entries,
                    normalized_source=self._normalized_source_for_group_role("xlsx", role),
                    normalized_group="xlsx",
                    normalized_doc_role=role,
                )
            )
        return packed

    def _normalized_groups_for_source_type(self, source_type: str) -> List[str]:
        normalized_type = (source_type or "").strip().lower()
        if normalized_type in {"txt", "pdf", "hwpx"}:
            return ["txt"]
        if normalized_type == "xlsx":
            return ["xlsx"]
        return []

    def _build_normalized_chunks_for_group(self, group: str) -> List[Dict[str, Any]]:
        normalized_group = (group or "").strip().lower()
        if normalized_group == "txt":
            return self._build_normalized_txt_chunks()
        if normalized_group == "xlsx":
            return self._build_normalized_xlsx_chunks()
        return []

    def _refresh_normalized_chunks_and_index(
        self,
        affected_groups: Optional[List[str]] = None,
    ) -> Dict[str, List[int]]:
        groups = [str(group).strip().lower() for group in (affected_groups or ["txt", "xlsx"]) if str(group).strip()]
        if not groups:
            return {"inserted_chunk_ids": [], "deleted_chunk_ids": []}

        inserted_ids: List[int] = []
        deleted_ids: List[int] = []
        conn = self._connect_db()
        c = conn.cursor()
        try:
            c.execute("BEGIN IMMEDIATE")
            norm_version = f"norm-{int(time.time())}"
            for normalized_group in groups:
                c.execute(
                    """
                    SELECT id
                    FROM chunks
                    WHERE COALESCE(is_normalized, 0) = 1
                      AND normalized_group = ?
                    ORDER BY id ASC
                    """,
                    (normalized_group,),
                )
                deleted_ids.extend(int(row[0]) for row in c.fetchall())
                c.execute(
                    """
                    DELETE FROM chunks
                    WHERE COALESCE(is_normalized, 0) = 1
                      AND normalized_group = ?
                    """,
                    (normalized_group,),
                )

                normalized_items = self._build_normalized_chunks_for_group(normalized_group)
                for idx, item in enumerate(normalized_items, start=1):
                    chunk_id = f"normalized:{normalized_group}:{idx:06d}"
                    c.execute(
                        """
                        INSERT INTO chunks
                            (chunk_id, kb_id, source_path, source_type, doc_role, sheet, row, row_end,
                             line_start, line_end, section, doc_version, text,
                             is_normalized, normalized_group, source_updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            self.kb_id,
                            item.get("source_path", "") or "",
                            item.get("source_type", "") or "",
                            self._normalize_doc_role(item.get("doc_role", "")),
                            item.get("sheet", "") or "",
                            int(item.get("row", 0) or 0),
                            int(item.get("row_end", item.get("row", 0)) or 0),
                            int(item.get("line_start", 0) or 0),
                            int(item.get("line_end", item.get("line_start", 0)) or 0),
                            item.get("section", "") or "",
                            norm_version,
                            item.get("text", "") or "",
                            1,
                            normalized_group,
                            int(item.get("source_updated_at", 0) or 0),
                        ),
                    )
                    inserted_ids.append(int(c.lastrowid))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return {
            "inserted_chunk_ids": inserted_ids,
            "deleted_chunk_ids": deleted_ids,
        }

    def _sanitize_for_cache(self, text: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", text)

    def _cache_paths(self, source_path: str, file_hash: str) -> Tuple[str, str]:
        safe_source = self._sanitize_for_cache(source_path)
        sig_hash = hashlib.sha1(self._parser_signature().encode("utf-8")).hexdigest()[:10]
        key = f"{safe_source}.{file_hash[:12]}.{sig_hash}"
        items_path = os.path.join(self.cache_dir, f"{key}.items.pkl")
        large_path = os.path.join(self.cache_dir, f"{key}.large.npy")
        return items_path, large_path

    def _get_file_cache_meta(self, source_path: str) -> Optional[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM file_cache WHERE source_path = ?", (source_path,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def _get_file_cache_meta_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT * FROM file_cache
            WHERE file_hash = ? AND parser_sig = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (file_hash, self._parser_signature()),
        )
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def _upsert_file_cache_meta(
        self,
        source_path: str,
        file_hash: str,
        items_cache_path: str,
        emb_large_cache_path: str,
        item_count: int,
    ):
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO file_cache
                (source_path, file_hash, parser_sig, items_cache_path,
                 embeddings_cache_path, emb_small_cache_path, emb_large_cache_path,
                 item_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_hash = excluded.file_hash,
                parser_sig = excluded.parser_sig,
                items_cache_path = excluded.items_cache_path,
                embeddings_cache_path = excluded.embeddings_cache_path,
                emb_small_cache_path = excluded.emb_small_cache_path,
                emb_large_cache_path = excluded.emb_large_cache_path,
                item_count = excluded.item_count,
                updated_at = excluded.updated_at
            """,
            (
                source_path,
                file_hash,
                self._parser_signature(),
                items_cache_path,
                emb_large_cache_path,
                None,
                emb_large_cache_path,
                int(item_count),
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()

    def _load_cached_payload(
        self,
        source_path: str,
        file_hash: str,
    ) -> Optional[Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]]:
        meta = self._get_file_cache_meta(source_path) or self._get_file_cache_meta_by_hash(file_hash)
        if not meta:
            return None
        if meta.get("file_hash") != file_hash:
            return None
        if meta.get("parser_sig") != self._parser_signature():
            return None

        items_path = meta.get("items_cache_path", "")
        if not items_path:
            return None
        if not os.path.exists(items_path):
            return None

        try:
            with open(items_path, "rb") as f:
                cached_payload = pickle.load(f)
            payload_meta: Dict[str, Any] = {}
            if isinstance(cached_payload, dict):
                items = cached_payload.get("items", [])
                payload_meta = _normalize_pdf_ingest_stats(cached_payload.get("meta"))
            else:
                items = cached_payload
            if not isinstance(items, list):
                return None
            large_path = (
                meta.get("emb_large_cache_path")
                or meta.get("embeddings_cache_path", "")
                or meta.get("emb_small_cache_path", "")
            )
            if large_path and os.path.exists(large_path):
                emb_large = np.load(large_path)
                if (
                    emb_large.ndim == 2
                    and len(items) == emb_large.shape[0]
                    and emb_large.shape[1] == self.dim_large
                ):
                    return items, emb_large, payload_meta
            return items, np.empty((0, self.dim_large), dtype=np.float32), payload_meta
        except Exception:
            return None

    def _save_cached_payload(
        self,
        source_path: str,
        file_hash: str,
        items: List[Dict[str, Any]],
        emb_large: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        previous_meta = self._get_file_cache_meta(source_path) or {}
        items_path, large_path = self._cache_paths(source_path, file_hash)
        payload_to_store: Dict[str, Any] = {"items": items}
        normalized_meta = _normalize_pdf_ingest_stats(meta)
        if normalized_meta:
            payload_to_store["meta"] = normalized_meta
        with open(items_path, "wb") as f:
            pickle.dump(payload_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        emb_path_for_meta = ""
        if emb_large is not None:
            np.save(large_path, emb_large)
            emb_path_for_meta = large_path
        self._upsert_file_cache_meta(
            source_path=source_path,
            file_hash=file_hash,
            items_cache_path=items_path,
            emb_large_cache_path=emb_path_for_meta,
            item_count=len(items),
        )
        active_paths = {
            os.path.realpath(os.path.abspath(value))
            for value in (items_path, emb_path_for_meta)
            if value
        }
        for key in (
            "items_cache_path",
            "embeddings_cache_path",
            "emb_small_cache_path",
            "emb_large_cache_path",
        ):
            previous_path = str(previous_meta.get(key, "") or "").strip()
            if not previous_path:
                continue
            if os.path.realpath(os.path.abspath(previous_path)) in active_paths:
                continue
            _remove_managed_file(previous_path, [self.cache_dir])

    def _lazy_ocr_cache_dir(self) -> str:
        lazy_dir = os.path.join(self.cache_dir, "lazy_ocr")
        os.makedirs(lazy_dir, exist_ok=True)
        return lazy_dir

    def _lazy_ocr_cache_path(self, source_path: str, file_hash: str, page_no: int) -> str:
        safe_source = self._sanitize_for_cache(source_path)
        sig_hash = hashlib.sha1(self._parser_signature().encode("utf-8")).hexdigest()[:10]
        key = f"{safe_source}.{file_hash[:12]}.page{max(1, int(page_no))}.{sig_hash}.json"
        return os.path.join(self._lazy_ocr_cache_dir(), key)

    def _load_lazy_ocr_cache_text(self, source_path: str, file_hash: str, page_no: int) -> str:
        cache_path = self._lazy_ocr_cache_path(source_path, file_hash, page_no)
        if not os.path.exists(cache_path):
            return ""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return ""
        if str(payload.get("file_hash", "") or "") != str(file_hash or ""):
            return ""
        try:
            cached_page_no = int(payload.get("page_no", 0) or 0)
        except (TypeError, ValueError):
            cached_page_no = 0
        if cached_page_no != int(page_no):
            return ""
        return str(payload.get("text", "") or "").strip()

    def _save_lazy_ocr_cache_text(self, source_path: str, file_hash: str, page_no: int, text: str):
        cache_path = self._lazy_ocr_cache_path(source_path, file_hash, page_no)
        payload = {
            "source_path": source_path,
            "file_hash": file_hash,
            "page_no": int(page_no),
            "text": str(text or ""),
            "saved_at": int(time.time()),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _resolve_lazy_ocr_source_file(self, source_path: str) -> Dict[str, Any]:
        normalized_source = (source_path or "").strip()
        if not normalized_source:
            return {}
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        try:
            c.execute(
                """
                SELECT stored_path, sha256
                FROM files
                WHERE source_path = ?
                LIMIT 1
                """,
                (normalized_source,),
            )
            row = c.fetchone()
        finally:
            conn.close()
        if row:
            stored_path = str(row["stored_path"] or "").strip()
            file_hash = str(row["sha256"] or "").strip()
            if stored_path:
                return {"stored_path": stored_path, "file_hash": file_hash}
        if os.path.exists(normalized_source):
            return {"stored_path": normalized_source, "file_hash": ""}
        return {}

    def _extract_pdf_page_no_from_row(self, row: Dict[str, Any]) -> int:
        try:
            page_no = int(row.get("page_no", 0) or 0)
        except (TypeError, ValueError):
            page_no = 0
        if page_no > 0:
            return page_no
        section = str(row.get("section", "") or "")
        match = re.search(r"PDF page\s+(\d+)", section)
        if match:
            return max(1, int(match.group(1)))
        return 0

    def _should_use_lazy_ocr_for_row(self, row: Dict[str, Any]) -> bool:
        if str(row.get("source_type", "") or "").strip().lower() != "pdf":
            return False
        if not str(row.get("source_path", "") or "").strip():
            return False
        if self._extract_pdf_page_no_from_row(row) <= 0:
            return False
        text = str(row.get("text", "") or "")
        if is_weak_ocr_hint_text(text):
            return True
        return any(marker in text for marker in TABLE_HINT_MARKERS)

    def get_lazy_pdf_page_text(self, source_path: str, page_no: int) -> str:
        normalized_source = (source_path or "").strip()
        safe_page_no = max(1, int(page_no or 0))
        if not normalized_source or safe_page_no <= 0:
            return ""
        source_meta = self._resolve_lazy_ocr_source_file(normalized_source)
        stored_path = str(source_meta.get("stored_path", "") or "").strip()
        file_hash = str(source_meta.get("file_hash", source_meta.get("sha256", "")) or "").strip()
        if not stored_path:
            return ""

        cached_text = self._load_lazy_ocr_cache_text(normalized_source, file_hash, safe_page_no)
        if cached_text:
            return cached_text

        answer_path_ocr_enabled = _env_bool("PDF_ANSWER_PATH_LAZY_OCR_ENABLED", False)
        if not answer_path_ocr_enabled:
            return ""

        try:
            ocr_pages = extract_pdf_pages_with_paddleocr_vl(
                stored_path,
                page_numbers=[safe_page_no],
                total_document_pages=safe_page_no,
                completed_pages_base=0,
            )
        except Exception as exc:
            print(f"[LAZY_OCR][ERROR] source={normalized_source} page={safe_page_no} error={exc}")
            return ""
        finally:
            release_cached_ocr_model()

        resolved_text = ""
        for page in list(ocr_pages or []):
            try:
                resolved_page_no = int(page.get("page_no", 0) or 0)
            except (TypeError, ValueError):
                resolved_page_no = 0
            page_text = str(page.get("text", "") or "").strip()
            if not page_text:
                continue
            if resolved_page_no in {0, safe_page_no}:
                resolved_text = page_text
                break

        if resolved_text:
            self._save_lazy_ocr_cache_text(normalized_source, file_hash, safe_page_no, resolved_text)
        return resolved_text

    def get_lazy_pdf_page_text_for_row(self, row: Dict[str, Any]) -> str:
        if not self._should_use_lazy_ocr_for_row(row):
            return ""
        return self.get_lazy_pdf_page_text(
            str(row.get("source_path", "") or ""),
            self._extract_pdf_page_no_from_row(row),
        )

    def _delete_chunks_for_source(self, source_path: str) -> Tuple[int, List[int]]:
        like_prefix = f"{self._source_child_prefix(source_path)}%"
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            SELECT id
            FROM chunks
            WHERE COALESCE(is_normalized, 0) = 0
              AND (source_path = ? OR source_path LIKE ?)
            ORDER BY id ASC
            """,
            (source_path, like_prefix),
        )
        deleted_ids = [int(row[0]) for row in c.fetchall()]
        if deleted_ids:
            c.execute(
                """
                DELETE FROM chunks
                WHERE COALESCE(is_normalized, 0) = 0
                  AND (source_path = ? OR source_path LIKE ?)
                """,
                (source_path, like_prefix),
            )
        conn.commit()
        conn.close()
        return len(deleted_ids), deleted_ids

    def _delete_source_upload_meta_for_source(self, source_path: str):
        like_prefix = f"{self._source_child_prefix(source_path)}%"
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            "DELETE FROM source_uploads WHERE source_path = ? OR source_path LIKE ?",
            (source_path, like_prefix),
        )
        conn.commit()
        conn.close()

    def _query_cache_key(
        self,
        query: str,
        top_k: int,
        index_name: str,
        doc_roles: Optional[List[str]] = None,
    ) -> str:
        norm = " ".join((query or "").strip().lower().split())
        normalized_roles = self._normalize_doc_roles_filter(doc_roles)
        role_key = ",".join(normalized_roles) if normalized_roles else "*"
        return f"{index_name}|{top_k}|{role_key}|{norm}"

    def _get_cached_query_result(
        self,
        query: str,
        top_k: int,
        index_name: str,
        doc_roles: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        key = self._query_cache_key(query, top_k, index_name, doc_roles=doc_roles)
        cached = self.query_cache.get(key)
        if cached is None:
            return None
        return [dict(x) for x in cached]

    def _set_cached_query_result(
        self,
        query: str,
        top_k: int,
        index_name: str,
        results: List[Dict[str, Any]],
        doc_roles: Optional[List[str]] = None,
    ):
        key = self._query_cache_key(query, top_k, index_name, doc_roles=doc_roles)
        self.query_cache[key] = [dict(x) for x in results]
        if len(self.query_cache) > self.query_cache_limit:
            oldest = next(iter(self.query_cache))
            del self.query_cache[oldest]

    def _append_to_index(self, index: hnswlib.Index, embeddings: np.ndarray, labels: np.ndarray):
        needed = index.element_count + int(len(labels))
        if needed > index.get_max_elements():
            target = max(needed + 1024, int(index.get_max_elements() * 1.5))
            index.resize_index(target)
        index.add_items(embeddings, labels)

    def _emit_ingest_progress(
        self,
        progress_callback: Optional[Callable[..., None]],
        percent: int,
        message: str,
        stage: Optional[str] = None,
        **progress_meta: Any,
    ) -> None:
        if progress_callback is None:
            return
        stage_value = stage or "processing"
        if progress_meta:
            try:
                progress_callback(int(percent), message, stage_value, **progress_meta)
                return
            except TypeError:
                pass
        try:
            progress_callback(int(percent), message, stage_value)
        except TypeError:
            progress_callback(int(percent), message)

    def prepare_ingest_payload(
        self,
        file_path: str,
        *,
        original_filename: Optional[str] = None,
        document_role: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
        force_pdf_ocr: bool = False,
        pdf_ocr_mode: str = "",
    ) -> Dict[str, Any]:
        """Prepare OCR/text chunks for ingest without mutating chunk/index state."""
        phase_timings: Dict[str, float] = {"ocr_duration_seconds": 0.0}

        def _progress(percent: int, message: str, stage: str, **progress_meta: Any) -> None:
            self._emit_ingest_progress(progress_callback, percent, message, stage, **progress_meta)

        print(f"RAGEngine: Ingesting {file_path}...")
        ext = os.path.splitext(file_path)[1].lower()
        _progress(12, "파일 형식과 버전을 확인하는 중입니다.", "inspect_file")
        source_path = self._source_name(file_path, original_filename=original_filename)
        file_hash = self._compute_file_hash(file_path)
        _progress(18, "기존 처리 결과를 확인하는 중입니다.", "check_cache")
        source_type = (
            "txt"
            if ext == ".txt"
            else "xlsx"
            if ext == ".xlsx"
            else "pdf"
            if ext == ".pdf"
            else "hwpx"
            if ext == ".hwpx"
            else "unknown"
        )
        if source_type == "unknown":
            return {"status": "unsupported", "source_path": source_path, "used_cache": False}

        source_label = (original_filename or source_path or "").strip() or source_path
        previous_source_records = self._find_previous_source_records(
            source_path=source_path,
            original_filename=source_label,
        )
        explicit_doc_role = self._normalize_doc_role(document_role)
        inferred_doc_role = (
            explicit_doc_role
            if explicit_doc_role != DOC_ROLE_UNKNOWN
            else self._infer_doc_role(source_type=source_type, source_name=source_label)
        )
        txt_target_tokens = int(getattr(self, "txt_target_tokens", 640) or 640)
        txt_max_tokens = int(getattr(self, "txt_max_tokens", 900) or 900)
        pdf_target_tokens = int(getattr(self, "pdf_target_tokens", txt_target_tokens) or txt_target_tokens)
        pdf_max_tokens = int(getattr(self, "pdf_max_tokens", txt_max_tokens) or txt_max_tokens)
        hwpx_target_tokens = int(getattr(self, "hwpx_target_tokens", 220) or 220)
        hwpx_max_tokens = int(getattr(self, "hwpx_max_tokens", 320) or 320)
        txt_target_tokens_real = int(getattr(self, "txt_target_tokens_real", txt_target_tokens) or txt_target_tokens)
        txt_max_tokens_real = int(getattr(self, "txt_max_tokens_real", txt_max_tokens) or txt_max_tokens)
        pdf_target_tokens_real = int(getattr(self, "pdf_target_tokens_real", pdf_target_tokens) or pdf_target_tokens)
        pdf_max_tokens_real = int(getattr(self, "pdf_max_tokens_real", pdf_max_tokens) or pdf_max_tokens)
        hwpx_target_tokens_real = int(getattr(self, "hwpx_target_tokens_real", hwpx_target_tokens) or hwpx_target_tokens)
        hwpx_max_tokens_real = int(getattr(self, "hwpx_max_tokens_real", hwpx_max_tokens) or hwpx_max_tokens)
        used_cache = False
        uploaded_at = int(time.time())
        version = file_hash[:12]
        row_items: Optional[List[Dict[str, Any]]] = None
        pdf_ingest_stats: Dict[str, Any] = {}
        parser_name = "txt_plain"
        source_entries: List[Dict[str, str]] = [
            {
                "source_path": source_path,
                "source_label": source_label,
                "source_type": source_type,
                "source_role": inferred_doc_role,
            }
        ]

        high_quality_pdf_ocr = str(pdf_ocr_mode or "").strip().lower().replace("-", "_") in {
            "high_quality",
            "highquality",
            "quality",
            "vl",
            "vl_only",
        }
        if source_type == "pdf" and (force_pdf_ocr or high_quality_pdf_ocr):
            cached = None
        else:
            with self._engine_lock:
                cached = self._load_cached_payload(source_path=source_path, file_hash=file_hash)
        if cached:
            items, _emb_large, cached_payload_meta = cached
            used_cache = True
            _progress(36, "이전에 읽은 결과를 불러왔습니다. 저장 상태를 갱신하는 중입니다.", "cache_hit")
            print(f"RAGEngine: cache hit for {source_path} ({len(items)} chunks)")
            if source_type == "pdf":
                pdf_ingest_stats = _normalize_pdf_ingest_stats(cached_payload_meta, fallback_parser="paddleocr_vl")
                parser_name = str(pdf_ingest_stats.get("pdf_parser", "") or parser_name)
            if ext == ".xlsx":
                # Keep canonical table_cells populated even on cache hits.
                try:
                    row_items = load_xlsx(
                        file_path,
                        merged_cell_policy=self.xlsx_merged_cell_policy,
                        comment_policy=self.xlsx_comment_policy,
                    )
                except Exception:
                    row_items = None
            entry_map: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
            for item in items:
                item_source = (item.get("source_path", "") or source_path).strip() or source_path
                if item_source not in entry_map:
                    item_label = source_label
                    m = re.search(r"::part:(\d+)", item_source)
                    if m:
                        item_label = f"{source_label} [{int(m.group(1)):03d}]"
                    entry_map[item_source] = {
                        "source_path": item_source,
                        "source_label": item_label,
                        "source_type": source_type,
                        "source_role": inferred_doc_role,
                    }
            source_entries = list(entry_map.values()) or source_entries
        else:
            if ext == ".txt":
                _progress(30, "텍스트 문서를 읽는 중입니다.", "read_text")
                raw_lines = load_txt(file_path)
                split_parts: List[Dict[str, Any]] = []
                if self.txt_split_enabled and len(raw_lines) >= self.txt_split_trigger_lines:
                    _progress(42, "문단과 섹션을 나누는 중입니다.", "split_text")
                    docs = self._split_txt_lines_for_ingest(raw_lines)
                    split_parts = self._materialize_split_txt_docs(
                        source_path=source_path,
                        source_label=source_label,
                        docs=docs,
                    )

                if split_parts:
                    items = []
                    source_entries = []
                    _progress(54, "텍스트 조각을 정리하는 중입니다.", "chunk_text")
                    for part in split_parts:
                        part_items = self._split_lines_for_real_token_budget(
                            part["lines"],
                            target_tokens=txt_target_tokens_real,
                            min_tokens=self.txt_min_tokens,
                            max_tokens=txt_max_tokens_real,
                        )
                        if not part_items:
                            continue
                        source_entries.append(
                            {
                                "source_path": part["source_path"],
                                "source_label": part["source_label"],
                                "source_type": source_type,
                                "source_role": inferred_doc_role,
                            }
                        )
                        for p_item in part_items:
                            p_item["source_path"] = part["source_path"]
                        items.extend(part_items)
                else:
                    _progress(54, "텍스트 조각을 정리하는 중입니다.", "chunk_text")
                    items = self._split_lines_for_real_token_budget(
                        raw_lines,
                        target_tokens=txt_target_tokens_real,
                        min_tokens=self.txt_min_tokens,
                        max_tokens=txt_max_tokens_real,
                    )
                    for item in items:
                        item["source_path"] = source_path
                parser_name = "txt_plain"
            elif ext == ".xlsx":
                _progress(30, "엑셀 문서를 읽는 중입니다.", "read_sheet")
                row_items = load_xlsx(
                    file_path,
                    merged_cell_policy=self.xlsx_merged_cell_policy,
                    comment_policy=self.xlsx_comment_policy,
                    structure_v2=bool(self.structure_rag_v2_enabled and self.xlsx_structure_rag_v2_enabled),
                )
                _progress(48, "표 내용을 검색용 조각으로 정리하는 중입니다.", "chunk_sheet")
                if self.structure_rag_v2_enabled and self.xlsx_structure_rag_v2_enabled:
                    structure_rows: List[Dict[str, Any]] = []
                    for row_item in row_items:
                        structure_rows.append(row_item)
                        fact_text = str(row_item.get("fact_text", "") or "").strip()
                        if fact_text:
                            structure_rows.append(
                                {
                                    **row_item,
                                    "text": fact_text,
                                    "chunk_kind": "table_summary",
                                    "is_derived": True,
                                }
                            )
                    items = chunk_structure_records(
                        structure_rows,
                        source_type="xlsx",
                        doc_role=inferred_doc_role,
                        source_path=source_path,
                    )
                else:
                    items = chunk_xlsx_rows(
                        row_items,
                        group_min_rows=self.xlsx_group_min_rows,
                        group_max_rows=self.xlsx_group_max_rows,
                        overlap_rows=self.xlsx_overlap_rows,
                        target_tokens=self.xlsx_target_tokens,
                        max_tokens=self.xlsx_max_tokens,
                    )
                for item in items:
                    item["source_path"] = source_path
                parser_name = "xlsx_parser"
            elif ext == ".hwpx":
                if not self.hwpx_extract_enabled:
                    return {"status": "unsupported", "source_path": source_path, "used_cache": False}
                _progress(30, "한글 HWPX 문서 구조를 읽는 중입니다.", "read_hwpx")
                raw_lines = load_hwpx_records(
                    file_path,
                    include_tables=self.hwpx_include_tables,
                    structure_v2=bool(self.structure_rag_v2_enabled and self.hwpx_structure_rag_v2_enabled),
                )
                _progress(54, "HWPX 문단과 표 내용을 검색용 조각으로 정리하는 중입니다.", "chunk_hwpx")
                if self.structure_rag_v2_enabled and self.hwpx_structure_rag_v2_enabled:
                    items = chunk_structure_records(
                        raw_lines,
                        source_type="hwpx",
                        doc_role=inferred_doc_role,
                        source_path=source_path,
                    )
                else:
                    items = self._split_lines_for_real_token_budget(
                        raw_lines,
                        target_tokens=hwpx_target_tokens_real,
                        min_tokens=self.hwpx_min_tokens,
                        max_tokens=hwpx_max_tokens_real,
                    )
                for item in items:
                    item["source_path"] = source_path
                parser_name = "python_hwpx"
            elif ext == ".pdf":
                _progress(28, "PDF 문서 구조를 확인하는 중입니다.", "inspect_pdf")
                ocr_started = time.perf_counter()
                worker_release_seconds = 0.0
                worker_released = False
                worker_pids: List[int] = []
                worker_alive_after_shutdown: List[int] = []
                worker_shutdown_confirmed = True
                try:
                    pdf_result = extract_pdf_pages(
                        file_path,
                        progress_callback=progress_callback,
                        force_upload_ocr=bool(force_pdf_ocr),
                        pdf_ocr_mode=pdf_ocr_mode,
                    )
                finally:
                    phase_timings["ocr_duration_seconds"] = max(0.0, time.perf_counter() - ocr_started)
                    release_cached_ocr_model()
                release_started = time.perf_counter()
                _progress(71, "PDF OCR worker를 종료하고 GPU 메모리를 회수하는 중입니다.", "release_pdf_ocr_worker")
                try:
                    shutdown_info = shutdown_persistent_ocr_worker() or {}
                    for raw_pid in list(shutdown_info.get("worker_pids", []) or []):
                        try:
                            pid = int(raw_pid)
                        except (TypeError, ValueError):
                            continue
                        if pid > 0:
                            worker_pids.append(pid)
                    for raw_pid in list(shutdown_info.get("alive_after_shutdown", []) or []):
                        try:
                            pid = int(raw_pid)
                        except (TypeError, ValueError):
                            continue
                        if pid > 0:
                            worker_alive_after_shutdown.append(pid)
                    worker_shutdown_confirmed = bool(shutdown_info.get("shutdown_confirmed", True))
                    worker_released = True
                finally:
                    worker_release_seconds = max(0.0, time.perf_counter() - release_started)
                worker_pids = sorted(set(worker_pids))
                worker_alive_after_shutdown = sorted(set(worker_alive_after_shutdown))
                if not worker_shutdown_confirmed:
                    print(
                        "[UPLOAD][WARN] stage=release_pdf_ocr_worker "
                        f"source_path={source_path} worker_pids={worker_pids} "
                        f"alive_after_shutdown={worker_alive_after_shutdown}",
                        file=sys.stderr,
                    )
                _progress(
                    72,
                    "PDF OCR worker release completed.",
                    "release_pdf_ocr_worker",
                    ocr_worker_released=worker_released,
                    ocr_worker_release_seconds=round(worker_release_seconds, 3),
                    ocr_worker_pids=worker_pids,
                    ocr_worker_shutdown_confirmed=worker_shutdown_confirmed,
                    ocr_worker_alive_after_shutdown=worker_alive_after_shutdown,
                )
                page_records = list(pdf_result.get("pages", []) or [])
                parser_name = str(pdf_result.get("parser", "") or "paddleocr_vl")
                pdf_ingest_stats = {
                    "pdf_parser": parser_name,
                    "pdf_total_pages": int(pdf_result.get("total_pages", len(page_records)) or len(page_records)),
                    "pdf_text_pages": int(pdf_result.get("text_pages", 0) or 0),
                    "pdf_ocr_pages": int(pdf_result.get("ocr_pages", 0) or 0),
                    "pdf_attempted_ocr_pages": int(pdf_result.get("attempted_ocr_pages", 0) or 0),
                    "pdf_failed_pages": int(pdf_result.get("failed_pages", 0) or 0),
                    "pdf_table_like_pages": sum(1 for page in page_records if bool(page.get("table_like", False))),
                    "pdf_warnings": list(pdf_result.get("warnings", []) or []),
                    "ocr_device_attempted": str(pdf_result.get("ocr_device_attempted", "") or "").strip(),
                    "ocr_device_effective": str(pdf_result.get("ocr_device_effective", "") or "").strip(),
                    "ocr_gpu_fallback_used": bool(pdf_result.get("ocr_gpu_fallback_used", False)),
                    "ocr_gpu_failure_reason": str(pdf_result.get("ocr_gpu_failure_reason", "") or "").strip(),
                    "ocr_elapsed_seconds": pdf_result.get("ocr_elapsed_seconds"),
                    "ocr_pages_processed": pdf_result.get("ocr_pages_processed"),
                    "ocr_pages_per_minute": pdf_result.get("ocr_pages_per_minute"),
                    "ocr_pages_attempted": pdf_result.get("ocr_pages_attempted"),
                    "ocr_pages_emitted": pdf_result.get("ocr_pages_emitted"),
                    "ocr_pages_skipped_empty": pdf_result.get("ocr_pages_skipped_empty"),
                    "ocr_pages_skipped_short_text": pdf_result.get("ocr_pages_skipped_short_text"),
                    "ocr_attempted_pages_per_minute": pdf_result.get("ocr_attempted_pages_per_minute"),
                    "ocr_emitted_pages_per_minute": pdf_result.get("ocr_emitted_pages_per_minute"),
                    "ocr_worker_released": worker_released,
                    "ocr_worker_release_seconds": round(worker_release_seconds, 3),
                    "ocr_worker_pids": worker_pids,
                    "ocr_worker_shutdown_confirmed": worker_shutdown_confirmed,
                    "ocr_worker_alive_after_shutdown": worker_alive_after_shutdown,
                    "ocr_target_pages": pdf_result.get("ocr_target_pages"),
                    "ocr_target_seconds": pdf_result.get("ocr_target_seconds"),
                    "ocr_target_met": pdf_result.get("ocr_target_met"),
                    "ocr_subset_build_seconds": pdf_result.get("ocr_subset_build_seconds"),
                    "ocr_model_load_seconds": pdf_result.get("ocr_model_load_seconds"),
                    "ocr_predict_seconds": pdf_result.get("ocr_predict_seconds"),
                    "ocr_output_materialize_seconds": pdf_result.get("ocr_output_materialize_seconds"),
                    "ocr_payload_convert_seconds": pdf_result.get("ocr_payload_convert_seconds"),
                    "ocr_fragment_collect_seconds": pdf_result.get("ocr_fragment_collect_seconds"),
                    "ocr_page_dedupe_seconds": pdf_result.get("ocr_page_dedupe_seconds"),
                    "ocr_page_join_seconds": pdf_result.get("ocr_page_join_seconds"),
                    "ocr_text_merge_seconds": pdf_result.get("ocr_text_merge_seconds"),
                    "ocr_merge_seconds": pdf_result.get("ocr_merge_seconds"),
                    "ocr_batch_count": pdf_result.get("ocr_batch_count"),
                    "ocr_backend": pdf_result.get("ocr_backend"),
                    "ocr_backend_attempted": pdf_result.get("ocr_backend_attempted"),
                    "ocr_backend_effective": pdf_result.get("ocr_backend_effective"),
                    "ocr_backend_fallback_used": pdf_result.get("ocr_backend_fallback_used"),
                    "ocr_fast_pages": pdf_result.get("ocr_fast_pages"),
                    "ocr_vl_pages": pdf_result.get("ocr_vl_pages"),
                    "ocr_fast_seconds": pdf_result.get("ocr_fast_seconds"),
                    "ocr_vl_seconds": pdf_result.get("ocr_vl_seconds"),
                    "ocr_fast_avg_score": pdf_result.get("ocr_fast_avg_score"),
                    "ocr_fast_pair_ratio": pdf_result.get("ocr_fast_pair_ratio"),
                    "ocr_fast_orphan_ratio": pdf_result.get("ocr_fast_orphan_ratio"),
                    "ocr_high_quality_requested": pdf_result.get("ocr_high_quality_requested"),
                }
                items = []
                page_count = max(1, len(page_records))
                compaction_totals = {
                    "chars_before_compaction": 0,
                    "chars_after_compaction": 0,
                    "lines_before_dedupe": 0,
                    "lines_after_dedupe": 0,
                    "hints_dropped": 0,
                }
                embedding_compaction_max_chunk_chars = max(
                    400,
                    int(getattr(self, "embedding_compaction_max_chunk_chars", 1800) or 1800),
                )
                for page_index, page in enumerate(page_records, start=1):
                    page_no = max(1, int(page.get("page_no", 0) or 0))
                    total_pdf_pages = max(
                        page_count,
                        int(pdf_ingest_stats.get("pdf_total_pages", page_count) or page_count),
                    )
                    progress_page_no = min(max(0, page_no), total_pdf_pages)
                    page_progress = 58 + int(round((page_index / page_count) * 14))
                    _progress(
                        page_progress,
                        f"PDF 페이지 내용을 검색용 조각으로 정리하는 중입니다. ({progress_page_no}/{total_pdf_pages})",
                        "prepare_pdf_chunks",
                    )
                    page_text = str(page.get("text", "") or "").strip()
                    page_parser = str(page.get("parser", "") or parser_name).strip() or parser_name
                    table_hint_lines = [
                        re.sub(r"\s+", " ", (hint or "").strip())
                        for hint in list(page.get("table_hints", []) or [])
                        if re.sub(r"\s+", " ", (hint or "").strip())
                    ]
                    lazy_ocr_hint_lines = [
                        re.sub(r"\s+", " ", (hint or "").strip())
                        for hint in list(page.get("lazy_ocr_hints", []) or [])
                        if re.sub(r"\s+", " ", (hint or "").strip())
                    ]
                    if not page_text and not table_hint_lines and not lazy_ocr_hint_lines:
                        continue

                    compact_lines, compact_stats = self._compact_pdf_embedding_payload(
                        page_text,
                        table_hint_lines,
                        lazy_ocr_hint_lines,
                    )
                    for stat_key, stat_value in compact_stats.items():
                        compaction_totals[stat_key] += int(stat_value or 0)
                    if not compact_lines:
                        continue

                    page_lines: List[Dict[str, Any]] = []
                    for line_idx, normalized_line in enumerate(compact_lines, start=1):
                        page_lines.append(
                            {
                                "text": normalized_line,
                                "line_start": line_idx,
                                "line_end": line_idx,
                                "file_path": source_label,
                                "is_section": False,
                                }
                            )
                    if not page_lines:
                        continue

                    page_items = self._split_lines_for_real_token_budget(
                        page_lines,
                        target_tokens=pdf_target_tokens_real,
                        min_tokens=self.pdf_min_tokens,
                        max_tokens=pdf_max_tokens_real,
                    )
                    for chunk in page_items:
                        chunk["source_path"] = source_path
                        chunk["section"] = f"PDF page {page_no}"
                        chunk["page_no"] = page_no
                        chunk["page_parser"] = page_parser
                        chunk["parser_name"] = parser_name
                        chunk["embedding_text"] = build_embedding_text(
                            text=str(chunk.get("text", "") or "")[:embedding_compaction_max_chunk_chars],
                            source_path=source_label,
                            doc_role=inferred_doc_role,
                            heading_path=[f"PDF page {page_no}"],
                            chunk_kind="body",
                        )
                    items.extend(page_items)
                pdf_ingest_stats.update(compaction_totals)
                pdf_ingest_stats["chunks_created"] = int(len(items))

            if not items:
                return {"status": "empty", "source_path": source_path, "used_cache": False}

            if bool(getattr(self, "structure_rag_v2_enabled", False)) and source_type in {"hwpx", "xlsx"}:
                for item in items:
                    item["source_path"] = source_path
                    item["embedding_text"] = build_embedding_text(
                        text=str(item.get("text", "") or ""),
                        source_path=source_label,
                        doc_role=inferred_doc_role,
                        heading_path=item.get("heading_path", []),
                        chunk_kind=str(item.get("chunk_kind", "body") or "body"),
                    )

            _progress(72, "추출한 내용을 캐시에 저장하는 중입니다.", "store_ocr_cache")
            with self._engine_lock:
                self._save_cached_payload(
                    source_path=source_path,
                    file_hash=file_hash,
                    items=items,
                    meta=pdf_ingest_stats if source_type == "pdf" else None,
                )
            print(f"RAGEngine: cache stored for {source_path}")

        if source_type == "pdf":
            if not pdf_ingest_stats:
                pdf_ingest_stats = _summarize_pdf_chunk_pages(items, fallback_parser="paddleocr_vl")
                parser_name = str(pdf_ingest_stats.get("pdf_parser", "") or "paddleocr_vl")
            else:
                cached_stats = _summarize_pdf_chunk_pages(items, fallback_parser=parser_name)
                if not parser_name or parser_name == "txt_plain":
                    parser_name = str(
                        pdf_ingest_stats.get("pdf_parser", "")
                        or cached_stats.get("pdf_parser", "")
                        or "paddleocr_vl"
                    )
                if int(pdf_ingest_stats.get("pdf_total_pages", 0) or 0) <= 0:
                    pdf_ingest_stats = cached_stats
                else:
                    pdf_ingest_stats["pdf_parser"] = parser_name
                    if int(pdf_ingest_stats.get("pdf_text_pages", 0) or 0) == 0 and int(cached_stats.get("pdf_text_pages", 0) or 0) > 0:
                        pdf_ingest_stats["pdf_text_pages"] = int(cached_stats.get("pdf_text_pages", 0) or 0)
                    if int(pdf_ingest_stats.get("pdf_ocr_pages", 0) or 0) == 0 and int(cached_stats.get("pdf_ocr_pages", 0) or 0) > 0:
                        pdf_ingest_stats["pdf_ocr_pages"] = int(cached_stats.get("pdf_ocr_pages", 0) or 0)
                    if int(pdf_ingest_stats.get("pdf_total_pages", 0) or 0) == 0:
                        pdf_ingest_stats["pdf_total_pages"] = int(cached_stats.get("pdf_total_pages", 0) or 0)
        elif source_type == "xlsx":
            parser_name = "xlsx_parser"
        elif source_type == "hwpx":
            parser_name = "python_hwpx"
        else:
            parser_name = "txt_plain"


        return {
            "status": "prepared",
            "file_path": file_path,
            "source_path": source_path,
            "file_hash": file_hash,
            "source_type": source_type,
            "source_label": source_label,
            "inferred_doc_role": inferred_doc_role,
            "uploaded_at": uploaded_at,
            "version": version,
            "items": items,
            "row_items": row_items,
            "pdf_ingest_stats": pdf_ingest_stats,
            "parser_name": parser_name,
            "source_entries": source_entries,
            "previous_source_records": previous_source_records,
            "used_cache": used_cache,
            "phase_timings": phase_timings,
        }

    def commit_prepared_ingest(
        self,
        prepared: Dict[str, Any],
        *,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """Persist prepared chunks and refresh search artifacts."""
        with self._engine_lock:
            if not isinstance(prepared, dict) or prepared.get("status") != "prepared":
                return prepared

            def _progress(percent: int, message: str, stage: str, **progress_meta: Any) -> None:
                self._emit_ingest_progress(progress_callback, percent, message, stage, **progress_meta)

            file_path = str(prepared.get("file_path", "") or "")
            source_path = str(prepared.get("source_path", "") or "")
            file_hash = str(prepared.get("file_hash", "") or "")
            source_type = str(prepared.get("source_type", "") or "")
            source_label = str(prepared.get("source_label", "") or source_path)
            inferred_doc_role = str(prepared.get("inferred_doc_role", DOC_ROLE_UNKNOWN) or DOC_ROLE_UNKNOWN)
            uploaded_at = int(prepared.get("uploaded_at", int(time.time())) or int(time.time()))
            version = str(prepared.get("version", file_hash[:12]) or file_hash[:12])
            items = list(prepared.get("items", []) or [])
            row_items = prepared.get("row_items")
            pdf_ingest_stats = dict(prepared.get("pdf_ingest_stats", {}) or {})
            parser_name = str(prepared.get("parser_name", "") or "txt_plain")
            source_entries = list(prepared.get("source_entries", []) or [])
            previous_source_records = list(prepared.get("previous_source_records", []) or [])
            used_cache = bool(prepared.get("used_cache", False))
            phase_timings = dict(prepared.get("phase_timings", {}) or {})
            phase_timings.setdefault("ocr_duration_seconds", 0.0)
            phase_timings.setdefault("persist_duration_seconds", 0.0)
            phase_timings.setdefault("embedding_duration_seconds", 0.0)
            phase_timings.setdefault("index_duration_seconds", 0.0)
            phase_timings.setdefault("derived_sync_duration_seconds", 0.0)
            if not source_entries:
                source_entries = [
                    {
                        "source_path": source_path,
                        "source_label": source_label,
                        "source_type": source_type,
                        "source_role": inferred_doc_role,
                    }
                ]

            def _phase_log(stage: str, status: str, **fields: Any) -> None:
                field_parts = [f"source_path={source_path or '-'}"]
                for key, value in fields.items():
                    field_parts.append(f"{key}={value}")
                print(f"[UPLOAD][PHASE] stage={stage} status={status} " + " ".join(field_parts), flush=True)

            def _phase_progress(percent: int, message: str, stage: str, **progress_meta: Any) -> None:
                meta = dict(progress_meta or {})
                meta.setdefault("phase_name_effective", stage)
                _progress(percent, message, stage, **meta)

            def _run_with_phase_heartbeat(
                fn: Callable[[], Any],
                *,
                percent: int,
                message: str,
                stage: str,
                interval_seconds: float = 25.0,
                **progress_meta: Any,
            ) -> Any:
                stop_event = threading.Event()
                meta = dict(progress_meta or {})

                def _heartbeat() -> None:
                    while not stop_event.wait(interval_seconds):
                        _phase_progress(percent, message, stage, **meta)

                heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
                _phase_progress(percent, message, stage, **meta)
                heartbeat_thread.start()
                try:
                    return fn()
                finally:
                    stop_event.set()
                    heartbeat_thread.join(timeout=0.2)

            persist_started = time.perf_counter()
            _phase_log("persist_meta", "start", rows=len(items))
            _phase_progress(
                78,
                "문서 메타데이터를 정리하는 중입니다.",
                "persist_meta",
                phase_rows_total=len(items),
                phase_rows_done=0,
            )
            superseded_sources = sorted(
                {
                    str(record.get("source_path", "") or "").strip()
                    for record in previous_source_records
                    if str(record.get("source_path", "") or "").strip()
                    and str(record.get("source_path", "") or "").strip() != source_path
                }
            )
            superseded_cleanup: Dict[str, Any] = {
                "chunk_ids": [],
                "cache_paths": [],
                "stored_paths": [],
            }
            if superseded_sources:
                cleanup_conn = self._connect_db()
                try:
                    cleanup_conn.execute("BEGIN IMMEDIATE")
                    superseded_cleanup = _delete_source_records(
                        cleanup_conn,
                        superseded_sources,
                        delete_normalized_groups=False,
                    )
                    cleanup_conn.commit()
                except Exception:
                    cleanup_conn.rollback()
                    raise
                finally:
                    cleanup_conn.close()
            file_id = self._upsert_file_record(
                source_path=source_path,
                file_hash=file_hash,
                file_path=file_path,
                source_label=source_label,
                source_type=source_type,
            )
            doc_id = self._upsert_document_record(
                file_id=file_id,
                source_path=source_path,
                source_type=source_type,
                parser_name=parser_name,
                parser_version=self._parser_signature(),
            )
            self._replace_canonical_rows(
                doc_id=doc_id,
                source_type=source_type,
                items=items,
                row_items=row_items,
            )

            deleted_count, deleted_chunk_ids = self._delete_chunks_for_source(source_path)
            superseded_chunk_ids = [
                int(value)
                for value in list(superseded_cleanup.get("chunk_ids", []) or [])
                if int(value or 0) > 0
            ]
            deleted_chunk_ids = sorted(set([*deleted_chunk_ids, *superseded_chunk_ids]))
            deleted_count += len(superseded_chunk_ids)
            self._delete_source_upload_meta_for_source(source_path)
            for entry in source_entries:
                self._upsert_source_upload_meta(
                    source_path=(entry.get("source_path", "") or source_path),
                    source_type=(entry.get("source_type", "") or source_type),
                    doc_role=(entry.get("source_role", "") or inferred_doc_role),
                    file_hash=file_hash,
                    doc_version=version,
                    uploaded_at=uploaded_at,
                    original_filename=(entry.get("source_label", "") or source_label),
                )

            conn = self._connect_db()
            c = conn.cursor()
            new_ids: List[int] = []
            ontology_chunk_ids: List[int] = []
            source_role_map = {
                (entry.get("source_path", "") or source_path): self._normalize_doc_role(
                    entry.get("source_role", "") or inferred_doc_role
                )
                for entry in source_entries
            }

            _phase_log("store_chunks", "start", rows=len(items))
            insert_step = max(1, len(items) // 4) if items else 1
            for idx, item in enumerate(items):
                if idx == 0 or idx + 1 == len(items) or ((idx + 1) % insert_step) == 0:
                    insert_percent = 84 + int(round(((idx + 1) / max(1, len(items))) * 8))
                    _phase_progress(
                        insert_percent,
                        f"검색 조각을 저장하는 중입니다. ({idx + 1}/{len(items)})",
                        "store_chunks",
                        phase_rows_total=len(items),
                        phase_rows_done=idx + 1,
                        phase_chunks_total=len(items),
                        phase_chunks_done=idx + 1,
                    )
                item_source_path = (item.get("source_path", "") or source_path).strip() or source_path
                item_doc_role = source_role_map.get(item_source_path, inferred_doc_role)
                chunk_id = f"{item_source_path}:{version}:{idx + 1:06d}"
                c.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, kb_id, source_path, source_type, doc_role, sheet, row, row_end,
                         page_no, line_start, line_end, section, doc_version, text, embedding_text,
                         chunk_kind, heading_path_json, parent_chunk_key, structure_path, table_id,
                         row_no, cell_no, is_derived, is_normalized, normalized_group, source_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        self.kb_id,
                        item_source_path,
                        source_type,
                        self._normalize_doc_role(item_doc_role),
                        item.get("sheet", "") or "",
                        int(item.get("row", 0) or 0),
                        int(item.get("row_end", item.get("row", 0)) or 0),
                        int(item.get("page_no", 0) or 0),
                        int(item.get("line_start", 0) or 0),
                        int(item.get("line_end", 0) or 0),
                        item.get("section", "") or "",
                        version,
                        item.get("text", "") or "",
                        item.get("embedding_text", "") or "",
                        item.get("chunk_kind", "") or "",
                        json.dumps(item.get("heading_path", []) or [], ensure_ascii=False),
                        item.get("parent_chunk_key", "") or "",
                        item.get("structure_path", "") or "",
                        item.get("table_id", "") or "",
                        int(item.get("row_no", item.get("row", 0)) or 0),
                        int(item.get("cell_no", 0) or 0),
                        1 if bool(item.get("is_derived", False)) else 0,
                        0,
                        "",
                        uploaded_at,
                    ),
                )
                new_ids.append(int(c.lastrowid))
                if not bool(item.get("is_derived", False)):
                    ontology_chunk_ids.append(int(c.lastrowid))

            conn.commit()
            conn.close()
            _phase_log("store_chunks", "done", rows=len(new_ids))
            phase_timings["persist_duration_seconds"] = max(0.0, time.perf_counter() - persist_started)
            _phase_log(
                "persist_meta",
                "done",
                rows=len(items),
                phase_duration_seconds=f"{phase_timings['persist_duration_seconds']:.3f}",
                elapsed_seconds=f"{phase_timings['persist_duration_seconds']:.3f}",
            )

            index_started = time.perf_counter()
            _phase_log("refresh_index", "start", rows=len(items))
            refresh_chunks_total = max(len(items), len(new_ids) + len(deleted_chunk_ids))
            normalized_refresh = _run_with_phase_heartbeat(
                lambda: self._refresh_normalized_chunks_and_index(
                    affected_groups=self._normalized_groups_for_source_type(source_type),
                ),
                percent=94,
                message="통합 조각과 검색 인덱스를 갱신하는 중입니다.",
                stage="refresh_index",
                phase_chunks_total=refresh_chunks_total,
                phase_chunks_done=0,
            )
            changed_normalized_ids = list(normalized_refresh.get("inserted_chunk_ids", []) or [])
            deleted_normalized_ids = list(normalized_refresh.get("deleted_chunk_ids", []) or [])
            phase_timings["index_duration_seconds"] = max(0.0, time.perf_counter() - index_started)
            _phase_log(
                "refresh_index",
                "done",
                rows=len(items),
                changed_chunks=len(changed_normalized_ids),
                deleted_chunks=len(deleted_normalized_ids),
                phase_duration_seconds=f"{phase_timings['index_duration_seconds']:.3f}",
                elapsed_seconds=f"{phase_timings['index_duration_seconds']:.3f}",
            )

            embedding_started = time.perf_counter()
            embed_chunk_ids = [*new_ids, *changed_normalized_ids]
            deleted_embed_chunk_ids = [*deleted_chunk_ids, *deleted_normalized_ids]
            total_embed_rows = len(embed_chunk_ids)
            _phase_log("embed_chunks", "start", rows=total_embed_rows)

            def _emit_embed_progress(
                batch_index: int,
                total_batches: int,
                rows_done: int,
                **embed_meta: Any,
            ) -> None:
                _phase_progress(
                    96,
                    (
                        "임베딩과 벡터 저장을 진행하는 중입니다. "
                        f"(batch {batch_index}/{max(1, total_batches)}, rows {rows_done}/{total_embed_rows})"
                    ),
                    "embed_chunks",
                    phase_rows_total=total_embed_rows,
                    phase_rows_done=rows_done,
                    phase_chunks_total=total_embed_rows,
                    phase_chunks_done=rows_done,
                    embed_batch=batch_index,
                    embed_batches=total_batches,
                    embed_rows_done=rows_done,
                    embed_rows_total=total_embed_rows,
                    **dict(embed_meta or {}),
                )

            _run_with_phase_heartbeat(
                lambda: self._sync_sqlite_search_artifacts(
                    changed_chunk_ids=embed_chunk_ids,
                    deleted_chunk_ids=deleted_embed_chunk_ids,
                    index_name="large",
                    vector_progress_callback=_emit_embed_progress,
                    vector_log_context={"source_path": source_path},
                ),
                percent=96,
                message="임베딩과 벡터 저장을 준비하는 중입니다.",
                stage="embed_chunks",
                phase_rows_total=total_embed_rows,
                phase_rows_done=0,
                phase_chunks_total=total_embed_rows,
                phase_chunks_done=0,
                embed_batch=0,
                embed_batches=0,
                embed_rows_done=0,
                embed_rows_total=total_embed_rows,
                embed_input_tokens_total=0,
                embed_input_tokens_done=0,
                embed_input_tokens_p95=0,
                embed_input_tokens_max=0,
                embed_truncated_rows=0,
                embed_effective_batch_tokens=0,
            )
            phase_timings["embedding_duration_seconds"] = max(0.0, time.perf_counter() - embedding_started)
            _phase_log(
                "embed_chunks",
                "done",
                rows=total_embed_rows,
                phase_duration_seconds=f"{phase_timings['embedding_duration_seconds']:.3f}",
                elapsed_seconds=f"{phase_timings['embedding_duration_seconds']:.3f}",
            )

            derived_started = time.perf_counter()
            _phase_log("sync_derived", "start", rows=len(new_ids))

            indexable_count_before_sync = self._count_indexable_chunks()
            if self.hnsw_enabled:
                subphase_started = time.perf_counter()
                _phase_log("sync_derived", "start", subphase="rebuild_index", rows=indexable_count_before_sync)
                _run_with_phase_heartbeat(
                    lambda: self._rebuild_index_from_db(chunk_count=indexable_count_before_sync),
                    percent=98,
                    message="파생 검색 구조를 동기화하는 중입니다. (rebuild_index)",
                    stage="sync_derived",
                    phase_name_effective="sync_derived_rebuild_index",
                    phase_chunks_total=indexable_count_before_sync,
                    phase_chunks_done=0,
                )
                _phase_log(
                    "sync_derived",
                    "done",
                    subphase="rebuild_index",
                    phase_duration_seconds=f"{max(0.0, time.perf_counter() - subphase_started):.3f}",
                )

            concept_subphase_started = time.perf_counter()
            _phase_log("sync_derived_concept_embedding", "start", subphase="concept_embedding", rows=len(new_ids))
            concept_sync = _run_with_phase_heartbeat(
                lambda: self._sync_concept_links(
                    changed_chunk_ids=new_ids,
                    deleted_chunk_ids=deleted_chunk_ids,
                ),
                percent=98,
                message="파생 검색 구조를 동기화하는 중입니다. (concept_embedding)",
                stage="sync_derived",
                phase_name_effective="sync_derived_concept_embedding",
                phase_chunks_total=len(new_ids),
                phase_chunks_done=0,
            )
            _phase_log(
                "sync_derived_concept_embedding",
                "done",
                subphase="concept_embedding",
                concept_links_added=int(concept_sync.get("concept_links_added", 0) or 0),
                phase_duration_seconds=f"{max(0.0, time.perf_counter() - concept_subphase_started):.3f}",
            )

            ontology_sync = {"ontology_facts_added": 0, "ontology_facts_deleted": 0}
            if DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED and getattr(self, "db_path", None):
                ontology_subphase_started = time.perf_counter()
                _phase_log("sync_derived", "start", subphase="ontology_facts", rows=len(new_ids))
                ontology_sync = _run_with_phase_heartbeat(
                    lambda: self._sync_ontology_facts(
                        changed_chunk_ids=new_ids,
                        deleted_chunk_ids=deleted_chunk_ids,
                    ),
                    percent=98,
                    message="파생 검색 구조를 동기화하는 중입니다. (ontology_facts)",
                    stage="sync_derived",
                    phase_name_effective="sync_derived_ontology_facts",
                    phase_chunks_total=len(new_ids),
                    phase_chunks_done=0,
                )
                _phase_log(
                    "sync_derived",
                    "done",
                    subphase="ontology_facts",
                    ontology_facts_added=int(ontology_sync.get("ontology_facts_added", 0) or 0),
                    phase_duration_seconds=f"{max(0.0, time.perf_counter() - ontology_subphase_started):.3f}",
                )

            wiki_compile_status = "skipped"
            if _env_bool("WIKI_PAGE_WORKFLOW_ENABLED", False):
                wiki_subphase_started = time.perf_counter()
                _phase_log("sync_derived", "start", subphase="compile_wiki", rows=1)

                def _compile_wiki() -> str:
                    WikiStore(self.db_path).compile_source_page(source_path, space_id=self.kb_id)
                    return "ok"

                try:
                    wiki_compile_status = _run_with_phase_heartbeat(
                        _compile_wiki,
                        percent=98,
                        message="파생 검색 구조를 동기화하는 중입니다. (compile_wiki)",
                        stage="sync_derived",
                        phase_name_effective="sync_derived_compile_wiki",
                        phase_rows_total=1,
                        phase_rows_done=0,
                    )
                except Exception as exc:
                    wiki_compile_status = "error"
                    print(f"[WIKI][WARN] source={source_path} compile_source_page failed: {exc}", file=sys.stderr)
                _phase_log(
                    "sync_derived",
                    "done",
                    subphase="compile_wiki",
                    status_detail=wiki_compile_status,
                    phase_duration_seconds=f"{max(0.0, time.perf_counter() - wiki_subphase_started):.3f}",
                )

            self.query_cache.clear()
            phase_timings["derived_sync_duration_seconds"] = max(0.0, time.perf_counter() - derived_started)
            _phase_log(
                "sync_derived",
                "done",
                rows=len(new_ids),
                concept_links_added=int(concept_sync.get("concept_links_added", 0) or 0),
                ontology_facts_added=int(ontology_sync.get("ontology_facts_added", 0) or 0),
                phase_duration_seconds=f"{phase_timings['derived_sync_duration_seconds']:.3f}",
                elapsed_seconds=f"{phase_timings['derived_sync_duration_seconds']:.3f}",
            )

            normalized_count = self._count_normalized_chunks()
            indexable_count = self._count_indexable_chunks()
            previous_stored_paths = {
                str(record.get("stored_path", "") or "").strip()
                for record in previous_source_records
                if str(record.get("stored_path", "") or "").strip()
            }
            previous_stored_paths.update(
                str(value or "").strip()
                for value in list(superseded_cleanup.get("stored_paths", []) or [])
                if str(value or "").strip()
            )
            current_stored_path = os.path.realpath(os.path.abspath(file_path)) if file_path else ""
            if previous_stored_paths:
                uploads_dir = os.path.join(self.data_dir, "uploads")
                for previous_path in sorted(previous_stored_paths):
                    previous_resolved = os.path.realpath(os.path.abspath(previous_path))
                    if current_stored_path and previous_resolved == current_stored_path:
                        continue
                    _remove_managed_file(previous_path, [uploads_dir])
            stale_cache_paths = list(superseded_cleanup.get("cache_paths", []) or [])
            if stale_cache_paths:
                for cache_path in stale_cache_paths:
                    _remove_managed_file(str(cache_path or ""), [self.cache_dir])
            _phase_log("done", "ok", rows=len(new_ids), chunks=len(new_ids))
            return {
                "status": "ok",
                "source_path": source_path,
                "chunks": len(new_ids),
                "ontology_chunk_ids": ontology_chunk_ids,
                "used_cache": used_cache,
                "replaced_chunks": deleted_count,
                "normalized_chunks": normalized_count,
                "indexable_chunks": indexable_count,
                "index_name": "large",
                "parser_signature": self._parser_signature(),
                "doc_role": inferred_doc_role,
                "wiki_compile_status": wiki_compile_status,
                **phase_timings,
                **concept_sync,
                **ontology_sync,
                **pdf_ingest_stats,
            }

    def ingest_file(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        document_role: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        force_pdf_ocr: bool = False,
        pdf_ocr_mode: str = "",
    ) -> Dict[str, Any]:
        """Ingest a file into the KB and build the large index."""
        prepared = self.prepare_ingest_payload(
            file_path,
            original_filename=original_filename,
            document_role=document_role,
            progress_callback=progress_callback,
            force_pdf_ocr=force_pdf_ocr,
            pdf_ocr_mode=pdf_ocr_mode,
        )
        if not isinstance(prepared, dict) or prepared.get("status") != "prepared":
            return prepared
        return self.commit_prepared_ingest(prepared, progress_callback=progress_callback)

    def _resolve_index(self, index_name: str) -> Tuple[Optional[hnswlib.Index], str]:
        key = self._normalize_index_name(index_name)
        return self.index_large, key

    def close(self):
        with self._engine_lock:
            self.query_cache.clear()
            self._last_concept_search_meta = {}
            self.index_large = None
            self.model_large = None

    def _fts_query_from_text(self, query: str) -> str:
        tokens = self._query_keywords(query)
        if not tokens:
            tokens = self._tokenize_for_overlap(query)
        terms = []
        seen = set()
        for tok in tokens:
            t = (tok or "").strip().lower()
            if len(t) < 2:
                continue
            if t in seen:
                continue
            seen.add(t)
            terms.append(t)
            if len(terms) >= 24:
                break
        if not terms:
            return ""
        escaped = ['"' + term.replace('"', '""') + '"' for term in terms]
        return " OR ".join(escaped)

    def _search_fts_candidates(self, query: str, candidate_limit: int) -> Dict[int, float]:
        if not getattr(self, "fts_available", False):
            return {}
        fts_query = self._fts_query_from_text(query)
        if not fts_query:
            return {}
        limit = max(8, int(candidate_limit))
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        try:
            c.execute(
                """
                SELECT rowid AS chunk_pk, bm25(chunk_fts) AS bm25_score
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (fts_query, limit),
            )
            rows = c.fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return {}
        conn.close()

        out: Dict[int, float] = {}
        for row in rows:
            chunk_pk = int(row["chunk_pk"])
            bm25_val = float(row["bm25_score"] or 0.0)
            if bm25_val < 0:
                bm25_val = 0.0
            out[chunk_pk] = 1.0 / (1.0 + bm25_val)
        return out

    def _search_dense_candidates_hnsw(
        self,
        index: Optional[hnswlib.Index],
        query_embedding: np.ndarray,
        candidate_k: int,
    ) -> Dict[int, float]:
        if (not self.hnsw_enabled) or index is None or index.element_count == 0:
            return {}
        k = min(max(1, int(candidate_k)), int(index.element_count))
        labels, distances = index.knn_query(query_embedding, k=k)
        out: Dict[int, float] = {}
        for label, dist in zip(labels[0], distances[0]):
            out[int(label)] = max(0.0, 1.0 - float(dist))
        return out

    def _search_dense_candidates_sqlite(
        self,
        index_name: str,
        query_vector: np.ndarray,
        candidate_k: int,
        candidate_ids: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        if not self.sqlite_dense_enabled:
            return {}
        k = max(1, int(candidate_k))
        conn = self._connect_db()
        c = conn.cursor()
        if candidate_ids:
            ordered_ids = [int(chunk_id) for chunk_id in candidate_ids if int(chunk_id) > 0]
            if not ordered_ids:
                conn.close()
                return {}
            placeholders = ",".join("?" for _ in ordered_ids)
            c.execute(
                f"""
                SELECT chunk_pk, dim, embedding
                FROM chunk_vec
                WHERE index_name = ?
                  AND chunk_pk IN ({placeholders})
                """,
                (index_name, *ordered_ids),
            )
        else:
            c.execute(
                "SELECT chunk_pk, dim, embedding FROM chunk_vec WHERE index_name = ?",
                (index_name,),
            )
        heap: List[Tuple[float, int]] = []
        expected_dim = int(query_vector.shape[0])

        for chunk_pk, dim, blob in c.fetchall():
            if blob is None:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            vec_dim = int(dim or 0)
            if vec_dim > 0 and vec.shape[0] != vec_dim:
                continue
            if vec.shape[0] != expected_dim:
                continue
            score = float(np.dot(query_vector, vec))
            if len(heap) < k:
                heapq.heappush(heap, (score, int(chunk_pk)))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, int(chunk_pk)))
        conn.close()

        if not heap:
            return {}
        heap.sort(key=lambda x: x[0], reverse=True)
        return {chunk_pk: float(score) for score, chunk_pk in heap}

    def _search_text_candidates_sqlite(
        self,
        query: str,
        candidate_limit: int,
    ) -> List[int]:
        tokens = self._query_keywords(query)
        if not tokens:
            tokens = self._tokenize_for_overlap(query)
        ordered_terms: List[str] = []
        seen_terms = set()
        for raw in tokens:
            term = (raw or "").strip().lower()
            if len(term) < 2 or term in seen_terms:
                continue
            seen_terms.add(term)
            ordered_terms.append(term)
            if len(ordered_terms) >= 6:
                break
        if not ordered_terms:
            return []

        predicates = ["instr(lower(text), ?) > 0" for _ in ordered_terms]
        params: List[Any] = [*ordered_terms]
        where_parts = [f"({' OR '.join(predicates)})"]
        if self._use_normalized_only_for_index():
            where_parts.append("COALESCE(is_normalized, 0) = 1")
        sql = (
            "SELECT id FROM chunks "
            f"WHERE {' AND '.join(where_parts)} "
            "ORDER BY COALESCE(source_updated_at, 0) DESC, id DESC "
            "LIMIT ?"
        )
        params.append(max(8, int(candidate_limit)))
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(sql, tuple(params))
        rows = [int(row[0]) for row in c.fetchall()]
        conn.close()
        return rows

    def _load_candidate_rows(self, candidate_ids: List[int]) -> List[Dict[str, Any]]:
        if not candidate_ids:
            return []
        placeholders = ",".join(["?"] * len(candidate_ids))
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            f"""
            SELECT
                c.*,
                COALESCE(NULLIF(s.original_filename, ''), c.source_path) AS source_display,
                COALESCE(NULLIF(s.doc_role, ''), NULLIF(c.doc_role, ''), ?) AS doc_role,
                COALESCE(s.uploaded_at, c.source_updated_at, 0) AS uploaded_at
            FROM chunks c
            LEFT JOIN source_uploads s
                ON c.source_path = s.source_path
            WHERE c.id IN ({placeholders})
            """,
            [DOC_ROLE_UNKNOWN, *candidate_ids],
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def _limit_results_by_parent(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not bool(getattr(self, "structure_rag_v2_enabled", False)):
            return list(results[:top_k])
        limit = max(1, int(getattr(self, "structure_rag_parent_result_limit", 1) or 1))
        parent_counts: Dict[str, int] = {}
        selected: List[Dict[str, Any]] = []
        for result in results:
            parent_key = str(result.get("parent_chunk_key", "") or "").strip()
            if parent_key:
                rank = parent_counts.get(parent_key, 0) + 1
                if rank > limit:
                    continue
                parent_counts[parent_key] = rank
                result["parent_result_rank"] = rank
            else:
                result["parent_result_rank"] = 0
            selected.append(result)
            if len(selected) >= int(top_k):
                break
        return selected

    def _ground_derived_results_to_parents(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        derived = [
            row for row in results
            if bool(row.get("is_derived", 0)) and str(row.get("parent_chunk_key", "") or "").strip()
        ]
        if not derived:
            return results
        parent_keys = list(dict.fromkeys(str(row.get("parent_chunk_key", "") or "").strip() for row in derived))
        placeholders = ",".join("?" for _ in parent_keys)
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT *
            FROM chunks
            WHERE COALESCE(is_derived, 0) = 0
              AND parent_chunk_key IN ({placeholders})
            ORDER BY id ASC
            """,
            tuple(parent_keys),
        ).fetchall()
        conn.close()
        parents = {str(row["parent_chunk_key"] or ""): dict(row) for row in rows}
        for result in results:
            if not bool(result.get("is_derived", 0)):
                continue
            parent = parents.get(str(result.get("parent_chunk_key", "") or ""))
            if not parent:
                continue
            result["derived_text"] = str(result.get("text", "") or "")
            result["parent_chunk_id"] = int(parent.get("id", 0) or 0)
            result["text"] = str(parent.get("text", "") or "")
            for key in (
                "sheet", "row", "row_end", "page_no", "line_start", "line_end",
                "section", "chunk_kind", "heading_path_json", "structure_path",
                "table_id", "row_no", "cell_no",
            ):
                if key in parent:
                    result[key] = parent[key]
            result["parent_expanded"] = True
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        index_name: str = "large",
        doc_roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search with large embedding index."""
        with self._engine_lock:
            index_key = self._normalize_index_name(index_name)
            role_filters = self._normalize_doc_roles_filter(doc_roles)
            cached = self._get_cached_query_result(query, top_k, index_key, doc_roles=role_filters)
            if cached is not None:
                return cached

            index, index_key = self._resolve_index(index_key)
            query_embedding = self._encode_texts(index_name=index_key, texts=[query], task="query")
            query_vector = np.asarray(query_embedding[0], dtype=np.float32)
            query_codes = self._extract_code_tokens(query)
            query_needs_code = self._is_code_or_class_query(query)
            query_literals = self._extract_query_literals(query)
            now_ts = int(time.time())

            candidate_target = max(top_k, self.search_candidates)
            if role_filters:
                candidate_target = max(candidate_target, top_k * 6)

            concept_candidates = self._search_concept_candidates(
                query=query,
                candidate_limit=max(candidate_target * 2, top_k * 4),
            )
            concept_meta = dict(self._last_concept_search_meta or {})
            ontology_candidates = self._search_ontology_candidates(
                query=query,
                candidate_limit=max(candidate_target * 2, top_k * 4),
            )
            ontology_meta = dict(self._last_ontology_search_meta or {})
            fts_candidates = self._search_fts_candidates(
                query=query,
                candidate_limit=max(candidate_target * 2, top_k * 4),
            )
            text_candidate_ids = self._search_text_candidates_sqlite(
                query=query,
                candidate_limit=max(candidate_target * 2, top_k * 4),
            )
            hnsw_candidates = self._search_dense_candidates_hnsw(
                index=index,
                query_embedding=query_embedding,
                candidate_k=candidate_target,
            )

            candidate_ids: List[int] = []
            seen_ids = set()
            for cid in (
                list(concept_candidates.keys())
                + list(ontology_candidates.keys())
                + list(fts_candidates.keys())
                + list(text_candidate_ids)
                + list(hnsw_candidates.keys())
            ):
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                candidate_ids.append(int(cid))

            dense_candidates = self._search_dense_candidates_sqlite(
                index_name=index_key,
                query_vector=query_vector,
                candidate_k=candidate_target,
                candidate_ids=candidate_ids or None,
            )
            if not dense_candidates and not candidate_ids:
                dense_candidates = self._search_dense_candidates_sqlite(
                    index_name=index_key,
                    query_vector=query_vector,
                    candidate_k=candidate_target,
                )
            if not dense_candidates and hnsw_candidates:
                dense_candidates = hnsw_candidates

            if not candidate_ids:
                candidate_ids = list(dense_candidates.keys())
            if not candidate_ids:
                return []

            rows = self._load_candidate_rows(candidate_ids)

            results: List[Dict[str, Any]] = []
            effective_fts_weight = self.hybrid_fts_weight if fts_candidates else 0.0
            dense_weight = max(0.0, 1.0 - self.lexical_weight - effective_fts_weight)
            if dense_weight <= 0 and dense_candidates:
                dense_weight = max(0.0, 1.0 - self.lexical_weight)
            for res in rows:
                chunk_id = int(res.get("id", 0) or 0)
                doc_role = self._normalize_doc_role(res.get("doc_role", ""))
                if doc_role == DOC_ROLE_UNKNOWN:
                    doc_role = self._infer_doc_role(
                        source_type=(res.get("source_type", "") or "").strip().lower(),
                        source_name=(res.get("source_display", res.get("source_path", "")) or ""),
                    )
                res["doc_role"] = doc_role
                if role_filters and doc_role not in role_filters:
                    continue
                text = res.get("text", "") or ""
                concept_score = float(concept_candidates.get(chunk_id, 0.0))
                ontology_fact_score = float(ontology_candidates.get(chunk_id, 0.0))
                semantic_score = float(dense_candidates.get(chunk_id, 0.0))
                fts_score = float(fts_candidates.get(chunk_id, 0.0))
                lexical_score = self._lexical_overlap_score(query, text)
                literal_score = self._literal_match_score(text, query_literals)
                weak_ocr_hint = is_weak_ocr_hint_text(text)
                uploaded_at = int(res.get("uploaded_at", res.get("source_updated_at", 0)) or 0)
                recency_score = self._recency_score(uploaded_at, now_ts=now_ts)
                score = dense_weight * semantic_score + self.lexical_weight * lexical_score
                score += self.concept_score_weight * concept_score
                score += float(getattr(self, "ontology_score_weight", 0.26)) * ontology_fact_score
                score += effective_fts_weight * fts_score
                score += self.literal_match_boost * literal_score
                score += self.recency_boost * recency_score
                try:
                    heading_path = json.loads(str(res.get("heading_path_json", "") or "[]"))
                except Exception:
                    heading_path = []
                if not isinstance(heading_path, list):
                    heading_path = []
                heading_path = [str(value).strip() for value in heading_path if str(value).strip()]
                heading_text = " ".join(heading_path)
                structure_boost = 0.0
                if heading_text:
                    heading_overlap = self._lexical_overlap_score(query, heading_text)
                    structure_boost = min(0.10, 0.10 * heading_overlap)
                    score += structure_boost
                res["heading_path"] = heading_path
                res["structure_boost"] = float(structure_boost)
                res["parent_expanded"] = False
                res["ontology_match"] = bool(ontology_fact_score > 0.0)
                wiki_memory_boost = 0.0
                wiki_targets = getattr(self, "wiki_memory_boost_targets", {"chunks": {}, "sources": {}, "table_cells": {}})
                wiki_chunk_weight = int((wiki_targets.get("chunks", {}) or {}).get(chunk_id, 0) or 0)
                wiki_source_weight = int(
                    (wiki_targets.get("sources", {}) or {}).get(str(res.get("source_path", "") or ""), 0)
                    or 0
                )
                wiki_table_cell_weight = int(
                    (wiki_targets.get("table_cells", {}) or {}).get(int(res.get("table_cell_id", 0) or 0), 0)
                    or 0
                )
                if wiki_chunk_weight or wiki_source_weight or wiki_table_cell_weight:
                    wiki_memory_boost = min(
                        0.05,
                        self.wiki_memory_boost_weight
                        * max(wiki_chunk_weight, wiki_source_weight, wiki_table_cell_weight),
                    )
                    score += wiki_memory_boost
                tabular_query_boost = 0.0
                has_table_markers = any(marker in text for marker in TABLE_HINT_MARKERS)
                row_summary_match = (
                    TABLE_ROW_SUMMARY_MARKER in text
                    or TABLE_SEMANTIC_ROW_MARKER in text
                )
                if (not row_summary_match) and str(res.get("source_type", "") or "") in {"pdf", "hwpx", "xlsx"}:
                    row_summary_match = "|" in text and len(text) >= 40
                header_match_boost = 0.0
                table_page_boost = 0.0
                alias_match_boost = 0.0
                if has_table_markers and row_summary_match:
                    tabular_query_boost += 0.08
                if is_numeric_evidence_query(query):
                    if has_table_markers:
                        tabular_query_boost += 0.08
                    if row_summary_match:
                        tabular_query_boost += 0.06

                    query_header_terms = (
                        "rate",
                        "amount",
                        "cost",
                        "period",
                        "date",
                        "cycle",
                        "count",
                        "base",
                        "month",
                        "quarter",
                        "year",
                        "interval",
                    )
                    header_matches = sum(
                        1 for term in query_header_terms if term in query and term in text
                    )
                    if header_matches > 0:
                        header_match_boost = min(0.12, 0.03 * header_matches)
                        tabular_query_boost += header_match_boost
                    elif row_summary_match and has_table_markers:
                        header_match_boost = 0.06
                        tabular_query_boost += header_match_boost

                    alias_match_boost = _survey_alias_match_boost(query, text)
                    tabular_query_boost += alias_match_boost

                    if res.get("source_type", "") in {"pdf", "hwpx", "xlsx"} and (
                        row_summary_match
                        or str(res.get("page_parser", "") or "").startswith("paddleocr")
                        or bool(res.get("table_like", False))
                    ):
                        table_page_boost = 0.03
                        tabular_query_boost += table_page_boost

                score += tabular_query_boost
                weak_ocr_hint_penalty = 0.0
                if weak_ocr_hint:
                    raw_signal = max(concept_score, semantic_score, fts_score, lexical_score, literal_score)
                    if raw_signal <= 0.001:
                        weak_ocr_hint_penalty = 0.08
                    elif raw_signal <= 0.03:
                        weak_ocr_hint_penalty = 0.04
                    score -= weak_ocr_hint_penalty
                res["numeric_table_boost"] = float(tabular_query_boost)
                res["tabular_query_boost"] = float(tabular_query_boost)
                res["row_summary_match"] = bool(row_summary_match)
                res["header_match_boost"] = float(header_match_boost)
                res["table_page_boost"] = float(table_page_boost)
                res["alias_match_boost"] = float(alias_match_boost)
                res["weak_ocr_hint"] = bool(weak_ocr_hint)
                res["weak_ocr_hint_penalty"] = float(weak_ocr_hint_penalty)
                res["evidence_strength"] = "weak" if weak_ocr_hint else "strong"
                if query_needs_code:
                    if query_codes and self._has_exact_code(text, query_codes):
                        score += self.code_match_boost
                        res["code_match"] = 2
                    elif (not query_codes) and self._has_any_code(text):
                        score += self.code_match_boost * self.code_hint_boost_ratio
                        res["code_match"] = 1
                    else:
                        res["code_match"] = 0
                else:
                    res["code_match"] = 0
                if int(res.get("is_normalized", 0) or 0) == 1:
                    score -= self.normalized_score_penalty
                    res["normalized_penalty"] = float(self.normalized_score_penalty)
                else:
                    res["normalized_penalty"] = 0.0
                res["concept_score"] = concept_score
                res["matched_concepts"] = list((concept_meta.get("chunk_labels", {}) or {}).get(chunk_id, []))
                res["ontology_fact_score"] = ontology_fact_score
                res["matched_ontology_facts"] = list((ontology_meta.get("chunk_labels", {}) or {}).get(chunk_id, []))
                ontology_chunk_meta = (ontology_meta.get("chunk_meta", {}) or {}).get(chunk_id, {})
                res["ontology_query_rewrite"] = str(
                    ontology_chunk_meta.get("ontology_query_rewrite")
                    or ontology_meta.get("ontology_query_rewrite")
                    or query
                )
                res["ontology_hop_count"] = int(ontology_chunk_meta.get("ontology_hop_count", 0) or 0)
                res["ontology_candidate_reason"] = str(ontology_chunk_meta.get("ontology_candidate_reason", "") or "")
                res["semantic_score"] = semantic_score
                res["fts_score"] = fts_score
                res["lexical_score"] = lexical_score
                res["literal_score"] = literal_score
                res["recency_score"] = recency_score
                res["wiki_memory_boost"] = float(wiki_memory_boost)
                res["uploaded_at"] = uploaded_at
                res["score"] = float(score)
                res["index_name"] = index_key
                results.append(res)

            results.sort(
                key=lambda x: (
                    float(x.get("score", 0.0)),
                    int(x.get("uploaded_at", x.get("source_updated_at", 0)) or 0),
                    int(x.get("id", 0) or 0),
                ),
                reverse=True,
            )
            sliced = self._limit_results_by_parent(results, top_k=top_k)
            sliced = self._ground_derived_results_to_parents(sliced)
            self._set_cached_query_result(query, top_k, index_key, sliced, doc_roles=role_filters)
            return sliced

    def _prune_operational_logs_if_due(
        self,
        *,
        force: bool = False,
        now_ts: Optional[int] = None,
    ) -> int:
        now_value = int(time.time()) if now_ts is None else int(now_ts)
        if (
            not force
            and now_value - int(getattr(self, "_last_log_prune_at", 0) or 0)
            < int(getattr(self, "log_prune_interval_seconds", 3600) or 3600)
        ):
            return 0
        expire_before = now_value - int(getattr(self, "log_retention_days", 30) or 30) * 86400
        conn = self._connect_db()
        removed = 0
        try:
            for table in ("retrieval_logs", "answer_logs"):
                if not _table_exists(conn, table):
                    continue
                removed += prune_timestamped_rows(
                    conn,
                    table=table,
                    id_column="log_id",
                    timestamp_column="created_at",
                    expire_before=expire_before,
                    max_rows=int(getattr(self, "log_retention_max_rows", 50000) or 50000),
                    batch_size=int(getattr(self, "log_prune_batch_size", 2000) or 2000),
                )
            conn.commit()
            self._last_log_prune_at = now_value
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return removed

    def log_retrieval(
        self,
        query_id: str,
        user_id: str,
        query_text: str,
        topk_ids: List[int],
        meta: Optional[Dict[str, Any]] = None,
    ):
        payload_meta = self._safe_json_dump(meta or {})
        topk_json = self._safe_json_dump([int(x) for x in topk_ids if int(x) > 0])
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO retrieval_logs
                (query_id, user_id, query_text, topk_ids_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (query_id or "").strip(),
                (user_id or "").strip(),
                query_text or "",
                topk_json,
                payload_meta,
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()
        self._prune_operational_logs_if_due()

    def log_answer(
        self,
        query_id: str,
        llm_model: str,
        prompt_hash: str,
        answer_text: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        citations_json = self._safe_json_dump(citations or [])
        metadata_json = self._safe_json_dump(metadata or {})
        conn = self._connect_db()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO answer_logs
                (query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (query_id or "").strip(),
                (llm_model or "").strip(),
                (prompt_hash or "").strip(),
                answer_text or "",
                citations_json,
                metadata_json,
                int(time.time()),
            ),
        )
        log_id = int(c.lastrowid)
        conn.commit()
        conn.close()
        self._prune_operational_logs_if_due()
        return log_id

    def get_recent_retrieval_logs(self, limit: int = 120) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT query_id, user_id, query_text, topk_ids_json, meta_json, created_at
            FROM retrieval_logs
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_recent_answer_logs(self, limit: int = 120) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at
            FROM answer_logs
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def _tokenize_for_overlap(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = _OVERLAP_TERM_RE.findall(text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _query_keywords(self, query: str) -> List[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "what",
            "how",
            "when",
            "where",
            "which",
            "guide",
            "manual",
            "summary",
            "document",
            "upload",
        }
        raw_tokens = self._tokenize_for_overlap(query)
        keywords: List[str] = []
        for token in raw_tokens:
            lowered = token.strip().lower()
            if len(lowered) < 2 or lowered in stopwords:
                continue
            keywords.append(lowered)
        return list(dict.fromkeys(keywords))

    def _extract_code_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return sorted(set(re.findall(r"(?<!\d)(\d{2,4})(?!\d)", text)))

    def _is_code_or_class_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", (query or "").lower())
        hints = ("code", "class", "category", "number", "id")
        return any(h in compact for h in hints)

    def _has_exact_code(self, text: str, codes: List[str]) -> bool:
        if not text or not codes:
            return False
        for code in codes:
            if re.search(rf"(?<!\d){re.escape(code)}(?!\d)", text):
                return True
        return False

    def _has_any_code(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"(?<!\d)\d{2,4}(?!\d)", text))

    def _extract_query_literals(self, query: str) -> List[str]:
        if not query:
            return []
        literals: List[str] = []

        # Keep quoted phrases as hard lexical anchors.
        for quoted in _QUOTED_LITERAL_RE.findall(query):
            q = re.sub(r"\s+", " ", quoted).strip().lower()
            if len(q) >= 2:
                literals.append(q)

        # Numeric codes are high-signal for manual-like Q&A.
        literals.extend(self._extract_code_tokens(query))

        # Preserve mixed alpha-numeric tokens (e.g., API names, IDs).
        for tok in _MIXED_LITERAL_RE.findall(query):
            t = tok.strip().lower()
            if not t:
                continue
            if re.search(r"[a-z]", t) and (re.search(r"\d", t) or "-" in t or "_" in t or "/" in t):
                literals.append(t)

        return list(dict.fromkeys(literals))

    def _literal_match_score(self, text: str, literals: List[str]) -> float:
        if not text or not literals:
            return 0.0
        lowered = text.lower()
        hits = 0
        for lit in literals:
            if not lit:
                continue
            if lit.isdigit():
                if re.search(rf"(?<!\d){re.escape(lit)}(?!\d)", lowered):
                    hits += 1
            elif lit in lowered:
                hits += 1
        return float(hits / max(1, len(literals)))

    def _recency_score(self, updated_at: int, now_ts: Optional[int] = None) -> float:
        if not updated_at:
            return 0.0
        now = int(now_ts or time.time())
        age_seconds = max(0, now - int(updated_at))
        age_days = age_seconds / 86400.0
        return float(pow(0.5, age_days / self.recency_half_life_days))

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        q_tokens = set(self._query_keywords(query))
        if not q_tokens:
            return 0.0
        t_tokens = set(self._tokenize_for_overlap(text))
        if not t_tokens:
            return 0.0
        overlap = float(len(q_tokens & t_tokens) / len(q_tokens))
        exact_hits = 0
        lowered = (text or "").lower()
        for tok in q_tokens:
            if len(tok) >= 3 and tok in lowered:
                exact_hits += 1
        if exact_hits <= 0:
            return overlap
        bonus = min(self.exact_keyword_boost, self.exact_keyword_boost * (exact_hits / max(1, len(q_tokens))))
        return min(1.0, overlap + bonus)

    def evaluate_answerability(self, query: str, results: List[Dict[str, Any]], coverage_top_k: int = 6) -> Dict[str, Any]:
        if not results:
            return {
                "top1": 0.0,
                "top2": 0.0,
                "margin": 0.0,
                "coverage": 0.0,
                "keyword_hits": 0,
                "keyword_total": 0,
                "unique_sources": 0,
                "latest_uploaded_at": 0,
                "doc_roles": [],
            }

        top1 = float(results[0].get("semantic_score", results[0].get("score", 0.0)))
        top2 = 0.0
        if len(results) >= 2:
            top2 = float(results[1].get("semantic_score", results[1].get("score", 0.0)))

        margin = top1 - top2

        keywords = self._query_keywords(query)
        if not keywords:
            coverage = 1.0
            hit_count = 0
            total = 0
        else:
            joined = "\n".join((r.get("text", "") or "") for r in results[: max(1, coverage_top_k)]).lower()
            overlap_terms = set(self._tokenize_for_overlap(joined))
            hit_count = len(set(keywords) & overlap_terms)
            total = len(set(keywords))
            coverage = float(hit_count / total) if total else 0.0

        top_window = results[: max(1, coverage_top_k)]
        unique_sources = len(
            {
                (r.get("source_path", "") or "").strip()
                for r in top_window
                if (r.get("source_path", "") or "").strip()
            }
        )
        latest_uploaded_at = max(
            int(r.get("uploaded_at", r.get("source_updated_at", 0)) or 0)
            for r in top_window
        )
        doc_roles = sorted(
            {
                self._normalize_doc_role(r.get("doc_role", ""))
                for r in top_window
                if self._normalize_doc_role(r.get("doc_role", "")) != DOC_ROLE_UNKNOWN
            }
        )

        return {
            "top1": top1,
            "top2": top2,
            "margin": margin,
            "coverage": coverage,
            "keyword_hits": hit_count,
            "keyword_total": total,
            "unique_sources": unique_sources,
            "latest_uploaded_at": latest_uploaded_at,
            "doc_roles": doc_roles,
        }

    def format_source_ref(self, res: Dict[str, Any]) -> str:
        role = self._normalize_doc_role(res.get("doc_role", ""))
        role_tail = f" / role={role}" if role != DOC_ROLE_UNKNOWN else ""
        if int(res.get("is_normalized", 0) or 0) == 1:
            group = (res.get("normalized_group", "") or res.get("source_type", "") or "data").strip().lower()
            source_updated_at = int(res.get("source_updated_at", 0) or 0)
            updated = self._format_timestamp(source_updated_at)
            if group == "txt":
                bundle_no = int(res.get("line_start", 0) or res.get("row", 0) or 0)
                if bundle_no > 0:
                    return (
                        f"{NORMALIZED_BUNDLE_LABEL} TXT {bundle_no}번 / "
                        f"{LABEL_LATEST_UPLOAD_REFLECTED} {updated}{role_tail}"
                    )
                return (
                    f"{NORMALIZED_BUNDLE_LABEL} TXT / "
                    f"{LABEL_LATEST_UPLOAD_REFLECTED} {updated}{role_tail}"
                )
            if group == "xlsx":
                bundle_no = int(res.get("row", 0) or res.get("line_start", 0) or 0)
                if bundle_no > 0:
                    return (
                        f"{NORMALIZED_BUNDLE_LABEL} XLSX {bundle_no}번 / "
                        f"{LABEL_LATEST_UPLOAD_REFLECTED} {updated}{role_tail}"
                    )
                return (
                    f"{NORMALIZED_BUNDLE_LABEL} XLSX / "
                    f"{LABEL_LATEST_UPLOAD_REFLECTED} {updated}{role_tail}"
                )
            return (
                f"{NORMALIZED_BUNDLE_LABEL} {group.upper()} / "
                f"{LABEL_LATEST_UPLOAD_REFLECTED} {updated}{role_tail}"
            )

        source = (res.get("source_display", "") or res.get("source_path", "") or "").strip()
        updated = self._format_timestamp(int(res.get("uploaded_at", res.get("source_updated_at", 0)) or 0))
        updated_tail = f" / {LABEL_UPLOAD} {updated}" if updated != "-" else ""
        section = (res.get("section", "") or "").strip()
        if res.get("sheet"):
            row = int(res.get("row", 0) or 0)
            row_end = int(res.get("row_end", row) or row)
            if row_end > row:
                return (
                    f"{source} / {res.get('sheet')} / "
                    f"{LABEL_ROW} {row}-{row_end}{updated_tail}{role_tail}"
                )
            return f"{source} / {res.get('sheet')} / {LABEL_ROW} {row}{updated_tail}{role_tail}"
        if res.get("line_start"):
            line_start = int(res.get("line_start", 0) or 0)
            line_end = int(res.get("line_end", line_start) or line_start)
            if section:
                return (
                    f"{source} / {section} / "
                    f"{LABEL_LINE} {line_start}-{line_end}{updated_tail}{role_tail}"
                )
            return f"{source} / {LABEL_LINE} {line_start}-{line_end}{updated_tail}{role_tail}"
        if section:
            return f"{source} / {section}{updated_tail}{role_tail}"
        if updated_tail:
            return f"{source}{updated_tail}{role_tail}"
        return source

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
        parts: List[str] = []
        for line in lines:
            segs = _SENTENCE_SPLIT_RE.split(line)
            for seg in segs:
                s = seg.strip()
                if s:
                    parts.append(s)
        return parts

    def _extract_relevant_snippet(self, text: str, query: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        q_tokens = set(self._query_keywords(query))
        if not q_tokens:
            return text[:max_chars] + "\n...[truncated]"

        sents = self._split_sentences(text)
        if not sents:
            return text[:max_chars] + "\n...[truncated]"

        scored: List[Any] = []
        for idx, sent in enumerate(sents):
            score = self._lexical_overlap_score(query, sent)
            if len(sent) < 24:
                score *= 0.7
            scored.append((score, idx, sent))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        selected_idx: List[int] = []
        used = 0
        for score, idx, sent in scored:
            if score <= 0 and selected_idx:
                continue
            add_len = len(sent) + 1
            if used + add_len > max_chars:
                continue
            selected_idx.append(idx)
            used += add_len
            if used >= int(max_chars * 0.85):
                break

        if not selected_idx:
            return text[:max_chars] + "\n...[truncated]"

        selected_idx.sort()
        snippet = " ".join(sents[i] for i in selected_idx)

        min_len = max(80, int(max_chars * 0.35))
        if len(snippet) < min_len:
            remain = max_chars - len(snippet) - 1
            if remain > 40:
                snippet = f"{snippet}\n{text[:remain]}"

        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
        if len(snippet) < len(text):
            snippet += "\n...[focused]"
        return snippet

    def get_context_string(
        self,
        results: List[Dict[str, Any]],
        query: str = "",
        max_chars: int = 2400,
        per_result_max_chars: int = 700,
        focus_relevant: bool = True,
        top1_score: Optional[float] = None,
    ) -> str:
        """Format search results into a bounded context string for LLM."""
        if not results:
            return ""

        context_parts: List[str] = []
        used = 0
        seen = set()

        files = [
            (r.get("source_display", "") or r.get("source_path", "") or "").strip()
            for r in results
            if (r.get("source_display", "") or r.get("source_path", "") or "").strip()
        ]
        sections = [
            (r.get("section", "") or r.get("sheet", "") or "").strip()
            for r in results
            if (r.get("section", "") or r.get("sheet", "") or "").strip()
        ]
        uniq_sections = list(dict.fromkeys(sections))
        uploaded_times = [
            int(r.get("uploaded_at", r.get("source_updated_at", 0)) or 0)
            for r in results
            if int(r.get("uploaded_at", r.get("source_updated_at", 0)) or 0) > 0
        ]
        roles = [
            self._normalize_doc_role(r.get("doc_role", ""))
            for r in results
            if self._normalize_doc_role(r.get("doc_role", "")) != DOC_ROLE_UNKNOWN
        ]
        summary_lines = [
            "[CONTEXT_SUMMARY]",
            f"- passages={len(results)} | files={len(set(files))}",
        ]
        if uniq_sections:
            summary_lines.append(f"- sections={', '.join(uniq_sections[:4])}")
        if uploaded_times:
            summary_lines.append(f"- latest_upload={self._format_timestamp(max(uploaded_times))}")
        if roles:
            uniq_roles = list(dict.fromkeys(roles))
            summary_lines.append(f"- doc_roles={', '.join(uniq_roles)}")
        if top1_score is not None:
            summary_lines.append(f"- top1_similarity={float(top1_score):.3f}")
        summary_block = "\n".join(summary_lines) + "\n\n"
        summary_block = summary_block[:max_chars]
        context_parts.append(summary_block)
        used += len(summary_block)
        doc_no = 0

        for res in results:
            text = res.get("text", "") or ""
            source_path = res.get("source_path", "")
            source_display = (res.get("source_display", "") or source_path or "").strip()
            key = (
                source_path,
                res.get("sheet", ""),
                int(res.get("row", 0) or 0),
                int(res.get("row_end", 0) or 0),
                int(res.get("line_start", 0) or 0),
                int(res.get("line_end", 0) or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            doc_no += 1

            if len(text) > per_result_max_chars:
                if focus_relevant:
                    text = self._extract_relevant_snippet(text, query, per_result_max_chars)
                else:
                    text = text[:per_result_max_chars] + "\n...[truncated]"

            doc_type = (res.get("source_type", "") or "").strip() or "unknown"
            doc_role = self._normalize_doc_role(res.get("doc_role", ""))
            section = (res.get("section", "") or "").strip()
            sheet = (res.get("sheet", "") or "").strip()
            row = int(res.get("row", 0) or 0)
            row_end = int(res.get("row_end", row) or row)
            line_start = int(res.get("line_start", 0) or 0)
            line_end = int(res.get("line_end", line_start) or line_start)
            uploaded_at = int(res.get("uploaded_at", res.get("source_updated_at", 0)) or 0)
            uploaded_label = self._format_timestamp(uploaded_at)
            doc_version = (res.get("doc_version", "") or "").strip() or "-"
            if row > 0 and row_end > row:
                row_label = f"{row}-{row_end}"
            elif row > 0:
                row_label = str(row)
            elif line_start > 0 and line_end >= line_start:
                row_label = f"L{line_start}-L{line_end}"
            else:
                row_label = "-"

            item = f"[DOC {doc_no}]\n"
            item += (
                f"file={source_display or source_path} | type={doc_type} | role={doc_role} | sheet={sheet or '-'} | "
                f"section={section or '-'} | row={row_label} | "
                f"uploaded={uploaded_label} | version={doc_version}\n"
            )
            item += f"text={text}\n\n"

            if used + len(item) > max_chars:
                if not context_parts and max_chars > 140:
                    header = f"[DOC {doc_no}]\nfile={source_path}\n"
                    remain = max(40, max_chars - len(header) - len("text=\n...[truncated]\n\n"))
                    compact_text = text[:remain]
                    compact_item = f"{header}text={compact_text}\n...[truncated]\n\n"
                    context_parts.append(compact_item[:max_chars])
                break

            context_parts.append(item)
            used += len(item)

        return "".join(context_parts)

    def get_source_overview_context(
        self,
        source_path: str,
        max_chars: int = 1200,
        max_rows: int = 24,
        per_item_max_chars: int = 260,
    ) -> str:
        """Build ordered context from one source file for overview questions."""
        if not source_path:
            return ""

        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT source_path, section, sheet, row, row_end, line_start, line_end, text
            FROM chunks
            WHERE source_path = ?
            ORDER BY
                CASE
                    WHEN line_start IS NOT NULL AND line_start > 0 THEN line_start
                    WHEN row IS NOT NULL AND row > 0 THEN row
                    ELSE 999999
                END ASC,
                id ASC
            LIMIT ?
            """,
            (source_path, max_rows),
        )
        rows = c.fetchall()
        conn.close()

        parts: List[str] = []
        used = 0
        for r in rows:
            txt = r["text"] or ""
            if len(txt) > per_item_max_chars:
                txt = txt[:per_item_max_chars] + "\n...[truncated]"

            line = f"[OVERVIEW] Source: {r['source_path']}\n"
            if r["section"]:
                line += f"Section: {r['section']}\n"
            if r["sheet"]:
                row = int(r["row"] or 0)
                row_end = int(r["row_end"] or row)
                if row_end > row:
                    line += f"Location: {r['sheet']} Row {row}-{row_end}\n"
                else:
                    line += f"Location: {r['sheet']} Row {row}\n"
            elif r["line_start"]:
                line += f"Location: Lines {r['line_start']}-{r['line_end']}\n"
            line += f"Content:\n{txt}\n\n"

            if used + len(line) > max_chars:
                break
            parts.append(line)
            used += len(line)

        return "".join(parts)

    def get_source_outline_context(
        self,
        source_path: str,
        max_chars: int = 1200,
        max_items: int = 18,
    ) -> str:
        """Build a compact outline for one source so the agent can navigate long documents cheaply."""
        if not source_path:
            return ""

        conn = self._connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT COUNT(1) AS chunk_count
            FROM chunks
            WHERE source_path = ? AND COALESCE(is_normalized, 0) = 0
            """,
            (source_path,),
        )
        count_row = c.fetchone()
        chunk_count = int((count_row["chunk_count"] if count_row else 0) or 0)

        c.execute(
            """
            SELECT source_path, source_type, section, sheet, row, row_end, line_start, line_end, text
            FROM chunks
            WHERE source_path = ? AND COALESCE(is_normalized, 0) = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (source_path, max(8, int(max_items) * 4)),
        )
        rows = c.fetchall()
        conn.close()

        if not rows:
            return ""

        lines = [f"SOURCE_OUTLINE: {source_path}", f"- chunk_count={chunk_count}"]
        used = sum(len(line) + 1 for line in lines)
        seen_locations: set[str] = set()
        item_no = 0

        for r in rows:
            section = (r["section"] or "").strip()
            sheet = (r["sheet"] or "").strip()
            row_start = int(r["row"] or 0)
            row_end = int(r["row_end"] or row_start)
            line_start = int(r["line_start"] or 0)
            line_end = int(r["line_end"] or line_start)
            preview = re.sub(r"\s+", " ", (r["text"] or "").strip())
            if len(preview) > 120:
                preview = preview[:120] + "..."

            if sheet:
                location = (
                    f"{sheet} / {LABEL_ROW} {row_start}-{row_end}"
                    if row_end > row_start
                    else f"{sheet} / {LABEL_ROW} {row_start}"
                )
            elif section and line_start > 0:
                location = f"{section} / {LABEL_LINE} {line_start}-{line_end}"
            elif section:
                location = section
            elif line_start > 0:
                location = f"{LABEL_LINE} {line_start}-{line_end}"
            else:
                location = LABEL_BODY

            if location in seen_locations:
                continue
            seen_locations.add(location)
            item_no += 1
            item_line = f"{item_no}. {location} | preview={preview or '-'}"
            if used + len(item_line) + 1 > max_chars:
                break
            lines.append(item_line)
            used += len(item_line) + 1
            if item_no >= max(1, int(max_items)):
                break

        return "\n".join(lines)
