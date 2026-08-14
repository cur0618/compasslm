import json
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.ontology_store import OntologyStore
from src.wiki_answer_compiler import compile_answer_memory
from src.wiki_page_builder import build_wiki_page_payload


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


def _safe_json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def _plain_summary(text: str, limit: int = 220) -> str:
    plain = re.sub(r"\[\[CITATION:\d+\|[^\]]+\]\]", "", str(text or ""))
    plain = re.sub(r"\[DOC\s+\d+\]", "", plain)
    plain = " ".join(plain.split())
    if len(plain) <= limit:
        return plain
    return plain[:limit].rstrip() + "..."


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").lower().split())[:500]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower()) if len(t) >= 2]


def _question_aliases(text: str) -> str:
    aliases = str(text or "")
    replacements = {
        "태양광": "태양열 발전기",
        "태양열": "태양열 발전기",
        "설치": "설치 처리 조사 방법",
        "처리": "조사 방법",
        "조사": "조사 방법",
    }
    for src, dst in replacements.items():
        if src in aliases:
            aliases += f" {dst}"
    return aliases


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _quality_flags(
    *,
    answer_text: str,
    citations: List[Dict[str, Any]],
    source_statuses: List[str],
    metadata: Dict[str, Any],
) -> List[str]:
    flags = set()
    if not citations:
        flags.add("citationless")
    if any(status == "broken" for status in source_statuses):
        flags.add("broken_source")
    if any(status == "invalid_page_no" for status in source_statuses):
        flags.add("invalid_page_no")
    outside_markers = (
        "문서 밖 참고",
        "문서 밖",
        "일반적으로",
        "추정",
        "고려할 수 있습니다",
    )
    if any(marker in str(answer_text or "") for marker in outside_markers):
        flags.add("outside_document_claim")
    quality_issue = str(metadata.get("response_quality_issue", "") or "")
    if quality_issue:
        flags.add(quality_issue)
    phase_events = metadata.get("phase_events", [])
    if isinstance(phase_events, list):
        for event in phase_events:
            if not isinstance(event, dict):
                continue
            status = str(event.get("status", "") or "")
            detail = str(event.get("detail", "") or "")
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
            if status == "fallback" or payload.get("helper_degraded"):
                flags.add("helper_degraded")
            if "Exceeded maximum retries" in detail:
                flags.add("validation_retry_failed")
    retrieval_metrics = metadata.get("retrieval_metrics", {})
    if isinstance(retrieval_metrics, dict) and retrieval_metrics.get("has_conflict"):
        flags.add("conflict_present")
    return sorted(flags)


class WikiMemoryStore:
    """Citation-backed answer memory derived from answer_logs.

    Raw chunks remain the source of truth. This store records user-approved
    answers and their provenance so retrieval can use them only after source
    revalidation.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_saved_answers
            (
                saved_answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                answer_log_id INTEGER,
                user_id TEXT,
                question_text TEXT,
                normalized_question TEXT,
                answer_text TEXT,
                answer_summary TEXT,
                citation_json TEXT,
                status TEXT,
                confidence_score REAL,
                source_count INTEGER,
                quality_flags_json TEXT DEFAULT '[]',
                created_at INTEGER,
                updated_at INTEGER,
                reused_count INTEGER DEFAULT 0,
                last_reused_at INTEGER DEFAULT 0
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_sources
            (
                answer_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_answer_id INTEGER,
                source_path TEXT,
                source_ref TEXT,
                page_no INTEGER,
                chunk_id INTEGER,
                table_cell_id INTEGER,
                citation_label TEXT,
                quote_or_excerpt TEXT,
                status TEXT,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_feedback
            (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_answer_id INTEGER,
                answer_log_id INTEGER,
                query_id TEXT,
                user_id TEXT,
                feedback_type TEXT,
                saved_to_wiki INTEGER,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_claims
            (
                claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_answer_id INTEGER,
                claim_text TEXT,
                normalized_claim TEXT,
                source_refs_json TEXT,
                confidence_score REAL,
                status TEXT,
                conflict_group_id TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_concepts
            (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT,
                aliases_json TEXT,
                description TEXT,
                related_saved_answers_json TEXT,
                related_sources_json TEXT,
                confidence_score REAL,
                status TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_procedures
            (
                procedure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_name TEXT,
                procedure_steps_json TEXT,
                conditions_json TEXT,
                exceptions_json TEXT,
                source_refs_json TEXT,
                related_saved_answer_id INTEGER,
                status TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_table_rules
            (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT,
                rule_text TEXT,
                source_path TEXT,
                sheet_name TEXT,
                table_range TEXT,
                row_refs_json TEXT,
                column_refs_json TEXT,
                related_saved_answer_id INTEGER,
                status TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_conflicts
            (
                conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_answer_id INTEGER,
                conflicting_saved_answer_id INTEGER,
                conflict_type TEXT,
                description TEXT,
                status TEXT,
                created_at INTEGER,
                resolved_at INTEGER DEFAULT 0
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_answer_usage_stats
            (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_answer_id INTEGER,
                query_text TEXT,
                used_source_refs_json TEXT,
                success INTEGER,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_memory_lint_findings
            (
                finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_type TEXT,
                target_type TEXT,
                target_id INTEGER,
                severity TEXT,
                message TEXT,
                status TEXT,
                metadata_json TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_saved_answer_query_user
            ON wiki_saved_answers(query_id, user_id)
            """
        )
        self._ensure_column(c, "wiki_saved_answers", "quality_flags_json", "TEXT DEFAULT '[]'")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_saved_status ON wiki_saved_answers(status, updated_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_answer_sources_saved ON wiki_answer_sources(saved_answer_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_answer_feedback_query ON wiki_answer_feedback(query_id, created_at)")
        conn.commit()
        conn.close()

    def _ensure_column(self, cursor: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
        rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
        if column not in {str(row[1]) for row in rows}:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _latest_answer_log(self, conn: sqlite3.Connection, query_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            """
            SELECT log_id, query_id, answer_text, citations_json, answer_meta_json, created_at
            FROM answer_logs
            WHERE query_id = ?
            ORDER BY created_at DESC, log_id DESC
            LIMIT 1
            """,
            (query_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def _original_review_chunk_ids(
        self,
        conn: sqlite3.Connection,
        *,
        query_id: str,
        user_id: str,
        citations: List[Dict[str, Any]],
        limit: int = 8,
    ) -> List[int]:
        safe_limit = max(1, min(8, int(limit or 8)))
        chunk_columns = _table_columns(conn, "chunks")
        if "id" not in chunk_columns:
            return []
        normalized_sql = "COALESCE(is_normalized, 0) = 0" if "is_normalized" in chunk_columns else "1 = 1"
        derived_sql = "COALESCE(is_derived, 0) = 0" if "is_derived" in chunk_columns else "1 = 1"

        def _filter_original(values: List[int]) -> List[int]:
            unique_values = list(dict.fromkeys(int(value) for value in values if int(value) > 0))
            if not unique_values:
                return []
            placeholders = ",".join("?" for _ in unique_values)
            rows = conn.execute(
                f"SELECT id FROM chunks WHERE id IN ({placeholders}) AND {normalized_sql} AND {derived_sql}",
                tuple(unique_values),
            ).fetchall()
            valid_ids = {int(row[0]) for row in rows}
            return [chunk_id for chunk_id in unique_values if chunk_id in valid_ids][:safe_limit]

        citation_candidates = [_int_value(item.get("chunk_id", 0)) for item in citations]
        selected = _filter_original(citation_candidates)
        if selected:
            return selected

        candidates: List[int] = []
        if _table_columns(conn, "retrieval_logs"):
            row = conn.execute(
                """
                SELECT topk_ids_json
                FROM retrieval_logs
                WHERE query_id = ? AND (user_id = ? OR COALESCE(user_id, '') = '')
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (str(query_id or ""), str(user_id or "")),
            ).fetchone()
            if row is not None:
                values = _safe_json_loads(str(row[0] or "[]"), [])
                if isinstance(values, list):
                    candidates = [_int_value(value) for value in values]
                    candidates = [chunk_id for chunk_id in candidates if chunk_id > 0]
        return _filter_original(candidates)

    def _chunk_exists(self, conn: sqlite3.Connection, chunk_id: int, source_path: str) -> bool:
        if chunk_id <= 0:
            return bool(source_path)
        try:
            row = conn.execute(
                "SELECT id FROM chunks WHERE id = ? LIMIT 1",
                (int(chunk_id),),
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            return bool(source_path)

    def _source_status(self, conn: sqlite3.Connection, citation: Dict[str, Any]) -> str:
        source_path = str(citation.get("source_path", "") or "").strip()
        chunk_id = _int_value(citation.get("chunk_id", 0))
        page_no = _int_value(citation.get("page_no", 0))
        if page_no > 1000:
            return "invalid_page_no"
        if not source_path and chunk_id <= 0:
            return "broken"
        if not self._chunk_exists(conn, chunk_id, source_path):
            return "broken"
        return "active"

    def save_answer_from_query_id(
        self,
        *,
        query_id: str,
        user_id: str,
        feedback_type: str = "save_to_wiki",
        fallback_question: str = "",
    ) -> Dict[str, Any]:
        now = int(time.time())
        safe_query_id = str(query_id or "").strip()
        safe_user_id = str(user_id or "").strip() or "anonymous"
        if not safe_query_id:
            raise ValueError("query_id is required")

        conn = self._connect()
        c = conn.cursor()
        answer_log = self._latest_answer_log(conn, safe_query_id)
        if answer_log is None:
            conn.close()
            raise LookupError(f"answer log not found for query_id={safe_query_id}")

        metadata = _safe_json_loads(str(answer_log.get("answer_meta_json", "") or "{}"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        citations = _safe_json_loads(str(answer_log.get("citations_json", "") or "[]"), [])
        if not isinstance(citations, list):
            citations = []
        clean_citations = [dict(item) for item in citations if isinstance(item, dict)]

        question = (
            str(metadata.get("original_user_message", "") or "").strip()
            or str(metadata.get("effective_user_message", "") or "").strip()
            or str(fallback_question or "").strip()
        )
        answer_text = str(answer_log.get("answer_text", "") or "")
        source_statuses = [self._source_status(conn, citation) for citation in clean_citations]
        source_count = len(clean_citations)
        quality_flags = _quality_flags(
            answer_text=answer_text,
            citations=clean_citations,
            source_statuses=source_statuses,
            metadata=metadata,
        )
        all_sources_active = bool(clean_citations) and all(status == "active" for status in source_statuses)
        status = "published" if all_sources_active and not quality_flags else "needs_review"
        if feedback_type == "report_citation_issue":
            status = "reported"

        existing = c.execute(
            """
            SELECT * FROM wiki_saved_answers
            WHERE query_id = ? AND user_id = ?
            LIMIT 1
            """,
            (safe_query_id, safe_user_id),
        ).fetchone()
        if existing is None:
            c.execute(
                """
                INSERT INTO wiki_saved_answers
                    (query_id, answer_log_id, user_id, question_text, normalized_question,
                     answer_text, answer_summary, citation_json, status, confidence_score,
                     source_count, quality_flags_json, created_at, updated_at, reused_count, last_reused_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_query_id,
                    int(answer_log["log_id"]),
                    safe_user_id,
                    question,
                    _normalize_question(question),
                    answer_text,
                    _plain_summary(answer_text),
                    _safe_json_dump(clean_citations),
                    status,
                    0.75 if status == "published" else 0.35,
                    source_count,
                    _safe_json_dump(quality_flags),
                    now,
                    now,
                    0,
                    0,
                ),
            )
            saved_answer_id = int(c.lastrowid)
            for citation, source_status in zip(clean_citations, source_statuses):
                c.execute(
                    """
                    INSERT INTO wiki_answer_sources
                        (saved_answer_id, source_path, source_ref, page_no, chunk_id, table_cell_id,
                         citation_label, quote_or_excerpt, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        saved_answer_id,
                        str(citation.get("source_path", "") or ""),
                        str(citation.get("source_ref", "") or citation.get("label", "") or ""),
                        _int_value(citation.get("page_no", 0)),
                        _int_value(citation.get("chunk_id", 0)),
                        _int_value(citation.get("table_cell_id", 0)),
                        str(citation.get("citation_label", "") or citation.get("label", "") or ""),
                        str(citation.get("quote", "") or citation.get("text", "") or "")[:800],
                        source_status,
                        now,
                    ),
                )
            if answer_text.strip():
                c.execute(
                    """
                    INSERT INTO wiki_answer_claims
                        (saved_answer_id, claim_text, normalized_claim, source_refs_json,
                         confidence_score, status, conflict_group_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        saved_answer_id,
                        _plain_summary(answer_text, limit=500),
                        _normalize_question(answer_text),
                        _safe_json_dump(clean_citations),
                        0.75 if status == "published" else 0.35,
                        status,
                        "",
                        now,
                        now,
                    ),
                )
        else:
            saved_answer_id = int(existing["saved_answer_id"])
            if feedback_type == "report_citation_issue":
                c.execute(
                    """
                    UPDATE wiki_saved_answers
                    SET status = ?, updated_at = ?
                    WHERE saved_answer_id = ?
                    """,
                    ("reported", now, saved_answer_id),
                )

        c.execute(
            """
            INSERT INTO wiki_answer_feedback
                (saved_answer_id, answer_log_id, query_id, user_id, feedback_type, saved_to_wiki, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                saved_answer_id,
                int(answer_log["log_id"]),
                safe_query_id,
                safe_user_id,
                str(feedback_type or "save_to_wiki"),
                0 if feedback_type == "report_citation_issue" else 1,
                now,
            ),
        )
        conn.commit()
        row = c.execute(
            "SELECT * FROM wiki_saved_answers WHERE saved_answer_id = ?",
            (saved_answer_id,),
        ).fetchone()
        ontology_review_chunk_ids = []
        if feedback_type == "report_citation_issue":
            ontology_review_chunk_ids = self._original_review_chunk_ids(
                conn,
                query_id=safe_query_id,
                user_id=safe_user_id,
                citations=clean_citations,
                limit=8,
            )
        conn.close()
        chunk_ids = [_int_value(citation.get("chunk_id", 0)) for citation in clean_citations]
        chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id > 0]
        if chunk_ids:
            ontology_signal = "reported" if feedback_type == "report_citation_issue" else str(dict(row).get("status", "") or "")
            if ontology_signal in {"published", "reported", "needs_review"}:
                OntologyStore(self.db_path).apply_wiki_signal(
                    chunk_ids=chunk_ids,
                    signal=ontology_signal,
                    boost=0.08,
                    feedback_key=f"saved_answer:{saved_answer_id}:{feedback_type or 'save_to_wiki'}",
                )
        result = dict(row)
        if feedback_type == "report_citation_issue":
            result["ontology_review_chunk_ids"] = ontology_review_chunk_ids
        return result

    def list_saved_answers(self, limit: int = 100, status: str = "") -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where_sql = ""
        params: List[Any] = []
        if safe_status:
            where_sql = "WHERE status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT saved_answer_id, query_id, user_id, question_text, answer_summary,
                       status, confidence_score, source_count, created_at, updated_at,
                       reused_count, last_reused_at, quality_flags_json
                FROM wiki_saved_answers
                {where_sql}
                ORDER BY updated_at DESC, saved_answer_id DESC
                LIMIT ?
                """,
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def get_saved_answer(self, saved_answer_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM wiki_saved_answers WHERE saved_answer_id = ?",
            (int(saved_answer_id),),
        ).fetchone()
        if row is None:
            conn.close()
            return None
        data = dict(row)
        data["sources"] = [
            dict(src)
            for src in conn.execute(
                "SELECT * FROM wiki_answer_sources WHERE saved_answer_id = ? ORDER BY answer_source_id",
                (int(saved_answer_id),),
            )
        ]
        data["claims"] = [
            dict(claim)
            for claim in conn.execute(
                "SELECT * FROM wiki_answer_claims WHERE saved_answer_id = ? ORDER BY claim_id",
                (int(saved_answer_id),),
            )
        ]
        conn.close()
        return data

    def update_saved_answer_status(self, saved_answer_id: int, status: str) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        safe_status = str(status or "").strip() or "needs_review"
        conn = self._connect()
        conn.execute(
            "UPDATE wiki_saved_answers SET status = ?, updated_at = ? WHERE saved_answer_id = ?",
            (safe_status, now, int(saved_answer_id)),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM wiki_saved_answers WHERE saved_answer_id = ?",
            (int(saved_answer_id),),
        ).fetchone()
        source_rows = conn.execute(
            "SELECT chunk_id FROM wiki_answer_sources WHERE saved_answer_id = ?",
            (int(saved_answer_id),),
        ).fetchall()
        conn.close()
        chunk_ids = [_int_value(row["chunk_id"]) for row in source_rows]
        if safe_status in {"published", "reported", "needs_review"} and chunk_ids:
            signal = "reported" if safe_status in {"reported", "needs_review"} else "published"
            OntologyStore(self.db_path).apply_wiki_signal(
                chunk_ids=[chunk_id for chunk_id in chunk_ids if chunk_id > 0],
                signal=signal,
                boost=0.08,
                feedback_key=f"saved_answer_status:{int(saved_answer_id)}:{safe_status}",
            )
        return dict(row) if row is not None else None

    def delete_saved_answer(self, saved_answer_id: int) -> bool:
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM wiki_saved_answers WHERE saved_answer_id = ?", (int(saved_answer_id),))
        deleted = c.rowcount > 0
        c.execute("DELETE FROM wiki_answer_sources WHERE saved_answer_id = ?", (int(saved_answer_id),))
        c.execute("DELETE FROM wiki_answer_feedback WHERE saved_answer_id = ?", (int(saved_answer_id),))
        c.execute("DELETE FROM wiki_answer_claims WHERE saved_answer_id = ?", (int(saved_answer_id),))
        conn.commit()
        conn.close()
        return deleted

    def search_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        tokens = _tokenize(_question_aliases(query))
        if not tokens:
            return []
        rows = self.list_saved_answers(limit=200, status="published")
        scored = []
        for row in rows:
            haystack = _question_aliases(f"{row.get('question_text', '')} {row.get('answer_summary', '')}").lower()
            hits = sum(1 for token in tokens if token in haystack)
            if hits:
                item = dict(row)
                reuse_boost = min(0.2, float(item.get("reused_count", 0) or 0) * 0.02)
                item["memory_score"] = (hits / max(1, len(tokens))) + reuse_boost
                scored.append(item)
        scored.sort(key=lambda item: (-float(item.get("memory_score", 0.0)), -int(item.get("updated_at", 0) or 0)))
        matches = scored[: max(1, int(limit or 3))]
        self.record_memory_usage(query, matches, success=1)
        return matches

    def record_memory_usage(self, query: str, matches: List[Dict[str, Any]], success: int = 1) -> None:
        if not matches:
            return
        now = int(time.time())
        conn = self._connect()
        refs = [
            {
                "saved_answer_id": int(item.get("saved_answer_id", 0) or 0),
                "score": float(item.get("memory_score", 0.0) or 0.0),
            }
            for item in matches
        ]
        for item in matches:
            saved_answer_id = int(item.get("saved_answer_id", 0) or 0)
            if saved_answer_id <= 0:
                continue
            conn.execute(
                """
                UPDATE wiki_saved_answers
                SET reused_count = COALESCE(reused_count, 0) + 1, last_reused_at = ?
                WHERE saved_answer_id = ?
                """,
                (now, saved_answer_id),
            )
            item["reused_count"] = int(item.get("reused_count", 0) or 0) + 1
            item["last_reused_at"] = now
        conn.execute(
            """
            INSERT INTO wiki_answer_usage_stats
                (saved_answer_id, query_text, used_source_refs_json, success, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(matches[0].get("saved_answer_id", 0) or 0),
                str(query or ""),
                _safe_json_dump(refs),
                int(success),
                now,
            ),
        )
        conn.commit()
        conn.close()

    def quality_summary(self) -> Dict[str, Any]:
        conn = self._connect()
        status_counts = {
            str(row["status"] or "unknown"): int(row["count"])
            for row in conn.execute(
                """
                SELECT COALESCE(status, 'unknown') AS status, COUNT(*) AS count
                FROM wiki_saved_answers
                GROUP BY COALESCE(status, 'unknown')
                """
            )
        }
        quality_flag_counts: Dict[str, int] = {}
        for row in conn.execute("SELECT quality_flags_json FROM wiki_saved_answers"):
            flags = _safe_json_loads(str(row["quality_flags_json"] or "[]"), [])
            if not isinstance(flags, list):
                continue
            for flag in flags:
                safe_flag = str(flag or "").strip()
                if safe_flag:
                    quality_flag_counts[safe_flag] = quality_flag_counts.get(safe_flag, 0) + 1
        usage_count = int(conn.execute("SELECT COUNT(*) FROM wiki_answer_usage_stats").fetchone()[0])
        conn.close()
        return {
            "status_counts": status_counts,
            "quality_flag_counts": quality_flag_counts,
            "usage_count": usage_count,
        }

    def run_lint(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM wiki_memory_lint_findings")
        findings: List[Dict[str, Any]] = []
        saved_rows = [
            dict(row)
            for row in c.execute(
                """
                SELECT saved_answer_id, status, source_count, citation_json, question_text, quality_flags_json
                FROM wiki_saved_answers
                WHERE COALESCE(status, '') != 'archived'
                """
            )
        ]
        for row in saved_rows:
            saved_answer_id = int(row.get("saved_answer_id", 0) or 0)
            citations = _safe_json_loads(str(row.get("citation_json", "") or "[]"), [])
            if not isinstance(citations, list) or not citations:
                findings.append(
                    {
                        "finding_type": "citationless_saved_answer",
                        "target_type": "saved_answer",
                        "target_id": saved_answer_id,
                        "severity": "high",
                        "message": "근거 citation이 없는 saved answer입니다.",
                    }
                )
            source_rows = [
                dict(src)
                for src in c.execute(
                    "SELECT * FROM wiki_answer_sources WHERE saved_answer_id = ?",
                    (saved_answer_id,),
                )
            ]
            for src in source_rows:
                if str(src.get("status", "") or "") == "broken":
                    findings.append(
                        {
                            "finding_type": "broken_source_reference",
                            "target_type": "answer_source",
                            "target_id": int(src.get("answer_source_id", 0) or 0),
                            "severity": "high",
                            "message": "원본 chunk/source reference가 깨진 saved answer source입니다.",
                        }
                    )
                if str(src.get("status", "") or "") == "invalid_page_no":
                    findings.append(
                        {
                            "finding_type": "invalid_page_no",
                            "target_type": "answer_source",
                            "target_id": int(src.get("answer_source_id", 0) or 0),
                            "severity": "high",
                            "message": "PDF page_no가 비정상적으로 큰 saved answer source입니다.",
                        }
                    )
            flags = _safe_json_loads(str(row.get("quality_flags_json", "") or "[]"), [])
            if isinstance(flags, list):
                for flag in flags:
                    if str(flag or "") == "outside_document_claim":
                        findings.append(
                            {
                                "finding_type": "outside_document_claim",
                                "target_type": "saved_answer",
                                "target_id": saved_answer_id,
                                "severity": "medium",
                                "message": "문서 밖 주장으로 보이는 표현이 포함된 saved answer입니다.",
                            }
                        )

        for finding in findings:
            c.execute(
                """
                INSERT INTO wiki_memory_lint_findings
                    (finding_type, target_type, target_id, severity, message, status,
                     metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding["finding_type"],
                    finding["target_type"],
                    int(finding.get("target_id", 0) or 0),
                    finding["severity"],
                    finding["message"],
                    "open",
                    "{}",
                    now,
                    now,
                ),
            )
            finding["finding_id"] = int(c.lastrowid)
            finding["status"] = "open"
            finding["created_at"] = now
            finding["updated_at"] = now
        conn.commit()
        conn.close()
        return findings

    def list_lint_findings(self, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where_sql = ""
        params: List[Any] = []
        if safe_status:
            where_sql = "WHERE status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT * FROM wiki_memory_lint_findings
                {where_sql}
                ORDER BY updated_at DESC, finding_id DESC
                LIMIT ?
                """,
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def resolve_lint_finding(self, finding_id: int) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        conn = self._connect()
        conn.execute(
            "UPDATE wiki_memory_lint_findings SET status = ?, updated_at = ? WHERE finding_id = ?",
            ("resolved", now, int(finding_id)),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM wiki_memory_lint_findings WHERE finding_id = ?",
            (int(finding_id),),
        ).fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def create_conflict(
        self,
        *,
        saved_answer_id: int,
        conflicting_saved_answer_id: int,
        conflict_type: str,
        description: str,
    ) -> Dict[str, Any]:
        now = int(time.time())
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO wiki_answer_conflicts
                (saved_answer_id, conflicting_saved_answer_id, conflict_type, description, status, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(saved_answer_id),
                int(conflicting_saved_answer_id),
                str(conflict_type or "conflict"),
                str(description or ""),
                "open",
                now,
                0,
            ),
        )
        conflict_id = int(c.lastrowid)
        conn.commit()
        row = conn.execute("SELECT * FROM wiki_answer_conflicts WHERE conflict_id = ?", (conflict_id,)).fetchone()
        conn.close()
        return dict(row)

    def list_conflicts(self, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where_sql = ""
        params: List[Any] = []
        if safe_status:
            where_sql = "WHERE status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT * FROM wiki_answer_conflicts
                {where_sql}
                ORDER BY created_at DESC, conflict_id DESC
                LIMIT ?
                """,
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def resolve_conflict(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        conn = self._connect()
        conn.execute(
            "UPDATE wiki_answer_conflicts SET status = ?, resolved_at = ? WHERE conflict_id = ?",
            ("resolved", now, int(conflict_id)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM wiki_answer_conflicts WHERE conflict_id = ?", (int(conflict_id),)).fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def _open_blocked_saved_answer_ids(self, conn: sqlite3.Connection) -> set[int]:
        blocked: set[int] = set()
        for row in conn.execute(
            """
            SELECT saved_answer_id, conflicting_saved_answer_id
            FROM wiki_answer_conflicts
            WHERE COALESCE(status, 'open') = 'open'
            """
        ):
            if int(row["saved_answer_id"] or 0) > 0:
                blocked.add(int(row["saved_answer_id"]))
            if int(row["conflicting_saved_answer_id"] or 0) > 0:
                blocked.add(int(row["conflicting_saved_answer_id"]))
        for row in conn.execute(
            """
            SELECT target_id
            FROM wiki_memory_lint_findings
            WHERE COALESCE(status, 'open') = 'open'
              AND target_type IN ('saved_answer', 'answer', 'wiki_saved_answer')
            """
        ):
            if int(row["target_id"] or 0) > 0:
                blocked.add(int(row["target_id"]))
        return blocked

    def build_wiki_page_candidates(self, min_repetitions: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        blocked_ids = self._open_blocked_saved_answer_ids(conn)
        candidates: List[Dict[str, Any]] = []

        def _sources_for_saved_ids(raw_ids: Any) -> List[Dict[str, Any]]:
            ids = raw_ids if isinstance(raw_ids, list) else _safe_json_loads(str(raw_ids or "[]"), [])
            if not isinstance(ids, list):
                ids = []
            sources: List[Dict[str, Any]] = []
            for saved_id in ids:
                try:
                    safe_id = int(saved_id)
                except Exception:
                    continue
                if safe_id in blocked_ids:
                    return []
                for src in conn.execute(
                    "SELECT * FROM wiki_answer_sources WHERE saved_answer_id = ? AND COALESCE(status, '') = 'active'",
                    (safe_id,),
                ):
                    sources.append(dict(src))
            return sources

        for row in conn.execute(
            """
            SELECT * FROM wiki_answer_concepts
            WHERE COALESCE(status, '') = 'published'
            ORDER BY updated_at DESC, concept_id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 100)),),
        ):
            concept = dict(row)
            related_ids = _safe_json_loads(str(concept.get("related_saved_answers_json", "[]") or "[]"), [])
            if isinstance(related_ids, list) and len(related_ids) < max(1, int(min_repetitions or 1)):
                continue
            sources = _sources_for_saved_ids(concept.get("related_saved_answers_json", "[]"))
            if not sources:
                continue
            candidates.append(
                build_wiki_page_payload(
                    page_type="concept",
                    title=str(concept.get("concept_name", "") or "concept"),
                    body=str(concept.get("description", "") or ""),
                    claims=[{"claim_text": str(concept.get("description", "") or ""), "source_refs_json": _safe_json_dump(sources)}],
                    sources=sources,
                    provenance={"candidate_type": "concept", "candidate_id": int(concept.get("concept_id", 0) or 0)},
                )
            )

        for row in conn.execute(
            """
            SELECT * FROM wiki_answer_procedures
            WHERE COALESCE(status, '') = 'published'
            ORDER BY updated_at DESC, procedure_id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 100)),),
        ):
            procedure = dict(row)
            saved_id = int(procedure.get("related_saved_answer_id", 0) or 0)
            if saved_id in blocked_ids:
                continue
            sources = _sources_for_saved_ids([saved_id])
            if not sources:
                continue
            steps = _safe_json_loads(str(procedure.get("procedure_steps_json", "[]") or "[]"), [])
            if not isinstance(steps, list):
                steps = []
            body = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps)) or str(procedure.get("procedure_name", "") or "")
            candidates.append(
                build_wiki_page_payload(
                    page_type="procedure",
                    title=str(procedure.get("procedure_name", "") or "procedure"),
                    body=body,
                    claims=[{"claim_text": body, "source_refs_json": _safe_json_dump(sources)}],
                    sources=sources,
                    provenance={"candidate_type": "procedure", "candidate_id": int(procedure.get("procedure_id", 0) or 0)},
                )
            )

        for row in conn.execute(
            """
            SELECT * FROM wiki_answer_table_rules
            WHERE COALESCE(status, '') = 'published'
            ORDER BY updated_at DESC, rule_id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 100)),),
        ):
            rule = dict(row)
            saved_id = int(rule.get("related_saved_answer_id", 0) or 0)
            if saved_id in blocked_ids:
                continue
            sources = _sources_for_saved_ids([saved_id])
            if not sources:
                continue
            body = str(rule.get("rule_text", "") or rule.get("rule_name", "") or "")
            candidates.append(
                build_wiki_page_payload(
                    page_type="table_rule",
                    title=str(rule.get("rule_name", "") or "table rule"),
                    body=body,
                    claims=[{"claim_text": body, "source_refs_json": _safe_json_dump(sources)}],
                    sources=sources,
                    provenance={"candidate_type": "table_rule", "candidate_id": int(rule.get("rule_id", 0) or 0)},
                )
            )

        conn.close()
        return candidates[: max(1, int(limit or 100))]

    def compile_saved_answer(self, saved_answer_id: int) -> Dict[str, List[Dict[str, Any]]]:
        detail = self.get_saved_answer(int(saved_answer_id))
        if detail is None:
            raise LookupError(f"saved answer not found: {saved_answer_id}")
        compiled = compile_answer_memory(detail)
        now = int(time.time())
        conn = self._connect()
        c = conn.cursor()
        claims: List[Dict[str, Any]] = []
        for claim in compiled["claims"]:
            existing = c.execute(
                """
                SELECT * FROM wiki_answer_claims
                WHERE saved_answer_id = ? AND normalized_claim = ?
                LIMIT 1
                """,
                (int(saved_answer_id), str(claim.get("normalized_claim", ""))),
            ).fetchone()
            if existing is None:
                c.execute(
                    """
                    INSERT INTO wiki_answer_claims
                        (saved_answer_id, claim_text, normalized_claim, source_refs_json,
                         confidence_score, status, conflict_group_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(saved_answer_id),
                        str(claim.get("claim_text", "")),
                        str(claim.get("normalized_claim", "")),
                        _safe_json_dump(_safe_json_loads(str(claim.get("source_refs_json", "[]")), [])),
                        float(claim.get("confidence_score", 0.35) or 0.35),
                        str(claim.get("status", "needs_review") or "needs_review"),
                        "",
                        now,
                        now,
                    ),
                )
                claim_id = int(c.lastrowid)
                row = c.execute("SELECT * FROM wiki_answer_claims WHERE claim_id = ?", (claim_id,)).fetchone()
            else:
                row = existing
            claims.append(dict(row))

        concepts: List[Dict[str, Any]] = []
        for concept in compiled["concepts"]:
            name = str(concept.get("concept_name", "") or "").strip()[:200] or "saved answer concept"
            existing = c.execute("SELECT * FROM wiki_answer_concepts WHERE concept_name = ? LIMIT 1", (name,)).fetchone()
            related_ids = [int(saved_answer_id)]
            if existing is not None:
                related_ids = _safe_json_loads(str(existing["related_saved_answers_json"] or "[]"), [])
                if not isinstance(related_ids, list):
                    related_ids = []
                if int(saved_answer_id) not in [int(x) for x in related_ids if str(x).isdigit()]:
                    related_ids.append(int(saved_answer_id))
                c.execute(
                    """
                    UPDATE wiki_answer_concepts
                    SET related_saved_answers_json = ?, updated_at = ?
                    WHERE concept_id = ?
                    """,
                    (_safe_json_dump(related_ids), now, int(existing["concept_id"])),
                )
                row = c.execute("SELECT * FROM wiki_answer_concepts WHERE concept_id = ?", (int(existing["concept_id"]),)).fetchone()
            else:
                c.execute(
                    """
                    INSERT INTO wiki_answer_concepts
                        (concept_name, aliases_json, description, related_saved_answers_json,
                         related_sources_json, confidence_score, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        _safe_json_dump(concept.get("aliases_json", [])),
                        str(concept.get("description", "") or ""),
                        _safe_json_dump(related_ids),
                        _safe_json_dump(concept.get("related_sources_json", [])),
                        float(concept.get("confidence_score", 0.35) or 0.35),
                        str(concept.get("status", "needs_review") or "needs_review"),
                        now,
                        now,
                    ),
                )
                row = c.execute("SELECT * FROM wiki_answer_concepts WHERE concept_id = ?", (int(c.lastrowid),)).fetchone()
            concepts.append(dict(row))

        procedures: List[Dict[str, Any]] = []
        for procedure in compiled["procedures"]:
            name = str(procedure.get("procedure_name", "") or "").strip()[:200] or "saved answer procedure"
            existing = c.execute(
                """
                SELECT * FROM wiki_answer_procedures
                WHERE procedure_name = ? AND related_saved_answer_id = ?
                LIMIT 1
                """,
                (name, int(saved_answer_id)),
            ).fetchone()
            if existing is None:
                c.execute(
                    """
                    INSERT INTO wiki_answer_procedures
                        (procedure_name, procedure_steps_json, conditions_json, exceptions_json,
                         source_refs_json, related_saved_answer_id, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        _safe_json_dump(procedure.get("procedure_steps_json", [])),
                        _safe_json_dump(procedure.get("conditions_json", [])),
                        _safe_json_dump(procedure.get("exceptions_json", [])),
                        _safe_json_dump(_safe_json_loads(str(procedure.get("source_refs_json", "[]")), [])),
                        int(saved_answer_id),
                        str(procedure.get("status", "needs_review") or "needs_review"),
                        now,
                        now,
                    ),
                )
                row = c.execute("SELECT * FROM wiki_answer_procedures WHERE procedure_id = ?", (int(c.lastrowid),)).fetchone()
            else:
                row = existing
            procedures.append(dict(row))

        table_rules: List[Dict[str, Any]] = []
        for rule in compiled["table_rules"]:
            name = str(rule.get("rule_name", "") or "").strip()[:200] or "saved answer table rule"
            existing = c.execute(
                """
                SELECT * FROM wiki_answer_table_rules
                WHERE rule_name = ? AND related_saved_answer_id = ?
                LIMIT 1
                """,
                (name, int(saved_answer_id)),
            ).fetchone()
            if existing is None:
                c.execute(
                    """
                    INSERT INTO wiki_answer_table_rules
                        (rule_name, rule_text, source_path, sheet_name, table_range,
                         row_refs_json, column_refs_json, related_saved_answer_id, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        str(rule.get("rule_text", "") or ""),
                        str(rule.get("source_path", "") or ""),
                        str(rule.get("sheet_name", "") or ""),
                        str(rule.get("table_range", "") or ""),
                        _safe_json_dump(rule.get("row_refs_json", [])),
                        _safe_json_dump(rule.get("column_refs_json", [])),
                        int(saved_answer_id),
                        str(rule.get("status", "needs_review") or "needs_review"),
                        now,
                        now,
                    ),
                )
                row = c.execute("SELECT * FROM wiki_answer_table_rules WHERE rule_id = ?", (int(c.lastrowid),)).fetchone()
            else:
                row = existing
            table_rules.append(dict(row))
        conn.commit()
        conn.close()
        return {"claims": claims, "concepts": concepts, "procedures": procedures, "table_rules": table_rules}

    def list_concepts(self, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_table("wiki_answer_concepts", "concept_id", status=status, limit=limit)

    def list_procedures(self, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_table("wiki_answer_procedures", "procedure_id", status=status, limit=limit)

    def list_table_rules(self, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_table("wiki_answer_table_rules", "rule_id", status=status, limit=limit)

    def _list_table(self, table: str, pk: str, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where_sql = ""
        params: List[Any] = []
        if safe_status:
            where_sql = "WHERE status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {table} {where_sql} ORDER BY updated_at DESC, {pk} DESC LIMIT ?",
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def retrieval_boost_targets(self) -> Dict[str, Dict[Any, int]]:
        conn = self._connect()
        chunks: Dict[int, int] = {}
        sources: Dict[str, int] = {}
        table_cells: Dict[int, int] = {}
        for row in conn.execute(
            """
            SELECT a.reused_count, s.source_path, s.chunk_id, s.table_cell_id, s.status AS source_status
            FROM wiki_saved_answers a
            JOIN wiki_answer_sources s ON s.saved_answer_id = a.saved_answer_id
            WHERE a.status = 'published'
              AND COALESCE(a.reused_count, 0) > 0
              AND COALESCE(s.status, '') = 'active'
            """
        ):
            weight = max(1, int(row["reused_count"] or 0))
            chunk_id = int(row["chunk_id"] or 0)
            table_cell_id = int(row["table_cell_id"] or 0)
            source_path = str(row["source_path"] or "")
            if chunk_id > 0:
                chunks[chunk_id] = chunks.get(chunk_id, 0) + weight
            if table_cell_id > 0:
                table_cells[table_cell_id] = table_cells.get(table_cell_id, 0) + weight
            if source_path:
                sources[source_path] = sources.get(source_path, 0) + weight
        table_names = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if {"wiki_pages", "wiki_page_sources"}.issubset(table_names):
            for row in conn.execute(
                """
                SELECT s.source_path, s.chunk_id
                FROM wiki_pages p
                JOIN wiki_page_sources s ON s.page_id = p.page_id
                WHERE COALESCE(p.status, '') = 'published'
                """
            ):
                chunk_id = int(row["chunk_id"] or 0)
                source_path = str(row["source_path"] or "")
                if chunk_id > 0:
                    chunks[chunk_id] = chunks.get(chunk_id, 0) + 1
                if source_path:
                    sources[source_path] = sources.get(source_path, 0) + 1
        conn.close()
        return {"chunks": chunks, "table_cells": table_cells, "sources": sources}

    def export_markdown(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        index_lines = ["# Wiki Answer Memory", ""]
        for row in self.list_saved_answers(limit=1000):
            saved_answer_id = int(row.get("saved_answer_id", 0) or 0)
            title = str(row.get("question_text", "") or f"saved-answer-{saved_answer_id}").strip()
            slug = f"wiki/saved_answers/{saved_answer_id}.md"
            index_lines.append(f"- [{title}]({saved_answer_id}.md)")
            detail = self.get_saved_answer(saved_answer_id) or row
            body = [
                "---",
                f"id: {saved_answer_id}",
                f"status: {detail.get('status', '')}",
                f"source_count: {detail.get('source_count', 0)}",
                "---",
                "",
                f"# {title}",
                "",
                "## Answer",
                str(detail.get("answer_text", "") or detail.get("answer_summary", "") or "").strip(),
            ]
            sources = detail.get("sources", []) if isinstance(detail, dict) else []
            if sources:
                body.extend(["", "## Sources"])
                for source in sources:
                    ref = str(source.get("source_ref", "") or source.get("source_path", "") or "").strip()
                    if ref:
                        body.append(f"- {ref}")
            files[slug] = "\n".join(body).strip() + "\n"
        files["wiki/saved_answers/index.md"] = "\n".join(index_lines).strip() + "\n"
        concept_lines = ["# Wiki Concepts", ""]
        for concept in self.list_concepts(limit=1000):
            concept_lines.append(f"- {concept.get('concept_name', '')} ({concept.get('status', '')})")
        files["wiki/concepts/index.md"] = "\n".join(concept_lines).strip() + "\n"
        procedure_lines = ["# Wiki Procedures", ""]
        for procedure in self.list_procedures(limit=1000):
            procedure_lines.append(f"- {procedure.get('procedure_name', '')} ({procedure.get('status', '')})")
        files["wiki/procedures/index.md"] = "\n".join(procedure_lines).strip() + "\n"
        rule_lines = ["# Wiki Table Rules", ""]
        for rule in self.list_table_rules(limit=1000):
            rule_lines.append(f"- {rule.get('rule_name', '')} ({rule.get('status', '')})")
        files["wiki/table_rules/index.md"] = "\n".join(rule_lines).strip() + "\n"
        files["wiki/review_queue.md"] = "# Wiki Review Queue\n\n" + "\n".join(
            f"- {row.get('question_text', '')} ({row.get('status', '')})"
            for row in self.list_saved_answers(status="needs_review", limit=1000)
        ).strip() + "\n"
        summary = self.quality_summary()
        files["wiki/overview.md"] = (
            "# Wiki Overview\n\n"
            f"- status_counts: {_safe_json_dump(summary.get('status_counts', {}))}\n"
            f"- quality_flag_counts: {_safe_json_dump(summary.get('quality_flag_counts', {}))}\n"
            f"- usage_count: {int(summary.get('usage_count', 0) or 0)}\n"
        )
        lint_lines = ["# Wiki Lint", ""]
        for finding in self.list_lint_findings(limit=1000):
            lint_lines.append(f"- {finding.get('severity', '')}: {finding.get('message', '')} [{finding.get('status', '')}]")
        files["wiki/lint.md"] = "\n".join(lint_lines).strip() + "\n"
        conflict_lines = ["# Wiki Conflicts", ""]
        for conflict in self.list_conflicts(limit=1000):
            conflict_lines.append(f"- {conflict.get('conflict_type', '')}: {conflict.get('description', '')} [{conflict.get('status', '')}]")
        files["wiki/conflicts.md"] = "\n".join(conflict_lines).strip() + "\n"
        return files
