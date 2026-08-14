from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.ontology_extractor import (
    coerce_limited_llm_facts_from_chunk,
    extract_deterministic_facts_from_text,
    normalize_entity_key,
    normalize_entity_text,
)
from src.ontology_alias_registry import expand_ontology_query_tokens
from src.ontology_quality import score_ontology_candidate


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "[]"


def _safe_json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text or "")) if len(token) >= 2]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _contains_compact(haystack: str, needle: str) -> bool:
    compact_haystack = _compact(haystack)
    compact_needle = _compact(needle)
    return bool(compact_needle and compact_needle in compact_haystack)


def _is_duplicate_column_error(exc: sqlite3.OperationalError) -> bool:
    return "duplicate column name" in str(exc).lower()


class OntologyStore:
    def __init__(self, db_path: str, kb_id: str = ""):
        self.db_path = str(db_path)
        self.kb_id = str(kb_id or "")
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
            CREATE TABLE IF NOT EXISTS ontology_entities
            (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT,
                normalized_key TEXT,
                display_text TEXT,
                entity_type TEXT,
                aliases_json TEXT,
                confidence REAL,
                status TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                UNIQUE(kb_id, normalized_key)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_relations
            (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT,
                relation_key TEXT,
                label TEXT,
                inverse_key TEXT,
                domain_type TEXT,
                range_type TEXT,
                extraction_policy TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                UNIQUE(kb_id, relation_key)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_facts
            (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT,
                subject_entity_id INTEGER,
                predicate TEXT,
                object_entity_id INTEGER,
                object_value TEXT,
                fact_kind TEXT,
                extraction_method TEXT,
                confidence REAL,
                status TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_fact_sources
            (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                chunk_id INTEGER,
                source_path TEXT,
                source_ref TEXT,
                page_no INTEGER,
                line_start INTEGER,
                line_end INTEGER,
                table_cell_id INTEGER,
                created_at INTEGER
            )
            """
        )
        for statement in (
            "ALTER TABLE ontology_fact_sources ADD COLUMN evidence_quote TEXT DEFAULT ''",
            "ALTER TABLE ontology_fact_sources ADD COLUMN evidence_span_json TEXT DEFAULT ''",
        ):
            try:
                c.execute(statement)
            except sqlite3.OperationalError as exc:
                if not _is_duplicate_column_error(exc):
                    raise
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_fact_feedback
            (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT,
                fact_id INTEGER,
                feedback_key TEXT,
                signal TEXT,
                created_at INTEGER,
                UNIQUE(kb_id, fact_id, feedback_key)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_fact_history
            (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT,
                fact_id INTEGER,
                signal TEXT,
                previous_status TEXT,
                new_status TEXT,
                previous_confidence REAL,
                new_confidence REAL,
                source TEXT,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_query_logs
            (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                matched_fact_ids_json TEXT,
                meta_json TEXT,
                created_at INTEGER
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_entities_key ON ontology_entities(kb_id, normalized_key)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_facts_subject ON ontology_facts(kb_id, subject_entity_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_facts_predicate ON ontology_facts(kb_id, predicate)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_fact_sources_chunk ON ontology_fact_sources(chunk_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_feedback_fact ON ontology_fact_feedback(kb_id, fact_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ontology_history_fact ON ontology_fact_history(kb_id, fact_id, history_id)")
        conn.commit()
        conn.close()

    def _upsert_entity(
        self,
        conn: sqlite3.Connection,
        text: str,
        *,
        aliases: Optional[List[str]] = None,
        entity_type: str = "concept",
        confidence: float = 0.70,
    ) -> int:
        display = normalize_entity_text(text)
        key = normalize_entity_key(display)
        if not key:
            raise ValueError("entity text is required")
        clean_aliases: List[str] = []
        seen = set()
        for alias in [display, *(aliases or [])]:
            normalized = normalize_entity_text(alias)
            alias_key = normalize_entity_key(normalized)
            if not normalized or alias_key in seen:
                continue
            seen.add(alias_key)
            clean_aliases.append(normalized)
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO ontology_entities
                (kb_id, normalized_key, display_text, entity_type, aliases_json, confidence, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kb_id, normalized_key) DO UPDATE SET
                display_text = excluded.display_text,
                aliases_json = excluded.aliases_json,
                confidence = MAX(ontology_entities.confidence, excluded.confidence),
                status = CASE
                    WHEN ontology_entities.status = 'archived' THEN ontology_entities.status
                    ELSE excluded.status
                END,
                updated_at = excluded.updated_at
            """,
            (
                self.kb_id,
                key,
                display,
                entity_type,
                _safe_json_dump(clean_aliases),
                float(confidence),
                "active",
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT entity_id FROM ontology_entities WHERE kb_id = ? AND normalized_key = ?",
            (self.kb_id, key),
        ).fetchone()
        return int(row["entity_id"])

    def _upsert_relation(self, conn: sqlite3.Connection, predicate: str) -> None:
        relation_key = normalize_entity_key(predicate)
        if not relation_key:
            return
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO ontology_relations
                (kb_id, relation_key, label, inverse_key, domain_type, range_type, extraction_policy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kb_id, relation_key) DO UPDATE SET
                label = excluded.label,
                updated_at = excluded.updated_at
            """,
            (self.kb_id, relation_key, predicate, "", "concept", "literal_or_concept", "deterministic_or_limited_llm", now, now),
        )

    def _delete_facts_for_chunks(self, conn: sqlite3.Connection, chunk_ids: List[int]) -> int:
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT DISTINCT fact_id FROM ontology_fact_sources WHERE chunk_id IN ({placeholders})",
            tuple(int(chunk_id) for chunk_id in chunk_ids),
        ).fetchall()
        fact_ids = [int(row["fact_id"]) for row in rows]
        if not fact_ids:
            return 0
        fact_placeholders = ",".join("?" for _ in fact_ids)
        conn.execute(
            f"DELETE FROM ontology_fact_sources WHERE fact_id IN ({fact_placeholders})",
            tuple(fact_ids),
        )
        conn.execute(
            f"DELETE FROM ontology_facts WHERE fact_id IN ({fact_placeholders})",
            tuple(fact_ids),
        )
        return len(fact_ids)

    def sync_facts_for_chunks(
        self,
        changed_chunk_ids: List[int],
        deleted_chunk_ids: List[int],
        *,
        llm_payloads_by_chunk: Optional[Dict[int, Any]] = None,
        min_confidence: float = 0.62,
        llm_fact_status: str = "",
    ) -> Dict[str, int]:
        changed_ids = [int(chunk_id) for chunk_id in changed_chunk_ids or [] if int(chunk_id) > 0]
        deleted_ids = [int(chunk_id) for chunk_id in deleted_chunk_ids or [] if int(chunk_id) > 0]
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        deleted_count = self._delete_facts_for_chunks(conn, [*changed_ids, *deleted_ids])
        inserted_count = 0
        if changed_ids:
            placeholders = ",".join("?" for _ in changed_ids)
            chunk_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            chunk_kind_sql = "chunk_kind" if "chunk_kind" in chunk_columns else "'' AS chunk_kind"
            heading_path_sql = "heading_path_json" if "heading_path_json" in chunk_columns else "'[]' AS heading_path_json"
            is_derived_sql = "is_derived" if "is_derived" in chunk_columns else "0 AS is_derived"
            rows = conn.execute(
                f"""
                SELECT id, source_path, source_type, page_no, line_start, line_end, section, text,
                       {chunk_kind_sql}, {heading_path_sql}, {is_derived_sql}
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                tuple(changed_ids),
            ).fetchall()
            now = int(time.time())
            for row in rows:
                chunk_id = int(row["id"])
                try:
                    heading_path = json.loads(str(row["heading_path_json"] or "[]"))
                except Exception:
                    heading_path = []
                deterministic_facts = extract_deterministic_facts_from_text(
                    str(row["text"] or ""),
                    chunk_kind=str(row["chunk_kind"] or ""),
                    heading_path=heading_path,
                    is_derived=bool(row["is_derived"] or 0),
                )
                llm_facts = coerce_limited_llm_facts_from_chunk(
                    (llm_payloads_by_chunk or {}).get(chunk_id, []),
                    str(row["text"] or ""),
                    min_confidence=float(min_confidence),
                )
                forced_llm_status = normalize_entity_text(llm_fact_status).lower()
                if forced_llm_status in {"active", "published", "needs_review", "archived"}:
                    llm_facts = [{**fact, "status": forced_llm_status} for fact in llm_facts]
                for fact in [*deterministic_facts, *llm_facts]:
                    subject_id = self._upsert_entity(
                        conn,
                        str(fact.get("subject", "") or ""),
                        aliases=list(fact.get("subject_aliases", []) or []),
                        confidence=float(fact.get("confidence", 0.70) or 0.70),
                    )
                    object_entity_text = str(fact.get("object_entity", "") or "")
                    object_entity_id = 0
                    if object_entity_text:
                        object_entity_id = self._upsert_entity(
                            conn,
                            object_entity_text,
                            aliases=list(fact.get("object_aliases", []) or []),
                            confidence=float(fact.get("confidence", 0.70) or 0.70),
                        )
                    predicate = normalize_entity_text(str(fact.get("predicate", "") or ""))
                    if not predicate:
                        continue
                    self._upsert_relation(conn, predicate)
                    conn.execute(
                        """
                        INSERT INTO ontology_facts
                            (kb_id, subject_entity_id, predicate, object_entity_id, object_value,
                             fact_kind, extraction_method, confidence, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.kb_id,
                            subject_id,
                            predicate,
                            object_entity_id,
                            normalize_entity_text(str(fact.get("object_value", "") or "")),
                            normalize_entity_text(str(fact.get("fact_kind", "") or "")),
                            normalize_entity_text(str(fact.get("extraction_method", "") or "")),
                            float(fact.get("confidence", 0.70) or 0.70),
                            normalize_entity_text(str(fact.get("status", "active") or "active")) or "active",
                            now,
                            now,
                        ),
                    )
                    fact_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                    conn.execute(
                        """
                        INSERT INTO ontology_fact_sources
                            (fact_id, chunk_id, source_path, source_ref, page_no, line_start, line_end,
                             table_cell_id, evidence_quote, evidence_span_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fact_id,
                            chunk_id,
                            str(row["source_path"] or ""),
                            str(row["section"] or row["source_path"] or ""),
                            int(row["page_no"] or 0),
                            int(row["line_start"] or 0),
                            int(row["line_end"] or 0),
                            0,
                            normalize_entity_text(str(fact.get("evidence_quote", "") or "")),
                            _safe_json_dump(fact.get("evidence_span", [])),
                            now,
                        ),
                    )
                    inserted_count += 1
        conn.commit()
        conn.close()
        return {
            "ontology_facts_added": int(inserted_count),
            "ontology_facts_deleted": int(deleted_count),
        }

    def _expand_query_terms(self, query_text: str) -> Dict[str, Any]:
        terms = expand_ontology_query_tokens(_tokens(query_text))
        if not terms:
            terms = [query_text]
        seen: set[str] = set()
        normalized_terms: List[str] = []
        for term in terms:
            clean = normalize_entity_text(term)
            key = normalize_entity_key(clean)
            if not clean or key in seen:
                continue
            seen.add(key)
            normalized_terms.append(clean)
        rewritten = " ".join(normalized_terms)
        return {
            "terms": normalized_terms,
            "rewritten_query": rewritten,
        }

    def search_facts(
        self,
        query: str,
        limit: int = 20,
        max_hops: int = 2,
        min_confidence: float = 0.62,
        allowed_extraction_methods: Optional[set[str]] = None,
        experiment_mode: str = "runtime",
    ) -> List[Dict[str, Any]]:
        search_started = time.perf_counter()
        query_text = str(query or "")
        normalized_query = self._expand_query_terms(query_text)
        rewritten_query = str(normalized_query.get("rewritten_query") or query_text)
        query_tokens = _tokens(rewritten_query)
        query_compact = _compact(rewritten_query)
        if not query_tokens and not query_compact:
            return []
        max_hops = max(1, min(2, int(max_hops or 1)))
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT
                f.fact_id,
                f.predicate,
                f.object_value,
                f.fact_kind,
                f.extraction_method,
                f.confidence,
                f.status,
                s.chunk_id,
                s.source_path,
                s.source_ref,
                s.page_no,
                e.display_text AS subject,
                e.aliases_json AS subject_aliases,
                oe.display_text AS object_entity
            FROM ontology_facts f
            JOIN ontology_entities e ON e.entity_id = f.subject_entity_id
            JOIN ontology_fact_sources s ON s.fact_id = f.fact_id
            LEFT JOIN ontology_entities oe ON oe.entity_id = f.object_entity_id
            WHERE f.kb_id = ?
              AND COALESCE(f.status, '') IN ('active', 'published')
            """,
            (self.kb_id,),
        ).fetchall()
        entity_links: Dict[str, set[str]] = {}
        for row in rows:
            subject = str(row["subject"] or "")
            object_entity = str(row["object_entity"] or "")
            predicate = str(row["predicate"] or "")
            if predicate == "상위개념" and subject and object_entity:
                entity_links.setdefault(normalize_entity_key(object_entity), set()).add(subject)
        matches: List[Dict[str, Any]] = []
        hit_totals = {
            "direct_hits": 0,
            "predicate_hits": 0,
            "value_hits": 0,
            "relation_hits": 0,
        }
        for row in rows:
            status = str(row["status"] or "")
            confidence = float(row["confidence"] or 0.0)
            extraction_method = str(row["extraction_method"] or "")
            if allowed_extraction_methods is not None and extraction_method not in allowed_extraction_methods:
                continue
            if status != "published" and confidence < float(min_confidence):
                continue
            aliases = _safe_json_loads(str(row["subject_aliases"] or "[]"), [])
            subject = str(row["subject"] or "")
            predicate = str(row["predicate"] or "")
            object_value = str(row["object_value"] or "")
            object_entity = str(row["object_entity"] or "")
            searchable_parts = [subject, predicate, object_value, object_entity, *[str(a) for a in aliases if a]]
            direct_hits = 0
            predicate_hits = 0
            value_hits = 0
            relation_hits = 0
            for part in searchable_parts:
                if not part:
                    continue
                if _contains_compact(query_compact, part):
                    if part == subject or part in aliases:
                        direct_hits += 2
                    elif part == predicate:
                        predicate_hits += 1
                    else:
                        value_hits += 1
            part_text = " ".join(searchable_parts).lower()
            for token in query_tokens:
                if token in part_text:
                    if token in _compact(subject) or any(token in _compact(alias) for alias in aliases):
                        direct_hits += 1
                    elif token in _compact(predicate):
                        predicate_hits += 1
                    else:
                        value_hits += 1
            if max_hops >= 2 and direct_hits <= 0:
                linked_subjects: set[str] = set()
                for token in query_tokens:
                    linked_subjects.update(entity_links.get(normalize_entity_key(token), set()))
                for linked in linked_subjects:
                    if normalize_entity_key(linked) == normalize_entity_key(subject):
                        relation_hits += 1
                        break
            scored = score_ontology_candidate(
                direct_hits=direct_hits,
                predicate_hits=predicate_hits,
                value_hits=value_hits,
                relation_hits=relation_hits,
                confidence=confidence,
                max_hops=max_hops,
            )
            if scored is None:
                continue
            hit_totals["direct_hits"] += direct_hits
            hit_totals["predicate_hits"] += predicate_hits
            hit_totals["value_hits"] += value_hits
            hit_totals["relation_hits"] += relation_hits
            hop_count = int(scored["hop_count"])
            score = float(scored["score"])
            matches.append(
                {
                    "fact_id": int(row["fact_id"]),
                    "subject": subject,
                    "predicate": predicate,
                    "object_value": object_value,
                    "object_entity": object_entity,
                    "fact_kind": str(row["fact_kind"] or ""),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "status": status,
                    "chunk_id": int(row["chunk_id"] or 0),
                    "source_path": str(row["source_path"] or ""),
                    "source_ref": str(row["source_ref"] or ""),
                    "page_no": int(row["page_no"] or 0),
                    "score": float(score),
                    "label": f"{subject} --{predicate}--> {object_entity or object_value}",
                    "ontology_query_rewrite": rewritten_query,
                    "ontology_hop_count": hop_count,
                    "ontology_candidate_reason": str(scored["reason"]),
                }
            )
        matches.sort(key=lambda item: (float(item["score"]), int(item["fact_id"])), reverse=True)
        limited = matches[: max(1, int(limit or 20))]
        search_latency_ms = (time.perf_counter() - search_started) * 1000.0
        conn.execute(
            """
            INSERT INTO ontology_query_logs (query_text, matched_fact_ids_json, meta_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                query_text,
                _safe_json_dump([int(item["fact_id"]) for item in limited]),
                _safe_json_dump({
                    "match_count": len(limited),
                    "ontology_query_rewrite": rewritten_query,
                    "max_hops": max_hops,
                    "experiment_mode": str(experiment_mode or "runtime"),
                    "allowed_extraction_methods": sorted(allowed_extraction_methods) if allowed_extraction_methods is not None else [],
                    "returned_fact_ids": [int(item["fact_id"]) for item in limited],
                    "returned_chunk_ids": [int(item["chunk_id"]) for item in limited],
                    "hit_totals": hit_totals,
                    "search_latency_ms": search_latency_ms,
                }),
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()
        return limited

    def apply_wiki_signal(
        self,
        *,
        chunk_ids: List[int],
        signal: str,
        boost: float = 0.08,
        feedback_key: str = "",
    ) -> int:
        safe_chunk_ids = [int(chunk_id) for chunk_id in chunk_ids or [] if int(chunk_id) > 0]
        if not safe_chunk_ids:
            return 0
        placeholders = ",".join("?" for _ in safe_chunk_ids)
        conn = self._connect()
        rows = conn.execute(
            f"SELECT DISTINCT fact_id FROM ontology_fact_sources WHERE chunk_id IN ({placeholders})",
            tuple(safe_chunk_ids),
        ).fetchall()
        fact_ids = [int(row["fact_id"]) for row in rows]
        if not fact_ids:
            conn.close()
            return 0
        if feedback_key:
            fresh_fact_ids: List[int] = []
            now = int(time.time())
            for fact_id in fact_ids:
                try:
                    conn.execute(
                        """
                        INSERT INTO ontology_fact_feedback (kb_id, fact_id, feedback_key, signal, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (self.kb_id, int(fact_id), str(feedback_key), str(signal), now),
                    )
                    fresh_fact_ids.append(int(fact_id))
                except sqlite3.IntegrityError:
                    pass
            if not fresh_fact_ids:
                conn.commit()
                conn.close()
                return 0
            fact_ids = fresh_fact_ids
        fact_placeholders = ",".join("?" for _ in fact_ids)
        now = int(time.time())
        previous_rows = {
            int(row["fact_id"]): (str(row["status"] or ""), float(row["confidence"] or 0.0))
            for row in conn.execute(
                f"SELECT fact_id, status, confidence FROM ontology_facts WHERE fact_id IN ({fact_placeholders})",
                tuple(fact_ids),
            )
        }
        if signal == "published":
            conn.execute(
                f"""
                UPDATE ontology_facts
                SET confidence = MIN(1.0, COALESCE(confidence, 0) + ?),
                    status = 'published',
                    updated_at = ?
                WHERE fact_id IN ({fact_placeholders})
                """,
                (float(boost), now, *fact_ids),
            )
        elif signal in {"reported", "needs_review"}:
            conn.execute(
                f"""
                UPDATE ontology_facts
                SET confidence = MAX(0.0, COALESCE(confidence, 0) - ?),
                    status = 'needs_review',
                    updated_at = ?
                WHERE fact_id IN ({fact_placeholders})
                """,
                (float(boost), now, *fact_ids),
            )
        updated_rows = {
            int(row["fact_id"]): (str(row["status"] or ""), float(row["confidence"] or 0.0))
            for row in conn.execute(
                f"SELECT fact_id, status, confidence FROM ontology_facts WHERE fact_id IN ({fact_placeholders})",
                tuple(fact_ids),
            )
        }
        for fact_id in fact_ids:
            previous = previous_rows.get(int(fact_id))
            updated = updated_rows.get(int(fact_id))
            if previous is None or updated is None or previous == updated:
                continue
            conn.execute(
                """
                INSERT INTO ontology_fact_history
                    (kb_id, fact_id, signal, previous_status, new_status,
                     previous_confidence, new_confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.kb_id,
                    int(fact_id),
                    str(signal),
                    previous[0],
                    updated[0],
                    previous[1],
                    updated[1],
                    "wiki_feedback",
                    now,
                ),
            )
        conn.commit()
        conn.close()
        return len(fact_ids)

    def get_fact_detail(self, fact_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                f.fact_id,
                e.display_text AS subject,
                f.predicate,
                oe.display_text AS object_entity,
                f.object_value,
                f.fact_kind,
                f.extraction_method,
                f.confidence,
                f.status,
                f.created_at,
                f.updated_at
            FROM ontology_facts f
            JOIN ontology_entities e ON e.entity_id = f.subject_entity_id
            LEFT JOIN ontology_entities oe ON oe.entity_id = f.object_entity_id
            WHERE f.kb_id = ? AND f.fact_id = ?
            """,
            (self.kb_id, int(fact_id)),
        ).fetchone()
        if row is None:
            conn.close()
            return None
        sources = [
            dict(source)
            for source in conn.execute(
                """
                SELECT chunk_id, source_path, source_ref, page_no, line_start, line_end,
                       table_cell_id, evidence_quote, evidence_span_json
                FROM ontology_fact_sources
                WHERE fact_id = ?
                ORDER BY source_id ASC
                """,
                (int(fact_id),),
            )
        ]
        history = [
            dict(history_row)
            for history_row in conn.execute(
                """
                SELECT history_id, signal, previous_status, new_status,
                       previous_confidence, new_confidence, source, created_at
                FROM ontology_fact_history
                WHERE kb_id = ? AND fact_id = ?
                ORDER BY history_id DESC
                """,
                (self.kb_id, int(fact_id)),
            )
        ]
        conn.close()
        detail = dict(row)
        detail["sources"] = sources
        detail["history"] = history
        return detail

    def list_entities(self, *, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where = "WHERE kb_id = ?"
        params: List[Any] = [self.kb_id]
        if safe_status:
            where += " AND status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT entity_id, normalized_key, display_text, entity_type, aliases_json,
                       confidence, status, created_at, updated_at
                FROM ontology_entities
                {where}
                ORDER BY updated_at DESC, entity_id DESC
                LIMIT ?
                """,
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def list_facts(self, *, status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        safe_status = str(status or "").strip()
        where = "WHERE f.kb_id = ?"
        params: List[Any] = [self.kb_id]
        if safe_status:
            where += " AND f.status = ?"
            params.append(safe_status)
        params.append(max(1, int(limit or 100)))
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT
                    f.fact_id,
                    e.display_text AS subject,
                    f.predicate,
                    oe.display_text AS object_entity,
                    f.object_value,
                    f.fact_kind,
                    f.extraction_method,
                    f.confidence,
                    f.status,
                    f.created_at,
                    f.updated_at
                FROM ontology_facts f
                JOIN ontology_entities e ON e.entity_id = f.subject_entity_id
                LEFT JOIN ontology_entities oe ON oe.entity_id = f.object_entity_id
                {where}
                ORDER BY f.updated_at DESC, f.fact_id DESC
                LIMIT ?
                """,
                tuple(params),
            )
        ]
        conn.close()
        return rows

    def overview(self) -> Dict[str, Any]:
        conn = self._connect()
        entity_count = int(
            conn.execute("SELECT COUNT(*) FROM ontology_entities WHERE kb_id = ?", (self.kb_id,)).fetchone()[0]
        )
        fact_count = int(
            conn.execute("SELECT COUNT(*) FROM ontology_facts WHERE kb_id = ?", (self.kb_id,)).fetchone()[0]
        )
        relation_count = int(
            conn.execute("SELECT COUNT(*) FROM ontology_relations WHERE kb_id = ?", (self.kb_id,)).fetchone()[0]
        )
        status_counts = {
            str(row["status"] or ""): int(row["count"] or 0)
            for row in conn.execute(
                """
                SELECT COALESCE(status, '') AS status, COUNT(*) AS count
                FROM ontology_facts
                WHERE kb_id = ?
                GROUP BY COALESCE(status, '')
                """,
                (self.kb_id,),
            )
        }
        top_relations = [
            {"predicate": str(row["predicate"] or ""), "count": int(row["count"] or 0)}
            for row in conn.execute(
                """
                SELECT predicate, COUNT(*) AS count
                FROM ontology_facts
                WHERE kb_id = ?
                GROUP BY predicate
                ORDER BY count DESC, predicate ASC
                LIMIT 20
                """,
                (self.kb_id,),
            )
        ]
        conn.close()
        return {
            "entity_count": entity_count,
            "fact_count": fact_count,
            "relation_count": relation_count,
            "fact_status_counts": status_counts,
            "top_relations": top_relations,
        }

    def rebuild_from_chunks(self) -> Dict[str, int]:
        conn = self._connect()
        rows = conn.execute("SELECT id FROM chunks WHERE COALESCE(is_normalized, 0) = 0 ORDER BY id ASC").fetchall()
        chunk_ids = [int(row["id"]) for row in rows]
        conn.close()
        return self.sync_facts_for_chunks(chunk_ids, chunk_ids)

    def update_fact_status(self, fact_id: int, status: str, *, source: str = "admin") -> Optional[Dict[str, Any]]:
        safe_status = str(status or "").strip()
        if safe_status not in {"active", "published", "needs_review", "archived"}:
            raise ValueError(f"unsupported ontology fact status: {safe_status}")
        now = int(time.time())
        conn = self._connect()
        previous = conn.execute(
            "SELECT status, confidence FROM ontology_facts WHERE kb_id = ? AND fact_id = ?",
            (self.kb_id, int(fact_id)),
        ).fetchone()
        conn.execute(
            """
            UPDATE ontology_facts
            SET status = ?,
                confidence = CASE
                    WHEN ? = 'published' THEN MIN(1.0, COALESCE(confidence, 0) + 0.08)
                    WHEN ? IN ('needs_review', 'archived') THEN MAX(0.0, COALESCE(confidence, 0) - 0.08)
                    ELSE confidence
                END,
                updated_at = ?
            WHERE kb_id = ? AND fact_id = ?
            """,
            (safe_status, safe_status, safe_status, now, self.kb_id, int(fact_id)),
        )
        updated = conn.execute(
            "SELECT status, confidence FROM ontology_facts WHERE kb_id = ? AND fact_id = ?",
            (self.kb_id, int(fact_id)),
        ).fetchone()
        if previous is not None and updated is not None:
            previous_status = str(previous["status"] or "")
            new_status = str(updated["status"] or "")
            previous_confidence = float(previous["confidence"] or 0.0)
            new_confidence = float(updated["confidence"] or 0.0)
            if (previous_status, previous_confidence) != (new_status, new_confidence):
                conn.execute(
                    """
                    INSERT INTO ontology_fact_history
                        (kb_id, fact_id, signal, previous_status, new_status,
                         previous_confidence, new_confidence, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.kb_id,
                        int(fact_id),
                        safe_status,
                        previous_status,
                        new_status,
                        previous_confidence,
                        new_confidence,
                        str(source or "admin"),
                        now,
                    ),
                )
        conn.commit()
        row = conn.execute(
            """
            SELECT
                f.fact_id,
                e.display_text AS subject,
                f.predicate,
                oe.display_text AS object_entity,
                f.object_value,
                f.fact_kind,
                f.extraction_method,
                f.confidence,
                f.status,
                f.created_at,
                f.updated_at
            FROM ontology_facts f
            JOIN ontology_entities e ON e.entity_id = f.subject_entity_id
            LEFT JOIN ontology_entities oe ON oe.entity_id = f.object_entity_id
            WHERE f.kb_id = ? AND f.fact_id = ?
            """,
            (self.kb_id, int(fact_id)),
        ).fetchone()
        conn.close()
        return dict(row) if row is not None else None
