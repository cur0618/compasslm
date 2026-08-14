import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Mapping


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


def _clean_slug(value: str) -> str:
    raw = re.sub(r"\\+", "/", str(value or "").strip())
    raw = re.sub(r"/+", "/", raw).strip("/")
    if not raw:
        return "index"
    safe_parts = []
    for part in raw.split("/"):
        safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "-", part).strip("-._")
        safe_parts.append(safe or "page")
    return "/".join(safe_parts)


def _clean_title_from_path(value: str) -> str:
    basename = os.path.basename(str(value or "").strip())
    stem, _ext = os.path.splitext(basename)
    return stem or basename or "문서"


def _markdown_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value or "").replace("\n", " ").replace(":", "-").strip()
    return text


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
    if column not in {str(row[1]) for row in rows}:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


class WikiStore:
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
            CREATE TABLE IF NOT EXISTS wiki_pages
            (
                page_id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE,
                title TEXT,
                page_type TEXT,
                body TEXT,
                metadata_json TEXT,
                status TEXT DEFAULT 'draft',
                provenance_json TEXT DEFAULT '{}',
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_links
            (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_page_id INTEGER,
                to_slug TEXT,
                link_type TEXT,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_claims
            (
                claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER,
                claim_text TEXT,
                status TEXT,
                citations_json TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_page_sources
            (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER,
                source_path TEXT,
                source_ref TEXT,
                page_no INTEGER,
                chunk_id INTEGER,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_update_log
            (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                page_id INTEGER,
                message TEXT,
                metadata_json TEXT,
                created_at INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_lint_findings
            (
                finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_type TEXT,
                page_id INTEGER,
                severity TEXT,
                message TEXT,
                status TEXT,
                metadata_json TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_pages_type ON wiki_pages(page_type, updated_at)")
        _ensure_column(c, "wiki_pages", "status", "TEXT DEFAULT 'draft'")
        _ensure_column(c, "wiki_pages", "provenance_json", "TEXT DEFAULT '{}'")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_pages_status ON wiki_pages(status, updated_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_claims_page ON wiki_claims(page_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wiki_sources_page ON wiki_page_sources(page_id)")
        conn.commit()
        conn.close()

    def upsert_page(
        self,
        *,
        slug: str,
        title: str,
        page_type: str,
        body: str,
        metadata: Mapping[str, Any] | None = None,
        status: str = "draft",
        provenance: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        now = int(time.time())
        safe_slug = _clean_slug(slug)
        safe_status = str(status or "draft").strip() or "draft"
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO wiki_pages (slug, title, page_type, body, metadata_json, status, provenance_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title = excluded.title,
                page_type = excluded.page_type,
                body = excluded.body,
                metadata_json = excluded.metadata_json,
                status = excluded.status,
                provenance_json = excluded.provenance_json,
                updated_at = excluded.updated_at
            """,
            (
                safe_slug,
                str(title or safe_slug),
                str(page_type or "note"),
                str(body or ""),
                _safe_json_dump(dict(metadata or {})),
                safe_status,
                _safe_json_dump(dict(provenance or {})),
                now,
                now,
            ),
        )
        c.execute("SELECT * FROM wiki_pages WHERE slug = ?", (safe_slug,))
        row = dict(c.fetchone())
        c.execute(
            """
            INSERT INTO wiki_update_log (event_type, page_id, message, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("page_upsert", int(row["page_id"]), f"updated {safe_slug}", "{}", now),
        )
        conn.commit()
        conn.close()
        return row

    def get_page(self, slug: str) -> Dict[str, Any] | None:
        safe_slug = _clean_slug(slug)
        conn = self._connect()
        row = conn.execute("SELECT * FROM wiki_pages WHERE slug = ?", (safe_slug,)).fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def update_page_status(self, slug: str, status: str) -> Dict[str, Any] | None:
        safe_slug = _clean_slug(slug)
        safe_status = str(status or "").strip()
        if safe_status not in {"draft", "published", "needs_review", "archived"}:
            raise ValueError(f"unsupported wiki page status: {safe_status}")
        now = int(time.time())
        conn = self._connect()
        conn.execute(
            "UPDATE wiki_pages SET status = ?, updated_at = ? WHERE slug = ?",
            (safe_status, now, safe_slug),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM wiki_pages WHERE slug = ?", (safe_slug,)).fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def add_page_source(
        self,
        *,
        page_id: int,
        source_path: str,
        source_ref: str,
        page_no: int = 0,
        chunk_id: int = 0,
    ) -> None:
        now = int(time.time())
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO wiki_page_sources (page_id, source_path, source_ref, page_no, chunk_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(page_id), source_path, source_ref, int(page_no or 0), int(chunk_id or 0), now),
        )
        conn.commit()
        conn.close()

    def replace_page_sources(self, *, page_id: int, sources: List[Dict[str, Any]]) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM wiki_page_sources WHERE page_id = ?", (int(page_id),))
        conn.commit()
        conn.close()
        for source in sources:
            self.add_page_source(
                page_id=int(page_id),
                source_path=str(source.get("source_path", "") or ""),
                source_ref=str(source.get("source_ref", "") or source.get("source_path", "") or ""),
                page_no=int(source.get("page_no", 0) or 0),
                chunk_id=int(source.get("chunk_id", 0) or 0),
            )

    def add_claim(
        self,
        *,
        page_id: int,
        claim_text: str,
        citations: List[Dict[str, Any]],
        status: str = "active",
    ) -> Dict[str, Any]:
        now = int(time.time())
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO wiki_claims (page_id, claim_text, status, citations_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(page_id), claim_text, status, _safe_json_dump(list(citations or [])), now, now),
        )
        claim_id = int(c.lastrowid)
        conn.commit()
        conn.close()
        return {"claim_id": claim_id, "page_id": int(page_id), "status": status}

    def replace_page_claims(self, *, page_id: int, claims: List[Dict[str, Any]]) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM wiki_claims WHERE page_id = ?", (int(page_id),))
        conn.commit()
        conn.close()
        for claim in claims:
            citations = _safe_json_loads(str(claim.get("source_refs_json", "[]") or "[]"), [])
            if not isinstance(citations, list):
                citations = []
            self.add_claim(
                page_id=int(page_id),
                claim_text=str(claim.get("claim_text", "") or ""),
                citations=[dict(item) for item in citations if isinstance(item, dict)],
                status=str(claim.get("status", "active") or "active"),
            )

    def list_pages(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        rows = [dict(row) for row in conn.execute("SELECT * FROM wiki_pages ORDER BY page_type, title, slug")]
        conn.close()
        return rows

    def space_summary(self) -> Dict[str, Any]:
        conn = self._connect()
        table_names = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        doc_role_counts: Dict[str, int] = {}
        guide_source_paths: List[str] = []
        casebook_source_paths: List[str] = []
        if "source_uploads" in table_names:
            for row in conn.execute(
                """
                SELECT COALESCE(NULLIF(doc_role, ''), 'unknown') AS doc_role, COUNT(*) AS count
                FROM source_uploads
                GROUP BY COALESCE(NULLIF(doc_role, ''), 'unknown')
                """
            ):
                doc_role_counts[str(row["doc_role"] or "unknown")] = int(row["count"] or 0)
            for row in conn.execute(
                """
                SELECT source_path, COALESCE(NULLIF(doc_role, ''), 'unknown') AS doc_role
                FROM source_uploads
                ORDER BY uploaded_at DESC, source_path ASC
                """
            ):
                doc_role = str(row["doc_role"] or "unknown")
                source_path = str(row["source_path"] or "")
                if doc_role == "guide" and source_path:
                    guide_source_paths.append(source_path)
                elif doc_role == "casebook" and source_path:
                    casebook_source_paths.append(source_path)

        page_status_counts: Dict[str, int] = {}
        page_type_counts: Dict[str, int] = {}
        for row in conn.execute(
            """
            SELECT COALESCE(NULLIF(status, ''), 'draft') AS status, COUNT(*) AS count
            FROM wiki_pages
            GROUP BY COALESCE(NULLIF(status, ''), 'draft')
            """
        ):
            page_status_counts[str(row["status"] or "draft")] = int(row["count"] or 0)
        for row in conn.execute(
            """
            SELECT COALESCE(NULLIF(page_type, ''), 'note') AS page_type, COUNT(*) AS count
            FROM wiki_pages
            GROUP BY COALESCE(NULLIF(page_type, ''), 'note')
            """
        ):
            page_type_counts[str(row["page_type"] or "note")] = int(row["count"] or 0)
        conn.close()
        return {
            "doc_role_counts": doc_role_counts,
            "guide_file_count": int(doc_role_counts.get("guide", 0)),
            "casebook_file_count": int(doc_role_counts.get("casebook", 0)),
            "guide_source_paths": guide_source_paths,
            "casebook_source_paths": casebook_source_paths,
            "page_status_counts": page_status_counts,
            "page_type_counts": page_type_counts,
        }

    def _page_sources(self, page_id: int) -> List[Dict[str, Any]]:
        conn = self._connect()
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM wiki_page_sources WHERE page_id = ? ORDER BY source_id ASC",
                (int(page_id),),
            )
        ]
        conn.close()
        return rows

    def _page_claims(self, page_id: int) -> List[Dict[str, Any]]:
        conn = self._connect()
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM wiki_claims WHERE page_id = ? ORDER BY claim_id ASC",
                (int(page_id),),
            )
        ]
        conn.close()
        return rows

    def published_page_hints(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        conn = self._connect()
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM wiki_pages
                WHERE COALESCE(status, '') = 'published'
                ORDER BY updated_at DESC, page_id DESC
                LIMIT ?
                """,
                (max(1, int(limit or 5)),),
            )
        ]
        hints: List[Dict[str, Any]] = []
        for row in rows:
            sources = self._page_sources(int(row["page_id"]))
            body = " ".join(str(row.get("body", "") or "").split())
            hints.append(
                {
                    "slug": str(row.get("slug", "") or ""),
                    "title": str(row.get("title", "") or ""),
                    "page_type": str(row.get("page_type", "") or ""),
                    "summary": body[:240],
                    "sources": [
                        {
                            "source_path": str(source.get("source_path", "") or ""),
                            "source_ref": str(source.get("source_ref", "") or ""),
                            "chunk_id": int(source.get("chunk_id", 0) or 0),
                            "page_no": int(source.get("page_no", 0) or 0),
                        }
                        for source in sources[:5]
                    ],
                }
            )
        conn.close()
        return hints

    def export_markdown(self, *, space_name: str = "", space_id: str = "") -> Dict[str, str]:
        pages = self.list_pages()
        exports: Dict[str, str] = {}
        summary = self.space_summary()
        display_space = str(space_name or space_id or "지침서 공간").strip()
        index_lines = [f"# {display_space} Wiki Index", ""]
        if space_id:
            index_lines.extend([f"- space_id: {space_id}", ""])
        index_lines.extend(
            [
                f"- guide_files: {summary.get('guide_file_count', 0)}",
                f"- casebook_files: {summary.get('casebook_file_count', 0)}",
                "",
            ]
        )
        for page in pages:
            slug = str(page.get("slug", "") or "index")
            title = str(page.get("title", "") or slug)
            index_lines.append(f"- [{title}]({slug}.md)")
            metadata = _safe_json_loads(str(page.get("metadata_json", "") or "{}"), {})
            frontmatter = {
                "title": title,
                "type": str(page.get("page_type", "") or "note"),
                "status": str(page.get("status", "") or "draft"),
                "space_name": display_space,
                "space_id": str(space_id or ""),
                **(metadata if isinstance(metadata, dict) else {}),
            }
            body_lines = ["---"]
            for key, value in frontmatter.items():
                body_lines.append(f"{key}: {_markdown_scalar(value)}")
            body_lines.extend(["---", "", f"# {title}", "", str(page.get("body", "") or "").strip()])
            sources = self._page_sources(int(page["page_id"]))
            if sources:
                body_lines.extend(["", "## Sources"])
                for source in sources:
                    ref = str(source.get("source_ref", "") or source.get("source_path", "") or "").strip()
                    if ref:
                        body_lines.append(f"- 근거: {ref}")
            claims = self._page_claims(int(page["page_id"]))
            if claims:
                body_lines.extend(["", "## Claims"])
                for claim in claims:
                    body_lines.append(f"- {str(claim.get('claim_text', '') or '').strip()}")
            exports[f"wiki/{slug}.md"] = "\n".join(body_lines).strip() + "\n"
        exports["wiki/index.md"] = "\n".join(index_lines).strip() + "\n"
        return exports

    def compile_source_page(self, source_path: str, *, max_chunks: int = 8, space_id: str = "") -> Dict[str, Any]:
        conn = self._connect()
        source_row = conn.execute(
            """
            SELECT source_path, source_type, doc_role, uploaded_at, original_filename
            FROM source_uploads
            WHERE source_path = ?
            LIMIT 1
            """,
            (source_path,),
        ).fetchone()
        chunk_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, source_path, source_type, section, line_start, line_end, text, source_updated_at
                FROM chunks
                WHERE source_path = ?
                  AND COALESCE(text, '') != ''
                ORDER BY id ASC
                LIMIT ?
                """,
                (source_path, max(1, int(max_chunks or 8))),
            )
        ]
        conn.close()

        if source_row is not None:
            source_meta = dict(source_row)
            title = _clean_title_from_path(str(source_meta.get("original_filename", "") or source_path))
            source_type = str(source_meta.get("source_type", "") or "")
            doc_role = str(source_meta.get("doc_role", "") or "unknown")
        else:
            source_meta = {"source_path": source_path}
            title = _clean_title_from_path(source_path)
            source_type = ""
            doc_role = "unknown"

        slug = f"sources/{_clean_slug(title)}"
        snippets = []
        for row in chunk_rows:
            text = " ".join(str(row.get("text", "") or "").split())
            if text:
                snippets.append(f"- {text[:500]}")
        body = "\n".join(snippets) if snippets else "아직 wiki 요약으로 컴파일할 텍스트가 없습니다."
        page = self.upsert_page(
            slug=slug,
            title=title,
            page_type="source",
            body=body,
            metadata={
                "source_count": 1,
                "chunk_count": len(chunk_rows),
                "quality": "draft",
                "source_type": source_type,
                "doc_role": doc_role,
                "doc_roles": [doc_role],
                "source_paths": [source_path],
                "space_id": str(space_id or ""),
            },
            provenance={
                "space_id": str(space_id or ""),
                "source_paths": [source_path],
                "doc_roles": [doc_role],
                "compiled_from": "source_upload",
            },
        )
        for row in chunk_rows[: max(1, int(max_chunks or 8))]:
            section = str(row.get("section", "") or "").strip()
            source_ref = f"{title}.pdf / {section}" if source_type == "pdf" and section else title
            page_no = 0
            match = re.search(r"PDF page\s+(\d+)", section, flags=re.IGNORECASE)
            if match:
                page_no = int(match.group(1))
            self.add_page_source(
                page_id=int(page["page_id"]),
                source_path=source_path,
                source_ref=source_ref,
                page_no=page_no,
                chunk_id=int(row.get("id", 0) or 0),
            )
        return page

    def save_answer_page(
        self,
        *,
        query_id: str,
        question: str,
        answer_text: str,
        citations: List[Dict[str, Any]],
        page_type: str = "query_note",
    ) -> Dict[str, Any]:
        safe_query_id = _clean_slug(query_id or f"answer-{int(time.time())}")
        title = (question or "저장된 답변").strip()
        if len(title) > 80:
            title = title[:80].rstrip() + "..."
        body = f"질문: {question.strip()}\n\n답변:\n{answer_text.strip()}"
        page = self.upsert_page(
            slug=f"claims/{safe_query_id}",
            title=title,
            page_type=page_type,
            body=body,
            metadata={
                "query_id": query_id,
                "source_count": len(citations or []),
                "claim_count": 1,
                "quality": "draft",
            },
        )
        self.add_claim(
            page_id=int(page["page_id"]),
            claim_text=answer_text.strip(),
            citations=list(citations or []),
        )
        for citation in list(citations or []):
            self.add_page_source(
                page_id=int(page["page_id"]),
                source_path=str(citation.get("source_path", "") or ""),
                source_ref=str(citation.get("source_ref", "") or citation.get("source_path", "") or ""),
                page_no=int(citation.get("page_no", 0) or 0),
                chunk_id=int(citation.get("chunk_id", 0) or 0),
            )
        return page

    def run_lint(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM wiki_lint_findings")
        findings: List[Dict[str, Any]] = []

        claim_rows = [
            dict(row)
            for row in c.execute(
                """
                SELECT claim_id, page_id, claim_text, citations_json
                FROM wiki_claims
                WHERE COALESCE(status, 'active') = 'active'
                """
            )
        ]
        for claim in claim_rows:
            citations = _safe_json_loads(str(claim.get("citations_json", "") or "[]"), [])
            if not isinstance(citations, list) or not citations:
                findings.append(
                    {
                        "finding_type": "citationless_claim",
                        "page_id": int(claim.get("page_id", 0) or 0),
                        "severity": "high",
                        "message": "근거 citation이 없는 wiki claim입니다.",
                        "metadata": {
                            "claim_id": int(claim.get("claim_id", 0) or 0),
                            "claim_preview": str(claim.get("claim_text", "") or "")[:240],
                        },
                    }
                )

        orphan_rows = [
            dict(row)
            for row in c.execute(
                """
                SELECT p.page_id, p.slug, p.title
                FROM wiki_pages p
                LEFT JOIN wiki_page_sources s ON s.page_id = p.page_id
                LEFT JOIN wiki_claims cl ON cl.page_id = p.page_id
                WHERE s.source_id IS NULL
                  AND cl.claim_id IS NULL
                """
            )
        ]
        for page in orphan_rows:
            findings.append(
                {
                    "finding_type": "orphan_page",
                    "page_id": int(page.get("page_id", 0) or 0),
                    "severity": "medium",
                    "message": "근거 source나 claim이 연결되지 않은 wiki page입니다.",
                    "metadata": {
                        "slug": str(page.get("slug", "") or ""),
                        "title": str(page.get("title", "") or ""),
                    },
                }
            )

        broken_source_rows = [
            dict(row)
            for row in c.execute(
                """
                SELECT p.page_id, p.slug, p.title, s.source_id, s.source_path, s.source_ref, s.chunk_id
                FROM wiki_pages p
                JOIN wiki_page_sources s ON s.page_id = p.page_id
                WHERE COALESCE(p.status, 'draft') = 'published'
                  AND COALESCE(s.source_path, '') = ''
                  AND COALESCE(s.chunk_id, 0) <= 0
                """
            )
        ]
        for row in broken_source_rows:
            page_id = int(row.get("page_id", 0) or 0)
            c.execute("UPDATE wiki_pages SET status = ?, updated_at = ? WHERE page_id = ?", ("needs_review", now, page_id))
            findings.append(
                {
                    "finding_type": "broken_page_source",
                    "page_id": page_id,
                    "severity": "high",
                    "message": "published wiki page에 깨진 source reference가 있습니다.",
                    "metadata": {
                        "slug": str(row.get("slug", "") or ""),
                        "source_id": int(row.get("source_id", 0) or 0),
                    },
                }
            )

        for finding in findings:
            c.execute(
                """
                INSERT INTO wiki_lint_findings
                    (finding_type, page_id, severity, message, status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(finding["finding_type"]),
                    int(finding.get("page_id", 0) or 0),
                    str(finding["severity"]),
                    str(finding["message"]),
                    "open",
                    _safe_json_dump(finding.get("metadata", {})),
                    now,
                    now,
                ),
            )
            finding["status"] = "open"
        conn.commit()
        conn.close()
        return findings
