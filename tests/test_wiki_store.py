import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.wiki_store import WikiStore


class WikiStoreTests(unittest.TestCase):
    def test_schema_page_upsert_claim_and_markdown_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiStore(str(db_path))

            page = store.upsert_page(
                slug="sources/2024-casebook",
                title="2024 사례집",
                page_type="source",
                body="태양열 발전기 사례 요약",
                metadata={"source_count": 1, "quality": "draft"},
            )
            store.add_page_source(
                page_id=page["page_id"],
                source_path="/uploads/2024농가경제조사사례집.pdf",
                source_ref="2024농가경제조사사례집.pdf / PDF page 14",
                page_no=14,
                chunk_id=42,
            )
            store.add_claim(
                page_id=page["page_id"],
                claim_text="태양열 발전기는 설치 형태에 따라 분류한다.",
                citations=[
                    {
                        "source_path": "/uploads/2024농가경제조사사례집.pdf",
                        "source_ref": "2024농가경제조사사례집.pdf / PDF page 14",
                        "page_no": 14,
                        "chunk_id": 42,
                    }
                ],
            )

            index_md = store.export_markdown()["wiki/index.md"]
            source_md = store.export_markdown()["wiki/sources/2024-casebook.md"]

            self.assertIn("- [2024 사례집](sources/2024-casebook.md)", index_md)
            self.assertIn("type: source", source_md)
            self.assertIn("source_count: 1", source_md)
            self.assertIn("태양열 발전기 사례 요약", source_md)
            self.assertIn("근거: 2024농가경제조사사례집.pdf / PDF page 14", source_md)

            with sqlite3.connect(str(db_path)) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'wiki_%'"
                    )
                }
            self.assertTrue(
                {
                    "wiki_pages",
                    "wiki_links",
                    "wiki_claims",
                    "wiki_page_sources",
                    "wiki_update_log",
                    "wiki_lint_findings",
                }.issubset(tables)
            )

    def test_compile_source_page_from_existing_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE chunks
                    (
                        id INTEGER PRIMARY KEY,
                        source_path TEXT,
                        source_type TEXT,
                        section TEXT,
                        line_start INTEGER,
                        line_end INTEGER,
                        text TEXT,
                        source_updated_at INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE source_uploads
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
                conn.execute(
                    """
                    INSERT INTO source_uploads
                        (source_path, source_type, doc_role, file_hash, doc_version, uploaded_at, original_filename)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("/uploads/casebook.pdf", "pdf", "casebook", "abc", "v1", 123, "2024농가경제조사사례집.pdf"),
                )
                conn.execute(
                    """
                    INSERT INTO chunks (id, source_path, source_type, section, line_start, line_end, text, source_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (42, "/uploads/casebook.pdf", "pdf", "PDF page 14", 1, 3, "태양열 발전기 설치 형태 사례", 123),
                )

            store = WikiStore(str(db_path))
            page = store.compile_source_page("/uploads/casebook.pdf", space_id="space-2024-farm")

            self.assertEqual(page["slug"], "sources/2024농가경제조사사례집")
            metadata = json.loads(page["metadata_json"])
            provenance = json.loads(page["provenance_json"])
            self.assertEqual(metadata["space_id"], "space-2024-farm")
            self.assertEqual(metadata["doc_role"], "casebook")
            self.assertEqual(metadata["source_paths"], ["/uploads/casebook.pdf"])
            self.assertEqual(provenance["space_id"], "space-2024-farm")
            exported = store.export_markdown()["wiki/sources/2024농가경제조사사례집.md"]
            self.assertIn("태양열 발전기 설치 형태 사례", exported)
            self.assertIn("근거: 2024농가경제조사사례집.pdf / PDF page 14", exported)

    def test_space_summary_and_export_mark_current_guide_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE source_uploads
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
                conn.executemany(
                    """
                    INSERT INTO source_uploads
                        (source_path, source_type, doc_role, file_hash, doc_version, uploaded_at, original_filename)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("/uploads/guide-a.pdf", "pdf", "guide", "a", "v1", 100, "지침서 A.pdf"),
                        ("/uploads/guide-b.pdf", "pdf", "guide", "b", "v1", 110, "지침서 B.pdf"),
                        ("/uploads/casebook.pdf", "pdf", "casebook", "c", "v1", 120, "사례집.pdf"),
                    ],
                )

            store = WikiStore(str(db_path))
            store.upsert_page(
                slug="concepts/test",
                title="공간 테스트",
                page_type="concept",
                body="같은 지침서 공간 안에서 누적된 페이지입니다.",
                metadata={"source_count": 2},
                status="published",
            )

            summary = store.space_summary()
            self.assertEqual(summary["guide_file_count"], 2)
            self.assertEqual(summary["casebook_file_count"], 1)
            self.assertEqual(summary["page_status_counts"]["published"], 1)

            exports = store.export_markdown(space_name="농가경제", space_id="kb-alice-1")
            self.assertIn("# 농가경제 Wiki Index", exports["wiki/index.md"])
            self.assertIn("guide_files: 2", exports["wiki/index.md"])
            self.assertIn("space_name: 농가경제", exports["wiki/concepts/test.md"])
            self.assertIn("space_id: kb-alice-1", exports["wiki/concepts/test.md"])

    def test_save_answer_page_preserves_answer_citations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiStore(str(db_path))

            page = store.save_answer_page(
                query_id="q123",
                question="태양열 발전기는 어떻게 조사해?",
                answer_text="설치 형태에 따라 분류합니다.",
                citations=[
                    {
                        "source_path": "/uploads/casebook.pdf",
                        "source_ref": "2024농가경제조사사례집.pdf / PDF page 14",
                        "page_no": 14,
                        "chunk_id": 42,
                    }
                ],
            )

            self.assertEqual(page["slug"], "claims/q123")
            exported = store.export_markdown()["wiki/claims/q123.md"]
            self.assertIn("type: query_note", exported)
            self.assertIn("태양열 발전기는 어떻게 조사해?", exported)
            self.assertIn("설치 형태에 따라 분류합니다.", exported)
            self.assertIn("2024농가경제조사사례집.pdf / PDF page 14", exported)

    def test_run_lint_records_citationless_claims_and_orphan_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiStore(str(db_path))
            page = store.upsert_page(
                slug="concepts/solar-generator",
                title="태양열 발전기",
                page_type="concept",
                body="근거 정리가 필요한 개념 페이지",
                metadata={"quality": "draft"},
            )
            store.add_claim(
                page_id=int(page["page_id"]),
                claim_text="태양열 발전기는 설치 형태에 따라 분류한다.",
                citations=[],
            )
            store.upsert_page(
                slug="concepts/orphan",
                title="연결 없는 페이지",
                page_type="concept",
                body="아직 근거와 claim이 연결되지 않은 페이지",
                metadata={"quality": "draft"},
            )

            findings = store.run_lint()

            finding_types = {finding["finding_type"] for finding in findings}
            self.assertIn("citationless_claim", finding_types)
            self.assertIn("orphan_page", finding_types)
            with sqlite3.connect(str(db_path)) as conn:
                stored_count = conn.execute("SELECT COUNT(*) FROM wiki_lint_findings").fetchone()[0]
            self.assertEqual(stored_count, len(findings))

    def test_page_status_migration_publish_archive_and_broken_source_lint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiStore(str(db_path))
            page = store.upsert_page(
                slug="concepts/status-test",
                title="상태 테스트",
                page_type="concept",
                body="상태 전환을 확인합니다.",
                metadata={"source_count": 1},
                status="draft",
                provenance={"candidate_type": "concept"},
            )

            self.assertEqual(page["status"], "draft")
            published = store.update_page_status("concepts/status-test", "published")
            archived = store.update_page_status("concepts/status-test", "archived")

            self.assertEqual(published["status"], "published")
            self.assertEqual(archived["status"], "archived")

            broken = store.upsert_page(
                slug="concepts/broken-source",
                title="깨진 출처",
                page_type="concept",
                body="깨진 출처를 가진 페이지입니다.",
                metadata={},
                status="published",
            )
            store.add_page_source(
                page_id=broken["page_id"],
                source_path="",
                source_ref="",
                page_no=0,
                chunk_id=0,
            )

            findings = store.run_lint()

            self.assertIn("broken_page_source", {finding["finding_type"] for finding in findings})
            self.assertEqual(store.get_page("concepts/broken-source")["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
