import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.ontology_store import OntologyStore
from src.wiki_memory_store import WikiMemoryStore


def _seed_answer_log(db_path: Path, *, query_id: str, citations, metadata=None, answer_text="답변입니다."):
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE answer_logs
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
        conn.execute(
            """
            CREATE TABLE chunks
            (
                id INTEGER PRIMARY KEY,
                source_path TEXT,
                text TEXT,
                source_updated_at INTEGER,
                is_normalized INTEGER DEFAULT 0,
                is_derived INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE retrieval_logs
            (
                query_id TEXT,
                user_id TEXT,
                query_text TEXT,
                topk_ids_json TEXT,
                meta_json TEXT,
                created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (id, source_path, text, source_updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (42, "/uploads/casebook.pdf", "근거 문장", 123),
        )
        conn.execute(
            """
            INSERT INTO answer_logs
                (query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                "local-llm",
                "hash",
                answer_text,
                json.dumps(citations, ensure_ascii=False),
                json.dumps(
                    {
                        "original_user_message": "태양열 발전기는 어떻게 조사해?",
                        "effective_user_message": "태양열 발전기 조사 방법",
                        **(metadata or {}),
                    },
                    ensure_ascii=False,
                ),
                1234,
            ),
        )


class WikiMemoryStoreTests(unittest.TestCase):
    def test_report_returns_original_citation_chunks_for_llm_recheck(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-report",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
            )

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-report",
                user_id="user-1",
                feedback_type="report_citation_issue",
            )

            self.assertEqual(saved["status"], "reported")
            self.assertEqual(saved["ontology_review_chunk_ids"], [42])

    def test_report_falls_back_to_retrieval_chunks_and_filters_derived_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(db_path, query_id="q-fallback", citations=[])
            with sqlite3.connect(str(db_path)) as conn:
                for chunk_id, is_normalized, is_derived in [
                    (43, 0, 0),
                    (44, 0, 1),
                    (45, 1, 0),
                    (46, 0, 0),
                ]:
                    conn.execute(
                        """
                        INSERT INTO chunks
                            (id, source_path, text, source_updated_at, is_normalized, is_derived)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (chunk_id, f"/uploads/{chunk_id}.xlsx", f"근거 {chunk_id}", 123, is_normalized, is_derived),
                    )
                conn.execute(
                    """
                    INSERT INTO retrieval_logs
                        (query_id, user_id, query_text, topk_ids_json, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("q-fallback", "user-1", "질문", json.dumps([44, 43, 45, 46]), "{}", 1235),
                )

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-fallback",
                user_id="user-1",
                feedback_type="report_citation_issue",
            )

            self.assertEqual(saved["ontology_review_chunk_ids"], [43, 46])

    def test_report_falls_back_when_citation_chunk_is_broken(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-broken",
                citations=[{"source_path": "/uploads/missing.pdf", "chunk_id": 9999}],
            )
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "INSERT INTO chunks (id, source_path, text, source_updated_at) VALUES (?, ?, ?, ?)",
                    (43, "/uploads/casebook.xlsx", "대체 근거", 123),
                )
                conn.execute(
                    """
                    INSERT INTO retrieval_logs
                        (query_id, user_id, query_text, topk_ids_json, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("q-broken", "user-1", "질문", "[43]", "{}", 1235),
                )

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-broken",
                user_id="user-1",
                feedback_type="report_citation_issue",
            )

            self.assertEqual(saved["ontology_review_chunk_ids"], [43])

    def test_report_limits_llm_recheck_to_eight_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(db_path, query_id="q-limit", citations=[])
            with sqlite3.connect(str(db_path)) as conn:
                chunk_ids = list(range(50, 62))
                for chunk_id in chunk_ids:
                    conn.execute(
                        "INSERT INTO chunks (id, source_path, text, source_updated_at) VALUES (?, ?, ?, ?)",
                        (chunk_id, f"/uploads/{chunk_id}.hwpx", f"근거 {chunk_id}", 123),
                    )
                conn.execute(
                    """
                    INSERT INTO retrieval_logs
                        (query_id, user_id, query_text, topk_ids_json, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("q-limit", "user-1", "질문", json.dumps(chunk_ids), "{}", 1235),
                )

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-limit",
                user_id="user-1",
                feedback_type="report_citation_issue",
            )

            self.assertEqual(saved["ontology_review_chunk_ids"], list(range(50, 58)))

    def test_save_answer_from_query_id_publishes_citation_backed_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q123",
                citations=[
                    {
                        "source_path": "/uploads/casebook.pdf",
                        "source_ref": "사례집.pdf / PDF page 14",
                        "page_no": 14,
                        "chunk_id": 42,
                    }
                ],
                answer_text="설치 형태에 따라 분류합니다. [[CITATION:1|사례집 14페이지]]",
            )

            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(
                query_id="q123",
                user_id="user-1",
                feedback_type="save_to_wiki",
            )

            self.assertEqual(saved["status"], "published")
            self.assertEqual(saved["question_text"], "태양열 발전기는 어떻게 조사해?")
            self.assertEqual(saved["source_count"], 1)
            with sqlite3.connect(str(db_path)) as conn:
                answer_count = conn.execute("SELECT COUNT(*) FROM wiki_saved_answers").fetchone()[0]
                source_count = conn.execute("SELECT COUNT(*) FROM wiki_answer_sources").fetchone()[0]
                feedback_count = conn.execute("SELECT COUNT(*) FROM wiki_answer_feedback").fetchone()[0]
            self.assertEqual(answer_count, 1)
            self.assertEqual(source_count, 1)
            self.assertEqual(feedback_count, 1)

    def test_save_answer_from_query_id_is_idempotent_for_same_user_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q123",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
            )
            store = WikiMemoryStore(str(db_path))

            first = store.save_answer_from_query_id(query_id="q123", user_id="user-1")
            second = store.save_answer_from_query_id(query_id="q123", user_id="user-1")

            self.assertEqual(first["saved_answer_id"], second["saved_answer_id"])
            with sqlite3.connect(str(db_path)) as conn:
                answer_count = conn.execute("SELECT COUNT(*) FROM wiki_saved_answers").fetchone()[0]
                feedback_count = conn.execute("SELECT COUNT(*) FROM wiki_answer_feedback").fetchone()[0]
            self.assertEqual(answer_count, 1)
            self.assertEqual(feedback_count, 2)

    def test_citationless_answer_is_saved_for_review_not_published(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(db_path, query_id="q-no-cite", citations=[])

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-no-cite",
                user_id="user-1",
            )

            self.assertEqual(saved["status"], "needs_review")
            self.assertEqual(saved["source_count"], 0)
            self.assertIn("citationless", saved["quality_flags_json"])

    def test_outside_document_claim_is_saved_for_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-outside",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
                answer_text="문서 밖 참고 정보로, 일반적으로 농산물 건조 시설입니다. [[CITATION:1|사례집]]",
            )

            saved = WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-outside",
                user_id="user-1",
            )

            self.assertEqual(saved["status"], "needs_review")
            self.assertIn("outside_document_claim", saved["quality_flags_json"])

    def test_invalid_page_number_is_saved_for_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-bad-page",
                citations=[
                    {
                        "source_path": "/uploads/casebook.pdf",
                        "source_ref": "사례집.pdf / PDF page 9843",
                        "page_no": 9843,
                        "chunk_id": 42,
                    }
                ],
            )

            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-bad-page", user_id="user-1")
            detail = store.get_saved_answer(saved["saved_answer_id"])

            self.assertEqual(saved["status"], "needs_review")
            self.assertIn("invalid_page_no", saved["quality_flags_json"])
            self.assertEqual(detail["sources"][0]["status"], "invalid_page_no")

    def test_search_memory_records_usage_stats_and_boosts_successful_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-success",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
                answer_text="태양열 발전기는 설치 위치와 고정식 여부에 따라 조사합니다. [[CITATION:1|사례집]]",
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-success", user_id="user-1")

            matches = store.search_memory("농가 태양광 설치는 어떻게 처리해?", limit=3)

            self.assertEqual(matches[0]["saved_answer_id"], saved["saved_answer_id"])
            self.assertGreater(matches[0]["reused_count"], 0)
            with sqlite3.connect(str(db_path)) as conn:
                usage_count = conn.execute("SELECT COUNT(*) FROM wiki_answer_usage_stats").fetchone()[0]
            self.assertEqual(usage_count, 1)

    def test_published_answer_boosts_linked_ontology_fact_confidence(self):
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
                        page_no INTEGER,
                        line_start INTEGER,
                        line_end INTEGER,
                        section TEXT,
                        text TEXT,
                        source_updated_at INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO chunks
                        (id, source_path, source_type, page_no, line_start, line_end, section, text, source_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        42,
                        "/uploads/pay.xlsx",
                        "xlsx",
                        0,
                        0,
                        0,
                        "기준",
                        "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원",
                        123,
                    ),
                )
                conn.execute(
                    """
                    CREATE TABLE answer_logs
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
                conn.execute(
                    """
                    INSERT INTO answer_logs
                        (query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "q-ontology",
                        "local-llm",
                        "hash",
                        "농가경제조사의 지급단가는 40천원입니다. [[CITATION:1|기준]]",
                        json.dumps([{"source_path": "/uploads/pay.xlsx", "chunk_id": 42}], ensure_ascii=False),
                        json.dumps({"original_user_message": "농가경제조사 지급단가"}, ensure_ascii=False),
                        1234,
                    ),
                )
            ontology = OntologyStore(str(db_path), kb_id="")
            ontology.sync_facts_for_chunks([42], [])
            before = ontology.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]

            WikiMemoryStore(str(db_path)).save_answer_from_query_id(
                query_id="q-ontology",
                user_id="user-1",
            )

            after = ontology.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]
            self.assertGreater(after, before)

    def test_duplicate_saved_answer_does_not_boost_ontology_fact_twice(self):
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
                        page_no INTEGER,
                        line_start INTEGER,
                        line_end INTEGER,
                        section TEXT,
                        text TEXT,
                        source_updated_at INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO chunks
                        (id, source_path, source_type, page_no, line_start, line_end, section, text, source_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        42,
                        "/uploads/pay.xlsx",
                        "xlsx",
                        0,
                        0,
                        0,
                        "기준",
                        "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원",
                        123,
                    ),
                )
                conn.execute(
                    """
                    CREATE TABLE answer_logs
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
                conn.execute(
                    """
                    INSERT INTO answer_logs
                        (query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "q-ontology-dupe",
                        "local-llm",
                        "hash",
                        "농가경제조사의 지급단가는 40천원입니다. [[CITATION:1|기준]]",
                        json.dumps([{"source_path": "/uploads/pay.xlsx", "chunk_id": 42}], ensure_ascii=False),
                        json.dumps({"original_user_message": "농가경제조사 지급단가"}, ensure_ascii=False),
                        1234,
                    ),
                )
            ontology = OntologyStore(str(db_path), kb_id="")
            ontology.sync_facts_for_chunks([42], [])
            store = WikiMemoryStore(str(db_path))

            store.save_answer_from_query_id(query_id="q-ontology-dupe", user_id="user-1")
            after_first = ontology.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]
            store.save_answer_from_query_id(query_id="q-ontology-dupe", user_id="user-1")
            after_second = ontology.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]

            self.assertAlmostEqual(after_second, after_first)

    def test_list_saved_answers_supports_status_filter_and_quality_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(db_path, query_id="q-review", citations=[])
            store = WikiMemoryStore(str(db_path))
            store.save_answer_from_query_id(query_id="q-review", user_id="user-1")

            review_rows = store.list_saved_answers(status="needs_review")
            summary = store.quality_summary()

            self.assertEqual(len(review_rows), 1)
            self.assertEqual(summary["status_counts"]["needs_review"], 1)
            self.assertEqual(summary["quality_flag_counts"]["citationless"], 1)

    def test_lint_findings_can_be_listed_and_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(db_path, query_id="q-lint", citations=[])
            store = WikiMemoryStore(str(db_path))
            store.save_answer_from_query_id(query_id="q-lint", user_id="user-1")
            store.run_lint()

            open_findings = store.list_lint_findings(status="open")
            resolved = store.resolve_lint_finding(open_findings[0]["finding_id"])

            self.assertEqual(resolved["status"], "resolved")
            self.assertEqual(store.list_lint_findings(status="open"), [])

    def test_compile_saved_answer_creates_structured_memory_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-compile",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
                answer_text=(
                    "태양열 발전기 처리 절차는 다음과 같습니다. "
                    "1. 건물 위 고정식 여부를 확인합니다. "
                    "2. 재조달가 10% 이상이면 투입자본액으로 처리합니다. "
                    "표 기준: 고정식은 대수리, 이동식은 기타구축물입니다. [[CITATION:1|사례집]]"
                ),
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-compile", user_id="user-1")

            compiled = store.compile_saved_answer(saved["saved_answer_id"])

            self.assertGreaterEqual(len(compiled["claims"]), 1)
            self.assertGreaterEqual(len(compiled["concepts"]), 1)
            self.assertGreaterEqual(len(compiled["procedures"]), 1)
            self.assertGreaterEqual(len(compiled["table_rules"]), 1)
            self.assertEqual(store.list_concepts()[0]["status"], "published")
            self.assertEqual(store.list_procedures()[0]["status"], "published")
            self.assertEqual(store.list_table_rules()[0]["status"], "published")

    def test_duplicate_compile_merges_existing_concepts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-compile",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
                answer_text="태양열 발전기 처리 절차입니다. 1. 설치 위치를 확인합니다. [[CITATION:1|사례집]]",
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-compile", user_id="user-1")

            store.compile_saved_answer(saved["saved_answer_id"])
            store.compile_saved_answer(saved["saved_answer_id"])

            self.assertEqual(len(store.list_concepts()), 1)

    def test_conflicts_can_be_listed_and_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiMemoryStore(str(db_path))
            conflict = store.create_conflict(
                saved_answer_id=1,
                conflicting_saved_answer_id=2,
                conflict_type="same_question_different_answer",
                description="서로 다른 답변입니다.",
            )

            resolved = store.resolve_conflict(conflict["conflict_id"])

            self.assertEqual(resolved["status"], "resolved")
            self.assertEqual(store.list_conflicts(status="open"), [])

    def test_retrieval_boost_targets_exclude_review_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-boost",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-boost", user_id="user-1")
            store.search_memory("태양열 발전기 조사", limit=1)

            targets = store.retrieval_boost_targets()
            store.update_saved_answer_status(saved["saved_answer_id"], "needs_review")

            self.assertEqual(targets["chunks"][42], 1)
            self.assertEqual(store.retrieval_boost_targets()["chunks"], {})

    def test_published_wiki_page_sources_are_retrieval_boost_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            wiki_store = __import__("src.wiki_store", fromlist=["WikiStore"]).WikiStore(str(db_path))
            page = wiki_store.upsert_page(
                slug="concepts/solar",
                title="태양열 발전기",
                page_type="concept",
                body="검토된 위키 페이지",
                metadata={},
                status="published",
            )
            wiki_store.add_page_source(
                page_id=page["page_id"],
                source_path="/uploads/casebook.pdf",
                source_ref="사례집",
                page_no=14,
                chunk_id=42,
            )

            targets = WikiMemoryStore(str(db_path)).retrieval_boost_targets()

            self.assertEqual(targets["chunks"][42], 1)
            self.assertEqual(targets["sources"]["/uploads/casebook.pdf"], 1)

    def test_retrieval_boost_targets_are_isolated_by_guide_space_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            WikiStore = __import__("src.wiki_store", fromlist=["WikiStore"]).WikiStore
            db_a = Path(tmpdir) / "space-a.sqlite"
            db_b = Path(tmpdir) / "space-b.sqlite"

            page_a = WikiStore(str(db_a)).upsert_page(
                slug="concepts/shared-term",
                title="공통 용어",
                page_type="concept",
                body="A 공간의 의미",
                metadata={},
                status="published",
            )
            WikiStore(str(db_a)).add_page_source(
                page_id=page_a["page_id"],
                source_path="/uploads/a-guide.pdf",
                source_ref="A 지침서",
                chunk_id=101,
            )
            page_b = WikiStore(str(db_b)).upsert_page(
                slug="concepts/shared-term",
                title="공통 용어",
                page_type="concept",
                body="B 공간의 의미",
                metadata={},
                status="published",
            )
            WikiStore(str(db_b)).add_page_source(
                page_id=page_b["page_id"],
                source_path="/uploads/b-guide.pdf",
                source_ref="B 지침서",
                chunk_id=202,
            )

            targets_a = WikiMemoryStore(str(db_a)).retrieval_boost_targets()
            targets_b = WikiMemoryStore(str(db_b)).retrieval_boost_targets()

            self.assertEqual(set(targets_a["chunks"]), {101})
            self.assertEqual(set(targets_b["chunks"]), {202})
            self.assertNotIn("/uploads/b-guide.pdf", targets_a["sources"])
            self.assertNotIn("/uploads/a-guide.pdf", targets_b["sources"])

    def test_export_markdown_includes_structured_memory_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-export",
                citations=[{"source_path": "/uploads/casebook.pdf", "chunk_id": 42}],
                answer_text="태양열 발전기 처리 절차입니다. 1. 설치 위치를 확인합니다. [[CITATION:1|사례집]]",
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-export", user_id="user-1")
            store.compile_saved_answer(saved["saved_answer_id"])

            files = store.export_markdown()

            self.assertIn("wiki/concepts/index.md", files)
            self.assertIn("wiki/procedures/index.md", files)
            self.assertIn("wiki/table_rules/index.md", files)
            self.assertIn("wiki/overview.md", files)
            self.assertIn("wiki/review_queue.md", files)
            self.assertIn("wiki/lint.md", files)
            self.assertIn("wiki/conflicts.md", files)

    def test_lint_flags_broken_saved_answer_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-broken",
                citations=[{"source_path": "/uploads/missing.pdf", "chunk_id": 999}],
            )
            store = WikiMemoryStore(str(db_path))
            store.save_answer_from_query_id(query_id="q-broken", user_id="user-1")

            findings = store.run_lint()

            self.assertIn("broken_source_reference", {item["finding_type"] for item in findings})


if __name__ == "__main__":
    unittest.main()
