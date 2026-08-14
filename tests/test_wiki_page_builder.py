import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.wiki_memory_store import WikiMemoryStore
from src.wiki_page_builder import WikiPageBuilder, build_wiki_page_payload
from src.wiki_store import WikiStore
from tests.test_wiki_memory_store import _seed_answer_log


class WikiPageBuilderTests(unittest.TestCase):
    def test_payload_preserves_claims_and_source_refs_for_concept_page(self):
        payload = build_wiki_page_payload(
            page_type="concept",
            title="태양열 발전기",
            body="태양열 발전기는 설치 형태에 따라 조사합니다.",
            claims=[{"claim_text": "설치 형태에 따라 조사합니다.", "source_refs_json": "[]"}],
            sources=[{"source_path": "/uploads/casebook.pdf", "source_ref": "사례집.pdf / PDF page 14", "chunk_id": 42}],
            provenance={"candidate_type": "concept", "candidate_id": 7},
        )

        self.assertEqual(payload["status"], "draft")
        self.assertEqual(payload["page_type"], "concept")
        self.assertIn("태양열-발전기", payload["slug"])
        self.assertEqual(payload["source_count"], 1)
        self.assertEqual(payload["claim_count"], 1)
        self.assertEqual(payload["sources"][0]["chunk_id"], 42)

    def test_payload_without_sources_is_needs_review(self):
        payload = build_wiki_page_payload(
            page_type="procedure",
            title="출처 없는 절차",
            body="검토가 필요합니다.",
            claims=[],
            sources=[],
            provenance={"candidate_type": "procedure", "candidate_id": 3},
        )

        self.assertEqual(payload["status"], "needs_review")
        self.assertEqual(payload["source_count"], 0)

    def test_builder_persists_page_claims_and_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            store = WikiStore(str(db_path))
            builder = WikiPageBuilder(store)

            page = builder.persist_page(
                build_wiki_page_payload(
                    page_type="table_rule",
                    title="농가경제조사 단가 기준",
                    body="농가경제조사 단가는 표 기준을 따릅니다.",
                    claims=[{"claim_text": "단가는 표 기준을 따릅니다.", "source_refs_json": "[]"}],
                    sources=[
                        {
                            "source_path": "/uploads/rule.hwpx",
                            "source_ref": "지침서 표 3",
                            "page_no": 0,
                            "chunk_id": 42,
                        }
                    ],
                    provenance={"candidate_type": "table_rule", "candidate_id": 9},
                )
            )

            self.assertEqual(page["status"], "draft")
            exported = store.export_markdown()["wiki/table_rules/농가경제조사-단가-기준.md"]
            self.assertIn("status: draft", exported)
            self.assertIn("지침서 표 3", exported)
            self.assertIn("단가는 표 기준을 따릅니다.", exported)

    def test_memory_candidates_skip_open_lint_and_conflict_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "meta.sqlite"
            _seed_answer_log(
                db_path,
                query_id="q-ok",
                citations=[{"source_path": "/uploads/casebook.pdf", "source_ref": "사례집", "chunk_id": 42}],
                answer_text="태양열 발전기 처리 절차입니다. 1. 설치 위치를 확인합니다. [[CITATION:1|사례집]]",
            )
            store = WikiMemoryStore(str(db_path))
            saved = store.save_answer_from_query_id(query_id="q-ok", user_id="user-1")
            store.compile_saved_answer(saved["saved_answer_id"])
            candidate_types = {item["page_type"] for item in store.build_wiki_page_candidates()}
            self.assertIn("concept", candidate_types)
            self.assertIn("procedure", candidate_types)

            store.create_conflict(
                saved_answer_id=saved["saved_answer_id"],
                conflicting_saved_answer_id=999,
                conflict_type="same_question_different_answer",
                description="검토 전까지 page 후보에서 제외",
            )
            self.assertEqual(store.build_wiki_page_candidates(), [])

            store.resolve_conflict(1)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO wiki_memory_lint_findings
                        (finding_type, target_type, target_id, severity, message, status, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("manual_review", "saved_answer", saved["saved_answer_id"], "medium", "검토 필요", "open", "{}", 1, 1),
                )
            self.assertEqual(store.build_wiki_page_candidates(), [])


if __name__ == "__main__":
    unittest.main()
