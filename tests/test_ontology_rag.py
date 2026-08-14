import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests

sys.modules.setdefault(
    "hnswlib",
    SimpleNamespace(Index=type("DummyIndex", (), {})),
)
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=object),
)

from src.rag import RAGEngine
from src.ontology_store import OntologyStore


class OntologyRagTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = Path(self.tmpdir.name) / "kb"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_patch = patch.object(RAGEngine, "_load_embedding_model", return_value=SimpleNamespace())
        self.dim_patch = patch.object(RAGEngine, "_infer_embedding_dim", return_value=3)
        self.model_patch.start()
        self.dim_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.addCleanup(self.dim_patch.stop)

    def _build_engine(self) -> RAGEngine:
        engine = RAGEngine(kb_id="default", data_dir=str(self.data_dir))

        def fake_encode(index_name: str, texts, task: str):
            return np.vstack([np.array([0.1, 0.0, 0.0], dtype=np.float32) for _ in texts])

        engine._encode_texts = fake_encode
        return engine

    def _insert_raw_chunk(self, engine: RAGEngine, text: str, source_path: str) -> int:
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text, is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"{source_path}:1", engine.kb_id, source_path, "xlsx", "guide", text, 0, 1),
        )
        chunk_id = int(cur.lastrowid)
        conn.commit()
        conn.close()
        return chunk_id

    def test_search_uses_ontology_fact_graph_as_retrieval_candidate(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "표의미: kind=table_row | subject=농가경제조사 | aliases=농가경제조사, 농어가경제조사 | 지급단가=40천원",
            "pay.xlsx",
        )
        self._insert_raw_chunk(
            engine,
            "태양열 발전기는 기타 구축물로 조사합니다.",
            "casebook.txt",
        )
        engine._sync_ontology_facts(changed_chunk_ids=[target_chunk], deleted_chunk_ids=[])

        results = engine.search("농가경제조사 지급단가은?", top_k=1)

        self.assertEqual(results[0]["id"], target_chunk)
        self.assertGreater(results[0]["ontology_fact_score"], 0.0)
        self.assertIn("농가경제조사", results[0]["matched_ontology_facts"][0])

    def test_search_exposes_ontology_query_rewrite_and_hop_metadata(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "표의미: kind=table_row | subject=농가경제조사 | aliases=농가경제조사, 농어가경제조사 | 지급단가=40천원",
            "pay.xlsx",
        )
        engine._sync_ontology_facts(changed_chunk_ids=[target_chunk], deleted_chunk_ids=[])

        results = engine.search("농어가경제조사 답례품 얼마야?", top_k=1)

        self.assertEqual(results[0]["id"], target_chunk)
        self.assertIn("지급단가", results[0]["ontology_query_rewrite"])
        self.assertGreaterEqual(results[0]["ontology_hop_count"], 1)
        self.assertTrue(results[0]["ontology_candidate_reason"])

    def test_search_forwards_evaluation_extraction_filter_to_ontology_store(self):
        engine = self._build_engine()
        engine.ontology_allowed_extraction_methods = {"deterministic_table_fact"}
        engine.ontology_experiment_mode = "deterministic"

        with patch("src.rag.OntologyStore.search_facts", return_value=[]) as search_facts:
            engine._search_ontology_candidates("농가경제조사", candidate_limit=5)

        self.assertEqual(
            search_facts.call_args.kwargs["allowed_extraction_methods"],
            {"deterministic_table_fact"},
        )
        self.assertEqual(search_facts.call_args.kwargs["experiment_mode"], "deterministic")

    def test_include_llm_sync_reports_disabled_reason_when_extraction_is_off(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원",
            "pay.xlsx",
        )

        with patch("src.rag.ONTOLOGY_LLM_EXTRACTION_ENABLED", False):
            summary = engine._sync_ontology_facts(
                changed_chunk_ids=[target_chunk],
                deleted_chunk_ids=[],
                include_llm=True,
            )

        self.assertEqual(summary["ontology_facts_added"], 1)
        self.assertTrue(summary["ontology_extraction_disabled"])
        self.assertEqual(summary["ontology_extraction_disabled_reason"], "ONTOLOGY_LLM_EXTRACTION_ENABLED=0")

    def test_deterministic_sync_does_not_call_llm_or_embedding(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원",
            "pay.xlsx",
        )

        with patch("src.rag.requests.post") as llm_post, patch.object(engine, "_encode_texts") as encode:
            summary = engine._sync_ontology_facts(
                changed_chunk_ids=[target_chunk],
                deleted_chunk_ids=[],
                include_llm=False,
            )

        self.assertEqual(summary["ontology_facts_added"], 1)
        llm_post.assert_not_called()
        encode.assert_not_called()

    def test_limited_llm_extraction_returns_empty_for_empty_choices(self):
        engine = self._build_engine()

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": []}

        with patch("src.rag.requests.post", return_value=FakeResponse()):
            payload = engine._extract_limited_llm_ontology_payload("농가경제조사 지급단가는 40천원입니다.")

        self.assertEqual(payload, [])

    def test_limited_llm_malformed_json_is_counted_as_extraction_error(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "농가경제조사 지급단가는 40천원입니다.",
            "pay.xlsx",
        )

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "not json"}}]}

        with patch("src.rag.ONTOLOGY_LLM_EXTRACTION_ENABLED", True), patch("src.rag.requests.post", return_value=FakeResponse()):
            summary = engine._sync_ontology_facts(
                changed_chunk_ids=[target_chunk],
                deleted_chunk_ids=[],
                include_llm=True,
            )

        self.assertEqual(summary["ontology_extraction_errors"], 1)

    def test_limited_llm_timeout_is_counted_as_extraction_error(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "농가경제조사 지급단가는 40천원입니다.",
            "pay.xlsx",
        )

        with patch("src.rag.ONTOLOGY_LLM_EXTRACTION_ENABLED", True), patch("src.rag.requests.post", side_effect=requests.Timeout("slow llm")):
            summary = engine._sync_ontology_facts(
                changed_chunk_ids=[target_chunk],
                deleted_chunk_ids=[],
                include_llm=True,
            )

        self.assertEqual(summary["ontology_extraction_errors"], 1)

    def test_reported_answer_llm_facts_are_forced_to_needs_review(self):
        engine = self._build_engine()
        target_chunk = self._insert_raw_chunk(
            engine,
            "농가경제조사 지급단가는 40천원입니다.",
            "pay.xlsx",
        )

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '[{"subject":"농가경제조사","predicate":"지급단가",'
                                    '"object":"40천원","confidence":0.91,'
                                    '"evidence_quote":"지급단가는 40천원"}]'
                                )
                            }
                        }
                    ]
                }

        with patch("src.rag.ONTOLOGY_LLM_EXTRACTION_ENABLED", True), patch(
            "src.rag.requests.post", return_value=FakeResponse()
        ):
            engine._sync_ontology_facts(
                changed_chunk_ids=[target_chunk],
                deleted_chunk_ids=[],
                include_llm=True,
                llm_fact_status="needs_review",
            )

        conn = engine._connect_db()
        row = conn.execute(
            "SELECT status FROM ontology_facts WHERE extraction_method = 'limited_llm'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "needs_review")
        self.assertEqual(
            OntologyStore(str(engine.db_path), kb_id=engine.kb_id).search_facts("농가경제조사 지급단가"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
