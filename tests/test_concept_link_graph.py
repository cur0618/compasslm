import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.modules.setdefault(
    "hnswlib",
    SimpleNamespace(Index=type("DummyIndex", (), {})),
)
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=object),
)

from src.rag import RAGEngine


class ConceptLinkGraphTests(unittest.TestCase):
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

    def _build_engine(self, kb_id: str = "default") -> RAGEngine:
        engine = RAGEngine(kb_id=kb_id, data_dir=str(self.data_dir))

        def fake_encode(index_name: str, texts, task: str):
            rows = []
            for text in texts:
                normalized = (text or "").strip().lower()
                if "농사" in normalized or "영농" in normalized:
                    rows.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                elif "지원" in normalized:
                    rows.append(np.array([0.8, 0.2, 0.0], dtype=np.float32))
                else:
                    value = float(len(normalized) or 1)
                    rows.append(np.array([value / 10.0, value / 100.0, value / 1000.0], dtype=np.float32))
            return np.vstack(rows) if rows else np.empty((0, 3), dtype=np.float32)

        engine._encode_texts = fake_encode
        return engine

    def _insert_raw_chunk(self, engine: RAGEngine, source_path: str, text: str) -> int:
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text, is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"{source_path}:1", engine.kb_id, source_path, "txt", "guide", text, 0, 1),
        )
        chunk_id = int(cur.lastrowid)
        conn.commit()
        conn.close()
        return chunk_id

    def test_concept_nodes_are_reused_within_same_kb(self):
        engine = self._build_engine()
        chunk_a = self._insert_raw_chunk(engine, "a.txt", "농사 지원 기준을 안내합니다.")
        chunk_b = self._insert_raw_chunk(engine, "b.txt", "다른 파일에서도 농사 절차를 설명합니다.")

        engine._sync_concept_links(changed_chunk_ids=[chunk_a, chunk_b], deleted_chunk_ids=[])

        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM concept_nodes WHERE normalized_key = ?", ("농사",))
        node_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM chunk_concept_edges")
        edge_count = int(cur.fetchone()[0])
        conn.close()

        self.assertEqual(node_count, 1)
        self.assertGreaterEqual(edge_count, 2)

    def test_semantic_query_expansion_can_reach_linked_chunks_in_same_kb(self):
        engine = self._build_engine()
        chunk_a = self._insert_raw_chunk(engine, "a.txt", "영농 지원 대상과 절차를 설명합니다.")
        chunk_b = self._insert_raw_chunk(engine, "b.txt", "농사 준비 서류를 정리합니다.")

        engine._sync_concept_links(changed_chunk_ids=[chunk_a, chunk_b], deleted_chunk_ids=[])
        concept_candidates = engine._search_concept_candidates("농사", candidate_limit=10)

        self.assertIn(chunk_a, concept_candidates)
        self.assertIn(chunk_b, concept_candidates)

    def test_concept_links_do_not_cross_kb_boundaries(self):
        engine_a = self._build_engine("test1")
        engine_b = self._build_engine("test2")
        chunk_a = self._insert_raw_chunk(engine_a, "a.txt", "농사 지원 기준입니다.")

        engine_a._sync_concept_links(changed_chunk_ids=[chunk_a], deleted_chunk_ids=[])
        concept_candidates = engine_b._search_concept_candidates("농사", candidate_limit=10)

        self.assertEqual(concept_candidates, {})


if __name__ == "__main__":
    unittest.main()
