import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
chardet_stub = ModuleType("chardet")
chardet_stub.detect = lambda _data: {"encoding": "utf-8"}
sys.modules.setdefault("chardet", chardet_stub)

try:
    import openpyxl  # noqa: F401
except ImportError:
    sys.modules.setdefault("openpyxl", ModuleType("openpyxl"))

from src.rag import RAGEngine


CONCEPT_TEXT = "\ud14d\uc2a4\ud2b8"
GUIDE_TEXT = "\uc9c0\uce68"
SURVEY_TEXT = "\ub18d\uac00\uacbd\uc81c\uc870\uc0ac"


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
                if CONCEPT_TEXT in normalized or "\uc0c1\uc18d" in normalized:
                    rows.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                elif GUIDE_TEXT in normalized:
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
        chunk_a = self._insert_raw_chunk(
            engine,
            "a.txt",
            f"{CONCEPT_TEXT} {GUIDE_TEXT} \uae30\ubc18 \uc548\ub0b4\ubb38\uc785\ub2c8\ub2e4.",
        )
        chunk_b = self._insert_raw_chunk(
            engine,
            "b.txt",
            f"\ub2e4\ub978 \ud30c\uc77c\uc5d0\uc11c\ub3c4 {CONCEPT_TEXT} \uc608\uc2dc\ub97c \uc124\uba85\ud569\ub2c8\ub2e4.",
        )

        engine._sync_concept_links(changed_chunk_ids=[chunk_a, chunk_b], deleted_chunk_ids=[])

        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM concept_nodes WHERE normalized_key = ?", (CONCEPT_TEXT,))
        node_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM chunk_concept_edges")
        edge_count = int(cur.fetchone()[0])
        conn.close()

        self.assertEqual(node_count, 1)
        self.assertGreaterEqual(edge_count, 2)

    def test_semantic_query_expansion_can_reach_linked_chunks_in_same_kb(self):
        engine = self._build_engine()
        chunk_a = self._insert_raw_chunk(
            engine,
            "a.txt",
            "\uc0c1\uc18d \uc9c0\uce68\uc758 \uc808\ucc28\uacfc \uc608\uc2dc\ub97c \uc124\uba85\ud569\ub2c8\ub2e4.",
        )
        chunk_b = self._insert_raw_chunk(
            engine,
            "b.txt",
            f"{CONCEPT_TEXT} \uc900\ube44 \uc11c\ub958\ub97c \uc815\ub9ac\ud569\ub2c8\ub2e4.",
        )

        engine._sync_concept_links(changed_chunk_ids=[chunk_a, chunk_b], deleted_chunk_ids=[])
        concept_candidates = engine._search_concept_candidates(CONCEPT_TEXT, candidate_limit=10)

        self.assertIn(chunk_a, concept_candidates)
        self.assertIn(chunk_b, concept_candidates)

    def test_concept_links_do_not_cross_kb_boundaries(self):
        engine_a = self._build_engine("test1")
        engine_b = self._build_engine("test2")
        chunk_a = self._insert_raw_chunk(
            engine_a,
            "a.txt",
            f"{CONCEPT_TEXT} {GUIDE_TEXT} \uae30\ubc18 \ubb38\uc11c\uc785\ub2c8\ub2e4.",
        )

        engine_a._sync_concept_links(changed_chunk_ids=[chunk_a], deleted_chunk_ids=[])
        concept_candidates = engine_b._search_concept_candidates(CONCEPT_TEXT, candidate_limit=10)

        self.assertEqual(concept_candidates, {})

    def test_concept_links_accept_korean_ocr_text_without_regex_error(self):
        engine = self._build_engine()
        chunk_id = self._insert_raw_chunk(
            engine,
            "ocr.txt",
            "2026\ub144 \ub18d\uac00\uacbd\uc81c\uc870\uc0ac \uc9c0\uce68\uc11c\ub97c \ubcf4\uace0 "
            "\uc870\uc0ac\ud45c \uc791\uc131 \uc808\ucc28\uc640 \uc81c\ucd9c \uc21c\uc11c\ub97c "
            "\uc815\ub9ac\ud569\ub2c8\ub2e4.",
        )

        engine._sync_concept_links(changed_chunk_ids=[chunk_id], deleted_chunk_ids=[])

        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM concept_nodes WHERE normalized_key = ?",
            (SURVEY_TEXT,),
        )
        node_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM chunk_concept_edges WHERE chunk_pk = ?", (chunk_id,))
        edge_count = int(cur.fetchone()[0])
        conn.close()

        self.assertGreaterEqual(node_count, 1)
        self.assertGreater(edge_count, 0)

    def test_query_literal_helpers_accept_unicode_quotes_and_mixed_ids(self):
        engine = self._build_engine()

        literals = engine._extract_query_literals(
            '\u201c\ub18d\uac00\uacbd\uc81c\uc870\uc0ac \uc9c0\uce68\uc11c\u201d API-2026/07 \uc5c5\ub85c\ub4dc'
        )

        self.assertIn("\ub18d\uac00\uacbd\uc81c\uc870\uc0ac \uc9c0\uce68\uc11c", literals)
        self.assertIn("api-2026/07", literals)


if __name__ == "__main__":
    unittest.main()
