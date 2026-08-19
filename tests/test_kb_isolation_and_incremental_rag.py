import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class IncrementalRAGTests(unittest.TestCase):
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
                value = float(len((text or "").strip()) or 1)
                rows.append(np.array([value, value / 10.0, value / 100.0], dtype=np.float32))
            return np.vstack(rows) if rows else np.empty((0, 3), dtype=np.float32)

        engine._encode_texts = fake_encode
        return engine

    def test_ingest_does_not_trigger_full_sqlite_rebuilds(self):
        engine = self._build_engine()
        file_one = Path(self.tmpdir.name) / "first.txt"
        file_two = Path(self.tmpdir.name) / "second.txt"
        file_one.write_text("alpha source\n", encoding="utf-8")
        file_two.write_text("beta source\n", encoding="utf-8")

        engine._rebuild_chunk_vectors_from_db = MagicMock(side_effect=AssertionError("full vector rebuild should not run"))
        engine._rebuild_fts_index_from_db = MagicMock(side_effect=AssertionError("full fts rebuild should not run"))

        engine.ingest_file(str(file_one), original_filename="first.txt")
        engine.ingest_file(str(file_two), original_filename="second.txt")

    def test_dense_sqlite_scoring_can_be_limited_to_candidate_ids(self):
        engine = self._build_engine()
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text, is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("c1", engine.kb_id, "a.txt", "txt", "guide", "alpha", 0, 1),
                ("c2", engine.kb_id, "b.txt", "txt", "guide", "beta", 0, 1),
                ("c3", engine.kb_id, "c.txt", "txt", "guide", "gamma", 0, 1),
            ],
        )
        chunk_ids = [int(row[0]) for row in cur.execute("SELECT id FROM chunks ORDER BY id ASC").fetchall()]
        embeddings = {
            chunk_ids[0]: np.array([0.9, 0.0, 0.0], dtype=np.float32),
            chunk_ids[1]: np.array([0.1, 0.0, 0.0], dtype=np.float32),
            chunk_ids[2]: np.array([0.8, 0.0, 0.0], dtype=np.float32),
        }
        cur.executemany(
            """
            INSERT INTO chunk_vec (chunk_pk, index_name, dim, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (chunk_pk, "large", int(vec.shape[0]), vec.tobytes(), 1)
                for chunk_pk, vec in embeddings.items()
            ],
        )
        conn.commit()
        conn.close()

        results = engine._search_dense_candidates_sqlite(
            index_name="large",
            query_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_k=5,
            candidate_ids=[chunk_ids[1], chunk_ids[2]],
        )

        self.assertEqual(set(results.keys()), {chunk_ids[1], chunk_ids[2]})
        self.assertNotIn(chunk_ids[0], results)

    def test_chunks_schema_migrates_structure_columns_idempotently(self):
        engine = self._build_engine()
        engine._init_db()

        conn = engine._connect_db()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        conn.close()

        self.assertTrue(
            {
                "embedding_text",
                "chunk_kind",
                "heading_path_json",
                "parent_chunk_key",
                "structure_path",
                "table_id",
                "row_no",
                "cell_no",
                "is_derived",
            }.issubset(columns)
        )

    def test_search_artifacts_embed_structure_text_but_keep_original_chunk_text(self):
        engine = self._build_engine()
        captured = []

        def capture_encode(index_name: str, texts, task: str):
            captured.extend(texts)
            return np.vstack([np.array([0.1, 0.0, 0.0], dtype=np.float32) for _ in texts])

        engine._encode_texts = capture_encode
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text, embedding_text,
                 is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "structured:1",
                engine.kb_id,
                "guide.hwpx",
                "hwpx",
                "guide",
                "원문 문장",
                "문서: guide.hwpx\n유형: 예외\n내용: 원문 문장",
                0,
                1,
            ),
        )
        chunk_pk = int(cur.lastrowid)
        conn.commit()
        conn.close()

        engine._sync_sqlite_search_artifacts(changed_chunk_ids=[chunk_pk])

        self.assertEqual(captured, ["문서: guide.hwpx\n유형: 예외\n내용: 원문 문장"])
        conn = engine._connect_db()
        stored_text = conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_pk,)).fetchone()[0]
        conn.close()
        self.assertEqual(stored_text, "원문 문장")

    def test_structure_enabled_ingest_persists_structure_columns(self):
        engine = self._build_engine()
        engine.structure_rag_v2_enabled = True
        engine.xlsx_structure_rag_v2_enabled = True
        file_path = Path(self.tmpdir.name) / "guide.xlsx"

        try:
            import openpyxl
        except Exception:
            self.skipTest("openpyxl unavailable")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "지급기준"
        sheet.append(["조사명", "지급단가"])
        sheet.append(["농가경제조사", 40])
        workbook.save(file_path)

        result = engine.ingest_file(str(file_path), original_filename="지급기준.xlsx", document_role="guide")

        conn = engine._connect_db()
        row = conn.execute(
            """
            SELECT text, embedding_text, chunk_kind, heading_path_json,
                   parent_chunk_key, table_id, row_no, is_derived
            FROM chunks
            WHERE source_type = 'xlsx' AND COALESCE(is_normalized, 0) = 0
            ORDER BY id ASC
            LIMIT 1
            """
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertIn("농가경제조사", row[0])
        self.assertIn("path=지급기준", row[1])
        self.assertEqual(row[2], "table_row")
        self.assertEqual(row[3], '["지급기준"]')
        self.assertTrue(row[4])
        self.assertEqual(row[5], "지급기준:table:1")
        self.assertEqual(row[6], 2)
        self.assertEqual(row[7], 0)
        self.assertEqual(len(result["ontology_chunk_ids"]), 1)
        self.assertTrue(all(isinstance(value, int) for value in result["ontology_chunk_ids"]))

    def test_parent_result_limit_keeps_best_result_per_parent(self):
        engine = self._build_engine()
        engine.structure_rag_v2_enabled = True
        engine.structure_rag_parent_result_limit = 1
        results = [
            {"id": 1, "score": 0.9, "parent_chunk_key": "parent-a", "is_derived": 1},
            {"id": 2, "score": 0.8, "parent_chunk_key": "parent-a", "is_derived": 0},
            {"id": 3, "score": 0.7, "parent_chunk_key": "parent-b", "is_derived": 0},
            {"id": 4, "score": 0.6, "parent_chunk_key": "", "is_derived": 0},
        ]

        limited = engine._limit_results_by_parent(results, top_k=4)

        self.assertEqual([row["id"] for row in limited], [1, 3, 4])
        self.assertEqual(limited[0]["parent_result_rank"], 1)

    def test_structure_metadata_is_exposed_on_search_result(self):
        engine = self._build_engine()
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text, embedding_text,
                 chunk_kind, heading_path_json, parent_chunk_key, table_id, row_no,
                 is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "structure-search:1",
                engine.kb_id,
                "guide.hwpx",
                "hwpx",
                "guide",
                "해외 농가는 제외한다.",
                "문서: guide.hwpx\n경로: 제2조 조사대상\n유형: 예외\n내용: 해외 농가는 제외한다.",
                "exception",
                '["제2조 조사대상"]',
                "parent-1",
                "",
                0,
                0,
                1,
            ),
        )
        chunk_pk = int(cur.lastrowid)
        conn.commit()
        conn.close()
        engine._sync_sqlite_search_artifacts(changed_chunk_ids=[chunk_pk])

        results = engine.search("제2조 조사대상 예외", top_k=1)

        self.assertEqual(results[0]["chunk_kind"], "exception")
        self.assertEqual(results[0]["heading_path"], ["제2조 조사대상"])
        self.assertEqual(results[0]["parent_chunk_key"], "parent-1")
        self.assertGreater(results[0]["structure_boost"], 0.0)
        self.assertIn("ontology_match", results[0])

    def test_derived_result_is_grounded_with_original_parent_text(self):
        engine = self._build_engine()
        conn = engine._connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text,
                 parent_chunk_key, chunk_kind, is_derived, is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("parent", engine.kb_id, "guide.xlsx", "xlsx", "guide", "농가경제조사 | 40", "row-2", "table_row", 0, 0, 1),
        )
        parent_id = int(cur.lastrowid)
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, kb_id, source_path, source_type, doc_role, text,
                 parent_chunk_key, chunk_kind, is_derived, is_normalized, source_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("derived", engine.kb_id, "guide.xlsx", "xlsx", "guide", "표의미: 지급단가=40천원", "row-2", "table_summary", 1, 0, 1),
        )
        derived_id = int(cur.lastrowid)
        conn.commit()
        conn.close()

        expanded = engine._ground_derived_results_to_parents(
            [{"id": derived_id, "text": "표의미: 지급단가=40천원", "parent_chunk_key": "row-2", "is_derived": 1}]
        )

        self.assertEqual(expanded[0]["text"], "농가경제조사 | 40")
        self.assertEqual(expanded[0]["parent_chunk_id"], parent_id)
        self.assertTrue(expanded[0]["parent_expanded"])
        self.assertEqual(expanded[0]["derived_text"], "표의미: 지급단가=40천원")


class KBEngineRegistryTests(unittest.TestCase):
    def test_registry_evicts_inactive_kbs_when_limit_is_one(self):
        from src.kb_engine_registry import KBEngineRegistry

        registry = KBEngineRegistry(max_loaded_kbs=1, idle_ttl_seconds=3600)

        created = []

        def factory(kb_name: str):
            engine = SimpleNamespace(kb_id=kb_name, close=MagicMock())
            created.append(engine)
            return engine

        first = registry.get_or_create("test1", factory)
        second = registry.get_or_create("test2", factory)

        self.assertIsNot(first, second)
        self.assertEqual(list(registry._instances.keys()), ["test2"])
        first.close.assert_called_once()
        self.assertFalse(second.close.called)

    def test_registry_lease_prevents_eviction_until_request_finishes(self):
        from src.kb_engine_registry import KBEngineRegistry

        registry = KBEngineRegistry(max_loaded_kbs=1, idle_ttl_seconds=3600)

        def factory(kb_name: str):
            return SimpleNamespace(kb_id=kb_name, close=MagicMock())

        with registry.lease("test1", factory) as first:
            second = registry.get_or_create("test2", factory)
            self.assertFalse(first.close.called)
            self.assertFalse(second.close.called)
            self.assertEqual(registry.snapshot_count(), 2)

        first.close.assert_called_once()
        self.assertEqual(list(registry._instances.keys()), ["test2"])
        self.assertFalse(second.close.called)

    def test_remove_defers_close_for_a_leased_engine(self):
        from src.kb_engine_registry import KBEngineRegistry

        registry = KBEngineRegistry(max_loaded_kbs=1, idle_ttl_seconds=3600)

        def factory(kb_name: str):
            return SimpleNamespace(kb_id=kb_name, close=MagicMock())

        with registry.lease("test1", factory) as first:
            registry.remove("test1")
            self.assertFalse(first.close.called)
            replacement = registry.get_or_create("test1", factory)
            self.assertIsNot(first, replacement)

        first.close.assert_called_once()
        self.assertFalse(replacement.close.called)


if __name__ == "__main__":
    unittest.main()
