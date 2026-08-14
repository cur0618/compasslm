import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from src.ontology_store import OntologyStore, _is_duplicate_column_error


class OntologyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "meta.sqlite"

    def _insert_chunk(self, text: str, source_path: str = "guide.xlsx") -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT,
                source_type TEXT,
                page_no INTEGER,
                line_start INTEGER,
                line_end INTEGER,
                section TEXT,
                text TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT INTO chunks (source_path, source_type, page_no, line_start, line_end, section, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_path, "xlsx", 0, 0, 0, "기준", text),
        )
        chunk_id = int(cur.lastrowid)
        conn.commit()
        conn.close()
        return chunk_id

    def _insert_linked_chunks(self):
        parent_chunk = self._insert_chunk(
            "표의미: kind=definition_block | subject=농업보조금 > 농가경제조사 | 지급단가=40천원",
            source_path="parent.txt",
        )
        child_chunk = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | 지급횟수=12",
            source_path="child.xlsx",
        )
        return parent_chunk, child_chunk

    def test_table_fact_line_becomes_queryable_subject_predicate_value_fact(self):
        chunk_id = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | aliases=농가경제조사, 농어가경제조사 | 지급단가=40천원 | 지급횟수=12"
        )
        store = OntologyStore(str(self.db_path), kb_id="default")

        summary = store.sync_facts_for_chunks([chunk_id], [])
        matches = store.search_facts("농가경제조사 지급단가", limit=5)

        self.assertEqual(summary["ontology_facts_added"], 2)
        self.assertEqual(matches[0]["subject"], "농가경제조사")
        self.assertEqual(matches[0]["predicate"], "지급단가")
        self.assertEqual(matches[0]["object_value"], "40천원")
        self.assertEqual(matches[0]["chunk_id"], chunk_id)

    def test_wiki_publish_signal_boosts_fact_confidence_for_same_source_chunk(self):
        chunk_id = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원"
        )
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([chunk_id], [])

        before = store.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]
        store.apply_wiki_signal(chunk_ids=[chunk_id], signal="published", boost=0.08)
        after = store.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]

        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.0)

    def test_active_low_confidence_fact_is_excluded_until_published(self):
        chunk_id = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원"
        )
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([chunk_id], [])
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE ontology_facts SET confidence = 0.20 WHERE kb_id = ?", ("default",))
        conn.commit()
        conn.close()

        self.assertEqual(store.search_facts("농가경제조사 지급단가", limit=5), [])
        fact = store.list_facts(limit=1)[0]
        store.update_fact_status(int(fact["fact_id"]), "published")

        matches = store.search_facts("농가경제조사 지급단가", limit=5)

        self.assertEqual(matches[0]["status"], "published")

    def test_max_hops_two_finds_related_object_entity_with_lower_score(self):
        parent_chunk, _child_chunk = self._insert_linked_chunks()
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([parent_chunk], [])

        one_hop = store.search_facts("농업보조금 지급단가", limit=5, max_hops=1)
        two_hop = store.search_facts("농업보조금 지급단가", limit=5, max_hops=2)

        self.assertEqual(one_hop, [])
        self.assertEqual(two_hop[0]["subject"], "농가경제조사")
        self.assertEqual(two_hop[0]["ontology_hop_count"], 2)
        self.assertLess(two_hop[0]["score"], 0.78)

    def test_two_hop_does_not_return_relation_only_false_positive(self):
        parent_chunk, _child_chunk = self._insert_linked_chunks()
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([parent_chunk], [])

        matches = store.search_facts("농업보조금 날씨", limit=5, max_hops=2)

        self.assertEqual(matches, [])

    def test_wiki_feedback_event_is_not_applied_twice_to_same_fact(self):
        chunk_id = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원"
        )
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([chunk_id], [])

        before = store.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]
        self.assertEqual(
            store.apply_wiki_signal(chunk_ids=[chunk_id], signal="published", boost=0.08, feedback_key="answer-1"),
            1,
        )
        self.assertEqual(
            store.apply_wiki_signal(chunk_ids=[chunk_id], signal="published", boost=0.08, feedback_key="answer-1"),
            0,
        )
        after = store.search_facts("농가경제조사 지급단가", limit=1)[0]["confidence"]

        self.assertAlmostEqual(after, before + 0.08)

    def test_fact_detail_includes_status_and_confidence_history(self):
        chunk_id = self._insert_chunk(
            "표의미: kind=table_row | subject=농가경제조사 | 지급단가=40천원"
        )
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks([chunk_id], [])
        fact = store.list_facts(limit=1)[0]

        store.update_fact_status(int(fact["fact_id"]), "published", source="admin")
        store.apply_wiki_signal(
            chunk_ids=[chunk_id],
            signal="needs_review",
            feedback_key="answer-history-1",
        )

        detail = store.get_fact_detail(int(fact["fact_id"]))

        self.assertEqual([row["signal"] for row in detail["history"]], ["needs_review", "published"])
        self.assertEqual(detail["history"][0]["source"], "wiki_feedback")
        self.assertEqual(detail["history"][1]["source"], "admin")
        self.assertGreater(
            detail["history"][1]["new_confidence"],
            detail["history"][1]["previous_confidence"],
        )

    def test_query_aliases_are_loaded_from_registry(self):
        store = OntologyStore(str(self.db_path), kb_id="default")

        expanded = store._expand_query_terms("농어가경제조사 답례품 얼마")

        self.assertIn("농가경제조사", expanded["terms"])
        self.assertIn("지급단가", expanded["terms"])
        self.assertIn("금액", expanded["terms"])

    def test_duplicate_column_detection_does_not_hide_other_schema_errors(self):
        self.assertTrue(_is_duplicate_column_error(sqlite3.OperationalError("duplicate column name: evidence_quote")))
        self.assertFalse(_is_duplicate_column_error(sqlite3.OperationalError("disk I/O error")))

    def test_search_can_filter_llm_extraction_and_logs_diagnostics(self):
        chunk_id = self._insert_chunk(
            "농가경제조사 지급단가는 40천원입니다."
        )
        store = OntologyStore(str(self.db_path), kb_id="default")
        store.sync_facts_for_chunks(
            [chunk_id],
            [],
            llm_payloads_by_chunk={
                chunk_id: [{
                    "subject": "농가경제조사",
                    "predicate": "지급단가",
                    "object": "40천원",
                    "confidence": 0.9,
                    "evidence_quote": "농가경제조사 지급단가는 40천원입니다.",
                }]
            },
        )

        deterministic = store.search_facts(
            "농가경제조사 지급단가",
            allowed_extraction_methods={"deterministic_table_fact", "deterministic_definition_path"},
            experiment_mode="deterministic",
        )
        llm = store.search_facts(
            "농가경제조사 지급단가",
            allowed_extraction_methods=None,
            experiment_mode="llm",
        )

        self.assertEqual(deterministic, [])
        self.assertEqual(llm[0]["extraction_method"], "limited_llm")
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT meta_json FROM ontology_query_logs ORDER BY rowid DESC LIMIT 2"
        ).fetchall()
        conn.close()
        latest = json.loads(rows[0][0])
        self.assertEqual(latest["experiment_mode"], "llm")
        self.assertEqual(latest["returned_fact_ids"], [llm[0]["fact_id"]])
        self.assertEqual(latest["returned_chunk_ids"], [chunk_id])
        self.assertGreaterEqual(latest["search_latency_ms"], 0.0)
        self.assertIn("direct_hits", latest["hit_totals"])


if __name__ == "__main__":
    unittest.main()
