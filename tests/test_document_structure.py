import unittest

from src.document_structure import build_embedding_text, chunk_structure_records, normalize_structure_record


class DocumentStructureTests(unittest.TestCase):
    def test_build_embedding_text_compacts_repeated_metadata(self):
        value = build_embedding_text(
            text="Payment support amount is 40000 won.",
            source_path="guide.hwpx",
            doc_role="guide",
            heading_path=["Chapter 1", "Support Rules"],
            chunk_kind="table_row",
        )

        self.assertIn("source=guide.hwpx", value)
        self.assertIn("role=guide", value)
        self.assertIn("path=Chapter 1 > Support Rules", value)
        self.assertIn("kind=table_row", value)
        self.assertTrue(value.endswith("Payment support amount is 40000 won."))

    def test_normalize_structure_record_keeps_original_text_and_builds_embedding_text(self):
        record = normalize_structure_record(
            {
                "text": "This exception applies only to direct support.",
                "file_path": "rule.hwpx",
                "heading_path": ["Article 5"],
                "chunk_kind": "exception",
                "line_start": 8,
                "line_end": 8,
            },
            source_type="hwpx",
            doc_role="guide",
        )

        self.assertEqual(record["text"], "This exception applies only to direct support.")
        self.assertIn("kind=exception", record["embedding_text"])
        self.assertIn("path=Article 5", record["embedding_text"])
        self.assertEqual(record["heading_path"], ["Article 5"])
        self.assertFalse(record["is_derived"])

    def test_chunk_structure_records_merges_table_cells_without_crossing_sections(self):
        records = [
            {"text": "Purpose", "chunk_kind": "heading", "heading_path": ["Purpose"], "parent_chunk_key": "p1"},
            {"text": "First sentence", "chunk_kind": "body", "heading_path": ["Purpose"], "parent_chunk_key": "p2"},
            {"text": "Eligibility", "chunk_kind": "heading", "heading_path": ["Eligibility"], "parent_chunk_key": "p3"},
            {"text": "Second sentence", "chunk_kind": "body", "heading_path": ["Eligibility"], "parent_chunk_key": "p4"},
            {"text": "Crop exchange", "chunk_kind": "table_row", "heading_path": ["Eligibility"], "parent_chunk_key": "t1", "table_id": "tbl[1]", "row_no": 2, "cell_no": 1},
            {"text": "40", "chunk_kind": "table_row", "heading_path": ["Eligibility"], "parent_chunk_key": "t1", "table_id": "tbl[1]", "row_no": 2, "cell_no": 2},
        ]

        chunks = chunk_structure_records(records, source_type="hwpx", doc_role="guide", source_path="sample.hwpx")

        self.assertFalse(any("First sentence" in c["text"] and "Second sentence" in c["text"] for c in chunks))
        table_chunks = [c for c in chunks if c["parent_chunk_key"] == "t1"]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0]["text"], "Crop exchange | 40")
        self.assertEqual(table_chunks[0]["chunk_kind"], "table_row")


if __name__ == "__main__":
    unittest.main()
