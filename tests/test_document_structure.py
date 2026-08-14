import unittest

from src.document_structure import build_embedding_text, chunk_structure_records, normalize_structure_record


class DocumentStructureTests(unittest.TestCase):
    def test_build_embedding_text_uses_document_role_path_kind_and_original_text(self):
        value = build_embedding_text(
            text="농가경제조사의 지급단가는 40천원이다.",
            source_path="지침서.hwpx",
            doc_role="guide",
            heading_path=["제2장 조사", "제4조 지급"],
            chunk_kind="table_row",
        )

        self.assertEqual(
            value,
            "문서: 지침서.hwpx\n"
            "역할: guide\n"
            "경로: 제2장 조사 > 제4조 지급\n"
            "유형: 표 행\n"
            "내용: 농가경제조사의 지급단가는 40천원이다.",
        )

    def test_normalize_structure_record_keeps_original_text_and_builds_embedding_text(self):
        record = normalize_structure_record(
            {
                "text": "다만 해외 표본은 제외한다.",
                "file_path": "규정.hwpx",
                "heading_path": ["제3조 대상"],
                "chunk_kind": "exception",
                "line_start": 8,
                "line_end": 8,
            },
            source_type="hwpx",
            doc_role="guide",
        )

        self.assertEqual(record["text"], "다만 해외 표본은 제외한다.")
        self.assertIn("유형: 예외", record["embedding_text"])
        self.assertIn("경로: 제3조 대상", record["embedding_text"])
        self.assertEqual(record["heading_path"], ["제3조 대상"])
        self.assertFalse(record["is_derived"])

    def test_chunk_structure_records_does_not_cross_heading_or_table_parent(self):
        records = [
            {"text": "제1조 목적", "chunk_kind": "heading", "heading_path": ["제1조 목적"], "parent_chunk_key": "p1"},
            {"text": "첫 문장", "chunk_kind": "body", "heading_path": ["제1조 목적"], "parent_chunk_key": "p2"},
            {"text": "제2조 대상", "chunk_kind": "heading", "heading_path": ["제2조 대상"], "parent_chunk_key": "p3"},
            {"text": "둘째 문장", "chunk_kind": "body", "heading_path": ["제2조 대상"], "parent_chunk_key": "p4"},
            {"text": "농가경제조사", "chunk_kind": "table_row", "heading_path": ["제2조 대상"], "parent_chunk_key": "t1", "table_id": "tbl[1]", "row_no": 2, "cell_no": 1},
            {"text": "40", "chunk_kind": "table_row", "heading_path": ["제2조 대상"], "parent_chunk_key": "t1", "table_id": "tbl[1]", "row_no": 2, "cell_no": 2},
        ]

        chunks = chunk_structure_records(records, source_type="hwpx", doc_role="guide", source_path="sample.hwpx")

        self.assertFalse(any("첫 문장" in c["text"] and "둘째 문장" in c["text"] for c in chunks))
        table_chunks = [c for c in chunks if c["parent_chunk_key"] == "t1"]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0]["text"], "농가경제조사 | 40")
        self.assertEqual(table_chunks[0]["chunk_kind"], "table_row")


if __name__ == "__main__":
    unittest.main()
