import importlib.util
import sys
import types
import unittest
from pathlib import Path

from src.rag_text_markers import (
    EMBEDDING_DIMENSION_PROBE_TEXT,
    TABLE_HINT_MARKERS,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_rag_module():
    stubs = {
        "hnswlib": types.SimpleNamespace(Index=object),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "src.pdf_ocr": types.SimpleNamespace(
            extract_pdf_pages=lambda *args, **kwargs: {},
            extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
            release_cached_ocr_model=lambda *args, **kwargs: None,
            shutdown_persistent_ocr_worker=lambda *args, **kwargs: None,
        ),
        "src.hwpx_loader": types.SimpleNamespace(load_hwpx_records=lambda *args, **kwargs: []),
        "src.utils": types.SimpleNamespace(
            chunk_txt_items=lambda *args, **kwargs: [],
            chunk_xlsx_rows=lambda *args, **kwargs: [],
            load_txt=lambda *args, **kwargs: [],
            load_xlsx=lambda *args, **kwargs: [],
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            "codex_test_rag_text_markers",
            ROOT / "src" / "rag.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


class RagTextMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag_module = _load_rag_module()

    def setUp(self):
        self.engine = object.__new__(self.rag_module.RAGEngine)

    def test_embedding_diagnostics_use_readable_korean(self):
        self.assertEqual(EMBEDDING_DIMENSION_PROBE_TEXT, "임베딩 차원 확인")
        hint = self.rag_module._embedding_model_requirements_hint(
            "jinaai/jina-embeddings-v5-text-small"
        )
        self.assertIn("환경이 필요할 수 있습니다", hint)

    def test_korean_question_and_answer_columns_are_detected(self):
        records = self.engine._extract_xlsx_qa_records(
            "[Sheet: 현장사례]\nRow 3: 질문=품앗이는 어떻게 조사합니까? | 답변=노동 교환으로 처리합니다.",
            default_sheet="",
            default_row=0,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question"], "품앗이는 어떻게 조사합니까?")
        self.assertEqual(records[0]["answer"], "노동 교환으로 처리합니다.")
        self.assertEqual(records[0]["sheet"], "현장사례")
        self.assertEqual(records[0]["row"], 3)

    def test_location_and_source_references_are_readable(self):
        location = self.engine._build_txt_location_label(
            {"line_start": 10, "line_end": 12, "section": ""}
        )
        self.assertEqual(location, "라인 10-12")

        self.engine._format_timestamp = lambda _value: "2026-07-31 09:00"
        source_ref = self.engine.format_source_ref(
            {
                "is_normalized": 1,
                "normalized_group": "xlsx",
                "source_type": "xlsx",
                "row": 2,
                "source_updated_at": 1,
                "doc_role": "",
            }
        )
        self.assertEqual(
            source_ref,
            "통합 정리 XLSX 2번 / 최신 업로드 반영 2026-07-31 09:00",
        )

    def test_table_markers_match_the_ocr_and_table_fact_contract(self):
        self.assertEqual(
            TABLE_HINT_MARKERS,
            ("표행:", "표행요약:", "표헤더:", "표값:", "표의미:"),
        )
        row = {
            "source_type": "pdf",
            "source_path": "guide.pdf",
            "page_no": 1,
            "text": "표행요약: 조사명 지급단가 | 농가경제조사 40천원",
        }
        self.assertTrue(self.engine._should_use_lazy_ocr_for_row(row))


if __name__ == "__main__":
    unittest.main()
