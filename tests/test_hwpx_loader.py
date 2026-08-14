import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path


class _FakeParagraph:
    def __init__(self, section_index, paragraph_index, path, text, is_nested=False, style_name="", outline_level=0):
        self.section = types.SimpleNamespace(index=section_index)
        self.index = paragraph_index
        self.path = path
        self.is_nested = is_nested
        self.style_name = style_name
        self.outline_level = outline_level
        self._text = text

    def text(self):
        return self._text

    @property
    def _text(self):
        return self.__dict__["text_value"]

    @_text.setter
    def _text(self, value):
        self.__dict__["text_value"] = value


class _FakeTextExtractor:
    paragraphs = []

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_document_paragraphs(self, include_nested=True):
        return iter(self.paragraphs if include_nested else [p for p in self.paragraphs if not p.is_nested])


def _install_fake_hwpx():
    module = types.ModuleType("hwpx")
    module.TextExtractor = _FakeTextExtractor
    sys.modules["hwpx"] = module


def _write_hwpx_zip(path: Path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/hwp+zip")
        zf.writestr("Contents/content.hpf", "<package/>")
        zf.writestr("Contents/section0.xml", "<section/>")


class HwpxLoaderTests(unittest.TestCase):
    def setUp(self):
        _install_fake_hwpx()
        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "제1조 목적", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/p[2]", "이 문서는 HWPX 테스트입니다.", False),
            _FakeParagraph(0, 2, "Contents/section0.xml/tbl/tr/tc/p[1]", "품목 단가", True),
            _FakeParagraph(1, 0, "Contents/section1.xml/p[1]", "다음 section 첫 문단", False),
            _FakeParagraph(1, 1, "Contents/section1.xml/p[2]", "   ", False),
        ]

    def tearDown(self):
        sys.modules.pop("hwpx", None)
        sys.modules.pop("src.hwpx_loader", None)

    def test_load_hwpx_records_preserves_paragraph_and_nested_table_metadata(self):
        from src.hwpx_loader import load_hwpx_records

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)

            records = load_hwpx_records(str(hwpx_path))

        self.assertEqual(
            [record["text"] for record in records],
            ["제1조 목적", "이 문서는 HWPX 테스트입니다.", "[표] 품목 단가", "다음 section 첫 문단"],
        )
        self.assertEqual(records[0]["section"], "제1조 목적")
        self.assertEqual(records[1]["section"], "제1조 목적")
        self.assertEqual(records[2]["section"], "제1조 목적")
        self.assertEqual(records[2]["hwpx_path"], "Contents/section0.xml/tbl/tr/tc/p[1]")
        self.assertTrue(records[2]["is_nested"])
        self.assertEqual(records[2]["line_start"], 3)
        self.assertEqual(records[3]["hwpx_paragraph_index"], 0)
        self.assertEqual(records[3]["line_start"], 4)

    def test_load_hwpx_records_can_exclude_nested_table_paragraphs(self):
        from src.hwpx_loader import load_hwpx_records

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)

            records = load_hwpx_records(str(hwpx_path), include_tables=False)

        self.assertEqual([record["text"] for record in records], ["제1조 목적", "이 문서는 HWPX 테스트입니다.", "다음 section 첫 문단"])

    def test_load_hwpx_records_adds_table_row_summary_and_survey_aliases(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "답례품 지급 계획", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/tbl[1]/tr[1]/tc[1]/p[1]", "답례품", True),
            _FakeParagraph(0, 2, "Contents/section0.xml/tbl[1]/tr[1]/tc[2]/p[1]", "조사명", True),
            _FakeParagraph(0, 3, "Contents/section0.xml/tbl[1]/tr[1]/tc[3]/p[1]", "지급단가", True),
            _FakeParagraph(0, 4, "Contents/section0.xml/tbl[1]/tr[2]/tc[1]/p[1]", "지류, 현금", True),
            _FakeParagraph(0, 5, "Contents/section0.xml/tbl[1]/tr[2]/tc[2]/p[1]", "농어가경제조사", True),
            _FakeParagraph(0, 6, "Contents/section0.xml/tbl[1]/tr[2]/tc[3]/p[1]", "40", True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)

            records = load_hwpx_records(str(hwpx_path))

        summary_records = [record for record in records if record["text"].startswith("표행요약:")]
        value_records = [record for record in records if record["text"].startswith("표값:")]
        self.assertTrue(any("농어가경제조사" in record["text"] for record in summary_records))
        self.assertTrue(any("농가경제조사" in record["text"] for record in summary_records))
        self.assertTrue(any("지급단가=40천원" in record["text"] for record in summary_records))
        self.assertTrue(any("농어가경제조사 지급단가 40천원" in record["text"] for record in value_records))

    def test_load_hwpx_records_adds_flattened_table_summary_when_path_has_no_indexes(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "답례품 지급 계획", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/tbl/tr/tc/p", "지류, 현금", True),
            _FakeParagraph(0, 2, "Contents/section0.xml/tbl/tr/tc/p", "농어가경제조사", True),
            _FakeParagraph(0, 3, "Contents/section0.xml/tbl/tr/tc/p", "4,300", True),
            _FakeParagraph(0, 4, "Contents/section0.xml/tbl/tr/tc/p", "40", True),
            _FakeParagraph(0, 5, "Contents/section0.xml/tbl/tr/tc/p", "12", True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)

            records = load_hwpx_records(str(hwpx_path))

        summary_text = "\n".join(record["text"] for record in records if record["text"].startswith("표행요약:"))
        self.assertIn("조사명=농어가경제조사", summary_text)
        self.assertIn("명칭별칭=농어가경제조사, 농가경제조사, 어가경제조사", summary_text)
        self.assertIn("지급단가=40천원", summary_text)

    def test_load_hwpx_records_generates_generic_name_aliases_from_table_values(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "답례품 지급 계획", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/tbl[1]/tr[1]/tc[1]/p[1]", "조사명", True),
            _FakeParagraph(0, 2, "Contents/section0.xml/tbl[1]/tr[1]/tc[2]/p[1]", "지급단가", True),
            _FakeParagraph(0, 3, "Contents/section0.xml/tbl[1]/tr[2]/tc[1]/p[1]", "농·어가경제조사", True),
            _FakeParagraph(0, 4, "Contents/section0.xml/tbl[1]/tr[2]/tc[2]/p[1]", "40", True),
            _FakeParagraph(0, 5, "Contents/section0.xml/tbl[1]/tr[3]/tc[1]/p[1]", "경제활동인구조사(부가조사)", True),
            _FakeParagraph(0, 6, "Contents/section0.xml/tbl[1]/tr[3]/tc[2]/p[1]", "20", True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)

            records = load_hwpx_records(str(hwpx_path))

        summary_text = "\n".join(record["text"] for record in records if record["text"].startswith("표행요약:"))
        self.assertIn("명칭별칭=농·어가경제조사, 농어가경제조사, 농가경제조사, 어가경제조사", summary_text)
        self.assertIn("명칭별칭=경제활동인구조사(부가조사), 경제활동인구조사, 부가조사", summary_text)
        self.assertNotIn("조사명별칭=", summary_text)

    def test_load_hwpx_records_restores_heading_path_and_body_kinds(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "제1장 총칙", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/p[2]", "제2조 조사대상", False),
            _FakeParagraph(0, 2, "Contents/section0.xml/p[3]", "국내 농가를 대상으로 한다.", False),
            _FakeParagraph(0, 3, "Contents/section0.xml/p[4]", "다만 해외 농가는 제외한다.", False),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)
            records = load_hwpx_records(str(hwpx_path), structure_v2=True)

        self.assertEqual(records[2]["heading_path"], ["제1장 총칙", "제2조 조사대상"])
        self.assertEqual(records[2]["chunk_kind"], "body")
        self.assertEqual(records[3]["chunk_kind"], "exception")
        self.assertEqual(records[0]["chunk_kind"], "heading")

    def test_load_hwpx_records_prefers_style_heading_when_text_has_no_heading_number(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "조사 개요", False, style_name="Heading 1", outline_level=1),
            _FakeParagraph(0, 1, "Contents/section0.xml/p[2]", "조사 대상", False, style_name="Heading 2", outline_level=2),
            _FakeParagraph(0, 2, "Contents/section0.xml/p[3]", "국내 농가를 대상으로 한다.", False),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)
            records = load_hwpx_records(str(hwpx_path), structure_v2=True)

        self.assertEqual(records[2]["heading_path"], ["조사 개요", "조사 대상"])
        self.assertEqual(records[0]["heading_source"], "style")

    def test_load_hwpx_records_links_table_rows_and_derived_summaries_to_parent(self):
        from src.hwpx_loader import load_hwpx_records

        _FakeTextExtractor.paragraphs = [
            _FakeParagraph(0, 0, "Contents/section0.xml/p[1]", "제1장 지급", False),
            _FakeParagraph(0, 1, "Contents/section0.xml/tbl[1]/tr[1]/tc[1]/p[1]", "조사명", True),
            _FakeParagraph(0, 2, "Contents/section0.xml/tbl[1]/tr[1]/tc[2]/p[1]", "지급단가", True),
            _FakeParagraph(0, 3, "Contents/section0.xml/tbl[1]/tr[2]/tc[1]/p[1]", "농가경제조사", True),
            _FakeParagraph(0, 4, "Contents/section0.xml/tbl[1]/tr[2]/tc[2]/p[1]", "40", True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "sample.hwpx"
            _write_hwpx_zip(hwpx_path)
            records = load_hwpx_records(str(hwpx_path), structure_v2=True)

        row_cells = [r for r in records if r.get("table_id") == "tbl[1]" and r.get("row_no") == 2 and not r.get("is_derived")]
        summaries = [r for r in records if r.get("chunk_kind") == "table_summary" and r.get("row_no") == 2]
        self.assertTrue(row_cells)
        self.assertTrue(summaries)
        self.assertEqual(summaries[0]["parent_chunk_key"], row_cells[0]["parent_chunk_key"])
        self.assertEqual(summaries[0]["heading_path"], ["제1장 지급"])


if __name__ == "__main__":
    unittest.main()
