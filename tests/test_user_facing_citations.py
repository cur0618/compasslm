import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "citation_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_citation_labels", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UserFacingCitationTests(unittest.TestCase):
    def test_pdf_citation_becomes_filename_and_page(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "section": "PDF page 27",
                "page_no": 27,
            }
        )
        self.assertEqual(label, "2024농가경제조사지침서 27페이지")

    def test_prefixed_pdf_citation_hides_generated_upload_prefix(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "20260402102553_63bbf2d0e3__2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 11,
            }
        )
        self.assertEqual(label, "2024농가경제조사지침서 11페이지")

    def test_pdf_citation_ignores_implausible_section_page_without_explicit_page_no(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "2024농가경제조사사례집.pdf",
                "source_type": "pdf",
                "section": "PDF page 67418",
            }
        )
        self.assertEqual(label, "2024농가경제조사사례집")

    def test_single_underscore_prefixed_pdf_citation_hides_generated_upload_prefix(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "20260403_e123g2234_2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 11,
            }
        )
        self.assertEqual(label, "2024농가경제조사지침서 11페이지")

    def test_source_display_takes_priority_over_stored_source_path(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_display": "2024농가경제조사지침서.pdf",
                "source_path": "20260402102553_63bbf2d0e3__2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 11,
            }
        )
        self.assertEqual(label, "2024농가경제조사지침서 11페이지")

    def test_nested_raw_row_source_display_takes_priority_over_stored_source_path(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "metadata": {
                    "raw_row": {
                        "source_display": "2024농가경제조사지침서.pdf",
                    }
                },
                "source_path": "20260402102553_63bbf2d0e3__2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 11,
            }
        )
        self.assertEqual(label, "2024농가경제조사지침서 11페이지")

    def test_xlsx_citation_becomes_filename_sheet_and_row(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "농가조사표.xlsx",
                "source_type": "xlsx",
                "sheet": "기본정보",
                "row": 12,
                "row_end": 12,
            }
        )
        self.assertEqual(label, "농가조사표 / 기본정보 / 12행")

    def test_txt_citation_becomes_filename_and_line_range(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        label = module.build_user_facing_citation_label(
            {
                "source_path": "양봉메모.txt",
                "source_type": "txt",
                "line_start": 34,
                "line_end": 36,
            }
        )
        self.assertEqual(label, "양봉메모 / 34-36라인")

    def test_final_user_visible_answer_hides_internal_doc_labels(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        answer = "처리 방법은 환풍기 기준으로 보시면 됩니다. [DOC 1]"
        docs = {
            1: {
                "source_path": "2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "section": "PDF page 27",
                "page_no": 27,
            }
        }

        rendered = module.replace_doc_citations(answer, docs)

        self.assertNotIn("[DOC 1]", rendered)
        self.assertIn("[[CITATION:1|2024농가경제조사지침서 27페이지]]", rendered)

    def test_final_user_visible_answer_normalizes_legacy_doc_tokens(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        answer = "처리 방법은 환풍기 기준으로 보시면 됩니다. [[DOC 1|저장용이름]]"
        docs = {
            1: {
                "source_path": "20260402102553_63bbf2d0e3__2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 27,
            }
        }

        rendered = module.replace_doc_citations(answer, docs)

        self.assertNotIn("[[DOC 1|저장용이름]]", rendered)
        self.assertIn("[[CITATION:1|2024농가경제조사지침서 27페이지]]", rendered)

    def test_final_user_visible_answer_normalizes_legacy_citation_tokens(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        answer = "처리 방법은 환풍기 기준으로 보시면 됩니다. [[CITATION:1|저장용이름]]"
        docs = {
            1: {
                "source_path": "20260402102553_63bbf2d0e3__2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 27,
            }
        }

        rendered = module.replace_doc_citations(answer, docs)

        self.assertNotIn("[[CITATION:1|저장용이름]]", rendered)
        self.assertIn("[[CITATION:1|2024농가경제조사지침서 27페이지]]", rendered)

    def test_final_user_visible_answer_normalizes_loose_citation_tokens(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        answer = "양곡소비량조사는 10천원입니다. [CITATION:3|저장용이름]"
        docs = {
            3: {
                "source_path": "20260511141142_c61e6cd704__2026년_통계조사답례품_지급_관리_계획_안_3_.hwpx",
                "source_type": "hwpx",
                "line_start": 641,
                "line_end": 882,
            }
        }

        rendered = module.replace_doc_citations(answer, docs)

        self.assertNotIn("[CITATION:3|저장용이름]", rendered)
        self.assertIn("[[CITATION:3|2026년_통계조사답례품_지급_관리_계획_안_3_ / 641-882라인]]", rendered)

    def test_citation_token_uses_doc_number_and_hover_label(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        citation_marker = module.build_user_facing_citation_token(
            7,
            {
                "source_path": "2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "section": "PDF page 25",
                "page_no": 25,
            },
        )
        self.assertEqual(citation_marker, "[[CITATION:7|2024농가경제조사지침서 25페이지]]")

    def test_answer_citations_move_to_bottom_and_renumber_by_first_appearance(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        answer = "첫 판단입니다. [DOC 4]\n두 번째 판단입니다. [DOC 5]\n첫 근거를 다시 씁니다. [DOC 4]\n예전 근거도 섞였습니다. [[CITATION:1|저장용이름]]"
        docs = {
            1: {
                "source_path": "2024농가경제조사사례집.pdf",
                "source_type": "pdf",
                "page_no": 21,
            },
            4: {
                "source_path": "2024농가경제조사지침서.pdf",
                "source_type": "pdf",
                "page_no": 27,
            },
            5: {
                "source_path": "농가조사표.xlsx",
                "source_type": "xlsx",
                "sheet": "기본정보",
                "row": 12,
            },
        }

        rendered = module.render_answer_with_bottom_citations(answer, docs)

        self.assertNotIn("[DOC 4]", rendered)
        self.assertNotIn("[DOC 5]", rendered)
        self.assertNotIn("저장용이름", rendered)
        self.assertIn("첫 판단입니다.", rendered)
        self.assertIn("두 번째 판단입니다.", rendered)
        self.assertTrue(
            rendered.endswith(
                "근거: [[CITATION:1|2024농가경제조사지침서 27페이지]], [[CITATION:2|농가조사표 / 기본정보 / 12행]], [[CITATION:3|2024농가경제조사사례집 21페이지]]"
            )
        )


if __name__ == "__main__":
    unittest.main()
