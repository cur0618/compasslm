import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chat_policy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_chat_policy", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BroadSummaryPolicyTests(unittest.TestCase):
    def test_detects_broad_summary_request(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        self.assertTrue(module.is_broad_summary_request("업로드한 문서를 확인해서 주의사항을 정리해줘"))
        self.assertFalse(module.is_broad_summary_request("농가경제조사 답례품 단가 알려줘"))

    def test_prompts_narrower_when_pdf_sources_are_many(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        results = [
            {"source_type": "pdf", "source_display": "A.pdf"},
            {"source_type": "pdf", "source_display": "B.pdf"},
            {"source_type": "pdf", "source_display": "C.pdf"},
            {"source_type": "pdf", "source_display": "D.pdf"},
        ]
        self.assertTrue(
            module.should_prompt_for_narrower_summary(
                "업로드한 문서를 확인해서 주의사항을 정리해줘",
                metrics={"unique_sources": 4, "top1": 0.31, "coverage": 0.19},
                results=results,
                kb_file_count=4,
                overview_mode=True,
            )
        )

    def test_allows_summary_when_pdf_sources_are_few(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        results = [
            {"source_type": "pdf", "source_display": "2024농가경제조사지침서.pdf"},
            {"source_type": "pdf", "source_display": "2024농가경제조사지침서.pdf"},
        ]
        self.assertFalse(
            module.should_prompt_for_narrower_summary(
                "업로드한 문서를 확인해서 주의사항을 정리해줘",
                metrics={"unique_sources": 1, "top1": 0.62, "coverage": 0.81},
                results=results,
                kb_file_count=1,
                overview_mode=False,
            )
        )

    def test_scope_narrowing_response_includes_refinement_examples(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        text = module.build_scope_narrowing_response(
            "업로드한 문서를 확인해서 주의사항을 정리해줘",
            results=[{"source_type": "pdf", "source_display": "2024농가경제조사지침서.pdf"}],
        )
        self.assertIn("범위를 조금만 좁혀", text)
        self.assertIn("25페이지", text)
        self.assertIn("2024농가경제조사지침서", text)
