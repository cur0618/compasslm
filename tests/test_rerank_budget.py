import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "rerank_budget.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_rerank_budget", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RerankBudgetTests(unittest.TestCase):
    def test_trim_rerank_candidate_lines_keeps_prompt_within_context_limit(self):
        rerank_module = _load_module()
        user_message = "긴 질의 " * 400
        candidate_lines = [
            f"{idx}) file=sample-{idx}.txt | loc=- | uploaded=2026-03-12 10:00 | score=0.500 | norm=0 | code=0 | text="
            + ("본문 " * 180)
            for idx in range(1, 73)
        ]

        trimmed_lines, meta = rerank_module.trim_rerank_candidate_lines_to_budget(
            user_message=user_message,
            candidate_lines=candidate_lines,
            keep_n=12,
            llm_context_limit=8192,
            helper_max_tokens=120,
            prompt_overhead_tokens=280,
            safety_margin=120,
        )

        prompt = rerank_module.build_rerank_prompt(user_message, trimmed_lines)
        estimated_prompt_tokens = rerank_module.estimate_mixed_tokens(prompt) + 280

        self.assertGreaterEqual(len(trimmed_lines), 12)
        self.assertLess(len(trimmed_lines), len(candidate_lines))
        self.assertLessEqual(estimated_prompt_tokens + 120, 8192)
        self.assertEqual(meta["selected_count"], len(trimmed_lines))
        self.assertGreater(meta["trimmed_count"], 0)

    def test_trim_rerank_candidate_lines_can_truncate_line_width_when_needed(self):
        rerank_module = _load_module()
        user_message = "매우 긴 질문 " * 300
        candidate_lines = [
            f"{idx}) file=sample-{idx}.txt | loc=- | uploaded=2026-03-12 10:00 | score=0.500 | norm=0 | code=0 | text="
            + ("긴본문 " * 300)
            for idx in range(1, 13)
        ]

        trimmed_lines, meta = rerank_module.trim_rerank_candidate_lines_to_budget(
            user_message=user_message,
            candidate_lines=candidate_lines,
            keep_n=12,
            llm_context_limit=4096,
            helper_max_tokens=120,
            prompt_overhead_tokens=280,
            safety_margin=120,
        )

        self.assertEqual(len(trimmed_lines), 12)
        self.assertLess(meta["line_char_cap"], len(candidate_lines[0]))
        self.assertTrue(all(len(line) <= meta["line_char_cap"] for line in trimmed_lines))


if __name__ == "__main__":
    unittest.main()
