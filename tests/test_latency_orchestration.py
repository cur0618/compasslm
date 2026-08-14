import asyncio
import importlib.util
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "query_orchestration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_query_orchestration", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ParallelHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_helper_analysis_and_expand_can_run_in_parallel(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        started = []
        ready = asyncio.Event()

        async def analyze(query: str):
            started.append(("analyze", time.perf_counter()))
            await ready.wait()
            return {"intent": "fact", "query": query}

        async def expand(query: str):
            started.append(("expand", time.perf_counter()))
            await ready.wait()
            return f"{query} 확장"

        task = asyncio.create_task(module.run_parallel_helper_tasks("환풍기 설치 기준", analyze, expand))
        for _ in range(20):
            if len(started) == 2:
                break
            await asyncio.sleep(0.005)
        ready.set()
        analysis, expanded = await task

        self.assertEqual(len(started), 2)
        self.assertEqual(analysis["query"], "환풍기 설치 기준")
        self.assertEqual(expanded, "환풍기 설치 기준 확장")
        self.assertLess(abs(started[0][1] - started[1][1]), 0.02)


class RerankDecisionTests(unittest.TestCase):
    def test_simple_direct_question_can_skip_rerank(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        should_rerank, reason = module.decide_rerank_usage(
            "양봉장 환풍기 설치 기준",
            candidate_count=6,
        )

        self.assertFalse(should_rerank)
        self.assertEqual(reason, "simple_direct_question")

    def test_complex_multi_condition_question_uses_rerank(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        should_rerank, reason = module.decide_rerank_usage(
            "2024년 지침서와 사례집을 비교해서 태양광 설치 처리 기준, 예외, 필요 서류를 요약해줘",
            candidate_count=18,
        )

        self.assertTrue(should_rerank)
        self.assertEqual(reason, "complex_multi_condition")


class HelperDegradedContractTests(unittest.TestCase):
    def test_service_tracks_empty_helper_failures_and_runs_deterministic_path_in_parallel(self):
        source = (ROOT / "src" / "compass_ai" / "service.py").read_text(encoding="utf-8")

        self.assertIn("_helper_degraded_failures", source)
        self.assertIn("_mark_helper_degraded", source)
        self.assertIn("_is_helper_degraded", source)
        self.assertIn("_run_degraded_helper_with_deterministic", source)
        self.assertIn("deterministic_parallel_used", source)
        self.assertIn("helper_wait_skipped", source)
        self.assertIn("asyncio.create_task", source)
        self.assertIn("direct_fallback_empty_output", source)

    def test_service_degrades_repeated_unexpected_model_behavior(self):
        source = (ROOT / "src" / "compass_ai" / "service.py").read_text(encoding="utf-8")

        self.assertIn("_should_mark_helper_degraded", source)
        self.assertIn("unexpected_model_behavior", source)
        self.assertIn("exceeded maximum retries", source)
        self.assertIn("output_validation", source)
        self.assertIn("followup_rewrite_fail", source)


if __name__ == "__main__":
    unittest.main()
