import unittest


from src.structure_evaluation import evaluate_structure_mode, summarize_structure_comparison


class StructureEvaluationTests(unittest.TestCase):
    def test_evaluate_structure_mode_computes_grounding_and_parent_metrics(self):
        cases = [
            {
                "id": "h1",
                "source_type": "hwpx",
                "answerable": True,
                "expected_refs": ["rules.hwpx#제3조"],
                "accepted_answers": ["7일 이내"],
                "numeric_signatures": ["7일"],
            },
            {
                "id": "x1",
                "source_type": "xlsx",
                "answerable": False,
                "expected_refs": [],
            },
        ]
        results = {
            "h1": {
                "ranked_refs": ["rules.hwpx#제3조", "rules.hwpx#제4조"],
                "ranked_parent_keys": ["p3", "p3"],
                "answer": "신청은 7일 이내에 해야 합니다.",
                "citation_refs": ["rules.hwpx#제3조"],
                "retrieval_ms": 12.0,
                "total_ms": 30.0,
            },
            "x1": {
                "ranked_refs": [],
                "ranked_parent_keys": [],
                "answer": "문서에서 확인할 수 없습니다.",
                "citation_refs": [],
                "retrieval_ms": 18.0,
                "total_ms": 40.0,
            },
        }

        metrics = evaluate_structure_mode(cases, results)

        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertEqual(metrics["numeric_accuracy"], 1.0)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["parent_duplicate_rate"], 0.5)
        self.assertEqual(metrics["p95_total_ms"], 40.0)

    def test_comparison_uses_format_specific_release_gates(self):
        baseline = {
            "recall_at_3": 0.70,
            "answer_accuracy": 0.70,
            "numeric_accuracy": 0.70,
            "citation_accuracy": 0.97,
            "false_positive_rate": 0.02,
            "p95_total_ms": 100.0,
        }
        candidate = {
            "recall_at_3": 0.82,
            "answer_accuracy": 0.82,
            "numeric_accuracy": 0.82,
            "citation_accuracy": 0.99,
            "false_positive_rate": 0.01,
            "p95_total_ms": 112.0,
        }

        comparison = summarize_structure_comparison(baseline, candidate, source_type="xlsx")

        self.assertTrue(comparison["gates"]["quality_improvement"])
        self.assertTrue(comparison["gates"]["numeric_accuracy"])
        self.assertTrue(comparison["gates"]["citation_accuracy"])
        self.assertTrue(comparison["gates"]["false_positive_rate"])
        self.assertTrue(comparison["gates"]["latency"])
        self.assertTrue(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
