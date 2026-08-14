import unittest


from src.ontology_evaluation import evaluate_mode, summarize_comparison


class OntologyEvaluationTests(unittest.TestCase):
    def test_evaluate_mode_computes_retrieval_and_safety_metrics(self):
        cases = [
            {
                "id": "q1",
                "answerable": True,
                "expected_chunk_ids": [],
                "expected_sources": ["pay.xlsx"],
                "expected_answer": "40천원",
                "category": "table_numeric",
            },
            {
                "id": "q2",
                "answerable": False,
                "expected_chunk_ids": [],
                "expected_answer": "",
                "category": "unanswerable",
            },
        ]
        results = {
            "q1": {
                "ranked_chunk_ids": [11, 12],
                "ranked_sources": ["/kb/pay.xlsx", "/kb/other.txt"],
                "answer": "지급단가는 40천원입니다.",
                "ontology_used": True,
                "ontology_hop_count": 1,
                "retrieval_ms": 10.0,
                "total_ms": 30.0,
            },
            "q2": {
                "ranked_chunk_ids": [99],
                "ranked_sources": ["unrelated.txt"],
                "answer": "관련 없어 보이는 검색 후보",
                "ontology_used": False,
                "ontology_hop_count": 0,
                "retrieval_ms": 20.0,
                "total_ms": 50.0,
            },
        }

        metrics = evaluate_mode(cases, results, top_ks=(1, 3, 5))

        self.assertEqual(metrics["case_count"], 2)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["ontology_candidate_usage_rate"], 0.5)
        self.assertEqual(metrics["one_hop_count"], 1)
        self.assertEqual(metrics["p95_total_ms"], 50.0)

    def test_summarize_comparison_applies_plan_acceptance_gates(self):
        baseline = {
            "recall_at_3": 0.60,
            "answer_accuracy": 0.60,
            "numeric_accuracy": 0.60,
            "recovered_failure_rate": 0.0,
            "false_positive_rate": 0.01,
            "p95_total_ms": 100.0,
        }
        candidate = {
            "recall_at_3": 0.72,
            "answer_accuracy": 0.72,
            "numeric_accuracy": 0.68,
            "recovered_failure_rate": 0.35,
            "false_positive_rate": 0.02,
            "p95_total_ms": 112.0,
        }

        comparison = summarize_comparison(baseline, candidate)

        self.assertTrue(comparison["gates"]["retrieval_improvement"])
        self.assertTrue(comparison["gates"]["numeric_accuracy"])
        self.assertTrue(comparison["gates"]["false_positive_rate"])
        self.assertTrue(comparison["gates"]["latency"])
        self.assertTrue(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
