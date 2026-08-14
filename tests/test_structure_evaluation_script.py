import json
import tempfile
import unittest
from pathlib import Path


from scripts.evaluate_structure_rag import load_cases, load_mode_results, run_report


class StructureEvaluationScriptTests(unittest.TestCase):
    def test_load_cases_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"id": "same", "query": "q1", "source_type": "hwpx"}),
                    json.dumps({"id": "same", "query": "q2", "source_type": "xlsx"}),
                ]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cases(path)

    def test_run_report_writes_metrics_without_fabricating_missing_modes(self):
        cases = [{
            "id": "x1",
            "query": "지급단가는?",
            "source_type": "xlsx",
            "answerable": True,
            "expected_refs": ["pay.xlsx#Sheet1/table:1/row:2"],
            "accepted_answers": ["40천원"],
            "numeric_signatures": ["40천원"],
        }]
        baseline = {"x1": {
            "ranked_refs": ["other.xlsx#Sheet1/table:1/row:2"],
            "ranked_parent_keys": ["other"],
            "answer": "다른 값",
            "citation_refs": [],
            "retrieval_ms": 10,
            "total_ms": 20,
        }}
        candidate = {"x1": {
            "ranked_refs": ["pay.xlsx#Sheet1/table:1/row:2"],
            "ranked_parent_keys": ["pay-row-2"],
            "answer": "지급단가는 40천원입니다.",
            "citation_refs": ["pay.xlsx#Sheet1/table:1/row:2"],
            "retrieval_ms": 11,
            "total_ms": 22,
        }}
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report"
            summary = run_report(
                cases,
                {"baseline": baseline, "structure_v2": candidate},
                output_dir=output,
            )

            self.assertEqual(summary["modes"]["structure_v2"]["recall_at_1"], 1.0)
            self.assertIn("xlsx", summary["comparisons"])
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "summary.md").exists())

    def test_load_mode_results_accepts_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(json.dumps({"case_id": "q1", "ranked_refs": []}), encoding="utf-8")

            self.assertEqual(load_mode_results(path), {"q1": {"case_id": "q1", "ranked_refs": []}})


if __name__ == "__main__":
    unittest.main()
