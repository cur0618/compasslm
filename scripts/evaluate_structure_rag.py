from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.structure_evaluation import evaluate_structure_mode, summarize_structure_comparison


def _load_json_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"expected JSON array: {path}")
        return [dict(row) for row in payload]
    rows = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = _load_json_rows(path)
    seen = set()
    for case in cases:
        case_id = str(case.get("id", "") or "").strip()
        query = str(case.get("query", "") or "").strip()
        source_type = str(case.get("source_type", "") or "").lower()
        if not case_id or not query or source_type not in {"hwpx", "xlsx"}:
            raise ValueError("each evaluation case requires id, query, and source_type=hwpx|xlsx")
        if case_id in seen:
            raise ValueError(f"duplicate evaluation case id: {case_id}")
        seen.add(case_id)
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def load_mode_results(path: Path) -> Dict[str, Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
            return {str(key): dict(value) for key, value in payload.items()}
    rows = _load_json_rows(path)
    results: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", row.get("id", "")) or "").strip()
        if not case_id:
            raise ValueError(f"result row requires case_id: {path}")
        if case_id in results:
            raise ValueError(f"duplicate result case id: {case_id}")
        results[case_id] = row
    return results


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Structure RAG Evaluation", "", f"Cases: {summary['case_count']}", "", "## Modes", ""]
    for mode, metrics in summary.get("modes", {}).items():
        lines.extend([
            f"### {mode}",
            f"- Recall@3: {float(metrics.get('recall_at_3', 0.0)):.3f}",
            f"- MRR: {float(metrics.get('mrr', 0.0)):.3f}",
            f"- Answer accuracy: {float(metrics.get('answer_accuracy', 0.0)):.3f}",
            f"- Citation accuracy: {float(metrics.get('citation_accuracy', 0.0) or 0.0):.3f}",
            f"- False positive rate: {float(metrics.get('false_positive_rate', 0.0)):.3f}",
            f"- Parent duplicate rate: {float(metrics.get('parent_duplicate_rate', 0.0)):.3f}",
            f"- p95 total: {float(metrics.get('p95_total_ms', 0.0)):.1f} ms",
            "",
        ])
    lines.extend(["## Release Gates", ""])
    for source_type, comparison in summary.get("comparisons", {}).items():
        lines.append(f"- {source_type}: {'PASS' if comparison.get('passed') else 'FAIL'}")
    return "\n".join(lines).rstrip() + "\n"


def run_report(
    cases: Sequence[Mapping[str, Any]],
    mode_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_metrics = {mode: evaluate_structure_mode(cases, results) for mode, results in mode_results.items()}
    comparisons: Dict[str, Any] = {}
    if "baseline" in mode_results and "structure_v2" in mode_results:
        for source_type in ("hwpx", "xlsx"):
            selected = [case for case in cases if str(case.get("source_type", "")).lower() == source_type]
            if not selected:
                continue
            baseline_metrics = evaluate_structure_mode(selected, mode_results["baseline"])
            candidate_metrics = evaluate_structure_mode(selected, mode_results["structure_v2"])
            comparisons[source_type] = summarize_structure_comparison(
                baseline_metrics,
                candidate_metrics,
                source_type=source_type,
            )
    summary = {"case_count": len(cases), "modes": mode_metrics, "comparisons": comparisons}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate CompassLM HWPX/XLSX structure RAG evaluation results")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--baseline-results", required=True, type=Path)
    parser.add_argument("--structure-v2-results", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("reports/structure-rag-eval") / datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = run_report(
        load_cases(args.cases),
        {
            "baseline": load_mode_results(args.baseline_results),
            "structure_v2": load_mode_results(args.structure_v2_results),
        },
        output_dir=output_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
