from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ontology_evaluation import evaluate_mode, summarize_comparison


SUPPORTED_MODES = ("off", "deterministic", "llm")


def parse_float_list(raw: str) -> List[float]:
    values = [float(value.strip()) for value in str(raw or "").split(",") if value.strip()]
    if not values:
        raise ValueError("at least one tuning value is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("tuning values must be between 0 and 1")
    return values


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        case = json.loads(line)
        case_id = str(case.get("id", "") or "").strip()
        query = str(case.get("query", "") or "").strip()
        if not case_id or not query:
            raise ValueError(f"evaluation case at line {line_number} requires id and query")
        if case_id in seen:
            raise ValueError(f"duplicate evaluation case id: {case_id}")
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def _configure_mode(engine: Any, mode: str) -> None:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported ontology evaluation mode: {mode}")
    engine.ontology_rag_enabled = mode != "off"
    engine.ontology_experiment_mode = mode
    engine.ontology_allowed_extraction_methods = (
        {
            "deterministic_table_fact",
            "deterministic_definition_path",
            "deterministic_structure_fact",
        }
        if mode == "deterministic"
        else None
    )
    query_cache = getattr(engine, "query_cache", None)
    if query_cache is not None and hasattr(query_cache, "clear"):
        query_cache.clear()


def _run_case(engine: Any, case: Mapping[str, Any], top_k: int) -> Dict[str, Any]:
    started = time.perf_counter()
    results = engine.search(str(case.get("query", "") or ""), top_k=top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000.0
    top_texts = [str(row.get("text", "") or "") for row in results[:3]]
    ontology_rows = [row for row in results if float(row.get("ontology_fact_score", 0.0) or 0.0) > 0]
    return {
        "ranked_chunk_ids": [int(row.get("id", 0) or 0) for row in results],
        "ranked_sources": [str(row.get("source_path", row.get("source_display", "")) or "") for row in results],
        "answer": "\n".join(top_texts),
        "ontology_used": bool(ontology_rows),
        "ontology_hop_count": min(
            [int(row.get("ontology_hop_count", 0) or 0) for row in ontology_rows] or [0]
        ),
        "retrieval_ms": retrieval_ms,
        "total_ms": retrieval_ms,
    }


def _comparison_with_recovery(
    cases: Sequence[Mapping[str, Any]],
    baseline_results: Mapping[str, Mapping[str, Any]],
    candidate_results: Mapping[str, Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    failed = []
    recovered = 0
    for case in cases:
        if not bool(case.get("answerable", True)):
            continue
        case_id = str(case.get("id", "") or "")
        single = [case]
        baseline_correct = evaluate_mode(single, {case_id: baseline_results.get(case_id, {})})["answer_accuracy"] == 1.0
        candidate_correct = evaluate_mode(single, {case_id: candidate_results.get(case_id, {})})["answer_accuracy"] == 1.0
        if not baseline_correct:
            failed.append(case_id)
            recovered += int(candidate_correct)
    enriched = dict(candidate_metrics)
    enriched["recovered_failure_rate"] = (recovered / len(failed)) if failed else 0.0
    comparison = summarize_comparison(baseline_metrics, enriched)
    comparison["baseline_failure_count"] = len(failed)
    comparison["recovered_failure_count"] = recovered
    comparison["recovered_failure_rate"] = enriched["recovered_failure_rate"]
    return comparison


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Ontology RAG Evaluation", "", f"Cases: {summary['case_count']}", "", "## Modes", ""]
    for mode, metrics in summary.get("modes", {}).items():
        lines.extend([
            f"### {mode}",
            f"- Recall@3: {float(metrics.get('recall_at_3', 0.0)):.3f}",
            f"- MRR: {float(metrics.get('mrr', 0.0)):.3f}",
            f"- Answer accuracy: {float(metrics.get('answer_accuracy', 0.0)):.3f}",
            f"- Numeric accuracy: {float(metrics.get('numeric_accuracy', 0.0)):.3f}",
            f"- False positive rate: {float(metrics.get('false_positive_rate', 0.0)):.3f}",
            f"- p95 retrieval: {float(metrics.get('p95_retrieval_ms', 0.0)):.1f} ms",
            "",
        ])
    lines.extend(["## Comparisons", ""])
    for name, comparison in summary.get("comparisons", {}).items():
        lines.append(f"- {name}: {'PASS' if comparison.get('passed') else 'FAIL'}")
    return "\n".join(lines).rstrip() + "\n"


def run_evaluation(
    cases: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
    output_dir: Path,
    engine: Any,
    top_k: int = 5,
    score_weight: float = 0.26,
    min_confidence: float = 0.62,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.ontology_score_weight = float(score_weight)
    engine.ontology_min_fact_confidence = float(min_confidence)
    mode_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    mode_metrics: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        _configure_mode(engine, mode)
        results = {
            str(case["id"]): _run_case(engine, case, top_k)
            for case in cases
        }
        mode_results[mode] = results
        mode_metrics[mode] = evaluate_mode(cases, results)
        (output_dir / f"{mode}-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    comparisons: Dict[str, Any] = {}
    if "off" in mode_metrics:
        for mode in modes:
            if mode == "off":
                continue
            comparisons[f"{mode}_vs_off"] = _comparison_with_recovery(
                cases,
                mode_results["off"],
                mode_results[mode],
                mode_metrics["off"],
                mode_metrics[mode],
            )
    summary = {
        "case_count": len(cases),
        "configuration": {
            "ontology_score_weight": float(score_weight),
            "ontology_min_fact_confidence": float(min_confidence),
        },
        "modes": mode_metrics,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _parse_modes(raw: str) -> List[str]:
    modes = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if invalid:
        raise ValueError(f"unsupported modes: {', '.join(invalid)}")
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CompassLM retrieval with ontology modes")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--modes", default="off,deterministic,llm")
    parser.add_argument("--data-dir", default="data/kb/default")
    parser.add_argument("--kb-id", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--score-weights", default="0.10,0.18,0.26,0.34")
    parser.add_argument("--confidence-thresholds", default="0.55,0.62,0.70")
    args = parser.parse_args()

    from src.rag import RAGEngine

    output_dir = args.output_dir or Path("reports/ontology-eval") / datetime.now().strftime("%Y%m%d_%H%M%S")
    engine = RAGEngine(kb_id=args.kb_id, data_dir=args.data_dir)
    cases = load_dataset(args.dataset)
    modes = _parse_modes(args.modes)
    matrix: Dict[str, Any] = {}
    for score_weight in parse_float_list(args.score_weights):
        for min_confidence in parse_float_list(args.confidence_thresholds):
            key = f"weight_{score_weight:.2f}__confidence_{min_confidence:.2f}"
            matrix[key] = run_evaluation(
                cases,
                modes=modes,
                output_dir=output_dir / key,
                engine=engine,
                top_k=max(1, args.top_k),
                score_weight=score_weight,
                min_confidence=min_confidence,
            )
    matrix_summary = {"output_dir": str(output_dir), "case_count": len(cases), "matrix": matrix}
    (output_dir / "matrix-summary.json").write_text(
        json.dumps(matrix_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(matrix_summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
