from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Mapping, Sequence


ABSTENTION_MARKERS = (
    "확인할 수 없습니다",
    "근거가 부족",
    "문서에 없습니다",
    "알 수 없습니다",
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _is_abstention(answer: str) -> bool:
    compact = _compact(answer)
    return any(_compact(marker) in compact for marker in ABSTENTION_MARKERS)


def _answer_is_correct(case: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    answerable = bool(case.get("answerable", True))
    answer = str(result.get("answer", "") or "")
    if not answerable:
        return not result.get("ranked_chunk_ids") and _is_abstention(answer)
    accepted = case.get("accepted_answers") or [case.get("expected_answer", "")]
    expected = [_compact(value) for value in accepted if _compact(value)]
    compact_answer = _compact(answer)
    return bool(expected) and any(value in compact_answer for value in expected)


def evaluate_mode(
    cases: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    *,
    top_ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    answerable_cases = [case for case in cases if bool(case.get("answerable", True))]
    reciprocal_ranks = []
    recall_hits = {int(k): 0 for k in top_ks}
    correct_answers = 0
    numeric_total = 0
    numeric_correct = 0
    unanswerable_total = 0
    false_positives = 0
    abstention_evaluated = 0
    correct_abstentions = 0
    ontology_used = 0
    one_hop_count = 0
    two_hop_count = 0
    retrieval_times = []
    total_times = []

    for case in cases:
        case_id = str(case.get("id", "") or "")
        result = results.get(case_id, {})
        ranked = [int(value) for value in result.get("ranked_chunk_ids", [])]
        expected = {int(value) for value in case.get("expected_chunk_ids", [])}
        ranked_sources = [_compact(value) for value in result.get("ranked_sources", [])]
        expected_sources = [_compact(value) for value in case.get("expected_sources", []) if _compact(value)]
        if bool(case.get("answerable", True)):
            rank = next((index for index, chunk_id in enumerate(ranked, 1) if chunk_id in expected), 0)
            if not rank and expected_sources:
                rank = next(
                    (
                        index
                        for index, source in enumerate(ranked_sources, 1)
                        if any(expected_source in source for expected_source in expected_sources)
                    ),
                    0,
                )
            reciprocal_ranks.append((1.0 / rank) if rank else 0.0)
            for k in recall_hits:
                if rank and rank <= k:
                    recall_hits[k] += 1
            correct = _answer_is_correct(case, result)
            correct_answers += int(correct)
            if str(case.get("category", "") or "") in {"table_numeric", "numeric"}:
                numeric_total += 1
                numeric_correct += int(correct)
        else:
            unanswerable_total += 1
            if bool(result.get("ontology_used", False)):
                false_positives += 1
            if "abstained" in result:
                abstention_evaluated += 1
                correct_abstentions += int(bool(result.get("abstained")))
        if bool(result.get("ontology_used", False)):
            ontology_used += 1
        hop_count = int(result.get("ontology_hop_count", 0) or 0)
        one_hop_count += int(hop_count == 1)
        two_hop_count += int(hop_count == 2)
        retrieval_times.append(float(result.get("retrieval_ms", 0.0) or 0.0))
        total_times.append(float(result.get("total_ms", 0.0) or 0.0))

    answerable_count = max(1, len(answerable_cases))
    case_count = max(1, len(cases))
    metrics: Dict[str, Any] = {
        "case_count": len(cases),
        "answerable_count": len(answerable_cases),
        "answer_accuracy": correct_answers / answerable_count,
        "numeric_accuracy": (numeric_correct / numeric_total) if numeric_total else 0.0,
        "false_positive_rate": (false_positives / unanswerable_total) if unanswerable_total else 0.0,
        "abstention_evaluated_count": abstention_evaluated,
        "abstention_accuracy": (correct_abstentions / abstention_evaluated) if abstention_evaluated else None,
        "ontology_candidate_usage_rate": ontology_used / case_count,
        "one_hop_count": one_hop_count,
        "two_hop_count": two_hop_count,
        "mrr": sum(reciprocal_ranks) / answerable_count,
        "p50_retrieval_ms": _percentile(retrieval_times, 0.50),
        "p95_retrieval_ms": _percentile(retrieval_times, 0.95),
        "p50_total_ms": _percentile(total_times, 0.50),
        "p95_total_ms": _percentile(total_times, 0.95),
    }
    for k, count in recall_hits.items():
        metrics[f"recall_at_{k}"] = count / answerable_count
    return metrics


def summarize_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    recall_gain = float(candidate.get("recall_at_3", 0.0)) - float(baseline.get("recall_at_3", 0.0))
    recovered = float(candidate.get("recovered_failure_rate", 0.0))
    numeric_gain = float(candidate.get("numeric_accuracy", 0.0)) - float(baseline.get("numeric_accuracy", 0.0))
    baseline_latency = float(baseline.get("p95_total_ms", 0.0) or 0.0)
    candidate_latency = float(candidate.get("p95_total_ms", 0.0) or 0.0)
    latency_ratio = (candidate_latency / baseline_latency) if baseline_latency > 0 else 1.0
    gates = {
        "retrieval_improvement": recall_gain >= 0.10 or recovered >= 0.30,
        "numeric_accuracy": numeric_gain >= 0.05,
        "false_positive_rate": float(candidate.get("false_positive_rate", 0.0)) <= 0.02,
        "latency": latency_ratio <= 1.15,
    }
    return {
        "recall_at_3_gain": recall_gain,
        "numeric_accuracy_gain": numeric_gain,
        "latency_ratio": latency_ratio,
        "gates": gates,
        "passed": all(gates.values()),
    }
