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


def _matches_reference(actual: Any, expected: Any) -> bool:
    actual_value = _compact(actual)
    expected_value = _compact(expected)
    return bool(actual_value and expected_value) and (
        actual_value == expected_value
        or expected_value in actual_value
        or actual_value in expected_value
    )


def _is_abstention(answer: Any) -> bool:
    compact = _compact(answer)
    return any(_compact(marker) in compact for marker in ABSTENTION_MARKERS)


def _answer_is_correct(case: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    if not bool(case.get("answerable", True)):
        return not result.get("ranked_refs") and _is_abstention(result.get("answer", ""))
    accepted = case.get("accepted_answers") or [case.get("expected_answer", "")]
    answer = _compact(result.get("answer", ""))
    expected = [_compact(value) for value in accepted if _compact(value)]
    return bool(expected) and any(value in answer for value in expected)


def evaluate_structure_mode(
    cases: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    *,
    top_ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    answerable = [case for case in cases if bool(case.get("answerable", True))]
    recall_hits = {int(k): 0 for k in top_ks}
    reciprocal_ranks = []
    answer_correct = 0
    numeric_total = 0
    numeric_correct = 0
    citation_total = 0
    citation_correct = 0
    unanswerable_total = 0
    false_positives = 0
    duplicate_parent_entries = 0
    parent_entries = 0
    retrieval_times = []
    total_times = []

    for case in cases:
        case_id = str(case.get("id", "") or "")
        result = results.get(case_id, {})
        ranked_refs = list(result.get("ranked_refs", []) or [])
        expected_refs = list(case.get("expected_refs", []) or [])
        if bool(case.get("answerable", True)):
            rank = next(
                (
                    index
                    for index, actual in enumerate(ranked_refs, 1)
                    if any(_matches_reference(actual, expected) for expected in expected_refs)
                ),
                0,
            )
            reciprocal_ranks.append((1.0 / rank) if rank else 0.0)
            for k in recall_hits:
                recall_hits[k] += int(bool(rank and rank <= k))
            correct = _answer_is_correct(case, result)
            answer_correct += int(correct)
            signatures = [_compact(value) for value in case.get("numeric_signatures", []) if _compact(value)]
            if signatures:
                numeric_total += 1
                answer = _compact(result.get("answer", ""))
                numeric_correct += int(all(signature in answer for signature in signatures))
            citation_total += 1
            citations = list(result.get("citation_refs", []) or [])
            citation_correct += int(
                bool(expected_refs)
                and bool(citations)
                and all(any(_matches_reference(citation, expected) for expected in expected_refs) for citation in citations)
                and all(any(_matches_reference(citation, expected) for citation in citations) for expected in expected_refs)
            )
        else:
            unanswerable_total += 1
            false_positives += int(bool(ranked_refs) or not _is_abstention(result.get("answer", "")))

        parent_keys = [str(value) for value in result.get("ranked_parent_keys", []) if str(value)]
        parent_entries += len(parent_keys)
        duplicate_parent_entries += len(parent_keys) - len(set(parent_keys))
        retrieval_times.append(float(result.get("retrieval_ms", 0.0) or 0.0))
        total_times.append(float(result.get("total_ms", 0.0) or 0.0))

    answerable_count = max(1, len(answerable))
    metrics: Dict[str, Any] = {
        "case_count": len(cases),
        "answerable_count": len(answerable),
        "answer_accuracy": answer_correct / answerable_count,
        "numeric_accuracy": (numeric_correct / numeric_total) if numeric_total else None,
        "citation_accuracy": (citation_correct / citation_total) if citation_total else None,
        "false_positive_rate": (false_positives / unanswerable_total) if unanswerable_total else 0.0,
        "parent_duplicate_rate": (duplicate_parent_entries / parent_entries) if parent_entries else 0.0,
        "mrr": sum(reciprocal_ranks) / answerable_count,
        "p50_retrieval_ms": _percentile(retrieval_times, 0.50),
        "p95_retrieval_ms": _percentile(retrieval_times, 0.95),
        "p50_total_ms": _percentile(total_times, 0.50),
        "p95_total_ms": _percentile(total_times, 0.95),
    }
    for k, count in recall_hits.items():
        metrics[f"recall_at_{k}"] = count / answerable_count
    return metrics


def summarize_structure_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    source_type: str,
) -> Dict[str, Any]:
    source_type = str(source_type or "").lower()
    answer_gain = float(candidate.get("answer_accuracy", 0.0) or 0.0) - float(baseline.get("answer_accuracy", 0.0) or 0.0)
    recall_gain = float(candidate.get("recall_at_3", 0.0) or 0.0) - float(baseline.get("recall_at_3", 0.0) or 0.0)
    baseline_numeric = baseline.get("numeric_accuracy")
    candidate_numeric = candidate.get("numeric_accuracy")
    numeric_gain = (
        float(candidate_numeric) - float(baseline_numeric)
        if baseline_numeric is not None and candidate_numeric is not None
        else None
    )
    baseline_latency = float(baseline.get("p95_total_ms", 0.0) or 0.0)
    candidate_latency = float(candidate.get("p95_total_ms", 0.0) or 0.0)
    latency_ratio = candidate_latency / baseline_latency if baseline_latency > 0 else 1.0
    required_gain = 0.10 if source_type == "xlsx" else 0.05
    quality_gain = max(answer_gain, recall_gain)
    gates = {
        "quality_improvement": quality_gain >= required_gain,
        "numeric_accuracy": numeric_gain is None or numeric_gain >= required_gain,
        "citation_accuracy": float(candidate.get("citation_accuracy", 0.0) or 0.0) >= 0.98,
        "false_positive_rate": float(candidate.get("false_positive_rate", 0.0) or 0.0) <= 0.02,
        "latency": latency_ratio <= 1.15,
    }
    return {
        "source_type": source_type,
        "answer_accuracy_gain": answer_gain,
        "recall_at_3_gain": recall_gain,
        "numeric_accuracy_gain": numeric_gain,
        "latency_ratio": latency_ratio,
        "gates": gates,
        "passed": all(gates.values()),
    }
