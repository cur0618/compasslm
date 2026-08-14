from __future__ import annotations

from typing import Any, Dict, Optional


def score_ontology_candidate(
    *,
    direct_hits: int,
    predicate_hits: int,
    value_hits: int,
    relation_hits: int,
    confidence: float,
    max_hops: int,
) -> Optional[Dict[str, Any]]:
    if direct_hits > 0:
        hop_count = 1
        reason = "direct_match"
    elif max_hops >= 2 and relation_hits > 0 and predicate_hits > 0:
        hop_count = 2
        reason = "two_hop_entity_link"
    else:
        return None
    hits = direct_hits + predicate_hits + value_hits + relation_hits
    if hits <= 0:
        return None
    score = min(1.0, (0.12 * hits) + (0.45 * float(confidence)))
    if hop_count == 2:
        score *= 0.72
    return {
        "score": float(score),
        "hop_count": hop_count,
        "reason": reason,
    }
