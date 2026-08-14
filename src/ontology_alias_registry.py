from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_ONTOLOGY_QUERY_ALIASES: Dict[str, Sequence[str]] = {
    "답례품": ("지급단가", "단가", "금액", "답례품"),
    "얼마": ("지급단가", "단가", "금액"),
    "농어가경제조사": ("농가경제조사", "농어가경제조사"),
}


def expand_ontology_query_tokens(
    tokens: Iterable[str],
    *,
    aliases: Mapping[str, Sequence[str]] = DEFAULT_ONTOLOGY_QUERY_ALIASES,
) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(aliases.get(token, ()))
    return expanded
