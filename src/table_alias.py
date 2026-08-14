import re
from typing import Iterable, List, Set


SPACE_PATTERN = re.compile(r"\s+")
PARENTHETICAL_PATTERN = re.compile(r"^(.+?)\((.+?)\)$")
CONNECTOR_PATTERN = re.compile(r"[·ㆍ/]")
NAME_HINT_PATTERN = re.compile(r"(조사|사업|항목|품목|대상|구분|명)")
NAME_HEADER_PATTERN = re.compile(r"(조사명|사업명|항목명|품목|대상|구분|명칭|이름|명$)")
STRIP_FOR_MATCH_PATTERN = re.compile(r"[\s·ㆍ/\\,()\[\]{}<>「」『』'\"`_\-]+")
QUERY_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{3,}")
COMPOUND_PAIR_PREFIXES = ("농어", "남여", "내외")


def normalize_alias_text(value: str) -> str:
    return STRIP_FOR_MATCH_PATTERN.sub("", (value or "").lower())


def _normalize_display(value: str) -> str:
    return SPACE_PATTERN.sub(" ", (value or "").strip())


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_display(item)
        match_key = normalize_alias_text(normalized)
        key = normalized.lower()
        if not normalized or len(match_key) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _split_connector_variants(value: str) -> List[str]:
    normalized = _normalize_display(value)
    parts = [part for part in CONNECTOR_PATTERN.split(normalized) if part]
    if len(parts) < 2:
        return []

    variants: List[str] = [CONNECTOR_PATTERN.sub("", normalized)]
    tail = parts[-1]
    if len(tail) >= 2:
        suffix = tail[1:]
        for prefix in parts[:-1]:
            if len(prefix) <= 2 and suffix:
                variants.append(prefix + suffix)
        variants.append(tail)
    return variants


def _split_compound_pair_variants(value: str) -> List[str]:
    normalized = _normalize_display(value)
    compact = normalize_alias_text(normalized)
    if len(compact) < 5:
        return []
    if not any(compact.startswith(prefix) for prefix in COMPOUND_PAIR_PREFIXES):
        return []
    suffix = compact[2:]
    return [compact[0] + suffix, compact[1] + suffix]


def generate_name_aliases(value: str) -> List[str]:
    normalized = _normalize_display(value)
    if not normalized:
        return []

    candidates: List[str] = [normalized]
    paren_match = PARENTHETICAL_PATTERN.match(normalized)
    if paren_match:
        outer = _normalize_display(paren_match.group(1))
        inner = _normalize_display(paren_match.group(2))
        candidates.extend([outer, inner])

    candidates.extend(_split_connector_variants(normalized))
    candidates.extend(_split_compound_pair_variants(normalized))
    return _dedupe(candidates)


def is_name_header(header: str) -> bool:
    return bool(NAME_HEADER_PATTERN.search(_normalize_display(header)))


def should_emit_name_alias(value: str, *, header: str = "") -> bool:
    aliases = generate_name_aliases(value)
    if len(aliases) <= 1:
        return False
    if is_name_header(header):
        return True
    return bool(NAME_HINT_PATTERN.search(_normalize_display(value)))


def alias_match_boost(query: str, text: str) -> float:
    query_key = normalize_alias_text(query)
    if len(query_key) < 2:
        return 0.0
    if not any(marker in (text or "") for marker in ("표행요약:", "표값:", "표의미:")):
        return 0.0

    text_key = normalize_alias_text(text)
    if query_key and query_key in text_key:
        return 0.06
    for token in QUERY_TOKEN_PATTERN.findall(query or ""):
        normalized_term = normalize_alias_text(token)
        if len(normalized_term) >= 3 and normalized_term in text_key:
            return 0.06

    query_alias_keys = {normalize_alias_text(alias) for alias in generate_name_aliases(query)}
    text_alias_keys = {normalize_alias_text(alias) for alias in generate_name_aliases(text)}
    query_alias_keys.discard("")
    text_alias_keys.discard("")
    if query_alias_keys & text_alias_keys:
        return 0.04
    return 0.0
