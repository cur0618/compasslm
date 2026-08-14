from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from src.table_alias import generate_name_aliases, is_name_header, normalize_alias_text, should_emit_name_alias


SPACE_PATTERN = re.compile(r"\s+")
NUMERIC_ONLY_PATTERN = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
PRICE_HEADER_PATTERN = re.compile(r"(단가|금액|비용|가격|수수료|요금)")
NAME_LIKE_HEADER_PATTERN = re.compile(r"(조사명|사업명|항목명|품목|대상|구분|명칭|이름|명$)")
DEFINITION_CODE_PATTERN = re.compile(r"^(?:[①-⑳㉑-㊿]|\(?\d{1,2}\)?[.)]?|[가-힣A-Za-z]\))\s*.+")
PAGE_CHROME_EXACT = {
    "header",
    "footer",
    "image",
    "text",
    "paragraph_title",
    "statistics korea",
}
PAGE_CHROME_COMPACT = {
    "Ⅰ",
    "Ⅱ",
    "Ⅲ",
    "Ⅳ",
    "Ⅴ",
    "조사개요",
    "현장조사",
    "작성요령",
    "원부조사표",
    "내검ㆍ입력",
    "내검입력",
    "부록",
    "목차",
    "차례",
    "요",
    "표",
    "검",
    "력",
}


def normalize_text(value: str) -> str:
    return SPACE_PATTERN.sub(" ", (value or "").strip())


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _format_field_value(header: str, value: str, row_text: str = "") -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    if PRICE_HEADER_PATTERN.search(header or "") and NUMERIC_ONLY_PATTERN.match(normalized):
        if "천원" not in row_text and "만원" not in row_text and "원" not in normalized:
            return f"{normalized}천원"
    return normalized


def _choose_subject(headers: Sequence[str], values: Sequence[str]) -> Tuple[str, int]:
    for idx, header in enumerate(headers):
        if idx < len(values) and is_name_header(header) and values[idx]:
            return values[idx], idx
    for idx, value in enumerate(values):
        if value and not NUMERIC_ONLY_PATTERN.match(value.replace(",", "")):
            return value, idx
    return (values[0] if values else "", 0)


def build_table_row_fact_text(headers: Sequence[str], values: Sequence[str]) -> str:
    normalized_headers = [normalize_text(str(header or "")) for header in headers]
    normalized_values = [normalize_text(str(value or "")) for value in values]
    row_text = " ".join(normalized_values)
    pairs: List[str] = ["표의미: kind=table_row"]
    subject, subject_idx = _choose_subject(normalized_headers, normalized_values)
    aliases: List[str] = []

    if subject:
        pairs.append(f"subject={subject}")
        subject_header = normalized_headers[subject_idx] if subject_idx < len(normalized_headers) else ""
        aliases.extend(generate_name_aliases(subject) if should_emit_name_alias(subject, header=subject_header) else [subject])

    for idx, header in enumerate(normalized_headers):
        if idx >= len(normalized_values):
            continue
        value = _format_field_value(header, normalized_values[idx], row_text)
        if not header or not value:
            continue
        pairs.append(f"{header}={value}")
        if NAME_LIKE_HEADER_PATTERN.search(header) or is_name_header(header):
            aliases.extend(generate_name_aliases(value))

    aliases = _dedupe(aliases)
    if aliases:
        pairs.insert(2 if subject else 1, f"aliases={', '.join(aliases)}")
    return " | ".join(pairs)


def build_flat_table_row_fact_text(header_line: str, row_line: str) -> str:
    headers = [token for token in normalize_text(header_line).split() if token]
    values = [token for token in normalize_text(row_line).split() if token]
    if not headers or not values:
        return ""
    if len(values) > len(headers) and is_name_header(headers[0]):
        extra = len(values) - len(headers)
        values = [" ".join(values[: extra + 1]), *values[extra + 1 :]]
    return build_table_row_fact_text(headers, values)


def _is_definition_heading(line: str) -> bool:
    text = normalize_text(line)
    if not text or len(text) > 12:
        return False
    if _is_page_chrome_line(text):
        return False
    if DEFINITION_CODE_PATTERN.match(text):
        return False
    if re.search(r"[.!?。]$", text):
        return False
    return bool(re.search(r"[A-Za-z가-힣]", text))


def _is_definition_code(line: str) -> bool:
    return bool(DEFINITION_CODE_PATTERN.match(normalize_text(line)))


def _compact_definition_text(line: str) -> str:
    return re.sub(r"\s+", "", normalize_text(line))


def _is_page_chrome_line(line: str) -> bool:
    text = normalize_text(line)
    if not text:
        return True
    lowered = text.lower()
    compact = _compact_definition_text(text)
    if lowered in PAGE_CHROME_EXACT or compact in PAGE_CHROME_COMPACT:
        return True
    if lowered.startswith(("http://", "https://")) or "kostat.go.kr" in lowered:
        return True
    if re.fullmatch(r"[·ㆍ∙.\-–—_ ]{3,}", text):
        return True
    if lowered.startswith("imgs/") or re.search(r"\.(?:jpg|jpeg|png|gif|webp)\b", lowered):
        return True
    if lowered.startswith("표의미:"):
        return True
    if re.fullmatch(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", compact):
        return True
    if re.fullmatch(r"\d+\s*[∙.-]\s*통계청", text):
        return True
    return False


def _is_valid_definition_subject(subject: str) -> bool:
    text = normalize_text(subject)
    if not text or _is_page_chrome_line(text):
        return False
    compact = _compact_definition_text(text)
    if ">" not in text and len(compact) <= 1:
        return False
    if compact in PAGE_CHROME_COMPACT:
        return False
    return bool(re.search(r"[가-힣A-Za-z①-⑳㉑-㊿]", text))


def _is_valid_definition_body(body: str) -> bool:
    text = normalize_text(body)
    if len(text) < 8 or _is_page_chrome_line(text):
        return False
    if not re.search(r"[가-힣]", text):
        return False
    return True


def build_definition_facts_from_lines(lines: Sequence[str]) -> List[str]:
    facts: List[str] = []
    current_heading = ""
    current_subject = ""
    current_body: List[str] = []

    def flush() -> None:
        nonlocal current_subject, current_body
        if current_subject and current_body:
            body = normalize_text(" ".join(line for line in current_body if not _is_page_chrome_line(line)))
            if _is_valid_definition_subject(current_subject) and _is_valid_definition_body(body):
                facts.append(f"표의미: kind=definition_block | subject={current_subject} | 정의={body}")
        current_subject = ""
        current_body = []

    for raw in lines:
        line = normalize_text(raw)
        if not line or _is_page_chrome_line(line):
            continue
        if _is_definition_heading(line):
            flush()
            current_heading = line
            current_subject = line
            current_body = []
            continue
        if _is_definition_code(line):
            flush()
            current_subject = f"{current_heading} > {line}" if current_heading else line
            current_body = []
            continue
        if current_subject:
            current_body.append(line)
    flush()
    return _dedupe(facts)


def parse_table_fact_line(line: str) -> Dict[str, str]:
    text = normalize_text(line)
    if not text.startswith("표의미:"):
        return {}
    result: Dict[str, str] = {}
    for part in [piece.strip() for piece in text.split("|") if piece.strip()]:
        if part.startswith("표의미:"):
            part = part.replace("표의미:", "", 1).strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[normalize_text(key)] = normalize_text(value)
    return result


def table_fact_matches_query(fact: Dict[str, str], query: str) -> bool:
    query_key = normalize_alias_text(query)
    if not query_key:
        return False
    candidates: List[str] = []
    subject = fact.get("subject", "")
    if subject:
        candidates.append(subject)
    aliases = fact.get("aliases", "") or fact.get("명칭별칭", "")
    candidates.extend(alias.strip() for alias in aliases.split(",") if alias.strip())
    for candidate in candidates:
        key = normalize_alias_text(candidate)
        if len(key) >= 3 and key in query_key:
            return True
    return False
