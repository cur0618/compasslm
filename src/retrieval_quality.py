from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
CRITICAL_TERM_STOPWORDS = {
    "가구",
    "경우",
    "관련",
    "기준",
    "내용",
    "내용을",
    "농가",
    "대한",
    "대상",
    "되어",
    "되나요",
    "문서",
    "문의",
    "방법",
    "방법은",
    "설명",
    "소유",
    "요약",
    "자가",
    "자가소유",
    "정리",
    "정리해줘",
    "중요",
    "중요한",
    "조사",
    "조사해야해",
    "조사해야",
    "조사방법",
    "질문",
    "처리",
    "처리방법",
    "처리방법은",
    "해야",
    "해야해",
    "해줘",
    "해주세요",
    "했는데",
    "어떻게",
    "알려줘",
}


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "") if len(token) >= 2]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _normalize_query_term(token: str) -> str:
    term = (token or "").strip().lower()
    for suffix in ("으로는", "에서는", "에게는", "부터는", "까지는", "이라는", "은", "는", "이", "가", "을", "를", "에", "의", "도", "만"):
        if len(term) > len(suffix) + 1 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def extract_critical_terms(query: str, *, limit: int = 6) -> List[str]:
    tokens = _tokens(query)
    terms: List[str] = []
    seen: Set[str] = set()
    for raw_token in tokens:
        token = _normalize_query_term(raw_token)
        if token in CRITICAL_TERM_STOPWORDS:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max(1, int(limit or 6)):
            break
    return terms


def _result_text_for_gate(result: Dict[str, Any]) -> str:
    parts = [
        str(result.get("section", "") or ""),
        str(result.get("source_ref", "") or ""),
        str(result.get("source_display", "") or ""),
        str(result.get("source_path", "") or ""),
        str(result.get("text", "") or result.get("snippet", "") or ""),
    ]
    return "\n".join(part for part in parts if part)


def _result_has_term(result: Dict[str, Any], term: str) -> bool:
    compact_term = _compact(term)
    if not compact_term:
        return False
    return compact_term in _compact(_result_text_for_gate(result))


def _result_is_normalized(result: Dict[str, Any]) -> bool:
    return int(result.get("is_normalized", 0) or 0) == 1


def _literal_title_hit(result: Dict[str, Any], critical_terms: List[str]) -> bool:
    titleish = "\n".join(
        [
            str(result.get("section", "") or ""),
            str(result.get("source_ref", "") or ""),
            str(result.get("text", "") or "").splitlines()[0] if str(result.get("text", "") or "").strip() else "",
        ]
    )
    if not titleish.strip():
        return False
    compact_title = _compact(titleish)
    has_title_marker = bool(re.search(r"(?:^|\s)q\s*\d+|문\s*\d+|처리방법|질문", titleish, re.IGNORECASE))
    return has_title_marker and any(_compact(term) in compact_title for term in critical_terms)


def _related_term_suggestions(results: List[Dict[str, Any]], critical_terms: List[str], *, limit: int = 5) -> List[str]:
    critical_set = set(critical_terms)
    counts: Dict[str, int] = {}
    for result in results[:8]:
        for token in _tokens(_result_text_for_gate(result)):
            if token in CRITICAL_TERM_STOPWORDS or token in critical_set:
                continue
            if len(token) < 2 or re.fullmatch(r"\d+", token):
                continue
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    return [term for term, _ in ordered[: max(0, int(limit or 5))]]


def apply_critical_term_gate(
    query: str,
    results: List[Dict[str, Any]],
    *,
    enabled: bool = True,
    require_raw_backing_for_normalized: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    critical_terms = extract_critical_terms(query)
    meta: Dict[str, Any] = {
        "critical_terms": critical_terms,
        "critical_term_hits": {},
        "critical_term_gate_passed": True,
        "literal_title_hit_count": 0,
        "normalized_only_blocked": False,
        "related_terms_suggestions": [],
    }
    if not enabled or not results or not critical_terms:
        return list(results), meta

    hit_rows: List[Dict[str, Any]] = []
    critical_hits: Dict[str, List[int]] = {term: [] for term in critical_terms}
    literal_title_hit_count = 0
    raw_hit_exists = False
    for result in results:
        row_hits = [term for term in critical_terms if _result_has_term(result, term)]
        if not row_hits:
            continue
        row_id = int(result.get("id", 0) or 0)
        for term in row_hits:
            critical_hits.setdefault(term, []).append(row_id)
        if _literal_title_hit(result, critical_terms):
            literal_title_hit_count += 1
        if not _result_is_normalized(result):
            raw_hit_exists = True
        hit_rows.append(result)

    filtered: List[Dict[str, Any]] = []
    normalized_only_blocked = False
    for result in hit_rows:
        if _result_is_normalized(result) and require_raw_backing_for_normalized and not raw_hit_exists:
            normalized_only_blocked = True
            continue
        filtered.append(result)

    gate_passed = bool(filtered)
    meta.update(
        {
            "critical_term_hits": {term: ids for term, ids in critical_hits.items() if ids},
            "critical_term_gate_passed": gate_passed,
            "literal_title_hit_count": int(literal_title_hit_count),
            "normalized_only_blocked": bool(normalized_only_blocked),
            "related_terms_suggestions": [] if gate_passed else _related_term_suggestions(results, critical_terms),
        }
    )
    return filtered, meta


def _anchor_score(query: str, text: str) -> float:
    compact_query = _compact(query)
    compact_text = _compact(text)
    score = 0.0

    if "조사제외" in compact_query:
        if "조사제외" in compact_text or "제외가구" in compact_text or "조사대상이아닌" in compact_text:
            score += 0.55
        if "조사대상" in compact_text and ("제외가구" in compact_text or "제외기준" in compact_text):
            score += 0.25
    if "조사대상" in compact_query:
        if "조사대상" in compact_text:
            score += 0.35
        if "조사대상가구" in compact_text:
            score += 0.25
    if "처리" in compact_query or "어떻게조사" in compact_query:
        if "조사" in compact_text and ("처리" in compact_text or "기입" in compact_text or "조사" in compact_text):
            score += 0.20
    if "기준" in compact_query and "기준" in compact_text:
        score += 0.15
    return score


def _fact_quality_score(text: str) -> float:
    compact_text = _compact(text)
    if "표의미:kind=table_row" in compact_text:
        return 0.18
    if "표의미:kind=definition_block" not in compact_text:
        return 0.0
    noisy_subjects = ("subject=Ⅰ", "subject=Ⅱ", "subject=Ⅲ", "subject=Ⅳ", "subject=Ⅴ", "subject=header", "subject=image")
    if any(_compact(value) in compact_text for value in noisy_subjects):
        return -0.35
    return 0.08


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokens(text))
    overlap = len(query_tokens & text_tokens)
    return min(0.35, overlap / max(1, len(query_tokens)) * 0.35)


def grounded_answer_rank_score(query: str, result: Dict[str, Any]) -> float:
    text = str(result.get("text", "") or result.get("snippet", "") or "")
    base_score = float(result.get("score", 0.0) or 0.0) * 0.05
    score = base_score
    score += _lexical_score(query, text)
    score += _anchor_score(query, text)
    score += _fact_quality_score(text)
    if bool(result.get("weak_ocr_hint", False)):
        score -= 0.20
    if int(result.get("is_normalized", 0) or 0):
        score -= 0.04
    return score


def rerank_results_for_grounded_answer(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    indexed = list(enumerate(results))
    indexed.sort(
        key=lambda item: (
            grounded_answer_rank_score(query, item[1]),
            float(item[1].get("score", 0.0) or 0.0),
            -item[0],
        ),
        reverse=True,
    )
    return [item[1] for item in indexed]


def _record_raw_row(record: Any) -> Dict[str, Any]:
    metadata = getattr(record, "metadata", {}) or {}
    if isinstance(record, dict):
        metadata = record.get("metadata", {}) or {}
    raw = metadata.get("raw_row", {}) if isinstance(metadata, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _record_value(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_role(record: Any) -> str:
    raw = _record_raw_row(record)
    for key in ("doc_role", "role", "source_role"):
        value = str(raw.get(key, "") or "").strip().lower()
        if value:
            return value
    text = " ".join(
        [
            str(_record_value(record, "source_path", "") or ""),
            str(_record_value(record, "source_ref", "") or ""),
        ]
    ).lower()
    if "사례집" in text or "casebook" in text or "q&a" in text:
        return "casebook"
    if "지침서" in text or "guide" in text:
        return "guide"
    return ""


def _record_page_key(record: Any) -> Tuple[str, str]:
    raw = _record_raw_row(record)
    source_path = str(_record_value(record, "source_path", "") or raw.get("source_path", "") or "").strip()
    source_ref = str(_record_value(record, "source_ref", "") or "").strip()
    page = str(raw.get("page_no", "") or raw.get("page", "") or "").strip()
    if not page:
        match = re.search(r"PDF page\s+(\d+)", source_ref, re.IGNORECASE)
        if match:
            page = match.group(1)
    if not page:
        page = "|".join(
            [
                str(_record_value(record, "section", "") or raw.get("section", "") or "").strip(),
                str(_record_value(record, "sheet", "") or raw.get("sheet", "") or "").strip(),
                str(_record_value(record, "line_start", "") or raw.get("line_start", "") or "").strip(),
                str(_record_value(record, "row", "") or raw.get("row", "") or "").strip(),
            ]
        )
    return (source_path or source_ref, page)


def _procedure_case_query(query: str) -> bool:
    compact = _compact(query)
    return any(token in compact for token in ("처리방법", "어떻게처리", "경우", "사례", "처리"))


def _auto_prefetch_rank(query: str, record: Any) -> Tuple[float, ...]:
    raw = _record_raw_row(record)
    weak = int(bool(raw.get("weak_ocr_hint", False)))
    text = str(_record_value(record, "text", "") or "")
    role_bonus = 0
    if _procedure_case_query(query) and _record_role(record) == "casebook":
        role_bonus = 1
    fact_bonus = 1 if "표의미: kind=table_row" in text else 0
    numeric_boost = float(raw.get("numeric_table_boost", 0.0) or 0.0)
    score = float(_record_value(record, "score", 0.0) or 0.0)
    doc_no = int(_record_value(record, "doc_no", 0) or 0)
    return (-weak, role_bonus, fact_bonus, numeric_boost, score, -doc_no)


def select_auto_prefetch_documents(query: str, records: List[Any], limit: int = 2) -> List[Any]:
    if not records or limit <= 0:
        return []
    ranked = sorted(records, key=lambda record: _auto_prefetch_rank(query, record), reverse=True)
    selected: List[Any] = []
    seen_pages: Set[Tuple[str, str]] = set()
    seen_roles: Set[str] = set()

    def add(record: Any) -> None:
        if len(selected) >= limit:
            return
        page_key = _record_page_key(record)
        if page_key in seen_pages:
            return
        selected.append(record)
        seen_pages.add(page_key)
        role = _record_role(record)
        if role:
            seen_roles.add(role)

    if _procedure_case_query(query):
        for desired_role in ("guide", "casebook"):
            if len(selected) >= limit:
                break
            for record in ranked:
                if _record_role(record) == desired_role:
                    add(record)
                    break

    for record in ranked:
        add(record)
        if len(selected) >= limit:
            break

    if _procedure_case_query(query) and "casebook" not in seen_roles:
        for record in ranked:
            if _record_role(record) == "casebook":
                add(record)
                break
    return selected[:limit]
