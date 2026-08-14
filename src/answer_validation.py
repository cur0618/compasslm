import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Set

from src.table_facts import parse_table_fact_line, table_fact_matches_query


SEEDED_EVIDENCE_TOP1_MIN = 0.33
SEEDED_EVIDENCE_COVERAGE_MIN = 0.22
EVIDENCE_ALIGNMENT_MIN = 0.34

NUMERIC_EVIDENCE_HINTS = (
    "단가",
    "금액",
    "비용",
    "얼마",
    "가격",
    "예산",
    "수수료",
    "요금",
    "답례품",
    "천원",
    "만원",
    "억원",
    "%",
    "퍼센트",
    "비율",
    "건수",
    "몇 건",
    "몇건",
    "몇 명",
    "몇명",
    "총액",
    "합계",
    "수량",
    "언제",
    "시기",
    "주기",
    "일정",
    "기일",
    "월별",
    "기준월",
    "지급시기",
    "지급 시기",
    "지급주기",
    "지급 주기",
    "지급기준월",
    "지급 기준월",
    "지급대상월",
    "보고기일",
    "보고 기일",
    "조사시기",
    "조사 시기",
    "조사주기",
    "조사 주기",
    "연간",
    "반기",
    "분기",
    "월",
    "일",
)
NON_NUMERIC_INFORMATION_HINTS = (
    "종류",
    "품목",
    "무엇",
    "뭐",
    "어떤",
)
STRONG_NUMERIC_OR_TEMPORAL_HINTS = tuple(
    hint
    for hint in NUMERIC_EVIDENCE_HINTS
    if hint not in {"답례품", "월", "일"}
)
HIGH_RISK_FALLBACK_HINTS = (
    "법",
    "법률",
    "세금",
    "세무",
    "의료",
    "병원",
    "치료",
    "약",
    "복용",
    "투자",
    "주식",
    "대출",
    "계약",
    "판결",
    "판례",
    "규정",
    "정책",
    "안전",
    "보안",
    "개인정보",
    "보험",
)
DOC_LABEL_PATTERN = re.compile(r"\[DOC\s+\d+\]")
SPACE_PATTERN = re.compile(r"\s+")
DISALLOWED_MARKDOWN_PATTERNS = (
    re.compile(r"(^|\n)\s{0,3}#{1,6}\s+\S"),
    re.compile(r"(^|\n)\s{0,3}[-*+]\s+\S"),
    re.compile(r"(^|\n)\s{0,3}\d+\.\s+\S"),
    re.compile(r"\*\*[^\n]+?\*\*"),
    re.compile(r"__[^\n]+?__"),
    re.compile(r"`[^`\n]+`"),
)
ALIGNMENT_STOP_TOKENS = {
    "알려줘",
    "어떻게",
    "조사",
    "해야해",
    "해줘",
    "경우",
    "처리",
    "방법",
    "문서",
    "근거",
    "대한",
    "있는",
    "없는",
    "합니다",
}
WEAK_OCR_HINT_REQUIRED_MARKERS = ("OCR후보:", "원문 확인 필요")
STRONG_TABLE_MARKERS = ("표행:", "표행요약:", "표값:", "표헤더:")
GROUNDED_ABSTENTION_HINTS = (
    "문서근거부족",
    "근거부족으로단정할수없",
    "단정할수없",
    "문서상확인되지않",
    "확인되지않",
    "찾지못했",
    "근거가부족",
)
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?<![\d.])"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s*(?:\(\s*(천원|만원|억원|백만원|십만원|원|퍼센트|%)\s*\)|(천원|만원|억원|백만원|십만원|원|퍼센트|%|명|건|개|호|회))"
)
TEMPORAL_LITERAL_PATTERN = re.compile(r"(?<![\d.])(\d{1,4})\s*(년|월|일|분기)")
TEMPORAL_KEYWORD_PATTERN = re.compile(r"(상반기|하반기|반기|분기|연간|월간|주간|매월|매년|초순|중순|하순)")

MONEY_UNIT_FACTORS = {
    "원": Decimal("1"),
    "천원": Decimal("1000"),
    "만원": Decimal("10000"),
    "십만원": Decimal("100000"),
    "백만원": Decimal("1000000"),
    "억원": Decimal("100000000"),
}
PERCENT_UNITS = {"%", "퍼센트"}
COUNT_UNITS = {"명", "건", "개", "호", "회"}


def _compact_text(text: str) -> str:
    return SPACE_PATTERN.sub("", (text or "").lower())


def is_grounded_abstention_text(text: str) -> bool:
    compact = _compact_text(text)
    return any(hint in compact for hint in GROUNDED_ABSTENTION_HINTS)


def is_numeric_evidence_query(query: str) -> bool:
    compact = _compact_text(query)
    if any(hint in compact for hint in NON_NUMERIC_INFORMATION_HINTS) and not any(
        hint in compact for hint in STRONG_NUMERIC_OR_TEMPORAL_HINTS
    ):
        return False
    return any(hint in compact for hint in NUMERIC_EVIDENCE_HINTS)


def is_weak_ocr_hint_text(text: str) -> bool:
    payload = (text or "").strip()
    if not payload:
        return False
    if any(marker not in payload for marker in WEAK_OCR_HINT_REQUIRED_MARKERS):
        return False
    if any(marker in payload for marker in STRONG_TABLE_MARKERS):
        return False
    nonempty_lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not nonempty_lines:
        return False
    if all(line.startswith("OCR후보:") for line in nonempty_lines):
        return True
    return len(payload) <= 240


def summarize_evidence_strength(rows: Iterable[Dict[str, Any]]) -> Dict[str, int | bool]:
    weak_count = 0
    strong_count = 0
    total_count = 0
    for row in rows:
        total_count += 1
        explicit_strength = str((row or {}).get("evidence_strength", "") or "").strip().lower()
        if explicit_strength == "weak":
            weak_count += 1
            continue
        if explicit_strength == "strong":
            strong_count += 1
            continue
        if bool((row or {}).get("weak_ocr_hint", False)) or is_weak_ocr_hint_text(str((row or {}).get("text", "") or "")):
            weak_count += 1
            continue
        strong_count += 1

    return {
        "total_evidence_count": total_count,
        "weak_evidence_count": weak_count,
        "strong_evidence_count": strong_count,
        "weak_evidence_only": bool(total_count > 0 and strong_count == 0 and weak_count > 0),
    }


def has_seeded_retrieval_evidence(
    *,
    docs_available: int,
    metrics: Dict[str, Any] | None = None,
) -> bool:
    if int(docs_available or 0) <= 0:
        return False
    payload = metrics or {}
    top1 = float(payload.get("top1", 0.0) or 0.0)
    coverage = float(payload.get("coverage", 0.0) or 0.0)
    return top1 >= SEEDED_EVIDENCE_TOP1_MIN or coverage >= SEEDED_EVIDENCE_COVERAGE_MIN


def should_treat_abstention_as_quality_issue(query: str, metrics: Dict[str, Any] | None = None) -> bool:
    if is_numeric_evidence_query(query):
        return False
    payload = metrics or {}
    top1 = float(payload.get("top1", 0.0) or 0.0)
    coverage = float(payload.get("coverage", 0.0) or 0.0)
    return top1 >= max(0.38, SEEDED_EVIDENCE_TOP1_MIN) and coverage >= max(0.24, SEEDED_EVIDENCE_COVERAGE_MIN)


def build_tool_recheck_debug_payload(
    *,
    require_tool_evidence: bool,
    allow_retrieval_tool: bool,
    docs_available: int,
    metrics: Dict[str, Any] | None,
    new_tool_event_count: int,
    numeric_evidence_required: bool = False,
    auto_prefetch_satisfied: bool = False,
    candidate_is_grounded_abstention: bool = False,
    query_text: str = "",
    evidence_texts: Iterable[str] | None = None,
) -> Dict[str, Any]:
    payload = metrics or {}
    alignment = evaluate_evidence_alignment(query_text, evidence_texts or [])
    normalized = {
        "require_tool_evidence": bool(require_tool_evidence),
        "allow_retrieval_tool": bool(allow_retrieval_tool),
        "docs_available": max(0, int(docs_available or 0)),
        "new_tool_event_count": max(0, int(new_tool_event_count or 0)),
        "numeric_evidence_required": bool(numeric_evidence_required),
        "auto_prefetch_satisfied": bool(auto_prefetch_satisfied),
        "candidate_is_grounded_abstention": bool(candidate_is_grounded_abstention),
        "metrics_top1": float(payload.get("top1", 0.0) or 0.0),
        "metrics_coverage": float(payload.get("coverage", 0.0) or 0.0),
        "metrics_unique_sources": max(0, int(payload.get("unique_sources", 0) or 0)),
        "top1_threshold": SEEDED_EVIDENCE_TOP1_MIN,
        "coverage_threshold": SEEDED_EVIDENCE_COVERAGE_MIN,
        "evidence_alignment_ok": bool(alignment["ok"]),
        "evidence_alignment_score": float(alignment["score"]),
        "evidence_alignment_threshold": EVIDENCE_ALIGNMENT_MIN,
        "query_terms": alignment["query_terms"],
        "matched_terms": alignment["matched_terms"],
    }
    normalized["seeded_retrieval_evidence_ok"] = has_seeded_retrieval_evidence(
        docs_available=normalized["docs_available"],
        metrics=payload,
    )
    normalized["should_require_tool_recheck"] = should_require_tool_recheck(
        require_tool_evidence=normalized["require_tool_evidence"],
        allow_retrieval_tool=normalized["allow_retrieval_tool"],
        docs_available=normalized["docs_available"],
        metrics=payload,
        new_tool_event_count=normalized["new_tool_event_count"],
        numeric_evidence_required=normalized["numeric_evidence_required"],
        auto_prefetch_satisfied=normalized["auto_prefetch_satisfied"],
        candidate_is_grounded_abstention=normalized["candidate_is_grounded_abstention"],
        evidence_alignment_ok=normalized["evidence_alignment_ok"],
    )
    return normalized


def _alignment_tokens(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or "").lower())
    tokens: Set[str] = set()
    for token in raw_tokens:
        if token in ALIGNMENT_STOP_TOKENS:
            continue
        if len(token) < 2:
            continue
        tokens.add(token)
    return tokens


def evaluate_evidence_alignment(query_text: str, evidence_texts: Iterable[str]) -> Dict[str, Any]:
    query_terms = _alignment_tokens(query_text)
    if not query_terms:
        return {"ok": True, "score": 1.0, "query_terms": [], "matched_terms": []}

    evidence_terms: Set[str] = set()
    for text in evidence_texts:
        evidence_terms.update(_alignment_tokens(text))

    if not evidence_terms:
        return {
            "ok": False,
            "score": 0.0,
            "query_terms": sorted(query_terms),
            "matched_terms": [],
        }

    matched = {
        query_term
        for query_term in query_terms
        if query_term in evidence_terms or any(query_term in evidence_term or evidence_term in query_term for evidence_term in evidence_terms)
    }
    score = len(matched) / max(1, len(query_terms))
    return {
        "ok": score >= EVIDENCE_ALIGNMENT_MIN,
        "score": score,
        "query_terms": sorted(query_terms),
        "matched_terms": sorted(matched),
    }


def should_require_tool_recheck(
    *,
    require_tool_evidence: bool,
    allow_retrieval_tool: bool,
    docs_available: int,
    metrics: Dict[str, Any] | None,
    new_tool_event_count: int,
    numeric_evidence_required: bool = False,
    auto_prefetch_satisfied: bool = False,
    candidate_is_grounded_abstention: bool = False,
    evidence_alignment_ok: bool = True,
) -> bool:
    if not require_tool_evidence or not allow_retrieval_tool:
        return False
    if auto_prefetch_satisfied:
        return False
    if int(new_tool_event_count or 0) > 0:
        return False
    seeded_retrieval_evidence = has_seeded_retrieval_evidence(
        docs_available=docs_available,
        metrics=metrics,
    )
    if candidate_is_grounded_abstention and not numeric_evidence_required and not seeded_retrieval_evidence:
        return False
    if numeric_evidence_required:
        return True
    if seeded_retrieval_evidence and not evidence_alignment_ok:
        return True
    return not seeded_retrieval_evidence


def should_auto_prefetch_numeric_evidence(
    *,
    require_tool_evidence: bool,
    allow_retrieval_tool: bool,
    docs_available: int,
    metrics: Dict[str, Any] | None,
    new_tool_event_count: int,
    numeric_evidence_required: bool = False,
) -> bool:
    if not numeric_evidence_required:
        return False
    if not allow_retrieval_tool:
        return False
    if int(docs_available or 0) <= 0:
        return False
    if int(new_tool_event_count or 0) > 0:
        return False
    if require_tool_evidence:
        return True
    return has_seeded_retrieval_evidence(
        docs_available=docs_available,
        metrics=metrics,
    )


def build_outline_recheck_debug_payload(
    *,
    use_source_outline: bool,
    outline_tool_used: bool,
    docs_available: int,
    metrics: Dict[str, Any] | None,
    numeric_evidence_required: bool = False,
) -> Dict[str, Any]:
    payload = metrics or {}
    normalized = {
        "use_source_outline": bool(use_source_outline),
        "outline_tool_used": bool(outline_tool_used),
        "docs_available": max(0, int(docs_available or 0)),
        "numeric_evidence_required": bool(numeric_evidence_required),
        "metrics_top1": float(payload.get("top1", 0.0) or 0.0),
        "metrics_coverage": float(payload.get("coverage", 0.0) or 0.0),
        "metrics_unique_sources": max(0, int(payload.get("unique_sources", 0) or 0)),
        "top1_threshold": SEEDED_EVIDENCE_TOP1_MIN,
        "coverage_threshold": SEEDED_EVIDENCE_COVERAGE_MIN,
    }
    normalized["seeded_retrieval_evidence_ok"] = has_seeded_retrieval_evidence(
        docs_available=normalized["docs_available"],
        metrics=payload,
    )
    normalized["should_require_outline_recheck"] = should_require_outline_recheck(
        use_source_outline=normalized["use_source_outline"],
        outline_tool_used=normalized["outline_tool_used"],
        docs_available=normalized["docs_available"],
        metrics=payload,
        numeric_evidence_required=normalized["numeric_evidence_required"],
    )
    return normalized


def should_require_outline_recheck(
    *,
    use_source_outline: bool,
    outline_tool_used: bool,
    docs_available: int,
    metrics: Dict[str, Any] | None,
    numeric_evidence_required: bool = False,
) -> bool:
    if numeric_evidence_required and not outline_tool_used:
        return True
    if not use_source_outline:
        return False
    if outline_tool_used:
        return False
    return not has_seeded_retrieval_evidence(
        docs_available=docs_available,
        metrics=metrics,
    )


def should_allow_general_knowledge_fallback(
    query: str,
    *,
    kb_has_docs: bool,
) -> bool:
    compact = _compact_text(query)
    if kb_has_docs:
        return False
    if is_numeric_evidence_query(query):
        return False
    return not any(hint in compact for hint in HIGH_RISK_FALLBACK_HINTS)


def _parse_decimal(raw: str) -> Decimal | None:
    cleaned = (raw or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _normalize_numeric_signature(value: Decimal, unit: str) -> str:
    normalized_unit = (unit or "").strip()
    if normalized_unit in MONEY_UNIT_FACTORS:
        amount = value * MONEY_UNIT_FACTORS[normalized_unit]
        if amount == amount.to_integral_value():
            return f"money:{int(amount)}"
        return f"money:{amount.normalize()}"
    if normalized_unit in PERCENT_UNITS:
        return f"percent:{value.normalize()}"
    if normalized_unit in COUNT_UNITS:
        return f"count:{value.normalize()}:{normalized_unit}"
    return ""


def extract_numeric_signatures(text: str) -> Set[str]:
    payload = DOC_LABEL_PATTERN.sub(" ", text or "")
    signatures: Set[str] = set()
    for match in NUMBER_WITH_UNIT_PATTERN.finditer(payload):
        value = _parse_decimal(match.group(1))
        if value is None:
            continue
        unit = (match.group(2) or match.group(3) or "").strip()
        signature = _normalize_numeric_signature(value, unit)
        if signature:
            signatures.add(signature)
    for match in re.finditer(r"(?:단가|금액|비용|가격|수수료|요금)\s*=\s*(\d[\d,]*(?:\.\d+)?)\b(?!\s*(?:원|천원|만원|억원|백만원|십만원))", payload):
        value = _parse_decimal(match.group(1))
        if value is None:
            continue
        signature = _normalize_numeric_signature(value, "천원")
        if signature:
            signatures.add(signature)
    return signatures


def extract_temporal_signatures(text: str) -> Set[str]:
    payload = DOC_LABEL_PATTERN.sub(" ", text or "")
    signatures: Set[str] = set()
    for match in TEMPORAL_LITERAL_PATTERN.finditer(payload):
        value = (match.group(1) or "").strip()
        unit = (match.group(2) or "").strip()
        if value and unit:
            signatures.add(f"time:{value}{unit}")
    for match in TEMPORAL_KEYWORD_PATTERN.finditer(payload):
        token = (match.group(1) or "").strip()
        if token:
            signatures.add(f"timekw:{token}")
    return signatures


def has_grounded_numeric_answer(
    *,
    query: str,
    answer_text: str,
    evidence_texts: Iterable[str],
) -> bool:
    if not is_numeric_evidence_query(query):
        return True

    answer_signatures = extract_numeric_signatures(answer_text)
    answer_temporal = extract_temporal_signatures(answer_text)
    if not answer_signatures and not answer_temporal:
        return False

    evidence_items = list(evidence_texts)
    matching_fact_signatures: Set[str] = set()
    matching_fact_temporal: Set[str] = set()
    matching_fact_seen = False
    for text in evidence_items:
        for line in str(text or "").splitlines():
            fact = parse_table_fact_line(line)
            if not fact or fact.get("kind") != "table_row":
                continue
            if not table_fact_matches_query(fact, query):
                continue
            matching_fact_seen = True
            matching_fact_signatures.update(extract_numeric_signatures(line))
            matching_fact_temporal.update(extract_temporal_signatures(line))

    if matching_fact_seen:
        if answer_signatures and not matching_fact_signatures:
            return False
        if answer_temporal and not matching_fact_temporal:
            return False
        if answer_signatures and not answer_signatures.issubset(matching_fact_signatures):
            return False
        if answer_temporal and not answer_temporal.issubset(matching_fact_temporal):
            return False
        return True

    evidence_signatures: Set[str] = set()
    evidence_temporal: Set[str] = set()
    for text in evidence_items:
        evidence_signatures.update(extract_numeric_signatures(text))
        evidence_temporal.update(extract_temporal_signatures(text))

    if answer_signatures and not evidence_signatures:
        return False
    if answer_temporal and not evidence_temporal:
        return False
    if answer_signatures and not answer_signatures.issubset(evidence_signatures):
        return False
    if answer_temporal and not answer_temporal.issubset(evidence_temporal):
        return False
    return True


def contains_disallowed_markdown(text: str) -> bool:
    payload = DOC_LABEL_PATTERN.sub(" ", text or "")
    return any(pattern.search(payload) for pattern in DISALLOWED_MARKDOWN_PATTERNS)
