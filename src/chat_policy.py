import os
import re
from typing import Any, Iterable, Mapping


SUMMARY_HINTS = (
    "요약",
    "정리",
    "핵심",
    "주의사항",
    "유의사항",
    "체크포인트",
)
GENERIC_SCOPE_HINTS = (
    "업로드한문서",
    "업로드문서",
    "문서를확인",
    "문서확인",
    "전체",
    "전부",
    "전반",
    "전반적",
    "전체적",
    "공통",
)


def _compact_text(text: str) -> str:
    return "".join((text or "").strip().lower().split())


def is_broad_summary_request(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if not any(hint in compact for hint in SUMMARY_HINTS):
        return False
    if any(token in compact for token in ("단가", "금액", "얼마", "페이지", "쪽", "행", "라인", "기준")):
        return False
    if any(re.search(pattern, compact) for pattern in (r"\d+페이지", r"\d+쪽", r"\d+행", r"\d+라인")):
        return False
    return any(hint in compact for hint in GENERIC_SCOPE_HINTS) or len(compact) <= 24


def _unique_source_count(results: Iterable[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    for row in results or []:
        source = str(
            row.get("source_display", "")
            or row.get("source_path", "")
            or row.get("source_ref", "")
            or ""
        ).strip()
        if source:
            seen.add(source)
    return len(seen)


def _unique_pdf_source_count(results: Iterable[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    for row in results or []:
        if str(row.get("source_type", "") or "").strip().lower() != "pdf":
            continue
        source = str(
            row.get("source_display", "")
            or row.get("source_path", "")
            or row.get("source_ref", "")
            or ""
        ).strip()
        if source:
            seen.add(source)
    return len(seen)


def explain_scope_nudge_reason(
    metrics: Mapping[str, Any] | None,
    results: Iterable[Mapping[str, Any]],
    *,
    kb_file_count: int = 0,
    overview_mode: bool = False,
) -> str:
    unique_sources = max(int((metrics or {}).get("unique_sources", 0) or 0), _unique_source_count(results))
    pdf_sources = _unique_pdf_source_count(results)
    if pdf_sources >= 4:
        return "many_pdf_sources"
    if overview_mode and pdf_sources >= 3:
        return "overview_many_pdf_sources"
    if unique_sources >= 5:
        return "many_unique_sources"
    if int(kb_file_count or 0) >= 4 and unique_sources >= 3:
        return "broad_kb_scope"
    return "broad_summary_scope"


def should_prompt_for_narrower_summary(
    query: str,
    *,
    metrics: Mapping[str, Any] | None,
    results: Iterable[Mapping[str, Any]],
    kb_file_count: int = 0,
    overview_mode: bool = False,
) -> bool:
    if not is_broad_summary_request(query):
        return False
    rows = list(results or [])
    if not rows:
        return False

    unique_sources = max(int((metrics or {}).get("unique_sources", 0) or 0), _unique_source_count(rows))
    pdf_sources = _unique_pdf_source_count(rows)
    top1 = float((metrics or {}).get("top1", 0.0) or 0.0)
    coverage = float((metrics or {}).get("coverage", 0.0) or 0.0)
    kb_files = max(0, int(kb_file_count or 0))

    if pdf_sources >= 4:
        return True
    if unique_sources >= 5:
        return True
    if overview_mode and (pdf_sources >= 3 or unique_sources >= 4):
        return True
    if kb_files >= 4 and unique_sources >= 3 and coverage < 0.45:
        return True
    if pdf_sources >= 3 and (top1 < 0.45 or coverage < 0.55):
        return True
    return False


def _clean_source_name(results: Iterable[Mapping[str, Any]]) -> str:
    for row in results or []:
        raw = str(
            row.get("source_display", "")
            or row.get("source_path", "")
            or row.get("source_ref", "")
            or ""
        ).strip()
        if not raw:
            continue
        basename = os.path.basename(raw)
        stem, _ = os.path.splitext(basename)
        return stem or basename
    return "문서"


def build_scope_narrowing_response(query: str, *, results: Iterable[Mapping[str, Any]]) -> str:
    source_name = _clean_source_name(results)
    compact = _compact_text(query)
    if "주의사항" in compact or "유의사항" in compact:
        topic_example = "안전 관련 주의사항만 정리해 주세요."
        focused_example = "답례품 단가 관련 주의사항만 알려 주세요."
    else:
        topic_example = "핵심 내용만 3가지로 정리해 주세요."
        focused_example = "업무 처리 기준만 정리해 주세요."

    return (
        "업로드된 문서 범위가 넓어서 한 번에 정확하게 정리하면 기준이 서로 섞일 수 있습니다. "
        "범위를 조금만 좁혀 주시면 그 부분만 바로 정리해 드리겠습니다.\n"
        f"예를 들면 [{source_name} 25페이지 기준으로만 정리해 주세요.]\n"
        f"[{topic_example}]\n"
        f"[{focused_example}]"
    )
