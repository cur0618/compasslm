import os
import re
from typing import Any, Mapping


DOC_LABEL_PATTERN = re.compile(r"\[DOC\s+(\d+)\]")
LEGACY_DOC_LABEL_PATTERN = re.compile(r"\[\[\s*DOC\s+(\d+)(?:\|[^\]\n]*)?\s*\]\]", flags=re.IGNORECASE)
LEGACY_CITATION_LABEL_PATTERN = re.compile(r"\[\[\s*CITATION\s*:\s*(\d+)(?:\|[^\]\n]*)?\s*\]\]", flags=re.IGNORECASE)
LOOSE_CITATION_LABEL_PATTERN = re.compile(r"\[\s*CITATION\s*:\s*(\d+)(?:\|[^\]\n]*)?\s*\]", flags=re.IGNORECASE)
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s,，.。;；:：()\[\]{}<>·ㆍ/-]*$")


def _read(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _clean_filename(source_path: str) -> str:
    raw_value = str(source_path or "").strip()
    if " / " in raw_value:
        raw_value = raw_value.split(" / ", 1)[0].strip()
    basename = os.path.basename(raw_value)
    if not basename:
        return "문서"
    if "__" in basename:
        prefix, remainder = basename.split("__", 1)
        if re.fullmatch(r"\d{8,14}_[0-9A-Za-z]{6,}", prefix) and remainder:
            basename = remainder
    parts = basename.split("_")
    if (
        len(parts) >= 3
        and re.fullmatch(r"\d{8,14}", parts[0])
        and re.fullmatch(r"[0-9A-Za-z]{6,}", parts[1])
    ):
        candidate = "_".join(parts[2:]).strip("_")
        if candidate:
            basename = candidate
    stem, _ext = os.path.splitext(basename)
    return stem or basename


def _resolve_source_name(record: Any) -> str:
    top_level_source_display = str(_read(record, "source_display", "") or "").strip()
    if top_level_source_display:
        return _clean_filename(top_level_source_display)

    metadata = _read(record, "metadata", {})
    if isinstance(metadata, Mapping):
        raw_row = metadata.get("raw_row", {})
        if isinstance(raw_row, Mapping):
            nested_source_display = str(raw_row.get("source_display", "") or "").strip()
            if nested_source_display:
                return _clean_filename(nested_source_display)

    source_path = str(_read(record, "source_path", "") or "").strip()
    if source_path:
        return _clean_filename(source_path)

    source_ref = str(_read(record, "source_ref", "") or "").strip()
    if source_ref:
        return _clean_filename(source_ref)

    return "문서"


def _extract_page_no(record: Any) -> int:
    for key in ("page_no", "page"):
        try:
            value = int(_read(record, key, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    section = str(_read(record, "section", "") or "").strip()
    match = re.search(r"page\s*(\d+)", section, flags=re.IGNORECASE)
    if match:
        page_no = int(match.group(1))
        max_fallback_page = 1000
        try:
            max_fallback_page = int(os.getenv("CITATION_MAX_SECTION_PAGE_FALLBACK", "1000") or "1000")
        except Exception:
            max_fallback_page = 1000
        if page_no <= max(1, max_fallback_page):
            return page_no
    return 0


def build_user_facing_citation_label(record: Any) -> str:
    source_name = _resolve_source_name(record)
    source_type = str(_read(record, "source_type", "") or "").strip().lower()
    sheet = str(_read(record, "sheet", "") or "").strip()

    if source_type == "pdf" or _extract_page_no(record) > 0:
        page_no = _extract_page_no(record)
        return f"{source_name} {page_no}페이지" if page_no > 0 else source_name

    if sheet:
        row = max(0, int(_read(record, "row", 0) or 0))
        row_end = max(row, int(_read(record, "row_end", row) or row))
        if row > 0 and row_end > row:
            return f"{source_name} / {sheet} / {row}-{row_end}행"
        if row > 0:
            return f"{source_name} / {sheet} / {row}행"
        return f"{source_name} / {sheet}"

    line_start = max(0, int(_read(record, "line_start", 0) or 0))
    line_end = max(line_start, int(_read(record, "line_end", line_start) or line_start))
    if line_start > 0 and line_end > line_start:
        return f"{source_name} / {line_start}-{line_end}라인"
    if line_start > 0:
        return f"{source_name} / {line_start}라인"

    return source_name


def _sanitize_citation_label_for_token(label: str) -> str:
    cleaned = " ".join(str(label or "").split())
    if not cleaned:
        return "문서"
    return cleaned.replace("|", "/").replace("[", "(").replace("]", ")")


def build_user_facing_citation_token(doc_no: int, record: Any) -> str:
    label = _sanitize_citation_label_for_token(build_user_facing_citation_label(record))
    return f"[[CITATION:{int(doc_no)}|{label}]]"


def canonicalize_doc_citations(answer_text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return f"[DOC {int(match.group(1))}]"

    normalized = LEGACY_DOC_LABEL_PATTERN.sub(_replace, answer_text or "")
    normalized = LEGACY_CITATION_LABEL_PATTERN.sub(_replace, normalized)
    return LOOSE_CITATION_LABEL_PATTERN.sub(_replace, normalized)


def replace_doc_citations(answer_text: str, docs_by_no: Mapping[int, Any]) -> str:
    normalized_answer = canonicalize_doc_citations(answer_text or "")

    def _replace(match: re.Match[str]) -> str:
        doc_no = int(match.group(1))
        record = docs_by_no.get(doc_no)
        if record is None:
            return match.group(0)
        return build_user_facing_citation_token(doc_no, record)

    return DOC_LABEL_PATTERN.sub(_replace, normalized_answer)


def _strip_inline_citation_artifacts(text: str) -> str:
    cleaned_lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if re.match(r"^(?:문서\s*)?근거\s*[:：]", stripped) and PUNCTUATION_ONLY_PATTERN.fullmatch(
            re.sub(r"^(?:문서\s*)?근거\s*[:：]", "", stripped).strip()
        ):
            continue
        if PUNCTUATION_ONLY_PATTERN.fullmatch(stripped):
            continue
        cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"[ \t]+([,，.。;；:：])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def render_answer_with_bottom_citations(answer_text: str, docs_by_no: Mapping[int, Any]) -> str:
    normalized_answer = canonicalize_doc_citations(answer_text or "")
    ordered_doc_numbers = []
    seen = set()
    for match in DOC_LABEL_PATTERN.finditer(normalized_answer):
        doc_no = int(match.group(1))
        if doc_no in seen or docs_by_no.get(doc_no) is None:
            continue
        ordered_doc_numbers.append(doc_no)
        seen.add(doc_no)

    body = DOC_LABEL_PATTERN.sub("", normalized_answer)
    body = _strip_inline_citation_artifacts(body)
    if not ordered_doc_numbers:
        return body

    tokens = [
        build_user_facing_citation_token(display_no, docs_by_no[doc_no])
        for display_no, doc_no in enumerate(ordered_doc_numbers, start=1)
    ]
    citation_line = "근거: " + ", ".join(tokens)
    if not body:
        return citation_line
    return f"{body}\n\n{citation_line}"
