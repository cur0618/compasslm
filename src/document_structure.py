import os
import re
from typing import Any, Dict, Iterable, List, Optional


WHITESPACE_RE = re.compile(r"\s+")
CHUNK_KIND_LABELS = {
    "heading": "제목",
    "body": "본문",
    "definition": "정의",
    "condition": "조건",
    "exception": "예외",
    "table_title": "표 제목",
    "table_header": "표 헤더",
    "table_row": "표 행",
    "table_summary": "표 요약",
}


def _clean(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "").strip())


def normalize_heading_path(value: Any) -> List[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [part for part in value.split(">")]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    result: List[str] = []
    for item in values:
        cleaned = _clean(item)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def build_embedding_text(
    *,
    text: str,
    source_path: str = "",
    doc_role: str = "",
    heading_path: Optional[Iterable[str]] = None,
    chunk_kind: str = "body",
) -> str:
    original = str(text or "").strip()
    headings = normalize_heading_path(list(heading_path or []))
    lines: List[str] = []
    filename = os.path.basename(str(source_path or "").strip())
    if filename:
        lines.append(f"문서: {filename}")
    role = _clean(doc_role)
    if role:
        lines.append(f"역할: {role}")
    if headings:
        lines.append(f"경로: {' > '.join(headings)}")
    kind = _clean(chunk_kind) or "body"
    lines.append(f"유형: {CHUNK_KIND_LABELS.get(kind, kind)}")
    lines.append(f"내용: {original}")
    return "\n".join(lines)


def normalize_structure_record(
    record: Dict[str, Any],
    *,
    source_type: str,
    doc_role: str,
) -> Dict[str, Any]:
    normalized = dict(record or {})
    text = str(normalized.get("text", "") or "").strip()
    source_path = str(
        normalized.get("source_path", "")
        or normalized.get("file_path", "")
        or ""
    ).strip()
    heading_path = normalize_heading_path(normalized.get("heading_path", []))
    chunk_kind = _clean(normalized.get("chunk_kind", "body")) or "body"
    normalized.update(
        {
            "text": text,
            "source_path": source_path,
            "source_type": _clean(source_type),
            "doc_role": _clean(doc_role),
            "heading_path": heading_path,
            "chunk_kind": chunk_kind,
            "parent_chunk_key": _clean(normalized.get("parent_chunk_key", "")),
            "structure_path": _clean(
                normalized.get("structure_path", "")
                or normalized.get("hwpx_path", "")
            ),
            "table_id": _clean(normalized.get("table_id", "")),
            "row_no": int(normalized.get("row_no", normalized.get("row", 0)) or 0),
            "cell_no": int(normalized.get("cell_no", 0) or 0),
            "is_derived": bool(normalized.get("is_derived", False)),
        }
    )
    normalized["embedding_text"] = str(normalized.get("embedding_text", "") or "").strip() or build_embedding_text(
        text=text,
        source_path=source_path,
        doc_role=doc_role,
        heading_path=heading_path,
        chunk_kind=chunk_kind,
    )
    return normalized


def chunk_structure_records(
    records: List[Dict[str, Any]],
    *,
    source_type: str,
    doc_role: str,
    source_path: str,
) -> List[Dict[str, Any]]:
    normalized = [
        normalize_structure_record(
            {**record, "source_path": source_path},
            source_type=source_type,
            doc_role=doc_role,
        )
        for record in records
        if str(record.get("text", "") or "").strip()
    ]
    chunks: List[Dict[str, Any]] = []
    table_groups: Dict[str, List[Dict[str, Any]]] = {}
    table_order: List[str] = []

    for record in normalized:
        if record.get("chunk_kind") == "table_row" and not record.get("is_derived"):
            key = record.get("parent_chunk_key") or f"table:{len(table_order) + 1}"
            if key not in table_groups:
                table_groups[key] = []
                table_order.append(key)
            table_groups[key].append(record)
            continue
        chunks.append(record)

    for key in table_order:
        cells = table_groups[key]
        cells.sort(key=lambda item: (int(item.get("cell_no", 0) or 0), int(item.get("line_start", 0) or 0)))
        first = cells[0]
        text_parts = [str(item.get("raw_text", "") or item.get("text", "") or "").strip() for item in cells]
        merged = {
            **first,
            "text": " | ".join(part for part in text_parts if part),
            "line_start": min(int(item.get("line_start", 0) or 0) for item in cells),
            "line_end": max(int(item.get("line_end", item.get("line_start", 0)) or 0) for item in cells),
            "cell_no": 0,
            "parent_chunk_key": key,
        }
        merged["embedding_text"] = build_embedding_text(
            text=merged["text"],
            source_path=source_path,
            doc_role=doc_role,
            heading_path=merged.get("heading_path", []),
            chunk_kind="table_row",
        )
        chunks.append(merged)

    chunks.sort(
        key=lambda item: (
            int(item.get("line_start", 0) or 0),
            int(item.get("row_no", item.get("row", 0)) or 0),
            1 if item.get("is_derived") else 0,
        )
    )
    return chunks
