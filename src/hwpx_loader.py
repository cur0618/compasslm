import os
import re
from typing import Any, Dict, List, Tuple

from src.table_facts import build_table_row_fact_text
from src.table_alias import generate_name_aliases, is_name_header, should_emit_name_alias
from src.document_structure import normalize_structure_record


SECTION_PATTERN = re.compile(r"^(#{1,6}\s+.+|(?:\d+(?:\.\d+){0,3}|[가-힣A-Za-z]\))\s+.+|제\s*\d+\s*(장|절|항|조).+)$")
WHITESPACE_PATTERN = re.compile(r"\s+")
TABLE_INDEX_PATTERN = re.compile(r"(tbl|table|tr|row|tc|cell)\[(\d+)\]", re.IGNORECASE)
BARE_TABLE_TOKEN_PATTERN = re.compile(r"(^|[/#:])(tbl|table|tr|row|tc|cell)(?=$|[/#:])", re.IGNORECASE)
NUMERIC_ONLY_PATTERN = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
PRICE_HEADER_PATTERN = re.compile(r"(단가|금액|비용|가격|수수료|요금)")
HEADING_LEVEL_PATTERN = re.compile(r"^제\s*\d+\s*(장|절|조|항)")
CONDITION_PATTERN = re.compile(r"(경우|때|요건|조건|한하여)")
EXCEPTION_PATTERN = re.compile(r"(다만|제외|예외|불구하고)")
DEFINITION_PATTERN = re.compile(r"(이란|란\s|이라 함은|정의한다)")
HEADING_LEVELS = {"장": 1, "절": 2, "조": 3, "항": 4}
STYLE_HEADING_PATTERN = re.compile(r"(?:heading|제목|개요)\s*([1-6])?", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", (value or "").strip())


def _is_section_text(value: str) -> bool:
    return bool(SECTION_PATTERN.match(value or ""))


def _heading_level(value: str) -> int:
    match = HEADING_LEVEL_PATTERN.match(value or "")
    if match:
        return HEADING_LEVELS.get(match.group(1), 4)
    return 4


def _paragraph_style_heading_level(paragraph: Any) -> int:
    for attr in ("outline_level", "heading_level"):
        try:
            level = int(getattr(paragraph, attr, 0) or 0)
        except (TypeError, ValueError):
            level = 0
        if 1 <= level <= 6:
            return level
    style = getattr(paragraph, "style", None)
    style_values = [
        getattr(paragraph, "style_name", ""),
        getattr(paragraph, "style_id", ""),
        getattr(style, "name", "") if style is not None else "",
        getattr(style, "id", "") if style is not None else "",
    ]
    for raw in style_values:
        match = STYLE_HEADING_PATTERN.search(str(raw or ""))
        if match:
            return int(match.group(1) or 1)
    return 0


def _classify_paragraph_kind(text: str, *, is_heading: bool, is_table: bool) -> str:
    if is_heading:
        return "heading"
    if is_table:
        return "table_row"
    if EXCEPTION_PATTERN.search(text or ""):
        return "exception"
    if CONDITION_PATTERN.search(text or ""):
        return "condition"
    if DEFINITION_PATTERN.search(text or ""):
        return "definition"
    return "body"


def _is_table_path(path: str) -> bool:
    lowered = (path or "").lower()
    return any(marker in lowered for marker in ("/tbl", ":tbl", "/tc", ":tc", "table", "cell"))


def _table_path_indexes(path: str) -> Tuple[str, int, int]:
    indexes: Dict[str, int] = {}
    table_id_parts: List[str] = []
    for match in TABLE_INDEX_PATTERN.finditer(path or ""):
        token = match.group(1).lower()
        value = int(match.group(2))
        if token in {"tbl", "table"}:
            indexes["table"] = value
            table_id_parts.append(match.group(0))
        elif token in {"tr", "row"}:
            indexes["row"] = value
        elif token in {"tc", "cell"}:
            indexes["cell"] = value

    table_id = ":".join(table_id_parts) if table_id_parts else "table"
    return table_id, int(indexes.get("row", 0) or 0), int(indexes.get("cell", 0) or 0)


def _is_explicit_table_cell_path(path: str) -> bool:
    lowered = path or ""
    return bool(TABLE_INDEX_PATTERN.search(lowered) or BARE_TABLE_TOKEN_PATTERN.search(lowered))


def _name_alias_text(value: str, *, header: str = "") -> str:
    if should_emit_name_alias(value, header=header):
        return ", ".join(generate_name_aliases(value))
    return ""


def _dedupe_aliases(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        key = _normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _format_table_value(header: str, value: str, row_text: str) -> str:
    normalized_value = _normalize_text(value)
    if PRICE_HEADER_PATTERN.search(header or "") and NUMERIC_ONLY_PATTERN.match(normalized_value):
        if "천원" not in row_text and "만원" not in row_text and "원" not in normalized_value:
            return f"{normalized_value}천원"
    return normalized_value


def _looks_like_thousand_won_price(value: str) -> bool:
    cleaned = (value or "").replace(",", "").strip()
    if not NUMERIC_ONLY_PATTERN.match(cleaned):
        return False
    try:
        numeric = float(cleaned)
    except ValueError:
        return False
    return 0 < numeric <= 500


def _build_flattened_survey_hint_records(
    table_cells: List[Dict[str, Any]],
    *,
    filename: str,
    structure_v2: bool = False,
) -> List[Dict[str, Any]]:
    unindexed = [
        cell
        for cell in table_cells
        if _is_explicit_table_cell_path(str(cell.get("hwpx_path", "") or ""))
        and _table_path_indexes(str(cell.get("hwpx_path", "") or ""))[1:] == (0, 0)
    ]
    if not unindexed:
        return []

    groups: List[List[Dict[str, Any]]] = []
    for cell in sorted(unindexed, key=lambda item: (int(item.get("hwpx_section_index", 0) or 0), int(item.get("line_start", 0) or 0))):
        if (
            groups
            and int(groups[-1][-1].get("hwpx_section_index", 0) or 0) == int(cell.get("hwpx_section_index", 0) or 0)
            and int(cell.get("line_start", 0) or 0) <= int(groups[-1][-1].get("line_end", 0) or 0) + 1
        ):
            groups[-1].append(cell)
        else:
            groups.append([cell])

    hint_records: List[Dict[str, Any]] = []
    for group in groups:
        values = [_normalize_text(str(cell.get("raw_text", "") or "")) for cell in group]
        for index, value in enumerate(values):
            if not _name_alias_text(value, header="조사명"):
                continue
            price = ""
            for candidate in values[index + 1 : index + 6]:
                if _looks_like_thousand_won_price(candidate):
                    price = f"{candidate}천원"
                    break
            if not price:
                continue
            aliases = _name_alias_text(value, header="조사명")
            pairs = [f"조사명={value}", f"지급단가={price}", f"명칭별칭={aliases}"]
            if index > 0 and values[index - 1]:
                pairs.insert(0, f"답례품={values[index - 1]}")
            fact_headers = ["답례품", "조사명", "지급단가"] if index > 0 else ["조사명", "지급단가"]
            fact_values = [values[index - 1], value, price] if index > 0 else [value, price]
            row_start = min(int(cell.get("line_start", 0) or 0) for cell in group)
            row_end = max(int(cell.get("line_end", 0) or 0) for cell in group)
            first_cell = group[0]
            base_record = {
                "file_path": filename,
                "is_section": False,
                "section": first_cell.get("section", ""),
                "hwpx_section_index": first_cell.get("hwpx_section_index", 0),
                "hwpx_path": first_cell.get("hwpx_path", ""),
                "is_nested": True,
                "is_table": True,
                "is_table_summary": True,
                "line_start": row_start,
                "line_end": row_end,
                "hwpx_paragraph_index": min(int(cell.get("hwpx_paragraph_index", 0) or 0) for cell in group),
            }
            parent_key = f"hwpx:{first_cell.get('hwpx_section_index', 0)}:flat:{row_start}-{row_end}"
            extra = {
                "chunk_kind": "table_summary",
                "parent_chunk_key": parent_key,
                "is_derived": True,
                "heading_path": list(first_cell.get("heading_path", []) or []),
                "row_no": 0,
                "cell_no": 0,
            } if structure_v2 else {}
            hint_records.append({**base_record, **extra, "text": f"표행요약: {' | '.join(pairs)}"})
            hint_records.append({**base_record, **extra, "text": build_table_row_fact_text(fact_headers, fact_values)})
            hint_records.append({**base_record, **extra, "text": f"표값: {value} 지급단가 {price}"})

    return hint_records


def _build_hwpx_table_hint_records(
    table_cells: List[Dict[str, Any]],
    *,
    filename: str,
    structure_v2: bool = False,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for cell in table_cells:
        if not _is_explicit_table_cell_path(str(cell.get("hwpx_path", "") or "")):
            continue
        table_id, row_no, cell_no = _table_path_indexes(str(cell.get("hwpx_path", "") or ""))
        if row_no <= 0 or cell_no <= 0:
            continue
        grouped.setdefault((int(cell.get("hwpx_section_index", 0) or 0), table_id), []).append(
            {**cell, "row_no": row_no, "cell_no": cell_no}
        )

    hint_records: List[Dict[str, Any]] = []
    for (_section_index, _table_id), cells in grouped.items():
        rows: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for cell in cells:
            rows.setdefault(int(cell["row_no"]), {})[int(cell["cell_no"])] = cell
        if len(rows) < 2:
            continue

        ordered_row_numbers = sorted(rows)
        header_row = rows[ordered_row_numbers[0]]
        headers = [
            _normalize_text(header_row[cell_no].get("raw_text", ""))
            for cell_no in sorted(header_row)
            if _normalize_text(header_row[cell_no].get("raw_text", ""))
        ]
        if len(headers) < 2:
            continue

        header_cells = list(sorted(header_row))
        first_cell = cells[0]
        base_record = {
            "file_path": filename,
            "is_section": False,
            "section": first_cell.get("section", ""),
            "hwpx_section_index": first_cell.get("hwpx_section_index", 0),
            "hwpx_path": first_cell.get("hwpx_path", ""),
            "is_nested": True,
            "is_table": True,
            "is_table_summary": True,
        }
        if structure_v2:
            base_record.update(
                {
                    "table_id": _table_id,
                    "heading_path": list(first_cell.get("heading_path", []) or []),
                    "is_derived": True,
                }
            )
        line_start = min(int(cell.get("line_start", 0) or 0) for cell in cells)
        line_end = max(int(cell.get("line_end", 0) or 0) for cell in cells)
        hint_records.append(
            {
                **base_record,
                "text": f"표헤더: {' | '.join(headers)}",
                "line_start": line_start,
                "line_end": line_start,
                "hwpx_paragraph_index": first_cell.get("hwpx_paragraph_index", 0),
                **({"chunk_kind": "table_header", "row_no": ordered_row_numbers[0], "cell_no": 0, "parent_chunk_key": f"hwpx:{_section_index}:{_table_id}:row:{ordered_row_numbers[0]}"} if structure_v2 else {}),
            }
        )

        for row_no in ordered_row_numbers[1:]:
            row = rows[row_no]
            raw_values = [
                _normalize_text(row[cell_no].get("raw_text", ""))
                for cell_no in sorted(row)
                if _normalize_text(row[cell_no].get("raw_text", ""))
            ]
            if len(raw_values) < 2:
                continue
            raw_row_text = " ".join(raw_values)
            pairs: List[str] = []
            value_lines: List[str] = []
            row_aliases: List[str] = []
            for index, header_cell_no in enumerate(header_cells):
                if index >= len(headers):
                    continue
                value_cell = row.get(header_cell_no)
                if not value_cell:
                    continue
                header = headers[index]
                value = _format_table_value(header, str(value_cell.get("raw_text", "") or ""), raw_row_text)
                if not value:
                    continue
                pairs.append(f"{header}={value}")
                aliases = _name_alias_text(value, header=header)
                if aliases and is_name_header(header):
                    row_aliases.extend(generate_name_aliases(value))
                if PRICE_HEADER_PATTERN.search(header):
                    survey_name = next((candidate for candidate in raw_values if "조사" in candidate), "")
                    if survey_name:
                        value_lines.append(f"표값: {survey_name} {header} {value}")
            if row_aliases:
                pairs.append(f"명칭별칭={', '.join(_dedupe_aliases(row_aliases))}")

            row_cells = list(row.values())
            row_start = min(int(cell.get("line_start", 0) or 0) for cell in row_cells)
            row_end = max(int(cell.get("line_end", 0) or 0) for cell in row_cells)
            hint_records.append(
                {
                    **base_record,
                    "text": f"표행요약: {' | '.join(pairs)}",
                    "line_start": row_start,
                    "line_end": row_end,
                    "hwpx_paragraph_index": min(int(cell.get("hwpx_paragraph_index", 0) or 0) for cell in row_cells),
                    **({"chunk_kind": "table_summary", "row_no": row_no, "cell_no": 0, "parent_chunk_key": f"hwpx:{_section_index}:{_table_id}:row:{row_no}"} if structure_v2 else {}),
                }
            )
            hint_records.append(
                {
                    **base_record,
                    "text": build_table_row_fact_text(headers, raw_values),
                    "line_start": row_start,
                    "line_end": row_end,
                    "hwpx_paragraph_index": min(int(cell.get("hwpx_paragraph_index", 0) or 0) for cell in row_cells),
                    **({"chunk_kind": "table_summary", "row_no": row_no, "cell_no": 0, "parent_chunk_key": f"hwpx:{_section_index}:{_table_id}:row:{row_no}"} if structure_v2 else {}),
                }
            )
            for value_line in value_lines:
                hint_records.append(
                    {
                        **base_record,
                        "text": value_line,
                        "line_start": row_start,
                        "line_end": row_end,
                        "hwpx_paragraph_index": min(int(cell.get("hwpx_paragraph_index", 0) or 0) for cell in row_cells),
                        **({"chunk_kind": "table_summary", "row_no": row_no, "cell_no": 0, "parent_chunk_key": f"hwpx:{_section_index}:{_table_id}:row:{row_no}"} if structure_v2 else {}),
                    }
                )

    hint_records.extend(_build_flattened_survey_hint_records(table_cells, filename=filename, structure_v2=structure_v2))
    return hint_records


def load_hwpx_records(
    file_path: str,
    *,
    include_tables: bool = True,
    structure_v2: bool = False,
) -> List[Dict[str, Any]]:
    """Load HWPX paragraphs while preserving section and nested-table hints."""
    try:
        from hwpx import TextExtractor
    except Exception as exc:
        raise RuntimeError(
            "HWPX 문서를 읽으려면 python-hwpx 패키지가 필요합니다. "
            "오프라인 번들에 python_hwpx wheel을 포함한 뒤 backend를 다시 설치해 주세요."
        ) from exc

    records: List[Dict[str, Any]] = []
    table_cells: List[Dict[str, Any]] = []
    current_section_by_hwpx_section: Dict[int, str] = {}
    heading_stack_by_hwpx_section: Dict[int, List[str]] = {}
    filename = os.path.basename(file_path)

    with TextExtractor(str(file_path)) as extractor:
        for global_line_no, paragraph in enumerate(
            extractor.iter_document_paragraphs(include_nested=bool(include_tables)),
            start=1,
        ):
            raw_text = paragraph.text() if callable(getattr(paragraph, "text", None)) else ""
            text = _normalize_text(raw_text)
            if not text:
                continue

            section_index = int(getattr(getattr(paragraph, "section", None), "index", 0) or 0)
            paragraph_index = int(getattr(paragraph, "index", global_line_no - 1) or 0)
            path = str(getattr(paragraph, "path", "") or "")
            is_nested = bool(getattr(paragraph, "is_nested", False))
            is_table = is_nested or _is_table_path(path)
            if is_table and not include_tables:
                continue

            style_heading_level = _paragraph_style_heading_level(paragraph) if structure_v2 else 0
            is_section = bool(style_heading_level) or _is_section_text(text)
            if is_section:
                current_section_by_hwpx_section[section_index] = text
                if structure_v2:
                    level = style_heading_level or _heading_level(text)
                    stack = list(heading_stack_by_hwpx_section.get(section_index, []))
                    stack = stack[: max(0, level - 1)]
                    stack.append(text)
                    heading_stack_by_hwpx_section[section_index] = stack
            section = current_section_by_hwpx_section.get(section_index) or f"HWPX section {section_index + 1}"
            indexed_text = f"[표] {text}" if is_table and not text.startswith("[표]") else text

            record = {
                "text": indexed_text,
                "raw_text": text,
                "line_start": global_line_no,
                "line_end": global_line_no,
                "file_path": filename,
                "is_section": is_section,
                "section": section,
                "hwpx_section_index": section_index,
                "hwpx_paragraph_index": paragraph_index,
                "hwpx_path": path,
                "is_nested": is_nested,
                "is_table": is_table,
            }
            if structure_v2:
                table_id, row_no, cell_no = _table_path_indexes(path) if is_table else ("", 0, 0)
                parent_key = (
                    f"hwpx:{section_index}:{table_id}:row:{row_no}"
                    if is_table
                    else f"hwpx:{section_index}:paragraph:{paragraph_index}"
                )
                record.update(
                    {
                        "heading_path": list(heading_stack_by_hwpx_section.get(section_index, [])),
                        "chunk_kind": _classify_paragraph_kind(text, is_heading=is_section, is_table=is_table),
                        "parent_chunk_key": parent_key,
                        "structure_path": path,
                        "table_id": table_id,
                        "row_no": row_no,
                        "cell_no": cell_no,
                        "is_derived": False,
                        "heading_source": "style" if style_heading_level else ("regex" if is_section else ""),
                    }
                )
                record = normalize_structure_record(record, source_type="hwpx", doc_role="")
            records.append(record)
            if is_table:
                table_cells.append(record)

    hint_records = _build_hwpx_table_hint_records(table_cells, filename=filename, structure_v2=structure_v2)
    if structure_v2:
        hint_records = [normalize_structure_record(item, source_type="hwpx", doc_role="") for item in hint_records]
    records.extend(hint_records)
    return records
