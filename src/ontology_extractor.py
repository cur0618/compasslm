from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.table_facts import parse_table_fact_line


RESERVED_TABLE_FACT_KEYS = {"kind", "subject", "aliases"}
SUPPORTED_FACT_KINDS = {"table_row", "definition_block"}
STRUCTURE_PREDICATES = {
    "definition": "정의",
    "condition": "조건",
    "exception": "예외",
}


def normalize_entity_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_entity_key(value: str) -> str:
    return re.sub(r"\s+", "", normalize_entity_text(value).lower())


def split_aliases(value: str) -> List[str]:
    aliases: List[str] = []
    seen: set[str] = set()
    for raw in str(value or "").split(","):
        alias = normalize_entity_text(raw)
        key = normalize_entity_key(alias)
        if not alias or key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def extract_deterministic_facts_from_text(
    text: str,
    *,
    chunk_kind: str = "",
    heading_path: Any = None,
    is_derived: bool = False,
) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    normalized_text = normalize_entity_text(text)
    structure_kind = normalize_entity_text(chunk_kind).lower()
    structure_predicate = STRUCTURE_PREDICATES.get(structure_kind, "")
    headings = heading_path if isinstance(heading_path, list) else []
    headings = [normalize_entity_text(item) for item in headings if normalize_entity_text(item)]
    if structure_predicate and normalized_text and headings and not bool(is_derived):
        subject = headings[-1]
        facts.append(
            {
                "subject": subject,
                "subject_aliases": [subject],
                "predicate": structure_predicate,
                "object_value": normalized_text,
                "object_aliases": [],
                "object_entity": "",
                "fact_kind": structure_kind,
                "extraction_method": "deterministic_structure_fact",
                "confidence": 0.82,
                "evidence_quote": normalized_text,
            }
        )
    for raw_line in str(text or "").splitlines():
        line = normalize_entity_text(raw_line)
        if not line.startswith("표의미:"):
            continue
        parsed = parse_table_fact_line(line)
        kind = parsed.get("kind", "")
        subject = normalize_entity_text(parsed.get("subject", ""))
        if kind not in SUPPORTED_FACT_KINDS or not subject:
            continue
        parent_subject = ""
        if kind == "definition_block" and " > " in subject:
            parent_subject, child_subject = [normalize_entity_text(part) for part in subject.split(" > ", 1)]
            if child_subject:
                subject = child_subject
        aliases = split_aliases(parsed.get("aliases", ""))
        if subject and subject not in aliases:
            aliases.insert(0, subject)
        for key, value in parsed.items():
            predicate = normalize_entity_text(key)
            object_value = normalize_entity_text(value)
            if predicate in RESERVED_TABLE_FACT_KEYS or not object_value:
                continue
            facts.append(
                {
                    "subject": subject,
                    "subject_aliases": aliases,
                    "predicate": predicate,
                    "object_value": object_value,
                    "object_aliases": [],
                    "object_entity": "",
                    "fact_kind": kind,
                    "extraction_method": "deterministic_table_fact",
                    "confidence": 0.78,
                }
            )
        if kind == "definition_block" and parent_subject:
            parent = parent_subject
            child = subject
            if parent and child:
                facts.append(
                    {
                        "subject": child,
                        "subject_aliases": [child],
                        "predicate": "상위개념",
                        "object_value": parent,
                        "object_aliases": [parent],
                        "object_entity": parent,
                        "fact_kind": "hierarchy",
                        "extraction_method": "deterministic_definition_path",
                        "confidence": 0.72,
                    }
                )
    return facts


def validate_limited_llm_fact_payload(payload: Any) -> List[Dict[str, Any]]:
    """Validate a restricted JSON-like fact payload from a local LLM.

    This intentionally accepts only facts with explicit provenance-ready fields.
    The caller is responsible for attaching chunk/page/line provenance.
    """
    if not isinstance(payload, list):
        return []
    facts: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        subject = normalize_entity_text(str(item.get("subject", "") or ""))
        predicate = normalize_entity_text(str(item.get("predicate", "") or ""))
        object_value = normalize_entity_text(str(item.get("object", item.get("object_value", "")) or ""))
        evidence_quote = normalize_entity_text(str(item.get("evidence_quote", "") or ""))
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if not subject or not predicate or not object_value or confidence < 0.50:
            continue
        facts.append(
            {
                "subject": subject,
                "subject_aliases": split_aliases(str(item.get("subject_aliases", "") or "")) or [subject],
                "predicate": predicate,
                "object_value": object_value,
                "object_aliases": split_aliases(str(item.get("object_aliases", "") or "")),
                "object_entity": normalize_entity_text(str(item.get("object_entity", "") or "")),
                "fact_kind": normalize_entity_text(str(item.get("fact_kind", "llm_relation") or "llm_relation")),
                "extraction_method": "limited_llm",
                "confidence": min(0.95, max(0.0, confidence)),
                "evidence_quote": evidence_quote,
            }
        )
    return facts


def parse_limited_llm_fact_response(text: str) -> List[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
    return validate_limited_llm_fact_payload(payload)


def _quote_in_chunk(evidence_quote: str, chunk_text: str) -> bool:
    quote_key = normalize_entity_key(evidence_quote)
    chunk_key = normalize_entity_key(chunk_text)
    return bool(quote_key and quote_key in chunk_key)


def coerce_limited_llm_facts_from_chunk(
    payload: Any,
    chunk_text: str,
    *,
    min_confidence: float = 0.62,
) -> List[Dict[str, Any]]:
    facts = validate_limited_llm_fact_payload(payload)
    out: List[Dict[str, Any]] = []
    for fact in facts:
        evidence_quote = normalize_entity_text(str(fact.get("evidence_quote", "") or ""))
        if not _quote_in_chunk(evidence_quote, chunk_text):
            continue
        confidence = float(fact.get("confidence", 0.0) or 0.0)
        out.append({
            **fact,
            "evidence_quote": evidence_quote,
            "status": "active" if confidence >= float(min_confidence) else "needs_review",
        })
    return out
