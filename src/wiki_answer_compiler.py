import re
from typing import Any, Dict, List


def _plain(text: str) -> str:
    text = re.sub(r"\[\[CITATION:\d+\|[^\]]+\]\]", "", str(text or ""))
    return " ".join(text.split())


def _sentences(text: str) -> List[str]:
    plain = _plain(text)
    parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s*|(?<=요\.)\s*", plain)
    return [part.strip(" .") for part in parts if part and len(part.strip()) >= 8]


def _concept_name(question: str, answer: str) -> str:
    source = f"{question} {answer}"
    candidates = re.findall(r"[0-9A-Za-z가-힣]+(?:\s+[0-9A-Za-z가-힣]+){0,3}", source)
    for candidate in candidates:
        clean = candidate.strip()
        if any(marker in clean for marker in ("태양열", "태양광", "폐건물", "원두막", "농가경제조사")):
            return clean[:80]
    return (_plain(question) or _plain(answer) or "saved answer concept")[:80]


def compile_answer_memory(answer: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build deterministic structured memory candidates from a saved answer.

    This deliberately avoids LLM extraction. Candidates remain reviewable and
    must carry the saved answer's source refs before they are used downstream.
    """

    question = str(answer.get("question_text", "") or "")
    answer_text = str(answer.get("answer_text", "") or answer.get("answer_summary", "") or "")
    source_refs = answer.get("sources") or []
    status = "published" if answer.get("status") == "published" and source_refs else "needs_review"
    source_refs_json = answer.get("citation_json", "[]") or "[]"
    sentences = _sentences(answer_text)
    concept_name = _concept_name(question, answer_text)

    claims = [
        {
            "claim_text": sentence[:500],
            "normalized_claim": " ".join(sentence.lower().split())[:500],
            "source_refs_json": source_refs_json,
            "confidence_score": 0.7 if status == "published" else 0.35,
            "status": status,
        }
        for sentence in sentences[:5]
    ]
    if not claims and answer_text.strip():
        claims.append(
            {
                "claim_text": _plain(answer_text)[:500],
                "normalized_claim": _plain(answer_text).lower()[:500],
                "source_refs_json": source_refs_json,
                "confidence_score": 0.35,
                "status": "needs_review",
            }
        )

    procedure_markers = ("절차", "방법", "1.", "2.", "확인", "처리", "단계")
    procedures: List[Dict[str, Any]] = []
    if any(marker in answer_text for marker in procedure_markers):
        steps = sentences[:6] or [_plain(answer_text)[:300]]
        procedures.append(
            {
                "procedure_name": concept_name,
                "procedure_steps_json": steps,
                "conditions_json": [],
                "exceptions_json": [],
                "source_refs_json": source_refs_json,
                "status": status,
            }
        )

    table_markers = ("표", "sheet", "range", "열", "행", "기준", "분류", "단가")
    table_rules: List[Dict[str, Any]] = []
    if any(marker in answer_text for marker in table_markers):
        table_rules.append(
            {
                "rule_name": concept_name,
                "rule_text": _plain(answer_text)[:500],
                "source_path": str(source_refs[0].get("source_path", "") or "") if source_refs else "",
                "sheet_name": "",
                "table_range": "",
                "row_refs_json": [],
                "column_refs_json": [],
                "status": status,
            }
        )

    concepts = [
        {
            "concept_name": concept_name,
            "aliases_json": [concept_name],
            "description": (claims[0]["claim_text"] if claims else _plain(answer_text)[:220]),
            "related_sources_json": source_refs,
            "confidence_score": 0.7 if status == "published" else 0.35,
            "status": status,
        }
    ]

    return {
        "claims": claims,
        "concepts": concepts,
        "procedures": procedures,
        "table_rules": table_rules,
    }
