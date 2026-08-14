import json
import re
from typing import Any, Dict, List, Mapping

from src.wiki_store import WikiStore


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _clean_slug_part(value: str) -> str:
    raw = str(value or "").strip()
    safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "-", raw).strip("-._")
    return safe or "page"


def _normalize_sources(sources: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, Mapping):
            continue
        normalized.append(
            {
                "source_path": str(source.get("source_path", "") or ""),
                "source_ref": str(source.get("source_ref", "") or source.get("citation_label", "") or source.get("label", "") or ""),
                "page_no": int(source.get("page_no", 0) or 0),
                "chunk_id": int(source.get("chunk_id", 0) or 0),
                "table_cell_id": int(source.get("table_cell_id", 0) or 0),
            }
        )
    return normalized


def _unique_nonempty(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_wiki_page_payload(
    *,
    page_type: str,
    title: str,
    body: str,
    claims: List[Dict[str, Any]] | None,
    sources: List[Dict[str, Any]] | None,
    provenance: Mapping[str, Any] | None = None,
    status: str = "",
) -> Dict[str, Any]:
    safe_type = str(page_type or "concept").strip() or "concept"
    safe_title = str(title or safe_type).strip()[:200] or safe_type
    safe_sources = _normalize_sources(sources or [])
    safe_claims = [dict(claim) for claim in claims or [] if isinstance(claim, Mapping)]
    default_status = "draft" if safe_sources else "needs_review"
    safe_status = str(status or default_status).strip() or default_status
    base_slug = {
        "concept": "concepts",
        "procedure": "procedures",
        "table_rule": "table_rules",
        "source_summary": "sources",
    }.get(safe_type, "pages")
    return {
        "slug": f"{base_slug}/{_clean_slug_part(safe_title)}",
        "title": safe_title,
        "page_type": safe_type,
        "body": str(body or "").strip() or "검토 가능한 wiki page 초안입니다.",
        "status": safe_status,
        "claims": safe_claims,
        "sources": safe_sources,
        "source_count": len(safe_sources),
        "claim_count": len(safe_claims),
        "provenance": dict(provenance or {}),
    }


class WikiPageBuilder:
    def __init__(self, wiki_store: WikiStore):
        self.wiki_store = wiki_store

    def persist_page(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        page = self.wiki_store.upsert_page(
            slug=str(payload.get("slug", "") or ""),
            title=str(payload.get("title", "") or ""),
            page_type=str(payload.get("page_type", "") or "concept"),
            body=str(payload.get("body", "") or ""),
            status=str(payload.get("status", "") or "draft"),
            metadata={
                "source_count": int(payload.get("source_count", 0) or 0),
                "claim_count": int(payload.get("claim_count", 0) or 0),
                "quality": str(payload.get("status", "") or "draft"),
                "source_paths": _unique_nonempty(
                    [source.get("source_path", "") for source in list(payload.get("sources", []) or []) if isinstance(source, Mapping)]
                ),
                "space_id": str((payload.get("provenance", {}) or {}).get("space_id", "") or ""),
            },
            provenance=dict(payload.get("provenance", {}) or {}),
        )
        page_id = int(page["page_id"])
        self.wiki_store.replace_page_sources(page_id=page_id, sources=list(payload.get("sources", []) or []))
        self.wiki_store.replace_page_claims(page_id=page_id, claims=list(payload.get("claims", []) or []))
        return self.wiki_store.get_page(str(page["slug"])) or page

    def build_pages(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.persist_page(candidate) for candidate in candidates]
