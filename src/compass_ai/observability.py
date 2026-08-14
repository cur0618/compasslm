from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping

from src.compass_ai.models import PhaseEventRecord


def trim_preview(value: str, limit: int = 500) -> str:
    text = " ".join((value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def run_usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}

    details = getattr(usage, "details", {}) or {}
    try:
        normalized_details = {str(k): int(v or 0) for k, v in dict(details).items()}
    except Exception:
        normalized_details = {}

    def _read_int(name: str) -> int:
        try:
            return int(getattr(usage, name, 0) or 0)
        except Exception:
            return 0

    return {
        "requests": _read_int("requests"),
        "tool_calls": _read_int("tool_calls"),
        "input_tokens": _read_int("input_tokens"),
        "cache_write_tokens": _read_int("cache_write_tokens"),
        "cache_read_tokens": _read_int("cache_read_tokens"),
        "output_tokens": _read_int("output_tokens"),
        "input_audio_tokens": _read_int("input_audio_tokens"),
        "cache_audio_read_tokens": _read_int("cache_audio_read_tokens"),
        "output_audio_tokens": _read_int("output_audio_tokens"),
        "details": normalized_details,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_phase_event(
    phase: str,
    event_name: str,
    *,
    status: str = "ok",
    detail: str = "",
    payload: Mapping[str, Any] | None = None,
) -> PhaseEventRecord:
    normalized_payload = dict(payload or {})
    return PhaseEventRecord(
        phase=(phase or "unknown").strip() or "unknown",
        event_name=(event_name or "event").strip() or "event",
        status=(status or "ok").strip() or "ok",
        detail=trim_preview(detail or "", 400),
        payload=normalized_payload,
        created_at_iso=utc_now_iso(),
    )


def phase_events_to_dicts(events: Iterable[PhaseEventRecord], limit: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = list(events)
    for event in rows[-max(1, int(limit or 1)) :]:
        out.append(
            {
                "phase": event.phase,
                "event_name": event.event_name,
                "status": event.status,
                "detail": event.detail,
                "payload": dict(event.payload or {}),
                "created_at_iso": event.created_at_iso,
            }
        )
    return out


def compact_chat_history_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    turn_limit: int = 8,
    char_limit: int = 1400,
) -> str:
    normalized: List[str] = []
    for row in rows:
        role = str((row or {}).get("role", "") or "").strip().lower()
        text = trim_preview(str((row or {}).get("text", "") or ""), 500)
        if not text:
            continue
        label = "USER" if role == "user" else "ASSISTANT"
        normalized.append(f"{label}: {text}")

    if not normalized:
        return ""

    recent = normalized[-max(1, int(turn_limit or 1)) :]
    if char_limit <= 0:
        return "[RECENT_CHAT_HISTORY]\n" + "\n".join(recent)

    kept: List[str] = []
    current_chars = 0
    for line in reversed(recent):
        remaining = max(48, char_limit - current_chars)
        clipped = trim_preview(line, remaining)
        addition = len(clipped) + (1 if kept else 0)
        if kept and current_chars + addition > char_limit:
            break
        kept.append(clipped)
        current_chars += addition
        if current_chars >= char_limit:
            break

    if not kept:
        return ""
    kept.reverse()
    return "[RECENT_CHAT_HISTORY]\n" + "\n".join(kept)
