from typing import Any


SMALL_TALK_HINTS = (
    "고마워",
    "감사",
    "땡큐",
    "알겠",
    "오케이",
    "ok",
    "ㅇㅋ",
    "좋아",
    "응",
    "네",
    "안녕",
    "반가워",
)

CORRECTION_HINTS = (
    "아니라",
    "말고",
    "정정",
    "수정",
    "오타",
    "대신",
    "였어",
    "였고",
    "이아니라",
    "그게아니라",
)

CONTINUATION_HINTS = (
    "그럼",
    "그러면",
    "그건",
    "그게",
    "그거",
    "이건",
    "이게",
    "이거",
    "저건",
    "저게",
    "저거",
    "추가로",
    "그리고",
)


def _compact_text(text: str) -> str:
    return "".join((text or "").strip().lower().split())


def is_small_talk_text(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if len(compact) > 24:
        return False
    if any(symbol in compact for symbol in ("?", "처리", "방법", "기준", "설치", "요건", "서류")):
        return False
    return any(hint in compact for hint in SMALL_TALK_HINTS)


def should_attempt_followup_rewrite(text: str) -> bool:
    compact = _compact_text(text)
    if not compact or is_small_talk_text(text):
        return False
    if len(compact) > 80:
        return False
    if any(hint in compact for hint in CORRECTION_HINTS):
        return True
    return len(compact) <= 40 and any(hint in compact for hint in CONTINUATION_HINTS)


def resolve_effective_query(user_message: str, followup_analysis: Any) -> str:
    original = " ".join((user_message or "").split())
    if not original or followup_analysis is None:
        return original

    followup_type = str(getattr(followup_analysis, "followup_type", "standalone") or "standalone").strip().lower()
    rewritten_query = " ".join(str(getattr(followup_analysis, "rewritten_query", "") or "").split())
    should_use_history = bool(getattr(followup_analysis, "should_use_history", False))
    is_small_talk = bool(getattr(followup_analysis, "is_small_talk", False))

    if is_small_talk:
        return original
    if followup_type not in {"correction", "continuation"}:
        return original
    if not should_use_history or not rewritten_query:
        return original
    return rewritten_query
