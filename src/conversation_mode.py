import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.query_rewrite import is_small_talk_text


ACTIVE_CONVERSATION_MODES = {"casual_chat", "document_qa"}
NON_ANCHOR_CONVERSATION_MODES = {"history_control", "identity"}

DOCUMENT_INTENT_HINTS = (
    "문서",
    "업로드",
    "파일",
    "근거",
    "페이지",
    "시트",
    "행",
    "라인",
    "요약",
    "정리",
    "규정",
    "기준",
    "요건",
    "절차",
    "단가",
    "금액",
    "비율",
    "건수",
    "코드",
    "답례품",
    "지원대상",
    "신청",
    "주의사항",
    "유의사항",
)

CASUAL_EXTRA_HINTS = (
    "오늘어때",
    "뭐해",
    "뭐하고있",
    "잘지내",
    "심심",
    "재밌는이야기",
    "농담",
    "기분어때",
    "배고파",
    "졸려",
    "피곤",
)

CASUAL_PATTERN_GROUPS = (
    (r"(오늘|요즘|지금).*(어때|어떠)"),
    (r"(뭐해|뭐하고있|뭐 하고 있)"),
    (r"(재밌는|재미있는).*(이야기|얘기)"),
    (r"(농담|심심)"),
)

LIVE_INFO_HINTS = (
    "날씨",
    "기온",
    "강수",
    "비와",
    "비오",
    "눈와",
    "눈오",
    "뉴스",
    "속보",
    "주가",
    "코스피",
    "코스닥",
    "환율",
    "비트코인",
    "btc",
    "이더리움",
    "eth",
    "실시간",
)

SHORT_FOLLOWUP_QUESTION_HINTS = (
    "얼마",
    "언제",
    "왜",
    "몇",
    "어떻게",
    "어느",
)

CONTEXTUAL_FOLLOWUP_HINTS = (
    "그건",
    "그게",
    "그거",
    "이건",
    "이게",
    "이거",
    "저건",
    "저게",
    "저거",
    "그럼",
    "그러면",
    "그리고",
    "추가로",
    "위에서",
    "앞에서",
    "방금",
    "그중",
    "그 중",
    "일부만",
    "이 경우",
    "그 경우",
    "저 경우",
    "경우는",
    "적용되는 경우",
    "제외하면",
    "빼면",
    "왜",
)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(str(raw or ""))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


@dataclass
class ConversationModeDecision:
    mode: str
    reason: str


@dataclass
class RecentConversationState:
    last_active_mode: Optional[str] = None
    mode_anchor_run_id: int = 0
    casual_turn_streak: int = 0


def is_contextual_followup_message(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if len(compact) > 48 and "위에서" not in compact and "일부만" not in compact:
        return False
    return any(hint.replace(" ", "") in compact for hint in CONTEXTUAL_FOLLOWUP_HINTS)


def is_explicit_document_message(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if re.search(r"\d+\s*(페이지|쪽|행|라인|시트)", compact):
        return True
    if any(hint in compact for hint in ("[doc", "pdf", "xlsx", "txt", "kb")):
        return True
    return any(hint.replace(" ", "") in compact for hint in DOCUMENT_INTENT_HINTS)


def is_explicit_casual_message(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if is_small_talk_text(text):
        return True
    if any(re.search(pattern, compact) for pattern in CASUAL_PATTERN_GROUPS):
        return True
    return any(hint in compact for hint in CASUAL_EXTRA_HINTS)


def is_live_info_request(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if is_explicit_document_message(text):
        return False
    return any(hint in compact for hint in LIVE_INFO_HINTS)


def should_force_followup_rewrite(text: str, *, last_active_mode: Optional[str]) -> bool:
    if last_active_mode != "document_qa":
        return False
    if is_contextual_followup_message(text):
        return True
    compact = _compact_text(text)
    if len(compact) > 12 or "?" not in text:
        return False
    return any(hint in compact for hint in SHORT_FOLLOWUP_QUESTION_HINTS)


def summarize_recent_conversation_state(rows: Iterable[dict[str, Any]]) -> RecentConversationState:
    state = RecentConversationState()
    streak = 0

    for row in rows:
        metadata = _safe_json_loads(row.get("metadata_json", ""))
        mode = str(metadata.get("conversation_mode", "") or "").strip()
        if not mode:
            continue
        if state.last_active_mode is None and mode in ACTIVE_CONVERSATION_MODES:
            state.last_active_mode = mode
            state.mode_anchor_run_id = int(row.get("run_id", 0) or 0)
        if mode in NON_ANCHOR_CONVERSATION_MODES:
            continue
        if mode == "casual_chat":
            streak += 1
            continue
        break

    state.casual_turn_streak = streak
    return state


def resolve_conversation_mode(
    user_message: str,
    *,
    kb_has_docs: bool,
    last_active_mode: Optional[str],
    followup_type: str = "standalone",
    is_small_talk: bool = False,
) -> ConversationModeDecision:
    if is_explicit_document_message(user_message):
        return ConversationModeDecision(mode="document_qa", reason="explicit_document_intent")

    if is_live_info_request(user_message):
        return ConversationModeDecision(mode="casual_chat", reason="live_info_request")

    if is_explicit_casual_message(user_message) or is_small_talk:
        return ConversationModeDecision(mode="casual_chat", reason="explicit_casual_intent")

    if is_contextual_followup_message(user_message):
        if last_active_mode in ACTIVE_CONVERSATION_MODES:
            return ConversationModeDecision(mode=str(last_active_mode), reason="inherited_from_last_assistant_mode")
        return ConversationModeDecision(mode="casual_chat", reason="standalone_without_history_anchor")

    if followup_type in {"correction", "continuation"} and last_active_mode == "document_qa":
        return ConversationModeDecision(mode="document_qa", reason="inherited_from_last_assistant_mode")

    if kb_has_docs:
        return ConversationModeDecision(mode="document_qa", reason="default_document_when_kb_present")

    return ConversationModeDecision(mode="casual_chat", reason="default_casual_without_kb")
