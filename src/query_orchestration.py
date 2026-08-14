import asyncio
import re
from typing import Awaitable, Callable, Tuple, TypeVar


T = TypeVar("T")
U = TypeVar("U")

OVERVIEW_HINTS = ("요약", "정리", "전체", "흐름", "개요", "비교", "차이", "outline", "summary")
MULTI_CONDITION_HINTS = ("그리고", "또는", "예외", "조건", "서류", "기준", "절차", "처리", ",")


async def run_parallel_helper_tasks(
    user_message: str,
    analyze_fn: Callable[[str], Awaitable[T]],
    expand_fn: Callable[[str], Awaitable[U]],
) -> Tuple[T, U]:
    analysis_result, expand_result = await asyncio.gather(
        analyze_fn(user_message),
        expand_fn(user_message),
        return_exceptions=True,
    )
    return analysis_result, expand_result


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def decide_rerank_usage(query: str, *, candidate_count: int) -> tuple[bool, str]:
    compact = _compact_text(query)
    if candidate_count <= 0:
        return False, "no_candidates"

    overview_hits = sum(1 for hint in OVERVIEW_HINTS if hint in compact)
    multi_condition_hits = sum(1 for hint in MULTI_CONDITION_HINTS if hint in compact)
    is_long_query = len(compact) >= 28
    has_comparison = any(hint in compact for hint in ("비교", "차이", "정리", "요약"))

    if (has_comparison and multi_condition_hits >= 2) or (overview_hits >= 1 and multi_condition_hits >= 2):
        return True, "complex_multi_condition"
    if candidate_count >= 16:
        return True, "large_candidate_pool"
    if overview_hits >= 1 and (is_long_query or candidate_count >= 10):
        return True, "overview_request"
    if multi_condition_hits >= 3 and is_long_query:
        return True, "complex_multi_condition"
    return False, "simple_direct_question"
