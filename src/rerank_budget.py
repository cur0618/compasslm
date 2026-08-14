import math
import re
from typing import Any, Dict, List, Tuple


ASCII_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
KOREAN_CHAR_PATTERN = re.compile(r"[가-힣]")
SYMBOL_PATTERN = re.compile(r"[^\w\s가-힣]")

RERANK_PROMPT_HEADER = (
    "질문에 가장 관련 높은 순서로 후보 번호만 출력해줘.\n"
    "- 질문의 숫자/코드/고유명사와 정확히 맞는 후보를 우선\n"
    "- 같은 내용이면 uploaded가 최신인 후보를 우선\n"
    "- norm=0(원문) 후보를 norm=1(통합정리)보다 우선\n"
    "- 설명 금지\n"
    "출력 형식: 3,1,5,2\n"
    "번호 외 텍스트 금지.\n\n"
)
RERANK_MIN_LINE_CHAR_CAP = 96


def estimate_mixed_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_words = len(ASCII_WORD_PATTERN.findall(text))
    ko_chars = len(KOREAN_CHAR_PATTERN.findall(text))
    symbols = len(SYMBOL_PATTERN.findall(text))
    est = (ascii_words * 1.1) + (ko_chars * 0.65) + (symbols * 0.3)
    return max(1, int(math.ceil(est)))


def build_rerank_prompt(user_message: str, candidate_lines: List[str]) -> str:
    return (
        f"{RERANK_PROMPT_HEADER}"
        f"질문: {user_message}\n\n"
        "후보:\n"
        + "\n".join(candidate_lines)
    )


def _clip_line(line: str, char_cap: int) -> str:
    if char_cap <= 0:
        return ""
    if len(line) <= char_cap:
        return line
    suffix = "..."
    keep = max(1, char_cap - len(suffix))
    return line[:keep] + suffix


def _fit_prefix_count(
    user_message: str,
    clipped_lines: List[str],
    *,
    prompt_token_budget: int,
) -> int:
    count = 0
    for idx in range(1, len(clipped_lines) + 1):
        prompt = build_rerank_prompt(user_message, clipped_lines[:idx])
        if estimate_mixed_tokens(prompt) > prompt_token_budget:
            break
        count = idx
    return count


def _build_line_char_caps(candidate_lines: List[str]) -> List[int]:
    if not candidate_lines:
        return [RERANK_MIN_LINE_CHAR_CAP]

    original_cap = max(len(line) for line in candidate_lines)
    caps = {
        original_cap,
        640,
        560,
        480,
        420,
        360,
        320,
        280,
        240,
        220,
        200,
        180,
        160,
        140,
        120,
        RERANK_MIN_LINE_CHAR_CAP,
    }
    for ratio in (0.85, 0.7, 0.6, 0.5, 0.4, 0.33, 0.28, 0.24, 0.2):
        caps.add(max(RERANK_MIN_LINE_CHAR_CAP, int(original_cap * ratio)))
    return sorted((cap for cap in caps if cap > 0), reverse=True)


def trim_rerank_candidate_lines_to_budget(
    *,
    user_message: str,
    candidate_lines: List[str],
    keep_n: int,
    llm_context_limit: int,
    helper_max_tokens: int,
    prompt_overhead_tokens: int,
    safety_margin: int,
) -> Tuple[List[str], Dict[str, Any]]:
    desired_keep = max(1, min(keep_n, len(candidate_lines)))
    prompt_token_budget = max(
        64,
        llm_context_limit - helper_max_tokens - prompt_overhead_tokens - safety_margin,
    )
    base_prompt_tokens = estimate_mixed_tokens(build_rerank_prompt(user_message, []))
    meta: Dict[str, Any] = {
        "original_count": len(candidate_lines),
        "selected_count": 0,
        "trimmed_count": 0,
        "line_char_cap": 0,
        "prompt_token_budget": prompt_token_budget,
        "base_prompt_tokens": base_prompt_tokens,
    }

    if not candidate_lines:
        return [], meta

    best_lines: List[str] = []
    best_meta: Dict[str, Any] = {}

    for char_cap in _build_line_char_caps(candidate_lines):
        clipped_lines = [_clip_line(line, char_cap) for line in candidate_lines]
        selected_count = _fit_prefix_count(
            user_message,
            clipped_lines,
            prompt_token_budget=prompt_token_budget,
        )
        if selected_count <= 0:
            continue

        candidate_meta = {
            "selected_count": selected_count,
            "trimmed_count": max(0, len(candidate_lines) - selected_count),
            "line_char_cap": char_cap,
        }
        if selected_count >= desired_keep:
            best_lines = clipped_lines[:selected_count]
            best_meta = candidate_meta
            break
        if not best_lines or selected_count > best_meta.get("selected_count", 0):
            best_lines = clipped_lines[:selected_count]
            best_meta = candidate_meta

    if not best_lines and desired_keep > 0:
        fallback_cap = min(max(len(line) for line in candidate_lines), RERANK_MIN_LINE_CHAR_CAP)
        best_lines = [_clip_line(line, fallback_cap) for line in candidate_lines[:desired_keep]]
        best_meta = {
            "selected_count": len(best_lines),
            "trimmed_count": max(0, len(candidate_lines) - len(best_lines)),
            "line_char_cap": fallback_cap,
        }

    meta.update(best_meta)
    return best_lines, meta
