import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def derive_openai_base_url(api_url: str) -> str:
    raw = (api_url or "").strip()
    if not raw:
        return "http://127.0.0.1:8003/v1"

    parsed = urlsplit(raw)
    path = parsed.path or ""
    for suffix in ("/chat/completions", "/completions", "/responses"):
        path = _strip_suffix(path, suffix)
    if not path:
        path = "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


def normalize_provider_kind(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in {"", "openai", "openai_compatible", "openai_compat", "llama_cpp_openai", "vllm_openai"}:
        return "openai_compatible"
    return normalized or "openai_compatible"


def resolve_provider_label(base_url: str) -> str:
    explicit = (os.getenv("PYDANTIC_AI_PROVIDER_LABEL", "") or "").strip()
    if explicit:
        return explicit

    parsed = urlsplit((base_url or "").strip())
    netloc = (parsed.netloc or "").strip()
    if netloc:
        return netloc
    path = (parsed.path or "").strip().strip("/")
    if path:
        return path
    return "local-openai-compatible"


def normalize_history_strategy(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in {"", "compact", "compact_text", "summary"}:
        return "compact_text"
    if normalized in {"pydantic", "pydantic_messages", "raw_messages"}:
        return "pydantic_messages"
    return "compact_text"


def resolve_model_name() -> str:
    for env_name in ("PYDANTIC_AI_MODEL_NAME", "LLM_MODEL_NAME"):
        value = (os.getenv(env_name, "") or "").strip()
        if value:
            return value

    model_path = (os.getenv("LLM_MODEL_PATH", "") or "").strip()
    if model_path:
        basename = Path(model_path).name
        stem = Path(basename).stem.strip()
        if stem:
            return stem

    return "qwen3.5-9b-q4_k_m"


def resolve_api_key() -> str:
    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        value = (os.getenv(env_name, "") or "").strip()
        if value:
            return value
    return "compasslm-local-key"


@dataclass(frozen=True)
class CompassAISettings:
    api_url: str
    base_url: str
    provider_kind: str
    provider_label: str
    model_name: str
    api_key: str
    output_retries: int
    request_limit: int
    tool_calls_limit: int
    enable_retrieval_tool: bool
    tool_timeout_seconds: float
    max_concurrency: int
    history_strategy: str
    compact_history_turn_limit: int
    compact_history_char_limit: int
    enable_phase_events: bool

    @classmethod
    def from_env(cls) -> "CompassAISettings":
        api_url = (os.getenv("LLM_API_URL", "http://127.0.0.1:8003/v1/chat/completions") or "").strip()
        quality_retry_enabled = _env_bool("LLM_QUALITY_RETRY_ENABLED", True)
        base_url = ((os.getenv("LLM_OPENAI_BASE_URL", "") or "").strip() or derive_openai_base_url(api_url))
        return cls(
            api_url=api_url,
            base_url=base_url,
            provider_kind=normalize_provider_kind(os.getenv("PYDANTIC_AI_PROVIDER_KIND", "openai_compatible")),
            provider_label=resolve_provider_label(base_url),
            model_name=resolve_model_name(),
            api_key=resolve_api_key(),
            output_retries=(
                max(0, int(os.getenv("LLM_QUALITY_MAX_RETRY", "2")))
                if quality_retry_enabled
                else 0
            ),
            request_limit=max(3, int(os.getenv("PYDANTIC_AI_REQUEST_LIMIT", "6"))),
            tool_calls_limit=max(0, int(os.getenv("PYDANTIC_AI_TOOL_CALLS_LIMIT", "8"))),
            enable_retrieval_tool=_env_bool("PYDANTIC_AI_ENABLE_RETRIEVAL_TOOL", True),
            tool_timeout_seconds=max(5.0, float(os.getenv("PYDANTIC_AI_TOOL_TIMEOUT_SECONDS", "20"))),
            max_concurrency=max(1, int(os.getenv("PYDANTIC_AI_MAX_CONCURRENCY", "1"))),
            history_strategy=normalize_history_strategy(os.getenv("PYDANTIC_AI_HISTORY_STRATEGY", "compact_text")),
            compact_history_turn_limit=max(2, int(os.getenv("PYDANTIC_AI_COMPACT_HISTORY_TURNS", "10"))),
            compact_history_char_limit=max(240, int(os.getenv("PYDANTIC_AI_COMPACT_HISTORY_CHARS", "1400"))),
            enable_phase_events=_env_bool("PYDANTIC_AI_ENABLE_PHASE_EVENTS", True),
        )
