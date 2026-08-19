import asyncio
import re
from typing import Any, Callable, Dict, List, Optional

import requests

from pydantic_ai import (
    Agent,
    InlineDefsJsonSchemaTransformer,
    ModelRetry,
    PromptedOutput,
    RunContext,
    ToolDefinition,
    UnexpectedModelBehavior,
    UsageLimits,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from src.compass_ai.models import (
    CompassAgentDeps,
    FollowupAnalysis,
    HelperRunDiagnostics,
    OpsReview,
    QuestionAnalysis,
    RetrievedDocRecord,
)
from src.answer_validation import (
    build_outline_recheck_debug_payload,
    build_tool_recheck_debug_payload,
    has_grounded_numeric_answer,
    is_grounded_abstention_text,
    is_numeric_evidence_query,
    repair_answer_text_format,
    should_require_outline_recheck,
    should_require_tool_recheck,
)
from src.citation_labels import canonicalize_doc_citations
from src.compass_ai.prompts import ANSWER_SYSTEM_INSTRUCTIONS, HELPER_BASE_INSTRUCTIONS
from src.compass_ai.observability import build_phase_event, trim_preview
from src.compass_ai.settings import CompassAISettings
from src.rerank_budget import build_rerank_prompt


class PydanticAIService:
    def __init__(self, settings: Optional[CompassAISettings] = None):
        self.settings = settings or CompassAISettings.from_env()
        model = self._build_model()

        self.model_name = self.settings.model_name
        self.api_url = self.settings.api_url
        self.base_url = self.settings.base_url
        self.provider_kind = self.settings.provider_kind
        self.provider_label = self.settings.provider_label
        self._helper_degraded_failures: Dict[str, int] = {}

        self._helper_agent = Agent(
            model,
            output_type=str,
            name="compass_helper",
            instructions=HELPER_BASE_INSTRUCTIONS,
        )
        self._analysis_agent = Agent(
            model,
            output_type=PromptedOutput(QuestionAnalysis),
            name="compass_question_analysis",
            instructions=HELPER_BASE_INSTRUCTIONS,
        )
        self._ops_review_agent = Agent(
            model,
            output_type=PromptedOutput(OpsReview),
            name="compass_ops_review",
            instructions=HELPER_BASE_INSTRUCTIONS,
        )
        self._answer_agent = Agent(
            model,
            output_type=str,
            deps_type=CompassAgentDeps,
            name="compass_answer",
            instructions=ANSWER_SYSTEM_INSTRUCTIONS,
            output_retries=self.settings.output_retries,
            prepare_tools=self._prepare_tools,
            tool_timeout=self.settings.tool_timeout_seconds,
            max_concurrency=self.settings.max_concurrency,
        )
        self._register_answer_agent_hooks()

    def _build_model(self) -> OpenAIChatModel:
        if self.settings.provider_kind != "openai_compatible":
            raise ValueError(f"Unsupported PYDANTIC_AI_PROVIDER_KIND: {self.settings.provider_kind}")

        provider = OpenAIProvider(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
        )
        profile = OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
        )
        return OpenAIChatModel(
            self.settings.model_name,
            provider=provider,
            profile=profile,
        )

    @staticmethod
    def _parse_bool_flag(value: str) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _helper_kind(default_failure_code: str) -> str:
        value = (default_failure_code or "helper_run_fail").strip()
        if value.endswith("_fail"):
            return value[: -len("_fail")]
        return value

    def _mark_helper_degraded(self, default_failure_code: str, failure_code: str = "") -> None:
        marker = (failure_code or "").strip()
        if not self._should_mark_helper_degraded(marker):
            return
        key = self._helper_kind(default_failure_code)
        self._helper_degraded_failures[key] = self._helper_degraded_failures.get(key, 0) + 1

    def _is_helper_degraded(self, default_failure_code: str) -> bool:
        return self._helper_degraded_failures.get(self._helper_kind(default_failure_code), 0) > 0

    @staticmethod
    def _should_mark_helper_degraded(failure_code_or_detail: str) -> bool:
        marker = (failure_code_or_detail or "").strip().lower()
        return any(
            token in marker
            for token in (
                "empty_output",
                "unexpected_model_behavior",
                "output_validation",
                "output validation",
                "exceeded maximum retries",
                "direct_fallback_fail",
            )
        )

    @staticmethod
    def _consume_degraded_helper_result(task: "asyncio.Task[tuple[str, HelperRunDiagnostics]]") -> None:
        try:
            task.result()
        except Exception:
            return

    async def _run_degraded_helper_with_deterministic(
        self,
        *,
        prompt: str,
        instructions: str,
        max_tokens: int,
        deterministic: Callable[[], Any],
        temperature: float = 0.0,
        timeout: int = 45,
        default_failure_code: str = "helper_run_fail",
    ) -> tuple[Any, HelperRunDiagnostics]:
        helper_task = asyncio.create_task(
            self.run_helper_diagnostic(
                prompt=prompt,
                instructions=instructions,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                default_failure_code=default_failure_code,
            )
        )
        helper_task.add_done_callback(self._consume_degraded_helper_result)
        value = deterministic()
        if asyncio.iscoroutine(value):
            value = await value
        return value, HelperRunDiagnostics(
            status="fallback",
            failure_code=f"{default_failure_code}_degraded_deterministic_parallel",
            error_type="HelperDegraded",
            error_detail="Helper previously returned empty output; used deterministic result while helper runs in parallel.",
            fallback_used=True,
            helper_degraded=True,
            deterministic_parallel_used=True,
            helper_wait_skipped=True,
        )

    @staticmethod
    def _split_helper_items(value: Any, limit: int = 4) -> List[str]:
        items: List[str] = []
        raw_values = value if isinstance(value, list) else [value]
        for entry in raw_values:
            for raw in str(entry or "").split("||"):
                cleaned = " ".join((raw or "").strip().split())
                if not cleaned or cleaned == "-":
                    continue
                if cleaned not in items:
                    items.append(cleaned)
                if len(items) >= max(1, limit):
                    return items
        return items

    def _normalize_question_analysis(self, payload: QuestionAnalysis | Dict[str, Any] | None) -> QuestionAnalysis:
        base = payload if isinstance(payload, QuestionAnalysis) else QuestionAnalysis.model_validate(payload or {})
        data = base.model_dump()
        data["intent_type"] = str(data.get("intent_type", "mixed") or "mixed").strip().lower() or "mixed"
        data["search_queries"] = self._split_helper_items(data.get("search_queries", []), limit=5)
        data["answer_focus"] = self._split_helper_items(data.get("answer_focus", []), limit=4)
        data["literal_first"] = self._parse_bool_flag(str(data.get("literal_first", False)))
        data["prefer_recent_sources"] = self._parse_bool_flag(str(data.get("prefer_recent_sources", False)))
        data["use_source_outline"] = self._parse_bool_flag(str(data.get("use_source_outline", False)))
        require_tool = data.get("require_tool_evidence", True)
        data["require_tool_evidence"] = self._parse_bool_flag(str(require_tool)) if not isinstance(require_tool, bool) else bool(require_tool)
        numeric_required = data.get("numeric_evidence_required", False)
        data["numeric_evidence_required"] = (
            self._parse_bool_flag(str(numeric_required))
            if not isinstance(numeric_required, bool)
            else bool(numeric_required)
        )
        return QuestionAnalysis.model_validate(data)

    def _normalize_followup_analysis(
        self,
        payload: FollowupAnalysis | Dict[str, Any] | None,
    ) -> FollowupAnalysis:
        base = payload if isinstance(payload, FollowupAnalysis) else FollowupAnalysis.model_validate(payload or {})
        data = base.model_dump()
        data["followup_type"] = (
            str(data.get("followup_type", "standalone") or "standalone").strip().lower() or "standalone"
        )
        data["rewritten_query"] = " ".join(str(data.get("rewritten_query", "") or "").split())
        data["should_use_history"] = self._parse_bool_flag(str(data.get("should_use_history", False)))
        data["is_small_talk"] = self._parse_bool_flag(str(data.get("is_small_talk", False)))
        return FollowupAnalysis.model_validate(data)

    def _parse_question_analysis(self, content: str) -> QuestionAnalysis:
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        values: Dict[str, str] = {}
        for line in lines:
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            values[key.strip().lower()] = raw_value.strip()

        return self._normalize_question_analysis(
            {
                "intent_type": (values.get("intent", "mixed") or "mixed").strip().lower(),
                "search_queries": self._split_helper_items(values.get("queries", "")),
                "answer_focus": self._split_helper_items(values.get("focus", "")),
                "literal_first": self._parse_bool_flag(values.get("literal_first", "")),
                "prefer_recent_sources": self._parse_bool_flag(values.get("prefer_recent", "")),
                "use_source_outline": self._parse_bool_flag(values.get("use_source_outline", values.get("outline_first", ""))),
                "require_tool_evidence": self._parse_bool_flag(values.get("require_tool_evidence", "yes")),
                "numeric_evidence_required": self._parse_bool_flag(values.get("numeric_evidence_required", "no")),
            }
        )

    def _deterministic_query_expansion(self, user_message: str) -> str:
        cleaned = " ".join(str(user_message or "").split())
        if not cleaned:
            return ""
        tokens = re.findall(r"[0-9A-Za-z가-힣_.%/()·ㆍ-]{2,}", cleaned)
        items: List[str] = []
        for value in [cleaned, *tokens]:
            if value and value not in items:
                items.append(value)
            if len(items) >= 12:
                break
        return " ".join(items)

    def _deterministic_question_analysis(self, user_message: str) -> QuestionAnalysis:
        cleaned = " ".join(str(user_message or "").split())
        lowered = cleaned.lower()
        intent = "mixed"
        if any(word in cleaned for word in ("요약", "정리", "핵심", "전체")):
            intent = "summary"
        elif any(word in cleaned for word in ("정의", "의미", "뭐야", "무엇")):
            intent = "definition"
        elif any(word in cleaned for word in ("방법", "절차", "어떻게")):
            intent = "procedure"
        elif any(word in cleaned for word in ("비교", "차이")):
            intent = "comparison"
        elif cleaned:
            intent = "fact"
        return self._normalize_question_analysis(
            {
                "intent_type": intent,
                "search_queries": [cleaned] if cleaned else [],
                "answer_focus": [cleaned] if cleaned else [],
                "literal_first": any(word in cleaned for word in ("요약", "정리", "전체", "핵심")),
                "prefer_recent_sources": any(word in cleaned for word in ("방금", "최근", "업로드")),
                "use_source_outline": len(cleaned) >= 90 or any(word in cleaned for word in ("전체", "흐름", "구조")),
                "require_tool_evidence": True,
                "numeric_evidence_required": is_numeric_evidence_query(lowered),
            }
        )

    def _parse_followup_analysis(self, content: str) -> FollowupAnalysis:
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        values: Dict[str, str] = {}
        for line in lines:
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            values[key.strip().lower()] = raw_value.strip()

        return self._normalize_followup_analysis(
            {
                "followup_type": values.get("followup_type", "standalone"),
                "rewritten_query": values.get("rewritten_query", ""),
                "should_use_history": values.get("should_use_history", "no"),
                "is_small_talk": values.get("is_small_talk", "no"),
            }
        )

    @staticmethod
    def _helper_failure_code(error: BaseException, default_failure_code: str) -> str:
        detail = str(error or "").strip().lower()
        if isinstance(error, UnexpectedModelBehavior):
            return f"{default_failure_code}_unexpected_model_behavior"
        if "timeout" in detail:
            return f"{default_failure_code}_timeout"
        if "connection" in detail or "connect" in detail:
            return f"{default_failure_code}_connection_fail"
        return default_failure_code

    def _build_helper_diagnostics(
        self,
        error: Optional[BaseException],
        *,
        default_failure_code: str,
        status: str = "error",
        fallback_used: bool = False,
    ) -> HelperRunDiagnostics:
        if error is None:
            return HelperRunDiagnostics(status=status, fallback_used=fallback_used)
        return HelperRunDiagnostics(
            status=status,
            failure_code=self._helper_failure_code(error, default_failure_code),
            error_type=type(error).__name__,
            error_detail=str(error or "").strip(),
            fallback_used=fallback_used,
        )

    @staticmethod
    def _should_use_direct_text_completion_fallback(error: BaseException) -> bool:
        detail = str(error or "").strip().lower()
        return isinstance(error, UnexpectedModelBehavior) and (
            "output validation" in detail
            or "exceeded maximum retries" in detail
        )

    def _run_openai_compatible_text_completion_sync(
        self,
        *,
        prompt: str,
        instructions: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
    ) -> str:
        response = requests.post(
            self.settings.api_url,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.model_name,
                "messages": [
                    {"role": "system", "content": instructions or HELPER_BASE_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
            timeout=max(1, int(timeout or 45)),
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content", first.get("text", ""))
        return str(content or "").strip()

    async def _run_openai_compatible_text_completion(
        self,
        *,
        prompt: str,
        instructions: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
    ) -> str:
        return await asyncio.to_thread(
            self._run_openai_compatible_text_completion_sync,
            prompt=prompt,
            instructions=instructions,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    def _register_answer_agent_hooks(self):
        agent = self._answer_agent

        def _append_validation_retry_event(
            deps: CompassAgentDeps,
            *,
            validator: str,
            detail: str,
            payload: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not self.settings.enable_phase_events:
                return
            body = {"validator": (validator or "").strip() or "unknown"}
            if payload:
                body.update(dict(payload))
            deps.run_state.phase_events.append(
                build_phase_event(
                    "output_validation",
                    "retry",
                    status="retry",
                    detail=detail,
                    payload=body,
                )
            )

        def _format_current_sources(deps: CompassAgentDeps) -> str:
            if not deps.run_state.docs:
                return "(아직 없음)"
            lines: List[str] = []
            for doc_no in sorted(deps.run_state.docs):
                record: RetrievedDocRecord = deps.run_state.docs[doc_no]
                lines.append(f"[DOC {doc_no}] {record.source_ref}")
            return "\n".join(lines)

        @agent.instructions
        def _runtime_context(ctx: RunContext[CompassAgentDeps]) -> str:
            deps = ctx.deps
            retrieval = deps.retrieval
            analysis = deps.question_analysis or QuestionAnalysis()
            lines = [
                "[RUNTIME]",
                f"- date={deps.runtime_date_iso}",
                f"- kb_name={deps.kb_name}",
                f"- query_id={deps.query_id}",
                f"- query_doc_intent={retrieval.query_doc_intent or 'mixed'}",
                f"- retrieval_role_filter={retrieval.retrieval_role_filter or 'all'}",
                f"- search_query={retrieval.search_query or deps.user_message}",
                f"- overview_mode={int(retrieval.overview_mode)}",
                f"- seeded_docs={len(deps.run_state.docs)}",
                f"- question_intent={analysis.intent_type}",
                f"- literal_first={int(bool(analysis.literal_first))}",
                f"- prefer_recent_sources={int(bool(analysis.prefer_recent_sources))}",
                f"- use_source_outline={int(bool(analysis.use_source_outline))}",
                f"- require_tool_evidence={int(bool(deps.require_tool_evidence))}",
                f"- numeric_evidence_required={int(bool(analysis.numeric_evidence_required))}",
            ]
            if analysis.answer_focus:
                lines.append(f"- answer_focus={' | '.join(analysis.answer_focus[:3])}")
            return "\n".join(lines)

        @agent.instructions
        def _retrieval_blocks(ctx: RunContext[CompassAgentDeps]) -> str:
            deps = ctx.deps
            retrieval = deps.retrieval
            retrieval_meta = retrieval.retrieval_meta or "RETRIEVAL_META:\n(없음)"
            policy_lines = [
                "- 현재 확보된 근거가 질문에 직접 대응하면 추가 도구 호출 없이 답한다.",
                "- SEEDED_CONTEXT와 CURRENT_SOURCES에 질문 문장 또는 같은 의미의 처리방법이 있으면 도구를 호출하지 말고 그 근거로 바로 답한다.",
                "- SEEDED_CONTEXT와 CURRENT_SOURCES를 먼저 읽고, 각 핵심 주장에 이미 확보된 [DOC n]를 연결한다.",
                "- 질문이 길거나 문서가 길어 보이면 `get_source_outline` 또는 `get_source_overview`로 구조를 먼저 확인한다.",
                "- 더 읽어야 할 문서가 있으면 `open_document` 또는 `get_source_overview`를 호출한다.",
                "- 도구가 꼭 필요해도 `search_knowledge_base`는 최대 1회, `open_document`는 최대 2개 문서까지만 사용한다.",
                "- 현재 확보된 문서 번호만 인용한다. 없는 [DOC n]를 만들면 안 된다.",
                "- 근거가 질문과 어긋날 때만 검색어를 바꿔 `search_knowledge_base`를 재호출한다.",
            ]
            if not deps.allow_retrieval_tool or deps.search_tool is None:
                policy_lines = [
                    "- 도구 호출 없이 현재 제공된 SEEDED_CONTEXT와 CURRENT_SOURCES만으로 답한다.",
                    "- 현재 확보된 근거가 질문에 직접 대응하면 추가 도구 호출 없이 답한다.",
                    "- 각 핵심 주장에 CURRENT_SOURCES의 [DOC n] 인용을 연결한다.",
                    "- 근거가 부족하면 도구를 호출하지 말고 확인된 범위만 말한다.",
                ]
            return (
                f"{retrieval_meta}\n\n"
                "SOURCE_HINTS:\n"
                f"{retrieval.source_hint or '(없음)'}\n\n"
                "REFERENCE_HINT:\n"
                f"{retrieval.reference_hint or '(없음)'}\n\n"
                "SEEDED_CONTEXT:\n"
                f"{retrieval.context or '(없음)'}\n\n"
                "CURRENT_SOURCES:\n"
                f"{_format_current_sources(deps)}\n\n"
                "[TOOL_POLICY]\n"
                f"{chr(10).join(policy_lines)}"
            )

        @agent.tool(
            sequential=True,
            retries=1,
            timeout=self.settings.tool_timeout_seconds,
        )
        def search_knowledge_base(
            ctx: RunContext[CompassAgentDeps],
            search_query: str,
            top_k: int = 8,
            doc_role: str = "auto",
        ) -> str:
            """Search the knowledge base and register stable [DOC n] citations for the current run."""
            deps = ctx.deps
            if not deps.allow_retrieval_tool or deps.search_tool is None:
                raise ModelRetry("검색 도구는 현재 비활성화되어 있다. 현재 확보한 근거만으로 답하라.")
            recent_events = deps.run_state.tool_events[max(0, int(deps.tool_event_baseline or 0)) :]
            search_count = sum(1 for event in recent_events if event.tool_name == "search_knowledge_base")
            if search_count >= 1:
                return (
                    "SEARCH_LIMIT_REACHED: search_knowledge_base는 이 답변에서 이미 1회 사용했다. "
                    "추가 검색을 반복하지 말고 CURRENT_SOURCES와 이미 확보한 [DOC n] 근거로 답하라.\n"
                    f"{_format_current_sources(deps)}"
                )

            query = (search_query or "").strip() or deps.user_message
            role_hint = (doc_role or "auto").strip().lower() or "auto"
            try:
                k = min(12, max(3, int(top_k)))
            except Exception as exc:
                raise ModelRetry("top_k는 3 이상 12 이하의 정수여야 한다.") from exc

            payload = deps.search_tool(query, k, role_hint)
            if not (payload or "").strip():
                raise ModelRetry("검색 결과가 비어 있다. 검색어를 더 구체화해서 다시 시도하라.")
            return payload

        @agent.tool(
            sequential=True,
            retries=1,
            timeout=self.settings.tool_timeout_seconds,
        )
        def open_document(ctx: RunContext[CompassAgentDeps], doc_no: int, max_chars: int = 1600) -> str:
            """Open one discovered document chunk with fuller grounded context."""
            deps = ctx.deps
            if not deps.allow_retrieval_tool or deps.open_document_tool is None:
                raise ModelRetry("문서 열람 도구는 현재 비활성화되어 있다.")
            try:
                target_doc_no = int(doc_no)
                limit = min(2600, max(400, int(max_chars)))
            except Exception as exc:
                raise ModelRetry("doc_no와 max_chars는 정수여야 한다.") from exc
            recent_events = deps.run_state.tool_events[max(0, int(deps.tool_event_baseline or 0)) :]
            opened_docs = [
                int((event.arguments or {}).get("doc_no", 0) or 0)
                for event in recent_events
                if event.tool_name == "open_document"
            ]
            if target_doc_no not in opened_docs and len([doc_no for doc_no in opened_docs if doc_no > 0]) >= 2:
                return (
                    "OPEN_DOCUMENT_LIMIT_REACHED: open_document는 이 답변에서 최대 2개 문서까지만 열 수 있다. "
                    "추가 문서를 열지 말고 이미 열람한 문서와 CURRENT_SOURCES의 [DOC n] 근거로 답하라.\n"
                    f"{_format_current_sources(deps)}"
                )
            payload = deps.open_document_tool(target_doc_no, limit)
            if not (payload or "").strip():
                raise ModelRetry("해당 문서를 열지 못했다. 먼저 검색 결과의 [DOC n]를 확인하라.")
            return payload

        @agent.tool(
            sequential=True,
            retries=1,
            timeout=self.settings.tool_timeout_seconds,
        )
        def get_source_overview(ctx: RunContext[CompassAgentDeps], doc_no: int, max_chars: int = 1000) -> str:
            """Get broader source overview for a discovered document."""
            deps = ctx.deps
            if not deps.allow_retrieval_tool or deps.source_overview_tool is None:
                raise ModelRetry("소스 개요 도구는 현재 비활성화되어 있다.")
            try:
                target_doc_no = int(doc_no)
                limit = min(1800, max(300, int(max_chars)))
            except Exception as exc:
                raise ModelRetry("doc_no와 max_chars는 정수여야 한다.") from exc
            payload = deps.source_overview_tool(target_doc_no, limit)
            if not (payload or "").strip():
                raise ModelRetry("해당 소스 개요를 찾지 못했다.")
            return payload

        @agent.tool(
            sequential=True,
            retries=1,
            timeout=self.settings.tool_timeout_seconds,
        )
        def get_source_outline(ctx: RunContext[CompassAgentDeps], doc_no: int, max_chars: int = 1200) -> str:
            """Get a compact outline for navigating a long document before opening detailed chunks."""
            deps = ctx.deps
            if not deps.allow_retrieval_tool or deps.source_outline_tool is None:
                raise ModelRetry("소스 개요 도구는 현재 비활성화되어 있다.")
            try:
                target_doc_no = int(doc_no)
                limit = min(2000, max(400, int(max_chars)))
            except Exception as exc:
                raise ModelRetry("doc_no와 max_chars는 정수여야 한다.") from exc
            payload = deps.source_outline_tool(target_doc_no, limit)
            if not (payload or "").strip():
                raise ModelRetry("해당 소스 outline을 찾지 못했다.")
            return payload

        @agent.tool(
            sequential=True,
            retries=1,
            timeout=self.settings.tool_timeout_seconds,
        )
        def list_current_sources(ctx: RunContext[CompassAgentDeps]) -> str:
            """List discovered sources for stable [DOC n] citation references."""
            deps = ctx.deps
            if deps.list_sources_tool is not None:
                payload = deps.list_sources_tool()
                if (payload or "").strip():
                    return payload
            return _format_current_sources(deps)

        @agent.output_validator
        def _validate_answer(ctx: RunContext[CompassAgentDeps], output: str) -> str:
            deps = ctx.deps
            canonical_output = canonicalize_doc_citations(output or "")
            repaired_format_output = canonicalize_doc_citations(repair_answer_text_format(canonical_output) or "").strip()
            if repaired_format_output and repaired_format_output != canonical_output.strip():
                _append_validation_retry_event(
                    deps,
                    validator="answer_format_repair",
                    detail="markdown/style wording repaired before citation validation",
                    payload={
                        "candidate_preview": trim_preview(canonical_output, 240),
                        "repaired_preview": trim_preview(repaired_format_output, 240),
                    },
                )
                canonical_output = repaired_format_output
            candidate_preview = trim_preview(canonical_output, 240)
            candidate_is_grounded_abstention = is_grounded_abstention_text(canonical_output)
            if deps.citation_validator is not None:
                citation_issue = deps.citation_validator(canonical_output)
                if citation_issue:
                    repaired_output = ""
                    repaired_issue = ""
                    if deps.citation_repairer is not None:
                        try:
                            repaired_output = canonicalize_doc_citations(deps.citation_repairer(canonical_output) or "").strip()
                        except Exception as exc:
                            _append_validation_retry_event(
                                deps,
                                validator="citation_repair",
                                detail=f"citation repair failed: {exc}",
                                payload={
                                    "docs_available": len(deps.run_state.docs),
                                    "available_doc_numbers": sorted(deps.run_state.docs.keys())[:12],
                                    "candidate_preview": candidate_preview,
                                },
                            )
                            repaired_output = ""
                        if repaired_output and repaired_output != canonical_output.strip():
                            repaired_issue = deps.citation_validator(repaired_output)
                            if not repaired_issue:
                                _append_validation_retry_event(
                                    deps,
                                    validator="citation_repair",
                                    detail="missing citations repaired from current sources",
                                    payload={
                                        "docs_available": len(deps.run_state.docs),
                                        "available_doc_numbers": sorted(deps.run_state.docs.keys())[:12],
                                        "candidate_preview": candidate_preview,
                                        "repaired_preview": trim_preview(repaired_output, 240),
                                        "previous_issue": citation_issue,
                                    },
                                )
                                canonical_output = repaired_output
                                candidate_preview = trim_preview(canonical_output, 240)
                                citation_issue = ""
                    if not citation_issue:
                        pass
                    else:
                        effective_issue = repaired_issue or citation_issue
                        retry_payload = {
                            "docs_available": len(deps.run_state.docs),
                            "available_doc_numbers": sorted(deps.run_state.docs.keys())[:12],
                            "candidate_preview": candidate_preview,
                        }
                        if repaired_issue:
                            retry_payload["repaired_issue"] = repaired_issue
                            retry_payload["repaired_preview"] = trim_preview(repaired_output, 240)
                        _append_validation_retry_event(
                            deps,
                            validator="citation",
                            detail=effective_issue,
                            payload=retry_payload,
                        )
                        raise ModelRetry(effective_issue)

            new_events = deps.run_state.tool_events[max(0, int(deps.tool_event_baseline or 0)) :]
            used_tool_names = {event.tool_name for event in new_events}
            analysis = deps.question_analysis or QuestionAnalysis()
            metrics = deps.run_state.latest_metrics or deps.retrieval.metrics or {}
            if deps.answer_sanitizer is not None:
                try:
                    sanitized_output = canonicalize_doc_citations(deps.answer_sanitizer(canonical_output) or "").strip()
                except Exception as exc:
                    _append_validation_retry_event(
                        deps,
                        validator="answer_sanitizer",
                        detail=f"answer sanitizer failed: {exc}",
                        payload={"candidate_preview": candidate_preview},
                    )
                    sanitized_output = ""
                if sanitized_output and sanitized_output != canonical_output.strip():
                    _append_validation_retry_event(
                        deps,
                        validator="answer_sanitizer",
                        detail="outside-document paragraph removed",
                        payload={
                            "candidate_preview": candidate_preview,
                            "sanitized_preview": trim_preview(sanitized_output, 240),
                        },
                    )
                    canonical_output = sanitized_output
                    candidate_preview = trim_preview(canonical_output, 240)
                    candidate_is_grounded_abstention = is_grounded_abstention_text(canonical_output)
                    if deps.citation_validator is not None:
                        sanitized_citation_issue = deps.citation_validator(canonical_output)
                        if sanitized_citation_issue:
                            _append_validation_retry_event(
                                deps,
                                validator="citation",
                                detail=sanitized_citation_issue,
                                payload={
                                    "candidate_preview": candidate_preview,
                                    "reason": "sanitized_output_lost_citation",
                                },
                            )
                            raise ModelRetry(sanitized_citation_issue)
            auto_prefetch_satisfied = bool(
                {"open_document", "get_source_outline", "get_source_overview"} & used_tool_names
            )
            evidence_texts = [
                f"{record.source_ref}\n{record.text}"
                for record in deps.run_state.docs.values()
            ]
            tool_recheck_payload = build_tool_recheck_debug_payload(
                require_tool_evidence=deps.require_tool_evidence,
                allow_retrieval_tool=deps.allow_retrieval_tool,
                docs_available=len(deps.run_state.docs),
                metrics=metrics,
                new_tool_event_count=len(new_events),
                numeric_evidence_required=bool(analysis.numeric_evidence_required),
                auto_prefetch_satisfied=auto_prefetch_satisfied,
                candidate_is_grounded_abstention=candidate_is_grounded_abstention,
                query_text=deps.user_message,
                evidence_texts=evidence_texts,
            )
            candidate_numeric_grounded = has_grounded_numeric_answer(
                query=deps.user_message,
                answer_text=canonical_output,
                evidence_texts=evidence_texts,
            )

            if should_require_tool_recheck(
                require_tool_evidence=deps.require_tool_evidence,
                allow_retrieval_tool=deps.allow_retrieval_tool,
                docs_available=len(deps.run_state.docs),
                metrics=metrics,
                new_tool_event_count=len(new_events),
                numeric_evidence_required=bool(analysis.numeric_evidence_required),
                auto_prefetch_satisfied=auto_prefetch_satisfied,
                candidate_is_grounded_abstention=candidate_is_grounded_abstention,
                evidence_alignment_ok=bool(tool_recheck_payload.get("evidence_alignment_ok", True)),
            ):
                _append_validation_retry_event(
                    deps,
                    validator="tool_recheck",
                    detail=(
                        "답변 전에 최소 한 번 search_knowledge_base, list_current_sources, "
                        "get_source_outline, get_source_overview, open_document 중 하나로 근거를 다시 확인하라."
                    ),
                    payload={
                        **tool_recheck_payload,
                        "auto_prefetch_satisfied": auto_prefetch_satisfied,
                        "candidate_is_grounded_abstention": candidate_is_grounded_abstention,
                        "used_tool_names": sorted(used_tool_names),
                        "candidate_preview": candidate_preview,
                    },
                )
                raise ModelRetry(
                    "답변 전에 최소 한 번 search_knowledge_base, list_current_sources, get_source_outline, get_source_overview, open_document 중 하나로 근거를 다시 확인하라."
                )

            if (
                analysis.numeric_evidence_required
                and deps.allow_retrieval_tool
                and not ({"open_document", "get_source_outline", "get_source_overview"} & used_tool_names)
            ):
                if candidate_numeric_grounded and bool(tool_recheck_payload.get("seeded_retrieval_evidence_ok", False)):
                    return canonical_output
                _append_validation_retry_event(
                    deps,
                    validator="numeric_recheck",
                    detail=(
                        "숫자/단가/표 기반 질문이다. 답변 전에 open_document 또는 "
                        "get_source_outline/get_source_overview로 숫자와 단위를 다시 확인하라."
                    ),
                    payload={
                        "numeric_evidence_required": True,
                        "docs_available": len(deps.run_state.docs),
                        "used_tool_names": sorted(used_tool_names),
                        "metrics_top1": float(metrics.get("top1", 0.0) or 0.0),
                        "metrics_coverage": float(metrics.get("coverage", 0.0) or 0.0),
                        "metrics_unique_sources": int(metrics.get("unique_sources", 0) or 0),
                        "candidate_preview": candidate_preview,
                    },
                )
                raise ModelRetry(
                    "숫자/단가/표 기반 질문이다. 답변 전에 open_document 또는 get_source_outline/get_source_overview로 숫자와 단위를 다시 확인하라."
                )

            if deps.require_tool_evidence and should_require_outline_recheck(
                use_source_outline=analysis.use_source_outline,
                outline_tool_used=bool({"get_source_outline", "get_source_overview", "open_document"} & used_tool_names),
                docs_available=len(deps.run_state.docs),
                metrics=metrics,
                numeric_evidence_required=bool(analysis.numeric_evidence_required),
            ):
                if deps.allow_retrieval_tool:
                    _append_validation_retry_event(
                        deps,
                        validator="outline_recheck",
                        detail=(
                            "긴 문서 또는 구조 파악형 질문이다. 답변 전에 get_source_outline 또는 "
                            "get_source_overview로 문서 구조를 먼저 확인하라."
                        ),
                        payload={
                            **build_outline_recheck_debug_payload(
                                use_source_outline=analysis.use_source_outline,
                                outline_tool_used=bool({"get_source_outline", "get_source_overview", "open_document"} & used_tool_names),
                                docs_available=len(deps.run_state.docs),
                                metrics=metrics,
                                numeric_evidence_required=bool(analysis.numeric_evidence_required),
                            ),
                            "used_tool_names": sorted(used_tool_names),
                            "candidate_preview": candidate_preview,
                        },
                    )
                    raise ModelRetry(
                        "긴 문서 또는 구조 파악형 질문이다. 답변 전에 get_source_outline 또는 get_source_overview로 문서 구조를 먼저 확인하라."
                    )

            checker = deps.quality_checker
            if checker is None:
                return canonical_output

            issue = checker(
                deps.user_message,
                canonical_output,
                metrics,
            )
            if not issue:
                return canonical_output

            hint_builder = deps.quality_hint_builder
            retry_hint = (
                hint_builder(issue)
                if hint_builder is not None
                else "근거 연결과 답변 충실도를 다시 점검해 재작성하라."
            )
            if deps.allow_retrieval_tool and deps.search_tool:
                retry_hint = (
                    f"{retry_hint}\n"
                    "필요하면 search_knowledge_base를 다시 호출하고, 특정 근거가 필요하면 open_document를 사용하라."
                )
            _append_validation_retry_event(
                deps,
                validator="quality_checker",
                detail=retry_hint,
                payload={
                    "quality_issue": issue,
                    "docs_available": len(deps.run_state.docs),
                    "used_tool_names": sorted(used_tool_names),
                    "metrics_top1": float(metrics.get("top1", 0.0) or 0.0),
                    "metrics_coverage": float(metrics.get("coverage", 0.0) or 0.0),
                    "metrics_unique_sources": int(metrics.get("unique_sources", 0) or 0),
                    "candidate_preview": candidate_preview,
                },
            )
            raise ModelRetry(retry_hint)

    async def _prepare_tools(
        self,
        ctx: RunContext[CompassAgentDeps],
        tool_defs: List[ToolDefinition],
    ) -> List[ToolDefinition] | None:
        deps = ctx.deps
        enabled: List[ToolDefinition] = []
        for tool in tool_defs:
            if tool.name == "search_knowledge_base":
                if deps.allow_retrieval_tool and deps.search_tool is not None:
                    enabled.append(tool)
            elif tool.name == "open_document":
                if deps.allow_retrieval_tool and deps.open_document_tool is not None and deps.run_state.docs:
                    enabled.append(tool)
            elif tool.name == "get_source_overview":
                if deps.allow_retrieval_tool and deps.source_overview_tool is not None and deps.run_state.docs:
                    enabled.append(tool)
            elif tool.name == "get_source_outline":
                if deps.allow_retrieval_tool and deps.source_outline_tool is not None and deps.run_state.docs:
                    enabled.append(tool)
            elif tool.name == "list_current_sources":
                if deps.allow_retrieval_tool and deps.run_state.docs:
                    enabled.append(tool)
            else:
                enabled.append(tool)
        return enabled

    async def run_helper_diagnostic(
        self,
        *,
        prompt: str,
        instructions: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout: int = 45,
        default_failure_code: str = "helper_run_fail",
    ) -> tuple[str, HelperRunDiagnostics]:
        fallback_candidate: Optional[BaseException] = None
        try:
            result = await self._helper_agent.run(
                prompt,
                instructions=instructions,
                model_settings=ModelSettings(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=float(timeout),
                ),
            )
        except UnexpectedModelBehavior as exc:
            fallback_candidate = exc
        except Exception as exc:
            fallback_candidate = exc

        if fallback_candidate is not None:
            if self._should_use_direct_text_completion_fallback(fallback_candidate):
                try:
                    output = await self._run_openai_compatible_text_completion(
                        prompt=prompt,
                        instructions=instructions,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                    )
                    if output:
                        return output, HelperRunDiagnostics(
                            status="ok",
                            failure_code=self._helper_failure_code(
                                fallback_candidate,
                                default_failure_code,
                            ),
                            error_type=type(fallback_candidate).__name__,
                            error_detail=(
                                "PydanticAI helper output validation failed; "
                                "used direct OpenAI-compatible text completion fallback."
                            ),
                            fallback_used=True,
                        )
                    failure_code = f"{default_failure_code}_direct_fallback_empty_output"
                    self._mark_helper_degraded(default_failure_code, failure_code)
                    return "", HelperRunDiagnostics(
                        status="error",
                        failure_code=failure_code,
                        error_type="EmptyOutput",
                        error_detail=(
                            "PydanticAI helper output validation failed; "
                            "direct OpenAI-compatible text completion fallback returned empty output."
                        ),
                        fallback_used=True,
                    )
                except Exception as fallback_exc:
                    failure_code = f"{default_failure_code}_direct_fallback_fail"
                    self._mark_helper_degraded(default_failure_code, failure_code)
                    return "", HelperRunDiagnostics(
                        status="error",
                        failure_code=failure_code,
                        error_type=type(fallback_exc).__name__,
                        error_detail=(
                            f"{str(fallback_candidate or '').strip()} | "
                            f"direct fallback failed: {str(fallback_exc or '').strip()}"
                        ),
                        fallback_used=True,
                    )
            failure_code = self._helper_failure_code(fallback_candidate, default_failure_code)
            self._mark_helper_degraded(default_failure_code, failure_code)
            return "", self._build_helper_diagnostics(
                fallback_candidate,
                default_failure_code=default_failure_code,
            )
        output = (result.output or "").strip()
        if not output:
            failure_code = f"{default_failure_code}_empty_output"
            self._mark_helper_degraded(default_failure_code, failure_code)
            return "", HelperRunDiagnostics(
                status="error",
                failure_code=failure_code,
                error_type="EmptyOutput",
                error_detail="helper returned empty output",
            )
        return output, HelperRunDiagnostics(status="ok")

    async def run_helper(
        self,
        *,
        prompt: str,
        instructions: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout: int = 45,
    ) -> str:
        output, _ = await self.run_helper_diagnostic(
            prompt=prompt,
            instructions=instructions,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        return output

    async def greeting_text(self, user_message: str, *, timeout: int = 20) -> str:
        content, _ = await self.greeting_text_diagnostic(
            user_message,
            timeout=timeout,
        )
        return content

    async def greeting_text_diagnostic(
        self,
        user_message: str,
        *,
        timeout: int = 20,
    ) -> tuple[str, HelperRunDiagnostics]:
        content, diag = await self.run_helper_diagnostic(
            prompt=f"사용자 인사: {user_message}\n1~2문장으로 자연스럽고 친절한 존댓말 인사만 답변해줘.",
            instructions=(
                "너는 한국어 비서다. 사용자가 인사하면 짧고 자연스럽고 친절한 존댓말 인사로만 답한다. "
                "추가 정보나 추측은 하지 않는다."
            ),
            max_tokens=80,
            temperature=0.2,
            timeout=timeout,
            default_failure_code="greeting_fail",
        )
        return content or "안녕하세요. 편하게 말씀해 주시면 제가 도와드리겠습니다.", diag

    async def casual_chat_diagnostic(
        self,
        user_message: str,
        *,
        recent_history: str = "",
        timeout: int = 20,
    ) -> tuple[str, HelperRunDiagnostics]:
        content, diag = await self.run_helper_diagnostic(
            prompt=(
                "다음 최근 대화를 참고해서 사용자의 일상 대화에 짧고 자연스럽게 답해줘.\n"
                "- 최근 대화가 없으면 현재 메시지만 보고 답변\n"
                "- 답변은 1~3문장 존댓말\n"
                "- 문서 검색, 근거, 인용, 업로드, 시스템 설명은 먼저 꺼내지 말 것\n"
                "- 마크다운 제목/불릿/굵게/백틱 금지\n"
                "- 강조가 필요하면 괄호나 대괄호만 사용\n\n"
                "[RECENT_HISTORY]\n"
                f"{recent_history or '(없음)'}\n\n"
                f"[USER_MESSAGE]\n{user_message}"
            ),
            instructions=(
                "너는 한국어 비서다. 가벼운 인사, 감사, 짧은 잡담, 맥락 있는 일상 후속 질문에 "
                "공손하고 자연스러운 존댓말로만 짧게 답한다."
            ),
            max_tokens=140,
            temperature=0.3,
            timeout=timeout,
            default_failure_code="casual_chat_fail",
        )
        return content or "네, 편하게 말씀해 주세요.", diag

    async def expand_query(
        self,
        user_message: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> str:
        content, _ = await self.expand_query_diagnostic(
            user_message,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return content

    async def expand_query_diagnostic(
        self,
        user_message: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, HelperRunDiagnostics]:
        prompt = (
            "아래 질문을 검색용 키워드로 확장해줘.\n"
            "- 출력은 한 줄(키워드 나열)만\n"
            "- 설명/부연 금지\n"
            "- 질문의 고유명사/숫자/코드/연도/단위/약어는 절대 바꾸거나 삭제하지 말 것\n"
            "- 원문 키워드는 유지하고, 필요한 동의어/현업표현만 보강\n"
            "- 추상 질문이면 실제 문서에서 찾기 쉬운 구체 표현으로 보강\n"
            "- 최대 24개 키워드\n\n"
            f"질문: {user_message}"
        )
        instructions = (
            "너는 RAG 검색 질의 재작성기다. "
            "질문의 의미는 유지하면서 검색 재현율과 정밀도를 높이는 키워드만 생성한다."
        )
        default_failure_code = "query_expand_fail"
        if self._is_helper_degraded(default_failure_code):
            return await self._run_degraded_helper_with_deterministic(
                prompt=prompt,
                instructions=instructions,
                max_tokens=max_tokens,
                deterministic=lambda: self._deterministic_query_expansion(user_message),
                temperature=0.0,
                timeout=timeout,
                default_failure_code=default_failure_code,
            )
        content, diag = await self.run_helper_diagnostic(
            prompt=prompt,
            instructions=instructions,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
            default_failure_code=default_failure_code,
        )
        if content:
            return content, diag
        fallback = self._deterministic_query_expansion(user_message)
        if fallback:
            return fallback, HelperRunDiagnostics(
                status="fallback",
                failure_code=diag.failure_code or "query_expand_deterministic_fallback",
                error_type=diag.error_type,
                error_detail="LLM query expansion failed; used deterministic query terms.",
                fallback_used=True,
            )
        return "", diag

    async def analyze_question(
        self,
        user_message: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> QuestionAnalysis:
        analysis, _ = await self.analyze_question_diagnostic(
            user_message,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return analysis

    async def analyze_question_diagnostic(
        self,
        user_message: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> tuple[QuestionAnalysis, HelperRunDiagnostics]:
        helper_prompt = (
            "아래 사용자 질문을 문서 기반 질의응답용으로 분석해줘.\n"
            "- 긴 문장이어도 실제 문서 검색에 바로 쓸 수 있는 짧은 검색 구문으로 압축\n"
            "- 질문이 요약/정리 성격이면 literal_first=yes\n"
            "- 방금 업로드한 문서의 핵심 정리를 우선 봐야 하면 prefer_recent=yes\n"
            "- 문서가 길거나 전체 구조/흐름 파악이 중요하면 use_source_outline=yes\n"
            "- 근거 확인이 필요하면 require_tool_evidence=yes\n"
            "- 숫자/단가/금액/비율/건수 질문이면 numeric_evidence_required=yes\n"
            "- 출력은 정확히 아래 8줄 형식만 사용\n"
            "intent=summary|fact|definition|identity|comparison|procedure|analysis|mixed\n"
            "literal_first=yes|no\n"
            "prefer_recent=yes|no\n"
            "use_source_outline=yes|no\n"
            "require_tool_evidence=yes|no\n"
            "numeric_evidence_required=yes|no\n"
            "queries=검색구문1 || 검색구문2 || 검색구문3\n"
            "focus=답변핵심1 || 답변핵심2 || 답변핵심3\n\n"
            f"질문: {user_message}"
        )
        helper_instructions = (
            "너는 문서 검색 전략 분석기다. "
            "사용자 질문의 의도를 압축해서 검색 재현율과 답변 적합도를 높이는 실용적인 힌트만 만든다."
        )
        helper_failure_code = "question_analysis_helper_fail"
        if self._is_helper_degraded(helper_failure_code):
            return await self._run_degraded_helper_with_deterministic(
                prompt=helper_prompt,
                instructions=helper_instructions,
                max_tokens=max_tokens,
                deterministic=lambda: self._deterministic_question_analysis(user_message),
                temperature=0.0,
                timeout=timeout,
                default_failure_code=helper_failure_code,
            )
        primary_diag = HelperRunDiagnostics(status="ok")
        try:
            result = await self._analysis_agent.run(
                (
                    "사용자 질문을 문서 기반 질의응답용으로 분석해줘.\n"
                    "- 긴 문장이어도 실제 검색용으로 바로 쓸 수 있는 짧은 구문으로 분해\n"
                    "- 요약/정리/전체 흐름 질문이면 literal_first=true\n"
                    "- 최신 업로드 문서가 우선일 가능성이 높으면 prefer_recent_sources=true\n"
                    "- 긴 문서 탐색형 질문이면 use_source_outline=true\n"
                    "- 문서 근거 확인이 필요한 질문은 require_tool_evidence=true 유지\n"
                    "- 숫자/단가/금액/비율/건수 같은 질문이면 numeric_evidence_required=true\n"
                    "- search_queries는 최대 5개, answer_focus는 최대 4개\n\n"
                    f"질문: {user_message}"
                ),
                model_settings=ModelSettings(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=float(timeout),
                ),
            )
            return self._normalize_question_analysis(result.output), primary_diag
        except UnexpectedModelBehavior as exc:
            primary_diag = self._build_helper_diagnostics(
                exc,
                default_failure_code="question_analysis_agent_fail",
                status="fallback",
                fallback_used=True,
            )
        except Exception as exc:
            primary_diag = self._build_helper_diagnostics(
                exc,
                default_failure_code="question_analysis_agent_fail",
                status="fallback",
                fallback_used=True,
            )

        content, helper_diag = await self.run_helper_diagnostic(
            prompt=helper_prompt,
            instructions=helper_instructions,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
            default_failure_code=helper_failure_code,
        )
        if not content:
            failure_diag = helper_diag if helper_diag.failure_code else primary_diag
            return self._deterministic_question_analysis(user_message), HelperRunDiagnostics(
                status="fallback",
                failure_code=failure_diag.failure_code or "question_analysis_fail",
                error_type=failure_diag.error_type or primary_diag.error_type,
                error_detail="LLM question analysis failed; used deterministic question analysis.",
                fallback_used=True,
            )
        if primary_diag.fallback_used and not helper_diag.fallback_used:
            helper_diag = HelperRunDiagnostics(
                status="ok",
                failure_code=primary_diag.failure_code,
                error_type=primary_diag.error_type,
                error_detail="Structured question analysis failed; used text helper fallback.",
                fallback_used=True,
            )
        return self._parse_question_analysis(content), helper_diag

    async def followup_rewrite(
        self,
        user_message: str,
        *,
        recent_history: str,
        max_tokens: int,
        timeout: int,
    ) -> FollowupAnalysis:
        analysis, _ = await self.followup_rewrite_diagnostic(
            user_message,
            recent_history=recent_history,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return analysis

    async def followup_rewrite_diagnostic(
        self,
        user_message: str,
        *,
        recent_history: str,
        max_tokens: int,
        timeout: int,
    ) -> tuple[FollowupAnalysis, HelperRunDiagnostics]:
        prompt = (
            "다음 최근 대화를 바탕으로 사용자의 새 발화를 해석해줘.\n"
            "- 사용자가 직전 질문을 정정, 치환, 보완, 이어서 묻는지 판단\n"
            "- 잡담/인사/감사면 small_talk로 분류\n"
            "- 독립 질문이면 standalone\n"
            "- 출력은 정확히 아래 4줄 형식만 사용\n"
            "followup_type=standalone|correction|continuation|small_talk|meta\n"
            "rewritten_query=완결된 단독 질문 또는 빈 문자열\n"
            "should_use_history=yes|no\n"
            "is_small_talk=yes|no\n\n"
            "[RECENT_HISTORY]\n"
            f"{recent_history or '(없음)'}\n\n"
            f"[NEW_MESSAGE]\n{user_message}"
        )
        instructions = (
            "너는 한국어 대화 문맥 해석기다. "
            "새 발화가 이전 질문의 정정이나 후속 설명이면, 문서 검색에 바로 쓸 수 있는 완결된 질문으로 다시 써라."
        )
        default_failure_code = "followup_rewrite_fail"
        if self._is_helper_degraded(default_failure_code):
            return await self._run_degraded_helper_with_deterministic(
                prompt=prompt,
                instructions=instructions,
                max_tokens=max_tokens,
                deterministic=lambda: FollowupAnalysis(),
                temperature=0.0,
                timeout=timeout,
                default_failure_code=default_failure_code,
            )
        content, diag = await self.run_helper_diagnostic(
            prompt=prompt,
            instructions=instructions,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
            default_failure_code=default_failure_code,
        )
        if not content:
            return FollowupAnalysis(), diag
        return self._parse_followup_analysis(content), diag

    async def review_operations(
        self,
        report_text: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> OpsReview:
        review, _ = await self.review_operations_diagnostic(
            report_text,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return review

    async def review_operations_diagnostic(
        self,
        report_text: str,
        *,
        max_tokens: int,
        timeout: int,
    ) -> tuple[OpsReview, HelperRunDiagnostics]:
        try:
            result = await self._ops_review_agent.run(
                (
                    "다음 운영 로그 요약을 보고, 문서 근거형 에이전트 품질 개선을 위한 핵심 패턴과 조치를 정리해줘.\n"
                    "- top_patterns는 실제 실패 패턴만 간단명료하게 적는다.\n"
                    "- recommended_actions는 바로 실행 가능한 개선 항목만 적는다.\n"
                    "- monitoring_checks는 다음 배포 때 다시 확인할 점검 항목만 적는다.\n\n"
                    f"{report_text}"
                ),
                model_settings=ModelSettings(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=float(timeout),
                ),
            )
            return OpsReview.model_validate(result.output), HelperRunDiagnostics(status="ok")
        except Exception as exc:
            return OpsReview(), self._build_helper_diagnostics(
                exc,
                default_failure_code="ops_review_fail",
            )

    async def rerank_candidates(
        self,
        *,
        user_message: str,
        candidate_lines: List[str],
        max_tokens: int,
        timeout: int,
    ) -> str:
        content, _ = await self.rerank_candidates_diagnostic(
            user_message=user_message,
            candidate_lines=candidate_lines,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return content

    async def rerank_candidates_diagnostic(
        self,
        *,
        user_message: str,
        candidate_lines: List[str],
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, HelperRunDiagnostics]:
        prompt = build_rerank_prompt(user_message, candidate_lines)
        instructions = "너는 검색 재랭커다. 사용자 질문에 직접 답을 주는 원문 근거를 우선 정렬한다."
        default_failure_code = "rerank_fail"
        if self._is_helper_degraded(default_failure_code):
            return await self._run_degraded_helper_with_deterministic(
                prompt=prompt,
                instructions=instructions,
                max_tokens=max_tokens,
                deterministic=lambda: "",
                temperature=0.0,
                timeout=timeout,
                default_failure_code=default_failure_code,
            )
        return await self.run_helper_diagnostic(
            prompt=prompt,
            instructions=instructions,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
            default_failure_code=default_failure_code,
        )

    async def answer_question(
        self,
        deps: CompassAgentDeps,
        *,
        message_history: Optional[List[ModelMessage]] = None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: int,
        runtime_instructions: str = "",
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        has_runtime_tools = bool(
            deps.allow_retrieval_tool
            and (
                deps.search_tool
                or deps.open_document_tool
                or deps.source_overview_tool
                or deps.source_outline_tool
                or deps.list_sources_tool
            )
        )
        tool_calls_limit = (
            self.settings.tool_calls_limit
            if has_runtime_tools
            else 0
        )
        usage_limits = UsageLimits(
            request_limit=self.settings.request_limit,
            tool_calls_limit=tool_calls_limit,
        )
        prompt = (
            f"QUESTION:\n{deps.user_message}\n\n"
            "TASK:\n"
            "- 시스템 지침에 따라 자연스러운 한국어 답변으로 작성한다.\n"
            "- 문서 답변이면 현재 SEEDED_CONTEXT와 CURRENT_SOURCES를 먼저 검토한다.\n"
            "- 현재 확보된 근거가 질문에 직접 대응하면 추가 도구 호출 없이 답한다.\n"
            "- 근거가 질문과 어긋날 때만 검색어를 바꿔 search_knowledge_base를 호출한다.\n"
            "- 문서가 길거나 전체 구조가 중요하면 get_source_outline 또는 get_source_overview로 먼저 범위를 좁힌다.\n"
            "- 각 핵심 주장에 [DOC i] 근거를 연결한다.\n"
            "- 질문 문장을 그대로 반복하지 말고 결론부터 작성한다.\n"
            "- 필요하면 2~4문장 또는 짧은 줄바꿈으로 정리하되, 마크다운 제목/목록/굵게/백틱은 쓰지 않는다.\n"
            "- 강조가 필요하면 괄호나 대괄호만 사용한다.\n"
            "- 근거가 부족하면 문서 근거 부족이라고 명확히 말하고 확인된 범위만 답한다."
        )
        metadata = {
            "kb_name": deps.kb_name,
            "query_id": deps.query_id,
        }
        if run_metadata:
            metadata.update(run_metadata)
        return await self._answer_agent.run(
            prompt,
            deps=deps,
            message_history=message_history,
            instructions=runtime_instructions or None,
            model_settings=ModelSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=float(timeout),
            ),
            usage_limits=usage_limits,
            metadata=metadata,
        )
