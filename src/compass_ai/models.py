from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


QualityChecker = Callable[[str, str, Dict[str, Any]], str]
QualityHintBuilder = Callable[[str], str]
SearchToolExecutor = Callable[[str, int, str], str]
OpenDocToolExecutor = Callable[[int, int], str]
OverviewToolExecutor = Callable[[int, int], str]
SourceOutlineToolExecutor = Callable[[int, int], str]
ListSourcesToolExecutor = Callable[[], str]
CitationValidator = Callable[[str], str]


@dataclass
class RetrievalSnapshot:
    context: str
    retrieval_meta: str
    source_hint: str
    reference_hint: str = ""
    query_doc_intent: str = ""
    retrieval_role_filter: str = ""
    search_query: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    overview_mode: bool = False


class QuestionAnalysis(BaseModel):
    intent_type: str = Field(default="mixed")
    search_queries: List[str] = Field(default_factory=list)
    answer_focus: List[str] = Field(default_factory=list)
    literal_first: bool = Field(default=False)
    prefer_recent_sources: bool = Field(default=False)
    use_source_outline: bool = Field(default=False)
    require_tool_evidence: bool = Field(default=True)
    numeric_evidence_required: bool = Field(default=False)


class FollowupAnalysis(BaseModel):
    followup_type: str = Field(default="standalone")
    rewritten_query: str = Field(default="")
    should_use_history: bool = Field(default=False)
    is_small_talk: bool = Field(default=False)


class OpsReview(BaseModel):
    top_patterns: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    monitoring_checks: List[str] = Field(default_factory=list)


@dataclass
class RetrievedDocRecord:
    doc_no: int
    chunk_id: int
    source_path: str
    source_ref: str
    text: str
    source_type: str = ""
    section: str = ""
    sheet: str = ""
    row: int = 0
    row_end: int = 0
    page_no: int = 0
    line_start: int = 0
    line_end: int = 0
    uploaded_at: int = 0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEventRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result_preview: str
    created_at_iso: str


@dataclass
class PhaseEventRecord:
    phase: str
    event_name: str
    status: str = "ok"
    detail: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at_iso: str = ""


@dataclass
class HelperRunDiagnostics:
    status: str = "ok"
    failure_code: str = ""
    error_type: str = ""
    error_detail: str = ""
    fallback_used: bool = False
    helper_degraded: bool = False
    deterministic_parallel_used: bool = False
    helper_wait_skipped: bool = False


@dataclass
class AgentRunState:
    docs: Dict[int, RetrievedDocRecord] = field(default_factory=dict)
    doc_numbers_by_chunk_id: Dict[int, int] = field(default_factory=dict)
    tool_events: List[ToolEventRecord] = field(default_factory=list)
    phase_events: List[PhaseEventRecord] = field(default_factory=list)
    latest_search_query: str = ""
    latest_role_filter: str = "all"
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    latest_result_count: int = 0


@dataclass
class CompassAgentDeps:
    kb_name: str
    query_id: str
    user_message: str
    retrieval: RetrievalSnapshot
    runtime_date_iso: str
    question_analysis: Optional[QuestionAnalysis] = None
    run_state: AgentRunState = field(default_factory=AgentRunState)
    quality_checker: Optional[QualityChecker] = None
    quality_hint_builder: Optional[QualityHintBuilder] = None
    search_tool: Optional[SearchToolExecutor] = None
    open_document_tool: Optional[OpenDocToolExecutor] = None
    source_overview_tool: Optional[OverviewToolExecutor] = None
    source_outline_tool: Optional[SourceOutlineToolExecutor] = None
    list_sources_tool: Optional[ListSourcesToolExecutor] = None
    citation_validator: Optional[CitationValidator] = None
    require_tool_evidence: bool = False
    tool_event_baseline: int = 0
    allow_retrieval_tool: bool = False
