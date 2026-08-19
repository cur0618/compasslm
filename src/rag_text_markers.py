"""Canonical Korean labels shared by RAG ingest, retrieval, and diagnostics."""

EMBEDDING_DIMENSION_PROBE_TEXT = "임베딩 차원 확인"

QUESTION_COLUMN_MARKERS = ("질문", "question", "문의", "query", "q")
ANSWER_COLUMN_MARKERS = ("답변", "answer", "해설", "조치", "결과", "a")

TABLE_HINT_MARKERS = ("표행:", "표행요약:", "표헤더:", "표값:", "표의미:")
TABLE_ROW_SUMMARY_MARKER = "표행요약:"
TABLE_SEMANTIC_ROW_MARKER = "표의미: kind=table_row"

LABEL_ANSWER_PRIORITY = "답변 우선순위"
LABEL_BODY = "본문"
LABEL_LATEST = "최신"
LABEL_LATEST_UPLOAD_REFLECTED = "최신 업로드 반영"
LABEL_LATEST_UPLOAD_REFLECTED_AT = "최신 업로드 반영 시각"
LABEL_LINE = "라인"
LABEL_LOCATION = "위치"
LABEL_NO_LOCATION = "위치 정보 없음"
LABEL_QUESTION = "질문"
LABEL_ROW = "행"
LABEL_SOURCE = "출처"
LABEL_SOURCE_SUMMARY = "출처 요약"
LABEL_UPLOAD = "업로드"
NORMALIZED_BUNDLE_LABEL = "통합 정리"


def normalized_bundle_header(group: str) -> str:
    return f"[통합정리-{(group or '').upper()}]"


def normalized_bundle_section(group: str) -> str:
    return f"통합정리-{(group or '').upper()} (최신우선)"
