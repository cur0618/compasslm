import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "answer_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_answer_validation", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnswerValidationRuleTests(unittest.TestCase):
    def test_detects_weak_ocr_hint_text(self):
        module = _load_module()
        self.assertTrue(
            module.is_weak_ocr_hint_text(
                "OCR후보: 텍스트 부족 페이지 | 페이지 160 | 이미지 스캔 표 숫자 단가 금액 확인\nOCR후보: 텍스트 부족 페이지 | 페이지 160 | 원문 확인 필요"
            )
        )
        self.assertFalse(
            module.is_weak_ocr_hint_text(
                "표행요약: 조사명 단가 횟수 지급대상월 | 가계동향조사 본조사 80 12 1 2 3"
            )
        )

    def test_summarize_evidence_strength_flags_weak_only_sets(self):
        module = _load_module()
        weak_only = module.summarize_evidence_strength(
            [
                {"text": "OCR후보: 텍스트 부족 페이지 | 페이지 160 | 원문 확인 필요"},
                {"text": "OCR후보: 텍스트 부족 페이지 | 페이지 159 | 원문 확인 필요"},
            ]
        )
        self.assertTrue(weak_only["weak_evidence_only"])
        self.assertEqual(weak_only["strong_evidence_count"], 0)
        self.assertEqual(weak_only["weak_evidence_count"], 2)

        mixed = module.summarize_evidence_strength(
            [
                {"text": "OCR후보: 텍스트 부족 페이지 | 페이지 160 | 원문 확인 필요"},
                {"text": "답례품 단가는 4만원입니다. [DOC 1]"},
            ]
        )
        self.assertFalse(mixed["weak_evidence_only"])
        self.assertEqual(mixed["strong_evidence_count"], 1)
        self.assertEqual(mixed["weak_evidence_count"], 1)

    def test_seeded_retrieval_evidence_skips_tool_recheck(self):
        module = _load_module()
        self.assertFalse(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=24,
                metrics={"top1": 0.49, "coverage": 0.75},
                new_tool_event_count=0,
            )
        )

    def test_missing_seeded_evidence_still_requires_tool_recheck(self):
        module = _load_module()
        self.assertTrue(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=0,
                metrics={"top1": 0.0, "coverage": 0.0},
                new_tool_event_count=0,
            )
        )

    def test_grounded_abstention_without_seeded_evidence_skips_tool_recheck(self):
        module = _load_module()
        self.assertFalse(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=8,
                metrics={"top1": 0.0, "coverage": 0.0},
                new_tool_event_count=0,
                candidate_is_grounded_abstention=True,
            )
        )

    def test_existing_tool_event_always_satisfies_tool_recheck(self):
        module = _load_module()
        self.assertFalse(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=0,
                metrics={"top1": 0.0, "coverage": 0.0},
                new_tool_event_count=1,
            )
        )

    def test_numeric_query_requires_tool_recheck_even_with_seeded_evidence(self):
        module = _load_module()
        self.assertTrue(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=24,
                metrics={"top1": 0.49, "coverage": 0.75},
                new_tool_event_count=0,
                numeric_evidence_required=True,
            )
        )

    def test_numeric_grounded_abstention_still_requires_tool_recheck(self):
        module = _load_module()
        self.assertTrue(
            module.should_require_tool_recheck(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=8,
                metrics={"top1": 0.0, "coverage": 0.0},
                new_tool_event_count=0,
                numeric_evidence_required=True,
                candidate_is_grounded_abstention=True,
            )
        )

    def test_numeric_query_can_auto_prefetch_document_evidence(self):
        module = _load_module()
        self.assertTrue(
            module.should_auto_prefetch_numeric_evidence(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=8,
                metrics={"top1": 0.64, "coverage": 1.0},
                new_tool_event_count=0,
                numeric_evidence_required=True,
            )
        )

    def test_tabular_schedule_query_is_treated_as_evidence_query(self):
        module = _load_module()
        self.assertTrue(module.is_numeric_evidence_query("봄배추는 언제 지급하는지 알려줘"))
        self.assertTrue(module.is_numeric_evidence_query("경제활동인구조사 지급 주기 알려줘"))
        self.assertFalse(module.is_numeric_evidence_query("답례품 종류 알려줘"))

    def test_auto_prefetch_skips_when_numeric_tool_has_already_run(self):
        module = _load_module()
        self.assertFalse(
            module.should_auto_prefetch_numeric_evidence(
                require_tool_evidence=True,
                allow_retrieval_tool=True,
                docs_available=8,
                metrics={"top1": 0.64, "coverage": 1.0},
                new_tool_event_count=1,
                numeric_evidence_required=True,
            )
        )

    def test_tool_recheck_debug_payload_exposes_auto_prefetch_state(self):
        module = _load_module()
        payload = module.build_tool_recheck_debug_payload(
            require_tool_evidence=True,
            allow_retrieval_tool=True,
            docs_available=8,
            metrics={"top1": 0.18, "coverage": 0.14, "unique_sources": 2},
            new_tool_event_count=1,
            numeric_evidence_required=True,
            auto_prefetch_satisfied=True,
        )
        self.assertTrue(payload["auto_prefetch_satisfied"])
        self.assertFalse(payload["should_require_tool_recheck"])

    def test_tool_recheck_debug_payload_exposes_retry_reason(self):
        module = _load_module()
        payload = module.build_tool_recheck_debug_payload(
            require_tool_evidence=True,
            allow_retrieval_tool=True,
            docs_available=24,
            metrics={"top1": 0.64, "coverage": 1.0, "unique_sources": 3},
            new_tool_event_count=0,
            numeric_evidence_required=True,
        )
        self.assertEqual(payload["metrics_top1"], 0.64)
        self.assertEqual(payload["metrics_coverage"], 1.0)
        self.assertEqual(payload["metrics_unique_sources"], 3)
        self.assertTrue(payload["seeded_retrieval_evidence_ok"])
        self.assertTrue(payload["should_require_tool_recheck"])
        self.assertEqual(payload["top1_threshold"], module.SEEDED_EVIDENCE_TOP1_MIN)
        self.assertEqual(payload["coverage_threshold"], module.SEEDED_EVIDENCE_COVERAGE_MIN)

    def test_seeded_evidence_alignment_rejects_unrelated_top_results(self):
        module = _load_module()
        evidence_texts = [
            "Q8 마을버스 등기이사로 5억 투자하고 매월 200만원 수입이 있는데 농업외임금으로 조사하는지 여부",
            "부모와 자녀가 함께 생활하다 부모가 분가한 경우 농기계와 토지 담보 부채 파악방법",
        ]
        payload = module.build_tool_recheck_debug_payload(
            require_tool_evidence=True,
            allow_retrieval_tool=True,
            docs_available=24,
            metrics={"top1": 0.49, "coverage": 0.75, "unique_sources": 3},
            new_tool_event_count=0,
            numeric_evidence_required=False,
            query_text="자녀가 중학교 졸업 후 고등학교 들어가기 전 어떻게 조사해야해?",
            evidence_texts=evidence_texts,
        )
        self.assertFalse(payload["evidence_alignment_ok"])
        self.assertTrue(payload["seeded_retrieval_evidence_ok"])
        self.assertTrue(payload["should_require_tool_recheck"])

    def test_seeded_evidence_alignment_accepts_matching_school_terms(self):
        module = _load_module()
        payload = module.build_tool_recheck_debug_payload(
            require_tool_evidence=True,
            allow_retrieval_tool=True,
            docs_available=24,
            metrics={"top1": 0.49, "coverage": 0.75, "unique_sources": 3},
            new_tool_event_count=0,
            numeric_evidence_required=False,
            query_text="자녀가 중학교 졸업 후 고등학교 들어가기 전 어떻게 조사해야해?",
            evidence_texts=[
                "2월 중학교 졸업 후 3월 고등학교 진학 전에는 재학 상태와 교육 정도를 월별로 확인한다."
            ],
        )
        self.assertTrue(payload["evidence_alignment_ok"])
        self.assertFalse(payload["should_require_tool_recheck"])

    def test_numeric_query_abstention_is_not_treated_as_quality_issue(self):
        module = _load_module()
        self.assertFalse(
            module.should_treat_abstention_as_quality_issue(
                query="타인의 소를 위탁 사육하고 받은 수수료는 어떻게 조사해야해?",
                metrics={"top1": 0.54, "coverage": 0.29},
            )
        )

    def test_non_numeric_query_abstention_with_strong_evidence_remains_quality_issue(self):
        module = _load_module()
        self.assertTrue(
            module.should_treat_abstention_as_quality_issue(
                query="산에서 난로용 장작나무를 베어온 경우 조사 방법을 알려줘",
                metrics={"top1": 0.38, "coverage": 0.67},
            )
        )

    def test_outline_recheck_debug_payload_exposes_outline_state(self):
        module = _load_module()
        payload = module.build_outline_recheck_debug_payload(
            use_source_outline=True,
            outline_tool_used=False,
            docs_available=24,
            metrics={"top1": 0.15, "coverage": 0.1, "unique_sources": 2},
            numeric_evidence_required=False,
        )
        self.assertTrue(payload["use_source_outline"])
        self.assertFalse(payload["outline_tool_used"])
        self.assertFalse(payload["seeded_retrieval_evidence_ok"])
        self.assertTrue(payload["should_require_outline_recheck"])

    def test_numeric_amount_response_must_match_grounded_evidence(self):
        module = _load_module()
        grounded = module.has_grounded_numeric_answer(
            query="농가경제조사 단가 알려줘",
            answer_text="농가경제조사 답례품 단가는 4만원입니다. [DOC 1]",
            evidence_texts=["농가경제조사 답례품 단가는 40(천원) 또는 4만원 입니다."],
        )
        hallucinated = module.has_grounded_numeric_answer(
            query="농가경제조사 단가 알려줘",
            answer_text="농가경제조사 단가는 4300원입니다. [DOC 1]",
            evidence_texts=["농가경제조사 답례품 단가는 40(천원) 또는 4만원 입니다."],
        )
        self.assertTrue(grounded)
        self.assertFalse(hallucinated)

    def test_hwpx_table_amount_infers_thousand_won_unit_from_price_header(self):
        module = _load_module()
        self.assertTrue(
            module.has_grounded_numeric_answer(
                query="농가경제조사 단가 알려줘",
                answer_text="농가경제조사 답례품 단가는 4만원입니다. [DOC 1]",
                evidence_texts=[
                    "표행요약: 답례품=지류, 현금 | 조사명=농어가경제조사 | 명칭별칭=농어가경제조사, 농가경제조사, 어가경제조사 | 지급단가=40"
                ],
            )
        )

    def test_numeric_amount_must_match_same_subject_table_fact(self):
        module = _load_module()
        mixed_evidence = [
            "표의미: kind=table_row | subject=농어가경제조사 | aliases=농어가경제조사, 농가경제조사, 어가경제조사 | 지급단가=40천원 | 지급횟수=12회",
            "표의미: kind=table_row | subject=가계동향조사 | aliases=가계동향조사 | 구분=1인가구 | 지급단가=4만원",
            "표의미: kind=table_row | subject=가계동향조사 | aliases=가계동향조사 | 구분=2인이상 | 지급단가=2만원",
        ]
        grounded = module.has_grounded_numeric_answer(
            query="농가경제조사 답례품 단가 알려줘",
            answer_text="농가경제조사 답례품 단가는 4만원입니다. [DOC 1]",
            evidence_texts=mixed_evidence,
        )
        wrong_cross_row = module.has_grounded_numeric_answer(
            query="농가경제조사 답례품 단가는 가구원수에 따라 알려줘",
            answer_text="농가경제조사는 1인가구 4만원, 2인 이상 2만원입니다. [DOC 1]",
            evidence_texts=mixed_evidence,
        )
        self.assertTrue(grounded)
        self.assertFalse(wrong_cross_row)

    def test_tabular_schedule_response_must_match_grounded_time_evidence(self):
        module = _load_module()
        grounded = module.has_grounded_numeric_answer(
            query="봄배추는 언제 지급하는지 알려줘",
            answer_text="봄배추는 4월 지급 기준입니다. [DOC 1]",
            evidence_texts=[
                "표행요약: 작물명 조사 시기 보고 기일 지급 기준월 지급 단가 | 봄배추 4월 중순 ~ 6월 초순 6월 5일 4월 20천원 포구"
            ],
        )
        hallucinated = module.has_grounded_numeric_answer(
            query="봄배추는 언제 지급하는지 알려줘",
            answer_text="봄배추는 7월 지급 기준입니다. [DOC 1]",
            evidence_texts=[
                "표행요약: 작물명 조사 시기 보고 기일 지급 기준월 지급 단가 | 봄배추 4월 중순 ~ 6월 초순 6월 5일 4월 20천원 포구"
            ],
        )
        self.assertTrue(grounded)
        self.assertFalse(hallucinated)

    def test_general_knowledge_fallback_is_disabled_when_kb_has_docs(self):
        module = _load_module()
        self.assertFalse(
            module.should_allow_general_knowledge_fallback(
                "농가경제조사 단가 알려줘",
                kb_has_docs=True,
            )
        )
        self.assertTrue(
            module.should_allow_general_knowledge_fallback(
                "오늘 날씨 어때",
                kb_has_docs=False,
            )
        )

    def test_markdown_emphasis_is_rejected(self):
        module = _load_module()
        self.assertTrue(module.contains_disallowed_markdown("답은 **4만원**입니다. [DOC 1]"))

    def test_markdown_list_is_rejected(self):
        module = _load_module()
        self.assertTrue(module.contains_disallowed_markdown("- 단가는 4만원입니다. [DOC 1]"))

    def test_plain_text_with_citation_is_allowed(self):
        module = _load_module()
        self.assertFalse(module.contains_disallowed_markdown("단가는 [4만원]입니다. [DOC 1]"))


if __name__ == "__main__":
    unittest.main()
