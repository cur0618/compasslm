import unittest
from pathlib import Path
from types import SimpleNamespace

from src.retrieval_quality import apply_critical_term_gate
from src.retrieval_quality import rerank_results_for_grounded_answer
from src.retrieval_quality import select_auto_prefetch_documents


ROOT = Path(__file__).resolve().parents[1]


class RetrievalQualityTests(unittest.TestCase):
    def test_direct_anchor_evidence_beats_high_scoring_off_topic_definition_fact(self):
        results = [
            {
                "id": 1,
                "score": 0.82,
                "text": "표의미: kind=definition_block | subject=Ⅱ. 현장조사 > 1. 농가경제조사의 목적은 무엇인가요? | 정의=농가소득 구성 현황을 파악한다.",
            },
            {
                "id": 2,
                "score": 0.73,
                "text": "Ⅰ. 조사개요 7. 조사대상 ○조사대상: 3,300가구 조사대상 가구 농가경제조사 제외가구 10a(1,000㎡) 이상의 농지를 직접 경작하는 가구 연간 직접 생산한 농축산물 판매액이 120만원 이상인 가구",
            },
            {
                "id": 3,
                "score": 0.80,
                "text": "대식물에서는 조사제외(조사표의 작물재배현황에는 조사함)",
            },
        ]

        ranked = rerank_results_for_grounded_answer(
            "농가경제조사의 조사제외 기준을 알려줄래?",
            results,
        )

        self.assertEqual(ranked[0]["id"], 2)

    def test_auto_prefetch_dedupes_same_source_page_and_mixes_casebook_for_procedure_question(self):
        records = [
            SimpleNamespace(
                doc_no=1,
                chunk_id=101,
                source_path="2024농가경제조사지침서.pdf",
                source_ref="2024농가경제조사지침서.pdf / PDF page 79",
                text="포도 40kg 판매 금액",
                score=0.91,
                metadata={"raw_row": {"page": 79, "doc_role": "guide"}},
            ),
            SimpleNamespace(
                doc_no=2,
                chunk_id=102,
                source_path="2024농가경제조사지침서.pdf",
                source_ref="2024농가경제조사지침서.pdf / PDF page 79",
                text="포도 40kg 판매 금액 중복 chunk",
                score=0.88,
                metadata={"raw_row": {"page": 79, "doc_role": "guide"}},
            ),
            SimpleNamespace(
                doc_no=3,
                chunk_id=201,
                source_path="2024농가경제조사사례집.pdf",
                source_ref="2024농가경제조사사례집.pdf / PDF page 812",
                text="농가에서 포도를 판매하고 일부 가공하여 판매한 경우 처리방법 사례",
                score=0.74,
                metadata={"raw_row": {"page": 812, "doc_role": "casebook"}},
            ),
        ]

        selected = select_auto_prefetch_documents(
            "농가에서 포도 40kg을 판매하고 40kg은 가공하여 판매한 경우의 처리방법은?",
            records,
            limit=2,
        )

        self.assertEqual([record.doc_no for record in selected], [1, 3])

    def test_main_dedupe_key_uses_evidence_fingerprint_before_chunk_id(self):
        source = ROOT / "src" / "main.py"
        text = source.read_text(encoding="utf-8")

        self.assertIn("_result_evidence_fingerprint", text)
        key_body = text[text.index("def _result_dedupe_key") : text.index("def _merge_search_candidates")]
        self.assertLess(key_body.index("_result_evidence_fingerprint"), key_body.index("chunk_id"))

    def test_critical_term_gate_blocks_off_topic_casebook_result(self):
        results = [
            {
                "id": 1,
                "score": 0.88,
                "doc_role": "casebook",
                "text": "Q13 태양열 발전기를 설치한 경우의 처리방법은? 태양열 발전기는 기타 구축물로 조사합니다.",
            }
        ]

        filtered, meta = apply_critical_term_gate(
            "자가소유 폐건물은 어떻게 조사해야해?",
            results,
        )

        self.assertEqual(filtered, [])
        self.assertFalse(meta["critical_term_gate_passed"])
        self.assertIn("폐건물", meta["critical_terms"])

    def test_critical_term_gate_keeps_matching_title_result(self):
        results = [
            {
                "id": 2,
                "score": 0.79,
                "doc_role": "casebook",
                "section": "Q21 폐건물을 소유한 경우의 처리방법은?",
                "text": "Q21 폐건물을 소유한 경우의 처리방법은? 실제 사용 여부와 철거 가능성을 확인하여 조사합니다.",
            }
        ]

        filtered, meta = apply_critical_term_gate(
            "자가소유 폐건물은 어떻게 조사해야해?",
            results,
        )

        self.assertEqual([row["id"] for row in filtered], [2])
        self.assertTrue(meta["critical_term_gate_passed"])
        self.assertEqual(meta["literal_title_hit_count"], 1)

    def test_critical_term_gate_blocks_normalized_only_without_raw_backing(self):
        results = [
            {
                "id": 3,
                "score": 0.93,
                "is_normalized": 1,
                "normalized_group": "txt",
                "source_path": "__normalized_txt_guide__",
                "text": "[통합정리-TXT]\n폐건물 처리 관련 요약: 실제 이용 여부를 확인합니다.",
            }
        ]

        filtered, meta = apply_critical_term_gate(
            "자가소유 폐건물은 어떻게 조사해야해?",
            results,
        )

        self.assertEqual(filtered, [])
        self.assertTrue(meta["normalized_only_blocked"])
        self.assertFalse(meta["critical_term_gate_passed"])

    def test_critical_term_gate_does_not_apply_to_broad_summary_request(self):
        results = [
            {
                "id": 4,
                "score": 0.51,
                "text": "25년 2월 7일은 날씨가 맑았음",
            }
        ]

        filtered, meta = apply_critical_term_gate("중요한 내용을 정리해줘", results)

        self.assertEqual([row["id"] for row in filtered], [4])
        self.assertEqual(meta["critical_terms"], [])
        self.assertTrue(meta["critical_term_gate_passed"])


if __name__ == "__main__":
    unittest.main()
