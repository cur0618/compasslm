import unittest

from src.ontology_extractor import (
    coerce_limited_llm_facts_from_chunk,
    extract_deterministic_facts_from_text,
    validate_limited_llm_fact_payload,
)


class OntologyExtractorTests(unittest.TestCase):
    def test_llm_payload_keeps_only_facts_with_quotes_present_in_chunk(self):
        chunk_text = "농가경제조사의 지급단가는 40천원이며 지급횟수는 12회입니다."
        payload = [
            {
                "subject": "농가경제조사",
                "predicate": "지급단가",
                "object": "40천원",
                "confidence": 0.86,
                "evidence_quote": "지급단가는 40천원",
            },
            {
                "subject": "농가경제조사",
                "predicate": "지급대상",
                "object": "2인 이상",
                "confidence": 0.91,
                "evidence_quote": "2인 이상",
            },
        ]

        facts = coerce_limited_llm_facts_from_chunk(payload, chunk_text, min_confidence=0.62)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["predicate"], "지급단가")
        self.assertEqual(facts[0]["object_value"], "40천원")
        self.assertEqual(facts[0]["status"], "active")
        self.assertEqual(facts[0]["evidence_quote"], "지급단가는 40천원")

    def test_low_confidence_llm_fact_is_kept_for_review_when_quote_is_grounded(self):
        chunk_text = "농가경제조사의 지급단가는 40천원입니다."
        payload = [
            {
                "subject": "농가경제조사",
                "predicate": "지급단가",
                "object": "40천원",
                "confidence": 0.55,
                "evidence_quote": "지급단가는 40천원",
            }
        ]

        facts = coerce_limited_llm_facts_from_chunk(payload, chunk_text, min_confidence=0.62)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["status"], "needs_review")

    def test_validator_rejects_payload_without_required_spo_shape(self):
        payload = [{"subject": "농가경제조사", "predicate": "", "object": "40천원", "confidence": 0.8}]

        facts = validate_limited_llm_fact_payload(payload)

        self.assertEqual(facts, [])

    def test_structure_exception_chunk_creates_grounded_deterministic_fact(self):
        text = "다만 해외 농가는 조사대상에서 제외한다."

        facts = extract_deterministic_facts_from_text(
            text,
            chunk_kind="exception",
            heading_path=["제1장 총칙", "제2조 조사대상"],
        )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["subject"], "제2조 조사대상")
        self.assertEqual(facts[0]["predicate"], "예외")
        self.assertEqual(facts[0]["object_value"], text)
        self.assertEqual(facts[0]["evidence_quote"], text)
        self.assertEqual(facts[0]["extraction_method"], "deterministic_structure_fact")

    def test_structure_fact_is_not_created_for_derived_chunk(self):
        facts = extract_deterministic_facts_from_text(
            "다만 해외 농가는 제외한다.",
            chunk_kind="exception",
            heading_path=["제2조 조사대상"],
            is_derived=True,
        )

        self.assertEqual(facts, [])


if __name__ == "__main__":
    unittest.main()
