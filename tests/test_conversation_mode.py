import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "conversation_mode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_test_conversation_mode", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConversationModeTests(unittest.TestCase):
    def test_contextual_followup_phrase_is_detected(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        self.assertTrue(module.is_contextual_followup_message("위에서 말한 것 중에 일부만 적용되는 경우는?"))
        self.assertFalse(module.is_contextual_followup_message("주소?"))

    def test_contextual_followup_inherits_document_mode(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "위에서 말한 것 중에 일부만 적용되는 경우는?",
            kb_has_docs=True,
            last_active_mode="document_qa",
        )
        self.assertEqual(decision.mode, "document_qa")
        self.assertEqual(decision.reason, "inherited_from_last_assistant_mode")

    def test_contextual_followup_inherits_casual_mode(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "그건?",
            kb_has_docs=True,
            last_active_mode="casual_chat",
        )
        self.assertEqual(decision.mode, "casual_chat")
        self.assertEqual(decision.reason, "inherited_from_last_assistant_mode")

    def test_live_info_request_routes_to_casual_even_when_kb_has_docs(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "오늘 날씨 어때?",
            kb_has_docs=True,
            last_active_mode=None,
        )
        self.assertEqual(decision.mode, "casual_chat")
        self.assertEqual(decision.reason, "live_info_request")

    def test_explicit_document_query_overrides_previous_casual_mode(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "농가경제조사 단가 알려줘",
            kb_has_docs=True,
            last_active_mode="casual_chat",
        )
        self.assertEqual(decision.mode, "document_qa")
        self.assertEqual(decision.reason, "explicit_document_intent")

    def test_live_info_question_breaks_document_mode_inheritance(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "날씨?",
            kb_has_docs=True,
            last_active_mode="document_qa",
            followup_type="standalone",
        )
        self.assertEqual(decision.mode, "casual_chat")
        self.assertEqual(decision.reason, "live_info_request")

    def test_ambiguous_followup_without_anchor_does_not_force_kb(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        decision = module.resolve_conversation_mode(
            "그건?",
            kb_has_docs=True,
            last_active_mode=None,
        )
        self.assertEqual(decision.mode, "casual_chat")
        self.assertEqual(decision.reason, "standalone_without_history_anchor")

    def test_recent_agent_runs_determine_last_mode_and_casual_streak(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        rows = [
            {
                "run_id": 14,
                "metadata_json": json.dumps({"conversation_mode": "identity"}, ensure_ascii=False),
            },
            {
                "run_id": 13,
                "metadata_json": json.dumps({"conversation_mode": "casual_chat"}, ensure_ascii=False),
            },
            {
                "run_id": 12,
                "metadata_json": json.dumps({"conversation_mode": "casual_chat"}, ensure_ascii=False),
            },
            {
                "run_id": 11,
                "metadata_json": json.dumps({"conversation_mode": "document_qa"}, ensure_ascii=False),
            },
        ]

        state = module.summarize_recent_conversation_state(rows)
        self.assertEqual(state.last_active_mode, "casual_chat")
        self.assertEqual(state.mode_anchor_run_id, 13)
        self.assertEqual(state.casual_turn_streak, 2)

    def test_document_mode_can_force_followup_rewrite_for_short_question_words(self):
        self.assertTrue(MODULE_PATH.exists())
        if not MODULE_PATH.exists():
            return

        module = _load_module()
        self.assertTrue(module.should_force_followup_rewrite("얼마야?", last_active_mode="document_qa"))
        self.assertFalse(module.should_force_followup_rewrite("얼마야?", last_active_mode="casual_chat"))


if __name__ == "__main__":
    unittest.main()
