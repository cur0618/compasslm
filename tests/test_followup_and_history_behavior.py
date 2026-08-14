import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault(
    "pydantic_ai.messages",
    SimpleNamespace(ModelMessage=object, ModelMessagesTypeAdapter=SimpleNamespace(validate_json=lambda payload: [])),
)

from src.chat_store import ChatStore


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_MODULE_PATH = ROOT / "src" / "compass_ai" / "settings.py"
QUERY_REWRITE_MODULE_PATH = ROOT / "src" / "query_rewrite.py"


def _load_settings_module():
    spec = importlib.util.spec_from_file_location("codex_test_compass_ai_settings", str(SETTINGS_MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_query_rewrite_module():
    spec = importlib.util.spec_from_file_location("codex_test_query_rewrite", str(QUERY_REWRITE_MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HistoryDefaultsTests(unittest.TestCase):
    def test_settings_default_compact_history_turn_limit_is_10(self):
        with patch.dict(os.environ, {}, clear=True):
            settings_module = _load_settings_module()
            settings = settings_module.CompassAISettings.from_env()
        self.assertEqual(settings.compact_history_turn_limit, 10)

    def test_new_chat_sessions_enable_history_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChatStore(Path(tmpdir) / "chat.sqlite")
            session = store.ensure_session("session-a")

        self.assertTrue(session["history_enabled"])

    def test_chat_history_remains_kb_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChatStore(Path(tmpdir) / "chat.sqlite")
            store.set_history_enabled("session-a", True)
            store.append_chat_message("session-a", "kb-alpha", "user", "alpha 질문")
            store.append_chat_message("session-a", "kb-alpha", "assistant", "alpha 답변")
            store.append_chat_message("session-a", "kb-beta", "user", "beta 질문")

            alpha_rows = store.get_chat_history("session-a", "kb-alpha")
            beta_rows = store.get_chat_history("session-a", "kb-beta")

        self.assertEqual([row["text"] for row in alpha_rows], ["alpha 질문", "alpha 답변"])
        self.assertEqual([row["text"] for row in beta_rows], ["beta 질문"])


class FollowupContractTests(unittest.TestCase):
    def test_models_define_followup_analysis(self):
        source = (ROOT / "src" / "compass_ai" / "models.py").read_text(encoding="utf-8")
        self.assertIn("class FollowupAnalysis", source)

    def test_service_exposes_followup_rewrite_diagnostic(self):
        source = (ROOT / "src" / "compass_ai" / "service.py").read_text(encoding="utf-8")
        self.assertIn("async def followup_rewrite_diagnostic", source)


class FollowupBehaviorTests(unittest.TestCase):
    def test_short_correction_followup_is_detected_and_rewritten_query_is_preferred(self):
        self.assertTrue(QUERY_REWRITE_MODULE_PATH.exists())
        if not QUERY_REWRITE_MODULE_PATH.exists():
            return

        module = _load_query_rewrite_module()
        analysis = SimpleNamespace(
            followup_type="correction",
            rewritten_query="양봉장에 환풍기를 설치했을 때 처리 방법",
            should_use_history=True,
            is_small_talk=False,
        )

        self.assertTrue(module.should_attempt_followup_rewrite("태양광이 아니라 환풍기였어"))
        self.assertEqual(
            module.resolve_effective_query("태양광이 아니라 환풍기였어", analysis),
            "양봉장에 환풍기를 설치했을 때 처리 방법",
        )

    def test_small_talk_followup_does_not_override_user_query(self):
        self.assertTrue(QUERY_REWRITE_MODULE_PATH.exists())
        if not QUERY_REWRITE_MODULE_PATH.exists():
            return

        module = _load_query_rewrite_module()
        analysis = SimpleNamespace(
            followup_type="small_talk",
            rewritten_query="",
            should_use_history=False,
            is_small_talk=True,
        )

        self.assertFalse(module.should_attempt_followup_rewrite("고마워"))
        self.assertEqual(module.resolve_effective_query("고마워", analysis), "고마워")


if __name__ == "__main__":
    unittest.main()
