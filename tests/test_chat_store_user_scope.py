import tempfile
import unittest
from pathlib import Path

from src.chat_store import ChatStore, is_failed_history_answer_text


class ChatStoreUserScopeTests(unittest.TestCase):
    def test_chat_history_can_be_partitioned_by_user_id(self):
        with tempfile.TemporaryDirectory() as td:
            store = ChatStore(str(Path(td) / "app.sqlite"), history_limit=20)

            store.append_chat_message("browser-session", "shared-kb", "user", "alice private", user_id="alice")
            store.append_chat_message("browser-session", "shared-kb", "user", "bob private", user_id="bob")

            self.assertEqual(
                store.get_chat_history("browser-session", "shared-kb", user_id="alice"),
                [{"role": "user", "text": "alice private"}],
            )
            self.assertEqual(
                store.get_chat_history("browser-session", "shared-kb", user_id="bob"),
                [{"role": "user", "text": "bob private"}],
            )

    def test_recent_agent_runs_can_be_partitioned_by_user_id(self):
        with tempfile.TemporaryDirectory() as td:
            store = ChatStore(str(Path(td) / "app.sqlite"), agent_run_limit=20)

            store.append_agent_run("s", "kb", "q1", "alice q", "alice a", b"[]", user_id="alice")
            store.append_agent_run("s", "kb", "q2", "bob q", "bob a", b"[]", user_id="bob")

            alice_runs = store.get_recent_agent_runs(kb_name="kb", user_id="alice")
            bob_runs = store.get_recent_agent_runs(kb_name="kb", user_id="bob")

            self.assertEqual([run["query_id"] for run in alice_runs], ["q1"])
            self.assertEqual([run["query_id"] for run in bob_runs], ["q2"])

    def test_failed_no_evidence_answer_text_is_marked_for_prompt_history_filtering(self):
        self.assertTrue(
            is_failed_history_answer_text(
                "\ubb38\uc11c \uadfc\uac70\uac00 \ubd80\uc871\ud574 \ud655\uc815\uc801\uc73c\ub85c \uc548\ub0b4\ud558\uae30 \uc5b4\ub835\uc2b5\ub2c8\ub2e4."
            )
        )
        self.assertFalse(
            is_failed_history_answer_text(
                "\ucc98\ub9ac: \uc815\uae30\uc801\uc73c\ub85c \ubc1b\ub294 \uc218\ub2f9\uc740 \uc218\uc785\uc73c\ub85c \uc870\uc0ac\ud569\ub2c8\ub2e4. [DOC 1]"
            )
        )


if __name__ == "__main__":
    unittest.main()
