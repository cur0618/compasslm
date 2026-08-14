import tempfile
import unittest
from pathlib import Path

from src.chat_store import ChatStore


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


if __name__ == "__main__":
    unittest.main()
