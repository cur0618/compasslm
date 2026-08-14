import tempfile
import time
import unittest
from pathlib import Path

from src.auth_store import AuthStore, build_scoped_kb_id, hash_password, verify_password


class AuthStoreTests(unittest.TestCase):
    def test_password_hash_uses_pbkdf2_and_verifies(self):
        encoded = hash_password("correct horse battery staple")

        self.assertTrue(encoded.startswith("pbkdf2_sha256$"))
        self.assertNotIn("correct horse", encoded)
        self.assertTrue(verify_password("correct horse battery staple", encoded))
        self.assertFalse(verify_password("wrong", encoded))

    def test_user_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            user = store.create_user(
                login_id="admin",
                password="replace-with-strong-secret",
                display_name="Admin",
                role="admin",
            )

            self.assertEqual(user["login_id"], "admin")
            self.assertEqual(user["role"], "admin")
            self.assertIsNotNone(store.authenticate("admin", "replace-with-strong-secret"))
            self.assertIsNone(store.authenticate("admin", "bad-password"))

            token = store.create_session(user["user_id"], ttl_seconds=60)
            session_user = store.get_user_by_session(token)
            self.assertIsNotNone(session_user)
            self.assertEqual(session_user["user_id"], user["user_id"])

            store.revoke_session(token)
            self.assertIsNone(store.get_user_by_session(token))

    def test_duplicate_login_ids_are_rejected_for_registration(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            store.create_user("writer", "pw", display_name="Writer")

            with self.assertRaises(ValueError) as ctx:
                store.create_user("writer", "pw2", display_name="Other")

            self.assertIn("login_id already exists", str(ctx.exception))

    def test_expired_or_inactive_sessions_do_not_authenticate(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            user = store.create_user("writer", "pw", role="user")
            expired = store.create_session(user["user_id"], ttl_seconds=-1)
            self.assertIsNone(store.get_user_by_session(expired, now_ts=int(time.time()) + 10))

            active = store.create_session(user["user_id"], ttl_seconds=60)
            store.set_user_active(user["user_id"], False)
            self.assertIsNone(store.get_user_by_session(active))

    def test_kbs_are_listed_only_for_their_owner(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            admin = store.create_user("admin", "pw", role="admin")
            alice = store.create_user("alice", "pw")
            bob = store.create_user("bob", "pw")

            admin_kb = store.create_kb(admin["user_id"], "legacy", internal_kb_id="legacy")
            alice_kb = store.create_kb(alice["user_id"], "지침서")
            bob_kb = store.create_kb(bob["user_id"], "지침서")

            self.assertEqual([kb["display_name"] for kb in store.list_kbs(alice["user_id"])], ["지침서"])
            self.assertEqual(store.get_kb(alice["user_id"], "지침서")["kb_id"], alice_kb["kb_id"])
            self.assertIsNone(store.get_kb(alice["user_id"], "legacy"))
            self.assertIsNone(store.get_kb(alice["user_id"], bob_kb["display_name"], internal_kb_id=bob_kb["internal_kb_id"]))
            self.assertEqual(store.get_kb(admin["user_id"], "legacy")["kb_id"], admin_kb["kb_id"])

    def test_rename_kb_rejects_duplicate_display_name_without_sqlite_exception(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            user = store.create_user("alice", "pw")
            store.create_kb(user["user_id"], "지침서")
            store.create_kb(user["user_id"], "사례집")

            with self.assertRaises(ValueError) as ctx:
                store.rename_kb(user["user_id"], "지침서", "사례집")

            self.assertIn("kb display_name already exists", str(ctx.exception))
            self.assertEqual([kb["display_name"] for kb in store.list_kbs(user["user_id"])], ["사례집", "지침서"])
            self.assertIsNotNone(store.authenticate("alice", "pw"))

    def test_rename_kb_returns_none_when_old_display_name_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            user = store.create_user("alice", "pw")
            store.create_kb(user["user_id"], "지침서")

            self.assertIsNone(store.rename_kb(user["user_id"], "없는공간", "새이름"))
            self.assertEqual([kb["display_name"] for kb in store.list_kbs(user["user_id"])], ["지침서"])

    def test_legacy_kb_sync_reuses_existing_internal_id_after_display_rename(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            admin = store.create_user("admin", "pw", role="admin")
            original = store.create_kb(admin["user_id"], "default", internal_kb_id="default")
            renamed = store.rename_kb(admin["user_id"], "default", "운영 지침서")

            synced = store.create_kb(admin["user_id"], "default", internal_kb_id="default")

            self.assertEqual(synced["kb_id"], original["kb_id"])
            self.assertEqual(synced["display_name"], "운영 지침서")
            self.assertEqual(synced["internal_kb_id"], "default")
            self.assertEqual(renamed["kb_id"], original["kb_id"])
            self.assertEqual([kb["display_name"] for kb in store.list_kbs(admin["user_id"])], ["운영 지침서"])

    def test_create_kb_rejects_internal_id_owned_by_another_user_without_leaking_lock(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            alice = store.create_user("alice", "pw")
            bob = store.create_user("bob", "pw")
            store.create_kb(alice["user_id"], "legacy", internal_kb_id="legacy")

            with self.assertRaises(ValueError) as ctx:
                store.create_kb(bob["user_id"], "legacy", internal_kb_id="legacy")

            self.assertIn("kb internal_kb_id already exists", str(ctx.exception))
            self.assertIsNotNone(store.authenticate("bob", "pw"))

    def test_admin_legacy_kb_sync_skips_internal_ids_owned_by_other_users(self):
        with tempfile.TemporaryDirectory() as td:
            store = AuthStore(str(Path(td) / "app.sqlite"))
            admin = store.create_user("admin", "pw", role="admin")
            alice = store.create_user("alice", "pw")
            store.create_kb(alice["user_id"], "농가경제조사", internal_kb_id="농가경제조사")

            synced = store.ensure_legacy_kbs_for_admin(admin["user_id"], ["default", "농가경제조사"])

            self.assertEqual([kb["display_name"] for kb in synced], ["default"])
            self.assertEqual([kb["display_name"] for kb in store.list_kbs(admin["user_id"])], ["default"])
            self.assertEqual([kb["display_name"] for kb in store.list_kbs(alice["user_id"])], ["농가경제조사"])

    def test_scoped_kb_ids_reject_path_traversal_and_are_stable(self):
        first = build_scoped_kb_id("user-123", "지침서 2026")
        second = build_scoped_kb_id("user-123", "지침서 2026")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("user_123__"))
        self.assertNotIn("/", first)
        self.assertNotIn("..", first)
        with self.assertRaises(ValueError):
            build_scoped_kb_id("user-123", "../secret")


if __name__ == "__main__":
    unittest.main()
