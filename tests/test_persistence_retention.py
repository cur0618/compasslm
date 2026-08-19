import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from scripts.prune_runtime_logs import prune_runtime_log_dirs
from src.persistence_retention import prune_timestamped_rows, rotate_file_if_oversize


class PersistenceRetentionTests(unittest.TestCase):
    def test_timestamped_rows_apply_age_and_count_limits(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE answer_logs (log_id INTEGER PRIMARY KEY, created_at INTEGER)"
        )
        conn.executemany(
            "INSERT INTO answer_logs (log_id, created_at) VALUES (?, ?)",
            [(1, 10), (2, 20), (3, 80), (4, 90)],
        )

        removed = prune_timestamped_rows(
            conn,
            table="answer_logs",
            id_column="log_id",
            timestamp_column="created_at",
            expire_before=50,
            max_rows=1,
            batch_size=10,
        )

        self.assertEqual(removed, 3)
        self.assertEqual(
            conn.execute("SELECT log_id FROM answer_logs ORDER BY log_id").fetchall(),
            [(4,)],
        )
        conn.close()

    def test_runtime_log_prune_only_removes_managed_timestamp_directories(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_dir = root / "20260701_010101"
            recent_dir = root / "20260731_010101"
            active_dir = root / "20260731_020202"
            unrelated_dir = root / "manual-notes"
            for directory in (old_dir, recent_dir, active_dir, unrelated_dir):
                directory.mkdir()
            old_time = time.time() - (40 * 86400)
            os.utime(old_dir, (old_time, old_time))

            removed = prune_runtime_log_dirs(
                root,
                retention_days=14,
                max_dirs=10,
                active_dir=active_dir,
                now=time.time(),
            )

            self.assertEqual(removed, [str(old_dir.resolve())])
            self.assertFalse(old_dir.exists())
            self.assertTrue(recent_dir.exists())
            self.assertTrue(active_dir.exists())
            self.assertTrue(unrelated_dir.exists())

    def test_jsonl_rotation_keeps_a_bounded_number_of_backups(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rag_trace.jsonl"
            log_path.write_text("x" * 20, encoding="utf-8")

            self.assertTrue(
                rotate_file_if_oversize(
                    log_path,
                    max_bytes=10,
                    backup_count=2,
                )
            )
            self.assertFalse(log_path.exists())
            self.assertTrue(Path(f"{log_path}.1").exists())

            log_path.write_text("y" * 20, encoding="utf-8")
            rotate_file_if_oversize(log_path, max_bytes=10, backup_count=2)
            self.assertTrue(Path(f"{log_path}.1").exists())
            self.assertTrue(Path(f"{log_path}.2").exists())


if __name__ == "__main__":
    unittest.main()
