import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault(
    "hnswlib",
    SimpleNamespace(Index=type("DummyIndex", (), {})),
)
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=object),
)

from src.rag import delete_file_from_kb, get_kb_files


def _init_meta_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE files (
            file_id TEXT PRIMARY KEY,
            source_path TEXT UNIQUE,
            orig_name TEXT,
            stored_path TEXT,
            uploaded_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE source_uploads (
            source_path TEXT PRIMARY KEY,
            original_filename TEXT
        )
        """
    )
    conn.commit()
    conn.close()


class FileDisplayNameTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = Path(self.tmpdir.name) / "kb"
        self.kb_dir = self.data_dir / "alpha"
        self.upload_dir = self.kb_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.kb_dir / "meta.sqlite"
        _init_meta_db(self.db_path)

    def test_kb_file_listing_uses_original_name_for_display(self):
        stored_one = self.upload_dir / "20260330_101010_a1b2c3_report.pdf"
        stored_two = self.upload_dir / "20260330_101120_d4e5f6_report.pdf"
        stored_one.write_text("alpha", encoding="utf-8")
        stored_two.write_text("beta", encoding="utf-8")

        conn = sqlite3.connect(str(self.db_path))
        conn.executemany(
            """
            INSERT INTO files (file_id, source_path, orig_name, stored_path, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("file-1", "doc-a.pdf", "2024농가경제조사지침서.pdf", str(stored_one), 100),
                ("file-2", "doc-b.pdf", "2024농가경제조사지침서.pdf", str(stored_two), 200),
            ],
        )
        conn.executemany(
            """
            INSERT INTO source_uploads (source_path, original_filename)
            VALUES (?, ?)
            """,
            [
                ("doc-a.pdf", "2024농가경제조사지침서.pdf"),
                ("doc-b.pdf", "2024농가경제조사지침서.pdf"),
            ],
        )
        conn.commit()
        conn.close()

        files = get_kb_files("alpha", data_dir=str(self.data_dir))

        self.assertEqual([entry["display_name"] for entry in files], ["2024농가경제조사지침서.pdf", "2024농가경제조사지침서.pdf"])
        self.assertEqual([entry["file_id"] for entry in files], ["file-2", "file-1"])
        self.assertTrue(all(entry["stored_name"].startswith("20260330_") for entry in files))
        self.assertTrue(all(entry["stored_name"] != entry["display_name"] for entry in files))

    def test_duplicate_original_names_stay_deletable_via_file_id(self):
        stored_one = self.upload_dir / "20260330_keep.pdf"
        stored_two = self.upload_dir / "20260330_delete.pdf"
        stored_one.write_text("keep", encoding="utf-8")
        stored_two.write_text("delete", encoding="utf-8")

        conn = sqlite3.connect(str(self.db_path))
        conn.executemany(
            """
            INSERT INTO files (file_id, source_path, orig_name, stored_path, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("file-1", "doc-a.pdf", "중복문서.pdf", str(stored_one), 100),
                ("file-2", "doc-b.pdf", "중복문서.pdf", str(stored_two), 200),
            ],
        )
        conn.executemany(
            """
            INSERT INTO source_uploads (source_path, original_filename)
            VALUES (?, ?)
            """,
            [
                ("doc-a.pdf", "중복문서.pdf"),
                ("doc-b.pdf", "중복문서.pdf"),
            ],
        )
        conn.commit()
        conn.close()

        delete_file_from_kb("alpha", "file-2", data_dir=str(self.data_dir))
        files = get_kb_files("alpha", data_dir=str(self.data_dir))

        self.assertTrue(stored_one.exists())
        self.assertFalse(stored_two.exists())
        self.assertEqual([entry["file_id"] for entry in files], ["file-1"])
        self.assertEqual(files[0]["display_name"], "중복문서.pdf")


if __name__ == "__main__":
    unittest.main()
