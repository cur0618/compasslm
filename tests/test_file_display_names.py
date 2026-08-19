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
sys.modules.setdefault(
    "src.pdf_ocr",
    SimpleNamespace(
        extract_pdf_pages=lambda *args, **kwargs: {},
        extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
        release_cached_ocr_model=lambda *args, **kwargs: None,
        shutdown_persistent_ocr_worker=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "src.utils",
    SimpleNamespace(
        chunk_txt_items=lambda *args, **kwargs: [],
        chunk_xlsx_rows=lambda *args, **kwargs: [],
        load_txt=lambda *args, **kwargs: [],
        load_xlsx=lambda *args, **kwargs: [],
    ),
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


    def test_delete_cascades_search_derived_wiki_and_cache_rows(self):
        target_file = self.upload_dir / "target.pdf"
        keep_file = self.upload_dir / "keep.pdf"
        target_file.write_text("target", encoding="utf-8")
        keep_file.write_text("keep", encoding="utf-8")
        cache_dir = self.kb_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_cache = cache_dir / "target.items.pkl"
        target_vector = cache_dir / "target.large.npy"
        target_cache.write_bytes(b"cache")
        target_vector.write_bytes(b"vector")

        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(
            """
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_path TEXT, source_type TEXT, is_normalized INTEGER DEFAULT 0);
            CREATE TABLE chunk_vec (chunk_pk INTEGER, index_name TEXT);
            CREATE TABLE chunk_fts (text TEXT);
            CREATE TABLE chunk_concept_edges (chunk_pk INTEGER, concept_id INTEGER);
            CREATE TABLE concept_nodes (concept_id INTEGER PRIMARY KEY, normalized_key TEXT);
            CREATE TABLE file_cache (
                source_path TEXT PRIMARY KEY,
                items_cache_path TEXT,
                embeddings_cache_path TEXT,
                emb_small_cache_path TEXT,
                emb_large_cache_path TEXT
            );
            CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_path TEXT, file_id TEXT);
            CREATE TABLE doc_blocks (doc_id TEXT);
            CREATE TABLE doc_table_cells (doc_id TEXT);
            CREATE TABLE ontology_facts (fact_id INTEGER PRIMARY KEY);
            CREATE TABLE ontology_fact_sources (
                source_id INTEGER PRIMARY KEY,
                fact_id INTEGER,
                chunk_id INTEGER,
                source_path TEXT
            );
            CREATE TABLE ontology_fact_feedback (fact_id INTEGER);
            CREATE TABLE ontology_fact_history (fact_id INTEGER);
            CREATE TABLE wiki_pages (page_id INTEGER PRIMARY KEY, page_type TEXT);
            CREATE TABLE wiki_page_sources (
                source_id INTEGER PRIMARY KEY,
                page_id INTEGER,
                source_path TEXT,
                chunk_id INTEGER
            );
            CREATE TABLE wiki_links (from_page_id INTEGER);
            CREATE TABLE wiki_claims (page_id INTEGER);
            CREATE TABLE wiki_update_log (page_id INTEGER);
            CREATE TABLE wiki_lint_findings (page_id INTEGER);
            """
        )
        conn.executemany(
            """
            INSERT INTO files (file_id, source_path, orig_name, stored_path, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("target-file", "target.pdf", "target.pdf", str(target_file), 100),
                ("keep-file", "keep.pdf", "keep.pdf", str(keep_file), 100),
            ],
        )
        conn.executemany(
            "INSERT INTO source_uploads (source_path, original_filename) VALUES (?, ?)",
            [("target.pdf", "target.pdf"), ("keep.pdf", "keep.pdf")],
        )
        conn.executemany(
            "INSERT INTO chunks (id, source_path, source_type, is_normalized) VALUES (?, ?, ?, 0)",
            [(1, "target.pdf", "pdf"), (2, "keep.pdf", "pdf")],
        )
        conn.executemany(
            "INSERT INTO chunk_vec (chunk_pk, index_name) VALUES (?, 'large')",
            [(1,), (2,)],
        )
        conn.executemany(
            "INSERT INTO chunk_fts (rowid, text) VALUES (?, ?)",
            [(1, "target"), (2, "keep")],
        )
        conn.executemany(
            "INSERT INTO concept_nodes (concept_id, normalized_key) VALUES (?, ?)",
            [(10, "shared"), (11, "target-only")],
        )
        conn.executemany(
            "INSERT INTO chunk_concept_edges (chunk_pk, concept_id) VALUES (?, ?)",
            [(1, 10), (2, 10), (1, 11)],
        )
        conn.execute(
            """
            INSERT INTO file_cache
                (source_path, items_cache_path, embeddings_cache_path, emb_small_cache_path, emb_large_cache_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("target.pdf", str(target_cache), "", "", str(target_vector)),
        )
        conn.executemany(
            "INSERT INTO documents (doc_id, source_path, file_id) VALUES (?, ?, ?)",
            [("target-doc", "target.pdf", "target-file"), ("keep-doc", "keep.pdf", "keep-file")],
        )
        conn.execute("INSERT INTO doc_blocks (doc_id) VALUES ('target-doc')")
        conn.execute("INSERT INTO doc_table_cells (doc_id) VALUES ('target-doc')")
        conn.executemany("INSERT INTO ontology_facts (fact_id) VALUES (?)", [(20,), (21,)])
        conn.executemany(
            """
            INSERT INTO ontology_fact_sources (source_id, fact_id, chunk_id, source_path)
            VALUES (?, ?, ?, ?)
            """,
            [
                (1, 20, 1, "target.pdf"),
                (2, 20, 2, "keep.pdf"),
                (3, 21, 1, "target.pdf"),
            ],
        )
        conn.execute("INSERT INTO ontology_fact_feedback (fact_id) VALUES (21)")
        conn.execute("INSERT INTO ontology_fact_history (fact_id) VALUES (21)")
        conn.execute("INSERT INTO wiki_pages (page_id, page_type) VALUES (30, 'source')")
        conn.execute(
            """
            INSERT INTO wiki_page_sources (source_id, page_id, source_path, chunk_id)
            VALUES (1, 30, 'target.pdf', 1)
            """
        )
        conn.execute("INSERT INTO wiki_links (from_page_id) VALUES (30)")
        conn.execute("INSERT INTO wiki_claims (page_id) VALUES (30)")
        conn.execute("INSERT INTO wiki_update_log (page_id) VALUES (30)")
        conn.execute("INSERT INTO wiki_lint_findings (page_id) VALUES (30)")
        conn.commit()
        conn.close()

        delete_file_from_kb("alpha", "target-file", data_dir=str(self.data_dir))

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks WHERE id = 1").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunk_vec WHERE chunk_pk = 1").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE rowid = 1").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunk_concept_edges WHERE chunk_pk = 1").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM concept_nodes WHERE concept_id = 11").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM concept_nodes WHERE concept_id = 10").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ontology_fact_sources WHERE source_id = 1").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ontology_facts WHERE fact_id = 20").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ontology_facts WHERE fact_id = 21").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM wiki_pages WHERE page_id = 30").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM file_cache WHERE source_path = 'target.pdf'").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks WHERE id = 2").fetchone()[0], 1)
        finally:
            conn.close()

        self.assertFalse(target_file.exists())
        self.assertTrue(keep_file.exists())
        self.assertFalse(target_cache.exists())
        self.assertFalse(target_vector.exists())


if __name__ == "__main__":
    unittest.main()
