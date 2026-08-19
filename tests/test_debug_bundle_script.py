import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "project-gpu" / "collect_debug_bundle.sh"


def _create_app_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chat_messages
        (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kb_name TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE agent_runs
        (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            session_id TEXT NOT NULL,
            kb_name TEXT NOT NULL,
            user_message TEXT,
            answer_text TEXT,
            new_messages_json BLOB NOT NULL,
            metadata_json TEXT,
            response_quality_issue TEXT,
            usage_json TEXT,
            request_count INTEGER NOT NULL DEFAULT 0,
            tool_call_count INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            context_chars INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_app_activity(
    db_path: Path,
    *,
    session_id: str,
    kb_name: str,
    query_id: str,
    user_text: str,
    answer_text: str,
    created_at: int,
):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO chat_messages (session_id, kb_name, role, text, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, kb_name, "user", user_text, created_at),
    )
    conn.execute(
        """
        INSERT INTO chat_messages (session_id, kb_name, role, text, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, kb_name, "assistant", answer_text, created_at + 1),
    )
    conn.execute(
        """
        INSERT INTO agent_runs
        (
            query_id, session_id, kb_name, user_message, answer_text, new_messages_json,
            metadata_json, response_quality_issue, usage_json, request_count, tool_call_count,
            input_tokens, output_tokens, context_chars, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            session_id,
            kb_name,
            user_text,
            answer_text,
            "[]",
            json.dumps({"conversation_mode": "document_grounded"}, ensure_ascii=False),
            "",
            json.dumps({"requests": 1}, ensure_ascii=False),
            1,
            1,
            120,
            80,
            640,
            created_at + 1,
        ),
    )
    conn.commit()
    conn.close()


def _create_kb_meta_db(
    db_path: Path,
    *,
    query_id: str,
    session_id: str,
    query_text: str,
    answer_text: str,
    created_at: int,
):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE retrieval_logs
        (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            user_id TEXT,
            query_text TEXT,
            topk_ids_json TEXT,
            meta_json TEXT,
            created_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE answer_logs
        (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            llm_model TEXT,
            prompt_hash TEXT,
            answer_text TEXT,
            citations_json TEXT,
            answer_meta_json TEXT,
            created_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE wiki_saved_answers
        (
            saved_answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            answer_log_id INTEGER,
            user_id TEXT,
            question_text TEXT,
            answer_text TEXT,
            answer_summary TEXT,
            citation_json TEXT,
            status TEXT,
            quality_flags_json TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE wiki_answer_feedback
        (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_answer_id INTEGER,
            answer_log_id INTEGER,
            query_id TEXT,
            user_id TEXT,
            feedback_type TEXT,
            saved_to_wiki INTEGER,
            created_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_entities
        (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id TEXT,
            display_text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_facts
        (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id TEXT,
            subject_entity_id INTEGER,
            predicate TEXT,
            object_entity_id INTEGER,
            object_value TEXT,
            fact_kind TEXT,
            extraction_method TEXT,
            confidence REAL,
            status TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ontology_fact_sources
        (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id INTEGER,
            chunk_id INTEGER,
            source_path TEXT,
            source_ref TEXT,
            page_no INTEGER,
            line_start INTEGER,
            line_end INTEGER,
            evidence_quote TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO retrieval_logs (query_id, user_id, query_text, topk_ids_json, meta_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            session_id,
            query_text,
            json.dumps([101, 102]),
            json.dumps({"top1": 0.77}, ensure_ascii=False),
            created_at + 1,
        ),
    )
    conn.execute(
        """
        INSERT INTO answer_logs (query_id, llm_model, prompt_hash, answer_text, citations_json, answer_meta_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            "qwen-test",
            "prompt-hash-1",
            answer_text,
            json.dumps([{"label": "[1]"}], ensure_ascii=False),
            json.dumps({"grounded": True}, ensure_ascii=False),
            created_at + 2,
        ),
    )
    conn.execute(
        """
        INSERT INTO wiki_saved_answers
            (saved_answer_id, query_id, answer_log_id, user_id, question_text, answer_text,
             answer_summary, citation_json, status, quality_flags_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            query_id,
            1,
            session_id,
            query_text,
            answer_text,
            "신고된 답변 요약",
            json.dumps([{"chunk_id": 101, "label": "[1]"}], ensure_ascii=False),
            "reported",
            json.dumps(["reported"], ensure_ascii=False),
            created_at + 3,
            created_at + 3,
        ),
    )
    conn.execute(
        """
        INSERT INTO wiki_answer_feedback
            (saved_answer_id, answer_log_id, query_id, user_id, feedback_type, saved_to_wiki, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, 1, query_id, session_id, "report_citation_issue", 0, created_at + 4),
    )
    conn.execute(
        "INSERT INTO ontology_entities (entity_id, kb_id, display_text) VALUES (?, ?, ?)",
        (1, "default", "농가경제조사"),
    )
    conn.execute(
        """
        INSERT INTO ontology_facts
            (fact_id, kb_id, subject_entity_id, predicate, object_entity_id, object_value,
             fact_kind, extraction_method, confidence, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (7, "default", 1, "지급단가", 0, "40천원", "literal", "limited_llm", 0.44, "reported", created_at + 5, created_at + 6),
    )
    conn.execute(
        """
        INSERT INTO ontology_fact_sources
            (fact_id, chunk_id, source_path, source_ref, page_no, line_start, line_end, evidence_quote)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (7, 101, "/uploads/guide.pdf", "guide.pdf 4페이지", 4, 10, 12, "농가경제조사 지급단가 40천원"),
    )
    conn.commit()
    conn.close()


class DebugBundleScriptTests(unittest.TestCase):
    def test_collect_debug_bundle_creates_single_integrated_json_without_args(self):
        self.assertTrue(SCRIPT_PATH.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            logs_dir = tmp / "logs"
            data_dir = tmp / "data"
            kb_dir = data_dir / "kb" / "default"
            logs_dir.mkdir(parents=True)
            kb_dir.mkdir(parents=True)

            app_db_path = data_dir / "app.sqlite"
            kb_db_path = kb_dir / "meta.sqlite"
            _create_app_db(app_db_path)
            _insert_app_activity(
                app_db_path,
                session_id="sess-1",
                kb_name="default",
                query_id="query-1",
                user_text="문서 핵심 요약",
                answer_text="문서에 따르면 안전수칙이 핵심입니다.",
                created_at=1712100000,
            )
            _create_kb_meta_db(
                kb_db_path,
                query_id="query-1",
                session_id="sess-1",
                query_text="문서 핵심 요약",
                answer_text="문서에 따르면 안전수칙이 핵심입니다.",
                created_at=1712100000,
            )

            (logs_dir / "admin_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-03T00:00:00Z",
                        "kb_name": "default",
                        "is_correct": False,
                        "question": "문서 핵심 요약",
                        "answer": "잘못된 답변",
                        "expected_answer": "문서에 따르면 안전수칙이 핵심입니다.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (logs_dir / "rag_trace.jsonl").write_text(
                json.dumps(
                    {
                        "kb_name": "default",
                        "query": "문서 핵심 요약",
                        "top_results": [{"chunk_id": 101, "score": 0.77}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "COMPASSLM_LOGS_DIR": str(logs_dir),
                    "COMPASSLM_APP_DB_PATH": str(app_db_path),
                    "KB_DATA_DIR": str(data_dir / "kb"),
                    "COMPASSLM_DEBUG_API_BASE_URL": "http://127.0.0.1:9",
                    "COMPASSLM_DEBUG_CAPTURE_NOW": "20260403_101500",
                }
            )

            result = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            bundle_dir = logs_dir / "debug-captures"
            bundle_files = sorted(bundle_dir.glob("*.json"))
            self.assertEqual(len(bundle_files), 1, result.stdout + result.stderr)

            bundle = json.loads(bundle_files[0].read_text(encoding="utf-8"))
            self.assertEqual(bundle["resolved_kb_name"], "default")
            self.assertEqual(bundle["capture_id"], "20260403_101500")
            self.assertIn("startup_snapshot", bundle)
            self.assertIn("api_results", bundle)
            self.assertIn("recent_chat_messages", bundle)
            self.assertIn("recent_agent_runs", bundle)
            self.assertIn("recent_retrieval_logs", bundle)
            self.assertIn("recent_answer_logs", bundle)
            self.assertIn("recent_no_evidence_reports", bundle)
            self.assertIn("recent_reported_ontology_facts", bundle)
            self.assertIn("log_tails", bundle)
            self.assertGreaterEqual(len(bundle["recent_chat_messages"]), 1)
            self.assertGreaterEqual(len(bundle["recent_agent_runs"]), 1)
            self.assertGreaterEqual(len(bundle["recent_retrieval_logs"]), 1)
            self.assertGreaterEqual(len(bundle["recent_answer_logs"]), 1)
            self.assertGreaterEqual(len(bundle["recent_no_evidence_reports"]), 1)
            self.assertGreaterEqual(len(bundle["recent_reported_ontology_facts"]), 1)
            self.assertEqual(bundle["summary"]["recent_no_evidence_report_count"], 1)
            self.assertEqual(bundle["summary"]["recent_reported_ontology_fact_count"], 1)
            self.assertIn("validator_diagnostics", bundle["summary"])
            self.assertIn("validator_diagnostics", bundle["recent_kb_snapshots"][0])
            self.assertIn("report_citation_issue", json.dumps(bundle["recent_no_evidence_reports"], ensure_ascii=False))
            self.assertIn("농가경제조사", json.dumps(bundle["recent_reported_ontology_facts"], ensure_ascii=False))
            self.assertIn("document_grounded", json.dumps(bundle["recent_agent_runs"], ensure_ascii=False))
            self.assertIn("문서 핵심 요약", result.stdout)

    def test_collect_debug_bundle_groups_recent_logs_for_multiple_kbs_by_latest_activity(self):
        self.assertTrue(SCRIPT_PATH.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            logs_dir = tmp / "logs"
            data_dir = tmp / "data"
            logs_dir.mkdir(parents=True)
            (data_dir / "kb" / "nong1").mkdir(parents=True)
            (data_dir / "kb" / "답례품").mkdir(parents=True)

            app_db_path = data_dir / "app.sqlite"
            _create_app_db(app_db_path)
            _insert_app_activity(
                app_db_path,
                session_id="sess-nong1",
                kb_name="nong1",
                query_id="query-nong1",
                user_text="농지 문서 기준을 요약해줘",
                answer_text="nong1 문서 기준으로 정리했습니다.",
                created_at=1712100100,
            )
            _insert_app_activity(
                app_db_path,
                session_id="sess-gift",
                kb_name="답례품",
                query_id="query-gift",
                user_text="답례품 기준을 알려줘",
                answer_text="답례품 KB 기준으로 안내드립니다.",
                created_at=1712100200,
            )

            _create_kb_meta_db(
                data_dir / "kb" / "nong1" / "meta.sqlite",
                query_id="query-nong1",
                session_id="sess-nong1",
                query_text="농지 문서 기준을 요약해줘",
                answer_text="nong1 문서 기준으로 정리했습니다.",
                created_at=1712100100,
            )
            _create_kb_meta_db(
                data_dir / "kb" / "답례품" / "meta.sqlite",
                query_id="query-gift",
                session_id="sess-gift",
                query_text="답례품 기준을 알려줘",
                answer_text="답례품 KB 기준으로 안내드립니다.",
                created_at=1712100200,
            )

            env = os.environ.copy()
            env.update(
                {
                    "COMPASSLM_LOGS_DIR": str(logs_dir),
                    "COMPASSLM_APP_DB_PATH": str(app_db_path),
                    "KB_DATA_DIR": str(data_dir / "kb"),
                    "COMPASSLM_DEBUG_API_BASE_URL": "http://127.0.0.1:9",
                    "COMPASSLM_DEBUG_CAPTURE_NOW": "20260403_103000",
                }
            )

            subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            bundle = json.loads((logs_dir / "debug-captures" / "20260403_103000.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["resolved_kb_name"], "답례품")
            self.assertEqual(bundle["summary"]["kb_count"], 2)
            self.assertEqual(bundle["summary"]["kb_names_by_recent_activity"], ["답례품", "nong1"])
            self.assertEqual(bundle["summary"]["kb_last_questions"]["답례품"], "답례품 기준을 알려줘")
            self.assertEqual(bundle["summary"]["kb_last_questions"]["nong1"], "농지 문서 기준을 요약해줘")

            snapshots = bundle["recent_kb_snapshots"]
            self.assertEqual([item["kb_name"] for item in snapshots], ["답례품", "nong1"])
            self.assertEqual(snapshots[0]["latest_query_id"], "query-gift")
            self.assertEqual(snapshots[1]["latest_query_id"], "query-nong1")
            self.assertIn("답례품 기준", json.dumps(snapshots[0], ensure_ascii=False))
            self.assertIn("농지 문서 기준", json.dumps(snapshots[1], ensure_ascii=False))

            snapshots_by_name = bundle["kb_snapshots_by_name"]
            self.assertEqual(sorted(snapshots_by_name.keys()), ["nong1", "답례품"])
            self.assertEqual(snapshots_by_name["답례품"]["latest_query_id"], "query-gift")
            self.assertEqual(snapshots_by_name["nong1"]["latest_query_id"], "query-nong1")
            self.assertEqual(
                snapshots_by_name["답례품"]["recent_chat_messages"][0]["text"],
                "답례품 기준을 알려줘",
            )

    def test_collect_debug_bundle_captures_up_to_fifty_chat_messages_and_agent_runs(self):
        self.assertTrue(SCRIPT_PATH.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            logs_dir = tmp / "logs"
            data_dir = tmp / "data"
            logs_dir.mkdir(parents=True)
            (data_dir / "kb" / "nong1").mkdir(parents=True)

            app_db_path = data_dir / "app.sqlite"
            _create_app_db(app_db_path)
            for idx in range(60):
                _insert_app_activity(
                    app_db_path,
                    session_id="sess-nong1",
                    kb_name="nong1",
                    query_id=f"query-{idx}",
                    user_text=f"질문 {idx}",
                    answer_text=f"답변 {idx}",
                    created_at=1712101000 + idx * 10,
                )

            _create_kb_meta_db(
                data_dir / "kb" / "nong1" / "meta.sqlite",
                query_id="query-59",
                session_id="sess-nong1",
                query_text="질문 59",
                answer_text="답변 59",
                created_at=1712101590,
            )

            env = os.environ.copy()
            env.update(
                {
                    "COMPASSLM_LOGS_DIR": str(logs_dir),
                    "COMPASSLM_APP_DB_PATH": str(app_db_path),
                    "KB_DATA_DIR": str(data_dir / "kb"),
                    "COMPASSLM_DEBUG_API_BASE_URL": "http://127.0.0.1:9",
                    "COMPASSLM_DEBUG_CAPTURE_NOW": "20260403_111500",
                }
            )

            subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            bundle = json.loads((logs_dir / "debug-captures" / "20260403_111500.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["summary"]["recent_chat_message_count"], 50)
            self.assertEqual(bundle["summary"]["recent_agent_run_count"], 50)
            self.assertEqual(len(bundle["recent_chat_messages"]), 50)
            self.assertEqual(len(bundle["recent_agent_runs"]), 50)
            self.assertEqual(bundle["recent_chat_messages"][0]["text"], "질문 35")
            self.assertEqual(bundle["recent_chat_messages"][-1]["text"], "답변 59")
            self.assertEqual(bundle["recent_agent_runs"][0]["query_id"], "query-59")
            self.assertEqual(bundle["recent_agent_runs"][-1]["query_id"], "query-10")

    def test_collect_debug_bundle_includes_kb_directories_even_without_meta_sqlite(self):
        self.assertTrue(SCRIPT_PATH.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            logs_dir = tmp / "logs"
            data_dir = tmp / "data"
            logs_dir.mkdir(parents=True)
            (data_dir / "kb" / "nong1").mkdir(parents=True)
            (data_dir / "kb" / "답례품" / "uploads").mkdir(parents=True)

            app_db_path = data_dir / "app.sqlite"
            _create_app_db(app_db_path)
            _insert_app_activity(
                app_db_path,
                session_id="sess-nong1",
                kb_name="nong1",
                query_id="query-nong1",
                user_text="농지 문서 기준을 요약해줘",
                answer_text="nong1 문서 기준으로 정리했습니다.",
                created_at=1712100100,
            )
            _create_kb_meta_db(
                data_dir / "kb" / "nong1" / "meta.sqlite",
                query_id="query-nong1",
                session_id="sess-nong1",
                query_text="농지 문서 기준을 요약해줘",
                answer_text="nong1 문서 기준으로 정리했습니다.",
                created_at=1712100100,
            )

            env = os.environ.copy()
            env.update(
                {
                    "COMPASSLM_LOGS_DIR": str(logs_dir),
                    "COMPASSLM_APP_DB_PATH": str(app_db_path),
                    "KB_DATA_DIR": str(data_dir / "kb"),
                    "COMPASSLM_DEBUG_API_BASE_URL": "http://127.0.0.1:9",
                    "COMPASSLM_DEBUG_CAPTURE_NOW": "20260403_104500",
                }
            )

            subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            bundle = json.loads((logs_dir / "debug-captures" / "20260403_104500.json").read_text(encoding="utf-8"))
            kb_names = [item["kb_name"] for item in bundle["recent_kb_snapshots"]]
            self.assertIn("nong1", kb_names)
            self.assertIn("답례품", kb_names)

    def test_collect_debug_bundle_documents_cookie_auth_for_protected_api(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("COMPASSLM_DEBUG_AUTH_COOKIE='cookie-value'", script)
        self.assertIn("COMPASSLM_DEBUG_COOKIE_HEADER='compass_auth_session=cookie-value; other=value'", script)
        self.assertIn("COMPASSLM_DEBUG_PYTHON_BIN", script)
        self.assertIn("DEBUG_PYTHON_BIN=\"python\"", script)
        self.assertIn("ops_failure_patterns_ok=false", script)
        self.assertIn("api_auth_missing=true", script)
        self.assertIn("/ops/* plus /kbs/* API diagnostics will be omitted", script)


if __name__ == "__main__":
    unittest.main()
