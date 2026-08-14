import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
except Exception:
    ModelMessage = Any

    class ModelMessagesTypeAdapter:
        @staticmethod
        def validate_json(value: Any) -> List[Any]:
            return []


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


class ChatStore:
    def __init__(
        self,
        db_path: str,
        *,
        history_limit: int = 200,
        agent_run_limit: int = 12,
        timeout_seconds: float = 30.0,
        busy_timeout_ms: int = 30000,
    ):
        self.db_path = Path(db_path).resolve()
        self.history_limit = max(20, int(history_limit))
        self.agent_run_limit = max(1, int(agent_run_limit))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.busy_timeout_ms = max(5000, int(busy_timeout_ms))
        self._lock = threading.RLock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
        default_sql: str = "",
    ):
        existing = self._table_columns(conn, table_name)
        if column_name in existing:
            return
        default_clause = f" DEFAULT {default_sql}" if default_sql else ""
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}{default_clause}"
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout_seconds)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions
                (
                    session_id TEXT PRIMARY KEY,
                    history_enabled INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages
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
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs
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
            self._ensure_column(c.connection, "agent_runs", "response_quality_issue", "TEXT")
            self._ensure_column(c.connection, "agent_runs", "usage_json", "TEXT")
            self._ensure_column(c.connection, "agent_runs", "request_count", "INTEGER", "0")
            self._ensure_column(c.connection, "agent_runs", "tool_call_count", "INTEGER", "0")
            self._ensure_column(c.connection, "agent_runs", "input_tokens", "INTEGER", "0")
            self._ensure_column(c.connection, "agent_runs", "output_tokens", "INTEGER", "0")
            self._ensure_column(c.connection, "agent_runs", "context_chars", "INTEGER", "0")
            self._ensure_column(c.connection, "chat_messages", "user_id", "TEXT", "''")
            self._ensure_column(c.connection, "agent_runs", "user_id", "TEXT", "''")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_kb "
                "ON chat_messages(session_id, kb_name, message_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_user_session_kb "
                "ON chat_messages(user_id, session_id, kb_name, message_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_session_kb "
                "ON agent_runs(session_id, kb_name, run_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_user_session_kb "
                "ON agent_runs(user_id, session_id, kb_name, run_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_query "
                "ON agent_runs(query_id, created_at)"
            )
            conn.commit()
            conn.close()

    def ensure_session(self, session_id: str) -> Dict[str, Any]:
        now_ts = int(time.time())
        with self._lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO chat_sessions (session_id, history_enabled, created_at, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at
                """,
                (session_id, now_ts, now_ts),
            )
            c.execute(
                "SELECT session_id, history_enabled, created_at, updated_at FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = c.fetchone()
            conn.commit()
            conn.close()
        return {
            "session_id": session_id,
            "history_enabled": bool(row["history_enabled"]) if row else True,
            "created_at": int(row["created_at"]) if row else now_ts,
            "updated_at": int(row["updated_at"]) if row else now_ts,
        }

    def set_history_enabled(self, session_id: str, enabled: bool):
        now_ts = int(time.time())
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                UPDATE chat_sessions
                SET history_enabled = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (1 if enabled else 0, now_ts, session_id),
            )
            conn.commit()
            conn.close()

    def is_history_enabled(self, session_id: str) -> bool:
        with self._lock:
            row = self.ensure_session(session_id)
        return bool(row.get("history_enabled", False))

    def clear_history(self, session_id: str, kb_name: Optional[str] = None, *, user_id: str = ""):
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            if kb_name is None:
                c.execute("DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?", (session_id, user_id or ""))
                c.execute("DELETE FROM agent_runs WHERE session_id = ? AND user_id = ?", (session_id, user_id or ""))
            else:
                c.execute(
                    "DELETE FROM chat_messages WHERE session_id = ? AND kb_name = ? AND user_id = ?",
                    (session_id, kb_name, user_id or ""),
                )
                c.execute(
                    "DELETE FROM agent_runs WHERE session_id = ? AND kb_name = ? AND user_id = ?",
                    (session_id, kb_name, user_id or ""),
                )
            c.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (int(time.time()), session_id),
            )
            conn.commit()
            conn.close()

    def append_chat_message(self, session_id: str, kb_name: str, role: str, text: str, *, user_id: str = ""):
        message = (text or "").strip()
        if not message:
            return
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO chat_messages (session_id, kb_name, role, text, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, kb_name, role, message, int(time.time()), user_id or ""),
            )
            c.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = ? AND kb_name = ? AND user_id = ? AND message_id NOT IN (
                    SELECT message_id
                    FROM chat_messages
                    WHERE session_id = ? AND kb_name = ? AND user_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                """,
                (session_id, kb_name, user_id or "", session_id, kb_name, user_id or "", self.history_limit),
            )
            c.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (int(time.time()), session_id),
            )
            conn.commit()
            conn.close()

    def get_chat_history(self, session_id: str, kb_name: str, *, user_id: str = "") -> List[Dict[str, str]]:
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                SELECT role, text
                FROM chat_messages
                WHERE session_id = ? AND kb_name = ? AND user_id = ?
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (session_id, kb_name, user_id or "", self.history_limit),
            )
            rows = c.fetchall()
            conn.close()
        rows = list(reversed(rows))
        return [{"role": str(row["role"]), "text": str(row["text"])} for row in rows]

    def append_agent_run(
        self,
        session_id: str,
        kb_name: str,
        query_id: str,
        user_message: str,
        answer_text: str,
        new_messages_json: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        response_quality_issue: str = "",
        usage: Optional[Dict[str, Any]] = None,
        context_chars: int = 0,
        user_id: str = "",
    ):
        payload_blob = new_messages_json if new_messages_json else b"[]"
        usage_payload = usage if isinstance(usage, dict) else {}
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO agent_runs
                    (
                        query_id,
                        session_id,
                        kb_name,
                        user_message,
                        answer_text,
                        new_messages_json,
                        metadata_json,
                        response_quality_issue,
                        usage_json,
                        request_count,
                        tool_call_count,
                        input_tokens,
                        output_tokens,
                        context_chars,
                        user_id,
                        created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    session_id,
                    kb_name,
                    user_message or "",
                    answer_text or "",
                    sqlite3.Binary(payload_blob),
                    _safe_json_dump(metadata or {}),
                    (response_quality_issue or "").strip(),
                    _safe_json_dump(usage_payload),
                    int(usage_payload.get("requests", 0) or 0),
                    int(usage_payload.get("tool_calls", 0) or 0),
                    int(usage_payload.get("input_tokens", 0) or 0),
                    int(usage_payload.get("output_tokens", 0) or 0),
                    max(0, int(context_chars or 0)),
                    user_id or "",
                    int(time.time()),
                ),
            )
            c.execute(
                """
                DELETE FROM agent_runs
                WHERE session_id = ? AND kb_name = ? AND user_id = ? AND run_id NOT IN (
                    SELECT run_id
                    FROM agent_runs
                    WHERE session_id = ? AND kb_name = ? AND user_id = ?
                    ORDER BY run_id DESC
                    LIMIT ?
                )
                """,
                (session_id, kb_name, user_id or "", session_id, kb_name, user_id or "", self.agent_run_limit),
            )
            c.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (int(time.time()), session_id),
            )
            conn.commit()
            conn.close()

    def load_agent_message_history(self, session_id: str, kb_name: str, *, user_id: str = "") -> List[ModelMessage]:
        with self._lock:
            self.ensure_session(session_id)
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                """
                SELECT new_messages_json
                FROM agent_runs
                WHERE session_id = ? AND kb_name = ? AND user_id = ?
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (session_id, kb_name, user_id or "", self.agent_run_limit),
            )
            rows = c.fetchall()
            conn.close()

        out: List[ModelMessage] = []
        for row in reversed(rows):
            payload = row["new_messages_json"]
            if payload is None:
                continue
            try:
                out.extend(ModelMessagesTypeAdapter.validate_json(payload))
            except Exception:
                continue
        return out

    def get_recent_agent_runs(
        self,
        *,
        kb_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        query = [
            "SELECT run_id, query_id, session_id, kb_name, user_id, user_message, answer_text, metadata_json,",
            "response_quality_issue, usage_json, request_count, tool_call_count, input_tokens, output_tokens,",
            "context_chars, created_at",
            "FROM agent_runs",
        ]
        params: List[Any] = []
        where_parts: List[str] = []
        if kb_name:
            where_parts.append("kb_name = ?")
            params.append(kb_name)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id or "")
        if where_parts:
            query.append("WHERE " + " AND ".join(where_parts))
        query.append("ORDER BY created_at DESC, run_id DESC LIMIT ?")
        params.append(max(1, int(limit)))

        with self._lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("\n".join(query), tuple(params))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
        return rows
