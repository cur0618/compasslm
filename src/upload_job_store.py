import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


def _safe_json_load(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class UploadJobStore:
    def __init__(self, db_path: str, *, timeout_seconds: float = 30.0, busy_timeout_ms: int = 30000):
        self.db_path = Path(db_path).resolve()
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.busy_timeout_ms = max(5000, int(busy_timeout_ms))
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout_seconds)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_jobs
                (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    kb_id TEXT NOT NULL DEFAULT '',
                    file_id TEXT NOT NULL DEFAULT '',
                    original_filename TEXT NOT NULL DEFAULT '',
                    stored_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress_percent INTEGER NOT NULL DEFAULT 0,
                    progress_stage TEXT NOT NULL DEFAULT '',
                    failure_code TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL DEFAULT 0,
                    queued_at INTEGER NOT NULL DEFAULT 0,
                    processing_started_at INTEGER NOT NULL DEFAULT 0,
                    completed_at INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upload_jobs_user_updated ON upload_jobs(user_id, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upload_jobs_kb_status ON upload_jobs(kb_id, status, updated_at)")
            conn.commit()
            conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> Dict[str, Any]:
        payload = _safe_json_load(str(row["payload_json"] or "{}"))
        payload.update(
            {
                "job_id": str(row["job_id"] or ""),
                "user_id": str(row["user_id"] or ""),
                "kb_name": str(row["kb_id"] or ""),
                "stored_filename": str(row["file_id"] or ""),
                "original_filename": str(row["original_filename"] or ""),
                "stored_path": str(row["stored_path"] or ""),
                "status": str(row["status"] or "queued"),
                "progress_percent": int(row["progress_percent"] or 0),
                "progress_stage": str(row["progress_stage"] or ""),
                "failure_code": str(row["failure_code"] or ""),
                "message": str(row["message"] or ""),
                "created_at": int(row["created_at"] or 0),
                "queued_at": int(row["queued_at"] or 0),
                "processing_started_at": int(row["processing_started_at"] or 0),
                "completed_at": int(row["completed_at"] or 0),
                "updated_at": int(row["updated_at"] or 0),
                "version": int(row["version"] or 1),
            }
        )
        return payload

    def save_job(self, job: Dict[str, Any]):
        payload = dict(job or {})
        job_id = str(payload.get("job_id", "") or "").strip()
        if not job_id:
            raise ValueError("job_id must not be empty")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO upload_jobs
                    (
                        job_id, user_id, kb_id, file_id, original_filename, stored_path,
                        status, progress_percent, progress_stage, failure_code, message,
                        created_at, queued_at, processing_started_at, completed_at, updated_at,
                        version, payload_json
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    kb_id = excluded.kb_id,
                    file_id = excluded.file_id,
                    original_filename = excluded.original_filename,
                    stored_path = excluded.stored_path,
                    status = excluded.status,
                    progress_percent = excluded.progress_percent,
                    progress_stage = excluded.progress_stage,
                    failure_code = excluded.failure_code,
                    message = excluded.message,
                    queued_at = excluded.queued_at,
                    processing_started_at = excluded.processing_started_at,
                    completed_at = excluded.completed_at,
                    updated_at = excluded.updated_at,
                    version = excluded.version,
                    payload_json = excluded.payload_json
                """,
                (
                    job_id,
                    str(payload.get("user_id", "") or ""),
                    str(payload.get("kb_name", payload.get("kb_id", "")) or ""),
                    str(payload.get("stored_filename", payload.get("file_id", "")) or ""),
                    str(payload.get("original_filename", "") or ""),
                    str(payload.get("stored_path", "") or ""),
                    str(payload.get("status", "queued") or "queued"),
                    int(payload.get("progress_percent", 0) or 0),
                    str(payload.get("progress_stage", "") or ""),
                    str(payload.get("failure_code", "") or ""),
                    str(payload.get("message", "") or ""),
                    int(payload.get("created_at", 0) or 0),
                    int(payload.get("queued_at", 0) or 0),
                    int(payload.get("processing_started_at", 0) or 0),
                    int(payload.get("completed_at", 0) or 0),
                    int(payload.get("updated_at", 0) or 0),
                    int(payload.get("version", 1) or 1),
                    _safe_json_dump(payload),
                ),
            )
            conn.commit()
            conn.close()

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = self.get_job(job_id)
        if not current:
            return None
        current.update(dict(updates or {}))
        self.save_job(current)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM upload_jobs WHERE job_id = ?", (job_id,)).fetchone()
            conn.close()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        user_id: str = "",
        kb_name: str = "",
        include_terminal: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = ["SELECT * FROM upload_jobs"]
        params: List[Any] = []
        where: List[str] = []
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if kb_name:
            where.append("kb_id = ?")
            params.append(kb_name)
        if not include_terminal:
            where.append("status NOT IN ('success', 'error', 'timeout', 'not_found', 'cancelled')")
        if where:
            query.append("WHERE " + " AND ".join(where))
        query.append("ORDER BY updated_at DESC, created_at DESC LIMIT ?")
        params.append(max(1, int(limit or 100)))
        with self._lock:
            conn = self._connect()
            rows = conn.execute("\n".join(query), tuple(params)).fetchall()
            conn.close()
        return [self._row_to_job(row) for row in rows]

    def list_incomplete_jobs(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT *
                FROM upload_jobs
                WHERE status IN ('queued', 'processing')
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit or 100)),),
            ).fetchall()
            conn.close()
        return [self._row_to_job(row) for row in rows]

    def prune_terminal_jobs(
        self,
        *,
        expire_before: int,
        max_terminal_rows: int = 5000,
        batch_size: int = 1000,
    ) -> int:
        terminal_statuses = ("success", "error", "timeout", "not_found", "cancelled")
        limit = max(1, int(batch_size or 1))
        removed_ids: List[str] = []
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" for _ in terminal_statuses)
                if int(expire_before or 0) > 0:
                    removed_ids.extend(
                        str(row[0])
                        for row in conn.execute(
                            f"""
                            SELECT job_id FROM upload_jobs
                            WHERE status IN ({placeholders})
                              AND updated_at < ?
                            ORDER BY updated_at ASC, created_at ASC
                            LIMIT ?
                            """,
                            (*terminal_statuses, int(expire_before), limit),
                        ).fetchall()
                    )
                if removed_ids:
                    delete_placeholders = ",".join("?" for _ in removed_ids)
                    conn.execute(
                        f"DELETE FROM upload_jobs WHERE job_id IN ({delete_placeholders})",
                        tuple(removed_ids),
                    )

                resolved_max = max(0, int(max_terminal_rows or 0))
                if resolved_max > 0:
                    terminal_count = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM upload_jobs WHERE status IN ({placeholders})",
                            terminal_statuses,
                        ).fetchone()[0]
                    )
                    overflow = max(0, terminal_count - resolved_max)
                    if overflow:
                        cap_ids = [
                            str(row[0])
                            for row in conn.execute(
                                f"""
                                SELECT job_id FROM upload_jobs
                                WHERE status IN ({placeholders})
                                ORDER BY updated_at ASC, created_at ASC
                                LIMIT ?
                                """,
                                (*terminal_statuses, min(limit, overflow)),
                            ).fetchall()
                        ]
                        if cap_ids:
                            cap_placeholders = ",".join("?" for _ in cap_ids)
                            conn.execute(
                                f"DELETE FROM upload_jobs WHERE job_id IN ({cap_placeholders})",
                                tuple(cap_ids),
                            )
                            removed_ids.extend(cap_ids)
                conn.commit()
            finally:
                conn.close()
        return len(set(removed_ids))
