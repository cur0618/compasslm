import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260_000


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _slug(value: str, *, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or fallback


def _reject_path_like(value: str):
    raw = (value or "").strip()
    if not raw:
        raise ValueError("value must not be empty")
    if "/" in raw or "\\" in raw or raw in {".", ".."} or ".." in raw.split("/"):
        raise ValueError("path-like values are not allowed")
    if ".." in raw:
        raise ValueError("path traversal is not allowed")


def build_scoped_kb_id(user_id: str, display_name: str) -> str:
    _reject_path_like(user_id)
    _reject_path_like(display_name)
    user_part = _slug(user_id, fallback="user")
    name_part = _slug(display_name, fallback="kb")
    digest = hashlib.sha256(f"{user_id}\0{display_name}".encode("utf-8")).hexdigest()[:12]
    return f"{user_part}__{name_part}__{digest}"


def build_fresh_scoped_kb_id(user_id: str, display_name: str) -> str:
    return f"{build_scoped_kb_id(user_id, display_name)}__{uuid.uuid4().hex[:10]}"


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return f"{PBKDF2_ALGORITHM}${int(iterations)}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = (encoded or "").split("$", 3)
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        salt = _b64decode(salt_raw)
        expected = _b64decode(digest_raw)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _hash_session_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


class AuthStore:
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
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users
                (
                    user_id TEXT PRIMARY KEY,
                    login_id TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    display_name TEXT,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    last_login_at INTEGER
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions
                (
                    session_token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS user_kbs
                (
                    kb_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    internal_kb_id TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(owner_user_id) REFERENCES users(user_id),
                    UNIQUE(owner_user_id, display_name)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs
                (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    detail_json TEXT,
                    created_at INTEGER NOT NULL
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id, expires_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_user_kbs_owner ON user_kbs(owner_user_id, is_deleted, display_name)")
            conn.commit()
            conn.close()

    def create_user(
        self,
        login_id: str,
        password: str,
        *,
        display_name: str = "",
        role: str = "user",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        login = (login_id or "").strip()
        if not login:
            raise ValueError("login_id must not be empty")
        if role not in {"admin", "user"}:
            raise ValueError("role must be admin or user")
        now_ts = int(time.time())
        uid = user_id or f"u_{uuid.uuid4().hex[:16]}"
        with self._lock:
            conn = self._connect()
            c = conn.cursor()
            try:
                c.execute(
                    """
                    INSERT INTO users (user_id, login_id, password_hash, display_name, role, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (uid, login, hash_password(password), display_name or login, role, now_ts),
                )
            except sqlite3.IntegrityError as exc:
                conn.close()
                if "users.login_id" in str(exc):
                    raise ValueError("login_id already exists") from exc
                raise
            conn.commit()
            row = c.execute("SELECT * FROM users WHERE user_id = ?", (uid,)).fetchone()
            conn.close()
        return dict(row)

    def get_user_by_login(self, login_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM users WHERE login_id = ?", ((login_id or "").strip(),)).fetchone()
            conn.close()
        return dict(row) if row else None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            conn.close()
        return dict(row) if row else None

    def set_user_active(self, user_id: str, active: bool):
        with self._lock:
            conn = self._connect()
            conn.execute("UPDATE users SET is_active = ? WHERE user_id = ?", (1 if active else 0, user_id))
            conn.commit()
            conn.close()

    def authenticate(self, login_id: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_login(login_id)
        if not user or not int(user.get("is_active") or 0):
            return None
        if not verify_password(password or "", str(user.get("password_hash") or "")):
            return None
        with self._lock:
            conn = self._connect()
            conn.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?", (int(time.time()), user["user_id"]))
            conn.commit()
            conn.close()
        user.pop("password_hash", None)
        return user

    def create_session(
        self,
        user_id: str,
        *,
        ttl_seconds: int = 86400,
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        token = secrets.token_urlsafe(32)
        now_ts = int(time.time())
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO user_sessions
                    (session_token_hash, user_id, expires_at, created_at, last_seen_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _hash_session_token(token),
                    user_id,
                    now_ts + int(ttl_seconds),
                    now_ts,
                    now_ts,
                    ip_address or "",
                    user_agent or "",
                ),
            )
            conn.commit()
            conn.close()
        return token

    def get_user_by_session(self, token: str, *, now_ts: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        now_value = int(time.time()) if now_ts is None else int(now_ts)
        token_hash = _hash_session_token(token)
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                """
                SELECT users.*
                FROM user_sessions
                JOIN users ON users.user_id = user_sessions.user_id
                WHERE user_sessions.session_token_hash = ?
                    AND user_sessions.expires_at > ?
                    AND users.is_active = 1
                """,
                (token_hash, now_value),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE user_sessions SET last_seen_at = ? WHERE session_token_hash = ?",
                    (now_value, token_hash),
                )
                conn.commit()
            conn.close()
        if not row:
            return None
        user = dict(row)
        user.pop("password_hash", None)
        return user

    def revoke_session(self, token: str):
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM user_sessions WHERE session_token_hash = ?", (_hash_session_token(token),))
            conn.commit()
            conn.close()

    def prune_expired_sessions(
        self,
        *,
        now_ts: Optional[int] = None,
        limit: int = 1000,
    ) -> int:
        now_value = int(time.time()) if now_ts is None else int(now_ts)
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT session_token_hash
                FROM user_sessions
                WHERE expires_at <= ?
                ORDER BY expires_at ASC
                LIMIT ?
                """,
                (now_value, max(1, int(limit or 1))),
            ).fetchall()
            session_hashes = [str(row[0]) for row in rows if str(row[0] or "")]
            if session_hashes:
                placeholders = ",".join("?" for _ in session_hashes)
                conn.execute(
                    f"DELETE FROM user_sessions WHERE session_token_hash IN ({placeholders})",
                    tuple(session_hashes),
                )
                conn.commit()
            conn.close()
        return len(session_hashes)

    def create_kb(
        self,
        owner_user_id: str,
        display_name: str,
        *,
        internal_kb_id: Optional[str] = None,
        kb_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _reject_path_like(display_name)
        now_ts = int(time.time())
        explicit_internal_id = internal_kb_id is not None
        resolved_internal = internal_kb_id or build_scoped_kb_id(owner_user_id, display_name)
        _reject_path_like(resolved_internal)
        resolved_kb_id = kb_id or f"kb_{uuid.uuid4().hex[:16]}"
        with self._lock:
            conn = self._connect()
            try:
                c = conn.cursor()
                existing_internal = c.execute(
                    "SELECT * FROM user_kbs WHERE internal_kb_id = ?",
                    (resolved_internal,),
                ).fetchone()
                if existing_internal:
                    if str(existing_internal["owner_user_id"]) != str(owner_user_id):
                        raise ValueError("kb internal_kb_id already exists")
                    if explicit_internal_id:
                        c.execute(
                            """
                            UPDATE user_kbs
                            SET is_deleted = 0, updated_at = ?
                            WHERE kb_id = ?
                            """,
                            (now_ts, existing_internal["kb_id"]),
                        )
                        conn.commit()
                        row = c.execute(
                            "SELECT * FROM user_kbs WHERE kb_id = ? AND is_deleted = 0",
                            (existing_internal["kb_id"],),
                        ).fetchone()
                        return dict(row)
                    if int(existing_internal["is_deleted"] or 0) == 0:
                        return dict(existing_internal)
                    resolved_internal = build_fresh_scoped_kb_id(owner_user_id, display_name)

                existing_display = c.execute(
                    """
                    SELECT *
                    FROM user_kbs
                    WHERE owner_user_id = ? AND display_name = ?
                    """,
                    (owner_user_id, display_name),
                ).fetchone()
                if existing_display and int(existing_display["is_deleted"] or 0) != 0 and not explicit_internal_id:
                    c.execute(
                        """
                        UPDATE user_kbs
                        SET kb_id = ?, internal_kb_id = ?, created_at = ?, updated_at = ?, is_deleted = 0
                        WHERE owner_user_id = ? AND display_name = ?
                        """,
                        (resolved_kb_id, resolved_internal, now_ts, now_ts, owner_user_id, display_name),
                    )
                    conn.commit()
                    row = c.execute(
                        "SELECT * FROM user_kbs WHERE kb_id = ? AND is_deleted = 0",
                        (resolved_kb_id,),
                    ).fetchone()
                    return dict(row)

                c.execute(
                    """
                    INSERT INTO user_kbs
                        (kb_id, owner_user_id, display_name, internal_kb_id, created_at, updated_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(owner_user_id, display_name) DO UPDATE SET
                        is_deleted = 0,
                        updated_at = excluded.updated_at
                    """,
                    (resolved_kb_id, owner_user_id, display_name, resolved_internal, now_ts, now_ts),
                )
                conn.commit()
                row = c.execute(
                    """
                    SELECT *
                    FROM user_kbs
                    WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0
                    """,
                    (owner_user_id, display_name),
                ).fetchone()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                if "user_kbs.internal_kb_id" in str(exc):
                    raise ValueError("kb internal_kb_id already exists") from exc
                raise
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return dict(row)

    def list_kbs(self, owner_user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT *
                FROM user_kbs
                WHERE owner_user_id = ? AND is_deleted = 0
                ORDER BY display_name COLLATE NOCASE
                """,
                (owner_user_id,),
            ).fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def get_kb(
        self,
        owner_user_id: str,
        display_name: str,
        *,
        internal_kb_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        query = [
            "SELECT * FROM user_kbs",
            "WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0",
        ]
        params: List[Any] = [owner_user_id, display_name]
        if internal_kb_id:
            query.append("AND internal_kb_id = ?")
            params.append(internal_kb_id)
        with self._lock:
            conn = self._connect()
            row = conn.execute("\n".join(query), tuple(params)).fetchone()
            conn.close()
        return dict(row) if row else None

    def rename_kb(self, owner_user_id: str, old_display_name: str, new_display_name: str) -> Optional[Dict[str, Any]]:
        _reject_path_like(new_display_name)
        now_ts = int(time.time())
        with self._lock:
            conn = self._connect()
            try:
                c = conn.cursor()
                if old_display_name != new_display_name:
                    existing = c.execute(
                        """
                        SELECT kb_id FROM user_kbs
                        WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0
                        """,
                        (owner_user_id, new_display_name),
                    ).fetchone()
                    if existing:
                        raise ValueError("kb display_name already exists")
                c.execute(
                    """
                    UPDATE user_kbs
                    SET display_name = ?, updated_at = ?
                    WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0
                    """,
                    (new_display_name, now_ts, owner_user_id, old_display_name),
                )
                if c.rowcount <= 0:
                    conn.rollback()
                    return None
                conn.commit()
                row = c.execute(
                    "SELECT * FROM user_kbs WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0",
                    (owner_user_id, new_display_name),
                ).fetchone()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                if "user_kbs.owner_user_id, user_kbs.display_name" in str(exc):
                    raise ValueError("kb display_name already exists") from exc
                raise
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return dict(row) if row else None

    def soft_delete_kb(self, owner_user_id: str, display_name: str) -> bool:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                UPDATE user_kbs
                SET is_deleted = 1, updated_at = ?
                WHERE owner_user_id = ? AND display_name = ? AND is_deleted = 0
                """,
                (int(time.time()), owner_user_id, display_name),
            )
            conn.commit()
            changed = cur.rowcount > 0
            conn.close()
        return changed

    def ensure_legacy_kbs_for_admin(self, admin_user_id: str, kb_names: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for kb_name in kb_names:
            if not kb_name:
                continue
            try:
                out.append(self.create_kb(admin_user_id, kb_name, internal_kb_id=kb_name))
            except ValueError as exc:
                if "kb internal_kb_id already exists" in str(exc):
                    continue
                raise
        return out
