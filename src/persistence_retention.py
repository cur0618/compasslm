"""Bounded retention helpers for SQLite-backed operational records."""

import re
import sqlite3
from pathlib import Path
from typing import List


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str) -> str:
    if not _SQL_IDENTIFIER_RE.fullmatch(str(value or "")):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return str(value)


def _delete_ids(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    ids: List[object],
) -> int:
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"DELETE FROM {table} WHERE {id_column} IN ({placeholders})",
        tuple(ids),
    )
    return len(ids)


def prune_timestamped_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    timestamp_column: str,
    expire_before: int = 0,
    max_rows: int = 0,
    batch_size: int = 1000,
) -> int:
    """Delete old rows, then cap the table while retaining the newest rows."""
    safe_table = _identifier(table)
    safe_id = _identifier(id_column)
    safe_timestamp = _identifier(timestamp_column)
    limit = max(1, int(batch_size or 1))
    removed = 0

    if int(expire_before or 0) > 0:
        expired_ids = [
            row[0]
            for row in conn.execute(
                f"""
                SELECT {safe_id}
                FROM {safe_table}
                WHERE {safe_timestamp} < ?
                ORDER BY {safe_timestamp} ASC, {safe_id} ASC
                LIMIT ?
                """,
                (int(expire_before), limit),
            ).fetchall()
        ]
        removed += _delete_ids(
            conn,
            table=safe_table,
            id_column=safe_id,
            ids=expired_ids,
        )

    resolved_max_rows = max(0, int(max_rows or 0))
    if resolved_max_rows > 0:
        row_count = int(
            conn.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[0]
        )
        overflow = max(0, row_count - resolved_max_rows)
        if overflow:
            cap_ids = [
                row[0]
                for row in conn.execute(
                    f"""
                    SELECT {safe_id}
                    FROM {safe_table}
                    ORDER BY {safe_timestamp} ASC, {safe_id} ASC
                    LIMIT ?
                    """,
                    (min(limit, overflow),),
                ).fetchall()
            ]
            removed += _delete_ids(
                conn,
                table=safe_table,
                id_column=safe_id,
                ids=cap_ids,
            )
    return removed


def rotate_file_if_oversize(
    path: Path,
    *,
    max_bytes: int,
    backup_count: int,
) -> bool:
    resolved = Path(path)
    if not resolved.is_file() or resolved.stat().st_size < max(1, int(max_bytes or 1)):
        return False
    backups = max(1, int(backup_count or 1))
    oldest = Path(f"{resolved}.{backups}")
    if oldest.exists():
        oldest.unlink()
    for index in range(backups - 1, 0, -1):
        source = Path(f"{resolved}.{index}")
        if source.exists():
            source.replace(Path(f"{resolved}.{index + 1}"))
    resolved.replace(Path(f"{resolved}.1"))
    return True
