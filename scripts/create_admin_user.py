#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.auth_store import AuthStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "app.sqlite"
DEFAULT_KB_ROOT = ROOT / "data" / "kb"
DEFAULT_ADMIN_CREDENTIAL_ENV = "COMPASSLM_ADMIN_PASSWORD"


def _legacy_kb_names(kb_root: Path) -> list[str]:
    if not kb_root.exists():
        return []
    return sorted(
        path.name
        for path in kb_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _read_existing_user(db_path: Path, login_id: str) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT user_id, login_id, role, is_active FROM users WHERE login_id = ?",
                (login_id,),
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    return dict(row) if row else None


def provision_admin(
    *,
    db_path: Path,
    kb_root: Path,
    login_id: str,
    password: Optional[str],
    display_name: str = "Administrator",
    dry_run: bool = False,
) -> Dict[str, Any]:
    login = (login_id or "").strip()
    if not login:
        raise ValueError("login_id must not be empty")
    db_path = Path(db_path).resolve()
    kb_root = Path(kb_root).resolve()
    legacy_names = _legacy_kb_names(kb_root)

    if dry_run:
        existing = _read_existing_user(db_path, login)
        if existing and str(existing.get("role") or "") != "admin":
            raise ValueError(f"existing login '{login}' is not an admin")
        return {
            "created": existing is None,
            "dry_run": True,
            "login_id": login,
            "role": "admin",
            "db_path": str(db_path),
            "legacy_kbs": legacy_names,
        }

    store = AuthStore(str(db_path))
    user = store.get_user_by_login(login)
    created = user is None
    if user is None:
        if not password:
            raise ValueError("password is required when creating a new admin")
        user = store.create_user(
            login,
            password,
            display_name=(display_name or login).strip(),
            role="admin",
        )
    elif str(user.get("role") or "") != "admin":
        raise ValueError(f"existing login '{login}' is not an admin")

    store.ensure_legacy_kbs_for_admin(str(user["user_id"]), legacy_names)
    registered = {
        str(record["internal_kb_id"])
        for record in store.list_kbs(str(user["user_id"]))
    }
    return {
        "created": created,
        "dry_run": False,
        "login_id": login,
        "role": "admin",
        "db_path": str(db_path),
        "legacy_kbs": [name for name in legacy_names if name in registered],
    }


def _resolve_password(args: argparse.Namespace) -> Optional[str]:
    if args.dry_run:
        return None
    env_value = os.environ.get(args.password_env, "")
    if env_value:
        return env_value
    existing = _read_existing_user(Path(args.db), args.login)
    if existing:
        return None
    if not sys.stdin.isatty():
        raise ValueError(
            f"set {args.password_env} or run interactively to create the admin"
        )
    first = getpass.getpass("Admin password: ")
    second = getpass.getpass("Confirm admin password: ")
    if first != second:
        raise ValueError("password confirmation does not match")
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the CompassLM admin and register existing legacy KBs.",
    )
    parser.add_argument("--login", default="admin", help="Admin login ID.")
    parser.add_argument("--display-name", default="Administrator")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Auth SQLite path.")
    parser.add_argument(
        "--kb-root",
        default=str(DEFAULT_KB_ROOT),
        help="Directory containing existing legacy KB folders.",
    )
    parser.add_argument(
        "--password-env",
        default=DEFAULT_ADMIN_CREDENTIAL_ENV,
        help="Environment variable used for a new admin password.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the intended action without modifying the database.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        resolved_value = _resolve_password(args)
        result = provision_admin(
            db_path=Path(args.db),
            kb_root=Path(args.kb_root),
            login_id=args.login,
            password=resolved_value,
            display_name=args.display_name,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
