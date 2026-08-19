#!/usr/bin/env python3
"""Safely prune CompassLM timestamped runtime log directories."""

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional


_RUNTIME_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def prune_runtime_log_dirs(
    root: Path,
    *,
    retention_days: int,
    max_dirs: int,
    active_dir: Optional[Path] = None,
    now: Optional[float] = None,
    dry_run: bool = False,
) -> List[str]:
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.exists():
        return []
    if not resolved_root.is_dir():
        raise ValueError(f"runtime log root is not a directory: {resolved_root}")

    active = Path(active_dir).expanduser().resolve() if active_dir else None
    timestamp_dirs = sorted(
        (
            path.resolve()
            for path in resolved_root.iterdir()
            if path.is_dir() and _RUNTIME_DIR_RE.fullmatch(path.name)
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    keep_count = max(1, int(max_dirs or 1))
    cutoff = float(now if now is not None else time.time()) - max(
        1,
        int(retention_days or 1),
    ) * 86400
    removed: List[str] = []
    for index, path in enumerate(timestamp_dirs):
        if active is not None and path == active:
            continue
        too_old = path.stat().st_mtime < cutoff
        exceeds_count = index >= keep_count
        if not (too_old or exceeds_count):
            continue
        if path.parent != resolved_root:
            raise ValueError(f"refusing to prune path outside runtime root: {path}")
        removed.append(str(path))
        if not dry_run:
            shutil.rmtree(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--retention-days", type=int, default=14)
    parser.add_argument("--max-dirs", type=int, default=30)
    parser.add_argument("--active-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    removed = prune_runtime_log_dirs(
        args.root,
        retention_days=args.retention_days,
        max_dirs=args.max_dirs,
        active_dir=args.active_dir,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps({"removed": removed, "count": len(removed)}, ensure_ascii=False))
    else:
        print(f"[RETENTION] runtime_log_dirs_removed={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
