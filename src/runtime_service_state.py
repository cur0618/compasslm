from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict


def runtime_state_file() -> Path:
    explicit = str(os.getenv("COMPASS_PORT_STATE_FILE", "") or "").strip()
    if explicit:
        return Path(explicit)
    state_dir = str(os.getenv("COMPASS_RUNTIME_STATE_DIR", "") or "").strip()
    if state_dir:
        return Path(state_dir) / "ports.env"
    project_root = Path(
        str(os.getenv("COMPASSLM_HOME", "") or Path(__file__).resolve().parents[1])
    ).resolve()
    project_hash = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:12]
    runtime_root = Path(str(os.getenv("XDG_RUNTIME_DIR", "") or "/tmp"))
    return runtime_root / f"compasslm-{os.getuid()}-{project_hash}" / "ports.env"


def read_runtime_state(path: Path | None = None) -> Dict[str, str]:
    state_path = path or runtime_state_file()
    if not state_path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in state_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _pid_is_alive(raw_pid: str) -> bool:
    try:
        pid = int(raw_pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def resolve_ready_backend_url() -> str:
    state_path = runtime_state_file()
    state = read_runtime_state(state_path)
    url = str(state.get("COMPASSLM_BASE_URL_SELECTED", "") or "").strip().rstrip("/")
    port = str(state.get("API_PORT_SELECTED", "") or "").strip()
    pid = str(state.get("API_PID", "") or "").strip()
    if not url or not port or not pid:
        raise ValueError(f"backend ready state is missing: {state_path}")
    if not _pid_is_alive(pid):
        raise ValueError(f"backend ready state process is not alive: pid={pid}")
    return url
