from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.runtime_service_state import resolve_ready_backend_url


@dataclass(frozen=True)
class SmokeConfig:
    base_url: str
    admin_login: str
    admin_password: str
    kb_name: str
    timeout_seconds: float = 300.0
    poll_interval_seconds: float = 2.0


def load_config() -> SmokeConfig:
    configured_base_url = os.getenv("COMPASSLM_BASE_URL", "").strip()
    if not configured_base_url:
        try:
            configured_base_url = resolve_ready_backend_url()
        except ValueError:
            configured_base_url = ""
    required = {
        "COMPASSLM_BASE_URL": configured_base_url,
        "COMPASSLM_ADMIN_LOGIN": os.getenv("COMPASSLM_ADMIN_LOGIN", "").strip(),
        "COMPASSLM_ADMIN_PASSWORD": os.getenv("COMPASSLM_ADMIN_PASSWORD", ""),
        "COMPASSLM_SMOKE_KB": os.getenv("COMPASSLM_SMOKE_KB", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")
    return SmokeConfig(
        base_url=required["COMPASSLM_BASE_URL"].rstrip("/"),
        admin_login=required["COMPASSLM_ADMIN_LOGIN"],
        admin_password=required["COMPASSLM_ADMIN_PASSWORD"],
        kb_name=required["COMPASSLM_SMOKE_KB"],
        timeout_seconds=float(os.getenv("COMPASSLM_SMOKE_TIMEOUT_SECONDS", "300")),
        poll_interval_seconds=float(os.getenv("COMPASSLM_SMOKE_POLL_SECONDS", "2")),
    )


def _json(response: Any) -> Dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("CompassLM API response must be a JSON object")
    return payload


def run_smoke(
    config: SmokeConfig,
    *,
    session: Optional[Any] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    client = session or requests.Session()
    base_url = config.base_url.rstrip("/")
    kb = quote(config.kb_name, safe="")
    _json(client.post(
        f"{base_url}/auth/login",
        json={"login_id": config.admin_login, "password": config.admin_password},
        timeout=30,
    ))
    accepted = _json(client.post(
        f"{base_url}/kbs/{kb}/ontology/rebuild?include_llm=1",
        timeout=30,
    ))
    job_id = str(accepted.get("ontology_rebuild_job_id", "") or "")
    if not job_id:
        raise RuntimeError(f"ontology rebuild was not queued: {accepted}")

    deadline = time.monotonic() + max(1.0, config.timeout_seconds)
    job: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        job_payload = _json(client.get(
            f"{base_url}/kbs/{kb}/ontology/rebuild/jobs/{quote(job_id, safe='')}",
            timeout=30,
        ))
        job = job_payload.get("job") if isinstance(job_payload.get("job"), dict) else {}
        status = str(job.get("status", "") or "").lower()
        if status in {"success", "error", "cancelled"}:
            break
        sleep(max(0.0, config.poll_interval_seconds))
    else:
        raise TimeoutError(f"ontology rebuild job timed out: {job_id}")

    if str(job.get("status", "") or "").lower() != "success":
        raise RuntimeError(f"ontology rebuild job failed: {job}")
    if int(job.get("progress_percent", 0) or 0) != 100:
        raise RuntimeError(f"ontology rebuild completed without 100% progress: {job}")
    if bool(job.get("ontology_extraction_disabled", False)):
        reason = str(job.get("ontology_extraction_disabled_reason", "") or "")
        if not reason:
            raise RuntimeError("LLM extraction was disabled without a reason")

    facts_payload = _json(client.get(
        f"{base_url}/kbs/{kb}/ontology/facts?limit=20",
        timeout=30,
    ))
    facts = facts_payload.get("facts") if isinstance(facts_payload.get("facts"), list) else []
    if not facts:
        raise RuntimeError("ontology rebuild completed but returned no facts")
    fact_id = int(facts[0].get("fact_id", 0) or 0)
    detail_payload = _json(client.get(
        f"{base_url}/kbs/{kb}/ontology/facts/{fact_id}",
        timeout=30,
    ))
    fact = detail_payload.get("fact") if isinstance(detail_payload.get("fact"), dict) else {}
    sources = fact.get("sources") if isinstance(fact.get("sources"), list) else []
    if not sources or not any(
        int(source.get("chunk_id", 0) or 0) > 0
        and bool(str(source.get("evidence_quote", "") or "").strip())
        for source in sources
        if isinstance(source, dict)
    ):
        raise RuntimeError("fact detail did not expose chunk evidence")
    subject = str(fact.get("subject", "") or "").strip()
    predicate = str(fact.get("predicate", "") or "").strip()
    positive_query = " ".join(value for value in (subject, predicate) if value).strip()
    if not positive_query:
        raise RuntimeError("fact detail did not expose subject/predicate for search smoke")
    positive_payload = _json(client.get(
        f"{base_url}/kbs/{kb}/ontology/search?{urlencode({'query': positive_query, 'limit': 5})}",
        timeout=30,
    ))
    positive_matches = positive_payload.get("matches") if isinstance(positive_payload.get("matches"), list) else []
    if not any(int(match.get("fact_id", 0) or 0) == fact_id for match in positive_matches if isinstance(match, dict)):
        raise RuntimeError("ontology positive query did not return the inspected fact")
    negative_query = "__ontology_smoke_no_match_7f3c9d1e__"
    negative_payload = _json(client.get(
        f"{base_url}/kbs/{kb}/ontology/search?{urlencode({'query': negative_query, 'limit': 5})}",
        timeout=30,
    ))
    negative_matches = negative_payload.get("matches") if isinstance(negative_payload.get("matches"), list) else []
    if negative_matches:
        raise RuntimeError("ontology negative query returned unexpected matches")
    return {
        "status": "success",
        "job_id": job_id,
        "fact_id": fact_id,
        "ontology_extraction_errors": int(job.get("ontology_extraction_errors", 0) or 0),
        "ontology_extraction_disabled": bool(job.get("ontology_extraction_disabled", False)),
        "ontology_extraction_disabled_reason": str(job.get("ontology_extraction_disabled_reason", "") or ""),
        "positive_query": positive_query,
        "positive_query_match_count": len(positive_matches),
        "negative_query_match_count": len(negative_matches),
    }


def main() -> int:
    try:
        config = load_config()
        result = run_smoke(config)
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    output = {"config": {**asdict(config), "admin_password": "***"}, **result}
    output_text = json.dumps(output, ensure_ascii=False, sort_keys=True)
    output_path = os.getenv("COMPASSLM_SMOKE_OUTPUT", "").strip()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
