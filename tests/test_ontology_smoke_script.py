import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.smoke_ontology_rebuild_llm import SmokeConfig, load_config, run_smoke


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.job_reads = 0

    def post(self, url, **kwargs):
        if url.endswith("/auth/login"):
            return _FakeResponse({"status": "success"})
        if "/ontology/rebuild?include_llm=1" in url:
            return _FakeResponse({
                "status": "accepted",
                "ontology_rebuild_job_id": "job-1",
            })
        raise AssertionError(url)

    def get(self, url, **kwargs):
        if url.endswith("/ontology/rebuild/jobs/job-1"):
            self.job_reads += 1
            status = "processing" if self.job_reads == 1 else "success"
            return _FakeResponse({
                "job": {
                    "status": status,
                    "progress_percent": 50 if status == "processing" else 100,
                    "ontology_extraction_errors": 0,
                    "ontology_extraction_disabled": False,
                }
            })
        if "/ontology/facts?" in url:
            return _FakeResponse({"facts": [{"fact_id": 7}]})
        if url.endswith("/ontology/facts/7"):
            return _FakeResponse({
                "fact": {
                    "fact_id": 7,
                    "subject": "농가경제조사",
                    "predicate": "지급단가",
                    "sources": [{"chunk_id": 1, "evidence_quote": "지급단가는 40천원"}],
                }
            })
        if "/ontology/search?" in url:
            if "__ontology_smoke_no_match_" in url:
                return _FakeResponse({"matches": []})
            return _FakeResponse({
                "matches": [{"fact_id": 7, "chunk_id": 1, "ontology_hop_count": 1}]
            })
        raise AssertionError(url)


class OntologySmokeScriptTests(unittest.TestCase):
    def test_load_config_requires_operational_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "COMPASSLM_BASE_URL"):
                load_config()

    def test_load_config_uses_ready_backend_state_when_base_url_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "ports.env"
            state_path.write_text(
                "\n".join([
                    "API_PORT_SELECTED=8124",
                    "COMPASSLM_BASE_URL_SELECTED=http://127.0.0.1:8124",
                    f"API_PID={os.getpid()}",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "COMPASS_PORT_STATE_FILE": str(state_path),
                "COMPASSLM_ADMIN_LOGIN": "admin",
                "COMPASSLM_ADMIN_PASSWORD": "replace-with-strong-secret",
                "COMPASSLM_SMOKE_KB": "default",
            }, clear=True):
                config = load_config()

        self.assertEqual(config.base_url, "http://127.0.0.1:8124")

    def test_run_smoke_logs_in_polls_job_and_checks_fact_evidence(self):
        config = SmokeConfig(
            base_url="http://127.0.0.1:8000",
            admin_login="admin",
            admin_password="replace-with-strong-secret",
            kb_name="default",
            timeout_seconds=5.0,
            poll_interval_seconds=0.0,
        )

        result = run_smoke(config, session=_FakeSession(), sleep=lambda _seconds: None)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(result["fact_id"], 7)
        self.assertEqual(result["ontology_extraction_errors"], 0)
        self.assertEqual(result["positive_query_match_count"], 1)
        self.assertEqual(result["negative_query_match_count"], 0)

    def test_main_writes_smoke_result_when_output_path_is_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "smoke.json"
            with patch.dict(os.environ, {
                "COMPASSLM_BASE_URL": "http://127.0.0.1:8000",
                "COMPASSLM_ADMIN_LOGIN": "admin",
                "COMPASSLM_ADMIN_PASSWORD": "replace-with-strong-secret",
                "COMPASSLM_SMOKE_KB": "default",
                "COMPASSLM_SMOKE_OUTPUT": str(output_path),
            }, clear=True), patch(
                "scripts.smoke_ontology_rebuild_llm.requests.Session",
                return_value=_FakeSession(),
            ), patch("scripts.smoke_ontology_rebuild_llm.time.sleep", return_value=None):
                from scripts.smoke_ontology_rebuild_llm import main

                self.assertEqual(main(), 0)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["config"]["admin_password"], "***")


if __name__ == "__main__":
    unittest.main()
