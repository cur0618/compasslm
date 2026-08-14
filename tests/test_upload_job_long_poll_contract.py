import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UploadJobLongPollContractTests(unittest.TestCase):
    def test_main_exposes_upload_job_long_poll_contract(self):
        source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
        self.assertIn('"version"', source)
        self.assertIn("since_version", source)
        self.assertIn("wait_seconds", source)
        self.assertIn("status_code=204", source)
        self.assertIn("return Response(status_code=204)", source)


if __name__ == "__main__":
    unittest.main()
