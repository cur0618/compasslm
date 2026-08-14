import tempfile
import unittest
from pathlib import Path

from src.upload_job_store import UploadJobStore


class UploadJobStoreTests(unittest.TestCase):
    def test_upload_job_snapshots_persist_by_user_and_kb(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "app.sqlite")
            store = UploadJobStore(db_path)

            store.save_job(
                {
                    "job_id": "job-a",
                    "user_id": "alice",
                    "kb_name": "alice__kb",
                    "stored_filename": "file_a.pdf",
                    "original_filename": "a.pdf",
                    "status": "queued",
                    "progress_percent": 0,
                    "progress_stage": "queued",
                    "created_at": 10,
                    "updated_at": 10,
                }
            )
            store.save_job(
                {
                    "job_id": "job-b",
                    "user_id": "bob",
                    "kb_name": "bob__kb",
                    "stored_filename": "file_b.pdf",
                    "original_filename": "b.pdf",
                    "status": "success",
                    "progress_percent": 100,
                    "progress_stage": "done",
                    "created_at": 20,
                    "updated_at": 20,
                }
            )

            reloaded = UploadJobStore(db_path)
            self.assertEqual(reloaded.get_job("job-a")["user_id"], "alice")
            self.assertEqual([job["job_id"] for job in reloaded.list_jobs(user_id="alice")], ["job-a"])
            self.assertEqual([job["job_id"] for job in reloaded.list_jobs(user_id="bob")], ["job-b"])
            self.assertEqual(reloaded.list_jobs(user_id="alice", kb_name="bob__kb"), [])

    def test_upload_job_updates_status_and_timing_columns(self):
        with tempfile.TemporaryDirectory() as td:
            store = UploadJobStore(str(Path(td) / "app.sqlite"))
            store.save_job(
                {
                    "job_id": "job-a",
                    "user_id": "alice",
                    "kb_name": "kb",
                    "stored_filename": "file.pdf",
                    "status": "queued",
                    "created_at": 10,
                    "updated_at": 10,
                }
            )

            store.update_job(
                "job-a",
                {
                    "status": "processing",
                    "progress_percent": 40,
                    "progress_stage": "run_pdf_ocr",
                    "processing_started_at": 15,
                    "updated_at": 16,
                },
            )
            row = store.get_job("job-a")
            self.assertEqual(row["status"], "processing")
            self.assertEqual(row["progress_percent"], 40)
            self.assertEqual(row["progress_stage"], "run_pdf_ocr")
            self.assertEqual(row["processing_started_at"], 15)

            store.update_job(
                "job-a",
                {
                    "status": "error",
                    "failure_code": "ocr_runtime_missing",
                    "message": "failed",
                    "completed_at": 30,
                    "updated_at": 31,
                },
            )
            row = store.get_job("job-a")
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["failure_code"], "ocr_runtime_missing")
            self.assertEqual(row["message"], "failed")
            self.assertEqual(row["completed_at"], 30)

    def test_incomplete_jobs_returns_only_restart_recoverable_statuses(self):
        with tempfile.TemporaryDirectory() as td:
            store = UploadJobStore(str(Path(td) / "app.sqlite"))
            for job_id, status, updated_at in [
                ("job-queued", "queued", 10),
                ("job-processing", "processing", 20),
                ("job-success", "success", 30),
                ("job-error", "error", 40),
            ]:
                store.save_job(
                    {
                        "job_id": job_id,
                        "user_id": "alice",
                        "kb_name": "kb",
                        "stored_filename": f"{job_id}.pdf",
                        "original_filename": f"{job_id}.pdf",
                        "stored_path": str(Path(td) / f"{job_id}.pdf"),
                        "status": status,
                        "progress_stage": status,
                        "created_at": updated_at,
                        "updated_at": updated_at,
                    }
                )

            self.assertEqual(
                [job["job_id"] for job in store.list_incomplete_jobs()],
                ["job-processing", "job-queued"],
            )


if __name__ == "__main__":
    unittest.main()
