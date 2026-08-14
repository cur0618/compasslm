import unittest

from src.upload_progress import build_upload_stall_state, estimate_display_progress_percent
from src.upload_progress import estimate_background_ocr_progress_percent
from src.upload_progress import update_upload_phase_state, upload_failure_default_for_stage


class UploadProgressEstimationTests(unittest.TestCase):
    def test_two_phase_upload_state_tracks_ocr_and_index_completion_separately(self):
        state = {
            "ocr_completed": False,
            "index_completed": False,
            "embedding_started_at": 0,
            "embedding_completed_at": 0,
            "stage": "queued",
        }

        state = update_upload_phase_state(state, "store_ocr_cache", now_ts=100)
        self.assertTrue(state["ocr_completed"])
        self.assertFalse(state["index_completed"])
        self.assertEqual(state["embedding_started_at"], 0)

        state = update_upload_phase_state(state, "embed_chunks", now_ts=120)
        self.assertTrue(state["ocr_completed"])
        self.assertFalse(state["index_completed"])
        self.assertEqual(state["embedding_started_at"], 120)
        self.assertEqual(state["embedding_completed_at"], 0)

        state = update_upload_phase_state(state, "done", now_ts=150)
        self.assertTrue(state["index_completed"])
        self.assertEqual(state["embedding_started_at"], 120)
        self.assertEqual(state["embedding_completed_at"], 150)

    def test_two_phase_failure_default_distinguishes_embedding_and_index_failures(self):
        self.assertEqual(
            upload_failure_default_for_stage("run_pdf_ocr", ocr_completed=False),
            "upload_ingest_fail",
        )
        self.assertEqual(
            upload_failure_default_for_stage("embed_chunks", ocr_completed=True),
            "upload_embedding_fail",
        )
        self.assertEqual(
            upload_failure_default_for_stage("refresh_index", ocr_completed=True),
            "upload_index_fail",
        )

    def test_run_pdf_ocr_stage_creeps_forward_with_elapsed_time(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 42,
            "processing_started_at": 100,
            "current_page": 0,
            "total_pages": 24,
        }

        estimated = estimate_display_progress_percent(payload, now_ts=190)

        self.assertGreater(estimated, 42)
        self.assertLessEqual(estimated, 76)

    def test_chunk_pdf_stage_uses_page_ratio_when_available(self):
        payload = {
            "status": "processing",
            "progress_stage": "chunk_pdf",
            "progress_percent": 58,
            "processing_started_at": 100,
            "current_page": 5,
            "total_pages": 10,
        }

        estimated = estimate_display_progress_percent(payload, now_ts=140)

        self.assertGreaterEqual(estimated, 66)
        self.assertLessEqual(estimated, 92)

    def test_run_pdf_ocr_stage_uses_page_ratio_when_available(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 42,
            "processing_started_at": 100,
            "current_page": 30,
            "total_pages": 40,
        }

        estimated = estimate_display_progress_percent(payload, now_ts=140)

        self.assertGreaterEqual(estimated, 68)
        self.assertLessEqual(estimated, 76)

    def test_background_ocr_progress_uses_ocr_target_ratio_not_upload_stage_percent(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 75,
            "ocr_completed_pages": 0,
            "ocr_target_pages": 49,
        }

        self.assertEqual(estimate_background_ocr_progress_percent(payload), 5)

        payload["ocr_completed_pages"] = 25
        self.assertGreaterEqual(estimate_background_ocr_progress_percent(payload), 52)
        self.assertLessEqual(estimate_background_ocr_progress_percent(payload), 54)

        payload["ocr_completed_pages"] = 49
        self.assertEqual(estimate_background_ocr_progress_percent(payload), 99)

    def test_terminal_status_keeps_original_percent(self):
        payload = {
            "status": "success",
            "progress_stage": "done",
            "progress_percent": 100,
            "processing_started_at": 100,
        }

        estimated = estimate_display_progress_percent(payload, now_ts=180)

        self.assertEqual(estimated, 100)

    def test_run_pdf_ocr_job_is_marked_stalled_after_missing_heartbeat(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 52,
            "processing_started_at": 100,
            "last_progress_at": 100,
            "total_pages": 24,
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=900,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNotNone(stalled)
        self.assertEqual(stalled["status"], "error")
        self.assertEqual(stalled["progress_stage"], "error")
        self.assertEqual(stalled["failure_code"], "upload_job_stalled")
        self.assertIn("PDF OCR", stalled["message"])
        self.assertGreaterEqual(int(stalled["stall_seconds"]), 800)

    def test_run_pdf_ocr_stall_message_mentions_gpu_fallback_controls(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 52,
            "processing_started_at": 100,
            "last_progress_at": 100,
            "total_pages": 24,
            "ocr_device_attempted": "gpu:0",
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=900,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNotNone(stalled)
        self.assertIn("GPU", stalled["message"])
        self.assertIn("CPU fallback", stalled["message"])

    def test_queued_job_is_marked_stalled_after_excessive_wait(self):
        payload = {
            "status": "queued",
            "progress_stage": "queued",
            "created_at": 100,
            "last_progress_at": 100,
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=400,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNotNone(stalled)
        self.assertEqual(stalled["failure_code"], "upload_job_stalled")
        self.assertIn("대기", stalled["message"])

    def test_processing_job_before_timeout_is_not_marked_stalled(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 52,
            "processing_started_at": 100,
            "last_progress_at": 430,
            "total_pages": 24,
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=600,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNone(stalled)

    def test_run_pdf_ocr_job_is_not_marked_stalled_when_recent_ocr_heartbeat_exists(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 52,
            "processing_started_at": 100,
            "last_progress_at": 100,
            "ocr_heartbeat_at": 860,
            "total_pages": 24,
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=900,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNone(stalled)

    def test_run_pdf_ocr_job_is_not_marked_stalled_when_recent_batch_completion_exists(self):
        payload = {
            "status": "processing",
            "progress_stage": "run_pdf_ocr",
            "progress_percent": 52,
            "processing_started_at": 100,
            "last_progress_at": 100,
            "ocr_heartbeat_at": 100,
            "ocr_last_batch_completed_at": 860,
            "total_pages": 24,
        }

        stalled = build_upload_stall_state(
            payload,
            now_ts=900,
            processing_timeout_seconds=480,
            queue_timeout_seconds=180,
        )

        self.assertIsNone(stalled)


if __name__ == "__main__":
    unittest.main()
