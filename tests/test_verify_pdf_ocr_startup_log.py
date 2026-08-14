import tempfile
import unittest
from pathlib import Path

from scripts.verify_pdf_ocr_startup_log import verify_pdf_ocr_startup_log


class VerifyPdfOcrStartupLogTests(unittest.TestCase):
    def _write_log(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".log", encoding="utf-8", delete=False)
        with handle:
            handle.write(text)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_accepts_fully_warmed_bounded_rolling_window(self):
        path = self._write_log(
            "\n".join(
                [
                    "[PDF_OCR][WARMUP] status=ready device=gpu:0 worker_count=3 worker_pids=101,102,103 model_load_seconds=5.0",
                    "[PDF_OCR][START] device=gpu:0 total_pages=200 worker_count=3 exec_batch_pages=3 batch_count=67",
                    "[PDF_OCR][BATCH_SUBMIT] batch=1 original_pages=1-3",
                    "[PDF_OCR][BATCH_SUBMIT] batch=2 original_pages=4-6",
                    "[PDF_OCR][BATCH_SUBMIT] batch=3 original_pages=7-9",
                    "[PDF_OCR][SERIAL_PREDICT_DONE] original_pages=1-3",
                    "[PDF_OCR][BATCH] device=gpu:0 mode=parallel_gpu batch=1/67 pages=1-3",
                    "[PDF_OCR][BATCH_SUBMIT] batch=4 original_pages=10-12",
                ]
            )
        )

        result = verify_pdf_ocr_startup_log(path, expected_workers=3)

        self.assertTrue(result["ok"])
        self.assertEqual(result["initial_submit_count"], 3)
        self.assertEqual(result["warmup_worker_count"], 3)
        self.assertEqual(result["warmup_worker_pids"], [101, 102, 103])

    def test_accepts_single_warmed_worker_and_single_gpu_batch(self):
        path = self._write_log(
            "\n".join(
                [
                    "[PDF_OCR][WARMUP] status=ready device=gpu:0 worker_count=1 worker_pids=101 model_load_seconds=5.0",
                    "[PDF_OCR][START] device=gpu:0 total_pages=400 worker_count=1 exec_batch_pages=3 batch_count=134",
                    "[PDF_OCR][BATCH_SUBMIT] batch=1 original_pages=1-3",
                    "[PDF_OCR][SERIAL_PREDICT_DONE] original_pages=1-3",
                    "[PDF_OCR][BATCH] device=gpu:0 mode=single_gpu batch=1/134 pages=1-3",
                    "[PDF_OCR][BATCH_SUBMIT] batch=2 original_pages=4-6",
                ]
            )
        )

        result = verify_pdf_ocr_startup_log(path, expected_workers=1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["initial_submit_count"], 1)
        self.assertEqual(result["warmup_worker_count"], 1)
        self.assertEqual(result["warmup_worker_pids"], [101])

    def test_rejects_original_unbounded_submission_pattern(self):
        path = self._write_log(
            "\n".join(
                [
                    "[PDF_OCR][WARMUP] status=ready device=gpu:0 model_load_seconds=5.0",
                    "[PDF_OCR][START] device=gpu:0 total_pages=200 worker_count=3 exec_batch_pages=3 batch_count=67",
                    *[
                        f"[PDF_OCR][BATCH_SUBMIT] batch={index} original_pages={index}-{index}"
                        for index in range(1, 68)
                    ],
                    "Creating model: ('PaddleOCR-VL', '/models/ocr')",
                    "[UPLOAD][OCR_PROGRESS] retry_reason=gpu_timeout",
                    "[PDF_OCR][BATCH] device=cpu mode=cpu batch=1/67 pages=1-3",
                ]
            )
        )

        result = verify_pdf_ocr_startup_log(path, expected_workers=3)

        self.assertFalse(result["ok"])
        self.assertGreater(result["initial_submit_count"], 3)
        self.assertIn("warmup_worker_count_missing", result["errors"])
        self.assertIn("initial_submit_window_exceeded", result["errors"])
        self.assertIn("model_reloaded_after_ocr_start", result["errors"])
        self.assertIn("gpu_timeout_before_first_batch", result["errors"])

    def test_requires_a_completed_gpu_batch(self):
        path = self._write_log(
            "\n".join(
                [
                    "[PDF_OCR][WARMUP] status=ready device=gpu:0 worker_count=3 worker_pids=101,102,103 model_load_seconds=5.0",
                    "[PDF_OCR][START] device=gpu:0 total_pages=200 worker_count=3 exec_batch_pages=3 batch_count=67",
                    "[PDF_OCR][BATCH_SUBMIT] batch=1 original_pages=1-3",
                    "[PDF_OCR][BATCH_SUBMIT] batch=2 original_pages=4-6",
                    "[PDF_OCR][BATCH_SUBMIT] batch=3 original_pages=7-9",
                ]
            )
        )

        result = verify_pdf_ocr_startup_log(path, expected_workers=3)

        self.assertFalse(result["ok"])
        self.assertIn("first_gpu_batch_missing", result["errors"])

    def test_rejects_duplicate_warmup_worker_pids(self):
        path = self._write_log(
            "\n".join(
                [
                    "[PDF_OCR][WARMUP] status=ready device=gpu:0 worker_count=3 worker_pids=101,101,102 model_load_seconds=5.0",
                    "[PDF_OCR][START] device=gpu:0 total_pages=3 worker_count=3 exec_batch_pages=1 batch_count=3",
                    "[PDF_OCR][BATCH_SUBMIT] batch=1 original_pages=1-1",
                    "[PDF_OCR][BATCH_SUBMIT] batch=2 original_pages=2-2",
                    "[PDF_OCR][BATCH_SUBMIT] batch=3 original_pages=3-3",
                    "[PDF_OCR][BATCH] device=gpu:0 mode=parallel_gpu batch=1/3 pages=1-1",
                ]
            )
        )

        result = verify_pdf_ocr_startup_log(path, expected_workers=3)

        self.assertFalse(result["ok"])
        self.assertIn("warmup_worker_pids_mismatch", result["errors"])


if __name__ == "__main__":
    unittest.main()
