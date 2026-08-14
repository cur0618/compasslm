import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src import pdf_ocr


class PdfOCRHPSTests(unittest.TestCase):
    def test_hps_backend_skips_local_persistent_worker_warmup(self):
        with mock.patch.dict("os.environ", {"PDF_OCR_BACKEND": "hps"}, clear=False):
            result = pdf_ocr.warmup_persistent_ocr_worker(device="gpu:0")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "hps_backend_selected")

    def test_batch_timing_summary_reports_percentiles(self):
        summary = pdf_ocr._summarize_batch_wall_seconds([1.0, 2.0, 3.0, 10.0])

        self.assertEqual(summary["ocr_batch_wall_seconds_mean"], 4.0)
        self.assertEqual(summary["ocr_batch_wall_seconds_p50"], 2.5)
        self.assertEqual(summary["ocr_batch_wall_seconds_p95"], 8.95)
        self.assertEqual(summary["ocr_batch_wall_seconds_max"], 10.0)

    def test_materialize_output_separates_generator_time(self):
        def _lazy_output():
            time.sleep(0.02)
            yield {"page_no": 1, "text": "first"}
            time.sleep(0.02)
            yield {"page_no": 2, "text": "second"}

        items, elapsed = pdf_ocr._materialize_ocr_output(_lazy_output())

        self.assertEqual(len(items), 2)
        self.assertGreaterEqual(elapsed, 0.03)

    def test_normalize_hps_response_preserves_page_order_and_markdown(self):
        response = {
            "result": {
                "layoutParsingResults": [
                    {"pageNo": 2, "markdown": {"text": "page two"}},
                    {"pageNo": 1, "markdown": {"text": "page one"}},
                ]
            }
        }

        pages = pdf_ocr._normalize_hps_layout_response(response, min_text_chars=4)

        self.assertEqual(
            pages,
            [
                {"page_no": 1, "text": "page one"},
                {"page_no": 2, "text": "page two"},
            ],
        )

    def test_hps_backend_falls_back_to_local_when_enabled(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            with mock.patch.dict(
                "os.environ",
                {
                    "PDF_OCR_BACKEND": "hps",
                    "PDF_OCR_HPS_FALLBACK_TO_LOCAL": "1",
                    "PDF_OCR_HPS_URL": "http://127.0.0.1:18080",
                },
                clear=False,
            ), mock.patch.object(
                pdf_ocr,
                "_execute_hps_ocr_with_runtime",
                side_effect=RuntimeError("HPS unavailable"),
            ), mock.patch.object(
                pdf_ocr,
                "_execute_local_paddleocr_vl_with_runtime",
                return_value=([{"page_no": 1, "text": "local"}], {"ocr_backend": "local"}),
            ) as local_call:
                pages, runtime = pdf_ocr._execute_paddleocr_vl_with_runtime(handle.name)

        self.assertEqual(pages, [{"page_no": 1, "text": "local"}])
        self.assertTrue(runtime["ocr_backend_fallback_used"])
        self.assertEqual(runtime["ocr_backend_attempted"], "hps")
        local_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
