import tempfile
import unittest
import zipfile
from pathlib import Path

from src.upload_helpers import is_hwpx_signature, validate_upload_meta


class HwpxUploadValidationTests(unittest.TestCase):
    def test_validate_upload_meta_accepts_hwpx_extension_and_zip_mime(self):
        safe_name = validate_upload_meta(
            "보고서.hwpx",
            "application/zip",
            allowed_extensions={".txt", ".xlsx", ".pdf", ".hwpx"},
            allowed_mime_by_ext={
                ".hwpx": {
                    "",
                    "application/zip",
                    "application/octet-stream",
                    "application/haansofthwpx",
                }
            },
        )

        self.assertEqual(safe_name, "보고서.hwpx")

    def test_validate_upload_meta_accepts_hancom_hwpx_mime(self):
        safe_name = validate_upload_meta(
            "보고서.hwpx",
            "application/haansofthwpx",
            allowed_extensions={".txt", ".xlsx", ".pdf", ".hwpx"},
            allowed_mime_by_ext={".hwpx": {"", "application/haansofthwpx"}},
        )

        self.assertEqual(safe_name, "보고서.hwpx")

    def test_is_hwpx_signature_accepts_owpml_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            hwpx_path = Path(tmp) / "valid.hwpx"
            with zipfile.ZipFile(hwpx_path, "w") as zf:
                zf.writestr("mimetype", "application/hwp+zip")
                zf.writestr("Contents/content.hpf", "<package/>")
                zf.writestr("Contents/section0.xml", "<section/>")

            self.assertTrue(is_hwpx_signature(str(hwpx_path)))

    def test_is_hwpx_signature_rejects_plain_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "plain.hwpx"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("hello.txt", "not an hwpx document")

            self.assertFalse(is_hwpx_signature(str(zip_path)))


if __name__ == "__main__":
    unittest.main()
