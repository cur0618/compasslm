import os
import re
import time
import uuid
import zipfile
from typing import Dict, Iterable, Optional, Set


def safe_upload_filename(name: str) -> str:
    normalized = (name or "").strip().replace("\\", "/")
    base = os.path.basename(normalized).replace("\x00", "")
    if not base:
        raise ValueError("파일 이름이 비어 있습니다.")
    safe = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", base)
    safe = safe.strip("._")
    if not safe:
        raise ValueError("파일 이름 형식이 올바르지 않습니다.")
    return safe


def build_stored_upload_name(original_name: str) -> str:
    safe = safe_upload_filename(original_name)
    stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    suffix = uuid.uuid4().hex[:10]
    return f"{stamp}_{suffix}__{safe}"


def validate_upload_meta(
    filename: str,
    content_type: Optional[str],
    allowed_extensions: Iterable[str],
    allowed_mime_by_ext: Dict[str, Set[str]],
) -> str:
    safe_name = safe_upload_filename(filename or "")
    ext = os.path.splitext(safe_name)[1].lower()
    allowed_ext_set = {x.lower() for x in allowed_extensions}
    if ext not in allowed_ext_set:
        raise ValueError(".txt, .xlsx, .pdf, .hwpx 파일만 올릴 수 있습니다.")

    mime = ((content_type or "").split(";")[0]).strip().lower()
    allowed_mime = allowed_mime_by_ext.get(ext, {""})
    if mime and mime not in allowed_mime:
        raise ValueError(f"파일 형식이 맞지 않습니다. ({ext}, {mime})")

    return safe_name


def is_zip_signature(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return len(sig) >= 2 and sig[:2] == b"PK"
    except Exception:
        return False


def is_hwpx_signature(path: str) -> bool:
    try:
        if not is_zip_signature(path):
            return False
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            name_by_lower = {name.lower(): name for name in names}
            lowered = set(name_by_lower)
            has_section = any(
                (
                    name.startswith("contents/section")
                    or name.startswith("contents/body/section")
                    or name.startswith("contents/sections/")
                )
                and name.endswith(".xml")
                for name in lowered
            )
            has_package = any(
                name == "contents/content.hpf"
                or name == "contents/header.xml"
                or name == "meta-inf/container.xml"
                or name.endswith("/content.hpf")
                or name.endswith("/container.xml")
                or name.endswith(".hpf")
                for name in lowered
            )
            has_owpml_hint = any(
                marker in name
                for name in lowered
                for marker in (
                    "owpml",
                    "contents/",
                    "meta-inf/",
                )
            )
            if "mimetype" in lowered:
                try:
                    mime_text = zf.read(name_by_lower["mimetype"]).decode("utf-8", errors="ignore").lower()
                except Exception:
                    mime_text = ""
                if "hwp" in mime_text or "hwpx" in mime_text:
                    return has_section or has_package or has_owpml_hint
            return has_section and has_package
    except Exception:
        return False


def is_pdf_signature(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(5)
        return sig == b"%PDF-"
    except Exception:
        return False
