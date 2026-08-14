from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_upload_route_allows_hwpx_and_routes_it_to_fast_lane():
    main_source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")

    assert 'ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".xlsx", ".pdf", ".hwpx"}' in main_source
    assert 'FAST_LANE_UPLOAD_EXTENSIONS = {".txt", ".xlsx", ".hwpx"}' in main_source
    assert "is_hwpx_signature" in main_source
    assert 'if ext == ".hwpx" and not _is_hwpx_signature(stored_path):' in main_source


def test_rag_ingest_has_hwpx_source_type_and_parser_contract():
    rag_source = (ROOT / "src" / "rag.py").read_text(encoding="utf-8")

    assert "from src.hwpx_loader import load_hwpx_records" in rag_source
    assert 'if ext == ".hwpx"' in rag_source
    assert 'else "hwpx"' in rag_source
    assert 'parser_name = "python_hwpx"' in rag_source
    assert 'source_type == "hwpx"' in rag_source


def test_offline_bundle_scripts_require_python_hwpx_wheel_and_import():
    prepare_source = (ROOT / "project-gpu" / "embedding-gpu-server" / "prepare_offline_packages_linux.sh").read_text(encoding="utf-8")
    check_source = (ROOT / "project-gpu" / "check_gpu_assets.sh").read_text(encoding="utf-8")
    setup_source = (ROOT / "project-gpu" / "setup_gpu_track.sh").read_text(encoding="utf-8")

    assert "PYTHON_HWPX_COUNT" in prepare_source
    assert "python_hwpx-*.whl" in prepare_source
    assert "python_hwpx-*.whl" in check_source
    assert '"hwpx": "python-hwpx"' in setup_source
    assert "TextExtractor" in setup_source
