import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_EXAMPLE = ROOT / "project-gpu" / "runtime.env.example"
BACKEND_EXAMPLE = ROOT / "project-gpu" / "main-backend" / ".env.example"
EMBEDDING_EXAMPLE = ROOT / "project-gpu" / "embedding-gpu-server" / ".env.example"
SETUP_SCRIPT = ROOT / "project-gpu" / "setup_gpu_track.sh"
SHELL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_shell_assignments(text: str) -> dict[str, str]:
    """Parse active KEY=value lines as text; never execute or expand them."""
    assignments: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if SHELL_IDENTIFIER.fullmatch(key):
            assignments[key] = value.strip()
    return assignments


def read_assignments(path: Path) -> dict[str, str]:
    return parse_shell_assignments(path.read_text(encoding="utf-8"))


def extract_heredoc(script: str, opening_line: str) -> str:
    lines = script.splitlines()
    try:
        start = lines.index(opening_line) + 1
    except ValueError as exc:
        raise AssertionError(f"missing heredoc opening: {opening_line}") from exc
    try:
        end = lines.index("EOF", start)
    except ValueError as exc:
        raise AssertionError(f"unterminated heredoc: {opening_line}") from exc
    return "\n".join(lines[start:end]) + "\n"


class PublicConfigurationValuesTests(unittest.TestCase):
    def assert_shared_values(self, values: dict[str, str]) -> None:
        self.assertEqual(values["EMBED_HOST"], "127.0.0.1")
        self.assertEqual(values["LLM_HOST"], "127.0.0.1")
        self.assertEqual(values["API_HOST"], "127.0.0.1")
        self.assertEqual(values["EMBEDDING_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(values["LLM_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(values["EMBEDDING_API_URL"], "http://127.0.0.1:8002")
        self.assertEqual(
            values["EMBEDDING_MODEL_LARGE_PATH"],
            "${EMBEDDING_SERVER_HOME}/models/Qwen/Qwen3-Embedding-0.6B",
        )
        self.assertEqual(values["EMBED_BATCH_SIZE"], "16")
        self.assertEqual(values["EMBEDDING_API_BATCH_SIZE"], "16")
        self.assertEqual(
            values["LLM_MODEL_PATH"],
            "${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
        )
        self.assertEqual(values["LLM_MODEL_NAME"], "qwen3.5-9b-q4_k_m")
        self.assertEqual(values["LLM_CTX_SIZE"], "131072")
        self.assertEqual(values["LLM_CONTEXT_LIMIT"], "131072")
        self.assertEqual(
            values["PDF_OCR_MODEL_NAME"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            values["PDF_OCR_VL_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            values["PDF_OCR_LAYOUT_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3",
        )
        self.assertEqual(
            values["PDF_OCR_DOC_ORIENTATION_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori",
        )
        self.assertEqual(
            values["PDF_OCR_DOC_UNWARP_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/UVDoc",
        )
        self.assertEqual(values["PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK"], "0")
        self.assertEqual(values["PDF_OCR_MAX_PAGES"], "400")

    def test_shared_example_uses_canonical_public_values(self) -> None:
        self.assert_shared_values(read_assignments(SHARED_EXAMPLE))

    def test_shell_parser_uses_ascii_identifiers_and_last_value_wins(self) -> None:
        values = parse_shell_assignments(
            "VALID=first\n"
            "VALID=second\n"
            "_ALSO_VALID2=yes\n"
            "NÁME=unicode\n"
            "9INVALID=number\n"
            "BAD-DASH=no\n"
        )
        self.assertEqual(values, {"VALID": "second", "_ALSO_VALID2": "yes"})

    def test_service_examples_match_the_shared_baseline(self) -> None:
        embedding = read_assignments(EMBEDDING_EXAMPLE)
        self.assertEqual(embedding["EMBED_HOST"], "127.0.0.1")
        self.assertEqual(embedding["EMBEDDING_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(
            embedding["EMBEDDING_MODEL_LARGE_PATH"],
            "./models/Qwen/Qwen3-Embedding-0.6B",
        )
        self.assertEqual(embedding["EMBED_BATCH_SIZE"], "16")

        backend = read_assignments(BACKEND_EXAMPLE)
        self.assertEqual(backend["LLM_HOST"], "127.0.0.1")
        self.assertEqual(backend["LLM_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(backend["EMBEDDING_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(backend["EMBEDDING_API_URL"], "http://127.0.0.1:8002")
        self.assertEqual(backend["EMBEDDING_API_BATCH_SIZE"], "16")
        self.assertEqual(
            backend["LLM_MODEL_PATH"],
            "${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
        )
        self.assertEqual(backend["LLM_MODEL_NAME"], "qwen3.5-9b-q4_k_m")
        self.assertEqual(backend["LLM_CTX_SIZE"], "131072")
        self.assertEqual(backend["LLM_CONTEXT_LIMIT"], "131072")
        self.assertEqual(
            backend["PDF_OCR_MODEL_NAME"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend["PDF_OCR_VL_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend["PDF_OCR_LAYOUT_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3",
        )
        self.assertEqual(
            backend["PDF_OCR_DOC_ORIENTATION_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori",
        )
        self.assertEqual(
            backend["PDF_OCR_DOC_UNWARP_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/UVDoc",
        )
        self.assertEqual(backend["PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK"], "0")
        self.assertEqual(backend["PDF_OCR_MAX_PAGES"], "400")

    def test_setup_generator_emits_canonical_values_without_private_defaults(self) -> None:
        script = SETUP_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("compasslm-local-key", script)
        self.assertNotIn("LLM_HOST=0.0.0.0", script)
        private_user_path = "/user/" + "cur0618/"
        self.assertNotIn(private_user_path, script)

        embedding_body = extract_heredoc(
            script,
            '  cat > "${EMBEDDING_SERVER_HOME}/.env.auto" <<EOF',
        )
        embedding_raw = parse_shell_assignments(embedding_body)
        self.assertEqual(
            embedding_raw["EMBEDDING_MODEL_LARGE_PATH"],
            r"\${EMBEDDING_SERVER_HOME}/models/Qwen/Qwen3-Embedding-0.6B",
        )
        embedding = parse_shell_assignments(
            embedding_body.replace(r"\$", "$")
        )
        self.assertEqual(embedding["EMBED_HOST"], "127.0.0.1")
        self.assertEqual(embedding["EMBEDDING_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(
            embedding["EMBEDDING_MODEL_LARGE_PATH"],
            "${EMBEDDING_SERVER_HOME}/models/Qwen/Qwen3-Embedding-0.6B",
        )
        self.assertEqual(embedding["EMBED_BATCH_SIZE"], "16")

        backend_body = extract_heredoc(
            script,
            '  cat > "${MAIN_BACKEND_HOME}/.env.auto" <<EOF',
        )
        backend_raw = parse_shell_assignments(backend_body)
        self.assertEqual(
            backend_raw["LLM_MODELS_DIR"],
            r"\${MAIN_BACKEND_HOME}/models/llm",
        )
        self.assertEqual(
            backend_raw["LLM_MODEL_PATH"],
            r"\${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
        )
        self.assertEqual(
            backend_raw["LLM_RUNTIME"],
            r"\${MAIN_BACKEND_HOME}/runtime/llama-server",
        )
        self.assertEqual(
            backend_raw["PDF_OCR_MODEL_NAME"],
            r"\${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend_raw["PDF_OCR_VL_MODEL_DIR"],
            r"\${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend_raw["PDF_OCR_LAYOUT_MODEL_DIR"],
            r"\${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3",
        )
        self.assertEqual(
            backend_raw["PDF_OCR_DOC_ORIENTATION_MODEL_DIR"],
            r"\${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori",
        )
        self.assertEqual(
            backend_raw["PDF_OCR_DOC_UNWARP_MODEL_DIR"],
            r"\${MAIN_BACKEND_HOME}/models/ocr/UVDoc",
        )
        backend = parse_shell_assignments(
            backend_body.replace(r"\$", "$")
        )
        self.assertEqual(backend["LLM_HOST"], "127.0.0.1")
        self.assertEqual(backend["API_HOST"], "127.0.0.1")
        self.assertEqual(backend["LLM_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(backend["EMBEDDING_API_KEY"], "replace-with-strong-secret")
        self.assertEqual(backend["EMBEDDING_API_URL"], "http://127.0.0.1:8002")
        self.assertEqual(backend["EMBEDDING_API_BATCH_SIZE"], "16")
        self.assertEqual(
            backend["LLM_MODEL_PATH"],
            "${MAIN_BACKEND_HOME}/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
        )
        self.assertEqual(backend["LLM_MODEL_NAME"], "qwen3.5-9b-q4_k_m")
        self.assertEqual(backend["LLM_CTX_SIZE"], "131072")
        self.assertEqual(backend["LLM_CONTEXT_LIMIT"], "131072")
        self.assertEqual(
            backend["PDF_OCR_MODEL_NAME"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend["PDF_OCR_VL_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PaddleOCR-VL",
        )
        self.assertEqual(
            backend["PDF_OCR_LAYOUT_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-DocLayoutV3",
        )
        self.assertEqual(
            backend["PDF_OCR_DOC_ORIENTATION_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/PP-LCNet_x1_0_doc_ori",
        )
        self.assertEqual(
            backend["PDF_OCR_DOC_UNWARP_MODEL_DIR"],
            "${MAIN_BACKEND_HOME}/models/ocr/UVDoc",
        )
        self.assertEqual(backend["PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK"], "0")
        self.assertEqual(backend["PDF_OCR_MAX_PAGES"], "400")

        shared = parse_shell_assignments(
            extract_heredoc(
                script,
                '  cat > "${PROJECT_GPU_HOME}/runtime.env.example" <<\'EOF\'',
            )
        )
        self.assert_shared_values(shared)

    def test_generated_shared_example_matches_committed_example_byte_for_byte(self) -> None:
        script = SETUP_SCRIPT.read_text(encoding="utf-8")
        generated = extract_heredoc(
            script,
            '  cat > "${PROJECT_GPU_HOME}/runtime.env.example" <<\'EOF\'',
        )
        self.assertNotIn(r"\${", generated)
        self.assertEqual(generated, SHARED_EXAMPLE.read_text(encoding="utf-8"))

    def test_online_backend_setup_installs_the_selected_paddle_runtime(self) -> None:
        script = SETUP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("install_backend_online_paddle_runtime()", script)
        self.assertIn('PADDLE_RUNTIME_VERSION:-3.3.0', script)
        self.assertIn('resolve_backend_paddle_runtime_kind', script)
        self.assertIn('resolve_backend_paddle_cuda_track', script)
        self.assertIn(
            'https://paddle-whl.bj.bcebos.com/stable/${paddle_cuda_track}/paddlepaddle-gpu/'
            'paddlepaddle_gpu-${paddle_runtime_version}-cp311-cp311-linux_x86_64.whl',
            script,
        )
        self.assertIn('PADDLE_GPU_WHL_URL:-${default_gpu_whl_url}', script)
        self.assertIn('paddlepaddle==${paddle_runtime_version}', script)
        self.assertIn("online PaddlePaddle GPU wheel requires Python 3.11", script)

        backend_section = script[script.index('if [[ "${SKIP_BACKEND}" != "1" ]]'):]
        online_call = (
            'install_backend_online_paddle_runtime '
            '"${BACKEND_VENV}/bin/python"'
        )
        self.assertEqual(backend_section.count(online_call), 1)
        requirements_install = (
            '"${BACKEND_VENV}/bin/python" -m pip install -r "${BACKEND_REQ}"'
        )
        self.assertLess(
            backend_section.index(requirements_install),
            backend_section.index(online_call),
        )
        self.assertLess(
            backend_section.index(online_call),
            backend_section.index(
                'echo "[INFO] Verifying main-backend OCR runtime imports..."'
            ),
        )
        self.assertEqual(
            backend_section.count(
                'install_backend_offline_paddle_runtime '
                '"${BACKEND_VENV}/bin/python" "${OFFLINE_DIR}"'
            ),
            2,
        )


class PublicDocumentationContractTests(unittest.TestCase):
    def assert_contains_all(self, text: str, required: tuple[str, ...]) -> None:
        normalized = " ".join(text.split())
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(" ".join(fragment.split()), normalized)

    def test_root_readme_documents_the_reproducible_public_workflow(self) -> None:
        path = ROOT / "README.md"
        self.assertTrue(path.is_file(), "root README.md must exist")
        readme = path.read_text(encoding="utf-8")
        self.assert_contains_all(
            readme,
            (
                "Python 3.11+",
                "Linux",
                "NVIDIA GPU",
                "CUDA",
                "Bash",
                "bash project-gpu/setup_gpu_track.sh --python python3.11",
                "project-gpu/embedding-gpu-server/compassvenv",
                "project-gpu/main-backend/compassvenv",
                "python3.11 -m venv .venv",
                "python -m pip install -r requirements.txt",
                "서비스 실행용이 아닙니다",
                "cp project-gpu/runtime.env.example project-gpu/runtime.env",
                "docs/MODELS_AND_ASSETS.md",
                "bash project-gpu/check_gpu_assets.sh",
                "[MISS]",
                "exit code 0",
                "bash project-gpu/run_embedding_server.sh",
                "bash project-gpu/run_llm_server.sh",
                "bash project-gpu/run_backend_api.sh",
                "[READY] embedding",
                "[READY] llm",
                "[READY] backend",
                "http://127.0.0.1:8002/health",
                "http://127.0.0.1:8003/v1/models",
                'read -rsp "LLM API key: " LLM_API_KEY',
                "export LLM_API_KEY",
                'Authorization: Bearer ${LLM_API_KEY}',
                "unset LLM_API_KEY",
                '"data"',
                "http://127.0.0.1:8004/health",
                '{"status":"ok","service":"compasslm-backend"}',
                "python3 -m unittest discover -s tests -p 'test_*.py'",
                "python3 scripts/check_public_release.py --manifest PUBLIC_RELEASE_MANIFEST.txt",
                "PUBLIC_RELEASE_MANIFEST.txt",
                "127.0.0.1",
                "replace-with-strong-secret",
            ),
        )
        for excluded in (
            "모델 가중치",
            "임베딩/OCR 가중치",
            "LLM 런타임",
            "사용자 데이터/업로드",
            "로그",
            "데이터베이스",
            "오프라인 번들/패키지",
            "캐시",
        ):
            with self.subTest(excluded=excluded):
                self.assertIn(excluded, readme)
        self.assertRegex(
            readme,
            r"서비스별 `?\.env`?\s*(?:→|->)\s*서비스별 `?\.env\.auto`?"
            r"\s*(?:→|->)\s*project-gpu/runtime\.env\s*(?:→|->)\s*실행 셸 환경변수",
        )
        self.assertIn("낮은 우선순위", readme)
        self.assertIn("덮어씁니다", readme)
        self.assertIn("직접 수정하지", readme)

        normalized = " ".join(readme.split())
        self.assert_contains_all(
            readme,
            (
                "사용자/업로드 → backend/UI orchestrator",
                "backend → embedding API",
                "색인·질의 벡터",
                "backend → llama.cpp LLM",
                "답변 생성",
                "local data/SQLite/index → retrieval",
                "인용(citations) 포함 응답 → 사용자/UI",
            ),
        )
        self.assertNotIn(
            "embedding API :8002 → llama.cpp LLM :8003 → backend/UI :8004",
            normalized,
        )

    def test_root_readme_documents_the_current_operator_and_repository_workflow(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assert_contains_all(
            readme,
            (
                "TXT, XLSX, HWPX, PDF",
                "계정별 지식공간",
                "비동기 2단계",
                "Wiki answer memory",
                "Ontology-RAG",
                "scripts/create_admin_user.py",
                "project-gpu/compass_up.sh",
                "project-gpu/compass_status.sh",
                "project-gpu/compass_logs.sh backend -f",
                "project-gpu/compass_down.sh",
                "GitHub를 기준 저장소",
                "공공 GitLab에 자동 반영",
                ".github/workflows/mirror-to-aigov-gitlab.yml",
                "GitLab Auto DevOps",
            ),
        )

    def test_model_asset_guide_maps_formats_paths_and_configuration(self) -> None:
        path = ROOT / "docs" / "MODELS_AND_ASSETS.md"
        self.assertTrue(path.is_file(), "public model/asset guide must exist")
        guide = path.read_text(encoding="utf-8")
        self.assert_contains_all(
            guide,
            (
                "project-gpu/main-backend/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
                "project-gpu/main-backend/runtime/llama-server",
                "LLM_MODELS_DIR",
                "LLM_MODEL_PATH",
                "비어 있지 않으면",
                "재귀 탐지",
                "LLM_MODEL_NAME",
                "LLM_RUNTIME",
                "LLM_CTX_SIZE",
                "LLM_CONTEXT_LIMIT",
                "131072",
                "LLM_API_KEY",
                "project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-0.6B/",
                "config.json",
                "tokenizer",
                ".safetensors",
                "EMBEDDING_MODEL_LARGE_PATH",
                "DEFAULT_INDEX",
                "API_LARGE_ALIAS",
                "MODEL_DEVICE",
                "EMBED_BATCH_SIZE",
                "EMBEDDING_API_URL",
                "API_BATCH_SIZE",
                "API_KEY",
                "llama.cpp LLM 입력이 아니다",
                "project-gpu/main-backend/models/ocr/PaddleOCR-VL",
                "project-gpu/main-backend/models/ocr/PP-DocLayoutV3",
                "project-gpu/main-backend/models/ocr/PP-LCNet_x1_0_doc_ori",
                "project-gpu/main-backend/models/ocr/UVDoc",
                "PDF_OCR_MODEL_NAME",
                "PDF_OCR_VL_MODEL_DIR",
                "PDF_OCR_LAYOUT_MODEL_DIR",
                "PDF_OCR_DOC_ORIENTATION_MODEL_DIR",
                "PDF_OCR_DOC_UNWARP_MODEL_DIR",
                "PDF_OCR_BACKEND",
                "PDF_OCR_DEVICE",
                "PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK=0",
                "서비스별 .env → 서비스별 .env.auto → project-gpu/runtime.env → 실행 셸 환경변수",
                "나중에 읽은 파일이 앞선 값을 덮어쓴다",
                "127.0.0.1",
                "project-gpu/install_llama_cuda_runtime.sh",
                "project-gpu/build_llama_runtime_offline.sh",
                "bash project-gpu/check_gpu_assets.sh",
                "[MISS]",
                "exit code가 0",
            ),
        )
        self.assertIn("같은 값", guide)

    def test_project_gpu_guide_uses_public_paths_and_canonical_config(self) -> None:
        guide = (ROOT / "project-gpu" / "README.md").read_text(encoding="utf-8")
        private_user_path = "/user/" + "cur0618/"
        self.assertNotIn(private_user_path, guide)
        self.assertIn("/user/<user>/", guide)
        self.assertIn("runtime.env.example", guide)
        self.assertIn("공통 기준", guide)
        self.assertIn("../docs/MODELS_AND_ASSETS.md", guide)
        remote_server_example = guide[
            guide.index("임베딩 서버의 최종"):
            guide.index("백엔드의 최종")
        ]
        self.assertIn("COMPASS_AUTO_PORT=0", remote_server_example)
        self.assert_contains_all(
            guide,
            (
                "서비스별 .env < 서비스별 .env.auto < runtime.env < 실행 셸 환경변수",
                "임베딩 서버의 최종 `project-gpu/runtime.env`",
                "EMBED_HOST=0.0.0.0",
                "EMBED_PORT=8002",
                "백엔드의 최종 `project-gpu/runtime.env`",
                "COMPASS_AUTO_PORT=0",
                "EMBEDDING_API_URL=https://",
                "`runtime.env`가 서비스 `.env`와 `.env.auto`보다 우선",
                "firewall allowlist",
                "plaintext Bearer",
                "public Internet",
                "TLS reverse proxy",
                "VPN",
                "private tunnel",
                "127.0.0.1",
            ),
        )

    def test_repository_boundary_guide_uses_the_public_manifest_and_guard(self) -> None:
        guide = (ROOT / "docs" / "repository-boundaries.md").read_text(encoding="utf-8")

        self.assertIn("PUBLIC_RELEASE_MANIFEST.txt", guide)
        self.assertIn(
            "python3 scripts/check_public_release.py --manifest PUBLIC_RELEASE_MANIFEST.txt",
            guide,
        )
        self.assertNotIn("transfer_manifests/", guide)
        self.assertNotIn("scripts/run_lumin_audit.sh", guide)

    def test_license_is_the_standard_mit_license(self) -> None:
        expected = """MIT License

Copyright (c) 2026 cur0618

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        path = ROOT / "LICENSE"
        self.assertTrue(path.is_file(), "LICENSE must exist")
        self.assertEqual(path.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
