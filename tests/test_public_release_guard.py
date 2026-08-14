from __future__ import annotations

import contextlib
import io
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_public_release import main, read_manifest, validate_manifest


class PublicReleaseGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write(self, relative: str, content: str = "safe\n") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def manifest(self, entries: list[str]) -> Path:
        path = self.root / "PUBLIC_RELEASE_MANIFEST.txt"
        path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        return path

    def issues_for(self, entries: list[str]) -> list[str]:
        return validate_manifest(self.root, self.manifest(entries))

    def test_valid_manifest_with_self_entry_passes(self) -> None:
        self.write("README.md")
        manifest = self.manifest(["PUBLIC_RELEASE_MANIFEST.txt", "README.md"])

        self.assertEqual(read_manifest(manifest), ["PUBLIC_RELEASE_MANIFEST.txt", "README.md"])
        self.assertEqual(validate_manifest(self.root, manifest), [])

    def test_manifest_requires_self_entry_and_strict_ordering(self) -> None:
        self.write("a.txt")
        self.write("b.txt")
        cases = {
            "missing self": (["a.txt"], "must include PUBLIC_RELEASE_MANIFEST.txt"),
            "unsorted": (
                ["PUBLIC_RELEASE_MANIFEST.txt", "b.txt", "a.txt"],
                "sorted",
            ),
            "duplicate": (
                ["PUBLIC_RELEASE_MANIFEST.txt", "a.txt", "a.txt"],
                "duplicate",
            ),
            "blank": (["PUBLIC_RELEASE_MANIFEST.txt", "", "a.txt"], "blank"),
        }
        for label, (entries, expected) in cases.items():
            with self.subTest(label=label):
                issues = self.issues_for(entries)
                self.assertTrue(any(expected in issue for issue in issues), issues)

    def test_manifest_rejects_absolute_traversal_and_unnormalized_paths(self) -> None:
        for entry in [
            "/etc/hosts",
            "../outside.txt",
            "docs/../README.md",
            "docs\\README.md",
            " README.md",
            "README.md ",
            "docs/*.md",
            "docs/file?.md",
            "docs/[ab].md",
        ]:
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("normalized POSIX-relative" in issue for issue in issues), issues)

    def test_manifest_rejects_missing_and_directory_entries(self) -> None:
        self.write("folder/item.txt")
        cases = {
            "missing.txt": "missing",
            "folder": "regular file",
        }
        for entry, expected in cases.items():
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any(expected in issue for issue in issues), issues)

    def test_manifest_rejects_a_symlink_entry(self) -> None:
        target = self.write("target.txt")
        link = self.root / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not supported on this platform")

        issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", "link.txt"])

        self.assertTrue(any("symlink" in issue for issue in issues), issues)

    def test_manifest_rejects_a_symlinked_parent_directory(self) -> None:
        self.write("real/item.txt")
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.root / "real", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not supported on this platform")

        issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", "alias/item.txt"])

        self.assertTrue(any("symlinked parent" in issue for issue in issues), issues)

    def test_live_env_is_rejected_but_env_examples_are_allowed(self) -> None:
        self.write("service/.env", "SAFE=true\n")
        self.write("service/.env.example", "SERVICE_API_KEY=replace-with-strong-secret\n")
        self.write("service/local.env.example", "SERVICE_API_KEY=${SERVICE_API_KEY}\n")

        live_issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", "service/.env"])
        self.assertTrue(any("live environment" in issue for issue in live_issues), live_issues)
        example_entries = [
            "PUBLIC_RELEASE_MANIFEST.txt",
            "service/.env.example",
            "service/local.env.example",
        ]
        self.assertEqual(self.issues_for(example_entries), [])

        for entry in ["service/prod.env.local", "service/app.env.prod", "service/tool.env.example.backup"]:
            self.write(entry, "SAFE=true\n")
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("live environment" in issue for issue in issues), issues)

    def test_forbidden_path_components_are_rejected(self) -> None:
        components = [
            ".agents",
            ".audit",
            ".cache",
            ".codex",
            ".git",
            ".idea",
            ".lumin",
            ".mypy_cache",
            ".pytest_cache",
            ".roo",
            ".ruff_cache",
            ".superpowers",
            ".tools",
            ".venv",
            ".vscode",
            ".worktrees",
            "__pycache__",
            "archive",
            "build",
            "cache",
            "compassvenv",
            "data",
            "dist",
            "evaluation_results",
            "fin_result",
            "logs",
            "models",
            "node_modules",
            "offline_assets",
            "offline_bundle",
            "offline_packages",
            "packages",
            "results",
            "runtime",
            "tools",
            "uploads",
            "venv",
            "wheels",
        ]
        for component in components:
            entry = f"service/{component}/file.txt"
            self.write(entry)
            with self.subTest(component=component):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("forbidden component" in issue for issue in issues), issues)

    def test_weight_tokenizer_and_native_runtime_files_are_rejected(self) -> None:
        entries = [
            "release-files/model.gguf",
            "release-files/model.safetensors",
            "release-files/model.bin",
            "release-files/model.pt",
            "release-files/model.pth",
            "release-files/model.ckpt",
            "release-files/model.onnx",
            "release-files/model.index.json",
            "release-files/tokenizer.json",
            "release-files/tokenizer.model",
            "release-files/spiece.model",
            "release-files/merges.txt",
            "release-files/vocab.json",
            "release-files/native.so",
            "release-files/native.so.1",
            "release-files/native.dll",
            "release-files/native.dylib",
            "release-files/native.exe",
            "release-files/private.pem",
            "release-files/private.key",
            "release-files/private.p12",
            "release-files/private.pfx",
            "release-files/llama-server",
            "release-files/rpc-server",
        ]
        for entry in entries:
            self.write(entry)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("forbidden filename or format" in issue for issue in issues), issues)

    def test_archives_state_databases_logs_and_fonts_are_rejected(self) -> None:
        entries = [
            "release-files/archive.tar",
            "release-files/archive.tar.gz",
            "release-files/archive.tgz",
            "release-files/archive.zip",
            "release-files/package.whl",
            "release-files/evaluation.jsonl",
            "release-files/state.sqlite",
            "release-files/state.sqlite3",
            "release-files/state.db",
            "release-files/runtime.log",
            "release-files/cache.pyc",
            "release-files/font.woff",
            "release-files/font.woff2",
            "release-files/font.ttf",
            "release-files/font.otf",
        ]
        for entry in entries:
            self.write(entry)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("forbidden filename or format" in issue for issue in issues), issues)

    def test_operational_names_and_editor_temps_are_rejected(self) -> None:
        entries = [
            "notes/.DS_Store",
            "notes/Desktop.ini",
            "notes/WORKPLAN.md",
            "notes/usage.md",
            "notes/workflow.md",
            "notes/workflow.xml",
            "notes/transfer_private.txt",
            "notes/file.txt~",
            "notes/.#draft.txt",
            "notes/.draft.swp",
            "notes/.draft.swo",
            "notes/draft.tmp",
        ]
        for entry in entries:
            self.write(entry)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("forbidden" in issue for issue in issues), issues)

    def test_supported_public_text_file_types_pass(self) -> None:
        entries = [
            ".gitignore",
            "Dockerfile",
            "LICENSE",
            "README.md",
            "config/app.cfg",
            "config/app.ini",
            "config/app.toml",
            "config/app.yaml",
            "config/app.yml",
            "config/runtime.env.example",
            "requirements.txt",
            "src/app.js",
            "src/app.py",
            "src/index.html",
            "src/style.css",
            "scripts/start.sh",
            "values.json",
        ]
        for entry in entries:
            self.write(entry, "{}\n" if entry.endswith(".json") else "example\n")

        issues = self.issues_for(sorted(["PUBLIC_RELEASE_MANIFEST.txt", *entries]))

        self.assertEqual(issues, [])

    def test_unknown_extensions_and_binary_like_payloads_are_rejected(self) -> None:
        payloads = {
            "release-files/blob.dat": (b"safe ascii\n", ["unsupported public text type"]),
            "release-files/nul.dat": (
                b"safe\x00ascii\n",
                ["unsupported public text type", "control byte"],
            ),
            "release-files/document.pdf": (
                b"%PDF-1.7\nplain ascii body\n",
                ["unsupported public text type", "binary signature"],
            ),
            "release-files/disguised.txt": (
                b"%PDF-1.7\nplain ascii body\n",
                ["binary signature"],
            ),
            "release-files/control.txt": (b"safe\x01text\n", ["control byte"]),
        }
        for entry, (payload, expected_markers) in payloads.items():
            path = self.root / entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                for marker in expected_markers:
                    self.assertTrue(any(marker in issue for issue in issues), issues)

    def test_gitignore_excludes_private_key_artifacts(self) -> None:
        gitignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")

        for pattern in ["*.pem", "*.key", "*.p12", "*.pfx"]:
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, gitignore.splitlines())

    def test_public_shell_scripts_are_executable(self) -> None:
        repository = Path(__file__).parents[1]
        manifest = read_manifest(repository / "PUBLIC_RELEASE_MANIFEST.txt")
        shell_scripts = [entry for entry in manifest if entry.endswith(".sh")]

        self.assertTrue(shell_scripts)
        indexed_modes: dict[str, str] = {}
        has_git_metadata = (repository / ".git").exists()
        if has_git_metadata:
            result = subprocess.run(
                ["git", "ls-files", "--stage", "--", *shell_scripts],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            indexed_modes = {
                line.split(maxsplit=1)[1].split("\t", 1)[1]: line.split(maxsplit=1)[0]
                for line in result.stdout.splitlines()
            }
            self.assertEqual(set(indexed_modes), set(shell_scripts))
        for entry in shell_scripts:
            with self.subTest(entry=entry):
                mode = (repository / entry).stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR, f"{entry} must be executable")
                if has_git_metadata:
                    self.assertEqual(indexed_modes.get(entry), "100755")

    def test_files_over_the_size_limit_are_rejected(self) -> None:
        large = self.root / "large.txt"
        large.write_bytes(b"x" * 33)
        issues = validate_manifest(
            self.root,
            self.manifest(["PUBLIC_RELEASE_MANIFEST.txt", "large.txt"]),
            max_bytes=32,
        )

        self.assertTrue(any("size limit" in issue for issue in issues), issues)

    def test_local_paths_and_private_network_hosts_are_rejected(self) -> None:
        unsafe_contents = [
            "path=" + "/home/" + "alice/project\n",
            "path=" + "/mnt/c/Users/" + "alice/project\n",
            "path=" + "C:" + "\\Users\\" + "alice\\project\n",
            "owner=" + "ae" + "lag\n",
            "endpoint=http://" + "10." + "0.0.8:8000\n",
            "endpoint=http://" + "172." + "16.0.8:8000\n",
            "endpoint=http://" + "192." + "168.1.8:8000\n",
            "endpoint=http://service." + "internal/api\n",
            "endpoint=http://service." + "local" + "domain/api\n",
        ]
        for index, content in enumerate(unsafe_contents):
            entry = f"docs/unsafe-{index}.txt"
            self.write(entry, content)
            with self.subTest(content=content):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("private content" in issue for issue in issues), issues)

    def test_package_versions_that_resemble_private_ips_are_allowed(self) -> None:
        cuda_10_3_5_147 = "10.3.5." + "147"
        cuda_10_9_0_58 = "10.9.0." + "58"
        cuda_10_3_0_86 = "10.3.0." + "86"
        cuda_10_3_7_77 = "10.3.7." + "77"
        setup_lines = [
            f'require_offline_artifact "${{dir}}" "nvidia_curand_cu12-{cuda_10_3_5_147}-*.whl" "runtime"',
            f'require_offline_artifact "${{dir}}" "nvidia_cufft_cu11-{cuda_10_9_0_58}-*.whl" "runtime"',
            f'require_offline_artifact "${{dir}}" "nvidia_curand_cu11-{cuda_10_3_0_86}-*.whl" "runtime"',
            f'require_offline_artifact "${{dir}}" "nvidia_curand_cu12-{cuda_10_3_7_77}-*.whl" "runtime"',
            f"nvidia-curand-cu12=={cuda_10_3_7_77}",
            f"'nvidia-curand-cu12[cuda,server]~=1!{cuda_10_3_7_77}.post1'",
            f'"nvidia-curand-cu12>={cuda_10_3_7_77}"',
            f"nvidia-curand-cu12<={cuda_10_3_7_77}",
            f"nvidia-curand-cu12!={cuda_10_3_7_77}",
            f"nvidia-curand-cu12>{cuda_10_3_7_77}",
            f"nvidia-curand-cu12<{cuda_10_3_7_77}",
            f"CUDA_RUNTIME_VERSION={cuda_10_3_7_77}",
            f'EMBEDDING_VERSION="{cuda_10_3_5_147}.post1"',
            f"nvidia_curand_cu12-{cuda_10_3_7_77}-py3-none-manylinux_x86_64.whl",
            f'ok_or_missing_glob "${{OFFLINE_DIR}}/nvidia_curand_cu12-{cuda_10_3_7_77}-*.whl" "runtime"',
        ]
        self.write("scripts/setup.sh", "\n".join(setup_lines) + "\n")

        issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", "scripts/setup.sh"])

        self.assertEqual(issues, [])

    def test_private_ips_in_network_assignments_urls_and_prose_are_rejected(self) -> None:
        unsafe_contents = [
            "EMBED_" + "HOST=10." + "1.2.3\n",
            "URL=http://" + "192." + "168.1.2\n",
            "SERVICE_" + "IP=172." + "16.0.1\n",
            "BIND_" + "ADDR=10." + "4.5.6\n",
            "LISTEN_" + "ADDRESS=192." + "168.2.3\n",
            "connect to " + "10." + "2.3.4 for inference\n",
            "endpoint http://" + "172." + "31.4.5/api\n",
        ]
        for index, content in enumerate(unsafe_contents):
            entry = f"docs/network-{index}.txt"
            self.write(entry, content)
            with self.subTest(content=content):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("internal network host" in issue for issue in issues), issues)

    def test_query_and_prose_punctuation_do_not_hide_private_ips(self) -> None:
        unsafe_contents = [
            "https://public.example/redirect?next=" + "10." + "1.2.3\n",
            "Alert!" + "10." + "1.2.3 is private\n",
            "if [ $HOST == " + "10." + "1.2.3 ]; then\n",
            "https://public.example/?next==" + "10." + "1.2.3\n",
            "assert ENDPOINT != " + "192." + "168.1.8\n",
            "SERVER>=" + "172." + "16.4.5\n",
        ]
        for index, content in enumerate(unsafe_contents):
            entry = f"docs/punctuation-{index}.txt"
            self.write(entry, content)
            with self.subTest(content=content):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("internal network host" in issue for issue in issues), issues)

    def test_private_key_headers_are_rejected(self) -> None:
        headers = [
            "-----BEGIN " + "PRIVATE KEY-----\n",
            "-----BEGIN " + "OPEN" + "SSH PRIVATE KEY-----\n",
            "-----BEGIN " + "RSA PRIVATE KEY-----\n",
        ]
        for index, header in enumerate(headers):
            entry = f"docs/key-{index}.txt"
            self.write(entry, header)
            with self.subTest(header=header):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("private-key" in issue for issue in issues), issues)

    def test_real_credentials_are_rejected_and_placeholders_are_allowed(self) -> None:
        unsafe_assignments = [
            "SERVICE_API_" + "KEY=live-value-123\n",
            "API_" + "KEY=live-value-123\n",
            "TOKEN" + "=live-value-123\n",
            "PASSWORD" + "=live-value-123\n",
            "SECRET" + "=live-value-123\n",
            "service_api_" + "key=live-value-123\n",
            "token" + "=live-value-123\n",
            "password" + "=live-value-123\n",
            "secret" + "=live-value-123\n",
            "API_" + "KEY=${API_" + "KEY:-live-production-" + "secret}\n",
            "TOKEN" + "=production-" + "example-token-7f9a\n",
            'password="admin-password"\n',
            'password="admin-password",\n',
            'token="secret"\n',
            "TOKEN" + '=os.getenv("TOKEN", "admin-password")\n',
            "PASSWORD" + '=str("admin-password")\n',
            "SECRET" + '=required["secret"]\n',
            "TOKEN" + '=eval("secret")\n',
            "TOKEN" + "=eval(candidate)\n",
            "TOKEN" + '=eval("secret").strip()\n',
            "TOKEN" + "=unknown_factory()\n",
            "TOKEN" + '=secrets.token_urlsafe(b"secret")\n',
            "API_TOKEN" + "=-lsecret\n",
        ]
        for index, unsafe in enumerate(unsafe_assignments):
            entry = f"config/unsafe-{index}.txt"
            self.write(entry, unsafe)
            with self.subTest(unsafe=unsafe):
                unsafe_issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in unsafe_issues), unsafe_issues)

        allowed_values = [
            "",
            "${SERVICE_API_KEY}",
            "replace-with-strong-secret",
            "change-me",
            "example",
            "placeholder",
            "your-token",
        ]
        for index, value in enumerate(allowed_values):
            entry = f"config/safe-{index}.txt"
            variable = "service_api_" + "key" if index % 2 else "API_" + "KEY"
            self.write(entry, f"{variable}={value}\n")
            with self.subTest(value=value):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertEqual(issues, [])

    def test_computed_credentials_and_linker_tokens_are_allowed(self) -> None:
        safe_python_assignments = [
            'EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()',
            'ONTOLOGY_LLM_API_KEY = os.getenv("ONTOLOGY_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))',
            'admin_password=required["COMPASSLM_ADMIN_PASSWORD"]',
            "token = secrets.token_urlsafe(32)",
            'password = str(data.get("password", "") or "")',
            'token = request.cookies.get(AUTH_SESSION_COOKIE_NAME, "")',
            'password = "replace-with-strong-secret"',
            "token = auth_store.create_session(\n"
            '    user["user_id"],\n'
            "    ttl_seconds=60,\n"
            ")",
            "service = build_service(\n"
            "    api_key=self.settings.api_key,\n"
            '    password="replace-with-strong-secret",\n'
            ")",
        ]
        safe_shell_assignments = [
            'export DEBUG_AUTH_TOKEN="${DEBUG_AUTH_TOKEN:-}"',
            'STDCXXFS_LINK_TOKEN="-lstdc++fs"',
            "python3 - <<'PY'\n"
            'DEBUG_AUTH_TOKEN = os.environ.get("COMPASSLM_DEBUG_AUTH_TOKEN", "").strip()\n'
            "PY",
        ]
        self.write("src/computed.py", "\n".join(safe_python_assignments) + "\n")
        self.write("scripts/computed.sh", "\n".join(safe_shell_assignments) + "\n")

        issues = self.issues_for(
            ["PUBLIC_RELEASE_MANIFEST.txt", "scripts/computed.sh", "src/computed.py"]
        )

        self.assertEqual(issues, [])

    def test_multiline_python_credentials_inspect_complete_expressions(self) -> None:
        unsafe_sources = [
            'TOKEN = resolve_api_key(\n    "admin-password",\n)\n',
            'TOKEN = secrets.token_urlsafe(\n    b"secret",\n)\n',
            "TOKEN = module.build_user_facing_citation_token(\n"
            "    7,\n"
            '    {"source_path": "admin-password"},\n'
            ")\n",
        ]
        for index, source in enumerate(unsafe_sources):
            entry = f"src/multiline-{index}.py"
            self.write(entry, source)
            with self.subTest(source=source):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_non_python_credentials_reject_python_expression_disguises(self) -> None:
        unsafe_values = [
            "secret",
            "secret,",
            "resolve_api_key(candidate)",
        ]
        for index, value in enumerate(unsafe_values):
            entry = f"config/runtime-{index}.env.example"
            self.write(entry, f"TOKEN={value}\n")
            with self.subTest(value=value):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_shell_declaration_prefixes_and_python_annotations_are_inspected(self) -> None:
        unsafe_sources = {
            "scripts/local.sh": 'local API_TOKEN="admin-password"\n',
            "scripts/readonly.sh": 'readonly API_TOKEN="secret"\n',
            "scripts/declare.sh": 'declare -r API_TOKEN="secret"\n',
            "src/annotated.py": 'api_key: str = "admin-password"\n',
        }
        for entry, source in unsafe_sources.items():
            self.write(entry, source)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_incremental_credential_assignments_are_rejected(self) -> None:
        unsafe_sources = {
            "src/augmented.py": 'API_TOKEN = ""\nAPI_TOKEN += "admin-password"\n',
            "scripts/augmented.sh": 'API_TOKEN=""\nAPI_TOKEN+="secret"\n',
        }
        for entry, source in unsafe_sources.items():
            self.write(entry, source)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_python_mapping_and_mutation_credentials_are_rejected(self) -> None:
        unsafe_sources = [
            'config["API_KEY"] = "live-secret"\n',
            'config[f"API_{\'KEY\'}"] = "live-secret"\n',
            'config = {"apiKey": "live-secret"}\n',
            'config = {**{"api-key": "live-secret"}}\n',
            'config = dict(apiKey="live-secret")\n',
            'config = dict([("api-key", "live-secret")])\n',
            'config = dict(zip(["APIKEY"], ["live-secret"]))\n',
            'config.setdefault("AUTH_TOKEN_KEY", "live-secret")\n',
            'config.update({"TOKEN_SET": "live-secret"})\n',
            'os.environ.setdefault("CLIENT_SECRET_KEY", "live-secret")\n',
            'os.environ.update({"AWS_SECRET_ACCESS_KEY": "live-secret"})\n',
            'os.putenv("accessToken", "live-secret")\n',
            'setattr(config, "adminPassword", "live-secret")\n',
        ]
        for index, source in enumerate(unsafe_sources):
            entry = f"src/container-{index}.py"
            self.write(entry, source)
            with self.subTest(source=source):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_credential_name_variants_are_rejected(self) -> None:
        names = [
            "API_KEY",
            "APIKEY",
            "apiKey",
            "api-key",
            "TOKEN",
            "accessToken",
            "AUTH_TOKEN_KEY",
            "TOKEN_KEY",
            "TOKEN_SET",
            "PASSWORD",
            "adminPassword",
            "SECRET",
            "clientSecret",
            "AWS_SECRET_ACCESS_KEY",
            "CLIENT_SECRET_KEY",
        ]
        for index, name in enumerate(names):
            suffix = ".py" if "-" not in name else ".env.example"
            entry = f"config/name-{index}{suffix}"
            if suffix == ".py":
                self.write(entry, f'{name} = "live-secret"\n')
            else:
                self.write(entry, f'{name}="live-secret"\n')
            with self.subTest(name=name):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_python_credential_defaults_and_complete_expressions_are_rejected(self) -> None:
        unsafe_sources = [
            'TOKEN = os.getenv("TOKEN", "live-secret")\n',
            'TOKEN = str("live-secret")\n',
            'TOKEN = eval("live-secret")\n',
            'TOKEN = required["secret"]\n',
            'def connect(API_KEY="live-secret"):\n    pass\n',
            'def connect(*, accessToken: str = "live-secret"):\n    pass\n',
            'API_KEY: str = "live-secret"\n',
            'API_KEY = ""\nAPI_KEY += "live-secret"\n',
            'service(\n    clientSecret="live-secret",\n)\n',
        ]
        for index, source in enumerate(unsafe_sources):
            entry = f"src/expression-{index}.py"
            self.write(entry, source)
            with self.subTest(source=source):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_structured_config_credentials_and_json_duplicates_are_rejected(self) -> None:
        unsafe_sources = {
            "config/duplicate.json": '{"apiKey":"replace-with-strong-secret","apiKey":"live-secret"}\n',
            "config/camel.json": '{"clientSecret":"live-secret"}\n',
            "config/kebab.json": '{"api-key":"live-secret"}\n',
            "config/quoted.yaml": '"apiKey": "live-secret"\n',
            "config/block.yaml": "clientSecret: |\n  live-secret\n",
            "config/flow.yml": '{api-key: "live-secret"}\n',
            "config/quoted.ini": "'adminPassword' = 'live-secret'\n",
            "config/colon.ini": "adminPassword: live-secret\n",
            "config/trailing-comma.json": '{"apiKey":"live-secret",}\n',
        }
        for entry, source in unsafe_sources.items():
            self.write(entry, source)
            with self.subTest(entry=entry):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(
                    any(
                        marker in issue
                        for marker in ("credential", "duplicate JSON key", "invalid JSON")
                        for issue in issues
                    ),
                    issues,
                )

    def test_shell_credential_writes_and_parameter_mutations_are_rejected(self) -> None:
        unsafe_sources = [
            'local -r -x -- API_TOKEN="live-secret"\n',
            'readonly -a -- AUTH_TOKEN_KEY="live-secret"\n',
            'declare -g -r -- CLIENT_SECRET_KEY="live-secret"\n',
            'typeset -x -r APIKEY="live-secret"\n',
            'echo ready;API_KEY="live-secret"\n',
            'env -i -- SAFE=1 accessToken="live-secret" command true\n',
            'API_TOKEN=""\nAPI_TOKEN+="live-secret"\n',
            ': "${API_TOKEN:=live-secret}"\n',
            ': "${API_TOKEN=live-secret}"\n',
            'printf -v API_TOKEN %s live-secret\n',
            'read API_TOKEN <<< "live-secret"\n',
        ]
        for index, source in enumerate(unsafe_sources):
            entry = f"scripts/shell-{index}.sh"
            self.write(entry, source)
            with self.subTest(source=source):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_shell_heredocs_route_python_and_data_bodies_safely(self) -> None:
        python_commands = [
            "python3 -",
            "python3 -I -",
            "env SAFE=1 python3 -",
            "$PYTHON_BIN -",
            "${VENV_PY} -",
            "${python_bin} -",
            "${JUPYTER_PYTHON} -",
            "${python_cmd} -",
            "${VENV_ROOT}/bin/python -",
            "uv run python3 -",
            'printf "%s" seed | python3 -I -',
        ]
        starts = [
            "<<PY",
            "<<'PY'",
            '<<"PY"',
            "<<\\PY",
            "<<$'PY'",
            '<<$"PY"',
            "<<-PY",
        ]
        for index, (command, start) in enumerate(zip(python_commands, starts * 2)):
            entry = f"scripts/python-heredoc-{index}.sh"
            self.write(entry, f'{command} {start}\nTOKEN = "live-secret"\nPY\n')
            with self.subTest(command=command, start=start):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

        data_commands = ["${cmd}", "${NOT_AN_INTERPRETER_PY}", "ssh host", "read payload", "cat", "tee output"]
        for index, command in enumerate(data_commands):
            entry = f"scripts/data-heredoc-{index}.sh"
            self.write(entry, f'{command} <<\'DATA\'\nAPI_KEY="live-secret"\nDATA\n')
            with self.subTest(command=command):
                issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", entry])
                self.assertTrue(any("credential" in issue for issue in issues), issues)

    def test_safe_credential_expressions_remain_allowed(self) -> None:
        source = "\n".join(
            [
                'API_KEY = "replace-with-strong-secret"',
                'TOKEN = os.getenv("TOKEN", "")',
                'PASSWORD = required["COMPASSLM_ADMIN_PASSWORD"]',
                "SECRET = secrets.token_urlsafe(32)",
                "token_hash = _hash_session_token(token)",
                "token = auth_store.create_session(user_id, ttl_seconds=60)",
                'config = {"clientSecret": "${CLIENT_SECRET}"}',
            ]
        )
        self.write("src/safe-credentials.py", source + "\n")
        self.write(
            "scripts/safe-credentials.sh",
            "\n".join(
                [
                    'API_KEY="${API_KEY}"',
                    'TOKEN="${TOKEN:-replace-with-strong-secret}"',
                    'PASSWORD="***"',
                    'STDCXXFS_LINK_TOKEN="-lstdc++fs"',
                ]
            )
            + "\n",
        )

        issues = self.issues_for(
            [
                "PUBLIC_RELEASE_MANIFEST.txt",
                "scripts/safe-credentials.sh",
                "src/safe-credentials.py",
            ]
        )

        self.assertEqual(issues, [])

    def test_invalid_utf8_is_reported(self) -> None:
        bad = self.root / "bad.txt"
        bad.write_bytes(b"\xff\xfe")

        issues = self.issues_for(["PUBLIC_RELEASE_MANIFEST.txt", "bad.txt"])

        self.assertTrue(any("UTF-8" in issue for issue in issues), issues)

    def test_invalid_utf8_manifest_is_reported(self) -> None:
        manifest = self.root / "PUBLIC_RELEASE_MANIFEST.txt"
        manifest.write_bytes(b"\xff\xfe")

        issues = validate_manifest(self.root, manifest)

        self.assertTrue(any("manifest must be UTF-8" in issue for issue in issues), issues)

    def test_cli_prints_success_and_error_contracts(self) -> None:
        self.write("README.md")
        manifest = self.manifest(["PUBLIC_RELEASE_MANIFEST.txt", "README.md"])
        success_output = io.StringIO()
        with contextlib.redirect_stdout(success_output):
            success_code = main(["--root", os.fspath(self.root), "--manifest", os.fspath(manifest)])
        self.assertEqual(success_code, 0)
        self.assertIn("[PUBLIC_RELEASE][OK]", success_output.getvalue())
        self.assertIn("2 files", success_output.getvalue())

        self.write("secret.env", "SERVICE_API_" + "KEY=live-value\n")
        self.manifest(["PUBLIC_RELEASE_MANIFEST.txt", "secret.env"])
        error_output = io.StringIO()
        with contextlib.redirect_stdout(error_output):
            error_code = main(["--root", os.fspath(self.root), "--manifest", os.fspath(manifest)])
        self.assertEqual(error_code, 1)
        self.assertIn("[PUBLIC_RELEASE][ERROR]", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
