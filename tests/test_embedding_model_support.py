import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_PATH = ROOT / "src" / "rag.py"
EMBED_SERVER_PATH = ROOT / "project-gpu" / "embedding-gpu-server" / "src" / "main.py"
BASH_EXE = (
    "C:\\Program Files\\Git\\bin\\bash.exe"
    if os.name == "nt"
    else os.environ.get("BASH_EXE", "bash")
)


def _to_bash_path(value: Path | str) -> str:
    normalized = str(value).replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        return f"/{normalized[0].lower()}{normalized[2:]}"
    return normalized


def _run_bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = {**os.environ, "MSYS2_ARG_CONV_EXCL": "*"}
    if env:
        merged_env.update(env)
    return subprocess.run(
        [BASH_EXE, "-lc", script],
        check=True,
        capture_output=True,
        text=True,
        env=merged_env,
    )


class FakeArray:
    def __init__(self, data):
        self._data = data
        self._data = data
        if isinstance(data, list) and data and isinstance(data[0], list):
            self.ndim = 2
            self.shape = (len(data), len(data[0]))
        elif isinstance(data, list):
            self.ndim = 1
            self.shape = (len(data),)
        else:
            self.ndim = 0
            self.shape = ()

    def reshape(self, *_shape):
        return self

    def __getitem__(self, index):
        return self._data[index]

    def tolist(self):
        return self._data

    def astype(self, *_args, **_kwargs):
        return self


class FakeModel:
    def __init__(self):
        self.prompts = {"query": "query", "document": "document"}
        self.calls = []
        class Tokenizer:
            def __call__(self_inner, texts, **kwargs):
                return {"input_ids": [[idx + 1 for idx, _ in enumerate(str(text).split() or [str(text)])] for text in texts]}

            def decode(self_inner, ids, skip_special_tokens=True):
                return " ".join(f"tok{token}" for token in ids)

        self.tokenizer = Tokenizer()

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [[0.1, 0.2] for _ in texts]


def _stub_numpy_module():
    module = types.ModuleType("numpy")
    module.float32 = "float32"
    module.ndarray = object
    module.asarray = lambda data, dtype=None: FakeArray(data)
    module.empty = lambda shape, dtype=None: []
    module.concatenate = lambda arrays, axis=0: sum(arrays, [])
    module.vstack = lambda arrays: sum(arrays, [])
    return module


def _stub_sentence_transformers_module():
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.prompts = {"query": "query", "document": "document"}
            self.tokenizer = FakeModel().tokenizer

        def encode(self, texts, **kwargs):
            return [[0.1, 0.2] for _ in texts]

    module.SentenceTransformer = SentenceTransformer
    return module


def _stub_fastapi_module():
    module = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

        def on_event(self, *args, **kwargs):
            return lambda func: func

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    module.FastAPI = FastAPI
    module.Header = lambda default=None, **kwargs: default
    module.HTTPException = HTTPException
    module.status = types.SimpleNamespace(
        HTTP_401_UNAUTHORIZED=401,
        HTTP_503_SERVICE_UNAVAILABLE=503,
    )
    return module


def _stub_pydantic_module():
    module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.BaseModel = BaseModel
    return module


def _load_module(module_name: str, file_path: Path, extra_modules: dict[str, types.ModuleType]):
    old_modules: dict[str, object] = {}
    try:
        for name, module in extra_modules.items():
            old_modules[name] = sys.modules.get(name)
            sys.modules[name] = module
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _write_fake_model_dir(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")
    return model_dir


class EmbeddingModelSupportTests(unittest.TestCase):
    def test_rag_engine_uses_jina_v5_retrieval_args(self):
        rag_module = _load_module(
            "codex_test_rag",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *a, **k: {
                        "parser": "paddleocr_vl",
                        "pages": [],
                        "total_pages": 0,
                        "text_pages": 0,
                        "ocr_pages": 0,
                        "failed_pages": 0,
                        "warnings": [],
                    },
                    release_cached_ocr_model=lambda *a, **k: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *a, **k: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *a, **k: [],
                    chunk_xlsx_rows=lambda *a, **k: [],
                    load_txt=lambda *a, **k: [],
                    load_xlsx=lambda *a, **k: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine.embed_batch_size = 8
        engine.embedding_task_prefix_mode = engine._resolve_task_prefix_mode(
            "jinaai/jina-embeddings-v5-text-small",
            "auto",
        )

        model = FakeModel()
        rag_module.RAGEngine._encode_texts_local(engine, model, ["질문"], task="query")
        rag_module.RAGEngine._encode_texts_local(engine, model, ["문서"], task="passage")

        self.assertEqual(engine.embedding_task_prefix_mode, "jina_v5")
        self.assertEqual(model.calls[0]["task"], "retrieval")
        self.assertEqual(model.calls[0]["prompt_name"], "query")
        self.assertEqual(model.calls[1]["task"], "retrieval")
        self.assertEqual(model.calls[1]["prompt_name"], "document")

    def test_embedding_server_uses_jina_v5_retrieval_args(self):
        embed_module = _load_module(
            "codex_test_embed_server",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        model = FakeModel()
        mode = embed_module._resolve_task_prefix_mode(
            "jinaai/jina-embeddings-v5-text-small",
            "auto",
        )
        embed_module._encode_with_mode(model, ["질문"], task="query", mode=mode)
        embed_module._encode_with_mode(model, ["문서"], task="passage", mode=mode)

        self.assertEqual(mode, "jina_v5")
        self.assertEqual(model.calls[0]["task"], "retrieval")
        self.assertEqual(model.calls[0]["prompt_name"], "query")
        self.assertEqual(model.calls[1]["task"], "retrieval")
        self.assertEqual(model.calls[1]["prompt_name"], "document")

    def test_embedding_server_tokenize_lengths_reports_effective_lengths(self):
        embed_module = _load_module(
            "codex_test_embed_tokenize",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        class Registry:
            def get(self, index_name):
                return FakeModel()

            def task_prefix_mode(self, index_name):
                return "none"

            def model_id(self, index_name):
                return "fake-model"

        embed_module.registry = Registry()
        embed_module.EMBEDDING_DISABLE_AUTH = True
        payload = embed_module.TokenizeLengthsRequest(
            texts=["one two three", "one two"],
            task="passage",
            index_name="large",
        )

        response = embed_module.tokenize_lengths(payload)

        self.assertEqual(response.lengths, [3, 2])
        self.assertEqual(response.model_id, "fake-model")

    def test_embedding_server_uses_model_tokenize_when_tokenizer_has_no_input_ids(self):
        embed_module = _load_module(
            "codex_test_embed_tokenize_fallback",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        class TensorLike:
            def __init__(self, data):
                self.data = data

            def tolist(self):
                return self.data

        class ModelWithSentenceTransformerTokenize(FakeModel):
            class BrokenTokenizer:
                def __call__(self, texts, **kwargs):
                    return {"attention_mask": [[1] for _ in texts]}

                def decode(self, ids, skip_special_tokens=True):
                    return " ".join(f"tok{token}" for token in ids)

            def __init__(self):
                super().__init__()
                self.tokenizer = self.BrokenTokenizer()

            def tokenize(self, texts):
                return {"input_ids": TensorLike([[1, 2, 3], [4, 5]])}

        prepared = embed_module._build_tokenized_inputs(
            ModelWithSentenceTransformerTokenize(),
            ["one two three", "one two"],
            task="passage",
            mode="none",
        )

        self.assertEqual(prepared["effective_lengths"], [3, 2])

    def test_embedding_server_falls_back_to_heuristic_lengths_when_token_ids_are_unavailable(self):
        embed_module = _load_module(
            "codex_test_embed_tokenize_heuristic",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        class ModelWithBrokenTokenizer(FakeModel):
            class BrokenTokenizer:
                def __call__(self, texts, **kwargs):
                    return {"attention_mask": [[1] for _ in texts]}

            def __init__(self):
                super().__init__()
                self.tokenizer = self.BrokenTokenizer()

        prepared = embed_module._build_tokenized_inputs(
            ModelWithBrokenTokenizer(),
            ["one two three"],
            task="passage",
            mode="none",
        )

        self.assertGreaterEqual(prepared["effective_lengths"][0], 3)
        self.assertEqual(len(prepared["texts"]), 1)

    def test_embedding_server_prefers_qwen_0_6b_before_kure(self):
        embed_module = _load_module(
            "codex_test_embed_server_pick",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "models"
            qwen_dir = _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-0.6B")
            _write_fake_model_dir(model_root / "kure-v1")

            picked = embed_module._pick_model_dir_from_root(str(model_root))

            self.assertEqual(Path(picked), qwen_dir)

    def test_load_gpu_env_prefers_qwen_0_6b_before_kure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "models"
            qwen_dir = _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-0.6B")
            _write_fake_model_dir(model_root / "kure-v1")

            load_env = _to_bash_path(ROOT / "project-gpu" / "load_gpu_env.sh")
            model_root_bash = _to_bash_path(model_root)
            script = f"""
set -euo pipefail
source '{load_env}'
result="$(compass_resolve_embedding_model_path '{model_root_bash}')"
printf '%s\\n' "$result"
"""
            completed = _run_bash(
                script,
                env={
                    "COMPASSLM_HOME": _to_bash_path(ROOT),
                    "LLM_MODELS_DIR": "/tmp/llm-models",
                    "LLM_RUNTIME": "/tmp/llama-server",
                    "LLM_MODEL_PATH": "/tmp/llm-model.gguf",
                },
            )

            self.assertEqual(completed.stdout.strip(), _to_bash_path(qwen_dir))

    def test_embedding_server_priority_is_qwen_0_6b_then_kure_then_multilingual_then_qwen_4b(self):
        embed_module = _load_module(
            "codex_test_embed_server_priority",
            EMBED_SERVER_PATH,
            extra_modules={
                "numpy": _stub_numpy_module(),
                "fastapi": _stub_fastapi_module(),
                "pydantic": _stub_pydantic_module(),
                "sentence_transformers": _stub_sentence_transformers_module(),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "models"
            qwen06_dir = _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-0.6B")
            _write_fake_model_dir(model_root / "kure-v1")
            _write_fake_model_dir(model_root / "multilingual-e5-large")
            _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-4B")

            picked = embed_module._pick_model_dir_from_root(str(model_root))

            self.assertEqual(Path(picked), qwen06_dir)

    def test_load_gpu_env_priority_is_qwen_0_6b_then_kure_then_multilingual_then_qwen_4b(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "models"
            qwen06_dir = _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-0.6B")
            _write_fake_model_dir(model_root / "kure-v1")
            _write_fake_model_dir(model_root / "multilingual-e5-large")
            _write_fake_model_dir(model_root / "Qwen" / "Qwen3-Embedding-4B")

            load_env = _to_bash_path(ROOT / "project-gpu" / "load_gpu_env.sh")
            model_root_bash = _to_bash_path(model_root)
            script = f"""
set -euo pipefail
source '{load_env}'
result="$(compass_resolve_embedding_model_path '{model_root_bash}')"
printf '%s\\n' "$result"
"""
            completed = _run_bash(
                script,
                env={
                    "COMPASSLM_HOME": _to_bash_path(ROOT),
                    "LLM_MODELS_DIR": "/tmp/llm-models",
                    "LLM_RUNTIME": "/tmp/llama-server",
                    "LLM_MODEL_PATH": "/tmp/llm-model.gguf",
                },
            )

            self.assertEqual(completed.stdout.strip(), _to_bash_path(qwen06_dir))


if __name__ == "__main__":
    unittest.main()
