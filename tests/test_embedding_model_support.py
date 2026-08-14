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


class FakeArray:
    def __init__(self, data):
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


class FakeModel:
    def __init__(self):
        self.prompts = {"query": "query", "document": "document"}
        self.calls = []

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
    module.status = types.SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
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

            script = f"""
set -euo pipefail
source '{ROOT / "project-gpu" / "load_gpu_env.sh"}'
result="$(compass_resolve_embedding_model_path '{model_root}')"
printf '%s\\n' "$result"
"""
            completed = subprocess.run(
                ["bash", "-lc", script],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "COMPASSLM_HOME": str(ROOT)},
            )

            self.assertEqual(Path(completed.stdout.strip()), qwen_dir)

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

            script = f"""
set -euo pipefail
source '{ROOT / "project-gpu" / "load_gpu_env.sh"}'
result="$(compass_resolve_embedding_model_path '{model_root}')"
printf '%s\\n' "$result"
"""
            completed = subprocess.run(
                ["bash", "-lc", script],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "COMPASSLM_HOME": str(ROOT)},
            )

            self.assertEqual(Path(completed.stdout.strip()), qwen06_dir)


if __name__ == "__main__":
    unittest.main()
