import gc
import hmac
import os
import traceback
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:  # pragma: no cover - keep the server importable for diagnostics
    torch = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_local_path(value: str) -> bool:
    if not value:
        return False
    v = value.strip()
    if "${" in v or v.startswith("$"):
        return True
    if v.startswith(("/", "./", "../", "~/")):
        return True
    if len(v) >= 2 and v[1] == ":":
        return True
    return os.path.exists(os.path.expanduser(v))


def _strip_wrapping_quotes(value: str) -> str:
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        return v[1:-1].strip()
    return v


def _dir_has_model_markers(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    names = set(os.listdir(model_dir))
    # Sentence-transformers style
    if "modules.json" in names or "config_sentence_transformers.json" in names:
        return True
    # HF style
    has_config = "config.json" in names
    has_weights = any(
        name == "model.safetensors"
        or name.endswith(".safetensors")
        or name == "pytorch_model.bin"
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        for name in names
    )
    return has_config and has_weights


def _embedding_model_load_kwargs(model_ref: str) -> dict:
    lowered = (model_ref or "").strip().lower()
    if "jina" in lowered and "embeddings-v5" in lowered:
        return {"trust_remote_code": True}
    return {}


def _embedding_model_requirements_hint(model_ref: str) -> str:
    lowered = (model_ref or "").strip().lower()
    if "jina" in lowered and "embeddings-v5" in lowered:
        return " Jina v5 계열은 transformers>=4.57.0, torch>=2.8.0, peft>=0.15.2 환경이 필요할 수 있습니다."
    return ""


def _pick_model_dir_from_root(root_dir: str, max_depth: int = 6) -> str:
    if not os.path.isdir(root_dir):
        return root_dir
    if _dir_has_model_markers(root_dir):
        return root_dir

    preferred_tokens = (
        "qwen3-embedding-0.6b",
        "qwen3-embedding-0-6b",
        "kure-v1",
        "kure_v1",
        "multilingual-e5-large",
        "multilingual-e5-base",
        "qwen3-embedding-4b",
        "jina-embeddings-v5-text-small",
    )
    preferred_candidates = {token: [] for token in preferred_tokens}
    fallback_candidates: List[str] = []

    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    for current_root, dirnames, _ in os.walk(root_dir):
        cur_depth = current_root.rstrip(os.sep).count(os.sep) - root_depth
        if cur_depth > max_depth:
            dirnames[:] = []
            continue

        if _dir_has_model_markers(current_root):
            lower = current_root.lower()
            matched = False
            for token in preferred_tokens:
                if token in lower:
                    preferred_candidates[token].append(current_root)
                    matched = True
                    break
            if not matched:
                fallback_candidates.append(current_root)

    for token in preferred_tokens:
        candidates = preferred_candidates[token]
        if candidates:
            return sorted(candidates)[0]
    if fallback_candidates:
        return sorted(fallback_candidates)[0]
    return root_dir


EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
EMBEDDING_DISABLE_AUTH = _env_bool("EMBEDDING_DISABLE_AUTH", False)
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "16")))
EMBED_MIN_BATCH_SIZE = max(1, int(os.getenv("EMBED_MIN_BATCH_SIZE", "4")))
EMBED_NORMALIZE = _env_bool("EMBED_NORMALIZE", True)
EMBEDDING_TASK_PREFIX_MODE_RAW = (os.getenv("EMBEDDING_TASK_PREFIX_MODE", "auto") or "auto").strip().lower()
EMBEDDING_QWEN_QUERY_INSTRUCTION = (
    os.getenv(
        "EMBEDDING_QWEN_QUERY_INSTRUCTION",
        "Given a user question, retrieve relevant passages that answer the question",
    ).strip()
    or "Given a user question, retrieve relevant passages that answer the question"
)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_SERVER_HOME = os.path.abspath(os.path.join(APP_DIR, ".."))
DEFAULT_LARGE_MODEL_PATH = os.path.join(EMBEDDING_SERVER_HOME, "models")
PROJECT_GPU_HOME = os.path.abspath(os.path.join(EMBEDDING_SERVER_HOME, ".."))
COMPASSLM_HOME = os.path.abspath(os.path.join(PROJECT_GPU_HOME, ".."))


if EMBED_MIN_BATCH_SIZE > EMBED_BATCH_SIZE:
    EMBED_MIN_BATCH_SIZE = EMBED_BATCH_SIZE


def _is_torch_oom(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in message
        for token in (
            "outofmemoryerror",
            "cuda out of memory",
            "out of memory",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
        )
    )


def _clear_torch_cuda_cache():
    gc.collect()
    if torch is None or not hasattr(torch, "cuda"):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return


def _resolve_large_model_id(raw_value: str) -> str:
    value = _strip_wrapping_quotes(raw_value)
    if not value:
        return DEFAULT_LARGE_MODEL_PATH

    # Resolve common shell-style variables that may be left unresolved depending on launcher.
    var_map = {
        "COMPASSLM_HOME": COMPASSLM_HOME,
        "PROJECT_GPU_HOME": PROJECT_GPU_HOME,
        "EMBEDDING_SERVER_HOME": EMBEDDING_SERVER_HOME,
    }
    for key, resolved in var_map.items():
        value = value.replace(f"${{{key}}}", resolved)
        value = value.replace(f"${key}", resolved)

    value = os.path.expanduser(os.path.expandvars(value))

    # Common relative style used in configs.
    if value == "models" or value.startswith("models/"):
        value = os.path.join(EMBEDDING_SERVER_HOME, value)
    elif value.startswith(("./", "../")):
        value = os.path.abspath(os.path.join(EMBEDDING_SERVER_HOME, value))

    return value


RAW_LARGE_MODEL_ID = (os.getenv("EMBEDDING_MODEL_LARGE_PATH", DEFAULT_LARGE_MODEL_PATH) or DEFAULT_LARGE_MODEL_PATH).strip()
LARGE_MODEL_ID = _resolve_large_model_id(RAW_LARGE_MODEL_ID)
DEFAULT_INDEX = (os.getenv("EMBEDDING_DEFAULT_INDEX", "large") or "large").strip().lower()
LARGE_ALIAS = (os.getenv("EMBEDDING_API_LARGE_ALIAS", "large") or "large").strip().lower()
MODEL_DEVICE = os.getenv("EMBEDDING_MODEL_DEVICE", "").strip() or None

if _looks_like_local_path(LARGE_MODEL_ID):
    _resolved = os.path.expanduser(LARGE_MODEL_ID)
    if not os.path.isabs(_resolved) and not (len(_resolved) >= 2 and _resolved[1] == ":"):
        _resolved = os.path.abspath(os.path.join(EMBEDDING_SERVER_HOME, _resolved))
    LARGE_MODEL_ID = _pick_model_dir_from_root(_resolved)

if RAW_LARGE_MODEL_ID != LARGE_MODEL_ID:
    print(f"[INFO] EMBEDDING_MODEL_LARGE_PATH resolved: {RAW_LARGE_MODEL_ID} -> {LARGE_MODEL_ID}")


def _resolve_task_prefix_mode(model_id: str, requested_mode: str) -> str:
    mode = (requested_mode or "auto").strip().lower()
    if mode in {"none", "e5", "qwen", "jina_v5"}:
        return mode
    if mode != "auto":
        print(f"[WARN] Unsupported EMBEDDING_TASK_PREFIX_MODE='{requested_mode}', fallback to auto.")
    lowered = (model_id or "").lower()
    if "jina" in lowered and "embeddings-v5" in lowered:
        return "jina_v5"
    if "qwen3-embedding" in lowered or ("qwen" in lowered and "embedding" in lowered):
        return "qwen"
    if "e5" in lowered:
        return "e5"
    return "none"


def _apply_task_prefix(text: str, task: str, mode: str) -> str:
    value = (text or "").strip()
    if mode == "qwen" and task == "query":
        return f"Instruct: {EMBEDDING_QWEN_QUERY_INSTRUCTION}\nQuery: {value}"
    if mode == "e5":
        return f"{task}: {value}"
    return value


def _prompt_name_aliases_for_task(task: str) -> List[str]:
    if task == "query":
        return ["query", "question", "qry"]
    return ["document", "doc", "passage", "text"]


def _extract_model_prompt_keys(model: SentenceTransformer) -> List[str]:
    prompts = getattr(model, "prompts", None)
    if isinstance(prompts, dict):
        return [str(key) for key in prompts.keys()]
    return []


def _resolve_prompt_name_candidates(model: SentenceTransformer, task: str) -> List[str]:
    aliases = _prompt_name_aliases_for_task(task)
    configured_keys = _extract_model_prompt_keys(model)
    if not configured_keys:
        return aliases

    lowered = {str(key).strip().lower(): str(key) for key in configured_keys}
    candidates: List[str] = []
    for alias in aliases:
        actual = lowered.get(alias)
        if actual and actual not in candidates:
            candidates.append(actual)

    if candidates:
        return candidates
    return configured_keys if task == "query" else configured_keys[::-1]


def _encode_attempt_options(model: SentenceTransformer, task: str, mode: str) -> List[dict]:
    if mode == "jina_v5":
        prompt_candidates = _resolve_prompt_name_candidates(model, task)
        attempts: List[dict] = [
            {"task": "retrieval", "prompt_name": prompt_name}
            for prompt_name in prompt_candidates
        ]
        attempts.append({"task": "retrieval"})
        attempts.extend({"prompt_name": prompt_name} for prompt_name in prompt_candidates)
        attempts.append({})
        return attempts
    if mode == "qwen":
        return [{"prompt_name": prompt_name} for prompt_name in _resolve_prompt_name_candidates(model, task)] + [{}]
    return [{}]


def _encode_with_mode(
    model: SentenceTransformer,
    texts: List[str],
    task: str,
    mode: str,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    kwargs = {
        "normalize_embeddings": EMBED_NORMALIZE,
        "convert_to_numpy": True,
        "batch_size": int(batch_size or EMBED_BATCH_SIZE),
        "show_progress_bar": False,
    }
    vectors = None
    last_error: Optional[Exception] = None
    for extra_kwargs in _encode_attempt_options(model, task, mode):
        attempt_kwargs = dict(kwargs)
        attempt_kwargs.update(extra_kwargs)
        try:
            vectors = model.encode(texts, **attempt_kwargs)
            break
        except TypeError as e:
            last_error = e
            continue
        except ValueError as e:
            if extra_kwargs and ("Prompt name" in str(e) or "task" in str(e).lower()):
                last_error = e
                continue
            raise

    if vectors is None:
        if last_error is not None:
            raise last_error
        vectors = model.encode(texts, **kwargs)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _encode_with_adaptive_batches(
    model: SentenceTransformer,
    texts: List[str],
    task: str,
    mode: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    current_batch_size = min(EMBED_BATCH_SIZE, len(texts))
    merged: List[np.ndarray] = []
    start = 0

    while start < len(texts):
        batch = texts[start : start + current_batch_size]
        try:
            arr = _encode_with_mode(
                model=model,
                texts=batch,
                task=task,
                mode=mode,
                batch_size=current_batch_size,
            )
            merged.append(arr)
            start += len(batch)
        except Exception as e:
            if not _is_torch_oom(e):
                raise

            _clear_torch_cuda_cache()
            if current_batch_size <= EMBED_MIN_BATCH_SIZE:
                raise RuntimeError(
                    "Embedding CUDA out of memory even after reducing batch size "
                    f"to {current_batch_size}. Lower EMBED_BATCH_SIZE/EMBEDDING_API_BATCH_SIZE "
                    "or free GPU memory from competing processes."
                ) from e

            next_batch_size = max(EMBED_MIN_BATCH_SIZE, current_batch_size // 2)
            if next_batch_size == current_batch_size:
                next_batch_size = max(1, current_batch_size - 1)
            print(
                "[WARN] Embedding CUDA OOM "
                f"task={task} start={start} request_count={len(texts)} "
                f"batch_count={len(batch)} batch_size={current_batch_size} -> retry {next_batch_size}"
            )
            current_batch_size = next_batch_size

    if len(merged) == 1:
        return merged[0]
    return np.concatenate(merged, axis=0)


class EmbedRequest(BaseModel):
    texts: List[str]
    task: str
    index_name: Optional[str] = None


class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    dim: int
    index_name: str
    model_id: str


def _normalize_index_name(name: Optional[str]) -> str:
    raw = (name or DEFAULT_INDEX or "large").strip().lower()
    if raw in {"large", "b", "secondary", "index_b", LARGE_ALIAS, "default"}:
        return "large"
    if raw in {"small", "a", "primary", "index_a"}:
        return "large"
    raise HTTPException(status_code=400, detail=f"Unsupported index_name: {name}")


class ModelRegistry:
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._dim: Optional[int] = None
        self._task_prefix_mode: Optional[str] = None

    def _load(self) -> SentenceTransformer:
        if _looks_like_local_path(LARGE_MODEL_ID):
            local_path = os.path.expanduser(LARGE_MODEL_ID)
            if not os.path.exists(local_path):
                raise RuntimeError(
                    f"Embedding model path not found: {LARGE_MODEL_ID}. "
                    "Set EMBEDDING_MODEL_LARGE_PATH to a valid local model directory."
                )
            if os.path.isdir(local_path) and not os.listdir(local_path):
                raise RuntimeError(
                    f"Embedding model directory is empty: {LARGE_MODEL_ID}. "
                    "Large model files may be missing on this machine."
                )

        kwargs = {"device": MODEL_DEVICE} if MODEL_DEVICE else {}
        kwargs.update(_embedding_model_load_kwargs(LARGE_MODEL_ID))
        try:
            model = SentenceTransformer(LARGE_MODEL_ID, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{LARGE_MODEL_ID}'. "
                "Check EMBEDDING_MODEL_LARGE_PATH and model files."
                f"{_embedding_model_requirements_hint(LARGE_MODEL_ID)}"
            ) from e
        task_prefix_mode = _resolve_task_prefix_mode(
            model_id=LARGE_MODEL_ID,
            requested_mode=EMBEDDING_TASK_PREFIX_MODE_RAW,
        )
        probe_text = _apply_task_prefix("dim probe", task="passage", mode=task_prefix_mode)
        probe = _encode_with_mode(
            model=model,
            texts=[probe_text],
            task="passage",
            mode=task_prefix_mode,
        )
        arr = np.asarray(probe)
        dim = int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])
        self._model = model
        self._dim = dim
        self._task_prefix_mode = task_prefix_mode
        print(f"[INFO] Embedding task_prefix_mode={task_prefix_mode} model={LARGE_MODEL_ID}")
        return model

    def get(self, index_name: str) -> SentenceTransformer:
        _ = _normalize_index_name(index_name)
        if self._model is not None:
            return self._model
        return self._load()

    def dim(self, index_name: str) -> int:
        _ = _normalize_index_name(index_name)
        if self._dim is None:
            self.get("large")
        return int(self._dim or 0)

    def model_id(self, index_name: str) -> str:
        _ = _normalize_index_name(index_name)
        return LARGE_MODEL_ID

    def task_prefix_mode(self, index_name: str) -> str:
        _ = _normalize_index_name(index_name)
        if self._task_prefix_mode is None:
            self.get("large")
        return self._task_prefix_mode or "none"


registry = ModelRegistry()
app = FastAPI(title="CompassLM Embedding GPU Server", version="0.1.0")


def _authorize(authorization: Optional[str]):
    if EMBEDDING_DISABLE_AUTH:
        return
    if not EMBEDDING_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EMBEDDING_API_KEY is not configured on server.",
        )
    expected = f"Bearer {EMBEDDING_API_KEY}"
    received = (authorization or "").strip()
    if not hmac.compare_digest(received, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _validate_request(payload: EmbedRequest):
    if payload.task not in {"query", "passage"}:
        raise HTTPException(status_code=400, detail="task must be 'query' or 'passage'")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if len(payload.texts) > 512:
        raise HTTPException(status_code=400, detail="texts exceeds max batch size (512)")


@app.on_event("startup")
def _startup():
    registry.get("large")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "auth_enabled": not EMBEDDING_DISABLE_AUTH,
        "default_index": "large",
        "indexes": {
            "large": {
                "model_id": registry.model_id("large"),
                "dim": registry.dim("large"),
                "task_prefix_mode": registry.task_prefix_mode("large"),
                "embed_batch_size": EMBED_BATCH_SIZE,
                "embed_min_batch_size": EMBED_MIN_BATCH_SIZE,
            },
        },
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    _validate_request(payload)

    try:
        index_name = _normalize_index_name(payload.index_name)
        model = registry.get(index_name)
        task_prefix_mode = registry.task_prefix_mode(index_name)
        prefixed = [_apply_task_prefix((text or ""), task=payload.task, mode=task_prefix_mode) for text in payload.texts]
        arr = _encode_with_adaptive_batches(
            model=model,
            texts=prefixed,
            task=payload.task,
            mode=task_prefix_mode,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(
            f"[EMBED][ERROR] task={payload.task} count={len(payload.texts)} "
            f"index={payload.index_name or DEFAULT_INDEX} error={type(e).__name__}: {e}"
        )
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return EmbedResponse(
        vectors=arr.tolist(),
        dim=int(arr.shape[1]),
        index_name=index_name,
        model_id=registry.model_id(index_name),
    )
