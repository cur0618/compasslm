import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PDF_OCR_PATH = ROOT / "src" / "pdf_ocr.py"
RAG_PATH = ROOT / "src" / "rag.py"


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


class _FakeFitzPage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self, kind: str = "text") -> str:
        return self._text


class _FakeFitzDocument:
    def __init__(self, page_texts: list[str]):
        self._pages = [_FakeFitzPage(text) for text in page_texts]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __len__(self):
        return len(self._pages)

    def __iter__(self):
        return iter(self._pages)

    def load_page(self, index: int):
        return self._pages[index]


class _FakeSubsetFitzDocument:
    def __init__(self):
        self._pages: list[_FakeFitzPage] = []

    def insert_pdf(self, source_doc, from_page: int, to_page: int):
        for page_index in range(from_page, to_page + 1):
            page = source_doc.load_page(page_index)
            self._pages.append(_FakeFitzPage(page.get_text("text")))

    def save(self, path: str):
        Path(path).write_text("\n".join(page.get_text("text") for page in self._pages), encoding="utf-8")

    def close(self):
        return None


def _stub_fitz_module(page_texts: list[str]) -> types.ModuleType:
    module = types.ModuleType("fitz")

    def _open(path=None):
        if path is None:
            return _FakeSubsetFitzDocument()
        return _FakeFitzDocument(page_texts)

    module.open = _open
    return module


def _stub_numpy_module():
    module = types.ModuleType("numpy")

    class _FakeArray(list):
        @property
        def ndim(self):
            if not self:
                return 1
            return 2 if isinstance(self[0], (list, tuple)) else 1

        @property
        def shape(self):
            if self.ndim == 2:
                return (len(self), len(self[0]) if self else 0)
            return (len(self),)

        def astype(self, *args, **kwargs):
            return self

    module.float32 = "float32"
    module.int64 = "int64"
    module.ndarray = object
    module.asarray = lambda data, dtype=None: _FakeArray(data)
    module.array = lambda data, dtype=None: _FakeArray(data)
    module.empty = lambda shape, dtype=None: _FakeArray([])
    module.concatenate = lambda arrays, axis=0: _FakeArray(sum(arrays, []))
    module.vstack = lambda arrays: _FakeArray(sum(arrays, []))
    return module


def _stub_sentence_transformers_module():
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def encode(self, texts, **kwargs):
            return [[0.1, 0.2] for _ in texts]

    module.SentenceTransformer = SentenceTransformer
    return module


class PdfOcrPageMetadataTests(unittest.TestCase):
    def test_extract_page_no_ignores_nested_non_page_numbers(self):
        pdf_module = _load_module(
            "codex_test_pdf_page_no_sanitizes_nested_noise",
            PDF_OCR_PATH,
            extra_modules={},
        )

        payload = {
            "text": "실제 OCR 텍스트",
            "blocks": [
                {
                    "index": 67418,
                    "bbox": [10, 20, 300, 400],
                    "text": "중첩된 블록 텍스트",
                }
            ],
            "metadata": {
                "image_path": "/tmp/rendered/document_page_99999_block_1.png",
            },
        }

        self.assertEqual(pdf_module._extract_page_no(payload, fallback=14, max_page=200), 14)

    def test_extract_page_no_accepts_explicit_page_within_document_range(self):
        pdf_module = _load_module(
            "codex_test_pdf_page_no_accepts_explicit_page",
            PDF_OCR_PATH,
            extra_modules={},
        )

        self.assertEqual(pdf_module._extract_page_no({"page_no": 14, "index": 67418}, fallback=1, max_page=200), 14)
        self.assertEqual(pdf_module._extract_page_no({"page_no": 67418}, fallback=7, max_page=200), 7)

    def test_pdf_ingest_stats_only_parse_strict_pdf_page_sections(self):
        rag_module = _load_module(
            "codex_test_rag_pdf_page_stats_strict_section",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        stats = rag_module._summarize_pdf_chunk_pages(
            [
                {"section": "PDF page 14", "page_parser": "paddleocr_vl"},
                {"section": "chapter 2024 example", "page_parser": "paddleocr_vl"},
            ]
        )

        self.assertEqual(stats["pdf_total_pages"], 1)
        self.assertEqual(stats["pdf_ocr_pages"], 1)


class HybridPdfIngestTests(unittest.TestCase):
    def test_ocr_progress_heartbeat_emits_during_blocking_work(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_heartbeat",
            PDF_OCR_PATH,
            extra_modules={},
        )

        callback_event = threading.Event()
        heartbeats: list[tuple[int, str, str]] = []

        def _progress_callback(percent: int, message: str, stage: str):
            heartbeats.append((percent, message, stage))
            callback_event.set()

        with pdf_module._ocr_progress_heartbeat(
            _progress_callback,
            percent=42,
            message="PDF OCR을 실행하는 중입니다.",
            stage="run_pdf_ocr",
            interval_seconds=0.01,
        ):
            self.assertTrue(callback_event.wait(0.2))

        recorded = len(heartbeats)
        threading.Event().wait(0.03)

        self.assertGreaterEqual(recorded, 1)
        self.assertEqual(heartbeats[0], (42, "PDF OCR을 실행하는 중입니다.", "run_pdf_ocr"))
        self.assertEqual(len(heartbeats), recorded)

    def test_available_gpu_memory_bytes_reads_nvidia_smi_and_respects_limit(self):
        pdf_module = _load_module(
            "codex_test_pdf_gpu_available_memory",
            PDF_OCR_PATH,
            extra_modules={},
        )

        completed = types.SimpleNamespace(
            returncode=0,
            stdout="24576, 32768\n",
            stderr="",
        )

        with mock.patch.dict(os.environ, {"PDF_OCR_GPU_BUDGET_GB": "10"}, clear=False):
            with mock.patch.object(pdf_module.subprocess, "run", return_value=completed) as run_mock:
                available = pdf_module._available_gpu_memory_bytes("gpu:0")

        self.assertEqual(available, 10 * 1024 * 1024 * 1024)
        run_mock.assert_called_once()

    def test_auto_parallel_ocr_workers_use_system_ram_for_cpu_and_vram_for_gpu(self):
        pdf_module = _load_module(
            "codex_test_pdf_auto_parallel_workers",
            PDF_OCR_PATH,
            extra_modules={},
        )

        with mock.patch.dict(os.environ, {"PDF_OCR_PARALLEL_MAX_WORKERS": "2"}, clear=False):
            with mock.patch.object(pdf_module, "_available_memory_bytes", return_value=48 * 1024 * 1024 * 1024) as ram_mock:
                with mock.patch.object(pdf_module, "_available_gpu_memory_bytes", return_value=24 * 1024 * 1024 * 1024) as vram_mock:
                    workers = pdf_module._recommended_parallel_ocr_workers(
                        device="cpu",
                        candidate_page_count=48,
                        cpu_count=16,
                    )
        self.assertEqual(workers, 2)
        ram_mock.assert_called_once_with()
        vram_mock.assert_not_called()

        with mock.patch.dict(os.environ, {"PDF_OCR_PARALLEL_MAX_WORKERS": "2"}, clear=False):
            workers_low_mem = pdf_module._recommended_parallel_ocr_workers(
                device="cpu",
                candidate_page_count=48,
                cpu_count=16,
                available_bytes=8 * 1024 * 1024 * 1024,
            )
        self.assertEqual(workers_low_mem, 1)

        with mock.patch.dict(
            os.environ,
            {
                "PDF_OCR_PARALLEL_GPU_MEM_GB_PER_WORKER": "4",
                "PDF_OCR_PARALLEL_GPU_MEM_RESERVE_GB": "2",
                "PDF_OCR_PARALLEL_MAX_WORKERS": "2",
            },
            clear=False,
        ):
            with mock.patch.object(pdf_module, "_available_memory_bytes", return_value=48 * 1024 * 1024 * 1024) as ram_mock:
                with mock.patch.object(pdf_module, "_available_gpu_memory_bytes", return_value=12 * 1024 * 1024 * 1024) as vram_mock:
                    workers_gpu = pdf_module._recommended_parallel_ocr_workers(
                        device="gpu:0",
                        candidate_page_count=48,
                        cpu_count=16,
                    )
        self.assertEqual(workers_gpu, 2)
        ram_mock.assert_not_called()
        vram_mock.assert_called_once_with("gpu:0")

    def test_parallel_ocr_batches_split_pages_contiguously(self):
        pdf_module = _load_module(
            "codex_test_pdf_parallel_batches",
            PDF_OCR_PATH,
            extra_modules={},
        )

        batches = pdf_module._split_parallel_ocr_page_batches(list(range(1, 11)), worker_count=2)

        self.assertEqual(batches, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    def test_selected_ocr_batches_report_exact_pdf_page_progress(self):
        pdf_module = _load_module(
            "codex_test_pdf_selected_ocr_progress",
            PDF_OCR_PATH,
            extra_modules={},
        )

        subset_pages_by_path: dict[str, list[int]] = {}
        temp_paths: list[str] = []
        progress_events: list[tuple[int, str, str, dict[str, int]]] = []

        def _fake_build_subset(pdf_path, page_numbers):
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.close()
            temp_paths.append(handle.name)
            subset_pages_by_path[handle.name] = list(page_numbers)
            return handle.name, {idx + 1: page_no for idx, page_no in enumerate(page_numbers)}

        def _fake_call_serial(pdf_path, progress_callback=None, *, model_name=None, max_pages=None, min_text_chars=None, device=None, release_after_run_override=None, progress_meta=None):
            return [
                {"page_no": idx + 1, "text": f"OCR {page_no}"}
                for idx, page_no in enumerate(subset_pages_by_path[pdf_path])
            ]

        def _fake_execute(tasks, **kwargs):
            completed = [
                (
                    task,
                    [
                        {"page_no": idx + 1, "text": f"OCR {page_no}"}
                        for idx, page_no in enumerate(task["page_numbers"])
                    ],
                )
                for task in tasks
            ]
            for task, pages in completed:
                kwargs["on_batch_completed"](task, pages, {})
            return {
                "completed": completed,
                "remaining_tasks": [],
                "failure_reason": "",
                "runtime_info": {},
            }

        pdf_module._build_pdf_subset_for_pages = _fake_build_subset
        pdf_module._call_serial_ocr = _fake_call_serial
        pdf_module._execute_ocr_subset_tasks = _fake_execute
        pdf_module._recommended_parallel_ocr_workers = lambda **kwargs: 1

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                with mock.patch.dict(os.environ, {"PDF_OCR_PROGRESS_BATCH_PAGES": "1"}, clear=False):
                    pages = pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        tmp_pdf.name,
                        page_numbers=[2, 4],
                        progress_callback=lambda percent, message, stage, **meta: progress_events.append((percent, message, stage, dict(meta))),
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        total_document_pages=4,
                        completed_pages_base=2,
                    )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        self.assertEqual([page["page_no"] for page in pages], [2, 4])
        run_events = [event for event in progress_events if event[2] == "run_pdf_ocr"]
        self.assertTrue(any(meta.get("current_page") == 2 and meta.get("total_pages") == 4 for _, _, _, meta in run_events))
        self.assertTrue(any(meta.get("current_page") == 3 and meta.get("ocr_completed_pages") == 1 for _, _, _, meta in run_events))
        self.assertTrue(any(meta.get("current_page") == 4 and meta.get("ocr_completed_pages") == 2 for _, _, _, meta in run_events))

    def test_selected_ocr_execution_batches_can_be_larger_than_progress_batch_env(self):
        pdf_module = _load_module(
            "codex_test_pdf_exec_batch_split",
            PDF_OCR_PATH,
            extra_modules={},
        )

        subset_pages_by_path: dict[str, list[int]] = {}
        temp_paths: list[str] = []
        observed_task_batches: list[list[list[int]]] = []

        def _fake_build_subset(pdf_path, page_numbers):
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.close()
            temp_paths.append(handle.name)
            subset_pages_by_path[handle.name] = list(page_numbers)
            return handle.name, {idx + 1: page_no for idx, page_no in enumerate(page_numbers)}

        def _fake_execute(tasks, **kwargs):
            observed_task_batches.append([list(task["page_numbers"]) for task in tasks])
            completed = [
                (
                    task,
                    [
                        {"page_no": idx + 1, "text": f"OCR {page_no}"}
                        for idx, page_no in enumerate(subset_pages_by_path[task["pdf_path"]])
                    ],
                )
                for task in tasks
            ]
            for task, pages in completed:
                kwargs["on_batch_completed"](task, pages, {})
            return {
                "completed": completed,
                "remaining_tasks": [],
                "failure_reason": "",
            }

        pdf_module._build_pdf_subset_for_pages = _fake_build_subset
        pdf_module._execute_ocr_subset_tasks = _fake_execute
        pdf_module._recommended_parallel_ocr_workers = lambda **kwargs: 1

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_EXEC_BATCH_PAGES": "4",
                        "PDF_OCR_PROGRESS_BATCH_PAGES": "1",
                    },
                    clear=False,
                ):
                    pages = pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        tmp_pdf.name,
                        page_numbers=[1, 2, 3, 4, 5, 6, 7],
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        total_document_pages=7,
                        completed_pages_base=0,
                    )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        self.assertEqual(observed_task_batches, [[[1, 2, 3, 4], [5, 6, 7]]])
        self.assertEqual([page["page_no"] for page in pages], [1, 2, 3, 4, 5, 6, 7])

    def test_parallel_ocr_worker_keeps_gpu_model_cached_between_subset_tasks(self):
        pdf_module = _load_module(
            "codex_test_pdf_parallel_worker_model_reuse",
            PDF_OCR_PATH,
            extra_modules={},
        )

        release_flags: list[bool | None] = []

        def _fake_serial(
            pdf_path,
            progress_callback=None,
            *,
            model_name=None,
            max_pages=None,
            min_text_chars=None,
            device_override=None,
            release_after_run_override=None,
            progress_meta=None,
        ):
            release_flags.append(release_after_run_override)
            return [{"page_no": 1, "text": f"OCR {Path(pdf_path).name}"}]

        pdf_module._extract_pdf_pages_with_paddleocr_vl_serial = _fake_serial

        first = pdf_module._parallel_ocr_subset_worker(
            {
                "pdf_path": "/tmp/subset-a.pdf",
                "model_name": "mock-model",
                "max_pages": 1,
                "min_text_chars": 4,
                "device": "gpu:0",
            }
        )
        second = pdf_module._parallel_ocr_subset_worker(
            {
                "pdf_path": "/tmp/subset-b.pdf",
                "model_name": "mock-model",
                "max_pages": 1,
                "min_text_chars": 4,
                "device": "gpu:0",
            }
        )

        self.assertEqual(
            [page["text"] for page in list(first["pages"]) + list(second["pages"])],
            ["OCR subset-a.pdf", "OCR subset-b.pdf"],
        )
        self.assertEqual(release_flags, [False, False])

    def test_persistent_gpu_ocr_worker_reuses_process_pool_between_uploads(self):
        pdf_module = _load_module(
            "codex_test_pdf_persistent_worker_reuse",
            PDF_OCR_PATH,
            extra_modules={},
        )

        created_executors = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.max_workers = max_workers
                self.mp_context = mp_context
                self.submitted = []
                self.shutdown_calls = 0
                created_executors.append(self)

            def submit(self, fn, task):
                self.submitted.append(dict(task))
                future = pdf_module.concurrent.futures.Future()
                future.set_result(
                    {
                        "pages": [{"page_no": 1, "text": f"OCR {Path(task['pdf_path']).name}"}],
                        "runtime_info": {"ocr_model_load_seconds": 0.0},
                    }
                )
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_calls += 1

        def _run_once(pdf_name):
            return pdf_module._execute_ocr_subset_tasks(
                [
                    {
                        "pdf_path": f"/tmp/{pdf_name}",
                        "page_numbers": [1],
                    }
                ],
                model_name="mock-model",
                min_text_chars=4,
                device="gpu:0",
                max_workers=1,
                stage="upload_pdf_ocr",
                batch_timeout_seconds=600,
            )

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                    "PDF_OCR_PERSISTENT_WORKER": "1",
                    "PDF_OCR_PERSISTENT_WORKERS": "1",
                },
                clear=False,
            ):
                first = _run_once("upload-a.pdf")
                second = _run_once("upload-b.pdf")
                pdf_module.shutdown_persistent_ocr_worker()

        self.assertEqual(len(created_executors), 1)
        self.assertEqual(
            [item["pdf_path"] for item in created_executors[0].submitted],
            ["/tmp/upload-a.pdf", "/tmp/upload-b.pdf"],
        )
        self.assertEqual(created_executors[0].shutdown_calls, 1)
        self.assertEqual(first["failure_reason"], "")
        self.assertEqual(second["failure_reason"], "")

    def test_ocr_batch_submit_emits_current_page_progress(self):
        pdf_module = _load_module(
            "codex_test_pdf_batch_submit_progress",
            PDF_OCR_PATH,
            extra_modules={},
        )

        progress_events = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.shutdown_calls = 0

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                future.set_result(
                    {
                        "pages": [{"page_no": 1, "text": "OCR page 160"}],
                        "runtime_info": {},
                    }
                )
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_calls += 1

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                    "PDF_OCR_PERSISTENT_WORKER": "0",
                },
                clear=False,
            ):
                result = pdf_module._execute_ocr_subset_tasks(
                    [
                        {
                            "pdf_path": "/tmp/subset-page-160.pdf",
                            "page_numbers": [160],
                        }
                    ],
                    model_name="mock-model",
                    min_text_chars=4,
                    device="gpu:0",
                    max_workers=1,
                    stage="run_pdf_ocr",
                    progress_callback=lambda percent, message, stage, **meta: progress_events.append(
                        (percent, message, stage, dict(meta))
                    ),
                    progress_percent_fn=lambda: 49,
                    progress_meta_fn=lambda: {
                        "current_page": 159,
                        "total_pages": 160,
                        "ocr_completed_pages": 159,
                        "ocr_target_pages": 160,
                    },
                    batch_timeout_seconds=240,
                )

        self.assertEqual(result["failure_reason"], "")
        self.assertTrue(progress_events)
        percent, message, stage, meta = progress_events[0]
        self.assertEqual(percent, 49)
        self.assertEqual(stage, "run_pdf_ocr")
        self.assertIn("160/160페이지", message)
        self.assertEqual(meta["current_page"], 160)
        self.assertEqual(meta["total_pages"], 160)
        self.assertEqual(meta["ocr_completed_pages"], 159)

    def test_persistent_gpu_ocr_warmup_loads_model_once_in_worker(self):
        pdf_module = _load_module(
            "codex_test_pdf_persistent_worker_warmup",
            PDF_OCR_PATH,
            extra_modules={},
        )

        load_calls = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.shutdown_calls = 0

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                future.set_result(fn(task))
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_calls += 1

        def _fake_load(model_name, device=None):
            load_calls.append((model_name, device))
            return object()

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(pdf_module, "_load_ocr_model", _fake_load):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_DEVICE": "gpu:0",
                        "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                        "PDF_OCR_PERSISTENT_WORKER": "1",
                        "PDF_OCR_PERSISTENT_WORKERS": "1",
                    },
                    clear=False,
                ):
                    first = pdf_module.warmup_persistent_ocr_worker(model_name="mock-model", device="gpu:0")
                    second = pdf_module.warmup_persistent_ocr_worker(model_name="mock-model", device="gpu:0")
                    pdf_module.shutdown_persistent_ocr_worker()

        self.assertEqual(load_calls, [("mock-model", "gpu:0")])
        self.assertEqual(first["status"], "ready")
        self.assertEqual(second["status"], "ready")
        self.assertEqual(first["device"], "gpu:0")

    def test_persistent_gpu_ocr_warmup_primes_every_configured_worker(self):
        pdf_module = _load_module(
            "codex_test_pdf_persistent_worker_warmup_all",
            PDF_OCR_PATH,
            extra_modules={},
        )

        submitted = []
        worker_pids = iter([101, 102, 103])

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.max_workers = max_workers

            def submit(self, fn, task):
                submitted.append(dict(task))
                future = pdf_module.concurrent.futures.Future()
                future.set_result(
                    {
                        "status": "ready",
                        "device": task["device"],
                        "model_name": task["model_name"],
                        "model_load_seconds": 1.0,
                        "worker_pid": next(worker_pids),
                    }
                )
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_DEVICE": "gpu:0",
                    "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                    "PDF_OCR_PERSISTENT_WORKER": "1",
                    "PDF_OCR_PERSISTENT_WORKERS": "3",
                },
                clear=False,
            ):
                result = pdf_module.warmup_persistent_ocr_worker(model_name="mock-model", device="gpu:0")
                pdf_module.shutdown_persistent_ocr_worker()

        self.assertEqual(len(submitted), 3)
        self.assertEqual(result["worker_count"], 3)
        self.assertEqual(result["worker_pids"], [101, 102, 103])

    def test_persistent_gpu_ocr_warmup_rejects_duplicate_worker_processes(self):
        pdf_module = _load_module(
            "codex_test_pdf_persistent_worker_warmup_duplicates",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                return None

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                future.set_result(
                    {
                        "status": "ready",
                        "device": task["device"],
                        "model_name": task["model_name"],
                        "model_load_seconds": 1.0,
                        "worker_pid": 101,
                    }
                )
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_DEVICE": "gpu:0",
                    "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                    "PDF_OCR_PERSISTENT_WORKER": "1",
                    "PDF_OCR_PERSISTENT_WORKERS": "3",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "distinct worker processes"):
                    pdf_module.warmup_persistent_ocr_worker(
                        model_name="mock-model",
                        device="gpu:0",
                        timeout_seconds=1,
                    )

    def test_ocr_batch_submission_is_bounded_by_worker_count(self):
        pdf_module = _load_module(
            "codex_test_pdf_bounded_batch_submission",
            PDF_OCR_PATH,
            extra_modules={},
        )

        submitted = []
        submissions_seen_at_wait = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                self.max_workers = max_workers

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                submitted.append((dict(task), future))
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        def _fake_wait(futures, timeout=None, return_when=None):
            submissions_seen_at_wait.append(len(submitted))
            future = next(iter(futures))
            future.set_result({"pages": [], "runtime_info": {}})
            return {future}, set(futures) - {future}

        tasks = [
            {
                "batch_index": index,
                "pdf_path": f"/tmp/subset-{index}.pdf",
                "page_numbers": [index],
            }
            for index in range(1, 6)
        ]

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(pdf_module.concurrent.futures, "wait", _fake_wait):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                        "PDF_OCR_PERSISTENT_WORKER": "0",
                    },
                    clear=False,
                ):
                    result = pdf_module._execute_ocr_subset_tasks(
                        tasks,
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        max_workers=2,
                        stage="run_pdf_ocr",
                        batch_timeout_seconds=240,
                    )

        self.assertEqual(result["failure_reason"], "")
        self.assertEqual(len(submitted), 5)
        self.assertEqual(submissions_seen_at_wait[0], 2)
        self.assertTrue(all(count <= 5 for count in submissions_seen_at_wait))

    def test_ocr_batch_failure_returns_submitted_and_unsubmitted_tasks(self):
        pdf_module = _load_module(
            "codex_test_pdf_bounded_batch_failure",
            PDF_OCR_PATH,
            extra_modules={},
        )

        submitted = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                return None

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                submitted.append(future)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        def _fake_wait(futures, timeout=None, return_when=None):
            future = next(iter(futures))
            future.set_exception(RuntimeError("CUDA out of memory"))
            return {future}, set(futures) - {future}

        tasks = [
            {
                "batch_index": index,
                "pdf_path": f"/tmp/subset-{index}.pdf",
                "page_numbers": [index],
            }
            for index in range(1, 6)
        ]

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(pdf_module.concurrent.futures, "wait", _fake_wait):
                with mock.patch.object(pdf_module, "_terminate_process_pool", lambda executor: None):
                    with mock.patch.dict(
                        os.environ,
                        {
                            "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                            "PDF_OCR_PERSISTENT_WORKER": "0",
                        },
                        clear=False,
                    ):
                        result = pdf_module._execute_ocr_subset_tasks(
                            tasks,
                            model_name="mock-model",
                            min_text_chars=4,
                            device="gpu:0",
                            max_workers=2,
                            stage="run_pdf_ocr",
                            batch_timeout_seconds=240,
                        )

        self.assertEqual(len(submitted), 2)
        self.assertEqual(result["failure_reason"], "gpu_oom")
        self.assertEqual(
            sorted(task["batch_index"] for task in result["remaining_tasks"]),
            [1, 2, 3, 4, 5],
        )

    def test_ocr_batch_timeout_is_not_masked_by_other_batch_completions(self):
        pdf_module = _load_module(
            "codex_test_pdf_individual_batch_timeout",
            PDF_OCR_PATH,
            extra_modules={},
        )

        submitted = []
        clock = [0.0]
        completed_batches = []

        class _FakeExecutor:
            def __init__(self, max_workers=None, mp_context=None):
                return None

            def submit(self, fn, task):
                future = pdf_module.concurrent.futures.Future()
                submitted.append((dict(task), future))
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        def _fake_wait(futures, timeout=None, return_when=None):
            clock[0] += 1.0
            for task, future in submitted[1:]:
                if future in futures and not future.done():
                    future.set_result({"pages": [{"page_no": task["subset_page_count"], "text": "done"}], "runtime_info": {}})
                    return {future}, set(futures) - {future}
            return set(), set(futures)

        tasks = [
            {
                "batch_index": index,
                "pdf_path": f"/tmp/subset-{index}.pdf",
                "page_numbers": [index],
            }
            for index in range(1, 6)
        ]

        with mock.patch.object(pdf_module.concurrent.futures, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(pdf_module.concurrent.futures, "wait", _fake_wait):
                with mock.patch.object(pdf_module.time, "monotonic", side_effect=lambda: clock[0]):
                    with mock.patch.object(pdf_module, "_terminate_process_pool", lambda executor: None):
                        with mock.patch.dict(
                            os.environ,
                            {
                                "PDF_OCR_GPU_PROCESS_ISOLATION": "1",
                                "PDF_OCR_PERSISTENT_WORKER": "0",
                            },
                            clear=False,
                        ):
                            result = pdf_module._execute_ocr_subset_tasks(
                                tasks,
                                model_name="mock-model",
                                min_text_chars=4,
                                device="gpu:0",
                                max_workers=2,
                                stage="run_pdf_ocr",
                                batch_timeout_seconds=3,
                                on_batch_completed=lambda task, pages, runtime: completed_batches.append(
                                    task["batch_index"]
                                ),
                            )

        self.assertEqual(result["failure_reason"], "gpu_timeout")
        self.assertEqual(completed_batches, [2, 3, 4])
        self.assertEqual(
            sorted(task["batch_index"] for task in result["remaining_tasks"]),
            [1, 5],
        )

    def test_persistent_gpu_ocr_shutdown_terminates_stuck_pool(self):
        pdf_module = _load_module(
            "codex_test_pdf_persistent_worker_shutdown",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakeExecutor:
            pass

        fake_executor = _FakeExecutor()
        terminated = []

        pdf_module._PERSISTENT_OCR_EXECUTOR = fake_executor
        pdf_module._PERSISTENT_OCR_WORKERS = 1
        pdf_module._PERSISTENT_OCR_DEVICE = "gpu:0"
        pdf_module._PERSISTENT_OCR_MODEL_NAME = "mock-model"

        with mock.patch.object(pdf_module, "_terminate_process_pool", lambda executor: terminated.append(executor)):
            pdf_module.shutdown_persistent_ocr_worker()

        self.assertEqual(terminated, [fake_executor])
        self.assertIsNone(pdf_module._PERSISTENT_OCR_EXECUTOR)
        self.assertEqual(pdf_module._PERSISTENT_OCR_DEVICE, "")

    def test_selected_ocr_applies_timeout_to_first_single_gpu_batch(self):
        pdf_module = _load_module(
            "codex_test_pdf_single_gpu_first_batch_timeout",
            PDF_OCR_PATH,
            extra_modules={},
        )

        observed_timeouts: list[float | None] = []
        temp_paths: list[str] = []

        def _fake_build_subset(pdf_path, page_numbers):
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.close()
            temp_paths.append(handle.name)
            return handle.name, {idx + 1: page_no for idx, page_no in enumerate(page_numbers)}

        def _fake_execute(tasks, *, batch_timeout_seconds=None, **kwargs):
            observed_timeouts.append(batch_timeout_seconds)
            completed = [
                (task, [{"page_no": 1, "text": "first page"}])
                for task in tasks
            ]
            for task, pages in completed:
                kwargs["on_batch_completed"](task, pages, {})
            return {
                "completed": completed,
                "remaining_tasks": [],
                "failure_reason": "",
                "runtime_info": {},
            }

        pdf_module._build_pdf_subset_for_pages = _fake_build_subset
        pdf_module._execute_ocr_subset_tasks = _fake_execute
        pdf_module._recommended_parallel_ocr_workers = lambda **kwargs: 1

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_EXEC_BATCH_PAGES": "1",
                        "PDF_OCR_GPU_SINGLE_BATCH_TIMEOUT_SECONDS": "360",
                    },
                    clear=False,
                ):
                    pages = pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        tmp_pdf.name,
                        page_numbers=[1],
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        total_document_pages=1,
                        completed_pages_base=0,
                    )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        self.assertEqual(pages, [{"page_no": 1, "text": "first page"}])
        self.assertEqual(observed_timeouts, [360.0])

    def test_selected_ocr_batches_retry_single_gpu_after_parallel_gpu_oom(self):
        pdf_module = _load_module(
            "codex_test_pdf_parallel_gpu_downgrade",
            PDF_OCR_PATH,
            extra_modules={},
        )

        subset_pages_by_path: dict[str, list[int]] = {}
        temp_paths: list[str] = []
        call_modes: list[tuple[str, str, int, list[list[int]]]] = []

        def _fake_build_subset(pdf_path, page_numbers):
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.close()
            temp_paths.append(handle.name)
            subset_pages_by_path[handle.name] = list(page_numbers)
            return handle.name, {idx + 1: page_no for idx, page_no in enumerate(page_numbers)}

        def _page_payload(task):
            return [
                {"page_no": idx + 1, "text": f"OCR {page_no}"}
                for idx, page_no in enumerate(task["page_numbers"])
            ]

        def _fake_execute(tasks, *, model_name, min_text_chars, device, max_workers, stage, progress_callback=None, progress_percent_fn=None, progress_meta_fn=None, batch_timeout_seconds=None, on_batch_completed=None):
            task_page_sets = [list(task["page_numbers"]) for task in tasks]
            mode = "parallel_gpu" if device == "gpu:0" and max_workers > 1 else "single_gpu" if device == "gpu:0" else "cpu"
            call_modes.append((mode, device, max_workers, task_page_sets))
            if mode == "parallel_gpu":
                on_batch_completed(tasks[0], _page_payload(tasks[0]), {})
                return {
                    "completed": [(tasks[0], _page_payload(tasks[0]))],
                    "remaining_tasks": tasks[1:],
                    "failure_reason": "gpu_oom",
                }
            completed = [(task, _page_payload(task)) for task in tasks]
            for task, pages in completed:
                on_batch_completed(task, pages, {})
            return {
                "completed": completed,
                "remaining_tasks": [],
                "failure_reason": "",
            }

        pdf_module._build_pdf_subset_for_pages = _fake_build_subset
        pdf_module._execute_ocr_subset_tasks = _fake_execute
        pdf_module._recommended_parallel_ocr_workers = lambda **kwargs: 2

        runtime_info = {
            "ocr_device_attempted": "gpu:0",
            "ocr_device_effective": "gpu:0",
            "ocr_gpu_fallback_used": False,
            "ocr_gpu_failure_reason": "",
        }
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                with mock.patch.dict(os.environ, {"PDF_OCR_PROGRESS_BATCH_PAGES": "2"}, clear=False):
                    pages = pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        tmp_pdf.name,
                        page_numbers=[1, 2, 3, 4],
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        total_document_pages=4,
                        completed_pages_base=0,
                        runtime_info=runtime_info,
                    )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        self.assertEqual(call_modes[0][:3], ("parallel_gpu", "gpu:0", 2))
        self.assertEqual(call_modes[1][:3], ("single_gpu", "gpu:0", 1))
        self.assertEqual(call_modes[1][3], [[3, 4]])
        self.assertEqual([page["page_no"] for page in pages], [1, 2, 3, 4])
        self.assertFalse(runtime_info["ocr_gpu_fallback_used"])
        self.assertEqual(runtime_info["ocr_device_effective"], "gpu:0")
        self.assertEqual(runtime_info["ocr_retry_reason"], "gpu_oom")

    def test_selected_ocr_batches_fallback_to_cpu_after_single_gpu_failure(self):
        pdf_module = _load_module(
            "codex_test_pdf_parallel_gpu_to_cpu_fallback",
            PDF_OCR_PATH,
            extra_modules={},
        )

        subset_pages_by_path: dict[str, list[int]] = {}
        temp_paths: list[str] = []
        call_modes: list[tuple[str, str, int, list[list[int]]]] = []

        def _fake_build_subset(pdf_path, page_numbers):
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.close()
            temp_paths.append(handle.name)
            subset_pages_by_path[handle.name] = list(page_numbers)
            return handle.name, {idx + 1: page_no for idx, page_no in enumerate(page_numbers)}

        def _page_payload(task):
            return [
                {"page_no": idx + 1, "text": f"OCR {page_no}"}
                for idx, page_no in enumerate(task["page_numbers"])
            ]

        def _fake_execute(tasks, *, model_name, min_text_chars, device, max_workers, stage, progress_callback=None, progress_percent_fn=None, progress_meta_fn=None, batch_timeout_seconds=None, on_batch_completed=None):
            task_page_sets = [list(task["page_numbers"]) for task in tasks]
            mode = "parallel_gpu" if device == "gpu:0" and max_workers > 1 else "single_gpu" if device == "gpu:0" else "cpu"
            call_modes.append((mode, device, max_workers, task_page_sets))
            if mode == "parallel_gpu":
                on_batch_completed(tasks[0], _page_payload(tasks[0]), {})
                return {
                    "completed": [(tasks[0], _page_payload(tasks[0]))],
                    "remaining_tasks": tasks[1:],
                    "failure_reason": "gpu_oom",
                }
            if mode == "single_gpu":
                return {
                    "completed": [],
                    "remaining_tasks": tasks,
                    "failure_reason": "gpu_oom",
                }
            completed = [(task, _page_payload(task)) for task in tasks]
            for task, pages in completed:
                on_batch_completed(task, pages, {})
            return {
                "completed": completed,
                "remaining_tasks": [],
                "failure_reason": "",
            }

        pdf_module._build_pdf_subset_for_pages = _fake_build_subset
        pdf_module._execute_ocr_subset_tasks = _fake_execute
        pdf_module._recommended_parallel_ocr_workers = lambda **kwargs: 2

        runtime_info = {
            "ocr_device_attempted": "gpu:0",
            "ocr_device_effective": "gpu:0",
            "ocr_gpu_fallback_used": False,
            "ocr_gpu_failure_reason": "",
        }
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_PROGRESS_BATCH_PAGES": "2",
                        "PDF_OCR_GPU_FALLBACK_TO_CPU": "1",
                    },
                    clear=False,
                ):
                    pages = pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages(
                        tmp_pdf.name,
                        page_numbers=[1, 2, 3, 4],
                        model_name="mock-model",
                        min_text_chars=4,
                        device="gpu:0",
                        total_document_pages=4,
                        completed_pages_base=0,
                        runtime_info=runtime_info,
                    )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        self.assertEqual([mode for mode, _, _, _ in call_modes], ["parallel_gpu", "single_gpu", "cpu"])
        self.assertEqual([page["page_no"] for page in pages], [1, 2, 3, 4])
        self.assertTrue(runtime_info["ocr_gpu_fallback_used"])
        self.assertEqual(runtime_info["ocr_device_effective"], "cpu")
        self.assertEqual(runtime_info["ocr_retry_reason"], "gpu_oom")

    def test_configure_paddle_gpu_memory_env_applies_best_effort_limit(self):
        pdf_module = _load_module(
            "codex_test_pdf_gpu_memory_env",
            PDF_OCR_PATH,
            extra_modules={},
        )

        with mock.patch.dict(
            os.environ,
            {
                "PDF_OCR_DEVICE": "gpu:0",
                "PDF_OCR_GPU_BUDGET_GB": "10",
                "PDF_OCR_GPU_INITIAL_MEMORY_MB": "2048",
                "PDF_OCR_GPU_REALLOCATE_MEMORY_MB": "1024",
                "PDF_OCR_GPU_ALLOCATOR_STRATEGY": "auto_growth",
            },
            clear=False,
        ):
            updates = pdf_module._configure_paddle_gpu_memory_env(device="gpu:0")

        self.assertEqual(updates["FLAGS_allocator_strategy"], "auto_growth")
        self.assertEqual(updates["FLAGS_initial_gpu_memory_in_mb"], "2048")
        self.assertEqual(updates["FLAGS_reallocate_gpu_memory_in_mb"], "1024")
        self.assertEqual(updates["FLAGS_fraction_of_gpu_memory_to_use"], "0.3125")

    def test_resolve_model_candidates_uses_local_models_without_online_fallback_by_default(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_offline_candidates",
            PDF_OCR_PATH,
            extra_modules={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ocr_root = Path(tmp_dir) / "models" / "ocr"
            local_vl = ocr_root / "PaddleOCR-VL"
            local_vl.mkdir(parents=True)

            with mock.patch.dict(
                os.environ,
                {
                    "MAIN_BACKEND_HOME": tmp_dir,
                    "PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK": "0",
                },
                clear=False,
            ):
                candidates = pdf_module._resolve_model_candidates("PaddleOCR-VL")

        self.assertEqual(candidates, [str(local_vl)])
        self.assertNotIn("PaddleOCR-VL-1.5", candidates)
        self.assertNotIn("PaddleOCR-VL", candidates[1:])

    def test_build_paddleocrvl_kwargs_disables_internal_queue_by_default(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_disable_internal_queue",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakePaddleOCRVL:
            def __init__(
                self,
                pipeline_version=None,
                layout_detection_model_dir=None,
                vl_rec_model_dir=None,
                use_queues=None,
                device=None,
            ):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            kwargs = pdf_module._build_paddleocrvl_kwargs(
                paddleocrvl_cls=_FakePaddleOCRVL,
                candidate=tmp_dir,
                requested_model_name="PaddleOCR-VL",
                device="gpu:0",
            )

        self.assertEqual(kwargs["use_queues"], False)
        self.assertEqual(kwargs["device"], "gpu:0")

    def test_build_paddleocrvl_kwargs_applies_signature_safe_tuning_env(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_tuning_kwargs",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakePaddleOCRVL:
            def __init__(
                self,
                pipeline_version=None,
                use_chart_recognition=None,
                use_seal_recognition=None,
                use_ocr_for_image_block=None,
                max_new_tokens=None,
                min_pixels=None,
                max_pixels=None,
                layout_shape_mode=None,
                engine=None,
                vlm_extra_args=None,
                unsupported=None,
            ):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_USE_CHART_RECOGNITION": "0",
                    "PDF_OCR_USE_SEAL_RECOGNITION": "false",
                    "PDF_OCR_USE_OCR_FOR_IMAGE_BLOCK": "1",
                    "PDF_OCR_MAX_NEW_TOKENS": "768",
                    "PDF_OCR_MIN_PIXELS": "3136",
                    "PDF_OCR_MAX_PIXELS": "786432",
                    "PDF_OCR_LAYOUT_SHAPE_MODE": "rect",
                    "PDF_OCR_ENGINE": "paddle",
                    "PDF_OCR_VLM_EXTRA_ARGS_JSON": '{"ocr_max_pixels": 262144, "table_max_pixels": 786432}',
                },
                clear=False,
            ):
                kwargs = pdf_module._build_paddleocrvl_kwargs(
                    paddleocrvl_cls=_FakePaddleOCRVL,
                    candidate=tmp_dir,
                    requested_model_name="PaddleOCR-VL",
                    device="gpu:0",
                )

        self.assertEqual(kwargs["use_chart_recognition"], False)
        self.assertEqual(kwargs["use_seal_recognition"], False)
        self.assertEqual(kwargs["use_ocr_for_image_block"], True)
        self.assertEqual(kwargs["max_new_tokens"], 768)
        self.assertEqual(kwargs["min_pixels"], 3136)
        self.assertEqual(kwargs["max_pixels"], 786432)
        self.assertEqual(kwargs["layout_shape_mode"], "rect")
        self.assertEqual(kwargs["engine"], "paddle")
        self.assertEqual(kwargs["vlm_extra_args"], {"ocr_max_pixels": 262144, "table_max_pixels": 786432})

    def test_build_paddleocrvl_kwargs_applies_v100_fast_profile_defaults(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_v100_profile",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakePaddleOCRVL:
            def __init__(
                self,
                pipeline_version=None,
                use_queues=None,
                use_chart_recognition=None,
                use_seal_recognition=None,
                use_ocr_for_image_block=None,
                max_new_tokens=None,
                min_pixels=None,
                max_pixels=None,
                layout_shape_mode=None,
                vl_rec_max_concurrency=None,
                device=None,
            ):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {"PDF_OCR_OPTIMIZATION_PROFILE": "v100_32gb_fast"},
                clear=True,
            ):
                kwargs = pdf_module._build_paddleocrvl_kwargs(
                    paddleocrvl_cls=_FakePaddleOCRVL,
                    candidate=tmp_dir,
                    requested_model_name="PaddleOCR-VL-1.5",
                    device="gpu:0",
                )

        self.assertEqual(kwargs["use_queues"], True)
        self.assertEqual(kwargs["use_chart_recognition"], False)
        self.assertEqual(kwargs["use_seal_recognition"], False)
        self.assertEqual(kwargs["use_ocr_for_image_block"], False)
        self.assertEqual(kwargs["max_new_tokens"], 512)
        self.assertEqual(kwargs["min_pixels"], 3136)
        self.assertEqual(kwargs["max_pixels"], 589824)
        self.assertEqual(kwargs["layout_shape_mode"], "rect")
        self.assertEqual(kwargs["vl_rec_max_concurrency"], 1)
        self.assertEqual(kwargs["device"], "gpu:0")

    def test_build_paddleocrvl_kwargs_keeps_explicit_env_over_v100_fast_profile(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_v100_profile_override",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakePaddleOCRVL:
            def __init__(
                self,
                pipeline_version=None,
                use_queues=None,
                use_chart_recognition=None,
                max_new_tokens=None,
                max_pixels=None,
                vl_rec_max_concurrency=None,
            ):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_OPTIMIZATION_PROFILE": "v100_32gb_fast",
                    "PDF_OCR_USE_INTERNAL_QUEUES": "0",
                    "PDF_OCR_USE_CHART_RECOGNITION": "1",
                    "PDF_OCR_MAX_NEW_TOKENS": "768",
                    "PDF_OCR_MAX_PIXELS": "786432",
                    "PDF_OCR_VL_REC_MAX_CONCURRENCY": "2",
                },
                clear=True,
            ):
                kwargs = pdf_module._build_paddleocrvl_kwargs(
                    paddleocrvl_cls=_FakePaddleOCRVL,
                    candidate=tmp_dir,
                    requested_model_name="PaddleOCR-VL-1.5",
                    device="",
                )

        self.assertEqual(kwargs["use_queues"], False)
        self.assertEqual(kwargs["use_chart_recognition"], True)
        self.assertEqual(kwargs["max_new_tokens"], 768)
        self.assertEqual(kwargs["max_pixels"], 786432)
        self.assertEqual(kwargs["vl_rec_max_concurrency"], 2)

    def test_build_paddleocrvl_kwargs_ignores_tuning_env_when_signature_lacks_parameter(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_tuning_signature_safe",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _MinimalPaddleOCRVL:
            def __init__(self, pipeline_version=None, device=None):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_USE_CHART_RECOGNITION": "0",
                    "PDF_OCR_MAX_NEW_TOKENS": "768",
                    "PDF_OCR_VLM_EXTRA_ARGS_JSON": '{"ocr_max_pixels": 262144}',
                },
                clear=False,
            ):
                kwargs = pdf_module._build_paddleocrvl_kwargs(
                    paddleocrvl_cls=_MinimalPaddleOCRVL,
                    candidate=tmp_dir,
                    requested_model_name="PaddleOCR-VL",
                    device="gpu:0",
                )

        self.assertNotIn("use_chart_recognition", kwargs)
        self.assertNotIn("max_new_tokens", kwargs)
        self.assertNotIn("vlm_extra_args", kwargs)
        self.assertEqual(kwargs["device"], "gpu:0")

    def test_build_paddleocrvl_kwargs_ignores_invalid_vlm_extra_args_json(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_tuning_invalid_json",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakePaddleOCRVL:
            def __init__(self, pipeline_version=None, vlm_extra_args=None):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {"PDF_OCR_VLM_EXTRA_ARGS_JSON": "not-json"},
                clear=False,
            ):
                kwargs = pdf_module._build_paddleocrvl_kwargs(
                    paddleocrvl_cls=_FakePaddleOCRVL,
                    candidate=tmp_dir,
                    requested_model_name="PaddleOCR-VL",
                    device="",
                )

        self.assertNotIn("vlm_extra_args", kwargs)

    def test_execute_paddleocr_runtime_records_elapsed_throughput_and_target_status(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_runtime_throughput",
            PDF_OCR_PATH,
            extra_modules={},
        )

        pdf_module._count_pdf_pages = lambda path: 200
        pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages = lambda *args, **kwargs: [
            {"page_no": page_no, "text": f"OCR page {page_no}"}
            for page_no in range(1, 201)
        ]

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.object(pdf_module.time, "monotonic", side_effect=[100.0, 250.0]):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_DEVICE": "gpu:0",
                        "PDF_OCR_MAX_PAGES": "200",
                        "PDF_OCR_TARGET_PAGES": "200",
                        "PDF_OCR_TARGET_SECONDS": "300",
                    },
                    clear=False,
                ):
                    pages, runtime_info = pdf_module._execute_paddleocr_vl_with_runtime(tmp_pdf.name)

        self.assertEqual(len(pages), 200)
        self.assertEqual(runtime_info["ocr_elapsed_seconds"], 150.0)
        self.assertEqual(runtime_info["ocr_pages_processed"], 200)
        self.assertAlmostEqual(runtime_info["ocr_pages_per_minute"], 80.0)
        self.assertEqual(runtime_info["ocr_target_pages"], 200)
        self.assertEqual(runtime_info["ocr_target_seconds"], 300.0)
        self.assertTrue(runtime_info["ocr_target_met"])

    def test_execute_paddleocr_runtime_records_timing_breakdown(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_runtime_timing_breakdown",
            PDF_OCR_PATH,
            extra_modules={},
        )

        def _fake_selected(*args, **kwargs):
            runtime_info = kwargs.get("runtime_info")
            runtime_info["ocr_subset_build_seconds"] = 1.5
            runtime_info["ocr_batch_count"] = 2
            runtime_info["ocr_model_load_seconds"] = 3.0
            runtime_info["ocr_predict_seconds"] = 12.0
            return [{"page_no": 1, "text": "OCR page 1"}]

        pdf_module._count_pdf_pages = lambda path: 2
        pdf_module._extract_pdf_pages_with_paddleocr_vl_selected_pages = _fake_selected

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.object(pdf_module.time, "monotonic", side_effect=[100.0, 120.0]):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_OCR_DEVICE": "gpu:0",
                        "PDF_OCR_MAX_PAGES": "2",
                    },
                    clear=False,
                ):
                    pages, runtime_info = pdf_module._execute_paddleocr_vl_with_runtime(tmp_pdf.name)

        self.assertEqual(len(pages), 1)
        self.assertEqual(runtime_info["ocr_elapsed_seconds"], 20.0)
        self.assertEqual(runtime_info["ocr_subset_build_seconds"], 1.5)
        self.assertEqual(runtime_info["ocr_batch_count"], 2)
        self.assertEqual(runtime_info["ocr_model_load_seconds"], 3.0)
        self.assertEqual(runtime_info["ocr_predict_seconds"], 12.0)

    def test_gpu_ocr_error_detection_covers_oom_and_init_failures(self):
        pdf_module = _load_module(
            "codex_test_pdf_gpu_failure_detection",
            PDF_OCR_PATH,
            extra_modules={},
        )

        self.assertFalse(pdf_module._ocr_failure_needs_cpu_fallback(RuntimeError("CUDA out of memory")))
        with mock.patch.dict(os.environ, {"PDF_OCR_GPU_FALLBACK_TO_CPU": "1"}, clear=False):
            self.assertTrue(pdf_module._ocr_failure_needs_cpu_fallback(RuntimeError("CUDA out of memory")))
            self.assertTrue(
                pdf_module._ocr_failure_needs_cpu_fallback(
                    RuntimeError("ResourceExhaustedError: Out of memory error on GPU")
                )
            )
            self.assertTrue(pdf_module._ocr_failure_needs_cpu_fallback(RuntimeError("cannot init GPU device 0")))
        self.assertEqual(
            pdf_module._ocr_failure_reason(
                RuntimeError("CUDA error: no kernel image is available for execution on the device")
            ),
            "gpu_kernel_incompatible",
        )
        self.assertFalse(
            pdf_module._ocr_failure_needs_cpu_fallback(
                RuntimeError("CUDA error: no kernel image is available for execution on the device")
            )
        )
        self.assertFalse(pdf_module._ocr_failure_needs_cpu_fallback(RuntimeError("No model source is available")))

    def test_extract_pdf_pages_with_paddleocr_vl_retries_on_cpu_after_gpu_oom(self):
        pdf_module = _load_module(
            "codex_test_pdf_gpu_fallback",
            PDF_OCR_PATH,
            extra_modules={},
        )
        calls: list[str] = []

        def _fake_serial(pdf_path, progress_callback=None, *, model_name=None, max_pages=None, min_text_chars=None):
            calls.append(os.getenv("PDF_OCR_DEVICE", ""))
            if len(calls) == 1:
                raise RuntimeError("PaddleOCR-VL PDF 인식 실패: CUDA out of memory")
            return [{"page_no": 1, "text": "cpu fallback text"}]

        pdf_module._extract_pdf_pages_with_paddleocr_vl_serial = _fake_serial
        pdf_module._count_pdf_pages = lambda path: 1

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_DEVICE": "gpu:0",
                    "PDF_OCR_GPU_FALLBACK_TO_CPU": "1",
                    "PDF_OCR_MAX_PAGES": "8",
                    "PDF_OCR_MIN_TEXT_CHARS": "4",
                },
                clear=False,
            ):
                pages = pdf_module.extract_pdf_pages_with_paddleocr_vl(tmp_pdf.name)

        self.assertEqual(calls, ["gpu:0", "cpu"])
        self.assertEqual(pages, [{"page_no": 1, "text": "cpu fallback text"}])

    def test_extract_pdf_pages_with_paddleocr_vl_raises_when_gpu_fallback_disabled(self):
        pdf_module = _load_module(
            "codex_test_pdf_gpu_no_fallback",
            PDF_OCR_PATH,
            extra_modules={},
        )

        def _fake_serial(pdf_path, progress_callback=None, *, model_name=None, max_pages=None, min_text_chars=None):
            raise RuntimeError("PaddleOCR-VL PDF 인식 실패: CUDA out of memory")

        pdf_module._extract_pdf_pages_with_paddleocr_vl_serial = _fake_serial
        pdf_module._count_pdf_pages = lambda path: 1

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_DEVICE": "gpu:0",
                    "PDF_OCR_GPU_FALLBACK_TO_CPU": "0",
                    "PDF_OCR_MAX_PAGES": "8",
                    "PDF_OCR_MIN_TEXT_CHARS": "4",
                },
                clear=False,
            ):
                with self.assertRaises(RuntimeError):
                    pdf_module.extract_pdf_pages_with_paddleocr_vl(tmp_pdf.name)

    def test_extract_pdf_pages_prefers_pymupdf_text_when_all_pages_are_textual(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_text_only",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR fallback should not run when PyMuPDF text is sufficient")
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module(["첫 페이지 텍스트", "둘째 페이지 텍스트"])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "pymupdf_text")
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["text_pages"], 2)
        self.assertEqual(result["ocr_pages"], 0)
        self.assertEqual(result["failed_pages"], 0)
        self.assertEqual([page["parser"] for page in result["pages"]], ["pymupdf_text", "pymupdf_text"])

    def test_extract_pdf_pages_ocr_first_runs_paddleocr_before_pymupdf(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_first_order",
            PDF_OCR_PATH,
            extra_modules={},
        )
        calls: list[str] = []

        def _fake_ocr(*args, **kwargs):
            calls.append("ocr")
            return [{"page_no": 1, "text": "OCR text"}]

        def _fake_pymupdf(*args, **kwargs):
            calls.append("pymupdf")
            return {"pages": [{"page_no": 1, "text": "PyMuPDF text", "parser": "pymupdf_text"}], "total_pages": 1}

        pdf_module.extract_pdf_pages_with_paddleocr_vl = _fake_ocr
        pdf_module._extract_pdf_pages_with_pymupdf = _fake_pymupdf
        pdf_module._count_pdf_pages = lambda path: 1

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_PARSE_MODE": "ocr_first",
                    "PDF_OCR_MAX_PAGES": "200",
                },
                clear=False,
            ):
                result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(calls, ["ocr"])
        self.assertEqual(result["parser"], "paddleocr_vl")
        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual(result["text_pages"], 0)

    def test_extract_pdf_pages_ocr_first_falls_back_to_pymupdf_when_ocr_empty(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_first_empty_fallback",
            PDF_OCR_PATH,
            extra_modules={},
        )
        calls: list[str] = []

        def _fake_ocr(*args, **kwargs):
            calls.append("ocr")
            return []

        def _fake_pymupdf(*args, **kwargs):
            calls.append("pymupdf")
            return {"pages": [{"page_no": 1, "text": "PyMuPDF fallback text", "parser": "pymupdf_text"}], "total_pages": 1}

        pdf_module.extract_pdf_pages_with_paddleocr_vl = _fake_ocr
        pdf_module._extract_pdf_pages_with_pymupdf = _fake_pymupdf
        pdf_module._count_pdf_pages = lambda path: 1

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(os.environ, {"PDF_PARSE_MODE": "ocr_first"}, clear=False):
                result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(calls, ["ocr", "pymupdf"])
        self.assertEqual(result["parser"], "hybrid_pdf")
        self.assertEqual(result["ocr_pages"], 0)
        self.assertEqual(result["text_pages"], 1)
        self.assertIn("ocr_first_empty_fallback_to_pymupdf", result["warnings"])

    def test_extract_pdf_pages_ocr_first_raises_on_gpu_kernel_incompatible_by_default(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_first_kernel_incompatible",
            PDF_OCR_PATH,
            extra_modules={},
        )

        def _fake_ocr(*args, **kwargs):
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

        pdf_module.extract_pdf_pages_with_paddleocr_vl = _fake_ocr
        pdf_module._count_pdf_pages = lambda path: 1

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(os.environ, {"PDF_PARSE_MODE": "ocr_first"}, clear=False):
                with self.assertRaises(RuntimeError):
                    pdf_module.extract_pdf_pages(tmp_pdf.name)

    def test_extract_pdf_pages_ocr_first_uses_pymupdf_for_pages_beyond_ocr_limit(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_first_over_limit",
            PDF_OCR_PATH,
            extra_modules={},
        )

        def _fake_ocr(*args, **kwargs):
            self.assertEqual(kwargs.get("page_numbers"), [1, 2])
            return [
                {"page_no": 1, "text": "OCR page 1"},
                {"page_no": 2, "text": "OCR page 2"},
            ]

        def _fake_pymupdf(*args, **kwargs):
            return {
                "pages": [
                    {"page_no": 1, "text": "PyMuPDF page 1", "parser": "pymupdf_text"},
                    {"page_no": 2, "text": "PyMuPDF page 2", "parser": "pymupdf_text"},
                    {"page_no": 3, "text": "PyMuPDF page 3", "parser": "pymupdf_text"},
                ],
                "total_pages": 3,
            }

        pdf_module.extract_pdf_pages_with_paddleocr_vl = _fake_ocr
        pdf_module._extract_pdf_pages_with_pymupdf = _fake_pymupdf
        pdf_module._count_pdf_pages = lambda path: 3

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_PARSE_MODE": "ocr_first",
                    "PDF_OCR_MAX_PAGES": "2",
                },
                clear=False,
            ):
                result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "hybrid_pdf")
        self.assertEqual(result["ocr_pages"], 2)
        self.assertEqual(result["text_pages"], 1)
        self.assertEqual([(page["page_no"], page["parser"]) for page in result["pages"]], [(1, "paddleocr_vl"), (2, "paddleocr_vl"), (3, "pymupdf_text")])
        self.assertIn("ocr_first_pymupdf_for_pages_over_ocr_limit", result["warnings"])

    def test_default_ocr_limit_selects_first_400_pages(self):
        pdf_module = _load_module(
            "codex_test_pdf_default_400_page_limit",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module._count_pdf_pages = lambda path: 401

        with mock.patch.dict(os.environ, {}, clear=True):
            total_pages, selected_pages, has_pages_over_limit = pdf_module._selected_ocr_first_page_numbers(
                "/tmp/mock.pdf"
            )

        self.assertEqual(total_pages, 401)
        self.assertEqual(len(selected_pages), 400)
        self.assertEqual(selected_pages[0], 1)
        self.assertEqual(selected_pages[-1], 400)
        self.assertTrue(has_pages_over_limit)

    def test_extract_pdf_pages_adds_table_hints_for_numeric_table_like_text(self):
        pdf_module = _load_module(
            "codex_test_pdf_table_hints",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR fallback should not run when PyMuPDF text is sufficient")
        )

        table_text = "조사항목 단가\n농가경제조사 40 (천원)"
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module([table_text])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "pymupdf_text")
        self.assertTrue(result["pages"][0]["table_like"])
        self.assertIn("표행: 농가경제조사 40 (천원)", result["pages"][0]["table_hints"])
        self.assertIn("표값: 40천원", result["pages"][0]["table_hints"])

    def test_extract_pdf_pages_defers_upload_ocr_and_keeps_lazy_hints_by_default(self):
        pdf_module = _load_module(
            "codex_test_pdf_lazy_ocr_default",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Upload OCR should be deferred by default")
        )

        table_text = "조사명 단가 횟수 지급대상월\n경제활동인구조사 본조사 10 12 1 2 3"
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module([table_text, ""])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "hybrid_pdf")
        self.assertEqual(result["ocr_pages"], 0)
        self.assertEqual(result["failed_pages"], 0)
        self.assertIn("lazy_ocr_deferred", result["warnings"])
        self.assertEqual([page["page_no"] for page in result["pages"]], [1, 2])
        self.assertEqual(result["pages"][0]["parser"], "pymupdf_text")
        self.assertTrue(
            any(hint.startswith("OCR후보: 표형 페이지") for hint in result["pages"][0]["lazy_ocr_hints"])
        )
        self.assertEqual(result["pages"][1]["parser"], "pymupdf_hint")
        self.assertEqual(result["pages"][1]["text"], "")
        self.assertTrue(
            any("OCR후보: 텍스트 부족 페이지" in hint for hint in result["pages"][1]["lazy_ocr_hints"])
        )

    def test_extract_pdf_pages_uses_ocr_only_for_pages_without_text(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_partial_ocr",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: [
            {"page_no": 2, "text": "둘째 페이지 OCR 텍스트"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module(["첫 페이지 텍스트", ""])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                        "PDF_UPLOAD_OCR_ENABLED": "1",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "hybrid_pdf")
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["text_pages"], 1)
        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual(result["failed_pages"], 0)
        self.assertEqual(
            [(page["page_no"], page["parser"]) for page in result["pages"]],
            [(1, "pymupdf_text"), (2, "paddleocr_vl")],
        )

    def test_extract_pdf_pages_runs_ocr_for_table_like_text_pages(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_table_ocr",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: [
            {
                "page_no": 1,
                "text": "조사명 단가 횟수 지급대상월\n경제활동인구조사 본조사 10 12 1 2 3",
            }
        ]

        table_text = "조사명 단가 횟수 지급대상월\n경제활동인구조사 본조사 10 12 1 2 3"
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module([table_text])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                        "PDF_UPLOAD_OCR_ENABLED": "1",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual(result["text_pages"], 0)
        self.assertEqual(result["pages"][0]["parser"], "paddleocr_vl")

    def test_extract_pdf_pages_adds_compact_row_hints_for_flattened_table_text(self):
        pdf_module = _load_module(
            "codex_test_pdf_table_row_summaries",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR fallback should not run when validating table summary hints")
        )

        table_text = (
            "조사명 단가 횟수 지급대상월 "
            "경제활동인구조사 본조사 10 12 1 2 3 "
            "가계동향조사 본조사 80 12 1 2 3"
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module([table_text])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "text_only",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        summary_hints = [hint for hint in result["pages"][0]["table_hints"] if hint.startswith("표행요약:")]
        self.assertEqual(len(summary_hints), 2)
        self.assertTrue(any("경제활동인구조사 본조사 10 12" in hint for hint in summary_hints))
        self.assertTrue(any("가계동향조사 본조사 80 12" in hint for hint in summary_hints))

    def test_extract_pdf_pages_adds_row_summaries_for_multiline_schedule_table(self):
        pdf_module = _load_module(
            "codex_test_pdf_schedule_row_summaries",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR fallback should not run when validating multiline schedule summaries")
        )

        table_text = (
            "작물명 조사 시기 보고 기일 지급 기준월 지급 단가 지급 단위\n"
            "봄배추 4월 중순 ~ 6월 초순 6월 5일 4월 20천원 포구\n"
            "고구마 8월 중순 ~ 12월 초순 12월 4일 8월 20천원 필지"
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module([table_text])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "text_only",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        summary_hints = [hint for hint in result["pages"][0]["table_hints"] if hint.startswith("표행요약:")]
        self.assertTrue(any("봄배추" in hint and "지급 기준월" in hint and "20천원" in hint for hint in summary_hints))
        self.assertTrue(any("고구마" in hint and "12월 4일" in hint for hint in summary_hints))

    def test_extract_pdf_pages_uses_subset_pdf_for_partial_ocr_fallback(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_subset_ocr",
            PDF_OCR_PATH,
            extra_modules={},
        )
        ocr_paths: list[str] = []

        def _fake_ocr(path, **kwargs):
            selected_pages = list(kwargs.get("page_numbers", []) or [])
            ocr_paths.append((path, selected_pages))
            page_no = selected_pages[0] if selected_pages else 1
            return [{"page_no": page_no, "text": "둘째 페이지 OCR 텍스트"}]

        pdf_module.extract_pdf_pages_with_paddleocr_vl = _fake_ocr

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module(["첫 페이지 텍스트", ""])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                        "PDF_UPLOAD_OCR_ENABLED": "1",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(len(ocr_paths), 1)
        self.assertEqual(ocr_paths[0][0], tmp_pdf.name)
        self.assertEqual(ocr_paths[0][1], [2])
        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual([(page["page_no"], page["parser"]) for page in result["pages"]], [(1, "pymupdf_text"), (2, "paddleocr_vl")])

    def test_extract_pdf_pages_keeps_partial_text_result_when_ocr_fallback_fails(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_partial_success",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("PaddleOCR-VL PDF 인식 실패: timeout")
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": _stub_fitz_module(["첫 페이지 텍스트", ""])}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "pymupdf",
                        "PDF_TEXT_MIN_CHARS": "4",
                        "PDF_TEXT_MIN_NONSPACE_RATIO": "0.20",
                        "PDF_UPLOAD_OCR_ENABLED": "1",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "hybrid_pdf")
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["text_pages"], 1)
        self.assertEqual(result["ocr_pages"], 0)
        self.assertEqual(result["failed_pages"], 1)
        self.assertEqual(result["warnings"], ["ocr_fallback_failed"])
        self.assertEqual([(page["page_no"], page["parser"]) for page in result["pages"]], [(1, "pymupdf_text")])

    def test_extract_pdf_pages_with_paddleocr_vl_handles_generator_output(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_generator_output",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakeModel:
            def predict(self, **kwargs):
                def _iter():
                    yield {"page_no": 1, "text": "첫 페이지 OCR 텍스트"}
                    yield {"page_no": 2, "text": "둘째 페이지 OCR 텍스트"}

                return _iter()

        pdf_module._load_ocr_model = lambda model_name=None: _FakeModel()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_MODEL_NAME": "PaddleOCR-VL",
                    "PDF_OCR_MAX_PAGES": "10",
                },
                clear=False,
            ):
                pages = pdf_module.extract_pdf_pages_with_paddleocr_vl(tmp_pdf.name)

        self.assertEqual([page["page_no"] for page in pages], [1, 2])
        self.assertEqual([page["text"] for page in pages], ["첫 페이지 OCR 텍스트", "둘째 페이지 OCR 텍스트"])

    def test_extract_pdf_pages_with_paddleocr_vl_releases_gpu_model_after_success(self):
        pdf_module = _load_module(
            "codex_test_pdf_ocr_release_after_success",
            PDF_OCR_PATH,
            extra_modules={},
        )

        class _FakeModel:
            def predict(self, **kwargs):
                return [{"page_no": 1, "text": "GPU OCR 텍스트"}]

        reset_calls: list[str] = []
        original_reset = pdf_module._reset_cached_ocr_model
        pdf_module._load_ocr_model = lambda *args, **kwargs: _FakeModel()
        pdf_module._reset_cached_ocr_model = lambda: (reset_calls.append("reset"), original_reset())[1]

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(
                os.environ,
                {
                    "PDF_OCR_DEVICE": "gpu:0",
                    "PDF_OCR_MODEL_NAME": "PaddleOCR-VL",
                    "PDF_OCR_RELEASE_GPU_MODEL_AFTER_RUN": "1",
                },
                clear=False,
            ):
                pages = pdf_module.extract_pdf_pages_with_paddleocr_vl(tmp_pdf.name)

        self.assertEqual(pages, [{"page_no": 1, "text": "GPU OCR 텍스트"}])
        self.assertEqual(reset_calls, ["reset"])

    def test_extract_pdf_pages_can_disable_text_extractor(self):
        pdf_module = _load_module(
            "codex_test_pdf_hybrid_disabled_text_extractor",
            PDF_OCR_PATH,
            extra_modules={},
        )
        pdf_module.extract_pdf_pages_with_paddleocr_vl = lambda *args, **kwargs: [
            {"page_no": 1, "text": "OCR 전용 텍스트"},
        ]

        def _unexpected_fitz_open(_path=None):
            raise AssertionError("PyMuPDF should not be used when PDF_TEXT_EXTRACTOR=disabled")

        fitz_module = types.ModuleType("fitz")
        fitz_module.open = _unexpected_fitz_open

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            with mock.patch.dict(sys.modules, {"fitz": fitz_module}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PDF_PARSE_MODE": "hybrid",
                        "PDF_TEXT_EXTRACTOR": "disabled",
                    },
                    clear=False,
                ):
                    result = pdf_module.extract_pdf_pages(tmp_pdf.name)

        self.assertEqual(result["parser"], "paddleocr_vl")
        self.assertEqual(result["text_pages"], 0)
        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual(result["warnings"], [])

    def test_rag_ingest_file_uses_hybrid_pdf_parser_and_returns_page_stats(self):
        captured_doc_meta = {}
        release_calls: list[str] = []
        progress_events: list[tuple[int, str, str]] = []

        rag_module = _load_module(
            "codex_test_rag_hybrid_pdf",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {
                        "parser": "hybrid_pdf",
                        "pages": [
                            {"page_no": 1, "text": "첫 페이지 텍스트", "parser": "pymupdf_text"},
                            {"page_no": 4, "text": "넷째 페이지 OCR 텍스트", "parser": "paddleocr_vl"},
                        ],
                        "total_pages": 4,
                        "text_pages": 1,
                        "ocr_pages": 1,
                        "failed_pages": 0,
                        "warnings": [],
                        "ocr_elapsed_seconds": 150.0,
                        "ocr_pages_processed": 200,
                        "ocr_pages_per_minute": 80.0,
                        "ocr_target_pages": 200,
                        "ocr_target_seconds": 300.0,
                        "ocr_target_met": True,
                    },
                    release_cached_ocr_model=lambda *args, **kwargs: release_calls.append("released"),
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("rag.py should use hybrid pdf extractor")
                    ),
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda lines, **kwargs: [
                        {
                            "text": " ".join(str(line.get("text", "") or "") for line in lines),
                            "line_start": 1,
                            "line_end": len(lines),
                        }
                    ],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        class _FakeCursor:
            def __init__(self):
                self.lastrowid = 0

            def execute(self, *args, **kwargs):
                self.lastrowid += 1
                return self

        class _FakeConn:
            def __init__(self):
                self.cursor_obj = _FakeCursor()

            def cursor(self):
                return self.cursor_obj

            def commit(self):
                return None

            def close(self):
                return None

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine.txt_target_tokens = 640
        engine.txt_min_tokens = 420
        engine.txt_max_tokens = 900
        engine.txt_overlap_ratio = 0.25
        engine.txt_split_enabled = False
        engine.txt_split_trigger_lines = 0
        engine.txt_split_target_tokens = 0
        engine.txt_split_min_tokens = 0
        engine.txt_split_max_tokens = 0
        engine.xlsx_group_min_rows = 2
        engine.xlsx_group_max_rows = 8
        engine.xlsx_overlap_rows = 1
        engine.xlsx_target_tokens = 640
        engine.xlsx_max_tokens = 900
        engine.xlsx_merged_cell_policy = "fill"
        engine.xlsx_comment_policy = "inline"
        engine.pdf_target_tokens = 640
        engine.pdf_min_tokens = 420
        engine.pdf_max_tokens = 900
        engine.kb_id = "default"
        engine._engine_lock = threading.Lock()
        engine.query_cache = {}
        engine._normalize_doc_role = lambda role: role or "unknown"
        engine._compute_file_hash = lambda path: "abc123def456"
        engine._load_cached_payload = lambda **kwargs: None
        engine._save_cached_payload = lambda **kwargs: None
        engine._upsert_file_record = lambda **kwargs: 1
        engine._upsert_document_record = lambda **kwargs: captured_doc_meta.update(kwargs) or 2
        engine._replace_canonical_rows = lambda **kwargs: None
        engine._delete_chunks_for_source = lambda source_path: (0, [])
        engine._delete_source_upload_meta_for_source = lambda source_path: None
        engine._upsert_source_upload_meta = lambda **kwargs: None
        engine._connect_db = lambda: _FakeConn()
        engine._normalized_groups_for_source_type = lambda source_type: ["txt"]
        engine._refresh_normalized_chunks_and_index = lambda affected_groups=None: {
            "inserted_chunk_ids": [],
            "deleted_chunk_ids": [],
        }
        engine._sync_sqlite_search_artifacts = lambda **kwargs: None
        engine.hnsw_enabled = False
        engine.concept_links_enabled = False
        engine._count_normalized_chunks = lambda: 2
        engine._count_indexable_chunks = lambda: 2
        engine._parser_signature = lambda: "sig"
        engine._safe_json_dump = lambda payload: str(payload)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            result = rag_module.RAGEngine.ingest_file(
                engine,
                tmp_pdf.name,
                original_filename="sample.pdf",
                document_role="guide",
                progress_callback=lambda percent, message, stage: progress_events.append((percent, message, stage)),
            )

        self.assertEqual(captured_doc_meta["parser_name"], "hybrid_pdf")
        self.assertEqual(result["pdf_parser"], "hybrid_pdf")
        self.assertEqual(result["pdf_total_pages"], 4)
        self.assertEqual(result["pdf_text_pages"], 1)
        self.assertEqual(result["pdf_ocr_pages"], 1)
        self.assertEqual(result["pdf_failed_pages"], 0)
        self.assertEqual(result["ocr_pages_per_minute"], 80.0)
        self.assertTrue(result["ocr_target_met"])
        self.assertEqual(release_calls, ["released"])
        chunk_messages = [message for _, message, stage in progress_events if stage == "prepare_pdf_chunks"]
        self.assertTrue(any("(1/4)" in message for message in chunk_messages))
        self.assertTrue(any("(4/4)" in message for message in chunk_messages))

    def test_rag_cache_payload_preserves_pdf_stats(self):
        import numpy as real_numpy

        rag_module = _load_module(
            "codex_test_rag_cache_pdf_meta",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": real_numpy,
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *a, **k: {},
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
        cache_meta: dict[str, dict] = {}
        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = tmpdir
            engine.dim_large = 2
            engine._parser_signature = lambda: "sig"
            engine._get_file_cache_meta = lambda source_path: cache_meta.get(source_path)
            engine._get_file_cache_meta_by_hash = lambda file_hash: next(
                (meta for meta in cache_meta.values() if meta.get("file_hash") == file_hash),
                None,
            )
            engine._upsert_file_cache_meta = lambda **kwargs: cache_meta.__setitem__(
                kwargs["source_path"],
                {
                    "source_path": kwargs["source_path"],
                    "file_hash": kwargs["file_hash"],
                    "parser_sig": engine._parser_signature(),
                    "items_cache_path": kwargs["items_cache_path"],
                    "emb_large_cache_path": kwargs["emb_large_cache_path"],
                    "embeddings_cache_path": kwargs["emb_large_cache_path"],
                    "emb_small_cache_path": None,
                },
            )

            items = [{"text": "첫 페이지", "page_no": 1, "page_parser": "pymupdf_text"}]
            pdf_stats = {
                "pdf_parser": "hybrid_pdf",
                "pdf_total_pages": 2,
                "pdf_text_pages": 1,
                "pdf_ocr_pages": 0,
                "pdf_failed_pages": 1,
                "pdf_warnings": ["ocr_fallback_failed"],
            }
            rag_module.RAGEngine._save_cached_payload(
                engine,
                source_path="sample.pdf",
                file_hash="abc123def456",
                items=items,
                meta=pdf_stats,
            )
            loaded = rag_module.RAGEngine._load_cached_payload(
                engine,
                source_path="sample.pdf",
                file_hash="abc123def456",
            )

        self.assertIsNotNone(loaded)
        loaded_items, _emb, loaded_meta = loaded
        self.assertEqual(loaded_items[0]["text"], "첫 페이지")
        self.assertEqual(loaded_meta["pdf_failed_pages"], 1)
        self.assertEqual(loaded_meta["pdf_warnings"], ["ocr_fallback_failed"])

    def test_rag_numeric_query_boosts_table_row_summary_markers(self):
        rag_module = _load_module(
            "codex_test_rag_numeric_table_row_summary",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine._engine_lock = threading.Lock()
        engine.search_candidates = 4
        engine.lexical_weight = 0.0
        engine.hybrid_fts_weight = 0.0
        engine.concept_score_weight = 0.0
        engine.literal_match_boost = 0.0
        engine.recency_boost = 0.0
        engine.code_match_boost = 0.0
        engine.code_hint_boost_ratio = 0.0
        engine.normalized_score_penalty = 0.0
        engine._last_concept_search_meta = {}
        engine._normalize_index_name = lambda index_name: index_name
        engine._normalize_doc_roles_filter = lambda doc_roles: None
        engine._get_cached_query_result = lambda *args, **kwargs: None
        engine._resolve_index = lambda index_name: (object(), index_name)
        engine._encode_texts = lambda **kwargs: [[0.1, 0.2]]
        engine._extract_code_tokens = lambda query: []
        engine._is_code_or_class_query = lambda query: False
        engine._extract_query_literals = lambda query: []
        engine._search_concept_candidates = lambda **kwargs: {}
        engine._search_fts_candidates = lambda **kwargs: {}
        engine._search_text_candidates_sqlite = lambda **kwargs: [1, 2]
        engine._search_dense_candidates_hnsw = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._search_dense_candidates_sqlite = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._load_candidate_rows = lambda candidate_ids: [
            {
                "id": 2,
                "doc_role": "guide",
                "text": "일반 설명 텍스트",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
            {
                "id": 1,
                "doc_role": "guide",
                "text": "표행요약: 조사명 단가 횟수 지급대상월 | 가계동향조사 본조사 80 12 1 2 3",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
        ]
        engine._normalize_doc_role = lambda role: role or "guide"
        engine._infer_doc_role = lambda **kwargs: "guide"
        engine._lexical_overlap_score = lambda query, text: 0.0
        engine._literal_match_score = lambda text, query_literals: 0.0
        engine._recency_score = lambda uploaded_at, now_ts=None: 0.0
        engine._set_cached_query_result = lambda *args, **kwargs: None

        results = rag_module.RAGEngine.search(
            engine,
            "가계동향조사 단가 알려줘",
            top_k=2,
            index_name="large",
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["numeric_table_boost"], 0.0)
        self.assertEqual(results[1]["numeric_table_boost"], 0.0)

    def test_rag_schedule_query_boosts_table_row_summary_markers(self):
        rag_module = _load_module(
            "codex_test_rag_schedule_table_row_summary",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine._engine_lock = threading.Lock()
        engine.search_candidates = 4
        engine.lexical_weight = 0.0
        engine.hybrid_fts_weight = 0.0
        engine.concept_score_weight = 0.0
        engine.literal_match_boost = 0.0
        engine.recency_boost = 0.0
        engine.code_match_boost = 0.0
        engine.code_hint_boost_ratio = 0.0
        engine.normalized_score_penalty = 0.0
        engine._last_concept_search_meta = {}
        engine._normalize_index_name = lambda index_name: index_name
        engine._normalize_doc_roles_filter = lambda doc_roles: None
        engine._get_cached_query_result = lambda *args, **kwargs: None
        engine._resolve_index = lambda index_name: (object(), index_name)
        engine._encode_texts = lambda **kwargs: [[0.1, 0.2]]
        engine._extract_code_tokens = lambda query: []
        engine._is_code_or_class_query = lambda query: False
        engine._extract_query_literals = lambda query: []
        engine._search_concept_candidates = lambda **kwargs: {}
        engine._search_fts_candidates = lambda **kwargs: {}
        engine._search_text_candidates_sqlite = lambda **kwargs: [2, 1]
        engine._search_dense_candidates_hnsw = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._search_dense_candidates_sqlite = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._load_candidate_rows = lambda candidate_ids: [
            {
                "id": 2,
                "doc_role": "guide",
                "text": "일반 설명 텍스트",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
            {
                "id": 1,
                "doc_role": "guide",
                "text": "표행요약: 작물명 조사 시기 보고 기일 지급 기준월 지급 단가 지급 단위 | 봄배추 4월 중순 6월 5일 4월 20천원 포구",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
        ]
        engine._normalize_doc_role = lambda role: role or "guide"
        engine._infer_doc_role = lambda **kwargs: "guide"
        engine._lexical_overlap_score = lambda query, text: 0.0
        engine._literal_match_score = lambda text, query_literals: 0.0
        engine._recency_score = lambda uploaded_at, now_ts=None: 0.0
        engine._set_cached_query_result = lambda *args, **kwargs: None

        results = rag_module.RAGEngine.search(
            engine,
            "봄배추는 언제 지급하는지 알려줘",
            top_k=2,
            index_name="large",
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["numeric_table_boost"], 0.0)
        self.assertEqual(results[1]["numeric_table_boost"], 0.0)

    def test_rag_alias_query_prefers_hwpx_table_row_over_pdf_term_only_chunk(self):
        rag_module = _load_module(
            "codex_test_rag_hwpx_alias_table_row_summary",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine._engine_lock = threading.Lock()
        engine.search_candidates = 4
        engine.lexical_weight = 0.0
        engine.hybrid_fts_weight = 0.0
        engine.concept_score_weight = 0.0
        engine.literal_match_boost = 0.0
        engine.recency_boost = 0.0
        engine.code_match_boost = 0.0
        engine.code_hint_boost_ratio = 0.0
        engine.normalized_score_penalty = 0.0
        engine._last_concept_search_meta = {}
        engine._normalize_index_name = lambda index_name: index_name
        engine._normalize_doc_roles_filter = lambda doc_roles: None
        engine._get_cached_query_result = lambda *args, **kwargs: None
        engine._resolve_index = lambda index_name: (object(), index_name)
        engine._encode_texts = lambda **kwargs: [[0.1, 0.2]]
        engine._extract_code_tokens = lambda query: []
        engine._is_code_or_class_query = lambda query: False
        engine._extract_query_literals = lambda query: []
        engine._search_concept_candidates = lambda **kwargs: {}
        engine._search_fts_candidates = lambda **kwargs: {}
        engine._search_text_candidates_sqlite = lambda **kwargs: [2, 1]
        engine._search_dense_candidates_hnsw = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._search_dense_candidates_sqlite = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._load_candidate_rows = lambda candidate_ids: [
            {
                "id": 2,
                "doc_role": "guide",
                "text": "농가경제조사 교육 자료입니다. 답례품 단가는 이 문단에 없습니다.",
                "source_type": "pdf",
                "source_display": "guide.pdf",
                "source_path": "/tmp/guide.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
            {
                "id": 1,
                "doc_role": "guide",
                "text": "표행요약: 답례품=지류, 현금 | 조사명=농어가경제조사 | 명칭별칭=농어가경제조사, 농가경제조사, 어가경제조사 | 지급단가=40천원",
                "source_type": "hwpx",
                "source_display": "plan.hwpx",
                "source_path": "/tmp/plan.hwpx",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
        ]
        engine._normalize_doc_role = lambda role: role or "guide"
        engine._infer_doc_role = lambda **kwargs: "guide"
        engine._lexical_overlap_score = lambda query, text: 0.0
        engine._literal_match_score = lambda text, query_literals: 0.0
        engine._recency_score = lambda uploaded_at, now_ts=None: 0.0
        engine._set_cached_query_result = lambda *args, **kwargs: None

        results = rag_module.RAGEngine.search(
            engine,
            "농가경제조사 답례품 단가 알려줘",
            top_k=2,
            index_name="large",
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["numeric_table_boost"], results[1]["numeric_table_boost"])

    def test_rag_generic_alias_query_prefers_hwpx_table_row_over_pdf_term_only_chunk(self):
        rag_module = _load_module(
            "codex_test_rag_generic_hwpx_alias_table_row_summary",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine._engine_lock = threading.Lock()
        engine.search_candidates = 4
        engine.lexical_weight = 0.0
        engine.hybrid_fts_weight = 0.0
        engine.concept_score_weight = 0.0
        engine.literal_match_boost = 0.0
        engine.recency_boost = 0.0
        engine.code_match_boost = 0.0
        engine.code_hint_boost_ratio = 0.0
        engine.normalized_score_penalty = 0.0
        engine._last_concept_search_meta = {}
        engine._normalize_index_name = lambda index_name: index_name
        engine._normalize_doc_roles_filter = lambda doc_roles: None
        engine._get_cached_query_result = lambda *args, **kwargs: None
        engine._resolve_index = lambda index_name: (object(), index_name)
        engine._encode_texts = lambda **kwargs: [[0.1, 0.2]]
        engine._extract_code_tokens = lambda query: []
        engine._is_code_or_class_query = lambda query: False
        engine._extract_query_literals = lambda query: []
        engine._search_concept_candidates = lambda **kwargs: {}
        engine._search_fts_candidates = lambda **kwargs: {}
        engine._search_text_candidates_sqlite = lambda **kwargs: [2, 1]
        engine._search_dense_candidates_hnsw = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._search_dense_candidates_sqlite = lambda **kwargs: {1: 0.25, 2: 0.25}
        engine._load_candidate_rows = lambda candidate_ids: [
            {
                "id": 2,
                "doc_role": "guide",
                "text": "남학생건강조사 교육 자료입니다. 단가는 이 문단에 없습니다.",
                "source_type": "pdf",
                "source_display": "guide.pdf",
                "source_path": "/tmp/guide.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
            {
                "id": 1,
                "doc_role": "guide",
                "text": "표행요약: 조사명=남·여학생건강조사 | 명칭별칭=남·여학생건강조사, 남여학생건강조사, 남학생건강조사, 여학생건강조사 | 지급단가=30천원",
                "source_type": "hwpx",
                "source_display": "plan.hwpx",
                "source_path": "/tmp/plan.hwpx",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
        ]
        engine._normalize_doc_role = lambda role: role or "guide"
        engine._infer_doc_role = lambda **kwargs: "guide"
        engine._lexical_overlap_score = lambda query, text: 0.0
        engine._literal_match_score = lambda text, query_literals: 0.0
        engine._recency_score = lambda uploaded_at, now_ts=None: 0.0
        engine._set_cached_query_result = lambda *args, **kwargs: None

        results = rag_module.RAGEngine.search(
            engine,
            "남학생건강조사 단가 알려줘",
            top_k=2,
            index_name="large",
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["alias_match_boost"], 0.0)

    def test_rag_search_penalizes_weak_ocr_hint_chunks_when_scores_tie(self):
        rag_module = _load_module(
            "codex_test_rag_weak_ocr_hint_penalty",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        engine._engine_lock = threading.Lock()
        engine.search_candidates = 4
        engine.lexical_weight = 0.0
        engine.hybrid_fts_weight = 0.0
        engine.concept_score_weight = 0.0
        engine.literal_match_boost = 0.0
        engine.recency_boost = 0.0
        engine.code_match_boost = 0.0
        engine.code_hint_boost_ratio = 0.0
        engine.normalized_score_penalty = 0.0
        engine._last_concept_search_meta = {}
        engine._normalize_index_name = lambda index_name: index_name
        engine._normalize_doc_roles_filter = lambda doc_roles: None
        engine._get_cached_query_result = lambda *args, **kwargs: None
        engine._resolve_index = lambda index_name: (object(), index_name)
        engine._encode_texts = lambda **kwargs: [[0.1, 0.2]]
        engine._extract_code_tokens = lambda query: []
        engine._is_code_or_class_query = lambda query: False
        engine._extract_query_literals = lambda query: []
        engine._search_concept_candidates = lambda **kwargs: {}
        engine._search_fts_candidates = lambda **kwargs: {}
        engine._search_text_candidates_sqlite = lambda **kwargs: [1, 2]
        engine._search_dense_candidates_hnsw = lambda **kwargs: {1: 0.0, 2: 0.0}
        engine._search_dense_candidates_sqlite = lambda **kwargs: {1: 0.0, 2: 0.0}
        engine._load_candidate_rows = lambda candidate_ids: [
            {
                "id": 2,
                "doc_role": "guide",
                "text": "OCR후보: 텍스트 부족 페이지 | 페이지 160 | 원문 확인 필요",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
            {
                "id": 1,
                "doc_role": "guide",
                "text": "답례품 관련 안내 문장은 문서에 없음",
                "source_type": "pdf",
                "source_display": "sample.pdf",
                "source_path": "/tmp/sample.pdf",
                "uploaded_at": 0,
                "source_updated_at": 0,
                "is_normalized": 0,
            },
        ]
        engine._normalize_doc_role = lambda role: role or "guide"
        engine._infer_doc_role = lambda **kwargs: "guide"
        engine._lexical_overlap_score = lambda query, text: 0.0
        engine._literal_match_score = lambda text, query_literals: 0.0
        engine._recency_score = lambda uploaded_at, now_ts=None: 0.0
        engine._set_cached_query_result = lambda *args, **kwargs: None

        results = rag_module.RAGEngine.search(
            engine,
            "답례품 단가 알려줘",
            top_k=2,
            index_name="large",
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertFalse(results[0]["weak_ocr_hint"])
        self.assertTrue(results[1]["weak_ocr_hint"])
        self.assertLess(results[1]["score"], results[0]["score"])

    def test_rag_lazy_pdf_page_text_uses_file_cache_when_enabled(self):
        ocr_calls: list[tuple[str, tuple[int, ...]]] = []
        release_calls: list[str] = []

        rag_module = _load_module(
            "codex_test_rag_lazy_pdf_cache",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: release_calls.append("released"),
                    extract_pdf_pages_with_paddleocr_vl=lambda path, **kwargs: (
                        ocr_calls.append((path, tuple(kwargs.get("page_numbers", []) or ()))),
                        [{"page_no": 3, "text": "셋째 페이지 OCR 텍스트"}],
                    )[1],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = tmpdir
            engine._parser_signature = lambda: "sig"
            engine._resolve_lazy_ocr_source_file = lambda source_path: {
                "stored_path": "/tmp/sample.pdf",
                "file_hash": "abc123def456",
            }

            with mock.patch.dict(
                os.environ,
                {
                    "PDF_LAZY_OCR_CACHE_ENABLED": "1",
                    "PDF_ANSWER_PATH_LAZY_OCR_ENABLED": "1",
                },
                clear=False,
            ):
                first = rag_module.RAGEngine.get_lazy_pdf_page_text(engine, "sample.pdf", 3)
                second = rag_module.RAGEngine.get_lazy_pdf_page_text(engine, "sample.pdf", 3)

        self.assertEqual(first, "셋째 페이지 OCR 텍스트")
        self.assertEqual(second, "셋째 페이지 OCR 텍스트")
        self.assertEqual(ocr_calls, [("/tmp/sample.pdf", (3,))])
        self.assertEqual(release_calls, ["released"])

    def test_rag_answer_path_lazy_pdf_ocr_uses_cache_without_running_ocr(self):
        ocr_calls: list[tuple[str, tuple[int, ...]]] = []

        rag_module = _load_module(
            "codex_test_rag_answer_path_lazy_pdf_cache_only",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=lambda *args, **kwargs: {},
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                    extract_pdf_pages_with_paddleocr_vl=lambda path, **kwargs: (
                        ocr_calls.append((path, tuple(kwargs.get("page_numbers", []) or ()))),
                        [{"page_no": 3, "text": "셋째 페이지 OCR 텍스트"}],
                    )[1],
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda *args, **kwargs: [],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        engine = rag_module.RAGEngine.__new__(rag_module.RAGEngine)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = tmpdir
            engine._parser_signature = lambda: "sig"
            engine._resolve_lazy_ocr_source_file = lambda source_path: {
                "stored_path": "/tmp/sample.pdf",
                "file_hash": "abc123def456",
            }
            engine._save_lazy_ocr_cache_text("sample.pdf", "abc123def456", 3, "캐시된 OCR 텍스트")

            with mock.patch.dict(
                os.environ,
                {
                    "PDF_LAZY_OCR_CACHE_ENABLED": "1",
                    "PDF_ANSWER_PATH_LAZY_OCR_ENABLED": "0",
                },
                clear=False,
            ):
                cached = rag_module.RAGEngine.get_lazy_pdf_page_text(engine, "sample.pdf", 3)

            with mock.patch.dict(
                os.environ,
                {
                    "PDF_LAZY_OCR_CACHE_ENABLED": "0",
                    "PDF_ANSWER_PATH_LAZY_OCR_ENABLED": "0",
                },
                clear=False,
            ):
                missing = rag_module.RAGEngine.get_lazy_pdf_page_text(engine, "sample.pdf", 4)

        self.assertEqual(cached, "캐시된 OCR 텍스트")
        self.assertEqual(missing, "")
        self.assertEqual(ocr_calls, [])

    def test_gpu_parallel_worker_recommendation_honors_one_page_minimum_for_h100_probe(self):
        pdf_module = _load_module(
            "codex_test_pdf_h100_parallel_min_pages",
            PDF_OCR_PATH,
            extra_modules={},
        )

        with mock.patch.dict(
            os.environ,
            {
                "PDF_OCR_PARALLEL_MAX_WORKERS": "3",
                "PDF_OCR_PARALLEL_MIN_PAGES": "1",
                "PDF_OCR_PARALLEL_CPU_PER_WORKER": "1",
                "PDF_OCR_PARALLEL_GPU_MEM_GB_PER_WORKER": "8",
                "PDF_OCR_PARALLEL_GPU_MEM_RESERVE_GB": "8",
            },
            clear=False,
        ):
            workers = pdf_module._recommended_parallel_ocr_workers(
                "gpu:0",
                candidate_page_count=3,
                cpu_count=32,
                available_bytes=70 * 1024 * 1024 * 1024,
            )

        self.assertEqual(workers, 3)


class PdfTwoPhaseIngestTests(unittest.TestCase):
    def _load_rag_module(self, events: list[str]):
        def _extract_pdf_pages(path, progress_callback=None, **kwargs):
            events.append("ocr")
            if progress_callback:
                progress_callback(
                    50,
                    "PDF OCR completed",
                    "run_pdf_ocr",
                    current_page=1,
                    total_pages=1,
                    ocr_completed_pages=1,
                    ocr_target_pages=1,
                )
            return {
                "parser": "paddleocr_vl",
                "pages": [{"page_no": 1, "text": "첫 페이지 OCR 텍스트", "parser": "paddleocr_vl"}],
                "total_pages": 1,
                "text_pages": 0,
                "ocr_pages": 1,
                "attempted_ocr_pages": 1,
                "failed_pages": 0,
            }

        def _chunk_txt_items(lines, **kwargs):
            text = " ".join(str(line.get("text", "") if isinstance(line, dict) else line).strip() for line in lines)
            return [
                {
                    "text": text or "empty",
                    "line_start": 1,
                    "line_end": max(1, len(lines)),
                    "section": "",
                }
            ]

        return _load_module(
            "codex_test_rag_two_phase_ingest",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=_extract_pdf_pages,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=_chunk_txt_items,
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

    def _new_engine(self, rag_module, tmpdir: str):
        return rag_module.RAGEngine(kb_id="two_phase_prepare", data_dir=tmpdir)

    def _chunk_count(self, engine) -> int:
        conn = engine._connect_db()
        try:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source_path = ? AND COALESCE(is_normalized, 0) = 0",
                    ("sample.pdf",),
                ).fetchone()[0]
            )
        finally:
            conn.close()

    def test_prepare_ingest_payload_finishes_pdf_ocr_without_indexing_or_db_chunks(self):
        events: list[str] = []
        rag_module = self._load_rag_module(events)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.dict(
                os.environ,
                {
                    "RAG_SQLITE_DENSE_ENABLED": "0",
                    "RAG_HNSW_ENABLED": "0",
                    "RAG_CONCEPT_LINKS_ENABLED": "0",
                    "DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED": "0",
                    "WIKI_PAGE_WORKFLOW_ENABLED": "0",
                },
                clear=False,
            ):
                engine = self._new_engine(rag_module, tmpdir)
                engine._sync_sqlite_search_artifacts = lambda **kwargs: events.append("index")

                prepared = engine.prepare_ingest_payload(str(pdf_path), original_filename="sample.pdf")

            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(prepared["source_type"], "pdf")
            self.assertEqual(len(prepared["items"]), 1)
            self.assertIn("ocr", events)
            self.assertNotIn("index", events)
            self.assertEqual(self._chunk_count(engine), 0)

    def test_commit_prepared_ingest_runs_indexing_after_ocr_payload_is_ready(self):
        events: list[str] = []
        rag_module = self._load_rag_module(events)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.dict(
                os.environ,
                {
                    "RAG_SQLITE_DENSE_ENABLED": "0",
                    "RAG_HNSW_ENABLED": "0",
                    "RAG_CONCEPT_LINKS_ENABLED": "0",
                    "DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED": "0",
                    "WIKI_PAGE_WORKFLOW_ENABLED": "0",
                },
                clear=False,
            ):
                engine = self._new_engine(rag_module, tmpdir)
                engine._sync_sqlite_search_artifacts = lambda **kwargs: events.append("index")

                prepared = engine.prepare_ingest_payload(str(pdf_path), original_filename="sample.pdf")
                result = engine.commit_prepared_ingest(prepared)

            self.assertEqual(result["status"], "ok")
            self.assertLess(events.index("ocr"), events.index("index"))
            self.assertEqual(self._chunk_count(engine), 1)

    def test_cached_pdf_prepare_skips_ocr_but_commit_refreshes_index(self):
        events: list[str] = []
        rag_module = self._load_rag_module(events)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.dict(
                os.environ,
                {
                    "RAG_SQLITE_DENSE_ENABLED": "0",
                    "RAG_HNSW_ENABLED": "0",
                    "RAG_CONCEPT_LINKS_ENABLED": "0",
                    "DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED": "0",
                    "WIKI_PAGE_WORKFLOW_ENABLED": "0",
                },
                clear=False,
            ):
                engine = self._new_engine(rag_module, tmpdir)
                engine._sync_sqlite_search_artifacts = lambda **kwargs: events.append("index")

                first = engine.prepare_ingest_payload(str(pdf_path), original_filename="sample.pdf")
                engine.commit_prepared_ingest(first)
                events.clear()

                second = engine.prepare_ingest_payload(str(pdf_path), original_filename="sample.pdf")
                result = engine.commit_prepared_ingest(second)

            self.assertEqual(result["status"], "ok")
        self.assertTrue(second["used_cache"])
        self.assertNotIn("ocr", events)
        self.assertIn("index", events)

    def test_prepare_ingest_payload_does_not_hold_engine_lock_during_pdf_ocr(self):
        events: list[str] = []
        lock_state = {"held": False}

        class TrackingLock:
            def __enter__(self):
                lock_state["held"] = True
                return self

            def __exit__(self, exc_type, exc, tb):
                lock_state["held"] = False
                return False

        def _extract_pdf_pages(path, progress_callback=None, **kwargs):
            events.append("ocr")
            self.assertFalse(lock_state["held"], "PDF OCR should run outside RAGEngine._engine_lock")
            return {
                "parser": "paddleocr_vl",
                "pages": [{"page_no": 1, "text": "OCR text", "parser": "paddleocr_vl"}],
                "total_pages": 1,
                "text_pages": 0,
                "ocr_pages": 1,
                "attempted_ocr_pages": 1,
                "failed_pages": 0,
            }

        rag_module = _load_module(
            "codex_test_rag_two_phase_lock_scope",
            RAG_PATH,
            extra_modules={
                "hnswlib": types.SimpleNamespace(Index=object),
                "numpy": _stub_numpy_module(),
                "requests": types.ModuleType("requests"),
                "sentence_transformers": _stub_sentence_transformers_module(),
                "src.pdf_ocr": types.SimpleNamespace(
                    extract_pdf_pages=_extract_pdf_pages,
                    extract_pdf_pages_with_paddleocr_vl=lambda *args, **kwargs: [],
                    release_cached_ocr_model=lambda *args, **kwargs: None,
                ),
                "src.utils": types.SimpleNamespace(
                    chunk_txt_items=lambda lines, **kwargs: [
                        {
                            "text": " ".join(str(line.get("text", "") if isinstance(line, dict) else line) for line in lines),
                            "line_start": 1,
                            "line_end": len(lines),
                        }
                    ],
                    chunk_xlsx_rows=lambda *args, **kwargs: [],
                    load_txt=lambda *args, **kwargs: [],
                    load_xlsx=lambda *args, **kwargs: [],
                ),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.dict(
                os.environ,
                {
                    "RAG_SQLITE_DENSE_ENABLED": "0",
                    "RAG_HNSW_ENABLED": "0",
                    "RAG_CONCEPT_LINKS_ENABLED": "0",
                    "DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED": "0",
                    "WIKI_PAGE_WORKFLOW_ENABLED": "0",
                },
                clear=False,
            ):
                engine = self._new_engine(rag_module, tmpdir)
                engine._engine_lock = TrackingLock()

                prepared = engine.prepare_ingest_payload(str(pdf_path), original_filename="sample.pdf")

        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(events, ["ocr"])

    def test_ingest_result_includes_phase_timing(self):
        events: list[str] = []
        rag_module = self._load_rag_module(events)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.dict(
                os.environ,
                {
                    "RAG_SQLITE_DENSE_ENABLED": "0",
                    "RAG_HNSW_ENABLED": "0",
                    "RAG_CONCEPT_LINKS_ENABLED": "0",
                    "DOCUMENT_UPLOAD_ONTOLOGY_JOB_ENABLED": "0",
                    "WIKI_PAGE_WORKFLOW_ENABLED": "0",
                },
                clear=False,
            ):
                engine = self._new_engine(rag_module, tmpdir)
                engine._sync_sqlite_search_artifacts = lambda **kwargs: events.append("index")
                result = engine.ingest_file(str(pdf_path), original_filename="sample.pdf")

        self.assertEqual(result["status"], "ok")
        for key in (
            "ocr_duration_seconds",
            "persist_duration_seconds",
            "embedding_duration_seconds",
            "index_duration_seconds",
            "derived_sync_duration_seconds",
        ):
            self.assertIn(key, result)
            self.assertGreaterEqual(float(result[key]), 0.0)


if __name__ == "__main__":
    unittest.main()
