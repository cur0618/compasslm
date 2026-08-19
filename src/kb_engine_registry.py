import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional, TypeVar


EngineT = TypeVar("EngineT")


class KBEngineRegistry:
    def __init__(self, max_loaded_kbs: int = 1, idle_ttl_seconds: int = 900):
        self.max_loaded_kbs = max(1, int(max_loaded_kbs or 1))
        self.idle_ttl_seconds = max(60, int(idle_ttl_seconds or 900))
        self._lock = threading.RLock()
        self._instances: "OrderedDict[str, EngineT]" = OrderedDict()
        self._last_access: Dict[str, float] = {}
        self._lease_counts: Dict[int, int] = {}
        self._retired_engines: Dict[int, EngineT] = {}

    def get(self, kb_name: str) -> Optional[EngineT]:
        with self._lock:
            engine = self._instances.get(kb_name)
            if engine is None:
                return None
            self._touch_locked(kb_name)
            return engine

    def get_or_create(self, kb_name: str, factory: Callable[[str], EngineT]) -> EngineT:
        with self._lock:
            existing = self._instances.get(kb_name)
            if existing is not None:
                self._touch_locked(kb_name)
                self._prune_locked(active_kb=kb_name)
                return existing

        created = factory(kb_name)

        with self._lock:
            existing = self._instances.get(kb_name)
            if existing is not None:
                self._close_engine(created)
                self._touch_locked(kb_name)
                self._prune_locked(active_kb=kb_name)
                return existing
            self._instances[kb_name] = created
            self._touch_locked(kb_name)
            self._prune_locked(active_kb=kb_name)
            return created

    def _acquire(self, kb_name: str, factory: Callable[[str], EngineT]) -> EngineT:
        with self._lock:
            existing = self._instances.get(kb_name)
            if existing is not None:
                self._lease_counts[id(existing)] = self._lease_counts.get(id(existing), 0) + 1
                self._touch_locked(kb_name)
                self._prune_locked(active_kb=kb_name)
                return existing

        created = factory(kb_name)
        close_created = False
        with self._lock:
            existing = self._instances.get(kb_name)
            if existing is not None:
                engine = existing
                close_created = True
            else:
                engine = created
                self._instances[kb_name] = engine
            self._lease_counts[id(engine)] = self._lease_counts.get(id(engine), 0) + 1
            self._touch_locked(kb_name)
            self._prune_locked(active_kb=kb_name)
        if close_created:
            self._close_engine(created)
        return engine

    @contextmanager
    def lease(
        self,
        kb_name: str,
        factory: Callable[[str], EngineT],
    ) -> Iterator[EngineT]:
        engine = self._acquire(kb_name, factory)
        try:
            yield engine
        finally:
            self.release(engine)

    def release(self, engine: EngineT) -> None:
        close_retired: Optional[EngineT] = None
        engine_key = id(engine)
        with self._lock:
            count = int(self._lease_counts.get(engine_key, 0) or 0)
            if count <= 1:
                self._lease_counts.pop(engine_key, None)
                close_retired = self._retired_engines.pop(engine_key, None)
            else:
                self._lease_counts[engine_key] = count - 1
            self._prune_locked()
        self._close_engine(close_retired)

    def remove(self, kb_name: str):
        close_now: Optional[EngineT] = None
        with self._lock:
            engine = self._instances.pop(kb_name, None)
            self._last_access.pop(kb_name, None)
            if engine is not None and self._lease_counts.get(id(engine), 0) > 0:
                self._retired_engines[id(engine)] = engine
            else:
                close_now = engine
        self._close_engine(close_now)

    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._instances)

    def _touch_locked(self, kb_name: str):
        self._last_access[kb_name] = time.time()
        if kb_name in self._instances:
            self._instances.move_to_end(kb_name)

    def _prune_locked(self, active_kb: Optional[str] = None):
        now = time.time()
        idle_expired = [
            kb_name
            for kb_name, last_access in self._last_access.items()
            if (
                kb_name != active_kb
                and self._lease_counts.get(id(self._instances.get(kb_name)), 0) <= 0
                and (now - float(last_access)) > self.idle_ttl_seconds
            )
        ]
        for kb_name in idle_expired:
            engine = self._instances.pop(kb_name, None)
            self._last_access.pop(kb_name, None)
            self._close_engine(engine)

        while len(self._instances) > self.max_loaded_kbs:
            oldest_kb = next(
                (
                    kb_name
                    for kb_name, engine in self._instances.items()
                    if kb_name != active_kb and self._lease_counts.get(id(engine), 0) <= 0
                ),
                None,
            )
            if oldest_kb is None:
                break
            engine = self._instances.pop(oldest_kb, None)
            self._last_access.pop(oldest_kb, None)
            self._close_engine(engine)

    def _close_engine(self, engine: Optional[EngineT]):
        if engine is None:
            return
        closer = getattr(engine, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
