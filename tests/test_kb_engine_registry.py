import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.kb_engine_registry import KBEngineRegistry


class KBEngineLeaseTests(unittest.TestCase):
    @staticmethod
    def _factory(kb_name: str):
        return SimpleNamespace(kb_id=kb_name, close=MagicMock())

    def test_lease_prevents_eviction_until_request_finishes(self):
        registry = KBEngineRegistry(max_loaded_kbs=1, idle_ttl_seconds=3600)

        with registry.lease("test1", self._factory) as first:
            second = registry.get_or_create("test2", self._factory)
            self.assertFalse(first.close.called)
            self.assertFalse(second.close.called)
            self.assertEqual(registry.snapshot_count(), 2)

        first.close.assert_called_once()
        self.assertEqual(list(registry._instances.keys()), ["test2"])
        self.assertFalse(second.close.called)

    def test_remove_defers_close_for_a_leased_engine(self):
        registry = KBEngineRegistry(max_loaded_kbs=1, idle_ttl_seconds=3600)

        with registry.lease("test1", self._factory) as first:
            registry.remove("test1")
            self.assertFalse(first.close.called)
            replacement = registry.get_or_create("test1", self._factory)
            self.assertIsNot(first, replacement)

        first.close.assert_called_once()
        self.assertFalse(replacement.close.called)


if __name__ == "__main__":
    unittest.main()
