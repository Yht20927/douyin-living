# -*- coding: utf-8 -*-
"""Unit tests for src.modelPool — focuses on LRU semantics and refcounting."""

import threading
import pytest

from src.modelPool import ModelPool


@pytest.fixture(autouse=True)
def _reset_pool():
    """Each test sees a clean singleton."""
    ModelPool.reset()
    yield
    ModelPool.reset()


def _make_factory(records: list, name: str):
    """Returns a factory that records each call so tests can assert
    how many times a model was actually instantiated.
    """
    def factory():
        records.append(name)
        return {"name": name}
    return factory


class TestSingleton:
    def test_instance_returns_same_object(self):
        a = ModelPool.instance()
        b = ModelPool.instance()
        assert a is b

    def test_reset_drops_singleton(self):
        a = ModelPool.instance()
        ModelPool.reset()
        b = ModelPool.instance()
        assert a is not b


class TestAcquireRelease:
    def test_factory_called_once_on_hit(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        f = _make_factory(records, "m")
        pool.acquire("m", f, gpu=True)
        pool.acquire("m", f, gpu=True)
        pool.acquire("m", f, gpu=True)
        assert records == ["m"]  # only first call hit factory

    def test_release_drops_refcount(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        pool.acquire("m", _make_factory(records, "m"), gpu=True)
        pool.acquire("m", _make_factory(records, "m"), gpu=True)
        pool.release("m")
        # Still in cache
        assert "m" in pool.resident

    def test_evict_refuses_in_use(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        pool.acquire("m", _make_factory(records, "m"))
        # refcount = 1 → eviction must refuse
        assert pool.evict("m") is False
        assert "m" in pool.resident

    def test_evict_succeeds_after_release(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        pool.acquire("m", _make_factory(records, "m"))
        pool.release("m")
        assert pool.evict("m") is True
        assert "m" not in pool.resident


class TestLRU:
    def test_oldest_evicted_first(self):
        pool = ModelPool(gpu_limit=2)
        records: list[str] = []
        # Load and release each so they are evictable
        for n in ("a", "b"):
            pool.acquire(n, _make_factory(records, n))
            pool.release(n)
        # Loading "c" forces eviction of LRU = "a"
        pool.acquire("c", _make_factory(records, "c"))
        assert "a" not in pool.resident
        assert set(pool.resident) == {"b", "c"}

    def test_acquire_marks_mru(self):
        pool = ModelPool(gpu_limit=2)
        records: list[str] = []
        for n in ("a", "b"):
            pool.acquire(n, _make_factory(records, n))
            pool.release(n)
        # Touch "a" so it becomes MRU
        pool.acquire("a", _make_factory(records, "a"))
        pool.release("a")
        # Now loading "c" should evict "b" instead
        pool.acquire("c", _make_factory(records, "c"))
        assert "b" not in pool.resident
        assert "a" in pool.resident

    def test_in_use_skipped_during_eviction(self):
        pool = ModelPool(gpu_limit=2)
        records: list[str] = []
        # "a" stays acquired (refcount=1) — must NOT be evicted
        pool.acquire("a", _make_factory(records, "a"))
        pool.acquire("b", _make_factory(records, "b"))
        pool.release("b")
        # Loading "c" — "a" is in-use, "b" is the only valid victim
        pool.acquire("c", _make_factory(records, "c"))
        assert "a" in pool.resident
        assert "b" not in pool.resident
        assert "c" in pool.resident

    def test_cpu_models_dont_count(self):
        pool = ModelPool(gpu_limit=1)
        records: list[str] = []
        # Three CPU-only models — no eviction even though gpu_limit=1
        for n in ("a", "b", "c"):
            pool.acquire(n, _make_factory(records, n), gpu=False)
            pool.release(n)
        assert pool.gpu_count == 0
        assert set(pool.resident) == {"a", "b", "c"}


class TestScope:
    def test_context_manager_releases(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        with pool.scope("m", _make_factory(records, "m")) as model:
            assert model == {"name": "m"}
        # After exit — model still cached but refcount=0 → evictable
        assert pool.evict("m") is True

    def test_scope_releases_on_exception(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []
        with pytest.raises(RuntimeError):
            with pool.scope("m", _make_factory(records, "m")):
                raise RuntimeError("boom")
        # Refcount should be 0 even though scope raised
        assert pool.evict("m") is True


class TestThreadSafety:
    def test_concurrent_acquire_loads_once(self):
        pool = ModelPool(gpu_limit=4)
        records: list[str] = []

        def worker():
            for _ in range(20):
                pool.acquire("shared", _make_factory(records, "shared"))
                pool.release("shared")

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 100 acquires across 5 threads → factory still only called ONCE
        assert records.count("shared") == 1
