# -*- coding: utf-8 -*-
"""ModelPool — singleton GPU/CPU model cache with LRU eviction.

Why this exists
---------------
The signal pipeline loads several heavy models (faster-whisper ~3GB,
panns ~200MB, FastText ~600MB, text2vec ~400MB, HF sentiment ~400MB).
Without coordination they all stay resident, blow past the ~4GB VRAM
budget on the smaller GPUs we target, and crash with OOM on the third
or fourth video.

ModelPool keeps a bounded set of models in VRAM, evicts the
least-recently-used one when a new model needs space, and forces a
``torch.cuda.empty_cache()`` after every eviction so the freed bytes
actually return to the allocator.  CPU models don't count against the
limit — they cost RAM, not VRAM, and the host has more of it.

Typical usage::

    pool = ModelPool.instance()
    with pool.scope("panns", lambda: AudioTagging(...), gpu=True) as model:
        model.inference(batch)
    # `panns` is now eligible for eviction; if another GPU model is
    # acquired and the limit (default 2) is exceeded, panns is unloaded.

The pool is process-wide and thread-safe so the parallel ThreadPool in
controller can use it without external coordination.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable

from src.log.logger import getLogger

log = getLogger(__name__)


# Cap of GPU-resident models. Tuned so that the two largest
# co-resident pairs (whisper + panns ≈ 3.2GB, fasttext + text2vec ≈ 1GB)
# fit on a 4GB card; raise this on bigger hardware.
DEFAULT_GPU_LIMIT = int(os.environ.get("MODELPOOL_GPU_LIMIT", "2"))


class _Entry:
    __slots__ = ("name", "model", "gpu", "refcount")

    def __init__(self, name: str, model: Any, gpu: bool):
        self.name = name
        self.model = model
        self.gpu = gpu
        # refcount > 0 means an active scope is using the model and it
        # must NOT be evicted even if it's the LRU tail.
        self.refcount = 0


class ModelPool:
    """Process-wide bounded cache for ML models."""

    _instance: "ModelPool | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, gpu_limit: int = DEFAULT_GPU_LIMIT):
        if gpu_limit < 1:
            raise ValueError("gpu_limit must be ≥ 1")
        self._gpu_limit = gpu_limit
        # OrderedDict gives us O(1) move-to-end for LRU semantics.
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    # ── singleton ──────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "ModelPool":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the global singleton (test-only — releases all models)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.evict_all()
            cls._instance = None

    # ── public API ─────────────────────────────────────────────────────

    def acquire(
        self,
        name: str,
        factory: Callable[[], Any],
        gpu: bool = True,
    ) -> Any:
        """Return the named model, loading it via ``factory`` if absent.

        ``gpu=True`` counts against the LRU budget; ``gpu=False`` is
        cached but never evicted.  The returned model has its refcount
        bumped — call ``release(name)`` (or use ``scope()``) to drop it.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is not None:
                entry.refcount += 1
                self._entries.move_to_end(name)  # mark as MRU
                log.debug(f"ModelPool: hit {name} (refcount={entry.refcount})")
                return entry.model

            # Miss — make room first if this is a GPU model.
            if gpu:
                self._evict_to_fit(reserve_for=name)

            log.info(f"ModelPool: loading {name} (gpu={gpu})")
            model = factory()
            entry = _Entry(name, model, gpu)
            entry.refcount = 1
            self._entries[name] = entry
            return model

    def release(self, name: str) -> None:
        """Decrement refcount; the model stays cached but becomes evictable."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.refcount = max(0, entry.refcount - 1)

    def evict(self, name: str) -> bool:
        """Force-evict a single named model.  Refuses if refcount > 0.

        Returns True if the model was evicted.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return False
            if entry.refcount > 0:
                log.debug(f"ModelPool: refusing to evict in-use {name}")
                return False
            del self._entries[name]
            del entry.model
            self._free_vram()
            log.info(f"ModelPool: evicted {name}")
            return True

    def evict_all(self) -> None:
        """Drop everything that isn't currently in use."""
        with self._lock:
            for name in [n for n, e in self._entries.items() if e.refcount == 0]:
                del self._entries[name]
            self._entries.clear()  # also drop in-use refs (test-only)
            self._free_vram()

    @contextmanager
    def scope(
        self,
        name: str,
        factory: Callable[[], Any],
        gpu: bool = True,
    ):
        """Context manager that acquires + releases automatically.

        Releasing on exit doesn't unload the model — it just marks it
        evictable.  The next acquire still hits the cache.
        """
        model = self.acquire(name, factory, gpu=gpu)
        try:
            yield model
        finally:
            self.release(name)

    @property
    def resident(self) -> list[str]:
        """List of names currently cached, in LRU order (oldest first)."""
        with self._lock:
            return list(self._entries.keys())

    @property
    def gpu_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.gpu)

    # ── internals ──────────────────────────────────────────────────────

    def _evict_to_fit(self, reserve_for: str) -> None:
        """Evict GPU LRU entries until there's room for one more.

        ``reserve_for`` is purely advisory — it only appears in log lines
        so users can tell which model triggered the eviction.
        """
        while self.gpu_count >= self._gpu_limit:
            victim = self._pick_victim()
            if victim is None:
                # Everything is in-use; nothing we can do.  The new model
                # will load anyway and exceed the limit briefly — better
                # to overshoot than to deadlock.
                log.warning(
                    f"ModelPool: GPU limit {self._gpu_limit} reached but "
                    f"all entries in-use; loading {reserve_for} anyway"
                )
                return
            log.info(
                f"ModelPool: evicting {victim} to make room for {reserve_for}"
            )
            entry = self._entries.pop(victim)
            del entry.model
            self._free_vram()

    def _pick_victim(self) -> str | None:
        """Return the LRU GPU entry with refcount==0, or None."""
        # OrderedDict iteration is insertion-order = LRU order
        for name, entry in self._entries.items():
            if entry.gpu and entry.refcount == 0:
                return name
        return None

    @staticmethod
    def _free_vram() -> None:
        """Best-effort VRAM reclaim — torch.cuda.empty_cache + gc.collect."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
