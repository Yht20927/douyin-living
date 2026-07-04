# -*- coding: utf-8 -*-
"""Integration test: extractFeatures must release every model it acquires.

The bug this guards against: in the WIP after Sprint 1, ``signalText``
acquired FastText through ``ModelPool`` but never called ``release()``,
leaving the model pinned at refcount=1 forever and defeating the LRU
eviction the pool exists to provide.

We don't have the real heavy dependencies (fasttext, text2vec,
transformers) installed in CI — and even if we did, loading them takes
seconds.  Instead we inject lightweight stubs into ``sys.modules`` so
the loader branches in ``signalText`` execute against fakes, then check
the ModelPool refcount discipline afterwards.
"""

import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest

from src.modelPool import ModelPool
from src import signalText


# ── Lightweight model stubs ──────────────────────────────────────────


class _StubFastTextModel:
    """get_sentence_vector returns a deterministic 4-d vector."""

    def get_sentence_vector(self, text: str) -> np.ndarray:
        # Hash to a stable 4-byte vector so different inputs differ.
        h = hash(text)
        return np.array(
            [(h >> i) & 0xFF for i in (0, 8, 16, 24)],
            dtype=np.float32,
        ) / 256.0


class _StubSentenceModel:
    """text2vec stand-in — encode → small numpy matrix."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        return np.array([[hash(t) & 0xFF, len(t)] for t in texts], dtype=np.float32)


class _StubPipeline:
    """transformers.pipeline stand-in — always positive, deterministic."""

    def __call__(self, text):
        return [{"label": "POSITIVE", "score": 0.9}]


# ── Test fixtures ────────────────────────────────────────────────────


@pytest.fixture
def _stub_modules(monkeypatch, tmp_path):
    """Inject stub fasttext/text2vec/transformers into sys.modules so the
    heavy import branches inside extractFeatures take the loaded path."""

    # Reset ModelPool — the singleton may be carrying state from earlier tests
    ModelPool.reset()

    # ── fasttext ─────────────────────────────────────────────────────
    fasttext_stub = types.ModuleType("fasttext")
    fasttext_stub.load_model = lambda path: _StubFastTextModel()
    monkeypatch.setitem(sys.modules, "fasttext", fasttext_stub)

    # FastText only loads if cc.zh.100.bin exists on disk — touch it.
    # The factory ignores the file's contents so an empty file is fine.
    bin_path = Path("cc.zh.100.bin")
    created = False
    if not bin_path.exists():
        bin_path.touch()
        created = True

    # ── text2vec ─────────────────────────────────────────────────────
    text2vec_stub = types.ModuleType("text2vec")
    text2vec_stub.SentenceModel = _StubSentenceModel
    monkeypatch.setitem(sys.modules, "text2vec", text2vec_stub)

    # ── transformers ─────────────────────────────────────────────────
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.pipeline = lambda *a, **k: _StubPipeline()
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    # Reset the module-level sentiment cache so each test re-loads.
    monkeypatch.setattr(signalText, "_sentimentPipeline", None)
    monkeypatch.setattr(signalText, "_sentimentAvailable", None)

    yield

    if created:
        bin_path.unlink(missing_ok=True)
    ModelPool.reset()


@pytest.fixture
def _tiny_inputs(tmp_path):
    """Minimal danmaku + ASR fixtures that exercise every branch."""
    danmaku = tmp_path / "dm.jsonl"
    asr = tmp_path / "asr.json"

    # Two danmaku messages a few seconds apart so the sentiment loop runs
    danmaku.write_text(
        json.dumps({"recvTimeSec": 0.5, "content": "哈哈太好笑了", "userId": "u1"}, ensure_ascii=False) + "\n"
        + json.dumps({"recvTimeSec": 2.0, "content": "牛 nice", "userId": "u2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Two ASR segments so the FastText acquisition path triggers
    asr.write_text(json.dumps({
        "segments": [
            {"start": 0, "end": 2, "text": "大家好欢迎来到直播间", "speaker": "S1"},
            {"start": 2, "end": 4, "text": "今天我们来聊聊最近的热点话题", "speaker": "S2"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    return str(danmaku), str(asr)


# ── Tests ────────────────────────────────────────────────────────────


class TestPoolReleaseDiscipline:
    def test_all_holdings_released_after_extract(self, _stub_modules, _tiny_inputs):
        """The whole point of Sprint 2: every acquired model must be
        evictable (refcount=0) after extractFeatures returns.
        """
        danmakuPath, asrPath = _tiny_inputs

        signalText.extractFeatures(danmakuPath, asrPath)

        pool = ModelPool.instance()
        # All three heavy models should be cached but not pinned.
        for name in (
            signalText.POOL_NAME_FASTTEXT,
            signalText.POOL_NAME_TEXT2VEC,
            signalText.POOL_NAME_HF_SENTIMENT,
        ):
            if name in pool.resident:
                # pool.evict() refuses if refcount > 0 — if it succeeds
                # we know refcount was 0, which is what we want.
                assert pool.evict(name), f"{name} not releasable — refcount > 0"

    def test_release_runs_even_on_exception(self, _stub_modules, _tiny_inputs):
        """If extraction blows up midway, the finally block still releases.

        We force the failure at output-write time so all three models
        have already been acquired by the time we raise.
        """
        danmakuPath, asrPath = _tiny_inputs

        # Pointing outputPath at a non-existent directory makes the
        # final json.dump raise — by then everything is acquired.
        with pytest.raises(FileNotFoundError):
            signalText.extractFeatures(
                danmakuPath, asrPath,
                outputPath="/nonexistent/dir/out.json",
            )

        pool = ModelPool.instance()
        for name in (
            signalText.POOL_NAME_FASTTEXT,
            signalText.POOL_NAME_TEXT2VEC,
            signalText.POOL_NAME_HF_SENTIMENT,
        ):
            if name in pool.resident:
                assert pool.evict(name), f"{name} leaked after exception"

    def test_consecutive_extracts_dont_inflate_refcount(self, _stub_modules, _tiny_inputs):
        """Three back-to-back runs must leave each model at refcount=0.

        This is the real-world failure mode for the Sprint 1 bug: the
        controller processes multiple videos in sequence, and each call
        to extractFeatures was bumping refcount by 1 with no matching
        release, so by video 3 the model was pinned 3-deep and could
        never be evicted — exactly the OOM the pool was designed to
        prevent.
        """
        danmakuPath, asrPath = _tiny_inputs

        for _ in range(3):
            signalText.extractFeatures(danmakuPath, asrPath)

        pool = ModelPool.instance()
        for name in (
            signalText.POOL_NAME_FASTTEXT,
            signalText.POOL_NAME_TEXT2VEC,
            signalText.POOL_NAME_HF_SENTIMENT,
        ):
            if name in pool.resident:
                # If acquire bumped refcount once per run without a
                # matching release, the count would now be 3 and
                # evict() would refuse.
                assert pool.evict(name), (
                    f"{name} accumulated refcount over consecutive extracts"
                )
