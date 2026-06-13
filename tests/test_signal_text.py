# -*- coding: utf-8 -*-
"""Unit tests for signalText helpers — pure-python paths only.

The full extractFeatures() loads jieba/FastText/text2vec, which we don't
exercise here.  We focus on the small algorithmic primitives that have
caused trouble: keyword fuzzy match, UTR sliding window, entropy, and
second derivative.
"""

import json
import numpy as np
import pytest

from src.signalText import (
    _fuzzyMatch,
    _fuzzyMatchCached,
    _computeUtr,
    _computeEntropyWindow,
    _secondDerivative,
    _parseTimestamp,
    _getContent,
    _getUserId,
    SIMILARITY_THRESHOLD,
)


class _StubModel:
    """Minimal FastText stand-in.  Each token gets a fixed unit vector by
    hashing its first character — collisions are intentional so we can
    drive the cosine similarity in tests deterministically.
    """

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._mapping = mapping

    def get_sentence_vector(self, text: str) -> np.ndarray:
        if text in self._mapping:
            return self._mapping[text]
        # Default: zeros (so cosine sim → 0)
        return np.zeros(4, dtype=np.float32)


class TestFuzzyMatchCached:
    def test_exact_match_short_circuits(self):
        # Even with a None model, exact substring should match instantly.
        assert _fuzzyMatchCached("我笑了哈哈", ["哈哈"], None, None, None) is True

    def test_no_match_no_model(self):
        assert _fuzzyMatchCached("早上好", ["哈哈"], None, None, None) is False

    def test_cosine_above_threshold(self):
        # text and keyword get identical vectors → cosine = 1.0 → above 0.7
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        model = _StubModel({"完全笑死": v, "哈哈": v})
        kwVecs = [v]
        kwNorms = np.array([np.linalg.norm(v) + 1e-10], dtype=np.float32)
        assert _fuzzyMatchCached("完全笑死", ["哈哈"], model, kwVecs, kwNorms) is True

    def test_cosine_below_threshold(self):
        # Orthogonal vectors → cosine = 0 → below threshold
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        model = _StubModel({"今天天气": v1, "哈哈": v2})
        kwVecs = [v2]
        kwNorms = np.array([1.0], dtype=np.float32)
        assert _fuzzyMatchCached("今天天气", ["哈哈"], model, kwVecs, kwNorms) is False

    def test_uses_pre_embedded_keywords(self):
        # The cached path must NOT call get_sentence_vector for keywords —
        # only for the input text.
        call_log: list[str] = []

        class CountingModel(_StubModel):
            def get_sentence_vector(self, text):
                call_log.append(text)
                return super().get_sentence_vector(text)

        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        model = CountingModel({"text": v})
        kwVecs = [v, v]
        kwNorms = np.array([1.0, 1.0], dtype=np.float32)
        _fuzzyMatchCached("text", ["kw1", "kw2"], model, kwVecs, kwNorms)
        # Only the input text should have been embedded once
        assert call_log == ["text"]


class TestComputeUtr:
    def test_empty_input(self):
        assert _computeUtr([]) == []

    def test_all_unique(self):
        # 5 seconds, each carrying one unique token → UTR ~ 1.0
        texts = [["a"], ["b"], ["c"], ["d"], ["e"]]
        utr = _computeUtr(texts)
        assert len(utr) == 5
        for v in utr:
            assert v == pytest.approx(1.0)

    def test_repeated_token_lowers_utr(self):
        # All seconds emit the same token → unique=1, total>=5 → low UTR
        texts = [["spam"]] * 10
        utr = _computeUtr(texts)
        for v in utr:
            assert v < 1.0
            assert v > 0.0

    def test_no_text_returns_one(self):
        # By convention, empty windows have neutral (1.0) UTR
        assert _computeUtr([[]] * 3) == [1.0, 1.0, 1.0]


class TestComputeEntropyWindow:
    def test_uniform_high_entropy(self):
        # Many distinct tokens around t=5 → entropy > 0
        texts: list[list[str]] = [[] for _ in range(20)]
        for i in range(20):
            texts[i] = [f"w{i}"]
        ent = _computeEntropyWindow(texts, 20)
        assert max(ent) > 0.0

    def test_repeated_zero_or_low(self):
        # All identical tokens → entropy = 0 (single bin)
        texts = [["same"] for _ in range(20)]
        ent = _computeEntropyWindow(texts, 20)
        for v in ent:
            assert v == pytest.approx(0.0, abs=1e-9)

    def test_empty_signal(self):
        ent = _computeEntropyWindow([[] for _ in range(10)], 10)
        assert ent == [0.0] * 10


class TestSecondDerivative:
    def test_constant_zero(self):
        assert _secondDerivative([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]

    def test_linear_zero_inside(self):
        # f(x) = x → f'' = 0 except at the boundaries (which we leave 0)
        assert _secondDerivative([0.0, 1.0, 2.0, 3.0]) == [0.0, 0.0, 0.0, 0.0]

    def test_quadratic_const_inside(self):
        # f(x) = x² → f'' = 2 at every interior sample
        out = _secondDerivative([0.0, 1.0, 4.0, 9.0, 16.0])
        assert out[0] == 0.0 and out[-1] == 0.0
        for v in out[1:-1]:
            assert v == pytest.approx(2.0)


class TestEntryHelpers:
    def test_parseTimestamp_priority(self):
        # recvTimeSec wins over timestamp / _time
        assert _parseTimestamp({"recvTimeSec": 1.5, "timestamp": 99}) == 1.5
        assert _parseTimestamp({"timestamp": 7, "_time": 99}) == 7.0
        assert _parseTimestamp({}) == 0.0

    def test_parseTimestamp_bad_value(self):
        # Non-numeric values should fall through, not crash
        assert _parseTimestamp({"recvTimeSec": "abc", "_time": 3}) == 3.0

    def test_getContent(self):
        assert _getContent({"content": "hi"}) == "hi"
        assert _getContent({"text": "fallback"}) == "fallback"
        assert _getContent({}) == ""

    def test_getUserId_hashed(self):
        # Same input → same hash; deterministic; 16 hex chars
        h1 = _getUserId({"userId": "abc"})
        h2 = _getUserId({"userId": "abc"})
        assert h1 == h2
        assert len(h1) == 16
        # And different inputs (almost always) hash differently
        assert _getUserId({"userId": "abc"}) != _getUserId({"userId": "xyz"})


class TestSimilarityThreshold:
    """Lock the threshold default — bumping it silently breaks recall."""

    def test_threshold_value(self):
        # Code paths use SIMILARITY_THRESHOLD; if anyone bumps it, this
        # test forces an explicit code review.
        assert SIMILARITY_THRESHOLD == 0.7
