# -*- coding: utf-8 -*-
"""Unit tests for signalText helpers — pure-python paths only.

The full extractFeatures() loads jieba/FastText/text2vec, which we don't
exercise here.  We focus on the small algorithmic primitives that have
caused trouble: keyword fuzzy match, UTR sliding window, entropy, and
second derivative.
"""

import numpy as np
import pytest

import src.signalText as signalText
from src.signalText import (
    KEYWORD_CATEGORY_FEATURES,
    _calibrateUtrThreshold,
    _canonicalKeywordCategory,
    _computeEntropyWindow,
    _computeKeywordCategorySignals,
    _computeMessageRate,
    _computeRepeatRate,
    _computeTopicChange,
    _computeUtr,
    _fastSentiment,
    _fuzzyMatchCached,
    _getContent,
    _getUserId,
    _parseTimestamp,
    _resolveDuration,
    _ruleSentiment,
    _secondDerivative,
    _sentimentForWords,
    _textHasEmoji,
    POOL_NAME_TEXT2VEC,
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


class TestResolveDuration:
    """The text timeline must come from real media, not ``len(danmaku)//10``."""

    def test_explicit_duration_wins(self):
        assert _resolveDuration(
            120.0,
            [{"end": 30.0}],
            [{"recvTimeSec": 90.0}],
        ) == 120.0

    def test_asr_end_used_without_explicit_duration(self):
        assert _resolveDuration(
            None,
            [{"end": 12.5}, {"end": 47.0}, {"end": None}],
            [],
        ) == 47.0

    def test_max_danmaku_timestamp_used(self):
        assert _resolveDuration(
            None,
            [],
            [{"recvTimeSec": 3.0}, {"timestamp": 61.5}, {"_time": 20.0}],
        ) == 61.5

    def test_longest_source_wins(self):
        assert _resolveDuration(
            None,
            [{"end": 15.0}],
            [{"recvTimeSec": 80.0}],
        ) == 80.0

    def test_invalid_values_ignored(self):
        assert _resolveDuration(
            None,
            [{"end": "bad"}, {"end": None}],
            [{"recvTimeSec": "bad"}, {}],
        ) == 0.0

    def test_empty_sources_return_zero(self):
        assert _resolveDuration(None, [], []) == 0.0

    def test_zero_or_negative_explicit_duration_falls_back(self):
        assert _resolveDuration(0, [{"end": 9.0}], []) == 9.0
        assert _resolveDuration(-5, [], [{"recvTimeSec": 9.0}]) == 9.0


class TestSimilarityThreshold:
    """Lock the threshold default — bumping it silently breaks recall."""

    def test_threshold_value(self):
        # Code paths use SIMILARITY_THRESHOLD; if anyone bumps it, this
        # test forces an explicit code review.
        assert SIMILARITY_THRESHOLD == 0.7


class TestFastSentiment:
    def test_empty_words_returns_zero(self):
        assert _fastSentiment([]) == 0.0

    def test_rule_fallback_positive(self, monkeypatch):
        monkeypatch.setattr(signalText, "_sentimentAvailable", False)
        monkeypatch.setattr(signalText, "_sentimentPipeline", None)
        assert _fastSentiment(["哈哈", "真", "好"]) == 0.5

    def test_rule_fallback_negative(self, monkeypatch):
        monkeypatch.setattr(signalText, "_sentimentAvailable", False)
        monkeypatch.setattr(signalText, "_sentimentPipeline", None)
        assert _fastSentiment(["无语", "翻车"]) == -0.5

    def test_rule_fallback_neutral(self, monkeypatch):
        monkeypatch.setattr(signalText, "_sentimentAvailable", False)
        monkeypatch.setattr(signalText, "_sentimentPipeline", None)
        assert _fastSentiment(["今天", "天气"]) == 0.0

    def test_pipeline_positive_and_negative(self, monkeypatch):
        monkeypatch.setattr(signalText, "_sentimentAvailable", True)

        class Pipeline:
            def __init__(self, label):
                self.label = label

            def __call__(self, text):
                return [{"label": self.label, "score": 0.9}]

        monkeypatch.setattr(signalText, "_sentimentPipeline", Pipeline("POSITIVE"))
        assert _fastSentiment(["哈哈"]) == 0.9
        monkeypatch.setattr(signalText, "_sentimentPipeline", Pipeline("NEGATIVE"))
        assert _fastSentiment(["哈哈"]) == -0.9

    def test_pipeline_failure_falls_back_to_rules(self, monkeypatch):
        monkeypatch.setattr(signalText, "_sentimentAvailable", True)

        def broken(text):
            raise RuntimeError("inference failed")

        monkeypatch.setattr(signalText, "_sentimentPipeline", broken)
        assert _fastSentiment(["哈哈"]) == 0.5


class TestRuleSentiment:
    def test_positive_negative_neutral(self):
        assert _ruleSentiment(["哈哈", "真", "好"]) == 0.5
        assert _ruleSentiment(["无语", "翻车"]) == -0.5
        assert _ruleSentiment(["今天", "天气"]) == 0.0
        assert _ruleSentiment([]) == 0.0

    def test_rules_mode_never_calls_model(self, monkeypatch):
        monkeypatch.setattr(signalText._cfg, "sentiment_mode", "rules")
        monkeypatch.setattr(
            signalText,
            "_fastSentiment",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("model must not be called")
            ),
        )
        assert _sentimentForWords(["哈哈"]) == 0.5
        assert _sentimentForWords(["今天"]) == 0.0

    def test_hybrid_only_calls_model_for_neutral(self, monkeypatch):
        calls = []

        def fakeFastSentiment(words, pool=None, holdings=None):
            calls.append(list(words))
            return 0.9

        monkeypatch.setattr(signalText._cfg, "sentiment_mode", "hybrid")
        monkeypatch.setattr(signalText, "_fastSentiment", fakeFastSentiment)
        assert _sentimentForWords(["哈哈"]) == 0.5
        assert calls == []
        assert _sentimentForWords(["今天"]) == 0.9
        assert calls == [["今天"]]


class TestDanmakuRates:
    def test_repeat_rate_detects_duplicates(self):
        texts = [["666", "666", "上链接"], [], []]
        rate = _computeRepeatRate(texts, radius=0)
        assert rate[0] == pytest.approx(1.0 / 3.0)
        assert rate[1:] == [0.0, 0.0]

    def test_repeat_rate_normalises_case_and_spaces(self):
        assert _computeRepeatRate([["Spam", " spam "]], radius=0) == [0.5]

    def test_repeat_rate_empty(self):
        assert _computeRepeatRate([], radius=5) == []
        assert _computeRepeatRate([[], []], radius=1) == [0.0, 0.0]

    def test_question_rate(self):
        texts = [["真的吗？"], ["好"], ["怎么弄?"]]
        rate = _computeMessageRate(
            texts,
            lambda text: any(marker in text for marker in ("?", "？")),
            radius=0,
        )
        assert rate == [1.0, 0.0, 1.0]

    def test_exclamation_and_emoji_rates(self):
        texts = [["好！"], ["🙂"], ["[笑哭]"], ["普通"]]
        exclamation = _computeMessageRate(
            texts,
            lambda text: any(marker in text for marker in ("!", "！")),
            radius=0,
        )
        emoji = _computeMessageRate(texts, _textHasEmoji, radius=0)
        assert exclamation == [1.0, 0.0, 0.0, 0.0]
        assert emoji == [0.0, 1.0, 1.0, 0.0]

    def test_emoji_detection(self):
        assert _textHasEmoji("😂")
        assert _textHasEmoji("[捂脸]")
        assert not _textHasEmoji("普通弹幕")


class TestKeywordCategories:
    def test_category_signals_combine_danmaku_and_asr(self):
        texts = [["666"], [], ["上链接"]]
        signals = _computeKeywordCategorySignals(
            texts,
            {
                "highEnergy": ["666"],
                "shopping": ["上链接", "库存"],
                "unknownCategory": ["x"],
            },
            asrSegments=[{"start": 1, "end": 2, "text": "库存最后一批"}],
            radius=0,
        )
        assert set(signals) == set(KEYWORD_CATEGORY_FEATURES.values())
        assert signals["keywordHighEnergy"][0] == pytest.approx(0.6)
        assert signals["keywordShopping"][0] == 0.0
        assert signals["keywordShopping"][1] == pytest.approx(0.4)
        assert signals["keywordShopping"][2] == pytest.approx(0.6)
        assert signals["keywordFunny"] == [0.0, 0.0, 0.0]

    def test_category_aliases(self):
        assert _canonicalKeywordCategory("highEnergy") == "highEnergy"
        assert _canonicalKeywordCategory("high_energy") == "highEnergy"
        assert _canonicalKeywordCategory("unknown") is None

    def test_empty_inputs_return_zero_arrays(self):
        signals = _computeKeywordCategorySignals([], {"shopping": ["上车"]})
        assert all(values == [] for values in signals.values())


class TestUtrCalibration:
    def test_sparse_data_keeps_configured_threshold(self):
        threshold, source = _calibrateUtrThreshold(
            [0.2] * 5, 0.3, percentile=20, minSamples=20, low=0.15, high=0.6
        )
        assert (threshold, source) == (0.3, "configured")

    def test_calibrates_from_distribution(self):
        values = [0.1 + index * 0.01 for index in range(100)]
        threshold, source = _calibrateUtrThreshold(
            values, 0.3, percentile=20, minSamples=50, low=0.15, high=0.6
        )
        assert source == "calibrated"
        assert threshold == pytest.approx(0.29, abs=0.02)

    def test_clamps_to_safe_bounds(self):
        threshold, _ = _calibrateUtrThreshold(
            [0.95] * 100, 0.3, percentile=20, minSamples=50, low=0.15, high=0.6
        )
        assert threshold == 0.6


class TestComputeTopicChange:
    def _installFakeDeps(self, monkeypatch, encode=None, labels=None):
        import sys
        import types

        class FakeSentenceModel:
            def __init__(self, name):
                self.name = name

            def encode(self, texts):
                if encode is not None:
                    return encode(texts)
                return np.array([[1.0, 0.0] for _ in texts])

        class FakeKMeans:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_predict(self, embeddings):
                if labels is not None:
                    return np.array(labels)
                return np.zeros(len(embeddings), dtype=int)

        text2vecModule = types.ModuleType("text2vec")
        text2vecModule.SentenceModel = FakeSentenceModel
        sklearnModule = types.ModuleType("sklearn")
        clusterModule = types.ModuleType("sklearn.cluster")
        clusterModule.KMeans = FakeKMeans
        sklearnModule.cluster = clusterModule
        monkeypatch.setitem(sys.modules, "text2vec", text2vecModule)
        monkeypatch.setitem(sys.modules, "sklearn", sklearnModule)
        monkeypatch.setitem(sys.modules, "sklearn.cluster", clusterModule)

    def test_cluster_change_marks_transition(self, monkeypatch):
        self._installFakeDeps(monkeypatch, labels=[0, 1])
        allText = [[f"topic{i // 30}"] for i in range(60)]
        result = _computeTopicChange(allText, 60)
        assert sum(result) == 1
        assert result[30] == 1

    def test_requires_two_non_empty_windows(self, monkeypatch):
        self._installFakeDeps(monkeypatch)
        allText = [[""] for _ in range(60)]
        assert _computeTopicChange(allText, 60) == [0] * 60

    def test_empty_embeddings_returns_zero(self, monkeypatch):
        self._installFakeDeps(monkeypatch, encode=lambda texts: [])
        allText = [[f"w{i // 30}"] for i in range(60)]
        assert _computeTopicChange(allText, 60) == [0] * 60

    def test_uses_pool_and_records_holding(self, monkeypatch):
        self._installFakeDeps(monkeypatch, labels=[0, 1])

        acquired = []

        class FakePool:
            def acquire(self, name, factory, gpu=False):
                acquired.append((name, gpu))
                return factory()

        holdings = []
        allText = [[f"w{i // 30}"] for i in range(60)]
        result = _computeTopicChange(allText, 60, pool=FakePool(), holdings=holdings)
        assert acquired == [(POOL_NAME_TEXT2VEC, True)]
        assert holdings == [POOL_NAME_TEXT2VEC]
        assert sum(result) == 1

    def test_missing_dependencies_returns_zero(self, monkeypatch):
        import builtins

        realImport = builtins.__import__

        def guardedImport(name, *args, **kwargs):
            if name in {"text2vec", "sklearn.cluster"}:
                raise ImportError("not installed")
            return realImport(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guardedImport)
        allText = [[f"w{i // 30}"] for i in range(60)]
        assert _computeTopicChange(allText, 60) == [0] * 60
