# -*- coding: utf-8 -*-
"""Window-level feature aggregation tests."""

import numpy as np

from src.pipeline.window_features import (
    computeWindowFeatures,
    durationFit,
    normalizeFeatureArrays,
    resolveProfileWeights,
)


class TestNormalisation:
    def test_constant_signal_becomes_zero(self):
        result = normalizeFeatureArrays({"rms": np.ones(50)})
        assert np.allclose(result["rms"], 0.0)

    def test_sparse_burst_is_amplified_and_clipped(self):
        signal = np.zeros(200)
        signal[100] = 10.0
        result = normalizeFeatureArrays({"rms": signal})["rms"]
        assert result[100] > 2.0
        assert np.max(result) <= 5.0
        assert abs(result[0]) < 1e-9

    def test_empty_signal_is_preserved(self):
        result = normalizeFeatureArrays({"empty": np.array([])})
        assert result["empty"].size == 0


class TestWeights:
    def test_profile_weights_are_normalised(self):
        weights = resolveProfileWeights("default", ["dmDensity", "rms"])
        assert set(weights) <= {"dmDensity", "rms"}
        assert sum(weights.values()) == 1.0

    def test_unknown_profile_falls_back_to_default(self):
        default = resolveProfileWeights("default", ["dmDensity", "rms"])
        unknown = resolveProfileWeights("does-not-exist", ["dmDensity", "rms"])
        assert default == unknown

    def test_unweighted_feature_set_uses_equal_weights(self):
        weights = resolveProfileWeights("default", ["brandNewSignal"])
        assert weights == {"brandNewSignal": 1.0}


class TestDurationFit:
    def test_inside_range(self):
        assert durationFit(20.0, 15.0, 30.0) == 1.0

    def test_too_short_decays_linearly(self):
        assert durationFit(7.5, 15.0, 30.0) == 0.5

    def test_too_long_decays_linearly(self):
        assert durationFit(60.0, 15.0, 30.0) == 0.5

    def test_zero_duration(self):
        assert durationFit(0.0, 15.0, 30.0) == 0.0


class TestWindowFeatures:
    def _signals(self):
        rms = np.zeros(300)
        dmDensity = np.zeros(300)
        rms[100:110] = 4.0
        dmDensity[100:106] = 3.0
        return {"rms": rms, "dmDensity": dmDensity}

    def test_statistics_and_agreement(self):
        window = computeWindowFeatures(
            self._signals(),
            start=90,
            end=120,
            weights={"rms": 0.5, "dmDensity": 0.5},
            minDuration=15,
            maxDuration=40,
        )
        assert window["peakScore"] > 1.0
        assert window["meanScore"] > 0.0
        assert window["areaScore"] > 0.0
        assert window["prePostDelta"] > 0.0
        assert window["multiSignalAgreement"] == 1.0
        assert window["durationFit"] == 1.0
        assert window["activeSignals"] == ["rms", "dmDensity"]
        assert set(window["signals"]) == {"rms", "dmDensity"}
        for stats in window["signals"].values():
            assert set(stats) >= {
                "weight", "peak", "mean", "area",
                "preDelta", "postDelta", "active", "contribution",
            }

    def test_novelty_is_positive_after_quiet_context(self):
        window = computeWindowFeatures(
            self._signals(),
            start=100,
            end=130,
            weights={"rms": 0.5, "dmDensity": 0.5},
        )
        assert window["novelty"] > 0.1

    def test_weights_control_peak_contribution(self):
        signals = {"rms": np.full(60, 3.0), "dmDensity": np.zeros(60)}
        audioOnly = computeWindowFeatures(
            signals, 10, 40, weights={"rms": 1.0, "dmDensity": 0.0}
        )
        textOnly = computeWindowFeatures(
            signals, 10, 40, weights={"rms": 0.0, "dmDensity": 1.0}
        )
        assert audioOnly["peakScore"] > textOnly["peakScore"]
        assert textOnly["activeSignals"] == []

    def test_empty_window_is_safe(self):
        window = computeWindowFeatures({"rms": np.zeros(10)}, 5, 5)
        assert window["duration"] == 0.0
        assert window["durationFit"] == 0.0
        assert window["activeSignals"] == []

    def test_is_deterministic(self):
        first = computeWindowFeatures(self._signals(), 90, 120)
        second = computeWindowFeatures(self._signals(), 90, 120)
        assert first == second
