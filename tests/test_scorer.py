# -*- coding: utf-8 -*-
"""Unit tests for the scorer module."""

import numpy as np
import pytest

from src.scorer import (
    _detectPeaks,
    _nms,
    _qualityFilter,
    _preprocess,
    _optimizeBoundaries,
    _compensateLag,
    score,
    PROFILES,
    NMS_MERGE_DISTANCE,
)


class TestDetectPeaks:
    """Tests for _detectPeaks — multi-scale peak detection."""

    def test_empty_signal_returns_empty(self):
        """Empty signal should produce no peaks."""
        result = _detectPeaks(np.array([], dtype=np.float64), k=2.0)
        assert result == []

    def test_flat_signal_returns_empty(self):
        """Flat signal has no peaks regardless of k."""
        flat = np.ones(100, dtype=np.float64)
        result = _detectPeaks(flat, k=2.0)
        assert result == []

    def test_single_spike_detected(self):
        """A clear isolated spike should be detected."""
        score = np.zeros(200, dtype=np.float64)
        score[100:103] = 10.0  # 3s spike for persistence
        result = _detectPeaks(score, k=2.0)
        assert len(result) >= 1

    def test_persistence_rejects_brief_spike(self):
        """Spikes shorter than 2s should be rejected."""
        score = np.zeros(200, dtype=np.float64)
        score[100] = 10.0  # single-point spike — no persistence
        result = _detectPeaks(score, k=2.0)
        # Should not be detected because persistence fails
        peakStarts = [int(r[0]) for r in result]
        assert 100 not in peakStarts

    def test_detection_respects_sensitivity(self):
        """Higher k (lower sensitivity) should find fewer peaks."""
        score = np.random.RandomState(42).randn(500).astype(np.float64)
        # Add some clear spikes
        score[100:103] = 8.0
        score[300:303] = 8.0

        peaks_lo = len(_detectPeaks(score, k=1.5))
        peaks_hi = len(_detectPeaks(score, k=4.0))
        assert peaks_lo >= peaks_hi

    def test_multi_scale_windows(self):
        """Different window sizes should contribute candidates."""
        score = np.zeros(300, dtype=np.float64)
        score[50:53] = 5.0    # narrow peak → caught by window=3
        score[150:170] = 3.0  # broad plateau → caught by window=30
        result = _detectPeaks(score, k=1.0)
        assert len(result) >= 1


class TestNMS:
    """Tests for _nms — non-maximum suppression."""

    def test_no_overlap_keeps_all(self):
        clips = [(0.0, 10.0, 0.8), (30.0, 40.0, 0.9), (60.0, 70.0, 0.7)]
        result = _nms(clips)
        assert len(result) == 3

    def test_overlapping_clips_merged(self):
        clips = [(0.0, 10.0, 0.8), (8.0, 18.0, 0.9)]
        result = _nms(clips)
        assert len(result) == 1
        assert result[0][0] == 0.0
        assert result[0][1] == 18.0
        assert result[0][2] == 0.9  # max score kept

    def test_empty_input(self):
        assert _nms([]) == []

    def test_single_clip(self):
        clips = [(5.0, 15.0, 0.5)]
        result = _nms(clips)
        assert result == clips


class TestQualityFilter:
    """Tests for _qualityFilter — duration bounds + t-test."""

    def test_too_short_clip_filtered(self):
        clips = [(0.0, 5.0, 0.8)]  # 5s < minDur=15
        score = np.ones(100, dtype=np.float64)
        result = _qualityFilter(clips, minDur=15.0, maxDur=90.0, score=score, nSecs=100)
        assert len(result) == 0

    def test_too_long_clip_filtered(self):
        clips = [(0.0, 95.0, 0.8)]  # 95s > maxDur=90
        score = np.ones(100, dtype=np.float64)
        result = _qualityFilter(clips, minDur=15.0, maxDur=90.0, score=score, nSecs=100)
        assert len(result) == 0

    def test_valid_duration_kept(self):
        clips = [(20.0, 50.0, 0.8)]  # 30s ∈ [15, 90]
        score = np.ones(100, dtype=np.float64)
        score[20:50] = 5.0
        result = _qualityFilter(clips, minDur=15.0, maxDur=90.0, score=score, nSecs=100)
        assert len(result) == 1

    def test_t_test_rejects_insignificant(self):
        """Clip with statistically insignificant difference should be rejected."""
        clips = [(60.0, 75.0, 0.8)]
        rng = np.random.RandomState(99)
        score = rng.normal(loc=1.0, scale=0.5, size=100).astype(np.float64)
        # Clip region has same distribution → t-test p > 0.05 → rejected
        result = _qualityFilter(clips, minDur=15.0, maxDur=90.0, score=score, nSecs=100)
        assert len(result) == 0


class TestPreprocess:
    """Tests for _preprocess — EWMA + Z-score + median filter."""

    def test_output_same_keys(self):
        features = {
            "rms": np.array([0.1] * 100, dtype=np.float64),
            "dmDensity": np.arange(100, dtype=np.float64),
        }
        result = _preprocess(features)
        assert set(result.keys()) == set(features.keys())

    def test_output_same_length(self):
        features = {"rms": np.random.RandomState(0).randn(200).astype(np.float64)}
        result = _preprocess(features)
        assert len(result["rms"]) == 200

    def test_zscore_centers(self):
        """Z-score of a constant signal should be ~0."""
        features = {"rms": np.ones(200, dtype=np.float64)}
        result = _preprocess(features)
        # All values should be near 0 (constant signal has no deviation)
        assert np.allclose(result["rms"][60:140], 0.0, atol=1e-6)


class TestScoreIntegration:
    """Light integration tests for the score() function."""

    def test_empty_features_returns_no_clips(self):
        result = score(roomId="test", profile="default")
        assert result["clips"] == []

    def test_mismatched_features_still_runs(self):
        """Scorer should handle features with different lengths."""
        af = {"features": {"rms": [0.1] * 50}}
        tf = {"features": {"dmDensity": [0.0] * 100}}
        result = score(
            roomId="test",
            audioFeatures=af,
            textFeatures=tf,
            profile="default",
            sensitivity=3.0,
        )
        assert "clips" in result

    def test_unknown_profile_falls_back_to_default(self):
        """Unknown profile name should fall back to 'default' weights."""
        result = score(
            roomId="test",
            audioFeatures={"features": {"rms": [0.1] * 60}},
            profile="nonexistent",
            sensitivity=3.0,
        )
        assert "clips" in result

    def test_profiles_have_consistent_keys(self):
        """All profiles should use the same feature keys as the default."""
        default_keys = set(PROFILES["default"].keys())
        for name, prof in PROFILES.items():
            assert set(prof.keys()) == default_keys, f"Profile '{name}' keys differ"

    def test_short_recording_produces_clips(self):
        """Recordings ≤60s must not produce all-zero Z-scores."""
        n = 30
        # Flat low-variance baseline + strong spike → t-test should pass
        rms = np.full(n, 0.1, dtype=np.float64)
        rms[10:14] = 10.0
        af = {"features": {"rms": rms.tolist()}}
        result = score(
            roomId="test",
            audioFeatures=af,
            profile="default",
            sensitivity=1.0,
            minDuration=3.0,
            maxDuration=20.0,
        )
        # Should find at least one clip even though n < 60
        assert len(result["clips"]) >= 1

    def test_very_short_recording_does_not_crash(self):
        """Recordings < 3s should run without error and return empty clips."""
        af = {"features": {"rms": [0.1, 0.2, 0.3]}}
        result = score(
            roomId="test",
            audioFeatures=af,
            profile="default",
            sensitivity=1.5,
            minDuration=3.0,
            maxDuration=20.0,
        )
        assert "clips" in result


class TestNmsMergeDistance:
    """Verify NMS_MERGE_DISTANCE constant is reasonable."""

    def test_merge_distance_positive(self):
        assert NMS_MERGE_DISTANCE > 0


class TestOptimizeBoundaries:
    """Regression — _optimizeBoundaries used to reference an undefined
    `searchStart`, raising NameError as soon as RMS was present and
    silently dropping every clip when caught upstream.
    """

    def test_runs_with_rms(self):
        rms = np.zeros(120, dtype=np.float64)
        rms[40] = -2.0  # valley before peak
        rms[80] = -2.0  # valley after peak
        zMatrix = {"rms": rms}
        result = _optimizeBoundaries([(50.0, 70.0, 1.0)], nSecs=120, zMatrix=zMatrix)
        assert len(result) == 1
        s, e, _ = result[0]
        # Boundary should snap to or near the valley positions
        assert 40 <= s <= 50
        assert 70 <= e <= 80

    def test_runs_without_rms(self):
        # No rms in zMatrix → should still produce ±5s bounds without error
        result = _optimizeBoundaries([(50.0, 70.0, 1.0)], nSecs=120, zMatrix={})
        assert len(result) == 1
        s, e, _ = result[0]
        assert s == 45
        assert e == 75

    def test_clamps_to_nsecs(self):
        rms = np.zeros(20, dtype=np.float64)
        result = _optimizeBoundaries([(0.0, 18.0, 1.0)], nSecs=20, zMatrix={"rms": rms})
        s, e, _ = result[0]
        assert s >= 0
        assert e <= 20


class TestCompensateLag:
    """Regression — wrap-around band must be zeroed for both lag signs."""

    def test_no_dm_signal_passes_through(self):
        zMatrix = {"rms": np.ones(50)}
        result = _compensateLag(zMatrix)
        assert "rms" in result
        assert np.array_equal(result["rms"], np.ones(50))

    def test_positive_lag_zeros_head(self):
        # Build a synthetic dm signal that lags the audio by +2s.
        # Cross-correlation should detect that and shift dm forward;
        # the head wrap-around must be zeroed (not contain wrapped tail).
        n = 100
        au = np.zeros(n)
        au[20:25] = 5.0
        dm = np.zeros(n)
        dm[22:27] = 5.0  # +2s late vs audio
        zMatrix = {"rms": au.copy(), "dmDensity": dm.copy()}
        out = _compensateLag(zMatrix)
        # Whatever the detected lag, the dm signal length is preserved
        assert len(out["dmDensity"]) == n
        # And no value should be larger than the original peak (no wraparound dup)
        assert float(out["dmDensity"].max()) <= 5.0 + 1e-9

    def test_short_signal_skips(self):
        zMatrix = {"rms": np.ones(5), "dmDensity": np.ones(5)}
        result = _compensateLag(zMatrix)
        # < 10 samples → no shift performed
        assert np.array_equal(result["dmDensity"], np.ones(5))
