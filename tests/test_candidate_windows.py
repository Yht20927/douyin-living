# -*- coding: utf-8 -*-
"""Candidate window generation tests — expand short, trim long, never lose all."""

import numpy as np

from src.scorer import (
    _bestSubwindow,
    _expandCandidates,
    _nms,
    _qualityFilter,
    score,
)


class TestCandidateExpansion:
    def test_short_candidate_expanded_to_min_duration(self):
        scoreCurve = np.zeros(200, dtype=np.float64)
        scoreCurve[50:52] = 5.0
        result = _expandCandidates(
            [(50.0, 52.0, 1.0)], nSecs=200, minDur=15.0, maxDur=90.0,
            score=scoreCurve,
        )
        assert len(result) == 1
        start, end, _ = result[0]
        assert end - start == 15.0
        assert start <= 50 and end >= 52

    def test_expansion_clamps_at_recording_start(self):
        scoreCurve = np.zeros(100, dtype=np.float64)
        result = _expandCandidates(
            [(0.0, 2.0, 1.0)], nSecs=100, minDur=15.0, maxDur=90.0,
            score=scoreCurve,
        )
        start, end, _ = result[0]
        assert start == 0.0
        assert end - start == 15.0

    def test_recording_shorter_than_min_uses_whole_source(self):
        scoreCurve = np.ones(10, dtype=np.float64)
        result = _expandCandidates(
            [(3.0, 5.0, 1.0)], nSecs=10, minDur=15.0, maxDur=90.0,
            score=scoreCurve,
        )
        assert result == [(0.0, 10.0, 1.0)]

    def test_long_candidate_trimmed_to_best_subwindow(self):
        scoreCurve = np.zeros(300, dtype=np.float64)
        scoreCurve[120:140] = 9.0
        start, end = _bestSubwindow(scoreCurve, 100, 200, maxDur=30.0)
        assert end - start == 30
        assert start <= 120 and end >= 140


class TestNmsDuration:
    def test_merge_over_max_duration_is_trimmed(self):
        scoreCurve = np.zeros(200, dtype=np.float64)
        scoreCurve[80:90] = 10.0
        merged = _nms(
            [(0.0, 60.0, 1.0), (50.0, 100.0, 2.0)],
            maxDur=90.0,
            score=scoreCurve,
        )
        assert len(merged) == 1
        start, end, _ = merged[0]
        assert end - start <= 90.0

    def test_duplicate_is_recorded(self):
        dropped = []
        _nms([(0.0, 10.0, 1.0), (8.0, 18.0, 2.0)], dropped=dropped)
        assert dropped and dropped[0]["reason"] == "duplicate"


class TestQualityFilterRepair:
    def test_t_too_short_is_expanded_not_dropped(self):
        scoreCurve = np.ones(100, dtype=np.float64)
        result = _qualityFilter(
            [(0.0, 5.0, 0.8)], minDur=15.0, maxDur=90.0,
            score=scoreCurve, nSecs=100,
        )
        assert len(result) == 1
        start, end, _ = result[0]
        assert end - start == 15.0

    def test_too_long_is_trimmed_not_dropped(self):
        scoreCurve = np.ones(200, dtype=np.float64)
        result = _qualityFilter(
            [(0.0, 120.0, 0.8)], minDur=15.0, maxDur=90.0,
            score=scoreCurve, nSecs=200,
        )
        assert len(result) == 1
        start, end, _ = result[0]
        assert end - start == 90.0

    def test_t_test_failure_kept_with_score_penalty(self):
        rng = np.random.RandomState(99)
        scoreCurve = rng.normal(loc=1.0, scale=0.5, size=200).astype(np.float64)
        adjustments = []
        result = _qualityFilter(
            [(60.0, 75.0, 0.8)], minDur=15.0, maxDur=90.0,
            score=scoreCurve, nSecs=200, adjustments=adjustments,
        )
        assert len(result) == 1
        assert result[0][2] < 0.8
        assert any(a["reason"] == "t_test_insignificant" for a in adjustments)

    def test_impossible_window_records_drop_reason(self):
        dropped = []
        result = _qualityFilter(
            [(0.0, 0.0, 0.8)], minDur=15.0, maxDur=90.0,
            score=np.zeros(100), nSecs=100, dropped=dropped,
        )
        assert result == []
        assert dropped[0]["reason"] == "too_short"


class TestHourLongRecall:
    def test_sparse_peaks_produce_many_candidates(self):
        """1h synthetic signal with 60 sparse peaks must yield ≥10 clips.

        Regression for the old pipeline where 59 candidates collapsed to
        0–1 after boundary/NMS/duration filtering.
        """
        rng = np.random.RandomState(0)
        n = 3600
        rms = rng.rand(n) * 0.2
        for t in range(20, n, 60):
            rms[t:t + 3] += 5.0

        result = score(
            audioFeatures={"features": {"rms": rms.tolist()}},
            sensitivity=2.5,
            minDuration=15.0,
            maxDuration=90.0,
        )
        assert len(result["clips"]) >= 10
        for clip in result["clips"]:
            assert 15.0 <= clip["duration"] <= 90.0
