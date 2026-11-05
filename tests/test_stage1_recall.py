# -*- coding: utf-8 -*-
"""Stage-1 recall tests — determinism, budget, anchors, output contract."""

import json

import numpy as np
import pytest

from src.pipeline.stage1_recall import recallCandidates


def _sparseFeatures(nSecs=3600, step=60, seed=0):
    rng = np.random.RandomState(seed)
    rms = rng.rand(nSecs) * 0.2
    dmDensity = rng.rand(nSecs) * 0.2
    for t in range(20, nSecs, step):
        rms[t:t + 3] += 5.0
        dmDensity[t:t + 5] += 2.0
    return {
        "audioFeatures": {"features": {"rms": rms.tolist()}},
        "textFeatures": {"features": {"dmDensity": dmDensity.tolist()}},
    }


class TestHourlyRecallBudget:
    def test_one_hour_produces_between_30_and_80_candidates(self):
        features = _sparseFeatures()
        result = recallCandidates(durationSec=3600, **features)
        assert 30 <= len(result["candidates"]) <= 80
        assert result["stats"]["candidateCount"] == len(result["candidates"])
        assert result["stats"]["candidateCap"] == 60

        for index, candidate in enumerate(result["candidates"], start=1):
            assert candidate["id"] == index
            assert candidate["start"] < candidate["end"]
            assert candidate["duration"] >= 15.0
            assert candidate["duration"] <= 90.0
            assert candidate["sources"]
            assert "rms" in candidate["sources"] or "dmDensity" in candidate["sources"]

    def test_recall_per_hour_bounds_the_output(self):
        features = _sparseFeatures(step=30)
        result = recallCandidates(
            durationSec=3600, recallPerHour=30, **features
        )
        assert len(result["candidates"]) <= 30

    def test_short_recording_uses_whole_source(self):
        result = recallCandidates(
            durationSec=10,
            audioFeatures={"features": {"rms": [0.0, 1.0, 0.0]}},
            minDuration=15,
            maxDuration=90,
        )
        assert result["candidates"]
        assert result["candidates"][0]["start"] == 0.0
        assert result["candidates"][0]["end"] == 10.0


class TestAnchors:
    def test_keyword_anchor_is_kept_even_without_composite_peak(self):
        n = 600
        keyword = np.zeros(n)
        keyword[200] = 1.0
        result = recallCandidates(
            durationSec=n,
            textFeatures={"features": {"asrKeyword": keyword.tolist()}},
            recallThreshold=5.0,
        )
        assert result["candidates"]
        assert any("asrKeyword" in c["sources"] for c in result["candidates"])
        assert any(c["anchor"] == "asrKeyword" for c in result["candidates"])

    def test_constant_signals_do_not_invent_candidates(self):
        result = recallCandidates(
            durationSec=120,
            audioFeatures={"features": {"rms": [0.1] * 120}},
            textFeatures={"features": {"dmDensity": [1.0] * 120}},
        )
        assert result["candidates"] == []


class TestRecallDocument:
    def test_deterministic_output(self):
        features = _sparseFeatures(nSecs=900, step=60)
        first = recallCandidates(durationSec=900, **features)
        second = recallCandidates(durationSec=900, **features)
        assert first["candidates"] == second["candidates"]
        assert first["recallCurve"] == second["recallCurve"]

    def test_writes_json_document(self, tmp_path):
        path = tmp_path / "recall.json"
        features = _sparseFeatures(nSecs=600, step=60)
        result = recallCandidates(
            durationSec=600, outputPath=str(path), **features
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["candidates"] == result["candidates"]
        assert len(saved["recallCurve"]) == 600
        assert saved["stats"]["candidateCount"] == len(result["candidates"])

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="minDuration"):
            recallCandidates(minDuration=90, maxDuration=15)
        with pytest.raises(ValueError, match="recallPerHour"):
            recallCandidates(recallPerHour=0)
        with pytest.raises(ValueError, match="minGapSec"):
            recallCandidates(minGapSec=0)
