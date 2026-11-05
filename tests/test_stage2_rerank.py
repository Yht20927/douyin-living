# -*- coding: utf-8 -*-
"""Stage-2 rerank and two-stage orchestration tests."""

import json

import numpy as np

from src.pipeline.stage2_rerank import rerankCandidates, twoStageScore
from tests.fixtures.synthetic import makeSyntheticFeatures


def _source(features):
    return {
        "audioFeatures": features["audio"],
        "textFeatures": features["text"],
        "visualFeatures": features["visual"],
    }


class TestTwoStageContract:
    def test_produces_legacy_compatible_time_table(self, tmp_path):
        features = makeSyntheticFeatures(durationSec=90, peakStarts=(20, 55))
        timeTable = tmp_path / "time_table.json"
        scores = tmp_path / "scores.json"
        candidates = tmp_path / "recall.json"

        result = twoStageScore(
            roomId="synthetic-90s",
            durationSec=90.0,
            minDuration=15.0,
            maxDuration=60.0,
            topK=5,
            outputPath=str(timeTable),
            scoresPath=str(scores),
            candidatesPath=str(candidates),
            **_source(features),
        )

        assert result["pipeline"] == "two_stage"
        assert result["duration"] == 90
        assert result["clips"], "synthetic highlights must rerank into clips"
        assert result["stages"]["recall"]["candidateCount"] >= 1
        assert result["stages"]["rerank"]["inputCount"] >= 1
        assert isinstance(result["stages"]["totalElapsedMs"], int)

        for clip in result["clips"]:
            assert clip["candidateSource"] == "two_stage"
            assert 15.0 <= clip["duration"] <= 60.0
            assert 0.0 <= clip["confidence"] <= 1.0
            assert clip["trigger"]
            assert isinstance(clip["reasons"], list)
            assert isinstance(clip["risks"], list)
            assert clip["dropReasons"] == []
            assert clip["windowFeatures"]["activeSignals"]
            assert "peakScore" in clip["windowFeatures"]
            assert "signals" in clip["windowFeatures"]
            assert clip["breakdown"]

        savedTable = json.loads(timeTable.read_text(encoding="utf-8"))
        assert savedTable["clips"] == result["clips"]
        assert savedTable["durationPolicy"] == {
            "dynamic": True,
            "boundarySearchSec": 8.0,
            "platform": "none",
        }
        savedScores = json.loads(scores.read_text(encoding="utf-8"))
        assert len(savedScores["scores"]) == 90
        assert savedScores["pipeline"] == "two_stage"
        assert savedScores["candidates"]
        assert "stages" in savedScores
        assert json.loads(candidates.read_text(encoding="utf-8"))["candidates"]

    def test_is_deterministic(self):
        features = makeSyntheticFeatures()
        kwargs = _source(features)
        first = twoStageScore(durationSec=90.0, minDuration=15.0, maxDuration=60.0, **kwargs)
        second = twoStageScore(durationSec=90.0, minDuration=15.0, maxDuration=60.0, **kwargs)
        assert [(c["start"], c["end"], c["score"]) for c in first["clips"]] == [
            (c["start"], c["end"], c["score"]) for c in second["clips"]
        ]

    def test_empty_features_return_empty_result(self):
        result = twoStageScore(durationSec=90.0)
        assert result["clips"] == []
        assert result["stages"]["recall"]["candidateCount"] == 0


class TestRerankRanking:
    def test_strong_window_outranks_noise(self):
        n = 300
        rng = np.random.RandomState(7)
        rms = rng.rand(n) * 0.1
        dmDensity = rng.rand(n) * 0.1
        eventLaughter = np.zeros(n)
        asrKeyword = np.zeros(n)

        # Strong highlight around 60s.
        rms[55:65] += 5.0
        dmDensity[54:68] += 15.0
        eventLaughter[58:64] = 1.0
        asrKeyword[60] = 1.0

        candidates = [
            {"id": 1, "start": 45.0, "end": 75.0, "score": 1.0, "sources": ["rms"]},
            {"id": 2, "start": 205.0, "end": 235.0, "score": 1.0, "sources": ["rms"]},
        ]
        result = rerankCandidates(
            candidates,
            durationSec=n,
            audioFeatures={"features": {
                "rms": rms.tolist(),
                "eventLaughter": eventLaughter.tolist(),
            }},
            textFeatures={"features": {
                "dmDensity": dmDensity.tolist(),
                "asrKeyword": asrKeyword.tolist(),
            }},
            minDuration=15.0,
            maxDuration=60.0,
            topK=5,
        )
        assert len(result["clips"]) == 2
        strong = next(c for c in result["clips"] if c["start"] < 100)
        quiet = next(c for c in result["clips"] if c["start"] > 100)
        assert strong["score"] > quiet["score"]
        assert "asrKeyword" in strong["reasons"]
        assert "eventLaughter" not in quiet["reasons"]

    def test_top_k_cut_is_recorded(self):
        n = 400
        rms = np.zeros(n)
        for t in (40, 140, 240, 340):
            rms[t:t + 4] = 5.0
        candidates = [
            {"id": i, "start": t - 15, "end": t + 15, "score": 1.0, "sources": ["rms"]}
            for i, t in enumerate((40, 140, 240, 340), start=1)
        ]
        result = rerankCandidates(
            candidates,
            durationSec=n,
            audioFeatures={"features": {"rms": rms.tolist()}},
            minDuration=15.0,
            maxDuration=60.0,
            topK=2,
        )
        assert len(result["clips"]) == 2
        assert any(item["reason"] == "top_k_cut" for item in result["dropped"])

    def test_overlap_is_suppressed(self):
        n = 300
        rms = np.zeros(n)
        rms[100:120] = 5.0
        candidates = [
            {"id": 1, "start": 90.0, "end": 120.0, "score": 2.0, "sources": ["rms"]},
            {"id": 2, "start": 95.0, "end": 125.0, "score": 1.0, "sources": ["rms"]},
        ]
        result = rerankCandidates(
            candidates,
            durationSec=n,
            audioFeatures={"features": {"rms": rms.tolist()}},
            minDuration=15.0,
            maxDuration=60.0,
            topK=5,
        )
        assert len(result["clips"]) == 1
        assert any(item["reason"] == "overlap" for item in result["dropped"])

    def test_duration_is_clamped_and_adjustment_recorded(self):
        n = 200
        rms = np.zeros(n)
        rms[50:100] = 2.0
        result = rerankCandidates(
            [{"start": 10.0, "end": 150.0, "score": 1.0, "sources": ["rms"]}],
            durationSec=n,
            audioFeatures={"features": {"rms": rms.tolist()}},
            minDuration=15.0,
            maxDuration=30.0,
        )
        clip = result["clips"][0]
        assert clip["duration"] <= 30.0
        assert any(
            item["reason"] == "stage2_duration_fit"
            for item in result["adjustments"]
        )
        assert "window_adjusted" in clip["risks"]

    def test_keyword_category_is_ranked_and_tagged(self):
        n = 200
        keywordShopping = np.zeros(n)
        keywordShopping[50:80] = 1.0
        result = rerankCandidates(
            [{
                "start": 50.0,
                "end": 80.0,
                "score": 1.0,
                "sources": ["keywordShopping"],
            }],
            durationSec=n,
            textFeatures={"features": {
                "keywordShopping": keywordShopping.tolist(),
            }},
            minDuration=15.0,
            maxDuration=60.0,
            boundarySearchSec=0.0,
        )
        clip = result["clips"][0]
        assert "keywordShopping" in clip["reasons"]
        assert "带货" in clip["tags"]
        assert clip["title"].startswith("🛒")
        assert clip["score"] > 0

    def test_missing_asr_is_reported_as_risk(self):
        n = 120
        rms = np.zeros(n)
        rms[50:60] = 5.0
        result = rerankCandidates(
            [{"start": 45.0, "end": 65.0, "score": 1.0, "sources": ["rms"]}],
            durationSec=n,
            audioFeatures={"features": {"rms": rms.tolist()}},
            minDuration=10.0,
            maxDuration=30.0,
        )
        assert "asr_missing" in result["clips"][0]["risks"]

    def test_invalid_durations_raise(self):
        import pytest

        with pytest.raises(ValueError, match="minDuration"):
            rerankCandidates([], minDuration=90.0, maxDuration=15.0)


class TestDynamicDurationAndBoundaries:
    def _rank(self, candidates, n, **overrides):
        rms = np.zeros(n)
        for candidate in candidates:
            start = int(candidate["start"])
            end = int(candidate["end"])
            rms[start:end] = 5.0
        audioFeatures = {"features": {
            "rms": rms.tolist(),
            "eventLaughter": overrides.pop("eventLaughter", np.zeros(n)).tolist(),
            "eventMusic": overrides.pop("eventMusic", np.zeros(n)).tolist(),
        }}
        textFeatures = {"features": {
            "asrKeyword": overrides.pop("asrKeyword", np.zeros(n)).tolist(),
        }}
        kwargs = {
            "durationSec": n,
            "audioFeatures": audioFeatures,
            "textFeatures": textFeatures,
            "minDuration": 15.0,
            "maxDuration": 120.0,
            "topK": 10,
            **overrides,
        }
        return rerankCandidates(candidates, **kwargs)

    def test_content_type_selects_dynamic_duration(self):
        n = 300
        eventLaughter = np.zeros(n)
        eventLaughter[95:125] = 1.0
        eventMusic = np.zeros(n)
        eventMusic[195:225] = 1.0
        candidates = [
            {"start": 90.0, "end": 120.0, "score": 1.0, "sources": ["eventLaughter"]},
            {"start": 195.0, "end": 225.0, "score": 1.0, "sources": ["eventMusic"]},
        ]
        result = self._rank(
            candidates,
            n,
            eventLaughter=eventLaughter,
            eventMusic=eventMusic,
            dynamicDuration=True,
            boundarySearchSec=0.0,
        )

        funny = next(c for c in result["clips"] if c["start"] < 150)
        talent = next(c for c in result["clips"] if c["start"] >= 150)
        assert funny["contentType"] == "funny"
        assert funny["targetDuration"] == [15.0, 35.0]
        assert talent["contentType"] == "talent"
        assert talent["targetDuration"] == [45.0, 120.0]
        assert talent["duration"] >= 45.0
        assert result["stats"]["contentTypes"] == {"funny": 1, "talent": 1}

    def test_secondary_signal_refines_default_to_funny(self):
        n = 200
        eventLaughter = np.zeros(n)
        eventLaughter[55:70] = 1.0
        result = self._rank(
            [{"start": 50.0, "end": 80.0, "score": 1.0, "sources": ["rms"]}],
            n,
            eventLaughter=eventLaughter,
            dynamicDuration=True,
            boundarySearchSec=0.0,
        )
        clip = result["clips"][0]
        assert clip["contentType"] == "funny"
        assert clip["targetDuration"] == [15.0, 35.0]

    def test_asr_sentence_boundary_is_applied_and_recorded(self):
        n = 200
        asrKeyword = np.zeros(n)
        asrKeyword[50:90] = 1.0
        result = self._rank(
            [{"start": 50.0, "end": 88.0, "score": 1.0, "sources": ["rms"]}],
            n,
            asrKeyword=asrKeyword,
            asrSegments=[{"start": 50.0, "end": 90.0}],
            minDuration=30.0,
            maxDuration=60.0,
            dynamicDuration=False,
            boundarySearchSec=10.0,
        )
        clip = result["clips"][0]
        assert clip["end"] == 90.0
        assert clip["boundary"]["endSource"] == "asrSentence"
        assert "boundary_asrSentence" in clip["boundary"]["reasons"]
        assert any(
            item["reason"] == "boundary_optimized"
            and (
                item["startSource"] == "asrSentence"
                or item["endSource"] == "asrSentence"
            )
            for item in result["adjustments"]
        )

    def test_platform_soft_limit_is_applied_to_talent(self):
        n = 300
        eventMusic = np.zeros(n)
        eventMusic[100:160] = 1.0
        result = self._rank(
            [{"start": 100.0, "end": 160.0, "score": 1.0, "sources": ["eventMusic"]}],
            n,
            eventMusic=eventMusic,
            dynamicDuration=True,
            boundarySearchSec=0.0,
            platform="douyin",
        )
        clip = result["clips"][0]
        assert clip["contentType"] == "talent"
        assert clip["targetDuration"] == [45.0, 60.0]
        assert 45.0 <= clip["duration"] <= 60.0
