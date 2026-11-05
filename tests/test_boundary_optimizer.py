# -*- coding: utf-8 -*-
"""Content-aware boundary optimization tests."""

import numpy as np

from src.pipeline.boundary_optimizer import (
    inferContentType,
    optimizeBoundaries,
    targetDurationRange,
)


class TestContentType:
    def test_profile_wins_for_shopping_and_talent(self):
        assert inferContentType("shopping", ["speakerChange"]) == "shopping"
        assert inferContentType("talent", ["eventLaughter"]) == "talent"

    def test_semantic_signals_select_specific_types(self):
        assert inferContentType("default", ["speakerChange"]) == "interaction"
        assert inferContentType("default", ["eventLaughter"]) == "funny"
        assert inferContentType("default", ["eventMusic"]) == "talent"

    def test_game_and_default_fallbacks(self):
        assert inferContentType("game", []) == "game"
        assert inferContentType("game", ["eventLaughter"]) == "game"
        assert inferContentType("default", ["rms"]) == "default"


class TestTargetDuration:
    def test_presets_are_clamped_by_user_bounds(self):
        assert targetDurationRange("default", 15, 90) == (15.0, 90.0)
        assert targetDurationRange("funny", 15, 90) == (15.0, 35.0)
        assert targetDurationRange("shopping", 15, 90) == (20.0, 60.0)
        assert targetDurationRange("interaction", 15, 90) == (30.0, 90.0)
        assert targetDurationRange("talent", 15, 90) == (45.0, 90.0)
        assert targetDurationRange("funny", 20, 30) == (20.0, 30.0)

    def test_impossible_preset_falls_back_to_user_bounds(self):
        assert targetDurationRange("interaction", 10, 20) == (10.0, 20.0)

    def test_platform_limits_are_soft(self):
        assert targetDurationRange("talent", 15, 90, "douyin") == (45.0, 60.0)
        # No overlap with bilibili's 60-180: keep the content preset.
        assert targetDurationRange("funny", 15, 90, "bilibili") == (15.0, 35.0)
        assert targetDurationRange("default", 15, 90, "none") == (15.0, 90.0)


class TestBoundaryOptimization:
    def _signals(self, n=200):
        rms = np.zeros(n)
        rms[48:78] = 5.0
        sceneChange = np.zeros(n)
        sceneChange[96] = 1.0
        return {"rms": rms, "sceneChange": sceneChange}

    def test_snaps_end_to_asr_sentence_boundary(self):
        n = 200
        rms = np.zeros(n)
        rms[45:75] = 5.0
        curve = rms.copy()
        result = optimizeBoundaries(
            48, 82, {"rms": rms}, curve,
            asrSegments=[{"start": 40, "end": 80}],
            targetMin=30, targetMax=90, searchSec=10,
            contentType="interaction",
        )
        assert result["changed"] is True
        assert result["end"] == 80.0
        assert result["endSource"] == "asrSentence"
        assert "boundary_asrSentence" in result["boundaryReasons"]

    def test_snaps_start_to_asr_sentence_boundary(self):
        n = 200
        rms = np.zeros(n)
        rms[45:75] = 5.0
        result = optimizeBoundaries(
            48, 82, {"rms": rms}, rms,
            asrSegments=[{"start": 50, "end": 82}],
            targetMin=30, targetMax=90, searchSec=10,
            contentType="interaction",
        )
        assert result["start"] == 50.0
        assert result["startSource"] == "asrSentence"

    def test_uses_silence_minimum(self):
        n = 120
        rms = np.ones(n)
        rms[49] = 0.0
        rms[50] = 0.0
        result = optimizeBoundaries(
            48, 78, {"rms": rms}, rms,
            targetMin=25, targetMax=90, searchSec=8,
        )
        assert result["startSource"] == "silence"

    def test_uses_scene_cut_at_end(self):
        n = 200
        signals = self._signals(n)
        curve = np.zeros(n)
        result = optimizeBoundaries(
            58, 92, signals, curve,
            targetMin=30, targetMax=90, searchSec=10,
        )
        assert result["end"] == 96.0
        assert result["endSource"] == "sceneChange"

    def test_duration_penalty_prevents_overshrinking(self):
        n = 200
        rms = np.zeros(n)
        rms[40:90] = 5.0
        result = optimizeBoundaries(
            48, 82, {"rms": rms}, rms,
            asrSegments=[{"start": 60, "end": 80}],
            targetMin=30, targetMax=35, searchSec=10,
            contentType="interaction",
        )
        assert 30.0 <= result["duration"] <= 35.0

    def test_no_semantic_candidates_keeps_original(self):
        n = 120
        flat = np.zeros(n)
        result = optimizeBoundaries(
            40, 70, {"rms": flat}, flat,
            targetMin=15, targetMax=90, searchSec=8,
        )
        assert result["changed"] is False
        assert result["start"] == 40.0
        assert result["end"] == 70.0

    def test_zero_search_disables_optimization(self):
        n = 120
        rms = np.zeros(n)
        rms[50] = 5.0
        result = optimizeBoundaries(
            40, 70, {"rms": rms}, rms,
            asrSegments=[{"start": 45, "end": 70}],
            searchSec=0,
        )
        assert result["changed"] is False
        assert result["targetDuration"] == [15.0, 90.0]

    def test_is_deterministic(self):
        n = 200
        signals = self._signals(n)
        first = optimizeBoundaries(48, 82, signals, signals["rms"], searchSec=10)
        second = optimizeBoundaries(48, 82, signals, signals["rms"], searchSec=10)
        assert first == second
