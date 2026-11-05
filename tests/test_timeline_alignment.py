# -*- coding: utf-8 -*-
"""Timeline contract tests — feature arrays must share one 1 Hz timeline."""

import json

import numpy as np

from src.scorer import alignFeatureArrays, score


class TestAlignFeatureArrays:
    def test_short_text_aligned_to_audio_length(self):
        features = {
            "rms": np.arange(20, dtype=np.float64),
            "dmDensity": np.arange(17, dtype=np.float64),
            "asrKeyword": np.ones(17, dtype=np.float64),
        }

        aligned, report, nSecs = alignFeatureArrays(features)

        assert nSecs == 20
        assert all(len(values) == 20 for values in aligned.values())
        assert len(report) == 2
        byName = {item["name"]: item for item in report}
        assert byName["dmDensity"]["method"] == "interpolate"
        assert byName["asrKeyword"]["method"] == "pad_zeros"
        # Event signal must not invent activity in the padded tail.
        assert aligned["asrKeyword"][-3:].tolist() == [0.0, 0.0, 0.0]
        # Continuous signal preserves its last observed value at the edge.
        assert aligned["dmDensity"][-1] == 16.0

    def test_long_signal_truncated(self):
        features = {
            "rms": np.arange(30, dtype=np.float64),
            "dmDensity": np.arange(20, dtype=np.float64),
        }
        aligned, report, nSecs = alignFeatureArrays(features)
        assert nSecs == 30
        assert len(aligned["dmDensity"]) == 30
        assert aligned["dmDensity"][-1] == 19.0
        assert {item["method"] for item in report} == {"interpolate"}

    def test_explicit_target_length(self):
        features = {"rms": np.arange(10, dtype=np.float64)}
        aligned, report, nSecs = alignFeatureArrays(features, targetLength=25)
        assert nSecs == 25
        assert len(aligned["rms"]) == 25
        assert report[0]["oldLen"] == 10
        assert report[0]["newLen"] == 25

    def test_single_value_signal(self):
        aligned, _, nSecs = alignFeatureArrays(
            {"rms": np.array([3.0]), "dmDensity": np.array([1.0])},
            targetLength=5,
        )
        assert nSecs == 5
        assert aligned["rms"].tolist() == [3.0] * 5

    def test_empty_features(self):
        assert alignFeatureArrays({}) == ({}, [], 0)


class TestScoreAlignment:
    def test_all_signals_participate_after_alignment(self, tmp_path):
        """17s text + 20s audio/visual must all contribute at 20s."""
        n = 20
        rms = np.full(n, 0.1, dtype=np.float64)
        rms[6:8] = 5.0
        dmDensity = np.zeros(17, dtype=np.float64)
        dmDensity[5:8] = 20.0
        visual = {"sceneChange": [0] * n, "motion": [0.1] * n}

        scoresPath = tmp_path / "scores.json"
        result = score(
            roomId="align-test",
            audioFeatures={"features": {"rms": rms.tolist()}},
            textFeatures={"features": {"dmDensity": dmDensity.tolist()}},
            visualFeatures={"features": visual},
            sensitivity=1.0,
            minDuration=15.0,
            maxDuration=60.0,
            durationSec=20.0,
            scoresPath=str(scoresPath),
        )

        assert result["duration"] == 20
        assert {item["name"] for item in result["alignment"]} == {"dmDensity"}
        assert result["alignment"][0]["newLen"] == 20

        saved = json.loads(scoresPath.read_text(encoding="utf-8"))
        assert "dmDensity" in saved["signalStats"]
        assert saved["weights"]["dmDensity"] > 0
        assert "dmDensity" in saved["topSignals"]

    def test_duration_sec_pads_all_features(self):
        result = score(
            audioFeatures={"features": {"rms": [0.1, 0.2, 0.3]}},
            sensitivity=5.0,
            minDuration=5.0,
            maxDuration=20.0,
            durationSec=8.0,
        )
        assert result["duration"] == 8
        assert len(result["alignment"]) == 1
        assert result["alignment"][0]["newLen"] == 8

    def test_all_canonical_features_are_visible(self, tmp_path):
        """Every extractor feature must survive alignment and appear in stats."""
        from src.util import FEATURE_NAMES

        n = 40
        textNames = [
            "dmDensity", "dmAcceleration", "dmEntropy", "dmSentiment",
            "dmUniqueUsers", "dmUtr", "dmRepeatRate", "dmQuestionRate",
            "dmExclamationRate", "dmEmojiRate", "keywordHighEnergy",
            "keywordFunny", "keywordControversy", "keywordInteractive",
            "keywordEmotional", "keywordShopping", "asrKeyword",
            "topicChange", "speakerChange",
        ]
        audioNames = [
            "rms", "spectralCentroid", "mfccDist", "zcr",
            "eventLaughter", "eventApplause", "eventMusic",
        ]
        visualNames = ["sceneChange", "motion", "faceCount"]

        def makeSignals(names, length):
            out = {}
            for i, name in enumerate(names):
                arr = np.zeros(length, dtype=np.float64)
                arr[10:14] = float(i + 1)
                out[name] = arr.tolist()
            return out

        scoresPath = tmp_path / "scores.json"
        result = score(
            roomId="all-features",
            audioFeatures={"features": makeSignals(audioNames, n)},
            textFeatures={"features": makeSignals(textNames, n - 7)},
            visualFeatures={"features": makeSignals(visualNames, n)},
            sensitivity=1.0,
            minDuration=10.0,
            maxDuration=30.0,
            durationSec=float(n),
            scoresPath=str(scoresPath),
        )

        assert result["duration"] == n
        alignedText = {
            item["name"] for item in result["alignment"]
            if item["name"] in textNames
        }
        assert alignedText == set(textNames)

        saved = json.loads(scoresPath.read_text(encoding="utf-8"))
        assert set(saved["signalStats"]) == set(FEATURE_NAMES)
        assert all(
            "weight" in stats and "meanAbsContribution" in stats
            for stats in saved["signalStats"].values()
        )
        assert all(name in saved["weights"] for name in saved["topSignals"])

