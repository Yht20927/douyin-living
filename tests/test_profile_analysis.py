# -*- coding: utf-8 -*-
"""Unit tests for scorer profile support analysis (S5-7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.profile_analysis import (
    casePositiveCount,
    compareProfiles,
    loadCase,
    loadFeatureFile,
    profileDistance,
)


def _writeCase(
    root: Path,
    clips: list[dict],
    annotations: list[dict],
    profile: str = "default",
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    timeTable = root / "time_table.json"
    annotationPath = root / "annotations.json"
    timeTable.write_text(
        json.dumps(
            {
                "roomId": root.name,
                "duration": 120,
                "profile": "default",
                "sensitivity": 2.5,
                "minDuration": 10,
                "maxDuration": 30,
                "clips": clips,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    annotationPath.write_text(
        json.dumps(
            {
                "roomId": root.name,
                "duration": 120,
                "profile": profile,
                "annotations": annotations,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return timeTable, annotationPath


def _writeSyntheticFeatures(root: Path) -> None:
    """Two clear 7-second spikes inside a 120-second timeline."""
    length = 120

    def spike() -> list[float]:
        values = [0.1] * length
        for start, end in ((20, 27), (70, 78)):
            for index in range(start, end):
                values[index] = 6.0
        return values

    root.mkdir(parents=True, exist_ok=True)
    featureSets = {
        "audio_features.json": {
            "sampleRate": 1,
            "duration": length,
            "features": {
                "rms": spike(),
                "eventLaughter": spike(),
                "spectralCentroid": spike(),
            },
        },
        "text_features.json": {
            "sampleRate": 1,
            "duration": length,
            "features": {
                "dmDensity": spike(),
                "dmUniqueUsers": spike(),
                "keywordHighEnergy": spike(),
            },
        },
        "visual_features.json": {
            "sampleRate": 1,
            "duration": length,
            "features": {"motion": spike(), "sceneChange": spike()},
        },
    }
    for name, payload in featureSets.items():
        (root / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )


def _makeCase(root: Path, profile: str) -> dict:
    _writeSyntheticFeatures(root)
    timeTable, annotationPath = _writeCase(
        root,
        [],
        [{"id": "a1", "start": 11, "end": 28, "quality": 5}],
        profile=profile,
    )
    return loadCase(str(timeTable), str(annotationPath))


class TestProfileDistance:
    def test_identical_vectors_have_zero_distance(self):
        weights = {"rms": 0.5, "motion": 0.5}

        distance = profileDistance(weights, dict(weights))

        assert distance["totalVariation"] == 0.0
        assert distance["cosine"] == pytest.approx(1.0)

    def test_disjoint_vectors_have_max_distance(self):
        distance = profileDistance({"rms": 1.0}, {"motion": 1.0})

        assert distance["totalVariation"] == 1.0
        assert distance["cosine"] == 0.0

    def test_distance_is_symmetric(self):
        left = {"rms": 0.7, "motion": 0.3}
        right = {"rms": 0.2, "motion": 0.8}

        assert profileDistance(left, right) == profileDistance(right, left)
        assert profileDistance(left, right)["totalVariation"] == pytest.approx(
            0.5
        )


class TestFeatureLoading:
    def test_missing_file_returns_empty(self, tmp_path):
        assert loadFeatureFile(tmp_path / "missing.json") == {}

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{bad", encoding="utf-8")

        with pytest.raises(ValueError, match=str(path)):
            loadFeatureFile(path)

    def test_non_object_features_raise(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"features": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="must be an object"):
            loadFeatureFile(path)

    def test_load_case_prefers_annotation_profile(self, tmp_path):
        case = _makeCase(tmp_path / "room", "game")

        assert case["profile"] == "game"
        assert case["durationSec"] == 120.0
        assert len(case["audio"]["rms"]) == 120

    def test_load_case_without_features_raises(self, tmp_path):
        timeTable, annotationPath = _writeCase(
            tmp_path / "room",
            [],
            [{"id": "a1", "start": 1, "end": 10, "quality": 5}],
        )

        with pytest.raises(ValueError, match="no non-empty feature files"):
            loadCase(str(timeTable), str(annotationPath))


class TestCasePositiveCount:
    def test_respects_quality_threshold_and_worth_clipping(self):
        case = {
            "annotations": [
                {"quality": 5, "worthClipping": True},
                {"quality": 2, "worthClipping": True},
                {"quality": 5, "worthClipping": False},
            ]
        }

        assert casePositiveCount(case) == 1
        assert casePositiveCount(case, minQuality=2) == 2


class TestCompareProfiles:
    def _cases(self, tmp_path: Path) -> list[dict]:
        return [
            _makeCase(tmp_path / "gameOne", "game"),
            _makeCase(tmp_path / "gameTwo", "game"),
            _makeCase(tmp_path / "shoppingOne", "shopping"),
        ]

    def test_support_counts_and_recommendations(self, tmp_path):
        report = compareProfiles(
            self._cases(tmp_path),
            profiles=["default", "game", "shopping", "talent"],
            k=2,
            minCases=2,
            minPositives=2,
            minDelta=-1.0,
        )

        game = report["profiles"]["game"]
        assert game["caseCount"] == 2
        assert game["positiveCount"] == 2
        assert game["sufficient"] is True
        assert game["recommendation"] == "keep"
        assert game["ownedCases"] == ["gameOne", "gameTwo"]
        assert game["weightSum"] == pytest.approx(1.0)
        assert game["topFeatures"]

        shopping = report["profiles"]["shopping"]
        assert shopping["caseCount"] == 1
        assert shopping["sufficient"] is False
        assert shopping["recommendation"] == "collect_more_or_merge"

        default = report["profiles"]["default"]
        assert default["caseCount"] == 0
        assert default["aggregate"] is None
        assert default["recommendation"] == "keep"

        assert "game" in report["summary"]["keep"]
        assert "shopping" in report["summary"]["collectMoreOrMerge"]

    def test_sufficient_but_flat_profile_is_merged(self, tmp_path):
        report = compareProfiles(
            self._cases(tmp_path),
            profiles=["default", "game"],
            k=2,
            minCases=2,
            minPositives=2,
            minDelta=1.0,
        )

        game = report["profiles"]["game"]
        assert game["sufficient"] is True
        assert game["deltaVsDefault"] is not None
        assert game["recommendation"] == "merge_into_default"
        assert "game" in report["summary"]["mergeIntoDefault"]

    def test_case_diagnostics_cover_every_profile(self, tmp_path):
        report = compareProfiles(
            self._cases(tmp_path),
            profiles=["default", "game"],
            k=2,
            minCases=1,
            minPositives=1,
        )

        assert len(report["cases"]) == 3
        for case in report["cases"]:
            assert set(case["metrics"]) == {"default", "game"}
            assert case["bestProfile"] in {"default", "game"}
            assert case["positives"] == 1

    def test_distances_cover_every_pair(self, tmp_path):
        profiles = ["default", "game", "shopping", "talent"]
        report = compareProfiles(
            self._cases(tmp_path),
            profiles=profiles,
            k=2,
            minCases=1,
            minPositives=1,
        )

        assert len(report["distances"]) == 6
        pairs = {
            frozenset((item["left"], item["right"]))
            for item in report["distances"]
        }
        assert len(pairs) == 6

    def test_duplicate_profile_names_are_deduped(self, tmp_path):
        report = compareProfiles(
            self._cases(tmp_path),
            profiles=["default", "game", "default"],
            k=2,
            minCases=1,
            minPositives=1,
        )

        assert list(report["profiles"]) == ["default", "game"]
        assert len(report["distances"]) == 1

    def test_duplicate_labels_raise(self, tmp_path):
        cases = self._cases(tmp_path)
        cases[1]["label"] = cases[0]["label"]

        with pytest.raises(ValueError, match="labels must be unique"):
            compareProfiles(cases, profiles=["default"], k=5)

    def test_unknown_profile_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown profile"):
            compareProfiles(
                self._cases(tmp_path),
                profiles=["default", "missing"],
                k=5,
            )

    def test_case_without_positives_raises(self, tmp_path):
        case = _makeCase(tmp_path / "room", "default")
        case["annotations"] = [
            {"quality": 1, "worthClipping": False}
        ]

        with pytest.raises(ValueError, match="no positive annotations"):
            compareProfiles([case], profiles=["default"], k=5)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"k": 0}, "k must be >= 1"),
            ({"minCases": 0}, "minCases must be >= 1"),
            ({"minPositives": 0}, "minPositives must be >= 1"),
            ({"matchIou": -0.1}, "matchIou"),
            ({"matchIou": 1.1}, "matchIou"),
            ({"duplicateIou": -0.1}, "duplicateIou"),
            ({"duplicateIou": 1.1}, "duplicateIou"),
            ({"minQuality": 0.5}, "minQuality"),
            ({"minQuality": 6}, "minQuality"),
        ],
    )
    def test_invalid_parameters_raise(self, tmp_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            compareProfiles(
                self._cases(tmp_path),
                profiles=["default"],
                **kwargs,
            )

    def test_empty_cases_raise(self):
        with pytest.raises(ValueError, match="no cases"):
            compareProfiles([], profiles=["default"])
