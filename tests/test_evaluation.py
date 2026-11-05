# -*- coding: utf-8 -*-
"""Unit tests for the offline evaluation core (S5-6)."""

import json
import math

import pytest

from src.evaluation import (
    DEFAULT_TARGETS,
    _maxMatching,
    aggregateMetrics,
    checkTargets,
    clipIou,
    computeMetrics,
    isPositive,
    loadAnnotations,
    loadTimeTable,
    normalizeAnnotations,
    normalizeWeights,
    objectiveValue,
    overlapSec,
    rankClips,
)


def _annotation(
    annotationId: str,
    start: float,
    end: float,
    quality: float = 5,
    worthClipping: bool = True,
    annotationType: str = "highlight",
) -> dict:
    return {
        "id": annotationId,
        "start": start,
        "end": end,
        "quality": quality,
        "worthClipping": worthClipping,
        "type": annotationType,
    }


class TestNormalizeAnnotations:
    def test_object_defaults_and_metadata(self):
        result = normalizeAnnotations(
            {
                "roomId": 123,
                "duration": 600,
                "profile": "game",
                "annotations": [{"start": 10, "end": 20}],
            }
        )

        assert result["roomId"] == "123"
        assert result["duration"] == 600.0
        assert result["profile"] == "game"
        assert result["positiveCount"] == 1
        assert result["negativeCount"] == 0
        item = result["annotations"][0]
        assert item["id"] == "a1"
        assert item["worthClipping"] is True
        assert item["quality"] == 5.0
        assert item["type"] == "highlight"
        assert item["notes"] == ""

    def test_bare_list_is_supported(self):
        result = normalizeAnnotations(
            [{"id": "x", "start": 0, "end": 5, "quality": 2}]
        )

        assert len(result["annotations"]) == 1
        assert result["positiveCount"] == 0
        assert result["negativeCount"] == 1

    @pytest.mark.parametrize(
        "value, expected",
        [
            (False, False),
            ("false", False),
            ("0", False),
            ("否", False),
            (True, True),
            ("true", True),
            ("1", True),
            ("是", True),
        ],
    )
    def test_worth_clipping_accepts_json_bool_and_strings(self, value, expected):
        result = normalizeAnnotations(
            [{"id": "x", "start": 0, "end": 5, "worthClipping": value}]
        )

        assert result["annotations"][0]["worthClipping"] is expected

    def test_negative_defaults_to_quality_one(self):
        result = normalizeAnnotations(
            [{"id": "n", "start": 0, "end": 5, "worthClipping": False}]
        )

        assert result["annotations"][0]["quality"] == 1.0
        assert result["negativeCount"] == 1

    @pytest.mark.parametrize(
        "data, message",
        [
            ("not-a-document", "object or a list"),
            ({"annotations": {}}, "must be a list"),
            ({"annotations": ["bad"]}, "must be an object"),
            ({"annotations": [{"start": 0}]}, "end must be a number"),
            (
                {"annotations": [{"start": 0, "end": 5, "worthClipping": "maybe"}]},
                "must be a boolean",
            ),
            (
                {"annotations": [{"start": 0, "end": 5, "quality": 6}]},
                "quality must be within 1..5",
            ),
            (
                {"annotations": [{"start": 10, "end": 10}]},
                "end must be greater than start",
            ),
            (
                {"annotations": [{"start": -1, "end": 5}]},
                "start must be >= 0",
            ),
            ({"duration": -1}, "duration must be >= 0"),
            (
                {
                    "annotations": [
                        {"id": "dup", "start": 0, "end": 5},
                        {"id": "dup", "start": 10, "end": 15},
                    ]
                },
                "duplicate annotation id",
            ),
            ({"duration": "soon"}, "duration must be a number"),
        ],
    )
    def test_invalid_documents_raise(self, data, message):
        with pytest.raises(ValueError, match=message):
            normalizeAnnotations(data)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_values_raise(self, value):
        with pytest.raises(ValueError, match="must be finite"):
            normalizeAnnotations([{"start": 0, "end": value}])


class TestLoadAnnotations:
    def test_loads_valid_file_and_keeps_path(self, tmp_path):
        path = tmp_path / "annotations.json"
        path.write_text(
            json.dumps(
                {
                    "roomId": "r1",
                    "annotations": [{"id": "a1", "start": 0, "end": 10}],
                }
            ),
            encoding="utf-8",
        )

        result = loadAnnotations(path)

        assert result["path"] == str(path)
        assert result["positiveCount"] == 1

    def test_missing_file_raises_with_path(self, tmp_path):
        path = tmp_path / "missing.json"
        with pytest.raises(ValueError, match=str(path)):
            loadAnnotations(path)

    def test_invalid_json_raises_with_path(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(ValueError, match=str(path)):
            loadAnnotations(path)

    def test_schema_error_mentions_path(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"annotations": [{"start": 1}]}), encoding="utf-8")
        with pytest.raises(ValueError, match=str(path)):
            loadAnnotations(path)


class TestLoadTimeTable:
    def test_defaults_missing_clips_to_empty_list(self, tmp_path):
        path = tmp_path / "time_table.json"
        path.write_text(json.dumps({"roomId": "r1"}), encoding="utf-8")

        table = loadTimeTable(path)

        assert table["clips"] == []

    def test_keeps_existing_clips(self, tmp_path):
        path = tmp_path / "time_table.json"
        path.write_text(
            json.dumps({"roomId": "r1", "clips": [{"start": 1, "end": 2}]}),
            encoding="utf-8",
        )

        table = loadTimeTable(path)

        assert len(table["clips"]) == 1

    @pytest.mark.parametrize(
        "payload, message",
        [
            ([], "must be a JSON object"),
            ({"clips": "bad"}, "'clips' must be a list"),
        ],
    )
    def test_invalid_shape_raises(self, tmp_path, payload, message):
        path = tmp_path / "time_table.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            loadTimeTable(path)

    def test_missing_file_raises(self, tmp_path):
        path = tmp_path / "missing.json"
        with pytest.raises(ValueError, match=str(path)):
            loadTimeTable(path)


class TestGeometry:
    def test_overlap_and_iou(self):
        assert overlapSec(0, 10, 5, 15) == 5.0
        assert overlapSec(0, 10, 10, 20) == 0.0
        assert clipIou({"start": 0, "end": 10}, {"start": 0, "end": 10}) == 1.0
        assert clipIou({"start": 0, "end": 10}, {"start": 5, "end": 15}) == pytest.approx(
            1 / 3
        )
        assert clipIou({"start": 0, "end": 10}, {"start": 20, "end": 30}) == 0.0

    def test_is_positive_uses_quality_threshold(self):
        assert isPositive(_annotation("a", 0, 1, quality=3)) is True
        assert isPositive(_annotation("a", 0, 1, quality=2)) is False
        assert isPositive(_annotation("a", 0, 1, quality=5, worthClipping=False)) is False
        assert isPositive(_annotation("a", 0, 1, quality=2), minQuality=2) is True

    def test_rank_clips_orders_by_score_then_start(self):
        clips = [
            {"id": "low", "score": 5, "start": 20},
            {"id": "b", "score": 10, "start": 10},
            {"id": "a", "score": 10, "start": 5},
        ]

        ranked = rankClips(clips)

        assert [clip["id"] for clip in ranked] == ["a", "b", "low"]


class TestMaxMatching:
    def test_augmenting_path_beats_greedy(self):
        """Greedy would give clip A annotation 1 and leave clip B unmatched."""
        positives = [
            _annotation("ann1", 0, 10),
            _annotation("ann2", 5, 15),
        ]
        clips = [
            {"id": "A", "start": 0, "end": 10, "score": 9},
            {"id": "B", "start": 0, "end": 8, "score": 8},
        ]

        pairs, unmatched = _maxMatching(clips, positives, matchIou=0.3)

        assert len(pairs) == 2
        assert unmatched == set()
        byClip = {clipIndex: annotationIndex for clipIndex, annotationIndex, _ in pairs}
        assert byClip == {0: 1, 1: 0}

    def test_no_overlap_leaves_every_clip_unmatched(self):
        pairs, unmatched = _maxMatching(
            [{"id": "A", "start": 0, "end": 5}],
            [_annotation("ann1", 100, 110)],
            matchIou=0.0,
        )

        assert pairs == []
        assert unmatched == {0}


class TestComputeMetrics:
    def test_perfect_match(self):
        clips = [{"id": 1, "start": 10, "end": 20, "score": 5}]
        annotations = [_annotation("a1", 10, 20)]

        metrics = computeMetrics(clips, annotations, k=5)

        assert metrics["precisionAtK"] == 1.0
        assert metrics["precisionAtKStrict"] == 0.2
        assert metrics["recallAtK"] == 1.0
        assert metrics["f1AtK"] == 1.0
        assert metrics["boundaryIou"] == 1.0
        assert metrics["editingCostSec"] == 0.0
        assert metrics["invalidRate"] == 0.0
        assert metrics["duplicateRate"] == 0.0
        assert metrics["negativeHitRate"] is None
        assert metrics["matches"][0]["annotationId"] == "a1"
        assert metrics["unmatchedClips"] == []

    def test_k_truncates_ranked_clips_and_recall_keeps_all_positives(self):
        clips = [
            {"id": "top", "start": 0, "end": 10, "score": 10},
            {"id": "second", "start": 50, "end": 60, "score": 9},
        ]
        annotations = [_annotation("a1", 0, 10), _annotation("a2", 50, 60)]

        metrics = computeMetrics(clips, annotations, k=1)

        assert metrics["clipCount"] == 2
        assert metrics["topKCount"] == 1
        assert metrics["eligibleAtK"] == 1
        assert metrics["precisionAtK"] == 1.0
        assert metrics["recallAtK"] == 0.5
        assert metrics["f1AtK"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["unmatchedClips"] == []
        assert metrics["matchedCount"] == 1

    def test_fewer_clips_than_k_uses_actual_clip_count(self):
        metrics = computeMetrics(
            [{"id": 1, "start": 0, "end": 10, "score": 5}],
            [_annotation("a1", 0, 10)],
            k=5,
        )

        assert metrics["eligibleAtK"] == 1
        assert metrics["precisionAtK"] == 1.0
        assert metrics["precisionAtKStrict"] == 0.2

    def test_unmatched_and_duplicate_clips_are_reported(self):
        clips = [
            {"id": "first", "start": 0, "end": 10, "score": 10},
            {"id": "copy", "start": 2, "end": 8, "score": 9},
        ]
        annotations = [_annotation("a1", 0, 10)]

        metrics = computeMetrics(clips, annotations, k=2)

        assert metrics["matchedCount"] == 1
        assert metrics["precisionAtK"] == 0.5
        assert metrics["invalidRate"] == 0.5
        assert metrics["duplicateRate"] == 0.5
        assert metrics["unmatchedCount"] == 1
        assert metrics["unmatchedClips"][0]["clipId"] == "copy"
        assert metrics["unmatchedClips"][0]["rank"] == 2
        assert metrics["duplicates"][0]["clipId"] == "copy"
        assert metrics["duplicates"][0]["duplicateOfClipId"] == "first"

    def test_negative_hits_do_not_count_as_matches(self):
        clips = [{"id": 1, "start": 25, "end": 28, "score": 5}]
        annotations = [
            _annotation("a1", 0, 10),
            _annotation("n1", 20, 30, quality=1, worthClipping=False, annotationType="dead_air"),
        ]

        metrics = computeMetrics(clips, annotations, k=1)

        assert metrics["precisionAtK"] == 0.0
        assert metrics["recallAtK"] == 0.0
        assert metrics["invalidRate"] == 1.0
        assert metrics["negativeHitRate"] == 1.0
        assert metrics["negativeHits"][0]["annotationId"] == "n1"
        assert metrics["boundaryIou"] is None
        assert metrics["editingCostSec"] is None

    def test_min_quality_reclassifies_low_quality_annotations(self):
        clips = [{"id": 1, "start": 0, "end": 10, "score": 5}]
        annotations = [_annotation("a1", 0, 10, quality=2)]

        with pytest.raises(ValueError, match="no positive highlights"):
            computeMetrics(clips, annotations, k=1)
        metrics = computeMetrics(clips, annotations, k=1, minQuality=2)
        assert metrics["precisionAtK"] == 1.0

    def test_match_iou_gate_rejects_weak_overlap(self):
        clips = [{"id": 1, "start": 0, "end": 10, "score": 5}]
        annotations = [_annotation("a1", 8, 30)]

        loose = computeMetrics(clips, annotations, k=1, matchIou=0.0)
        strict = computeMetrics(clips, annotations, k=1, matchIou=0.5)

        assert loose["matchedCount"] == 1
        assert strict["matchedCount"] == 0
        assert strict["boundaryIou"] is None

    def test_no_clips_reports_zero_precision(self):
        metrics = computeMetrics([], [_annotation("a1", 0, 10)], k=5)

        assert metrics["clipCount"] == 0
        assert metrics["eligibleAtK"] == 0
        assert metrics["precisionAtK"] == 0.0
        assert metrics["recallAtK"] == 0.0
        assert metrics["invalidRate"] == 0.0
        assert metrics["duplicateRate"] == 0.0
        assert metrics["boundaryIou"] is None

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"k": 0}, "k must be >= 1"),
            ({"matchIou": -0.1}, "matchIou"),
            ({"matchIou": 1.1}, "matchIou"),
            ({"duplicateIou": -0.1}, "duplicateIou"),
            ({"duplicateIou": 1.1}, "duplicateIou"),
        ],
    )
    def test_invalid_parameters_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            computeMetrics(
                [{"id": 1, "start": 0, "end": 10, "score": 5}],
                [_annotation("a1", 0, 10)],
                **kwargs,
            )

    def test_no_positive_annotations_raise(self):
        with pytest.raises(ValueError, match="no positive highlights"):
            computeMetrics(
                [{"id": 1, "start": 0, "end": 10, "score": 5}],
                [_annotation("n1", 0, 10, quality=1, worthClipping=False)],
            )


class TestTargetsAndAggregation:
    def test_check_targets_handles_direction_and_missing_values(self):
        checks = checkTargets(
            {
                "precisionAtK": 0.5,
                "recallAtK": 0.8,
                "invalidRate": 0.1,
                "boundaryIou": None,
            },
            {
                "precisionAtK": 0.6,
                "recallAtK": 0.7,
                "invalidRate": 0.2,
                "boundaryIou": 0.5,
            },
        )

        assert checks["precisionAtK"]["pass"] is False
        assert checks["recallAtK"]["pass"] is True
        assert checks["invalidRate"]["pass"] is True
        assert checks["boundaryIou"] == {
            "target": 0.5,
            "value": None,
            "pass": None,
        }

    def test_default_targets_cover_the_plan(self):
        assert set(DEFAULT_TARGETS) == {
            "precisionAtK",
            "recallAtK",
            "boundaryIou",
            "invalidRate",
            "duplicateRate",
        }

    def test_aggregate_reports_macro_and_micro(self):
        caseOne = computeMetrics(
            [{"id": 1, "start": 0, "end": 10, "score": 10}],
            [_annotation("a1", 0, 10), _annotation("a2", 50, 60)],
            k=2,
        )
        caseTwo = computeMetrics(
            [
                {"id": 1, "start": 0, "end": 10, "score": 10},
                {"id": 2, "start": 2, "end": 8, "score": 9},
            ],
            [
                _annotation("a1", 0, 10),
                _annotation(
                    "n1",
                    2,
                    8,
                    quality=1,
                    worthClipping=False,
                    annotationType="dead_air",
                ),
            ],
            k=2,
        )

        aggregate = aggregateMetrics([caseOne, caseTwo], k=2)

        assert aggregate["cases"] == 2
        assert aggregate["macro"]["precisionAtK"] == 0.75
        assert aggregate["macro"]["precisionAtKStrict"] == 0.5
        assert aggregate["macro"]["recallAtK"] == 0.75
        assert aggregate["macro"]["invalidRate"] == 0.25
        assert aggregate["macro"]["duplicateRate"] == 0.25
        assert aggregate["micro"]["precisionAtK"] == pytest.approx(2 / 3, abs=1e-4)
        assert aggregate["micro"]["precisionAtKStrict"] == 0.5
        assert aggregate["micro"]["recallAtK"] == pytest.approx(2 / 3, abs=1e-4)
        assert aggregate["micro"]["invalidRate"] == pytest.approx(1 / 3, abs=1e-4)
        assert aggregate["micro"]["duplicateRate"] == pytest.approx(1 / 3, abs=1e-4)
        assert aggregate["micro"]["boundaryIou"] == 1.0
        assert aggregate["micro"]["negativeHitRate"] == 1.0
        assert aggregate["macroTargets"]["recallAtK"]["pass"] is True
        assert aggregate["microTargets"]["invalidRate"]["pass"] is False

    def test_aggregate_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no case metrics"):
            aggregateMetrics([], k=5)

    def test_objective_value(self):
        metrics = {
            "precisionAtK": 0.5,
            "recallAtK": 0.6,
            "f1AtK": 0.55,
            "boundaryIou": 0.4,
            "editingCostSec": 12.0,
        }

        assert objectiveValue(metrics, "f1") == 0.55
        assert objectiveValue(metrics, "precision") == 0.5
        assert objectiveValue(metrics, "recall") == 0.6
        assert objectiveValue(metrics, "iou") == 0.4
        assert objectiveValue(metrics, "editing") == -12.0
        assert objectiveValue({"editingCostSec": None}, "editing") == -1e9
        assert objectiveValue({"boundaryIou": None}, "iou") == 0.0

        with pytest.raises(ValueError, match="objective must be one of"):
            objectiveValue(metrics, "unknown")


class TestNormalizeWeights:
    def test_drops_unknown_and_negative_and_normalizes(self):
        weights = normalizeWeights(
            {"rms": 2.0, "motion": 1.0, "unknownFeature": 100, "sceneChange": -5}
        )

        assert set(weights) == {"rms", "motion"}
        assert weights["rms"] == pytest.approx(2 / 3)
        assert weights["motion"] == pytest.approx(1 / 3)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_zero_only_weights_raise(self):
        with pytest.raises(ValueError, match="at least one positive"):
            normalizeWeights({"rms": 0, "motion": -1})
