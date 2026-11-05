# -*- coding: utf-8 -*-
"""Integration tests for the S5-6 command-line tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.evaluation import DEFAULT_TARGETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _loadScript(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"s56_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluateScript():
    return _loadScript("evaluate")


@pytest.fixture(scope="module")
def tuneScript():
    return _loadScript("tune_weights")


@pytest.fixture(scope="module")
def labelScript():
    return _loadScript("label_report")


def _writeCase(
    root: Path,
    clips: list[dict],
    annotations: list[dict],
    roomId: str = "room",
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    timeTable = root / "time_table.json"
    annotationPath = root / "annotations.json"
    timeTable.write_text(
        json.dumps(
            {
                "roomId": roomId,
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
                "roomId": roomId,
                "duration": 120,
                "profile": "default",
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


class TestEvaluateScript:
    def test_evaluate_case_reports_metrics(self, evaluateScript, tmp_path):
        timeTable, annotationPath = _writeCase(
            tmp_path / "case",
            [
                {"id": 1, "start": 10, "end": 40, "score": 9},
                {"id": 2, "start": 70, "end": 95, "score": 8},
            ],
            [
                {"id": "a1", "start": 10, "end": 40, "quality": 5},
                {"id": "a2", "start": 70, "end": 95, "quality": 4},
            ],
        )

        case = evaluateScript.evaluateCase(
            str(timeTable),
            str(annotationPath),
            "case",
            [1, 2],
            0.0,
            0.5,
            3.0,
            DEFAULT_TARGETS,
        )

        assert case["label"] == "case"
        assert case["clipCount"] == 2
        assert case["positiveAnnotations"] == 2
        assert case["metrics"]["1"]["precisionAtK"] == 1.0
        assert case["metrics"]["1"]["recallAtK"] == 0.5
        assert case["metrics"]["2"]["recallAtK"] == 1.0
        assert case["metrics"]["2"]["targets"]["recallAtK"]["pass"] is True

    def test_positive_count_uses_requested_min_quality(
        self, evaluateScript, tmp_path
    ):
        timeTable, annotationPath = _writeCase(
            tmp_path / "case",
            [{"id": 1, "start": 10, "end": 40, "score": 9}],
            [
                {"id": "a1", "start": 10, "end": 40, "quality": 5},
                {"id": "a2", "start": 70, "end": 95, "quality": 4},
            ],
        )

        case = evaluateScript.evaluateCase(
            str(timeTable),
            str(annotationPath),
            "case",
            [1],
            0.0,
            0.5,
            5.0,
            DEFAULT_TARGETS,
        )

        assert case["positiveAnnotations"] == 1
        assert case["negativeAnnotations"] == 1

    def test_main_writes_json_and_returns_zero(self, evaluateScript, tmp_path, capsys):
        timeTable, annotationPath = _writeCase(
            tmp_path / "case",
            [{"id": 1, "start": 10, "end": 40, "score": 9}],
            [{"id": "a1", "start": 10, "end": 40, "quality": 5}],
        )
        outputPath = tmp_path / "report.json"

        code = evaluateScript.main(
            [
                "--time-table",
                str(timeTable),
                "--annotations",
                str(annotationPath),
                "--name",
                "single",
                "--k",
                "1",
                "--json",
                str(outputPath),
                "--quiet",
            ]
        )

        assert code == 0
        report = json.loads(outputPath.read_text(encoding="utf-8"))
        assert report["cases"][0]["label"] == "single"
        assert report["aggregates"]["1"]["micro"]["precisionAtK"] == 1.0
        assert capsys.readouterr().out == ""

    def test_fail_on_target_returns_one(self, evaluateScript, tmp_path):
        timeTable, annotationPath = _writeCase(
            tmp_path / "case",
            [{"id": 1, "start": 10, "end": 40, "score": 9}],
            [
                {"id": "a1", "start": 10, "end": 40, "quality": 5},
                {"id": "a2", "start": 70, "end": 95, "quality": 5},
            ],
        )
        targetsPath = tmp_path / "targets.json"
        targetsPath.write_text(
            json.dumps({"recallAtK": 0.9}), encoding="utf-8"
        )

        code = evaluateScript.main(
            [
                "--case",
                str(timeTable),
                str(annotationPath),
                "--k",
                "1",
                "--targets",
                str(targetsPath),
                "--fail-on-target",
                "--quiet",
            ]
        )

        assert code == 1

    def test_mismatched_pairs_are_rejected(self, evaluateScript, tmp_path, capsys):
        with pytest.raises(SystemExit):
            evaluateScript.main(
                [
                    "--time-table",
                    str(tmp_path / "a.json"),
                    "--annotations",
                    str(tmp_path / "a.annotations.json"),
                    "--annotations",
                    str(tmp_path / "b.annotations.json"),
                ]
            )


class TestTuneWeightsScript:
    def test_load_case_and_score_cached_features(self, tuneScript, tmp_path):
        _writeSyntheticFeatures(tmp_path)
        timeTable, annotationPath = _writeCase(
            tmp_path,
            [],
            [
                {"id": "a1", "start": 11, "end": 28, "quality": 5},
                {"id": "a2", "start": 66, "end": 79, "quality": 4},
            ],
        )

        case = tuneScript.loadCase(str(timeTable), str(annotationPath))
        result = tuneScript.scoreCase(case, "default")

        assert len(case["audio"]["rms"]) == 120
        assert case["durationSec"] == 120.0
        assert len(result["clips"]) >= 1

    def test_evaluate_candidate_and_patched_profile_restore(
        self, tuneScript, tmp_path
    ):
        _writeSyntheticFeatures(tmp_path)
        timeTable, annotationPath = _writeCase(
            tmp_path,
            [],
            [
                {"id": "a1", "start": 11, "end": 28, "quality": 5},
                {"id": "a2", "start": 66, "end": 79, "quality": 4},
            ],
        )
        case = tuneScript.loadCase(str(timeTable), str(annotationPath))
        baseline = tuneScript.normalizeWeights(
            dict(tuneScript.scorer.PROFILES["default"])
        )
        before = dict(tuneScript.scorer.PROFILES["default"])

        result = tuneScript.evaluateCandidate(
            baseline,
            [case],
            "default",
            2,
            "f1",
            "micro",
            0.0,
            0.5,
            3.0,
            DEFAULT_TARGETS,
        )

        assert result["objective"] >= 0
        assert result["aggregate"]["micro"]["recallAtK"] > 0
        assert tuneScript.scorer.PROFILES["default"] == before

    def test_main_writes_yaml_and_json(self, tuneScript, tmp_path):
        _writeSyntheticFeatures(tmp_path)
        timeTable, annotationPath = _writeCase(
            tmp_path,
            [],
            [
                {"id": "a1", "start": 11, "end": 28, "quality": 5},
                {"id": "a2", "start": 66, "end": 79, "quality": 4},
            ],
        )
        yamlPath = tmp_path / "tuned.yaml"
        jsonPath = tmp_path / "tune.json"

        code = tuneScript.main(
            [
                "--case",
                str(timeTable),
                str(annotationPath),
                "--profile",
                "default",
                "--k",
                "2",
                "--strategy",
                "random",
                "--iterations",
                "3",
                "--seed",
                "1",
                "--output-yaml",
                str(yamlPath),
                "--json",
                str(jsonPath),
                "--quiet",
            ]
        )

        assert code == 0
        yamlText = yamlPath.read_text(encoding="utf-8")
        assert "default:" in yamlText
        report = json.loads(jsonPath.read_text(encoding="utf-8"))
        assert report["profile"] == "default"
        assert report["evaluated"] == 3
        assert report["best"]["weights"]
        assert sum(report["best"]["weights"].values()) == pytest.approx(1.0, abs=1e-4)


class TestLabelReportScript:
    def test_build_report_without_video(self, labelScript, tmp_path):
        timeTable, annotationPath = _writeCase(
            tmp_path / "room",
            [
                {
                    "id": 1,
                    "start": 10,
                    "end": 40,
                    "score": 9.5,
                    "title": "高能 <时刻>",
                    "tags": ["高能"],
                    "trigger": "哈哈",
                },
                {"id": 2, "start": 70, "end": 95, "score": 8.0},
            ],
            [{"id": "a1", "start": 10, "end": 40, "quality": 4}],
        )

        summary = labelScript.buildReport(
            str(timeTable),
            str(annotationPath),
            None,
            str(tmp_path / "report"),
            2,
            3.0,
            False,
            False,
        )

        htmlText = Path(summary["html"]).read_text(encoding="utf-8")
        template = json.loads(
            Path(summary["template"]).read_text(encoding="utf-8")
        )
        assert summary["candidates"] == 2
        assert summary["thumbnails"] == 0
        assert "高能 &lt;时刻&gt;" in htmlText
        assert "data-start=\"10.000\"" in htmlText
        assert 'data-id="a1"' in htmlText
        assert "值得切" in htmlText
        assert len(template["annotations"]) == 1
        assert len(template["candidates"]) == 2
        assert template["candidates"][0]["id"] == "c1"

    def test_empty_time_table_still_writes_page(self, labelScript, tmp_path):
        timeTable, annotationPath = _writeCase(tmp_path / "room", [], [])

        summary = labelScript.buildReport(
            str(timeTable),
            str(annotationPath),
            None,
            str(tmp_path / "report"),
            5,
            3.0,
            False,
            False,
        )

        htmlText = Path(summary["html"]).read_text(encoding="utf-8")
        assert summary["candidates"] == 0
        assert "没有候选片段" in htmlText
