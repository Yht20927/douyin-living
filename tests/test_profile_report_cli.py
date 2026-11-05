# -*- coding: utf-8 -*-
"""Integration tests for scripts/profile_report.py (S5-7)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _loadScript(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"s57_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def profileScript():
    return _loadScript("profile_report")


def _writeSyntheticFeatures(root: Path) -> None:
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


def _writeCase(root: Path, profile: str) -> tuple[Path, Path]:
    _writeSyntheticFeatures(root)
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
                "clips": [],
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
                "annotations": [
                    {"id": "a1", "start": 11, "end": 28, "quality": 5},
                    {"id": "a2", "start": 66, "end": 79, "quality": 4},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return timeTable, annotationPath


class TestLoadCases:
    def test_deduplicates_labels_and_records_inputs(self, profileScript, tmp_path):
        first = _writeCase(tmp_path / "a" / "room", "game")
        second = _writeCase(tmp_path / "b" / "room", "shopping")

        cases, inputs = profileScript.loadCases([first, second])

        assert [case["label"] for case in cases] == ["room", "room#2"]
        assert [item["profile"] for item in inputs] == ["game", "shopping"]
        assert inputs[0]["timeTable"] == str(first[0])

    def test_unknown_case_profile_raises(self, profileScript, tmp_path):
        timeTable, annotationPath = _writeCase(tmp_path / "room", "missing")

        with pytest.raises(ValueError, match="unknown profile"):
            profileScript.loadCases([(str(timeTable), str(annotationPath))])


class TestMain:
    def test_main_writes_json_and_prints_summary(self, profileScript, tmp_path):
        caseOne = _writeCase(tmp_path / "gameOne", "game")
        caseTwo = _writeCase(tmp_path / "gameTwo", "game")
        caseThree = _writeCase(tmp_path / "shoppingOne", "shopping")
        jsonPath = tmp_path / "report.json"

        code = profileScript.main(
            [
                "--case",
                str(caseOne[0]),
                str(caseOne[1]),
                "--case",
                str(caseTwo[0]),
                str(caseTwo[1]),
                "--case",
                str(caseThree[0]),
                str(caseThree[1]),
                "--k",
                "2",
                "--min-cases",
                "2",
                "--min-positives",
                "2",
                "--min-delta",
                "-1",
                "--json",
                str(jsonPath),
            ]
        )

        assert code == 0
        report = json.loads(jsonPath.read_text(encoding="utf-8"))
        assert report["k"] == 2
        assert set(report["profileReports"]) == {
            "default",
            "game",
            "shopping",
            "talent",
        }
        assert report["profileReports"]["game"]["caseCount"] == 2
        assert report["profileReports"]["game"]["sufficient"] is True
        assert report["profileReports"]["shopping"]["sufficient"] is False
        assert report["profileReports"]["default"]["caseCount"] == 0
        assert len(report["cases"]) == 3
        assert len(report["distances"]) == 6
        assert "game" in report["summary"]["keep"]

    def test_quiet_suppresses_text_output(self, profileScript, tmp_path, capsys):
        caseOne = _writeCase(tmp_path / "room", "game")

        code = profileScript.main(
            [
                "--case",
                str(caseOne[0]),
                str(caseOne[1]),
                "--k",
                "2",
                "--quiet",
            ]
        )

        assert code == 0
        assert capsys.readouterr().out == ""

    def test_profile_subset_is_honored(self, profileScript, tmp_path):
        caseOne = _writeCase(tmp_path / "room", "game")
        jsonPath = tmp_path / "report.json"

        code = profileScript.main(
            [
                "--case",
                str(caseOne[0]),
                str(caseOne[1]),
                "--profiles",
                "default,game",
                "--k",
                "2",
                "--json",
                str(jsonPath),
                "--quiet",
            ]
        )

        assert code == 0
        report = json.loads(jsonPath.read_text(encoding="utf-8"))
        assert set(report["profileReports"]) == {"default", "game"}
        assert len(report["distances"]) == 1
