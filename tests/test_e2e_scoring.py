# -*- coding: utf-8 -*-
"""End-to-end scoring regression on a deterministic 90s synthetic fixture."""

import json

from src.scorer import score
from tests.fixtures.synthetic import makeSyntheticFeatures


def test_fixture_produces_non_empty_time_table(tmp_path):
    features = makeSyntheticFeatures(durationSec=90, peakStarts=(20, 55))
    timeTablePath = tmp_path / "time_table.json"
    scoresPath = tmp_path / "scores.json"

    result = score(
        roomId="synthetic-90s",
        audioFeatures=features["audio"],
        textFeatures=features["text"],
        visualFeatures=features["visual"],
        sensitivity=2.0,
        minDuration=15.0,
        maxDuration=60.0,
        durationSec=90.0,
        outputPath=str(timeTablePath),
        scoresPath=str(scoresPath),
    )

    assert result["duration"] == 90
    assert result["clips"], "synthetic highlights must produce at least one clip"
    for clip in result["clips"]:
        assert 15.0 <= clip["duration"] <= 60.0
        assert clip["candidateSource"] == "peak"
        assert "dropReasons" in clip

    saved = json.loads(timeTablePath.read_text(encoding="utf-8"))
    assert saved["clips"]
    assert saved["duration"] == 90

    scores = json.loads(scoresPath.read_text(encoding="utf-8"))
    assert len(scores["scores"]) == 90
    assert scores["signalStats"]


def test_fixture_scoring_is_deterministic(tmp_path):
    features = makeSyntheticFeatures()
    first = score(
        roomId="synthetic-90s",
        audioFeatures=features["audio"],
        textFeatures=features["text"],
        visualFeatures=features["visual"],
        sensitivity=2.0,
        minDuration=15.0,
        maxDuration=60.0,
        durationSec=90.0,
    )
    second = score(
        roomId="synthetic-90s",
        audioFeatures=features["audio"],
        textFeatures=features["text"],
        visualFeatures=features["visual"],
        sensitivity=2.0,
        minDuration=15.0,
        maxDuration=60.0,
        durationSec=90.0,
    )
    assert [(c["start"], c["end"]) for c in first["clips"]] == [
        (c["start"], c["end"]) for c in second["clips"]
    ]
