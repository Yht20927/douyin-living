# -*- coding: utf-8 -*-
"""Tests for per-clip metadata artifacts (S6-2)."""

from __future__ import annotations

import json
from pathlib import Path

from src.clipper import clipOutputPath
from src.clipMeta import (
    buildClipMetadata,
    clipMetadataPath,
    suggestedAction,
    writeClipMetadata,
)

TWO_STAGE_CLIP = {
    "id": 1,
    "start": 342.0,
    "end": 367.0,
    "duration": 25.0,
    "score": 0.91,
    "confidence": 0.82,
    "title": "弹幕爆发 + 关键词命中",
    "tags": ["互动", "搞笑", "互动", ""],
    "reasons": ["dmDensity", "asrKeyword", "dmDensity"],
    "risks": ["asr_missing"],
    "dropReasons": [],
    "trigger": "dmDensity+asrKeyword",
    "breakdown": {"dmDensity": 0.31, "asrKeyword": 0.22},
}

LEGACY_CLIP = {
    "id": 3,
    "start": 12.4,
    "end": 40.2,
    "score": 1.23456,
    "title": "legacy moment",
    "tags": ["音量"],
    "trigger": "rms",
    "breakdown": {"rms": 0.4, "dmDensity": 0.2, "negative": -0.1},
    "dropReasons": [],
}


class TestBuildClipMetadata:
    def test_two_stage_fields_and_deduplication(self):
        metadata = buildClipMetadata(TWO_STAGE_CLIP)

        assert metadata["clipId"] == 1
        assert metadata["start"] == 342.0
        assert metadata["end"] == 367.0
        assert metadata["duration"] == 25.0
        assert metadata["score"] == 0.91
        assert metadata["confidence"] == 0.82
        assert metadata["title"] == "弹幕爆发 + 关键词命中"
        assert metadata["tags"] == ["互动", "搞笑"]
        assert metadata["reasons"] == ["dmDensity", "asrKeyword"]
        assert metadata["risks"] == ["asr_missing"]
        assert metadata["suggestedAction"] == "review"
        assert metadata["scoreBreakdown"] == {"dmDensity": 0.31, "asrKeyword": 0.22}
        assert metadata["assets"] == {"video": None, "srt": None, "thumbnail": None}

    def test_legacy_clip_derives_reasons_from_breakdown(self):
        metadata = buildClipMetadata(LEGACY_CLIP)

        assert metadata["confidence"] is None
        assert metadata["reasons"] == ["rms", "dmDensity"]
        assert metadata["suggestedAction"] == "keep"
        assert metadata["score"] == 1.2346
        assert metadata["duration"] == 27.8

    def test_clip_id_fallback(self):
        metadata = buildClipMetadata({"start": 1, "end": 2}, clipId=9)
        assert metadata["clipId"] == 9

    def test_explicit_drop_reasons_win(self):
        metadata = buildClipMetadata(
            {
                "start": 1,
                "end": 2,
                "risks": ["asr_missing"],
                "dropReasons": ["too_short"],
            }
        )
        assert metadata["suggestedAction"] == "drop"

    def test_suggested_action(self):
        assert suggestedAction([], []) == "keep"
        assert suggestedAction(["asr_missing"], []) == "review"
        assert suggestedAction(["asr_missing"], ["too_short"]) == "drop"

    def test_malformed_values_are_normalised(self):
        metadata = buildClipMetadata(
            {
                "id": "x",
                "start": "bad",
                "end": None,
                "duration": None,
                "score": "nan",
                "confidence": "inf",
                "title": None,
                "tags": "single",
                "reasons": "trigger",
                "risks": ["", "risk"],
                "breakdown": {"a": "bad", "b": 2},
            }
        )
        assert metadata["clipId"] == "x"
        assert metadata["start"] == 0.0
        assert metadata["end"] == 0.0
        assert metadata["duration"] == 0.0
        assert metadata["score"] == 0.0
        assert metadata["confidence"] is None
        assert metadata["title"] == ""
        assert metadata["tags"] == ["single"]
        assert metadata["reasons"] == ["trigger"]
        assert metadata["risks"] == ["risk"]
        assert metadata["scoreBreakdown"] == {"a": 0.0, "b": 2.0}


class TestWriteClipMetadata:
    def test_writes_next_to_clip_with_asset_flags(self, tmp_path):
        highlights = tmp_path / "highlights"
        highlights.mkdir()
        clipPath = Path(clipOutputPath(TWO_STAGE_CLIP, str(highlights)))
        clipPath.write_bytes(b"fake-mp4")
        clipPath.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        clipPath.with_suffix(".jpg").write_bytes(b"fake-jpg")

        metadataPath = Path(writeClipMetadata(TWO_STAGE_CLIP, clipPath))

        assert metadataPath == clipPath.with_suffix(".json")
        document = json.loads(metadataPath.read_text(encoding="utf-8"))
        assert document["clipId"] == 1
        assert document["assets"] == {
            "video": clipPath.name,
            "srt": clipPath.with_suffix(".srt").name,
            "thumbnail": clipPath.with_suffix(".jpg").name,
        }

    def test_missing_assets_are_null(self, tmp_path):
        clipPath = tmp_path / "clip_1s_test.mp4"
        clipPath.write_bytes(b"fake-mp4")

        metadataPath = Path(writeClipMetadata(LEGACY_CLIP, clipPath))
        document = json.loads(metadataPath.read_text(encoding="utf-8"))

        assert document["assets"]["video"] == clipPath.name
        assert document["assets"]["srt"] is None
        assert document["assets"]["thumbnail"] is None

    def test_metadata_path_matches_clip_stem(self, tmp_path):
        expected = Path(
            clipOutputPath(LEGACY_CLIP, str(tmp_path))
        ).with_suffix(".json")
        assert Path(clipMetadataPath(LEGACY_CLIP, str(tmp_path))) == expected

    def test_rewrite_is_atomic_and_keeps_valid_json(self, tmp_path):
        clipPath = tmp_path / "clip_1s_test.mp4"
        clipPath.write_bytes(b"fake-mp4")

        first = Path(writeClipMetadata(LEGACY_CLIP, clipPath))
        second = Path(
            writeClipMetadata({**LEGACY_CLIP, "title": "updated"}, clipPath)
        )

        assert first == second
        document = json.loads(second.read_text(encoding="utf-8"))
        assert document["title"] == "updated"
        assert not list(tmp_path.glob("*.tmp"))
