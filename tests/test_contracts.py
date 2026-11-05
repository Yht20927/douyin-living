# -*- coding: utf-8 -*-
"""Contract tests — hashing, atomic IO, metadata and artifact inspection."""

import json

import pytest

from src.contracts import (
    RecordingMeta,
    artifactFingerprint,
    atomicWriteJson,
    hashJson,
    inspectAsrArtifact,
    inspectFeatureArtifact,
    probeDuration,
    probeMedia,
    settingsHash,
    sha256File,
    sha256Text,
)


class TestHashing:
    def test_sha256_text_matches_file(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("hello 世界", encoding="utf-8")
        assert sha256File(path) == sha256Text("hello 世界")

    def test_hash_json_is_key_order_independent(self):
        assert hashJson({"b": 2, "a": 1}) == hashJson({"a": 1, "b": 2})
        assert hashJson({"a": 1}) != hashJson({"a": 2})

    def test_artifact_fingerprint_small_file(self, tmp_path):
        path = tmp_path / "small.bin"
        path.write_bytes(b"abc")
        info = artifactFingerprint(path)
        assert info["sizeBytes"] == 3
        assert info["sha256"] == sha256Text("abc")

    def test_artifact_fingerprint_large_file_skips_hash(self, tmp_path):
        path = tmp_path / "large.bin"
        path.write_bytes(b"x" * 64)
        info = artifactFingerprint(path, hashLimitBytes=8)
        assert info["sha256"] is None
        assert info["note"] == "large-file: hash skipped"

    def test_artifact_fingerprint_missing(self, tmp_path):
        assert artifactFingerprint(tmp_path / "nope") is None


class TestAtomicWrite:
    def test_roundtrip_and_parent_creation(self, tmp_path):
        path = tmp_path / "nested" / "data.json"
        atomicWriteJson(path, {"b": 2, "a": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}

    def test_serialization_failure_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / "data.json"
        with pytest.raises(TypeError):
            atomicWriteJson(path, {"bad": object()})
        assert not path.exists()
        assert list(tmp_path.glob("*.tmp")) == []


class TestSettingsHash:
    def test_changes_when_config_changes(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: 1\n", encoding="utf-8")
        first = settingsHash(tmp_path)
        (tmp_path / "a.yaml").write_text("x: 2\n", encoding="utf-8")
        assert settingsHash(tmp_path) != first

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "a.json").write_text("{}", encoding="utf-8")
        first = settingsHash(tmp_path)
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        assert settingsHash(tmp_path) == first

    def test_missing_directory_returns_stable_hash(self, tmp_path):
        missing = tmp_path / "nope"
        assert settingsHash(missing) == settingsHash(missing)


class TestRecordingMeta:
    def test_save_load_roundtrip(self, tmp_path):
        meta = RecordingMeta(
            roomId="room-1",
            videoDurationSec=12.5,
            segments=["a.flv", "b.flv"],
        )
        path = tmp_path / "recording_meta.json"
        meta.save(path)
        loaded = RecordingMeta.load(path)
        assert loaded is not None
        assert loaded.roomId == "room-1"
        assert loaded.videoDurationSec == 12.5
        assert loaded.segments == ["a.flv", "b.flv"]

    def test_load_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / "recording_meta.json"
        path.write_text("not json", encoding="utf-8")
        assert RecordingMeta.load(path) is None

    def test_load_non_object_returns_none(self, tmp_path):
        path = tmp_path / "recording_meta.json"
        path.write_text("[]", encoding="utf-8")
        assert RecordingMeta.load(path) is None

    def test_load_ignores_unknown_fields(self, tmp_path):
        path = tmp_path / "recording_meta.json"
        path.write_text('{"roomId": "r", "futureField": 1}', encoding="utf-8")
        loaded = RecordingMeta.load(path)
        assert loaded is not None
        assert loaded.roomId == "r"


class TestProbeMedia:
    def test_missing_file_returns_empty(self, tmp_path):
        assert probeMedia(tmp_path / "nope.mp4") == {}
        assert probeDuration(tmp_path / "nope.mp4") == 0.0

    def test_parses_ffprobe_document(self, tmp_path, monkeypatch):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"fake")
        document = {
            "format": {"duration": "12.5", "size": "4", "bit_rate": "1000"},
            "streams": [
                {
                    "index": 0, "codec_type": "video", "codec_name": "h264",
                    "duration": "12.0", "avg_frame_rate": "30000/1001",
                    "width": 1280, "height": 720,
                },
                {
                    "index": 1, "codec_type": "audio", "codec_name": "aac",
                    "duration": "12.5",
                },
            ],
        }

        class Proc:
            returncode = 0
            stdout = json.dumps(document)

        monkeypatch.setattr("src.contracts.subprocess.run", lambda *a, **k: Proc())
        info = probeMedia(path)
        assert info["durationSec"] == 12.0
        assert info["videoDurationSec"] == 12.0
        assert info["audioDurationSec"] == 12.5
        assert info["formatDurationSec"] == 12.5
        assert info["hasVideo"] is True
        assert info["hasAudio"] is True
        assert info["fps"] == 29.97
        assert info["sizeBytes"] == 4

    def test_ffprobe_failure_returns_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"fake")

        class Proc:
            returncode = 1
            stdout = ""

        monkeypatch.setattr("src.contracts.subprocess.run", lambda *a, **k: Proc())
        assert probeMedia(path) == {}


class TestFeatureArtifactInspection:
    def test_top_level_list_invalid(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("[]", encoding="utf-8")
        assert inspectFeatureArtifact(path)["status"] == "invalid"

    def test_features_not_object_invalid(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text('{"duration": 5, "features": []}', encoding="utf-8")
        result = inspectFeatureArtifact(path)
        assert result["status"] == "invalid"
        assert "'features' must be an object" in result["errors"][0]

    def test_non_array_feature_invalid(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(
            '{"duration": 5, "features": {"rms": 1.0}}', encoding="utf-8"
        )
        result = inspectFeatureArtifact(path)
        assert result["status"] == "invalid"
        assert "rms" in result["errors"][0]

    def test_misaligned_lengths_reported_but_ok(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(
            '{"duration": 5, "features": {"rms": [1, 2, 3]}}', encoding="utf-8"
        )
        result = inspectFeatureArtifact(path)
        assert result["status"] == "ok"
        assert result["featureNames"] == ["rms"]
        assert result["lengths"] == {"rms": 3}
        assert result["errors"] and "differ from declared duration" in result["errors"][0]

    def test_fractional_duration_rounds_up(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(
            '{"duration": 5.2, "features": {"rms": [1, 1, 1, 1, 1, 1]}}',
            encoding="utf-8",
        )
        result = inspectFeatureArtifact(path)
        assert result["status"] == "ok"
        assert result["maxLength"] == 6


class TestAsrArtifactInspection:
    def test_top_level_scalar_invalid(self, tmp_path):
        path = tmp_path / "asr.json"
        path.write_text("42", encoding="utf-8")
        result = inspectAsrArtifact(path)
        assert result["status"] == "invalid"

    def test_segments_not_array_invalid(self, tmp_path):
        path = tmp_path / "asr.json"
        path.write_text('{"segments": {}}', encoding="utf-8")
        result = inspectAsrArtifact(path)
        assert result["status"] == "invalid"

    def test_empty_segments(self, tmp_path):
        path = tmp_path / "asr.json"
        path.write_text('{"segments": []}', encoding="utf-8")
        result = inspectAsrArtifact(path)
        assert result["status"] == "empty"

    def test_bad_end_values_do_not_crash(self, tmp_path):
        path = tmp_path / "asr.json"
        path.write_text(
            '{"segments": [{"end": "bad"}, {"end": null}, "not-a-dict"]}',
            encoding="utf-8",
        )
        result = inspectAsrArtifact(path)
        assert result["status"] == "ok"
        assert result["duration"] == 0.0
