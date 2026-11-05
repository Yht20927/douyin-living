# -*- coding: utf-8 -*-
"""Status-machine and artifact-contract tests."""

import json

import pytest

from src.contracts import (
    RecordingMeta,
    artifactFingerprint,
    atomicWriteJson,
    inspectAsrArtifact,
    inspectFeatureArtifact,
    probeMedia,
    settingsHash,
)
from src.pipeline import PipelineStatus, StepStatus


class TestPipelineStatus:
    def test_step_lifecycle_records_timing_and_hashes(self, tmp_path):
        output = tmp_path / "features.json"
        atomicWriteJson(output, {"features": {"rms": [1.0, 2.0]}})

        status = PipelineStatus("room-1", outputDir=tmp_path)
        status.startStep("signal", inputs=["input.json"])
        status.finishStep(
            "signal", StepStatus.OK,
            outputs=[str(output)],
            meta={"featureCount": 1},
        )
        status.save()

        saved = json.loads((tmp_path / "pipeline_status.json").read_text())
        step = saved["steps"]["signal"]
        assert step["status"] == "ok"
        assert step["elapsedMs"] is not None
        assert step["artifacts"][str(output)]["sha256"]
        assert saved["runId"]
        assert saved["settingsHash"]
        assert saved["environment"]["python"]

    def test_failed_step_enters_failure_summary(self, tmp_path):
        status = PipelineStatus("room-1", outputDir=tmp_path)
        status.setStep("asr", StepStatus.FAILED, error="model missing")
        status.save()

        saved = json.loads((tmp_path / "pipeline_status.json").read_text())
        assert saved["steps"]["asr"]["status"] == "failed"
        assert any("asr" in item for item in saved["failureSummary"])

    def test_unknown_status_rejected(self):
        status = PipelineStatus("room-1")
        with pytest.raises(ValueError, match="unknown step status"):
            status.finishStep("x", "definitely-not-a-status")

    def test_empty_and_skipped_are_distinct(self, tmp_path):
        status = PipelineStatus("room-1", outputDir=tmp_path)
        status.setStep("empty", StepStatus.EMPTY)
        status.setStep("skipped", StepStatus.SKIPPED, error="reused")
        status.save()

        saved = json.loads((tmp_path / "pipeline_status.json").read_text())
        assert saved["steps"]["empty"]["status"] == "empty"
        assert saved["steps"]["skipped"]["status"] == "skipped"
        assert saved["failureSummary"] == []

    def test_on_step_callback_receives_completed_record(self, tmp_path):
        seen = []
        status = PipelineStatus(
            "room-1",
            outputDir=tmp_path,
            onStep=lambda name, record: seen.append((name, record)),
        )
        status.startStep("signal")
        status.finishStep("signal", StepStatus.OK, meta={"reused": False})

        assert seen[0][0] == "signal"
        assert seen[0][1]["status"] == "ok"
        assert seen[0][1]["meta"] == {"reused": False}
        assert seen[0][1]["elapsedMs"] is not None

        status.setStep("scored", StepStatus.OK, elapsedMs=123)
        assert seen[-1][0] == "scored"
        assert seen[-1][1]["elapsedMs"] == 123

    def test_on_step_callback_errors_do_not_break_pipeline(self, tmp_path):
        def boom(name, record):
            raise RuntimeError("event sink down")

        status = PipelineStatus("room-1", outputDir=tmp_path, onStep=boom)
        status.setStep("asr", StepStatus.OK)
        assert status.steps["asr"]["status"] == "ok"


class TestFeatureArtifactInspection:
    def test_ok(self, tmp_path):
        path = tmp_path / "features.json"
        atomicWriteJson(path, {
            "duration": 10,
            "features": {"rms": [0.1] * 10, "dmDensity": [0.0] * 10},
        })
        info = inspectFeatureArtifact(path)
        assert info["status"] == "ok"
        assert info["featureCount"] == 2
        assert info["maxLength"] == 10

    def test_empty(self, tmp_path):
        path = tmp_path / "features.json"
        atomicWriteJson(path, {
            "duration": 10,
            "features": {"rms": [], "dmDensity": []},
        })
        info = inspectFeatureArtifact(path)
        assert info["status"] == "empty"
        assert info["featureCount"] == 0

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "features.json"
        path.write_text("not json")
        info = inspectFeatureArtifact(path)
        assert info["status"] == "invalid"

    def test_missing_duration_is_invalid(self, tmp_path):
        path = tmp_path / "features.json"
        atomicWriteJson(path, {"duration": 0, "features": {"rms": [1.0]}})
        info = inspectFeatureArtifact(path)
        assert info["status"] == "invalid"
        assert any("duration" in error for error in info["errors"])

    def test_missing_file(self, tmp_path):
        info = inspectFeatureArtifact(tmp_path / "nope.json")
        assert info["status"] == "missing"


class TestAsrArtifactInspection:
    def test_plain_list_is_ok(self, tmp_path):
        path = tmp_path / "asr.json"
        atomicWriteJson(path, [{"start": 0, "end": 3, "text": "hi"}])
        info = inspectAsrArtifact(path)
        assert info["status"] == "ok"
        assert info["segments"] == 1
        assert info["duration"] == 3.0

    def test_failed_status_is_failed(self, tmp_path):
        path = tmp_path / "asr.json"
        atomicWriteJson(path, {"status": "failed", "segments": [], "error": "OOM"})
        info = inspectAsrArtifact(path)
        assert info["status"] == "failed"
        assert "OOM" in info["errors"][0]

    def test_empty_segments(self, tmp_path):
        path = tmp_path / "asr.json"
        atomicWriteJson(path, {"segments": []})
        info = inspectAsrArtifact(path)
        assert info["status"] == "empty"


class TestRecordingMeta:
    def test_roundtrip(self, tmp_path):
        meta = RecordingMeta(
            roomId="123",
            videoDurationSec=42.5,
            audioDurationSec=42.0,
            segments=["a.flv", "b.flv"],
            sourceHashes={"a.flv": {"sha256": "abc"}},
        )
        path = tmp_path / "recording_meta.json"
        meta.save(path)

        loaded = RecordingMeta.load(path)
        assert loaded is not None
        assert loaded.roomId == "123"
        assert loaded.videoDurationSec == 42.5
        assert loaded.segments == ["a.flv", "b.flv"]

    def test_load_invalid_returns_none(self, tmp_path):
        path = tmp_path / "recording_meta.json"
        path.write_text("not json")
        assert RecordingMeta.load(path) is None


class TestAtomicWrite:
    def test_no_temp_files_left(self, tmp_path):
        path = tmp_path / "out.json"
        atomicWriteJson(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
        assert list(tmp_path.glob("*.tmp")) == []


class TestHelpers:
    def test_settings_hash_depends_on_content(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: 1\n")
        first = settingsHash(tmp_path)
        (tmp_path / "a.yaml").write_text("x: 2\n")
        second = settingsHash(tmp_path)
        assert first != second
        assert len(first) == 64

    def test_probe_missing_file_returns_empty(self, tmp_path):
        assert probeMedia(tmp_path / "missing.mp4") == {}

    def test_artifact_fingerprint_missing(self, tmp_path):
        assert artifactFingerprint(tmp_path / "missing") is None
