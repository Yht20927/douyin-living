# -*- coding: utf-8 -*-
"""S5-5 tests — ASR runtime resolution, VAD wiring, chunk resume."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import asr
from src.config import AsrSettings


class TestResolveRuntime:
    def test_cpu_auto_uses_base(self):
        runtime = asr.resolveAsrRuntime(
            AsrSettings(model="auto", device="auto"), cudaAvailable=False
        )
        assert runtime["model"] == "base"
        assert runtime["device"] == "cpu"
        assert "auto→base" in runtime["selection"]

    def test_cuda_auto_uses_small(self):
        runtime = asr.resolveAsrRuntime(
            AsrSettings(model="auto", device="auto"), cudaAvailable=True
        )
        assert runtime["model"] == "small"
        assert runtime["device"] == "cuda"

    def test_explicit_model_is_kept(self):
        runtime = asr.resolveAsrRuntime(
            AsrSettings(model="large-v3", device="auto"), cudaAvailable=False
        )
        assert runtime["model"] == "large-v3"

    def test_cpu_rejects_cuda_only_compute(self):
        runtime = asr.resolveAsrRuntime(
            AsrSettings(device="cpu", compute_type="float16"),
            cudaAvailable=True,
        )
        assert runtime["computeType"] == "int8"

    def test_language_auto_becomes_none(self):
        runtime = asr.resolveAsrRuntime(
            AsrSettings(language="auto"), cudaAvailable=False
        )
        assert runtime["language"] is None


def _installFakeWhisper(monkeypatch, handler):
    module = types.ModuleType("faster_whisper")

    class FakeWhisperModel:
        def __init__(self, modelSize, device, compute_type):
            self.modelSize = modelSize
            self.device = device
            self.computeType = compute_type

        def transcribe(self, audioPath, **kwargs):
            return handler(audioPath, kwargs)

    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


class TestTranscribeSinglePass:
    def test_vad_params_and_outputs(self, tmp_path, monkeypatch):
        calls = {}

        def handler(audioPath, kwargs):
            calls["audio"] = audioPath
            calls["kwargs"] = kwargs
            return (
                iter([SimpleNamespace(start=0.04, end=1.26, text="你好")]),
                SimpleNamespace(language="zh", language_probability=0.99),
            )

        _installFakeWhisper(monkeypatch, handler)
        outPath = tmp_path / "audio_asr"
        results = asr.transcribe(
            "audio.aac",
            modelSize="small",
            language="zh",
            device="cpu",
            computeType="int8",
            diarize=False,
            outputPath=str(outPath),
            batchSize=16,
            vadFilter=True,
            vadMinSilenceMs=700,
            vadSpeechPadMs=120,
            beamSize=4,
            chunkSec=0,
        )

        assert results == [{"start": 0.0, "end": 1.3, "text": "你好"}]
        assert calls["kwargs"]["language"] == "zh"
        assert calls["kwargs"]["vad_filter"] is True
        assert calls["kwargs"]["vad_parameters"] == {
            "min_silence_duration_ms": 700,
            "speech_pad_ms": 120,
        }
        assert calls["kwargs"]["beam_size"] == 4

        saved = json.loads(Path(str(outPath) + ".json").read_text(encoding="utf-8"))
        assert saved == results
        assert Path(str(outPath) + ".srt").exists()

    def test_vad_can_be_disabled(self, tmp_path, monkeypatch):
        calls = {}

        def handler(audioPath, kwargs):
            calls.update(kwargs)
            return (
                iter([SimpleNamespace(start=0, end=1, text="x")]),
                SimpleNamespace(language="zh", language_probability=1.0),
            )

        _installFakeWhisper(monkeypatch, handler)
        asr.transcribe(
            "audio.aac",
            diarize=False,
            outputPath=str(tmp_path / "out"),
            vadFilter=False,
            chunkSec=0,
        )
        assert calls["vad_filter"] is False
        assert calls["vad_parameters"] is None


class TestChunkResume:
    def _fakeProbe(self, audioPath):
        name = Path(audioPath).name
        if name == "audio.aac":
            return {"durationSec": 120.0}
        return {"durationSec": 60.0}

    def test_failed_chunk_resumes_without_retranscribing(self, tmp_path, monkeypatch):
        audio = tmp_path / "audio.aac"
        audio.write_bytes(b"fake-audio")
        state = {"calls": [], "failSecond": True}

        def handler(audioPath, kwargs):
            name = Path(audioPath).name
            state["calls"].append(name)
            if name == "chunk_0001.m4a" and state["failSecond"]:
                raise RuntimeError("simulated chunk failure")
            if name == "chunk_0000.m4a":
                text = "first"
            else:
                text = "second"
            return (
                iter([SimpleNamespace(start=0.0, end=5.0, text=text)]),
                SimpleNamespace(language="zh", language_probability=1.0),
            )

        _installFakeWhisper(monkeypatch, handler)
        monkeypatch.setattr(asr, "probeMedia", self._fakeProbe)

        def fakeSplit(audioPath, chunkSec, workDir):
            paths = []
            for index in range(2):
                path = Path(workDir) / f"chunk_{index:04d}.m4a"
                path.write_bytes(b"chunk")
                paths.append(str(path))
            return paths

        monkeypatch.setattr(asr, "_splitAudio", fakeSplit)

        outputPath = str(tmp_path / "audio_asr.json")
        with pytest.raises(RuntimeError, match="simulated chunk failure"):
            asr.transcribe(
                str(audio),
                diarize=False,
                outputPath=outputPath,
                chunkSec=60,
            )

        checkpoint = Path(asr._checkpointPath(outputPath))
        assert checkpoint.exists()
        assert not Path(outputPath).exists()

        state["failSecond"] = False
        results = asr.transcribe(
            str(audio),
            diarize=False,
            outputPath=outputPath,
            chunkSec=60,
        )

        assert [seg["text"] for seg in results] == ["first", "second"]
        assert [seg["start"] for seg in results] == [0.0, 60.0]
        assert state["calls"] == [
            "chunk_0000.m4a", "chunk_0001.m4a", "chunk_0001.m4a"
        ]
        assert not checkpoint.exists()
        saved = json.loads(Path(outputPath).read_text(encoding="utf-8"))
        assert saved == results

    def test_checkpoint_invalidated_when_audio_changes(self, tmp_path, monkeypatch):
        audio = tmp_path / "audio.aac"
        audio.write_bytes(b"fake-audio")
        state = {"calls": []}

        def handler(audioPath, kwargs):
            state["calls"].append(Path(audioPath).name)
            return (
                iter([SimpleNamespace(start=0, end=1, text="x")]),
                SimpleNamespace(language="zh", language_probability=1.0),
            )

        _installFakeWhisper(monkeypatch, handler)
        monkeypatch.setattr(asr, "probeMedia", self._fakeProbe)

        def fakeSplit(audioPath, chunkSec, workDir):
            paths = []
            for index in range(2):
                path = Path(workDir) / f"chunk_{index:04d}.m4a"
                path.write_bytes(b"chunk")
                paths.append(str(path))
            return paths

        monkeypatch.setattr(asr, "_splitAudio", fakeSplit)
        outputPath = str(tmp_path / "audio_asr.json")

        asr.transcribe(str(audio), diarize=False, outputPath=outputPath, chunkSec=60)
        signature = asr._audioSignature(
            str(audio),
            {
                "chunkSec": 60.0,
                "model": "base",
                "device": "cuda",
                "computeType": "int8",
                "language": "zh",
                "diarize": False,
                "batchSize": 32,
                "vadFilter": True,
                "vadMinSilenceMs": 500,
                "vadSpeechPadMs": 200,
                "beamSize": 5,
            },
        )
        asr._saveCheckpoint(outputPath, signature, {"0": [{"start": 0, "end": 1, "text": "old"}]})

        audio.write_bytes(b"changed-audio")
        asr.transcribe(str(audio), diarize=False, outputPath=outputPath, chunkSec=60)
        # Both chunks must be transcribed again after the audio changed.
        assert state["calls"].count("chunk_0000.m4a") == 2


class TestSplitAudio:
    def test_missing_ffmpeg_raises(self, tmp_path, monkeypatch):
        def missing(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(asr.subprocess, "run", missing)
        with pytest.raises(RuntimeError, match="ffmpeg not installed"):
            asr._splitAudio("audio.aac", 60, str(tmp_path))
