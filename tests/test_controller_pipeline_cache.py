# -*- coding: utf-8 -*-
"""End-to-end artifact cache tests for the controller pipeline.

The pipeline is exercised with fake ffmpeg/ASR/signal/scorer/clipper
implementations so the tests stay fast and dependency-free while still
covering every real cache decision.
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from src.controller import Controller, normalizeStepName
from src.pipeline.artifacts import ArtifactCache


class FakeProc:
    returncode = 0

    async def communicate(self):
        return b"", b""


@pytest.fixture
def pipelineEnv(tmp_path, monkeypatch):
    room = "room"
    video = tmp_path / f"{room}_merged.mp4"
    video.write_bytes(b"fake-video")
    danmaku = tmp_path / f"{room}_danmaku.jsonl"
    danmaku.write_text('{"type": "chat", "content": "hi"}\n', encoding="utf-8")
    audio = tmp_path / f"{room}_merged.aac"

    calls = {
        "audio_extract": 0,
        "asr": 0,
        "audio_signal": 0,
        "text_signal": 0,
        "visual_signal": 0,
        "score": 0,
        "two_stage": 0,
        "two_stage_last": None,
        "clip": 0,
        "srt": 0,
        "thumbnail": 0,
    }

    async def fakeExec(*args, **kwargs):
        calls["audio_extract"] += 1
        for arg in args:
            if isinstance(arg, str) and arg.endswith(".aac"):
                Path(arg).write_bytes(b"fake-audio")
                break
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fakeExec)
    monkeypatch.setattr(
        "src.controller.probeMedia",
        lambda path: {
            "durationSec": 30.0,
            "videoDurationSec": 30.0,
            "audioDurationSec": 30.0,
        },
    )

    def fakeTranscribe(audioPath, **kwargs):
        calls["asr"] += 1
        Path(kwargs["outputPath"]).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "segments": [{"start": 0.0, "end": 5.0, "text": "hello"}],
                }
            ),
            encoding="utf-8",
        )

    asrModule = types.ModuleType("asr")
    asrModule.transcribe = fakeTranscribe
    monkeypatch.setitem(sys.modules, "asr", asrModule)
    torchModule = types.ModuleType("torch")
    torchModule.Tensor = type("Tensor", (), {})
    torchModule.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torchModule)

    def writeFeatures(outputPath, features, durationSec):
        Path(outputPath).write_text(
            json.dumps({"duration": durationSec or 30.0, "features": features}),
            encoding="utf-8",
        )

    def fakeAudioSignal(audioPath, outputPath=None, durationSec=None):
        calls["audio_signal"] += 1
        writeFeatures(outputPath, {"rms": [0.1] * 30}, durationSec)

    def fakeTextSignal(danmakuPath, asrPath, outputPath=None, durationSec=None):
        calls["text_signal"] += 1
        writeFeatures(outputPath, {"dmDensity": [1.0] * 30}, durationSec)

    def fakeVisualSignal(videoPath, outputPath=None, durationSec=None):
        calls["visual_signal"] += 1
        writeFeatures(outputPath, {"sceneChange": [0.0] * 30}, durationSec)

    monkeypatch.setattr("src.signalAudio.extractFeatures", fakeAudioSignal)
    monkeypatch.setattr("src.signalText.extractFeatures", fakeTextSignal)
    monkeypatch.setattr("src.signalVisual.extractFeatures", fakeVisualSignal)

    def fakeScore(**kwargs):
        calls["score"] += 1
        clips = [
            {
                "id": 1,
                "start": 5.0,
                "end": 20.0,
                "duration": 15.0,
                "score": 1.0,
                "trigger": "funny",
                "title": "moment",
                "tags": [],
                "thumbnailTime": 6.0,
            },
            {
                "id": 2,
                "start": 30.0,
                "end": 50.0,
                "duration": 20.0,
                "score": 0.9,
                "trigger": "highlight",
                "title": "moment",
                "tags": [],
                "thumbnailTime": 31.0,
            },
        ]
        result = {
            "clips": clips,
            "dropped": [],
            "adjustments": [],
            "alignment": [],
        }
        Path(kwargs["outputPath"]).write_text(json.dumps(result), encoding="utf-8")
        Path(kwargs["scoresPath"]).write_text(
            json.dumps({"scores": []}), encoding="utf-8"
        )
        return result

    monkeypatch.setattr("src.scorer.score", fakeScore)

    def fakeTwoStage(**kwargs):
        calls["two_stage"] += 1
        calls["two_stage_last"] = kwargs
        clips = [
            {
                "id": 1,
                "start": 4.0,
                "end": 24.0,
                "duration": 20.0,
                "score": 0.8,
                "confidence": 0.7,
                "trigger": "rms",
                "title": "two-stage moment",
                "tags": ["音量"],
                "candidateSource": "two_stage",
                "reasons": ["rms"],
                "risks": [],
                "dropReasons": [],
                "thumbnailTime": 5.0,
            }
        ]
        result = {
            "pipeline": "two_stage",
            "stages": {
                "recall": {"candidateCount": 3, "elapsedMs": 4},
                "rerank": {"inputCount": 3, "outputCount": 1, "elapsedMs": 2},
                "totalElapsedMs": 6,
            },
            "clips": clips,
            "dropped": [],
            "adjustments": [],
            "alignment": [],
        }
        Path(kwargs["outputPath"]).write_text(json.dumps(result), encoding="utf-8")
        Path(kwargs["scoresPath"]).write_text(
            json.dumps({"pipeline": "two_stage", "scores": []}), encoding="utf-8"
        )
        Path(kwargs["candidatesPath"]).write_text(
            json.dumps({"candidates": [1, 2, 3]}), encoding="utf-8"
        )
        return result

    monkeypatch.setattr("src.pipeline.stage2_rerank.twoStageScore", fakeTwoStage)

    from src.clipper import clipOutputPath

    def fakeClipVideo(inputPath, timeTablePath, outputDir, dryRun=False, resolution="720p"):
        calls["clip"] += 1
        data = json.loads(Path(timeTablePath).read_text(encoding="utf-8"))
        outputs = []
        for clip in data["clips"]:
            path = Path(clipOutputPath(clip, outputDir))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-clip")
            outputs.append((str(path), clip))
        return outputs

    def fakeThumbnail(inputPath, timeSec, outputPath, dryRun=False):
        calls["thumbnail"] += 1
        Path(outputPath).write_bytes(b"fake-jpg")

    def fakeClipSrt(srtPath, clipStart, clipEnd, outputPath):
        calls["srt"] += 1
        Path(outputPath).write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8"
        )

    monkeypatch.setattr("src.clipper.clipVideo", fakeClipVideo)
    monkeypatch.setattr("src.clipper.extractThumbnail", fakeThumbnail)
    monkeypatch.setattr("src.clipper.generateClipSrt", fakeClipSrt)

    return {
        "room": room,
        "outputDir": tmp_path,
        "video": video,
        "audio": audio,
        "danmaku": danmaku,
        "calls": calls,
    }


def runPipeline(outputDir, **kwargs):
    ctrl = Controller("room", clipOnly=True, outputDir=str(outputDir), **kwargs)
    asyncio.run(ctrl._runClippingPipeline())
    return ctrl


def loadStatus(outputDir):
    return json.loads((Path(outputDir) / "pipeline_status.json").read_text(encoding="utf-8"))


class TestStepSelection:
    def test_normalize_aliases(self):
        assert normalizeStepName("audio") == "audio"
        assert normalizeStepName("1_audio_extract") == "audio"
        assert normalizeStepName("5_text_signal") == "signal"
        assert normalizeStepName("7_scorer") == "score"
        assert normalizeStepName("8_clipping") == "clip"
        assert normalizeStepName(None) is None
        with pytest.raises(ValueError, match="unknown pipeline step"):
            normalizeStepName("not-a-step")

    def test_only_step_implies_from_step(self):
        ctrl = Controller("room", onlyStep="score")
        assert ctrl.onlyStep == "score"
        assert ctrl.fromStep == "score"
        assert ctrl._stopAfter("7_scorer") is True

    def test_rebuild_window(self):
        plain = Controller("room")
        assert plain._shouldRebuild("1_audio_extract") is False

        forced = Controller("room", force=True)
        assert forced._shouldRebuild("1_audio_extract") is True
        assert forced._shouldRebuild("8_clipping") is True

        resumed = Controller("room", fromStep="signal")
        assert resumed._shouldRebuild("1_audio_extract") is False
        assert resumed._shouldRebuild("2_asr") is False
        assert resumed._shouldRebuild("4_audio_signal") is True
        assert resumed._shouldRebuild("7_scorer") is True
        assert resumed._shouldRebuild("8_clipping") is True


class TestPipelineCache:
    def test_first_run_writes_metadata_then_second_run_reuses(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]

        runPipeline(outputDir)
        assert calls["asr"] == 1
        assert calls["audio_signal"] == calls["text_signal"] == calls["visual_signal"] == 1
        assert calls["score"] == 1
        assert calls["clip"] == 1

        for name in (
            "room_merged.aac.meta.json",
            "room_merged_asr.json.meta.json",
            "audio_features.json.meta.json",
            "text_features.json.meta.json",
            "visual_features.json.meta.json",
            "time_table.json.meta.json",
        ):
            assert (outputDir / name).exists(), name
        assert len(list((outputDir / "highlights").glob("*.mp4.meta.json"))) == 2

        first = loadStatus(outputDir)
        assert first["cacheStats"]["writes"] >= 8

        runPipeline(outputDir)
        assert calls["asr"] == 1
        assert calls["audio_signal"] == calls["text_signal"] == calls["visual_signal"] == 1
        assert calls["score"] == 1
        assert calls["clip"] == 1
        assert calls["srt"] == 2
        assert calls["thumbnail"] == 2

        second = loadStatus(outputDir)
        assert second["steps"]["1_audio_extract"]["status"] == "skipped"
        assert second["steps"]["2_asr"]["status"] == "skipped"
        assert second["steps"]["4_audio_signal"]["status"] == "skipped"
        assert second["steps"]["5_text_signal"]["status"] == "skipped"
        assert second["steps"]["6_visual_signal"]["status"] == "skipped"
        assert second["steps"]["7_scorer"]["status"] == "skipped"
        assert second["steps"]["8_clipping"]["status"] == "skipped"
        assert second["steps"]["9_subtitles_thumbnails"]["status"] == "skipped"
        assert second["cacheStats"]["hits"] >= 8

    def test_force_rebuilds_every_stage(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)
        before = dict(calls)

        runPipeline(outputDir, force=True)

        assert calls["audio_extract"] == before["audio_extract"] + 1
        assert calls["asr"] == before["asr"] + 1
        assert calls["audio_signal"] == before["audio_signal"] + 1
        assert calls["text_signal"] == before["text_signal"] + 1
        assert calls["visual_signal"] == before["visual_signal"] + 1
        assert calls["score"] == before["score"] + 1
        assert calls["clip"] == before["clip"] + 1
        assert calls["srt"] == before["srt"] + 2
        assert calls["thumbnail"] == before["thumbnail"] + 2

        saved = loadStatus(outputDir)
        assert saved["steps"]["1_audio_extract"]["meta"]["cacheReason"] == "forced rebuild"

    def test_from_step_reuses_earlier_stages(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)
        before = dict(calls)

        runPipeline(outputDir, fromStep="score")

        assert calls["asr"] == before["asr"]
        assert calls["audio_signal"] == before["audio_signal"]
        assert calls["text_signal"] == before["text_signal"]
        assert calls["visual_signal"] == before["visual_signal"]
        assert calls["score"] == before["score"] + 1
        assert calls["clip"] == before["clip"] + 1

    def test_only_step_stops_after_named_stage(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)
        before = dict(calls)

        runPipeline(outputDir, onlyStep="asr")

        assert calls["asr"] == before["asr"] + 1
        assert calls["audio_signal"] == before["audio_signal"]
        assert calls["score"] == before["score"]
        saved = loadStatus(outputDir)
        assert "7_scorer" not in saved["steps"]

    def test_two_stage_scoring_is_selectable_and_cached(
        self, pipelineEnv, monkeypatch
    ):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        events = []

        def fakeLogEvent(event, **fields):
            payload = {"event": event, **fields}
            events.append(payload)
            return payload

        monkeypatch.setattr("src.controller.logEvent", fakeLogEvent)

        # Seed the output directory with a legacy result first: switching
        # modes must invalidate the cached time table because the scoring
        # parameters changed.
        runPipeline(outputDir)
        assert calls["score"] == 1
        assert calls["two_stage"] == 0

        runPipeline(outputDir, twoStage=True)
        assert calls["score"] == 1
        assert calls["two_stage"] == 1
        assert (outputDir / "recall_candidates.json").exists()

        twoStageKwargs = calls["two_stage_last"]
        assert twoStageKwargs["asrSegments"] == [
            {"start": 0.0, "end": 5.0, "text": "hello"}
        ]
        assert twoStageKwargs["dynamicDuration"] is True
        assert twoStageKwargs["boundarySearchSec"] == 8.0
        assert twoStageKwargs["platform"] == "none"

        status = loadStatus(outputDir)
        step = status["steps"]["7_scorer"]
        assert step["meta"]["pipeline"] == "two_stage"
        assert step["meta"]["stages"]["recall"]["candidateCount"] == 3
        assert any(str(outputDir / "recall_candidates.json") == p for p in step["artifacts"])

        scoreEvents = [
            event for event in events
            if event.get("event") == "pipeline.step" and event.get("step") == "7_scorer"
        ]
        assert scoreEvents[-1]["scoring"] == "two_stage"
        assert scoreEvents[-1]["stages"]["rerank"]["outputCount"] == 1

        # A second two-stage run reuses the time table; the legacy scorer must
        # never be called.
        runPipeline(outputDir, twoStage=True)
        assert calls["two_stage"] == 1
        assert loadStatus(outputDir)["steps"]["7_scorer"]["status"] == "skipped"

    def test_two_stage_scoring_records_asr_as_cache_input(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]

        runPipeline(outputDir, twoStage=True)
        assert calls["two_stage"] == 1

        meta = json.loads(
            (outputDir / "time_table.json.meta.json").read_text(encoding="utf-8")
        )
        asrPath = str(outputDir / "room_merged_asr.json")
        assert asrPath in meta["sourceHashes"]
        assert meta["params"]["scoring"] == "two_stage"

    def test_two_stage_cache_key_tracks_boundary_settings(
        self, pipelineEnv, monkeypatch
    ):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]

        runPipeline(outputDir, twoStage=True)
        assert calls["two_stage"] == 1

        monkeypatch.setattr("src.controller._cfg.two_stage_boundary_search_sec", 4.0)
        runPipeline(outputDir, twoStage=True)
        assert calls["two_stage"] == 2
        assert calls["two_stage_last"]["boundarySearchSec"] == 4.0

    def test_failed_asr_is_retried(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)

        asrPath = outputDir / "room_merged_asr.json"
        asrPath.write_text(
            json.dumps({"status": "failed", "segments": [], "error": "boom"}),
            encoding="utf-8",
        )
        # Keep the sidecar parameters identical to the controller's so this
        # test exercises the validator (failed artifact), not a params miss.
        asrMeta = json.loads(
            (outputDir / "room_merged_asr.json.meta.json").read_text(encoding="utf-8")
        )
        cache = ArtifactCache()
        cache.record(
            str(asrPath),
            "2_asr",
            [str(pipelineEnv["audio"])],
            params=asrMeta["params"],
            extra={"status": "failed"},
        )

        beforeAsr = calls["asr"]
        runPipeline(outputDir)
        assert calls["asr"] == beforeAsr + 1

    def test_asr_failure_degrades_text_and_reports_status(
        self, pipelineEnv, monkeypatch
    ):
        outputDir = pipelineEnv["outputDir"]
        asrModule = sys.modules["asr"]

        def failing(*args, **kwargs):
            raise RuntimeError("simulated ASR crash")

        monkeypatch.setattr(asrModule, "transcribe", failing)
        runPipeline(outputDir)

        saved = loadStatus(outputDir)
        assert saved["steps"]["2_asr"]["status"] == "failed"
        asrDoc = json.loads(
            (outputDir / "room_merged_asr.json").read_text(encoding="utf-8")
        )
        assert asrDoc["status"] == "failed"
        assert asrDoc["segments"] == []
        assert "simulated ASR crash" in asrDoc["error"]

        textStep = saved["steps"]["5_text_signal"]
        assert textStep["status"] == "ok"
        assert textStep["meta"]["asrStatus"] == "failed"

    def test_asr_cache_key_uses_resolved_runtime(self, pipelineEnv, monkeypatch):
        import src.controller as controllerModule

        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)
        beforeAsr = calls["asr"]

        resolved = dict(
            controllerModule.resolveAsrRuntime(controllerModule._asrCfg)
        )
        resolved["model"] = "medium"
        monkeypatch.setattr(
            controllerModule, "resolveAsrRuntime", lambda cfg: resolved
        )

        runPipeline(outputDir)
        assert calls["asr"] == beforeAsr + 1

        # Same resolved runtime → cache hit again.
        runPipeline(outputDir)
        assert calls["asr"] == beforeAsr + 1

    def test_missing_or_corrupt_meta_triggers_rebuild(self, pipelineEnv):
        outputDir = pipelineEnv["outputDir"]
        calls = pipelineEnv["calls"]
        runPipeline(outputDir)

        # Missing sidecar: audio is re-extracted.
        (outputDir / "room_merged.aac.meta.json").unlink()
        beforeAudio = calls["audio_extract"]
        beforeAsr = calls["asr"]
        runPipeline(outputDir)
        assert calls["audio_extract"] == beforeAudio + 1
        # Audio bytes are identical, so downstream can still reuse.
        assert calls["asr"] == beforeAsr

        # Corrupt sidecar: same fallback, no exception.
        (outputDir / "audio_features.json.meta.json").write_text(
            "not-json", encoding="utf-8"
        )
        beforeSignal = calls["audio_signal"]
        runPipeline(outputDir)
        assert calls["audio_signal"] == beforeSignal + 1

    def test_pipeline_emits_structured_events(self, pipelineEnv, monkeypatch):
        events = []

        def fakeLogEvent(event, **fields):
            payload = {"event": event, **fields}
            events.append(payload)
            return payload

        monkeypatch.setattr("src.controller.logEvent", fakeLogEvent)
        runPipeline(pipelineEnv["outputDir"])

        assert events[0]["event"] == "pipeline.start"
        assert events[0]["roomId"] == "room"
        assert events[-1]["event"] == "pipeline.finish"
        assert events[-1]["runId"] == events[0]["runId"]
        assert events[-1]["status"] == "ok"
        assert events[-1]["failures"] == 0
        assert isinstance(events[-1]["elapsedMs"], int)

        firstSteps = {
            event["step"]: event
            for event in events
            if event["event"] == "pipeline.step"
        }
        assert {"1_audio_extract", "2_asr", "7_scorer", "8_clipping"} <= set(
            firstSteps
        )
        assert firstSteps["2_asr"]["status"] == "ok"
        assert firstSteps["2_asr"]["reused"] is False
        assert isinstance(firstSteps["2_asr"]["elapsedMs"], int)
        assert firstSteps["8_clipping"]["status"] == "ok"

        events.clear()
        runPipeline(pipelineEnv["outputDir"])
        secondSteps = {
            event["step"]: event
            for event in events
            if event["event"] == "pipeline.step"
        }
        assert secondSteps["2_asr"]["status"] == "skipped"
        assert secondSteps["2_asr"]["reused"] is True
        assert secondSteps["2_asr"]["cacheReason"] == "cache hit"
        assert events[-1]["event"] == "pipeline.finish"
        assert events[-1]["cacheHits"] >= 1


class TestDryRunReport:
    def test_dry_run_writes_html_report(self, pipelineEnv, tmp_path, monkeypatch):
        # The fixture video is fake bytes; skip real ffmpeg extraction.
        monkeypatch.setattr(
            "src.report._extractThumbnail", lambda *args, **kwargs: False
        )
        reportPath = tmp_path / "report" / "index.html"

        runPipeline(
            pipelineEnv["outputDir"],
            dryRun=True,
            reportHtml=str(reportPath),
        )

        assert reportPath.exists()
        text = reportPath.read_text(encoding="utf-8")
        assert "切片 Dry-run 报告" in text
        assert "moment" in text
        assert "ffmpeg -hide_banner" in text
        assert "复制时间码" in text
        assert pipelineEnv["calls"]["clip"] == 0
        status = loadStatus(pipelineEnv["outputDir"])
        assert status["steps"]["8_clipping"]["meta"]["dryRun"] is True

    def test_report_failure_does_not_abort_dry_run(
        self, pipelineEnv, tmp_path, monkeypatch
    ):
        reportPath = tmp_path / "report" / "index.html"

        def explode(*args, **kwargs):
            raise RuntimeError("report exploded")

        monkeypatch.setattr("src.report.buildReport", explode)

        runPipeline(
            pipelineEnv["outputDir"],
            dryRun=True,
            reportHtml=str(reportPath),
        )

        assert not reportPath.exists()
        status = loadStatus(pipelineEnv["outputDir"])
        assert status["steps"]["7_scorer"]["status"] == "ok"
        assert status["steps"]["8_clipping"]["meta"]["dryRun"] is True
