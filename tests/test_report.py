# -*- coding: utf-8 -*-
"""Tests for the S6-1 dry-run HTML report."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src import report
from src.controller import Controller


def _writeArtifacts(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    timeTable = root / "time_table.json"
    timeTable.write_text(
        json.dumps(
            {
                "roomId": "room<&>",
                "duration": 120,
                "profile": "default",
                "pipeline": "two_stage",
                "clips": [
                    {
                        "id": 1,
                        "start": 10.0,
                        "end": 40.0,
                        "duration": 30.0,
                        "score": 9.5,
                        "confidence": 0.8,
                        "trigger": "dmDensity",
                        "title": "高能 <script>alert(1)</script>",
                        "tags": ["互动", "高能"],
                        "candidateSource": "two_stage",
                        "reasons": ["dmDensity", "asrKeyword"],
                        "risks": ["asr_missing"],
                        "dropReasons": ["manual_mark"],
                        "thumbnailTime": 20.0,
                        "breakdown": {"dmDensity": 0.31, "rms": -0.12},
                    },
                    {
                        "id": 2,
                        "start": 70.0,
                        "end": 95.0,
                        "duration": 25.0,
                        "score": 8.0,
                        "trigger": "rms",
                        "title": "第二个候选",
                        "tags": ["音量"],
                        "thumbnailTime": 75.0,
                    },
                ],
                "dropped": [
                    {
                        "start": 50.0,
                        "end": 60.0,
                        "score": 2.0,
                        "reason": "overlap",
                        "keptStart": 10.0,
                        "keptEnd": 40.0,
                    }
                ],
                "adjustments": [
                    {
                        "start": 70.0,
                        "end": 95.0,
                        "originalStart": 68.0,
                        "originalEnd": 98.0,
                        "reason": "boundary_optimized",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scores = root / "scores.json"
    values = [0.1] * 120
    for index in range(10, 40):
        values[index] = 0.6
    for index in range(70, 95):
        values[index] = 0.4
    scores.write_text(
        json.dumps(
            {
                "roomId": "room<&>",
                "duration": 120,
                "pipeline": "two_stage",
                "scores": values,
                "topSignals": ["dmDensity", "rms"],
                "signalStats": {
                    "dmDensity": {
                        "weight": 0.3,
                        "meanZ": 1.2,
                        "maxZ": 4.0,
                        "meanAbsContribution": 0.36,
                    },
                    "rms": {
                        "weight": 0.2,
                        "meanZ": 0.5,
                        "maxZ": 2.0,
                        "meanAbsContribution": 0.1,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    candidates = root / "recall_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "roomId": "room<&>",
                "duration": 120,
                "candidates": [
                    {
                        "id": 1,
                        "start": 10.0,
                        "end": 40.0,
                        "duration": 30.0,
                        "score": 0.9,
                        "peakTime": 20.0,
                        "anchor": "recallCurve",
                        "sources": ["dmDensity"],
                    }
                ],
                "stats": {
                    "candidateCap": 10,
                    "recallPerHour": 60,
                    "recallThreshold": 0.8,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    asr = root / "audio_asr.json"
    asr.write_text(
        json.dumps(
            {
                "status": "ok",
                "language": "zh",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "你好 <b>世界</b>"},
                    {"start": 2.0, "end": 4.0, "text": "测试内容"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    danmaku = root / "room_danmaku.jsonl"
    danmaku.write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "chat", "content": "好吃 好吃 玉米", "recvTimeSec": 1.0},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"type": "chat", "content": "空气炸锅 好吃", "recvTimeSec": 2.0},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "type": "chat",
                        "content": "这个 什么 特别好吃",
                        "recvTimeSec": 3.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"type": "like", "count": 3}, ensure_ascii=False),
                "{not-json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    highlights = root / "highlights"
    highlights.mkdir()
    (highlights / "clip_10s_dmDensity.jpg").write_bytes(b"\xff\xd8fake-jpg")
    return {
        "root": root,
        "timeTable": timeTable,
        "scores": scores,
        "candidates": candidates,
        "asr": asr,
        "danmaku": danmaku,
        "highlights": highlights,
    }


def _build(paths: dict[str, Path], output: Path, **kwargs) -> dict:
    return report.buildReport(
        paths["timeTable"],
        scoresPath=paths["scores"],
        candidatesPath=paths["candidates"],
        asrPath=paths["asr"],
        danmakuPath=paths["danmaku"],
        outputPath=output,
        **kwargs,
    )


class TestBuildReport:
    def test_renders_all_sections_and_escapes_text(self, tmp_path):
        paths = _writeArtifacts(tmp_path / "room")
        output = tmp_path / "report" / "index.html"

        summary = _build(paths, output, makeThumbnails=False)

        text = output.read_text(encoding="utf-8")
        assert output.exists()
        assert summary["html"] == str(output)
        assert summary["mediaDir"] == str(output.parent / "index_media")
        assert summary["candidates"] == 2
        assert summary["dropped"] == 1
        assert summary["scorePoints"] == 120
        assert summary["asrSegments"] == 2
        assert summary["thumbnails"] == 1

        # Titles / ASR / room ids must be escaped.
        assert "高能 &lt;script&gt;alert(1)&lt;/script&gt;" in text
        assert "<script>alert(1)</script>" not in text
        assert "你好 &lt;b&gt;世界&lt;/b&gt;" in text
        assert "room&lt;&amp;&gt;" in text

        # Score curve, clip bands, candidates and copyable commands.
        assert '<polyline class="curve-line"' in text
        assert 'class="clip-band"' in text
        assert "复制 ffmpeg 命令" in text
        assert "ffmpeg -hide_banner -ss 10.000" in text
        assert "复制时间码" in text
        assert "00:10 – 00:40" in text
        assert "#1" in text and "#2" in text
        assert "confidence 0.80" in text

        # Reasons / risks / drops / adjustments.
        assert "触发原因" in text
        assert "asrKeyword" in text
        assert "asr_missing" in text
        assert "manual_mark" in text
        assert "overlap" in text
        assert "boundary_optimized" in text

        # Danmaku top words and signal contributions.
        assert "好吃" in text
        assert "玉米" in text
        assert "空气炸锅" in text
        assert "dmDensity" in text
        assert "|贡献|" in text

        # Two-stage recall section.
        assert "Stage 1 召回候选" in text
        assert "recallCurve" in text

        # One reused thumbnail is embedded as a data URI.
        assert "data:image/jpeg;base64," in text
        assert "无缩略帧" in text

    def test_directory_output_writes_index_and_media(self, tmp_path):
        paths = _writeArtifacts(tmp_path / "room")
        outputDir = tmp_path / "report-dir"

        summary = _build(paths, outputDir, makeThumbnails=False)

        assert Path(summary["html"]) == outputDir / "index.html"
        assert Path(summary["mediaDir"]) == outputDir / "media"
        assert (outputDir / "index.html").exists()
        assert (outputDir / "media").is_dir()

    def test_missing_optional_artifacts_are_tolerated(self, tmp_path):
        root = tmp_path / "room"
        root.mkdir()
        timeTable = root / "time_table.json"
        timeTable.write_text(
            json.dumps(
                {
                    "roomId": "room",
                    "duration": 30,
                    "clips": [
                        {
                            "id": 1,
                            "start": 5,
                            "end": 20,
                            "score": 1.2,
                            "trigger": "rms",
                            "title": "only clip",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        summary = report.buildReport(
            timeTable,
            outputPath=tmp_path / "report",
            makeThumbnails=False,
        )

        text = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
        assert summary["candidates"] == 1
        assert summary["scorePoints"] == 0
        assert summary["thumbnails"] == 0
        assert "scores.json 缺失" in text
        assert "没有 ASR 产物" in text
        assert "没有弹幕数据" in text
        assert "未使用两阶段评分" in text
        assert "only clip" in text

    def test_invalid_time_table_raises(self, tmp_path):
        broken = tmp_path / "time_table.json"
        broken.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="time_table.json is missing or invalid"):
            report.buildReport(broken, outputPath=tmp_path / "report")

    def test_make_thumbnails_uses_source_video(self, tmp_path, monkeypatch):
        paths = _writeArtifacts(tmp_path / "room")
        video = tmp_path / "room.mp4"
        video.write_bytes(b"fake-video")
        calls: list[tuple[Path, float, Path]] = []

        def fakeExtract(source, timeSec, outputPath):
            calls.append((source, timeSec, outputPath))
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            outputPath.write_bytes(b"\xff\xd8extracted")
            return True

        monkeypatch.setattr(report, "_extractThumbnail", fakeExtract)
        summary = _build(paths, tmp_path / "report", videoPath=video)

        assert len(calls) == 2
        assert calls[0][1] == 20.0
        assert calls[1][1] == 75.0
        assert summary["thumbnails"] == 2
        assert (tmp_path / "report" / "media" / "c01.jpg").exists()
        assert (tmp_path / "report" / "media" / "c02.jpg").exists()

    def test_thumbnail_extraction_failure_falls_back(self, tmp_path, monkeypatch):
        paths = _writeArtifacts(tmp_path / "room")
        video = tmp_path / "room.mp4"
        video.write_bytes(b"fake-video")
        monkeypatch.setattr(report, "_extractThumbnail", lambda *args, **kwargs: False)

        summary = _build(paths, tmp_path / "report", videoPath=video)

        # clip 1 falls back to highlights/clip_10s_dmDensity.jpg.
        assert summary["thumbnails"] == 1
        text = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," in text

    def test_asr_failed_and_empty_statuses(self, tmp_path):
        paths = _writeArtifacts(tmp_path / "room")
        failed = tmp_path / "failed_asr.json"
        failed.write_text(
            json.dumps({"status": "failed", "segments": [], "error": "OOM"}),
            encoding="utf-8",
        )
        empty = tmp_path / "empty_asr.json"
        empty.write_text(json.dumps({"segments": []}), encoding="utf-8")

        failedSummary = report.buildReport(
            paths["timeTable"],
            asrPath=failed,
            outputPath=tmp_path / "failed",
            makeThumbnails=False,
        )
        emptySummary = report.buildReport(
            paths["timeTable"],
            asrPath=empty,
            outputPath=tmp_path / "empty",
            makeThumbnails=False,
        )

        failedText = Path(failedSummary["html"]).read_text(encoding="utf-8")
        emptyText = Path(emptySummary["html"]).read_text(encoding="utf-8")
        assert "ASR 失败：OOM" in failedText
        assert "ASR 产物为空" in emptyText


class TestTokenizers:
    def test_tokenize_strips_brackets_and_keeps_cjk_words(self):
        tokens = report._tokenize("[赞] 空气炸锅 好吃 hello123")
        assert "空气炸锅" in tokens
        assert "好吃" in tokens
        assert "hello123" in tokens
        assert "赞" not in tokens

    def test_top_words_filters_stopwords(self):
        words = dict(
            report._topWords(
                [
                    {"content": "这个 什么 好吃"},
                    {"content": "好吃"},
                    {"content": "了 的 吗"},
                ]
            )
        )
        assert words.get("好吃") == 2
        assert "这个" not in words
        assert "什么" not in words
        assert "了" not in words

    def test_chat_messages_skip_unrelated_events(self):
        rows = report._chatMessages(
            [
                {"type": "chat", "content": "hello"},
                {"type": "like", "content": "should not count"},
                {"type": "chat", "content": ""},
                {"content": "legacy"},
                {"type": "chat", "text": "fallback"},
            ]
        )
        assert [row.get("content") or row.get("text") for row in rows] == [
            "hello",
            "legacy",
            "fallback",
        ]


class TestControllerReportHook:
    def test_hook_passes_artifacts_and_skips_missing(self, tmp_path, monkeypatch):
        root = tmp_path / "room"
        root.mkdir()
        timeTable = root / "time_table.json"
        timeTable.write_text("{}", encoding="utf-8")
        scores = root / "scores.json"
        scores.write_text("{}", encoding="utf-8")
        danmaku = root / "room_danmaku.jsonl"
        danmaku.write_text("", encoding="utf-8")
        video = root / "room.mp4"
        video.write_bytes(b"fake")
        captured: dict = {}

        def fakeBuild(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return {"html": "x", "candidates": 0, "thumbnails": 0}

        monkeypatch.setattr("src.report.buildReport", fakeBuild)
        ctrl = Controller("room", reportHtml=str(tmp_path / "report.html"))

        asyncio.run(
            ctrl._writeDryRunReport(
                timeTablePath=str(timeTable),
                scoresPath=str(scores),
                candidatesPath=str(root / "missing_candidates.json"),
                asrPath=str(root / "missing_asr.json"),
                danmakuPath=str(danmaku),
                videoPath=str(video),
            )
        )

        assert captured["args"] == (str(timeTable),)
        assert captured["scoresPath"] == str(scores)
        assert captured["candidatesPath"] is None
        assert captured["asrPath"] is None
        assert captured["danmakuPath"] == str(danmaku)
        assert captured["videoPath"] == str(video)
        assert captured["outputPath"] == str(tmp_path / "report.html")

    def test_hook_failure_is_swallowed(self, tmp_path, monkeypatch):
        ctrl = Controller("room", reportHtml=str(tmp_path / "report.html"))

        def explode(*args, **kwargs):
            raise RuntimeError("report exploded")

        monkeypatch.setattr("src.report.buildReport", explode)
        asyncio.run(
            ctrl._writeDryRunReport(
                timeTablePath=str(tmp_path / "time_table.json"),
                scoresPath=str(tmp_path / "scores.json"),
                candidatesPath=str(tmp_path / "candidates.json"),
                asrPath=str(tmp_path / "asr.json"),
                danmakuPath=str(tmp_path / "danmaku.jsonl"),
                videoPath=None,
            )
        )

    def test_hook_is_noop_without_report_path(self, tmp_path, monkeypatch):
        ctrl = Controller("room")
        called = False

        def fakeBuild(*args, **kwargs):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr("src.report.buildReport", fakeBuild)
        asyncio.run(
            ctrl._writeDryRunReport(
                timeTablePath="t",
                scoresPath="s",
                candidatesPath="c",
                asrPath="a",
                danmakuPath="d",
                videoPath=None,
            )
        )
        assert called is False

    def test_cli_passes_report_html(self, monkeypatch):
        from src import controller

        with patch("src.controller.Controller") as controllerClass:
            controllerClass.return_value.start = AsyncMock()
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "controller",
                    "12345",
                    "--clip-only",
                    "--dry-run",
                    "--report-html",
                    "data/12345/report/index.html",
                ],
            )
            asyncio.run(controller.main())

        kwargs = controllerClass.call_args.kwargs
        assert kwargs["reportHtml"] == "data/12345/report/index.html"
        assert kwargs["dryRun"] is True
        assert kwargs["clipOnly"] is True
