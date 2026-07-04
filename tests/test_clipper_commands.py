# -*- coding: utf-8 -*-
"""Tests for clipper — ffmpeg command assembly, dryRun, timeout paths."""

import json
import os
import subprocess
import pytest
from src.clipper import clipVideo, extractThumbnail, generateClipSrt, burnSubtitles


@pytest.fixture
def timeTable(tmp_path):
    p = tmp_path / "time_table.json"
    p.write_text(json.dumps({
        "clips": [
            {"start": 10.0, "end": 30.0, "trigger": "funny moment"},
            {"start": 60.0, "end": 90.0, "trigger": "highlight"},
        ]
    }), encoding="utf-8")
    return str(p)


@pytest.fixture
def dummyVideo(tmp_path):
    p = tmp_path / "dummy.flv"
    p.write_bytes(b"\x00" * 100)
    return str(p)


class TestClipVideoCommands:
    def test_dryRun_does_not_create_clip_files(self, timeTable, dummyVideo, tmp_path):
        outDir = str(tmp_path / "clips_dry")
        result = clipVideo(dummyVideo, timeTable, outDir, dryRun=True)
        # dryRun still returns clip metadata but doesn't create actual files
        assert len(result) == 2  # two clips in timeTable
        # The output directory may be created but no actual clip files exist
        clipFiles = [f for f in os.listdir(outDir) if f.endswith(".mp4")] if os.path.isdir(outDir) else []
        assert len(clipFiles) == 0

    def test_720p_uses_stream_copy(self, timeTable, dummyVideo, tmp_path, monkeypatch):
        """720p resolution should use -c copy (no re-encode)."""
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # Simulate successful ffmpeg
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        outDir = str(tmp_path / "clips_720")
        result = clipVideo(dummyVideo, timeTable, outDir, resolution="720p")
        assert len(result) == 2
        # First cmd should use -c copy
        assert "-c" in calls[0]
        copyIndex = calls[0].index("-c")
        assert calls[0][copyIndex + 1] == "copy"

    def test_1080p_uses_scale_filter(self, timeTable, dummyVideo, tmp_path, monkeypatch):
        """1080p resolution should re-encode with scale filter."""
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        outDir = str(tmp_path / "clips_1080")
        result = clipVideo(dummyVideo, timeTable, outDir, resolution="1080p")
        assert len(result) == 2
        # Should have scale filter
        assert "-vf" in calls[0]

    def test_clip_shorter_than_1s_skipped(self, dummyVideo, tmp_path, monkeypatch):
        """Clips < 1s duration are skipped."""
        timeTablePath = tmp_path / "short.json"
        timeTablePath.write_text(json.dumps({
            "clips": [{"start": 10.0, "end": 10.5, "trigger": "too short"}]
        }), encoding="utf-8")

        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
        outDir = str(tmp_path / "clips_skip")
        result = clipVideo(dummyVideo, str(timeTablePath), outDir)
        assert len(result) == 0

    def test_timeout_returns_partial_results(self, timeTable, dummyVideo, tmp_path, monkeypatch):
        """Timeout on one clip should skip it but continue with others."""
        callCount = [0]
        def fake_run(cmd, **kwargs):
            callCount[0] += 1
            if callCount[0] == 1:
                raise subprocess.TimeoutExpired(cmd, 300)
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        outDir = str(tmp_path / "clips_timeout")
        result = clipVideo(dummyVideo, timeTable, outDir)
        # Second clip should still succeed
        assert len(result) == 1

    def test_called_process_error_skipped(self, timeTable, dummyVideo, tmp_path, monkeypatch):
        """ffmpeg error on one clip should skip it."""
        callCount = [0]
        def fake_run(cmd, **kwargs):
            callCount[0] += 1
            if callCount[0] == 1:
                raise subprocess.CalledProcessError(1, cmd, stderr=b"error")
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        outDir = str(tmp_path / "clips_error")
        result = clipVideo(dummyVideo, timeTable, outDir)
        assert len(result) == 1


class TestExtractThumbnail:
    def test_dryRun_does_not_call_ffmpeg(self, dummyVideo, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
        outPath = str(tmp_path / "thumb.jpg")
        extractThumbnail(dummyVideo, 5.0, outPath, dryRun=True)
        assert len(calls) == 0
        assert not os.path.exists(outPath)

    def test_successful_extraction(self, dummyVideo, tmp_path, monkeypatch):
        """Successful thumbnail extraction should log and return."""
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # Simulate successful thumbnail creation
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake jpg")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        outPath = str(tmp_path / "thumb.jpg")
        extractThumbnail(dummyVideo, 5.0, outPath)
        assert len(calls) == 1
        assert os.path.exists(outPath)

    def test_timeout_handled(self, dummyVideo, tmp_path, monkeypatch):
        """Timeout in extractThumbnail should not raise."""
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)
        monkeypatch.setattr(subprocess, "run", fake_run)
        outPath = str(tmp_path / "thumb.jpg")
        # Should not raise
        extractThumbnail(dummyVideo, 5.0, outPath)


class TestGenerateClipSrt:
    def test_missing_srt_returns_early(self, tmp_path):
        """Missing SRT file should return without error."""
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt("/nonexistent/path.srt", 0, 60, outPath)
        assert not os.path.exists(outPath)

    def test_extracts_correct_range(self, tmp_path):
        srtPath = tmp_path / "test.srt"
        srtPath.write_text(
            "1\n00:00:05,000 --> 00:00:10,000\nHello\n\n"
            "2\n00:00:15,000 --> 00:00:20,000\nWorld\n\n"
            "3\n00:00:25,000 --> 00:00:30,000\nTest\n",
            encoding="utf-8",
        )
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt(str(srtPath), 8.0, 22.0, outPath)

        with open(outPath, encoding="utf-8") as f:
            content = f.read()
        # Entry 1 (5-10s) overlaps with [8,22] → included
        # Entry 2 (15-20s) fully within → included
        # Entry 3 (25-30s) outside → excluded
        assert "Hello" in content
        assert "World" in content
        assert "Test" not in content

    def test_windows_line_endings(self, tmp_path):
        """Windows \\r\\n line endings should be handled."""
        srtPath = tmp_path / "win.srt"
        srtPath.write_text(
            "1\r\n00:00:05,000 --> 00:00:10,000\r\nHello\r\n",
            encoding="utf-8",
        )
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt(str(srtPath), 0, 60, outPath)
        with open(outPath, encoding="utf-8") as f:
            content = f.read()
        assert "Hello" in content

    def test_safeName_preserves_plus(self, timeTable, dummyVideo, tmp_path, monkeypatch):
        """safeName should preserve '+' characters in filenames."""
        timeTablePath = tmp_path / "table.json"
        timeTablePath.write_text(json.dumps({
            "clips": [{"start": 10.0, "end": 30.0, "trigger": "test+clip"}]
        }), encoding="utf-8")
        outDir = str(tmp_path / "clips_plus")

        def fake_run(cmd, **kwargs):
            outPath = cmd[cmd.index("-y") + 1]
            with open(outPath, "wb") as f:
                f.write(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = clipVideo(dummyVideo, str(timeTablePath), outDir)
        assert len(result) == 1
        # The output filename should contain the + character
        assert "+" in result[0][0]

    def test_partial_overlap_truncated(self, tmp_path):
        """Subtitles that partially overlap clip boundaries should be truncated."""
        srtPath = tmp_path / "partial.srt"
        srtPath.write_text(
            "1\n00:00:03,000 --> 00:00:08,000\nEarly\n\n"
            "2\n00:00:18,000 --> 00:00:25,000\nLate\n",
            encoding="utf-8",
        )
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt(str(srtPath), 5.0, 20.0, outPath)

        with open(outPath, encoding="utf-8") as f:
            content = f.read()
        # Entry 1: 3-8s, clip 5-20s → newStart=0, newEnd=3
        assert "Early" in content
        assert "00:00:00,000 --> 00:00:03,000" in content
        # Entry 2: 18-25s, clip 5-20s → newStart=13, newEnd=15
        assert "Late" in content
        assert "00:00:13,000 --> 00:00:15,000" in content

    def test_empty_srt_file(self, tmp_path):
        """Empty SRT file should not crash."""
        srtPath = tmp_path / "empty.srt"
        srtPath.write_text("", encoding="utf-8")
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt(str(srtPath), 0, 60, outPath)
        assert not os.path.exists(outPath)

    def test_carriage_return_only(self, tmp_path):
        """Old-Mac-style \\r line endings should be handled."""
        srtPath = tmp_path / "mac.srt"
        srtPath.write_text(
            "1\r00:00:05,000 --> 00:00:10,000\rHello\r",
            encoding="utf-8",
        )
        outPath = str(tmp_path / "clip.srt")
        generateClipSrt(str(srtPath), 0, 60, outPath)
        with open(outPath, encoding="utf-8") as f:
            content = f.read()
        assert "Hello" in content


class TestBurnSubtitles:
    def test_dryRun(self, dummyVideo, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
        outPath = str(tmp_path / "burned.mp4")
        burnSubtitles(dummyVideo, "/fake/subtitles.srt", outPath, dryRun=True)
        assert len(calls) == 0

    def test_ffmpeg_failure_not_fatal(self, dummyVideo, tmp_path, monkeypatch):
        """ffmpeg failure in burnSubtitles should log a warning, not raise."""
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr=b"burn error")
        monkeypatch.setattr(subprocess, "run", fake_run)
        outPath = str(tmp_path / "burned.mp4")
        # Should not raise
        burnSubtitles(dummyVideo, "/fake/subtitles.srt", outPath)

    def test_timeout_not_fatal(self, dummyVideo, tmp_path, monkeypatch):
        """Timeout in burnSubtitles should log a warning, not raise."""
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 300)
        monkeypatch.setattr(subprocess, "run", fake_run)
        outPath = str(tmp_path / "burned.mp4")
        burnSubtitles(dummyVideo, "/fake/subtitles.srt", outPath)

    def test_successful_burn(self, dummyVideo, tmp_path, monkeypatch):
        """Successful subtitle burn should log and return."""
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        monkeypatch.setattr(subprocess, "run", fake_run)
        outPath = str(tmp_path / "burned.mp4")
        burnSubtitles(dummyVideo, "/fake/subtitles.srt", outPath)
        assert len(calls) == 1
