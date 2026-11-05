# -*- coding: utf-8 -*-
"""FlvRecorder resilience tests — atomic segments, validation and disk guards."""

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.flvRecorder as flvRecorder
from src.flvRecorder import FlvIdleTimeout, FlvRecorder


def _make(**kwargs) -> FlvRecorder:
    return FlvRecorder(
        roomId="room",
        url="http://example.com/live.flv",
        outputDir=kwargs.pop("outputDir", None),
        **kwargs,
    )


class TestIdleDetection:
    def test_next_chunk_raises_when_stream_stalls(self):
        recorder = _make()
        recorder._idleTimeout = 0.01

        async def stall():
            await asyncio.sleep(5)
            yield b"never"

        async def main():
            iterator = stall().__aiter__()
            with pytest.raises(FlvIdleTimeout, match="no data"):
                await recorder._nextChunk(iterator)

        asyncio.run(main())

    def test_next_chunk_returns_data_then_eof(self):
        recorder = _make()

        async def chunks():
            yield b"a"
            yield b"b"

        async def main():
            iterator = chunks().__aiter__()
            assert await recorder._nextChunk(iterator) == b"a"
            assert await recorder._nextChunk(iterator) == b"b"
            with pytest.raises(StopAsyncIteration):
                await recorder._nextChunk(iterator)

        asyncio.run(main())


class TestSegmentFinalize:
    def _writePart(self, tmp_path, name="room_hd_1.flv.part", size=128):
        part = tmp_path / name
        part.write_bytes(b"x" * size)
        return part

    def test_valid_part_is_atomically_published(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        part = self._writePart(tmp_path)
        final = tmp_path / "room_hd_1.flv"
        monkeypatch.setattr(flvRecorder.shutil, "which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr(
            flvRecorder,
            "probeMedia",
            lambda path: {"durationSec": 12.5, "hasVideo": True, "hasAudio": True},
        )

        recorder._finalizeSegment(str(part), str(final))

        assert not part.exists()
        assert final.exists()
        assert recorder.stats["downloadedFiles"] == [str(final)]
        assert recorder.stats["corruptFiles"] == []

    def test_corrupt_part_is_quarantined(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        part = self._writePart(tmp_path)
        final = tmp_path / "room_hd_1.flv"
        monkeypatch.setattr(flvRecorder.shutil, "which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr(flvRecorder, "probeMedia", lambda path: {})

        recorder._finalizeSegment(str(part), str(final))

        quarantined = tmp_path / "corrupt" / "room_hd_1.flv"
        assert not part.exists()
        assert not final.exists()
        assert quarantined.exists()
        assert recorder.stats["downloadedFiles"] == []
        assert recorder.stats["corruptFiles"] == [str(quarantined)]
        assert "corrupt segment" in recorder.stats["lastError"]

    def test_zero_duration_part_is_quarantined(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        part = self._writePart(tmp_path)
        monkeypatch.setattr(flvRecorder.shutil, "which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr(
            flvRecorder,
            "probeMedia",
            lambda path: {"durationSec": 0, "hasVideo": True, "hasAudio": True},
        )

        recorder._finalizeSegment(str(part), str(tmp_path / "room_hd_1.flv"))

        assert (tmp_path / "corrupt" / "room_hd_1.flv").exists()

    def test_missing_ffprobe_accepts_segment(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        part = self._writePart(tmp_path)
        final = tmp_path / "room_hd_1.flv"
        monkeypatch.setattr(flvRecorder.shutil, "which", lambda name: None)
        monkeypatch.setattr(flvRecorder, "probeMedia", lambda path: {})

        recorder._finalizeSegment(str(part), str(final))

        assert final.exists()
        assert recorder.stats["downloadedFiles"] == [str(final)]

    def test_empty_part_is_removed(self, tmp_path):
        recorder = _make(outputDir=str(tmp_path))
        part = self._writePart(tmp_path, size=0)

        recorder._finalizeSegment(str(part), str(tmp_path / "room_hd_1.flv"))

        assert not part.exists()
        assert recorder.stats["downloadedFiles"] == []


class TestDiskGuard:
    def test_low_disk_space_is_detected(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        recorder._minFreeBytes = 2 * 1024 ** 3
        monkeypatch.setattr(
            flvRecorder.shutil,
            "disk_usage",
            lambda path: SimpleNamespace(
                total=100 * 1024 ** 3, used=99 * 1024 ** 3, free=1024 ** 3
            ),
        )

        assert recorder._hasDiskSpace() is False
        assert "low disk space" in recorder.stats["lastError"]

    def test_enough_disk_space_passes(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        recorder._minFreeBytes = 2 * 1024 ** 3
        monkeypatch.setattr(
            flvRecorder.shutil,
            "disk_usage",
            lambda path: SimpleNamespace(
                total=100 * 1024 ** 3, used=1 * 1024 ** 3, free=99 * 1024 ** 3
            ),
        )

        assert recorder._hasDiskSpace() is True
        assert recorder.stats["lastError"] is None

    def test_low_disk_stops_before_download(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        monkeypatch.setattr(recorder, "_hasDiskSpace", lambda: False)
        called = False

        async def fakeSegment():
            nonlocal called
            called = True
            return "rotation"

        monkeypatch.setattr(recorder, "_downloadSegment", fakeSegment)

        asyncio.run(recorder._downloadLoop())

        assert called is False
        assert recorder.stats["stopReason"] == "low_disk"


class TestHousekeeping:
    def test_stale_parts_are_removed_but_fresh_parts_kept(self, tmp_path):
        recorder = _make(outputDir=str(tmp_path))
        old = tmp_path / "old.flv.part"
        fresh = tmp_path / "fresh.flv.part"
        old.write_bytes(b"old")
        fresh.write_bytes(b"fresh")
        oldTime = time.time() - flvRecorder._cfg.stale_part_max_age_secs - 60
        os.utime(old, (oldTime, oldTime))

        recorder._cleanupStaleParts()

        assert not old.exists()
        assert fresh.exists()

    def test_stats_exposes_resilience_fields(self):
        stats = _make().stats
        for key in (
            "downloadedFiles",
            "corruptFiles",
            "idleReconnects",
            "streamEnded",
            "stopReason",
            "lastError",
        ):
            assert key in stats


class _FakeContent:
    def __init__(self, chunks, delay=0.0):
        self._chunks = chunks
        self._delay = delay

    def iter_chunked(self, chunkSize):
        async def generator():
            for chunk in self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield chunk

        return generator()


class _FakeResponse:
    def __init__(self, chunks, delay=0.0):
        self.content = _FakeContent(chunks, delay)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None


def _installFakeAiohttp(monkeypatch, chunks, delay=0.0):
    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url):
            return _FakeResponse(chunks, delay)

    monkeypatch.setattr(flvRecorder.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(flvRecorder.aiohttp, "TCPConnector", lambda **kwargs: None)
    monkeypatch.setattr(flvRecorder.aiohttp, "ClientTimeout", lambda **kwargs: None)


def _acceptSegments(monkeypatch):
    monkeypatch.setattr(flvRecorder.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(flvRecorder, "probeMedia", _validProbe)


def _validProbe(path):
    return {"durationSec": 5.0, "hasVideo": True, "hasAudio": True}


class TestDownloadSegment:
    def test_eof_publishes_atomically(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        _installFakeAiohttp(monkeypatch, [b"a" * 64, b"b" * 64])
        _acceptSegments(monkeypatch)

        reason = asyncio.run(recorder._downloadSegment())

        files = list(tmp_path.glob("room_hd_*.flv"))
        assert reason == "eof"
        assert len(files) == 1
        assert files[0].stat().st_size == 128
        assert list(tmp_path.glob("*.part")) == []
        assert recorder.stats["downloadedFiles"] == [str(files[0])]
        assert recorder.stats["streamEnded"] is True
        assert recorder.stats["bytesWritten"] == 128
        assert recorder.stats["currentFile"] is None

    def test_rotation_interval_stops_segment(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path), rotationSecs=0)
        _installFakeAiohttp(monkeypatch, [b"a" * 64, b"b" * 64, b"c" * 64])
        _acceptSegments(monkeypatch)

        reason = asyncio.run(recorder._downloadSegment())

        files = list(tmp_path.glob("room_hd_*.flv"))
        assert reason == "rotation"
        assert len(files) == 1
        assert files[0].stat().st_size == 64
        assert recorder.stats["streamEnded"] is False

    def test_size_cap_stops_segment(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        recorder._maxSegmentBytes = 100
        _installFakeAiohttp(monkeypatch, [b"a" * 64, b"b" * 64, b"c" * 64])
        _acceptSegments(monkeypatch)

        reason = asyncio.run(recorder._downloadSegment())

        files = list(tmp_path.glob("room_hd_*.flv"))
        assert reason == "size_cap"
        assert files[0].stat().st_size == 128

    def test_low_disk_mid_segment_stops_and_keeps_data(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        monkeypatch.setattr(flvRecorder, "_DISK_CHECK_INTERVAL_BYTES", 1)
        monkeypatch.setattr(recorder, "_hasDiskSpace", lambda: False)
        _installFakeAiohttp(monkeypatch, [b"a" * 64, b"b" * 64])
        _acceptSegments(monkeypatch)

        reason = asyncio.run(recorder._downloadSegment())

        files = list(tmp_path.glob("room_hd_*.flv"))
        assert reason == "low_disk"
        assert len(files) == 1
        assert files[0].stat().st_size == 64

    def test_idle_stream_raises_after_cleaning_empty_part(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        recorder._idleTimeout = 0.01
        _installFakeAiohttp(monkeypatch, [b"late"], delay=0.1)
        _acceptSegments(monkeypatch)

        with pytest.raises(FlvIdleTimeout):
            asyncio.run(recorder._downloadSegment())

        assert list(tmp_path.glob("*.flv")) == []
        assert list(tmp_path.glob("*.part")) == []


class TestDownloadLoop:
    def test_eof_retries_exhausted_assumes_live_ended(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        calls = 0

        async def fakeSegment():
            nonlocal calls
            calls += 1
            recorder._streamEnded = True
            return "eof"

        monkeypatch.setattr(recorder, "_downloadSegment", fakeSegment)
        monkeypatch.setattr(recorder, "_waitOrStop", AsyncMock())

        asyncio.run(recorder._downloadLoop())

        assert calls == flvRecorder._cfg.max_retries + 1
        assert recorder.stats["stopReason"] == "stream_ended"
        assert recorder.stats["streamEnded"] is True

    def test_stop_event_ends_loop_cleanly(self, tmp_path, monkeypatch):
        recorder = _make(outputDir=str(tmp_path))
        recorder._stopEvent.set()

        async def fakeSegment():
            raise AssertionError("should not be called after stop")

        monkeypatch.setattr(recorder, "_downloadSegment", fakeSegment)

        asyncio.run(recorder._downloadLoop())

        assert recorder.stats["stopReason"] == "stopped"
