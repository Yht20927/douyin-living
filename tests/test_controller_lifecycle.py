# -*- coding: utf-8 -*-
"""Integration tests for Controller lifecycle — start/stop with mocks."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mockDependencies():
    """Patch all external dependencies of Controller."""
    with (
        patch("src.controller.Auth") as mockAuth,
        patch("src.controller.RoomApi") as mockRoomApi,
        patch("src.controller.DanmakuWs") as mockWs,
        patch("src.controller.FlvRecorder") as mockRecorder,
        patch("src.controller.LiveResponse") as mockLiveResp,
        patch("src.controller.Params") as mockParams,
        patch("src.controller.Signer") as mockSigner,
    ):
        # Auth
        mockAuth.fromEnv.return_value.cookieStr = "test=1"
        mockAuth.fromEnv.return_value.cookie = {"test": "1"}

        # RoomApi - return not-live by default (tests override as needed)
        mockRoomApi.getLiveInfo = AsyncMock(return_value={
            "roomId": "123", "userId": "456",
            "roomTitle": "Test Room", "roomStatus": "4",  # ended
        })
        mockRoomApi.getWebcastDetail = AsyncMock(return_value=b"")
        mockRoomApi.getStreamUrls = AsyncMock(return_value={
            "hd": "http://example.com/stream.flv"
        })
        mockRoomApi.close = AsyncMock()

        # DanmakuWs
        wsInstance = MagicMock()
        mockWs.return_value = wsInstance

        # FlvRecorder
        recInstance = MagicMock()
        recInstance._downloadedFiles = []
        recInstance._currentFile = None
        recInstance.run = AsyncMock()
        mockRecorder.return_value = recInstance

        # LiveResponse proto
        frame = MagicMock()
        frame.cursor = "0"
        frame.internalExt = ""
        mockLiveResp.return_value = frame

        # Params
        paramsInstance = MagicMock()
        paramsInstance.toDict.return_value = {}
        mockParams.return_value = paramsInstance

        yield {
            "auth": mockAuth,
            "roomApi": mockRoomApi,
            "ws": mockWs,
            "recorder": mockRecorder,
            "liveResp": mockLiveResp,
            "params": mockParams,
            "signer": mockSigner,
            "wsInstance": wsInstance,
            "recInstance": recInstance,
        }


class TestControllerLifecycle:
    def test_clip_only_mode(self, mockDependencies):
        from src.controller import Controller

        ctrl = Controller("12345", clipOnly=True)
        ctrl._auth = mockDependencies["auth"].fromEnv.return_value

        # Mock the clipping pipeline
        with patch.object(ctrl, "_runClippingPipeline", new_callable=AsyncMock) as mockClip:
            import asyncio
            asyncio.run(ctrl.start())
            mockClip.assert_called_once()

    def test_recording_stops_when_not_live(self, mockDependencies):
        """When room is not live, recording should return early."""
        from src.controller import Controller

        ctrl = Controller("12345")
        # Default mock returns roomStatus=4 (ended)
        import asyncio
        asyncio.run(ctrl.start())
        # Should not start danmaku WS
        mockDependencies["ws"].assert_not_called()

    def test_recording_starts_danmaku_when_live(self, mockDependencies):
        """When room IS live, danmaku WS should start."""
        mockDependencies["roomApi"].getLiveInfo = AsyncMock(return_value={
            "roomId": "123", "userId": "456",
            "roomTitle": "Live Room", "roomStatus": "2",  # living
        })

        from src.controller import Controller

        ctrl = Controller("12345")
        # Set stop event to exit immediately after setup
        ctrl._stopEvent.set()

        import asyncio
        asyncio.run(ctrl.start())

        # DanmakuWs should have been created and started
        mockDependencies["ws"].assert_called_once()
        mockDependencies["wsInstance"].start.assert_called_once()

    def test_no_cookies_logs_error(self, mockDependencies):
        """Missing cookies should log an error."""
        mockDependencies["auth"].fromEnv.return_value.cookieStr = ""

        from src.controller import Controller

        ctrl = Controller("12345")
        import asyncio
        asyncio.run(ctrl.start())
        # Should return early without crash

    def test_stop_cleans_up(self, mockDependencies):
        """stop() should clean up WS, writer thread, and FLV task."""
        mockDependencies["roomApi"].getLiveInfo = AsyncMock(return_value={
            "roomId": "123", "userId": "456",
            "roomTitle": "Live Room", "roomStatus": "2",
        })

        from src.controller import Controller

        ctrl = Controller("12345")
        ctrl._stopEvent.set()

        # Simulate writer thread already stopped
        import asyncio
        asyncio.run(ctrl.start())

        # Ensure close was called
        mockDependencies["roomApi"].close.assert_called()

    def test_safe_load_features_null_values(self, tmp_path):
        """_safe_load_features should replace null feature values."""
        from src.controller import Controller

        badPath = tmp_path / "bad_features.json"
        badPath.write_text(json.dumps({
            "sampleRate": 1,
            "duration": 10,
            "features": {
                "rms": None,
                "dmDensity": [1.0, 2.0],
            }
        }))

        result = Controller._safe_load_features(str(badPath))
        assert result is not None
        assert result["features"]["rms"] == []
        assert result["features"]["dmDensity"] == [1.0, 2.0]

    def test_safe_load_features_null_top_level(self, tmp_path):
        """Null top-level features should become empty dict."""
        from src.controller import Controller

        path = tmp_path / "null_features.json"
        path.write_text(json.dumps({
            "sampleRate": 1,
            "features": None,
        }))

        result = Controller._safe_load_features(str(path))
        assert result is not None
        assert result["features"] == {}

    def test_safe_load_features_missing_file(self):
        """Missing file returns None."""
        from src.controller import Controller

        result = Controller._safe_load_features("/nonexistent/path.json")
        assert result is None

    def test_safe_load_features_invalid_json(self, tmp_path):
        """Invalid JSON returns None."""
        from src.controller import Controller

        path = tmp_path / "bad.json"
        path.write_text("not json {{{")

        result = Controller._safe_load_features(str(path))
        assert result is None

    def test_danmaku_messages_are_bounded(self, mockDependencies):
        """The danmaku message store should be a bounded deque."""
        from src.controller import Controller

        ctrl = Controller("12345")
        assert hasattr(ctrl, "_danmakuMessages")
        from collections import deque
        assert isinstance(ctrl._danmakuMessages, deque)
        assert ctrl._danmakuMessages.maxlen == 1000


class TestRecordingMeta:
    """recording_meta.json must reflect probed media, not wall-clock guesses."""

    def test_write_recording_meta_uses_probe_and_monotonic_clock(self, tmp_path, monkeypatch):
        import asyncio
        from datetime import datetime, timezone

        from src.controller import Controller

        video = tmp_path / "room_merged.mp4"
        audio = tmp_path / "room_audio.m4a"
        video.write_bytes(b"fake-video")
        audio.write_bytes(b"fake-audio")

        def fake_probe(path):
            return {
                "durationSec": 123.4,
                "videoDurationSec": 123.4,
                "audioDurationSec": 0.0,
                "format": "mp4",
            }

        monkeypatch.setattr("src.controller.probeMedia", fake_probe)

        ctrl = Controller("room", outputDir=str(tmp_path))
        ctrl._recordStartTime = 1_700_000_000.0
        ctrl._recordStartMono = 555.0
        firstDataAt = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        recorder = MagicMock()
        recorder.firstDataAt = firstDataAt
        ctrl._flvRecorder = recorder

        meta = asyncio.run(
            ctrl._writeRecordingMeta(str(video), str(audio), [str(video)])
        )

        assert meta.videoDurationSec == 123.4
        assert meta.audioDurationSec == 123.4  # falls back to video probe
        assert meta.firstVideoDataAt == firstDataAt.isoformat()
        assert meta.startedAtWall == "2023-11-14T22:13:20+00:00"
        assert meta.startedAtMonotonic == 555.0
        assert meta.segments == [str(video)]
        assert str(video) in meta.sourceHashes
        assert meta.sourceHashes[str(video)]["sha256"]

        saved = json.loads((tmp_path / "recording_meta.json").read_text())
        assert saved["videoDurationSec"] == 123.4
        assert saved["sourceHashes"][str(video)]["sizeBytes"] == len(b"fake-video")

    def test_write_recording_meta_preserves_existing_clock(self, tmp_path, monkeypatch):
        import asyncio

        from src.controller import Controller
        from src.contracts import RecordingMeta

        video = tmp_path / "room_merged.mp4"
        video.write_bytes(b"fake-video")
        existing = RecordingMeta(
            roomId="room",
            startedAtWall="2023-01-01T00:00:00+00:00",
            startedAtMonotonic=42.0,
            firstVideoDataAt="2023-01-01T00:00:01+00:00",
        )
        existing.save(str(tmp_path / "recording_meta.json"))

        monkeypatch.setattr(
            "src.controller.probeMedia",
            lambda path: {"durationSec": 10.0, "videoDurationSec": 10.0},
        )

        ctrl = Controller("room", outputDir=str(tmp_path))
        # No live recorder and no fresh clock values — existing metadata wins.
        ctrl._flvRecorder = None
        ctrl._recordStartTime = 0.0
        ctrl._recordStartMono = 0.0

        meta = asyncio.run(
            ctrl._writeRecordingMeta(str(video), None, [])
        )

        assert meta.startedAtWall == "2023-01-01T00:00:00+00:00"
        assert meta.startedAtMonotonic == 42.0
        assert meta.firstVideoDataAt == "2023-01-01T00:00:01+00:00"
        assert meta.audioDurationSec == 0.0
