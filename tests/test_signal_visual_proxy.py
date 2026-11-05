# -*- coding: utf-8 -*-
"""S5-4 tests — proxy generation, single-pass scene/motion, fallback contract."""

import os
from pathlib import Path

import numpy as np
import pytest

import src.signalVisual as signalVisual


class _FakeCapture:
    """Deterministic in-memory capture used for the proxy pass."""

    def __init__(self, frames, fps=2.0):
        self._frames = list(frames)
        self._fps = fps
        self._index = 0

    def get(self, prop):
        cv2 = pytest.importorskip("cv2")
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        pass


class TestSceneScore:
    def test_identical_frames_score_zero(self):
        cv2 = pytest.importorskip("cv2")
        frame = np.full((24, 32, 3), 120, dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        assert signalVisual._sceneChangeScore(hsv, hsv) == 0.0

    def test_brightness_cut_exceeds_default_threshold(self):
        cv2 = pytest.importorskip("cv2")
        dark = np.full((24, 32, 3), 20, dtype=np.uint8)
        bright = np.full((24, 32, 3), 240, dtype=np.uint8)
        score = signalVisual._sceneChangeScore(
            cv2.cvtColor(dark, cv2.COLOR_BGR2HSV),
            cv2.cvtColor(bright, cv2.COLOR_BGR2HSV),
        )
        assert score >= 27.0


class TestBuildProxy:
    def test_missing_input_returns_error(self):
        path, info = signalVisual._buildProxy("/tmp/does-not-exist-s54.flv")
        assert path is None
        assert "not found" in info["error"]

    def test_ffmpeg_missing_returns_error(self, tmp_path, monkeypatch):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"not a real video")

        def missing(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(signalVisual.subprocess, "run", missing)
        path, info = signalVisual._buildProxy(str(source))
        assert path is None
        assert info["error"] == "ffmpeg not installed"

    def test_command_uses_bounded_proxy(self, tmp_path, monkeypatch):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"not a real video")
        seen = []

        def fakeRun(command, **kwargs):
            seen.append(command)
            Path(command[-1]).write_bytes(b"proxy")
            return type("Proc", (), {"returncode": 0, "stderr": ""})()

        monkeypatch.setattr(signalVisual.subprocess, "run", fakeRun)
        path, info = signalVisual._buildProxy(
            str(source), sourceHeight=1080, sourceFps=30.0
        )
        try:
            assert path is not None
            assert info["encoder"] == "libx264"
            assert info["height"] == 480
            assert info["sampleFps"] == 2.0
            videoFilter = seen[0][seen[0].index("-vf") + 1]
            assert "fps=2" in videoFilter
            assert "scale=-2:480" in videoFilter
        finally:
            if path:
                os.remove(path)


class TestExtractProxySignals:
    def test_single_pass_detects_cut_and_motion(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        frames = [
            np.full((240, 320, 3), 20, dtype=np.uint8),
            np.full((240, 320, 3), 20, dtype=np.uint8),
            np.full((240, 320, 3), 20, dtype=np.uint8),
            np.full((240, 320, 3), 240, dtype=np.uint8),
            np.full((240, 320, 3), 240, dtype=np.uint8),
            np.full((240, 320, 3), 240, dtype=np.uint8),
        ]
        fake = _FakeCapture(frames, fps=2.0)
        monkeypatch.setattr(cv2, "VideoCapture", lambda path: fake)

        def fakeFlow(previous, current, *args, **kwargs):
            return np.full((*previous.shape, 2), 0.5, dtype=np.float32)

        monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", fakeFlow)

        result = signalVisual._extractProxySignals("proxy.mp4")
        assert result["duration"] == 3
        assert result["sceneChange"] == [0, 1, 0]
        assert result["sceneCount"] == 1
        assert result["frames"] == 6
        assert all(value > 0 for value in result["motion"])

    def test_expected_timeline_pads_short_proxy(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fake = _FakeCapture([frame, frame], fps=2.0)
        monkeypatch.setattr(cv2, "VideoCapture", lambda path: fake)
        monkeypatch.setattr(
            cv2,
            "calcOpticalFlowFarneback",
            lambda *args, **kwargs: np.zeros((240, 320, 2), dtype=np.float32),
        )

        result = signalVisual._extractProxySignals("proxy.mp4", expectedSecs=5)
        assert result["duration"] == 5
        assert result["sceneChange"] == [0] * 5
        assert result["motion"] == [0.0] * 5


class TestExtractFeaturesModes:
    def _fakeResult(self, seconds=4):
        return {
            "sceneChange": [0, 1, 0, 0],
            "motion": [0.1, 0.2, 0.3, 0.4],
            "faceCount": [0, 0, 0, 0],
            "duration": seconds,
            "sceneCount": 1,
            "frames": seconds * 2,
            "fps": 2.0,
        }

    def test_proxy_path_cleans_up_and_reports_stats(self, tmp_path, monkeypatch):
        proxy = tmp_path / "proxy.mp4"
        proxy.write_bytes(b"proxy")
        monkeypatch.setattr(
            signalVisual,
            "_probeVideo",
            lambda path: {
                "duration": 0.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "frameCount": 0.0,
            },
        )
        monkeypatch.setattr(signalVisual._cfg, "proxy_enabled", True)
        monkeypatch.setattr(
            signalVisual,
            "_buildProxy",
            lambda *args, **kwargs: (
                str(proxy),
                {
                    "encoder": "libx264",
                    "height": 480,
                    "sampleFps": 2.0,
                },
            ),
        )
        monkeypatch.setattr(
            signalVisual,
            "_extractProxySignals",
            lambda path, expected: self._fakeResult(expected or 4),
        )

        outPath = tmp_path / "visual.json"
        result = signalVisual.extractFeatures(
            "source.mp4", outputPath=str(outPath), durationSec=4.0
        )

        assert not proxy.exists(), "temporary proxy must be removed"
        assert result["duration"] == 4
        assert result["stats"]["mode"] == "proxy"
        assert result["stats"]["sceneDetector"] == "hsv-diff"
        assert result["stats"]["sourceDecodePasses"] == 1
        assert result["stats"]["proxy"]["enabled"] is True
        assert result["stats"]["proxy"]["encoder"] == "libx264"
        assert outPath.exists()

    def test_proxy_failure_falls_back_to_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            signalVisual,
            "_probeVideo",
            lambda path: {
                "duration": 4.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "frameCount": 120.0,
            },
        )
        monkeypatch.setattr(signalVisual._cfg, "proxy_enabled", True)
        monkeypatch.setattr(
            signalVisual,
            "_buildProxy",
            lambda *args, **kwargs: (None, {"error": "simulated ffmpeg failure"}),
        )
        monkeypatch.setattr(
            signalVisual,
            "_extractLegacySignals",
            lambda path, expected: self._fakeResult(expected or 4),
        )

        result = signalVisual.extractFeatures("source.mp4")

        assert result["stats"]["mode"] == "legacy"
        assert result["stats"]["sceneDetector"] == "pyscenedetect"
        assert result["stats"]["sourceDecodePasses"] == 2
        assert result["stats"]["proxy"]["enabled"] is False
        assert result["stats"]["proxy"]["requested"] is True
        assert "simulated" in result["stats"]["proxy"]["fallbackReason"]

    def test_proxy_decode_failure_falls_back(self, tmp_path, monkeypatch):
        proxy = tmp_path / "proxy.mp4"
        proxy.write_bytes(b"proxy")
        monkeypatch.setattr(
            signalVisual,
            "_probeVideo",
            lambda path: {
                "duration": 4.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "frameCount": 120.0,
            },
        )
        monkeypatch.setattr(signalVisual._cfg, "proxy_enabled", True)
        monkeypatch.setattr(
            signalVisual,
            "_buildProxy",
            lambda *args, **kwargs: (str(proxy), {"height": 480, "sampleFps": 2.0}),
        )

        def broken(path, expected):
            raise RuntimeError("cannot decode proxy")

        monkeypatch.setattr(signalVisual, "_extractProxySignals", broken)
        monkeypatch.setattr(
            signalVisual,
            "_extractLegacySignals",
            lambda path, expected: self._fakeResult(expected or 4),
        )

        result = signalVisual.extractFeatures("source.mp4")

        assert not proxy.exists()
        assert result["stats"]["mode"] == "legacy"
        assert "cannot decode proxy" in result["stats"]["proxy"]["fallbackReason"]

    def test_disabled_proxy_skips_generation(self, monkeypatch):
        monkeypatch.setattr(
            signalVisual,
            "_probeVideo",
            lambda path: {
                "duration": 4.0,
                "fps": 30.0,
                "width": 1280,
                "height": 720,
                "frameCount": 120.0,
            },
        )
        monkeypatch.setattr(signalVisual._cfg, "proxy_enabled", False)

        def mustNotRun(*args, **kwargs):
            raise AssertionError("proxy must not be generated")

        monkeypatch.setattr(signalVisual, "_buildProxy", mustNotRun)
        monkeypatch.setattr(
            signalVisual,
            "_extractLegacySignals",
            lambda path, expected: self._fakeResult(expected or 4),
        )

        result = signalVisual.extractFeatures("source.mp4")
        assert result["stats"]["mode"] == "legacy"
        assert result["stats"]["proxy"]["requested"] is False
