# -*- coding: utf-8 -*-
"""Integration test for signalVisual — mock scenedetect + OpenCV."""

import json
import sys
import numpy as np
import pytest


class FakeVideoCapture:
    """Mock cv2.VideoCapture that returns synthetic frames."""
    def __init__(self, path):
        self._path = path
        self._frameCount = 0
        self._maxFrames = 30  # 1 second at 30fps

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._maxFrames)
        return 0

    def read(self):
        if self._frameCount >= self._maxFrames:
            return (False, None)
        # Return a small synthetic BGR frame
        frame = np.random.RandomState(self._frameCount).randint(
            0, 255, (240, 320, 3), dtype=np.uint8
        )
        self._frameCount += 1
        return (True, frame)

    def release(self):
        pass


class FakeSceneManager:
    def __init__(self):
        self.detectors = []

    def add_detector(self, detector):
        self.detectors.append(detector)

    def detect_scenes(self, video):
        pass

    def get_scene_list(self):
        return []  # No scene cuts


class TestSignalVisualIntegration:
    def test_extract_features_with_mocks(self, tmp_path, monkeypatch):
        """Full visual feature extraction with mocked dependencies."""
        # Mock scenedetect (imported inside extractFeatures function)
        import scenedetect

        fakeVideo = type("FakeVideo", (), {"duration": 1.0})()
        monkeypatch.setattr(scenedetect, "open_video", lambda path: fakeVideo)
        monkeypatch.setattr(scenedetect, "SceneManager", FakeSceneManager)
        FakeContentDetector = type("ContentDetector", (), {"__init__": lambda self, threshold=27.0: None})
        monkeypatch.setattr(scenedetect, "ContentDetector", FakeContentDetector)

        # Mock cv2 (also imported inside the function)
        import cv2
        monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
        # Also mock cv2 in sys.modules so the local import gets the patched version
        # The function does `import cv2` which returns sys.modules['cv2']

        from src.signalVisual import extractFeatures

        outPath = str(tmp_path / "visual_features.json")
        result = extractFeatures("fake.flv", outputPath=outPath)

        assert result["sampleRate"] == 1
        assert result["duration"] > 0
        assert "sceneChange" in result["features"]
        assert "motion" in result["features"]
        assert "faceCount" in result["features"]

        # Verify JSON output
        with open(outPath) as f:
            saved = json.load(f)
        assert saved["sampleRate"] == 1

    def test_extract_features_top_level_exception(self, tmp_path, monkeypatch):
        """When scenedetect throws, extractFeatures should return empty structure."""
        import scenedetect
        monkeypatch.setattr(
            scenedetect,
            "open_video",
            lambda path: (_ for _ in ()).throw(RuntimeError("simulated failure")),
        )

        from src.signalVisual import extractFeatures

        outPath = str(tmp_path / "visual_features.json")
        result = extractFeatures("fake.flv", outputPath=outPath)

        assert result["duration"] == 0
        assert result["features"]["sceneChange"] == []

        with open(outPath) as f:
            saved = json.load(f)
        assert saved["duration"] == 0
