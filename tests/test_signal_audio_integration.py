# -*- coding: utf-8 -*-
"""Integration test for signalAudio — mock librosa.load and AudioTagging."""

import json
import numpy as np
import pytest


class TestSignalAudioIntegration:
    def test_extract_features_with_mock_librosa(self, tmp_path, monkeypatch):
        """Full audio feature extraction with mocked dependencies."""
        # Mock librosa.load to return synthetic audio
        fakeAudio = np.random.RandomState(42).randn(16000 * 5).astype(np.float32)  # 5s

        class FakeLibrosa:
            @staticmethod
            def load(path, sr=None, mono=True):
                actual_sr = sr or 16000
                return fakeAudio, actual_sr

            @staticmethod
            def get_duration(y=None, sr=None):
                return len(fakeAudio) / (sr or 16000)

            class feature:
                @staticmethod
                def rms(y=None, frame_length=2048, hop_length=512):
                    frames = (len(y) + hop_length - 1) // hop_length if y is not None else 100
                    return np.array([np.ones(frames, dtype=np.float32)])

                @staticmethod
                def zero_crossing_rate(y, hop_length=512):
                    frames = (len(y) + hop_length - 1) // hop_length
                    return np.array([np.full(frames, 0.01, dtype=np.float32)])

                @staticmethod
                def spectral_centroid(y=None, sr=None, hop_length=512):
                    frames = (len(y) + hop_length - 1) // hop_length if y is not None else 100
                    return np.array([np.full(frames, 2000.0, dtype=np.float32)])

                @staticmethod
                def mfcc(y=None, sr=None, n_mfcc=13, hop_length=512):
                    frames = (len(y) + hop_length - 1) // hop_length if y is not None else 100
                    return np.ones((n_mfcc, frames), dtype=np.float32)

        monkeypatch.setattr("src.signalAudio.librosa", FakeLibrosa)

        # Also mock panns import to skip event detection
        monkeypatch.setitem(__import__("sys").modules, "panns_inference", None)

        import sys
        sys.modules["panns_inference"] = type(sys)("panns_inference")
        sys.modules["panns_inference"].AudioTagging = None

        from src.signalAudio import extractFeatures

        outPath = str(tmp_path / "audio_features.json")
        result = extractFeatures("fake.aac", outputPath=outPath)

        assert result["sampleRate"] == 1
        assert result["duration"] > 0
        assert "rms" in result["features"]
        assert "eventLaughter" in result["features"]

        # Verify JSON output
        with open(outPath) as f:
            saved = json.load(f)
        assert saved["sampleRate"] == 1

    def test_extract_features_top_level_exception(self, tmp_path, monkeypatch):
        """When librosa.load throws, extractFeatures should return empty structure."""
        class BrokenLibrosa:
            @staticmethod
            def load(path, sr=None, mono=True):
                raise RuntimeError("simulated load failure")

        monkeypatch.setattr("src.signalAudio.librosa", BrokenLibrosa)

        from src.signalAudio import extractFeatures

        outPath = str(tmp_path / "audio_features.json")
        result = extractFeatures("fake.aac", outputPath=outPath)

        assert result["duration"] == 0
        assert result["features"]["rms"] == []
        assert result["features"]["eventLaughter"] == []

        # Should have written empty JSON
        with open(outPath) as f:
            saved = json.load(f)
        assert saved["duration"] == 0

    def test_empty_audio_produces_valid_output(self, tmp_path, monkeypatch):
        """Very short audio (< 1s) should not crash."""
        fakeAudio = np.zeros(100, dtype=np.float32)

        class TinyLibrosa:
            @staticmethod
            def load(path, sr=None, mono=True):
                return fakeAudio, 16000

            @staticmethod
            def get_duration(y=None, sr=None):
                return len(fakeAudio) / 16000

            class feature:
                @staticmethod
                def rms(y=None, frame_length=2048, hop_length=512):
                    return np.array([np.zeros(1, dtype=np.float32)])

                @staticmethod
                def zero_crossing_rate(y, hop_length=512):
                    return np.array([np.zeros(1, dtype=np.float32)])

                @staticmethod
                def spectral_centroid(y=None, sr=None, hop_length=512):
                    return np.array([np.zeros(1, dtype=np.float32)])

                @staticmethod
                def mfcc(y=None, sr=None, n_mfcc=13, hop_length=512):
                    return np.zeros((n_mfcc, 1), dtype=np.float32)

        monkeypatch.setattr("src.signalAudio.librosa", TinyLibrosa)
        monkeypatch.setitem(__import__("sys").modules, "panns_inference", None)

        from src.signalAudio import extractFeatures

        result = extractFeatures("tiny.aac")
        assert result["sampleRate"] == 1
        assert result["duration"] > 0
