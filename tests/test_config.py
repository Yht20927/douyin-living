# -*- coding: utf-8 -*-
"""Tests for config module — settings loading, defaults, env overrides."""

import os
import pytest
from src.config import (
    load_settings,
    reload_settings,
    ScorerSettings,
    AudioSettings,
    TextSettings,
    VisualSettings,
    RecordingSettings,
    PipelineSettings,
)


class TestLoadSettings:
    def test_loads_all_sections(self):
        s = load_settings()
        assert s.scorer is not None
        assert s.audio is not None
        assert s.text is not None
        assert s.visual is not None
        assert s.recording is not None
        assert s.pipeline is not None

    def test_profiles_loaded(self):
        s = load_settings()
        assert "default" in s.scorer.profiles
        assert "game" in s.scorer.profiles

    def test_cached_result(self):
        """load_settings() is cached — same object returned."""
        s1 = load_settings()
        s2 = load_settings()
        assert s1 is s2

    def test_reload_clears_cache(self):
        """reload_settings() clears the cache."""
        s1 = load_settings()
        reload_settings()
        s2 = load_settings()
        # After cache clear, a new instance is created
        assert s1 is not s2


class TestScorerSettingsDefaults:
    def test_detection_defaults(self):
        s = ScorerSettings()
        assert s.detection.windows == [3, 10, 30]
        assert s.detection.nms_merge_distance == 15.0
        assert s.detection.min_clip_duration == 15.0
        assert s.detection.max_clip_duration == 90.0

    def test_preprocessing_defaults(self):
        s = ScorerSettings()
        assert s.preprocessing.ewma_alpha == 0.3
        assert s.preprocessing.zscore_window_sec == 60
        assert s.preprocessing.median_filter_size == 5


class TestAudioSettingsDefaults:
    def test_defaults(self):
        s = AudioSettings()
        assert s.sample_rate == 16000
        assert s.hop_length == 512
        assert s.panns_rms_threshold == 2.0
        assert s.panns_batch_size_gpu == 32
        assert s.panns_batch_size_cpu == 8

    def test_event_indices(self):
        s = AudioSettings()
        assert s.event_laughter_idx == 32
        assert s.event_applause_idx == 10
        assert s.event_music_idx == 137


class TestTextSettingsDefaults:
    def test_defaults(self):
        s = TextSettings()
        assert s.similarity_threshold == 0.7
        assert s.utr_threshold == 0.3
        assert s.sentiment_max_len == 512


class TestVisualSettingsDefaults:
    def test_defaults(self):
        s = VisualSettings()
        assert s.scene_threshold == 27.0
        assert s.fps_fallback == 30
        assert s.opticalflow_resolution == (320, 240)
        assert s.opticalflow_sample_interval == 5


class TestRecordingSettingsDefaults:
    def test_defaults(self):
        s = RecordingSettings()
        assert s.rotation_secs == 1800
        assert s.max_retries == 5
        assert s.chunk_size == 65536


class TestPipelineSettingsDefaults:
    def test_defaults(self):
        s = PipelineSettings()
        assert s.sensitivity_default == 2.5
        assert s.signal_pool_workers == 3
        assert s.ffmpeg_timeout_clip == 300
