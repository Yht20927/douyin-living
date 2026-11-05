# -*- coding: utf-8 -*-
"""Tests for config module — settings loading, defaults, env overrides."""

import pytest
from pydantic import ValidationError

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
        assert s.asr_model == "base"
        assert s.asr_device == "auto"
        assert s.asr_compute_type == "int8"
        assert s.asr_diarize is True


class TestEnvironmentOverrides:
    """YAML must be a *default*, not a constructor argument that outranks env."""

    def test_nested_scorer_env_override(self, monkeypatch):
        monkeypatch.setenv("SCORER_DETECTION__WINDOWS", "[5,15,45]")
        settings = ScorerSettings()
        assert settings.detection.windows == [5, 15, 45]

    def test_flat_env_override(self, monkeypatch):
        monkeypatch.setenv("AUDIO_PANNS_RMS_THRESHOLD", "9.9")
        settings = AudioSettings()
        assert settings.panns_rms_threshold == 9.9

    def test_pipeline_min_duration_env(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_MIN_DURATION_DEFAULT", "8.5")
        settings = PipelineSettings()
        assert settings.min_duration_default == 8.5

    def test_pipeline_asr_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_ASR_MODEL", "tiny")
        monkeypatch.setenv("PIPELINE_ASR_DEVICE", "cpu")
        monkeypatch.setenv("PIPELINE_ASR_COMPUTE_TYPE", "float32")
        monkeypatch.setenv("PIPELINE_ASR_DIARIZE", "false")
        settings = PipelineSettings()
        assert settings.asr_model == "tiny"
        assert settings.asr_device == "cpu"
        assert settings.asr_compute_type == "float32"
        assert settings.asr_diarize is False

    def test_explicit_init_still_wins(self, monkeypatch):
        monkeypatch.setenv("AUDIO_PANNS_RMS_THRESHOLD", "9.9")
        settings = AudioSettings(panns_rms_threshold=1.25)
        assert settings.panns_rms_threshold == 1.25


class TestProfileValidation:
    def test_profiles_are_normalized(self):
        settings = ScorerSettings(
            profiles={"p": {"dmDensity": 1.0, "rms": 3.0}},
        )
        assert sum(settings.profiles["p"].values()) == pytest.approx(1.0)
        assert settings.profiles["p"]["rms"] == pytest.approx(0.75)

    def test_non_unit_sum_rejected_when_normalization_disabled(self):
        with pytest.raises(ValidationError, match="weights sum"):
            ScorerSettings(
                normalize_profiles=False,
                profiles={"p": {"dmDensity": 0.5, "rms": 0.3}},
            )

    def test_unknown_feature_key_rejected(self):
        with pytest.raises(ValidationError, match="unknown feature"):
            ScorerSettings(profiles={"p": {"notAFeature": 1.0}})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError, match="negative weight"):
            ScorerSettings(profiles={"p": {"dmDensity": -0.5, "rms": 1.5}})

    def test_empty_profile_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ScorerSettings(profiles={"p": {}})


class TestStructuralValidation:
    def test_min_duration_must_be_below_max(self):
        with pytest.raises(ValidationError, match="min_clip_duration"):
            ScorerSettings(
                detection={"min_clip_duration": 90.0, "max_clip_duration": 15.0}
            )

    def test_windows_must_be_sorted(self):
        with pytest.raises(ValidationError, match="sorted"):
            ScorerSettings(detection={"windows": [30, 10, 3]})

    def test_windows_must_not_be_empty(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            ScorerSettings(detection={"windows": []})

    def test_pipeline_duration_bounds(self):
        with pytest.raises(ValidationError, match="min_duration_default"):
            PipelineSettings(min_duration_default=90.0, max_duration_default=15.0)

    def test_pipeline_asr_choices_validated(self):
        with pytest.raises(ValidationError, match="asr_model"):
            PipelineSettings(asr_model="huge")
        with pytest.raises(ValidationError, match="asr_device"):
            PipelineSettings(asr_device="tpu")

    def test_text_cluster_bounds(self):
        with pytest.raises(ValidationError, match="topic_kmeans_clusters_min"):
            TextSettings(
                topic_kmeans_clusters_min=6,
                topic_kmeans_clusters_max=3,
            )
