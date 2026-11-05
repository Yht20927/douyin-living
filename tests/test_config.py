# -*- coding: utf-8 -*-
"""Tests for config module — settings loading, defaults, env overrides."""

import pytest
from pydantic import ValidationError

from src.config import (
    load_settings,
    reload_settings,
    ScorerSettings,
    AudioSettings,
    AsrSettings,
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
        assert s.asr is not None
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
        assert s.utr_auto_calibrate is True
        assert s.utr_calibration_percentile == 20.0
        assert s.utr_calibration_min_samples == 60
        assert s.utr_calibration_min == 0.15
        assert s.utr_calibration_max == 0.6
        assert s.dm_rate_window_radius == 5
        assert s.sentiment_mode == "rules"


class TestVisualSettingsDefaults:
    def test_defaults(self):
        s = VisualSettings()
        assert s.scene_threshold == 27.0
        assert s.scene_min_gap_sec == 1.0
        assert s.fps_fallback == 30
        assert s.proxy_enabled is True
        assert s.proxy_height == 480
        assert s.proxy_timeout_sec == 1800
        assert s.sample_fps == 2.0
        assert s.enable_face is False
        assert s.opticalflow_resolution == (320, 240)
        assert s.opticalflow_sample_interval == 5


class TestAsrSettingsDefaults:
    def test_defaults(self):
        s = AsrSettings()
        assert s.model == "auto"
        assert s.auto_model_cpu == "base"
        assert s.auto_model_cuda == "small"
        assert s.device == "auto"
        assert s.compute_type == "int8"
        assert s.language == "zh"
        assert s.diarize is True
        assert s.batch_size == 32
        assert s.vad_filter is True
        assert s.vad_min_silence_ms == 500
        assert s.vad_speech_pad_ms == 200
        assert s.beam_size == 5
        assert s.chunk_sec == 1800.0
        assert s.resume is True

    def test_new_env_overrides(self, monkeypatch):
        monkeypatch.setenv("ASR_MODEL", "tiny")
        monkeypatch.setenv("ASR_DEVICE", "cpu")
        monkeypatch.setenv("ASR_COMPUTE_TYPE", "float32")
        monkeypatch.setenv("ASR_LANGUAGE", "auto")
        monkeypatch.setenv("ASR_DIARIZE", "false")
        monkeypatch.setenv("ASR_BATCH_SIZE", "8")
        monkeypatch.setenv("ASR_VAD_FILTER", "false")
        monkeypatch.setenv("ASR_VAD_MIN_SILENCE_MS", "700")
        monkeypatch.setenv("ASR_VAD_SPEECH_PAD_MS", "100")
        monkeypatch.setenv("ASR_BEAM_SIZE", "3")
        monkeypatch.setenv("ASR_CHUNK_SEC", "600")
        monkeypatch.setenv("ASR_RESUME", "false")
        s = AsrSettings()
        assert s.model == "tiny"
        assert s.device == "cpu"
        assert s.compute_type == "float32"
        assert s.language == "auto"
        assert s.diarize is False
        assert s.batch_size == 8
        assert s.vad_filter is False
        assert s.vad_min_silence_ms == 700
        assert s.vad_speech_pad_ms == 100
        assert s.beam_size == 3
        assert s.chunk_sec == 600.0
        assert s.resume is False

    def test_legacy_pipeline_env_aliases(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_ASR_MODEL", "medium")
        monkeypatch.setenv("PIPELINE_ASR_DEVICE", "cpu")
        monkeypatch.setenv("PIPELINE_ASR_COMPUTE_TYPE", "float32")
        monkeypatch.setenv("PIPELINE_ASR_DIARIZE", "false")
        s = AsrSettings()
        assert s.model == "medium"
        assert s.device == "cpu"
        assert s.compute_type == "float32"
        assert s.diarize is False

    def test_invalid_model_rejected(self):
        with pytest.raises(ValidationError, match="model"):
            AsrSettings(model="huge")


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
        assert s.two_stage_enabled is False
        assert s.two_stage_recall_per_hour == 60.0
        assert s.two_stage_top_k == 20
        assert s.two_stage_min_gap_sec == 12.0
        assert s.two_stage_recall_threshold == 0.8
        assert s.two_stage_window_sec == 30.0
        assert s.two_stage_min_score == 0.0
        assert s.two_stage_dynamic_duration is True
        assert s.two_stage_boundary_search_sec == 8.0
        assert s.two_stage_platform == "none"


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

    def test_text_engagement_env_overrides(self, monkeypatch):
        monkeypatch.setenv("TEXT_UTR_AUTO_CALIBRATE", "false")
        monkeypatch.setenv("TEXT_UTR_CALIBRATION_PERCENTILE", "10")
        monkeypatch.setenv("TEXT_UTR_CALIBRATION_MIN_SAMPLES", "30")
        monkeypatch.setenv("TEXT_UTR_CALIBRATION_MIN", "0.2")
        monkeypatch.setenv("TEXT_UTR_CALIBRATION_MAX", "0.5")
        monkeypatch.setenv("TEXT_DM_RATE_WINDOW_RADIUS", "3")
        monkeypatch.setenv("TEXT_SENTIMENT_MODE", "hybrid")
        settings = TextSettings()
        assert settings.utr_auto_calibrate is False
        assert settings.utr_calibration_percentile == 10.0
        assert settings.utr_calibration_min_samples == 30
        assert settings.utr_calibration_min == 0.2
        assert settings.utr_calibration_max == 0.5
        assert settings.dm_rate_window_radius == 3
        assert settings.sentiment_mode == "hybrid"

    def test_visual_proxy_env_overrides(self, monkeypatch):
        monkeypatch.setenv("VISUAL_PROXY_ENABLED", "false")
        monkeypatch.setenv("VISUAL_PROXY_HEIGHT", "360")
        monkeypatch.setenv("VISUAL_PROXY_TIMEOUT_SEC", "600")
        monkeypatch.setenv("VISUAL_SAMPLE_FPS", "4")
        monkeypatch.setenv("VISUAL_ENABLE_FACE", "true")
        monkeypatch.setenv("VISUAL_SCENE_MIN_GAP_SEC", "2.5")
        settings = VisualSettings()
        assert settings.proxy_enabled is False
        assert settings.proxy_height == 360
        assert settings.proxy_timeout_sec == 600
        assert settings.sample_fps == 4.0
        assert settings.enable_face is True
        assert settings.scene_min_gap_sec == 2.5

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

    def test_pipeline_two_stage_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_TWO_STAGE_ENABLED", "true")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_RECALL_PER_HOUR", "45")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_TOP_K", "12")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_MIN_GAP_SEC", "8.5")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_RECALL_THRESHOLD", "1.1")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_WINDOW_SEC", "25")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_MIN_SCORE", "0.15")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_DYNAMIC_DURATION", "false")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_BOUNDARY_SEARCH_SEC", "5.5")
        monkeypatch.setenv("PIPELINE_TWO_STAGE_PLATFORM", "douyin")
        settings = PipelineSettings()
        assert settings.two_stage_enabled is True
        assert settings.two_stage_recall_per_hour == 45.0
        assert settings.two_stage_top_k == 12
        assert settings.two_stage_min_gap_sec == 8.5
        assert settings.two_stage_recall_threshold == 1.1
        assert settings.two_stage_window_sec == 25.0
        assert settings.two_stage_min_score == 0.15
        assert settings.two_stage_dynamic_duration is False
        assert settings.two_stage_boundary_search_sec == 5.5
        assert settings.two_stage_platform == "douyin"

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

    def test_text_utr_calibration_bounds(self):
        with pytest.raises(ValidationError, match="utr_calibration_min"):
            TextSettings(utr_calibration_min=0.8, utr_calibration_max=0.4)
