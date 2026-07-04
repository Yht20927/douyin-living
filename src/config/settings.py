# -*- coding: utf-8 -*-
"""Pydantic-based settings — loads from config/*.yaml with env overrides.

Usage::

    from src.config import load_settings
    settings = load_settings()
    print(settings.scorer.profiles["default"])
    print(settings.audio.panns_rms_threshold)

Environment variable override::

    export SCORER_DETECTION_WINDOWS='[5,15,45]'
    export AUDIO_PANNS_RMS_THRESHOLD=1.5
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ── helpers ──────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (mutates base)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


# ── sub-settings ─────────────────────────────────────────────────

class _ScorerDetection(BaseModel):
    windows: list[int] = Field(default=[3, 10, 30])
    nms_merge_distance: float = Field(default=15.0)
    min_clip_duration: float = Field(default=15.0)
    max_clip_duration: float = Field(default=90.0)
    sensitivity_default: float = Field(default=2.5)
    peak_persistence_frames: int = Field(default=3)
    candidate_gap_sec: float = Field(default=3.0)
    trigger_zscore_threshold: float = Field(default=2.0)
    window_factor: int = Field(default=2)


class _ScorerPreprocessing(BaseModel):
    ewma_alpha: float = Field(default=0.3)
    zscore_window_sec: int = Field(default=60)
    zscore_eps: float = Field(default=1e-10)
    median_filter_size: int = Field(default=5)


class _ScorerBoundary(BaseModel):
    pad_sec: int = Field(default=5)
    valley_search_range: int = Field(default=10)
    valley_window_radius: int = Field(default=5)


class _ScorerTTest(BaseModel):
    pre_window_sec: int = Field(default=5)
    pvalue_threshold: float = Field(default=0.05)


class _ScorerGlobalDecay(BaseModel):
    coeff: float = Field(default=0.1)


class _ScorerThumbnail(BaseModel):
    offset_sec: float = Field(default=3.0)


class _ScorerBreakdown(BaseModel):
    top_n: int = Field(default=5)


class _ScorerLagCompensation(BaseModel):
    min_len: int = Field(default=10)
    max_shift: int = Field(default=5)


class _ScorerBoost(BaseModel):
    tanh_divisor: float = Field(default=2.0)


class ScorerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCORER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    profiles: dict[str, dict[str, float]] = Field(default={})
    preprocessing: _ScorerPreprocessing = Field(default_factory=_ScorerPreprocessing)
    detection: _ScorerDetection = Field(default_factory=_ScorerDetection)
    boundary: _ScorerBoundary = Field(default_factory=_ScorerBoundary)
    t_test: _ScorerTTest = Field(default_factory=_ScorerTTest)
    global_decay: _ScorerGlobalDecay = Field(default_factory=_ScorerGlobalDecay)
    thumbnail: _ScorerThumbnail = Field(default_factory=_ScorerThumbnail)
    breakdown: _ScorerBreakdown = Field(default_factory=_ScorerBreakdown)
    lag_compensation: _ScorerLagCompensation = Field(
        default_factory=_ScorerLagCompensation
    )
    boost: _ScorerBoost = Field(default_factory=_ScorerBoost)


class AudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_", extra="ignore")

    sample_rate: int = Field(default=16000)
    hop_length: int = Field(default=512)
    frame_length: int = Field(default=2048)
    n_mfcc: int = Field(default=13)
    panns_rms_threshold: float = Field(default=2.0)
    panns_hop_secs: float = Field(default=0.5)
    panns_chunk_secs: float = Field(default=1.5)
    panns_min_chunk_secs: float = Field(default=0.5)
    panns_batch_size_gpu: int = Field(default=32)
    panns_batch_size_cpu: int = Field(default=8)
    min_trigger_gap_sec: int = Field(default=2)
    event_laughter_idx: int = Field(default=32)
    event_applause_idx: int = Field(default=10)
    event_music_idx: int = Field(default=137)


class TextSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TEXT_", extra="ignore")

    similarity_threshold: float = Field(default=0.7)
    utr_threshold: float = Field(default=0.3)
    keyword_config_path: str = Field(default="config/keywords.json")
    density_estimate_divisor: int = Field(default=10)
    utr_penalty_max: float = Field(default=1.0)
    userid_hash_len: int = Field(default=16)
    sentiment_model: str = Field(
        default="uer/roberta-base-finetuned-jd-binary-chinese"
    )
    sentiment_max_len: int = Field(default=512)
    fasttext_model_path: str = Field(default="models/cc.zh.100.bin")
    text2vec_model: str = Field(default="shibing624/text2vec-base-chinese")
    utr_window_radius: int = Field(default=30)
    entropy_window_radius: int = Field(default=5)
    entropy_msg_cap: int = Field(default=200)
    entropy_top_words: int = Field(default=50)
    topic_window_size: int = Field(default=30)
    topic_msg_cap: int = Field(default=50)
    topic_kmeans_clusters_min: int = Field(default=2)
    topic_kmeans_clusters_max: int = Field(default=5)
    topic_kmeans_n_init: int = Field(default=10)
    topic_kmeans_random_state: int = Field(default=42)


class VisualSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISUAL_", extra="ignore")

    scene_threshold: float = Field(default=27.0)
    fps_fallback: int = Field(default=30)
    opticalflow_resolution: tuple[int, int] = Field(default=(320, 240))
    opticalflow_sample_interval: int = Field(default=5)
    farneback_pyr_scale: float = Field(default=0.5)
    farneback_levels: int = Field(default=3)
    farneback_winsize: int = Field(default=15)
    farneback_iterations: int = Field(default=3)
    farneback_poly_n: int = Field(default=5)
    farneback_poly_sigma: float = Field(default=1.2)
    farneback_flags: int = Field(default=0)
    face_score_threshold: float = Field(default=0.6)


class RecordingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECORDER_", extra="ignore")

    rotation_secs: int = Field(default=1800)
    backoff_base: int = Field(default=2)
    backoff_max: int = Field(default=30)
    unexpected_retry_wait: int = Field(default=10)
    max_retries: int = Field(default=5)
    http_timeout_total: int | None = Field(default=None)
    http_timeout_sock_read: float = Field(default=30.0)
    chunk_size: int = Field(default=65536)
    ws_ping_interval: float = Field(default=5.0)
    ws_reconnect_delay: float = Field(default=3.0)


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", extra="ignore")

    sensitivity_default: float = Field(default=2.5)
    min_duration_default: float = Field(default=15.0)
    max_duration_default: float = Field(default=90.0)
    resolution_default: str = Field(default="720p")
    ws_join_timeout: float = Field(default=3.0)
    writer_join_timeout: float = Field(default=3.0)
    queue_timeout: float = Field(default=0.5)
    signal_pool_workers: int = Field(default=3)
    ffmpeg_timeout_clip: int = Field(default=300)
    ffmpeg_timeout_thumbnail: int = Field(default=60)
    ffmpeg_timeout_burn: int = Field(default=300)


# ── aggregate settings ───────────────────────────────────────────

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    scorer: ScorerSettings = Field(default_factory=ScorerSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    visual: VisualSettings = Field(default_factory=VisualSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


# ── loader ───────────────────────────────────────────────────────

def _load_section(name: str, cls: type) -> Any:
    """Load a single section from its YAML file, then overlay env vars."""
    yaml_path = _CONFIG_DIR / f"{name}.yaml"
    yaml_data = _load_yaml(yaml_path)
    return cls(**yaml_data)


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load all settings from config/*.yaml with environment overrides.

    The result is cached; call ``load_settings.cache_clear()`` to force reload.
    """
    return AppSettings(
        scorer=_load_section("scorer", ScorerSettings),
        audio=_load_section("audio", AudioSettings),
        text=_load_section("text", TextSettings),
        visual=_load_section("visual", VisualSettings),
        recording=_load_section("recording", RecordingSettings),
        pipeline=_load_section("pipeline", PipelineSettings),
    )
