# -*- coding: utf-8 -*-
"""Pydantic-based settings — loads from config/*.yaml with env overrides.

Priority (highest first):
    1. explicit constructor arguments (tests / programmatic callers)
    2. environment variables
    3. ``.env`` file
    4. ``config/<section>.yaml``
    5. field defaults

Usage::

    from src.config import load_settings
    settings = load_settings()
    print(settings.scorer.profiles["default"])
    print(settings.audio.panns_rms_threshold)

Environment variable override::

    export SCORER_DETECTION__WINDOWS='[5,15,45]'
    export AUDIO_PANNS_RMS_THRESHOLD=1.5
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ── helpers ──────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Settings source that reads a section from ``config/<name>.yaml``."""

    def __init__(self, settings_cls: type[BaseSettings], yamlPath: Path):
        super().__init__(settings_cls)
        self._yamlPath = yamlPath

    def get_field_value(self, field, field_name):  # pragma: no cover - unused path
        data = _load_yaml(self._yamlPath)
        return data.get(field_name), field_name, self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        return _load_yaml(self._yamlPath)


class _YamlSectionSettings(BaseSettings):
    """Base class: YAML defaults are lower priority than environment vars."""

    model_config = SettingsConfigDict(extra="ignore")

    _yaml_name: ClassVar[str] = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = _CONFIG_DIR / f"{settings_cls._yaml_name}.yaml"
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSettingsSource(settings_cls, yaml_path),
            file_secret_settings,
        )


# ── sub-settings ─────────────────────────────────────────────────

class _ScorerDetection(BaseModel):
    windows: list[int] = Field(default=[3, 10, 30])
    nms_merge_distance: float = Field(default=15.0, ge=0.0)
    min_clip_duration: float = Field(default=15.0, gt=0.0)
    max_clip_duration: float = Field(default=90.0, gt=0.0)
    sensitivity_default: float = Field(default=2.5, gt=0.0)
    peak_persistence_frames: int = Field(default=3, ge=1)
    candidate_gap_sec: float = Field(default=3.0, ge=0.0)
    trigger_zscore_threshold: float = Field(default=2.0)
    window_factor: int = Field(default=2, ge=1)

    @field_validator("windows")
    @classmethod
    def _validateWindows(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("detection.windows must not be empty")
        if any(w <= 0 for w in value):
            raise ValueError("detection.windows must contain positive integers")
        if len(set(value)) != len(value):
            raise ValueError("detection.windows must not contain duplicates")
        if value != sorted(value):
            raise ValueError("detection.windows must be sorted ascending")
        return value

    @model_validator(mode="after")
    def _validateDurations(self):
        if self.min_clip_duration >= self.max_clip_duration:
            raise ValueError(
                "detection.min_clip_duration must be smaller than max_clip_duration"
            )
        return self


class _ScorerPreprocessing(BaseModel):
    ewma_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    zscore_window_sec: int = Field(default=60, ge=2)
    zscore_eps: float = Field(default=1e-10, gt=0.0)
    median_filter_size: int = Field(default=5, ge=1)


class _ScorerBoundary(BaseModel):
    pad_sec: int = Field(default=5, ge=0)
    valley_search_range: int = Field(default=10, ge=0)
    valley_window_radius: int = Field(default=5, ge=1)


class _ScorerTTest(BaseModel):
    pre_window_sec: int = Field(default=5, ge=1)
    pvalue_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class _ScorerGlobalDecay(BaseModel):
    coeff: float = Field(default=0.1, ge=0.0)


class _ScorerThumbnail(BaseModel):
    offset_sec: float = Field(default=3.0, ge=0.0)


class _ScorerBreakdown(BaseModel):
    top_n: int = Field(default=5, ge=1)


class _ScorerLagCompensation(BaseModel):
    min_len: int = Field(default=10, ge=1)
    max_shift: int = Field(default=5, ge=0)


class _ScorerBoost(BaseModel):
    tanh_divisor: float = Field(default=2.0, gt=0.0)


class ScorerSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCORER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    _yaml_name: ClassVar[str] = "scorer"

    profiles: dict[str, dict[str, float]] = Field(default={})
    normalize_profiles: bool = Field(default=True)
    weight_sum_tolerance: float = Field(default=0.01, ge=0.0)
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

    @model_validator(mode="after")
    def _validateProfiles(self):
        if not self.profiles:
            return self

        # Imported lazily to avoid a settings → logger → settings cycle.
        from src.util import FEATURE_NAMES

        normalized: dict[str, dict[str, float]] = {}
        for profileName, weights in self.profiles.items():
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"profile '{profileName}' must be a non-empty mapping")

            unknown = set(weights) - FEATURE_NAMES
            if unknown:
                raise ValueError(
                    f"profile '{profileName}' contains unknown feature key(s): "
                    f"{sorted(unknown)}"
                )
            negative = [k for k, v in weights.items() if v < 0]
            if negative:
                raise ValueError(
                    f"profile '{profileName}' contains negative weight(s): {negative}"
                )

            total = sum(weights.values())
            if total <= 0:
                raise ValueError(f"profile '{profileName}' weights must sum to > 0")
            if self.normalize_profiles:
                weights = {k: v / total for k, v in weights.items()}
                total = 1.0
            if abs(total - 1.0) > self.weight_sum_tolerance:
                raise ValueError(
                    f"profile '{profileName}' weights sum to {total:.3f}; "
                    f"expected 1.0 ± {self.weight_sum_tolerance} "
                    "(set SCORER_NORMALIZE_PROFILES=true to auto-normalize)"
                )
            normalized[profileName] = weights

        self.profiles = normalized
        return self


class AudioSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_", extra="ignore")

    _yaml_name: ClassVar[str] = "audio"

    sample_rate: int = Field(default=16000, gt=0)
    hop_length: int = Field(default=512, gt=0)
    frame_length: int = Field(default=2048, gt=0)
    n_mfcc: int = Field(default=13, ge=1)
    panns_rms_threshold: float = Field(default=2.0, ge=0.0)
    panns_hop_secs: float = Field(default=0.5, gt=0.0)
    panns_chunk_secs: float = Field(default=1.5, gt=0.0)
    panns_min_chunk_secs: float = Field(default=0.5, gt=0.0)
    panns_batch_size_gpu: int = Field(default=32, ge=1)
    panns_batch_size_cpu: int = Field(default=8, ge=1)
    min_trigger_gap_sec: int = Field(default=2, ge=0)
    event_laughter_idx: int = Field(default=32, ge=0)
    event_applause_idx: int = Field(default=10, ge=0)
    event_music_idx: int = Field(default=137, ge=0)

    @model_validator(mode="after")
    def _validateChunks(self):
        if self.panns_min_chunk_secs > self.panns_chunk_secs:
            raise ValueError("panns_min_chunk_secs must not exceed panns_chunk_secs")
        return self


class TextSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(env_prefix="TEXT_", extra="ignore")

    _yaml_name: ClassVar[str] = "text"

    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    utr_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    keyword_config_path: str = Field(default="config/keywords.json")
    density_estimate_divisor: int = Field(default=10, ge=1)
    utr_penalty_max: float = Field(default=1.0, ge=0.0)
    userid_hash_len: int = Field(default=16, ge=1)
    sentiment_model: str = Field(
        default="uer/roberta-base-finetuned-jd-binary-chinese"
    )
    sentiment_max_len: int = Field(default=512, gt=0)
    fasttext_model_path: str = Field(default="models/cc.zh.300.bin")
    text2vec_model: str = Field(default="shibing624/text2vec-base-chinese")
    utr_window_radius: int = Field(default=30, ge=1)
    utr_auto_calibrate: bool = True
    utr_calibration_percentile: float = Field(default=20.0, ge=1.0, le=50.0)
    utr_calibration_min_samples: int = Field(default=60, ge=1)
    utr_calibration_min: float = Field(default=0.15, gt=0.0, le=1.0)
    utr_calibration_max: float = Field(default=0.6, gt=0.0, le=1.0)
    dm_rate_window_radius: int = Field(default=5, ge=0)
    sentiment_mode: str = Field(
        default="rules", pattern=r"^(rules|hybrid|hf)$"
    )
    entropy_window_radius: int = Field(default=5, ge=0)
    entropy_msg_cap: int = Field(default=200, ge=1)
    entropy_top_words: int = Field(default=50, ge=1)
    topic_window_size: int = Field(default=30, ge=1)
    topic_msg_cap: int = Field(default=50, ge=1)
    topic_kmeans_clusters_min: int = Field(default=2, ge=1)
    topic_kmeans_clusters_max: int = Field(default=5, ge=1)
    topic_kmeans_n_init: int = Field(default=10, ge=1)
    topic_kmeans_random_state: int = Field(default=42)

    @model_validator(mode="after")
    def _validateClusters(self):
        if self.topic_kmeans_clusters_min > self.topic_kmeans_clusters_max:
            raise ValueError(
                "topic_kmeans_clusters_min must not exceed topic_kmeans_clusters_max"
            )
        if self.utr_calibration_min > self.utr_calibration_max:
            raise ValueError(
                "utr_calibration_min must not exceed utr_calibration_max"
            )
        return self


class VisualSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(env_prefix="VISUAL_", extra="ignore")

    _yaml_name: ClassVar[str] = "visual"

    scene_threshold: float = Field(default=27.0, gt=0.0)
    scene_min_gap_sec: float = Field(default=1.0, ge=0.0)
    fps_fallback: int = Field(default=30, gt=0)
    proxy_enabled: bool = True
    proxy_height: int = Field(default=480, ge=120)
    proxy_timeout_sec: int = Field(default=1800, gt=0)
    sample_fps: float = Field(default=2.0, gt=0.0, le=30.0)
    enable_face: bool = False
    opticalflow_resolution: tuple[int, int] = Field(default=(320, 240))
    opticalflow_sample_interval: int = Field(default=5, ge=1)
    farneback_pyr_scale: float = Field(default=0.5, gt=0.0, lt=1.0)
    farneback_levels: int = Field(default=3, ge=1)
    farneback_winsize: int = Field(default=15, ge=1)
    farneback_iterations: int = Field(default=3, ge=1)
    farneback_poly_n: int = Field(default=5, ge=1)
    farneback_poly_sigma: float = Field(default=1.2, gt=0.0)
    farneback_flags: int = Field(default=0)
    face_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("opticalflow_resolution")
    @classmethod
    def _validateResolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError("opticalflow_resolution must be a positive (width, height)")
        return value


class AsrSettings(_YamlSectionSettings):
    """ASR runtime knobs (config/asr.yaml).

    ``model=auto`` resolves to ``auto_model_cpu`` on CPU and
    ``auto_model_cuda`` on CUDA.  Legacy ``PIPELINE_ASR_*`` environment
    variables remain accepted as aliases for the fields that used to live in
    ``PipelineSettings``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        extra="ignore",
        populate_by_name=True,
    )

    _yaml_name: ClassVar[str] = "asr"

    model: str = Field(
        default="auto",
        validation_alias=AliasChoices("ASR_MODEL", "PIPELINE_ASR_MODEL"),
        pattern=r"^(auto|tiny|base|small|medium|large-v3)$",
    )
    auto_model_cpu: str = Field(
        default="base", pattern=r"^(tiny|base|small|medium|large-v3)$"
    )
    auto_model_cuda: str = Field(
        default="small", pattern=r"^(tiny|base|small|medium|large-v3)$"
    )
    device: str = Field(
        default="auto",
        validation_alias=AliasChoices("ASR_DEVICE", "PIPELINE_ASR_DEVICE"),
        pattern=r"^(auto|cuda|cpu)$",
    )
    compute_type: str = Field(
        default="int8",
        validation_alias=AliasChoices(
            "ASR_COMPUTE_TYPE", "PIPELINE_ASR_COMPUTE_TYPE"
        ),
        pattern=(
            r"^(default|auto|int8|int8_float32|int8_float16|int8_bfloat16"
            r"|int16|float16|bfloat16|float32)$"
        ),
    )
    language: str = Field(
        default="zh",
        pattern=r"^(auto|[a-zA-Z]{2,3}(-[a-zA-Z]{2,4})?)$",
    )
    diarize: bool = Field(
        default=True,
        validation_alias=AliasChoices("ASR_DIARIZE", "PIPELINE_ASR_DIARIZE"),
    )
    batch_size: int = Field(default=32, ge=1)
    vad_filter: bool = True
    vad_min_silence_ms: int = Field(default=500, ge=0)
    vad_speech_pad_ms: int = Field(default=200, ge=0)
    beam_size: int = Field(default=5, ge=1, le=10)
    # Long recordings are split into chunks; 0 disables chunking.
    chunk_sec: float = Field(default=1800.0, ge=0.0)
    resume: bool = True


class RecordingSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(env_prefix="RECORDER_", extra="ignore")

    _yaml_name: ClassVar[str] = "recording"

    rotation_secs: int = Field(default=1800, gt=0)
    backoff_base: int = Field(default=2, ge=1)
    backoff_max: int = Field(default=30, ge=1)
    unexpected_retry_wait: int = Field(default=10, ge=0)
    max_retries: int = Field(default=5, ge=0)
    http_timeout_total: int | None = Field(default=None, gt=0)
    http_timeout_sock_read: float = Field(default=30.0, gt=0.0)
    chunk_size: int = Field(default=65536, gt=0)
    ws_ping_interval: float = Field(default=5.0, gt=0.0)
    ws_ping_timeout: float = Field(default=10.0, gt=0.0)
    ws_reconnect_delay: float = Field(default=3.0, ge=0.0)
    idle_timeout_secs: float = Field(default=60.0, gt=0.0)
    min_free_disk_gb: float = Field(default=2.0, ge=0.0)
    max_segment_bytes: int = Field(default=4 * 1024 ** 3, gt=0)
    stale_part_max_age_secs: int = Field(default=86400, ge=0)

    @model_validator(mode="after")
    def _validateBackoff(self):
        if self.backoff_max < self.backoff_base:
            raise ValueError("backoff_max must be >= backoff_base")
        return self


class PipelineSettings(_YamlSectionSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", extra="ignore")

    _yaml_name: ClassVar[str] = "pipeline"

    sensitivity_default: float = Field(default=2.5, gt=0.0)
    min_duration_default: float = Field(default=15.0, gt=0.0)
    max_duration_default: float = Field(default=90.0, gt=0.0)
    resolution_default: str = Field(default="720p")
    ws_join_timeout: float = Field(default=3.0, ge=0.0)
    writer_join_timeout: float = Field(default=3.0, ge=0.0)
    queue_timeout: float = Field(default=0.5, gt=0.0)
    signal_pool_workers: int = Field(default=3, ge=1)
    ffmpeg_timeout_clip: int = Field(default=300, gt=0)
    ffmpeg_timeout_thumbnail: int = Field(default=60, gt=0)
    ffmpeg_timeout_burn: int = Field(default=300, gt=0)

    # ASR knobs are also the supported OOM escape hatch: lower the model,
    # force CPU, or disable diarization without editing controller code.
    asr_model: str = Field(
        default="base", pattern=r"^(tiny|base|small|medium|large-v3)$"
    )
    asr_device: str = Field(default="auto", pattern=r"^(auto|cuda|cpu)$")
    asr_compute_type: str = Field(
        default="int8",
        pattern=(
            r"^(default|auto|int8|int8_float32|int8_float16|int8_bfloat16"
            r"|int16|float16|bfloat16|float32)$"
        ),
    )
    asr_diarize: bool = True

    # Two-stage recall/rerank (Sprint 5).  Disabled by default so the legacy
    # scorer remains the production path until the new pipeline is validated.
    two_stage_enabled: bool = False
    two_stage_recall_per_hour: float = Field(default=60.0, gt=0.0)
    two_stage_top_k: int = Field(default=20, ge=1)
    two_stage_min_gap_sec: float = Field(default=12.0, gt=0.0)
    two_stage_recall_threshold: float = Field(default=0.8, ge=0.0)
    two_stage_window_sec: float = Field(default=30.0, gt=0.0)
    two_stage_min_score: float = Field(default=0.0, ge=0.0)
    two_stage_dynamic_duration: bool = True
    two_stage_boundary_search_sec: float = Field(default=8.0, ge=0.0)
    two_stage_platform: str = Field(
        default="none", pattern=r"^(none|douyin|bilibili|xiaohongshu)$"
    )

    @model_validator(mode="after")
    def _validateDurations(self):
        if self.min_duration_default >= self.max_duration_default:
            raise ValueError(
                "min_duration_default must be smaller than max_duration_default"
            )
        return self


# ── aggregate settings ───────────────────────────────────────────

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    scorer: ScorerSettings = Field(default_factory=ScorerSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    visual: VisualSettings = Field(default_factory=VisualSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


# ── loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load all settings from config/*.yaml with environment overrides.

    The result is cached; call ``load_settings.cache_clear()`` to force a
    reload.  Note that modules which snapshot ``_cfg`` at import time still
    need a process restart to pick up changes (see reload_settings()).
    """
    return AppSettings()
