# -*- coding: utf-8 -*-
"""ASR runtime resolution — auto model/device selection (S5-5).

Kept separate from the top-level ``asr.py`` entry point so callers can fake
``asr.transcribe`` without also providing configuration helpers.
"""

from src.config import load_settings
from src.log.logger import getLogger

log = getLogger(__name__)

# Compute types that only make sense on CUDA.  CPU runs are downgraded to int8
# so a stale GPU config does not fail after the CUDA→CPU fallback.
_CUDA_ONLY_COMPUTE = frozenset({"float16", "int8_float16", "bfloat16"})


def _detectCuda() -> tuple[bool, float]:
    """Return (cudaAvailable, vramGb) without hard-depending on torch."""
    try:
        import torch
    except ImportError:
        return False, 0.0
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        return False, 0.0
    if not available:
        return False, 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        vramGb = float(props.total_memory) / (1024 ** 3)
    except Exception:
        vramGb = 0.0
    return True, vramGb


def resolveAsrRuntime(cfg=None, cudaAvailable: bool | None = None) -> dict:
    """Resolve ``auto`` model/device/compute settings into concrete values.

    Args:
        cfg: ``AsrSettings`` instance; loads ``config/asr.yaml`` when omitted.
        cudaAvailable: Test/override hook; detected via torch when omitted.

    Returns:
        Dict with model, device, computeType, language and selection metadata.
    """
    cfg = cfg or load_settings().asr
    if cudaAvailable is None:
        cudaAvailable, vramGb = _detectCuda()
    else:
        vramGb = 0.0

    device = str(cfg.device)
    selection = []
    if device == "auto":
        device = "cuda" if cudaAvailable else "cpu"
        selection.append(f"device=auto→{device}")
    elif device == "cuda" and not cudaAvailable:
        log.warning("ASR device=cuda requested but CUDA is unavailable — using CPU")
        device = "cpu"
        selection.append("device=cuda unavailable→cpu")

    model = str(cfg.model)
    if model == "auto":
        model = cfg.auto_model_cuda if device == "cuda" else cfg.auto_model_cpu
        selection.append(f"model=auto→{model}")
    if vramGb:
        selection.append(f"vram={vramGb:.1f}GB")

    computeType = str(cfg.compute_type)
    if device == "cpu" and computeType in _CUDA_ONLY_COMPUTE:
        log.warning(f"ASR compute_type={computeType} is CUDA-only — using int8")
        computeType = "int8"
        selection.append(f"compute={cfg.compute_type}→int8")

    language = str(cfg.language)
    if language.lower() == "auto":
        language = None

    return {
        "model": model,
        "device": device,
        "computeType": computeType,
        "language": language,
        "diarize": bool(cfg.diarize),
        "batchSize": int(cfg.batch_size),
        "vadFilter": bool(cfg.vad_filter),
        "vadMinSilenceMs": int(cfg.vad_min_silence_ms),
        "vadSpeechPadMs": int(cfg.vad_speech_pad_ms),
        "beamSize": int(cfg.beam_size),
        "chunkSec": float(cfg.chunk_sec),
        "resume": bool(cfg.resume),
        "selection": "; ".join(selection) or "configured",
    }
