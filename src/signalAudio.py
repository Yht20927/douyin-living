# -*- coding: utf-8 -*-
"""Audio signal extraction — librosa features + panns event detection."""

import json
import numpy as np
import librosa
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().audio

# Audio event label indices from AudioSet (panns CNN14)
EVENT_LAUGHTER_IDX = _cfg.event_laughter_idx
EVENT_APPLAUSE_IDX = _cfg.event_applause_idx
EVENT_MUSIC_IDX = _cfg.event_music_idx
EVENT_NAMES = ["laughter", "applause", "music"]

# RMS threshold for triggering panns (relative to mean)
PANNS_RMS_THRESHOLD = _cfg.panns_rms_threshold


def extractFeatures(
    audioPath: str,
    sr: int | None = None,
    outputPath: str | None = None,
) -> dict:
    """Extract audio features from an audio file.

    Args:
        audioPath: Path to AAC/MP3/WAV audio file.
        sr: Sample rate for librosa (16kHz balances quality and speed).
        outputPath: If set, save features JSON to this path.

    Returns:
        dict with 'sampleRate', 'duration', 'features' keys.
    """
    log.info(f"Extracting audio features: {audioPath}")
    try:
        return _extractFeaturesImpl(audioPath, sr, outputPath)
    except Exception:
        log.exception(f"Audio feature extraction failed for {audioPath}")
        # Return an empty feature structure so the pipeline can continue
        # with other signal sources instead of crashing entirely.
        empty = {
            "sampleRate": 1,
            "duration": 0,
            "features": {
                "rms": [], "spectralCentroid": [], "mfccDist": [],
                "zcr": [], "eventLaughter": [], "eventApplause": [], "eventMusic": [],
            },
        }
        if outputPath:
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False)
        return empty


def _extractFeaturesImpl(
    audioPath: str,
    sr: int | None = None,
    outputPath: str | None = None,
) -> dict:
    """Inner implementation — wrapped by extractFeatures with try/except."""
    sr = sr or _cfg.sample_rate
    y, sr = librosa.load(audioPath, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    nSecs = int(np.ceil(duration))

    # ── Frame-level features (librosa frame rate) ────────────────
    hop_length = _cfg.hop_length
    frameRate = sr / hop_length

    log.debug("Computing RMS + ZCR + spectral centroid + MFCC...")

    rms = librosa.feature.rms(y=y, frame_length=_cfg.frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    spectralCentroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_cfg.n_mfcc, hop_length=hop_length)

    # MFCC distance (frame-to-frame change)
    mfccDist = np.zeros(len(rms))
    mfccDist[1:] = np.sqrt(np.sum((mfcc[:, 1:] - mfcc[:, :-1]) ** 2, axis=0))

    # ── Downsample to 1Hz ──────────────────────────────────────
    def _to1Hz(feature: np.ndarray) -> list[float]:
        """Average frame-level features to 1-second resolution."""
        framesPerSec = int(frameRate)
        result = []
        for t in range(nSecs):
            start = t * framesPerSec
            end = min(start + framesPerSec, len(feature))
            if end > start:
                result.append(float(np.mean(feature[start:end])))
            else:
                result.append(0.0)
        return result

    rms1Hz = _to1Hz(rms)
    zcr1Hz = _to1Hz(zcr)
    centroid1Hz = _to1Hz(spectralCentroid)
    mfccDist1Hz = _to1Hz(mfccDist)

    # ── panns event detection (lazy-loaded, RMS-triggered) ──────
    eventLaughter = [0.0] * nSecs
    eventApplause = [0.0] * nSecs
    eventMusic = [0.0] * nSecs

    rmsMean = np.mean(rms1Hz)
    rmsStd = np.std(rms1Hz)
    triggerCount = 0

    # Find RMS peaks to trigger panns
    for t in range(nSecs):
        if rms1Hz[t] > rmsMean + PANNS_RMS_THRESHOLD * rmsStd:
            triggerCount += 1

    if triggerCount > 0:
        log.debug(f"RMS peaks: {triggerCount}/{nSecs} triggering panns...")
        try:
            from panns_inference import AudioTagging
            import torch
            from src.modelPool import ModelPool

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pool = ModelPool.instance()

            # Pull panns through ModelPool so a long pipeline (multiple
            # videos in sequence) doesn't keep stacking event-detection
            # checkpoints in VRAM.  scope() releases on exit, leaving the
            # model evictable if another module needs the slot.
            with pool.scope(
                "panns:AudioTagging",
                lambda: AudioTagging(checkpoint_path=None, device=device, sr=_cfg.sample_rate),
                gpu=(device == "cuda"),
            ) as at:
                hopSamples = int(sr * _cfg.panns_hop_secs)

                # Collect every triggered chunk into a batch first; we'd
                # rather do one (or a few) GPU forward passes than
                # `triggerCount` of them.  Each chunk is ~1.5s of audio.
                # panns wants uniform-length tensors, so right-pad short tails with zeros.
                chunkLen = int(sr * _cfg.panns_chunk_secs)
                triggered: list[int] = []
                chunks: list[np.ndarray] = []

                lastTriggered = -999  # enforce minimum gap between triggers
                for t in range(nSecs):
                    if rms1Hz[t] <= rmsMean + PANNS_RMS_THRESHOLD * rmsStd:
                        continue
                    # Skip if too close to the previous trigger — adjacent
                    # seconds share most of their chunk audio, so running
                    # panns again would be redundant.
                    if t - lastTriggered < _cfg.min_trigger_gap_sec:
                        continue
                    lastTriggered = t
                    startSample = max(0, (t * int(sr)) - hopSamples)
                    endSample = min(len(y), (t * int(sr)) + int(sr))
                    chunk = y[startSample:endSample]
                    if len(chunk) < int(sr * _cfg.panns_min_chunk_secs):
                        continue
                    if len(chunk) < chunkLen:
                        chunk = np.pad(chunk, (0, chunkLen - len(chunk)), mode="constant")
                    else:
                        chunk = chunk[:chunkLen]
                    triggered.append(t)
                    chunks.append(chunk.astype(np.float32))

                if chunks:
                    # Batch in groups so GPU memory stays bounded.
                    batchSize = _cfg.panns_batch_size_gpu if device == "cuda" else _cfg.panns_batch_size_cpu
                    for i in range(0, len(chunks), batchSize):
                        batch = np.stack(chunks[i:i + batchSize])
                        clipwise, _ = at.inference(batch)
                        for j, t in enumerate(triggered[i:i + batchSize]):
                            eventLaughter[t] = float(clipwise[j, EVENT_LAUGHTER_IDX])
                            eventApplause[t] = float(clipwise[j, EVENT_APPLAUSE_IDX])
                            eventMusic[t] = float(clipwise[j, EVENT_MUSIC_IDX])
                    log.debug(
                        f"panns batch inference: {len(chunks)} chunks "
                        f"in {(len(chunks) + batchSize - 1) // batchSize} batches"
                    )
        except ImportError:
            log.warning("panns-inference not installed — skipping event detection")
        except Exception:
            log.warning("panns inference failed", exc_info=True)
    else:
        log.debug("No significant RMS peaks — skipping panns (saves ~2s)")

    # ── Build output ───────────────────────────────────────────
    features = {
        "sampleRate": 1,
        "duration": duration,
        "features": {
            "rms": rms1Hz,
            "spectralCentroid": centroid1Hz,
            "mfccDist": mfccDist1Hz,
            "zcr": zcr1Hz,
            "eventLaughter": eventLaughter,
            "eventApplause": eventApplause,
            "eventMusic": eventMusic,
        },
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False)
        log.info(f"Audio features saved: {outputPath}")

    log.info(f"Audio features: {nSecs}s, {len(list(features['features'].keys()))} features")
    return features
