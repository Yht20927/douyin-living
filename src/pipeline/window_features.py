# -*- coding: utf-8 -*-
"""Window-level feature aggregation for two-stage recall and reranking.

The legacy scorer works on per-second point scores.  The two-stage pipeline
instead needs a compact, explainable description of a *window*: how strong
each signal is inside it, how it differs from its neighbours, how many
modalities agree, and whether the duration fits the requested range.

All functions in this module are pure and deterministic.  Input matrices are
expected on a shared 1 Hz timeline (see :func:`src.scorer.alignFeatureArrays`).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Event-like signals are sparse by nature.  Without clipping, a single event
# second can dominate a window aggregate after robust normalisation.
DEFAULT_Z_CLIP = 5.0


def normalizeFeatureArrays(
    features: dict[str, np.ndarray],
    clip: float = DEFAULT_Z_CLIP,
) -> dict[str, np.ndarray]:
    """Robustly normalise every signal to a clipped Z-score scale.

    Median/MAD is used instead of mean/std so a long quiet stream with a few
    bursts does not compress the bursts into a tiny range.  Constant signals
    become all-zero arrays.
    """
    normalized: dict[str, np.ndarray] = {}
    for name, values in features.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            normalized[name] = arr
            continue

        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = float(np.std(arr))
        if not math.isfinite(scale) or scale <= 1e-9:
            normalized[name] = np.zeros_like(arr)
            continue

        z = (arr - median) / scale
        normalized[name] = np.clip(z, -clip, clip)
    return normalized


def resolveProfileWeights(
    profile: str,
    signalNames: Iterable[str],
) -> dict[str, float]:
    """Return positive weights for the available signals.

    Profiles come from the legacy scorer configuration.  Unknown profiles fall
    back to ``default``; when a profile has no overlap with the available
    signals, equal weights are used so recall still works on new feature sets.
    """
    names = list(signalNames)
    try:
        # Imported lazily: settings → logger → config must stay cycle-free.
        from src.scorer import PROFILES

        configured = PROFILES.get(profile) or PROFILES.get("default") or {}
    except Exception:  # pragma: no cover - defensive import guard
        configured = {}

    weights = {name: float(configured.get(name, 0.0)) for name in names}
    total = sum(weights.values())
    if total <= 0.0:
        if not names:
            return {}
        equal = 1.0 / len(names)
        return {name: equal for name in names}

    return {name: value / total for name, value in weights.items() if value > 0.0}


def durationFit(duration: float, minDuration: float, maxDuration: float) -> float:
    """Score how well ``duration`` fits the requested range (0..1)."""
    if duration <= 0:
        return 0.0
    if minDuration <= duration <= maxDuration:
        return 1.0
    if duration < minDuration:
        return max(0.0, duration / minDuration) if minDuration > 0 else 0.0
    return max(0.0, maxDuration / duration) if maxDuration > 0 else 0.0


def _sliceMean(signals: dict[str, np.ndarray], start: int, end: int) -> dict[str, float]:
    means: dict[str, float] = {}
    for name, arr in signals.items():
        if end > start and start < len(arr):
            means[name] = float(np.mean(arr[start:min(end, len(arr))]))
        else:
            means[name] = 0.0
    return means


def computeWindowFeatures(
    signals: dict[str, np.ndarray],
    start: float,
    end: float,
    weights: dict[str, float] | None = None,
    minDuration: float = 0.0,
    maxDuration: float = 1e9,
    agreementZ: float = 1.0,
) -> dict:
    """Describe one candidate window using the normalised signal matrix.

    Args:
        signals: Normalised 1 Hz signals (see :func:`normalizeFeatureArrays`).
        start, end: Window boundaries in seconds.
        weights: Signal weights; ``None`` means equal weights.
        minDuration, maxDuration: Duration range used by ``durationFit``.
        agreementZ: A signal counts as active when its window peak reaches
            this Z-score.

    Returns:
        A JSON-serialisable dict with per-signal statistics and window-level
        aggregates (``peakScore``, ``meanScore``, ``prePostDelta``,
        ``multiSignalAgreement``, ``novelty``, ``durationFit``).
    """
    nSecs = max((len(arr) for arr in signals.values()), default=0)
    s = max(0, min(int(round(start)), nSecs))
    e = max(0, min(int(round(end)), nSecs))
    if e < s:
        s, e = e, s
    windowLen = max(0, e - s)

    if weights is None:
        usable = [name for name, arr in signals.items() if len(arr) > 0]
        if usable:
            equal = 1.0 / len(usable)
            weights = {name: equal for name in usable}
        else:
            weights = {}
    else:
        weights = {
            name: float(weight)
            for name, weight in weights.items()
            if weight > 0 and name in signals
        }

    weightSum = sum(weights.values())
    if weightSum <= 0.0:
        weights = {}
        weightSum = 0.0

    perSignal: dict[str, dict] = {}
    peakScore = 0.0
    meanScore = 0.0
    areaScore = 0.0
    prePostDelta = 0.0
    agreementWeight = 0.0

    for name, arr in signals.items():
        weight = weights.get(name, 0.0)
        weightNorm = weight / weightSum if weightSum > 0 else 0.0

        if windowLen <= 0 or s >= len(arr):
            peak = mean = area = 0.0
            preDelta = postDelta = 0.0
        else:
            segment = arr[s:min(e, len(arr))]
            if segment.size == 0:
                peak = mean = area = 0.0
            else:
                peak = float(np.max(segment))
                mean = float(np.mean(segment))
                area = float(np.sum(np.maximum(segment, 0.0)))

            # Compare with equally sized neighbours so the metric is not
            # biased by window length.
            preStart = max(0, s - windowLen)
            preSegment = arr[preStart:s]
            preMean = float(np.mean(preSegment)) if preSegment.size else 0.0
            postEnd = min(len(arr), e + windowLen)
            postSegment = arr[e:postEnd]
            postMean = float(np.mean(postSegment)) if postSegment.size else 0.0
            preDelta = mean - preMean
            postDelta = mean - postMean

        active = bool(
            windowLen > 0
            and weightNorm > 0
            and (
                peak >= agreementZ
                or (mean >= agreementZ * 0.5 and peak > 0.1)
            )
        )
        if active and weightNorm > 0:
            agreementWeight += weightNorm

        contribution = weightNorm * mean
        perSignal[name] = {
            "weight": round(weightNorm, 6),
            "peak": round(peak, 6),
            "mean": round(mean, 6),
            "area": round(area, 6),
            "preDelta": round(preDelta, 6),
            "postDelta": round(postDelta, 6),
            "active": active,
            "contribution": round(contribution, 6),
        }

        peakScore += weightNorm * peak
        meanScore += weightNorm * mean
        areaScore += weightNorm * (area / windowLen if windowLen > 0 else 0.0)
        prePostDelta += weightNorm * ((preDelta + postDelta) / 2.0)

    # Novelty: how different is this window from the immediately preceding
    # window of the same length?  A quiet setup followed by an explosion
    # scores higher than a uniformly busy stretch.
    novelty = 0.0
    if windowLen > 0 and s > 0 and weights:
        prevStart = max(0, s - windowLen)
        currentMeans = _sliceMean(signals, s, e)
        previousMeans = _sliceMean(signals, prevStart, s)
        diffs = [
            (currentMeans[name] - previousMeans[name]) ** 2
            for name in weights
        ]
        if diffs:
            novelty = float(math.sqrt(sum(diffs) / len(diffs)))

    duration = float(e - s)
    activeSignals = sorted(
        [name for name, stats in perSignal.items() if stats["active"]],
        key=lambda name: (-perSignal[name]["contribution"], name),
    )

    return {
        "start": round(float(s), 2),
        "end": round(float(e), 2),
        "duration": round(duration, 2),
        "peakScore": round(float(peakScore), 6),
        "meanScore": round(float(meanScore), 6),
        "areaScore": round(float(areaScore), 6),
        "prePostDelta": round(float(prePostDelta), 6),
        "multiSignalAgreement": round(float(agreementWeight), 6),
        "novelty": round(float(novelty), 6),
        "durationFit": round(durationFit(duration, minDuration, maxDuration), 6),
        "targetDuration": [float(minDuration), float(maxDuration)],
        "activeSignals": activeSignals,
        "signals": perSignal,
    }
