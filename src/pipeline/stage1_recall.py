# -*- coding: utf-8 -*-
"""Stage 1 — low-cost, high-recall candidate window generation.

This stage deliberately avoids the expensive parts of the legacy scorer
(EWMA/median filtering, multi-scale persistence, t-tests).  It:

1. aligns the three feature artifacts onto one 1 Hz timeline;
2. robustly normalises each signal;
3. builds a weighted recall curve;
4. collects composite peaks plus signal-specific anchors (keywords, laughter,
   applause, speaker changes, scene cuts);
5. de-duplicates anchors and emits a bounded number of candidate windows.

The output is *not* a final clip list — stage 2 reranks these candidates with
window-level features.
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks

from src.contracts import atomicWriteJson
from src.log.events import logEvent
from src.log.logger import getLogger
from src.pipeline.window_features import (
    normalizeFeatureArrays,
    resolveProfileWeights,
)
from src.scorer import alignFeatureArrays
from src.util import validateFeatures

log = getLogger(__name__)

DEFAULT_ANCHOR_SIGNALS = (
    "asrKeyword",
    "speakerChange",
    "eventLaughter",
    "eventApplause",
    "eventMusic",
    "sceneChange",
)


def _collectRawFeatures(
    audioFeatures: dict | None,
    textFeatures: dict | None,
    visualFeatures: dict | None,
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    for source in (audioFeatures, textFeatures, visualFeatures):
        if not source:
            continue
        for name, values in (source.get("features") or {}).items():
            if isinstance(values, list) and values:
                raw[name] = np.asarray(values, dtype=np.float64)
    return raw


def _buildRecallCurve(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted signal sum plus a bounded onset bonus."""
    if not signals:
        return np.zeros(0, dtype=np.float64)

    first = next(iter(signals.values()))
    curve = np.zeros(len(first), dtype=np.float64)
    for name, z in signals.items():
        weight = weights.get(name, 0.0)
        if weight <= 0:
            continue
        if len(z) != len(curve):
            continue
        curve += weight * z

    # Onsets matter for live highlights: a signal that suddenly jumps is
    # often more interesting than one that was already high.  Keep this cheap
    # and bounded so it cannot dominate the level component.
    onset = np.zeros_like(curve)
    for name, z in signals.items():
        weight = weights.get(name, 0.0)
        if weight <= 0 or len(z) != len(curve):
            continue
        diff = np.diff(z, prepend=z[0])
        onset += weight * np.maximum(diff, 0.0)
    return curve + 0.2 * np.tanh(onset)


def _sourcesAt(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
    index: int,
    sourceZ: float,
) -> list[str]:
    sources = []
    for name, z in signals.items():
        if weights.get(name, 0.0) <= 0 or len(z) == 0:
            continue
        lo = max(0, index - 1)
        hi = min(len(z), index + 2)
        localPeak = float(np.max(z[lo:hi])) if hi > lo else 0.0
        if localPeak >= sourceZ:
            sources.append(name)
    return sorted(sources)


def _makeWindow(
    centre: float,
    nSecs: int,
    windowSeconds: float,
    minDuration: float,
    maxDuration: float,
) -> tuple[float, float]:
    if nSecs <= 0:
        return 0.0, 0.0
    if nSecs <= minDuration:
        return 0.0, float(nSecs)

    target = min(maxDuration, max(minDuration, windowSeconds))
    target = min(target, float(nSecs))
    start = centre - target / 2.0
    start = max(0.0, min(start, float(nSecs) - target))
    return round(start, 2), round(start + target, 2)


def recallCandidates(
    roomId: str = "",
    audioFeatures: dict | None = None,
    textFeatures: dict | None = None,
    visualFeatures: dict | None = None,
    durationSec: float | None = None,
    profile: str = "default",
    recallPerHour: float = 60.0,
    minGapSec: float = 12.0,
    recallThreshold: float = 0.8,
    windowSeconds: float = 30.0,
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    agreementZ: float = 1.0,
    anchorSignals: Iterable[str] | None = None,
    outputPath: str | None = None,
    runId: str | None = None,
) -> dict:
    """Generate stage-1 candidate windows.

    Args:
        roomId: Room identifier (metadata only).
        audioFeatures, textFeatures, visualFeatures: Raw signal artifacts.
        durationSec: Authoritative recording duration.
        profile: Scorer weight profile used for the recall curve.
        recallPerHour: Candidate budget per hour of recording.
        minGapSec: Minimum distance between candidate centres.
        recallThreshold: Peak prominence on the recall curve.
        windowSeconds: Preferred candidate window length.
        minDuration, maxDuration: Allowed window bounds.
        agreementZ: Z-score at which a signal counts as active.
        anchorSignals: Signal names that always contribute an anchor.
        outputPath: Optional path for the full recall document.

    Returns:
        ``{"candidates": [...], "recallCurve": [...], ...}``.
    """
    startedAt = time.perf_counter()
    if minDuration > maxDuration:
        raise ValueError("minDuration must not exceed maxDuration")
    if recallPerHour <= 0:
        raise ValueError("recallPerHour must be positive")
    if minGapSec <= 0:
        raise ValueError("minGapSec must be positive")

    raw = _collectRawFeatures(audioFeatures, textFeatures, visualFeatures)
    if not raw:
        log.error("Stage 1: no features found")
        return {
            "roomId": roomId,
            "duration": 0,
            "candidates": [],
            "recallCurve": [],
            "alignment": [],
            "stats": {"candidateCount": 0, "anchorCount": 0, "elapsedMs": 0},
        }

    targetLength = int(math.ceil(durationSec)) if durationSec and durationSec > 0 else None
    aligned, alignment, nSecs = alignFeatureArrays(raw, targetLength)
    if nSecs == 0 or not aligned:
        log.error("Stage 1: no usable features after alignment")
        return {
            "roomId": roomId,
            "duration": 0,
            "candidates": [],
            "recallCurve": [],
            "alignment": alignment,
            "stats": {"candidateCount": 0, "anchorCount": 0, "elapsedMs": 0},
        }

    validateFeatures(aligned, sourceLabel="stage1 recall input")
    signals = normalizeFeatureArrays(aligned)
    weights = resolveProfileWeights(profile, signals.keys())
    curve = _buildRecallCurve(signals, weights)

    hours = max(nSecs / 3600.0, 1.0 / 60.0)
    cap = max(1, int(math.ceil(recallPerHour * hours)))
    distance = max(1, int(round(minGapSec)))

    anchors: list[dict] = []
    if nSecs >= 3 and curve.size == nSecs:
        peaks, properties = find_peaks(
            curve,
            distance=distance,
            prominence=max(0.0, recallThreshold),
        )
        for idx in peaks:
            anchors.append({
                "index": int(idx),
                "score": round(float(curve[idx]), 6),
                "anchor": "recallCurve",
            })

    anchorOffset = max(1, int(round(minGapSec / 2.0)))
    for name in (anchorSignals if anchorSignals is not None else DEFAULT_ANCHOR_SIGNALS):
        z = signals.get(name)
        if z is None or len(z) == 0 or weights.get(name, 0.0) <= 0:
            continue
        signalPeaks, _ = find_peaks(z, distance=max(1, anchorOffset), height=agreementZ)
        for idx in signalPeaks:
            anchors.append({
                "index": int(idx),
                "score": round(float(curve[idx]) if idx < len(curve) else z[idx], 6),
                "anchor": name,
            })

    # Fallback: a broad activity plateau may not produce a local maximum at
    # the requested prominence.  Keep recall from collapsing to zero.
    if not anchors and curve.size:
        positive = np.where(curve > 0)[0]
        for idx in positive[np.argsort(curve[positive])[::-1]][:cap]:
            anchors.append({
                "index": int(idx),
                "score": round(float(curve[idx]), 6),
                "anchor": "recallCurve",
            })

    anchors.sort(
        key=lambda item: (
            -item["score"],
            0 if item["anchor"] != "recallCurve" else 1,
            item["index"],
        )
    )
    selected: list[dict] = []
    for anchor in anchors:
        centre = float(anchor["index"])
        nearby = next(
            (
                item for item in selected
                if abs(centre - item["peakTime"]) < minGapSec
            ),
            None,
        )
        if nearby is not None:
            # Merge nearby anchors instead of discarding their provenance;
            # otherwise a keyword anchor a few seconds from a composite peak
            # silently loses its `asrKeyword` source.
            nearby["_indices"].append(anchor["index"])
            nearby["_anchorSignals"].add(anchor["anchor"])
            if nearby["anchor"] == "recallCurve" and anchor["anchor"] != "recallCurve":
                nearby["anchor"] = anchor["anchor"]
            nearby["score"] = max(nearby["score"], round(float(anchor["score"]), 6))
            continue
        if len(selected) >= cap:
            continue

        start, end = _makeWindow(
            centre, nSecs, windowSeconds, minDuration, maxDuration
        )
        selected.append({
            "start": start,
            "end": end,
            "duration": round(end - start, 2),
            "score": round(float(anchor["score"]), 6),
            "peakTime": round(centre, 2),
            "anchor": anchor["anchor"],
            "_indices": [anchor["index"]],
            "_anchorSignals": {anchor["anchor"]},
        })

    selected.sort(key=lambda item: (item["start"], item["end"]))
    candidates = []
    for index, candidate in enumerate(selected, start=1):
        indices = candidate.pop("_indices")
        anchorSignals = sorted(candidate.pop("_anchorSignals"))
        sources: list[str] = [
            name for name in anchorSignals if name != "recallCurve"
        ]
        for anchorIndex in indices:
            for name in _sourcesAt(signals, weights, anchorIndex, agreementZ):
                if name not in sources:
                    sources.append(name)
        if not sources:
            sources = ["recallCurve"]
        candidate["sources"] = sources
        candidates.append({"id": index, **candidate})

    elapsedMs = int((time.perf_counter() - startedAt) * 1000)
    document = {
        "roomId": roomId,
        "duration": int(nSecs),
        "profile": profile,
        "candidates": candidates,
        "recallCurve": [round(float(value), 6) for value in curve.tolist()],
        "alignment": alignment,
        "weights": {name: round(float(weight), 6) for name, weight in weights.items()},
        "stats": {
            "candidateCount": len(candidates),
            "anchorCount": len(anchors),
            "candidateCap": cap,
            "recallPerHour": recallPerHour,
            "minGapSec": minGapSec,
            "recallThreshold": recallThreshold,
            "windowSeconds": windowSeconds,
            "elapsedMs": elapsedMs,
        },
    }

    if outputPath:
        atomicWriteJson(outputPath, document)
        log.info(f"Stage 1 recall saved: {outputPath}")

    log.info(
        f"Stage 1 recall: {len(candidates)} candidates from {len(anchors)} anchors "
        f"({nSecs}s, {elapsedMs}ms)"
    )
    logEvent(
        "pipeline.stage",
        runId=runId,
        roomId=roomId,
        step="stage1_recall",
        status="ok" if candidates else "empty",
        elapsedMs=elapsedMs,
        candidateCount=len(candidates),
        anchorCount=len(anchors),
        durationSec=int(nSecs),
    )
    return document
