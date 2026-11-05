# -*- coding: utf-8 -*-
"""Content-aware boundary optimization for two-stage clip windows.

S5-2 upgrades the fixed-width candidate windows from stage 1 with:

* content-type duration presets (funny / shopping / interaction / talent);
* optional platform duration limits (soft, used only when ranges overlap);
* boundary candidates from ASR sentence edges, silence minima, scene cuts,
  score-curve valleys and the original boundary;
* a joint start/end search that balances semantic snapping, movement cost,
  duration fit and score retention.

The module is pure and deterministic; it does not import the scorer or touch
controller state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks

# Initial content-type targets from the Sprint 5 plan.  These are deliberately
# conservative and easy to tune once an annotated set exists.
CONTENT_DURATION_PRESETS: dict[str, tuple[float, float]] = {
    "funny": (12.0, 35.0),
    "shopping": (20.0, 60.0),
    "interaction": (30.0, 90.0),
    "talent": (45.0, 120.0),
    "game": (15.0, 60.0),
    "default": (15.0, 90.0),
}

# Soft platform preferences.  A platform range is applied only when it
# overlaps the content/user range; otherwise the content preset wins.
PLATFORM_DURATION_LIMITS: dict[str, tuple[float, float]] = {
    "douyin": (15.0, 60.0),
    "bilibili": (60.0, 180.0),
    "xiaohongshu": (15.0, 60.0),
}

# Higher weight = more trustworthy boundary.  The distance penalty is
# subtracted from this value during the search.
BOUNDARY_SOURCE_WEIGHTS: dict[str, float] = {
    "asrSentence": 1.0,
    "silence": 0.9,
    "sceneChange": 0.8,
    "scoreValley": 0.7,
    "original": 0.2,
}


def inferContentType(profile: str, signals: Iterable[str]) -> str:
    """Infer a coarse content type from the profile and active signals.

    The mapping is intentionally rule-based: Sprint 5 has no annotated set
    yet, so a learned classifier would be premature.
    """
    names = set(signals)
    # An explicit non-default profile is the strongest signal: a game stream
    # that happens to contain laughter should stay a game highlight.
    if profile in {"shopping", "talent", "game"}:
        return profile
    if "speakerChange" in names:
        return "interaction"
    if "eventLaughter" in names:
        return "funny"
    if "eventMusic" in names:
        return "talent"
    return "default"


def targetDurationRange(
    contentType: str,
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    platform: str | None = None,
) -> tuple[float, float]:
    """Return the effective ``(min, max)`` duration for one content type.

    User-provided bounds are hard constraints.  Platform limits are a soft
    preference and are ignored when they would make the range empty.
    """
    if minDuration > maxDuration:
        raise ValueError("minDuration must not exceed maxDuration")

    presetMin, presetMax = CONTENT_DURATION_PRESETS.get(
        contentType, CONTENT_DURATION_PRESETS["default"]
    )
    lo = max(float(minDuration), presetMin)
    hi = min(float(maxDuration), presetMax)
    if lo > hi:
        # The user's explicit bounds always win over a content preset.
        return float(minDuration), float(maxDuration)

    platformLimits = PLATFORM_DURATION_LIMITS.get(str(platform or "").lower())
    if platformLimits is not None:
        pLo, pHi = platformLimits
        candidateLo = max(lo, pLo)
        candidateHi = min(hi, pHi)
        if candidateLo <= candidateHi:
            lo, hi = candidateLo, candidateHi

    return float(lo), float(hi)


def _edgeCost(candidate: dict, edge: float, searchSec: float) -> float:
    distance = abs(float(candidate["time"]) - float(edge))
    return distance / max(searchSec, 1.0) - float(candidate["weight"])


def _edgeCandidates(
    edge: float,
    signals: dict[str, np.ndarray],
    scoreCurve: np.ndarray,
    asrSegments: list[dict] | None,
    searchSec: float,
) -> list[dict]:
    """Collect candidate boundary times within ±``searchSec`` of ``edge``."""
    nSecs = len(scoreCurve)
    if nSecs <= 0:
        return []

    lo = max(0, int(math.floor(edge - searchSec)))
    hi = min(nSecs, int(math.ceil(edge + searchSec)))
    signalHi = max(lo, min(nSecs - 1, hi))
    raw: list[dict] = []

    def add(timeValue: float, source: str) -> None:
        try:
            rounded = int(round(float(timeValue)))
        except (TypeError, ValueError):
            return
        if 0 <= rounded <= nSecs:
            raw.append({
                "time": float(rounded),
                "source": source,
                "weight": BOUNDARY_SOURCE_WEIGHTS[source],
            })

    add(edge, "original")

    # ASR sentence starts/ends are the strongest semantic boundary source.
    for segment in asrSegments or []:
        if not isinstance(segment, dict):
            continue
        for key in ("start", "end"):
            value = segment.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if lo <= numeric <= hi:
                add(numeric, "asrSentence")

    # Silence/pause: local minima of RMS, i.e. the quietest second nearby.
    # A perfectly flat region has no meaningful interior minimum, so it is
    # skipped instead of producing a false boundary that merely mirrors the
    # original edge.
    rms = signals.get("rms")
    if rms is not None and len(rms) >= 3:
        region = np.asarray(rms[lo:signalHi + 1], dtype=np.float64)
        if region.size >= 3 and float(np.ptp(region)) > 1e-9:
            minima, _ = find_peaks(-region, distance=2)
            for index in minima:
                add(lo + int(index), "silence")

    # Scene cuts are editorial-friendly boundaries.
    sceneChange = signals.get("sceneChange")
    if sceneChange is not None and len(sceneChange) >= 3:
        region = np.asarray(sceneChange[lo:signalHi + 1], dtype=np.float64)
        if region.size >= 3:
            cuts, _ = find_peaks(region, height=0.5, distance=1)
            for index in cuts:
                add(lo + int(index), "sceneChange")

    # Low points of the fused score curve: cutting in a trough loses less.
    # Flat segments carry no trough, so avoid manufacturing one.
    if len(scoreCurve) >= 3:
        region = np.asarray(scoreCurve[lo:signalHi + 1], dtype=np.float64)
        if region.size >= 3 and float(np.ptp(region)) > 1e-9:
            valleys, _ = find_peaks(-region, distance=2)
            for index in valleys:
                add(lo + int(index), "scoreValley")

    # De-duplicate times, keeping the strongest semantic source.
    bestByTime: dict[int, dict] = {}
    for candidate in raw:
        key = int(candidate["time"])
        existing = bestByTime.get(key)
        if existing is None or candidate["weight"] > existing["weight"]:
            bestByTime[key] = candidate
    return sorted(bestByTime.values(), key=lambda item: item["time"])


def _selectEdgeCandidates(
    candidates: list[dict],
    edge: float,
    searchSec: float,
    limit: int = 10,
) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (_edgeCost(item, edge, searchSec), item["time"]),
    )[:limit]
    # The original placement must always stay available as a fallback.
    if not any(abs(item["time"] - edge) < 0.5 for item in ranked):
        ranked.append({
            "time": float(round(edge)),
            "source": "original",
            "weight": BOUNDARY_SOURCE_WEIGHTS["original"],
        })
    return ranked


def optimizeBoundaries(
    start: float,
    end: float,
    signals: dict[str, np.ndarray],
    scoreCurve: np.ndarray,
    asrSegments: list[dict] | None = None,
    targetMin: float = 15.0,
    targetMax: float = 90.0,
    searchSec: float = 8.0,
    contentType: str = "default",
    platform: str | None = None,
) -> dict:
    """Jointly choose start/end boundaries near a candidate window.

    Returns a JSON-serialisable dict with the chosen boundaries, their
    sources, the inferred content type and the effective target duration.
    When no semantic boundary is worth moving to, the original window is
    returned unchanged.
    """
    nSecs = len(scoreCurve)
    if nSecs <= 0:
        return {
            "start": float(start),
            "end": float(end),
            "duration": max(0.0, float(end) - float(start)),
            "changed": False,
            "startSource": "original",
            "endSource": "original",
            "boundaryReasons": [],
            "contentType": contentType,
            "targetDuration": [float(targetMin), float(targetMax)],
        }

    origStart = max(0.0, min(float(start), float(nSecs)))
    origEnd = max(0.0, min(float(end), float(nSecs)))
    if origEnd < origStart:
        origStart, origEnd = origEnd, origStart

    if searchSec <= 0:
        return {
            "start": round(origStart, 2),
            "end": round(origEnd, 2),
            "duration": round(origEnd - origStart, 2),
            "changed": False,
            "startSource": "original",
            "endSource": "original",
            "boundaryReasons": [],
            "contentType": contentType,
            "targetDuration": [float(targetMin), float(targetMax)],
        }

    startCandidates = _selectEdgeCandidates(
        _edgeCandidates(origStart, signals, scoreCurve, asrSegments, searchSec),
        origStart,
        searchSec,
    )
    endCandidates = _selectEdgeCandidates(
        _edgeCandidates(origEnd, signals, scoreCurve, asrSegments, searchSec),
        origEnd,
        searchSec,
    )

    baseSum = float(np.sum(scoreCurve[int(origStart):int(origEnd)]))
    baseScale = max(abs(baseSum), 1.0)
    bestStart = origStart
    bestEnd = origEnd
    bestStartSource = "original"
    bestEndSource = "original"
    bestCost = float("inf")

    for startCandidate in startCandidates:
        startTime = float(startCandidate["time"])
        for endCandidate in endCandidates:
            endTime = float(endCandidate["time"])
            duration = endTime - startTime
            if duration < 1.0:
                continue

            if duration < targetMin:
                durationPenalty = (
                    1.0 + (targetMin - duration) / max(targetMin, 1.0) * 2.0
                )
            elif duration > targetMax:
                durationPenalty = (
                    1.0 + (duration - targetMax) / max(targetMax, 1.0) * 2.0
                )
            else:
                durationPenalty = 0.0

            newSum = float(np.sum(scoreCurve[int(startTime):int(endTime)]))
            retentionPenalty = max(0.0, baseSum - newSum) / baseScale * 0.3
            cost = (
                _edgeCost(startCandidate, origStart, searchSec)
                + _edgeCost(endCandidate, origEnd, searchSec)
                + durationPenalty
                + retentionPenalty
            )
            if cost < bestCost - 1e-9:
                bestCost = cost
                bestStart = startTime
                bestEnd = endTime
                bestStartSource = startCandidate["source"]
                bestEndSource = endCandidate["source"]

    changed = abs(bestStart - origStart) >= 0.5 or abs(bestEnd - origEnd) >= 0.5
    reasons: list[str] = []
    for source in (bestStartSource, bestEndSource):
        if source != "original":
            reason = f"boundary_{source}"
            if reason not in reasons:
                reasons.append(reason)

    return {
        "start": round(bestStart, 2),
        "end": round(bestEnd, 2),
        "duration": round(bestEnd - bestStart, 2),
        "changed": changed,
        "startSource": bestStartSource,
        "endSource": bestEndSource,
        "boundaryReasons": reasons,
        "contentType": contentType,
        "targetDuration": [float(targetMin), float(targetMax)],
    }
