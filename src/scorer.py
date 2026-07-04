# -*- coding: utf-8 -*-
"""Scorer — multimodal fusion scoring, peak detection, boundary optimization.

Design features NOT yet implemented (see docs/superpowers/specs/):
  TODO: DTW alignment — dynamic time warping for cross-modal time calibration
  TODO: Gaussian envelope boundary optimization — RMS gaussian fit for fade-in/out
  TODO: SSIM dedup — structural similarity to merge near-duplicate clips
  TODO: Local DP optimization — dynamic programming for optimal boundary search
  TODO: scores.json breakdown — per-second signal contributions for explainability
  TODO: HDBSCAN topic clustering — replace KMeans for adaptive cluster count
"""

import json
import numpy as np
from scipy import signal as sp_signal, stats as sp_stats
from src.log.logger import getLogger
from src.util import FEATURE_NAMES, validateFeatures

log = getLogger(__name__)

# Default weights per profile
PROFILES = {
    "default": {
        "dmDensity": 0.15, "dmAcceleration": 0.05, "dmEntropy": 0.05,
        "dmSentiment": 0.10, "dmUtr": 0.03,
        "asrKeyword": 0.10, "topicChange": 0.05, "speakerChange": 0.05,
        "rms": 0.10, "mfccDist": 0.02,
        "eventLaughter": 0.10, "eventApplause": 0.05, "eventMusic": 0.05,
        "sceneChange": 0.05, "motion": 0.05, "faceCount": 0.00,
    },
    "game": {
        "dmDensity": 0.20, "dmAcceleration": 0.10, "dmEntropy": 0.05,
        "dmSentiment": 0.05, "dmUtr": 0.03,
        "asrKeyword": 0.08, "topicChange": 0.05, "speakerChange": 0.03,
        "rms": 0.08, "mfccDist": 0.02,
        "eventLaughter": 0.10, "eventApplause": 0.10, "eventMusic": 0.03,
        "sceneChange": 0.05, "motion": 0.10, "faceCount": 0.00,
    },
    "shopping": {
        "dmDensity": 0.10, "dmAcceleration": 0.03, "dmEntropy": 0.03,
        "dmSentiment": 0.05, "dmUtr": 0.03,
        "asrKeyword": 0.20, "topicChange": 0.05, "speakerChange": 0.05,
        "rms": 0.05, "mfccDist": 0.02,
        "eventLaughter": 0.05, "eventApplause": 0.05, "eventMusic": 0.05,
        "sceneChange": 0.05, "motion": 0.03, "faceCount": 0.00,
    },
    "talent": {
        "dmDensity": 0.10, "dmAcceleration": 0.05, "dmEntropy": 0.03,
        "dmSentiment": 0.15, "dmUtr": 0.03,
        "asrKeyword": 0.10, "topicChange": 0.05, "speakerChange": 0.10,
        "rms": 0.20, "mfccDist": 0.02,
        "eventLaughter": 0.05, "eventApplause": 0.05, "eventMusic": 0.05,
        "sceneChange": 0.03, "motion": 0.03, "faceCount": 0.00,
    },
}

DETECTION_WINDOWS = [3, 10, 30]
NMS_MERGE_DISTANCE = 15.0
MIN_CLIP_DURATION = 15.0
MAX_CLIP_DURATION = 90.0


def score(
    roomId: str = "",
    audioFeatures: dict | None = None,
    textFeatures: dict | None = None,
    visualFeatures: dict | None = None,
    profile: str = "default",
    sensitivity: float = 2.5,
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    outputPath: str | None = None,
    scoresPath: str | None = None,
) -> dict:
    """Run the full scoring pipeline.

    Args:
        roomId: Room identifier (for output metadata).
        audioFeatures: Output of signalAudio.py.
        textFeatures: Output of signalText.py.
        visualFeatures: Output of signalVisual.py.
        profile: Weight profile name.
        sensitivity: k value for adaptive threshold.
        minDuration, maxDuration: Clip duration bounds.
        outputPath: Save time_table.json.
        scoresPath: Save per-second scores.json.

    Returns:
        dict with 'clips' list.
    """
    af = audioFeatures or {"features": {}}
    tf = textFeatures or {"features": {}}
    vf = visualFeatures or {"features": {}}

    allFeatures: dict[str, np.ndarray] = {}
    for src in [af, tf, vf]:
        for k, v in src.get("features", {}).items():
            if isinstance(v, list) and len(v) > 0:
                allFeatures[k] = np.array(v, dtype=np.float64)

    durations = [len(v) for v in allFeatures.values()]
    nSecs = max(durations) if durations else 0
    if nSecs == 0:
        log.error("No features found")
        return {"clips": []}

    # Validate feature names against canonical set (catch silent signal loss)
    validateFeatures(allFeatures, sourceLabel="scorer input")

    log.debug(f"Feature matrix: {nSecs}s × {len(allFeatures)} signals")

    # ── Step 1: Preprocess ────────────────────────────────────
    zMatrix = _preprocess(allFeatures)

    # ── Step 2: Time-lag compensation (cross-correlation) ──────
    zMatrix = _compensateLag(zMatrix)

    # ── Score ─────────────────────────────────────────────────
    weights = PROFILES.get(profile, PROFILES["default"])
    scoreBase = np.zeros(nSecs)
    for name, z in zMatrix.items():
        w = weights.get(name, 0.0)
        if w > 0 and len(z) == nSecs:
            scoreBase += w * z

    # Boost (tanh of signal change)
    boost = np.zeros(nSecs)
    for name, z in zMatrix.items():
        w = weights.get(name, 0.0)
        if w <= 0 or len(z) != nSecs:
            continue
        diff = np.abs(np.diff(z, prepend=z[0]))
        boost += w * np.tanh(diff / 2.0)

    # Time decay: exp(-|t - candidate| / τ), τ=5s
    # Applied per-candidate in _detectPeaks, but also apply globally:
    decayGlobal = np.exp(-np.arange(nSecs) / max(nSecs, 1) * 0.1)  # slow global decay
    finalScore = (scoreBase + boost) * decayGlobal
    finalScore = np.nan_to_num(finalScore, nan=0.0)

    # ── Score full timeline (for scores.json) ─────────────────
    scoresTimeline = finalScore.tolist()

    # ── Peak detection ────────────────────────────────────────
    clips = _detectPeaks(finalScore, sensitivity)

    # ── Boundary optimization ──────────────────────────────────
    clips = _optimizeBoundaries(clips, nSecs, zMatrix)

    # ── NMS ───────────────────────────────────────────────────
    clips = _nms(clips)

    # ── Quality checks ─────────────────────────────────────────
    clips = _qualityFilter(clips, minDuration, maxDuration, finalScore, nSecs)

    # ── Build output ──────────────────────────────────────────
    resultClips = []
    for i, (start, end, peakScore) in enumerate(clips):
        trigger = _findTrigger(start, end, zMatrix, allFeatures)
        title = _generateTitle(trigger, start, zMatrix, allFeatures)
        tags = _generateTags(trigger)

        resultClips.append({
            "id": i + 1,
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(end - start, 1),
            "score": round(float(peakScore), 3),
            "trigger": trigger,
            "title": title,
            "tags": tags,
            "thumbnailTime": round(start + 3.0, 1),  # start+3s
            "breakdown": _getBreakdown(start, end, zMatrix, weights),
        })

    output = {
        "roomId": roomId,
        "duration": nSecs,
        "profile": profile,
        "sensitivity": sensitivity,
        "clips": resultClips,
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        log.info(f"Time table saved: {outputPath}")

    if scoresPath:
        with open(scoresPath, "w", encoding="utf-8") as f:
            json.dump({"roomId": roomId, "duration": nSecs, "scores": scoresTimeline}, f, ensure_ascii=False)
        log.info(f"Scores saved: {scoresPath}")

    log.info(f"Scored: {len(resultClips)} clips")
    return output


# ── Preprocessing ──────────────────────────────────────────────

def _preprocess(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """EWMA → Z-Score → Median filter.

    Pipeline per signal:
      1. EWMA  (α=0.3) — O(n) iterative (not the bottleneck)
      2. Rolling Z-Score (60s window) via uniform_filter1d (vectorized O(n))
      3. Median filter (5s) via ndimage.median_filter (vectorized O(n))
    """
    from scipy.ndimage import uniform_filter1d, median_filter

    result = {}
    for name, arr in features.items():
        n = len(arr)
        if n < 2:
            result[name] = np.zeros(n)
            continue

        # 1. EWMA: y[t] = α·x[t] + (1-α)·y[t-1], y[0] = x[0]
        alpha = 0.3
        smoothed = np.empty_like(arr)
        smoothed[0] = arr[0]
        for t in range(1, n):
            smoothed[t] = alpha * arr[t] + (1.0 - alpha) * smoothed[t - 1]

        # 2. Rolling Z-Score: (x - μ) / σ over 60s window (vectorized)
        # For short recordings (≤60s) fall back to global Z-Score so
        # clips are still detectable instead of returning all zeros.
        window = min(60, n)
        if n >= 3:
            if n > window:
                rollMean = uniform_filter1d(smoothed, size=window + 1, mode="nearest")
                rollMeanSq = uniform_filter1d(smoothed ** 2, size=window + 1, mode="nearest")
            else:
                # Global mean / std fallback for short recordings
                rollMean = np.full(n, np.mean(smoothed))
                rollMeanSq = np.full(n, np.mean(smoothed ** 2))
            rollStd = np.sqrt(np.maximum(rollMeanSq - rollMean ** 2, 0.0))
            # Avoid div-by-zero: use np.divide with where mask
            with np.errstate(divide="ignore", invalid="ignore"):
                zScore = np.divide(
                    smoothed - rollMean, rollStd,
                    where=rollStd > 1e-10, out=np.zeros(n),
                )
        else:
            zScore = np.zeros(n)

        # 3. Median filter (5s window, vectorized)
        medianF = median_filter(zScore, size=5, mode="nearest")

        result[name] = medianF
    return result


def _compensateLag(zMatrix: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cross-correlation lag compensation between danmaku and audio.

    Estimates the integer shift (in seconds) that maximises correlation
    between the first dm-prefixed signal and the first audio signal,
    then rolls all dm signals by that shift, zeroing the wrap-around
    region so wrapped samples never re-enter as fake spikes.
    """
    dmKeys = [k for k in zMatrix if k.startswith("dm")]
    audioKeys = [k for k in zMatrix if k in ("rms", "eventLaughter", "eventApplause")]

    if not (dmKeys and audioKeys):
        return zMatrix

    dmSig = zMatrix[dmKeys[0]]
    auSig = zMatrix[audioKeys[0]]
    minLen = min(len(dmSig), len(auSig))
    if minLen <= 10:
        return zMatrix

    try:
        corr = sp_signal.correlate(dmSig[:minLen], auSig[:minLen])
        lag = int(np.argmax(corr) - (minLen - 1))
        if not (0 < abs(lag) <= 5):
            return zMatrix

        log.debug(f"Danmaku→Audio lag: {lag}s — shifting dm signals")
        # np.roll(x, -lag) shifts dm by `-lag`; zero the wrap-around band so
        # tail/head samples don't reappear at the opposite end as ghost peaks.
        shift = -lag
        for k in list(zMatrix.keys()):
            if not k.startswith("dm"):
                continue
            rolled = np.roll(zMatrix[k], shift)
            if shift > 0:
                rolled[:shift] = 0       # head wrapped from tail
            else:
                rolled[shift:] = 0       # tail wrapped from head (shift is negative)
            zMatrix[k] = rolled
    except Exception:
        log.debug("Lag compensation failed", exc_info=True)
    return zMatrix


# ── Detection ──────────────────────────────────────────────────

def _detectPeaks(score: np.ndarray, k: float) -> list[tuple[float, float, float]]:
    """Multi-scale peak detection with persistence requirement.

    Uses vectorized rolling mean/std via scipy's uniform_filter1d
    (O(n) per window instead of O(n*w)), ~10-100x faster for long recordings.
    """
    from scipy.ndimage import uniform_filter1d

    n = len(score)
    candidates: set[int] = set()

    for window in DETECTION_WINDOWS:
        if 2 * window >= n:
            continue  # window too large for this recording

        # Rolling mean and std via uniform filter (boxcar window)
        # size = 2*window for symmetric ±window neighborhood
        localMean = uniform_filter1d(score, size=2 * window, mode="nearest")
        localMeanSq = uniform_filter1d(score ** 2, size=2 * window, mode="nearest")
        localStd = np.sqrt(np.maximum(localMeanSq - localMean ** 2, 0.0))

        threshold = localMean + k * localStd

        # Element-wise: is score[t] > threshold[t]?
        above = score > threshold

        # Persistence: score[t], score[t+1], score[t+2] must ALL exceed threshold[t]
        # threshold[t] varies per position, so check each shift against the same threshold
        persistent = above.copy()
        persistent[:-1] &= score[1:] > threshold[:-1]  # t+1 > threshold[t]
        persistent[:-2] &= score[2:] > threshold[:-2]  # t+2 > threshold[t]
        persistent[-2:] = False  # can't have persistence near the end

        for t in np.where(persistent)[0]:
            candidates.add(int(t))

    sortedCands = sorted(candidates)
    if not sortedCands:
        return []

    # Cluster nearby candidate points into ranges
    ranges: list[tuple[int, int, float]] = []
    start = sortedCands[0]
    prev = start
    maxScore = score[start]
    for t in sortedCands[1:]:
        if t - prev > 3:
            ranges.append((start, prev, maxScore))
            start = t
            maxScore = score[t]
        else:
            maxScore = max(maxScore, score[t])
        prev = t
    ranges.append((start, prev, maxScore))

    return [(float(s), float(e), float(sc)) for s, e, sc in ranges]


def _optimizeBoundaries(
    clips: list[tuple[float, float, float]],
    nSecs: int,
    zMatrix: dict[str, np.ndarray],
) -> list[tuple[float, float, float]]:
    """Semantic boundary alignment via Z-Score valley detection."""
    optimized = []
    for s, e, sc in clips:
        newStart = max(0, s - 5)
        newEnd = min(nSecs, e + 5)

        # Search for valleys (quiet moments) near boundaries.
        # Use RMS Z-Score: a true valley sits below the local mean over ±5s.
        rms = zMatrix.get("rms")
        rmsLen = len(rms) if rms is not None else 0
        if rmsLen > 0:
            # Find valley before peak — walk back ±10s, snap to first dip
            for offset in range(10, 0, -1):
                t = int(s) - offset
                if 0 <= t < rmsLen:
                    window = rms[max(0, t - 5):min(rmsLen, t + 5)]
                    if window.size > 0 and rms[t] < float(np.mean(window)):
                        newStart = t
                        break

            # Find valley after peak — walk forward up to 10s
            for offset in range(10):
                t = int(e) + offset
                if 0 <= t < rmsLen:
                    window = rms[max(0, t - 5):min(rmsLen, t + 5)]
                    if window.size > 0 and rms[t] < float(np.mean(window)):
                        newEnd = t
                        break

        localBest = (max(0, newStart), min(nSecs, newEnd), sc)
        optimized.append(localBest)
    return optimized


def _nms(clips: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Non-maximum suppression."""
    if not clips:
        return []
    sortedClips = sorted(clips, key=lambda x: x[0])
    merged = [sortedClips[0]]
    for clip in sortedClips[1:]:
        last = merged[-1]
        if clip[0] - last[1] < NMS_MERGE_DISTANCE:
            merged[-1] = (min(last[0], clip[0]), max(last[1], clip[1]), max(last[2], clip[2]))
        else:
            merged.append(clip)
    return merged


def _qualityFilter(
    clips: list[tuple[float, float, float]],
    minDur: float,
    maxDur: float,
    score: np.ndarray,
    nSecs: int,
) -> list[tuple[float, float, float]]:
    """Quality filter: duration bounds + t-test."""
    filtered = []
    for s, e, sc in clips:
        dur = e - s
        if dur < minDur or dur > maxDur:
            continue

        # t-test: scores in clip vs scores immediately before.
        # Failure here (e.g. zero-variance scores → NaN) shouldn't kill the
        # clip — fall through to keep it; quality bounds already passed.
        clipScores = score[int(s):int(e)]
        preStart = max(0, int(s) - 5)
        preScores = score[preStart:int(s)]
        if len(preScores) > 2 and len(clipScores) > 2:
            try:
                _, p = sp_stats.ttest_ind(clipScores, preScores)
                if p > 0.05:
                    continue
            except Exception:
                log.debug(f"t-test failed for clip {s}-{e}", exc_info=True)

        filtered.append((s, e, sc))
    return filtered


# ── Trigger & title ────────────────────────────────────────────

def _findTrigger(
    start: float, end: float,
    zMatrix: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> str:
    """Determine trigger with keyword context."""
    parts = []
    firstSignalLen = len(next(iter(zMatrix.values()), [])) if zMatrix else 0
    s, e = int(start), min(firstSignalLen, int(np.ceil(end)))

    for name, z in zMatrix.items():
        if s >= len(z):
            continue
        peak = float(np.max(z[s:e]))
        if peak > 2.0:
            if name == "asrKeyword":
                # Find which keyword triggered
                asr = features.get("asrKeyword")
                if asr is not None and len(asr) > s:
                    parts.append(f"asrKeyword")
            else:
                parts.append(name)

    return "+".join(parts) if parts else "unknown"


def _generateTitle(
    trigger: str, start: float,
    zMatrix: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> str:
    """Generate descriptive title."""
    if "asrKeyword" in trigger:
        return f"💬 关键词片段 ({int(start)}s)"
    if "eventLaughter" in trigger:
        return f"😂 搞笑高能 ({int(start)}s)"
    if "eventApplause" in trigger:
        return f"👏 精彩时刻 ({int(start)}s)"
    if "dmDensity" in trigger or "dmAcceleration" in trigger:
        return f"🔥 弹幕爆发 ({int(start)}s)"
    if "speakerChange" in trigger:
        return f"🗣 连麦高光 ({int(start)}s)"
    if "rms" in trigger:
        return f"🔊 音量爆发 ({int(start)}s)"
    if "motion" in trigger:
        return f"📹 画面高能 ({int(start)}s)"
    return f"📹 高光时刻 ({int(start)}s)"


def _generateTags(trigger: str) -> list[str]:
    """Generate tags from trigger."""
    tagMap = {
        "asrKeyword": "关键词", "eventLaughter": "搞笑",
        "eventApplause": "精彩", "dmDensity": "互动",
        "dmAcceleration": "爆发", "speakerChange": "连麦",
        "rms": "音量", "motion": "动作", "sceneChange": "场景",
    }
    tags: list[str] = []
    for k, v in tagMap.items():
        if k in trigger:
            tags.append(v)
    return tags or ["高光"]


def _getBreakdown(
    start: float, end: float,
    zMatrix: dict[str, np.ndarray],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute score contribution breakdown."""
    s, e = int(start), int(end)
    if s >= e:
        return {}
    contribs = {}
    for name, z in zMatrix.items():
        w = weights.get(name, 0)
        if w > 0 and e <= len(z):
            meanZ = float(np.mean(z[s:e]))
            contribs[name] = round(w * meanZ, 4)
    return dict(sorted(contribs.items(), key=lambda x: -abs(x[1]))[:5])
