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
import math
import numpy as np
from scipy import signal as sp_signal, stats as sp_stats
from src.log.logger import getLogger
from src.util import validateFeatures
from src.config import load_settings

log = getLogger(__name__)

# Binary/event-like signals must be padded with zeros when a shorter
# artifact is aligned to the recording duration.  Interpolating them would
# invent events that never happened.
EVENT_FEATURES = frozenset({
    "sceneChange",
    "asrKeyword",
    "speakerChange",
    "topicChange",
    "eventLaughter",
    "eventApplause",
    "eventMusic",
    "faceCount",
})

# Load scorer configuration (cached; call load_settings.cache_clear() to reload)
_cfg = load_settings().scorer

# Re-export profiles for backward compatibility with existing tests
PROFILES = _cfg.profiles
DETECTION_WINDOWS = _cfg.detection.windows
NMS_MERGE_DISTANCE = _cfg.detection.nms_merge_distance
MIN_CLIP_DURATION = _cfg.detection.min_clip_duration
MAX_CLIP_DURATION = _cfg.detection.max_clip_duration


# ── Timeline alignment ─────────────────────────────────────────

def alignFeatureArrays(
    features: dict[str, np.ndarray],
    targetLength: int | None = None,
) -> tuple[dict[str, np.ndarray], list[dict], int]:
    """Align every feature array to one common 1 Hz timeline.

    Short continuous signals are linearly interpolated (edge values are
    preserved at both ends); event-like signals are zero-padded.  Long
    signals are truncated.  No signal is ever silently dropped because of a
    length mismatch.

    Returns ``(aligned, report, targetLength)``.
    """
    if not features:
        return {}, [], 0

    lengths = [len(np.asarray(v).ravel()) for v in features.values()]
    target = int(targetLength) if targetLength and targetLength > 0 else max(lengths)
    target = max(target, 0)
    if target == 0:
        return {}, [], 0

    aligned: dict[str, np.ndarray] = {}
    report: list[dict] = []

    for name, values in features.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        oldLen = len(arr)
        if oldLen == 0:
            continue

        if oldLen == target:
            aligned[name] = arr
            continue

        if oldLen > target:
            aligned[name] = arr[:target]
            method = "truncate"
        elif name in EVENT_FEATURES:
            aligned[name] = np.concatenate([arr, np.zeros(target - oldLen)])
            method = "pad_zeros"
        else:
            # np.interp keeps the first/last sample as edge values, then
            # linearly fills the interior — the least surprising behaviour
            # for continuous signals such as RMS or danmaku density.
            sourceX = np.arange(oldLen, dtype=np.float64)
            targetX = np.linspace(0.0, oldLen - 1.0, target)
            aligned[name] = np.interp(targetX, sourceX, arr)
            method = "interpolate"

        report.append({
            "name": name,
            "oldLen": oldLen,
            "newLen": target,
            "method": method,
        })

    if report:
        summary = ", ".join(
            f"{item['name']}:{item['oldLen']}→{item['newLen']}({item['method']})"
            for item in report
        )
        log.warning(f"Aligned features to {target}s: {summary}")

    return aligned, report, target


def score(
    roomId: str = "",
    audioFeatures: dict | None = None,
    textFeatures: dict | None = None,
    visualFeatures: dict | None = None,
    profile: str = "default",
    sensitivity: float = 2.5,
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    durationSec: float | None = None,
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
        durationSec: Authoritative recording duration.  When provided, every
            feature array is aligned to this 1 Hz timeline.  Otherwise the
            longest feature array defines it.
        outputPath: Save time_table.json.
        scoresPath: Save per-second scores.json.

    Returns:
        dict with 'clips' list.
    """
    af = audioFeatures or {"features": {}}
    tf = textFeatures or {"features": {}}
    vf = visualFeatures or {"features": {}}

    rawFeatures: dict[str, np.ndarray] = {}
    for src in [af, tf, vf]:
        for k, v in src.get("features", {}).items():
            if isinstance(v, list) and len(v) > 0:
                rawFeatures[k] = np.array(v, dtype=np.float64)

    if not rawFeatures:
        log.error("No features found")
        return {"clips": []}

    targetLength = int(math.ceil(durationSec)) if durationSec and durationSec > 0 else None
    allFeatures, alignment, nSecs = alignFeatureArrays(rawFeatures, targetLength)
    if nSecs == 0 or not allFeatures:
        log.error("No usable features after alignment")
        return {"clips": []}

    # Validate feature names against canonical set (catch silent signal loss)
    validateFeatures(allFeatures, sourceLabel="scorer input")

    log.debug(f"Feature matrix: {nSecs}s × {len(allFeatures)} signals")

    # ── Step 1: Preprocess ────────────────────────────────────
    zMatrix = _preprocess(allFeatures)

    # ── Step 2: Time-lag compensation (cross-correlation) ──────
    zMatrix = _compensateLag(zMatrix)

    # ── Score ─────────────────────────────────────────────────
    if profile not in PROFILES:
        log.warning(f"Unknown profile '{profile}' — falling back to default")
    weights = PROFILES.get(profile, PROFILES["default"])
    scoreBase = np.zeros(nSecs)
    for name, z in zMatrix.items():
        w = weights.get(name, 0.0)
        if w > 0:
            scoreBase += w * z

    # Boost (tanh of signal change)
    boost = np.zeros(nSecs)
    for name, z in zMatrix.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        diff = np.abs(np.diff(z, prepend=z[0]))
        boost += w * np.tanh(diff / _cfg.boost.tanh_divisor)

    # Time decay: exp(-|t - candidate| / τ)
    # Applied per-candidate in _detectPeaks, but also apply globally:
    decayGlobal = np.exp(-np.arange(nSecs) / max(nSecs, 1) * _cfg.global_decay.coeff)
    finalScore = (scoreBase + boost) * decayGlobal
    finalScore = np.nan_to_num(finalScore, nan=0.0)

    # ── Score full timeline (for scores.json) ─────────────────
    scoresTimeline = finalScore.tolist()

    # ── Peak detection ────────────────────────────────────────
    drops: list[dict] = []
    adjustments: list[dict] = []
    clips = _detectPeaks(finalScore, sensitivity)
    if not clips:
        log.info(
            "No peaks above adaptive threshold "
            f"(sensitivity={sensitivity:.2f}, maxScore={float(np.max(finalScore)):.3f}); "
            "consider --two-stage or a lower --sensitivity"
        )

    # ── Boundary optimization ──────────────────────────────────
    clips = _optimizeBoundaries(clips, nSecs, zMatrix)

    # ── Candidate window expansion / trimming ──────────────────
    clips = _expandCandidates(clips, nSecs, minDuration, maxDuration, finalScore)

    # ── NMS (merged windows are re-trimmed to maxDuration) ─────
    clips = _nms(
        clips,
        maxDur=maxDuration,
        score=finalScore,
        dropped=drops,
    )

    # ── Quality checks: prune before dropping ──────────────────
    clips = _qualityFilter(
        clips, minDuration, maxDuration, finalScore, nSecs,
        dropped=drops,
        adjustments=adjustments,
    )

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
            "candidateSource": "peak",
            "dropReasons": [],
            "thumbnailTime": round(start + _cfg.thumbnail.offset_sec, 1),
            "breakdown": _getBreakdown(start, end, zMatrix, weights),
        })

    output = {
        "roomId": roomId,
        "duration": nSecs,
        "profile": profile,
        "sensitivity": sensitivity,
        "minDuration": minDuration,
        "maxDuration": maxDuration,
        "clips": resultClips,
        "dropped": drops,
        "adjustments": adjustments,
        "alignment": alignment,
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        log.info(f"Time table saved: {outputPath}")

    if scoresPath:
        weightsUsed = {name: w for name, w in weights.items() if w > 0}
        signalStats = {}
        for name, z in zMatrix.items():
            w = weights.get(name, 0.0)
            signalStats[name] = {
                "weight": round(float(w), 4),
                "meanZ": round(float(np.mean(z)), 4),
                "maxZ": round(float(np.max(z)), 4),
                "meanAbsContribution": (
                    round(float(w * np.mean(np.abs(z))), 4) if w > 0 else 0.0
                ),
            }
        eligibleSignals = [
            name for name, stats in signalStats.items() if stats["weight"] > 0
        ]
        topSignals = sorted(
            eligibleSignals,
            key=lambda n: signalStats[n]["meanAbsContribution"],
            reverse=True,
        )[:_cfg.breakdown.top_n]
        scoresDoc = {
            "roomId": roomId,
            "duration": nSecs,
            "scores": scoresTimeline,
            "weights": weightsUsed,
            "topSignals": topSignals,
            "signalStats": signalStats,
            "alignment": alignment,
        }
        with open(scoresPath, "w", encoding="utf-8") as f:
            json.dump(scoresDoc, f, ensure_ascii=False, indent=2)
        log.info(f"Scores saved: {scoresPath}")

    log.info(
        f"Scored: {len(resultClips)} clips "
        f"({len(drops)} dropped, {len(adjustments)} adjusted)"
    )
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
        alpha = _cfg.preprocessing.ewma_alpha
        smoothed = np.empty_like(arr)
        smoothed[0] = arr[0]
        for t in range(1, n):
            smoothed[t] = alpha * arr[t] + (1.0 - alpha) * smoothed[t - 1]

        # 2. Rolling Z-Score: (x - μ) / σ over configurable window (vectorized)
        # For short recordings fall back to global Z-Score so
        # clips are still detectable instead of returning all zeros.
        window = min(_cfg.preprocessing.zscore_window_sec, n)
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
            eps = _cfg.preprocessing.zscore_eps
            with np.errstate(divide="ignore", invalid="ignore"):
                zScore = np.divide(
                    smoothed - rollMean, rollStd,
                    where=rollStd > eps, out=np.zeros(n),
                )
        else:
            zScore = np.zeros(n)

        # 3. Median filter (vectorized)
        median_size = _cfg.preprocessing.median_filter_size
        medianF = median_filter(zScore, size=median_size, mode="nearest")

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
    if minLen <= _cfg.lag_compensation.min_len:
        return zMatrix

    try:
        corr = sp_signal.correlate(dmSig[:minLen], auSig[:minLen])
        lag = int(np.argmax(corr) - (minLen - 1))
        if not (0 < abs(lag) <= _cfg.lag_compensation.max_shift):
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

    wf = _cfg.detection.window_factor
    persistence = _cfg.detection.peak_persistence_frames
    gap = _cfg.detection.candidate_gap_sec

    for window in DETECTION_WINDOWS:
        if wf * window >= n:
            continue  # window too large for this recording

        # Rolling mean and std via uniform filter (boxcar window)
        # size = wf*window for symmetric ±window neighborhood
        localMean = uniform_filter1d(score, size=wf * window, mode="nearest")
        localMeanSq = uniform_filter1d(score ** 2, size=wf * window, mode="nearest")
        localStd = np.sqrt(np.maximum(localMeanSq - localMean ** 2, 0.0))

        threshold = localMean + k * localStd

        # Element-wise: is score[t] > threshold[t]?
        above = score > threshold

        # Persistence: consecutive seconds must ALL exceed threshold
        persistent = above.copy()
        for shift in range(1, persistence):
            persistent[:-(shift)] &= score[shift:] > threshold[:-(shift)]
        persistent[-(persistence - 1):] = False

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
        if t - prev > gap:
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
    pad = _cfg.boundary.pad_sec
    search = _cfg.boundary.valley_search_range
    vw = _cfg.boundary.valley_window_radius

    for s, e, sc in clips:
        newStart = max(0, s - pad)
        newEnd = min(nSecs, e + pad)

        # Search for valleys (quiet moments) near boundaries.
        # Use RMS Z-Score: a true valley sits below the local mean over ±vw.
        rms = zMatrix.get("rms")
        rmsLen = len(rms) if rms is not None else 0
        if rmsLen > 0:
            # Find valley before peak
            for offset in range(search, 0, -1):
                t = int(s) - offset
                if 0 <= t < rmsLen:
                    window = rms[max(0, t - vw):min(rmsLen, t + vw)]
                    if window.size > 0 and rms[t] < float(np.mean(window)):
                        newStart = t
                        break

            # Find valley after peak
            for offset in range(search):
                t = int(e) + offset
                if 0 <= t < rmsLen:
                    window = rms[max(0, t - vw):min(rmsLen, t + vw)]
                    if window.size > 0 and rms[t] < float(np.mean(window)):
                        newEnd = t
                        break

        localBest = (max(0, newStart), min(nSecs, newEnd), sc)
        optimized.append(localBest)
    return optimized


def _clampWindow(
    start: float, end: float, nSecs: int,
) -> tuple[int, int]:
    s = max(0, min(int(round(start)), nSecs))
    e = max(0, min(int(round(end)), nSecs))
    if e < s:
        s, e = e, s
    return s, e


def _bestSubwindow(
    score: np.ndarray,
    start: int,
    end: int,
    maxDur: float,
) -> tuple[int, int]:
    """Return the highest-energy sub-window of at most ``maxDur`` seconds."""
    window = int(round(maxDur))
    start = max(0, start)
    end = min(len(score), end)
    if window <= 0 or end - start <= window:
        return start, end

    segment = score[start:end]
    if len(segment) <= window:
        return start, end

    cumulative = np.cumsum(np.insert(segment, 0, 0.0))
    windowSums = cumulative[window:] - cumulative[:-window]
    bestOffset = int(np.argmax(windowSums))
    return start + bestOffset, start + bestOffset + window


def _expandCandidates(
    clips: list[tuple[float, float, float]],
    nSecs: int,
    minDur: float,
    maxDur: float,
    score: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Expand short peak ranges to publishable windows, trim long ones.

    Peak detection intentionally returns narrow ranges (often 0–3 s).  The
    old pipeline expanded them by a fixed ±5 s and then dropped anything
    below ``minDur`` — which made "candidate found" and "clip produced"
    diverge.  Expansion is now duration-aware and centred on the peak.
    """
    expanded: list[tuple[float, float, float]] = []
    for s, e, sc in clips:
        s, e = _clampWindow(s, e, nSecs)
        if nSecs <= 0:
            continue
        if e <= s:
            e = min(nSecs, s + 1)

        duration = e - s
        if duration < minDur:
            if nSecs <= minDur:
                s, e = 0, nSecs
            else:
                centre = (s + e) / 2.0
                target = int(math.ceil(minDur))
                newStart = int(round(centre - target / 2.0))
                newStart = max(0, min(newStart, nSecs - target))
                s, e = newStart, newStart + target

        if e - s > maxDur:
            s, e = _bestSubwindow(score, s, e, maxDur)

        expanded.append((float(s), float(e), float(sc)))
    return expanded


def _nms(
    clips: list[tuple[float, float, float]],
    mergeDistance: float | None = None,
    maxDur: float | None = None,
    score: np.ndarray | None = None,
    dropped: list[dict] | None = None,
) -> list[tuple[float, float, float]]:
    """Non-maximum suppression with optional post-merge duration trimming."""
    if not clips:
        return []

    distance = NMS_MERGE_DISTANCE if mergeDistance is None else mergeDistance
    sortedClips = sorted(clips, key=lambda x: x[0])
    merged = [sortedClips[0]]
    for clip in sortedClips[1:]:
        last = merged[-1]
        if clip[0] - last[1] < distance:
            if dropped is not None:
                dropped.append({
                    "start": round(float(clip[0]), 1),
                    "end": round(float(clip[1]), 1),
                    "score": round(float(clip[2]), 3),
                    "candidateSource": "peak",
                    "reason": "duplicate",
                })
            merged[-1] = (
                min(last[0], clip[0]),
                max(last[1], clip[1]),
                max(last[2], clip[2]),
            )
        else:
            merged.append(clip)

    if maxDur is not None and score is not None:
        trimmed: list[tuple[float, float, float]] = []
        for s, e, sc in merged:
            if e - s > maxDur:
                s, e = _bestSubwindow(score, int(s), int(e), maxDur)
            trimmed.append((float(s), float(e), float(sc)))
        merged = trimmed

    return merged


def _qualityFilter(
    clips: list[tuple[float, float, float]],
    minDur: float,
    maxDur: float,
    score: np.ndarray,
    nSecs: int,
    dropped: list[dict] | None = None,
    adjustments: list[dict] | None = None,
) -> list[tuple[float, float, float]]:
    """Prune-to-fit quality filter.

    Duration violations are repaired where possible:
      * too short → expand around the peak (or keep the whole recording when
        the source itself is shorter than ``minDur``);
      * too long  → keep the highest-energy sub-window;
      * t-test failure → keep the clip but lower its score, recording the
        reason instead of silently dropping a real highlight.
    """
    filtered: list[tuple[float, float, float]] = []

    for s, e, sc in clips:
        s, e = _clampWindow(s, e, nSecs)
        if e <= s:
            if dropped is not None:
                dropped.append({
                    "start": float(s),
                    "end": float(e),
                    "score": round(float(sc), 3),
                    "candidateSource": "peak",
                    "reason": "too_short",
                })
            continue

        if e - s > maxDur:
            original = (s, e)
            s, e = _bestSubwindow(score, s, e, maxDur)
            if adjustments is not None:
                adjustments.append({
                    "start": round(float(s), 1),
                    "end": round(float(e), 1),
                    "originalStart": original[0],
                    "originalEnd": original[1],
                    "reason": "trimmed_too_long",
                })

        if e - s < minDur:
            if nSecs > 0 and nSecs <= minDur:
                # Source is shorter than the requested minimum — keep the
                # whole recording and make the limitation explicit.
                s, e = 0, nSecs
                if adjustments is not None:
                    adjustments.append({
                        "start": 0.0,
                        "end": float(nSecs),
                        "reason": "insufficient_context",
                    })
            else:
                centre = (s + e) / 2.0
                target = int(math.ceil(minDur))
                newStart = max(0, min(int(round(centre - target / 2.0)), nSecs - target))
                s, e = newStart, newStart + target

            if e - s < minDur:
                if dropped is not None:
                    dropped.append({
                        "start": float(s),
                        "end": float(e),
                        "score": round(float(sc), 3),
                        "candidateSource": "peak",
                        "reason": "too_short",
                    })
                continue

        # t-test: scores in clip vs scores immediately before.  A failed
        # test is evidence, not proof — downgrade the score and keep it.
        clipScores = score[int(s):int(e)]
        preStart = max(0, int(s) - _cfg.t_test.pre_window_sec)
        preScores = score[preStart:int(s)]
        if len(preScores) > 2 and len(clipScores) > 2:
            try:
                _, p = sp_stats.ttest_ind(clipScores, preScores)
                if p > _cfg.t_test.pvalue_threshold:
                    sc = float(sc) * 0.7
                    if adjustments is not None:
                        adjustments.append({
                            "start": round(float(s), 1),
                            "end": round(float(e), 1),
                            "reason": "t_test_insignificant",
                            "applied": "score_penalty",
                            "pvalue": round(float(p), 4),
                        })
            except Exception:
                log.debug(f"t-test failed for clip {s}-{e}", exc_info=True)

        filtered.append((float(s), float(e), float(sc)))

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
        if peak > _cfg.detection.trigger_zscore_threshold:
            if name == "asrKeyword":
                # Find which keyword triggered
                asr = features.get("asrKeyword")
                if asr is not None and len(asr) > s:
                    parts.append("asrKeyword")
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
    return dict(sorted(contribs.items(), key=lambda x: -abs(x[1]))[:_cfg.breakdown.top_n])
