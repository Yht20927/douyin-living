# -*- coding: utf-8 -*-
"""Stage 2 — rerank stage-1 candidate windows with window-level features.

The reranker keeps the exact output contract of :func:`src.scorer.score`
(``clips``/``dropped``/``adjustments``/``alignment``) so the controller and
clipper can consume it without changes.  In addition every clip carries
``confidence``, ``reasons``, ``risks`` and the full ``windowFeatures`` object.

Scoring combines six window-level properties:

* ``peakScore``       — strongest moment inside the window;
* ``meanScore``       — sustained activity;
* ``prePostDelta``    — contrast against equal-sized neighbouring windows;
* ``multiSignalAgreement`` — how many modalities fire together;
* ``novelty``         — difference from the preceding window;
* ``durationFit``     — match with the requested duration range.

Semantic events (laughter/applause/keywords/speaker changes) provide a
bounded bonus.  All parameters are plain function arguments; nothing here
mutates controller state or replaces the legacy scorer.
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from src.contracts import atomicWriteJson
from src.log.events import logEvent
from src.log.logger import getLogger
from src.pipeline.boundary_optimizer import (
    inferContentType,
    optimizeBoundaries,
    targetDurationRange,
)
from src.pipeline.stage1_recall import recallCandidates
from src.pipeline.window_features import (
    computeWindowFeatures,
    normalizeFeatureArrays,
    resolveProfileWeights,
)
from src.scorer import alignFeatureArrays
from src.util import validateFeatures

log = getLogger(__name__)

TRIGGER_TITLES = (
    ("asrKeyword", "💬 关键词片段"),
    ("eventLaughter", "😂 搞笑高能"),
    ("eventApplause", "👏 精彩时刻"),
    ("dmDensity", "🔥 弹幕爆发"),
    ("dmAcceleration", "🔥 弹幕爆发"),
    ("speakerChange", "🗣 连麦高光"),
    ("rms", "🔊 音量爆发"),
    ("motion", "📹 画面高能"),
    ("sceneChange", "🎬 场景切换"),
)

TAG_MAP = {
    "asrKeyword": "关键词",
    "eventLaughter": "搞笑",
    "eventApplause": "精彩",
    "dmDensity": "互动",
    "dmAcceleration": "爆发",
    "dmUniqueUsers": "互动",
    "speakerChange": "连麦",
    "topicChange": "话题",
    "rms": "音量",
    "motion": "动作",
    "sceneChange": "场景",
}

SEMANTIC_BOOSTS = {
    "eventLaughter": 0.18,
    "asrKeyword": 0.15,
    "speakerChange": 0.12,
    "eventApplause": 0.12,
    "eventMusic": 0.08,
    "topicChange": 0.06,
    "sceneChange": 0.05,
    "dmSentiment": 0.04,
    "dmUniqueUsers": 0.04,
    "motion": 0.03,
}


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


def _squash(value: float, scale: float) -> float:
    """Map an unbounded positive score into 0..1 without a hard cut-off."""
    if value <= 0 or scale <= 0:
        return 0.0
    return float(1.0 - math.exp(-value / scale))


def _selectWindow(
    curve: np.ndarray,
    start: float,
    end: float,
    minDuration: float,
    maxDuration: float,
    searchRadius: int = 8,
) -> tuple[float, float, bool]:
    """Fit the window to the duration range and nudge it toward the peak.

    The search is intentionally local (±``searchRadius`` seconds) so stage 1
    keeps control of candidate placement while stage 2 fixes obvious
    misalignment caused by a fixed-width window.
    """
    nSecs = len(curve)
    if nSecs <= 0:
        return 0.0, 0.0, False
    if nSecs <= minDuration:
        changed = abs(start) > 0.05 or abs(end - float(nSecs)) > 0.05
        return 0.0, float(nSecs), changed

    originalDuration = max(0.0, end - start)
    target = min(max(originalDuration, minDuration), maxDuration)
    targetInt = max(1, min(nSecs, int(round(target))))

    baseStart = int(round(start))
    maxStart = max(0, nSecs - targetInt)
    baseStart = min(max(0, baseStart), maxStart)

    radius = max(0, int(searchRadius))
    first = max(0, baseStart - radius)
    last = min(maxStart, baseStart + radius)
    if last < first or curve.size != nSecs or not np.any(curve):
        changed = (
            targetInt != int(round(originalDuration))
            or abs(float(baseStart) - start) > 0.05
        )
        return float(baseStart), float(baseStart + targetInt), changed

    starts = np.arange(first, last + 1, dtype=np.int64)
    prefix = np.concatenate(([0.0], np.cumsum(curve, dtype=np.float64)))
    sums = prefix[starts + targetInt] - prefix[starts]
    # Prefer the original placement when two positions are equally strong.
    adjusted = sums - np.abs(starts - baseStart) * 1e-9
    best = int(starts[int(np.argmax(adjusted))])
    changed = (
        targetInt != int(round(originalDuration))
        or abs(float(best) - start) > 0.05
    )
    return float(best), float(best + targetInt), changed


def _titleFor(trigger: str, start: float) -> str:
    for name, title in TRIGGER_TITLES:
        if name in trigger:
            return f"{title} ({int(start)}s)"
    return f"📹 高光时刻 ({int(start)}s)"


def _tagsFor(trigger: str) -> list[str]:
    tags = [tag for name, tag in TAG_MAP.items() if name in trigger]
    # De-duplicate while preserving order.
    return list(dict.fromkeys(tags)) or ["高光"]


def _risksFor(
    window: dict,
    adjustment: bool,
    hasAsrSignal: bool,
) -> list[str]:
    risks: list[str] = []
    if window["multiSignalAgreement"] < 0.3:
        risks.append("low_signal_agreement")
    if window["novelty"] < 0.1:
        risks.append("low_novelty")
    if window["durationFit"] < 0.9:
        risks.append("duration_mismatch")
    if adjustment:
        risks.append("window_adjusted")
    if not hasAsrSignal:
        risks.append("asr_missing")
    if not window["activeSignals"]:
        risks.append("no_active_signal")
    return risks


def _scoreWindow(window: dict, weights: dict[str, float]) -> tuple[float, float, dict]:
    """Return ``(score, confidence, breakdown)`` for one window."""
    peakN = _squash(window["peakScore"], 2.0)
    meanN = _squash(window["meanScore"], 1.5)
    deltaN = _squash(window["prePostDelta"], 1.0)
    noveltyN = _squash(window["novelty"], 2.0)

    base = (
        0.28 * peakN
        + 0.22 * meanN
        + 0.18 * deltaN
        + 0.14 * float(window["multiSignalAgreement"])
        + 0.08 * noveltyN
        + 0.10 * float(window["durationFit"])
    )

    semantic = 0.0
    for name, boost in SEMANTIC_BOOSTS.items():
        stats = window["signals"].get(name)
        if stats is None or weights.get(name, 0.0) <= 0:
            continue
        if stats["peak"] >= 1.0:
            semantic += boost
    semantic = min(0.45, semantic)

    score = base + semantic
    confidence = min(1.0, max(0.0, 0.6 * base + 0.4 * score))

    breakdown = {
        name: stats["contribution"]
        for name, stats in window["signals"].items()
        if stats["weight"] > 0
    }
    breakdown = dict(
        sorted(breakdown.items(), key=lambda item: (-abs(item[1]), item[0]))[:8]
    )
    return round(float(score), 4), round(float(confidence), 4), breakdown


def _suppressOverlaps(
    ranked: list[dict],
    dropped: list[dict],
    overlapRatio: float = 0.6,
) -> list[dict]:
    """Keep the higher-scoring clip when two windows overlap too much."""
    kept: list[dict] = []
    for clip in sorted(ranked, key=lambda item: (-item["score"], item["start"])):
        duplicate = None
        for other in kept:
            overlap = min(clip["end"], other["end"]) - max(clip["start"], other["start"])
            shorter = min(clip["duration"], other["duration"])
            if shorter > 0 and overlap / shorter >= overlapRatio:
                duplicate = other
                break
        if duplicate is None:
            kept.append(clip)
            continue
        dropped.append({
            "start": clip["start"],
            "end": clip["end"],
            "score": clip["score"],
            "candidateSource": "two_stage",
            "reason": "overlap",
            "keptStart": duplicate["start"],
            "keptEnd": duplicate["end"],
        })
    return kept


def rerankCandidates(
    candidates: list[dict],
    roomId: str = "",
    audioFeatures: dict | None = None,
    textFeatures: dict | None = None,
    visualFeatures: dict | None = None,
    durationSec: float | None = None,
    profile: str = "default",
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    topK: int = 20,
    minScore: float = 0.0,
    agreementZ: float = 1.0,
    semanticSignals: Iterable[str] | None = None,
    asrSegments: list[dict] | None = None,
    dynamicDuration: bool = True,
    boundarySearchSec: float = 8.0,
    platform: str = "none",
    runId: str | None = None,
) -> dict:
    """Rerank stage-1 candidates and return a time-table compatible result."""
    startedAt = time.perf_counter()
    if minDuration > maxDuration:
        raise ValueError("minDuration must not exceed maxDuration")

    raw = _collectRawFeatures(audioFeatures, textFeatures, visualFeatures)
    if not raw or not candidates:
        return {
            "clips": [],
            "dropped": [],
            "adjustments": [],
            "alignment": [],
            "stats": {
                "inputCount": len(candidates),
                "outputCount": 0,
                "elapsedMs": int((time.perf_counter() - startedAt) * 1000),
            },
        }

    targetLength = int(math.ceil(durationSec)) if durationSec and durationSec > 0 else None
    aligned, alignment, nSecs = alignFeatureArrays(raw, targetLength)
    if nSecs == 0 or not aligned:
        return {
            "clips": [],
            "dropped": [],
            "adjustments": [],
            "alignment": alignment,
            "stats": {
                "inputCount": len(candidates),
                "outputCount": 0,
                "elapsedMs": int((time.perf_counter() - startedAt) * 1000),
            },
        }

    validateFeatures(aligned, sourceLabel="stage2 rerank input")
    signals = normalizeFeatureArrays(aligned)
    weights = resolveProfileWeights(profile, signals.keys())
    recallCurve = np.zeros(nSecs, dtype=np.float64)
    for name, z in signals.items():
        weight = weights.get(name, 0.0)
        if weight > 0 and len(z) == nSecs:
            recallCurve += weight * z

    ranked: list[dict] = []
    dropped: list[dict] = []
    adjustments: list[dict] = []
    allowedSemantic = set(semanticSignals or SEMANTIC_BOOSTS)
    asrSignal = signals.get("asrKeyword")
    hasAsrSignal = bool(asrSignal is not None and asrSignal.size and np.any(asrSignal))

    def _windowAt(
        windowStart: float,
        windowEnd: float,
        low: float,
        high: float,
        kind: str,
    ) -> dict:
        features = computeWindowFeatures(
            signals,
            windowStart,
            windowEnd,
            weights=weights,
            minDuration=low,
            maxDuration=high,
            agreementZ=agreementZ,
        )
        features["contentType"] = kind
        return features

    for candidate in candidates:
        start = float(candidate.get("start", 0.0))
        end = float(candidate.get("end", 0.0))
        candidateSignals = list(candidate.get("sources") or [])

        contentType = (
            inferContentType(profile, candidateSignals)
            if dynamicDuration
            else "default"
        )
        if dynamicDuration:
            targetMin, targetMax = targetDurationRange(
                contentType, minDuration, maxDuration, platform
            )
        else:
            targetMin, targetMax = minDuration, maxDuration

        refinedStart, refinedEnd, adjusted = _selectWindow(
            recallCurve, start, end, targetMin, targetMax
        )
        if refinedEnd - refinedStart < 1.0:
            dropped.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": float(candidate.get("score", 0.0)),
                "candidateSource": "two_stage",
                "reason": "too_short",
            })
            continue

        window = _windowAt(
            refinedStart, refinedEnd, targetMin, targetMax, contentType
        )

        # Stage-1 sources are a hint; once the provisional window is known,
        # active in-window signals can reveal a more specific content type
        # (e.g. a generic `rms` candidate that is actually a music segment).
        if dynamicDuration:
            refinedType = inferContentType(
                profile, candidateSignals + list(window["activeSignals"])
            )
            if refinedType != contentType:
                contentType = refinedType
                targetMin, targetMax = targetDurationRange(
                    contentType, minDuration, maxDuration, platform
                )
                newStart, newEnd, durationAdjusted = _selectWindow(
                    recallCurve, refinedStart, refinedEnd, targetMin, targetMax
                )
                if newEnd - newStart >= 1.0:
                    adjusted = adjusted or durationAdjusted
                    refinedStart, refinedEnd = newStart, newEnd
                    window = _windowAt(
                        refinedStart, refinedEnd,
                        targetMin, targetMax, contentType,
                    )

        boundaryInfo = {
            "start": refinedStart,
            "end": refinedEnd,
            "duration": round(refinedEnd - refinedStart, 2),
            "changed": False,
            "startSource": "original",
            "endSource": "original",
            "boundaryReasons": [],
            "contentType": contentType,
            "targetDuration": [float(targetMin), float(targetMax)],
        }
        boundaryOriginalStart = refinedStart
        boundaryOriginalEnd = refinedEnd
        durationAdjusted = adjusted
        boundaryShifted = False
        if boundarySearchSec > 0:
            beforeStart, beforeEnd = refinedStart, refinedEnd
            boundaryInfo = optimizeBoundaries(
                refinedStart,
                refinedEnd,
                signals,
                recallCurve,
                asrSegments=asrSegments,
                targetMin=targetMin,
                targetMax=targetMax,
                searchSec=boundarySearchSec,
                contentType=contentType,
                platform=platform,
            )
            if boundaryInfo["changed"]:
                duration = boundaryInfo["duration"]
                if duration < targetMin - 0.5 or duration > targetMax + 0.5:
                    fallbackStart, fallbackEnd, _ = _selectWindow(
                        recallCurve,
                        boundaryInfo["start"],
                        boundaryInfo["end"],
                        targetMin,
                        targetMax,
                    )
                    boundaryInfo["start"] = round(fallbackStart, 2)
                    boundaryInfo["end"] = round(fallbackEnd, 2)
                    boundaryInfo["duration"] = round(fallbackEnd - fallbackStart, 2)
                    boundaryInfo["changed"] = (
                        abs(fallbackStart - beforeStart) >= 0.5
                        or abs(fallbackEnd - beforeEnd) >= 0.5
                    )
                    boundaryInfo["fallback"] = True
                    # The fallback is a duration re-fit, not a semantic
                    # boundary snap: do not attribute it to ASR/scene edges.
                    boundaryInfo["startSource"] = "original"
                    boundaryInfo["endSource"] = "original"
                    boundaryInfo["boundaryReasons"] = []
                    if boundaryInfo["changed"]:
                        refinedStart = boundaryInfo["start"]
                        refinedEnd = boundaryInfo["end"]
                        window = _windowAt(
                            refinedStart, refinedEnd,
                            targetMin, targetMax, contentType,
                        )
                        durationAdjusted = True
                else:
                    refinedStart = boundaryInfo["start"]
                    refinedEnd = boundaryInfo["end"]
                    window = _windowAt(
                        refinedStart, refinedEnd,
                        targetMin, targetMax, contentType,
                    )
                    boundaryShifted = True
        adjusted = durationAdjusted or boundaryShifted

        effectiveWeights = {
            name: weight
            for name, weight in weights.items()
            if name in allowedSemantic or name not in SEMANTIC_BOOSTS
        }
        score, confidence, breakdown = _scoreWindow(window, effectiveWeights)

        if score < minScore:
            dropped.append({
                "start": window["start"],
                "end": window["end"],
                "score": score,
                "candidateSource": "two_stage",
                "reason": "below_min_score",
            })
            continue

        sources = list(dict.fromkeys(
            list(candidate.get("sources") or []) + list(window["activeSignals"])
        ))
        boundaryReasons = list(boundaryInfo.get("boundaryReasons") or [])
        reasons = list(dict.fromkeys(sources + boundaryReasons))
        trigger = "+".join(sources[:5]) if sources else "unknown"
        risks = _risksFor(window, adjusted, hasAsrSignal=hasAsrSignal)
        if boundaryInfo.get("fallback") and boundaryInfo.get("changed"):
            risks.append("boundary_fallback")

        if durationAdjusted and (refinedStart != start or refinedEnd != end):
            adjustments.append({
                "start": window["start"],
                "end": window["end"],
                "originalStart": round(start, 2),
                "originalEnd": round(end, 2),
                "reason": "stage2_duration_fit",
                "contentType": contentType,
                "targetDuration": [round(targetMin, 2), round(targetMax, 2)],
            })

        if boundaryShifted:
            adjustments.append({
                "start": window["start"],
                "end": window["end"],
                "originalStart": round(boundaryOriginalStart, 2),
                "originalEnd": round(boundaryOriginalEnd, 2),
                "reason": "boundary_optimized",
                "boundaryReasons": boundaryReasons,
                "startSource": boundaryInfo.get("startSource"),
                "endSource": boundaryInfo.get("endSource"),
            })

        ranked.append({
            "start": window["start"],
            "end": window["end"],
            "duration": window["duration"],
            "score": score,
            "confidence": confidence,
            "contentType": contentType,
            "targetDuration": [round(targetMin, 2), round(targetMax, 2)],
            "boundary": {
                "startSource": boundaryInfo.get("startSource", "original"),
                "endSource": boundaryInfo.get("endSource", "original"),
                "reasons": boundaryReasons,
            },
            "trigger": trigger,
            "title": _titleFor(trigger, window["start"]),
            "tags": _tagsFor(trigger),
            "candidateSource": "two_stage",
            "sources": sources,
            "reasons": reasons,
            "risks": risks,
            "dropReasons": [],
            "thumbnailTime": round(window["start"] + 3.0, 1),
            "breakdown": breakdown,
            "windowFeatures": window,
            "recallScore": round(float(candidate.get("score", 0.0)), 4),
        })

    ranked = _suppressOverlaps(ranked, dropped)
    ranked.sort(key=lambda item: (-item["score"], item["start"]))

    if topK > 0 and len(ranked) > topK:
        for clip in ranked[topK:]:
            dropped.append({
                "start": clip["start"],
                "end": clip["end"],
                "score": clip["score"],
                "candidateSource": "two_stage",
                "reason": "top_k_cut",
            })
        ranked = ranked[:topK]

    ranked.sort(key=lambda item: (item["start"], item["end"]))
    for index, clip in enumerate(ranked, start=1):
        clip["id"] = index

    contentTypeCounts: dict[str, int] = {}
    boundaryAdjusted = 0
    for clip in ranked:
        kind = clip.get("contentType", "default")
        contentTypeCounts[kind] = contentTypeCounts.get(kind, 0) + 1
        if clip.get("boundary", {}).get("reasons"):
            boundaryAdjusted += 1

    elapsedMs = int((time.perf_counter() - startedAt) * 1000)
    log.info(
        f"Stage 2 rerank: {len(ranked)}/{len(candidates)} candidates kept "
        f"({elapsedMs}ms)"
    )
    logEvent(
        "pipeline.stage",
        runId=runId,
        roomId=roomId,
        step="stage2_rerank",
        status="ok" if ranked else "empty",
        elapsedMs=elapsedMs,
        inputCount=len(candidates),
        outputCount=len(ranked),
        droppedCount=len(dropped),
        topK=topK,
        boundaryAdjusted=boundaryAdjusted,
        contentTypes=contentTypeCounts,
    )
    return {
        "clips": ranked,
        "dropped": dropped,
        "adjustments": adjustments,
        "alignment": alignment,
        "stats": {
            "inputCount": len(candidates),
            "outputCount": len(ranked),
            "droppedCount": len(dropped),
            "topK": topK,
            "boundaryAdjusted": boundaryAdjusted,
            "contentTypes": contentTypeCounts,
            "elapsedMs": elapsedMs,
        },
    }


def _signalStats(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[dict[str, dict], list[str]]:
    stats: dict[str, dict] = {}
    for name, z in signals.items():
        weight = weights.get(name, 0.0)
        stats[name] = {
            "weight": round(float(weight), 4),
            "meanZ": round(float(np.mean(z)), 4) if len(z) else 0.0,
            "maxZ": round(float(np.max(z)), 4) if len(z) else 0.0,
            "meanAbsContribution": (
                round(float(weight * np.mean(np.abs(z))), 4) if weight > 0 and len(z) else 0.0
            ),
        }
    eligible = [name for name, item in stats.items() if item["weight"] > 0]
    topSignals = sorted(
        eligible,
        key=lambda name: stats[name]["meanAbsContribution"],
        reverse=True,
    )[:5]
    return stats, topSignals


def twoStageScore(
    roomId: str = "",
    audioFeatures: dict | None = None,
    textFeatures: dict | None = None,
    visualFeatures: dict | None = None,
    profile: str = "default",
    minDuration: float = 15.0,
    maxDuration: float = 90.0,
    durationSec: float | None = None,
    recallPerHour: float = 60.0,
    topK: int = 20,
    minGapSec: float = 12.0,
    recallThreshold: float = 0.8,
    windowSeconds: float = 30.0,
    minScore: float = 0.0,
    asrSegments: list[dict] | None = None,
    dynamicDuration: bool = True,
    boundarySearchSec: float = 8.0,
    platform: str = "none",
    outputPath: str | None = None,
    scoresPath: str | None = None,
    candidatesPath: str | None = None,
    runId: str | None = None,
) -> dict:
    """Run stage 1 + stage 2 and return the legacy time-table shape."""
    startedAt = time.perf_counter()

    recall = recallCandidates(
        roomId=roomId,
        audioFeatures=audioFeatures,
        textFeatures=textFeatures,
        visualFeatures=visualFeatures,
        durationSec=durationSec,
        profile=profile,
        recallPerHour=recallPerHour,
        minGapSec=minGapSec,
        recallThreshold=recallThreshold,
        windowSeconds=windowSeconds,
        minDuration=minDuration,
        maxDuration=maxDuration,
        outputPath=candidatesPath,
        runId=runId,
    )

    rerank = rerankCandidates(
        recall.get("candidates") or [],
        roomId=roomId,
        audioFeatures=audioFeatures,
        textFeatures=textFeatures,
        visualFeatures=visualFeatures,
        durationSec=durationSec,
        profile=profile,
        minDuration=minDuration,
        maxDuration=maxDuration,
        topK=topK,
        minScore=minScore,
        asrSegments=asrSegments,
        dynamicDuration=dynamicDuration,
        boundarySearchSec=boundarySearchSec,
        platform=platform,
        runId=runId,
    )

    elapsedMs = int((time.perf_counter() - startedAt) * 1000)
    output = {
        "roomId": roomId,
        "duration": recall.get("duration", 0),
        "profile": profile,
        "pipeline": "two_stage",
        "minDuration": minDuration,
        "maxDuration": maxDuration,
        "durationPolicy": {
            "dynamic": dynamicDuration,
            "boundarySearchSec": boundarySearchSec,
            "platform": platform,
        },
        "clips": rerank.get("clips") or [],
        "dropped": rerank.get("dropped") or [],
        "adjustments": rerank.get("adjustments") or [],
        "alignment": rerank.get("alignment") or recall.get("alignment") or [],
        "stages": {
            "recall": dict(recall.get("stats") or {}),
            "rerank": dict(rerank.get("stats") or {}),
            "totalElapsedMs": elapsedMs,
        },
    }

    if outputPath:
        atomicWriteJson(outputPath, output)
        log.info(f"Two-stage time table saved: {outputPath}")

    if scoresPath:
        raw = _collectRawFeatures(audioFeatures, textFeatures, visualFeatures)
        targetLength = (
            int(math.ceil(durationSec)) if durationSec and durationSec > 0 else None
        )
        aligned, alignment, _ = alignFeatureArrays(raw, targetLength) if raw else ({}, [], 0)
        signals = normalizeFeatureArrays(aligned) if aligned else {}
        weights = resolveProfileWeights(profile, signals.keys()) if signals else {}
        signalStats, topSignals = _signalStats(signals, weights)
        scoresDoc = {
            "roomId": roomId,
            "duration": output["duration"],
            "pipeline": "two_stage",
            "scores": recall.get("recallCurve") or [],
            "weights": {name: round(float(weight), 4) for name, weight in weights.items()},
            "topSignals": topSignals,
            "signalStats": signalStats,
            "candidates": [
                {
                    "id": clip["id"],
                    "start": clip["start"],
                    "end": clip["end"],
                    "score": clip["score"],
                    "confidence": clip["confidence"],
                }
                for clip in output["clips"]
            ],
            "stages": output["stages"],
            "alignment": alignment,
        }
        atomicWriteJson(scoresPath, scoresDoc)
        log.info(f"Two-stage scores saved: {scoresPath}")

    logEvent(
        "pipeline.two_stage",
        runId=runId,
        roomId=roomId,
        step="7_scorer",
        status="ok" if output["clips"] else "empty",
        elapsedMs=elapsedMs,
        recallElapsedMs=output["stages"]["recall"].get("elapsedMs"),
        rerankElapsedMs=output["stages"]["rerank"].get("elapsedMs"),
        candidates=len(recall.get("candidates") or []),
        clips=len(output["clips"]),
    )
    log.info(
        f"Two-stage scoring: {len(output['clips'])} clips from "
        f"{len(recall.get('candidates') or [])} candidates ({elapsedMs}ms)"
    )
    return output
