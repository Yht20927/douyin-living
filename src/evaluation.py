# -*- coding: utf-8 -*-
"""Offline evaluation of clip selection against human annotations (S5-6).

The evaluation is deliberately model-free: it takes the ``time_table.json``
produced by either the legacy scorer or the two-stage pipeline and compares it
with a human annotation file.  Matching is one-to-one (a system clip cannot
claim two highlights), so Precision@K and Recall@K remain meaningful even when
several candidates cover the same event.

Annotation schema (JSON object or bare list)::

    {
      "roomId": "123",
      "duration": 3600,
      "profile": "default",
      "annotations": [
        {
          "id": "a1",
          "start": 120.0,
          "end": 150.0,
          "type": "funny",
          "quality": 4,
          "worthClipping": true,
          "notes": "real laugh peak"
        },
        {
          "id": "n1",
          "start": 300.0,
          "end": 320.0,
          "type": "dead_air",
          "quality": 1,
          "worthClipping": false,
          "notes": "should not be clipped"
        }
      ]
    }

``worthClipping`` defaults to true, ``quality`` defaults to 5 for positives and
1 for negatives.  A positive annotation is counted only when
``worthClipping`` is true and ``quality >= minQuality`` (default 3).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.log.logger import getLogger
from src.util import FEATURE_NAMES

log = getLogger(__name__)

DEFAULT_MIN_QUALITY = 3.0
DEFAULT_DUPLICATE_IOU = 0.5

# Initial targets from the Sprint 5 plan.  They are reporting defaults, not
# hard gates: the first real annotated run replaces them with a measured
# baseline.
DEFAULT_TARGETS = {
    "precisionAtK": 0.60,
    "recallAtK": 0.70,
    "boundaryIou": 0.50,
    "invalidRate": 0.20,
    "duplicateRate": 0.10,
}


def _requireNumber(value, field: str, annotationId: str) -> float:
    """Convert a JSON value to float or raise a descriptive ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"annotation {annotationId}: {field} must be a number"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"annotation {annotationId}: {field} must be finite")
    return number


def _requireBool(value, field: str, annotationId: str) -> bool:
    """Parse a boolean that may arrive as JSON bool, 0/1 or a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "是"}:
            return True
        if lowered in {"false", "0", "no", "n", "否"}:
            return False
    raise ValueError(f"annotation {annotationId}: {field} must be a boolean")


def normalizeAnnotations(data) -> dict:
    """Validate and normalize an annotation document.

    Args:
        data: Parsed JSON object or bare list of annotation dicts.

    Returns:
        Dict with roomId, duration, profile, annotations and counts.

    Raises:
        ValueError: on malformed bounds, duplicate ids, or non-finite values.
    """
    if isinstance(data, list):
        metadata: dict = {}
        rawAnnotations = data
    elif isinstance(data, dict):
        metadata = data
        rawAnnotations = data.get("annotations")
        if rawAnnotations is None:
            rawAnnotations = []
    else:
        raise ValueError("annotation JSON must be an object or a list")

    if not isinstance(rawAnnotations, list):
        raise ValueError("'annotations' must be a list")

    annotations: list[dict] = []
    seenIds: set[str] = set()
    for index, raw in enumerate(rawAnnotations):
        if not isinstance(raw, dict):
            raise ValueError(f"annotation #{index + 1} must be an object")
        annotationId = str(raw.get("id") or f"a{index + 1}")
        if annotationId in seenIds:
            raise ValueError(f"duplicate annotation id: {annotationId}")
        seenIds.add(annotationId)

        start = _requireNumber(raw.get("start"), "start", annotationId)
        end = _requireNumber(raw.get("end"), "end", annotationId)
        if start < 0:
            raise ValueError(
                f"annotation {annotationId}: start must be >= 0"
            )
        if end <= start:
            raise ValueError(
                f"annotation {annotationId}: end must be greater than start"
            )

        worthClipping = (
            _requireBool(raw["worthClipping"], "worthClipping", annotationId)
            if "worthClipping" in raw
            else True
        )
        defaultQuality = 5 if worthClipping else 1
        quality = _requireNumber(
            raw.get("quality", defaultQuality), "quality", annotationId
        )
        if not 1 <= quality <= 5:
            raise ValueError(
                f"annotation {annotationId}: quality must be within 1..5"
            )

        annotations.append(
            {
                "id": annotationId,
                "start": round(start, 3),
                "end": round(end, 3),
                "type": str(raw.get("type") or "highlight"),
                "quality": quality,
                "worthClipping": worthClipping,
                "notes": str(raw.get("notes") or ""),
            }
        )

    duration = metadata.get("duration")
    if duration is not None:
        duration = _requireNumber(duration, "duration", "document")
        if duration < 0:
            raise ValueError("document: duration must be >= 0")

    positiveCount = sum(1 for item in annotations if isPositive(item))
    return {
        "roomId": str(metadata.get("roomId") or ""),
        "duration": duration,
        "profile": str(metadata.get("profile") or ""),
        "annotations": annotations,
        "positiveCount": positiveCount,
        "negativeCount": len(annotations) - positiveCount,
    }


def loadAnnotations(path: str | Path) -> dict:
    """Load and normalize an annotation JSON file."""
    source = Path(path)
    try:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read annotations {source}: {exc}") from exc
    try:
        result = normalizeAnnotations(data)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    result["path"] = str(source)
    return result


def loadTimeTable(path: str | Path) -> dict:
    """Load a ``time_table.json`` and validate its top-level shape."""
    source = Path(path)
    try:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read time table {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source}: time table must be a JSON object")

    clips = data.get("clips")
    if clips is None:
        data["clips"] = []
    elif not isinstance(clips, list):
        raise ValueError(f"{source}: 'clips' must be a list")
    return data


def isPositive(annotation: dict, minQuality: float = DEFAULT_MIN_QUALITY) -> bool:
    """Return True when an annotation counts as a clip-worthy highlight."""
    return bool(annotation.get("worthClipping", True)) and float(
        annotation.get("quality", 5)
    ) >= minQuality


def overlapSec(startA: float, endA: float, startB: float, endB: float) -> float:
    """Temporal overlap in seconds."""
    return max(0.0, min(endA, endB) - max(startA, startB))


def clipIou(clipA: dict, clipB: dict) -> float:
    """Temporal intersection-over-union for two windows."""
    startA = float(clipA.get("start", 0.0))
    endA = float(clipA.get("end", startA))
    startB = float(clipB.get("start", 0.0))
    endB = float(clipB.get("end", startB))
    intersection = overlapSec(startA, endA, startB, endB)
    if intersection <= 0:
        return 0.0
    union = (endA - startA) + (endB - startB) - intersection
    return intersection / union if union > 0 else 0.0


def rankClips(clips: list[dict]) -> list[dict]:
    """Sort clips by score desc, then start asc; stable for equal scores."""
    return sorted(
        clips,
        key=lambda clip: (
            -float(clip.get("score", 0.0)),
            float(clip.get("start", 0.0)),
        ),
    )


def _maxMatching(
    topClips: list[dict],
    positives: list[dict],
    matchIou: float,
) -> tuple[list[tuple[int, int, float]], set[int]]:
    """Maximum-cardinality one-to-one matching, preferring higher IoU.

    Returns (matched pairs as (clipIndex, annotationIndex, iou), unmatched
    clip indexes).  The graph is tiny (K is normally <= 20), so Kuhn's
    augmenting-path algorithm is more than sufficient.
    """
    adjacency: list[list[tuple[int, float]]] = []
    for clip in topClips:
        candidates = []
        for annotationIndex, annotation in enumerate(positives):
            iou = clipIou(clip, annotation)
            if iou > 0 and iou >= matchIou:
                candidates.append((annotationIndex, iou))
        candidates.sort(key=lambda item: item[1], reverse=True)
        adjacency.append(candidates)

    annotationToClip = [-1] * len(positives)

    def _augment(clipIndex: int, seen: list[bool]) -> bool:
        for annotationIndex, _iou in adjacency[clipIndex]:
            if seen[annotationIndex]:
                continue
            seen[annotationIndex] = True
            previousClip = annotationToClip[annotationIndex]
            if previousClip == -1 or _augment(previousClip, seen):
                annotationToClip[annotationIndex] = clipIndex
                return True
        return False

    for clipIndex in range(len(topClips)):
        _augment(clipIndex, [False] * len(positives))

    pairs: list[tuple[int, int, float]] = []
    matchedClipIndexes: set[int] = set()
    for annotationIndex, clipIndex in enumerate(annotationToClip):
        if clipIndex != -1:
            pairs.append(
                (
                    clipIndex,
                    annotationIndex,
                    clipIou(topClips[clipIndex], positives[annotationIndex]),
                )
            )
            matchedClipIndexes.add(clipIndex)
    pairs.sort(key=lambda item: item[0])

    unmatched = {
        index for index in range(len(topClips)) if index not in matchedClipIndexes
    }
    return pairs, unmatched


def _duplicatePairs(
    topClips: list[dict],
    duplicateIou: float,
) -> list[dict]:
    duplicates = []
    for index, clip in enumerate(topClips):
        for previousIndex in range(index):
            iou = clipIou(clip, topClips[previousIndex])
            if iou >= duplicateIou:
                duplicates.append(
                    {
                        "rank": index + 1,
                        "clipId": clip.get("id"),
                        "duplicateOfRank": previousIndex + 1,
                        "duplicateOfClipId": topClips[previousIndex].get("id"),
                        "iou": round(iou, 4),
                        "start": clip.get("start"),
                        "end": clip.get("end"),
                    }
                )
                break
    return duplicates


def _negativeHits(
    topClips: list[dict],
    negatives: list[dict],
) -> list[dict]:
    hits = []
    for annotation in negatives:
        ranks = [
            index + 1
            for index, clip in enumerate(topClips)
            if overlapSec(
                float(clip.get("start", 0.0)),
                float(clip.get("end", 0.0)),
                annotation["start"],
                annotation["end"],
            )
            > 0
        ]
        if ranks:
            hits.append(
                {
                    "annotationId": annotation["id"],
                    "start": annotation["start"],
                    "end": annotation["end"],
                    "type": annotation.get("type"),
                    "ranks": ranks,
                }
            )
    return hits


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def computeMetrics(
    clips: list[dict],
    annotations: list[dict],
    k: int = 10,
    matchIou: float = 0.0,
    duplicateIou: float = DEFAULT_DUPLICATE_IOU,
    minQuality: float = DEFAULT_MIN_QUALITY,
) -> dict:
    """Compute Precision@K, Recall@K and boundary metrics.

    Args:
        clips: System clips (time_table ``clips`` list).
        annotations: Normalized annotations from :func:`normalizeAnnotations`.
        k: Rank cut-off; must be >= 1.
        matchIou: Minimum IoU for a clip/annotation match (0 means any
            positive temporal overlap).
        duplicateIou: IoU at or above which a lower-ranked clip is a duplicate.
        minQuality: Minimum quality for a positive annotation.

    Returns:
        Metrics dict plus match/unmatched/duplicate details.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 <= matchIou <= 1.0:
        raise ValueError("matchIou must be within 0..1")
    if not 0.0 <= duplicateIou <= 1.0:
        raise ValueError("duplicateIou must be within 0..1")

    positives = [item for item in annotations if isPositive(item, minQuality)]
    negatives = [item for item in annotations if not isPositive(item, minQuality)]
    if not positives:
        raise ValueError("annotations contain no positive highlights")

    ranked = rankClips(clips)
    topClips = ranked[:k]
    pairs, unmatchedIndexes = _maxMatching(topClips, positives, matchIou)
    duplicates = _duplicatePairs(topClips, duplicateIou)
    negativeHits = _negativeHits(topClips, negatives)

    matchDetails = []
    for clipIndex, annotationIndex, iou in pairs:
        clip = topClips[clipIndex]
        annotation = positives[annotationIndex]
        boundaryCost = abs(
            float(clip.get("start", 0.0)) - annotation["start"]
        ) + abs(float(clip.get("end", 0.0)) - annotation["end"])
        matchDetails.append(
            {
                "rank": clipIndex + 1,
                "clipId": clip.get("id"),
                "clipStart": clip.get("start"),
                "clipEnd": clip.get("end"),
                "annotationId": annotation["id"],
                "annotationStart": annotation["start"],
                "annotationEnd": annotation["end"],
                "type": annotation.get("type"),
                "quality": annotation.get("quality"),
                "iou": round(iou, 4),
                "boundaryCostSec": round(boundaryCost, 3),
            }
        )

    eligible = min(k, len(topClips))
    matchedCount = len(matchDetails)
    precision = matchedCount / eligible if eligible else 0.0
    recall = matchedCount / len(positives)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    iouValues = [float(item["iou"]) for item in matchDetails]
    boundaryCosts = [float(item["boundaryCostSec"]) for item in matchDetails]
    clipCount = len(topClips)

    return {
        "k": k,
        "clipCount": len(clips),
        "topKCount": clipCount,
        "eligibleAtK": eligible,
        "precisionAtK": round(precision, 4),
        "precisionAtKStrict": round(matchedCount / k, 4),
        "recallAtK": round(recall, 4),
        "f1AtK": round(f1, 4),
        "boundaryIou": round(_mean(iouValues), 4) if iouValues else None,
        "editingCostSec": (
            round(_mean(boundaryCosts), 3) if boundaryCosts else None
        ),
        "invalidRate": (
            round(len(unmatchedIndexes) / clipCount, 4) if clipCount else 0.0
        ),
        "duplicateRate": (
            round(len(duplicates) / clipCount, 4) if clipCount else 0.0
        ),
        "negativeHitRate": (
            round(len(negativeHits) / len(negatives), 4) if negatives else None
        ),
        "matchedCount": matchedCount,
        "positiveAnnotations": len(positives),
        "negativeAnnotations": len(negatives),
        "unmatchedCount": len(unmatchedIndexes),
        "duplicateCount": len(duplicates),
        "negativeHitCount": len(negativeHits),
        "iouSum": round(sum(iouValues), 6),
        "boundaryCostSum": round(sum(boundaryCosts), 6),
        "matches": matchDetails,
        "unmatchedClips": [
            {
                "rank": index + 1,
                "clipId": topClips[index].get("id"),
                "start": topClips[index].get("start"),
                "end": topClips[index].get("end"),
                "score": topClips[index].get("score"),
            }
            for index in sorted(unmatchedIndexes)
        ],
        "duplicates": duplicates,
        "negativeHits": negativeHits,
    }


def checkTargets(
    metrics: dict,
    targets: dict | None = None,
) -> dict:
    """Compare metrics with the provisional Sprint 5 targets."""
    targets = targets or DEFAULT_TARGETS
    checks = {}
    for name, target in targets.items():
        value = metrics.get(name)
        if value is None:
            checks[name] = {"target": target, "value": None, "pass": None}
            continue
        if name in {"invalidRate", "duplicateRate"}:
            passed = float(value) <= target
        else:
            passed = float(value) >= target
        checks[name] = {
            "target": target,
            "value": value,
            "pass": passed,
        }
    return checks


def aggregateMetrics(caseMetrics: list[dict], k: int, targets: dict | None = None) -> dict:
    """Aggregate per-case metrics as both macro and micro averages.

    Args:
        caseMetrics: Metrics dicts (all evaluated at the same ``k``).
        k: Rank cut-off used for the evaluation.
        targets: Optional target overrides passed to :func:`checkTargets`.
    """
    if not caseMetrics:
        raise ValueError("no case metrics to aggregate")

    def _macro(name: str) -> float | None:
        return _mean(
            [
                float(metrics[name])
                for metrics in caseMetrics
                if metrics.get(name) is not None
            ]
        )

    totalEligible = sum(int(item["eligibleAtK"]) for item in caseMetrics)
    totalPositives = sum(int(item["positiveAnnotations"]) for item in caseMetrics)
    totalClips = sum(int(item["topKCount"]) for item in caseMetrics)
    totalMatched = sum(int(item["matchedCount"]) for item in caseMetrics)
    totalUnmatched = sum(int(item["unmatchedCount"]) for item in caseMetrics)
    totalDuplicates = sum(int(item["duplicateCount"]) for item in caseMetrics)
    totalNegatives = sum(int(item["negativeAnnotations"]) for item in caseMetrics)
    totalNegativeHits = sum(int(item["negativeHitCount"]) for item in caseMetrics)
    totalIou = sum(float(item["iouSum"]) for item in caseMetrics)
    totalBoundaryCost = sum(
        float(item["boundaryCostSum"]) for item in caseMetrics
    )

    microPrecision = totalMatched / totalEligible if totalEligible else 0.0
    microRecall = totalMatched / totalPositives if totalPositives else 0.0
    microF1 = (
        2 * microPrecision * microRecall / (microPrecision + microRecall)
        if microPrecision + microRecall > 0
        else 0.0
    )

    macro = {
        name: (
            round(value, 4) if (value := _macro(name)) is not None else None
        )
        for name in (
            "precisionAtK",
            "precisionAtKStrict",
            "recallAtK",
            "f1AtK",
            "boundaryIou",
            "editingCostSec",
            "invalidRate",
            "duplicateRate",
            "negativeHitRate",
        )
    }
    micro = {
        "precisionAtK": round(microPrecision, 4),
        "precisionAtKStrict": round(
            totalMatched / (k * len(caseMetrics)) if caseMetrics else 0.0, 4
        ),
        "recallAtK": round(microRecall, 4),
        "f1AtK": round(microF1, 4),
        "boundaryIou": (
            round(totalIou / totalMatched, 4) if totalMatched else None
        ),
        "editingCostSec": (
            round(totalBoundaryCost / totalMatched, 3) if totalMatched else None
        ),
        "invalidRate": (
            round(totalUnmatched / totalClips, 4) if totalClips else 0.0
        ),
        "duplicateRate": (
            round(totalDuplicates / totalClips, 4) if totalClips else 0.0
        ),
        "negativeHitRate": (
            round(totalNegativeHits / totalNegatives, 4)
            if totalNegatives
            else None
        ),
    }
    return {
        "k": k,
        "cases": len(caseMetrics),
        "macro": macro,
        "micro": micro,
        "macroTargets": checkTargets(macro, targets),
        "microTargets": checkTargets(micro, targets),
    }


def objectiveValue(metrics: dict, objective: str = "f1") -> float:
    """One scalar for weight search; higher is always better."""
    if objective in {"f1", "precision", "recall"}:
        return float(metrics.get(f"{objective}AtK", 0.0))
    if objective == "iou":
        return float(metrics.get("boundaryIou") or 0.0)
    if objective == "editing":
        value = metrics.get("editingCostSec")
        return -float(value) if value is not None else -1e9
    raise ValueError(
        "objective must be one of f1, precision, recall, iou, editing"
    )


def normalizeWeights(weights: dict[str, float]) -> dict[str, float]:
    """Keep known non-negative features and normalize to sum 1."""
    cleaned = {
        name: max(0.0, float(value))
        for name, value in weights.items()
        if name in FEATURE_NAMES
    }
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("weight vector must contain at least one positive value")
    return {name: value / total for name, value in cleaned.items() if value > 0}
