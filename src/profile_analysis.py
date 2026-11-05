# -*- coding: utf-8 -*-
"""Profile support analysis for scorer weight profiles (S5-7).

Profiles are only meaningful when annotated cases exist for their domain and
the domain-specific weights actually beat the default on those cases.  This
module re-scores cached feature artifacts under every profile, compares the
result against human annotations, and reports a support level and a
recommendation.  It never edits ``config/scorer.yaml`` and never downloads or
runs a model.

Typical use is through ``scripts/profile_report.py``; the functions below are
kept importable so tuning and reporting share exactly one loading path.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

from src import scorer
from src.evaluation import (
    aggregateMetrics,
    computeMetrics,
    isPositive,
    loadAnnotations,
    loadTimeTable,
)

FEATURE_FILES = {
    "audio": "audio_features.json",
    "text": "text_features.json",
    "visual": "visual_features.json",
}


class _SilentLogger:
    """Swallow per-candidate scorer INFO lines during batch analysis."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


@contextmanager
def suppressScorerLog():
    """Temporarily silence ``src.scorer`` logging, restoring it afterwards."""
    original = scorer.log
    scorer.log = _SilentLogger()
    try:
        yield
    finally:
        scorer.log = original


def loadFeatureFile(path: Path) -> dict:
    """Load one feature artifact and keep only non-empty arrays."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read features {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: feature file must be a JSON object")
    features = data.get("features")
    if features is None:
        return {}
    if not isinstance(features, dict):
        raise ValueError(f"{path}: 'features' must be an object")
    return {
        name: values
        for name, values in features.items()
        if isinstance(values, list) and values
    }


def loadCase(timeTablePath: str, annotationPath: str) -> dict:
    """Load a time table, annotations and the cached three-way features."""
    table = loadTimeTable(timeTablePath)
    annotations = loadAnnotations(annotationPath)
    roomDir = Path(timeTablePath).resolve().parent

    featureSets = {
        kind: loadFeatureFile(roomDir / filename)
        for kind, filename in FEATURE_FILES.items()
    }
    if not any(featureSets.values()):
        raise ValueError(
            f"{roomDir}: no non-empty feature files "
            f"({', '.join(FEATURE_FILES.values())})"
        )

    duration = table.get("duration") or annotations.get("duration")
    return {
        "label": roomDir.name or Path(timeTablePath).stem,
        "roomId": str(table.get("roomId") or annotations.get("roomId") or ""),
        "profile": str(
            annotations.get("profile") or table.get("profile") or "default"
        ),
        "audio": featureSets["audio"],
        "text": featureSets["text"],
        "visual": featureSets["visual"],
        "annotations": annotations["annotations"],
        "positiveAnnotations": annotations["positiveCount"],
        "negativeAnnotations": annotations["negativeCount"],
        "durationSec": float(duration) if duration else None,
        "sensitivity": float(table.get("sensitivity", 2.5)),
        "minDuration": float(table.get("minDuration", 15.0)),
        "maxDuration": float(table.get("maxDuration", 90.0)),
    }


def scoreCase(case: dict, profile: str) -> dict:
    """Run the scorer against one case's cached features."""
    return scorer.score(
        roomId=case["roomId"],
        audioFeatures=(
            {"features": case["audio"]} if case["audio"] else None
        ),
        textFeatures=(
            {"features": case["text"]} if case["text"] else None
        ),
        visualFeatures=(
            {"features": case["visual"]} if case["visual"] else None
        ),
        profile=profile,
        sensitivity=case["sensitivity"],
        minDuration=case["minDuration"],
        maxDuration=case["maxDuration"],
        durationSec=case["durationSec"],
    )


def casePositiveCount(case: dict, minQuality: float = 3.0) -> int:
    """Count annotations that qualify as positives at ``minQuality``."""
    return sum(
        1 for item in case["annotations"] if isPositive(item, minQuality)
    )


def profileDistance(left: dict[str, float], right: dict[str, float]) -> dict:
    """Compare two non-negative weight vectors.

    ``totalVariation`` is ``0.5 * L1`` and lies in ``0..1`` for vectors that
    sum to 1 (0 means identical, 1 means disjoint support).  ``cosine`` is the
    usual cosine similarity.
    """
    names = set(left) | set(right)
    l1 = sum(
        abs(float(left.get(name, 0.0)) - float(right.get(name, 0.0)))
        for name in names
    )
    dot = sum(
        float(left.get(name, 0.0)) * float(right.get(name, 0.0))
        for name in names
    )
    normLeft = math.sqrt(
        sum(float(left.get(name, 0.0)) ** 2 for name in names)
    )
    normRight = math.sqrt(
        sum(float(right.get(name, 0.0)) ** 2 for name in names)
    )
    cosine = dot / (normLeft * normRight) if normLeft and normRight else 0.0
    return {
        "totalVariation": round(l1 / 2, 6),
        "cosine": round(cosine, 6),
    }


def _topFeatures(weights: dict[str, float], limit: int = 5) -> list[dict]:
    ranked = sorted(
        ((name, float(value)) for name, value in weights.items() if value > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {"name": name, "weight": round(value, 6)}
        for name, value in ranked[:limit]
    ]


def _metricSummary(metrics: dict) -> dict:
    return {
        "precisionAtK": metrics.get("precisionAtK"),
        "recallAtK": metrics.get("recallAtK"),
        "f1AtK": metrics.get("f1AtK"),
        "boundaryIou": metrics.get("boundaryIou"),
        "editingCostSec": metrics.get("editingCostSec"),
        "invalidRate": metrics.get("invalidRate"),
        "duplicateRate": metrics.get("duplicateRate"),
        "negativeHitRate": metrics.get("negativeHitRate"),
        "clipCount": metrics.get("clipCount"),
        "matchedCount": metrics.get("matchedCount"),
    }


def compareProfiles(
    cases: list[dict],
    profiles: list[str] | None = None,
    k: int = 5,
    minCases: int = 2,
    minPositives: int = 5,
    minDelta: float = 0.02,
    matchIou: float = 0.0,
    duplicateIou: float = 0.5,
    minQuality: float = 3.0,
    targets: dict | None = None,
) -> dict:
    """Evaluate every profile on every case and summarize support.

    A profile is considered to have enough evidence when it owns at least
    ``minCases`` annotated cases and those cases contain at least
    ``minPositives`` positives.  Its recommendation is ``keep`` only when the
    evidence is sufficient and its micro F1 is at least ``minDelta`` above the
    default profile on the same owned cases.  Otherwise the report recommends
    collecting more data or merging the profile into ``default``; nothing is
    changed automatically.

    Args:
        cases: Output of :func:`loadCase`; labels must be unique.
        profiles: Profile names to compare (default: every scorer profile).
        k: Rank cut-off for Precision@K / Recall@K.
        minCases: Minimum owned annotated cases for a profile.
        minPositives: Minimum owned positive annotations for a profile.
        minDelta: Required micro-F1 gain over default to keep a profile.
        matchIou, duplicateIou, minQuality: Evaluation parameters.
        targets: Optional target overrides for macro/micro checks.

    Returns:
        Dict with per-profile reports, per-case diagnostics, weight distances
        and a keep/merge summary.
    """
    if not cases:
        raise ValueError("no cases to compare")
    if k < 1:
        raise ValueError("k must be >= 1")
    if minCases < 1:
        raise ValueError("minCases must be >= 1")
    if minPositives < 1:
        raise ValueError("minPositives must be >= 1")
    if not 0.0 <= matchIou <= 1.0:
        raise ValueError("matchIou must be within 0..1")
    if not 0.0 <= duplicateIou <= 1.0:
        raise ValueError("duplicateIou must be within 0..1")
    if not 1.0 <= minQuality <= 5.0:
        raise ValueError("minQuality must be within 1..5")

    labels = [case["label"] for case in cases]
    if len(set(labels)) != len(labels):
        raise ValueError("case labels must be unique")

    profileNames = list(dict.fromkeys(profiles or scorer.PROFILES))
    if not profileNames:
        raise ValueError("no profiles to compare")
    unknown = [name for name in profileNames if name not in scorer.PROFILES]
    if unknown:
        raise ValueError(
            f"unknown profile(s): {', '.join(sorted(unknown))} "
            f"(available: {', '.join(sorted(scorer.PROFILES))})"
        )

    for case in cases:
        if casePositiveCount(case, minQuality) == 0:
            raise ValueError(
                f"case '{case['label']}' has no positive annotations "
                f"at minQuality={minQuality}"
            )

    caseResults: dict[str, dict[str, dict]] = {
        name: {} for name in profileNames
    }
    with suppressScorerLog():
        for profileName in profileNames:
            for case in cases:
                scored = scoreCase(case, profileName)
                caseResults[profileName][case["label"]] = computeMetrics(
                    scored.get("clips") or [],
                    case["annotations"],
                    k=k,
                    matchIou=matchIou,
                    duplicateIou=duplicateIou,
                    minQuality=minQuality,
                )

    profileReports: dict[str, dict] = {}
    for profileName in profileNames:
        weights = dict(scorer.PROFILES[profileName])
        owned = [case for case in cases if case["profile"] == profileName]
        ownedMetrics = [
            caseResults[profileName][case["label"]] for case in owned
        ]
        aggregate = (
            aggregateMetrics(ownedMetrics, k, targets) if ownedMetrics else None
        )
        positives = sum(
            casePositiveCount(case, minQuality) for case in owned
        )
        sufficient = (
            len(owned) >= minCases and positives >= minPositives
        )

        delta = None
        if ownedMetrics and "default" in caseResults:
            defaultAggregate = aggregateMetrics(
                [caseResults["default"][case["label"]] for case in owned],
                k,
                targets,
            )
            delta = {
                "microF1": round(
                    aggregate["micro"]["f1AtK"]
                    - defaultAggregate["micro"]["f1AtK"],
                    4,
                ),
                "macroF1": round(
                    aggregate["macro"]["f1AtK"]
                    - defaultAggregate["macro"]["f1AtK"],
                    4,
                ),
            }

        if profileName == "default":
            recommendation = "keep"
        elif not sufficient:
            recommendation = "collect_more_or_merge"
        elif delta is not None and delta["microF1"] >= minDelta:
            recommendation = "keep"
        else:
            recommendation = "merge_into_default"

        profileReports[profileName] = {
            "weights": {
                name: round(float(value), 6)
                for name, value in sorted(weights.items())
            },
            "weightSum": round(sum(float(v) for v in weights.values()), 6),
            "topFeatures": _topFeatures(weights),
            "ownedCases": [case["label"] for case in owned],
            "caseCount": len(owned),
            "positiveCount": positives,
            "sufficient": sufficient,
            "aggregate": aggregate,
            "deltaVsDefault": delta,
            "recommendation": recommendation,
        }

    caseDiagnostics = []
    metricFields = (
        "precisionAtK",
        "recallAtK",
        "f1AtK",
        "boundaryIou",
        "invalidRate",
        "duplicateRate",
    )
    for case in cases:
        ownProfile = case["profile"]
        perProfile = {
            name: _metricSummary(caseResults[name][case["label"]])
            for name in profileNames
        }
        bestProfile = max(
            profileNames,
            key=lambda name: (
                perProfile[name]["f1AtK"]
                if perProfile[name]["f1AtK"] is not None
                else -1.0,
                perProfile[name]["precisionAtK"] or 0.0,
                name == ownProfile,
            ),
        )
        caseDiagnostics.append(
            {
                "label": case["label"],
                "roomId": case["roomId"],
                "profile": ownProfile,
                "positives": casePositiveCount(case, minQuality),
                "annotations": len(case["annotations"]),
                "ownF1": (
                    perProfile.get(ownProfile, {}).get("f1AtK")
                    if ownProfile in perProfile
                    else None
                ),
                "bestProfile": bestProfile,
                "bestF1": perProfile[bestProfile]["f1AtK"],
                "metrics": {
                    name: {
                        field: perProfile[name][field]
                        for field in metricFields
                    }
                    for name in profileNames
                },
            }
        )

    distances = []
    for index, left in enumerate(profileNames):
        for right in profileNames[index + 1:]:
            distances.append(
                {
                    "left": left,
                    "right": right,
                    **profileDistance(
                        scorer.PROFILES[left], scorer.PROFILES[right]
                    ),
                }
            )

    summary = {
        "keep": [
            name
            for name, report in profileReports.items()
            if report["recommendation"] == "keep"
        ],
        "collectMoreOrMerge": [
            name
            for name, report in profileReports.items()
            if report["recommendation"] == "collect_more_or_merge"
        ],
        "mergeIntoDefault": [
            name
            for name, report in profileReports.items()
            if report["recommendation"] == "merge_into_default"
        ],
    }

    return {
        "k": k,
        "settings": {
            "minCases": minCases,
            "minPositives": minPositives,
            "minDelta": minDelta,
            "matchIou": matchIou,
            "duplicateIou": duplicateIou,
            "minQuality": minQuality,
        },
        "profiles": profileReports,
        "cases": caseDiagnostics,
        "distances": distances,
        "summary": summary,
    }
