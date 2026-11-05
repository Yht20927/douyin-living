#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search scorer weights on fixed features without re-running models (S5-6).

Each case is a directory produced by the pipeline, containing
``time_table.json``, ``annotations.json`` and the cached
``audio_features.json`` / ``text_features.json`` / ``visual_features.json``.
For every candidate weight vector the script temporarily swaps the profile in
``src.scorer.PROFILES``, calls :func:`src.scorer.score`, and evaluates the
result with the offline metrics from :mod:`src.evaluation`.

Usage::

    python scripts/tune_weights.py \\
        --case data/room1/time_table.json data/room1/annotations.json \\
        --case data/room2/time_table.json data/room2/annotations.json \\
        --profile default --k 5 --strategy random --iterations 200 \\
        --output-yaml /tmp/default.tuned.yaml --json /tmp/tune.json

The search never mutates ``config/scorer.yaml``; the best fragment is printed
or written to ``--output-yaml`` for manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import scorer  # noqa: E402
from src.evaluation import (  # noqa: E402
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_MIN_QUALITY,
    DEFAULT_TARGETS,
    aggregateMetrics,
    computeMetrics,
    loadAnnotations,
    loadTimeTable,
    normalizeWeights,
    objectiveValue,
)
from src.util import FEATURE_NAMES  # noqa: E402

FEATURE_FILES = {
    "audio": "audio_features.json",
    "text": "text_features.json",
    "visual": "visual_features.json",
}


class _SilentLogger:
    """Suppress per-candidate scorer INFO lines during a search."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


@contextmanager
def patchedProfile(profile: str, weights: dict[str, float]):
    """Temporarily replace one scorer profile, restoring it afterwards."""
    original = scorer.PROFILES.get(profile)
    scorer.PROFILES[profile] = weights
    try:
        yield
    finally:
        if original is None:
            scorer.PROFILES.pop(profile, None)
        else:
            scorer.PROFILES[profile] = original


def _loadFeatureFile(path: Path) -> dict:
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
    """Load one time table, annotation file and its cached feature arrays."""
    table = loadTimeTable(timeTablePath)
    annotations = loadAnnotations(annotationPath)
    roomDir = Path(timeTablePath).resolve().parent

    featureSets = {
        kind: _loadFeatureFile(roomDir / filename)
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
            table.get("profile") or annotations.get("profile") or "default"
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
        audioFeatures={"features": case["audio"]} if case["audio"] else None,
        textFeatures={"features": case["text"]} if case["text"] else None,
        visualFeatures={"features": case["visual"]} if case["visual"] else None,
        profile=profile,
        sensitivity=case["sensitivity"],
        minDuration=case["minDuration"],
        maxDuration=case["maxDuration"],
        durationSec=case["durationSec"],
    )


def evaluateCandidate(
    weights: dict[str, float],
    cases: list[dict],
    profile: str,
    k: int,
    objective: str,
    aggregateScope: str,
    matchIou: float,
    duplicateIou: float,
    minQuality: float,
    targets: dict,
) -> dict:
    """Score and evaluate one candidate weight vector on every case."""
    caseMetrics = []
    with patchedProfile(profile, weights):
        for case in cases:
            result = scoreCase(case, profile)
            caseMetrics.append(
                computeMetrics(
                    result.get("clips") or [],
                    case["annotations"],
                    k=k,
                    matchIou=matchIou,
                    duplicateIou=duplicateIou,
                    minQuality=minQuality,
                )
            )

    aggregate = aggregateMetrics(caseMetrics, k, targets)
    score = objectiveValue(aggregate[aggregateScope], objective)
    return {
        "weights": dict(weights),
        "objective": round(score, 6),
        "aggregate": aggregate,
        "caseMetrics": caseMetrics,
    }


def _qualityKey(result: dict, objective: str, scope: str) -> tuple:
    """Tie-break equal objectives by precision, recall and fewer errors."""
    metrics = result["aggregate"][scope]
    return (
        result["objective"],
        float(metrics.get("precisionAtK") or 0.0),
        float(metrics.get("recallAtK") or 0.0),
        -float(metrics.get("invalidRate") or 0.0),
        -float(metrics.get("duplicateRate") or 0.0),
    )


def randomCandidate(
    rng: random.Random,
    baseline: dict[str, float],
    pool: list[str],
) -> dict[str, float]:
    """Multiplicative jitter, with a chance to drop or introduce features."""
    scale = sum(baseline.values()) or 1.0
    weights: dict[str, float] = {}
    for name in pool:
        baseValue = baseline.get(name, 0.0)
        if baseValue <= 0:
            if rng.random() < 0.15:
                weights[name] = rng.uniform(0.02, 0.15) * scale
            continue
        roll = rng.random()
        if roll < 0.08:
            continue
        if roll < 0.55:
            multiplier = math.exp(rng.uniform(math.log(0.4), 0.0))
        else:
            multiplier = math.exp(rng.uniform(0.0, math.log(2.5)))
        weights[name] = baseValue * multiplier
    if not any(value > 0 for value in weights.values()):
        weights = dict(baseline)
    return normalizeWeights(weights)


def multiplyWeight(
    current: dict[str, float],
    name: str,
    multiplier: float,
) -> dict[str, float] | None:
    """Return a normalized candidate with one weight changed."""
    weights = dict(current)
    baseValue = weights.get(name, 0.0)
    if baseValue <= 0:
        if multiplier <= 0:
            return None
        weights[name] = 0.05 * multiplier * (sum(weights.values()) or 1.0)
    else:
        weights[name] = baseValue * multiplier
    try:
        return normalizeWeights(weights)
    except ValueError:
        return None


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _formatAggregate(aggregate: dict, k: int, scope: str) -> str:
    metrics = aggregate[scope]
    return (
        f"k={k} {scope}: P={_fmt(metrics.get('precisionAtK'))} "
        f"R={_fmt(metrics.get('recallAtK'))} "
        f"F1={_fmt(metrics.get('f1AtK'))} "
        f"IoU={_fmt(metrics.get('boundaryIou'))} "
        f"invalid={_fmt(metrics.get('invalidRate'))} "
        f"dup={_fmt(metrics.get('duplicateRate'))} "
        f"edit={_fmt(metrics.get('editingCostSec'), 1)}s"
    )


def _cleanResult(result: dict) -> dict:
    return {
        "weights": {
            name: round(float(value), 6)
            for name, value in sorted(result["weights"].items())
        },
        "objective": result["objective"],
        "aggregate": result["aggregate"],
        "caseMetrics": [
            {
                "precisionAtK": metrics["precisionAtK"],
                "recallAtK": metrics["recallAtK"],
                "f1AtK": metrics["f1AtK"],
                "boundaryIou": metrics["boundaryIou"],
                "editingCostSec": metrics["editingCostSec"],
                "invalidRate": metrics["invalidRate"],
                "duplicateRate": metrics["duplicateRate"],
            }
            for metrics in result["caseMetrics"]
        ],
    }


def _writeYaml(path: Path, profile: str, weights: dict, args) -> None:
    header = (
        "# Generated by scripts/tune_weights.py\n"
        f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# profile={profile} k={args.k} objective={args.objective} "
        f"scope={args.aggregate} strategy={args.strategy}\n"
        "# Review before merging into config/scorer.yaml.\n"
    )
    body = yaml.safe_dump(
        {
            profile: {
                name: round(float(value), 6)
                for name, value in sorted(weights.items())
            }
        },
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search scorer weights on fixed features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        required=True,
        metavar=("TIME_TABLE", "ANNOTATIONS"),
        help="one case as two paths; repeat for multiple cases",
    )
    parser.add_argument(
        "--profile",
        help="profile to tune (default: profile of the first time table)",
    )
    parser.add_argument("--k", type=int, default=5, help="rank cut-off (default: 5)")
    parser.add_argument(
        "--objective",
        choices=["f1", "precision", "recall", "iou", "editing"],
        default="f1",
        help="scalar objective (default: f1)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["micro", "macro"],
        default="micro",
        help="aggregate used for the objective (default: micro)",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "grid"],
        default="random",
        help="random search or coordinate grid (default: random)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="random candidates including baseline (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="coordinate-search passes (default: 2)",
    )
    parser.add_argument(
        "--grid-multipliers",
        default="0,0.5,1.5,2",
        help="comma-separated multipliers for --strategy grid",
    )
    parser.add_argument(
        "--features",
        help="comma-separated extra feature names to explore",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="explore every known feature, not just baseline/--features",
    )
    parser.add_argument("--match-iou", type=float, default=0.0)
    parser.add_argument(
        "--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU
    )
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY)
    parser.add_argument("--output-yaml", help="write the best profile fragment here")
    parser.add_argument("--json", help="write the full search result here")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print progress every N candidates (0 disables)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if not 0.0 <= args.match_iou <= 1.0:
        parser.error("--match-iou must be within 0..1")
    if not 0.0 <= args.duplicate_iou <= 1.0:
        parser.error("--duplicate-iou must be within 0..1")
    if not 1.0 <= args.min_quality <= 5.0:
        parser.error("--min-quality must be within 1..5")

    try:
        multipliers = [
            float(item)
            for item in args.grid_multipliers.split(",")
            if item.strip()
        ]
    except ValueError:
        parser.error("--grid-multipliers must be comma-separated numbers")
        return 2
    if not multipliers or any(value < 0 for value in multipliers):
        parser.error("--grid-multipliers must be non-negative")

    try:
        cases = [
            loadCase(timeTablePath, annotationPath)
            for timeTablePath, annotationPath in args.case
        ]
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    profile = args.profile or cases[0]["profile"]
    if profile not in scorer.PROFILES:
        parser.error(
            f"unknown profile '{profile}' "
            f"(available: {', '.join(sorted(scorer.PROFILES))})"
        )
        return 2
    if any(case["profile"] != profile for case in cases):
        print(
            "warning: cases use different profiles; tuning only "
            f"'{profile}' weights",
            file=sys.stderr,
        )

    baseline = normalizeWeights(dict(scorer.PROFILES[profile]))
    pool = set(baseline)
    if args.features:
        pool.update(
            name.strip() for name in args.features.split(",") if name.strip()
        )
    if args.all_features:
        pool.update(FEATURE_NAMES)
    unknown = pool - FEATURE_NAMES
    if unknown:
        parser.error(f"unknown feature(s): {', '.join(sorted(unknown))}")
        return 2
    poolList = sorted(pool)

    originalLog = scorer.log
    scorer.log = _SilentLogger()
    try:
        baselineResult = evaluateCandidate(
            baseline,
            cases,
            profile,
            args.k,
            args.objective,
            args.aggregate,
            args.match_iou,
            args.duplicate_iou,
            args.min_quality,
            DEFAULT_TARGETS,
        )
        bestResult = baselineResult
        evaluated = 1
        rng = random.Random(args.seed)

        if args.strategy == "random":
            for index in range(1, args.iterations):
                candidate = randomCandidate(rng, baseline, poolList)
                result = evaluateCandidate(
                    candidate,
                    cases,
                    profile,
                    args.k,
                    args.objective,
                    args.aggregate,
                    args.match_iou,
                    args.duplicate_iou,
                    args.min_quality,
                    DEFAULT_TARGETS,
                )
                evaluated += 1
                if _qualityKey(result, args.objective, args.aggregate) > _qualityKey(
                    bestResult, args.objective, args.aggregate
                ):
                    bestResult = result
                if (
                    args.progress_every
                    and not args.quiet
                    and index % args.progress_every == 0
                ):
                    print(
                        f"  random {index}/{args.iterations - 1}: "
                        f"best={_fmt(bestResult['objective'], 4)}"
                    )
        else:
            current = baseline
            currentResult = baselineResult
            for roundIndex in range(1, args.rounds + 1):
                improved = False
                for name in poolList:
                    for multiplier in multipliers:
                        if multiplier == 1.0:
                            continue
                        candidate = multiplyWeight(current, name, multiplier)
                        if candidate is None:
                            continue
                        result = evaluateCandidate(
                            candidate,
                            cases,
                            profile,
                            args.k,
                            args.objective,
                            args.aggregate,
                            args.match_iou,
                            args.duplicate_iou,
                            args.min_quality,
                            DEFAULT_TARGETS,
                        )
                        evaluated += 1
                        if _qualityKey(
                            result, args.objective, args.aggregate
                        ) > _qualityKey(
                            currentResult, args.objective, args.aggregate
                        ):
                            current = candidate
                            currentResult = result
                            improved = True
                        if (
                            args.progress_every
                            and not args.quiet
                            and evaluated % args.progress_every == 0
                        ):
                            print(
                                f"  grid {evaluated}: "
                                f"best={_fmt(currentResult['objective'], 4)}"
                            )
                if _qualityKey(
                    currentResult, args.objective, args.aggregate
                ) > _qualityKey(bestResult, args.objective, args.aggregate):
                    bestResult = currentResult
                if not improved:
                    break
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        scorer.log = originalLog

    improvement = bestResult["objective"] - baselineResult["objective"]
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "objective": args.objective,
        "aggregateScope": args.aggregate,
        "k": args.k,
        "strategy": args.strategy,
        "seed": args.seed,
        "evaluated": evaluated,
        "cases": [
            {
                "label": case["label"],
                "roomId": case["roomId"],
                "profile": case["profile"],
                "positiveAnnotations": case["positiveAnnotations"],
                "negativeAnnotations": case["negativeAnnotations"],
            }
            for case in cases
        ],
        "settings": {
            "matchIou": args.match_iou,
            "duplicateIou": args.duplicate_iou,
            "minQuality": args.min_quality,
        },
        "baseline": _cleanResult(baselineResult),
        "best": _cleanResult(bestResult),
        "improvement": round(improvement, 6),
    }

    if not args.quiet:
        for label, result in (
            ("baseline", baselineResult),
            ("best", bestResult),
        ):
            print(f"[{label}] objective={_fmt(result['objective'], 4)}")
            print(
                "  "
                + _formatAggregate(result["aggregate"], args.k, args.aggregate)
            )
        print(f"improvement: {_fmt(improvement, 4)}")
        print(f"evaluated: {evaluated} candidate(s)")
        print("best weights (YAML):")
        print(
            yaml.safe_dump(
                {profile: report["best"]["weights"]},
                allow_unicode=True,
                sort_keys=True,
                default_flow_style=False,
            ).rstrip()
        )

    if args.output_yaml:
        outputPath = Path(args.output_yaml)
        _writeYaml(outputPath, profile, report["best"]["weights"], args)
        print(f"Best profile written: {outputPath}")

    if args.json:
        outputPath = Path(args.json)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Search report written: {outputPath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
