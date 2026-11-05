#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate clipping output against human annotations (S5-6).

The evaluation is offline and model-free: it reads one or more
``time_table.json`` files plus matching annotation JSON files and reports
Precision@K, Recall@K, Boundary IoU, Invalid Rate, Duplicate Rate and Editing
Cost.  Nothing is re-scored, so the numbers describe exactly the artifacts on
disk.

Usage::

    # One case, two cut-offs
    python scripts/evaluate.py \\
        --time-table data/room1/time_table.json \\
        --annotations data/room1/annotations.json \\
        --k 5 --k 10

    # Several cases; labels default to the parent directory name
    python scripts/evaluate.py \\
        --case data/room1/time_table.json data/room1/annotations.json \\
        --case data/room2/time_table.json data/room2/annotations.json \\
        --json /tmp/eval.json

The annotation schema is documented in ``src/evaluation.py`` and
``docs/annotation-guide.md``.  Exit code is 0 unless ``--fail-on-target`` is
set and a target is missed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_MIN_QUALITY,
    DEFAULT_TARGETS,
    aggregateMetrics,
    checkTargets,
    computeMetrics,
    isPositive,
    loadAnnotations,
    loadTimeTable,
)
from src.log.logger import getLogger  # noqa: E402

log = getLogger(__name__)


def loadTargets(path: str | Path | None) -> dict:
    """Load target overrides, falling back to the Sprint 5 defaults."""
    if path is None:
        return dict(DEFAULT_TARGETS)

    source = Path(path)
    try:
        with open(source, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read targets {source}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{source}: targets must be a non-empty JSON object")

    targets = {}
    for name, value in raw.items():
        try:
            targets[str(name)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: target '{name}' must be a number"
            ) from exc
    return targets


def _caseLabel(timeTablePath: str, name: str | None) -> str:
    if name:
        return name
    source = Path(timeTablePath)
    if source.parent.name:
        return source.parent.name
    return source.stem


def _uniqueLabel(label: str, used: dict[str, int]) -> str:
    used[label] = used.get(label, 0) + 1
    if used[label] == 1:
        return label
    return f"{label}#{used[label]}"


def evaluateCase(
    timeTablePath: str,
    annotationPath: str,
    label: str,
    kValues: list[int],
    matchIou: float,
    duplicateIou: float,
    minQuality: float,
    targets: dict,
) -> dict:
    """Evaluate one time table/annotation pair at every requested K."""
    table = loadTimeTable(timeTablePath)
    annotations = loadAnnotations(annotationPath)
    clips = table.get("clips") or []
    allAnnotations = annotations["annotations"]
    positiveCount = sum(
        1 for item in allAnnotations if isPositive(item, minQuality)
    )

    metricsByK = {}
    for k in kValues:
        metrics = computeMetrics(
            clips,
            allAnnotations,
            k=k,
            matchIou=matchIou,
            duplicateIou=duplicateIou,
            minQuality=minQuality,
        )
        metrics["targets"] = checkTargets(metrics, targets)
        metricsByK[str(k)] = metrics

    return {
        "label": label,
        "timeTable": str(timeTablePath),
        "annotations": str(annotationPath),
        "roomId": str(table.get("roomId") or annotations.get("roomId") or ""),
        "profile": str(
            annotations.get("profile") or table.get("profile") or ""
        ),
        "durationSec": table.get("duration") or annotations.get("duration"),
        "clipCount": len(clips),
        "positiveAnnotations": positiveCount,
        "negativeAnnotations": len(allAnnotations) - positiveCount,
        "metrics": metricsByK,
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _formatMetrics(prefix: str, metrics: dict, k: int) -> str:
    return (
        f"  {prefix}k={k}: "
        f"P={_fmt(metrics.get('precisionAtK'))} "
        f"(strict {_fmt(metrics.get('precisionAtKStrict'))}) "
        f"R={_fmt(metrics.get('recallAtK'))} "
        f"F1={_fmt(metrics.get('f1AtK'))} "
        f"IoU={_fmt(metrics.get('boundaryIou'))} "
        f"invalid={_fmt(metrics.get('invalidRate'))} "
        f"dup={_fmt(metrics.get('duplicateRate'))} "
        f"edit={_fmt(metrics.get('editingCostSec'), 1)}s "
        f"negHit={_fmt(metrics.get('negativeHitRate'))}"
    )


def _formatTargets(checks: dict) -> str:
    parts = []
    for name, check in checks.items():
        if check.get("pass") is None:
            state = "NO_DATA"
        else:
            state = "PASS" if check["pass"] else "FAIL"
        parts.append(f"{name}={state}")
    return "  targets: " + " ".join(parts)


def evaluateCases(cases: list[dict], kValues: list[int], targets: dict) -> dict:
    """Aggregate per-case metrics for every K value."""
    aggregates = {}
    for k in kValues:
        caseMetrics = [case["metrics"][str(k)] for case in cases]
        aggregates[str(k)] = aggregateMetrics(caseMetrics, k, targets)
    return aggregates


def buildReport(
    cases: list[dict],
    aggregates: dict,
    args,
    targets: dict,
) -> dict:
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "k": args.k,
            "matchIou": args.match_iou,
            "duplicateIou": args.duplicate_iou,
            "minQuality": args.min_quality,
        },
        "targets": targets,
        "cases": cases,
        "aggregates": aggregates,
    }


def _collectCaseArgs(args) -> list[tuple[str, str, str | None]]:
    """Return (timeTable, annotations, explicit label) tuples."""
    collected: list[tuple[str, str, str | None]] = []
    if args.time_table or args.annotations:
        if len(args.time_table) != len(args.annotations):
            raise ValueError(
                "--time-table and --annotations must be provided in pairs"
            )
        for timeTable, annotations in zip(
            args.time_table, args.annotations, strict=True
        ):
            collected.append((timeTable, annotations, None))
    for timeTable, annotations in args.case:
        collected.append((timeTable, annotations, None))
    if not collected:
        raise ValueError(
            "provide --time-table/--annotations or at least one --case"
        )
    if args.name:
        if len(collected) > 1:
            raise ValueError("--name only applies to a single evaluation case")
        timeTable, annotations, _ = collected[0]
        collected = [(timeTable, annotations, args.name)]
    return collected


def _targetFailures(aggregates: dict) -> list[str]:
    failures = []
    for k, aggregate in aggregates.items():
        for scope in ("macro", "micro"):
            for name, check in aggregate[f"{scope}Targets"].items():
                if check.get("pass") is not True:
                    failures.append(f"k={k} {scope}.{name}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate time_table clips against human annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--time-table",
        action="append",
        default=[],
        help="time_table.json path (repeat with --annotations)",
    )
    parser.add_argument(
        "--annotations",
        action="append",
        default=[],
        help="annotation JSON path (repeat with --time-table)",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        default=[],
        metavar=("TIME_TABLE", "ANNOTATIONS"),
        help="one case as two paths; repeat for multiple cases",
    )
    parser.add_argument("--name", help="label for a single --time-table case")
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        default=[],
        help="rank cut-off; repeat for multiple values (default: 5, 10)",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.0,
        help="minimum IoU for a clip/annotation match (default: 0.0)",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=DEFAULT_DUPLICATE_IOU,
        help=f"IoU that marks a duplicate (default: {DEFAULT_DUPLICATE_IOU})",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=DEFAULT_MIN_QUALITY,
        help=f"minimum quality for a positive annotation (default: {DEFAULT_MIN_QUALITY})",
    )
    parser.add_argument(
        "--targets",
        help="JSON object overriding the provisional targets",
    )
    parser.add_argument("--json", help="write the full report to this path")
    parser.add_argument(
        "--fail-on-target",
        action="store_true",
        help="exit 1 when any macro/micro target is missed",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress text output")
    args = parser.parse_args(argv)

    args.k = list(dict.fromkeys(args.k or [5, 10]))
    if any(value < 1 for value in args.k):
        parser.error("--k must be >= 1")
    if not 0.0 <= args.match_iou <= 1.0:
        parser.error("--match-iou must be within 0..1")
    if not 0.0 <= args.duplicate_iou <= 1.0:
        parser.error("--duplicate-iou must be within 0..1")
    if not 1.0 <= args.min_quality <= 5.0:
        parser.error("--min-quality must be within 1..5")

    try:
        caseArgs = _collectCaseArgs(args)
        targets = loadTargets(args.targets)
        usedLabels: dict[str, int] = {}
        cases = []
        for timeTablePath, annotationPath, name in caseArgs:
            label = _uniqueLabel(
                _caseLabel(timeTablePath, name), usedLabels
            )
            cases.append(
                evaluateCase(
                    timeTablePath,
                    annotationPath,
                    label,
                    args.k,
                    args.match_iou,
                    args.duplicate_iou,
                    args.min_quality,
                    targets,
                )
            )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    aggregates = evaluateCases(cases, args.k, targets)
    report = buildReport(cases, aggregates, args, targets)

    if not args.quiet:
        for case in cases:
            print(
                f"[{case['label']}] room={case['roomId'] or '?'} "
                f"profile={case['profile'] or '?'} clips={case['clipCount']} "
                f"positives={case['positiveAnnotations']} "
                f"negatives={case['negativeAnnotations']}"
            )
            for k in args.k:
                metrics = case["metrics"][str(k)]
                print(_formatMetrics("", metrics, k))
                print(_formatTargets(metrics["targets"]))
        print(f"Aggregate ({len(cases)} case(s))")
        for k in args.k:
            aggregate = aggregates[str(k)]
            print(_formatMetrics("macro ", aggregate["macro"], k))
            print(_formatMetrics("micro ", aggregate["micro"], k))
            print(_formatTargets(aggregate["microTargets"]))

    if args.json:
        outputPath = Path(args.json)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Evaluation report saved: {outputPath}")

    if args.fail_on_target:
        failures = _targetFailures(aggregates)
        if failures:
            print(
                "Target check failed: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
