#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare scorer profiles against annotated cases (S5-7).

The report answers three questions without re-running any model:

* does a profile own enough annotated cases to be trusted?
* on those cases, does its micro F1 beat the ``default`` profile?
* how different are the weight vectors in the first place?

Each case is a directory produced by the pipeline, containing
``time_table.json``, ``annotations.json`` and cached feature artifacts.  All
profiles are re-scored on the cached features, so the comparison is
deterministic and cheap.  The script recommends ``keep``,
``collect_more_or_merge`` or ``merge_into_default`` for every profile, but it
never edits ``config/scorer.yaml``.

Usage::

    python scripts/profile_report.py \\
        --case data/room1/time_table.json data/room1/annotations.json \\
        --case data/room2/time_table.json data/room2/annotations.json \\
        --k 5 --min-cases 2 --min-positives 5 \\
        --json /tmp/profile-report.json

Only profiles backed by enough annotated cases should be presented as
distinct; otherwise the report says so explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import scorer  # noqa: E402
from src.evaluation import (  # noqa: E402
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_MIN_QUALITY,
)
from src.profile_analysis import compareProfiles, loadCase  # noqa: E402


def _uniqueLabel(label: str, used: dict[str, int]) -> str:
    used[label] = used.get(label, 0) + 1
    if used[label] == 1:
        return label
    return f"{label}#{used[label]}"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def loadCases(
    caseArgs: list[tuple[str, str]],
) -> tuple[list[dict], list[dict]]:
    """Load all cases, making labels unique and validating their profile."""
    usedLabels: dict[str, int] = {}
    cases = []
    inputs = []
    for timeTablePath, annotationPath in caseArgs:
        case = loadCase(timeTablePath, annotationPath)
        case["label"] = _uniqueLabel(case["label"], usedLabels)
        if case["profile"] not in scorer.PROFILES:
            raise ValueError(
                f"{timeTablePath}: unknown profile '{case['profile']}' "
                f"(available: {', '.join(sorted(scorer.PROFILES))})"
            )
        cases.append(case)
        inputs.append(
            {
                "label": case["label"],
                "timeTable": str(timeTablePath),
                "annotations": str(annotationPath),
                "roomId": case["roomId"],
                "profile": case["profile"],
                "positiveAnnotations": case["positiveAnnotations"],
                "negativeAnnotations": case["negativeAnnotations"],
            }
        )
    return cases, inputs


def buildReport(
    cases: list[dict],
    inputs: list[dict],
    profiles: list[str],
    k: int,
    minCases: int,
    minPositives: int,
    minDelta: float,
    matchIou: float,
    duplicateIou: float,
    minQuality: float,
) -> dict:
    """Run the comparison and attach generation metadata."""
    result = compareProfiles(
        cases,
        profiles=profiles,
        k=k,
        minCases=minCases,
        minPositives=minPositives,
        minDelta=minDelta,
        matchIou=matchIou,
        duplicateIou=duplicateIou,
        minQuality=minQuality,
    )
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k,
        "profiles": profiles,
        "inputs": inputs,
        "settings": result["settings"],
        "profileReports": result["profiles"],
        "cases": result["cases"],
        "distances": result["distances"],
        "summary": result["summary"],
    }


def _formatMetrics(prefix: str, aggregate: dict, k: int, scope: str = "micro") -> str:
    metrics = aggregate[scope]
    return (
        f"  {prefix}{scope} k={k}: "
        f"P={_fmt(metrics.get('precisionAtK'))} "
        f"R={_fmt(metrics.get('recallAtK'))} "
        f"F1={_fmt(metrics.get('f1AtK'))} "
        f"IoU={_fmt(metrics.get('boundaryIou'))} "
        f"invalid={_fmt(metrics.get('invalidRate'))} "
        f"dup={_fmt(metrics.get('duplicateRate'))}"
    )


def printReport(report: dict, quiet: bool = False) -> None:
    """Print a human-readable summary of the profile comparison."""
    if quiet:
        return
    print(
        f"Profile comparison · k={report['k']} · "
        f"{len(report['cases'])} case(s) · "
        f"{len(report['profiles'])} profile(s)"
    )
    for name in report["profiles"]:
        profileReport = report["profileReports"][name]
        evidence = "ok" if profileReport["sufficient"] else "insufficient"
        print(
            f"[{name}] owned={profileReport['caseCount']} case(s), "
            f"positives={profileReport['positiveCount']}, "
            f"evidence={evidence}"
        )
        aggregate = profileReport["aggregate"]
        if aggregate:
            print(_formatMetrics("", aggregate, report["k"], "micro"))
            print(_formatMetrics("", aggregate, report["k"], "macro"))
            delta = profileReport["deltaVsDefault"]
            if delta is not None:
                print(
                    "  vs default on owned cases: "
                    f"dF1={delta['microF1']:+.4f} "
                    f"dMacroF1={delta['macroF1']:+.4f}"
                )
        else:
            print("  no owned annotated cases — collect data first")
        recommendation = profileReport["recommendation"]
        if name == "default":
            recommendation = "keep (default baseline)"
        print(f"  recommendation: {recommendation}")

    print("Cases:")
    for case in report["cases"]:
        ownF1 = case["ownF1"]
        if ownF1 is None:
            ownText = "—"
        else:
            ownText = f"{ownF1:.3f}"
        defaultMetrics = case["metrics"].get("default")
        defaultText = (
            _fmt(defaultMetrics["f1AtK"]) if defaultMetrics else "—"
        )
        print(
            f"  [{case['label']}] profile={case['profile']} "
            f"positives={case['positives']} ownF1={ownText} "
            f"best={case['bestProfile']}({_fmt(case['bestF1'])}) "
            f"default={defaultText}"
        )

    print("Weight distances (total variation, 0=identical):")
    for distance in report["distances"]:
        print(
            f"  {distance['left']}~{distance['right']} "
            f"TV={distance['totalVariation']:.3f} "
            f"cosine={distance['cosine']:.3f}"
        )

    summary = report["summary"]
    keep = ", ".join(summary["keep"]) or "(none)"
    collect = ", ".join(summary["collectMoreOrMerge"]) or "(none)"
    merge = ", ".join(summary["mergeIntoDefault"]) or "(none)"
    print(f"Summary: keep={keep}; collect_more_or_merge={collect}; merge={merge}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare scorer profiles on annotated cases",
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
        "--profiles",
        help="comma-separated profile names (default: every configured profile)",
    )
    parser.add_argument("--k", type=int, default=5, help="rank cut-off (default: 5)")
    parser.add_argument(
        "--min-cases",
        type=int,
        default=2,
        help="minimum owned annotated cases to trust a profile (default: 2)",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=5,
        help="minimum owned positive annotations (default: 5)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.02,
        help="required micro-F1 gain over default to keep a profile (default: 0.02)",
    )
    parser.add_argument("--match-iou", type=float, default=0.0)
    parser.add_argument(
        "--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU
    )
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY)
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument("--quiet", action="store_true", help="suppress text output")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.min_cases < 1:
        parser.error("--min-cases must be >= 1")
    if args.min_positives < 1:
        parser.error("--min-positives must be >= 1")
    if not 0.0 <= args.match_iou <= 1.0:
        parser.error("--match-iou must be within 0..1")
    if not 0.0 <= args.duplicate_iou <= 1.0:
        parser.error("--duplicate-iou must be within 0..1")
    if not 1.0 <= args.min_quality <= 5.0:
        parser.error("--min-quality must be within 1..5")

    profiles = (
        [name.strip() for name in args.profiles.split(",") if name.strip()]
        if args.profiles
        else sorted(
            scorer.PROFILES,
            key=lambda name: (name != "default", name),
        )
    )
    unknown = [name for name in profiles if name not in scorer.PROFILES]
    if unknown:
        parser.error(
            f"unknown profile(s): {', '.join(unknown)} "
            f"(available: {', '.join(sorted(scorer.PROFILES))})"
        )

    try:
        cases, inputs = loadCases(args.case)
        report = buildReport(
            cases,
            inputs,
            profiles,
            args.k,
            args.min_cases,
            args.min_positives,
            args.min_delta,
            args.match_iou,
            args.duplicate_iou,
            args.min_quality,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    printReport(report, quiet=args.quiet)

    if args.json:
        outputPath = Path(args.json)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Profile report written: {outputPath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
