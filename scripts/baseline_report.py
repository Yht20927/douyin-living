#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a reproducible baseline report for the clipping pipeline.

The report records what the current code actually produces on the checked-in
sample data, plus a deterministic one-hour synthetic recall benchmark.  It
does not download models and does not require AI dependencies.

Usage::

    python scripts/baseline_report.py
    python scripts/baseline_report.py --output docs/baseline-2026-11.md
    python scripts/baseline_report.py --sensitivity 2.0 --json /tmp/base.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scorer import score  # noqa: E402


FEATURE_FILES = {
    "audio": "audio_features.json",
    "text": "text_features.json",
    "visual": "visual_features.json",
}


def _loadJson(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gitCommit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return proc.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _probeDuration(roomDir: Path) -> float | None:
    """Best-effort authoritative duration for a room directory."""
    metaPath = roomDir / "recording_meta.json"
    if metaPath.exists():
        try:
            meta = _loadJson(metaPath)
            value = meta.get("videoDurationSec") or meta.get("durationSec")
            if value:
                return float(value)
        except (OSError, ValueError, TypeError):
            pass

    tablePath = roomDir / "time_table.json"
    if tablePath.exists():
        try:
            value = _loadJson(tablePath).get("duration")
            if value:
                return float(value)
        except (OSError, ValueError, TypeError):
            pass

    for pattern in ("*.mp4", "*.flv"):
        candidates = sorted(roomDir.glob(pattern))
        if not candidates:
            continue
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(candidates[0]),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return float(proc.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
    return None


def _maxFeatureLength(features: dict[str, dict]) -> int:
    lengths = []
    for artifact in features.values():
        for values in artifact.get("features", {}).values():
            if isinstance(values, list):
                lengths.append(len(values))
    return max(lengths) if lengths else 0


def _loadRoom(roomDir: Path) -> dict[str, dict]:
    loaded = {}
    for kind, filename in FEATURE_FILES.items():
        path = roomDir / filename
        if path.exists():
            loaded[kind] = _loadJson(path)
    return loaded


def _reasonCounts(items: list[dict]) -> dict[str, int]:
    return dict(Counter(str(item.get("reason", "unknown")) for item in items))


def runRoom(roomDir: Path, sensitivity: float, minDuration: float, maxDuration: float) -> dict:
    features = _loadRoom(roomDir)
    if not features:
        return {"roomId": roomDir.name, "status": "no_features"}

    duration = _probeDuration(roomDir) or _maxFeatureLength(features)
    started = time.perf_counter()
    try:
        result = score(
            roomId=roomDir.name,
            audioFeatures=features.get("audio"),
            textFeatures=features.get("text"),
            visualFeatures=features.get("visual"),
            sensitivity=sensitivity,
            minDuration=minDuration,
            maxDuration=maxDuration,
            durationSec=duration,
        )
        elapsedSec = time.perf_counter() - started
        clips = result.get("clips", [])
        return {
            "roomId": roomDir.name,
            "status": "ok",
            "durationSec": result.get("duration"),
            "elapsedSec": round(elapsedSec, 3),
            "clips": len(clips),
            "clipDurations": [clip["duration"] for clip in clips],
            "alignment": result.get("alignment", []),
            "dropped": len(result.get("dropped", [])),
            "dropReasons": _reasonCounts(result.get("dropped", [])),
            "adjusted": len(result.get("adjustments", [])),
            "adjustReasons": _reasonCounts(result.get("adjustments", [])),
        }
    except Exception as exc:  # baseline must report failures, not crash
        return {
            "roomId": roomDir.name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def runSyntheticHour(sensitivity: float, minDuration: float, maxDuration: float) -> dict:
    """Deterministic 60-peak, one-hour recall benchmark."""
    rng = np.random.RandomState(0)
    n = 3600
    rms = rng.rand(n) * 0.2
    for t in range(20, n, 60):
        rms[t:t + 3] += 5.0

    started = time.perf_counter()
    result = score(
        roomId="synthetic-hour",
        audioFeatures={"features": {"rms": rms.tolist()}},
        sensitivity=sensitivity,
        minDuration=minDuration,
        maxDuration=maxDuration,
    )
    elapsedSec = time.perf_counter() - started
    clips = result.get("clips", [])
    return {
        "status": "ok",
        "durationSec": result.get("duration"),
        "elapsedSec": round(elapsedSec, 3),
        "expectedPeaks": len(range(20, n, 60)),
        "clips": len(clips),
        "clipDurations": [clip["duration"] for clip in clips],
        "recall": round(len(clips) / len(range(20, n, 60)), 3),
    }


def loadCoverage() -> dict | None:
    """Read .coverage without assuming pytest-cov is installed."""
    dataFile = PROJECT_ROOT / ".coverage"
    if not dataFile.exists():
        return None
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        outPath = Path(tmp) / "coverage.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "coverage", "json", "-o", str(outPath)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not outPath.exists():
            return None
        try:
            doc = json.loads(outPath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {
            "percent": round(doc.get("totals", {}).get("percent_covered", 0.0), 1),
            "stmts": doc.get("totals", {}).get("num_statements"),
            "missing": doc.get("totals", {}).get("missing_lines"),
        }


def _renderRoom(room: dict) -> list[str]:
    if room["status"] != "ok":
        return [f"| {room['roomId']} | {room['status']} | - | - | - | - |"]
    reasons = ", ".join(f"{k}:{v}" for k, v in room["dropReasons"].items()) or "-"
    return [
        "| {room} | {status} | {duration} | {clips} | {elapsed} | {reasons} |".format(
            room=room["roomId"],
            status=room["status"],
            duration=room.get("durationSec", "-"),
            clips=room.get("clips", "-"),
            elapsed=room.get("elapsedSec", "-"),
            reasons=reasons,
        )
    ]


def renderMarkdown(report: dict) -> str:
    lines = [
        "# Clipping Pipeline Baseline",
        "",
        f"- Generated: {report['generatedAt']}",
        f"- Git commit: `{report['gitCommit']}`",
        f"- Python: {report['python']}",
        f"- Platform: {report['platform']}",
        f"- Sensitivity: {report['sensitivity']}",
        f"- Duration bounds: {report['minDuration']}–{report['maxDuration']}s",
        "",
        "## Sample rooms",
        "",
        "| Room | Status | Duration (s) | Clips | Elapsed (s) | Drop reasons |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for room in report["rooms"]:
        lines.extend(_renderRoom(room))

    synthetic = report["syntheticHour"]
    lines.extend([
        "",
        "## Synthetic one-hour recall",
        "",
        "| Expected peaks | Clips | Recall | Elapsed (s) |",
        "| ---: | ---: | ---: | ---: |",
        f"| {synthetic['expectedPeaks']} | {synthetic['clips']} | "
        f"{synthetic['recall']:.1%} | {synthetic['elapsedSec']} |",
    ])

    lines.extend(["", "## Alignment", ""])
    anyAlignment = False
    for room in report["rooms"]:
        if room.get("alignment"):
            anyAlignment = True
            lines.append(f"### {room['roomId']}")
            lines.append("")
            lines.append("| Signal | Old len | New len | Method |")
            lines.append("| --- | ---: | ---: | --- |")
            for item in room["alignment"]:
                lines.append(
                    f"| {item['name']} | {item['oldLen']} | {item['newLen']} | "
                    f"{item['method']} |"
                )
            lines.append("")
    if not anyAlignment:
        lines.append("No length mismatches detected.")

    lines.extend(["", "## Coverage", ""])
    coverage = report.get("coverage")
    if coverage:
        lines.append(
            f"- Total: **{coverage['percent']}%** "
            f"({coverage['stmts']} statements, {coverage['missing']} missing)"
        )
    else:
        lines.append("- No `.coverage` file found; run `make cov` first.")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Generated from checked-in feature artifacts; no model downloads are required.",
        "- A sample room with zero clips means the scorer found no candidate above the",
        "  current threshold, not that the pipeline crashed.",
        "- Re-run this script after changing scorer thresholds or weights.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pipeline baseline report")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "baseline-2026-11.md"))
    parser.add_argument("--json", default=None, help="also write the raw report as JSON")
    parser.add_argument("--sensitivity", type=float, default=2.5)
    parser.add_argument("--min-duration", type=float, default=15.0)
    parser.add_argument("--max-duration", type=float, default=90.0)
    args = parser.parse_args(argv)

    dataDir = Path(args.data_dir)
    roomDirs = []
    if dataDir.exists():
        roomDirs = sorted(
            path for path in dataDir.iterdir()
            if path.is_dir() and any((path / name).exists() for name in FEATURE_FILES.values())
        )

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gitCommit": _gitCommit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sensitivity": args.sensitivity,
        "minDuration": args.min_duration,
        "maxDuration": args.max_duration,
        "rooms": [
            runRoom(room, args.sensitivity, args.min_duration, args.max_duration)
            for room in roomDirs
        ],
        "syntheticHour": runSyntheticHour(
            args.sensitivity, args.min_duration, args.max_duration
        ),
        "coverage": loadCoverage(),
    }

    outputPath = Path(args.output)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(renderMarkdown(report), encoding="utf-8")
    print(f"✅ Baseline report written: {outputPath}")

    if args.json:
        jsonPath = Path(args.json)
        jsonPath.parent.mkdir(parents=True, exist_ok=True)
        jsonPath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Raw report written: {jsonPath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
