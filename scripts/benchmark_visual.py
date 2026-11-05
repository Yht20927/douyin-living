#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark visual feature extraction on a real recording.

Records wall-clock time, realtime factor, scene count and which decode path
(proxy vs legacy) was used.  Run it on a one-hour 1080p file to compare
S5-4 against the previous double-decode implementation::

    python scripts/benchmark_visual.py --video /data/1h-1080p.flv
    python scripts/benchmark_visual.py --video /data/1h-1080p.flv --legacy
    python scripts/benchmark_visual.py --video /data/1h-1080p.flv --json /tmp/visual.json

``--legacy`` forces the old PySceneDetect + OpenCV path so both modes can be
measured on the same machine and file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.signalVisual as signalVisual  # noqa: E402


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a source video")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Authoritative duration in seconds (optional)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Disable the proxy path and benchmark the legacy decoder",
    )
    parser.add_argument("--proxy-height", type=int, default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", help="Write per-run results to this JSON file")
    return parser.parse_args()


def _configure(args: argparse.Namespace) -> None:
    cfg = signalVisual._cfg
    if args.legacy:
        cfg.proxy_enabled = False
    if args.proxy_height is not None:
        cfg.proxy_height = args.proxy_height
    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps


def _runOnce(args: argparse.Namespace, index: int) -> dict:
    startedAt = time.perf_counter()
    result = signalVisual.extractFeatures(
        args.video, durationSec=args.duration
    )
    elapsedSec = time.perf_counter() - startedAt
    stats = result.get("stats", {})
    duration = float(result.get("duration") or 0.0)
    return {
        "run": index,
        "video": str(Path(args.video).resolve()),
        "status": result.get("status"),
        "durationSec": duration,
        "elapsedSec": round(elapsedSec, 3),
        "realtimeFactor": (
            round(duration / elapsedSec, 4) if elapsedSec > 0 else None
        ),
        "mode": stats.get("mode"),
        "sourceDecodePasses": stats.get("sourceDecodePasses"),
        "scenes": stats.get("scenes"),
        "proxy": stats.get("proxy"),
        "featureCount": result.get("featureCount"),
        "generatorVersion": result.get("generatorVersion"),
    }


def main() -> int:
    args = _parseArgs()
    if not Path(args.video).exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    _configure(args)
    runs = [_runOnce(args, index + 1) for index in range(args.repeat)]
    elapsed = [run["elapsedSec"] for run in runs]
    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "video": runs[0]["video"],
        "mode": runs[0]["mode"],
        "runs": runs,
        "elapsedSec": {
            "min": min(elapsed),
            "median": round(statistics.median(elapsed), 3),
            "max": max(elapsed),
        },
        "realtimeFactorMedian": (
            round(runs[0]["durationSec"] / statistics.median(elapsed), 4)
            if statistics.median(elapsed) > 0
            else None
        ),
    }

    for run in runs:
        print(
            "{run}: {mode} {duration}s video in {elapsed}s "
            "({factor}x realtime, scenes={scenes}, decodePasses={passes})".format(
                run=run["run"],
                mode=run["mode"],
                duration=run["durationSec"],
                elapsed=run["elapsedSec"],
                factor=run["realtimeFactor"],
                scenes=run["scenes"],
                passes=run["sourceDecodePasses"],
            )
        )
    print(
        "median {median}s, best {best}s, worst {worst}s".format(
            median=summary["elapsedSec"]["median"],
            best=summary["elapsedSec"]["min"],
            worst=summary["elapsedSec"]["max"],
        )
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON written: {args.json}")

    return 0 if all(run["status"] == "ok" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
