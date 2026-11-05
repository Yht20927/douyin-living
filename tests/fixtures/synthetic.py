# -*- coding: utf-8 -*-
"""Deterministic 1 Hz feature fixtures — no AI dependencies required."""

from __future__ import annotations

import numpy as np


def makeSyntheticFeatures(
    durationSec: int = 90,
    peakStarts: tuple[int, ...] = (20, 55),
    seed: int = 7,
) -> dict[str, dict]:
    """Build audio/text/visual artifacts with sparse, aligned highlights.

    Returns the same shape as the real extractors so it can be passed
    directly to ``src.scorer.score``.
    """
    n = int(durationSec)
    if n <= 0:
        raise ValueError("durationSec must be positive")

    rng = np.random.RandomState(seed)
    rms = 0.05 + rng.rand(n) * 0.02
    dmDensity = np.zeros(n, dtype=np.float64)
    motion = np.zeros(n, dtype=np.float64)
    sceneChange = np.zeros(n, dtype=np.float64)
    eventLaughter = np.zeros(n, dtype=np.float64)

    for start in peakStarts:
        if not 0 <= start < n:
            continue
        peak = int(start)
        rms[max(0, peak - 1):min(n, peak + 3)] += 3.0
        dmDensity[max(0, peak - 2):min(n, peak + 4)] += 20.0
        motion[max(0, peak - 1):min(n, peak + 3)] += 0.8
        eventLaughter[max(0, peak - 1):min(n, peak + 3)] += 1.0
        sceneChange[peak] = 1.0

    return {
        "audio": {
            "sampleRate": 1,
            "duration": float(n),
            "features": {
                "rms": rms.tolist(),
                "eventLaughter": eventLaughter.tolist(),
            },
        },
        "text": {
            "sampleRate": 1,
            "duration": float(n),
            "features": {"dmDensity": dmDensity.tolist()},
        },
        "visual": {
            "sampleRate": 1,
            "duration": float(n),
            "features": {
                "motion": motion.tolist(),
                "sceneChange": sceneChange.tolist(),
            },
        },
    }
