# -*- coding: utf-8 -*-
"""Shared utility functions used across the project."""

from __future__ import annotations

from typing import FrozenSet

from src.log.logger import getLogger

log = getLogger(__name__)


# ── String / formatting utilities ────────────────────────────────

def fmtSize(size: int) -> str:
    """Format byte count as human-readable string."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def fmtSrtTime(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Cookie / auth parsing ────────────────────────────────────────

def transCookies(cookieStr: str) -> dict[str, str]:
    """Convert 'k1=v1; k2=v2' string to dict.

    Handles both '; ' (standard HTTP) and ';' (browser export) separators.
    Returns empty dict for empty / whitespace-only input, logging a warning.
    """
    result: dict[str, str] = {}

    stripped = cookieStr.strip()
    if not stripped:
        log.warning("Empty cookie string — auth may fail")
        return result

    # Normalize separator: treat both "; " and ";" as valid
    # Strategy: split on ';' first, then strip each item
    for item in stripped.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, value = item.partition("=")
        result[key.strip()] = value

    return result


# ── Feature name validation ──────────────────────────────────────

# Canonical feature names used across signal extractors and scorer profiles.
# Any mismatch between extractor output keys and scorer weight keys will cause
# silent signal loss — add new names here whenever you add a feature.
FEATURE_NAMES: FrozenSet[str] = frozenset({
    # Text features (signalText.py)
    "dmDensity",
    "dmAcceleration",
    "dmEntropy",
    "dmSentiment",
    "dmUniqueUsers",
    "dmUtr",
    "asrKeyword",
    "topicChange",
    "speakerChange",
    # Audio features (signalAudio.py)
    "rms",
    "spectralCentroid",
    "mfccDist",
    "zcr",
    "eventLaughter",
    "eventApplause",
    "eventMusic",
    # Visual features (signalVisual.py)
    "sceneChange",
    "motion",
    "faceCount",
})


def validateFeatures(
    features: dict[str, object],
    sourceLabel: str = "features",
    knownNames: FrozenSet[str] = FEATURE_NAMES,
) -> list[str]:
    """Validate feature dict keys against the canonical set.

    Returns a list of unknown keys (empty = all good).  Logs a warning for
    each unknown key so mismatches don't cause silent signal loss.
    """
    unknown = []
    for key in features:
        if key not in knownNames:
            unknown.append(key)
            log.warning(
                f"Unknown feature '{key}' in {sourceLabel} — "
                f"may not have a weight in scorer profiles"
            )
    return unknown
