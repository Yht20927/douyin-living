# -*- coding: utf-8 -*-
"""Per-clip metadata artifacts (S6-2).

Each successful highlight gets a ``clip_<start>s_<trigger>.json`` file next to
its ``.mp4`` / ``.srt`` / ``.jpg`` assets.  The metadata is intentionally
derived from ``time_table.json`` (plus asset existence) so legacy and
two-stage clips share one schema.

Schema::

    {
      "clipId": 1,
      "start": 342.0,
      "end": 367.0,
      "duration": 25.0,
      "score": 0.91,
      "confidence": 0.82,
      "title": "弹幕爆发 + 关键词命中",
      "tags": ["互动", "搞笑"],
      "reasons": ["dmDensity", "asrKeyword"],
      "risks": ["asr_missing"],
      "suggestedAction": "review",
      "scoreBreakdown": {"dmDensity": 0.31},
      "assets": {
        "video": "clip_342s_dmDensity.mp4",
        "srt": "clip_342s_dmDensity.srt",
        "thumbnail": "clip_342s_dmDensity.jpg"
      }
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.clipper import clipOutputPath
from src.contracts import atomicWriteJson
from src.log.logger import getLogger

log = getLogger(__name__)

ASSET_KEYS = ("video", "srt", "thumbnail")


def _safeFloat(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _optionalFloat(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _stringList(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _reasonsFor(clip: dict[str, Any]) -> list[str]:
    """Prefer explicit reasons; derive them for legacy scorer clips."""
    reasons = _stringList(clip.get("reasons"))
    if reasons:
        return reasons
    trigger = clip.get("trigger")
    if trigger:
        reasons.append(str(trigger))
    breakdown = clip.get("breakdown")
    if isinstance(breakdown, dict):
        ordered = sorted(
            breakdown.items(),
            key=lambda item: -abs(_safeFloat(item[1])),
        )
        for name, value in ordered:
            text = str(name).strip()
            if text and _safeFloat(value) > 0 and text not in reasons:
                reasons.append(text)
    return reasons


def _scoreBreakdown(clip: dict[str, Any]) -> dict[str, float]:
    breakdown = clip.get("breakdown")
    if not isinstance(breakdown, dict):
        return {}
    return {
        str(name): _safeFloat(value)
        for name, value in breakdown.items()
        if str(name).strip()
    }


def _assetsFor(clipPath: str | os.PathLike[str]) -> dict[str, str | None]:
    video = Path(clipPath)
    stem = video.with_suffix("")
    assets: dict[str, str | None] = {"video": video.name}
    for key, suffix in (("srt", ".srt"), ("thumbnail", ".jpg")):
        candidate = stem.with_suffix(suffix)
        assets[key] = candidate.name if candidate.exists() else None
    return assets


def suggestedAction(
    risks: list[str],
    dropReasons: list[str],
) -> str:
    """Return ``drop`` / ``review`` / ``keep`` from explicit metadata.

    This is a deterministic triage hint, not a model decision: surviving clips
    normally have no ``dropReasons``, so risk-free clips are ``keep`` and any
    risk flag asks a human to review before publishing.
    """
    if dropReasons:
        return "drop"
    if risks:
        return "review"
    return "keep"


def buildClipMetadata(
    clip: dict[str, Any],
    *,
    clipId: Any | None = None,
    assets: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable metadata document for one clip."""
    start = _safeFloat(clip.get("start"))
    end = max(start, _safeFloat(clip.get("end"), start))
    duration = _safeFloat(clip.get("duration"), end - start)
    if duration <= 0:
        duration = end - start
    tags = _stringList(clip.get("tags"))
    reasons = _reasonsFor(clip)
    risks = _stringList(clip.get("risks"))
    dropReasons = _stringList(clip.get("dropReasons"))
    resolvedId = clipId if clipId is not None else clip.get("id")
    confidence = _optionalFloat(clip.get("confidence"))

    assetMap: dict[str, str | None] = {key: None for key in ASSET_KEYS}
    if assets:
        for key in ASSET_KEYS:
            value = assets.get(key)
            assetMap[key] = str(value) if value else None

    return {
        "clipId": resolvedId,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "score": round(_safeFloat(clip.get("score")), 4),
        "confidence": (
            round(confidence, 4) if confidence is not None else None
        ),
        "title": str(clip.get("title") or ""),
        "tags": tags,
        "reasons": reasons,
        "risks": risks,
        "suggestedAction": suggestedAction(risks, dropReasons),
        "scoreBreakdown": _scoreBreakdown(clip),
        "assets": assetMap,
    }


def clipMetadataPath(
    clip: dict[str, Any],
    outputDir: str | os.PathLike[str],
) -> str:
    """Return the deterministic metadata path for one clip."""
    clipPath = clipOutputPath(clip, str(outputDir))
    return str(Path(clipPath).with_suffix(".json"))


def writeClipMetadata(
    clip: dict[str, Any],
    clipPath: str | os.PathLike[str],
    *,
    clipId: Any | None = None,
) -> str:
    """Write metadata next to a clip and return its path.

    The metadata filename matches the clip's deterministic stem, so callers
    can predict it from ``time_table.json`` without scanning the directory.
    """
    metadataPath = str(Path(clipPath).with_suffix(".json"))
    document = buildClipMetadata(
        clip,
        clipId=clipId,
        assets=_assetsFor(clipPath),
    )
    atomicWriteJson(metadataPath, document)
    log.debug(f"Clip metadata saved: {metadataPath}")
    return metadataPath
