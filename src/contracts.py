# -*- coding: utf-8 -*-
"""Artifact contracts — recording metadata, feature artifacts, atomic writes.

These helpers are deliberately dependency-light (stdlib only) so every
pipeline stage can use them without importing heavy AI packages.

Schema versioning
-----------------
``SCHEMA_VERSION`` is bumped whenever a persisted JSON shape changes in a
backwards-incompatible way.  Consumers should reject unknown major versions
instead of silently mis-reading old artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "sprint3"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ── time / hashing / atomic IO ───────────────────────────────────


def utcNowIso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256File(path: str | os.PathLike[str], chunkSize: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunkSize)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256Text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hashJson(data: Any) -> str:
    """Stable hash of a JSON-serialisable object."""
    serialized = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256Text(serialized)


def artifactFingerprint(
    path: str | os.PathLike[str],
    hashLimitBytes: int = 256 * 1024 * 1024,
) -> dict[str, Any] | None:
    """Describe an artifact for status/caching.

    Small files are hashed with SHA-256.  Large media files only record
    size + mtime to avoid blocking the pipeline for tens of seconds; the
    caller can still opt into hashing separately.
    """
    p = Path(path)
    if not p.exists():
        return None

    stat = p.stat()
    info: dict[str, Any] = {
        "path": str(p),
        "sizeBytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }
    if stat.st_size <= hashLimitBytes:
        info["sha256"] = sha256File(p)
    else:
        info["sha256"] = None
        info["note"] = "large-file: hash skipped"
    return info


def atomicWriteJson(path: str | os.PathLike[str], data: Any) -> None:
    """Write JSON atomically (temp file in same directory + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpName = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmpName, target)
    except BaseException:
        try:
            os.unlink(tmpName)
        except OSError:
            pass
        raise


def settingsHash(configDir: str | os.PathLike[str] | None = None) -> str:
    """Hash every YAML/JSON config file so a run can be reproduced."""
    directory = Path(configDir) if configDir else _CONFIG_DIR
    digest = hashlib.sha256()
    if not directory.exists():
        return digest.hexdigest()

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"} or not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# ── ffprobe ──────────────────────────────────────────────────────


def _parseRate(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def probeMedia(path: str | os.PathLike[str], timeoutSec: float = 30.0) -> dict[str, Any]:
    """Run ffprobe and return a normalised media description.

    Returns an empty dict when ffprobe is unavailable or the file cannot be
    probed.  Callers should treat that as "unknown", not "zero duration".
    """
    if not Path(path).exists():
        return {}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeoutSec, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}

    try:
        raw = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}

    formatInfo = raw.get("format", {}) or {}
    streams = raw.get("streams", []) or []

    def _streamDuration(codecType: str) -> float:
        values = []
        for stream in streams:
            if stream.get("codec_type") != codecType:
                continue
            try:
                values.append(float(stream.get("duration")))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0.0

    try:
        formatDuration = float(formatInfo.get("duration"))
    except (TypeError, ValueError):
        formatDuration = 0.0

    videoDuration = _streamDuration("video") or formatDuration
    audioDuration = _streamDuration("audio") or formatDuration
    fps = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            fps = _parseRate(stream.get("avg_frame_rate")) or _parseRate(
                stream.get("r_frame_rate")
            )
            break

    return {
        "durationSec": round(videoDuration or formatDuration, 3),
        "videoDurationSec": round(videoDuration, 3),
        "audioDurationSec": round(audioDuration, 3),
        "formatDurationSec": round(formatDuration, 3),
        "hasVideo": any(s.get("codec_type") == "video" for s in streams),
        "hasAudio": any(s.get("codec_type") == "audio" for s in streams),
        "fps": round(fps, 3) if fps else None,
        "sizeBytes": int(formatInfo.get("size") or Path(path).stat().st_size),
        "bitRate": int(formatInfo.get("bit_rate") or 0),
        "streams": [
            {
                "index": s.get("index"),
                "codecType": s.get("codec_type"),
                "codecName": s.get("codec_name"),
                "width": s.get("width"),
                "height": s.get("height"),
            }
            for s in streams
        ],
    }


def probeDuration(path: str | os.PathLike[str], timeoutSec: float = 30.0) -> float:
    return float(probeMedia(path, timeoutSec=timeoutSec).get("durationSec") or 0.0)


# ── recording metadata ───────────────────────────────────────────


@dataclass
class RecordingMeta:
    roomId: str
    startedAtWall: str | None = None
    startedAtMonotonic: float | None = None
    firstVideoDataAt: str | None = None
    videoDurationSec: float = 0.0
    audioDurationSec: float = 0.0
    segments: list[str] = field(default_factory=list)
    sourceHashes: dict[str, dict[str, Any]] = field(default_factory=dict)
    ffprobe: dict[str, Any] = field(default_factory=dict)
    schemaVersion: str = SCHEMA_VERSION
    generatorVersion: str = GENERATOR_VERSION
    createdAt: str = field(default_factory=utcNowIso)

    def toDict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> None:
        atomicWriteJson(path, self.toDict())

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "RecordingMeta | None":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ── feature artifact inspection ──────────────────────────────────


def inspectFeatureArtifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a feature JSON file and classify it.

    Returns a dict with:
        status       missing | invalid | empty | ok
        duration     declared duration (seconds)
        featureCount number of non-empty feature arrays
        featureNames names of non-empty feature arrays
        maxLength    longest feature array
        errors       human-readable validation errors
        fingerprint  artifactFingerprint() output (or None)
    """
    p = Path(path)
    result: dict[str, Any] = {
        "path": str(p),
        "status": "missing",
        "duration": 0.0,
        "featureCount": 0,
        "featureNames": [],
        "maxLength": 0,
        "errors": [],
        "fingerprint": None,
    }
    if not p.exists():
        result["errors"].append("file not found")
        return result

    result["fingerprint"] = artifactFingerprint(p)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "invalid"
        result["errors"].append(f"JSON parse error: {exc}")
        return result

    if not isinstance(data, dict):
        result["status"] = "invalid"
        result["errors"].append("top-level JSON must be an object")
        return result

    features = data.get("features")
    if not isinstance(features, dict):
        result["status"] = "invalid"
        result["errors"].append("'features' must be an object")
        return result

    nonEmpty: dict[str, list] = {}
    invalidArrays: list[str] = []
    for name, values in features.items():
        if values is None:
            continue
        if not isinstance(values, list):
            invalidArrays.append(name)
            continue
        if values:
            nonEmpty[name] = values

    if invalidArrays:
        result["status"] = "invalid"
        result["errors"].append(
            "feature values must be arrays: " + ", ".join(sorted(invalidArrays))
        )
        return result

    try:
        duration = float(data.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    result["duration"] = duration

    if not nonEmpty:
        result["status"] = "empty"
        result["errors"].append("no non-empty feature arrays")
        return result

    lengths = {name: len(values) for name, values in nonEmpty.items()}
    maxLength = max(lengths.values())
    result["featureCount"] = len(nonEmpty)
    result["featureNames"] = sorted(nonEmpty)
    result["maxLength"] = maxLength
    result["lengths"] = lengths

    if duration <= 0:
        result["status"] = "invalid"
        result["errors"].append("duration must be > 0")
        return result

    expected = int(duration) + (1 if duration > int(duration) else 0)
    misaligned = {k: v for k, v in lengths.items() if abs(v - expected) > 1}
    if misaligned:
        result["errors"].append(
            "feature arrays differ from declared duration: "
            + ", ".join(f"{k}={v}" for k, v in sorted(misaligned.items()))
        )

    result["status"] = "ok"
    return result


def inspectAsrArtifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Classify an ASR JSON file (plain list or {status, segments} object)."""
    p = Path(path)
    result: dict[str, Any] = {
        "path": str(p),
        "status": "missing",
        "segments": 0,
        "duration": 0.0,
        "errors": [],
        "fingerprint": None,
    }
    if not p.exists():
        result["errors"].append("file not found")
        return result

    result["fingerprint"] = artifactFingerprint(p)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "invalid"
        result["errors"].append(f"JSON parse error: {exc}")
        return result

    if isinstance(data, list):
        segments = data
        declaredStatus = "ok"
        error = None
    elif isinstance(data, dict):
        segments = data.get("segments")
        if segments is None:
            segments = []
        declaredStatus = str(data.get("status") or "ok").lower()
        error = data.get("error")
    else:
        result["status"] = "invalid"
        result["errors"].append("top-level JSON must be a list or object")
        return result

    if not isinstance(segments, list):
        result["status"] = "invalid"
        result["errors"].append("'segments' must be an array")
        return result

    result["segments"] = len(segments)
    try:
        result["duration"] = max(
            (float(seg.get("end") or 0.0) for seg in segments if isinstance(seg, dict)),
            default=0.0,
        )
    except (TypeError, ValueError):
        result["duration"] = 0.0

    if declaredStatus == "failed":
        result["status"] = "failed"
        result["errors"].append(str(error or "ASR reported failure"))
    elif not segments:
        result["status"] = "empty"
        result["errors"].append("ASR returned no segments")
    else:
        result["status"] = "ok"
    return result
