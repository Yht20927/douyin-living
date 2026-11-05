# -*- coding: utf-8 -*-
"""Artifact cache — sidecar ``*.meta.json`` checks for pipeline stages.

A pipeline step may only be skipped when *all* of the following still match:

* the output artifact and its sidecar metadata exist and are readable;
* the artifact itself was not modified after the metadata was written;
* every declared source file has the same fingerprint;
* the config hash, step parameters and generator version are unchanged.

Any mismatch is a cache miss; the caller rebuilds the artifact and records a
fresh sidecar.  This replaces the old "file exists ⇒ reuse" heuristic, which
silently reused stale audio/ASR/features after configuration changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.contracts import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    ArtifactMeta,
    artifactFingerprint,
    artifactMetaPath,
    fingerprintMatches,
    hashJson,
    settingsHash,
)
from src.log.logger import getLogger

log = getLogger(__name__)


@dataclass
class CacheDecision:
    """Result of an artifact cache lookup."""

    hit: bool
    reason: str
    meta: ArtifactMeta | None = None


class ArtifactCache:
    """Read/write artifact metadata with an in-process fingerprint cache."""

    def __init__(
        self,
        settingsHashValue: str | None = None,
        generatorVersion: str = GENERATOR_VERSION,
    ):
        self.settingsHashValue = settingsHashValue or settingsHash()
        self.generatorVersion = generatorVersion
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._fingerprints: dict[tuple[str, int, int], dict[str, Any] | None] = {}

    # ── fingerprints ─────────────────────────────────────────────

    @staticmethod
    def _normalise(path: str | os.PathLike[str]) -> str:
        return os.path.abspath(os.fspath(path))

    def fingerprint(self, path: str | os.PathLike[str]) -> dict[str, Any] | None:
        """Fingerprint a file once per (path, size, mtime) tuple.

        Hashing a large recording more than once per run is wasteful; the
        key includes size/mtime so an in-place update invalidates the entry.
        """
        normalised = self._normalise(path)
        try:
            stat = Path(normalised).stat()
        except OSError:
            return None
        key = (normalised, stat.st_size, stat.st_mtime_ns)
        if key in self._fingerprints:
            return self._fingerprints[key]
        fingerprint = artifactFingerprint(normalised)
        self._fingerprints[key] = fingerprint
        return fingerprint

    def _sourceHashes(self, inputs: list[str]) -> dict[str, dict[str, Any]]:
        hashes: dict[str, dict[str, Any]] = {}
        for source in inputs:
            fingerprint = self.fingerprint(source)
            if fingerprint is not None:
                hashes[self._normalise(source)] = fingerprint
        return hashes

    # ── lookup / record ──────────────────────────────────────────

    def check(
        self,
        artifactPath: str | os.PathLike[str],
        step: str,
        inputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
        validator: Callable[[], tuple[bool, str]] | None = None,
    ) -> CacheDecision:
        """Return whether ``artifactPath`` can be reused for ``step``.

        ``validator`` runs after metadata matches but before the lookup is
        counted as a hit; callers use it for artifact-specific schema checks
        (e.g. "feature JSON is non-empty").  It must return ``(ok, reason)``.
        """
        artifact = Path(artifactPath)
        if not artifact.exists():
            return self._miss(f"artifact missing: {artifactPath}")

        metaPath = artifactMetaPath(artifactPath)
        meta = ArtifactMeta.load(metaPath)
        if meta is None:
            return self._miss(f"metadata missing or unreadable: {metaPath}")

        if meta.schemaVersion != SCHEMA_VERSION:
            return self._miss(
                f"schema changed: {meta.schemaVersion} != {SCHEMA_VERSION}"
            )
        if meta.generatorVersion != self.generatorVersion:
            return self._miss(
                f"generator changed: {meta.generatorVersion} != {self.generatorVersion}"
            )
        if meta.step != step:
            return self._miss(f"step changed: {meta.step} != {step}")
        if meta.settingsHash != self.settingsHashValue:
            return self._miss("settings hash changed")
        if meta.paramsHash != hashJson(params or {}):
            return self._miss("step parameters changed")

        if not fingerprintMatches(meta.artifact, self.fingerprint(artifactPath)):
            return self._miss("artifact modified after metadata was written")

        expectedInputs = {self._normalise(p) for p in (inputs or [])}
        recordedInputs = set(meta.sourceHashes)
        if recordedInputs != expectedInputs:
            missing = sorted(expectedInputs - recordedInputs)
            extra = sorted(recordedInputs - expectedInputs)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            return self._miss("source set changed (" + "; ".join(detail) + ")")

        for source in expectedInputs:
            current = self.fingerprint(source)
            if current is None:
                return self._miss(f"source missing: {source}")
            if not fingerprintMatches(meta.sourceHashes[source], current):
                return self._miss(f"source changed: {source}")

        if validator is not None:
            ok, reason = validator()
            if not ok:
                return self._miss(reason)

        self.hits += 1
        return CacheDecision(True, "cache hit", meta)

    def record(
        self,
        artifactPath: str | os.PathLike[str],
        step: str,
        inputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
        elapsedMs: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ArtifactMeta:
        """Write a fresh sidecar for a just-produced artifact."""
        resolvedParams = dict(params or {})
        meta = ArtifactMeta(
            artifactPath=self._normalise(artifactPath),
            step=step,
            sourceHashes=self._sourceHashes(list(inputs or [])),
            settingsHash=self.settingsHashValue,
            paramsHash=hashJson(resolvedParams),
            params=resolvedParams,
            artifact=self.fingerprint(artifactPath) or {},
            elapsedMs=elapsedMs,
            generatorVersion=self.generatorVersion,
            extra=dict(extra or {}),
        )
        meta.save(artifactMetaPath(artifactPath))
        self.writes += 1
        return meta

    def invalidate(self, artifactPath: str | os.PathLike[str]) -> None:
        """Drop a sidecar before rebuilding so a failed run cannot look fresh."""
        try:
            os.remove(artifactMetaPath(artifactPath))
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("Failed to remove artifact metadata: %s", artifactPath)

    # ── helpers ──────────────────────────────────────────────────

    def markMiss(self, reason: str) -> CacheDecision:
        """Record a caller-side cache miss (e.g. artifact missing)."""
        return self._miss(reason)

    def _miss(self, reason: str) -> CacheDecision:
        self.misses += 1
        return CacheDecision(False, reason)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
