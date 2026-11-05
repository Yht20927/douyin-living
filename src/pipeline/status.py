# -*- coding: utf-8 -*-
"""Explicit pipeline status machine.

Replaces the old "file exists ⇒ ok" heuristic.  A step is only ``ok`` when
its output exists *and* passes validation; empty/failed/invalid outputs are
visible in ``pipeline_status.json`` instead of silently degrading scoring.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.contracts import (
    SCHEMA_VERSION,
    artifactFingerprint,
    atomicWriteJson,
    settingsHash,
    utcNowIso,
)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALID = "invalid"


def _coerceStatus(value: StepStatus | str) -> StepStatus:
    if isinstance(value, StepStatus):
        return value
    try:
        return StepStatus(str(value).lower())
    except ValueError as exc:
        raise ValueError(f"unknown step status: {value!r}") from exc


def collectEnvironment() -> dict[str, Any]:
    """Best-effort runtime description recorded with every pipeline run."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        firstLine = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        env["ffmpeg"] = firstLine or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["ffmpeg"] = None
    return env


class PipelineStatus:
    """Mutable status document for one pipeline run."""

    def __init__(
        self,
        roomId: str,
        outputDir: str | os.PathLike[str] | None = None,
        runId: str | None = None,
        profile: str | None = None,
        sensitivity: float | None = None,
    ):
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.roomId = roomId
        self.outputDir = str(outputDir) if outputDir is not None else None
        self.runId = runId or f"{now}-{uuid.uuid4().hex[:8]}"
        self.profile = profile
        self.sensitivity = sensitivity
        self.startedAt = utcNowIso()
        self.finishedAt: str | None = None
        self.settingsHash = settingsHash()
        self.environment = collectEnvironment()
        self.steps: dict[str, dict[str, Any]] = {}
        self.artifactHashes: dict[str, Any] = {}
        self.failureSummary: list[str] = []
        self._monotonic: dict[str, float] = {}

    # ── step lifecycle ───────────────────────────────────────────

    def startStep(self, name: str, inputs: list[str] | None = None) -> None:
        self._monotonic[name] = time.monotonic()
        self.steps[name] = {
            "status": StepStatus.RUNNING.value,
            "startedAt": utcNowIso(),
            "finishedAt": None,
            "elapsedMs": None,
            "inputs": list(inputs or []),
            "outputs": [],
            "artifacts": {},
            "error": None,
            "meta": {},
        }

    def finishStep(
        self,
        name: str,
        status: StepStatus | str,
        outputs: list[str] | None = None,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        coerced = _coerceStatus(status)
        record = self.steps.get(name)
        if record is None:
            self.startStep(name)

        started = self._monotonic.get(name)
        elapsedMs = int((time.monotonic() - started) * 1000) if started else None

        outputPaths = [p for p in (outputs or []) if p]
        artifacts: dict[str, Any] = {}
        for output in outputPaths:
            fingerprint = artifactFingerprint(output)
            if fingerprint is not None:
                artifacts[output] = fingerprint
                self.artifactHashes[output] = fingerprint

        record.update(
            {
                "status": coerced.value,
                "finishedAt": utcNowIso(),
                "elapsedMs": elapsedMs,
                "outputs": outputPaths,
                "artifacts": artifacts,
                "error": error,
                "meta": dict(meta or {}),
            }
        )
        if coerced in {StepStatus.FAILED, StepStatus.INVALID}:
            self.failureSummary.append(f"{name}: {error or coerced.value}")

    def setStep(
        self,
        name: str,
        status: StepStatus | str,
        outputs: list[str] | None = None,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
        elapsedMs: int | None = None,
    ) -> None:
        """Record a completed step without a preceding startStep()."""
        self.startStep(name)
        self.finishStep(name, status, outputs=outputs, error=error, meta=meta)
        if elapsedMs is not None:
            self.steps[name]["elapsedMs"] = elapsedMs

    def finish(self) -> None:
        self.finishedAt = utcNowIso()

    def hasFailures(self) -> bool:
        return bool(self.failureSummary)

    # ── serialisation ────────────────────────────────────────────

    def toDict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.runId,
            "roomId": self.roomId,
            "outputDir": self.outputDir,
            "profile": self.profile,
            "sensitivity": self.sensitivity,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "settingsHash": self.settingsHash,
            "environment": self.environment,
            "steps": self.steps,
            "artifactHashes": self.artifactHashes,
            "failureSummary": self.failureSummary,
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> str:
        if path is None:
            if self.outputDir is None:
                raise ValueError("outputDir is required when no status path is given")
            path = Path(self.outputDir) / "pipeline_status.json"
        self.finish()
        atomicWriteJson(path, self.toDict())
        return str(path)
