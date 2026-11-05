# -*- coding: utf-8 -*-
"""Structured pipeline events for ``logs/events.jsonl``.

Each :func:`logEvent` call emits exactly one JSON object.  The event sink is
installed by :mod:`src.log.logger` and only accepts records tagged through
``extra["event"]``, so these entries never pollute the human-readable console
or ``*.log`` files.

Core schema (always present, ``null`` when not applicable)::

    ts, schemaVersion, event, runId, roomId, step, status, elapsedMs

Additional keyword arguments are merged into the object as extra fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.log.logger import logger

EVENT_SCHEMA_VERSION = "1.0"

_CORE_FIELDS = {
    "ts",
    "schemaVersion",
    "event",
    "runId",
    "roomId",
    "step",
    "status",
    "elapsedMs",
}


def _utcNowIso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonSafe(value: Any) -> Any:
    """Convert common Python values into JSON-serialisable ones."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonSafe(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonSafe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonSafe(item) for item in value]
    return str(value)


def _asText(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    return str(value)


def logEvent(
    event: str,
    *,
    runId: str | None = None,
    roomId: str | None = None,
    step: str | None = None,
    status: str | None = None,
    elapsedMs: int | None = None,
    level: str = "INFO",
    **fields: Any,
) -> dict[str, Any]:
    """Write one structured event to the JSONL sink.

    Args:
        event: Dot-separated event name, e.g. ``pipeline.step``.
        runId: Pipeline run identifier (``PipelineStatus.runId``).
        roomId: Douyin room identifier.
        step: Pipeline step name when the event describes one step.
        status: Step/run status (``ok``/``failed``/``skipped``/...).
        elapsedMs: Duration in milliseconds, when known.
        level: loguru level used for filtering; event sink accepts all levels.
        **fields: Extra context such as ``cacheReason`` or ``error``.

    Returns:
        The serialised payload (useful for tests and debugging).
    """
    if not event:
        raise ValueError("event name must be non-empty")

    payload: dict[str, Any] = {
        "ts": _utcNowIso(),
        "schemaVersion": EVENT_SCHEMA_VERSION,
        "event": str(event),
        "runId": _asText(runId),
        "roomId": _asText(roomId),
        "step": _asText(step),
        "status": _asText(status),
        "elapsedMs": _safeElapsed(elapsedMs),
    }
    for key, value in fields.items():
        if key in _CORE_FIELDS:
            continue
        payload[key] = _jsonSafe(value)

    # The sink writes ``{message}``, so the JSON object must be the message.
    logger.bind(**payload).log(
        str(level).strip().upper(),
        json.dumps(payload, ensure_ascii=False),
    )
    return payload


def _safeElapsed(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
