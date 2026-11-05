# -*- coding: utf-8 -*-
"""Tests for logger module — sinks, event JSONL and level changes."""

import json
import threading
from pathlib import Path

import pytest

from src.log.events import EVENT_SCHEMA_VERSION, logEvent
from src.log.logger import getLogger, setLogLevel, LoggerManager


@pytest.fixture
def isolatedLogs(tmp_path):
    """Point both sinks at temp files and restore defaults afterwards."""
    appLog = tmp_path / "app.log"
    eventLog = tmp_path / "events.jsonl"
    LoggerManager.setup(
        logFile=appLog,
        level="INFO",
        eventLogFile=eventLog,
        force=True,
    )
    yield appLog, eventLog
    LoggerManager.setup(level="INFO", force=True)


def _readEvents(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestGetLogger:
    def test_returns_logger_with_module_name(self):
        log = getLogger("test_module")
        assert log is not None

    def test_default_name_is_main(self):
        log = getLogger()
        assert log is not None

    def test_multiple_calls_return_different_bindings(self):
        log1 = getLogger("mod1")
        log2 = getLogger("mod2")
        assert log1 is not None
        assert log2 is not None


class TestSetLogLevel:
    def test_valid_levels(self):
        """All valid log levels should be accepted without error."""
        for level in ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            setLogLevel(level)

    def test_concurrent_access(self):
        """setLogLevel should be safe under concurrent access."""
        errors = []

        def _setLevel(level):
            try:
                setLogLevel(level)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_setLevel, args=(lvl,))
            for lvl in ["DEBUG", "INFO", "WARNING"] * 3
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors during concurrent setLogLevel: {errors}"

    def test_restores_info_after_test(self):
        """Restore INFO level so subsequent tests get expected output."""
        setLogLevel("INFO")


class TestLoggerManager:
    def test_initialized_flag(self):
        """LoggerManager should be initialized after import."""
        assert LoggerManager._initialized


class TestEventLog:
    def test_event_written_to_jsonl_only(self, isolatedLogs):
        appLog, eventLog = isolatedLogs

        logEvent(
            "pipeline.step",
            runId="run-1",
            roomId="room-1",
            step="2_asr",
            status="ok",
            elapsedMs=42,
            cacheReason="cache hit",
            outputPath=Path("data/room/out.json"),
        )
        getLogger("human").info("ordinary message")

        events = _readEvents(eventLog)
        assert len(events) == 1
        record = events[0]
        assert record["schemaVersion"] == EVENT_SCHEMA_VERSION
        assert record["event"] == "pipeline.step"
        assert record["runId"] == "run-1"
        assert record["roomId"] == "room-1"
        assert record["step"] == "2_asr"
        assert record["status"] == "ok"
        assert record["elapsedMs"] == 42
        assert record["cacheReason"] == "cache hit"
        assert record["outputPath"] == "data/room/out.json"
        assert record["ts"]

        human = appLog.read_text(encoding="utf-8")
        assert "ordinary message" in human
        # Event payloads must not leak into the human-readable sinks.
        assert "pipeline.step" not in human

    def test_event_sink_survives_set_log_level(self, isolatedLogs):
        _, eventLog = isolatedLogs
        logEvent("before.level", runId="run-1", roomId="room-1")
        setLogLevel("DEBUG")
        logEvent("after.level", runId="run-1", roomId="room-1")

        events = _readEvents(eventLog)
        assert [event["event"] for event in events] == ["before.level", "after.level"]

    def test_invalid_level_keeps_existing_sinks(self, isolatedLogs):
        _, eventLog = isolatedLogs
        with pytest.raises(ValueError, match="unknown log level"):
            setLogLevel("NOT-A-LEVEL")

        logEvent("still.works", runId="run-1", roomId="room-1")
        events = _readEvents(eventLog)
        assert events[-1]["event"] == "still.works"

    def test_empty_event_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            logEvent("")

    def test_core_fields_cannot_be_overridden(self, isolatedLogs):
        _, eventLog = isolatedLogs
        logEvent(
            "pipeline.finish",
            runId="run-1",
            roomId="room-1",
            elapsedMs="12",
            **{"ts": "fake", "schemaVersion": "9.9"},
        )
        record = _readEvents(eventLog)[-1]
        assert record["event"] == "pipeline.finish"
        assert record["schemaVersion"] == EVENT_SCHEMA_VERSION
        assert record["ts"] != "fake"
        assert record["elapsedMs"] == 12
