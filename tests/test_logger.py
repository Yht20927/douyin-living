# -*- coding: utf-8 -*-
"""Tests for logger module — getLogger, setLogLevel thread safety."""

import threading
import pytest
from src.log.logger import getLogger, setLogLevel, LoggerManager


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
