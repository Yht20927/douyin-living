# -*- coding: utf-8 -*-
"""Unit tests for src.util — formatting + cookie + feature validation."""

import pytest

from src.util import (
    fmtSize,
    fmtSrtTime,
    transCookies,
    validateFeatures,
    FEATURE_NAMES,
)


class TestFmtSize:
    def test_bytes(self):
        assert fmtSize(0) == "0 B"
        assert fmtSize(512) == "512 B"

    def test_kilobytes(self):
        assert fmtSize(1024) == "1.0 KB"
        assert fmtSize(2048) == "2.0 KB"

    def test_megabytes(self):
        assert fmtSize(1024 * 1024) == "1.0 MB"
        assert fmtSize(int(1.5 * 1024 * 1024)) == "1.5 MB"


class TestFmtSrtTime:
    def test_zero(self):
        assert fmtSrtTime(0.0) == "00:00:00,000"

    def test_subsecond(self):
        assert fmtSrtTime(0.5) == "00:00:00,500"

    def test_hours(self):
        # 1 h 2 m 3.456 s
        assert fmtSrtTime(3723.456) == "01:02:03,456"

    def test_round_trip_format(self):
        # Format must always be HH:MM:SS,mmm (12 chars, comma at index 8)
        s = fmtSrtTime(125.789)
        assert len(s) == 12
        assert s[8] == ","


class TestTransCookies:
    def test_standard_separator(self):
        assert transCookies("a=1; b=2") == {"a": "1", "b": "2"}

    def test_no_space_separator(self):
        # Browser exports often use ";" with no space
        assert transCookies("a=1;b=2;c=3") == {"a": "1", "b": "2", "c": "3"}

    def test_mixed_separators(self):
        assert transCookies("a=1;b=2; c=3") == {"a": "1", "b": "2", "c": "3"}

    def test_value_with_equals(self):
        # JWT-like values containing = should keep everything after the first =
        assert transCookies("token=abc=def==") == {"token": "abc=def=="}

    def test_empty_returns_empty(self):
        assert transCookies("") == {}
        assert transCookies("   ") == {}

    def test_skips_malformed_segments(self):
        # Items with no '=' should be silently skipped, not crash
        assert transCookies("a=1; junk; b=2") == {"a": "1", "b": "2"}


class TestValidateFeatures:
    def test_all_known(self):
        unknown = validateFeatures({"rms": 0, "dmDensity": 0})
        assert unknown == []

    def test_unknown_reported(self):
        unknown = validateFeatures({"rms": 0, "totallyMadeUp": 0})
        assert unknown == ["totallyMadeUp"]

    def test_canonical_names_includes_core(self):
        # Anchor a few core names so accidental removal triggers a test failure
        for name in ("rms", "dmDensity", "asrKeyword", "sceneChange"):
            assert name in FEATURE_NAMES
