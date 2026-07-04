# -*- coding: utf-8 -*-
"""Unit tests for Params — query parameter builder."""

import pytest
from src.params import Params


class TestParams:
    def test_empty_params(self):
        p = Params()
        assert p.toDict() == {}
        assert p.build() == ""

    def test_add_single_param(self):
        p = Params()
        p.add("key", "value")
        assert p.toDict() == {"key": "value"}

    def test_add_all(self):
        p = Params()
        p.addAll({"a": "1", "b": "2"})
        assert p.toDict()["a"] == "1"
        assert p.toDict()["b"] == "2"

    def test_build_returns_encoded_string(self):
        p = Params()
        p.add("room_id", "123")
        p.add("name", "测试")
        result = p.build()
        assert "room_id=123" in result
        # Chinese characters should be percent-encoded
        assert "测试" not in result or "%" in result

    def test_toDict_returns_copy(self):
        """toDict() should not expose internal state for mutation."""
        p = Params()
        p.add("a", "1")
        d = p.toDict()
        d["b"] = "2"
        # Original params unchanged
        assert "b" not in p.toDict()

    def test_fluent_chaining(self):
        p = Params()
        result = p.add("a", "1").add("b", "2")
        assert result is p  # returns self
        assert len(p.toDict()) == 2

    def test_withPlatform_adds_expected_keys(self):
        p = Params()
        p.withPlatform()
        d = p.toDict()
        assert d["device_platform"] == "webapp"
        assert d["aid"] == "6383"
        assert "browser_name" in d

    def test_withWebId_generates_if_not_provided(self):
        p = Params()
        p.withWebId()
        assert "webid" in p.toDict()
        assert len(p.toDict()["webid"]) == 19

    def test_withMsToken_generates_if_not_provided(self):
        p = Params()
        p.withMsToken()
        assert "msToken" in p.toDict()
        assert len(p.toDict()["msToken"]) == 107
