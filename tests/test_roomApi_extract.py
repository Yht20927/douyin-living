# -*- coding: utf-8 -*-
"""Unit tests for _extractRoomInfo — SSR HTML parsing."""

import pytest
from src.roomApi import _extractRoomInfo


class TestExtractRoomInfo:
    def test_empty_html(self):
        result = _extractRoomInfo("", "12345")
        assert result["roomId"] == ""
        assert result["userId"] == ""

    def test_no_script_tags(self):
        html = "<html><body>No scripts here</body></html>"
        result = _extractRoomInfo(html, "12345")
        assert result["roomId"] == ""

    def test_extracts_room_id(self):
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"987654321\",\"status\":2}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["roomId"] == "987654321"

    def test_extracts_user_id(self):
        # The regex looks for \\\" in the raw HTML (doubly-escaped in SSR).
        # The _extractRoomInfo function skips scripts that don't contain
        # 'roomId', so every test fixture must include a roomId.
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"987654321\",\"user_unique_id\":\"111222333\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["userId"] == "111222333"

    def test_extracts_room_status_and_title(self):
        html = (
            '<script nonce="abc">'
            r'\"roomId\":\"123\",\"roomInfo\":{\"room\":{\"id_str\":\"123\",\"status\":2,\"status_str\":\"LIVE\",\"title\":\"测试直播间\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["roomStatus"] == "2"
        assert result["roomTitle"] == "测试直播间"

    def test_extracts_sec_uid(self):
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"987654321\",\"sec_uid\":\"abc-def-ghi\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["secUid"] == "abc-def-ghi"

    def test_extracts_anchor_id(self):
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"987654321\",\"anchor\":{\"id_str\":\"444555666\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["anchorId"] == "444555666"

    def test_multiple_scripts_finds_first_with_roomId(self):
        html = (
            '<script>irrelevant</script>'
            '<script nonce="abc">'
            r'{\"roomId\":\"111\",\"user_unique_id\":\"222\"}'
            '</script>'
            '<script nonce="def">'
            r'{\"roomId\":\"333\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        # Should use the first script that has a roomId
        assert result["roomId"] == "111"

    def test_no_nonce_fallback(self):
        """When no nonce scripts, falls back to all script tags."""
        html = (
            '<script>'
            r'{\"roomId\":\"999\",\"user_unique_id\":\"888\"}'
            '</script>'
        )
        result = _extractRoomInfo(html, "12345")
        assert result["roomId"] == "999"
        assert result["userId"] == "888"

    def test_malformed_script_does_not_crash(self):
        """Invalid JSON/escaping in script should not crash."""
        html = '<script nonce="abc">garbage{{{[[[</script>'
        result = _extractRoomInfo(html, "12345")
        assert isinstance(result, dict)
