# -*- coding: utf-8 -*-
"""Douyin room parser tests — SSR JSON, URL rendering and payload mapping."""

import json
from urllib.parse import quote

from src.douyin.room_parser import (
    extractRoomInfo,
    extractStreamUrls,
    extractStreamUrlsFromJson,
)


class TestExtractRoomInfo:
    def test_render_data_url_encoded(self):
        payload = {
            "roomInfo": {
                "room": {"id_str": "123", "status": 2, "title": "测试直播间"}
            },
            "user_unique_id": "user-1",
            "anchor": {"id_str": "anchor-1", "sec_uid": "sec-1"},
        }
        encoded = quote(json.dumps(payload, ensure_ascii=False), safe="")
        html = (
            '<script id="RENDER_DATA" type="application/json">'
            f"{encoded}</script>"
        )

        info = extractRoomInfo(html, "web-1")

        assert info["roomId"] == "123"
        assert info["userId"] == "user-1"
        assert info["anchorId"] == "anchor-1"
        assert info["secUid"] == "sec-1"
        assert info["roomStatus"] == "2"
        assert info["roomTitle"] == "测试直播间"

    def test_initial_state_json(self):
        payload = {
            "roomInfo": {
                "room": {"id_str": "456", "status": 2, "title": "Initial"}
            },
            "user_unique_id": "u-2",
            "anchor": {"id_str": "a-2"},
        }
        html = (
            "<script>window.__INITIAL_STATE__ = "
            f"{json.dumps(payload, ensure_ascii=False)};</script>"
        )

        info = extractRoomInfo(html, "web-2")

        assert info["roomId"] == "456"
        assert info["userId"] == "u-2"
        assert info["anchorId"] == "a-2"
        assert info["roomTitle"] == "Initial"

    def test_escaped_nonce_payload(self):
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"789\",\"user_unique_id\":\"u-3\",'
            r'\"roomInfo\":{\"room\":{\"id_str\":\"789\",\"status\":2,'
            r'\"title\":\"Escaped\"}},\"anchor\":{\"id_str\":\"a-3\"}}'
            "</script>"
        )

        info = extractRoomInfo(html, "web-3")

        assert info["roomId"] == "789"
        assert info["userId"] == "u-3"
        assert info["anchorId"] == "a-3"
        assert info["roomStatus"] == "2"
        assert info["roomTitle"] == "Escaped"

    def test_partial_payload_uses_regex_fallback(self):
        html = (
            '<script nonce="abc">'
            r'\"roomId\":\"999\",\"roomInfo\":{\"room\":{\"id_str\":\"999\",'
            r'\"status\":2,\"status_str\":\"LIVE\",\"title\":\"碎片\"}'
            "</script>"
        )

        info = extractRoomInfo(html, "web-4")

        assert info["roomId"] == "999"
        assert info["roomStatus"] == "2"
        assert info["roomTitle"] == "碎片"

    def test_empty_html_returns_defaults(self):
        info = extractRoomInfo("", "web-5")
        assert info["roomId"] == ""
        assert info["roomTitle"] == ""


class TestExtractStreamUrls:
    def test_urls_from_json_payload(self):
        payload = {
            "roomInfo": {"room": {"id_str": "1"}},
            "stream": (
                "https://pull-flv-cdn.example.com/third/stream-123_hd.flv"
                "?expire=1&sign=abc"
            ),
        }
        html = f"<script>{json.dumps(payload)}</script>"

        urls = extractStreamUrls(html)

        assert urls == {
            "hd": (
                "https://pull-flv-cdn.example.com/third/stream-123_hd.flv"
                "?expire=1&sign=abc"
            )
        }

    def test_escaped_url_regex_fallback(self):
        html = (
            '<script nonce="abc">'
            r'var u = "https:\/\/pull-flv-cdn.example.com\/third\/stream-1_or4.flv'
            r'?expire=1\u0026sign=abc";'
            "</script>"
        )

        urls = extractStreamUrls(html)

        assert set(urls) == {"or4"}
        assert "sign=abc" in urls["or4"]

    def test_no_pull_flv_urls_returns_empty(self):
        assert extractStreamUrls("<html><body>nothing</body></html>") == {}


class TestExtractStreamUrlsFromJson:
    def test_enter_room_payload_quality_mapping(self):
        payload = {
            "data": {
                "data": [
                    {
                        "stream_url": {
                            "flv_pull_url": {
                                "FULL_HD1": "https://pull-flv/x_or4.flv?sign=1",
                                "HD1": "https://pull-flv/x_hd.flv?sign=2",
                                "SD1": "https://pull-flv/x_sd.flv?sign=3",
                            }
                        }
                    }
                ]
            }
        }

        urls = extractStreamUrlsFromJson(payload)

        assert urls["or4"].endswith("sign=1")
        assert urls["hd"].endswith("sign=2")
        assert urls["sd"].endswith("sign=3")

    def test_quality_inferred_from_url_suffix(self):
        payload = {"flv_pull_url": {"UNKNOWN": "https://pull-flv/x_ld.flv?sign=4"}}
        urls = extractStreamUrlsFromJson(payload)
        assert urls == {"ld": "https://pull-flv/x_ld.flv?sign=4"}

    def test_non_flv_payload_returns_empty(self):
        assert extractStreamUrlsFromJson({"data": {"room": {"id_str": "1"}}}) == {}
