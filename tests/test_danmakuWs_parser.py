# -*- coding: utf-8 -*-
"""Unit tests for DanmakuWs._parseMessage — protobuf message parsing."""

import pytest
from unittest.mock import MagicMock, patch
from src.danmakuWs import DanmakuWs


class TestParseMessage:
    """Test _parseMessage with mock protobuf message objects."""

    def _make_ws(self):
        """Create a minimal DanmakuWs instance for testing."""
        return DanmakuWs(
            params={"test": "1"},
            cookieStr="test=1",
            onDanmaku=None,
        )

    def _make_msg(self, method, parsed_obj):
        """Create a mock protobuf message."""
        msg = MagicMock()
        msg.method = method
        msg.payload = b"fake"
        return msg, parsed_obj

    def test_chat_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastChatMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.ChatMessage") as MockChat:
            mockChat = MagicMock()
            mockChat.content = "hello"
            mockChat.user.nickname = "user1"
            mockChat.user.id = "123"
            mockChat.user.sec_uid = "sec123"
            mockChat.HasField.return_value = True
            MockChat.return_value = mockChat

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "chat"
            assert result["content"] == "hello"
            assert result["userName"] == "user1"
            assert result["userId"] == "123"

    def test_gift_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastGiftMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.GiftMessage") as MockGift:
            mockGift = MagicMock()
            mockGift.gift.name = "rose"
            mockGift.comboCount = 5
            mockGift.user.nickname = "sender"
            mockGift.toUser.nickname = "receiver"
            mockGift.HasField.side_effect = lambda field: field in ("gift", "user", "toUser")
            MockGift.return_value = mockGift

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "gift"
            assert result["giftName"] == "rose"
            assert result["comboCount"] == 5
            assert result["userName"] == "sender"
            assert result["toUserName"] == "receiver"

    def test_like_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastLikeMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.LikeMessage") as MockLike:
            mockLike = MagicMock()
            mockLike.count = 10
            mockLike.total = 100
            mockLike.user.nickname = "liker"
            mockLike.HasField.return_value = True
            MockLike.return_value = mockLike

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "like"
            assert result["count"] == 10
            assert result["total"] == 100
            assert result["userName"] == "liker"

    def test_member_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastMemberMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.MemberMessage") as MockMember:
            mockMember = MagicMock()
            mockMember.user.nickname = "newcomer"
            mockMember.memberCount = 42
            mockMember.HasField.return_value = True
            MockMember.return_value = mockMember

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "member"
            assert result["userName"] == "newcomer"
            assert result["memberCount"] == 42

    def test_social_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastSocialMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.SocialMessage") as MockSocial:
            mockSocial = MagicMock()
            mockSocial.user.nickname = "sharer"
            mockSocial.action = "share"
            mockSocial.HasField.return_value = True
            MockSocial.return_value = mockSocial

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "social"
            assert result["userName"] == "sharer"
            assert result["action"] == "share"

    def test_room_stats_message(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastRoomStatsMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.RoomStatsMessage") as MockStats:
            mockStats = MagicMock()
            mockStats.displayLong = "100 viewers"
            mockStats.displayShort = "100"
            MockStats.return_value = mockStats

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["type"] == "roomStats"
            assert result["displayLong"] == "100 viewers"
            assert result["displayShort"] == "100"

    def test_unknown_method(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "SomeUnknownMethod"
        msg.payload = b""

        result = ws._parseMessage(msg)
        assert result is not None
        assert result["type"] == "unknown"

    def test_parse_error_returns_none(self):
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastChatMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.ChatMessage") as MockChat:
            MockChat.side_effect = Exception("parse error")
            result = ws._parseMessage(msg)
            assert result is None

    def test_null_user_guarded(self):
        """When user sub-message is absent, should not crash."""
        ws = self._make_ws()
        msg = MagicMock()
        msg.method = "WebcastChatMessage"
        msg.payload = b""

        with patch("src.danmakuWs.LivePb.ChatMessage") as MockChat:
            mockChat = MagicMock()
            mockChat.content = "hello"
            mockChat.HasField.return_value = False  # No user field
            MockChat.return_value = mockChat

            result = ws._parseMessage(msg)
            assert result is not None
            assert result["userName"] == ""
            assert result["userId"] == ""
