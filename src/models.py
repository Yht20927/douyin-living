# -*- coding: utf-8 -*-
"""Data models for douyin-live-recorder."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class RoomStatus(IntEnum):
    UNKNOWN = 0
    LIVING = 2
    ENDED = 4


class Quality:
    SD2 = "SD2"        # 540x960
    SD1 = "SD1"        # 720x1280
    HD1 = "HD1"        # 720x1280
    FULL_HD1 = "FULL_HD1"  # 1080x1920

    @classmethod
    def default(cls) -> str:
        return cls.HD1


@dataclass
class StreamInfo:
    flvUrls: dict[str, str] = field(default_factory=dict)
    hlsUrls: dict[str, str] = field(default_factory=dict)
    qualities: list[dict] = field(default_factory=list)
    defaultQuality: str = "sd"
    streamId: str = ""
    sessionId: str = ""

    def flvUrl(self, quality: str | None = None) -> str:
        q = quality or Quality.default()
        return self.flvUrls[q]

    def hlsUrl(self, quality: str | None = None) -> str:
        q = quality or Quality.default()
        return self.hlsUrls[q]


@dataclass
class RoomInfo:
    webRid: str = ""
    roomId: str = ""
    title: str = ""
    status: RoomStatus = RoomStatus.UNKNOWN
    userCount: int = 0
    anchorId: str = ""
    coverUrl: str = ""
    stream: Optional[StreamInfo] = None


@dataclass
class DanmakuMessage:
    roomId: str = ""
    msgType: str = ""
    content: str = ""
    userName: str = ""
    userId: str = ""
    rawPayload: Optional[dict] = None
    timestamp: str = ""
    extra: dict = field(default_factory=dict)
