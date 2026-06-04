# -*- coding: utf-8 -*-
"""General configuration for douyin-live-recorder.

This class holds app-wide constants and defaults.  Values are grouped by
concern — static or rarely-changed settings belong here; runtime options
(profile, sensitivity, etc.) belong in CLI args or env vars.

For log-level at runtime, use ``--log-level`` CLI arg or call
``setLogLevel()`` from ``src.log.logger``.
"""

from pathlib import Path
from datetime import datetime


class GeneralConfig:
    """Application-wide configuration values.

    Sections:
        Logging      — log path, level, rotation, retention
        Directories  — data/ and scripts/ defaults
        Douyin API   — app id, version, SDK constants
        Fingerprint  — browser UA / screen / locale for API requests
        WebSocket    — ping interval, reconnect delay
        Recording    — FLV rotation interval, health check period
    """

    # ── Logging ─────────────────────────────────────────────────
    logFilePath: str = str(Path(__file__).parent.parent.parent / "logs")
    defaultLogLevel: str = "INFO"   # Override via --log-level CLI arg
    rotation: str = "10 MB"
    retention: str = "7 days"

    # ── Output directories ──────────────────────────────────────
    dataDir: str = str(Path(__file__).parent.parent.parent / "data")
    scriptDir: str = str(Path(__file__).parent.parent.parent / "scripts")

    # ── Douyin API defaults ─────────────────────────────────────
    aid: int = 6383
    devicePlatform: str = "web"
    versionCode: str = "180800"
    webcastSdkVersion: str = "1.0.15"

    # ── Browser fingerprint (for API requests) ─────────────────
    userAgent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    screenWidth: int = 1920
    screenHeight: int = 1080
    browserLanguage: str = "zh-CN"
    browserPlatform: str = "Win32"

    # ── WebSocket defaults ──────────────────────────────────────
    wsPingInterval: int = 5      # seconds between pings
    wsReconnectDelay: int = 3    # seconds before reconnect

    # ── Recording ───────────────────────────────────────────────
    flvRotationSecs: int = 1800  # 30 minutes per FLV segment
    healthCheckInterval: int = 60  # seconds between health checks
