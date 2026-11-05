# -*- coding: utf-8 -*-
"""Runtime config reload — thin wrapper around settings.load_settings.

Important: most pipeline modules snapshot their section at import time
(``_cfg = load_settings().scorer``).  Clearing the cache is therefore **not**
enough to make a running process pick up YAML changes — restart the process
after editing ``config/*.yaml``.  Environment variables are read when the
settings object is constructed, so they must also be set before start-up.
"""

import os

from src.config.settings import load_settings

__all__ = ["reload_settings"]


def reload_settings():
    """Clear the settings cache so subsequent ``load_settings()`` calls reload.

    Existing module-level snapshots keep their old values until restart.
    If ``DOUYIN_CONFIG_STRICT_RELOAD`` is set, refuse to pretend this is a
    hot reload so callers notice the limitation instead of silently mixing
    old and new configuration.
    """
    if os.environ.get("DOUYIN_CONFIG_STRICT_RELOAD", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            "Configuration is snapshotted at import time; restart the process "
            "to apply config changes."
        )
    load_settings.cache_clear()
    return load_settings()
