# -*- coding: utf-8 -*-
"""Runtime config reload — thin wrapper around settings.load_settings."""

from src.config.settings import load_settings

__all__ = ["reload_settings"]


def reload_settings():
    """Clear the settings cache so the next access re-reads from disk."""
    load_settings.cache_clear()
