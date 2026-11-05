# -*- coding: utf-8 -*-
"""Loguru logging setup for the recorder and clipping pipeline.

Two sink families are managed here:

* human-readable logs — console plus a rotating ``logs/*.log`` file;
* machine-readable events — ``logs/events.jsonl``, one JSON object per line.

Event records are tagged through ``extra["event"]`` by
:func:`src.log.events.logEvent` and are filtered out of the human sinks.  The
same sink-registration path is used by :meth:`LoggerManager.setup` and
:func:`setLogLevel`, so changing the level at runtime never drops the event
stream.  Rotation and retention values come from
:class:`src.log.generalConfig.GeneralConfig`.
"""

import logging
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from src.log.generalConfig import GeneralConfig

# --- configuration ---------------------------------------------------
LOG_DIR = Path(GeneralConfig.logFilePath)
DEFAULT_LOG_LEVEL = GeneralConfig.defaultLogLevel
LOG_FILE_PATH = ""
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<lvl>{level: <10}</lvl> | "
    "<cyan>{extra[module_name]:<15}</cyan>:<cyan>{function:<18}</cyan>:<yellow>{line:<5}</yellow> "
    "- <lvl>{message}</lvl>"
)

# Protect logger reconfiguration from concurrent access.
_logLock = threading.Lock()


def _isEventRecord(record: dict[str, Any]) -> bool:
    """Only structured events reach ``events.jsonl``."""
    return bool(record["extra"].get("event"))


def _isHumanRecord(record: dict[str, Any]) -> bool:
    """Keep structured events out of the console/file logs."""
    return not record["extra"].get("event")


class LoggerManager:
    """Owns the loguru sink configuration and supports level changes."""

    _initialized = False
    _level: str = DEFAULT_LOG_LEVEL
    _logPath: Path | None = None
    _eventLogPath: Path | None = None
    _rotation: str = GeneralConfig.logRotation
    _retention: str = GeneralConfig.logRetention

    @classmethod
    def setup(
        cls,
        logFile: Optional[Union[str, Path]] = None,
        level: str = DEFAULT_LOG_LEVEL,
        rotation: str = GeneralConfig.logRotation,
        retention: str = GeneralConfig.logRetention,
        eventLogFile: Optional[Union[str, Path]] = None,
        force: bool = False,
    ) -> None:
        """Configure console, file and event sinks.

        Args:
            logFile: Explicit human log path; defaults to a timestamped file
                under ``logs/``.
            level: Minimum level for console/file sinks.
            rotation: loguru rotation policy (size or daily ``"00:00"``).
            retention: How long rotated log files are kept.
            eventLogFile: Explicit JSONL event path; defaults to
                ``logs/events.jsonl``.
            force: Reconfigure even when ``setup()`` already ran (tests and
                runtime level changes).
        """
        with _logLock:
            if cls._initialized and not force:
                return

            resolvedLevel = cls._normaliseLevel(level)
            logger.remove()
            cls._installLevelIcons()

            cls._level = resolvedLevel
            cls._rotation = rotation
            cls._retention = retention

            logPath = Path(logFile) if logFile else LOG_DIR / (
                f"{datetime.now().strftime('%Y-%m-%d-')}-{uuid.uuid4().hex}.log"
            )
            logPath.parent.mkdir(parents=True, exist_ok=True)

            global LOG_FILE_PATH
            LOG_FILE_PATH = str(logPath)
            cls._logPath = logPath
            cls._eventLogPath = (
                Path(eventLogFile)
                if eventLogFile is not None
                else LOG_DIR / GeneralConfig.eventLogFileName
            )

            cls._registerSinks(resolvedLevel)
            cls._interceptStandardLogging()

            cls._initialized = True
            logger.info(
                f"日志系统初始化完成，日志级别: {resolvedLevel}, 日志文件: {logPath}"
            )

    @classmethod
    def _registerSinks(cls, level: str) -> None:
        """Register console, file and event sinks with the given level.

        Called under ``_logLock`` from both :meth:`setup` and
        :func:`setLogLevel`.  Keeping a single registration path prevents the
        event sink from disappearing after a runtime level change.
        """
        logger.add(
            sink=sys.stdout,
            format=cls._getFormat,
            colorize=True,
            level=level,
            filter=_isHumanRecord,
            backtrace=True,
            diagnose=True,
        )

        if cls._logPath is not None:
            logger.add(
                sink=cls._logPath,
                format=cls._getFileFormat,
                level=level,
                filter=_isHumanRecord,
                rotation=cls._rotation,
                retention=cls._retention,
                compression="zip",
                encoding="utf-8",
                enqueue=False,  # avoid multiprocessing queue issues
            )

        if cls._eventLogPath is None:
            return
        try:
            cls._eventLogPath.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                sink=cls._eventLogPath,
                format="{message}",
                level="TRACE",
                filter=_isEventRecord,
                rotation=GeneralConfig.eventLogRotation,
                retention=GeneralConfig.eventLogRetention,
                compression="zip",
                encoding="utf-8",
                enqueue=False,
            )
        except OSError as exc:
            print(
                f"[logger] event log disabled ({cls._eventLogPath}): {exc}",
                file=sys.stderr,
            )

    @staticmethod
    def _normaliseLevel(level: str) -> str:
        normalised = str(level or DEFAULT_LOG_LEVEL).strip().upper()
        try:
            logger.level(normalised)
        except ValueError as exc:
            raise ValueError(f"unknown log level: {level!r}") from exc
        return normalised

    @staticmethod
    def _installLevelIcons() -> None:
        """Attach icons used by the console formatter."""
        logger.level("TRACE", icon="🔍")
        logger.level("DEBUG", icon="🐛")
        logger.level("INFO", icon="ℹ️")
        logger.level("SUCCESS", icon="✅")
        logger.level("WARNING", icon="⚠️")
        logger.level("ERROR", icon="❌")
        logger.level("CRITICAL", icon="🔥")

    @staticmethod
    def _getFormat(record):
        """Build the console format, truncating noisy fields."""
        if "module_name" not in record["extra"]:
            record["extra"]["module_name"] = record["name"]

        icon = record["level"].icon
        maxModuleNameLength = 15
        maxFunctionLength = 18

        moduleName = record["extra"].get("module_name", "")
        if len(moduleName) > maxModuleNameLength:
            record["extra"]["module_name"] = moduleName[:maxModuleNameLength - 3] + "..."

        function = record.get("function", "")
        if len(function) > maxFunctionLength:
            record["function"] = function[:maxFunctionLength - 3] + "..."

        return (
            "<bold><green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green></bold> "
            "<bold>│</bold> "
            f"<lvl>{icon} {{level: <8}}</lvl> "
            "<bold>│</bold> "
            "<cyan>{extra[module_name]:<15}</cyan>:<cyan>{function:<18}</cyan>:<yellow>{line:<10}</yellow> "
            "<bold>│</bold> "
            "<lvl>{message}</lvl>\n{exception}"
        )

    @staticmethod
    def _getFileFormat(record):
        """Build the plain-text file format (no ANSI colours)."""
        if "module_name" not in record["extra"]:
            record["extra"]["module_name"] = record["name"]

        maxModuleNameLength = 15
        maxFunctionLength = 18

        moduleName = record["extra"].get("module_name", "")
        if len(moduleName) > maxModuleNameLength:
            record["extra"]["module_name"] = moduleName[:maxModuleNameLength - 3] + "..."

        function = record.get("function", "")
        if len(function) > maxFunctionLength:
            record["function"] = function[:maxFunctionLength - 3] + "..."

        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <10} | "
            "{extra[module_name]:<15}:{function:<18}:{line:<10} | "
            "{message}\n{exception}"
        )

    @staticmethod
    def _interceptStandardLogging() -> None:
        """Route standard-library warnings through loguru, filtering noise."""

        FILTERED_LOGGERS = {
            'huggingface_hub',
            'urllib3',
            'requests',
            'httpx',
            'httpcore',
            'filelock',
            'PIL',
            'matplotlib',
            'tensorflow',
            'torch',
        }

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                if record.name in FILTERED_LOGGERS or any(
                    record.name.startswith(f"{prefix}.") for prefix in FILTERED_LOGGERS
                ):
                    return

                message = record.getMessage()
                if 'HTTP Request:' in message or 'HEAD' in message or 'GET' in message:
                    return

                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                logger.opt(depth=depth, exception=record.exc_info).log(level, message)

        # Only intercept WARNING and above to keep third-party noise out.
        logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)


def getLogger(name: Optional[str] = None):
    """Return a logger bound to ``name`` for formatted module output."""
    if not LoggerManager._initialized:
        LoggerManager.setup(level=GeneralConfig.defaultLogLevel)

    return logger.bind(module_name=name or "Main")


def setLogLevel(level: str):
    """Change the console/file log level without dropping the event sink.

    Args:
        level: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR or CRITICAL.
    """
    if not LoggerManager._initialized:
        LoggerManager.setup(level=level)
        return

    with _logLock:
        resolvedLevel = LoggerManager._normaliseLevel(level)
        logger.remove()
        LoggerManager._installLevelIcons()
        LoggerManager._level = resolvedLevel
        LoggerManager._registerSinks(resolvedLevel)

    logger.info(f"日志级别已设置为: {resolvedLevel}")


# Initialise on import so every module gets the same sink configuration.
LoggerManager.setup()
