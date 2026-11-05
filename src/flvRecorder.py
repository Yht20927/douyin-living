# -*- coding: utf-8 -*-
"""FLV stream recorder — pull FLV via HTTP and write to disk with rotation.

Segments are written to ``*.flv.part`` and renamed to ``*.flv`` only after
they finish, so a crash never leaves a half-written file that looks valid.
Every finished segment is probed with ffprobe; unreadable segments are moved
to ``corrupt/`` and never enter the merge list.
"""

import asyncio
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import aiofiles

from src.contracts import probeMedia
from src.models import StreamInfo
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().recording

ROTATION_SECS = _cfg.rotation_secs
_DISK_CHECK_INTERVAL_BYTES = 64 * 1024 * 1024


class FlvIdleTimeout(Exception):
    """Raised when the FLV stream produces no bytes for too long."""


class FlvRecorder:
    """Downloads an FLV stream and writes rotating files to disk.

    Usage::

        recorder = FlvRecorder(roomId="799525583657", stream=streamInfo)
        await recorder.run()   # blocks until stream ends or stopped
    """

    def __init__(
        self,
        roomId: str,
        stream: StreamInfo | None = None,
        url: str | None = None,
        quality: str | None = None,
        outputDir: str | None = None,
        rotationSecs: int = ROTATION_SECS,
        onUrlExpired: "callable | None" = None,
    ):
        if stream is None and url is None:
            raise ValueError("Either `stream` (StreamInfo) or `url` (str) must be provided")

        self.roomId = roomId
        self.stream = stream
        self._directUrl = url
        self.quality = quality or (stream.defaultQuality if stream else "hd")
        self.outputDir = outputDir or os.path.join("data", roomId)
        self.rotationSecs = rotationSecs
        self._onUrlExpired = onUrlExpired

        self._stopEvent = asyncio.Event()
        self._currentFile: str | None = None
        self._bytesWritten: int = 0
        self._startedAt: datetime | None = None
        self._firstDataAt: datetime | None = None
        self._downloadedFiles: list[str] = []
        self._corruptFiles: list[str] = []
        self._lastError: str | None = None
        self._stopReason: str | None = None
        self._streamEnded = False
        self._idleReconnects = 0
        self._idleTimeout = float(_cfg.idle_timeout_secs)
        self._minFreeBytes = int(float(_cfg.min_free_disk_gb) * 1024 ** 3)
        self._maxSegmentBytes = int(_cfg.max_segment_bytes)
        self._nextDiskCheckAt = _DISK_CHECK_INTERVAL_BYTES

    def updateUrl(self, url: str):
        """Replace the direct URL (used after expiry refresh)."""
        self._directUrl = url

    @property
    def flvUrl(self) -> str:
        """Return the FLV URL — direct URL takes precedence over StreamInfo."""
        if self._directUrl:
            return self._directUrl
        try:
            return self.stream.flvUrl(self.quality)
        except KeyError:
            if not self.stream.flvUrls:
                raise ValueError("No FLV URLs available in stream info")
            first = next(iter(self.stream.flvUrls.values()))
            log.warning(f"Quality {self.quality} not available, falling back")
            return first

    async def run(self):
        """Start recording. Blocks until stopped, live ends or disk is low."""
        self._startedAt = datetime.now(timezone.utc)
        self._stopReason = None
        self._stopEvent.clear()
        Path(self.outputDir).mkdir(parents=True, exist_ok=True)
        self._cleanupStaleParts()
        if not self._hasDiskSpace():
            self._stopReason = "low_disk"
            return
        await self._downloadLoop()

    def stop(self):
        """Signal the recorder to stop."""
        self._stopEvent.set()

    # ── disk / housekeeping ───────────────────────────────────────

    def _hasDiskSpace(self) -> bool:
        """Return False when free space is below the configured floor."""
        if self._minFreeBytes <= 0:
            return True
        try:
            free = shutil.disk_usage(self.outputDir).free
        except OSError:
            return True
        if free >= self._minFreeBytes:
            return True
        self._lastError = (
            f"low disk space: {free / 1024 ** 3:.2f} GiB free "
            f"< {self._minFreeBytes / 1024 ** 3:.2f} GiB"
        )
        log.error(self._lastError)
        return False

    def _cleanupStaleParts(self) -> None:
        """Remove abandoned ``.part`` files left behind by a previous crash."""
        maxAge = int(_cfg.stale_part_max_age_secs)
        if maxAge <= 0:
            return
        cutoff = time.time() - maxAge
        for path in Path(self.outputDir).glob("*.flv.part"):
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                log.warning(f"Removed stale partial segment: {path}")
            except OSError:
                log.debug(f"Could not remove stale partial: {path}", exc_info=True)

    # ── download loop ─────────────────────────────────────────────

    async def _downloadLoop(self):
        retries = 0
        eofRetries = 0
        urlRefreshed = False
        while not self._stopEvent.is_set():
            if not self._hasDiskSpace():
                self._stopReason = "low_disk"
                break

            try:
                reason = await self._downloadSegment()
                retries = 0
                urlRefreshed = False
                if reason in ("rotation", "size_cap"):
                    eofRetries = 0

                if reason == "low_disk":
                    self._stopReason = "low_disk"
                    break
                if self._stopEvent.is_set():
                    self._stopReason = "stopped"
                    break
                if reason == "eof":
                    eofRetries += 1
                    if eofRetries > _cfg.max_retries:
                        self._stopReason = "stream_ended"
                        log.warning(
                            "FLV stream ended and reconnect attempts were "
                            "exhausted — assuming the live has ended"
                        )
                        break
                    wait = min(_cfg.backoff_base ** eofRetries, _cfg.backoff_max)
                    log.warning(
                        "FLV stream ended (EOF) — reconnecting in %ss "
                        "(attempt %d/%d)",
                        wait,
                        eofRetries,
                        _cfg.max_retries,
                    )
                    await self._waitOrStop(wait)
            except FlvIdleTimeout as e:
                retries += 1
                self._idleReconnects += 1
                self._lastError = str(e)
                wait = min(_cfg.backoff_base ** retries, _cfg.backoff_max)
                log.error(f"FLV idle timeout (retry {retries} in {wait}s): {e}")
                await self._waitOrStop(wait)
            except aiohttp.ClientResponseError as e:
                # 403/404 usually mean the signed URL expired — try to refresh once.
                if e.status in (403, 404) and self._onUrlExpired and not urlRefreshed:
                    log.warning(f"URL expired (HTTP {e.status}), refreshing...")
                    try:
                        newUrl = await self._onUrlExpired()
                        if newUrl:
                            self.updateUrl(newUrl)
                            urlRefreshed = True
                            retries = 0
                            continue
                    except Exception:
                        log.exception("URL refresh failed")
                retries += 1
                self._lastError = f"HTTP {e.status}: {e}"
                wait = min(_cfg.backoff_base ** retries, _cfg.backoff_max)
                log.error(f"HTTP error (retry {retries} in {wait}s): {e}")
                await self._waitOrStop(wait)
            except aiohttp.ClientError as e:
                retries += 1
                self._lastError = f"{type(e).__name__}: {e}"
                wait = min(_cfg.backoff_base ** retries, _cfg.backoff_max)
                log.error(f"HTTP error (retry {retries} in {wait}s): {e}")
                await self._waitOrStop(wait)
            except asyncio.CancelledError:
                self._stopReason = "cancelled"
                raise
            except Exception as e:
                log.exception("Unexpected error in download loop")
                retries += 1
                self._lastError = f"{type(e).__name__}: {e}"
                await self._waitOrStop(_cfg.unexpected_retry_wait)

            if retries > _cfg.max_retries:
                self._stopReason = "max_retries"
                if self._streamEnded:
                    log.warning(
                        "FLV stream ended and reconnect attempts were exhausted "
                        "— assuming the live has ended"
                    )
                else:
                    log.error("Too many retries, giving up")
                break

        if self._stopReason is None:
            self._stopReason = "stopped" if self._stopEvent.is_set() else "ended"
        log.info(f"FLV download loop finished: {self._stopReason}")

    async def _waitOrStop(self, timeout: float) -> None:
        """Sleep for ``timeout`` seconds unless ``stop()`` is called first."""
        try:
            await asyncio.wait_for(self._stopEvent.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    async def _nextChunk(self, iterator):
        """Read one chunk, raising :class:`FlvIdleTimeout` on a dead stream."""
        try:
            return await asyncio.wait_for(anext(iterator), timeout=self._idleTimeout)
        except asyncio.TimeoutError as exc:
            raise FlvIdleTimeout(
                f"no data for {self._idleTimeout:.1f}s"
            ) from exc

    async def _downloadSegment(self) -> str:
        """Download one segment; returns the reason the segment ended."""
        url = self.flvUrl
        segmentStart = asyncio.get_event_loop().time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        fname = f"{self.roomId}_{self.quality}_{ts}.flv"
        fpath = os.path.join(self.outputDir, fname)
        partPath = fpath + ".part"

        self._currentFile = fpath
        self._bytesWritten = 0
        self._nextDiskCheckAt = _DISK_CHECK_INTERVAL_BYTES
        reason = "error"

        log.info(f"Recording FLV → {fpath}")

        timeout = aiohttp.ClientTimeout(
            total=_cfg.http_timeout_total,
            sock_read=_cfg.http_timeout_sock_read,
        )
        connector = aiohttp.TCPConnector(force_close=True)

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    async with aiofiles.open(partPath, "wb") as fh:
                        iterator = resp.content.iter_chunked(_cfg.chunk_size).__aiter__()
                        while True:
                            if self._stopEvent.is_set():
                                reason = "stopped"
                                break
                            try:
                                chunk = await self._nextChunk(iterator)
                            except StopAsyncIteration:
                                self._streamEnded = True
                                reason = "eof"
                                log.info("FLV stream reached EOF")
                                break
                            if not chunk:
                                continue

                            if self._streamEnded:
                                self._streamEnded = False
                            await fh.write(chunk)
                            if self._firstDataAt is None:
                                self._firstDataAt = datetime.now(timezone.utc)
                            self._bytesWritten += len(chunk)

                            elapsed = asyncio.get_event_loop().time() - segmentStart
                            if elapsed >= self.rotationSecs:
                                log.info(f"Rotation interval reached ({elapsed:.0f}s)")
                                reason = "rotation"
                                break
                            if self._bytesWritten >= self._maxSegmentBytes:
                                log.info(
                                    "Segment size cap reached (%.1f MiB)",
                                    self._bytesWritten / 1024 ** 2,
                                )
                                reason = "size_cap"
                                break
                            if self._bytesWritten >= self._nextDiskCheckAt:
                                self._nextDiskCheckAt += _DISK_CHECK_INTERVAL_BYTES
                                if not self._hasDiskSpace():
                                    reason = "low_disk"
                                    break
        finally:
            self._currentFile = None
            if os.path.exists(partPath):
                self._finalizeSegment(partPath, fpath)

        return reason

    # ── segment validation ────────────────────────────────────────

    def _finalizeSegment(self, partPath: str, fpath: str) -> None:
        """Validate a finished ``.part`` file and atomically publish it."""
        try:
            size = os.path.getsize(partPath)
        except OSError:
            return
        if size <= 0:
            self._removeQuietly(partPath)
            return

        if not self._validateSegment(partPath):
            target = self._quarantineSegment(partPath)
            if target:
                self._corruptFiles.append(target)
                self._lastError = f"corrupt segment quarantined: {os.path.basename(target)}"
                log.error(
                    "Corrupt FLV segment quarantined: %s → %s", partPath, target
                )
            return

        try:
            os.replace(partPath, fpath)
        except OSError as exc:
            self._lastError = f"failed to publish segment: {exc}"
            log.exception(f"Failed to publish segment {partPath}")
            return

        self._downloadedFiles.append(fpath)
        log.info(
            "Segment done: %s (%.1f MiB)",
            os.path.basename(fpath),
            size / 1024 ** 2,
        )

    def _validateSegment(self, path: str) -> bool:
        """Return True when ffprobe sees a usable media segment."""
        if shutil.which("ffprobe") is None:
            log.warning("ffprobe not installed — skipping segment validation")
            return True
        info = probeMedia(path)
        if not info:
            return False
        duration = float(info.get("durationSec") or 0.0)
        if duration <= 0:
            return False
        return bool(info.get("hasVideo") or info.get("hasAudio"))

    def _quarantineSegment(self, path: str) -> str | None:
        corruptDir = os.path.join(self.outputDir, "corrupt")
        try:
            os.makedirs(corruptDir, exist_ok=True)
            name = os.path.basename(path)
            if name.endswith(".part"):
                name = name[: -len(".part")]
            target = os.path.join(corruptDir, name)
            if os.path.exists(target):
                target = f"{target}.{int(time.time())}"
            os.replace(path, target)
            return target
        except OSError:
            log.exception(f"Failed to quarantine corrupt segment: {path}")
            return None

    @staticmethod
    def _removeQuietly(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    # ── properties ────────────────────────────────────────────────

    @property
    def firstDataAt(self) -> datetime | None:
        """UTC timestamp of the first byte written to disk (None before data)."""
        return self._firstDataAt

    @property
    def stopReason(self) -> str | None:
        """Why the download loop ended (stopped/eof/low_disk/max_retries/...)."""
        return self._stopReason

    @property
    def streamEnded(self) -> bool:
        """True when the last read ended with a clean stream EOF."""
        return self._streamEnded

    @property
    def stats(self) -> dict:
        return {
            "roomId": self.roomId,
            "quality": self.quality,
            "bytesWritten": self._bytesWritten,
            "currentFile": self._currentFile,
            "downloadedFiles": list(self._downloadedFiles),
            "corruptFiles": list(self._corruptFiles),
            "idleReconnects": self._idleReconnects,
            "streamEnded": self._streamEnded,
            "stopReason": self._stopReason,
            "lastError": self._lastError,
            "startedAt": self._startedAt.isoformat() if self._startedAt else None,
            "firstDataAt": self._firstDataAt.isoformat() if self._firstDataAt else None,
        }
