# -*- coding: utf-8 -*-
"""FLV stream recorder — pull FLV via HTTP and write to disk with rotation."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import aiofiles

from src.models import StreamInfo
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().recording

ROTATION_SECS = _cfg.rotation_secs


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
        self._downloadedFiles: list[str] = []

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
        """Start recording. Blocks until stopped or stream ends."""
        self._startedAt = datetime.now(timezone.utc)
        self._stopEvent.clear()
        Path(self.outputDir).mkdir(parents=True, exist_ok=True)
        await self._downloadLoop()

    def stop(self):
        """Signal the recorder to stop."""
        self._stopEvent.set()

    async def _downloadLoop(self):
        retries = 0
        urlRefreshed = False
        while not self._stopEvent.is_set():
            try:
                await self._downloadSegment()
                retries = 0
                urlRefreshed = False
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
                wait = min(_cfg.backoff_base ** retries, _cfg.backoff_max)
                log.error(f"HTTP error (retry {retries} in {wait}s): {e}")
                await asyncio.wait_for(self._stopEvent.wait(), timeout=wait)
            except aiohttp.ClientError as e:
                retries += 1
                wait = min(_cfg.backoff_base ** retries, _cfg.backoff_max)
                log.error(f"HTTP error (retry {retries} in {wait}s): {e}")
                await asyncio.wait_for(self._stopEvent.wait(), timeout=wait)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Unexpected error in download loop")
                retries += 1
                await asyncio.wait_for(self._stopEvent.wait(), timeout=_cfg.unexpected_retry_wait)

            if retries > _cfg.max_retries:
                log.error("Too many retries, giving up")
                break

    async def _downloadSegment(self):
        """Download one segment (rotates file on interval)."""
        url = self.flvUrl
        segmentStart = asyncio.get_event_loop().time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        fname = f"{self.roomId}_{self.quality}_{ts}.flv"
        fpath = os.path.join(self.outputDir, fname)

        log.info(f"Recording FLV → {fpath}")

        timeout = aiohttp.ClientTimeout(
            total=_cfg.http_timeout_total,
            sock_read=_cfg.http_timeout_sock_read,
        )
        connector = aiohttp.TCPConnector(force_close=True)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                self._currentFile = fpath
                self._bytesWritten = 0

                async with aiofiles.open(fpath, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(_cfg.chunk_size):
                        if self._stopEvent.is_set():
                            break
                        await fh.write(chunk)
                        self._bytesWritten += len(chunk)

                        elapsed = asyncio.get_event_loop().time() - segmentStart
                        if elapsed >= self.rotationSecs:
                            log.info(f"Rotation interval reached ({elapsed:.0f}s)")
                            break

        sizeMb = self._bytesWritten / (1024 * 1024)
        log.info(f"Segment done: {fname} ({sizeMb:.1f} MiB)")
        self._downloadedFiles.append(fpath)
        self._currentFile = None

    @property
    def stats(self) -> dict:
        return {
            "roomId": self.roomId,
            "quality": self.quality,
            "bytesWritten": self._bytesWritten,
            "currentFile": self._currentFile,
            "downloadedFiles": self._downloadedFiles,
            "startedAt": self._startedAt.isoformat() if self._startedAt else None,
        }
