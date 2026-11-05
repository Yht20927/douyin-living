# -*- coding: utf-8 -*-
"""Controller — records live rooms and optionally runs AI clipping pipeline.

Modes:
    python -m src.controller <roomId>               # Record only (existing behavior)
    python -m src.controller <roomId> --clip         # Record + AI clipping
    python -m src.controller <roomId> --clip-only    # AI clipping only (from existing files)
"""

import asyncio
import json
import os
import queue
import shutil
import signal
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.auth import Auth
from src.params import Params
from src.signer import Signer
from src.roomApi import RoomApi
from src.danmakuWs import DanmakuWs
from src.flvRecorder import FlvRecorder
from src.protobuf.Live_pb2 import LiveResponse
from src.log.events import logEvent
from src.log.logger import getLogger
from src.util import fmtSize
from src.config import load_settings
from src.contracts import (
    RecordingMeta,
    artifactFingerprint,
    artifactMetaPath,
    inspectAsrArtifact,
    inspectFeatureArtifact,
    probeMedia,
)
from src.pipeline import ArtifactCache, CacheDecision, PipelineStatus, StepStatus

log = getLogger(__name__)
_cfg = load_settings().pipeline

# Canonical resumable pipeline stages, in execution order.  Physical status
# steps (1_audio_extract, 4_audio_signal, ...) map onto these groups.
STEP_ORDER = ("audio", "asr", "signal", "score", "clip")

_STEP_ALIASES = {
    "audio": "audio",
    "audio_extract": "audio",
    "1_audio_extract": "audio",
    "asr": "asr",
    "2_asr": "asr",
    "signal": "signal",
    "signals": "signal",
    "3_signal": "signal",
    "4_audio_signal": "signal",
    "5_text_signal": "signal",
    "6_visual_signal": "signal",
    "score": "score",
    "scorer": "score",
    "7_scorer": "score",
    "clip": "clip",
    "clips": "clip",
    "clipping": "clip",
    "8_clipping": "clip",
    "9_subtitles_thumbnails": "clip",
}


def normalizeStepName(value: str | None) -> str | None:
    """Map CLI/status step names onto a canonical resumable stage."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    try:
        return _STEP_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown pipeline step: {value!r}; expected one of {', '.join(STEP_ORDER)}"
        ) from exc



class Controller:
    """Controls a single douyin live room: recording + optional AI clipping.

    Usage::

        ctrl = Controller("300294032039", clip=True)
        await ctrl.start()
    """

    def __init__(
        self,
        webRid: str,
        clip: bool = False,
        clipOnly: bool = False,
        profile: str = "default",
        sensitivity: float | None = None,
        minDuration: float | None = None,
        maxDuration: float | None = None,
        preprocess: bool = False,
        resolution: str | None = None,
        dryRun: bool = False,
        outputDir: str | None = None,
        force: bool = False,
        fromStep: str | None = None,
        onlyStep: str | None = None,
        twoStage: bool | None = None,
    ):
        self.webRid = webRid
        self.clip = clip
        self.clipOnly = clipOnly
        self.profile = profile
        self.sensitivity = sensitivity if sensitivity is not None else _cfg.sensitivity_default
        self.minDuration = minDuration if minDuration is not None else _cfg.min_duration_default
        self.maxDuration = maxDuration if maxDuration is not None else _cfg.max_duration_default
        self.preprocess = preprocess
        self.resolution = resolution or _cfg.resolution_default
        self.dryRun = dryRun
        self.outputDir = outputDir or os.path.join("data", webRid)
        self.force = force
        self.fromStep = normalizeStepName(fromStep)
        self.onlyStep = normalizeStepName(onlyStep)
        # None = follow config; explicit bool is used by CLI/tests.
        self.twoStage = _cfg.two_stage_enabled if twoStage is None else twoStage
        if self.onlyStep and not self.fromStep:
            self.fromStep = self.onlyStep
        self._artifactCache: ArtifactCache | None = None

        self._auth: Auth | None = None
        self._danmakuWs: DanmakuWs | None = None
        self._flvTask: asyncio.Task | None = None
        self._stopEvent = asyncio.Event()
        # Use bounded deque to prevent unbounded memory growth during
        # long recordings (previously a plain list that could hold
        # hundreds of thousands of dicts).
        self._danmakuMessages: deque[dict[str, Any]] = deque(maxlen=1000)
        self._danmakuCount: int = 0
        self._flvPath: str | None = None
        self._flvRecorder: FlvRecorder | None = None
        self._recordStartTime: float = 0.0  # wall-clock seconds when recording started
        self._recordStartMono: float = 0.0
        self._danmakuQueue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._writerThread: threading.Thread | None = None
        self._writerStop = threading.Event()

    # ── cache / resume helpers ─────────────────────────────────

    @property
    def _cache(self) -> ArtifactCache:
        if self._artifactCache is None:
            self._artifactCache = ArtifactCache()
        return self._artifactCache

    def _shouldRebuild(self, step: str) -> bool:
        """True when ``--force`` or ``--from-step`` invalidates this stage."""
        canonical = normalizeStepName(step)
        if self.force:
            return True
        if self.fromStep and canonical:
            return STEP_ORDER.index(canonical) >= STEP_ORDER.index(self.fromStep)
        return False

    def _stopAfter(self, step: str) -> bool:
        """True when ``--only-step`` names this canonical stage."""
        return self.onlyStep is not None and self.onlyStep == normalizeStepName(step)

    def _checkCache(
        self,
        stepName: str,
        artifactPath: str,
        inputs: list[str],
        params: dict[str, Any] | None = None,
        validator=None,
    ) -> CacheDecision:
        """Cache lookup honouring force/from-step; validators check schema."""
        if self._shouldRebuild(stepName):
            self._cache.misses += 1
            return CacheDecision(False, "forced rebuild")
        return self._cache.check(
            artifactPath,
            stepName,
            inputs,
            params=params,
            validator=validator,
        )

    @staticmethod
    def _cacheInfo(decision: CacheDecision) -> dict[str, Any]:
        info: dict[str, Any] = {
            "reused": decision.hit,
            "cacheReason": decision.reason,
        }
        if decision.meta is not None:
            info["metaPath"] = artifactMetaPath(decision.meta.artifactPath)
            info["generatedAt"] = decision.meta.generatedAt
            if decision.meta.elapsedMs is not None:
                info["elapsedMs"] = decision.meta.elapsedMs
        return info

    @staticmethod
    def _loadJsonDict(path: str) -> dict[str, Any] | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # ── start / stop ───────────────────────────────────────────

    async def start(self):
        if self.clipOnly:
            await self._runClippingOnly()
            return

        await self._runRecording()
        if self.clip:
            await self._runClippingPipeline()

    async def stop(self):
        log.info("Stopping controller...")
        if self._danmakuWs:
            self._danmakuWs.stop()
            self._danmakuWs.join(timeout=_cfg.ws_join_timeout)
        # Signal and drain the background writer thread so every queued
        # message is flushed to disk before we proceed.
        if self._writerThread and self._writerThread.is_alive():
            self._writerStop.set()
            self._writerThread.join(timeout=_cfg.writer_join_timeout)
        if self._flvTask:
            self._flvTask.cancel()
            try:
                await self._flvTask
            except asyncio.CancelledError:
                pass
        # Release the shared HTTP client to avoid connection pool leaks.
        await RoomApi.close()
        log.info(f"Danmaku: {self._danmakuCount} messages")
        log.info("Controller stopped.")

    # ── Recording phase (existing) ─────────────────────────────

    async def _runRecording(self):
        """Record live room: danmaku + FLV + audio."""
        log.info(f"Fetching room info for {self.webRid}...")

        # 1. Auth
        self._auth = Auth.fromEnv()
        if not self._auth.cookieStr:
            log.error("No cookies in .env")
            return

        # 2. Room info
        info = await RoomApi.getLiveInfo(self._auth, self.webRid)
        roomId = info.get("roomId", "")
        userId = info.get("userId", "")
        roomTitle = info.get("roomTitle", "")

        if info.get("roomStatus") != "2":
            log.warning("Room not live")
            return
        log.info(f"Room: {roomTitle}")

        # 3. Webcast detail for WS
        detailBytes = await RoomApi.getWebcastDetail(
            self._auth, userId, roomId,
            referer=f"https://live.douyin.com/{self.webRid}"
        )
        frame = LiveResponse()
        frame.ParseFromString(detailBytes)

        # 4. Build WS params
        wsParams = Params()
        wsParams.addAll({
            "app_name": "douyin_web", "version_code": "180800",
            "webcast_sdk_version": "1.0.15", "update_version_code": "1.0.15",
            "compress": "gzip", "device_platform": "web",
            "cookie_enabled": "true", "screen_width": "1920", "screen_height": "1080",
            "browser_language": "zh-CN", "browser_platform": "Win32",
            "browser_name": "Mozilla", "browser_version": "5.0",
            "browser_online": "true", "tz_name": "Etc/GMT-8",
            "cursor": str(frame.cursor), "internal_ext": frame.internalExt,
            "host": "https://live.douyin.com", "aid": "6383", "live_id": "1",
            "did_rule": "3", "endpoint": "live_pc", "support_wrds": "1",
            "user_unique_id": str(userId), "im_path": "/webcast/im/fetch/",
            "identity": "audience", "need_persist_msg_count": "15",
            "insert_task_id": "", "live_reason": "", "room_id": roomId,
            "heartbeatDuration": "0",
            "signature": Signer.generateLiveSignature(roomId, userId),
        })

        # 5. Danmaku WS
        self._recordStartTime = time.time()
        self._recordStartMono = time.monotonic()
        Path(self.outputDir).mkdir(parents=True, exist_ok=True)
        jsonlPath = os.path.join(self.outputDir, f"{self.webRid}_danmaku.jsonl")

        def onDanmaku(msg: dict):
            msg["recvTimeSec"] = round(time.time() - self._recordStartTime, 3)
            self._danmakuMessages.append(msg)
            self._danmakuCount += 1
            self._danmakuQueue.put(msg)
            msgType = msg.get("type", "?")
            if msgType == "chat":
                log.info(f"💬 {msg.get('userName','?')}: {msg.get('content','')}")
            elif msgType == "gift":
                log.info(f"🎁 {msg.get('userName','?')} → {msg.get('giftName','?')} x{msg.get('comboCount',1)}")
            elif msgType == "like":
                log.info(f"👍 {msg.get('userName','?')} x{msg.get('count',0)}")
            elif msgType == "member":
                log.info(f"🚪 {msg.get('userName','?')} 进入直播间")
            elif msgType == "roomStats":
                log.info(f"📊 {msg.get('displayShort','')}")

        # Start background writer thread so the WS read loop never blocks on disk I/O.
        def _writerLoop():
            with open(jsonlPath, "a", encoding="utf-8") as fh:
                while not self._writerStop.is_set():
                    try:
                        msg = self._danmakuQueue.get(timeout=_cfg.queue_timeout)
                    except queue.Empty:
                        continue
                    fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    fh.flush()

        self._writerThread = threading.Thread(target=_writerLoop, daemon=True)
        self._writerThread.start()

        self._danmakuWs = DanmakuWs(
            params=wsParams.toDict(),
            cookieStr=self._auth.cookieStr,
            onDanmaku=onDanmaku,
        )
        self._danmakuWs.start()

        # 6. FLV recording — extract signed URLs from page SSR, use FlvRecorder
        async def _refreshStreamUrl() -> str:
            """Callback invoked by FlvRecorder when the signed URL expires."""
            refreshed = await RoomApi.getStreamUrls(self._auth, self.webRid)
            if refreshed:
                return refreshed.get("or4") or refreshed.get("hd") or refreshed.get("sd") or list(refreshed.values())[0]
            return ""

        try:
            streamUrls = await RoomApi.getStreamUrls(self._auth, self.webRid)
            if streamUrls:
                flvUrl = streamUrls.get("or4") or streamUrls.get("hd") or streamUrls.get("sd") or list(streamUrls.values())[0]
                self._flvRecorder = FlvRecorder(
                    roomId=self.webRid,
                    url=flvUrl,
                    outputDir=self.outputDir,
                    onUrlExpired=_refreshStreamUrl,
                )
                # Share the same stop event so Controller.stop() stops the recorder
                self._flvRecorder._stopEvent = self._stopEvent
                self._flvTask = asyncio.create_task(self._flvRecorder.run())
                log.info(f"FLV started: {flvUrl[:80]}...")
            else:
                log.warning("No FLV URLs found")
        except Exception as e:
            log.warning(f"FLV not available: {e}")

        # 7. Wait for stop, or for the recorder to end on its own (live
        # ended / unrecoverable network failure / low disk).  Without this
        # race a dead stream would leave the controller waiting forever.
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: self._stopEvent.set())
        except NotImplementedError:
            pass

        stopTask = asyncio.create_task(self._stopEvent.wait())
        try:
            if self._flvTask is not None:
                await asyncio.wait(
                    {stopTask, self._flvTask},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._flvTask.done() and not self._stopEvent.is_set():
                    if not self._flvTask.cancelled() and self._flvTask.exception():
                        log.error(
                            "FLV recorder failed: %r", self._flvTask.exception()
                        )
                    stopReason = getattr(self._flvRecorder, "stopReason", None)
                    log.info(
                        "FLV recorder finished (%s); finalizing recording",
                        stopReason or "ended",
                    )
            else:
                await stopTask
        finally:
            stopTask.cancel()
        await self.stop()

        # Collect every FLV segment that was rotated out, plus the in-progress one.
        # Multi-segment recordings (>30min) MUST all be processed — we merge
        # them losslessly into a single MP4 so the downstream pipeline sees one
        # continuous timeline.
        segments: list[str] = []
        if self._flvRecorder:
            segments.extend(self._flvRecorder._downloadedFiles or [])
            current = self._flvRecorder._currentFile
            if current and current not in segments and os.path.exists(current):
                segments.append(current)

        merged = await self._mergeFlvSegments(segments)
        if merged:
            self._flvPath = merged
            audioPath = merged.rsplit(".", 1)[0] + ".aac"
            await self._extractAudio(merged, audioPath)
            await self._writeRecordingMeta(merged, audioPath, segments)

    # ── Clipping pipeline ──────────────────────────────────────

    async def _runClippingOnly(self):
        """Run clipping pipeline on existing recorded files."""
        self._auth = Auth.fromEnv()
        await self._runClippingPipeline()

    async def _runClippingPipeline(self):
        """Run the full AI clipping pipeline on recorded data.

        The status document is written even when the pipeline fails or
        produces zero clips.  Every step records timing, inputs/outputs,
        artifact hashes and an explicit status (ok/empty/failed/skipped).
        """
        log.info("=" * 50)
        log.info("Starting AI clipping pipeline")
        log.info(
            "Options: force={} fromStep={} onlyStep={} scoring={}",
            self.force,
            self.fromStep,
            self.onlyStep,
            "two_stage" if self.twoStage else "legacy",
        )
        log.info("=" * 50)
        os.makedirs(self.outputDir, exist_ok=True)
        self._artifactCache = ArtifactCache()
        runStartedAt = time.monotonic()

        statusPath = os.path.join(self.outputDir, "pipeline_status.json")

        def _onStep(name: str, record: dict[str, Any]) -> None:
            """Mirror every completed step into the structured event log."""
            meta = record.get("meta") or {}
            extra: dict[str, Any] = {}
            if meta.get("reused") is not None:
                extra["reused"] = bool(meta["reused"])
            if meta.get("cacheReason"):
                extra["cacheReason"] = meta["cacheReason"]
            if record.get("error"):
                extra["error"] = record["error"]
            if meta.get("pipeline"):
                extra["scoring"] = meta["pipeline"]
            if meta.get("stages"):
                extra["stages"] = meta["stages"]
            logEvent(
                "pipeline.step",
                runId=status.runId,
                roomId=self.webRid,
                step=name,
                status=record.get("status"),
                elapsedMs=record.get("elapsedMs"),
                **extra,
            )

        status = PipelineStatus(
            roomId=self.webRid,
            outputDir=self.outputDir,
            profile=self.profile,
            sensitivity=self.sensitivity,
            onStep=_onStep,
        )
        logEvent(
            "pipeline.start",
            runId=status.runId,
            roomId=self.webRid,
            status="running",
            outputDir=self.outputDir,
            profile=self.profile,
            sensitivity=self.sensitivity,
            scoring="two_stage" if self.twoStage else "legacy",
            force=self.force,
            fromStep=self.fromStep,
            onlyStep=self.onlyStep,
        )
        try:
            await self._runClippingPipelineInner(status)
        except Exception as exc:
            log.exception("Clipping pipeline crashed")
            status.setStep(
                "0_pipeline",
                StepStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            status.cacheStats = self._cache.stats()
            status.save(statusPath)
            logEvent(
                "pipeline.finish",
                runId=status.runId,
                roomId=self.webRid,
                status="failed" if status.hasFailures() else "ok",
                elapsedMs=int((time.monotonic() - runStartedAt) * 1000),
                failures=len(status.failureSummary),
                cacheHits=status.cacheStats.get("hits", 0),
                cacheMisses=status.cacheStats.get("misses", 0),
                cacheWrites=status.cacheStats.get("writes", 0),
            )
            log.info(f"Pipeline status: {statusPath}")
            log.info(f"Artifact cache: {status.cacheStats}")

    async def _runClippingPipelineInner(self, status: PipelineStatus):
        """Actual pipeline body — always called under status accounting."""
        # Resolve the source video. Prefer a previously merged file
        # (data/{roomId}/{roomId}_merged.mp4); otherwise auto-merge any FLV
        # segments we find on disk so multi-segment recordings work end-to-end.
        mergedPath = os.path.join(self.outputDir, f"{self.webRid}_merged.mp4")
        if os.path.exists(mergedPath):
            videoPath = mergedPath
        else:
            flvFiles = sorted(Path(self.outputDir).glob("*.flv"))
            if not flvFiles:
                log.error(f"No FLV files in {self.outputDir}")
                status.setStep(
                    "0_input",
                    StepStatus.FAILED,
                    error=f"no FLV/MP4 input in {self.outputDir}",
                )
                return
            if len(flvFiles) == 1:
                # Re-mux single FLV to MP4 so downstream tools (OpenCV,
                # scenedetect) get a clean container with valid duration.
                singleFlv = str(flvFiles[0])
                remuxed = singleFlv.rsplit(".", 1)[0] + ".mp4"
                if not os.path.exists(remuxed):
                    log.info(f"Re-muxing FLV → MP4: {remuxed}")
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-i", singleFlv, "-c", "copy", "-y", remuxed,
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await proc.communicate()
                    if proc.returncode != 0 or not os.path.exists(remuxed):
                        err = stderr.decode(errors="replace")[-200:] if stderr else ""
                        log.warning(f"FLV re-mux failed: {err}")
                        videoPath = singleFlv
                    else:
                        videoPath = remuxed
                else:
                    videoPath = remuxed
            else:
                # Multi-segment on-disk recording — merge before continuing.
                merged = await self._mergeFlvSegments([str(p) for p in flvFiles])
                videoPath = merged or str(flvFiles[-1])
                if not merged:
                    log.warning(
                        f"FLV merge unavailable — falling back to last segment only: {videoPath}"
                    )
        audioPath = videoPath.rsplit(".", 1)[0] + ".aac"
        danmakuPath = os.path.join(self.outputDir, f"{self.webRid}_danmaku.jsonl")

        log.info(f"Video: {videoPath}")
        log.info(f"Audio: {audioPath}")
        log.info(f"Profile: {self.profile}, Sensitivity: {self.sensitivity}")

        # ── Step 1: Extract audio ──────────────────────────────
        status.startStep("1_audio_extract", inputs=[videoPath])
        audioParams = {"codec": "copy"}
        audioDecision = self._checkCache(
            "1_audio_extract",
            audioPath,
            [videoPath],
            params=audioParams,
            validator=lambda: (
                os.path.exists(audioPath) and os.path.getsize(audioPath) > 0,
                "cached audio is missing or empty",
            ),
        )
        if audioDecision.hit:
            log.info(f"Step 1: audio cache hit → {audioPath}")
            status.finishStep(
                "1_audio_extract",
                StepStatus.SKIPPED,
                outputs=[audioPath],
                meta=self._cacheInfo(audioDecision),
            )
        else:
            log.info(f"Step 1: {audioDecision.reason} — extracting audio")
            self._cache.invalidate(audioPath)
            startedAt = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", videoPath, "-vn", "-acodec", "copy", "-y", audioPath,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not os.path.exists(audioPath):
                err = stderr.decode(errors="replace")[-300:] if stderr else ""
                log.error(f"Audio extraction failed: {err}")
                status.finishStep(
                    "1_audio_extract",
                    StepStatus.FAILED,
                    error=err or "ffmpeg produced no audio file",
                )
                return
            elapsedMs = int((time.monotonic() - startedAt) * 1000)
            self._cache.record(
                audioPath,
                "1_audio_extract",
                [videoPath],
                params=audioParams,
                elapsedMs=elapsedMs,
                extra={"sizeBytes": os.path.getsize(audioPath)},
            )
            log.info(f"Audio: {audioPath}")
            status.finishStep(
                "1_audio_extract",
                StepStatus.OK,
                outputs=[audioPath],
                meta={
                    "reused": False,
                    "cacheReason": audioDecision.reason,
                    "elapsedMs": elapsedMs,
                },
            )

        # Probe the real media duration and persist the recording contract.
        recordingMeta = await self._writeRecordingMeta(
            videoPath, audioPath,
            [str(p) for p in sorted(Path(self.outputDir).glob("*.flv"))],
        )
        durationSec = recordingMeta.videoDurationSec or recordingMeta.ffprobe.get(
            "durationSec", 0.0
        )
        log.info(f"Recording duration: {durationSec:.1f}s")

        if self._stopAfter("audio"):
            log.info("--only-step audio: stopping before ASR")
            return

        # ── Step 2: ASR + diarization ──────────────────────────
        log.info("Step 2: ASR...")
        asrPath = audioPath.rsplit(".", 1)[0] + "_asr.json"
        asrParams = {
            "model": _cfg.asr_model,
            "computeType": _cfg.asr_compute_type,
            "diarize": _cfg.asr_diarize,
        }
        status.startStep("2_asr", inputs=[audioPath])

        def _asrValidator() -> tuple[bool, str]:
            info = inspectAsrArtifact(asrPath)
            if info["status"] in {"ok", "empty"}:
                return True, "cache hit"
            return False, f"previous ASR artifact is {info['status']}"

        asrDecision = self._checkCache(
            "2_asr",
            asrPath,
            [audioPath],
            params=asrParams,
            validator=_asrValidator,
        )
        if asrDecision.hit:
            asrInfo = inspectAsrArtifact(asrPath)
            log.info(f"Step 2: ASR cache hit → {asrPath} (status={asrInfo['status']})")
            status.finishStep(
                "2_asr",
                StepStatus.SKIPPED,
                outputs=[asrPath],
                error="; ".join(asrInfo["errors"]) or None,
                meta={
                    **self._cacheInfo(asrDecision),
                    "asrStatus": asrInfo["status"],
                    "segments": asrInfo["segments"],
                    "duration": asrInfo["duration"],
                },
            )
        else:
            log.info(f"Step 2: {asrDecision.reason} — running ASR")
            self._cache.invalidate(asrPath)
            startedAt = time.monotonic()
            try:
                from asr import transcribe
                device = _cfg.asr_device
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif device == "cuda":
                    import torch
                    if not torch.cuda.is_available():
                        log.warning(
                            "ASR device=cuda requested but CUDA is unavailable "
                            "— falling back to CPU"
                        )
                        device = "cpu"
                log.info(
                    f"ASR model: {_cfg.asr_model}, device: {device}, "
                    f"compute: {_cfg.asr_compute_type}, diarize: {_cfg.asr_diarize}"
                )
                # Offload blocking ASR to thread pool so the event loop stays alive
                await asyncio.to_thread(
                    transcribe,
                    audioPath,
                    modelSize=_cfg.asr_model,
                    device=device,
                    computeType=_cfg.asr_compute_type,
                    diarize=_cfg.asr_diarize,
                    outputPath=asrPath,
                )
                asrInfo = inspectAsrArtifact(asrPath)
                elapsedMs = int((time.monotonic() - startedAt) * 1000)
                self._cache.record(
                    asrPath,
                    "2_asr",
                    [audioPath],
                    params=asrParams,
                    elapsedMs=elapsedMs,
                    extra={
                        "status": asrInfo["status"],
                        "segments": asrInfo["segments"],
                    },
                )
                stepStatus = (
                    StepStatus.FAILED
                    if asrInfo["status"] == "missing"
                    else asrInfo["status"]
                )
                status.finishStep(
                    "2_asr",
                    stepStatus,
                    outputs=[asrPath],
                    error="; ".join(asrInfo["errors"]) or None,
                    meta={
                        "segments": asrInfo["segments"],
                        "duration": asrInfo["duration"],
                        "reused": False,
                        "cacheReason": asrDecision.reason,
                        "elapsedMs": elapsedMs,
                    },
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                elapsedMs = int((time.monotonic() - startedAt) * 1000)
                log.warning(f"ASR failed — marking as unavailable: {err}", exc_info=True)
                with open(asrPath, "w", encoding="utf-8") as f:
                    json.dump({"status": "failed", "segments": [], "error": err}, f)
                # Record the failed attempt too: the validator will reject it
                # next run, but the sidecar keeps timing/version diagnostics.
                self._cache.record(
                    asrPath,
                    "2_asr",
                    [audioPath],
                    params=asrParams,
                    elapsedMs=elapsedMs,
                    extra={"status": "failed", "error": err},
                )
                status.finishStep(
                    "2_asr",
                    StepStatus.FAILED,
                    outputs=[asrPath],
                    error=err,
                    meta={
                        "reused": False,
                        "cacheReason": asrDecision.reason,
                        "elapsedMs": elapsedMs,
                    },
                )

        if self._stopAfter("asr"):
            log.info("--only-step asr: stopping before signal extraction")
            return

        # Step 3: Preprocess (auto-editor, optional)
        # Skipped in Phase 1 — requires --preprocess flag

        # ── Step 4-6: Signal extraction (parallel) ─────────────
        log.info("Steps 4-6: Extracting signals (parallel)...")
        audioFeatPath = os.path.join(self.outputDir, "audio_features.json")
        textFeatPath = os.path.join(self.outputDir, "text_features.json")
        visualFeatPath = os.path.join(self.outputDir, "visual_features.json")

        def _runAudio(audPath: str, outPath: str):
            from src.signalAudio import extractFeatures
            extractFeatures(audPath, outputPath=outPath, durationSec=durationSec or None)

        def _runText(danPath: str, asrFilePath: str, outPath: str):
            from src.signalText import extractFeatures
            extractFeatures(
                danPath, asrFilePath, outputPath=outPath,
                durationSec=durationSec or None,
            )

        def _runVisual(vidPath: str, outPath: str):
            from src.signalVisual import extractFeatures
            extractFeatures(vidPath, outputPath=outPath, durationSec=durationSec or None)

        signalSpecs = [
            ("4_audio_signal", audioFeatPath, [audioPath], _runAudio),
            ("5_text_signal", textFeatPath, [danmakuPath, asrPath], _runText),
            ("6_visual_signal", visualFeatPath, [videoPath], _runVisual),
        ]
        signalParams = {
            "durationSec": round(float(durationSec), 3) if durationSec else None,
        }
        signalErrors: dict[str, str] = {}
        artifactStatus: dict[str, str] = {}
        futures: dict[str, Any] = {}
        decisionByStep: dict[str, CacheDecision] = {}
        elapsedByStep: dict[str, int] = {}

        def _timed(stepName: str, runner, *args):
            startedAt = time.monotonic()
            try:
                return runner(*args)
            finally:
                elapsedByStep[stepName] = int((time.monotonic() - startedAt) * 1000)

        def _featureValidator(outPath: str):
            def _validate() -> tuple[bool, str]:
                info = inspectFeatureArtifact(outPath)
                if info["status"] == "ok":
                    return True, "cache hit"
                detail = "; ".join(info["errors"]) or info["status"]
                return False, f"feature artifact is {info['status']} ({detail})"

            return _validate

        with ThreadPoolExecutor(max_workers=_cfg.signal_pool_workers) as pool:
            for stepName, outPath, inputs, runner in signalSpecs:
                decision = self._checkCache(
                    stepName,
                    outPath,
                    inputs,
                    params=signalParams,
                    validator=_featureValidator(outPath),
                )
                if decision.hit:
                    info = inspectFeatureArtifact(outPath)
                    log.info(
                        f"  {stepName}: cache hit "
                        f"({info['featureCount']} features, {info['duration']:.1f}s)"
                    )
                    artifactStatus[stepName] = "ok"
                    status.setStep(
                        stepName,
                        StepStatus.SKIPPED,
                        outputs=[outPath],
                        meta={
                            **self._cacheInfo(decision),
                            "featureCount": info["featureCount"],
                            "duration": info["duration"],
                            "featureNames": info["featureNames"],
                        },
                    )
                    continue

                log.info(f"  {stepName}: {decision.reason} — extracting")
                decisionByStep[stepName] = decision
                missing = [p for p in inputs if not os.path.exists(p)]
                if missing:
                    reason = "missing inputs: " + ", ".join(missing)
                    log.warning(f"  {stepName}: {reason}")
                    signalErrors[stepName] = reason
                    artifactStatus[stepName] = "skipped"
                    status.setStep(stepName, StepStatus.SKIPPED, error=reason)
                    continue

                status.startStep(stepName, inputs=inputs)
                self._cache.invalidate(outPath)
                if stepName == "4_audio_signal":
                    futures[stepName] = pool.submit(
                        _timed, stepName, runner, inputs[0], outPath
                    )
                elif stepName == "5_text_signal":
                    futures[stepName] = pool.submit(
                        _timed, stepName, runner, inputs[0], inputs[1], outPath
                    )
                else:
                    futures[stepName] = pool.submit(
                        _timed, stepName, runner, inputs[0], outPath
                    )

            outPathByStep = {name: out for name, out, _, _ in signalSpecs}
            inputsByStep = {name: inputs for name, _, inputs, _ in signalSpecs}
            for stepName, future in futures.items():
                outPath = outPathByStep[stepName]
                try:
                    future.result()
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    log.error(f"{stepName} signal extraction failed", exc_info=True)
                    signalErrors[stepName] = err
                    artifactStatus[stepName] = "failed"
                    status.finishStep(stepName, StepStatus.FAILED, error=err)
                    continue

                info = inspectFeatureArtifact(outPath)
                artifactStatus[stepName] = info["status"]
                elapsedMs = elapsedByStep.get(stepName)
                if info["status"] != "missing":
                    self._cache.record(
                        outPath,
                        stepName,
                        inputsByStep[stepName],
                        params=signalParams,
                        elapsedMs=elapsedMs,
                        extra={
                            "status": info["status"],
                            "featureCount": info["featureCount"],
                        },
                    )
                stepStatus = (
                    StepStatus.FAILED if info["status"] == "missing" else info["status"]
                )
                status.finishStep(
                    stepName,
                    stepStatus,
                    outputs=[outPath],
                    error="; ".join(info["errors"]) or None,
                    meta={
                        "featureCount": info["featureCount"],
                        "duration": info["duration"],
                        "featureNames": info["featureNames"],
                        "reused": False,
                        "cacheReason": decisionByStep.get(
                            stepName, CacheDecision(False, "rebuilt")
                        ).reason,
                        "elapsedMs": elapsedMs,
                    },
                )

        if self._stopAfter("signal"):
            log.info("--only-step signal: stopping before scoring")
            return

        # Load only artifacts that passed validation.  An empty/invalid file
        # is never handed to the scorer as if it were a real signal.
        def _loadSignal(outPath: str, stepName: str) -> dict:
            if artifactStatus.get(stepName) != "ok":
                return {"features": {}}
            loaded = self._safe_load_features(outPath)
            return loaded or {"features": {}}

        audioFeat = _loadSignal(audioFeatPath, "4_audio_signal")
        textFeat = _loadSignal(textFeatPath, "5_text_signal")
        visualFeat = _loadSignal(visualFeatPath, "6_visual_signal")

        if not any(feat.get("features") for feat in (audioFeat, textFeat, visualFeat)):
            log.error("No valid features extracted — cannot score")
            status.setStep(
                "7_scorer",
                StepStatus.FAILED,
                error="no valid feature artifacts",
            )
            return

        # ── Step 7: Scoring ────────────────────────────────────
        log.info("Step 7: Scoring...")
        useTwoStage = bool(self.twoStage)
        timeTablePath = os.path.join(self.outputDir, "time_table.json")
        scoresPath = os.path.join(self.outputDir, "scores.json")
        candidatesPath = os.path.join(self.outputDir, "recall_candidates.json")
        scoringInputs = [
            path
            for path, stepName in (
                (audioFeatPath, "4_audio_signal"),
                (textFeatPath, "5_text_signal"),
                (visualFeatPath, "6_visual_signal"),
            )
            if artifactStatus.get(stepName) == "ok"
        ]
        asrSegments: list[dict] = []
        if useTwoStage:
            # Sentence boundaries are only needed by the two-stage boundary
            # optimizer; the legacy scorer never read the raw ASR document.
            asrDoc = self._loadJsonDict(asrPath) or {}
            if isinstance(asrDoc.get("segments"), list):
                asrSegments = [
                    segment for segment in asrDoc["segments"]
                    if isinstance(segment, dict)
                ]
            if os.path.exists(asrPath):
                scoringInputs.append(asrPath)
        scoreParams = {
            "profile": self.profile,
            "sensitivity": self.sensitivity,
            "minDuration": self.minDuration,
            "maxDuration": self.maxDuration,
            "durationSec": round(float(durationSec), 3) if durationSec else None,
        }
        if useTwoStage:
            # Extra keys on purpose: switching scoring modes must invalidate
            # the cached time table, while legacy cache entries stay valid.
            scoreParams["scoring"] = "two_stage"
            scoreParams["twoStage"] = {
                "recallPerHour": _cfg.two_stage_recall_per_hour,
                "topK": _cfg.two_stage_top_k,
                "minGapSec": _cfg.two_stage_min_gap_sec,
                "recallThreshold": _cfg.two_stage_recall_threshold,
                "windowSeconds": _cfg.two_stage_window_sec,
                "minScore": _cfg.two_stage_min_score,
                "dynamicDuration": _cfg.two_stage_dynamic_duration,
                "boundarySearchSec": _cfg.two_stage_boundary_search_sec,
                "platform": _cfg.two_stage_platform,
            }

        def _scoreOutputs(pipeline: str = "") -> list[str]:
            outputs = [timeTablePath, scoresPath]
            if pipeline == "two_stage" and os.path.exists(candidatesPath):
                outputs.append(candidatesPath)
            return outputs

        def _timeTableValidator() -> tuple[bool, str]:
            if not os.path.exists(scoresPath):
                return False, "scores.json is missing"
            data = self._loadJsonDict(timeTablePath)
            if data is None or not isinstance(data.get("clips"), list):
                return False, "time_table.json is unreadable"
            return True, "cache hit"

        scoreDecision = self._checkCache(
            "7_scorer",
            timeTablePath,
            scoringInputs,
            params=scoreParams,
            validator=_timeTableValidator,
        )
        if scoreDecision.hit:
            result = self._loadJsonDict(timeTablePath) or {"clips": []}
            clips = result.get("clips") or []
            pipelineLabel = result.get("pipeline", "legacy")
            stages = result.get("stages") or {}
            log.info(f"Step 7: scoring cache hit ({len(clips)} clips)")
            status.setStep(
                "7_scorer",
                StepStatus.SKIPPED if clips else StepStatus.EMPTY,
                outputs=_scoreOutputs(pipelineLabel),
                error=None if clips else "no candidate clips survived detection",
                meta={
                    **self._cacheInfo(scoreDecision),
                    "pipeline": pipelineLabel,
                    "stages": stages,
                    "clips": len(clips),
                    "dropped": len(result.get("dropped") or []),
                    "adjustments": len(result.get("adjustments") or []),
                    "alignedSignals": len(result.get("alignment") or []),
                },
            )
        else:
            log.info(f"Step 7: {scoreDecision.reason} — scoring")
            self._cache.invalidate(timeTablePath)
            status.startStep("7_scorer", inputs=scoringInputs)
            startedAt = time.monotonic()
            try:
                if useTwoStage:
                    from src.pipeline.stage2_rerank import twoStageScore

                    result = twoStageScore(
                        roomId=self.webRid,
                        audioFeatures=audioFeat,
                        textFeatures=textFeat,
                        visualFeatures=visualFeat,
                        profile=self.profile,
                        minDuration=self.minDuration,
                        maxDuration=self.maxDuration,
                        durationSec=durationSec or None,
                        recallPerHour=_cfg.two_stage_recall_per_hour,
                        topK=_cfg.two_stage_top_k,
                        minGapSec=_cfg.two_stage_min_gap_sec,
                        recallThreshold=_cfg.two_stage_recall_threshold,
                        windowSeconds=_cfg.two_stage_window_sec,
                        minScore=_cfg.two_stage_min_score,
                        asrSegments=asrSegments,
                        dynamicDuration=_cfg.two_stage_dynamic_duration,
                        boundarySearchSec=_cfg.two_stage_boundary_search_sec,
                        platform=_cfg.two_stage_platform,
                        outputPath=timeTablePath,
                        scoresPath=scoresPath,
                        candidatesPath=candidatesPath,
                        runId=status.runId,
                    )
                else:
                    from src.scorer import score

                    result = score(
                        roomId=self.webRid,
                        audioFeatures=audioFeat,
                        textFeatures=textFeat,
                        visualFeatures=visualFeat,
                        profile=self.profile,
                        sensitivity=self.sensitivity,
                        minDuration=self.minDuration,
                        maxDuration=self.maxDuration,
                        durationSec=durationSec or None,
                        outputPath=timeTablePath,
                        scoresPath=scoresPath,
                    )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                log.exception("Scoring failed")
                status.finishStep("7_scorer", StepStatus.FAILED, error=err)
                return

            elapsedMs = int((time.monotonic() - startedAt) * 1000)
            clips = result.get("clips") or []
            pipelineLabel = result.get("pipeline", "legacy")
            stages = result.get("stages") or {}
            self._cache.record(
                timeTablePath,
                "7_scorer",
                scoringInputs,
                params=scoreParams,
                elapsedMs=elapsedMs,
                extra={
                    "pipeline": pipelineLabel,
                    "stages": stages,
                    "clips": len(clips),
                    "dropped": len(result.get("dropped") or []),
                    "adjustments": len(result.get("adjustments") or []),
                },
            )
            status.finishStep(
                "7_scorer",
                StepStatus.OK if clips else StepStatus.EMPTY,
                outputs=_scoreOutputs(pipelineLabel),
                error=None if clips else "no candidate clips survived detection",
                meta={
                    "pipeline": pipelineLabel,
                    "stages": stages,
                    "clips": len(clips),
                    "dropped": len(result.get("dropped") or []),
                    "adjustments": len(result.get("adjustments") or []),
                    "alignedSignals": len(result.get("alignment") or []),
                    "reused": False,
                    "cacheReason": scoreDecision.reason,
                    "elapsedMs": elapsedMs,
                },
            )

        if not clips:
            log.warning("No clips detected — see time_table.json dropped reasons")
            status.setStep(
                "8_clipping",
                StepStatus.SKIPPED,
                error="no clips to cut",
            )
            return

        if self._stopAfter("score"):
            log.info("--only-step score: stopping before clipping")
            return

        # ── Step 8: Clipping ───────────────────────────────────
        log.info("Step 8: Clipping video...")
        highlightsDir = os.path.join(self.outputDir, "highlights")
        from src.clipper import (
            clipOutputPath,
            clipVideo,
            extractThumbnail,
            generateClipSrt,
        )
        status.startStep("8_clipping", inputs=[videoPath, timeTablePath])
        if self.dryRun:
            log.info("--dry-run mode: skipping video cutting")
            status.finishStep(
                "8_clipping",
                StepStatus.SKIPPED,
                meta={"dryRun": True, "candidates": len(clips)},
            )
            log.info(f"Candidate report saved: {timeTablePath}")
            log.info(f"Scores timeline saved: {scoresPath}")
            return

        clipInputs = [videoPath, timeTablePath]
        expectedClips = [
            clip
            for clip in clips
            if float(clip.get("end", 0)) - float(clip.get("start", 0)) >= 1.0
        ]

        def _clipParams(clip: dict) -> dict[str, Any]:
            return {
                "resolution": self.resolution,
                "start": clip.get("start"),
                "end": clip.get("end"),
                "trigger": clip.get("trigger"),
            }

        if not expectedClips:
            status.finishStep(
                "8_clipping",
                StepStatus.EMPTY,
                error="all candidate clips are shorter than 1s",
            )
            return

        clipPaths: list[tuple[str, dict]] = []
        clipsReused = False
        clipDecisions = [
            self._checkCache(
                "8_clipping",
                clipOutputPath(clip, highlightsDir),
                clipInputs,
                params=_clipParams(clip),
                validator=(
                    lambda path=clipOutputPath(clip, highlightsDir): (
                        os.path.exists(path) and os.path.getsize(path) > 0,
                        "clip is missing or empty",
                    )
                ),
            )
            for clip in expectedClips
        ]
        if all(decision.hit for decision in clipDecisions):
            clipsReused = True
            clipPaths = [
                (clipOutputPath(clip, highlightsDir), clip) for clip in expectedClips
            ]
            log.info(f"Step 8: {len(clipPaths)} clips are cache hits")
            status.setStep(
                "8_clipping",
                StepStatus.SKIPPED,
                outputs=[path for path, _ in clipPaths],
                meta={
                    **self._cacheInfo(clipDecisions[0]),
                    "clips": len(clipPaths),
                },
            )
        else:
            firstMiss = next(
                (decision for decision in clipDecisions if not decision.hit), None
            )
            firstMissReason = firstMiss.reason if firstMiss else "clip rebuild"
            log.info(f"Step 8: {firstMissReason} — cutting {len(expectedClips)} clips")
            for clip in expectedClips:
                self._cache.invalidate(clipOutputPath(clip, highlightsDir))
            startedAt = time.monotonic()
            try:
                # Offload blocking ffmpeg calls to thread pool
                clipPaths = await asyncio.to_thread(
                    clipVideo, videoPath, timeTablePath, highlightsDir,
                    dryRun=False, resolution=self.resolution,
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                log.exception("Video clipping failed")
                status.finishStep("8_clipping", StepStatus.FAILED, error=err)
                return

            elapsedMs = int((time.monotonic() - startedAt) * 1000)
            for clipPath, clip in clipPaths:
                self._cache.record(
                    clipPath,
                    "8_clipping",
                    clipInputs,
                    params=_clipParams(clip),
                    elapsedMs=elapsedMs,
                    extra={"clips": len(clipPaths)},
                )
            status.finishStep(
                "8_clipping",
                StepStatus.OK if clipPaths else StepStatus.EMPTY,
                outputs=[path for path, _ in clipPaths],
                error=None if clipPaths else "ffmpeg produced no clip files",
                meta={
                    "clips": len(clipPaths),
                    "reused": False,
                    "cacheReason": firstMissReason,
                    "elapsedMs": elapsedMs,
                },
            )

        # ── Step 9: segment-level SRT + thumbnails ─────────────
        log.info("Step 9: Generating subtitles and thumbnails...")
        status.startStep("9_subtitles_thumbnails")
        srtPath = asrPath.rsplit(".", 1)[0] + ".srt"
        srtFailures = 0
        thumbFailures = 0
        reusedArtifacts = 0
        generated: list[str] = []
        for clipPath, clip in clipPaths:
            # Generate clip-specific SRT — best effort, the clip itself is
            # already produced and usable without subtitles.
            clipSrtPath = clipPath.rsplit(".", 1)[0] + ".srt"
            if clipsReused and os.path.exists(clipSrtPath):
                generated.append(clipSrtPath)
                reusedArtifacts += 1
            else:
                try:
                    os.remove(clipSrtPath)
                except OSError:
                    pass
                try:
                    await asyncio.to_thread(
                        generateClipSrt, srtPath, clip["start"], clip["end"], clipSrtPath,
                    )
                    generated.append(clipSrtPath)
                except Exception:
                    srtFailures += 1
                    log.debug(f"Clip SRT failed for {clipPath}", exc_info=True)
            # Thumbnail — same: best effort, offloaded to thread
            thumbPath = clipPath.rsplit(".", 1)[0] + ".jpg"
            if clipsReused and os.path.exists(thumbPath):
                generated.append(thumbPath)
                reusedArtifacts += 1
            else:
                try:
                    os.remove(thumbPath)
                except OSError:
                    pass
                try:
                    await asyncio.to_thread(
                        extractThumbnail, videoPath, clip["thumbnailTime"], thumbPath,
                    )
                    generated.append(thumbPath)
                except Exception:
                    thumbFailures += 1
                    log.debug(f"Thumbnail failed for {clipPath}", exc_info=True)
        if srtFailures:
            log.warning(f"{srtFailures}/{len(clipPaths)} clips have no SRT")
        if thumbFailures:
            log.warning(f"{thumbFailures}/{len(clipPaths)} clips have no thumbnail")
        allReused = clipsReused and reusedArtifacts == len(clipPaths) * 2
        status.finishStep(
            "9_subtitles_thumbnails",
            (
                StepStatus.SKIPPED
                if allReused
                else StepStatus.OK
                if not (srtFailures and thumbFailures)
                else StepStatus.EMPTY
            ),
            outputs=generated,
            meta={
                "srtFailures": srtFailures,
                "thumbnailFailures": thumbFailures,
                "reusedArtifacts": reusedArtifacts,
            },
        )

        log.info("=" * 50)
        log.info(f"Clipping complete: {len(clipPaths)} clips → {highlightsDir}")
        log.info("=" * 50)

    async def _writeRecordingMeta(
        self,
        videoPath: str,
        audioPath: str | None,
        segments: list[str],
    ) -> RecordingMeta:
        """Probe real media durations and write recording_meta.json atomically."""
        metaPath = os.path.join(self.outputDir, "recording_meta.json")
        probe = await asyncio.to_thread(probeMedia, videoPath)
        audioProbe: dict[str, Any] = {}
        if audioPath and os.path.exists(audioPath):
            audioProbe = await asyncio.to_thread(probeMedia, audioPath)

        existing = RecordingMeta.load(metaPath)
        sourceHashes: dict[str, Any] = {}
        for segment in segments:
            fingerprint = artifactFingerprint(segment)
            if fingerprint is not None:
                sourceHashes[segment] = fingerprint

        recorder = self._flvRecorder
        firstData = getattr(recorder, "firstDataAt", None) if recorder else None
        if hasattr(firstData, "isoformat"):
            firstData = firstData.isoformat()
        elif not isinstance(firstData, str):
            firstData = existing.firstVideoDataAt if existing else None

        startedAtWall = existing.startedAtWall if existing else None
        if not startedAtWall and self._recordStartTime:
            startedAtWall = datetime.fromtimestamp(
                self._recordStartTime, timezone.utc
            ).isoformat(timespec="seconds")

        meta = RecordingMeta(
            roomId=self.webRid,
            startedAtWall=startedAtWall,
            startedAtMonotonic=(
                existing.startedAtMonotonic
                if existing
                else (self._recordStartMono or None)
            ),
            firstVideoDataAt=firstData,
            videoDurationSec=float(
                probe.get("videoDurationSec") or probe.get("durationSec") or 0.0
            ),
            audioDurationSec=float(
                audioProbe.get("audioDurationSec")
                or audioProbe.get("durationSec")
                or probe.get("audioDurationSec")
                or 0.0
            ),
            segments=list(segments),
            sourceHashes=sourceHashes,
            ffprobe=probe,
        )
        meta.save(metaPath)
        log.info(f"Recording metadata: {metaPath}")
        return meta

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _safe_load_features(path: str) -> dict | None:
        """Load a JSON features file, replacing null values with safe defaults.

        Returns None if the file doesn't exist; otherwise returns the parsed
        dict with any null feature values replaced by empty dicts/lists.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to parse {path}: {e}")
            return None

        # Guard: if the top-level "features" key is null/missing, default it.
        if isinstance(data, dict):
            features = data.get("features")
            if features is None:
                data["features"] = {}
            elif isinstance(features, dict):
                # Replace any null feature arrays with empty lists
                for key, val in list(features.items()):
                    if val is None:
                        features[key] = []
        return data

    async def _extractAudio(self, flvPath: str, audioPath: str):
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", flvPath, "-vn", "-acodec", "copy", "-y", audioPath,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0 and os.path.exists(audioPath):
                log.info(f"Audio: {audioPath} ({fmtSize(os.path.getsize(audioPath))})")
            else:
                errMsg = stderr.decode(errors="replace")[-200:] if stderr else ""
                log.warning(f"Audio extraction failed (code={proc.returncode}): {errMsg}")
        except FileNotFoundError:
            log.warning("ffmpeg not installed")

    async def _mergeFlvSegments(self, segments: list[str]) -> str | None:
        """Concat FLV segments losslessly into a single MP4.

        Why merge instead of feeding ffmpeg a list each time downstream?
        Every later step (audio extract, ASR, signal extraction, clipping)
        assumes ONE continuous timeline.  Per-segment processing would
        force every stage to thread offsets through, which is exactly the
        kind of "manual time alignment" that goes wrong silently.

        Uses ffmpeg's concat demuxer with stream copy — no re-encode, no
        quality loss, takes seconds even for hours of footage.  Returns
        the merged path on success, or the single input path when there's
        only one segment, or None when ffmpeg isn't available.
        """
        valid = [s for s in segments if s and os.path.exists(s)]
        if not valid:
            return None

        # Never feed a half-written or corrupt segment into concat: it would
        # poison the whole merged timeline.  Quarantine bad inputs first.
        if shutil.which("ffprobe"):
            probed: list[str] = []
            for seg in valid:
                info = probeMedia(seg)
                duration = float(info.get("durationSec") or 0.0) if info else 0.0
                if duration <= 0:
                    target = self._quarantineSegment(seg)
                    log.warning(
                        "Quarantined corrupt FLV segment: %s → %s", seg, target
                    )
                    continue
                probed.append(seg)
            valid = probed
            if not valid:
                log.error("No usable FLV segments after validation")
                return None
        else:
            log.warning("ffprobe not installed — merging segments without validation")

        if len(valid) == 1:
            return valid[0]

        outPath = os.path.join(self.outputDir, f"{self.webRid}_merged.mp4")
        if os.path.exists(outPath):
            log.info(f"Reusing merged file: {outPath}")
            return outPath

        # ffmpeg concat demuxer requires a list file with `file '<path>'` lines.
        listPath = os.path.join(self.outputDir, ".concat_list.txt")
        try:
            with open(listPath, "w", encoding="utf-8") as fh:
                for seg in valid:
                    # Absolute path + single-quote escape (rare in our names but safe)
                    abs_seg = os.path.abspath(seg).replace("'", "'\\''")
                    fh.write(f"file '{abs_seg}'\n")

            log.info(f"Merging {len(valid)} FLV segments → {outPath}")
            ok = await self._concatFfmpeg(listPath, outPath, reencode=False)
            if not ok:
                log.warning("Lossless FLV merge failed — retrying with re-encode")
                try:
                    os.remove(outPath)
                except OSError:
                    pass
                ok = await self._concatFfmpeg(listPath, outPath, reencode=True)
            if not ok:
                return None
            log.info(f"Merged: {outPath} ({fmtSize(os.path.getsize(outPath))})")
            return outPath
        finally:
            try:
                os.remove(listPath)
            except OSError:
                pass

    async def _concatFfmpeg(
        self, listPath: str, outPath: str, reencode: bool
    ) -> bool:
        """Run one ffmpeg concat attempt; returns True on success."""
        codecArgs = (
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k"]
            if reencode
            else ["-c", "copy", "-bsf:a", "aac_adtstoasc"]
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", listPath, *codecArgs, "-y", outPath,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.warning("ffmpeg not installed — cannot merge FLV segments")
            return False
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not os.path.exists(outPath):
            errMsg = stderr.decode(errors="replace")[-300:] if stderr else ""
            mode = "re-encode" if reencode else "stream-copy"
            log.warning(
                f"FLV merge failed ({mode}, code={proc.returncode}): {errMsg}"
            )
            return False
        return True

    def _quarantineSegment(self, path: str) -> str | None:
        """Move an unreadable segment out of the merge path."""
        corruptDir = os.path.join(self.outputDir, "corrupt")
        try:
            os.makedirs(corruptDir, exist_ok=True)
            target = os.path.join(corruptDir, os.path.basename(path))
            if os.path.exists(target):
                target = f"{target}.{int(time.time())}"
            os.replace(path, target)
            return target
        except OSError:
            log.exception(f"Failed to quarantine segment: {path}")
            return None


# ── CLI ──────────────────────────────────────────────────────────

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Douyin Live AI Clipper")
    parser.add_argument("room_id", help="Live room ID")
    parser.add_argument("--clip", action="store_true", help="Run AI clipping after recording")
    parser.add_argument("--clip-only", action="store_true", help="Only run AI clipping on recorded files")
    parser.add_argument("--profile", default="default", choices=["default", "game", "shopping", "talent"],
                        help="Weight profile for scoring")
    parser.add_argument("--sensitivity", type=float, default=None, help="Peak detection sensitivity (2.0-3.5)")
    parser.add_argument("--min-duration", type=float, default=None, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum clip duration in seconds")
    parser.add_argument("--preprocess", action="store_true", help="Enable auto-editor preprocessing")
    parser.add_argument("--resolution", default=None, choices=["720p", "1080p", "4k"],
                        help="Output resolution")
    parser.add_argument("--dry-run", action="store_true", help="Only output time table, skip clipping")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every derived artifact, ignoring cached metadata",
    )
    parser.add_argument(
        "--from-step",
        choices=STEP_ORDER,
        help="Reuse earlier stages, rebuild this stage and every later one",
    )
    parser.add_argument(
        "--only-step",
        choices=STEP_ORDER,
        help="Stop after this stage (prerequisites are reused when valid)",
    )
    scoringGroup = parser.add_mutually_exclusive_group()
    scoringGroup.add_argument(
        "--two-stage",
        action="store_true",
        dest="twoStage",
        help="Use stage-1 recall + stage-2 reranking instead of the legacy scorer",
    )
    scoringGroup.add_argument(
        "--legacy-scoring",
        action="store_true",
        dest="legacyScoring",
        help="Force the legacy point-scoring path even when config enables two-stage",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
                        help="Log level (default: INFO)")
    args, _ = parser.parse_known_args()

    # Apply log level before anything else
    if args.log_level != "INFO":
        from src.log.logger import setLogLevel
        setLogLevel(args.log_level)

    ctrl = Controller(
        webRid=args.room_id,
        clip=args.clip,
        clipOnly=(
            args.clip_only
            or (bool(args.from_step or args.only_step) and not args.clip)
        ),
        profile=args.profile,
        sensitivity=args.sensitivity,
        minDuration=args.min_duration,
        maxDuration=args.max_duration,
        preprocess=args.preprocess,
        resolution=args.resolution,
        dryRun=args.dry_run,
        outputDir=args.output_dir,
        force=args.force,
        fromStep=args.from_step,
        onlyStep=args.only_step,
        twoStage=(
            True if args.twoStage else (False if args.legacyScoring else None)
        ),
    )
    await ctrl.start()


if __name__ == "__main__":
    asyncio.run(main())
