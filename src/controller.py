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
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.auth import Auth
from src.params import Params
from src.signer import Signer
from src.roomApi import RoomApi
from src.danmakuWs import DanmakuWs
from src.flvRecorder import FlvRecorder
from src.protobuf.Live_pb2 import LiveResponse
from src.log.logger import getLogger
from src.util import fmtSize

log = getLogger(__name__)


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
        sensitivity: float = 2.5,
        minDuration: float = 15.0,
        maxDuration: float = 90.0,
        preprocess: bool = False,
        resolution: str = "720p",
        dryRun: bool = False,
        outputDir: str | None = None,
    ):
        self.webRid = webRid
        self.clip = clip
        self.clipOnly = clipOnly
        self.profile = profile
        self.sensitivity = sensitivity
        self.minDuration = minDuration
        self.maxDuration = maxDuration
        self.preprocess = preprocess
        self.resolution = resolution
        self.dryRun = dryRun
        self.outputDir = outputDir or os.path.join("data", webRid)

        self._auth: Auth | None = None
        self._danmakuWs: DanmakuWs | None = None
        self._flvTask: asyncio.Task | None = None
        self._stopEvent = asyncio.Event()
        self._danmakuMessages: list[dict[str, Any]] = []
        self._flvPath: str | None = None
        self._flvRecorder: FlvRecorder | None = None
        self._recordStartTime: float = 0.0  # wall-clock seconds when recording started

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
            self._danmakuWs.join(timeout=3)
        if self._flvTask:
            self._flvTask.cancel()
            try:
                await self._flvTask
            except asyncio.CancelledError:
                pass
        log.info(f"Danmaku: {len(self._danmakuMessages)} messages")
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
        Path(self.outputDir).mkdir(parents=True, exist_ok=True)
        jsonlPath = os.path.join(self.outputDir, f"{self.webRid}_danmaku.jsonl")

        def onDanmaku(msg: dict):
            msg["recvTimeSec"] = round(time.time() - self._recordStartTime, 3)
            self._danmakuMessages.append(msg)
            with open(jsonlPath, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
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

        self._danmakuWs = DanmakuWs(
            params=wsParams.toDict(),
            cookieStr=self._auth.cookieStr,
            onDanmaku=onDanmaku,
        )
        self._danmakuWs.start()

        # 6. FLV recording — extract signed URLs from page SSR, use FlvRecorder
        try:
            streamUrls = await RoomApi.getStreamUrls(self._auth, self.webRid)
            if streamUrls:
                flvUrl = streamUrls.get("or4") or streamUrls.get("hd") or streamUrls.get("sd") or list(streamUrls.values())[0]
                self._flvRecorder = FlvRecorder(
                    roomId=self.webRid,
                    url=flvUrl,
                    outputDir=self.outputDir,
                )
                # Share the same stop event so Controller.stop() stops the recorder
                self._flvRecorder._stopEvent = self._stopEvent
                self._flvTask = asyncio.create_task(self._flvRecorder.run())
                log.info(f"FLV started: {flvUrl[:80]}...")
            else:
                log.warning("No FLV URLs found")
        except Exception as e:
            log.warning(f"FLV not available: {e}")

        # 7. Wait for stop
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: self._stopEvent.set())
        except NotImplementedError:
            pass

        await self._stopEvent.wait()
        await self.stop()

        # Extract audio from the last downloaded FLV segment
        if self._flvRecorder and self._flvRecorder._downloadedFiles:
            lastFlv = self._flvRecorder._downloadedFiles[-1]
            self._flvPath = lastFlv
            audioPath = lastFlv.rsplit(".", 1)[0] + ".aac"
            await self._extractAudio(lastFlv, audioPath)
        elif self._flvRecorder and self._flvRecorder._currentFile:
            # Rotation hasn't happened yet — use current file
            self._flvPath = self._flvRecorder._currentFile

    # ── Clipping pipeline ──────────────────────────────────────

    async def _runClippingOnly(self):
        """Run clipping pipeline on existing recorded files."""
        self._auth = Auth.fromEnv()
        await self._runClippingPipeline()

    async def _runClippingPipeline(self):
        """Run the full AI clipping pipeline on recorded data."""
        log.info("=" * 50)
        log.info("Starting AI clipping pipeline")
        log.info("=" * 50)

        # Find the latest FLV file
        flvFiles = sorted(Path(self.outputDir).glob("*.flv"))
        if not flvFiles:
            log.error(f"No FLV files in {self.outputDir}")
            return

        videoPath = str(flvFiles[-1])
        audioPath = videoPath.rsplit(".", 1)[0] + ".aac"
        danmakuPath = os.path.join(self.outputDir, f"{self.webRid}_danmaku.jsonl")

        log.info(f"Video: {videoPath}")
        log.info(f"Audio: {audioPath}")
        log.info(f"Profile: {self.profile}, Sensitivity: {self.sensitivity}")

        # Step 1: Extract audio (if not exists)
        if not os.path.exists(audioPath):
            log.info("Step 1: Extracting audio...")
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", videoPath, "-vn", "-acodec", "copy", "-y", audioPath,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if not os.path.exists(audioPath):
                log.error("Audio extraction failed")
                return
            log.info(f"Audio: {audioPath}")

        # Step 2: ASR + diarization (skip if model unavailable)
        log.info("Step 2: ASR...")
        asrPath = audioPath.rsplit(".", 1)[0] + "_asr.json"
        if not os.path.exists(asrPath):
            try:
                from asr import transcribe
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info(f"ASR device: {device}")
                transcribe(audioPath, device=device, diarize=True, outputPath=asrPath)
            except Exception:
                log.warning("ASR failed — marking as unavailable", exc_info=True)
                with open(asrPath, "w") as f:
                    json.dump({"status": "failed", "segments": []}, f)
        else:
            log.info(f"ASR exists: {asrPath}")

        # Step 3: Preprocess (auto-editor, optional)
        # Skipped in Phase 1 — requires --preprocess flag

        # Step 4-6: Signal extraction (parallel)
        log.info("Steps 4-6: Extracting signals (parallel)...")
        audioFeatPath = os.path.join(self.outputDir, "audio_features.json")
        textFeatPath = os.path.join(self.outputDir, "text_features.json")
        visualFeatPath = os.path.join(self.outputDir, "visual_features.json")

        def _runAudio(audPath: str, outPath: str):
            from src.signalAudio import extractFeatures
            extractFeatures(audPath, outputPath=outPath)

        def _runText(danPath: str, asrPath: str, outPath: str):
            from src.signalText import extractFeatures
            extractFeatures(danPath, asrPath, outputPath=outPath)

        def _runVisual(vidPath: str, outPath: str):
            from src.signalVisual import extractFeatures
            extractFeatures(vidPath, outputPath=outPath)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            if not os.path.exists(audioFeatPath):
                futures.append(pool.submit(_runAudio, audioPath, audioFeatPath))
            else:
                log.info(f"  Audio features exist, skipping: {audioFeatPath}")

            if not os.path.exists(textFeatPath):
                if os.path.exists(danmakuPath) and os.path.exists(asrPath):
                    futures.append(pool.submit(_runText, danmakuPath, asrPath, textFeatPath))
                else:
                    log.warning("  Missing danmaku or ASR — text features will be empty")

            if not os.path.exists(visualFeatPath):
                futures.append(pool.submit(_runVisual, videoPath, visualFeatPath))
            else:
                log.info(f"  Visual features exist, skipping: {visualFeatPath}")

            for f in futures:
                try:
                    f.result()
                except Exception:
                    log.error("Signal extraction failed", exc_info=True)

        # Load features for scorer
        audioFeat = None
        textFeat = None
        visualFeat = None

        if os.path.exists(audioFeatPath):
            with open(audioFeatPath) as f:
                audioFeat = json.load(f)
        if os.path.exists(textFeatPath):
            with open(textFeatPath) as f:
                textFeat = json.load(f)
        if os.path.exists(visualFeatPath):
            with open(visualFeatPath) as f:
                visualFeat = json.load(f)

        if not audioFeat and not textFeat and not visualFeat:
            log.error("No features extracted — cannot score")
            return

        # Step 7: Scoring
        log.info("Step 7: Scoring...")
        timeTablePath = os.path.join(self.outputDir, "time_table.json")
        scoresPath = os.path.join(self.outputDir, "scores.json")
        from src.scorer import score
        result = score(
            roomId=self.webRid,
            audioFeatures=audioFeat or {"features": {}},
            textFeatures=textFeat or {"features": {}},
            visualFeatures=visualFeat or {"features": {}},
            profile=self.profile,
            sensitivity=self.sensitivity,
            minDuration=self.minDuration,
            maxDuration=self.maxDuration,
            outputPath=timeTablePath,
            scoresPath=scoresPath,
        )

        if not result.get("clips"):
            log.warning("No clips detected — try lowering sensitivity")
            return

        # Step 8: Clipping
        log.info("Step 8: Clipping video...")
        highlightsDir = os.path.join(self.outputDir, "highlights")
        from src.clipper import clipVideo, extractThumbnail, generateClipSrt
        clipPaths = clipVideo(videoPath, timeTablePath, highlightsDir, dryRun=self.dryRun, resolution=self.resolution)

        if self.dryRun:
            log.info("--dry-run mode: skipping subtitle burn and thumbnails")
            log.info(f"Candidate report saved: {timeTablePath}")
            log.info(f"Scores timeline saved: {scoresPath}")
            return

        # Step 9: Generate segment-level SRT + thumbnails
        log.info("Step 9: Generating subtitles and thumbnails...")
        srtPath = asrPath.rsplit(".", 1)[0] + ".srt"
        for i, clipPath in enumerate(clipPaths):
            clip = result["clips"][i]
            # Generate clip-specific SRT
            clipSrtPath = clipPath.rsplit(".", 1)[0] + ".srt"
            try:
                generateClipSrt(srtPath, clip["start"], clip["end"], clipSrtPath)
            except Exception:
                pass
            # Thumbnail
            thumbPath = clipPath.rsplit(".", 1)[0] + ".jpg"
            try:
                extractThumbnail(videoPath, clip["thumbnailTime"], thumbPath)
            except Exception:
                pass

        log.info("=" * 50)
        log.info(f"Clipping complete: {len(clipPaths)} clips → {highlightsDir}")
        log.info("=" * 50)

        # Write pipeline status
        statusPath = os.path.join(self.outputDir, "pipeline_status.json")
        status = {
            "roomId": self.webRid,
            "profile": self.profile,
            "sensitivity": self.sensitivity,
            "steps": {
                "1_audio_extract": "ok",
                "2_asr": "ok" if os.path.exists(asrPath) else "skipped",
                "4_audio_signal": "ok" if os.path.exists(audioFeatPath) else "skipped",
                "5_text_signal": "ok" if os.path.exists(textFeatPath) else "skipped",
                "6_visual_signal": "ok" if os.path.exists(visualFeatPath) else "skipped",
                "7_scorer": "ok" if result.get("clips") else "no_clips",
                "8_clipping": "ok" if clipPaths else "no_clips",
            },
            "clips": len(clipPaths),
            "outputDir": highlightsDir,
        }
        with open(statusPath, "w") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        log.info(f"Pipeline status: {statusPath}")

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





# ── CLI ──────────────────────────────────────────────────────────

async def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Douyin Live AI Clipper")
    parser.add_argument("room_id", help="Live room ID")
    parser.add_argument("--clip", action="store_true", help="Run AI clipping after recording")
    parser.add_argument("--clip-only", action="store_true", help="Only run AI clipping on recorded files")
    parser.add_argument("--profile", default="default", choices=["default", "game", "shopping", "talent"],
                        help="Weight profile for scoring")
    parser.add_argument("--sensitivity", type=float, default=2.5, help="Peak detection sensitivity (2.0-3.5)")
    parser.add_argument("--min-duration", type=float, default=15.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=90.0, help="Maximum clip duration in seconds")
    parser.add_argument("--preprocess", action="store_true", help="Enable auto-editor preprocessing")
    parser.add_argument("--resolution", default="720p", choices=["720p", "1080p", "4k"],
                        help="Output resolution")
    parser.add_argument("--dry-run", action="store_true", help="Only output time table, skip clipping")
    parser.add_argument("--output-dir", help="Override output directory")
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
        clipOnly=args.clip_only,
        profile=args.profile,
        sensitivity=args.sensitivity,
        minDuration=args.min_duration,
        maxDuration=args.max_duration,
        preprocess=args.preprocess,
        resolution=args.resolution,
        dryRun=args.dry_run,
        outputDir=args.output_dir,
    )
    await ctrl.start()


if __name__ == "__main__":
    asyncio.run(main())
