#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a candidate review page and annotation template (S5-6).

The report shows the top-K clips from a ``time_table.json`` with thumbnails,
optional context videos, and controls for ``worthClipping`` / quality / type /
notes.  The page is self-contained (inline CSS/JS, base64 thumbnails) and can
export an annotation JSON from the browser.  A template JSON is written next
to it for manual editing or version control.

Usage::

    python scripts/label_report.py \\
        --time-table data/room1/time_table.json \\
        --video data/room1/room1_merged.mp4 \\
        --top 10 --output data/room1/label_report

Open ``data/room1/label_report/index.html`` in a browser, review every
candidate, then export the JSON (or edit ``annotations.template.json``).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    clipIou,
    isPositive,
    loadAnnotations,
    loadTimeTable,
    rankClips,
)
from src.log.logger import getLogger  # noqa: E402

log = getLogger(__name__)

TYPE_OPTIONS = [
    "highlight",
    "funny",
    "high_energy",
    "interactive",
    "shopping",
    "emotional",
    "controversy",
    "dead_air",
    "other",
]

PREVIEW_SEC = 6.0
MATCH_IOU = 0.5


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>切片候选标注 — __ROOM__</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #1d2433;
  --muted: #667085;
  --line: #e4e7ec;
  --accent: #2563eb;
  --accent-soft: #eff4ff;
  --danger: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 80px;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
        "PingFang SC", "Microsoft YaHei", sans-serif;
}
header, main, footer { max-width: 1180px; margin: 0 auto; }
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }
h2 { margin: 0 0 6px; font-size: 18px; line-height: 1.35; }
.meta { color: var(--muted); font-size: 14px; }
.notice {
  margin: 20px 0;
  padding: 14px 16px;
  border: 1px solid #d1e0ff;
  border-radius: 10px;
  background: var(--accent-soft);
  color: #1e3a8a;
  font-size: 14px;
}
.toolbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin: 18px 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(8px);
}
button {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 14px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}
button.primary { border-color: var(--accent); background: var(--accent); color: #fff; }
button:hover { filter: brightness(0.97); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 18px; }
.card {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel);
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.thumb { background: #101828; aspect-ratio: 16 / 9; }
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.thumb .empty {
  display: flex; align-items: center; justify-content: center;
  height: 100%; color: #98a2b3; font-size: 13px;
}
.card-body { padding: 14px; }
.row { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.rank { font: 600 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); }
.score { color: var(--muted); font-size: 13px; }
.times { margin: 4px 0 8px; color: var(--muted); font-size: 13px; }
.tags { min-height: 22px; margin-bottom: 10px; }
.tag {
  display: inline-block; margin: 0 5px 5px 0; padding: 2px 8px;
  border-radius: 999px; background: #f2f4f7; color: #475467; font-size: 12px;
}
.fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
label.field { display: block; font-size: 12px; color: var(--muted); }
label.field span { display: block; margin-bottom: 4px; }
select, input, textarea {
  width: 100%; border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 9px; background: #fff; color: var(--ink); font: inherit;
}
textarea { min-height: 62px; resize: vertical; }
.field.full { grid-column: 1 / -1; }
video { width: 100%; margin-top: 10px; border-radius: 8px; background: #000; }
details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 8px; }
.custom {
  margin-top: 32px; padding: 18px; border: 1px solid var(--line);
  border-radius: 14px; background: var(--panel);
}
.custom h2 { margin-top: 0; }
.custom-form { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.custom-form .wide { grid-column: span 2; }
.custom-list { margin: 14px 0 0; padding: 0; list-style: none; }
.custom-list li {
  display: flex; justify-content: space-between; gap: 12px; padding: 9px 0;
  border-top: 1px dashed var(--line); font-size: 14px;
}
.custom-list button { padding: 4px 9px; color: var(--danger); }
footer { margin-top: 36px; color: var(--muted); font-size: 13px; }
@media (max-width: 640px) {
  body { padding: 20px 12px 60px; }
  .custom-form { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>切片候选标注</h1>
  <div class="meta">__SUMMARY__</div>
  <div class="notice">
    对每个候选选择“值得切 / 不值得切”，并填写质量与类型。未判断的候选不会写入导出文件；
    导出会保留已有标注中未被本次候选覆盖的记录。如需补充 Top-K 之外的高光或负样本，
    请在页面底部“新增标注”中添加。
  </div>
  <div class="toolbar">
    <button class="primary" id="exportBtn">导出标注 JSON</button>
    <button id="copyBtn">复制 JSON</button>
    <span class="meta" id="decisionCount"></span>
  </div>
</header>
<main>
  <section class="grid" id="cards">__CARDS__</section>
  <section class="custom">
    <h2>新增标注（Top-K 之外）</h2>
    <div class="custom-form">
      <label class="field"><span>开始（秒）</span><input id="newStart" type="number" min="0" step="0.1"></label>
      <label class="field"><span>结束（秒）</span><input id="newEnd" type="number" min="0" step="0.1"></label>
      <label class="field"><span>类型</span><select id="newType">__TYPE_OPTIONS__</select></label>
      <label class="field"><span>质量 1–5</span><select id="newQuality">__QUALITY_OPTIONS__</select></label>
      <label class="field wide"><span>备注</span><input id="newNotes" placeholder="例如：真实笑点 / 不应切的冷场"></label>
      <label class="field"><span>判断</span><select id="newWorth"><option value="true">值得切</option><option value="false">不值得切</option></select></label>
    </div>
    <p><button id="addBtn">添加标注</button></p>
    <ul class="custom-list" id="customList"></ul>
  </section>
</main>
<footer>
  标注说明见 <code>docs/annotation-guide.md</code>；导出后可用
  <code>python scripts/evaluate.py</code> 计算指标。
</footer>
<script>
const REPORT_DATA = __REPORT_DATA__;
const customAnnotations = [];
let customSeq = 0;

function fmtTime(value) {
  const sec = Math.max(0, Math.round(Number(value) || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const pad = n => String(n).padStart(2, "0");
  return h ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function refreshCount() {
  const cards = [...document.querySelectorAll(".card")];
  const decided = cards.filter(card => card.querySelector(".worth").value !== "").length;
  document.getElementById("decisionCount").textContent =
    `${decided}/${cards.length} 个候选已判断 · 新增 ${customAnnotations.length} 条`;
}

function collectAnnotations() {
  const annotations = [];
  const decidedIds = new Set();
  let undecided = 0;
  for (const card of document.querySelectorAll(".card")) {
    const worth = card.querySelector(".worth").value;
    if (worth === "") { undecided += 1; continue; }
    decidedIds.add(card.dataset.id);
    annotations.push({
      id: card.dataset.id,
      start: Number(card.dataset.start),
      end: Number(card.dataset.end),
      type: card.querySelector(".type").value,
      quality: Number(card.querySelector(".quality").value),
      worthClipping: worth === "true",
      notes: card.querySelector(".notes").value.trim(),
    });
  }
  const baseAnnotations = (REPORT_DATA.annotations || [])
    .filter(item => !decidedIds.has(item.id));
  return {
    annotations: baseAnnotations.concat(annotations, customAnnotations),
    undecided,
  };
}

function buildDocument() {
  const { annotations, undecided } = collectAnnotations();
  if (undecided > 0) {
    const ok = confirm(`还有 ${undecided} 个候选未判断，仍要导出吗？`);
    if (!ok) return null;
  }
  if (!annotations.length) {
    alert("至少需要一条标注（候选判断或新增标注）。");
    return null;
  }
  return {
    roomId: REPORT_DATA.roomId,
    duration: REPORT_DATA.duration,
    profile: REPORT_DATA.profile,
    annotations: annotations,
    exportedAt: new Date().toISOString(),
  };
}

async function copyJson() {
  const doc = buildDocument();
  if (!doc) return;
  const text = JSON.stringify(doc, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    alert("标注 JSON 已复制到剪贴板。");
  } catch (err) {
    window.prompt("复制失败，请手动复制：", text);
  }
}

document.getElementById("exportBtn").addEventListener("click", () => {
  const doc = buildDocument();
  if (!doc) return;
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${REPORT_DATA.fileStem || "annotations"}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

document.getElementById("copyBtn").addEventListener("click", copyJson);

document.getElementById("addBtn").addEventListener("click", () => {
  const start = Number(document.getElementById("newStart").value);
  const end = Number(document.getElementById("newEnd").value);
  const type = document.getElementById("newType").value;
  const quality = Number(document.getElementById("newQuality").value);
  const worthClipping = document.getElementById("newWorth").value === "true";
  const notes = document.getElementById("newNotes").value.trim();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    alert("请填写有效的开始/结束时间（结束必须大于开始）。");
    return;
  }
  customSeq += 1;
  customAnnotations.push({
    id: `custom_${customSeq}`,
    start: start,
    end: end,
    type: type,
    quality: quality,
    worthClipping: worthClipping,
    notes: notes,
  });
  renderCustom();
  document.getElementById("newStart").value = "";
  document.getElementById("newEnd").value = "";
  document.getElementById("newNotes").value = "";
});

function renderCustom() {
  const list = document.getElementById("customList");
  list.innerHTML = "";
  customAnnotations.forEach((item, index) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent =
      `${fmtTime(item.start)}–${fmtTime(item.end)} · ${item.type} · ` +
      `质量 ${item.quality} · ${item.worthClipping ? "值得切" : "不值得切"}` +
      (item.notes ? ` · ${item.notes}` : "");
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.addEventListener("click", () => {
      customAnnotations.splice(index, 1);
      renderCustom();
    });
    li.appendChild(label);
    li.appendChild(remove);
    list.appendChild(li);
  });
  refreshCount();
}

document.querySelectorAll(".card .worth").forEach(select => {
  select.addEventListener("change", refreshCount);
});
refreshCount();
</script>
</body>
</html>
"""


def _fmtTime(value: float | None) -> str:
    if value is None:
        return "?"
    seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _detectVideo(roomDir: Path) -> Path | None:
    candidates = [
        path
        for pattern in ("*.mp4", "*.flv")
        for path in roomDir.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _runFfmpeg(command: list[str], timeout: int) -> bool:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except FileNotFoundError:
        log.warning("ffmpeg not installed — cannot generate report media")
        return False
    except subprocess.TimeoutExpired:
        log.warning(f"ffmpeg timed out after {timeout}s: {' '.join(command[:6])}")
        return False
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        log.warning(
            f"ffmpeg failed: {detail[-1] if detail else exc.returncode}"
        )
        return False


def extractThumbnail(
    video: Path, timeSec: float, outputPath: Path, timeout: int = 60
) -> bool:
    """Extract one JPEG frame, retrying with accurate seek on failure."""
    commands = [
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timeSec):.3f}",
            "-i",
            str(video),
            "-vframes",
            "1",
            "-q:v",
            "3",
            "-y",
            str(outputPath),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-ss",
            f"{max(0.0, timeSec):.3f}",
            "-vframes",
            "1",
            "-q:v",
            "3",
            "-y",
            str(outputPath),
        ],
    ]
    for command in commands:
        if _runFfmpeg(command, timeout) and outputPath.exists():
            return True
    return False


def extractContext(
    video: Path,
    start: float,
    end: float,
    outputPath: Path,
    paddingSec: float,
    timeout: int = 180,
) -> bool:
    """Cut a short context clip; stream-copy first, re-encode as fallback."""
    contextStart = max(0.0, start - paddingSec)
    duration = max(1.0, end - start + paddingSec * 2)
    copyCommand = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{contextStart:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-y",
        str(outputPath),
    ]
    if _runFfmpeg(copyCommand, timeout) and outputPath.exists():
        return True

    encodeCommand = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{contextStart:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale=-2:360",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-y",
        str(outputPath),
    ]
    return _runFfmpeg(encodeCommand, timeout) and outputPath.exists()


def _imageDataUri(path: Path) -> str | None:
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/jpeg;base64,{payload}"


def _bestExistingMatch(candidate: dict, annotations: list[dict]) -> dict | None:
    best = None
    bestIou = 0.0
    for annotation in annotations:
        iou = clipIou(candidate, annotation)
        if iou >= MATCH_IOU and iou > bestIou:
            best = annotation
            bestIou = iou
    return best


def _options(values: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{html.escape(value, quote=True)}"'
        f'{" selected" if value == selected else ""}>'
        f"{html.escape(value)}</option>"
        for value in values
    )


def _qualityOptions(selected: float) -> str:
    return "".join(
        f'<option value="{value}"{" selected" if value == int(selected) else ""}>'
        f"{value}</option>"
        for value in range(1, 6)
    )


def _worthOptions(selected: str) -> str:
    choices = [
        ("", "— 未判断 —"),
        ("true", "值得切"),
        ("false", "不值得切"),
    ]
    return "".join(
        f'<option value="{value}"{" selected" if value == selected else ""}>'
        f"{label}</option>"
        for value, label in choices
    )


def buildCard(
    candidate: dict,
    rank: int,
    annotationId: str,
    thumbnailUri: str | None,
    contextPath: str | None,
    existing: dict | None,
) -> str:
    title = html.escape(str(candidate.get("title") or "（无标题）"))
    trigger = html.escape(str(candidate.get("trigger") or ""))
    tags = candidate.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tagsHtml = "".join(
        f'<span class="tag">{html.escape(str(tag))}</span>' for tag in tags
    )
    if trigger:
        tagsHtml += f'<span class="tag">trigger: {trigger}</span>'

    if thumbnailUri:
        thumbHtml = f'<img src="{thumbnailUri}" alt="候选 {rank} 缩略图" loading="lazy">'
    else:
        thumbHtml = '<div class="empty">缩略图不可用</div>'

    if existing:
        worthValue = "true" if isPositive(existing) else "false"
        quality = float(existing.get("quality", 3))
        annotationType = str(existing.get("type") or "highlight")
        notes = html.escape(str(existing.get("notes") or ""))
    else:
        worthValue = ""
        quality = 3.0
        annotationType = "highlight"
        notes = ""

    typeOptions = _options(
        TYPE_OPTIONS + ([annotationType] if annotationType not in TYPE_OPTIONS else []),
        annotationType,
    )
    videoHtml = ""
    if contextPath:
        videoHtml = (
            f'<details><summary>上下文预览</summary>'
            f'<video controls preload="metadata" src="{html.escape(contextPath, quote=True)}"></video>'
            f"</details>"
        )

    start = float(candidate.get("start", 0.0))
    end = float(candidate.get("end", start))
    duration = float(candidate.get("duration") or (end - start))
    score = candidate.get("score")
    scoreText = f"score {float(score):.3f}" if score is not None else "score —"

    return (
        f'<article class="card" data-id="{html.escape(annotationId, quote=True)}" '
        f'data-start="{start:.3f}" data-end="{end:.3f}">'
        f'<div class="thumb">{thumbHtml}</div>'
        f'<div class="card-body">'
        f'<div class="row"><span class="rank">#{rank}</span>'
        f'<span class="score">{scoreText}</span></div>'
        f"<h2>{title}</h2>"
        f'<p class="times">{_fmtTime(start)} – {_fmtTime(end)} · {duration:.1f}s</p>'
        f'<div class="tags">{tagsHtml}</div>'
        f'<div class="fields">'
        f'<label class="field"><span>判断</span>'
        f'<select class="worth">{_worthOptions(worthValue)}</select></label>'
        f'<label class="field"><span>质量 1–5</span>'
        f'<select class="quality">{_qualityOptions(quality)}</select></label>'
        f'<label class="field full"><span>类型</span>'
        f'<select class="type">{typeOptions}</select></label>'
        f'<label class="field full"><span>备注</span>'
        f'<textarea class="notes" placeholder="例如：笑点真实 / 只是背景音乐">{notes}</textarea></label>'
        f"</div>{videoHtml}</div></article>"
    )


def writeTemplate(
    path: Path,
    table: dict,
    candidates: list[dict],
    existing: list[dict],
) -> None:
    document = {
        "roomId": str(table.get("roomId") or ""),
        "duration": table.get("duration"),
        "profile": str(table.get("profile") or ""),
        "instructions": (
            "positive: worthClipping=true 且 quality>=3；negative: 其余。"
            "先用 label_report 审核候选，再补充 Top-K 之外的高光与负样本。"
        ),
        "annotations": existing,
        "candidates": [
            {
                "id": f"c{index}",
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "score": candidate.get("score"),
                "title": candidate.get("title"),
                "tags": candidate.get("tags") or [],
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)


def buildReport(
    timeTablePath: str,
    annotationPath: str | None,
    videoPath: str | None,
    outputDir: str,
    top: int,
    contextSec: float,
    makeThumbnails: bool,
    makeContext: bool,
) -> dict:
    table = loadTimeTable(timeTablePath)
    existingDoc = loadAnnotations(annotationPath) if annotationPath else None
    existing = existingDoc["annotations"] if existingDoc else []
    clips = rankClips(table.get("clips") or [])[:top]

    outputPath = Path(outputDir)
    mediaDir = outputPath / "media"
    mediaDir.mkdir(parents=True, exist_ok=True)

    video = Path(videoPath) if videoPath else _detectVideo(Path(timeTablePath).parent)
    if video and not video.exists():
        raise ValueError(f"video not found: {video}")
    if video is None:
        log.warning("no video found — thumbnails and context previews are skipped")

    cards = []
    candidates = []
    usedExistingIds: set[str] = set()
    reservedIds = {str(annotation["id"]) for annotation in existing}
    usedCardIds: set[str] = set()
    for index, clip in enumerate(clips, start=1):
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", start))
        thumbnailPath = mediaDir / f"c{index}.jpg"
        contextPath = mediaDir / f"c{index}_context.mp4"
        # Remove stale artifacts from a previous run so the template never
        # points at media that this run failed to regenerate.
        thumbnailPath.unlink(missing_ok=True)
        contextPath.unlink(missing_ok=True)

        thumbnailOk = False
        if video and makeThumbnails:
            thumbnailTime = float(clip.get("thumbnailTime", (start + end) / 2))
            thumbnailOk = extractThumbnail(video, thumbnailTime, thumbnailPath)

        contextOk = False
        if video and makeContext:
            contextOk = extractContext(
                video, start, end, contextPath, contextSec
            )

        existingMatch = _bestExistingMatch(
            clip,
            [
                annotation
                for annotation in existing
                if annotation["id"] not in usedExistingIds
            ],
        )
        if existingMatch:
            usedExistingIds.add(str(existingMatch["id"]))
            annotationId = str(existingMatch["id"])
        else:
            suffix = index
            annotationId = f"cand_{suffix}"
            while annotationId in reservedIds or annotationId in usedCardIds:
                suffix += 1
                annotationId = f"cand_{suffix}"
        usedCardIds.add(annotationId)
        cards.append(
            buildCard(
                clip,
                index,
                annotationId,
                _imageDataUri(thumbnailPath) if thumbnailOk else None,
                (
                    contextPath.relative_to(outputPath).as_posix()
                    if contextOk
                    else None
                ),
                existingMatch,
            )
        )
        candidates.append(
            {
                "id": f"c{index}",
                "start": start,
                "end": end,
                "score": clip.get("score"),
                "title": clip.get("title"),
                "tags": clip.get("tags") or [],
                "thumbnail": (
                    thumbnailPath.relative_to(outputPath).as_posix()
                    if thumbnailOk
                    else None
                ),
                "context": (
                    contextPath.relative_to(outputPath).as_posix()
                    if contextOk
                    else None
                ),
            }
        )

    roomId = str(table.get("roomId") or (existingDoc or {}).get("roomId") or "")
    generatedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = (
        f"room={html.escape(roomId or '?')} · "
        f"profile={html.escape(str(table.get('profile') or '?'))} · "
        f"候选 {len(clips)} 个 · "
        f"已有标注 {len(existing)} 条 · "
        f"{html.escape(generatedAt)}"
    )
    if not clips:
        cardsHtml = (
            '<div class="notice">time_table.json 中没有候选片段。'
            "请先运行切片流程，或直接用页面底部表单添加人工标注。</div>"
        )
    else:
        cardsHtml = "".join(cards)

    reportData = {
        "roomId": roomId,
        "duration": table.get("duration"),
        "profile": str(table.get("profile") or ""),
        "fileStem": f"{roomId or 'annotations'}_annotations",
        "generatedAt": generatedAt,
        "annotations": existing,
    }
    scriptData = json.dumps(reportData, ensure_ascii=False).replace("</", "<\\/")
    htmlText = (
        HTML_TEMPLATE.replace("__ROOM__", html.escape(roomId or "?"))
        .replace("__SUMMARY__", summary)
        .replace("__TYPE_OPTIONS__", _options(TYPE_OPTIONS, "highlight"))
        .replace("__QUALITY_OPTIONS__", _qualityOptions(3))
        .replace("__CARDS__", cardsHtml)
        .replace("__REPORT_DATA__", scriptData)
    )
    htmlPath = outputPath / "index.html"
    htmlPath.write_text(htmlText, encoding="utf-8")

    templatePath = outputPath / "annotations.template.json"
    writeTemplate(templatePath, table, candidates, existing)

    return {
        "html": str(htmlPath),
        "template": str(templatePath),
        "outputDir": str(outputPath),
        "candidates": len(clips),
        "existingAnnotations": len(existing),
        "video": str(video) if video else None,
        "thumbnails": sum(1 for item in candidates if item["thumbnail"]),
        "contexts": sum(1 for item in candidates if item["context"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a candidate review page and annotation template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--time-table",
        required=True,
        help="time_table.json produced by the pipeline",
    )
    parser.add_argument(
        "--annotations",
        help="existing annotation JSON to prefill matching candidates",
    )
    parser.add_argument(
        "--video",
        help="source video (default: largest .mp4/.flv next to time_table.json)",
    )
    parser.add_argument(
        "--output",
        help="output directory (default: <time_table dir>/label_report)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of top-ranked candidates (default: 10)",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=PREVIEW_SEC,
        help=f"context padding in seconds (default: {PREVIEW_SEC})",
    )
    parser.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="skip thumbnail extraction",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="skip context video extraction",
    )
    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.context_sec < 0:
        parser.error("--context-sec must be >= 0")

    timeTablePath = Path(args.time_table)
    outputDir = (
        Path(args.output)
        if args.output
        else timeTablePath.parent / "label_report"
    )
    try:
        summary = buildReport(
            str(timeTablePath),
            args.annotations,
            args.video,
            str(outputDir),
            args.top,
            args.context_sec,
            not args.no_thumbnails,
            not args.no_context,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    print(f"Candidate review page: {summary['html']}")
    print(f"Annotation template:   {summary['template']}")
    print(
        f"Candidates: {summary['candidates']} · "
        f"thumbnails: {summary['thumbnails']} · "
        f"contexts: {summary['contexts']} · "
        f"existing annotations: {summary['existingAnnotations']}"
    )
    if summary["video"]:
        print(f"Video: {summary['video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
