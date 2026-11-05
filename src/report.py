# -*- coding: utf-8 -*-
"""Self-contained dry-run HTML report for scoring artifacts (S6-1).

The report is intentionally read-only: it consumes ``time_table.json``,
``scores.json``, ``recall_candidates.json``, the danmaku JSONL file and the
ASR JSON file produced by the pipeline, then renders one HTML file with inline
CSS/JS, an inline SVG score curve, base64 thumbnails, keyword summaries and
copy-ready ffmpeg commands.  No CDN or network access is required.

Typical usage from the controller::

    python -m src.controller 12345 --clip-only --dry-run \\
        --report-html data/12345/report/index.html
"""

from __future__ import annotations

import base64
import html
import json
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.log.logger import getLogger

log = getLogger(__name__)

# Tokens are kept short so the report rarely grows beyond a few MB even for
# hour-long streams.  Scores are bucket-max downsampled before drawing.
MAX_CURVE_POINTS = 1200
TOP_WORDS_DEFAULT = 20
ASR_EXCERPT_CHARS = 1500
ASR_SEGMENT_LIMIT = 200
RECALL_ROW_LIMIT = 50
DROPPED_ROW_LIMIT = 200

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")
_BRACKET_RE = re.compile(r"\[[^\]]{0,8}\]")
_TOKEN_RE = re.compile(
    r"__(?:ROOM|SUMMARY|STATS|SCORE_SVG|CANDIDATES|DROPPED|DANMAKU|SIGNALS|"
    r"ASR|RECALL|SOURCES)__"
)

_STOPWORDS = frozenset(
    {
        "什么", "怎么", "可以", "这个", "那个", "我们", "你们", "他们",
        "就是", "还是", "已经", "没有", "一个", "一下", "不是", "真的",
        "感觉", "东西", "现在", "因为", "所以", "但是", "然后", "如果",
        "知道", "喜欢", "谢谢", "主播", "直播", "链接", "多少", "时候",
        "大家", "老师", "兄弟", "姐妹", "宝宝", "朋友", "来了", "好的",
        "好吧", "哈哈", "哈哈哈", "呵呵", "嗯嗯", "对对", "是的", "收到",
        "关注", "点赞", "下单", "买买", "问一下", "我想", "有没有", "还有",
        "今天", "明天", "昨天", "这边", "那边", "这样", "那样", "一样",
        "非常", "特别", "比较", "应该", "可能", "不过", "而且", "或者",
        "以及", "对于", "关于", "里面", "外面", "上面", "下面", "一次",
        "两个", "几个", "真的", "好好", "一直", "已经", "开始", "觉得",
    }
)
_STOP_CHARS = frozenset("的了呢吗吧啊呀哦嗯哈是在有和就都也很还要会说我你他她它这那哪个不没")


# ── generic helpers ────────────────────────────────────────────

def _safeFloat(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _fmtTime(value: Any) -> str:
    """Format seconds as ``mm:ss`` or ``hh:mm:ss``."""
    seconds = max(0, int(round(_safeFloat(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _fmtSeconds(value: Any) -> str:
    return f"{max(0.0, _safeFloat(value)):.1f}s"


def _fmtTimeRange(start: Any, end: Any) -> str:
    return f"{_fmtTime(start)} – {_fmtTime(end)}"


def _loadJson(path: str | Path | None) -> Any | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        log.warning(f"Report: cannot read JSON artifact {path}")
        return None


def _loadJsonLines(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError:
        log.warning(f"Report: cannot read JSONL artifact {path}")
    return rows


def _empty(message: str) -> str:
    return f'<div class="empty">{html.escape(message)}</div>'


def _chips(values: Iterable[Any], className: str = "chip") -> str:
    items: list[str] = []
    for value in values:
        if isinstance(value, dict):
            text = ", ".join(f"{key}={item}" for key, item in value.items())
        else:
            text = str(value)
        if not text.strip():
            continue
        items.append(f'<span class="{className}">{html.escape(text)}</span>')
    return "".join(items) or '<span class="muted">—</span>'


def _resolveOutputPaths(
    outputPath: str | Path | None,
    timeTablePath: str | Path,
) -> tuple[Path, Path]:
    """Return ``(htmlPath, mediaDir)`` for a file or directory argument."""
    if outputPath is None:
        output = Path(timeTablePath).parent / "report"
    else:
        output = Path(outputPath)
    if output.suffix.lower() in (".html", ".htm"):
        htmlPath = output
        mediaDir = output.parent / f"{output.stem}_media"
    else:
        htmlPath = output / "index.html"
        mediaDir = output / "media"
    return htmlPath, mediaDir


# ── data extraction ────────────────────────────────────────────

def _scoreValues(scores: dict[str, Any] | None) -> list[float]:
    if not isinstance(scores, dict):
        return []
    raw = scores.get("scores")
    if not isinstance(raw, list):
        return []
    return [_safeFloat(value) for value in raw]


def _loadAsr(path: str | Path | None) -> dict[str, Any]:
    """Normalise a bare list or ``{status, segments, error}`` ASR document."""
    result: dict[str, Any] = {
        "status": "missing",
        "segments": [],
        "error": "",
        "language": "",
    }
    data = _loadJson(path)
    if data is None:
        return result
    declaredStatus = "ok"
    error = ""
    if isinstance(data, list):
        rawSegments = data
    elif isinstance(data, dict):
        rawSegments = data.get("segments")
        declaredStatus = str(data.get("status") or "ok").lower()
        error = str(data.get("error") or "")
        result["language"] = str(data.get("language") or "")
    else:
        result["status"] = "invalid"
        result["error"] = "顶级 JSON 必须是数组或对象"
        return result
    if not isinstance(rawSegments, list):
        result["status"] = "invalid"
        result["error"] = "segments 必须是数组"
        return result
    segments = [item for item in rawSegments if isinstance(item, dict)]
    result["segments"] = segments
    if declaredStatus == "failed":
        result["status"] = "failed"
        result["error"] = error or "ASR 报告失败"
    elif not segments:
        result["status"] = "empty"
        result["error"] = error or "ASR 没有返回片段"
    else:
        result["status"] = "ok"
    return result


def _asrText(segment: dict[str, Any]) -> str:
    for key in ("text", "content", "sentence"):
        value = segment.get(key)
        if value:
            return str(value).strip()
    return ""


def _tokenize(text: str) -> list[str]:
    """Best-effort tokenizer that works with or without jieba installed.

    jieba is an optional AI dependency, so the report falls back to CJK
    n-grams (2-3 chars) plus ASCII words.  ``_topWords`` removes stop words
    and single characters after tokenisation.
    """
    cleaned = _BRACKET_RE.sub(" ", text)
    candidates: list[str] = []
    for match in _CJK_RE.finditer(cleaned):
        run = match.group()
        if len(run) <= 4:
            candidates.append(run)
        else:
            candidates.extend(run[i:i + 2] for i in range(len(run) - 1))
            candidates.extend(run[i:i + 3] for i in range(len(run) - 2))
    candidates.extend(match.group().lower() for match in _ASCII_WORD_RE.finditer(cleaned))
    return candidates


def _topWords(
    messages: list[dict[str, Any]],
    limit: int = TOP_WORDS_DEFAULT,
) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for entry in messages:
        text = str(entry.get("content") or entry.get("text") or "").strip()
        if not text:
            continue
        for token in _tokenize(text):
            token = token.strip()
            if len(token) < 2:
                continue
            if token in _STOPWORDS:
                continue
            if any(char not in _STOP_CHARS for char in token):
                counter[token] += 1
    return counter.most_common(limit)


def _chatMessages(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or entry.get("text") or "").strip()
        if not content:
            continue
        if entry.get("type") not in (None, "", "chat", "danmaku"):
            continue
        chats.append(entry)
    return chats


# ── SVG score curve ────────────────────────────────────────────

def _downsample(values: list[float], target: int = MAX_CURVE_POINTS) -> list[float]:
    if len(values) <= target:
        return values
    bucket = len(values) / float(target)
    result: list[float] = []
    for index in range(target):
        start = int(index * bucket)
        end = max(start + 1, int((index + 1) * bucket))
        result.append(max(values[start:end]))
    return result


def _buildScoreSvg(
    values: list[float],
    clips: list[dict[str, Any]],
    duration: float,
) -> str:
    width, height = 1200, 280
    left, right, top, bottom = 58, 18, 18, 34
    plotWidth = width - left - right
    plotHeight = height - top - bottom
    if not values:
        return _empty("没有可绘制的分数时间轴（scores.json 缺失或为空）")

    sampled = _downsample(values)
    maxValue = max(sampled) if sampled else 0.0
    minValue = min(0.0, min(sampled) if sampled else 0.0)
    span = maxValue - minValue
    if span <= 0:
        span = 1.0
    duration = max(_safeFloat(duration), float(len(values)), 1.0)

    def xAt(second: float) -> float:
        return left + min(1.0, max(0.0, second / duration)) * plotWidth

    def yAt(value: float) -> float:
        ratio = (value - minValue) / span
        return top + (1.0 - ratio) * plotHeight

    points = " ".join(
        f"{left + index * plotWidth / max(1, len(sampled) - 1):.2f},"
        f"{yAt(value):.2f}"
        for index, value in enumerate(sampled)
    )
    bands: list[str] = []
    for clip in clips:
        start = _safeFloat(clip.get("start"))
        end = _safeFloat(clip.get("end"), start)
        x1, x2 = xAt(start), xAt(end)
        if x2 <= x1:
            continue
        clipId = html.escape(str(clip.get("id") or clip.get("rank") or ""))
        label = html.escape(_fmtTimeRange(start, end))
        bands.append(
            f'<rect class="clip-band" x="{x1:.2f}" y="{top}" '
            f'width="{max(1.5, x2 - x1):.2f}" height="{plotHeight}">'
            f"<title>{label} · #{clipId}</title></rect>"
        )

    grid: list[str] = []
    for step in range(5):
        ratio = step / 4.0
        y = top + ratio * plotHeight
        labelValue = maxValue - ratio * span
        grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{width - right}" y2="{y:.2f}"/>'
            f'<text class="axis" x="{left - 8}" y="{y + 4:.2f}" '
            f'text-anchor="end">{labelValue:.2f}</text>'
        )
    for step in range(5):
        ratio = step / 4.0
        x = left + ratio * plotWidth
        grid.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{top + plotHeight}"/>'
            f'<text class="axis" x="{x:.2f}" y="{height - 12}" '
            f'text-anchor="middle">{_fmtTime(ratio * duration)}</text>'
        )

    return (
        f'<svg class="curve" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="分数曲线">'
        + "".join(bands)
        + "".join(grid)
        + '<polyline class="curve-line" points="'
        + points
        + '"/></svg>'
        f'<div class="hint">横轴为时间，纵轴为分数；蓝色色带标记候选片段，'
        f"共 {len(clips)} 个候选。曲线已按桶最大值降采样至 "
        f"{len(sampled)} 个点。</div>"
    )


# ── HTML sections ──────────────────────────────────────────────

def _thumbsHtml(
    clip: dict[str, Any],
    rank: int,
    videoPath: str | None,
    mediaDir: Path,
    timeTableDir: Path,
    makeThumbnails: bool = True,
) -> tuple[str, str | None]:
    """Extract or reuse a thumbnail; return ``(html, relativePath)``."""
    thumbnailPath = mediaDir / f"c{rank:02d}.jpg"
    imagePath: Path | None = None
    if makeThumbnails:
        try:
            thumbnailPath.unlink(missing_ok=True)
        except OSError:
            pass
        thumbnailTime = clip.get("thumbnailTime")
        if thumbnailTime is None:
            thumbnailTime = (
                _safeFloat(clip.get("start")) + _safeFloat(clip.get("end"))
            ) / 2
        if videoPath and Path(videoPath).exists():
            if _extractThumbnail(
                Path(videoPath), _safeFloat(thumbnailTime), thumbnailPath
            ):
                imagePath = thumbnailPath
    if imagePath is None:
        imagePath = _findExistingThumbnail(timeTableDir, clip)
    if imagePath is None:
        return (
            '<div class="thumb"><div class="thumb-empty">无缩略帧</div></div>',
            None,
        )
    uri = _imageDataUri(imagePath)
    if uri is None:
        return (
            '<div class="thumb"><div class="thumb-empty">缩略帧不可读</div></div>',
            None,
        )
    return (
        f'<div class="thumb"><img src="{uri}" loading="lazy" '
        f'alt="候选 {rank} 缩略帧"></div>',
        imagePath.name,
    )


def _runFfmpeg(command: list[str], timeout: int = 30) -> bool:
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
        log.warning("Report: ffmpeg not installed — thumbnails are skipped")
        return False
    except subprocess.TimeoutExpired:
        log.warning(f"Report: ffmpeg timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        log.warning(f"Report: ffmpeg failed: {detail[-1] if detail else exc.returncode}")
        return False


def _extractThumbnail(video: Path, timeSec: float, outputPath: Path) -> bool:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timeSec):.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        "scale=640:-2",
        "-q:v",
        "4",
        "-y",
        str(outputPath),
    ]
    if _runFfmpeg(command) and outputPath.exists():
        return True
    # Accurate seek fallback for containers with unreliable indexes.
    fallback = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-ss",
        f"{max(0.0, timeSec):.3f}",
        "-frames:v",
        "1",
        "-vf",
        "scale=640:-2",
        "-q:v",
        "4",
        "-y",
        str(outputPath),
    ]
    return _runFfmpeg(fallback) and outputPath.exists()


def _findExistingThumbnail(
    timeTableDir: Path,
    clip: dict[str, Any],
) -> Path | None:
    highlights = timeTableDir / "highlights"
    if not highlights.is_dir():
        return None
    start = int(_safeFloat(clip.get("start")))
    matches = sorted(highlights.glob(f"clip_{start}s_*.jpg"))
    return matches[0] if matches else None


def _imageDataUri(path: Path) -> str | None:
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/jpeg;base64,{payload}"


def _breakdownHtml(breakdown: Any) -> str:
    if not isinstance(breakdown, dict) or not breakdown:
        return ""
    pairs = sorted(
        ((str(name), _safeFloat(value)) for name, value in breakdown.items()),
        key=lambda item: -abs(item[1]),
    )[:8]
    maxAbs = max((abs(value) for _, value in pairs), default=0.0) or 1.0
    rows: list[str] = []
    for name, value in pairs:
        width = max(2.0, abs(value) / maxAbs * 100.0)
        tone = "pos" if value >= 0 else "neg"
        rows.append(
            f'<div class="contrib-row"><code>{html.escape(name)}</code>'
            f'<span class="contrib-track"><i class="{tone}" '
            f'style="width:{width:.1f}%"></i></span>'
            f'<span class="contrib-value">{value:+.3f}</span></div>'
        )
    return (
        '<div class="contrib"><div class="subhead">信号贡献</div>'
        + "".join(rows)
        + "</div>"
    )


def _ffmpegCommand(
    videoPath: str | None,
    clip: dict[str, Any],
    rank: int,
) -> str:
    source = videoPath or "source.mp4"
    start = max(0.0, _safeFloat(clip.get("start")))
    end = max(start, _safeFloat(clip.get("end"), start))
    duration = max(0.1, end - start)
    trigger = str(clip.get("trigger") or f"clip{rank}")
    safeName = "".join(
        char for char in trigger if char.isalnum() or char in " _-+"
    )[:30] or f"clip{rank}"
    output = f"highlight_{rank:02d}_{int(start)}s_{safeName}.mp4"
    return (
        f"ffmpeg -hide_banner -ss {start:.3f} -i {shlex.quote(source)} "
        f"-t {duration:.3f} -c copy -y {shlex.quote(output)}"
    )


def _candidateCard(
    clip: dict[str, Any],
    rank: int,
    videoPath: str | None,
    mediaDir: Path,
    timeTableDir: Path,
    makeThumbnails: bool,
) -> tuple[str, str | None]:
    start = _safeFloat(clip.get("start"))
    end = max(start, _safeFloat(clip.get("end"), start))
    score = _safeFloat(clip.get("score"))
    confidence = clip.get("confidence")
    dropReasons = clip.get("dropReasons") or []
    if not isinstance(dropReasons, list):
        dropReasons = [dropReasons]

    reasons = clip.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        reasons = []
        trigger = clip.get("trigger")
        if trigger:
            reasons.append(str(trigger))
        breakdown = clip.get("breakdown")
        if isinstance(breakdown, dict):
            reasons.extend(
                name
                for name, value in sorted(
                    breakdown.items(), key=lambda item: -abs(_safeFloat(item[1]))
                )
                if _safeFloat(value) > 0
            )
    risks = clip.get("risks")
    if not isinstance(risks, list):
        risks = [risks] if risks else []

    tags = clip.get("tags") or []
    if not isinstance(tags, list):
        tags = [tags]
    title = str(clip.get("title") or f"候选片段 #{rank}")
    source = str(clip.get("candidateSource") or "")

    thumbHtml, thumbName = _thumbsHtml(
        clip, rank, videoPath, mediaDir, timeTableDir, makeThumbnails
    )
    command = _ffmpegCommand(videoPath, clip, rank)
    timecode = _fmtTimeRange(start, end)
    metaParts = [
        f"<code>{_fmtTime(start)}</code> → <code>{_fmtTime(end)}</code>",
        f"时长 {_fmtSeconds(end - start)}",
        f"分数 <b>{score:.3f}</b>",
    ]
    if confidence is not None:
        metaParts.append(f"confidence {_safeFloat(confidence):.2f}")
    if source:
        metaParts.append(f"来源 {html.escape(source)}")

    return (
        f'<article class="candidate" data-rank="{rank}">'
        f"{thumbHtml}"
        f'<div class="candidate-body">'
        f'<div class="candidate-head"><span class="rank">#{rank}</span>'
        f'<span class="score-pill">{score:.3f}</span></div>'
        f"<h3>{html.escape(title)}</h3>"
        f'<div class="meta">{" · ".join(metaParts)}</div>'
        f'<div class="tagrow">{_chips(tags)}</div>'
        f'<div class="facts">'
        f'<div class="fact"><span class="fact-label">触发原因</span>'
        f'<div>{_chips(reasons, "chip reasons")}</div></div>'
        f'<div class="fact"><span class="fact-label">风险</span>'
        f'<div>{_chips(risks, "chip risk")}</div></div>'
        f'<div class="fact"><span class="fact-label">丢弃原因</span>'
        f'<div>{_chips(dropReasons, "chip drop")}</div></div>'
        f"</div>"
        f"{_breakdownHtml(clip.get('breakdown'))}"
        f'<div class="actions">'
        f'<button type="button" data-copy="{html.escape(command, quote=True)}">'
        f"复制 ffmpeg 命令</button>"
        f'<button type="button" data-copy="{html.escape(timecode, quote=True)}">'
        f"复制时间码</button>"
        f"</div>"
        f'<pre class="cmd">{html.escape(command)}</pre>'
        f"</div></article>",
        thumbName,
    )


def _droppedHtml(dropped: Any, adjustments: Any) -> str:
    sections: list[str] = []
    if isinstance(dropped, list) and dropped:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in dropped:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or "unknown")
            groups.setdefault(reason, []).append(item)
        parts: list[str] = []
        for reason, items in sorted(groups.items(), key=lambda pair: -len(pair[1])):
            rows: list[str] = []
            for item in items[:DROPPED_ROW_LIMIT]:
                start = _safeFloat(item.get("start"))
                end = _safeFloat(item.get("end"), start)
                row = (
                    f"<li><code>{_fmtTimeRange(start, end)}</code> · "
                    f"{_fmtSeconds(end - start)} · score "
                    f"{_safeFloat(item.get('score')):.3f}"
                )
                if item.get("keptStart") is not None:
                    row += (
                        f" · 保留 {_fmtTimeRange(item.get('keptStart'), item.get('keptEnd'))}"
                    )
                rows.append(row + "</li>")
            parts.append(
                f'<div class="drop-group"><h3>{html.escape(reason)} '
                f'<span class="count">{len(items)}</span></h3>'
                f'<ul>{"".join(rows)}</ul></div>'
            )
        sections.append("".join(parts))
    if isinstance(adjustments, list) and adjustments:
        rows = []
        for item in adjustments[:DROPPED_ROW_LIMIT]:
            if not isinstance(item, dict):
                continue
            rows.append(
                f"<li><code>{_fmtTimeRange(item.get('start'), item.get('end'))}</code>"
                f" · 原始 "
                f"{_fmtTimeRange(item.get('originalStart'), item.get('originalEnd'))}"
                f" · {html.escape(str(item.get('reason') or 'adjusted'))}</li>"
            )
        sections.append(
            f'<details class="adjust"><summary>边界/时长调整 '
            f'({len(adjustments)})</summary><ul>{"".join(rows)}</ul></details>'
        )
    return "".join(sections) or _empty("没有候选被丢弃或调整")


def _danmakuHtml(
    entries: list[dict[str, Any]],
    topWords: list[tuple[str, int]],
) -> str:
    if not entries:
        return _empty("没有弹幕数据或弹幕文件缺失")
    chats = _chatMessages(entries)
    uniqueUsers = {
        str(entry.get("userId") or entry.get("user_id") or "")
        for entry in chats
    }
    uniqueUsers.discard("")
    stats = (
        f'<div class="inline-stats">'
        f"<span>消息 <b>{len(entries)}</b></span>"
        f"<span>可分析文本 <b>{len(chats)}</b></span>"
        f"<span>去重用户 <b>{len(uniqueUsers)}</b></span>"
        f"</div>"
    )
    if not topWords:
        return stats + _empty("弹幕文本不足以提取 Top 词")
    maxCount = max(count for _, count in topWords) or 1
    chips = []
    for word, count in topWords:
        level = 1 + min(3, int(count / maxCount * 3.99))
        chips.append(
            f'<span class="word w{level}" title="出现 {count} 次">'
            f"{html.escape(word)}<b>{count}</b></span>"
        )
    recent = []
    for entry in chats[-20:]:
        content = str(entry.get("content") or entry.get("text") or "").strip()
        if not content:
            continue
        timestamp = (
            entry.get("recvTimeSec", entry.get("timestamp", entry.get("_time", 0)))
        )
        recent.append(
            f"<li><code>{_fmtTime(timestamp)}</code> "
            f"{html.escape(content)}</li>"
        )
    recentHtml = (
        f'<details class="recent"><summary>最近弹幕</summary>'
        f'<ul>{"".join(recent)}</ul></details>'
        if recent
        else ""
    )
    return (
        stats
        + f'<div class="wordcloud">{"".join(chips)}</div>'
        + recentHtml
    )


def _signalHtml(scores: dict[str, Any] | None) -> str:
    if not isinstance(scores, dict):
        return _empty("scores.json 缺失，无法展示信号贡献")
    stats = scores.get("signalStats")
    if not isinstance(stats, dict) or not stats:
        return _empty("scores.json 中没有 signalStats")
    topSignals = scores.get("topSignals")
    if not isinstance(topSignals, list):
        topSignals = []
    ordered: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name in topSignals:
        item = stats.get(name)
        if isinstance(item, dict):
            ordered.append((str(name), item))
            seen.add(str(name))
    for name, item in sorted(
        stats.items(),
        key=lambda pair: -_safeFloat(
            pair[1].get("meanAbsContribution") if isinstance(pair[1], dict) else 0
        ),
    ):
        if name not in seen and isinstance(item, dict):
            ordered.append((str(name), item))
    ordered = ordered[:10]
    maxContribution = max(
        (_safeFloat(item.get("meanAbsContribution")) for _, item in ordered),
        default=0.0,
    ) or 1.0
    rows = []
    for name, item in ordered:
        contribution = _safeFloat(item.get("meanAbsContribution"))
        width = max(2.0, contribution / maxContribution * 100.0)
        rows.append(
            f'<div class="signal-row"><div class="signal-head">'
            f"<code>{html.escape(name)}</code>"
            f'<span class="muted">权重 {_safeFloat(item.get("weight")):.3f} · '
            f"meanZ {_safeFloat(item.get('meanZ')):+.3f} · "
            f"maxZ {_safeFloat(item.get('maxZ')):+.3f}</span></div>"
            f'<div class="signal-track"><i style="width:{width:.1f}%"></i></div>'
            f'<div class="signal-value">|贡献| {contribution:.4f}</div></div>'
        )
    return (
        '<div class="hint">Top-N 按全时间轴平均绝对贡献排序，仅用于解释'
        "评分权重，不代表单片段因果。</div>" + "".join(rows)
    )


def _asrHtml(asr: dict[str, Any]) -> str:
    status = asr.get("status")
    if status == "missing":
        return _empty("没有 ASR 产物（可能未运行、被缓存跳过或文件缺失）")
    if status == "failed":
        return _empty(f"ASR 失败：{asr.get('error') or '未知错误'}")
    if status == "invalid":
        return _empty(f"ASR 产物无效：{asr.get('error') or '格式错误'}")
    segments = asr.get("segments") or []
    if status == "empty" or not segments:
        return _empty("ASR 产物为空")

    speech = 0.0
    texts: list[str] = []
    rows: list[str] = []
    for segment in segments[:ASR_SEGMENT_LIMIT]:
        start = _safeFloat(segment.get("start"))
        end = _safeFloat(segment.get("end"), start)
        speech += max(0.0, end - start)
        text = _asrText(segment)
        if text:
            texts.append(text)
            rows.append(
                f"<li><code>{_fmtTimeRange(start, end)}</code> "
                f"{html.escape(text)}</li>"
            )
    fullText = " ".join(texts)
    excerpt = fullText[:ASR_EXCERPT_CHARS]
    if len(fullText) > ASR_EXCERPT_CHARS:
        excerpt += "…"
    language = f" · 语言 {html.escape(asr['language'])}" if asr.get("language") else ""
    stats = (
        f'<div class="inline-stats"><span>片段 <b>{len(segments)}</b></span>'
        f"<span>语音时长 <b>{_fmtSeconds(speech)}</b></span>"
        f"<span>字符 <b>{len(fullText)}</b></span></div>"
    )
    if not rows:
        return stats + _empty("ASR 片段没有文本字段")
    body = (
        stats
        + f'<p class="asr-text">{html.escape(excerpt)}{language}</p>'
        + f'<details class="recent"><summary>全部 ASR 片段 '
        f"({len(segments)})</summary><ul>{''.join(rows)}</ul></details>"
    )
    return body


def _recallHtml(
    recall: dict[str, Any] | None,
    candidatesPath: str | Path | None,
) -> str:
    if recall is None or not candidatesPath:
        return _empty("未使用两阶段评分或 recall_candidates.json 缺失")
    candidates = recall.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return _empty("Stage 1 没有召回候选")
    stats = recall.get("stats") or {}
    statLine = (
        f'<div class="inline-stats"><span>候选 <b>{len(candidates)}</b></span>'
        f"<span>上限 <b>{html.escape(str(stats.get('candidateCap', '?')))}</b></span>"
        f"<span>每小时候选 "
        f"<b>{html.escape(str(stats.get('recallPerHour', '?')))}</b></span>"
        f"<span>阈值 <b>{html.escape(str(stats.get('recallThreshold', '?')))}</b>"
        f"</span></div>"
    )
    ordered = sorted(
        (item for item in candidates if isinstance(item, dict)),
        key=lambda item: -_safeFloat(item.get("score")),
    )
    rows = []
    for item in ordered[:RECALL_ROW_LIMIT]:
        sources = item.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        rows.append(
            f"<tr><td><code>{_fmtTimeRange(item.get('start'), item.get('end'))}"
            f"</code></td><td>{_fmtSeconds(_safeFloat(item.get('duration')))}</td>"
            f"<td>{_safeFloat(item.get('score')):.4f}</td>"
            f"<td>{html.escape(str(item.get('anchor') or ''))}</td>"
            f"<td>{_chips(sources)}</td></tr>"
        )
    table = (
        '<div class="table-wrap"><table><thead><tr><th>时间</th><th>时长</th>'
        "<th>分数</th><th>anchor</th><th>来源</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    hidden = (
        f'<div class="hint">仅展示分数最高的 {RECALL_ROW_LIMIT} 个候选。</div>'
        if len(candidates) > RECALL_ROW_LIMIT
        else ""
    )
    return statLine + table + hidden


# ── template ───────────────────────────────────────────────────

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>切片 Dry-run 报告 — __ROOM__</title>
<style>
:root {
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #e3e8ef;
  --accent: #2563eb;
  --accent-soft: #eff4ff;
  --green: #067647;
  --amber: #b54708;
  --red: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 30px 20px 70px;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
        "PingFang SC", "Microsoft YaHei", sans-serif;
}
header, main, footer { max-width: 1240px; margin: 0 auto; }
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }
h2 { margin: 0 0 14px; font-size: 19px; }
h3 { margin: 0 0 8px; font-size: 16px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.meta { color: var(--muted); font-size: 13px; }
.muted { color: var(--muted); }
.hint { margin: 10px 0 0; color: var(--muted); font-size: 12.5px; }
.panel {
  margin-top: 18px; padding: 18px; border: 1px solid var(--line);
  border-radius: 14px; background: var(--panel);
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.stat-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px; margin-top: 16px;
}
.stat {
  padding: 12px 14px; border: 1px solid var(--line); border-radius: 12px;
  background: var(--panel);
}
.stat b { display: block; font-size: 22px; line-height: 1.2; }
.stat span { color: var(--muted); font-size: 12.5px; }
.inline-stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.inline-stats span { color: var(--muted); font-size: 13px; }
.inline-stats b { color: var(--ink); }
.empty {
  padding: 20px; border: 1px dashed var(--line); border-radius: 10px;
  color: var(--muted); text-align: center; font-size: 14px;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.curve { display: block; width: 100%; height: auto; }
.curve .grid { stroke: #eef1f5; stroke-width: 1; }
.curve .axis { fill: #7c8798; font-size: 11px; }
.curve .clip-band { fill: rgba(37, 99, 235, 0.14); }
.curve .curve-line {
  fill: none; stroke: #2563eb; stroke-width: 1.6;
  stroke-linejoin: round; stroke-linecap: round;
}
.candidate {
  display: grid; grid-template-columns: 300px 1fr; gap: 16px;
  padding: 16px 0; border-top: 1px solid var(--line);
}
.candidate:first-of-type { border-top: 0; padding-top: 0; }
.thumb {
  overflow: hidden; align-self: start; aspect-ratio: 16 / 9;
  border-radius: 10px; background: #101828;
}
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.thumb-empty {
  display: flex; align-items: center; justify-content: center;
  height: 100%; color: #98a2b3; font-size: 13px;
}
.candidate-head { display: flex; align-items: center; gap: 10px; }
.rank { color: var(--accent); font-weight: 700; font-size: 13px; }
.score-pill {
  padding: 1px 9px; border-radius: 999px; background: var(--accent-soft);
  color: #1e3a8a; font-size: 12px; font-weight: 600;
}
.candidate .meta { margin: 4px 0 10px; }
.candidate .meta code { color: #344054; }
.tagrow { margin-bottom: 10px; }
.chip {
  display: inline-block; margin: 0 6px 6px 0; padding: 2px 8px;
  border-radius: 999px; background: #f2f4f7; color: #475467; font-size: 12px;
}
.chip.reasons { background: #ecfdf3; color: #067647; }
.chip.risk { background: #fffaeb; color: #b54708; }
.chip.drop { background: #fef3f2; color: #b42318; }
.facts { display: grid; gap: 8px; margin-bottom: 10px; }
.fact { display: grid; grid-template-columns: 76px 1fr; gap: 10px; }
.fact-label { color: var(--muted); font-size: 12.5px; padding-top: 2px; }
.contrib { margin: 10px 0; padding: 10px 12px; border-radius: 10px; background: #f8fafc; }
.subhead { margin-bottom: 8px; color: var(--muted); font-size: 12px; }
.contrib-row {
  display: grid; grid-template-columns: 150px 1fr 68px; gap: 10px;
  align-items: center; margin-bottom: 5px; font-size: 12px;
}
.contrib-row code { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.contrib-track, .signal-track {
  display: block; overflow: hidden; height: 7px; border-radius: 999px;
  background: #e4e7ec;
}
.contrib-track i, .signal-track i { display: block; height: 100%; background: var(--accent); }
.contrib-track i.neg { background: var(--amber); }
.contrib-value { text-align: right; color: var(--muted); }
.actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 8px; }
button {
  border: 1px solid var(--line); border-radius: 8px; padding: 7px 12px;
  background: #fff; color: var(--ink); font: inherit; font-size: 13px;
  cursor: pointer;
}
button:hover { border-color: var(--accent); color: var(--accent); }
.cmd {
  margin: 0; padding: 9px 11px; overflow-x: auto; border-radius: 8px;
  background: #0f172a; color: #dbeafe; font-size: 12px; line-height: 1.5;
}
.drop-group { margin-top: 14px; }
.drop-group:first-child { margin-top: 0; }
.drop-group h3 { font-size: 14px; }
.count {
  display: inline-block; margin-left: 6px; padding: 0 7px; border-radius: 999px;
  background: #f2f4f7; color: var(--muted); font-size: 12px;
}
.drop-group ul, .recent ul, .adjust ul {
  margin: 6px 0 0; padding-left: 20px; color: #475467; font-size: 13px;
}
.drop-group li, .recent li, .adjust li { margin: 3px 0; }
details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 10px; }
.wordcloud { display: flex; flex-wrap: wrap; gap: 8px; }
.word {
  display: inline-flex; align-items: baseline; gap: 6px;
  padding: 4px 10px; border-radius: 999px; background: #f2f4f7;
  color: #344054; font-size: 13px;
}
.word b { color: var(--muted); font-size: 11px; font-weight: 600; }
.word.w2 { background: #e8efff; color: #1e3a8a; font-size: 14px; }
.word.w3 { background: #dbeafe; color: #1e40af; font-size: 16px; }
.word.w4 { background: #c7ddff; color: #1e3a8a; font-size: 18px; font-weight: 600; }
.asr-text {
  margin: 0 0 10px; padding: 12px 14px; border-radius: 10px;
  background: #f8fafc; color: #344054; font-size: 14px; white-space: pre-wrap;
}
.signal-row { margin-bottom: 12px; }
.signal-head { display: flex; justify-content: space-between; gap: 10px; font-size: 13px; }
.signal-head code { color: #1e3a8a; }
.signal-head .muted { font-size: 11.5px; }
.signal-track { height: 9px; margin: 6px 0 3px; }
.signal-value { color: var(--muted); font-size: 11.5px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
footer { margin-top: 28px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 820px) {
  body { padding: 20px 12px 50px; }
  .candidate { grid-template-columns: 1fr; }
  .grid2 { grid-template-columns: 1fr; }
  .thumb { max-width: 520px; }
  .contrib-row { grid-template-columns: 110px 1fr 58px; }
}
</style>
</head>
<body>
<header>
  <h1>切片 Dry-run 报告</h1>
  <div class="meta">__SUMMARY__</div>
  <div class="stat-grid">__STATS__</div>
</header>
<main>
  <section class="panel">
    <h2>分数曲线</h2>
    __SCORE_SVG__
  </section>
  <section class="panel">
    <h2>候选片段</h2>
    __CANDIDATES__
  </section>
  <section class="panel">
    <h2>丢弃与调整</h2>
    __DROPPED__
  </section>
  <section class="grid2">
    <section class="panel"><h2>弹幕 Top 词</h2>__DANMAKU__</section>
    <section class="panel"><h2>信号贡献 Top-N</h2>__SIGNALS__</section>
  </section>
  <section class="panel">
    <h2>ASR 摘要</h2>
    __ASR__
  </section>
  <section class="panel">
    <h2>Stage 1 召回候选</h2>
    __RECALL__
  </section>
</main>
<footer>
  本报告由 <code>--report-html</code> 生成，只读、自包含、无外部 CDN。
  数据源：__SOURCES__。片段分数仅用于 dry-run 审阅，未接入人工标注校准。
</footer>
<script>
function fallbackCopy(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
  area.remove();
  return ok;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    return fallbackCopy(text);
  }
}

document.querySelectorAll("button[data-copy]").forEach(button => {
  button.addEventListener("click", async () => {
    const text = button.getAttribute("data-copy") || "";
    const ok = await copyText(text);
    if (!ok) {
      window.prompt("复制失败，请手动复制：", text);
      return;
    }
    const original = button.textContent;
    button.textContent = "已复制";
    button.disabled = true;
    window.setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
    }, 1200);
  });
});
</script>
</body>
</html>
"""


# ── public API ─────────────────────────────────────────────────

def buildReport(
    timeTablePath: str | Path,
    *,
    scoresPath: str | Path | None = None,
    candidatesPath: str | Path | None = None,
    asrPath: str | Path | None = None,
    danmakuPath: str | Path | None = None,
    videoPath: str | Path | None = None,
    outputPath: str | Path | None = None,
    makeThumbnails: bool = True,
    topWords: int = TOP_WORDS_DEFAULT,
) -> dict[str, Any]:
    """Build a self-contained dry-run HTML report.

    Args:
        timeTablePath: ``time_table.json`` from the legacy or two-stage scorer.
        scoresPath: Optional ``scores.json`` with the score timeline.
        candidatesPath: Optional ``recall_candidates.json`` (two-stage).
        asrPath: Optional ASR JSON (bare list or ``{status, segments}``).
        danmakuPath: Optional danmaku JSONL.
        videoPath: Optional source video for thumbnail extraction.
        outputPath: Report file (``*.html``) or directory.  When a directory
            is given the report is written to ``<dir>/index.html``.
        makeThumbnails: Set ``False`` to skip ffmpeg extraction.
        topWords: Number of danmaku top words to render.

    Returns:
        Summary dict with the HTML path, media directory and artifact counts.
    """
    table = _loadJson(timeTablePath)
    if not isinstance(table, dict):
        raise ValueError(f"time_table.json is missing or invalid: {timeTablePath}")
    scores = _loadJson(scoresPath)
    recall = _loadJson(candidatesPath)
    asr = _loadAsr(asrPath)
    danmaku = _loadJsonLines(danmakuPath)

    htmlPath, mediaDir = _resolveOutputPaths(outputPath, timeTablePath)
    mediaDir.mkdir(parents=True, exist_ok=True)

    roomId = str(table.get("roomId") or "")
    if not roomId and isinstance(scores, dict):
        roomId = str(scores.get("roomId") or "")
    if not roomId:
        roomId = Path(timeTablePath).parent.name
    duration = _safeFloat(table.get("duration"))
    if duration <= 0:
        duration = float(len(_scoreValues(scores)))
    clips = [clip for clip in (table.get("clips") or []) if isinstance(clip, dict)]
    dropped = table.get("dropped") or []
    adjustments = table.get("adjustments") or []
    pipeline = str(table.get("pipeline") or (scores or {}).get("pipeline") or "legacy")
    profile = str(table.get("profile") or "default")

    chats = _chatMessages(danmaku)
    words = _topWords(chats, limit=topWords)
    scoreValues = _scoreValues(scores)

    timeTableDir = Path(timeTablePath).parent
    ranked = sorted(
        clips,
        key=lambda clip: (-_safeFloat(clip.get("score")), _safeFloat(clip.get("start"))),
    )
    cards: list[str] = []
    thumbnailCount = 0
    for rank, clip in enumerate(ranked, start=1):
        card, thumbnailName = _candidateCard(
            clip,
            rank,
            str(videoPath) if videoPath else None,
            mediaDir,
            timeTableDir,
            makeThumbnails,
        )
        if thumbnailName:
            thumbnailCount += 1
        cards.append(card)

    generatedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources = [
        f"time_table.json={Path(timeTablePath).name}",
    ]
    for label, path in (
        ("scores", scoresPath),
        ("recall", candidatesPath),
        ("asr", asrPath),
        ("danmaku", danmakuPath),
        ("video", videoPath),
    ):
        if path and Path(path).exists():
            sources.append(f"{label}={Path(path).name}")
    summary = (
        f"room={html.escape(roomId or '?')} · profile={html.escape(profile)} · "
        f"pipeline={html.escape(pipeline)} · 时长 {_fmtTime(duration)} · "
        f"生成于 {html.escape(generatedAt)}"
    )
    stats = [
        (str(len(clips)), "候选片段"),
        (str(len(dropped)), "丢弃记录"),
        (str(len(scoreValues)), "分数点"),
        (str(len(words)), "弹幕 Top 词"),
        (str(len(asr.get("segments") or [])), "ASR 片段"),
        (str(thumbnailCount), "缩略帧"),
    ]
    statsHtml = "".join(
        f'<div class="stat"><b>{html.escape(value)}</b>'
        f"<span>{html.escape(label)}</span></div>"
        for value, label in stats
    )
    candidatesHtml = (
        "".join(cards) if cards else _empty("time_table.json 中没有候选片段")
    )

    replacements = {
        "__ROOM__": html.escape(roomId or "?"),
        "__SUMMARY__": summary,
        "__STATS__": statsHtml,
        "__SCORE_SVG__": _buildScoreSvg(scoreValues, ranked, duration),
        "__CANDIDATES__": candidatesHtml,
        "__DROPPED__": _droppedHtml(dropped, adjustments),
        "__DANMAKU__": _danmakuHtml(danmaku, words),
        "__SIGNALS__": _signalHtml(scores if isinstance(scores, dict) else None),
        "__ASR__": _asrHtml(asr),
        "__RECALL__": _recallHtml(
            recall if isinstance(recall, dict) else None,
            candidatesPath,
        ),
        "__SOURCES__": html.escape(", ".join(sources)),
    }
    htmlText = _TOKEN_RE.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        HTML_TEMPLATE,
    )
    htmlPath.parent.mkdir(parents=True, exist_ok=True)
    htmlPath.write_text(htmlText, encoding="utf-8")
    log.info(
        f"Dry-run report saved: {htmlPath} "
        f"({len(clips)} candidates, {thumbnailCount} thumbnails)"
    )
    return {
        "html": str(htmlPath),
        "mediaDir": str(mediaDir),
        "roomId": roomId,
        "duration": duration,
        "pipeline": pipeline,
        "candidates": len(clips),
        "dropped": len(dropped) if isinstance(dropped, list) else 0,
        "thumbnails": thumbnailCount,
        "topWords": len(words),
        "asrSegments": len(asr.get("segments") or []),
        "scorePoints": len(scoreValues),
        "generatedAt": generatedAt,
    }
