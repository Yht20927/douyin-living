# 🎬 Douyin Live AI Clipper

抖音直播全自动录制 + AI 智能切片系统。

**录制** → **弹幕采集** → **语音识别** → **多信号融合打分** → **自动剪辑高光片段**

---

## ✨ 功能

| 模块 | 说明 |
|------|------|
| 📹 **直播录制** | 自动录制 FLV 视频流（4K 画质）+ AAC 音频 |
| 💬 **弹幕采集** | WebSocket 实时接收聊天/礼物/点赞/进场/房间统计 |
| 🎵 **音频提取** | `-acodec copy` 无损提取 AAC，保留原始 PTS 时间戳 |
| 🗣 **语音识别** | faster-whisper + WhisperX 说话人分离，输出 SRT 字幕 |
| 🧠 **AI 切片** | 3 层 AI + 7 项数学方法融合打分，自动定位高光时刻；可选两阶段召回/精排 |
| 🧾 **审核报告** | `--report-html` 输出自包含 dry-run 报告：分数曲线、候选卡片、缩略帧、弹幕 Top 词、ASR 摘要、可复制 ffmpeg 命令 |
| ✂️ **智能裁剪** | ffmpeg 无损剪辑 + 字幕烧录 + 封面缩略图 |

---

## 🚀 快速开始

### 1. 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.11+ | |
| Node.js | 18+ | 用于签名 JS 执行 |
| ffmpeg | 6.0+ | 视频处理 |
| CUDA | 12.x (可选) | GPU 加速 ASR |

### 2. 安装

```bash
# 克隆项目
git clone <repo-url>
cd douyin-living

# Python 依赖
pip install -r requirements.txt

# AI 切片依赖（仅录制可跳过）
pip install -r requirements-ai.txt

# Node.js 依赖（签名）
npm --prefix scripts install

# ffmpeg
winget install ffmpeg  # Windows
# brew install ffmpeg   # macOS
# apt install ffmpeg    # Linux
```

### 3. 配置 Cookie

打开 Chrome，访问 `live.douyin.com`，登录后按 F12 → Application → Cookies → 复制所有 Cookie。

创建 `.env` 文件：

```env
DY_COOKIES="你的Cookie字符串"
DY_LIVE_COOKIES="你的Cookie字符串"
```

也可以把原始 Cookie 字符串保存为项目根目录的 `cookie.txt`（或通过
`DY_COOKIE_FILE=/path/to/cookies.txt` 指定）。优先级：
`DY_LIVE_COOKIES` → `DY_COOKIES` → `DY_COOKIE_FILE` → `./cookie.txt`。

浏览器复制可能截断某个值（例如出现 `…`）。非 ASCII/格式非法的条目会被
跳过并只记录条目名，不会打印 Cookie 值；建议重新复制完整 Cookie 以恢复
登录态。`cookie.txt` 同时支持浏览器导出的 Netscape 格式。

### 4. 运行

```bash
# 仅录制直播（弹幕 + 视频）
python -m src.controller 300294032039

# 录制 + AI 切片（录制完成后自动分析）
python -m src.controller 300294032039 --clip

# 仅对已录制的视频做 AI 切片
python -m src.controller 300294032039 --clip-only

# 自定义参数
python -m src.controller 300294032039 --clip-only \
  --profile game \           # 直播类型: default/game/shopping/talent
  --sensitivity 2.0 \        # 灵敏度: 2.0-3.5 (越低切片越多)
  --min-duration 10 \        # 最小切片时长
  --max-duration 60          # 最大切片时长

# 断点续跑：复用 audio/asr/signal，从 score 重新打分并重新裁剪
python -m src.controller 300294032039 --from-step score

# 只重试失败或需要刷新的 ASR（前置 audio 命中缓存）
python -m src.controller 300294032039 --only-step asr

# 忽略所有 *.meta.json 缓存，强制重建全部派生文件
python -m src.controller 300294032039 --force

# Sprint 5 两阶段召回/精排（默认关闭，按需开启）
python -m src.controller 300294032039 --clip-only --two-stage

# 即使 config/pipeline.yaml 开启了 two_stage_enabled，也强制走旧评分
python -m src.controller 300294032039 --clip-only --legacy-scoring

# Sprint 6 dry-run 可视化报告（不裁剪，输出自包含 HTML）
python -m src.controller 300294032039 --clip-only --dry-run \
  --report-html data/300294032039/report/index.html
```

---

## 📦 独立工具

```bash
# 提取音频
python extractAudio.py video.flv           # → video.aac
python extractAudio.py video.flv out.mp3   # → out.mp3

# 语音识别
python asr.py audio.aac                    # → audio_asr.json + audio_asr.srt
python asr.py audio.aac --model tiny       # 快速模式
python asr.py audio.aac --no-diarize       # 跳过说话人分离
python asr.py audio.aac --chunk-sec 600    # 10 分钟分块 + 断点续跑
python asr.py audio.aac --no-resume        # 忽略已完成分块
```

---

## 📁 输出结构

```
data/300294032039/
├── 300294032039_20250603T144003.flv       ← 4K 视频录制
├── 300294032039_20250603T144003.aac       ← 无损音频
├── 300294032039_danmaku.jsonl             ← 弹幕数据
├── 300294032039_20250603T144003_asr.json  ← ASR 转录 + 说话人
├── 300294032039_20250603T144003_asr.srt   ← SRT 字幕
├── audio_features.json                    ← 音频分析结果
├── text_features.json                     ← 文本分析结果
├── visual_features.json                   ← 视觉分析结果
├── time_table.json                        ← 切片时间表
├── scores.json                            ← 全时间轴得分
├── recall_candidates.json                 ← 两阶段模式：阶段 A 候选窗口
├── pipeline_status.json                   ← 每步状态/耗时/缓存统计
├── annotations.json                       ← 人工标注（S5-6，手动维护）
├── eval.json                              ← 评估报告（S5-6，可选）
├── label_report/                          ← 候选审核页 + 标注模板（S5-6）
├── report/                                ← dry-run 可视化报告（S6-1，可选）
│   ├── index.html                         ← 自包含页面（内联 CSS/JS/SVG/缩略图）
│   └── media/                             ← 从源视频提取的缩略帧
├── *.meta.json                            ← 产物缓存元数据（源 hash/配置 hash）
└── highlights/
    ├── clip_342s_dmDensity+asrKeyword.mp4  ← 切片视频
    ├── clip_342s_dmDensity+asrKeyword.srt  ← 切片字幕
    ├── clip_342s_dmDensity+asrKeyword.jpg  ← 封面缩略图
    ├── clip_342s_dmDensity+asrKeyword.json ← 片段元数据/审核清单（S6-2）
    └── ...
```

---

## 🧠 AI 切片原理

### 多信号融合引擎

```
弹幕信号 ──→ 密度/加速度/熵值/情感/去重/反刷屏
             重复率/问号率/感叹号率/emoji 率/关键词类别强度
音频信号 ──→ RMS/频谱质心/MFCC/过零率/笑声/掌声
视觉信号 ──→ 场景切换/光流运动/人脸数量
ASR 信号 ──→ 关键词匹配/说话人交替/话题切换
    ↓
Z-Score 标准化 → 加权融合 → 时间衰减 → 突变奖励
    ↓
自适应阈值 + 多尺度检测 + NMS 去重
    ↓
语义边界对齐 + 高斯包络 + 局部 DP 优化
    ↓
时间表 → ffmpeg 批量无损剪辑
```

### 两阶段召回/精排（Sprint 5，可选）

开启 `--two-stage` 或 `PIPELINE_TWO_STAGE_ENABLED=true` 后：

```text
阶段 A（低成本全量召回）
  1 Hz 对齐 → median/MAD 鲁棒标准化 → profile 加权召回曲线
  + ASR 关键词/笑声/掌声/说话人切换/场景切点锚点
  → 每小时 30~80 个候选窗口（recall_candidates.json）

阶段 B（窗口级精排）
  peak / mean / area / 前后对比 / 多信号一致性 / novelty / durationFit
  + 语义事件 bonus → topK + 重叠抑制
  + 内容类型动态时长 + 边界联合优化
    （ASR 句界 / RMS 静音 / 场景切点 / 分数谷值）
  → confidence / reasons / risks / boundary → time_table.json
```

默认 `two_stage_enabled=false`，旧评分路径保留，便于用固定标注集做 A/B。
两阶段参数：`two_stage_recall_per_hour`、`two_stage_top_k`、`two_stage_min_gap_sec`、
`two_stage_recall_threshold`、`two_stage_window_sec`、`two_stage_min_score`，
均可在 `config/pipeline.yaml` 或 `PIPELINE_TWO_STAGE_*` 环境变量中配置。

S5-2 起阶段 B 还会根据 profile 与窗口内信号推断 `contentType`，并按内容类型收窄
目标时长：funny 12~35s、shopping 20~60s、interaction 30~90s、talent 45~120s、
game 15~60s、default 15~90s；用户 `--min-duration/--max-duration` 始终是硬边界。
可选的平台软限制由 `two_stage_platform`（`none/douyin/bilibili/xiaohongshu`）控制，
仅当平台区间与内容/用户区间有交集时生效。边界搜索半径由
`two_stage_boundary_search_sec` 控制，输出中的每个切片包含
`contentType/targetDuration/boundary.startSource/boundary.endSource/boundary.reasons`，
被移动的边界会额外记录 `boundary_optimized` adjustment。
`dynamicDuration`、`boundarySearchSec`、`platform` 也进入 scorer 缓存 key。

### 文本特征（Sprint 5 / S5-3）

- 新增按秒信号：`dmRepeatRate`、`dmQuestionRate`、`dmExclamationRate`、`dmEmojiRate`；
- 新增关键词类别强度：`keywordHighEnergy`、`keywordFunny`、`keywordControversy`、
  `keywordInteractive`、`keywordEmotional`、`keywordShopping`，由弹幕命中率与
  ASR 短语命中共同构成（0~1）；
- `config/keywords.json` 增加带货话术与直播常用短语，旧的 `asrKeyword` 契约保持不变；
- `dmUniqueUsers` 进入所有 profile 权重，真实互动人数不再只是诊断字段；
- `sentiment_mode=rules|hybrid|hf`：默认 `rules`，跳过逐条弹幕 HF 推理；
  `hybrid` 只对规则中性文本调用 HF，`hf` 保留旧行为；
- `utr_auto_calibrate=true` 时，从本场有弹幕秒的 UTR 分布取低分位数作为反刷屏阈值，
  并限制在 `utr_calibration_min~max`；样本不足时回退到 `utr_threshold`。
  实际阈值与来源写入 `text_features.json` 的 `stats.utr`。

相关参数位于 `config/text.yaml`，均支持 `TEXT_*` 环境变量。由于特征输出扩展，
`GENERATOR_VERSION` 已提升到 `sprint5`，旧派生产物会自动重建。

### 视觉性能（Sprint 5 / S5-4）

默认流程先用 ffmpeg 把源视频转成低帧率、低高度代理文件，再由单次 OpenCV
解码同时计算场景切点（HSV 帧差）与光流运动，源视频只解码一次：

```text
源视频 ──ffmpeg──→ proxy (sample_fps, proxy_height)
                         │
                         └─ 单次 OpenCV pass → sceneChange + motion
```

- `visual.proxy_enabled=true`：启用代理优先路径；ffmpeg 缺失或失败时自动回退旧的
  PySceneDetect + OpenCV 双解码路径，提取不会中断；
- `visual.sample_fps`（默认 2.0）与 `visual.proxy_height`（默认 480）控制代理大小；
- `visual.enable_face=false`：人脸检测留给候选窗口关键帧（Phase 2）；
- `visual_features.json` 在 `stats.mode` / `stats.sourceDecodePasses`
  中记录实际路径，`stats.proxy.fallbackReason` 记录回退原因。

基准脚本可在同一台机器上对比新旧路径：

```bash
python scripts/benchmark_visual.py --video /data/1h-1080p.flv
python scripts/benchmark_visual.py --video /data/1h-1080p.flv --legacy
python scripts/benchmark_visual.py --video /data/1h-1080p.flv --json /tmp/visual.json
```

建议用 1 小时 1080p 文件记录 `elapsedSec` 与 `realtimeFactor`，作为后续视觉改动的
回归指标。

### ASR 配置与质量（Sprint 5 / S5-5）

ASR 运行参数独立放在 `config/asr.yaml`，环境变量使用 `ASR_*`；旧
`PIPELINE_ASR_MODEL/DEVICE/COMPUTE_TYPE/DIARIZE` 仍作为别名兼容。

```yaml
# model=auto 时：CPU → auto_model_cpu（base），CUDA → auto_model_cuda（small）
model: "auto"
auto_model_cpu: "base"
auto_model_cuda: "small"
device: "auto"
compute_type: "int8"
language: "zh"
diarize: true
batch_size: 32
vad_filter: true
vad_min_silence_ms: 500
vad_speech_pad_ms: 200
beam_size: 5
chunk_sec: 1800
resume: true
```

- 默认不选 `large-v3`：4GB 显存和 CPU 环境无法稳定运行；需要时显式设置
  `ASR_MODEL=large-v3`；
- 当 `device=auto` 但 CUDA 不可用、或 CPU 上配置了 CUDA-only 的
  `float16/int8_float16/bfloat16` 时，会自动降级到 CPU/int8 并在日志中说明；
- 如果 CUDA 运行库缺失或无法加载（如 `libcublas.so.12`），ASR 会自动改用
  CPU/int8 重试并记录 warning；想避免首次失败和重复加载，可显式设置
  `ASR_DEVICE=cpu`；
- 超过 `chunk_sec`（默认 30 分钟）的音频会经 ffmpeg 重编码为 16 kHz 单声道分块；
  每完成一块就写 `<asr_json>.chunks.json`，进程中断后重跑会跳过已完成块；
  全部成功后删除检查点，失败则保留以便续跑。检查点签名包含音频大小/修改时间与
  全部 ASR 参数，音频或配置变化会自动失效；
- ASR 失败时 controller 仍写出 `{"status":"failed","segments":[],"error":...}`，
  `pipeline_status.json` 的 `2_asr` 标记为 `failed`；文本特征不会整体失败，而是
  降级为仅弹幕信号，并在 `text_features.json` 的
  `stats.asrStatus/asrDegraded/asrError` 记录原因。修复模型后：
  `--only-step asr` 重跑 ASR，再用 `--from-step signal` 继续；
- `faster-whisper` 固定在 1.x、`whisperx` 固定在 3.x，避免上游 API 漂移。

### 评估、调权与标注（Sprint 5 / S5-6）

S5-6 提供三套离线工具，全部读取已有产物，不重新跑模型：

```bash
# 1. 评估：time_table.json + 人工标注 JSON
python scripts/evaluate.py \
  --time-table data/300294032039/time_table.json \
  --annotations data/300294032039/annotations.json \
  --k 5 --k 10 --json /tmp/eval.json

# 2. 多场聚合（--case 可重复）
python scripts/evaluate.py \
  --case data/room1/time_table.json data/room1/annotations.json \
  --case data/room2/time_table.json data/room2/annotations.json \
  --k 5 --fail-on-target

# 3. 在固定特征上调权，不重跑音频/文本/视觉
python scripts/tune_weights.py \
  --case data/room1/time_table.json data/room1/annotations.json \
  --profile default --k 5 --strategy random --iterations 200 \
  --output-yaml /tmp/default.tuned.yaml --json /tmp/tune.json

# 4. 生成候选审核页和标注模板（自动选择同目录最大的 mp4/flv）
python scripts/label_report.py \
  --time-table data/300294032039/time_table.json \
  --top 10 --output data/300294032039/label_report
```

评估输出：

- Precision@K / Recall@K / F1@K；
- Boundary IoU 与 Editing Cost；
- Invalid Rate、Duplicate Rate、Negative Hit Rate；
- 逐 case 指标 + macro/micro 聚合 + 目标检查。

口径与标注规范见 `docs/annotation-guide.md`。初始目标（P@K≥0.6、R@K≥0.7、
Boundary IoU≥0.5、Invalid≤0.2、Duplicate≤0.1）只是建议值，必须用真实标注
校准；仓库当前尚未包含 5~10 场真实人工标注，不能把这些数字当对外 SLA。

### Profile 支持度分析（Sprint 5 / S5-7）

`scripts/profile_report.py` 用同一批标注集重评分所有 profile，回答“这个
profile 是否真的有数据支撑”：

```bash
python scripts/profile_report.py \
  --case data/room1/time_table.json data/room1/annotations.json \
  --case data/room2/time_table.json data/room2/annotations.json \
  --k 5 --min-cases 2 --min-positives 5 \
  --json /tmp/profile-report.json
```

判定规则：

- owned cases 少于 `--min-cases`（默认 2）或正样本少于 `--min-positives`
  （默认 5）→ `collect_more_or_merge`；
- 数据足够，但在 owned cases 上 micro F1 相对 `default` 的提升不足
  `--min-delta`（默认 0.02）→ `merge_into_default`；
- 否则 → `keep`；`default` 始终保留为兜底。

脚本只输出建议，不修改 `config/scorer.yaml`。当前仓库没有真实标注集，所有
profile 仍只是先验权重，不能宣称已被数据验证。

### Dry-run 可视化报告（Sprint 6 / S6-1）

`--report-html <path>` 在评分完成后生成只读报告，`--dry-run` 同样有效；
输出路径是 `.html`/`.htm` 文件时按文件写入，否则按目录写入 `index.html`：

```bash
python -m src.controller 300294032039 --clip-only --dry-run \
  --report-html data/300294032039/report/index.html
```

报告内容：

- 内联 SVG 分数曲线与候选时间带（无外部 CDN）；
- 候选卡片：起止时间、时长、分数、confidence、触发原因/风险/丢弃原因；
- 从源视频提取（失败时回退 `highlights/*.jpg`）的缩略帧，base64 内嵌；
- 弹幕 Top 词（可选 jieba，缺失时回退中文 n-gram）、最近弹幕；
- ASR 状态、片段数、语音时长、文本摘要与全部片段；
- 信号贡献 Top-N 与权重/meanZ/maxZ；
- 两阶段 `recall_candidates.json` 的召回窗口与来源；
- 一键复制 ffmpeg 命令与时间码。

报告生成失败只记录 warning，不会中断评分/裁剪主流程。报告中的分数尚未
经过人工标注校准，不能直接当作质量结论。

### 片段元数据（Sprint 6 / S6-2）

实际裁剪后，`highlights/` 中每个片段都会获得与 `.mp4` 同 stem 的 `.json`
审核清单；`--dry-run` 不生成：

```json
{
  "clipId": 1,
  "start": 342.0,
  "end": 367.0,
  "duration": 25.0,
  "score": 0.91,
  "confidence": 0.82,
  "title": "弹幕爆发 + 关键词命中",
  "tags": ["互动", "搞笑"],
  "reasons": ["dmDensity", "asrKeyword"],
  "risks": ["asr_missing"],
  "suggestedAction": "review",
  "scoreBreakdown": {"dmDensity": 0.31, "asrKeyword": 0.22},
  "assets": {
    "video": "clip_342s_dmDensity+asrKeyword.mp4",
    "srt": "clip_342s_dmDensity+asrKeyword.srt",
    "thumbnail": "clip_342s_dmDensity+asrKeyword.jpg"
  }
}
```

- `reasons` 优先使用两阶段评分给出的原因；legacy 片段回退到 `trigger` 与
  `breakdown` 中正贡献的 key；
- `confidence` 仅两阶段评分提供，legacy 片段为 `null`；
- `suggestedAction`：有 `dropReasons` → `drop`，有 `risks` → `review`，否则
  `keep`，只是人工审核提示；
- `assets` 记录同 stem 的实际文件，缺失的 SRT/封面为 `null`；
- 源 ASR SRT 缺失或片段区间无字幕时，`assets.srt` 为 `null`，并计入
  `9_subtitles_thumbnails.meta.srtFailures`；
- 元数据写入失败只记 warning（`pipeline_status.json` 的
  `9_subtitles_thumbnails.meta.metadataFailures`），不会中断裁剪。

### 直播类型预设

| Profile | 适用场景 | 核心权重 |
|---------|---------|---------|
| `default` | 通用 | 弹幕密度 + 笑声 + ASR 关键词 |
| `game` | 游戏直播 | 画面运动 + 弹幕爆发 + 掌声 |
| `shopping` | 带货直播 | ASR 关键词 + OCR（Phase 2）|
| `talent` | 才艺直播 | 音频能量 + 弹幕情感 + 说话人交替 |

---

## 🛠 技术栈

| 层 | 工具 | 用途 |
|----|------|------|
| 签名 | PyExecJS + dy_ab.js | a_bogus / X-Bogus |
| 弹幕 | websocket-client + Protobuf | LiveResponse 解码 |
| 音频 | librosa + panns-inference | RMS/MFCC + 事件检测 |
| ASR | faster-whisper + WhisperX | 语音识别 + 说话人分离 |
| NLP | jieba + FastText + text2vec + HF transformers | 分词/模糊匹配/聚类/情感 |
| 视觉 | ffmpeg 代理 + OpenCV（PySceneDetect 回退）+ insightface | 场景/光流/人脸 |
| 数学 | numpy + scipy + scikit-image | EWMA/Z-Score/互相关/SSIM/t-test |
| 剪辑 | ffmpeg | 无损裁剪 + 字幕烧录 + 缩略图 |

---

## 📋 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `room_id` | 必填 | 直播间 ID |
| `--clip` | off | 录制完成后运行 AI 切片 |
| `--clip-only` | off | 仅对已有文件做 AI 切片 |
| `--profile` | default | 直播类型: default/game/shopping/talent |
| `--sensitivity` | 2.5 | 峰值检测灵敏度 (2.0-3.5) |
| `--min-duration` | 15 | 最小切片时长 (秒) |
| `--max-duration` | 90 | 最大切片时长 (秒) |
| `--preprocess` | off | 启用 auto-editor 预处理 |
| `--resolution` | 720p | 输出分辨率: 720p/1080p/4k |
| `--dry-run` | off | 仅输出时间表不裁剪 |
| `--report-html` | off | 输出自包含 dry-run HTML 报告（文件或目录，配合 `--dry-run` 使用） |
| `--output-dir` | data/{roomId} | 自定义输出目录 |
| `--force` | off | 忽略缓存元数据，重建全部派生文件 |
| `--from-step` | 全部 | 复用此前步骤，从 audio/asr/signal/score/clip 重新执行 |
| `--only-step` | 关闭 | 只执行到指定步骤，用于单步重试 |
| `--two-stage` | off | 使用两阶段召回/精排评分（覆盖配置） |
| `--legacy-scoring` | off | 强制旧评分路径（覆盖配置） |

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| `docs/runbook.md` | 运行手册：Cookie/无流/ffmpeg/模型/磁盘/ASR OOM 与日志排查 |
| `docs/annotation-guide.md` | S5-6 标注规范、JSON 格式、指标口径与审核页用法 |
| `docs/superpowers/specs/2026-06-03-ai-clip-design.md` | AI 切片系统完整设计文档 |
| `config/keywords.json` | 关键词词典（高能/搞笑/争议/互动/情感/带货） |

---

## ⚠️ 注意事项

- **Cookie 有效期**：抖音 Cookie 通常 24 小时过期，需定期更新
- **GPU 加速**：ASR 推荐 CUDA 环境（conda 配置），CPU 模式较慢
- **模型下载**：首次运行会自动下载 faster-whisper/panns/FastText 等模型（共 ~5GB）
- **隐私**：用户 ID 使用 SHA256 哈希脱敏，人脸仅存储关键点坐标

---

## 🙏 致谢

本项目深度受益于 **[Douyin_Spider](https://github.com/cvv-cat/Douyin_Spider)**，感谢作者 **[cvv-cat](https://github.com/cvv-cat)** 在抖音直播协议逆向方面的开创性工作。

Douyin_Spider 为本项目提供了以下关键参考：

- 签名算法（a_bogus / X-Bogus / msToken）的 JS 实现
- WebSocket 弹幕 Protobuf 消息解码
- FLV 流 URL 提取与 SSR 页面解析
- Cookie 认证与 API 请求头构造

---

## 📄 License

MIT
