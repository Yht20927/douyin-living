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
├── *.meta.json                            ← 产物缓存元数据（源 hash/配置 hash）
└── highlights/
    ├── clip_342s_dmDensity+asrKeyword.mp4 ← 切片视频
    ├── clip_342s_dmDensity+asrKeyword.srt ← 切片字幕
    ├── clip_342s_dmDensity+asrKeyword.jpg ← 封面缩略图
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
| 视觉 | scenedetect + OpenCV + insightface | 场景/光流/人脸 |
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
