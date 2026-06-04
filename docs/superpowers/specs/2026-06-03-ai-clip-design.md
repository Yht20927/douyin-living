# 抖音直播 AI 智能切片系统 — 完整设计文档

> 版本: v3.0 | 日期: 2026-06-03 | 状态: 评审通过，待实施
>
> 本文档是唯一权威设计来源。替代已删除的 `design-review.md` 和 `tool-inventory.md`。

---

## 目录

1. [概述](#1-概述)
2. [架构总览](#2-架构总览)
3. [工具选型与验证](#3-工具选型与验证)
4. [信号提取层](#4-信号提取层)
5. [融合打分层（核心）](#5-融合打分层核心)
6. [执行流程](#6-执行流程)
7. [工程约束](#7-工程约束)
8. [实施计划](#8-实施计划)
9. [安装清单](#9-安装清单)
10. [风险矩阵与评审修正](#10-风险矩阵与评审修正)

---

## 1. 概述

### 1.1 目标

从抖音直播录制数据中自动识别"高光时刻"，裁剪为短视频片段，附带字幕、封面、标题、标签。

### 1.2 输入

| 数据 | 来源 | 格式 |
|------|------|------|
| 视频流 | `flvRecorder.py` | FLV (H.264, 最高 4K or4) |
| 音频流 | `extractAudio.py` | AAC 48kHz 立体声 |
| 弹幕流 | `danmakuWs.py` | JSONL（聊天/礼物/点赞/进场/房间统计） |
| ASR 文本 | `asr.py` + WhisperX | JSON（含 speaker 标签 + 句级时间戳） + SRT |

### 1.3 输出

```
data/{roomId}/highlights/
├── clip_0342s_dmSpike.mp4     ← 切片视频（720p/1080p 可选）
├── clip_0342s_dmSpike.srt     ← 字幕文件
├── clip_0342s_dmSpike.jpg     ← 封面缩略图
├── clip_0518s_gift.mp4
├── clip_0518s_gift.srt
├── ...
└── scores.json                ← 全时间轴得分（可追溯每帧打分依据）
```

### 1.4 原则

| 原则 | 含义 |
|------|------|
| **不重复造轮子** | 能用 GitHub 20k★ 项目的绝不自己写 |
| **先跑通再优化** | Phase 1 默认参数，Phase 2 数据驱动调优 |
| **显存友好** | 4GB GTX 1650 Ti 能跑全程 |
| **可解释** | 每个切片输出 `score_breakdown`，解释为什么切 |

---

## 2. 架构总览

### 2.1 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│                   录制阶段（已有，不改）                      │
│  controller.py → flvRecorder + danmakuWs                    │
│  extractAudio.py → audio.aac                                │
│  产出: xxx.flv + xxx.aac + xxx_danmaku.jsonl                │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ASR 阶段（asr.py 升级）                    │
│  faster-whisper large-v3 → 句级转录                          │
│  WhisperX diarization → 说话人标签 (SPEAKER_00/01/...)       │
│  产出: asr.json + subtitle.srt                              │
│  显存: ~3GB (INT8), 卸载后才进入下一步                        │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            信号提取阶段（并行 3 worker）                       │
│                                                             │
│  signalAudio.py    signalText.py     signalVisual.py         │
│  ┌─────────────┐   ┌──────────────┐  ┌──────────────────┐  │
│  │librosa:      │   │jieba: 分词   │  │PySceneDetect:    │  │
│  │ RMS/质心/    │   │FastText:     │  │ 场景切换检测     │  │
│  │ MFCC/ZCR     │   │ 模糊匹配     │  │                  │  │
│  │              │   │text2vec:     │  │OpenCV:           │  │
│  │panns:        │   │ 话题聚类     │  │ 光流幅值         │  │
│  │ 笑声/掌声/   │   │HF pipeline:  │  │                  │  │
│  │ 音乐检测     │   │ 情感分析     │  │insightface:      │  │
│  │              │   │              │  │ 人脸数量         │  │
│  └──────┬───────┘   └──────┬───────┘  └────────┬─────────┘  │
│         ↓                  ↓                   ↓            │
│  audio_features.json  text_features.json  visual_features.json│
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               融合打分阶段（scorer.py — 核心差异化）          │
│                                                             │
│  预处理: EWMA → Z-Score → 中值滤波 → 插值                    │
│  关系: 互相关(时滞补偿) + DTW(对齐)                           │
│  融合: 加权和 × 时间衰减 + tanh 突变奖励                      │
│  检测: 自适应阈值 + 多尺度(3s/10s/30s) + NMS                  │
│  边界: 语义对齐 + 高斯包络 + 局部DP                           │
│  质检: SSIM去重 + t-test + 评分追溯                           │
│                                                             │
│  产出: time_table.json                                      │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               执行阶段（ffmpeg 驱动）                         │
│                                                             │
│  clipper.py: 读取 time_table.json → ffmpeg 批量裁剪          │
│  ffmpeg: 字幕烧录 (subtitles=xxx.srt) + 缩略图提取            │
│  产出: highlights/*.mp4 + *.srt + *.jpg                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 自研 vs 复用

| 我们写（6 个模块） | 复用（8 个外部工具） |
|-------------------|---------------------|
| `scorer.py` — 融合打分（核心） | `ffmpeg` — 裁剪 + 字幕 + 缩略图 |
| `clipper.py` — ffmpeg 批量裁剪封装 | `PySceneDetect` (4.9k★) — 场景检测 |
| `signalAudio.py` — 音频特征提取 | `librosa` (7.5k★) — 音频分析 |
| `signalText.py` — 文本特征提取 | `jieba` (34k★) — 中文分词 |
| `asr.py` — 语音识别 + 说话人分离 | `FastText` (26k★) — 词向量模糊匹配 |
| `controller.py` — 并行流程编排 | `panns-inference` (1k★) — 音频事件 |
| | `insightface` (29k★) — 人脸检测 |
| | `WhisperX` (22k★) — 说话人分离 |

> **已评审否决**：lossless-cut（40k★ 但是 Electron GUI，CLI 不适合自动化）、videogrep（MP4 限定，FLV 需转换）

---

## 3. 工具选型与验证

### 3.1 已验证可行 ✅

| 工具 | 状态 | 验证方式 |
|------|------|---------|
| **ffmpeg** 视频裁剪 | ✅ | `ffmpeg -ss -t -c copy` 已验证 FLV→MP4 |
| **ffmpeg** 音频提取 | ✅ | `ffmpeg -vn -acodec copy` 已验证 FLV→AAC |
| **ffmpeg** 字幕烧录 | ✅ | `ffmpeg -vf subtitles=xxx.srt` |
| **librosa** RMS/MFCC/ZCR | ✅ | `pip install librosa` 即用 |
| **jieba** 中文分词 | ✅ | 34k★，行业标准 |
| **faster-whisper** ASR | ✅ | CPU tiny 模型 303s→20s，已集成 `asr.py` |
| **PySceneDetect** | ✅ | `pip install scenedetect[opencv]` |

### 3.2 已验证但有限制 ⚠️

| 工具 | 限制 | 缓解 |
|------|------|------|
| **panns-inference** | AudioSet 527 标签偏英文场景，"掌声"易误检 | 仅用 3 类：laughter/applause/music |
| **FastText** cc.zh.300.bin | 1GB 磁盘 | 提供轻量替代 cc.zh.100.bin (~350MB) |
| **WhisperX** diarization | 额外显存 ~500MB | 放在独立子进程，用完即卸载 |
| **auto-editor** | 可选预处理，会改变时间轴 | 默认关闭；若启用则输出 EDL 映射表 |

### 3.3 已否决 ❌

| 工具 | 否决原因 | 替代 |
|------|---------|------|
| **lossless-cut** | Electron GUI，CLI 不支持 JSON 驱动自动化 | ffmpeg `clipper.py` |
| **videogrep** | 原生不支持 FLV | ffmpeg 直接裁剪（我们不需要字幕搜索功能） |
| **mediapipe** | Google 已半废弃，API 不兼容 | insightface (29k★) |
| **paddleocr** (Phase 1) | 电商场景才需要 OCR，先不做 | 预留接口，Phase 2 再加 |

---

## 4. 信号提取层

### 4.1 signalAudio.py

**依赖**: `librosa` + `numpy` + `panns-inference`

**输入**: `audio.aac` (AAC 48kHz 立体声)

**输出**: `audio_features.json`，1Hz 采样，每字段长度 = 视频秒数

| 特征 | 计算方法 | 含义 | 计算频率 |
|------|---------|------|---------|
| `rms` | `librosa.feature.rms(y, frame_length=2048, hop_length=512)` | 音量能量 (dB) | 每 0.03s → 降采样至 1Hz |
| `spectralCentroid` | `librosa.feature.spectral_centroid(y=y, sr=sr)` | 频谱重心 (Hz)，高频=尖叫 | 同上 |
| `mfccDist` | `‖MFCC_t - MFCC_{t-1}‖₂` | 音色突变程度 | 同上 |
| `zcr` | `librosa.feature.zero_crossing_rate(y)` | 过零率，区分清音/浊音 | 同上 |
| `eventLaughter` | panns CNN14 → AudioSet class 32 概率 | 笑声（RMS 触发） | 事件驱动，仅 RMS > 阈值时 |
| `eventApplause` | panns CNN14 → AudioSet class 10 概率 | 掌声 | 事件驱动 |
| `eventMusic` | panns CNN14 → AudioSet class 137 概率 | 背景音乐 | 事件驱动 |

**panns 调用优化**：
```python
# 两步检测：先 RMS 快速扫描，仅峰值区域调 panns
if rms[t] > rms_mean + 2 * rms_std:  # 能量突增
    panns_result = at.inference(audio_chunk[t-0.5:t+1.0])
# 减少 80% 的 panns 调用量
```

**输出格式**：
```json
{
  "sampleRate": 1,
  "duration": 3600.0,
  "features": {
    "rms": [-32.5, -31.2, -28.7, ...],
    "spectralCentroid": [420.1, 385.3, 512.8, ...],
    "mfccDist": [0.12, 0.08, 0.45, ...],
    "zcr": [0.05, 0.04, 0.08, ...],
    "eventLaughter": [0.0, 0.0, 0.87, 0.92, 0.0, ...],
    "eventApplause": [0.0, 0.0, 0.0, 0.0, ...],
    "eventMusic": [0.3, 0.3, 0.3, 0.3, ...]
  }
}
```

### 4.2 signalText.py

**依赖**: `jieba` + `FastText` + `text2vec` + `transformers`

**输入**: `danmaku.jsonl` + `asr.json`

**输出**: `text_features.json`，1Hz 采样

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `dmDensity` | `count(messages in [t, t+1s]) × antiSpamPenalty` | 弹幕互动热度 |
| `dmAcceleration` | `d_{t+1} - 2d_t + d_{t-1}` | 弹幕爆发速度 |
| `dmEntropy` | `H = -Σ p(w)·log₂(p(w))` (10s 窗口，top-200，top-50词) | 话题多样性 |
| `dmSentiment` | HuggingFace `uer/roberta-base-finetuned-jd-binary-chinese` | 情感值 [-1,+1] |
| `dmUniqueUsers` | `count(unique userId in [t, t+1s])` | 真实活跃人数 |
| `dmUtr` | `unique tokens / total messages` (30s窗口) | 反刷屏：UTR<0.3触发惩罚 |
| `asrKeyword` | jieba 分词 + FastText 余弦相似度匹配 | 关键词命中 (0/1) |
| `topicChange` | text2vec embedding → KMeans label 变化 (每30s) | 话题切换 (0/1) |
| `speakerChange` | ASR speaker 标签切换 | 说话人交替 (0/1) |

**关键词词典**（可自定义 `config/keywords.json`）：
```json
{
  "high_energy": ["卧槽", "牛批", "666", "离谱", "绝了", "太强了", "无敌"],
  "funny": ["笑死", "哈哈哈哈", "蚌埠住了", "没绷住", "乐", "太搞笑了"],
  "controversy": ["无语", "下头", "恶心", "退钱", "翻车", "别买"],
  "interactive": ["抽我", "福袋", "左上角", "点关注", "加粉丝团"]
}
```

**FastText 模糊匹配**：
```python
# "我笑了" → cosine_sim > 0.7 → 匹配到 "笑死"、"哈哈哈哈" 等同义词簇
model = fasttext.load_model("cc.zh.300.bin")
sim = model.get_word_vector("笑死").dot(model.get_word_vector("爆笑"))
# > 0.75 → 视为同义
```

**反刷屏惩罚**：
```python
utr = len(set(tokens)) / total_messages
penalty = min(1.0, utr / 0.3)  # UTR<0.3 时惩罚
adjusted_density = raw_density * penalty
```

**输出格式**：
```json
{
  "sampleRate": 1,
  "duration": 3600.0,
  "features": {
    "dmDensity": [2.0, 3.0, 15.0, 12.5, ...],
    "dmEntropy": [2.1, 2.3, 3.8, 3.2, ...],
    "dmSentiment": [0.0, 0.1, 0.7, 0.5, ...],
    "dmUtr": [0.8, 0.7, 0.5, 0.6, ...],
    "asrKeyword": [0, 0, 1, 0, ...],
    "topicChange": [0, 0, 0, 0, ...],
    "speakerChange": [0, 0, 1, 0, ...]
  }
}
```

### 4.3 signalVisual.py

**依赖**: `PySceneDetect` + `OpenCV` + `insightface`

**输入**: `input.flv`

**输出**: `visual_features.json`，1Hz 采样

| 特征 | 计算方法 | 含义 | 计算频率 |
|------|---------|------|---------|
| `sceneChange` | PySceneDetect ContentDetector(threshold=27) | 场景切换 (0/1) | 全量 |
| `motion` | OpenCV Farneback 光流平均幅值 | 画面运动剧烈度 | 每 5 帧抽 1 帧 |
| `faceCount` | insightface FaceAnalysis | 画面中的人数 | **仅** score > 0.6 时触发 |

**执行策略**（控制显存）：
```
全量: sceneChange + motion（轻量，CPU/OpenCV）
按需: faceCount（重量，insightface GPU ~100MB，仅在 scorer 候选峰值区域激活）
Phase 2: OCR 文字（预留，电商直播场景）
```

---

## 5. 融合打分层（核心）

### 5.1 预处理管道

```
原始信号 (各模块 JSON)
    ↓
① 线性插值 → 统一采样率 1Hz
    ↓
② EWMA 平滑 (α=0.3) → 消除高频噪声
    ↓
③ Z-Score 标准化 (窗口 60s) → 消除量纲差异
    ↓
④ 中值滤波 (窗口 5) → 去除孤立尖峰
    ↓
标准化信号矩阵 z[t][i]  (t=秒, i=特征)
```

### 5.2 信号关系分析

| 分析 | 方法 | 产出 |
|------|------|------|
| 时滞补偿 | 互相关 `τ_max = argmax(R_{xy}(τ))` | 弹幕领先音频 2s → 所有信号时间戳校准 |

> Granger 因果检验和频域相干性已从设计中移除：前者计算量大性价比低，后者对 1Hz 采样无意义。

### 5.3 融合打分公式

```
Step 1 — 信号标准化:  z_i(t) = (x_i(t) - μ_i_60s) / σ_i_60s

Step 2 — 基础得分:    S_base(t) = Σ w_i · z_i(t)

Step 3 — 时间衰减:    decay(t, t_c) = exp(-|t - t_c| / 5)
                        (候选峰值 t_c 周围权重高，远处衰减)

Step 4 — 突变奖励:    boost_i(t) = tanh(|z_i(t) - z_i(t-1)| / 2)
                        (信号突变越大 bonus 越多)

Step 5 — 最终得分:    Score(t) = S_base(t) · decay(t) + Σ boost_i(t)
```

**默认权重**（可通过 `--profile` 切换）：

| 权重 | 默认 | game | shopping | talent |
|------|------|------|----------|--------|
| dmDensity | 0.15 | **0.20** | 0.10 | 0.10 |
| dmAcceleration | 0.05 | **0.10** | 0.03 | 0.05 |
| dmEntropy | 0.05 | 0.05 | 0.03 | 0.03 |
| dmSentiment | 0.10 | 0.05 | 0.05 | **0.15** |
| dmUtr | 0.03 | 0.03 | 0.03 | 0.03 |
| asrKeyword | 0.10 | 0.08 | **0.20** | 0.10 |
| topicChange | 0.05 | 0.05 | 0.03 | 0.03 |
| speakerChange | 0.05 | 0.03 | 0.05 | **0.10** |
| rms | 0.10 | 0.08 | 0.05 | **0.20** |
| mfccDist | 0.02 | 0.02 | 0.02 | 0.02 |
| eventLaughter | **0.10** | **0.10** | 0.05 | 0.05 |
| eventApplause | 0.05 | **0.10** | 0.05 | 0.05 |
| eventMusic | 0.05 | 0.03 | 0.05 | 0.05 |
| sceneChange | 0.05 | 0.05 | 0.05 | 0.03 |
| motion | 0.05 | **0.10** | 0.03 | 0.03 |
| faceCount | 0.00 | 0.00 | 0.00 | 0.00 |

> Phase 2 引入 LightGBM 直播类型分类器，根据前 5 分钟数据自动选择 profile。Phase 1 硬编码默认权重 + `--profile` CLI 参数。

### 5.4 峰值检测

```
① 自适应阈值: threshold(t) = μ_rolling(t) + k · σ_rolling(t)
   k 可通过 --sensitivity 调整 (默认 2.5)

② 多尺度检测:
   窗口 3s  → 短促爆发（笑声、拍桌）
   窗口 10s → 中等高潮（精彩连麦、翻车瞬间）
   窗口 30s → 整段精彩（大场面、结尾）

③ 持续性验证: Score(t) > threshold 需持续 ≥ 2s

④ 非极大值抑制 (NMS): 两峰值间距 < 15s → 合并取最高分
```

### 5.5 边界优化

```
① 语义对齐:
   峰值时刻 → 向前搜索 ASR segment 边界（句号/停顿）
            → 向后搜索 ASR segment 边界
   确保切片在句子边界开始/结束

② 高斯包络拟合:
   对 RMS 包络单高斯拟合 → μ ± 2σ 定界
   包含完整"爆发→衰减"过程

③ 局部动态规划:
   仅在峰值 ±30s 窗口内 DP 精调
   DP[L][R] = max( sum(scores[L:R]) - λ·|len - target| )
   target = 30s (抖音) / 60s (B站)
   复杂度: O(M²), M≈60 → 3600 次计算 (毫秒级)

④ 硬约束:
   片段时长 ∈ [15s, 90s]
```

### 5.6 质量保证

| 检查 | 方法 | 动作 |
|------|------|------|
| 重复检测 | 相邻片段 SSIM > 0.95 | 合并 |
| 有效性 | 片段前后 3s 得分 t-test (p<0.05) | 不通过则丢弃 |
| 可解释性 | 每切片输出 `score_breakdown` | 记录到 scores.json |
| 预览 | `--dry-run` 模式 | 仅输出时间表不裁剪 |

### 5.7 输出格式

**time_table.json** (scorer 产出)：

```json
{
  "roomId": "300294032039",
  "duration": 3600.0,
  "profile": "default",
  "sensitivity": 2.5,
  "clips": [
    {
      "id": 1,
      "start": 342.0,
      "end": 367.0,
      "duration": 25.0,
      "score": 0.87,
      "trigger": "dmSpike + asrKeyword(卧槽)",
      "title": "卧槽！主播竟然翻车了",
      "tags": ["搞笑", "翻车"],
      "thumbnailTime": 345.0,
      "breakdown": {
        "dmDensity": 0.30,
        "asrKeyword": 0.25,
        "rms": 0.20,
        "dmSentiment": 0.15,
        "eventLaughter": 0.10
      }
    }
  ]
}
```

---

## 6. 执行流程

### 6.1 10 步管线

```
Step 1  extractAudio.py      input.flv → audio.aac
Step 2  asr.py [+diarization] audio.aac → asr.json + subtitle.srt
                              (WhisperX 卸载 GPU 显存)
        ↓
Step 3  auto-editor (可选)    input.flv → cleaned.mp4 + edl.csv
                              (默认跳过，除非 --preprocess)
        ↓
Step 4  signalAudio.py ─┐
Step 5  signalText.py   ─┤ ProcessPoolExecutor(max_workers=3)
Step 6  signalVisual.py ─┘
        ↓
Step 7  scorer.py             *.json → time_table.json + scores.json
        ↓
Step 8  clipper.py (ffmpeg)   input.flv + time_table.json → highlights/*.mp4
Step 9  ffmpeg                字幕烧录: -vf subtitles=xxx.srt
Step 10 ffmpeg                缩略图: -ss thumbnailTime -vframes 1
```

### 6.2 controller.py 编排

```python
# 伪代码
async def runPipeline(videoPath, danmakuPath, outputDir, profile="default"):
    # Step 1-2
    audioPath = extractAudio(videoPath)
    asrPath, srtPath = runAsr(audioPath)  # 含 diarization
    
    # Step 3 (可选)
    if args.preprocess:
        cleanedPath, edlPath = runAutoEditor(videoPath)
        videoPath = cleanedPath
    
    # Step 4-6 并行
    with ProcessPoolExecutor(max_workers=3) as pool:
        f1 = pool.submit(runSignalAudio, audioPath)
        f2 = pool.submit(runSignalText, danmakuPath, asrPath)
        f3 = pool.submit(runSignalVisual, videoPath)
        audioFeatures = f1.result()
        textFeatures = f2.result()
        visualFeatures = f3.result()
    
    # Step 7
    timeTable = runScorer(audioFeatures, textFeatures, visualFeatures, profile)
    
    # Step 8-10
    clipVideo(videoPath, timeTable, outputDir)
    burnSubtitles(outputDir, srtPath)
    extractThumbnails(videoPath, timeTable, outputDir)
```

### 6.3 clipper.py（ffmpeg 封装，~30 行）

```python
import json, subprocess, os

def clipVideo(inputPath, timeTablePath, outputDir):
    """Read time_table.json, clip video with ffmpeg."""
    with open(timeTablePath) as f:
        data = json.load(f)
    
    os.makedirs(outputDir, exist_ok=True)
    
    for clip in data["clips"]:
        start, end = clip["start"], clip["end"]
        duration = end - start
        name = f"clip_{int(start)}s_{clip['trigger'].replace(' ', '_')}"
        outPath = os.path.join(outputDir, f"{name}.mp4")
        
        subprocess.run([
            "ffmpeg", "-ss", str(start), "-i", inputPath,
            "-t", str(duration), "-c", "copy", "-y", outPath
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"✓ {name}.mp4  ({duration:.0f}s)")
```

---

## 7. 工程约束

### 7.1 显存调度（4GB GTX 1650 Ti）

```
Phase A: 加载 faster-whisper large-v3 (INT8, ~3GB)
         → ASR 转录 + WhisperX diarization
         → del model; torch.cuda.empty_cache()
         → 显存释放

Phase B: 加载 panns (~200MB) + insightface (~100MB)
         → 并行特征提取
         → 文本模型 CPU 运行 (text2vec, FastText, jieba)
         → 总显存 < 600MB

Phase C: scorer (numpy/scipy, CPU only)
         → 0 显存

Phase D: ffmpeg (CPU)
         → 0 显存
```

### 7.2 时间对齐

```
统一基准: 视频 PTS (Presentation Timestamp)
弹幕校准: 弹幕服务器时间戳 → 线性回归 → PTS
ASR 对齐: -acodec copy 保留原始 PTS，天然对齐 ✓
验证方法: 找已知事件(如礼物动画)检查偏差 < 200ms
```

### 7.3 冷启动

```
预热期 (前 15 分钟):
  μ = 0.7 · μ_offline + 0.3 · μ_online
  σ² = 0.7 · σ²_offline + 0.3 · σ²_online

15 分钟后逐步过渡:
  μ = 0.1 · μ_offline + 0.9 · μ_online

离线统计来自预先计算的 100+ 场直播数据库
```

### 7.4 隐私合规

```
用户 ID: SHA256 哈希（JSONL 写入时即脱敏）
人脸数据: insightface 仅存储 468 个关键点坐标（浮点数），
          不存储原始人脸图像
ASR 文本: 仅保留高分片段的转录文本，不存完整对话
OCR: Phase 2 加入时仅提取商品名/价格，过滤手机号/姓名
```

### 7.5 显存泄漏防护

```python
# 每个大模型使用后
del model
torch.cuda.empty_cache()
import gc; gc.collect()

# WhisperX 使用独立子进程（最强隔离）
import multiprocessing as mp
p = mp.Process(target=runAsr, args=(audioPath,))
p.start(); p.join()  # 进程结束 → OS 回收所有显存
```

---

## 8. 实施计划

| Phase | 模块 | 工期 | 产出 | 依赖 |
|-------|------|------|------|------|
| **P1** | `signalAudio.py` | 1d | audio_features.json | librosa, panns |
| **P2** | `signalText.py` | 1d | text_features.json | jieba, FastText 模型下载 |
| **P3** | `scorer.py` | 2d | time_table.json | P1+P2 产出用于测试 |
| **P4** | `clipper.py` | 0.5d | ffmpeg 批量裁剪封装 | ffmpeg |
| **P5** | `asr.py` 升级 WhisperX | 0.5d | 说话人分离 | faster-whisper 已安装 |
| **P6** | `controller.py` 集成 | 1d | 并行编排 + CLI | P1-P5 |
| **P7** | 端到端验证 | 1d | 全管线测试 + 调参 | 完整录制数据 |

**总计: 7 天**

### 8.1 P1-P3 测试方式

每完成一个 Phase，用已有直播数据单独验证：
```bash
# P1 测试
python src/signalAudio.py --input data/300294032039/xxx.aac --output /tmp/audio.json

# P2 测试
python src/signalText.py --danmaku xxx.jsonl --asr xxx.json --output /tmp/text.json

# P3 测试
python src/scorer.py --audio /tmp/audio.json --text /tmp/text.json --output /tmp/time_table.json
```

### 8.2 P7 端到端测试

```bash
python -m src.controller 300294032039 --clip --profile default --sensitivity 2.5
# 预期产出: data/300294032039/highlights/ 下 3-15 个 mp4 文件
```

---

## 9. 安装清单

### 9.1 系统依赖

```bash
# ffmpeg (已安装)
winget install ffmpeg
```

### 9.2 Python 核心

```bash
pip install faster-whisper whisperx     # ASR + 说话人分离
pip install librosa panns-inference     # 音频分析
pip install jieba fasttext text2vec     # 中文 NLP
pip install transformers                # 情感分析
pip install scenedetect[opencv]         # 场景检测
pip install insightface                 # 人脸检测
```

### 9.3 模型下载（首次运行自动下载）

| 模型 | 大小 | 用途 |
|------|------|------|
| `faster-whisper large-v3` | ~3GB | ASR 转录 |
| `WhisperX diarization` | ~200MB | 说话人分离 |
| `panns CNN14` | ~100MB | 音频事件检测 |
| `FastText cc.zh.100.bin` | ~350MB | 词向量（轻量版） |
| `text2vec-base-chinese` | ~200MB | 语义聚类 |
| `insightface buffalo_l` | ~350MB | 人脸检测 |

> **国内用户**：模型托管在 HuggingFace，若下载失败，预先下载到 `~/.cache/` 或设置 `HF_MIRROR=https://hf-mirror.com`

---

## 10. 风险矩阵与评审修正

### 10.1 评审发现清单

| # | 发现 | 严重度 | 状态 | 修正 |
|---|------|--------|------|------|
| 1 | lossless-cut 是 GUI 不是 CLI 工具 | 🔴 | ✅ 已修正 | 换 ffmpeg `clipper.py` |
| 2 | 缺少说话人分离 | 🟡 | ✅ 已加入 | WhisperX diarization |
| 3 | Granger 因果检验性价比低 | 🟡 | ✅ 已删除 | 互相关足够 |
| 4 | 频域相干性对 1Hz 无意义 | 🟡 | ✅ 已删除 | — |
| 5 | panns 标签偏英文 | 🟡 | ✅ 已限制 | 仅用 3 类 |
| 6 | auto-editor 时间轴偏移 | 🟡 | ✅ 已处理 | 可选 + EDL 映射 |
| 7 | LightGBM 无训练数据 | 🟡 | ⏸ Phase 2 | Phase 1 用 `--profile` |
| 8 | FastText 1GB 太大 | 🟢 | ✅ 已提供 | cc.zh.100.bin 替代 |
| 9 | 显存泄漏风险 | 🟢 | ✅ 已防护 | del + empty_cache + 子进程 |
| 10 | 国内下载模型失败 | 🟢 | ✅ 已指引 | 手动下载 + 镜像 |

### 10.2 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| panns 中文场景误检 | 中 | 中 | 仅用 3 类，调低权重 |
| WhisperX diarization 显存超限 | 中 | 中 | 独立子进程 |
| 权重不适合特定直播类型 | 中 | 低 | `--profile` + `--sensitivity` |
| 模型下载失败（国内网） | 中 | 中 | 离线安装包指引 |
| 弹幕数据不足（低人气直播） | 低 | 中 | 冷启动策略 + 预热期 |
| 4K 视频处理太慢 | 低 | 低 | `--resolution 720p` 降分辨率 |

---

## 附录 A：CLI 参数设计

```
python -m src.controller <roomId> --clip [options]

  --profile      直播类型: default|game|shopping|talent (default: default)
  --sensitivity  峰值敏感度: 2.0-3.5 (default: 2.5, 越低越多切片)
  --min-duration 最小切片时长 (default: 15s)
  --max-duration 最大切片时长 (default: 90s)
  --preprocess   启用 auto-editor 预处理 (default: off)
  --resolution   输出分辨率: 720p|1080p|4k (default: 720p)
  --dry-run      仅输出时间表不裁剪
  --output-dir   输出目录 (default: data/{roomId}/highlights/)
```

## 附录 B：目录结构（最终）

```
douyin-living/
├── src/
│   ├── controller.py      ← 流程编排 + CLI
│   ├── scorer.py          ← 融合打分（核心）
│   ├── clipper.py         ← ffmpeg 批量裁剪
│   ├── signalAudio.py     ← 音频特征提取
│   ├── signalText.py      ← 文本特征提取
│   ├── signalVisual.py    ← 视觉特征提取
│   ├── asr.py             ← 语音识别 + 说话人分离（升级）
│   ├── danmakuWs.py       ← 弹幕 WebSocket
│   ├── flvRecorder.py     ← 视频录制
│   ├── roomApi.py         ← 房间 API
│   ├── signer.py          ← 签名生成
│   ├── auth.py            ← Cookie 认证
│   ├── params.py          ← 参数构造
│   ├── models.py          ← 数据模型
│   ├── util.py            ← 工具函数
│   └── protobuf/          ← Protobuf 定义
├── asr.py                 ← ASR 独立脚本
├── extractAudio.py        ← 音频提取独立脚本
├── logger.py              ← 日志系统
├── scripts/
│   ├── dyAb.js            ← a_bogus JS
│   └── dyLiveSign.js      ← 弹幕签名 JS
├── config/
│   └── keywords.json      ← 关键词词典（可自定义）
├── docs/
│   └── specs/
│       └── 2026-06-03-ai-clip-design.md  ← 本文档
└── tests/
    └── verify_modules.py
```
