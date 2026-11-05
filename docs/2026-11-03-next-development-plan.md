# 抖音直播 AI 切片系统 — 下一阶段开发计划

> 分析基线：`626eae0`（Sprint 2，2026-07-04）
> 文档日期：2026-11-03
> 适用周期：Sprint 3 ~ Sprint 6（约 9~11 周，单人全职估算）
> 文档状态：建议稿；第 12 节的产品/部署决策确认后可转为执行版

---

## 0. 结论摘要

1. **项目已经完成“从 0 到 1”**：录制、弹幕、音频提取、ASR、三路信号提取、融合打分、ffmpeg 裁剪、片段字幕/缩略图的主链路都已存在，Sprint 1/2 也完成了大量止血和工程化工作。
2. **当前最大瓶颈不是模型能力，而是“数据契约”和“评分链路”**：
   - 三路特征数组长度不一致时，scorer 会静默丢弃整类信号；
   - 候选片段先按秒级峰点生成，再只扩展 5~10 秒，最后被 15 秒最小时长过滤，容易“有候选但无成片”；
   - 配置的环境变量覆盖实际不生效，权重档案也没有做归一化校验；
   - 特征提取失败会写出空文件，但 pipeline_status 仍显示 `ok`。
3. **下一阶段应先做“可信管线”，而不是继续加模型**。建议 Sprint 3 用 2~3 周完成：
   - 环境可复现（CI / 完整依赖 / Docker / Makefile）；
   - 统一时间轴与特征长度；
   - 配置覆盖与权重校验；
   - 失败状态与产物哈希；
   - 候选窗口生成与时长约束修复。
4. **暂缓项**：LightGBM 自动分类、全量 OCR/人脸、Web UI、自动发布。没有标注集和评估指标之前，这些投入的收益不可验证。

**一句话：先把“切得对、切得出来、失败看得见、结果可复现”做扎实，再谈“切得更好”。**

---

## 1. 当前项目状态盘点

### 1.1 已具备能力

| 能力 | 当前状态 | 说明 |
|---|---|---|
| 直播录制 | ✅ 可用 | FLV 下载、文件轮转、URL 过期刷新 |
| 弹幕采集 | ✅ 可用 | WebSocket + Protobuf，支持聊天/礼物/点赞/进场/统计 |
| 音频提取 | ✅ 可用 | ffmpeg 无损提取 AAC |
| ASR | ⚠️ 部分可用 | faster-whisper + WhisperX，失败会降级但状态不准确 |
| 音频特征 | ✅ 基线可用 | RMS/ZCR/频谱质心/MFCC + panns 事件检测 |
| 文本特征 | ⚠️ 部分可用 | 密度/加速度/UTR/关键词/情感/熵/话题，但存在时间轴与性能问题 |
| 视觉特征 | ⚠️ 部分可用 | 场景切换 + 光流；人脸检测仍是占位 |
| 融合打分 | ⚠️ 部分可用 | 权重、EWMA、Z-Score、互相关、时间衰减已实现；DTW/SSIM/局部 DP 未实现 |
| 边界优化 | ❌ 较弱 | 仅 RMS 谷值检测，没有 ASR 句界/场景切点对齐 |
| 视频裁剪 | ✅ 可用 | ffmpeg 无损裁剪、分辨率转码、片段 SRT、缩略图 |
| 两阶段召回/精排 | ❌ 未实现 | 当前是全量特征 + 一次打分 |
| 评估闭环 | ❌ 未实现 | 没有标注集、指标脚本、权重搜索 |
| 工程化 | ⚠️ 部分 | 有 232 个测试、71% 覆盖率；无 CI、无 Docker、依赖声明不完整 |

### 1.2 工程指标（本次复核）

| 指标 | 数值 |
|---|---|
| Python 源码 | 约 4,790 行（不含 ASR/独立脚本） |
| 测试代码 | 约 2,706 行，232 个测试 |
| 测试通过 | 系统 Python 3.14 下 232 passed |
| 覆盖率 | 71% |
| 主要低覆盖模块 | `controller.py` 40%、`roomApi.py` 35%、`flvRecorder.py` 42%、`danmakuWs.py` 55%、`signalAudio.py` 60% |
| CI | 无 `.github/workflows` |
| 容器化 | 无 Dockerfile |
| 依赖锁定 | 无 lock/constraints 文件 |

### 1.3 本次分析确认的关键问题

#### P0-1 特征长度不一致，整类信号被静默丢弃

`src/scorer.py:71-94` 用所有特征的最大长度作为 `nSecs`，随后只对 `len(z) == nSecs` 的信号加权。长度不一致的信号不会报错、不会插值，也不会被记录，直接不参与打分。

当前样例数据已经复现：

```text
audio_features: 20s
visual_features: 20s
text_features:  17s
```

因此 `dmDensity / dmAcceleration / dmSentiment / asrKeyword / speakerChange` 等 9 个文本特征在打分时全部被忽略。文本信号是直播高光最重要的信号之一，这会让评分退化成“音频 + 画面”模型。

根因之一是 `src/signalText.py:133-137` 的时长估算：

```python
duration = max(
    asrSegments[-1].get("end", 0) if asrSegments else 0,
    len(danmakuEntries) // 10,  # rough estimate
)
```

该样例中 ASR 只到约 10 秒，171 条弹幕被估算成 17 秒，而弹幕实际 `recvTimeSec` 最大值约 27.9 秒，音频约 19.6 秒。文本时间轴被人为截短。

**影响**：评分结果不可解释、不可复现；文本信号形同虚设。

#### P0-2 候选 → 边界 → 时长过滤链路会“杀掉”候选

`src/scorer.py` 当前流程：

1. `_detectPeaks()` 返回的是秒级候选点的聚类范围，通常只有 0~3 秒；
2. `_optimizeBoundaries()` 只向前后扩展 `pad_sec=5`，谷值搜索范围 `10`；
3. `_qualityFilter()` 要求 `minDur=15`，不足 15 秒直接丢弃；超过 `maxDur=90` 也直接丢弃，而不是裁剪。

用 1 小时合成信号做快速实验：

```text
k=2.0：59 个候选 → 边界/NMS/时长过滤后仅 1 个
k=2.5：45 个候选 → 边界/NMS/时长过滤后 0 个
```

这解释了为什么真实样例 `data/887951935002/time_table.json` 是空数组，而 `highlights/` 里还留着上一轮生成的 `clip_2s_unknown.mp4`。

**影响**：用户看到“没有检测到片段”，但真正原因不是没有高光，而是候选没有扩展到可发布时长。

#### P0-3 配置环境变量覆盖失效，权重档案未校验

`src/config/settings.py:232-252` 把 YAML 数据作为构造参数传给 `BaseSettings`。pydantic-settings 的优先级中，初始化参数高于环境变量，因此文档宣称的：

```bash
export SCORER_DETECTION__WINDOWS='[5,15,45]'
export AUDIO_PANNS_RMS_THRESHOLD=1.5
```

实际不生效。本次验证：

```text
设置 SCORER_DETECTION__WINDOWS=[5,15,45]
读取结果仍是 [3, 10, 30]

设置 AUDIO_PANNS_RMS_THRESHOLD=9.9
读取结果仍是 2.0
```

另外：

- `config/scorer.yaml` 四套权重之和分别为：`default=1.00`、`game=1.07`、`shopping=0.84`、`talent=1.04`，没有校验或归一化；
- `src/scorer.py` 在导入时执行 `_cfg = load_settings().scorer`，`reload_settings()` 只清缓存，不会更新已经导入的模块常量，“热重载”实际不可用。

**影响**：运营调参、不同 profile 的灵敏度不可比，配置系统不可信。

#### P0-4 特征失败不可见，pipeline_status 会误报成功

- `src/signalAudio.py:41-56`、`src/signalText.py:56-70`、`src/signalVisual.py:31-41` 在异常时都会写出一份“空特征 JSON”；
- `src/controller.py:497-500` 的 `_stepStatus()` 只判断文件是否存在，因此空特征文件也会被标记为 `ok`；
- `src/controller.py:509` 的 ASR 状态同样只判断 `asrPath` 是否存在，ASR 失败写出的 `{"status":"failed","segments":[]}` 仍会被标记为 `ok`。

**影响**：失败被当成成功，后续 scorer 在缺少信号的情况下继续跑，最终“没有切片”无法定位原因。

#### P0-5 新环境无法按 README 复现

- `requirements.txt` 缺少 `aiofiles`、`pydantic`、`pydantic-settings`、`pyyaml`，而 `src/flvRecorder.py` 和 `src/config/settings.py` 在导入阶段就需要它们；
- `check_env.py` 也没有检查 `aiofiles/pydantic/yaml`；
- 当前工作区 `.venv` 只安装了 pytest/numpy/scipy/loguru，直接执行 `.venv/bin/python -m pytest` 会出现 9 个 collection error（`execjs`、`httpx`、`pydantic` 等缺失）；
- 系统 Python 3.14 下 232 个测试可以通过，但项目声明的是 Python >=3.11，没有 CI 固定版本。

**影响**：新开发者/新机器无法稳定复现，Sprint 2 的“71% 覆盖率”也无法在 CI 中守住。

#### P1-1 录制层仍有稳定性缺口

- `src/danmakuWs.py:114-123` 计算了指数退避 `delay`，但实际 sleep 用的是固定的 `self._reconnectDelay`；
- `src/danmakuWs.py:128-139` 每次重连的 `_onOpen` 都会启动一个新的 heartbeat 线程，旧线程不会随连接关闭而退出，长时间运行会累积线程；
- `src/flvRecorder.py:151-160` 在轮转时直接 `break` 出 `async for`，可能在 FLV tag 中间截断，后续 concat 可能产生损坏片段；
- `src/flvRecorder.py:92-94` 在正常 EOF 后把 `retries` 重置为 0，若服务端持续返回空响应，可能进入快速重连循环；
- 没有磁盘空间下限、单文件大小上限、空闲超时检测。

#### P1-2 视觉特征对长视频成本过高

`src/signalVisual.py` 对同一个视频完整解码两次：

- 第一次给 PySceneDetect；
- 第二次给 OpenCV 逐帧光流。

而且是原始分辨率解码。对于 2 小时 4K 直播，这会成为整个管线最慢的一步。设计文档里的“候选窗口再跑重模型”没有落地。

#### P1-3 文本/音频模型调用成本与语义问题

- `src/signalText.py:168-175` 对每条弹幕逐条调用 HF 情感模型；1 万条弹幕会产生 1 万次 transformer 推理；
- `src/signalText.py:258-260` 的 `asrKeyword` 是二值信号，没有区分关键词类别和强度；
- `dmUniqueUsers` 已经输出，但所有 profile 权重都是 0；
- `signalAudio` 的 panns 事件检测只在 RMS 峰值触发，短促事件可能漏检；
- `signalText` 的 UTR 阈值 0.3 偏低，正常聊天几乎不会被惩罚（旧审查已提示，仍未用真实数据校准）；
- 当前工作区没有 `models/` 目录，`config/text.yaml` 指向的 `models/cc.zh.100.bin` 不存在，FastText 模糊匹配实际已退化为精确匹配；text2vec/panns 依赖运行时下载。

#### P1-4 外部接口与裁剪精度仍有风险

- `src/roomApi.py:193-222` 依赖 SSR HTML 正则提取 FLV URL；页面结构变化会直接失效，已有的 `enterRoom()` JSON API 没有作为 fallback；
- `_get_client()` 没有并发保护，首次并发请求可能创建多个 httpx client；
- `src/clipper.py:69-73` 使用 `-ss` 在 `-i` 之前 + `-c copy`，会从输入最近的关键帧开始，片段起点可能比预期早几秒；
- `clipper.py` 的 ffmpeg 超时是硬编码 300/60 秒，没有使用 `config/pipeline.yaml` 中的配置。

#### P2-1 产品化与评估能力缺口

- `--preprocess` 参数存在但完全未使用（`src/controller.py:359` 注释“Phase 1 skipped”）；
- `--dry-run` 只输出 JSON，没有可视化报告；
- 没有片段级标题/标签/置信度/风险说明；
- 没有竖屏 9:16、平台预设、响度归一化；
- 没有标注集、Precision@K/Recall@K/Boundary IoU 等评估指标；
- 没有多房间批量录制/监听模式；
- README 写 Python 3.9+，但 `pyproject.toml` 与 `CLAUDE.md` 要求 3.11+，版本文档不一致。

### 1.4 当前“零切片”的根因链

以本次样例数据为例：

```text
弹幕最大时间 27.9s，音频 19.6s
        ↓
signalText 用 len(danmaku)//10 = 17s 估算时长
        ↓
text_features 只有 17s，audio/visual 是 20s
        ↓
scorer 只保留 len == 20s 的信号，9 个文本特征全部被丢弃
        ↓
剩余音频/视觉信号在默认 sensitivity=2.5 下没有形成稳定峰
        ↓
即使形成候选，候选点 ±5s 后仍不足 15s，被 qualityFilter 丢弃
        ↓
time_table.json = {"clips": []}
```

这不是某一个模型的问题，而是**数据契约 + 候选生成 + 状态反馈**三个环节共同造成的。Sprint 3 必须先把这条链路修好。

---

## 2. 下一阶段目标与原则

### 2.1 北极星目标

> 输入一场 1~2 小时的直播录制，稳定产出可人工审核、可发布的候选片段；每个片段都有可追溯的分数、触发原因和失败原因；在小型标注集上可以量化“比上一版更好”。

### 2.2 三个原则

1. **先可信，再智能**：没有可靠时间轴和状态反馈，增加模型只会增加不可解释性。
2. **先评估，再调参**：任何权重/阈值改动都必须能在固定标注集上回归，而不是靠单次观看感觉。
3. **先复用，再自研**：能通过 ffmpeg/PyAV/成熟库解决的长视频解码、切片、字幕问题，不自己写底层实现。

### 2.3 本阶段非目标

- ❌ 不引入 LightGBM/强化学习做自动 profile 识别（没有标注数据）。
- ❌ 不做全量 OCR/人脸检测（先做候选窗口精排，且仅在明确需要时）。
- ❌ 不做 Web UI / 多租户服务 / 自动发布（等核心链路稳定后再评估）。
- ❌ 不做大规模 controller 重写；采用“测试保护下的小步提取”。

---

## 3. 路线图总览

| Sprint | 版本主题 | 周期 | 核心目标 | 关键产出 |
|---|---|---|---|---|
| Sprint 3 | v0.2 可信管线 | 2~3 周 | 数据正确、环境可复现、失败可见 | 时间轴契约、特征对齐、配置修复、状态机、候选修复、CI |
| Sprint 4 | v0.3 稳定录制 | 1.5 周 | 长直播/弱网/页面变化下不丢数据 | WS 生命周期、FLV 轮转与空闲检测、API fallback、产物缓存 |
| Sprint 5 | v0.4 质量与评估 | 2~3 周 | 从“有切片”到“切片值得看” | 两阶段召回/精排、窗口级评分、ASR 边界、文本/视觉优化、评估脚本 |
| Sprint 6 | v0.5 创作者工作流 | 2 周 | 让结果可审核、可发布 | dry-run HTML 报告、片段元数据、竖屏/平台预设、批量模式、反馈闭环 |

> 单人全职估算约 9~11 周；如果只能业余投入，按 14~16 周排期。Sprint 3 完成前不建议启动 Sprint 5 的模型优化。

---

## 4. Sprint 3（P0）：把现有链路做“可信”

**目标**：同一份输入，所有信号都参与打分；任何失败都能在 `pipeline_status.json` 中看到；新机器按文档可以一次跑通。

### S3-1 环境与 CI 可复现

**改动**

- 修复 `requirements.txt`：至少补齐 `aiofiles`、`pydantic`、`pydantic-settings`、`pyyaml`；明确“录制核心依赖”和“AI 可选依赖”的边界。
- 新增 `requirements-dev.txt`：`pytest`、`pytest-asyncio`、`coverage`、`ruff`、`mypy`。
- 新增 `Makefile`：
  - `make setup`：创建 `.venv` 并安装依赖；
  - `make test`：运行测试；
  - `make cov`：覆盖率报告；
  - `make lint`：ruff；
  - `make e2e`：合成样例端到端；
  - `make clean`。
- 新增 `.github/workflows/ci.yml`：
  - Python 3.11 / 3.12；
  - 安装 `requirements.txt + requirements-dev.txt`；
  - 安装 ffmpeg、node；
  - `pytest --cov=src --cov-fail-under=75`；
  - `ruff check src tests`；
  - AI 依赖不进入每次 PR 的必跑任务，单独做定时 smoke job。
- 新增 CPU 版 `Dockerfile`（ffmpeg + node + Python 3.11），用于“无 GPU 也能录制 + 跑规则切片”。
- 新增 `scripts/download_models.py`（或 `make models`）：
  - 检查/下载 `models/cc.zh.100.bin`、text2vec、panns、faster-whisper；
  - 记录文件大小/校验和；
  - 支持 `HF_ENDPOINT` 镜像和离线目录。
- 修正 `check_env.py`：增加 `aiofiles/pydantic/yaml` 检查，并区分核心依赖与 AI 依赖；`--clip` 模式额外检查模型文件。

**验收**

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q          # 232+ passed
.venv/bin/python check_env.py          # 核心依赖全部通过
docker build -t douyin-live:cpu .      # 构建成功
```

**注意**：当前 `.venv` 是历史遗留环境，建议直接删除重建；不要把它当作依赖基准。

> **执行记录（2026-11-04）**
>
> - 已完成依赖拆分、`Makefile`、CI、`Dockerfile`、`.dockerignore`、模型下载脚本与 `check_env.py --clip`。
> - FastText 官方中文预训练模型只有 `cc.zh.300.bin`，配置与脚本已统一改为该文件（约 4.2 GB，需显式 `--fasttext` 才下载）。
> - 干净 Python 3.11 环境实测：`333 passed`；ruff 全绿；覆盖率 **76.6%**，CI 门禁已恢复为 **75%**。
> - 新增 `requirements-ci.txt`（仅额外安装 headless OpenCV + scenedetect），让视觉集成测试在 CI 中真正执行，同时不引入 torch/ASR 大依赖。
> - 新增 `tests/test_contracts.py`、`tests/test_roomApi_client.py`、`tests/test_e2e_scoring.py` 与 `tests/fixtures/synthetic.py`，并补齐 signalText 主题聚类/情感回退测试。
> - 新增 `scripts/baseline_report.py` 与 `docs/baseline-2026-11.md`：样例 20s 数据 0 切片（无峰值），合成 1 小时 60 峰召回 100%。
> - Dockerfile 已通过语法审查；当前机器未安装 buildx，未执行完整镜像构建。

### S3-2 统一时间轴与特征契约

**新增 `src/contracts.py`（或 `src/timeline.py`）**

定义三类结构：

1. `RecordingMeta`
   - `schemaVersion`
   - `roomId`
   - `startedAtWall` / `startedAtMonotonic`
   - `firstVideoDataAt`
   - `videoDurationSec` / `audioDurationSec`
   - `segments[]`、`sourceHashes{}`
   - `ffprobe` 得到的 fps/时长
2. `FeatureArtifactMeta`
   - `schemaVersion`
   - `sampleRate=1`
   - `duration`
   - `origin`（video/audio/danmaku/asr）
   - `sourceHash`
   - `generatorVersion`
3. `PipelineStatus`
   - `runId`、`startedAt`、`finishedAt`
   - `steps{}`（状态、耗时、错误、输入/输出、产物哈希）
   - `settingsHash`、`artifactHashes{}`

**改动**

- `controller` 录制开始时记录 wall clock / monotonic；第一段 FLV 真正写入时记录 `firstVideoDataAt`。
- 合并完成后用 `ffprobe` 探测视频/音频真实时长，写入 `recording_meta.json`（原子写：临时文件 + `os.replace`）。
- 所有 signal extractor 增加 `durationSec` 参数；`controller` 传入真实视频时长。
- `signalText`：
  - 优先使用 `durationSec`；
  - 回退到 `max(recvTimeSec)` / ASR 最大 end；
  - **删除 `len(danmaku) // 10` 的估算逻辑**；
  - 超出范围的弹幕记录计数并丢弃，不静默截断。
- `scorer` 增加 `alignFeatureArrays()`：
  - 短数组：连续型特征用边缘值/线性插值补齐，事件型特征补 0；
  - 长数组：截断；
  - 每次对齐都记录 `name / oldLen / newLen`；
  - **任何信号都不再因为长度不一致被静默丢弃**。

**验收**

- 用当前样例跑 scorer，19 个特征全部参与打分，日志中不再出现“文本特征被忽略”。
- 新增测试：17s 文本 + 20s 音频 + 20s 视觉 → 对齐后全部为 20s。
- `scores.json` 中包含每个信号的贡献/权重信息（至少 Top-N）。

### S3-3 修正配置系统

**改动**

- 修复环境变量覆盖：使用 pydantic-settings 的 `settings_customise_sources` + YAML source，或显式实现“YAML 默认值 → 环境变量覆盖”的合并；不要用“YAML 作为构造参数”的方式。
- 增加配置校验：
  - profile 权重之和必须为 `1.0 ± 0.01`（或显式 `normalize_profiles: true`）；
  - 权重 key 必须属于 `FEATURE_NAMES`；
  - 权重必须非负；
  - `min_duration < max_duration`；
  - 窗口、阈值、超时必须是合法范围。
- 修正 `config/scorer.yaml` 的权重：`game=1.07`、`shopping=0.84`、`talent=1.04` 需要归一化或补齐到 1.0。
- 移除模块级 `_cfg = load_settings()` 的“假热重载”：
  - 改成 `get_config()` 访问器，或在 `reload_settings()` 后显式 `rebind` 各模块；
  - 至少要在文档中明确“修改 YAML 需要重启进程”。
- 新增配置测试：环境变量覆盖、非法权重、未知特征 key、短/长时长边界。

**验收**

```bash
SCORER_DETECTION__WINDOWS='[5,15,45]' \
AUDIO_PANNS_RMS_THRESHOLD=9.9 \
python -c "from src.config import load_settings; s=load_settings(); print(s.scorer.detection.windows, s.audio.panns_rms_threshold)"
# 期望：[5, 15, 45] 9.9
```

### S3-4 状态机与失败语义

**改动**

- 定义步骤状态枚举：`pending / running / ok / empty / failed / skipped / invalid`。
- 每个 extractor 返回 `{status, duration, featureCount, error, elapsed}`，不再只写空文件。
- `controller` 读取特征文件时校验：
  - JSON 可解析；
  - `features` 是 dict；
  - 至少一个非空数组；
  - `duration > 0`。
- ASR 失败明确写 `status=failed`，`pipeline_status.steps.asr.status=failed`，并在报告中给出原因。
- `pipeline_status.json` 增加：
  - 每步耗时；
  - 输入/输出路径；
  - 产物 sha256；
  - 失败摘要；
  - 运行环境（Python/ffmpeg/模型名）。
- 产物缓存增加 source hash：只有源文件 hash 一致时才跳过；否则重跑。

**验收**

- 模拟 ASR 抛异常 → `pipeline_status` 显示 `failed`，不再显示 `ok`。
- 模拟空特征文件 → 显示 `empty`，scorer 不把它当成有效信号。
- 修改源视频后重新运行 → 不再复用旧特征。

### S3-5 修复候选生成与时长约束

**改动**

- `_detectPeaks()` 之后增加“候选窗口扩展”：
  - 以峰值为中心，按 profile/平台目标时长扩展到 `[minDuration, maxDuration]`；
  - 不足最小时长时优先向前后扩展，而不是直接丢弃；
  - 超过最大时长时裁剪到最大时长，而不是丢弃；
  - 记录 `dropReasons`（`too_short`、`too_long`、`low_confidence`、`duplicate`、`insufficient_context`）。
- `_qualityFilter()` 改为“修剪 + 过滤”：
  - 超长 → 保留高分窗口并裁剪；
  - 过短 → 能扩展则扩展，不能扩展才丢弃；
  - t-test 失败不直接丢弃，降低置信度并记录原因。
- `_nms()` 合并后如果超过最大时长，按分数切分或保留最高分子窗口。
- 在 `time_table.json` 中输出候选的 `dropReasons` 和 `candidateSource`。

**验收**

- 1 小时合成信号（60 个稀疏峰）在默认参数下产出 ≥ 10 个候选，而不是 0~1 个。
- 20 秒短样例不再因为“候选只有 2 秒”而必然为空；至少能输出 1 个带上下文的候选或被明确标记为 `too_short`。
- 新增测试覆盖：短候选扩展、长候选裁剪、NMS 后超长切分。

### S3-6 回归测试与基线报告

**改动**

- 新增 `tests/fixtures/`：用 ffmpeg 生成 60~120 秒合成视频/音频/弹幕，作为端到端固定输入。
- 新增 `tests/test_timeline_alignment.py`、`tests/test_pipeline_status.py`、`tests/test_candidate_windows.py`。
- 新增 `scripts/baseline_report.py`：对固定 fixtures 输出切片数量、耗时、各步骤状态、覆盖率。
- 把当前样例 `data/887951935002` 固化为回归用例（脱敏后只保留特征 JSON/时间表，不提交大视频）。

**验收**

- `pytest` 全绿；整体覆盖率 ≥ 75%，`scorer.py` ≥ 90%，`signalText.py` ≥ 80%。
- 端到端 fixture 在 `--dry-run` 下产出非空 `time_table.json`。
- CI 中可以看到每步耗时和失败原因。

**Sprint 3 完成定义**

- 新环境一条命令安装 + 测试通过；
- 所有特征长度一致，文本信号参与打分；
- 配置环境变量生效，权重和特征 key 有校验；
- 失败/空结果在状态文件中可见；
- 固定 fixtures 能稳定产出候选；
- 以上均有测试覆盖。

---

## 5. Sprint 4（P1）：录制与外部依赖韧性

**目标**：网络抖动、URL 过期、页面结构变化、长时间直播都不丢数据。

### S4-1 弹幕 WS 生命周期

- 每个连接独立的 `stopEvent`，`on_close` 时停止 heartbeat 线程；
- 真正使用指数退避（当前计算了 `delay` 但 sleep 固定值）；
- 增加连接超时、ping/pong 超时、重连次数和最后错误指标；
- `_parseMessage` 对未知消息记录类型计数，便于发现协议变化；
- 增加 mock 测试：连续 3 次重连后线程数不增长。

> **执行记录（2026-11-05）**
>
> - `DanmakuWs` 改为每个连接独立心跳线程与 `_hbStop` 事件；`on_close` / 重连前 / `stop()` 都会停止并回收心跳线程。
> - 重连退避改为真实指数退避，并使用 `stopEvent.wait(delay)`，`stop()` 可在 1 秒内打断最长 30 秒的等待。
> - 新增 `ws_ping_timeout` 配置（默认 10s），心跳检测到连续无数据时主动关闭连接并重连。
> - `stats` 暴露 `connected / connectAttempts / reconnects / consecutiveErrors / lastError / lastErrorAt / connectedAt / lastMessageAt / unknownMethods`。
> - `_parseMessage` 对未知 method 计数，`_onMessage` 记录最近活动时间。
> - 新增 `tests/test_danmakuWs_lifecycle.py`：连续 4 次连接后无心跳线程泄漏、长退避可中断、指数退避上限、错误指标与未知消息计数。
> - S4-1 完成时点实测：`343 passed`，ruff 全绿，覆盖率 **78.1%**。

### S4-2 FLV 录制韧性

- 轮转按 FLV tag 边界切分，或使用 ffmpeg segment muxer；写入 `.part` 后原子重命名；
- 每段完成后用 `ffprobe` 校验时长/流信息；损坏段隔离到 `corrupt/`，不参与 concat；
- 增加空闲检测：连续 N 秒无字节 → 重连；正常 EOF → 判断直播是否结束；
- 增加磁盘空间下限（如 <2GB 停止并告警）、单文件大小上限、过期文件清理策略；
- 合并失败时回退到“保留可用段 + 重新编码拼接”，并记录原因。

> **执行记录（2026-11-05）**
>
> - 每个 FLV 段先写 `*.flv.part`，完成后经 ffprobe 校验再 `os.replace()` 原子发布；启动时清理超过 24 小时的遗留 `.part`。
> - 校验失败的段移动到 `corrupt/`，不计入 `downloadedFiles`，也不会进入 concat 列表。
> - `_downloadSegment` 支持空闲检测（`idle_timeout_secs`，无字节超时→重连）、单段大小上限（`max_segment_bytes`）和下载中磁盘巡检；低于 `min_free_disk_gb` 时停止并记录 `stopReason=low_disk`。
> - 正常 EOF 会按 `max_retries` 重连；重连耗尽时标记 `stream_ended`，controller 不再无限等待，而是结束录制并进入合并。
> - controller 合并前再次用 ffprobe 过滤/隔离损坏段；流复制 concat 失败时自动回退到 `libx264/aac` 重编码，并记录实际模式。
> - 新增 `tests/test_flvRecorder_lifecycle.py`、`tests/test_controller_merge.py`，覆盖原子发布、隔离、空闲/磁盘/大小边界、EOF 重试耗尽、合并回退。
> - 干净 Python 3.11 环境实测：`371 passed`，ruff 全绿，覆盖率 **79.3%**。

### S4-3 Room API 适配层

- 优先解析页面 JSON（`RENDER_DATA` / `__INITIAL_STATE__` / SSR JSON），正则作为 fallback；
- `getStreamUrls()` 失败时自动调用 `enterRoom()` API；
- `httpx.AsyncClient` 加锁或使用生命周期管理，避免并发创建/泄漏；
- 增加 429/5xx 重试与抖动；Cookie 失效时给出明确提示；
- 把“页面结构解析”封装为 `src/douyin/room_parser.py`，便于未来协议变化时单点修改。

> **执行记录（2026-11-05）**
>
> - 新增 `src/douyin/room_parser.py`：优先解析 `RENDER_DATA`（URL 编码 JSON）、`__INITIAL_STATE__`/`__NUXT__`/`__NEXT_DATA__`、nonce script 中的 JSON；解析失败时回退到原正则，房间信息与 FLV URL 均适用。
> - `getStreamUrls()` 在页面 fetch 失败或 SSR 中无 URL 时自动调用 `enterRoom()`，并从 `flv_pull_url`（`FULL_HD1/HD1/SD1/SD2`）映射到 `or4/hd/sd/ld`。
> - `httpx.AsyncClient` 创建改为双重检查锁；新增统一 `_request()`：429/5xx 指数退避 + 抖动（最多 3 次），传输错误重试，401/403 明确提示刷新 Cookie。
> - 新增 `tests/test_room_parser.py`，并扩展 `tests/test_roomApi_client.py` 覆盖 RENDER_DATA、`__INITIAL_STATE__`、转义 URL、enterRoom 回退、429/5xx 重试与耗尽。
> - 干净 Python 3.11 环境实测：`389 passed`，ruff 全绿，覆盖率 **80.2%**。

### S4-4 产物缓存与可恢复执行

- 每个中间产物写 `*.meta.json`：源 hash、配置 hash、生成时间、耗时、版本；
- 重跑时只在 hash 一致时跳过；`--force` 强制重建；
- pipeline 支持从任意步骤恢复（audio/asr/signal/score/clip）；
- 失败步骤可单独重试，不必从录制开始。

### S4-5 可观测性与运行手册

- 结构化日志字段：`runId`、`roomId`、`step`、`elapsedMs`、`status`；
- `logs/` 增加按天/大小轮转和保留策略；
- 新增 `docs/runbook.md`：Cookie 过期、无流、ffmpeg 缺失、模型下载失败、磁盘满、ASR OOM 的处理步骤；
- `check_env.py` 增加 `--clip` 模式，检查 AI 依赖和模型文件。

**Sprint 4 完成定义**

- 模拟 403/404、空响应、10 秒断网、页面结构变化，均有可恢复路径；
- 连续重连 3 次后没有 heartbeat 线程泄漏；
- 损坏 FLV 段不会污染最终视频；
- 重跑只重建必要步骤。

---

## 6. Sprint 5（P1/P2）：从“有切片”到“切片值得看”

**目标**：建立“低成本全量召回 + 候选窗口精排”的两阶段架构，并用标注集量化效果。

### S5-1 两阶段架构

**阶段 A：全量低成本召回**

- 弹幕：密度、加速度、unique users、重复率、问号/感叹号比例；
- 音频：RMS、MFCC 距离、静音/停顿；
- 视觉：场景切点、低分辨率帧差/运动；
- ASR：精确关键词/短语匹配。

输出每小时 30~80 个候选窗口，记录来源和分数。

**阶段 B：候选窗口精排**

- panns 笑声/掌声/音乐；
- HF 情感（批量，而不是逐条弹幕）；
- text2vec 话题聚类；
- FastText 模糊关键词；
- 可选 face/OCR（只在明确需要时）；
- ASR 句界/说话人切换。

**代码建议**

```text
src/pipeline/
  stage1_recall.py
  stage2_rerank.py
  window_features.py
  status.py
```

不要一次性重写 controller；先让新模块以函数形式接入，旧路径保留一个 sprint。

### S5-2 窗口级评分与边界优化

- 从“秒级点分数”升级为“窗口级分数”：
  - `peak_score`、`mean_score`、`area_under_curve`；
  - `pre_post_delta`（片段前后对比）；
  - `multi_signal_agreement`（多模态共振）；
  - `novelty`（与前后窗口的差异）；
  - `duration_fit`（目标时长匹配度）。
- 边界优化融合：
  - ASR 句子开始/结束；
  - 静音/停顿；
  - 场景切换点；
  - 分数曲线谷值；
  - 目标平台时长偏好。
- 目标时长按内容类型动态化（初始建议）：
  - 搞笑/翻车：12~35s；
  - 商品爆点：20~60s；
  - 连麦/争议：30~90s；
  - 才艺/音乐：45~120s。
- 输出 `confidence`、`reason[]`、`risks[]`、`dropReasons[]`。

### S5-3 文本特征升级

- 增加 `dmUniqueUsers` 权重（真实热度比纯消息量更可靠）；
- 新增 `dmRepeatRate`、`dmQuestionRate`、`dmExclamationRate`、`dmEmojiRate`；
- 关键词按类别输出强度：`high_energy / funny / controversial / interactive / emotional / shopping`；
- 建立直播话术短语词典（“来了”“不是吧”“真的假的”“扣1”“上车”“库存”“最后一批”等）；
- 情感分析改为批量推理或“规则优先 + 候选窗口 HF”；
- UTR 阈值用真实数据分布校准，不继续拍脑袋。

### S5-4 视觉性能优化

- 用 ffmpeg 生成低分辨率代理流（如 `fps=2, scale=480:-2`），场景/运动只在代理上计算；
- 或者用 PyAV 单次解码，同时喂给场景检测和光流，避免解码两遍；
- 人脸/OCR 只对候选窗口的关键帧运行；
- 增加 `visual.proxy_height`、`visual.sample_fps`、`visual.enable_face` 配置；
- 在基准脚本中记录“1 小时 1080p 视频的视觉耗时”，作为回归指标。

### S5-5 ASR 配置与质量

- 新增 `config/asr.yaml`：模型大小、设备、compute type、语言、是否 diarize、batch size、VAD 参数；
- 默认值按硬件自适应：4GB GPU / CPU 默认 `base` 或 `small`，不默认 `large-v3`；
- 长音频分块 + 断点续跑；
- 明确“ASR 失败 → 文本特征降级 → pipeline 状态”的链路；
- 固定 faster-whisper/whisperx 版本，避免 API 漂移。

### S5-6 评估体系与标注集

**标注集**

- 选 5~10 场真实直播，覆盖 default/game/shopping/talent；
- 每场 30~120 分钟；
- 标注“值得切”的窗口、类型、1~5 分质量、边界；
- 同时标注“系统容易误判但不应切”的负样本。

**指标**

| 指标 | 定义 | 初始目标（需基线校准） |
|---|---|---|
| Precision@K | 前 K 个片段中人工认可比例 | ≥ 0.6 |
| Recall@K | 人工标注高光被召回比例 | ≥ 0.7 |
| Boundary IoU | 系统片段与人工片段重叠度 | ≥ 0.5 |
| Invalid Rate | 无意义片段比例 | ≤ 20% |
| Duplicate Rate | 重复/近似片段比例 | ≤ 10% |
| Editing Cost | 人工平均需调整的秒数 | 逐版下降 |

**工具**

- `scripts/evaluate.py`：输入 `time_table.json` + 标注 JSON，输出上述指标；
- `scripts/tune_weights.py`：在固定特征上做网格/随机搜索，不重新跑模型；
- `scripts/label_report.py`：生成待标注片段的缩略图和上下文，降低标注成本。

**注意**：上表目标是建议值，第一次运行后必须以真实基线为准，不要在没有基线时对外承诺。

### S5-7 Profile 与权重

- 先做权重归一化和手工网格搜索；
- 用标注集验证 `default/game/shopping/talent` 的差异是否真的有效；
- 若某 profile 没有足够样本，暂时合并，不要制造“看似有配置、实际没数据”的假精度；
- 自动 profile 识别（LightGBM/规则）放到 Sprint 6 之后，且必须有标注支撑。

**Sprint 5 完成定义**

- 两阶段架构在固定标注集上产出可复现指标；
- 视觉解码次数从 2 次降到 1 次，长视频耗时有基准对比；
- 片段边界包含 ASR 句界/静音/场景信息；
- 文本特征不再逐条弹幕跑 HF；
- 权重调优有脚本、有记录、有回归。

---

## 7. Sprint 6（P2）：创作者工作流与反馈闭环

**目标**：让运营/剪辑人员能快速审核、修改、导出，不必翻 JSON 和日志。

### S6-1 Dry-run 可视化报告

新增 `--report-html <path>`：

- 分数曲线（可用内嵌 SVG/Canvas，避免外部 CDN）；
- 候选片段列表：起止时间、时长、分数、触发原因、风险、丢弃原因；
- 每个候选的缩略帧；
- 弹幕 Top 词、ASR 摘要；
- 信号贡献 Top-N；
- 一键复制 ffmpeg 命令/时间码。

### S6-2 片段元数据与资产

每个片段输出 `clip_xxx.json`：

```json
{
  "clipId": 1,
  "start": 342.0,
  "end": 367.0,
  "duration": 25.0,
  "confidence": 0.82,
  "title": "弹幕爆发 + 关键词命中",
  "tags": ["互动", "搞笑"],
  "reasons": ["dmDensity", "asrKeyword", "eventLaughter"],
  "risks": ["ASR缺失", "边界可能偏短"],
  "suggestedAction": "keep",
  "scoreBreakdown": {"dmDensity": 0.31, "asrKeyword": 0.22}
}
```

同时保留：`mp4`、片段级 `srt`、`jpg` 封面。

### S6-3 竖屏与平台预设

新增 `--target-platform douyin|bilibili|xiaohongshu`：

| 平台 | 画幅 | 建议时长 | 默认处理 |
|---|---|---|---|
| douyin | 9:16 | 15~45s | 竖屏裁切/模糊背景、字幕烧录、响度归一化 |
| bilibili | 16:9 | 60~180s | 保留横屏、可选字幕 |
| xiaohongshu | 9:16 | 15~60s | 竖屏、封面优先 |

ffmpeg 竖屏实现建议：主体居中裁剪 + 模糊背景填充，避免简单拉伸；响度归一化使用 `loudnorm`。

### S6-4 批量与监听模式

- `--rooms-file rooms.txt`：批量处理多个房间；
- `--watch`：轮询直播间状态，开播自动录制；
- 每个房间独立目录、独立状态文件、独立锁；
- 失败重试和断点续跑；
- 不做多进程 GPU 抢占；重模型任务串行化。

### S6-5 人工反馈闭环

- `feedback.jsonl`：记录 keep/drop/edit 和调整后的边界；
- `scripts/apply_feedback.py`：把人工修改转成评估样本；
- 定期回顾：哪些信号经常误判、哪些类型召回不足；
- 反馈数据积累到一定量后，再考虑排序模型。

### S6-6 文档与发布

- 更新 `README.md`：真实安装步骤、CPU/GPU 两条路径、常见问题；
- 更新 `CLAUDE.md`：当前架构、配置、测试命令、已知限制；
- 新增 `docs/artifacts-schema.md`：特征/时间表/状态/片段元数据 schema；
- 新增 `CHANGELOG.md`，版本号从 `0.1.0` 起按 Sprint 递增；
- 在 `pyproject.toml` 中补 `ruff`/`mypy`/coverage 配置。

---

## 8. 优先级矩阵

| ID | 事项 | 优先级 | 影响 | 工作量 | 依赖 |
|---|---|---|---|---|---|
| P0-1 | 时间轴/特征长度对齐 | P0 | 极高 | 2~3 天 | 无 |
| P0-2 | 候选窗口扩展与时长约束 | P0 | 极高 | 2 天 | P0-1 |
| P0-3 | 配置环境变量/权重校验 | P0 | 高 | 1~2 天 | 无 |
| P0-4 | 状态机/失败语义 | P0 | 高 | 2 天 | 无 |
| P0-5 | 依赖/CI/Docker 可复现 | P0 | 高 | 1~2 天 | 无 |
| P1-1 | WS 生命周期/退避 | P1 | 高 | 1~2 天 | 无 |
| P1-2 | FLV 轮转/空闲/损坏段 | P1 | 高 | 2~3 天 | 无 |
| P1-3 | Room API fallback | P1 | 中高 | 1~2 天 | 无 |
| P1-4 | 视觉单次解码/代理流 | P1 | 高 | 3~5 天 | P0-1 |
| P1-5 | 文本特征升级/批量情感 | P1 | 高 | 3~5 天 | P0-1 |
| P1-6 | 两阶段召回/精排 | P1 | 极高 | 1~2 周 | P0-2、P1-4、P1-5 |
| P1-7 | 标注集/评估脚本 | P1 | 极高 | 1 周 | 需要真实数据 |
| P2-1 | Dry-run HTML 报告 | P2 | 中高 | 2~3 天 | P0-4 |
| P2-2 | 片段元数据/竖屏/平台预设 | P2 | 中高 | 3~5 天 | P1-6 |
| P2-3 | 批量/监听模式 | P2 | 中 | 3~5 天 | P1-1、P1-2 |
| P2-4 | Controller 拆分重构 | P2 | 中 | 3~5 天 | 测试覆盖 ≥80% |
| P3 | 自动 profile/排序模型 | P3 | 待验证 | 1~2 周 | P1-7 数据量足够 |

---

## 9. 测试与质量门禁

### 9.1 测试金字塔

| 层级 | 范围 | 目标 |
|---|---|---|
| 单元测试 | 纯函数、配置、对齐、候选窗口、状态机 | 快速、无外部依赖 |
| 集成测试 | controller + mock 外部依赖、recording session、pipeline runner | 覆盖失败/重试/降级 |
| E2E | 合成 60~120s 视频/音频/弹幕 | `--clip-only --dry-run` 产出非空时间表 |
| 回归测试 | 真实样例特征/时间表 golden | 防止评分行为漂移 |
| 性能基准 | 1h 1080p 的每步耗时/内存 | 记录基线，设回归预算 |

### 9.2 CI 门禁

```yaml
# 目标配置
- Python 3.11 / 3.12
- pip install -r requirements.txt -r requirements-dev.txt
- apt install ffmpeg; npm install
- ruff check src tests
- pytest --cov=src --cov-report=term-missing --cov-fail-under=75
- 可选：mypy src/contracts.py src/scorer.py
```

### 9.3 覆盖率目标

| 模块 | 当前 | Sprint 3 目标 | Sprint 5 目标 |
|---|---|---|---|
| 总体 | 71% | 75% | 80% |
| `scorer.py` | 89% | 90% | 92% |
| `signalText.py` | 72% | 80% | 85% |
| `controller.py` | 40% | 55% | 70% |
| `roomApi.py` | 35% | 50% | 70% |
| `flvRecorder.py` | 42% | 60% | 75% |
| `danmakuWs.py` | 55% | 65% | 75% |
| `signalAudio.py` | 60% | 70% | 80% |

> 覆盖率不是目的，但当前 controller/roomApi/flvRecorder 是“最容易改坏”的部分，必须补测试后再重构。

### 9.4 性能基线

新增 `scripts/benchmark.py`，记录：

- 录制 1 小时的文件大小、平均码率、分段数；
- 音频提取耗时；
- ASR 耗时（base/small/large，CPU/GPU）；
- 三路信号耗时、峰值 RSS/VRAM；
- 打分耗时；
- 裁剪耗时；
- 端到端总耗时。

第一版只记录，不设硬指标；Sprint 5 后再设回归预算（例如“视觉步骤不得超过基线的 120%”）。

---

## 10. 评估与数据闭环

### 10.1 标注文件格式（建议）

```json
{
  "sessionId": "room_2026xxxx",
  "roomId": "887951935002",
  "duration": 5400,
  "profile": "game",
  "clips": [
    {
      "start": 342.0,
      "end": 367.0,
      "type": "funny",
      "quality": 4,
      "boundary": "good",
      "notes": "笑点完整，开头有铺垫"
    }
  ],
  "negatives": [
    {"start": 800.0, "end": 830.0, "reason": "只有背景音乐，没有互动"}
  ]
}
```

### 10.2 评估流程

1. 选 5~10 场直播，先跑当前版本，导出候选和分数；
2. 人工标注“值得切/不值得切/边界是否合理”；
3. 运行 `scripts/evaluate.py` 得到基线；
4. 每次调参/改模型后重跑同一批特征，比较指标；
5. 指标下降的改动不得合并（除非有明确取舍并记录）。

### 10.3 记录“被丢弃原因”

每个候选即使没切成片，也要在 `time_table.json` 中保留：

- `score_below_threshold`
- `too_short`
- `too_long`
- `duplicate`
- `low_confidence`
- `t_test_insignificant`
- `missing_context`
- `missing_signal:<name>`

这比只看最终 mp4 更能定位“漏召回”。

---

## 11. 第一周执行清单

> 目标：一周内让“环境 + 时间轴 + 状态 + 候选”四个 P0 问题有可验证的闭环。

### Day 1：环境与基线

- [ ] 删除/重建 `.venv`，使用 Python 3.11；
- [ ] 修复 `requirements.txt`（补 `aiofiles/pydantic/pydantic-settings/pyyaml`）；
- [ ] 新增 `requirements-dev.txt`、`Makefile`；
- [ ] 跑通 `pytest` 并保存覆盖率基线；
- [ ] 记录当前样例 `--clip-only --dry-run` 的输出。

### Day 2：CI 与 Docker

- [ ] 新增 `.github/workflows/ci.yml`；
- [ ] 新增 CPU Dockerfile；
- [ ] `ruff` 配置并修复阻断项；
- [ ] `check_env.py` 增加核心依赖检查。

### Day 3：时间轴契约

- [ ] 新增 `src/contracts.py`；
- [ ] controller 写 `recording_meta.json`；
- [ ] `signalText` 接收 `durationSec`，删除 `len(danmaku)//10`；
- [ ] scorer 增加 `alignFeatureArrays()`；
- [ ] 新增对齐测试。

### Day 4：配置修复

- [ ] 修复 pydantic-settings 环境变量覆盖；
- [ ] 增加 profile 权重校验/归一化；
- [ ] 修正 `config/scorer.yaml` 的三套非 1.0 权重；
- [ ] 新增配置测试。

### Day 5：状态与失败语义

- [ ] 定义步骤状态枚举；
- [ ] extractor 返回状态/统计；
- [ ] `pipeline_status.json` 增加耗时、哈希、错误；
- [ ] ASR 失败状态修正；
- [ ] 新增空特征/失败特征测试。

### Day 6：候选窗口修复

- [ ] 候选扩展/裁剪逻辑；
- [ ] `_qualityFilter` 改为修剪优先；
- [ ] `time_table.json` 增加 `dropReasons`；
- [ ] 合成 1 小时信号验证候选数量。

### Day 7：端到端与复盘

- [ ] 生成 60~120 秒合成 fixture；
- [ ] `--clip-only --dry-run` 产出非空时间表；
- [ ] 写 `docs/baseline-2026-11.md`：测试、覆盖率、候选数、耗时、已知问题；
- [ ] 更新 `README.md` / `CLAUDE.md` 的安装与测试命令。

---

## 12. 待确认的产品/部署决策

以下问题会影响 Sprint 5/6 的优先级，建议在 Sprint 3 开始前确认：

1. **部署形态**：单机 CLI 工具，还是多房间常驻服务？
   - CLI：优先做稳定录制 + 批量脚本；
   - 服务：需要任务队列、并发控制、Web API，工作量明显增加。
2. **目标平台**：只做抖音竖屏，还是同时支持 B 站/小红书？
   - 只做抖音：Sprint 6 可以做 9:16 + 字幕烧录 + 响度；
   - 多平台：需要平台预设和不同的时长/画幅策略。
3. **硬件基线**：目标机器是 4GB GPU 还是 CPU-only？
   - 4GB：保留 panns/text2vec，默认 ASR `base/small`；
   - CPU-only：优先规则特征，重模型全部候选窗口内运行。
4. **标注资源**：能否提供 5~10 场真实直播和人工标注？
   - 能：Sprint 5 直接进入两阶段 + 评估；
   - 不能：先做 dry-run HTML + 人工 keep/drop 反馈，边用边积累。
5. **`--preprocess`**：保留并实现 auto-editor/EDL，还是直接删除该参数？
   - 建议短期删除，避免“参数存在但无效”的误导。
6. **发布边界**：是否需要自动发布/上传？
   - 建议本阶段只做导出包，不做自动发布，降低账号与合规风险。
7. **Python 版本**：是否必须支持 3.11，还是可以只支持 3.12+？
   - 建议 CI 覆盖 3.11 + 3.12，开发机可用 3.14，但不要把它作为唯一基线。

---

## 13. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| 抖音页面/签名/协议变化 | 高 | 录制中断 | 解析层隔离、fallback API、错误告警、快速适配 |
| Cookie 24h 过期 | 高 | 无法开播录制 | `check_env.py` 预检、明确提示、可选自动刷新方案 |
| 模型下载失败/显存不足 | 中 | 切片降级 | 核心依赖与 AI 依赖分离、CPU fallback、模型检查脚本 |
| 标注数据不足 | 高 | 无法证明效果 | 先用 dry-run + keep/drop 反馈积累，不急着上模型 |
| 长视频视觉处理过慢 | 高 | 端到端不可用 | 代理流、单次解码、候选窗口精排 |
| FLV 轮转损坏 | 中 | 视频不可用 | tag 边界切分、ffprobe 校验、损坏段隔离 |
| 重构引入回归 | 中 | 稳定性下降 | 先补测试再拆 controller，小步提交 |
| ASR/WhisperX API 漂移 | 中 | 说话人特征失效 | 固定版本、可选依赖、失败降级 |
| 自动发布合规风险 | 低 | 账号风险 | 本阶段不自动发布，只导出 |

---

## 14. 建议的目录演进（渐进式）

```text
src/
├── cli.py                  # CLI 参数解析（从 controller 拆出）
├── contracts.py            # RecordingMeta / FeatureArtifact / PipelineStatus
├── recording/
│   ├── session.py          # 录制编排
│   ├── danmaku_ws.py       # 现有 danmakuWs 迁移
│   └── flv_recorder.py     # 现有 flvRecorder 迁移
├── douyin/
│   ├── room_api.py         # 现有 roomApi
│   └── room_parser.py      # SSR/JSON 解析
├── signals/
│   ├── audio.py
│   ├── text.py
│   └── visual.py
├── scoring/
│   ├── preprocess.py
│   ├── candidates.py
│   ├── boundary.py
│   └── rank.py
├── pipeline/
│   ├── runner.py
│   ├── stage1_recall.py
│   ├── stage2_rerank.py
│   └── status.py
└── config/
```

**约束**：不要为了目录漂亮而大爆炸重写。每拆一个模块，先让旧入口转调新模块，测试全绿后再删旧代码。

---

## 附录 A：本次分析的关键证据

### A.1 测试与覆盖率

```text
系统 Python 3.14：232 passed
覆盖率：71%

低覆盖：
controller.py   40%
roomApi.py      35%
flvRecorder.py  42%
danmakuWs.py    55%
signalAudio.py  60%
```

当前 `.venv` 缺少 `execjs/httpx/pydantic` 等，直接跑 pytest 会有 9 个 collection error；说明环境需要重建并纳入 CI。

### A.2 特征长度不一致

```text
audio_features: 20s
visual_features: 20s
text_features:  17s
danmaku 最大 recvTimeSec: 27.9s
```

scorer 只对 `len == 20` 的信号加权，9 个文本特征被忽略。

### A.3 候选被时长过滤器杀掉

1 小时合成信号快速实验：

```text
k=2.0：59 候选 → 1 个成片
k=2.5：45 候选 → 0 个成片
```

### A.4 配置覆盖失效

```text
SCORER_DETECTION__WINDOWS=[5,15,45] → 实际 [3,10,30]
AUDIO_PANNS_RMS_THRESHOLD=9.9        → 实际 2.0
```

### A.5 权重和不为 1

```text
default  1.00
game     1.07
shopping 0.84
talent   1.04
```

### A.6 真实样例

```text
data/887951935002/time_table.json → {"clips": []}
highlights/ 中仍残留 clip_2s_unknown.mp4
```

这说明“旧产物残留 + 新结果为空”会进一步误导使用者，产物生命周期也需要管理。

### A.7 模型文件缺失

```text
models/ 目录不存在
config/text.yaml → models/cc.zh.100.bin 不存在
```

`signalText` 会记录 warning 并退化为精确关键词匹配；FastText 模糊匹配在当前环境实际未启用。

---

## 附录 B：问题到代码位置索引

| 问题 | 位置 |
|---|---|
| 特征长度不一致被丢弃 | `src/scorer.py:71-94` |
| 候选点 + 固定扩展 + 时长丢弃 | `src/scorer.py:271-372`、`399-402` |
| 文本时长估算 | `src/signalText.py:133-137` |
| 逐条弹幕情感推理 | `src/signalText.py:168-175`、`354-420` |
| 关键词二值化 | `src/signalText.py:258-260` |
| 空特征文件 | `src/signalAudio.py:41-56`、`src/signalText.py:56-70`、`src/signalVisual.py:31-41` |
| 状态只看文件存在 | `src/controller.py:497-500`、`509` |
| 配置环境变量失效 | `src/config/settings.py:232-252` |
| 模块级冻结配置 | `src/scorer.py:23`、`src/signalText.py:14` 等 |
| WS 退避未生效/心跳线程 | `src/danmakuWs.py:114-123`、`128-139` |
| FLV 轮转截断/空闲循环 | `src/flvRecorder.py:92-94`、`151-160` |
| 视觉双次全量解码 | `src/signalVisual.py:54-60`、`85-142` |
| SSR 正则提取 | `src/roomApi.py:193-222` |
| 共享 httpx client 无锁 | `src/roomApi.py:28-35` |
| `-c copy` 关键帧偏移 | `src/clipper.py:69-73` |
| `--preprocess` 未实现 | `src/controller.py:359`、`641` |
| 依赖缺失 | `requirements.txt`、`check_env.py` |

---

## 附录 C：建议新增/修改文件清单（Sprint 3）

**新增**

- `src/contracts.py`
- `src/pipeline/status.py`（或 `src/pipelineStatus.py`）
- `.github/workflows/ci.yml`
- `Dockerfile`
- `Makefile`
- `requirements-dev.txt`
- `tests/fixtures/`（合成视频/音频/弹幕）
- `tests/test_timeline_alignment.py`
- `tests/test_pipeline_status.py`
- `tests/test_candidate_windows.py`
- `scripts/baseline_report.py`
- `scripts/download_models.py`
- `docs/artifacts-schema.md`

**修改**

- `requirements.txt`
- `check_env.py`
- `src/config/settings.py`
- `src/config/loader.py`
- `src/scorer.py`
- `src/signalText.py`
- `src/signalAudio.py`
- `src/signalVisual.py`
- `src/controller.py`
- `config/scorer.yaml`
- `pyproject.toml`
- `README.md`
- `CLAUDE.md`

---

## 15. 最终判断

这个项目已经跨过了“能不能跑”的阶段，当前真正的问题是“结果是否可信、失败是否可定位、优化是否可衡量”。

下一阶段最划算的投入顺序是：

```text
可复现环境
  → 统一时间轴
  → 修正候选与边界
  → 失败可见
  → 录制韧性
  → 两阶段质量优化
  → 评估/反馈闭环
  → 产品化
```

只要 Sprint 3 的五个 P0 项完成，后续的模型优化、两阶段精排、平台适配才有可靠的地基；否则继续堆模型只会让“为什么没切出来/为什么切得不对”更难排查。
