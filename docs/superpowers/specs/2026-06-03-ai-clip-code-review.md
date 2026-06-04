# 抖音直播 AI 智能切片系统：设计与代码审阅报告

> 审阅对象：
> - 设计文档：`docs/superpowers/specs/2026-06-03-ai-clip-design.md`
> - 当前代码：`src/controller.py`、`src/scorer.py`、`src/signalAudio.py`、`src/signalText.py`、`src/signalVisual.py`、`src/clipper.py`、`asr.py` 等
>
> 审阅时间：2026-06-03
>
> 约束：本次只做审阅与建议，不改动业务代码。

## 1. 总体结论

当前代码已经具备 AI 切片系统的模块骨架：录制编排、音频特征、文本特征、视觉特征、融合打分、ffmpeg 裁剪、ASR 独立脚本都已存在，CLI 参数也基本覆盖设计文档的 Phase 1 入口。

但代码与设计文档的匹配度只能评为 **中低：约 55%-65%**。原因不是方向不对，而是当前实现更接近“可尝试跑通的 Phase 1 原型”，而设计文档描述的是一个更完整的 v3.0 系统。多个关键能力在代码里尚未真正落地，且存在若干会直接影响结果正确性的实现问题。

优先级最高的结论：

1. **弹幕时间轴目前不可用**：当前 JSONL 样例没有 `timestamp` / `_time` 字段，而 `signalText.py` 依赖该字段；结果会把弹幕集中到 0 秒附近，文本信号无法科学参与打分。
2. **文本特征提取存在确定性 bug**：`signalText.py` 在计算 UTR 后 `del allText30s`，随后又用它计算熵和话题变化，会导致文本特征提取失败。
3. **ASR 与说话人分离没有按设计接入管线**：`asr.py` 默认支持 diarization，但 `controller.py` 调用时显式 `diarize=False`，说话人切换特征基本失效。
4. **融合打分层只有一部分实现**：权重表已实现，但时滞补偿、时间衰减、DTW、语义边界、高斯包络、局部 DP、SSIM 去重等核心设计要么未实现，要么只是注释/占位。
5. **依赖声明严重滞后**：`requirements.txt` 和 `pyproject.toml` 没有声明 `librosa`、`faster-whisper`、`scenedetect`、`opencv`、`jieba`、`fasttext`、`text2vec`、`transformers`、`panns-inference`、`whisperx` 等设计依赖，干净环境无法按文档运行。

## 2. 文档与代码匹配度

### 2.1 已匹配的部分

| 设计项 | 当前代码状态 | 评价 |
|---|---|---|
| `controller.py` 统一编排 | 已有 `--clip`、`--clip-only`、`--profile`、`--sensitivity`、`--dry-run` 等 CLI 参数 | 入口基本对齐 |
| `extractAudio.py` / ffmpeg 音频提取 | 已有独立脚本，controller 内也会提取 AAC | 基本对齐 |
| `asr.py` faster-whisper | 独立脚本已实现 faster-whisper 转录 | 部分对齐 |
| `signalAudio.py` | 已实现 RMS、ZCR、频谱质心、MFCC 距离、panns 事件检测 | 基本对齐 |
| `signalText.py` | 已实现弹幕密度、加速度、UTR、关键词、情感、话题变化占位 | 方向对齐但有正确性问题 |
| `signalVisual.py` | 已实现 PySceneDetect、OpenCV 光流，保留 faceCount | 部分对齐 |
| `scorer.py` 权重 profile | default/game/shopping/talent 权重表基本等同设计 | 对齐 |
| `clipper.py` ffmpeg 裁剪、字幕烧录、缩略图 | 已实现基础封装 | 部分对齐 |
| `config/keywords.json` | 已有中文关键词词典，且比文档多了 emotional 类别 | 对齐并略有扩展 |

### 2.2 尚未匹配或不一致的部分

| 设计项 | 文档描述 | 当前代码 | 影响 |
|---|---|---|---|
| 弹幕时间戳与 PTS 对齐 | 弹幕服务器时间戳经线性回归校准到视频 PTS | `danmakuWs.py` 输出没有时间戳；`signalText.py` 解析不到则默认 0 | 文本信号时间轴基本失真 |
| ASR diarization | WhisperX 说话人分离，输出 speaker 标签 | `asr.py` 支持，但 `controller.py` 调用 `diarize=False` | `speakerChange` 特征失效 |
| 并行模型隔离 | `ProcessPoolExecutor(max_workers=3)`，大模型子进程释放显存 | `ThreadPoolExecutor(max_workers=3)` | 显存隔离、CPU/GPU 资源调度不符合设计 |
| auto-editor | 可选预处理并生成 EDL 映射 | `--preprocess` 参数存在，但流程未实现 | 文档功能未落地 |
| faceCount | 候选峰值区域按需 insightface | `faceDetection` 默认 False，controller 无二阶段调用 | 视觉人脸特征恒为 0 |
| 时滞补偿 | 互相关算出延迟后校准信号 | `_compensateLag()` 只计算和记录，不移动数组 | 对齐能力未生效 |
| DTW | 文档架构图写了 DTW 对齐 | 代码没有 DTW | 文档超前于实现 |
| 时间衰减 | `Score(t)=S_base(t)*decay + boost` | 代码 `finalScore = scoreBase + boost`，没有 decay | 打分公式不一致 |
| 多尺度峰值检测 | 3s/10s/30s 多尺度 | 有窗口循环，但阈值使用局部窗口且包含当前点，未形成多尺度片段评分 | 效果可能不稳定 |
| 语义边界 | ASR segment 句界/停顿对齐 | `_optimizeBoundaries()` 只看 RMS 谷值 | 切片可能截断语义 |
| 高斯包络与局部 DP | RMS 高斯拟合、局部 DP 精调 | 未实现 | 边界优化不足 |
| SSIM 去重 | 相邻片段 SSIM > 0.95 合并 | `ssim` 被 import 但未使用 | 去重设计未落地 |
| t-test 质检 | 片段前后 3s 对比 | 只和片段前 5s 做 t-test | 质检口径不一致 |
| 输出 `.srt` | 每个切片附带对应字幕文件 | 只尝试烧录全局 SRT 到 `_sub.mp4`，没有生成片段级 SRT | 输出不符合设计 |
| 输出分辨率 | `--resolution 720p|1080p|4k` | 参数存在但未传给 ffmpeg | CLI 参数无效 |

## 3. 关键问题清单

### P0：会直接导致结果错误或管线失败

1. **`signalText.py` 删除后复用 `allText30s`**

   代码在计算 UTR 后执行 `del allText30s`，随后又在构建输出前调用：

   - `dmEntropyRaw = _computeEntropyWindow(allText30s, nSecs)`
   - `topicVec = _computeTopicChange(allText30s, nSecs)`

   这会导致文本特征提取抛出 `UnboundLocalError`，controller 捕获后只记录 “Signal extraction failed”，后续 scorer 会在没有文本特征的情况下继续运行。

2. **弹幕 JSONL 缺少时间戳**

   当前样例弹幕行包含 `method/type/content/userName/userId`，没有 `timestamp`、`_time` 或服务器时间字段。`signalText.py` 的 `_parseTimestamp()` 取不到时间时返回 0.0，导致所有弹幕都落在第 0 秒。

   这与设计文档“弹幕服务器时间戳 → 线性回归 → PTS”的方案完全不匹配。弹幕密度、弹幕加速度、UTR、情感、熵、话题变化都会失真。

3. **ASR 失败后生成空 JSON，且下游静默降级**

   `controller.py` 中 ASR 失败会写入 `[]` 占位文件。当前数据目录里的 `_asr.json` 文件大小为 2 字节，符合空列表形态。这样 `asrKeyword`、`speakerChange`、字幕烧录都无法提供有效信号，但管线仍可能继续。

4. **测试入口不可直接运行**

   从仓库根目录执行 `python tests\verify_modules.py` 会出现 `ModuleNotFoundError: No module named 'src'`。设置 `PYTHONPATH` 后又遇到 Windows GBK 控制台无法输出 `✓/✗/ℹ` 等符号的问题。测试本身覆盖范围也只包含基础录制模块，不覆盖 AI 切片模块。

### P1：设计承诺未落地，明显影响效果

1. **`controller.py` 禁用说话人分离**

   `asr.py` 默认参数是 `diarize=True`，但 controller 调用时写死 `diarize=False`。这使设计中的 WhisperX diarization、speaker 标签、speakerChange 权重无法生效。

2. **依赖声明不完整**

   `requirements.txt` 只包含 `aiohttp/websockets/scipy/scikit-image/protobuf/httpx` 等少量依赖；`pyproject.toml` 更少。AI 切片实际需要的核心包没有被声明，安装清单只存在于设计文档中。

3. **视觉 sceneChange 语义错误**

   `signalVisual.py` 将 PySceneDetect 返回的每个 scene 的整段区间都标为 1，而不是只在场景切换点标 1。这样长场景会被持续加分，和“场景切换事件”含义不一致。

4. **情感模型加载方式不可接受**

   `_fastSentiment()` 每处理一条弹幕都可能创建一次 HuggingFace pipeline。若 transformers 可用，这会极慢且反复加载模型。设计里写的是 `HF pipeline`，但工程实现应是批量加载一次、批量推理或退化为轻量规则。

5. **融合打分的核心增强没有真正执行**

   `_compensateLag()` 只算 lag 不应用；时间衰减没有进入最终公式；DTW、局部 DP、高斯拟合、SSIM 去重都没有落地。当前 scorer 更像“滚动标准化 + 权重加和 + 简单候选检测”。

6. **片段长度生成逻辑容易把候选全部过滤掉**

   `_detectPeaks()` 返回的是候选点聚类范围，通常很短；`_optimizeBoundaries()` 只扩展约 5-10 秒；随后 `_qualityFilter()` 要求最小时长 15 秒。很多真实峰值可能在最后一关被过滤，出现 “No clips detected”。

7. **字幕和输出文件不符合文档**

   文档要求每个 `clip_xxx.mp4` 附带 `clip_xxx.srt`。当前代码只在存在全局 SRT 时烧录生成 `_sub.mp4`，没有生成片段级、时间重置后的 SRT，也没有替换原 clip 路径。

### P2：可维护性与工程一致性问题

1. `src/controller.py` import 了 `FlvRecorder`，但当前实现没有使用该类，而是内部 `_downloadFlv()` 下载。
2. `src/scorer.py` import 了 `ssim`，但未使用。
3. 多处 `except Exception: pass` 会吞掉字幕、缩略图、关键词匹配等错误，排查效果问题会很痛。
4. `resolution` 参数未使用，用户以为输出 720p/1080p 可控，实际 ffmpeg 直接 copy。
5. `scores.json` 只保存最终分数数组，没有保存每秒 breakdown、各信号贡献、阈值、候选峰值等可解释信息，和“可追溯每帧打分依据”的设计目标不一致。

## 4. 设计方案的科学性与效果优化建议

### 4.1 先修正时间轴，再谈模型

当前最大瓶颈不是模型不够强，而是多模态信号没有可靠共同时间基准。建议把时间轴改为系统第一优先级：

1. 录制开始时记录 `record_start_monotonic`、`record_start_wall_time`、视频文件首帧 PTS。
2. 每条弹幕写入：
   - `recv_monotonic_ms`
   - `recv_wall_time_ms`
   - 若 protobuf 有服务端时间，则写入 `server_time_ms`
   - `pts_sec` 或可回放映射所需字段
3. 所有 signal 输出保留统一 schema：`sampleRate`、`duration`、`origin`、`timeBase`、`sourceFile`、`sourceHash`、`generatedAt`。
4. scorer 只接受已经对齐到同一 PTS 轴的特征，避免各模块各自猜时间。

这是提升效果最划算的一步。

### 4.2 改成“两阶段候选生成 + 精排”

设计文档希望一次性并行抽全量音频/文本/视觉/人脸信号，但对 4GB 显存和长直播不够经济。更科学的方案：

1. **粗召回**：只用低成本信号全量跑，例如弹幕密度、弹幕加速度、音频 RMS/MFCC、简单 motion。
2. **候选合并**：得到 Top-K 或超过阈值的候选窗口，例如每小时 30-80 个。
3. **精排增强**：只在候选窗口内运行 panns、情感模型、text2vec、ASR 关键词、人脸、OCR。
4. **边界优化**：只在候选窗口内使用 ASR 句界、静音、分镜和局部分数曲线精调。

这样更符合“显存友好”和“先跑通再优化”，也能显著降低模型调用量。

### 4.3 scorer 应从“信号堆加”升级为“可解释候选模型”

当前 scorer 中所有 z-score 直接加权，容易出现几个问题：异常值支配、负分抵消、不同直播类型分布差异、短峰值被平滑掉。

建议 Phase 1 保持规则模型，但做以下优化：

1. 标准化从 rolling mean/std 改为 rolling median/MAD，抗异常刷屏更稳。
2. 每个信号先做单调变换和截断，例如 `clip(z, -3, 6)`，避免极端值支配。
3. 对“事件型特征”和“连续型特征”分开处理：事件型用脉冲扩散，连续型用窗口积分。
4. 候选分数用窗口级特征，而不是秒级点值，例如：
   - `peak_score`
   - `area_under_score`
   - `pre_post_delta`
   - `multi_signal_agreement`
   - `novelty`
5. 输出 `score_breakdown` 时记录正向贡献、负向惩罚、触发阈值、候选来源。

### 4.4 文本特征要更贴近直播语言

当前文本方案偏“通用 NLP”，对直播切片未必最高效。建议：

1. 弹幕关键词优先使用精确/正则/短语词典，FastText 只作为召回补充。
2. 对“哈哈哈”“666”“卧槽”“退钱”“链接”“库存”等高频直播模式做专门规则。
3. 情感模型改为批处理，或先用轻量词典 + emoji/标点特征；只有候选窗口再跑 HF。
4. UTR 不能只按整句去重，应做 token 级、用户级、时间级联合反刷屏：
   - 同用户重复惩罚
   - 同内容重复惩罚
   - 短时间突增但 unique user 不增长的惩罚
5. `dmUniqueUsers` 目前有输出但没有权重，可加入 scorer 作为比纯消息数更可靠的热度信号。

### 4.5 视觉特征应降低全量计算成本

当前 Farneback 光流全量跑长视频可能慢，且 4K 输入会更重。建议：

1. 对视频统一抽低分辨率预览流，例如 320p 或 480p。
2. motion 可先用帧差/直方图差作为粗特征，候选窗口再跑光流。
3. `sceneChange` 只记录切换点，不记录整段 scene。
4. faceCount/OCR 不要作为全量 Phase 1 特征，应进入候选精排阶段。

## 5. 整体优化路线

### P0：修正确实会坏的东西

1. 为弹幕 JSONL 增加录制时间戳，并让 `signalText.py` 基于真实时间轴聚合。
2. 修复 `allText30s` 删除后复用问题。
3. 明确 ASR 失败策略：失败应标记 `asrStatus=failed`，不要静默写空列表伪装成功。
4. 补齐依赖声明，至少拆成 `requirements-base.txt` 和 `requirements-ai.txt`。
5. 修复测试执行方式和 Windows 控制台编码问题。

### P1：让实现真正对齐设计 v3.0

1. controller 使用 ProcessPool 或独立子进程承载大模型任务。
2. 按设计启用/可配置 WhisperX diarization。
3. scorer 应实际应用时滞补偿、时间衰减、边界 DP 或把文档中未实现项降级为 Phase 2。
4. clipper 生成片段级 SRT，并支持 `--resolution` 转码。
5. `scores.json` 保存阈值、候选、各信号贡献和过滤原因。

### P2：提高切片效果

1. 建一个小型人工标注集，哪怕只有 5-10 场直播，每场标注“值得切/不值得切”窗口。
2. 加入离线评估指标：Precision@K、Recall@K、平均人工评分、重复率、无效片段率。
3. 对 profile 权重做网格搜索或贝叶斯优化，而不是手工猜。
4. 有数据后再引入 LightGBM；没有标签前不建议过早引入复杂模型。

### P3：提高性能和可运维性

1. 所有模型 lazy-load 后缓存到进程级单例，但在任务结束时可释放。
2. 文本模型批处理，避免逐条弹幕加载/推理。
3. 中间产物增加 schema version 和 source hash，避免旧特征误复用。
4. 每一步输出明确状态文件，例如 `pipeline_status.json`，记录 skipped/failed/success 和耗时。
5. 日志避免直接输出不兼容 GBK 的符号，或启动时强制 UTF-8。

## 6. 文档本身建议调整

1. 当前设计文档标注“评审通过，待实施”，但代码已有部分实现。建议文档增加“实现状态”列，区分 `implemented`、`partial`、`planned`。
2. 架构图里写了 DTW，但正文仅弱化为互相关；如果 Phase 1 不做 DTW，应从核心路径中移到 Phase 2。
3. “FLV (H.264, 最高 4K or4)” 中 `or4` 应解释为清晰度 key，避免看起来像格式笔误。
4. “GitHub 20k★ 项目”不适合做硬原则，建议改为“优先复用成熟、CLI 可自动化、许可证兼容的工具”。
5. 隐私合规里写 insightface 存 468 个关键点，但当前代码只输出 faceCount；如果未来仅需要人数，建议文档也改为“不保存人脸关键点，只保存聚合计数”。
6. 安装清单应与 `requirements`/`pyproject` 保持一致，并说明 CPU-only、GPU、国内镜像三种安装路径。

## 7. 建议的优先级排序

| 优先级 | 事项 | 原因 |
|---|---|---|
| P0 | 弹幕时间戳与 `signalText.py` bug | 没有可靠文本时间轴，切片核心信号不可用 |
| P0 | 依赖与测试入口 | 否则无法复现和验证 |
| P1 | ASR diarization 接入策略 | 文档权重和 speakerChange 依赖它 |
| P1 | scorer 实现与文档对齐 | 当前效果上限主要卡在这里 |
| P1 | 片段级字幕和分辨率 | 直接影响最终产物可用性 |
| P2 | 两阶段候选 + 精排 | 同时提升效果和性能 |
| P2 | 标注集与离线评估 | 没有评估集，权重优化会靠感觉 |

## 8. 最终判断

这个系统的方向是对的：弹幕、音频、ASR、视觉、多模态融合，是直播切片里比较合理的技术路线。当前代码也已经把主要文件结构搭起来了，不是空方案。

但要达到设计文档承诺的“智能切片系统”，不能继续只堆模型和特征。应先把 **时间轴、文本特征、ASR 状态、依赖复现、scorer 真实性** 这五件事打牢。它们修好后，再做候选精排、边界优化和模型评估，效果提升会更稳定，也更容易知道每一次优化到底有没有用。
