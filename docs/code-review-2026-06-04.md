# 代码审查报告

**日期**: 2026-06-04
**范围**: douyin-living 全部源码
**状态**: 已识别，待修复

---

## 1. 耦合度过高

### 1.1 Controller 是典型 God Object

**文件**: src/controller.py (489 行)
**问题**: 承担 8 个独立职责，违反单一职责原则

- CLI 参数解析 (L451-484) → 应在独立 src/cli.py
- Auth 管理 (L108-110) → 已有 auth.py，应解耦
- 房间信息获取 + WS 参数组装 (L114-150) → 应在 roomApi 或独立 pipeline
- FLV 录制 (L182-203, L406-438) → 绕过了 FlvRecorder 类
- 音频提取 (L235-240, L425-438) → 应在独立 audio 模块
- ASR 编排 (L247-260) → 应在 pipeline coordinator
- 信号提取并行编排 (L266-305) → 应在 pipeline coordinator
- 裁剪 + 字幕 + 状态写入 (L349-401) → 应在 pipeline coordinator

**关键发现**: FlvRecorder 类已存在（带重试、文件轮转、统计），但 Controller 的 _downloadFlv 方法完全绕过它，用更简单的方式重新实现了下载逻辑。

---

### 1.2 Cookie 解析逻辑重复 3 处

位置 1: auth.py:57-66 → _transCookies()
位置 2: roomApi.py:48-54 → 内联实现
位置 3: roomApi.py:140-145 → 内联实现 (再次)

所有实现都假设 cookie 用 "; " (分号+空格) 分隔

边界情况未处理:
- 浏览器导出的 cookie 可能用 ";" (只有分号) 分隔
- 空 cookie 字符串没有 warning

---

### 1.3 _fmtSize 函数重复 2 处

位置 1: controller.py:441-446
位置 2: clipper.py:159-164

---

### 1.4 硬编码的魔术字符串贯穿全链路

Scorer 的权重字典使用字符串键 (dmDensity, rms, eventLaughter)，这些字符串必须与 signal extractor 的输出 JSON 键完全一致才能工作，但没有任何编译时或运行时验证。

风险: 如果任何一处拼写错误，融合打分会静默丢失信号，不会报错。

---

## 2. 可提升空间

### 2.1 缺少 requirements.txt

README 写的是 pip install -r requirements.txt，但该文件不存在。

pyproject.toml 只声明了 4 个依赖，但实际运行需要 18+ 个包。

---

### 2.2 _computeTopicChange 是空壳实现

**文件**: src/signalText.py:372-410

代码声称做"话题聚类"，但实际实现是每 30 秒标记一次切换 (result[30]=1, result[60]=1, ...)。

text2vec + KMeans 模型加载了但完全没用上。对所有直播都产生完全相同的模式。

---

### 2.3 Scorer 峰值检测 O(n*w) 性能问题

**文件**: src/scorer.py:249-291

对于 1 小时录制 (nSecs=3600)，总迭代次数约 1,512,000 次。

建议使用 cumulative sum 向量化，性能提升 10-100x。

---

### 2.4 无测试套件

tests/verify_modules.py 只验证模块能否 import，没有真正的单元测试。关键模块如 scorer.py 完全没有可测试性保障。

---

### 2.5 FlvRecorder 和 Controller 的 _downloadFlv 功能重复

FlvRecorder: 有重试逻辑、文件轮转 (30min)、统计信息
Controller._downloadFlv: 无重试、无轮转、无统计

Controller 完全没有使用 FlvRecorder，导致 FlvRecorder 成为死代码。

---

### 2.6 ASR 模块 seg 对象类型处理脆弱

**文件**: asr.py:116-126

WhisperX 对齐后 segmentsList 的元素类型会从 Segment 变成 dict，代码用 isinstance 分支处理两种类型，脆弱且难读。

---

### 2.7 全局可变状态 _http 客户端

**文件**: src/roomApi.py:14

全局 httpx 客户端永远不会被关闭，长运行进程中可能泄漏连接。

---

### 2.8 _fmtSrtTime 重复 2 处

位置 1: asr.py:151-157
位置 2: clipper.py:144-149

---

## 3. 不适合之处

### 3.1 time.monotonic() vs ASR 时间轴对齐问题

**文件**: src/controller.py:153,158

弹幕使用 time.monotonic() 偏移量作为时间戳，但 ASR 使用的是音频文件内的绝对时间。如果录制过程中有暂停、网络抖动、或系统休眠，两者会产生漂移。

建议改用 time.time() (wall clock)。

---

### 3.2 Cookie 解析不处理边界情况

**文件**: src/auth.py:57-66

未处理的边界情况:
1. 如果 cookie 用 ";" 分隔（无空格），整个解析失败
2. 空 cookie 字符串会返回空 dict，没有 warning

---

### 3.3 _computeEntropy 未实现但保留

**文件**: src/signalText.py:308-311

死函数，已被 _computeEntropyWindow 替代，但仍保留在代码中。

---

### 3.4 _computeEntropyWindow 与设计文档不一致

设计文档说用 HDBSCAN 话题聚类，但实际实现是简单的 Counter + Shannon 熵。

---

### 3.5 UTR 阈值逻辑需要验证

**文件**: src/signalText.py:130-133

UTR_THRESHOLD = 0.3 非常低，正常聊天的 UTR 通常 > 0.7。这意味着几乎所有正常聊天都不会被惩罚，只有极端刷屏才会。

建议验证实际 UTR 分布，考虑调整阈值到 0.5-0.6。

---

### 3.6 设计文档与代码不一致

- HDBSCAN 话题聚类 → KMeans（且是空壳实现）→ 未实现
- DTW 对齐 → 未实现
- 高斯包络边界优化 → 仅用 RMS 谷值检测 → 部分实现
- SSIM 去重 → 未实现（scikit-image 已引入但未使用）
- 局部 DP 优化 → 仅做了简单的 +-10s 搜索 → 部分实现
- 情感分析用 uer/roberta → 已实现

---

### 3.7 GeneralConfig 类不适合当前用途

**文件**: src/log/generalConfig.py

纯静态配置类，所有值都是硬编码的，混合了三类不相关的配置:
1. 日志配置
2. API 常量
3. 浏览器指纹

日志级别无法通过 CLI 参数配置。

---

## 4. 修复优先级

### P0 - 必须立即修复

- 2.1 缺少 requirements.txt (影响: 新开发者无法运行, 难度: 低)
- 1.1 Controller 是 God Object (影响: 代码维护困难, 难度: 中)
- 1.4 硬编码魔术字符串 (影响: 静默丢信号, 难度: 低)
- 3.1 time.monotonic 时间漂移 (影响: ASR 和弹幕不对齐, 难度: 低)

### P1 - 高优先级

- 1.2 Cookie 解析重复 3 处 (影响: 维护困难 + 边界情况, 难度: 低)
- 1.3 _fmtSize 重复 2 处 (影响: 代码重复, 难度: 低)
- 2.2 _computeTopicChange 空壳 (影响: 浪费计算 + 无意义信号, 难度: 中)
- 2.5 FlvRecorder 死代码 (影响: 代码混乱, 难度: 中)
- 2.6 ASR seg 类型处理脆弱 (影响: 未来可能崩溃, 难度: 低)

### P2 - 中优先级

- 2.3 Scorer 峰值检测性能 (影响: 长视频性能差, 难度: 中)
- 2.4 无测试套件 (影响: 重构风险高, 难度: 中)
- 2.7 全局 _http 客户端 (影响: 连接泄漏风险, 难度: 低)
- 2.8 _fmtSrtTime 重复 2 处 (影响: 代码重复, 难度: 低)
- 3.3 _computeEntropy 死代码 (影响: 代码混乱, 难度: 低)

### P3 - 低优先级

- 3.4 设计文档不一致 (影响: 文档混乱, 难度: 低)
- 3.5 UTR 阈值验证 (影响: 可能不准确, 难度: 低)
- 3.6 设计文档多项未实现 (影响: 功能缺失, 难度: 高)
- 3.7 GeneralConfig 不适合 (影响: 配置管理困难, 难度: 中)

---

## 5. 建议的修复顺序

### Phase 1: 基础设施 (1-2 天)
1. 创建 requirements.txt
2. 提取共享工具函数到 src/util.py
3. 添加 feature name 常量或验证机制

### Phase 2: 解耦 Controller (2-3 天)
4. 提取 CLI 到 src/cli.py
5. 提取 AI 管道到 src/pipeline.py
6. 重构 Controller 只负责录制协调
7. 决定 FlvRecorder 命运（使用或删除）

### Phase 3: 修复具体问题 (1-2 天)
8. 统一 Cookie 解析
9. 修复 time.monotonic → time.time
10. 实现真正的 _computeTopicChange
11. 清理死代码

### Phase 4: 性能和测试 (2-3 天)
12. 向量化 Scorer 峰值检测
13. 为 Scorer 添加单元测试
14. 修复 ASR seg 类型处理

### Phase 5: 文档和配置 (1 天)
15. 更新设计文档
16. 重构 GeneralConfig

---

## 6. 附录：完整的依赖列表

### 核心依赖 (必须)
aiohttp>=3.9, websockets>=12.0, protobuf>=5.0, httpx>=0.27
python-dotenv>=1.0, loguru>=0.7, PyExecJS>=1.5
numpy>=1.24, scipy>=1.10

### 音频/ASR 依赖
librosa>=0.10, faster-whisper>=1.0, panns-inference>=0.1

### NLP 依赖
jieba>=0.42, transformers>=4.30

### 视觉依赖
scenedetect>=0.6, opencv-python>=4.8, scikit-image>=0.21

### 可选依赖 (功能增强)
whisperx>=3.1, fasttext>=0.9, text2vec>=1.0
scikit-learn>=1.3, insightface>=0.7, torch>=2.0

---

**报告生成**: 2026-06-04
**审查人**: Claude Code
**代码库**: douyin-living
