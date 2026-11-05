# 运行手册（Runbook）

> 适用版本：Sprint 5（2026-11-05）
> 覆盖范围：录制、弹幕、AI 切片、缓存恢复、评估/调权/标注、日志与事件排查

本手册面向实际运行和排障。先看“快速定位”，再按故障类型跳转。所有命令默认在项目根目录执行：

```bash
cd /path/to/douyin-living
ROOM=300294032039
DATA="data/$ROOM"
```

> ⚠️ 不要在日志、Issue、聊天记录中粘贴 `.env` / `cookie.txt` 的内容。Cookie 等同于账号凭证。

---

## 1. 快速定位

### 1.1 三个观测入口

| 入口 | 路径 | 用途 |
|------|------|------|
| 人类可读日志 | `logs/YYYY-MM-DD-<uuid>.log` | 步骤进度、ffmpeg/ASR 输出、异常堆栈 |
| 结构化事件 | `logs/events.jsonl` | 每行一个 JSON，按 `runId/step/status` 过滤 |
| 运行状态 | `data/{roomId}/pipeline_status.json` | 本次运行每一步的最终状态、耗时、错误、产物哈希、缓存统计 |

查看最近一次运行的整体结果：

```bash
jq '{
  runId, roomId, startedAt, finishedAt, settingsHash,
  cacheStats, failureSummary,
  steps: (.steps | map_values({status, elapsedMs, error}))
}' "$DATA/pipeline_status.json"
```

只看失败步骤：

```bash
jq -r '.failureSummary[]?' "$DATA/pipeline_status.json"
jq -r '.steps | to_entries[] | select(.value.status == "failed" or .value.status == "invalid")
       | [.key, .value.status, (.value.error // "")] | @tsv' "$DATA/pipeline_status.json"
```

### 1.2 步骤状态含义

| 状态 | 含义 | 处理 |
|------|------|------|
| `ok` | 产物存在且通过校验 | 正常 |
| `empty` | 步骤成功但结果为空（如 ASR 无语音、无候选片段） | 结合业务判断，通常不算故障 |
| `skipped` | 缓存命中或该步被跳过 | 看 `meta.cacheReason` / `meta.reused` |
| `failed` | 执行失败，或产物校验失败 | 看 `error`，按本手册处理 |
| `invalid` | 产物存在但格式/内容不合法 | 删除对应产物或使用 `--from-step` 重建 |
| `running` | 进程中断时遗留的运行态 | 说明上次运行未正常结束，重新执行即可 |
| `pending` | 尚未执行 | `--only-step` 提前停止时可能出现 |

### 1.3 结构化事件字段

`logs/events.jsonl` 每行是一个独立 JSON 对象，核心字段固定：

```json
{
  "ts": "2026-11-05T12:34:56+00:00",
  "schemaVersion": "1.0",
  "event": "pipeline.step",
  "runId": "20261105T123400Z-a1b2c3d4",
  "roomId": "300294032039",
  "step": "2_asr",
  "status": "ok",
  "elapsedMs": 4210,
  "reused": false,
  "cacheReason": "artifact missing: ..."
}
```

`error`、`reused`、`cacheReason` 等扩展字段只在有值时出现。

事件类型：

- `pipeline.start`：一次切片运行开始；
- `pipeline.step`：任一步骤结束（成功、空、失败、跳过都会发）；
- `pipeline.finish`：运行结束，带 `failures`、`cacheHits/cacheMisses/cacheWrites`。

常用查询：

```bash
# 实时跟踪步骤
tail -f logs/events.jsonl | jq -c 'select(.event == "pipeline.step")
  | {ts, step, status, elapsedMs, reused, cacheReason}'

# 最近一次运行的失败步骤
jq -r 'select(.event == "pipeline.step" and .status == "failed")
       | [.ts, .runId, .step, .error] | @tsv' logs/events.jsonl

# 最近 20 次运行的收尾状态
jq -r 'select(.event == "pipeline.finish")
       | [.ts, .runId, .status, .elapsedMs, .failures, .cacheHits] | @tsv' logs/events.jsonl \
  | tail -20
```

---

## 2. Cookie 过期 / 401 / 403

### 症状

- 日志出现 `No cookies in .env`；
- `Fetching room info` 之后报 401/403、空响应或 `Room not live`；
- 弹幕 WebSocket 连上后立即断开；
- 房间信息接口成功但拉流地址为空。

### 处理

1. 先运行环境检查（会读取 Cookie，但只打印条目数量，不打印内容）：

   ```bash
   python check_env.py
   ```

2. 在浏览器登录 `live.douyin.com`，重新复制 Cookie，更新 `.env`：

   ```env
   DY_LIVE_COOKIES="新的 Cookie 字符串"
   ```

   如果代码路径使用 `DY_COOKIES`，同步更新。不要把 Cookie 写进 `config/*.yaml` 或提交到 Git。

3. 重新运行：

   ```bash
   python -m src.controller "$ROOM" --clip-only
   ```

4. 抖音 Cookie 通常约 24 小时失效。长时间录制时建议在开播前更新，并把 Cookie 过期视为日常运维项，而不是代码故障。

5. 如果 Cookie 刚更新仍持续 403：确认房间 ID 正确、当前网络/IP 未被风控、请求频率没有异常；必要时更换网络后再试。

---

## 3. 未开播 / 无流 / 录制提前结束

### 症状

- 日志：`Room not live`、`No FLV URLs found`、`FLV not available`；
- `data/{roomId}/` 下没有 `*.flv`，或只有 `*.flv.part`；
- 切片日志：`No FLV files in ...`，`pipeline_status.json` 中 `0_input` 为 `failed`；
- 录制器 `stopReason` 为 `ended` / `stream_ended` / `stopped` / `low_disk` / `max_retries`。

### 处理

1. 确认直播间确实在播：

   ```bash
   python -m src.controller "$ROOM"
   ```

   日志会打印房间标题；也可以直接在浏览器打开 `https://live.douyin.com/$ROOM` 核对。

2. 检查已落盘文件：

   ```bash
   ls -lh "$DATA"/*.flv "$DATA"/*.flv.part 2>/dev/null
   find "$DATA" -maxdepth 2 -type f -name '*.flv' -printf '%s %p\n' | sort -n
   ```

3. `*.flv.part` 是未完成分段。程序启动时会清理超过 `stale_part_max_age_secs`（默认 24 小时）的残留；不要在录制进程仍在写入时手工删除。

4. 损坏分段会被移动到 `$DATA/corrupt/`，不会进入合并列表。可手工确认：

   ```bash
   ffprobe -v error -show_entries format=duration -of default=nw=1 "$DATA"/corrupt/*.flv
   ```

5. `max_retries` 表示网络/URL 重试耗尽。查看日志中的 `HTTP error`、`URL expired`、`FLV idle timeout`；确认网络稳定后重新录制。相关配置在 `config/recording.yaml`：

   - `max_retries`：重试次数；
   - `idle_timeout_secs`：无数据超时；
   - `backoff_base` / `backoff_max`：指数退避；
   - `rotation_secs`：FLV 分段时长。

6. 已经录到的分段仍然可用。直播结束后直接做切片：

   ```bash
   python -m src.controller "$ROOM" --clip-only
   ```

   多分段会先校验再合并；损坏分段被隔离，不会污染最终视频。

7. 如果是 `low_disk`，先按第 6 节清理磁盘，再重新录制。

---

## 4. ffmpeg / ffprobe 缺失

### 症状

- 日志：`ffmpeg not installed`、`Audio extraction failed`、`Video clipping failed`；
- `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`；
- `ffprobe not installed — skipping segment validation`。

### 处理

1. 在当前 shell 和虚拟环境中确认二进制可见：

   ```bash
   command -v ffmpeg ffprobe
   ffmpeg -version
   ffprobe -version
   ```

   推荐 ffmpeg 6.0+。

2. 安装：

   ```bash
   # Ubuntu / Debian
   sudo apt update && sudo apt install -y ffmpeg

   # macOS
   brew install ffmpeg

   # conda 环境
   conda install -c conda-forge ffmpeg
   ```

3. 安装后重新打开终端（或确认 `PATH` 包含安装目录），再运行：

   ```bash
   python check_env.py
   ```

4. 重试切片：

   ```bash
   python -m src.controller "$ROOM" --clip-only --only-step clip
   ```

   如果音频还没提取成功，使用 `--from-step audio`。

5. `ffprobe` 缺失时录制仍可继续，但 FLV 分段不会做完整性校验；生产环境应同时安装。

---

## 5. 模型下载失败

### 症状

- `check_env.py --clip` 报 HF 模型未缓存；
- `scripts/download_models.py` 报连接超时、SSL 错误、401；
- ASR 首次运行卡在下载，或文本特征步骤失败；
- `pipeline_status.json` 中 `2_asr` 为 `failed`，或文本/情感特征缺失。

### 处理

1. 检查依赖与模型资产：

   ```bash
   python check_env.py --clip
   python scripts/download_models.py --check --strict
   ```

2. 使用 Hugging Face 镜像重试（国内网络常见）：

   ```bash
   HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --hf
   HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --asr base
   ```

3. FastText 中文向量约 4 GB，不会自动下载：

   ```bash
   python scripts/download_models.py --fasttext
   ```

4. 确认磁盘和缓存目录：

   ```bash
   df -h .
   du -sh ~/.cache/huggingface ~/panns_data 2>/dev/null
   ```

   - faster-whisper / text2vec / 情感模型：`~/.cache/huggingface/`；
   - PANNs 音频事件模型：首次运行时由 `panns-inference` 下载到 `~/panns_data/`。

5. 企业代理环境：

   ```bash
   export HTTPS_PROXY=http://proxy.example.com:8080
   export HTTP_PROXY=http://proxy.example.com:8080
   export NO_PROXY=localhost,127.0.0.1
   ```

   先用 `curl -I https://huggingface.co` 验证网络，再下载。

6. 离线部署时，先在有网机器完成下载并把缓存目录同步到目标机器；确认 `scripts/download_models.py --check --strict` 通过后再设置 `HF_HUB_OFFLINE=1`。不要在模型尚未缓存时开启离线模式。

7. 模型缺失不一定会中断整个切片：ASR 失败会把 `*_asr.json` 标记为 `failed`，下次运行自动重试；文本特征会降级为仅弹幕信号（`text_features.json` 的 `stats.asrStatus/asrDegraded/asrError` 记录原因），视觉特征缺失会降低打分质量。修复模型后重跑：

   ```bash
   python -m src.controller "$ROOM" --clip-only --only-step asr
   # ASR 成功后再继续后续步骤
   python -m src.controller "$ROOM" --clip-only --from-step signal
   ```

---

## 6. 磁盘满

### 症状

- 录制日志：`low disk space`，`stopReason=low_disk`；
- 系统错误：`OSError: [Errno 28] No space left on device`；
- ffmpeg 写文件失败、`.part` 文件异常、合并失败。

### 处理

1. 查看空间和大户：

   ```bash
   df -h .
   du -sh data/* logs 2>/dev/null | sort -h
   find data -name '*.flv.part' -mtime +1 -printf '%TY-%Tm-%Td %s %p\n' 2>/dev/null
   ```

2. 优先清理：

   - 已发布/已归档的旧直播间目录；
   - `$DATA/corrupt/` 中确认无用的损坏分段；
   - `$DATA/highlights/` 中已导出的成片；
   - 过期的日志归档 `logs/*.zip`。

3. 录制保护配置在 `config/recording.yaml`：

   - `min_free_disk_gb`（默认 2 GB）：低于该值时录制器主动停止；
   - `max_segment_bytes`（默认 4 GB）：单个 FLV 分段上限。

   日志轮转/保留在 `src/log/generalConfig.py`：

   - `logRotation = "10 MB"`，`logRetention = "7 days"`；
   - `eventLogRotation = "00:00"`（每天轮转），`eventLogRetention = "30 days"`。

4. 换盘或扩容后，用 `--output-dir` 指向新位置：

   ```bash
   python -m src.controller "$ROOM" --clip-only --output-dir /mnt/bigdisk/douyin
   ```

   正在进行的直播需要重新启动录制才会使用新目录。

5. 清理后重试切片；若源视频仍在，不需要重新录制：

   ```bash
   python -m src.controller "$ROOM" --clip-only --from-step audio
   ```

---

## 7. ASR OOM / 进程被杀

### 症状

- 日志：`MemoryError`、`CUDA out of memory`、`torch.OutOfMemoryError`、`Killed`；
- `pipeline_status.json` 中 `2_asr` 为 `failed`；
- `*_asr.json` 内容为 `{"status": "failed", ...}`。

### 处理

1. 先确认是显存还是内存：

   ```bash
   nvidia-smi
   free -h
   dmesg | tail -30        # 查找 OOM killer / Killed process
   ```

2. 降低 ASR 资源占用。以下配置在 `config/asr.yaml`，也可用环境变量覆盖（无需改代码）。
   新变量使用 `ASR_*` 前缀；旧 `PIPELINE_ASR_*` 仍可作为别名使用：

   | 配置 | 环境变量 | 默认 | 说明 |
   |------|----------|------|------|
   | `model` | `ASR_MODEL`（旧 `PIPELINE_ASR_MODEL`） | `auto` | `auto` 时 CPU 用 `base`、CUDA 用 `small`；`tiny` 最省资源，`large-v3` 最吃显存 |
   | `device` | `ASR_DEVICE`（旧 `PIPELINE_ASR_DEVICE`） | `auto` | `auto/cuda/cpu`；显存不足时用 `cpu` |
   | `compute_type` | `ASR_COMPUTE_TYPE`（旧 `PIPELINE_ASR_COMPUTE_TYPE`） | `int8` | GPU 可选 `float16`，CPU 保持 `int8` |
   | `diarize` | `ASR_DIARIZE`（旧 `PIPELINE_ASR_DIARIZE`） | `true` | 设为 `false` 可关闭说话人分离，显著省内存 |
   | `chunk_sec` | `ASR_CHUNK_SEC` | `1800` | 长音频分块长度；`0` 关闭分块 |
   | `resume` | `ASR_RESUME` | `true` | 复用 `<asr_json>.chunks.json` 中已完成的分块 |

   最小资源组合示例（建议 `export` 后在同一个 shell 中完成重试）：

   ```bash
   export ASR_MODEL=tiny
   export ASR_DEVICE=cpu
   export ASR_COMPUTE_TYPE=int8
   export ASR_DIARIZE=false

   python -m src.controller "$ROOM" --clip-only --only-step asr
   ```

   如果这是长期配置，直接把对应值写入 `config/asr.yaml`，避免忘记设置环境变量导致下一次运行又回到默认模型。

3. CUDA OOM 的补充手段：

   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

   关闭占用 GPU 的其他进程；仍不够时降模型或改用 CPU。

4. 配置变化会改变 ASR 缓存参数，旧的失败产物会被 validator 拒绝并自动重建，不需要手工删除 `*_asr.json`。只重跑 ASR：

   ```bash
   python -m src.controller "$ROOM" --clip-only --only-step asr
   ```

5. ASR 成功后，后续步骤需要重算（ASR 产物指纹变了）：

   ```bash
   python -m src.controller "$ROOM" --clip-only --from-step signal
   ```

   上面两条命令必须使用与 ASR 相同的环境变量/配置（例如都设置 `ASR_MODEL=tiny`），否则 ASR 参数变化会再次触发重建。也可以直接跑完整 `--clip-only`，缓存会自动判断哪些步骤需要重建。

6. 如果仍然被 OOM killer 杀死，先用独立命令定位：

   ```bash
   python asr.py "$DATA/$(basename "$DATA")_merged.aac" \
     --model tiny --device cpu --no-diarize \
     --output "$DATA/$(basename "$DATA")_merged_asr.json"
   ```

   确认单进程可完成后，再把同样的参数用于 controller。注意：手工产物没有 `.meta.json` sidecar，controller 会视为缓存未命中；正式重跑仍应通过 controller 的配置项执行。

7. 内存长期紧张时增加 swap 只能作为兜底，CPU 推理会明显变慢；优先降模型和关闭 diarization。

---

## 8. 缓存与断点续跑

每个中间产物旁边有 `*.meta.json` sidecar，记录源文件指纹、配置 hash、步骤参数、生成时间、耗时和版本。只有全部一致时才复用。

### 常用命令

```bash
# 只重试 ASR（audio 命中缓存，执行完 ASR 停止）
python -m src.controller "$ROOM" --clip-only --only-step asr

# 从 signal 开始重建（复用 audio/asr）
python -m src.controller "$ROOM" --clip-only --from-step signal

# 全部忽略缓存，强制重建
python -m src.controller "$ROOM" --clip-only --force

# 只跑评分并停止
python -m src.controller "$ROOM" --clip-only --only-step score
```

### 排查缓存

```bash
jq '.cacheStats' "$DATA/pipeline_status.json"
jq '.steps | map_values(.meta)' "$DATA/pipeline_status.json"
jq -r '.steps | to_entries[] | select(.value.meta.reused == true)
       | [.key, .value.meta.cacheReason] | @tsv' "$DATA/pipeline_status.json"
```

- `cacheReason` 会说明未命中的原因（`artifact missing`、`settings hash changed`、`source changed`、`step parameters changed`、`generator changed` 等）；
- sidecar 损坏或缺失会自动重建，不会让流程崩溃；
- 不要手工编辑 `*.meta.json`；确实需要单步重建时用 `--from-step`，需要全量时用 `--force`。

---

## 9. 日志与事件运维

### 9.1 轮转与保留

| 日志 | 默认轮转 | 默认保留 | 配置位置 |
|------|----------|----------|----------|
| `logs/*.log` | 10 MB | 7 天 | `GeneralConfig.logRotation`, `logRetention` |
| `logs/events.jsonl` | 每天 00:00 | 30 天 | `GeneralConfig.eventLogRotation`, `eventLogRetention` |

轮转后的文件会自动压缩为 `.zip`。需要改成按大小/按天时，修改 `src/log/generalConfig.py` 后重启进程：

```python
logRotation = "00:00"      # 每天轮转；也可写成 "100 MB"
logRetention = "14 days"
eventLogRetention = "90 days"
```

### 9.2 普通日志

```bash
ls -lt logs/*.log | head
tail -f logs/$(ls -t logs/*.log | head -1 | xargs basename)
```

日志按模块名、函数名、行号格式化；异常会带堆栈。结构化事件不会写入普通日志，避免污染。

### 9.3 事件日志

```bash
# 查看某个 run 的全部事件
jq -c 'select(.runId == "20261105T123400Z-a1b2c3d4")' logs/events.jsonl

# 每个步骤的耗时排行（最近一次 run 可自行加 runId 过滤）
jq -r 'select(.event == "pipeline.step")
       | [.step, .status, (.elapsedMs // 0)] | @tsv' logs/events.jsonl \
  | sort -k3 -n -r | head -20
```

`pipeline.start` 记录 `force/fromStep/onlyStep/profile/sensitivity`；`pipeline.finish` 记录整体耗时、失败数和缓存命中统计，适合接入外部监控或日报。

---

## 10. 评估、调权与标注

### 10.1 评估已有切片

前置条件：

- `data/{roomId}/time_table.json` 已生成；
- 人工标注文件符合 `docs/annotation-guide.md`，至少包含一个正样本
  （`worthClipping=true` 且 `quality>=3`）。

```bash
# 单场，两个 K
python scripts/evaluate.py \
  --time-table "$DATA/time_table.json" \
  --annotations "$DATA/annotations.json" \
  --k 5 --k 10 --json "$DATA/eval.json"

# 多场聚合；任一 macro/micro 目标未达标时返回 1
python scripts/evaluate.py \
  --case data/room1/time_table.json data/room1/annotations.json \
  --case data/room2/time_table.json data/room2/annotations.json \
  --k 5 --fail-on-target

# 先用宽松匹配看召回，再用严格匹配看边界
python scripts/evaluate.py ... --match-iou 0.0
python scripts/evaluate.py ... --match-iou 0.3
```

读结果时注意：

- `precisionAtK` 分母是 `min(K, 实际片段数)`；跨场比较优先看
  `precisionAtKStrict` 或 micro 聚合；
- `recallAtK` 分母始终是全部正样本，K 越小召回越低是正常的；
- `boundaryIou` / `editingCostSec` 只统计匹配成功的片段；没有匹配时为 `—`；
- `invalidRate` 高说明前 K 里有大量片段没有对应高光；
- `duplicateRate` 高说明 NMS/去重不足；
- `negativeHitRate` 高说明系统仍在切“容易误判但不应切”的内容。

初始目标只是建议值。第一次真实标注跑完后，把实测 macro/micro 指标记录到
`docs/2026-11-03-next-development-plan.md` 或版本发布说明，再调整目标。

### 10.2 生成标注页

```bash
python scripts/label_report.py \
  --time-table "$DATA/time_table.json" \
  --top 10 \
  --output "$DATA/label_report"
```

- 浏览器打开 `label_report/index.html`；
- 缩略图内嵌在 HTML 中，上下文视频在 `label_report/media/`；
- 未判断的候选不会导出；页面底部可补 Top-K 之外的正/负样本；
- `--annotations` 可传入已有标注，自动预填匹配候选；
- 没有 ffmpeg 时加 `--no-thumbnails --no-context`，仍可生成纯表单页。

标注完成后把导出的 JSON 放到 `data/{roomId}/annotations.json`，再跑
`evaluate.py`。详细规范见 `docs/annotation-guide.md`。

### 10.3 调权

`tune_weights.py` 直接读取 `audio_features.json` / `text_features.json` /
`visual_features.json`，临时替换 `src.scorer.PROFILES` 中的权重后调用
`score()`，**不会重跑 ASR/视觉模型**。

```bash
# 随机搜索
python scripts/tune_weights.py \
  --case "$DATA/time_table.json" "$DATA/annotations.json" \
  --profile default --k 5 --objective f1 --aggregate micro \
  --strategy random --iterations 200 --seed 42 \
  --output-yaml /tmp/default.tuned.yaml --json /tmp/tune-default.json

# 坐标式网格搜索（适合特征少、想保守调整时）
python scripts/tune_weights.py \
  --case ... --strategy grid --rounds 2 \
  --grid-multipliers 0,0.5,1.5,2
```

输出中的 YAML 是 `config/scorer.yaml` 的片段，不会自动写入配置。合入前：

1. 确认权重和接近 1（脚本已归一化，四舍五入可能有 1e-6 误差）；
2. 用同一标注集重跑 `evaluate.py`，确认提升不是单场过拟合；
3. 至少留一场未参与调参的直播做验证；
4. 记录 `objective`、`baseline`、`best`、`evaluated` 和随机种子。

常见问题：

| 现象 | 原因 | 处理 |
|------|------|------|
| `no non-empty feature files` | 目录里没有三路特征产物 | 先跑完整 `--clip-only` 或至少 signal 步骤 |
| `annotations contain no positive highlights` | 标注全是负样本或质量都 <3 | 检查 `worthClipping/quality`，或调低 `--min-quality` |
| 所有候选 objective 都是 0 | 特征/短视频不足以产生片段 | 用至少 30~120 分钟真实场次；先确认 `time_table.json` 有候选 |
| 调权结果提升很大但评估下降 | 单场过拟合或标注边界不一致 | 增加场次、固定标注规范、留出验证集 |

### 10.4 Profile 对比与合并建议

`profile_report.py` 用同一批标注集重评分所有 profile，检查每个 profile 是否
有足够数据、是否真的优于 `default`：

```bash
python scripts/profile_report.py \
  --case data/room1/time_table.json data/room1/annotations.json \
  --case data/room2/time_table.json data/room2/annotations.json \
  --k 5 --min-cases 2 --min-positives 5 \
  --json /tmp/profile-report.json
```

判定规则：

| 条件 | 建议 | 含义 |
|------|------|------|
| owned cases < `--min-cases` 或正样本 < `--min-positives` | `collect_more_or_merge` | 证据不足，先补标注，不要把 profile 当已验证 |
| micro F1 相对 `default` 提升 < `--min-delta`（默认 0.02） | `merge_into_default` | 有数据但差异化权重没有收益 |
| 提升 ≥ `--min-delta` | `keep` | 可以保留独立 profile |

`default` 始终保留为兜底。报告同时输出每个 case 在全部 profile 下的
P/R/F1、最佳 profile，以及 profile 两两之间的 total-variation / cosine
距离，用于判断权重是否真的不同。脚本不会修改 `config/scorer.yaml`；合入前
必须确认 profile 在未参与调参的场次上仍然有效。

> 当前仓库没有真实人工标注，所有 profile 仍只是先验权重。`profile_report.py`
> 的结论必须以真实标注为准，不能仅凭权重距离宣称 profile 有效。

### 10.5 Dry-run 可视化报告（S6-1）

用已有录制做一次不裁剪的评分，并生成可直接在浏览器打开的 HTML：

```bash
DATA=data/300294032039
python -m src.controller "$(basename "$DATA")" --clip-only --dry-run \
  --report-html "$DATA/report/index.html"
```

- 报告在评分完成后生成，`--dry-run` 也会触发；生成失败只记录 warning，
  不会中断评分/裁剪主流程；
- 传入 `.html`/`.htm` 文件路径时，缩略帧写到同目录的 `<stem>_media/`；
  传入目录时写 `<dir>/index.html` 和 `<dir>/media/`；
- 页面自包含（内联 CSS/JS/SVG、base64 缩略图），不需要 CDN；
- 缩略帧优先用源视频通过 ffmpeg 提取，失败时回退
  `highlights/clip_<start>s_*.jpg`；
- 报告包含分数曲线、候选原因/风险/丢弃原因、弹幕 Top 词、ASR 摘要、
  信号贡献和可复制的 ffmpeg 命令/时间码；
- 没有 `scores.json` / ASR / 弹幕 / 两阶段候选时，对应区块显示缺失提示，
  不影响其他区块。

排障：

| 现象 | 原因 | 处理 |
|------|------|------|
| 报告没有生成，日志有 `Dry-run report generation failed` | 路径不可写、`time_table.json` 非法或模板异常 | 检查目录权限和 `time_table.json`；报告失败不影响主流程 |
| 缩略帧显示“无缩略帧” | 源视频缺失、ffmpeg 失败且没有同名 `highlights/*.jpg` | 检查 ffmpeg 和 `--output-dir`；或用 `highlights` 里的封面回退 |
| 弹幕 Top 词很少 | 该场次聊天文本少，或 jieba 未安装回退到 n-gram | 属于预期降级；可安装 `jieba` 后重生成报告 |
| 报告里看不到召回区块 | 使用 legacy 评分或缺少 `recall_candidates.json` | 加 `--two-stage` 重跑评分 |

---

## 11. 发布前 / 交付前检查

```bash
# 1. 环境与模型
python check_env.py --clip

# 2. 测试与覆盖率
pytest -q --cov=src --cov-fail-under=75
ruff check src tests scripts

# 3. 用已有录制做一次不裁剪的端到端验证
python -m src.controller "$ROOM" --clip-only --dry-run \
  --report-html "$DATA/report/index.html"

# 4. 核对报告与状态/事件
test -s "$DATA/report/index.html"
jq '.failureSummary, .cacheStats' "$DATA/pipeline_status.json"
jq -r 'select(.event == "pipeline.finish") | [.ts, .status, .failures] | @tsv' logs/events.jsonl | tail -1

# 5. 有真实标注时跑评估（没有标注则跳过并注明）
python scripts/evaluate.py \
  --time-table "$DATA/time_table.json" \
  --annotations "$DATA/annotations.json" \
  --k 5 --k 10 --json "$DATA/eval.json"
```

交付时应满足：

- `check_env.py --clip` 无 critical issue；
- 测试全绿，覆盖率不低于门禁；
- `pipeline_status.json` 中没有 `failed/invalid` 步骤（`empty` 需结合业务确认）；
- 最近一次 `pipeline.finish` 的 `status` 为 `ok`；
- 已确认 Cookie 有效期、磁盘余量和模型缓存；
- 有真实标注时，评估指标与初始目标的差异已记录，没有标注时明确说明
  “尚未建立真实基线”。
