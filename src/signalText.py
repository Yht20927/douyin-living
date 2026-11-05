# -*- coding: utf-8 -*-
"""Text signal extraction — danmaku + ASR text features."""

import json
import hashlib
import re
import threading
import numpy as np
from pathlib import Path
from collections import Counter
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().text

# FastText threshold for fuzzy match
SIMILARITY_THRESHOLD = _cfg.similarity_threshold

# Anti-spam UTR (Unique Token Ratio) threshold.
UTR_THRESHOLD = _cfg.utr_threshold

# Default keyword categories (loaded from config/keywords.json)
_KEYWORD_CONFIG_PATH = Path(_cfg.keyword_config_path)

# ── ModelPool name registry ──────────────────────────────────────────
POOL_NAME_FASTTEXT = "fasttext:cc.zh.300"
POOL_NAME_TEXT2VEC = f"text2vec:{_cfg.text2vec_model.replace('/', '-')}"
POOL_NAME_HF_SENTIMENT = f"hf:{_cfg.sentiment_model.replace('/', '-').replace('uer-', '')}"

# Keyword categories from config/keywords.json → canonical feature names.
KEYWORD_CATEGORY_FEATURES: dict[str, str] = {
    "highEnergy": "keywordHighEnergy",
    "funny": "keywordFunny",
    "controversy": "keywordControversy",
    "interactive": "keywordInteractive",
    "emotional": "keywordEmotional",
    "shopping": "keywordShopping",
}

_CATEGORY_ALIASES = {
    "highenergy": "highEnergy",
    "funny": "funny",
    "controversy": "controversy",
    "interactive": "interactive",
    "emotional": "emotional",
    "shopping": "shopping",
}

_QUESTION_MARKERS = ("?", "？")
_EXCLAMATION_MARKERS = ("!", "！")
_EMOJI_PATTERN = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]"
    r"|\[[^\[\]]{1,8}\]"
)

_RULE_POSITIVE = {
    "哈哈", "哈哈哈", "好", "nice", "牛", "牛批", "赞", "喜欢",
    "漂亮", "精彩", "笑", "笑死", "乐", "666", "绝了", "感动",
    "爱了", "厉害", "优秀", "谢谢", "开心", "可爱", "好笑",
}
_RULE_NEGATIVE = {
    "无语", "差", "垃圾", "退", "骗", "恶心", "翻车", "下头",
    "难受", "破防", "哭了", "失望", "生气", "离谱", "别买",
}


def extractFeatures(
    danmakuPath: str,
    asrPath: str,
    outputPath: str | None = None,
    keywordConfig: str | None = None,
    utrThreshold: float = UTR_THRESHOLD,
    durationSec: float | None = None,
) -> dict:
    """Extract text features from danmaku JSONL and ASR JSON.

    Args:
        danmakuPath: Path to danmaku JSONL file.
        asrPath: Path to ASR JSON (from asr.py, WhisperX format).
        outputPath: If set, save features JSON.
        keywordConfig: Path to keywords JSON, defaults to config/keywords.json.
        utrThreshold: UTR below which text is considered spammy.
            Default 0.3 (lenient); raise to 0.5–0.6 for stricter filtering.
        durationSec: Authoritative recording duration from the controller.
            When omitted, the maximum danmaku/ASR timestamp is used.  The old
            ``len(danmaku) // 10`` estimate is intentionally gone — it
            silently truncated the text timeline and dropped every text
            signal during fusion scoring.

    Returns:
        dict with 'sampleRate', 'duration', 'features', 'status' keys.
    """
    log.info(f"Extracting text features: {danmakuPath} + {asrPath}")
    try:
        return _extractTextFeaturesImpl(
            danmakuPath, asrPath, outputPath, keywordConfig, utrThreshold,
            durationSec=durationSec,
        )
    except Exception as exc:
        log.exception(f"Text feature extraction failed for {danmakuPath}")
        empty = {
            "sampleRate": 1,
            "duration": 0,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "generatorVersion": "signalText/3",
            "featureCount": 0,
            "features": {
                "dmDensity": [], "dmAcceleration": [], "dmEntropy": [],
                "dmSentiment": [], "dmUniqueUsers": [], "dmUtr": [],
                "dmRepeatRate": [], "dmQuestionRate": [],
                "dmExclamationRate": [], "dmEmojiRate": [],
                "keywordHighEnergy": [], "keywordFunny": [],
                "keywordControversy": [], "keywordInteractive": [],
                "keywordEmotional": [], "keywordShopping": [],
                "asrKeyword": [], "topicChange": [], "speakerChange": [],
            },
        }
        if outputPath:
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False)
        return empty


def _resolveDuration(
    durationSec: float | None,
    asrSegments: list[dict],
    danmakuEntries: list[dict],
) -> float:
    """Resolve the text timeline length without inventing one.

    Priority: controller-provided duration → ASR end → max danmaku timestamp.
    """
    if durationSec is not None and durationSec > 0:
        return float(durationSec)

    maxAsr = 0.0
    for seg in asrSegments:
        try:
            maxAsr = max(maxAsr, float(seg.get("end") or 0.0))
        except (TypeError, ValueError):
            continue

    maxDanmaku = 0.0
    for entry in danmakuEntries:
        try:
            maxDanmaku = max(maxDanmaku, _parseTimestamp(entry))
        except (TypeError, ValueError):
            continue

    return max(maxAsr, maxDanmaku, 0.0)


def _extractTextFeaturesImpl(
    danmakuPath: str,
    asrPath: str,
    outputPath: str | None = None,
    keywordConfig: str | None = None,
    utrThreshold: float = UTR_THRESHOLD,
    durationSec: float | None = None,
) -> dict:
    """Inner implementation — wrapped by extractFeatures with try/except."""

    # ── ModelPool bookkeeping ───────────────────────────────────
    # Every name handed to ``pool.acquire()`` below lands in
    # ``_holdings`` so the ``finally`` at the bottom can release them.
    # Without this list the heavy models (FastText, text2vec, HF
    # sentiment) stay pinned at refcount=1 forever — defeating the
    # LRU eviction the pool exists to provide.
    from src.modelPool import ModelPool
    _pool = ModelPool.instance()
    _holdings: list[str] = []

    try:
        # ── Load keyword config ─────────────────────────────────
        kwPath = keywordConfig or str(_KEYWORD_CONFIG_PATH)
        with open(kwPath, encoding="utf-8") as f:
            keywordDict = json.load(f)

        # Build flat keyword list
        keywords: list[str] = []
        for cat in keywordDict.values():
            keywords.extend(cat)
        keywords = list(set(keywords))
        log.debug(f"Loaded {len(keywords)} keywords from {len(keywordDict)} categories")

        # ── Load danmaku ────────────────────────────────────────
        log.debug("Loading danmaku...")
        danmakuEntries: list[dict] = []
        with open(danmakuPath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        danmakuEntries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        log.debug(f"Loaded {len(danmakuEntries)} danmaku messages")

        # ── Load ASR ────────────────────────────────────────────
        # ASR failure must degrade to danmaku-only, not abort text features.
        log.debug("Loading ASR...")
        asrStatus = "ok"
        asrError = None
        try:
            with open(asrPath, encoding="utf-8") as f:
                asrData = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            asrData = None
            asrStatus = "missing" if isinstance(exc, OSError) else "invalid"
            asrError = f"{type(exc).__name__}: {exc}"
            log.warning(f"ASR unavailable ({asrError}) — using danmaku only")

        # Defensive: ASR may output {"status":"failed","segments":[]} or a plain list.
        if isinstance(asrData, list):
            asrSegments = asrData
        elif isinstance(asrData, dict):
            asrSegments = asrData.get("segments", [])
            asrStatus = str(asrData.get("status") or asrStatus).lower()
            asrError = asrData.get("error") or asrError
        else:
            asrSegments = []

        if not isinstance(asrSegments, list):
            asrStatus = "invalid"
            asrError = asrError or "ASR 'segments' is not a list"
            asrSegments = []
        elif asrStatus == "ok" and not asrSegments:
            # Keep stats consistent with inspectAsrArtifact(): an otherwise
            # valid ASR file without segments is "empty", not "ok".
            asrStatus = "empty"

        asrDegraded = asrStatus not in {"ok", "empty"} or not asrSegments
        if asrStatus not in {"ok", "empty"}:
            log.warning(
                f"ASR status={asrStatus} — text features degrade to danmaku-only"
            )
        log.debug(f"Loaded {len(asrSegments)} ASR segments (status={asrStatus})")
        duration = _resolveDuration(durationSec, asrSegments, danmakuEntries)
        nSecs = int(np.ceil(duration))
        skippedOutOfRange = 0

        # ── Initialize feature arrays ───────────────────────────
        dmDensity = [0.0] * nSecs
        dmUniqueUsers: list[set] = [set() for _ in range(nSecs)]
        allText30s: list[list[str]] = [[] for _ in range(nSecs)]
        dmSentimentSum = [0.0] * nSecs
        dmSentimentCount = [0] * nSecs
        asrKeyword = [0] * nSecs
        speakerChange = [0] * nSecs
        lastSpeakerBySec: dict[int, str] = {}

        # ── Process danmaku ────────────────────────────────────
        _jieba_loaded = False

        for entry in danmakuEntries:
            ts = _parseTimestamp(entry)
            sec = int(ts)
            if sec < 0 or sec >= nSecs:
                skippedOutOfRange += 1
                continue

            content = _getContent(entry)
            userId = _getUserId(entry)

            dmDensity[sec] += 1.0
            dmUniqueUsers[sec].add(userId)

            # Text content for sliding window
            allText30s[sec].append(content)

            # Sentiment
            try:
                if not _jieba_loaded:
                    import jieba
                    _jieba_loaded = True
                words = list(jieba.cut(content))
                sentiment = _sentimentForWords(words, _pool, _holdings)
                dmSentimentSum[sec] += sentiment
                dmSentimentCount[sec] += 1
            except Exception:
                # One bad message must not derail the whole stream — log at debug
                # so the volume stays manageable but errors are still inspectable.
                log.debug("Sentiment failed for one message", exc_info=True)

        # ── Deduplicate users (privacy) ─────────────────────────
        uniqueUserCounts = [len(s) for s in dmUniqueUsers]
        del dmUniqueUsers  # free memory

        # Anti-spam: UTR per 30s window.  When enabled, calibrate the cut-off
        # from seconds that actually contain danmaku instead of trusting a
        # hard-coded threshold; sparse recordings keep the configured value.
        utrRaw = _computeUtr(allText30s)
        utrSamples = [
            utrRaw[t] for t in range(nSecs)
            if allText30s[t] and utrRaw[t] > 0
        ]
        if _cfg.utr_auto_calibrate and utrThreshold == UTR_THRESHOLD:
            effectiveUtr, utrSource = _calibrateUtrThreshold(
                utrSamples,
                configured=utrThreshold,
                percentile=_cfg.utr_calibration_percentile,
                minSamples=_cfg.utr_calibration_min_samples,
                low=_cfg.utr_calibration_min,
                high=_cfg.utr_calibration_max,
            )
        else:
            effectiveUtr, utrSource = float(utrThreshold), "configured"

        # Apply anti-spam penalty
        dmAdjustedDensity = []
        for t in range(nSecs):
            penalty = (
                min(1.0, utrRaw[t] / effectiveUtr)
                if utrRaw[t] > 0 and effectiveUtr > 0
                else 1.0
            )
            dmAdjustedDensity.append(dmDensity[t] * penalty)

        # ── Process ASR ────────────────────────────────────────
        log.debug("Processing ASR segments...")

        # Lazy-load FastText + pre-compute keyword vectors ONCE.
        # Without caching this is O(segments × keywords) get_sentence_vector
        # calls, which dominates the text pipeline on long recordings.
        _fastTextModel = None
        _kwVectors: list[np.ndarray] | None = None
        _kwNorms: np.ndarray | None = None

        def _ensureFastText():
            nonlocal _fastTextModel, _kwVectors, _kwNorms
            if _fastTextModel is not None or _kwVectors is not None:
                return
            try:
                import fasttext
            except ImportError:
                log.warning("fasttext not installed — falling back to exact keyword match")
                _fastTextModel = False  # mark as tried + failed (truthy check below)
                return
            # Resolve relative to project root so the model is found regardless
            # of the current working directory.
            modelPath = Path(__file__).resolve().parents[1] / _cfg.fasttext_model_path
            if not modelPath.exists():
                log.warning(f"FastText model {modelPath} not found — falling back to exact match")
                _fastTextModel = False
                return
            # FastText is CPU-only (≈600MB RAM) — gpu=False so it's
            # cached but never counts against the GPU LRU budget.
            _fastTextModel = _pool.acquire(
                POOL_NAME_FASTTEXT,
                lambda: fasttext.load_model(str(modelPath)),
                gpu=False,
            )
            _holdings.append(POOL_NAME_FASTTEXT)
            # Pre-embed every keyword once (kept as float32 vectors)
            vecs = [_fastTextModel.get_sentence_vector(kw) for kw in keywords]
            _kwVectors = [v.astype(np.float32) for v in vecs]
            _kwNorms = np.array(
                [np.linalg.norm(v) + 1e-10 for v in _kwVectors], dtype=np.float32
            )
            log.debug(f"FastText loaded; pre-embedded {len(_kwVectors)} keywords")

        for seg in asrSegments:
            text = seg.get("text", "")
            if not text:
                continue

            startSec = int(seg.get("start") or 0)
            endSec = int(np.ceil(seg.get("end") or 0))
            if endSec >= nSecs:
                endSec = nSecs - 1

            # Keyword matching — compute ONCE per segment, fan out to seconds
            try:
                _ensureFastText()

                if _fastTextModel:
                    # Fuzzy matching via FastText — uses pre-embedded keyword vectors
                    hit = _fuzzyMatchCached(text, keywords, _fastTextModel, _kwVectors, _kwNorms)
                else:
                    # Exact keyword match
                    hit = any(kw in text for kw in keywords)

                if hit:
                    for sec in range(startSec, endSec):
                        asrKeyword[sec] = 1
            except Exception:
                log.debug("Keyword matching failed for segment", exc_info=True)

            # Speaker change
            speaker = seg.get("speaker", "")
            if speaker:
                for sec in range(startSec, endSec):
                    if sec in lastSpeakerBySec and lastSpeakerBySec[sec] != speaker:
                        speakerChange[sec] = 1
                    lastSpeakerBySec[sec] = speaker

        # ── Build output ───────────────────────────────────────
        # Compute entropy from accumulated text
        dmEntropyRaw = _computeEntropyWindow(allText30s, nSecs)

        # Compute topic change (text2vec + KMeans across all windows)
        topicVec = _computeTopicChange(allText30s, nSecs, _pool, _holdings)

        # Danmaku engagement rates and category-specific keyword intensities.
        dmRepeatRate = _computeRepeatRate(allText30s)
        dmQuestionRate = _computeMessageRate(
            allText30s,
            lambda text: any(marker in text for marker in _QUESTION_MARKERS),
        )
        dmExclamationRate = _computeMessageRate(
            allText30s,
            lambda text: any(marker in text for marker in _EXCLAMATION_MARKERS),
        )
        dmEmojiRate = _computeMessageRate(allText30s, _textHasEmoji)
        keywordSignals = _computeKeywordCategorySignals(
            allText30s, keywordDict, asrSegments
        )

        featureDict = {
            "dmDensity": dmAdjustedDensity,
            "dmAcceleration": _secondDerivative(dmAdjustedDensity),
            "dmEntropy": dmEntropyRaw,
            "dmSentiment": [dmSentimentSum[i] / dmSentimentCount[i] if dmSentimentCount[i] > 0 else 0.0
                            for i in range(nSecs)],
            "dmUniqueUsers": uniqueUserCounts,
            "dmUtr": utrRaw,
            "dmRepeatRate": dmRepeatRate,
            "dmQuestionRate": dmQuestionRate,
            "dmExclamationRate": dmExclamationRate,
            "dmEmojiRate": dmEmojiRate,
            **keywordSignals,
            "asrKeyword": asrKeyword,
            "topicChange": topicVec,
            "speakerChange": speakerChange,
        }
        featureCount = sum(1 for values in featureDict.values() if values)
        features = {
            "sampleRate": 1,
            "duration": nSecs,
            "status": "ok" if featureCount else "empty",
            "generatorVersion": "signalText/3",
            "featureCount": featureCount,
            "stats": {
                "danmakuEntries": len(danmakuEntries),
                "skippedOutOfRange": skippedOutOfRange,
                "asrSegments": len(asrSegments),
                "asrStatus": asrStatus,
                "asrDegraded": asrDegraded,
                "asrError": str(asrError)[:300] if asrError else None,
                "durationSource": "explicit" if durationSec else "inferred",
                "sentimentMode": _cfg.sentiment_mode,
                "rateWindowRadius": _cfg.dm_rate_window_radius,
                "utr": {
                    "threshold": round(float(effectiveUtr), 4),
                    "source": utrSource,
                    "samples": len(utrSamples),
                },
                "keywordCategoryMax": {
                    name: round(max(values), 4) if values else 0.0
                    for name, values in keywordSignals.items()
                },
            },
            "features": featureDict,
        }

        if skippedOutOfRange:
            log.warning(
                f"{skippedOutOfRange}/{len(danmakuEntries)} danmaku entries "
                f"outside 0..{nSecs}s were skipped"
            )

        if outputPath:
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False)
            log.info(f"Text features saved: {outputPath}")

        log.info(f"Text features: {nSecs}s, {featureCount} non-empty features")
        return features

    finally:
        # Release every name we acquired, even if extraction raised.
        # Releases drop refcount to 0; the model stays cached until
        # ModelPool's LRU evicts it.
        for name in _holdings:
            _pool.release(name)


# ── Internal helpers ──────────────────────────────────────────────

def _parseTimestamp(entry: dict) -> float:
    """Extract timestamp from a danmaku JSONL entry.

    Priority: recvTimeSec (wall-clock offset from controller) >
    timestamp / _time (legacy) > defaults to 0.
    """
    for key in ("recvTimeSec", "timestamp", "_time"):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return 0.0


def _getContent(entry: dict) -> str:
    """Extract message content from a danmaku entry."""
    return entry.get("content", entry.get("text", ""))


def _getUserId(entry: dict) -> str:
    """Get user ID, SHA256-hashed for privacy."""
    raw = entry.get("userId", entry.get("user_id", str(entry.get("user_id", ""))))
    return hashlib.sha256(str(raw).encode()).hexdigest()[:16]


# Module-level caches for sentiment models.
# ``_sentimentAvailable`` is the tri-state load probe (None=untried,
# True=loaded, False=failed) — keeping it module-level means we only
# pay the import + load attempt once across the whole process.
# ``_sentimentPipeline`` mirrors what ``ModelPool`` actually owns;
# the pool keeps the pipeline alive (we hold a ref via ``acquire``)
# while this attribute lets ``_fastSentiment`` skip a pool lookup
# on every call.
_sentimentPipeline = None
_sentimentAvailable = None  # None=untried, True=loaded, False=failed
_sentimentLock = threading.Lock()


def _fastSentiment(words: list[str], pool=None, holdings: list[str] | None = None) -> float:
    """Sentiment analysis — tries HF pipeline once, falls back to rules.

    Args:
        words: jieba-tokenised text.
        pool: optional ModelPool to load the HF pipeline through.  When
            provided (every call from ``extractFeatures``), the pipeline
            counts against the GPU LRU budget and is releasable.  When
            ``None`` (legacy direct callers), the pipeline is loaded
            inline and stays resident — kept for backward compatibility
            of the unit tests that patch this helper directly.
        holdings: list to append the pool name to.  ``extractFeatures``
            walks this in its ``finally`` to release acquired models.
    """
    global _sentimentPipeline, _sentimentAvailable
    text = "".join(words)
    if not text.strip():
        return 0.0

    # Try to load HF pipeline exactly once, protected by lock
    # so concurrent ThreadPoolExecutor workers don't race to init.
    with _sentimentLock:
        if _sentimentAvailable is None:
            try:
                from transformers import pipeline

                def _loadPipeline():
                    return pipeline(
                        "sentiment-analysis",
                        model=_cfg.sentiment_model,
                    )

                if pool is not None:
                    # GPU=True: this is a 400MB transformer that benefits from
                    # CUDA when available — it's the canonical LRU candidate.
                    _sentimentPipeline = pool.acquire(
                        POOL_NAME_HF_SENTIMENT, _loadPipeline, gpu=True,
                    )
                    if holdings is not None:
                        holdings.append(POOL_NAME_HF_SENTIMENT)
                else:
                    _sentimentPipeline = _loadPipeline()
                _sentimentAvailable = True
            except Exception:
                log.debug("HF sentiment pipeline unavailable — using rule fallback", exc_info=True)
                _sentimentAvailable = False  # Mark as tried (even if failed)

    if _sentimentAvailable and _sentimentPipeline:
        try:
            result = _sentimentPipeline(text[:_cfg.sentiment_max_len])[0]
            score = result["score"]
            return score if result["label"].upper() == "POSITIVE" else -score
        except Exception:
            # Per-call failure → quietly degrade to rules; the pipeline itself
            # is still considered available so we keep trying subsequent texts.
            log.debug("HF sentiment inference failed for one text", exc_info=True)

    return _ruleSentiment(words)


def _ruleSentiment(words: list[str]) -> float:
    """Cheap lexicon sentiment used before any transformer inference."""
    if not words:
        return 0.0
    pos = sum(1 for word in words if word in _RULE_POSITIVE)
    neg = sum(1 for word in words if word in _RULE_NEGATIVE)
    if pos > neg:
        return 0.5
    if neg > pos:
        return -0.5
    return 0.0


def _sentimentForWords(
    words: list[str],
    pool=None,
    holdings: list[str] | None = None,
) -> float:
    """Apply the configured sentiment strategy.

    ``rules`` avoids per-message transformer calls entirely; ``hybrid`` only
    falls back to the model when the lexicon is neutral; ``hf`` preserves the
    legacy per-message behaviour.
    """
    ruleScore = _ruleSentiment(words)
    mode = str(_cfg.sentiment_mode).lower()
    if mode == "rules":
        return ruleScore
    if mode == "hybrid" and ruleScore != 0.0:
        return ruleScore
    return _fastSentiment(words, pool, holdings)


def _normalizeDanmaku(text: str) -> str:
    """Normalise a message for duplicate detection."""
    return re.sub(r"\s+", "", text.strip().lower())


def _textHasEmoji(text: str) -> bool:
    """Return True when a message contains a Unicode emoji or bracket emote."""
    return bool(_EMOJI_PATTERN.search(text))


def _prefixSums(values: list[float]) -> list[float]:
    """Inclusive prefix sums with a leading zero."""
    prefix = [0.0]
    total = 0.0
    for value in values:
        total += float(value)
        prefix.append(total)
    return prefix


def _computeMessageRate(
    allText30s: list[list[str]],
    predicate,
    radius: int | None = None,
) -> list[float]:
    """Sliding-window fraction of messages satisfying ``predicate``."""
    nSecs = len(allText30s)
    if nSecs == 0:
        return []
    windowRadius = (
        _cfg.dm_rate_window_radius if radius is None else max(0, int(radius))
    )
    hits = [0.0] * nSecs
    totals = [0.0] * nSecs
    for sec, texts in enumerate(allText30s):
        totals[sec] = float(len(texts))
        hits[sec] = float(sum(1 for text in texts if predicate(text)))

    hitPrefix = _prefixSums(hits)
    totalPrefix = _prefixSums(totals)
    result = [0.0] * nSecs
    for sec in range(nSecs):
        lo = max(0, sec - windowRadius)
        hi = min(nSecs, sec + windowRadius + 1)
        total = totalPrefix[hi] - totalPrefix[lo]
        if total > 0:
            result[sec] = (hitPrefix[hi] - hitPrefix[lo]) / total
    return result


def _computeRepeatRate(
    allText30s: list[list[str]],
    radius: int | None = None,
) -> list[float]:
    """Sliding-window fraction of duplicate (normalised) messages."""
    nSecs = len(allText30s)
    if nSecs == 0:
        return []
    windowRadius = (
        _cfg.dm_rate_window_radius if radius is None else max(0, int(radius))
    )
    result = [0.0] * nSecs
    for sec in range(nSecs):
        lo = max(0, sec - windowRadius)
        hi = min(nSecs, sec + windowRadius + 1)
        normalised = [
            _normalizeDanmaku(text)
            for texts in allText30s[lo:hi]
            for text in texts
            if text.strip()
        ]
        if normalised:
            result[sec] = (len(normalised) - len(set(normalised))) / len(normalised)
    return result


def _canonicalKeywordCategory(name: object) -> str | None:
    """Map config spellings (highEnergy / high_energy) to a canonical key."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    return _CATEGORY_ALIASES.get(key)


def _computeKeywordCategorySignals(
    allText30s: list[list[str]],
    keywordDict: dict,
    asrSegments: list[dict] | None = None,
    radius: int | None = None,
) -> dict[str, list[float]]:
    """Build per-category keyword intensity signals.

    Danmaku contribute a smoothed hit rate (60% weight); ASR segments that
    contain a category phrase add a binary event (40% weight).  The result
    stays in 0..1 so it can be compared across recordings.
    """
    nSecs = len(allText30s)
    if nSecs == 0:
        return {name: [] for name in KEYWORD_CATEGORY_FEATURES.values()}

    windowRadius = (
        _cfg.dm_rate_window_radius if radius is None else max(0, int(radius))
    )
    categoryKeywords: dict[str, list[str]] = {}
    for rawName, values in (keywordDict or {}).items():
        category = _canonicalKeywordCategory(rawName)
        if category is None or not isinstance(values, list):
            continue
        keywords = [
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        ]
        if keywords:
            categoryKeywords[category] = keywords

    dmHits = {category: [0.0] * nSecs for category in categoryKeywords}
    totals = [float(len(texts)) for texts in allText30s]
    for sec, texts in enumerate(allText30s):
        for text in texts:
            lower = text.lower()
            for category, keywords in categoryKeywords.items():
                if any(keyword in lower for keyword in keywords):
                    dmHits[category][sec] += 1.0

    asrHits = {category: [0.0] * nSecs for category in categoryKeywords}
    for segment in asrSegments or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").lower()
        if not text:
            continue
        matched = [
            category for category, keywords in categoryKeywords.items()
            if any(keyword in text for keyword in keywords)
        ]
        if not matched:
            continue
        try:
            start = int(segment.get("start") or 0)
            end = int(np.ceil(float(segment.get("end") or 0)))
        except (TypeError, ValueError):
            continue
        start = max(0, min(start, nSecs))
        end = max(start, min(end, nSecs))
        for sec in range(start, end):
            for category in matched:
                asrHits[category][sec] = 1.0

    totalPrefix = _prefixSums(totals)
    signals: dict[str, list[float]] = {}
    for category, featureName in KEYWORD_CATEGORY_FEATURES.items():
        hits = dmHits.get(category)
        if hits is None:
            signals[featureName] = [0.0] * nSecs
            continue
        hitPrefix = _prefixSums(hits)
        asr = asrHits.get(category, [0.0] * nSecs)
        values = [0.0] * nSecs
        for sec in range(nSecs):
            lo = max(0, sec - windowRadius)
            hi = min(nSecs, sec + windowRadius + 1)
            total = totalPrefix[hi] - totalPrefix[lo]
            dmRate = (hitPrefix[hi] - hitPrefix[lo]) / total if total > 0 else 0.0
            values[sec] = min(1.0, 0.6 * dmRate + 0.4 * asr[sec])
        signals[featureName] = values
    return signals


def _calibrateUtrThreshold(
    utrValues: list[float],
    configured: float,
    percentile: float | None = None,
    minSamples: int | None = None,
    low: float | None = None,
    high: float | None = None,
) -> tuple[float, str]:
    """Calibrate the anti-spam threshold from observed UTR values.

    Returns ``(threshold, source)`` where source is ``calibrated`` or
    ``configured``.  Sparse recordings fall back to the configured value.
    """
    percentile = (
        _cfg.utr_calibration_percentile if percentile is None else float(percentile)
    )
    minSamples = (
        _cfg.utr_calibration_min_samples if minSamples is None else int(minSamples)
    )
    low = _cfg.utr_calibration_min if low is None else float(low)
    high = _cfg.utr_calibration_max if high is None else float(high)

    values = [
        float(value) for value in utrValues
        if np.isfinite(value) and float(value) > 0
    ]
    if len(values) < max(1, minSamples):
        return float(configured), "configured"
    quantile = float(np.percentile(values, percentile))
    return min(high, max(low, quantile)), "calibrated"


def _computeUtr(allText30s: list[list[str]]) -> list[float]:
    """Compute Unique Token Ratio per 30s sliding window.

    Uses a sliding Counter for O(n * avg_tokens_per_step) instead of
    the naive O(n * 60 * avg_tokens).  At each step we add the incoming
    second's tokens and remove the outgoing second's tokens, then
    compute UTR = unique / total from the Counter.
    """
    from collections import Counter

    nSecs = len(allText30s)
    utr = [0.0] * nSecs
    windowRadius = _cfg.utr_window_radius

    # Pre-flatten: compute token list per second (materialize once)
    tokensPerSec: list[list[str]] = [
        " ".join(allText30s[t]).split() if allText30s[t] else []
        for t in range(nSecs)
    ]

    counter: Counter = Counter()
    # Initialise window for t=0: add seconds [0, windowRadius)
    for nt in range(min(windowRadius, nSecs)):
        for token in tokensPerSec[nt]:
            counter[token] += 1

    for t in range(nSecs):
        # Add right edge expanding into window
        ntAdd = t + windowRadius
        if ntAdd < nSecs:
            for token in tokensPerSec[ntAdd]:
                counter[token] += 1

        # Remove left edge leaving window
        ntRemove = t - windowRadius - 1
        if ntRemove >= 0:
            for token in tokensPerSec[ntRemove]:
                counter[token] -= 1
                if counter[token] <= 0:
                    del counter[token]

        total = sum(counter.values())
        if total > 0:
            utr[t] = len(counter) / total
        else:
            utr[t] = 1.0

    return utr


def _secondDerivative(arr: list[float]) -> list[float]:
    """Compute acceleration (second derivative) of a signal."""
    n = len(arr)
    result = [0.0] * n
    for i in range(1, n - 1):
        result[i] = arr[i + 1] - 2 * arr[i] + arr[i - 1]
    return result


def _fuzzyMatch(text: str, keywords: list[str], model) -> bool:
    """FastText fuzzy keyword matching (un-cached fallback path).

    Prefer ``_fuzzyMatchCached`` when scanning many segments — this version
    re-embeds every keyword on each call and is only kept for callers that
    don't pre-compute keyword vectors.
    """
    # Exact match first (fast path)
    for kw in keywords:
        if kw in text:
            return True
    # FastText similarity
    if model is not None:
        try:
            textVec = model.get_sentence_vector(text)
            textNorm = np.linalg.norm(textVec) + 1e-10
            for kw in keywords:
                kwVec = model.get_sentence_vector(kw)
                sim = float(np.dot(textVec, kwVec) / (textNorm * (np.linalg.norm(kwVec) + 1e-10)))
                if sim > SIMILARITY_THRESHOLD:
                    return True
        except Exception:
            pass
    return False


def _fuzzyMatchCached(
    text: str,
    keywords: list[str],
    model,
    kwVectors: list[np.ndarray] | None,
    kwNorms: np.ndarray | None,
) -> bool:
    """FastText fuzzy match using pre-embedded keyword vectors.

    Computes the text vector once, then dot-products it against the
    cached keyword matrix in a single vectorised step.  Drops the cost
    from O(segments × keywords) embeddings to O(segments) embeddings.
    """
    # Exact match first (fast path) — same as before, no embedding cost
    for kw in keywords:
        if kw in text:
            return True
    if model is None or kwVectors is None or kwNorms is None:
        return False
    try:
        textVec = model.get_sentence_vector(text).astype(np.float32)
        textNorm = float(np.linalg.norm(textVec)) + 1e-10
        # Stack once per call; cheap relative to embedding the text
        kwMatrix = np.stack(kwVectors)            # (K, D)
        sims = (kwMatrix @ textVec) / (kwNorms * textNorm)
        return bool(np.any(sims > SIMILARITY_THRESHOLD))
    except Exception:
        return False


def _computeEntropyWindow(allText30s: list[list[str]], nSecs: int) -> list[float]:
    """Compute Shannon entropy of text per 10s sliding window.

    Uses Counter + Shannon entropy H = -Σ p(w)·log₂(p(w)) on top-50 tokens
    from up to 200 messages per window.  High entropy = diverse discussion;
    low entropy = repetitive/spam.

    Note: design doc originally specified HDBSCAN for topic clustering;
    this implementation uses simpler Counter-based entropy which is
    faster and more practical for real-time danmaku streams.
    """
    result = [0.0] * nSecs
    radius = _cfg.entropy_window_radius
    for t in range(nSecs):
        texts = []
        for dt in range(-radius, radius + 1):
            nt = t + dt
            if 0 <= nt < nSecs:
                texts.extend(allText30s[nt][:_cfg.entropy_msg_cap])
        if not texts:
            continue
        allWords = " ".join(texts).split()
        wordCounts = Counter(allWords)
        topWords = [w for w, _ in wordCounts.most_common(_cfg.entropy_top_words)]
        total = sum(wordCounts.values())
        if total == 0:
            continue
        entropy = 0.0
        for w in topWords:
            p = wordCounts[w] / total
            if p > 0:
                entropy -= p * np.log2(p)
        result[t] = entropy
    return result


def _computeTopicChange(
    allText30s: list[list[str]],
    nSecs: int,
    pool=None,
    holdings: list[str] | None = None,
) -> list[int]:
    """Detect topic shifts using text2vec embeddings + KMeans clustering.

    Collects text from 30s windows, embeds each window via text2vec,
    clusters embeddings with KMeans (n_clusters = min(5, n_windows // 2)),
    and marks second-level transitions where the cluster label changes.

    Falls back to all-zero on ImportError (missing deps).

    When called from ``extractFeatures`` the text2vec model is acquired
    via ``pool`` so it shares the LRU budget; the pool name is appended
    to ``holdings`` so the caller's ``finally`` releases it.  Direct
    callers can pass ``pool=None`` and the model loads inline (legacy).
    """
    result = [0] * nSecs
    try:
        from text2vec import SentenceModel
        from sklearn.cluster import KMeans

        # ── Collect window texts and embed ─────────────────────
        windowSize = 30
        nWindows = max(1, (nSecs + windowSize - 1) // windowSize)
        windowTexts: list[str] = []

        for wi in range(nWindows):
            tStart = wi * windowSize
            texts: list[str] = []
            for dt in range(windowSize):
                nt = tStart + dt
                if 0 <= nt < nSecs:
                    texts.extend(allText30s[nt][:50])
            windowTexts.append(" ".join(texts) if texts else "")

        # Skip windows with no content
        nonEmptyIndices = [i for i, t in enumerate(windowTexts) if t.strip()]
        if len(nonEmptyIndices) < 2:
            return result

        nonEmptyTexts = [windowTexts[i] for i in nonEmptyIndices]

        # ── Embed ──────────────────────────────────────────────
        if pool is not None:
            # text2vec is a ~400MB transformer — runs on GPU when present.
            model = pool.acquire(
                POOL_NAME_TEXT2VEC,
                lambda: SentenceModel("shibing624/text2vec-base-chinese"),
                gpu=True,
            )
            if holdings is not None:
                holdings.append(POOL_NAME_TEXT2VEC)
        else:
            model = SentenceModel("shibing624/text2vec-base-chinese")
        embeddings = model.encode(nonEmptyTexts)
        if embeddings is None or len(embeddings) == 0:
            return result

        # ── Cluster ────────────────────────────────────────────
        nClusters = max(2, min(5, len(nonEmptyIndices) // 2))
        kmeans = KMeans(n_clusters=nClusters, n_init=10, random_state=42)
        clusterLabels = kmeans.fit_predict(embeddings)

        # ── Map back to per-second labels ──────────────────────
        labels = [0] * nSecs
        for i, ci in enumerate(nonEmptyIndices):
            tStart = ci * windowSize
            label = int(clusterLabels[i])
            for dt in range(windowSize):
                nt = tStart + dt
                if 0 <= nt < nSecs:
                    labels[nt] = label

        # ── Mark transitions ───────────────────────────────────
        for t in range(1, nSecs):
            if labels[t] != labels[t - 1]:
                result[t] = 1

        log.debug(f"TopicChange: {sum(result)} transitions, {nClusters} clusters")

    except ImportError:
        log.debug("text2vec/sklearn not installed — topicChange stays zero")
    except Exception:
        log.debug("topicChange computation failed", exc_info=True)
    return result
