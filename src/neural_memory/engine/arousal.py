"""Arousal detection — multilingual emotional intensity scoring.

Complements EmotionStep (NLP-based sentiment → valence/emotion tags) with
a fast regex scan that detects emotional *intensity* regardless of polarity.
High-arousal memories (production incidents, breakthroughs, security issues)
get encoding priority boosts and compression resistance.

Supports multiple languages via keyword patterns (English, Vietnamese) and
falls back to language-agnostic heuristics (punctuation density, caps ratio)
for unsupported languages.

Neuroscience basis: the amygdala tags memories with emotional significance,
making them easier to recall (flashbulb effect) and more resistant to
forgetting. Arousal is orthogonal to valence — a "breakthrough fix" and
a "critical outage" are both high-arousal despite opposite valence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from neural_memory.engine.pipeline import PipelineContext
from neural_memory.extraction.parser import detect_language

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Language-specific pattern registries ────────────────────
# To add a new language: add a key to _PATTERNS with "positive", "negative",
# and "high_arousal" regex lists. detect_language() in extraction/parser.py
# must also be updated to detect the new language.
# Unsupported languages fall back to language-agnostic heuristics.

_PATTERNS: dict[str, dict[str, list[str]]] = {
    "en": {
        "positive": [
            r"\bsolved\b",
            r"\bfixed\b",
            r"\bworks?\s+(?:well|great|perfectly)\b",
            r"\bfinally\b",
            r"\bbreakthrough\b",
            r"\bsuccess(?:ful(?:ly)?)?\b",
            r"\belegant\b",
            r"\boptimal\b",
            r"\bperfect(?:ly)?\b",
        ],
        "negative": [
            r"\bbug\b",
            r"\bcrash(?:ed|es|ing)?\b",
            r"\bfail(?:ed|s|ure)?\b",
            r"\bbroken\b",
            r"\bfrustr",
            r"\bwasted?\b",
            r"\bpainful\b",
            r"\bdanger(?:ous)?\b",
            r"\bcritical\b",
            r"\bsecurity\s+(?:vuln|issue|hole|breach)\b",
        ],
        "high_arousal": [
            r"\b(?:CRITICAL|URGENT|BREAKING|IMPORTANT)\b",
            r"!{2,}",
            r"\bdata\s+loss\b",
            r"\bproduction\s+(?:down|outage|incident)\b",
            r"\bsecurity\s+(?:breach|vulnerability|exploit)\b",
            r"\brollback\b",
            r"\bhotfix\b",
        ],
    },
    "vi": {
        "positive": [
            r"(?:^|\s)đã sửa(?:\s|$)",
            r"(?:^|\s)hoạt động tốt(?:\s|$)",
            r"(?:^|\s)thành công(?:\s|$)",
            r"(?:^|\s)tuyệt vời(?:\s|$)",
            r"(?:^|\s)đột phá(?:\s|$)",
            r"(?:^|\s)hoàn hảo(?:\s|$)",
            r"(?:^|\s)xuất sắc(?:\s|$)",
            r"(?:^|\s)tối ưu(?:\s|$)",
            r"(?:^|\s)cuối cùng(?:\s|$)",
        ],
        "negative": [
            r"(?:^|\s)lỗi(?:\s|$)",
            r"\bcrash(?:ed|es|ing)?\b",  # borrowed English in Vietnamese context
            r"(?:^|\s)thất bại(?:\s|$)",
            r"(?:^|\s)hỏng(?:\s|$)",
            r"(?:^|\s)nguy hiểm(?:\s|$)",
            r"(?:^|\s)lỗ hổng(?:\s|$)",
            r"(?:^|\s)sự cố(?:\s|$)",
            r"(?:^|\s)mất dữ liệu(?:\s|$)",
            r"(?:^|\s)nghiêm trọng(?:\s|$)",
            r"(?:^|\s)bảo mật(?:\s|$)",
        ],
        "high_arousal": [
            r"(?:^|\s)(?:KHẨN CẤP|SỰ CỐ|MẤT DỮ LIỆU|NGHIÊM TRỌNG)(?:\s|$)",
            r"!{2,}",
            r"(?:^|\s)production\s+(?:down|sập)(?:\s|$)",
            r"(?:^|\s)rollback(?:\s|$)",
            r"(?:^|\s)hotfix(?:\s|$)",
            r"(?:^|\s)cấp bách(?:\s|$)",
            r"(?:^|\s)khẩn(?:\s|$)",
        ],
    },
}


@lru_cache(maxsize=4)
def _compiled_patterns(lang: str, category: str) -> list[re.Pattern[str]]:
    """Compile and cache patterns for a language + category."""
    raw = _PATTERNS.get(lang, {}).get(category, [])
    return [re.compile(p, re.IGNORECASE) for p in raw]


def _language_agnostic_arousal(content: str) -> float:
    """Compute arousal using language-agnostic heuristics.

    Detects emotional intensity via:
    - Punctuation density (!! ?? !!!)
    - ALL-CAPS ratio
    - Exclamation/question mark clusters

    Returns a score in [0.0, 1.0].
    """
    if not content or len(content) < 10:
        return 0.0

    signals = 0.0

    # Exclamation clusters (!! or !!!)
    excl_clusters = len(re.findall(r"!{2,}", content))
    signals += min(excl_clusters * 0.3, 0.6)

    # Question mark clusters (?? or ???)
    quest_clusters = len(re.findall(r"\?{2,}", content))
    signals += min(quest_clusters * 0.2, 0.4)

    # ALL-CAPS words ratio (3+ char words)
    words = re.findall(r"\b[A-Z\u00C0-\u024F]{3,}\b", content)
    all_words = re.findall(r"\b\w{3,}\b", content)
    if all_words:
        caps_ratio = len(words) / len(all_words)
        if caps_ratio > 0.15:
            signals += min(caps_ratio * 2, 0.6)

    return min(1.0, signals)


def compute_arousal(content: str) -> float:
    """Compute emotional arousal from content.

    Returns a score in [0.0, 1.0] representing emotional intensity
    regardless of polarity. High arousal = emotionally charged content
    that the brain should encode more strongly.

    Uses keyword patterns for supported languages (en, vi) and falls
    back to language-agnostic heuristics for other languages.

    Args:
        content: Text content to analyze.

    Returns:
        Arousal score in [0.0, 1.0].
    """
    if not content:
        return 0.0

    lang = detect_language(content)

    # Keyword-based scoring for supported languages
    pos = sum(1 for p in _compiled_patterns(lang, "positive") if p.search(content))
    neg = sum(1 for p in _compiled_patterns(lang, "negative") if p.search(content))
    high = sum(1 for p in _compiled_patterns(lang, "high_arousal") if p.search(content))

    # Keyword arousal
    total = pos + neg + high * 2
    keyword_arousal = min(1.0, total / 5.0)

    # Language-agnostic arousal (always computed as supplement)
    agnostic_arousal = _language_agnostic_arousal(content)

    # Take the max — keywords are precise, agnostic catches edge cases
    arousal = max(keyword_arousal, agnostic_arousal)

    return arousal


@dataclass
class ArousalStep:
    """Pipeline step that computes arousal and stores it in metadata.

    Complements EmotionStep: EmotionStep handles valence + emotion tags,
    ArousalStep handles intensity detection for compression resistance
    and retrieval boosting.
    """

    @property
    def name(self) -> str:
        return "arousal"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if not getattr(config, "arousal_enabled", True):
            return ctx

        arousal = compute_arousal(ctx.content)
        if arousal > 0.0:
            ctx.effective_metadata["_arousal"] = round(arousal, 3)
            logger.debug("Arousal detected: %.3f for content: %.50s...", arousal, ctx.content)

        return ctx
