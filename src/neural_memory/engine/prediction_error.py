"""Prediction error encoding — surprise signal boosts memory priority.

When new information contradicts or differs substantially from existing
knowledge, the brain treats it as more important (prediction error signal).
This module detects:
  1. Semantic reversals (opposite claims about same entity)
  2. Within-topic novelty (SimHash distance from existing memories)
  3. Completely novel topics (no existing memories match)

Neuroscience basis: dopaminergic prediction error — unexpected stimuli
trigger stronger encoding via VTA/substantia nigra signaling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.engine.pipeline import PipelineContext
from neural_memory.extraction.parser import detect_language
from neural_memory.utils.simhash import hamming_distance, simhash

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.neuron import Neuron
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Multilingual reversal detection patterns ──────────────
# To add a new language: add entries to _NEGATION_PATTERNS,
# _NEGATION_STOPWORDS, and _OPPOSITE_PAIRS_REGISTRY.
# detect_language() in extraction/parser.py handles language detection.

_NEGATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": re.compile(
        r"\b(not|no|never|don'?t|doesn'?t|didn'?t|can'?t|cannot|won'?t|isn'?t|aren'?t|wasn'?t)\b",
        re.IGNORECASE,
    ),
    "vi": re.compile(
        r"(?:^|\s)(không|chưa|chẳng|đừng|không thể|chả|không bao giờ)(?:\s|$)",
        re.IGNORECASE,
    ),
}

_NEGATION_STOPWORDS: dict[str, set[str]] = {
    "en": {"not", "don", "doesn", "didn", "can", "cannot", "won", "isn", "aren", "wasn"},
    "vi": {"không", "chưa", "chẳng", "đừng", "chả", "thể", "bao", "giờ"},
}

_OPPOSITE_PAIRS_REGISTRY: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("fast", "slow"),
        ("good", "bad"),
        ("works", "broken"),
        ("stable", "unstable"),
        ("safe", "unsafe"),
        ("easy", "hard"),
        ("simple", "complex"),
        ("reliable", "unreliable"),
        ("efficient", "inefficient"),
        ("success", "failure"),
    ],
    "vi": [
        ("nhanh", "chậm"),
        ("tốt", "xấu"),
        ("ổn định", "không ổn định"),
        ("an toàn", "không an toàn"),
        ("dễ", "khó"),
        ("đơn giản", "phức tạp"),
        ("thành công", "thất bại"),
        ("hoạt động", "hỏng"),
        ("hiệu quả", "không hiệu quả"),
        ("tin cậy", "không tin cậy"),
    ],
}


def _detects_reversal(content_a: str, content_b: str) -> bool:
    """Detect if two texts express opposite claims about the same topic.

    Supports English and Vietnamese via multilingual pattern registries.
    Checks for:
      - Negation flip: "X works" vs "X doesn't work" / "X hoạt động" vs "X không hoạt động"
      - Opposite adjectives: "X is fast" vs "X is slow" / "X nhanh" vs "X chậm"
    """
    a_lower = content_a.lower()
    b_lower = content_b.lower()

    # Detect language from combined content
    lang = detect_language(content_a + " " + content_b)

    # Get patterns for detected language (fall back to English)
    neg_pattern = _NEGATION_PATTERNS.get(lang, _NEGATION_PATTERNS["en"])
    stopwords = _NEGATION_STOPWORDS.get(lang, _NEGATION_STOPWORDS["en"])
    opposite_pairs = _OPPOSITE_PAIRS_REGISTRY.get(lang, _OPPOSITE_PAIRS_REGISTRY["en"])

    # Negation flip: one has negation, the other doesn't,
    # but they share significant word overlap
    a_has_neg = bool(neg_pattern.search(a_lower))
    b_has_neg = bool(neg_pattern.search(b_lower))

    if a_has_neg != b_has_neg:
        # Check word overlap (excluding stop words / negations)
        a_words = set(re.findall(r"\b\w{2,}\b", a_lower)) - stopwords
        b_words = set(re.findall(r"\b\w{2,}\b", b_lower)) - stopwords
        if len(a_words & b_words) >= 2:
            return True

    # Opposite adjective pairs
    for pos, neg in opposite_pairs:
        if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
            return True

    # Cross-language: also try English patterns if content is Vietnamese
    # (Vietnamese devs often mix English terms like "crash", "bug", "works")
    if lang == "vi":
        en_neg = _NEGATION_PATTERNS["en"]
        a_has_en_neg = bool(en_neg.search(a_lower))
        b_has_en_neg = bool(en_neg.search(b_lower))
        if a_has_en_neg != b_has_en_neg:
            a_words = set(re.findall(r"\b\w{3,}\b", a_lower)) - _NEGATION_STOPWORDS["en"]
            b_words = set(re.findall(r"\b\w{3,}\b", b_lower)) - _NEGATION_STOPWORDS["en"]
            if len(a_words & b_words) >= 2:
                return True

        for pos, neg in _OPPOSITE_PAIRS_REGISTRY["en"]:
            if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                return True

    return False


async def compute_surprise_bonus(
    content: str,
    tags: set[str],
    content_hash: int,
    storage: NeuralStorage,
    config: BrainConfig,
) -> float:
    """Compute prediction error surprise bonus for new content.

    Returns:
        Surprise bonus in [0.0, 3.0]:
        - 0.0: redundant with existing memories
        - 1.0-1.5: novel topic (no existing memories match)
        - 2.0+: contradicts existing knowledge
    """
    if not tags:
        return 1.0  # No tags = moderate novelty (can't compare)

    # Find existing neurons with overlapping content
    # Use first few tags as search terms
    tag_list = sorted(tags)[:5]
    candidates: list[Neuron] = []
    for tag in tag_list:
        found = await storage.find_neurons(content_contains=tag, limit=10)
        candidates.extend(found)

    # Deduplicate by neuron id
    seen_ids: set[str] = set()
    unique: list[Neuron] = []
    for n in candidates:
        if n.id not in seen_ids:
            seen_ids.add(n.id)
            unique.append(n)

    if not unique:
        # Completely novel topic — moderate surprise
        logger.debug("Prediction error: novel topic (no matching neurons), surprise=1.5")
        return 1.5

    # Check for contradictions and compute novelty
    max_reversal_score = 0.0
    min_hamming = 64  # worst case

    for neuron in unique[:20]:  # cap to avoid perf issues
        # Contradiction check
        if _detects_reversal(content, neuron.content):
            max_reversal_score = 2.5
            logger.debug(
                "Prediction error: reversal detected vs neuron %s",
                neuron.id[:12],
            )
            break

        # SimHash novelty
        if neuron.content_hash and content_hash:
            dist = hamming_distance(content_hash, neuron.content_hash)
            min_hamming = min(min_hamming, dist)

    if max_reversal_score > 0:
        return min(max_reversal_score, 3.0)

    # Convert hamming distance to surprise:
    # dist < 5: very similar (near-duplicate) → 0.0
    # dist 5-15: moderate novelty → 0.5-1.0
    # dist 15-30: substantial novelty → 1.0-2.0
    # dist > 30: very different → 2.0
    if min_hamming < 5:
        surprise = 0.0
    elif min_hamming < 15:
        surprise = (min_hamming - 5) / 10.0  # 0.0 to 1.0
    elif min_hamming < 30:
        surprise = 1.0 + (min_hamming - 15) / 15.0  # 1.0 to 2.0
    else:
        surprise = 2.0

    logger.debug(
        "Prediction error: min_hamming=%d, surprise=%.2f",
        min_hamming,
        surprise,
    )
    return surprise


@dataclass
class PredictionErrorStep:
    """Pipeline step that adjusts priority based on prediction error.

    Must run AFTER AutoTagStep (needs tags) and AFTER DedupCheckStep
    (skip exact dupes). Adds surprise_bonus to auto_priority.
    """

    @property
    def name(self) -> str:
        return "prediction_error"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if not getattr(config, "prediction_error_enabled", True):
            return ctx

        # Skip if dedup found exact match (already handled)
        if ctx.effective_metadata.get("_dedup_reused_anchor") is not None:
            return ctx

        bonus = await compute_surprise_bonus(
            content=ctx.content,
            tags=ctx.merged_tags or ctx.auto_tags,
            content_hash=ctx.content_hash or simhash(ctx.content),
            storage=storage,
            config=config,
        )

        if bonus > 0:
            base_priority = ctx.effective_metadata.get("auto_priority", 5)
            raw = float(base_priority) if isinstance(base_priority, (int, float)) else 5.0
            new_priority = max(1, min(10, int(raw + bonus)))
            ctx.effective_metadata["auto_priority"] = new_priority
            ctx.effective_metadata["_surprise_bonus"] = round(bonus, 2)

        return ctx
