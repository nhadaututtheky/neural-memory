"""Arousal detection — lightweight regex-based emotional intensity scoring.

Complements EmotionStep (NLP-based sentiment → valence/emotion tags) with
a fast regex scan that detects emotional *intensity* regardless of polarity.
High-arousal memories (production incidents, breakthroughs, security issues)
get encoding priority boosts and compression resistance.

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

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _positive_patterns() -> list[re.Pattern[str]]:
    """Compile positive intensity patterns (cached)."""
    return [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\bsolved\b",
            r"\bfixed\b",
            r"\bworks?\s+(?:well|great|perfectly)\b",
            r"\bfinally\b",
            r"\bbreakthrough\b",
            r"\bsuccess(?:ful(?:ly)?)?\b",
            r"\belegant\b",
            r"\boptimal\b",
            r"\bperfect(?:ly)?\b",
        ]
    ]


@lru_cache(maxsize=1)
def _negative_patterns() -> list[re.Pattern[str]]:
    """Compile negative intensity patterns (cached)."""
    return [
        re.compile(p, re.IGNORECASE)
        for p in [
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
        ]
    ]


@lru_cache(maxsize=1)
def _high_arousal_patterns() -> list[re.Pattern[str]]:
    """Compile high-arousal amplifier patterns (cached)."""
    return [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(?:CRITICAL|URGENT|BREAKING|IMPORTANT)\b",
            r"!{2,}",
            r"\bdata\s+loss\b",
            r"\bproduction\s+(?:down|outage|incident)\b",
            r"\bsecurity\s+(?:breach|vulnerability|exploit)\b",
            r"\brollback\b",
            r"\bhotfix\b",
        ]
    ]


def compute_arousal(content: str) -> float:
    """Compute emotional arousal from content.

    Returns a score in [0.0, 1.0] representing emotional intensity
    regardless of polarity. High arousal = emotionally charged content
    that the brain should encode more strongly.

    Args:
        content: Text content to analyze.

    Returns:
        Arousal score in [0.0, 1.0].
    """
    if not content:
        return 0.0

    pos = sum(1 for p in _positive_patterns() if p.search(content))
    neg = sum(1 for p in _negative_patterns() if p.search(content))
    high = sum(1 for p in _high_arousal_patterns() if p.search(content))

    # Arousal = total intensity signals, normalized
    # High-arousal amplifiers count double
    total = pos + neg + high * 2
    arousal = min(1.0, total / 5.0)

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
