"""Fidelity layers engine — extractive essence generation and scoring.

Memories fade through fidelity levels (FULL -> SUMMARY -> ESSENCE -> GHOST)
based on decay score and context budget pressure. This module provides:
- Extractive essence generation (no LLM required)
- Optional LLM-enhanced abstractive essence
- Fidelity score computation
- Fidelity level selection
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber

# Type alias for LLM call function (async str -> str)
LLMCallFn = Callable[[str], Coroutine[None, None, str]]

logger = logging.getLogger(__name__)

# Sentence splitter: handles ., !, ? followed by whitespace or end-of-string.
# Preserves abbreviations like "e.g." and "Dr." by requiring capital after period.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])|(?<=[.!?])$")

# Entity-like patterns: technical terms, proper nouns, code refs.
_ENTITY_RE = re.compile(
    r"\b[A-Z]{2,}[a-z]*\b"  # Acronyms/tech terms: JSONB, PostgreSQL, API
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b"  # Multi-word proper nouns: Auth Module
    r"|\b[A-Z][a-z]{3,}\b"  # Single-word proper nouns: Redis, Docker, Python (4+ chars)
    r"|`[^`]+`"  # Code references
    r'|"[^"]+"'  # Quoted terms
)

# Common sentence starters to exclude from entity detection
_COMMON_STARTERS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Some",
        "Any",
        "All",
        "But",
        "And",
        "Also",
        "However",
        "Meanwhile",
        "Furthermore",
    }
)

MAX_ESSENCE_LENGTH = 150


def extract_essence(content: str) -> str:
    """Extract a single-sentence essence from content.

    Uses sentence-level scoring based on:
    - Entity density (more named entities = more informative)
    - Position bias (first and last sentences are more important)

    Args:
        content: The full text content to distill.

    Returns:
        A single sentence, max 150 characters.
        Falls back to truncated first sentence if no good candidate.
    """
    if not content or not content.strip():
        return ""

    content = content.strip()

    # Split into sentences
    sentences = _split_sentences(content)
    if not sentences:
        return _truncate(content, MAX_ESSENCE_LENGTH)

    if len(sentences) == 1:
        return _truncate(sentences[0], MAX_ESSENCE_LENGTH)

    # Score each sentence
    scored = []
    for i, sentence in enumerate(sentences):
        score = _score_sentence(sentence, i, len(sentences))
        scored.append((score, i, sentence))

    # Pick highest-scoring sentence
    scored.sort(key=lambda x: x[0], reverse=True)
    best_sentence = scored[0][2]

    return _truncate(best_sentence, MAX_ESSENCE_LENGTH)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Use regex split, then filter empty
    parts = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in parts if s and s.strip()]

    # If regex didn't split (no sentence boundaries), try newline split
    if len(sentences) <= 1 and "\n" in text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines

    return sentences


def _score_sentence(sentence: str, position: int, total: int) -> float:
    """Score a sentence for essence selection.

    Scoring factors:
    - Entity density: more entities per word = more informative
    - Position bias: first sentence gets 0.15 boost, last gets 0.10
    - Length penalty: very short (<5 words) or very long (>30 words) get penalized
    """
    words = sentence.split()
    word_count = len(words)

    # Entity density (0.0 - 1.0)
    raw_entities = _ENTITY_RE.findall(sentence)
    entities = [e for e in raw_entities if e.strip() not in _COMMON_STARTERS]
    entity_density = min(1.0, len(entities) / max(1, word_count) * 3)

    # Position bias
    position_score = 0.0
    if position == 0:
        position_score = 0.15  # First sentence bias
    elif position == total - 1:
        position_score = 0.1  # Last sentence bias

    # Length penalty: prefer 5-25 words
    length_score = 1.0
    if word_count < 5:
        length_score = 0.3
    elif word_count > 30:
        length_score = 0.7

    return entity_density * 0.5 + position_score + length_score * 0.3


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, breaking at word boundary."""
    if len(text) <= max_length:
        return text

    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.5:
        truncated = truncated[:last_space]

    return truncated.rstrip(".,;:- ") + "..."


# ── Fidelity Levels (Phase 2) ──────────────────────────────────────────


class FidelityLevel(StrEnum):
    """Memory fidelity levels, from richest to most compressed."""

    FULL = "full"
    SUMMARY = "summary"
    ESSENCE = "essence"
    GHOST = "ghost"


def compute_fidelity_score(
    *,
    activation: float,
    importance: float,
    hours_since_access: float,
    decay_rate: float,
    decay_floor: float = 0.05,
) -> float:
    """Compute fidelity score for a memory.

    Formula: max((importance + activation) * e^(-decay_rate * t), decay_floor)

    Args:
        activation: Spreading activation level from retrieval (0-1).
        importance: Normalized priority (0-1).
        hours_since_access: Hours since last access/conducted.
        decay_rate: Lambda decay rate from BrainConfig.
        decay_floor: Minimum score — ghost never reaches 0.

    Returns:
        Score in [decay_floor, 1.0].
    """
    base = min(1.0, importance + activation)
    decay = math.exp(-decay_rate * hours_since_access)
    return max(base * decay, decay_floor)


def select_fidelity(
    score: float,
    budget_pressure: float,
    *,
    full_threshold: float = 0.6,
    summary_threshold: float = 0.3,
    essence_threshold: float = 0.1,
) -> FidelityLevel:
    """Select fidelity level based on score and budget pressure.

    Budget pressure shifts thresholds upward — under pressure, more items
    get downgraded to cheaper fidelity levels.

    Args:
        score: Fidelity score from compute_fidelity_score (0-1).
        budget_pressure: Token budget pressure (0=plenty, 1=critical).
        full_threshold: Score above which FULL is selected.
        summary_threshold: Score above which SUMMARY is selected.
        essence_threshold: Score above which ESSENCE is selected.

    Returns:
        FidelityLevel for this memory.
    """
    # Validate threshold ordering
    if not (full_threshold >= summary_threshold >= essence_threshold >= 0.0):
        logger.warning(
            "Fidelity thresholds not in decreasing order: full=%.2f summary=%.2f essence=%.2f",
            full_threshold,
            summary_threshold,
            essence_threshold,
        )

    # Budget pressure shifts thresholds up (0.0 = no shift, 1.0 = +0.3 shift)
    shift = budget_pressure * 0.3

    if score >= full_threshold + shift:
        return FidelityLevel.FULL
    if score >= summary_threshold + shift:
        return FidelityLevel.SUMMARY
    if score >= essence_threshold + shift:
        return FidelityLevel.ESSENCE
    return FidelityLevel.GHOST


def render_at_fidelity(
    fiber: Fiber,
    level: FidelityLevel,
    anchor_content: str | None = None,
) -> str:
    """Render fiber content at the given fidelity level.

    Falls back gracefully: if requested level is unavailable, serves next
    richer level (ESSENCE → SUMMARY → FULL).

    Args:
        fiber: The fiber to render.
        level: Desired fidelity level.
        anchor_content: Full anchor neuron content (for FULL level).

    Returns:
        Rendered content string.
    """
    if level == FidelityLevel.GHOST:
        return _render_ghost(fiber)

    if level == FidelityLevel.ESSENCE:
        if fiber.essence:
            return fiber.essence
        # Fallback: try summary, then full
        if fiber.summary:
            return fiber.summary
        return anchor_content or ""

    if level == FidelityLevel.SUMMARY:
        if fiber.summary:
            return fiber.summary
        # Fallback: full content
        return anchor_content or ""

    # FULL
    result = anchor_content or fiber.summary or ""
    if not result:
        logger.warning("FULL fidelity render returned empty for fiber %s", fiber.id)
    return result


def _render_ghost(fiber: Fiber) -> str:
    """Render ghost view — interpretable metadata without content."""
    from neural_memory.utils.timeutils import utcnow

    try:
        age_hours = 0.0
        if fiber.created_at:
            delta = utcnow() - fiber.created_at
            age_hours = delta.total_seconds() / 3600

        if age_hours < 24:
            age_str = f"{age_hours:.0f}h"
        elif age_hours < 720:
            age_str = f"{age_hours / 24:.0f}d"
        else:
            age_str = f"{age_hours / 720:.0f}mo"

        tags = sorted(fiber.tags or [])[:3]
        tag_str = ", ".join(tags) if tags else "untagged"
        links = getattr(fiber, "synapse_count", 0) or 0

        return f"[~] {tag_str} | {age_str} ago | {links} links | recall:fiber:{fiber.id}"
    except Exception:
        logger.warning("Ghost render failed for fiber %s", fiber.id, exc_info=True)
        return f"[~] recall:fiber:{fiber.id}"


# ── Essence Generators (Phase 4) ──────────────────────────────────────


class EssenceGenerator(ABC):
    """Abstract base class for essence generation strategies."""

    @abstractmethod
    async def generate(self, content: str, *, priority: int = 5) -> str:
        """Generate a single-sentence essence from content.

        Args:
            content: Full text content to distill.
            priority: Memory priority (0-10). Implementations may skip
                      expensive operations for low-priority memories.

        Returns:
            Essence string (max ~150 chars), or empty string on failure.
        """


class ExtractiveEssenceGenerator(EssenceGenerator):
    """Extractive essence — sentence-level scoring, no LLM. Fast and free."""

    async def generate(self, content: str, *, priority: int = 5) -> str:
        return extract_essence(content)


class LLMEssenceGenerator(EssenceGenerator):
    """LLM-enhanced abstractive essence via configured provider.

    Cost guard: skips LLM for priority < 3. Falls back to extractive.
    """

    _PROMPT_TEMPLATE = (
        "Distill the following into exactly one sentence (max 30 words). "
        "Capture the core decision, fact, or insight. No filler.\n\n{content}"
    )
    _MIN_PRIORITY = 3
    _MAX_INPUT_CHARS = 2000

    def __init__(self, llm_call: LLMCallFn | None = None) -> None:
        self._llm_call = llm_call
        self._extractive = ExtractiveEssenceGenerator()

    async def generate(self, content: str, *, priority: int = 5) -> str:
        # Cost guard: low-priority memories get extractive only
        if priority < self._MIN_PRIORITY:
            return await self._extractive.generate(content, priority=priority)

        if not self._llm_call:
            return await self._extractive.generate(content, priority=priority)

        # Truncate input at word boundary to cap token cost
        truncated = (
            _truncate(content, self._MAX_INPUT_CHARS)
            if len(content) > self._MAX_INPUT_CHARS
            else content
        )
        prompt = self._PROMPT_TEMPLATE.format(content=truncated)

        try:
            result = await self._llm_call(prompt)
            if result and isinstance(result, str) and len(result.strip()) > 0:
                # Enforce max length
                essence = result.strip()
                if len(essence) > MAX_ESSENCE_LENGTH:
                    essence = _truncate(essence, MAX_ESSENCE_LENGTH)
                return essence
        except Exception:
            logger.debug("LLM essence generation failed, falling back to extractive", exc_info=True)

        # Fallback to extractive
        return await self._extractive.generate(content, priority=priority)


def get_essence_generator(
    strategy: str = "extractive", llm_call: LLMCallFn | None = None
) -> EssenceGenerator:
    """Factory for essence generators.

    Args:
        strategy: "extractive" (default, fast) or "llm" (quality, needs provider).
        llm_call: Async function(prompt: str) -> str for LLM strategy.

    Returns:
        EssenceGenerator instance.
    """
    if strategy == "llm":
        return LLMEssenceGenerator(llm_call=llm_call)
    if strategy != "extractive":
        logger.warning(
            "Unknown essence_generator strategy %r, falling back to extractive", strategy
        )
    return ExtractiveEssenceGenerator()
