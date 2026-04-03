"""Quality scorer — scores memory content richness and enforces write gate.

When write gate is enabled, memories below quality thresholds are rejected
before storage. When disabled, behaves as a soft gate (hints only, never rejects).

Addresses GitHub Issue #95: write-gate to improve brain purity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.unified_config import WriteGateConfig

# -- Cognitive richness word lists ------------------------------------------

_CAUSAL_PATTERN = re.compile(
    r"\b(because|caused|due to|led to|resulted in|reason|since|therefore"
    r"|so that|in order to|as a result|consequently)\b",
    re.IGNORECASE,
)

_TEMPORAL_PATTERN = re.compile(
    r"\b(after|before|then|when|while|during|until|once|first|next"
    r"|previously|subsequently|earlier|later|following)\b",
    re.IGNORECASE,
)

_COMPARATIVE_PATTERN = re.compile(
    r"\b(over|instead of|replaced|vs|versus|compared to|rather than"
    r"|better than|worse than|faster|slower|cheaper|preferred)\b",
    re.IGNORECASE,
)

# A8 Phase 3: Specificity patterns — concrete details that improve recall quality
_FILE_PATH_PATTERN = re.compile(
    r"[\w/\\]+\.\w{1,5}\b(?<!\.\w\.)"  # e.g. retrieval.py, src/auth/middleware.ts
    # Excludes abbreviations like "e.g." or "i.e."
)
_VERSION_PATTERN = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")  # v1.2.3, 4.28.0
_NUMBER_PATTERN = re.compile(r"\b\d{3,}\b")  # numbers with 3+ digits (skip 2-digit)
_STRUCTURE_PATTERN = re.compile(r"→|⟶|=>|->|-->")

# Generic filler patterns — content that is ONLY these words (no substance)
_GENERIC_FILLER = frozenset(
    {
        "done",
        "ok",
        "okay",
        "completed",
        "noted",
        "got it",
        "understood",
        "xong",
        "oke",
        "roger",
        "ack",
        "acknowledged",
        "yep",
        "yes",
        "no",
        "nope",
        "sure",
        "fine",
        "good",
        "great",
        "thanks",
        "thank you",
        "cam on",
        "hieu roi",
        "da",
        "vang",
    }
)

# -- Quality thresholds -----------------------------------------------------

_MIN_CONTENT_LENGTH = 10
_LOW_THRESHOLD = 4
_HIGH_THRESHOLD = 7


@dataclass(frozen=True)
class QualityResult:
    """Quality assessment for a memory."""

    score: int  # 0-10
    quality: str  # "low" | "medium" | "high"
    hints: tuple[str, ...] = field(default_factory=tuple)
    rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "quality": self.quality,
            "score": self.score,
        }
        if self.hints:
            result["hints"] = list(self.hints)
        if self.rejected:
            result["rejected"] = True
            result["rejection_reason"] = self.rejection_reason
        return result


def _is_generic_filler(content: str) -> bool:
    """Check if content is purely generic filler with no substance."""
    normalized = content.strip().lower().rstrip(".!?")
    return normalized in _GENERIC_FILLER


def score_memory(
    content: str,
    *,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    context: dict[str, object] | None = None,
) -> QualityResult:
    """Score memory content quality on a 0-10 scale.

    Scoring breakdown:
    - Content length (0-2): >=10 chars +1, >=50 chars +1
    - Context dict provided (+3)
    - Tags provided, at least 1 (+1)
    - Non-default type provided (+1)
    - Cognitive richness (0-3): causal +1, temporal +1, comparative +1

    Args:
        content: The memory content text.
        memory_type: Memory type string (e.g. "decision", "fact").
        tags: List of tags.
        context: Structured context dict from agent.

    Returns:
        QualityResult with score, quality label, and improvement hints.
    """
    points = 0
    hints: list[str] = []

    # 1. Content length (0-2 points)
    content_len = len(content.strip())
    if content_len >= _MIN_CONTENT_LENGTH:
        points += 1
    else:
        hints.append(
            f"Content is very short ({content_len} chars) — add more detail for better recall"
        )
    if content_len >= 50:
        points += 1

    # 2. Context dict (+3 points)
    if context and len(context) > 0:
        points += 3
    else:
        hints.append("Add context dict (reason, alternatives, cause) for richer neural connections")

    # 3. Tags (+1 point)
    if tags and len(tags) > 0:
        points += 1
    else:
        hints.append("Add tags for better retrieval")

    # 4. Non-default type (+1 point)
    if memory_type and memory_type != "fact":
        points += 1
    elif not memory_type:
        hints.append("Specify memory type (decision, insight, error, workflow)")

    # 5. Cognitive richness (0-3 points)
    has_causal = bool(_CAUSAL_PATTERN.search(content))
    has_temporal = bool(_TEMPORAL_PATTERN.search(content))
    has_comparative = bool(_COMPARATIVE_PATTERN.search(content))

    if has_causal:
        points += 1
    if has_temporal:
        points += 1
    if has_comparative:
        points += 1

    cognitive_count = sum([has_causal, has_temporal, has_comparative])
    if cognitive_count == 0:
        hints.append(
            "Add reasoning: why (because...), when (after...), or comparison (over X, instead of Y)"
        )

    # A8 Phase 3: Specificity bonus (+1 for file paths/versions/numbers)
    has_specificity = (
        bool(_FILE_PATH_PATTERN.search(content))
        or bool(_VERSION_PATTERN.search(content))
        or bool(_NUMBER_PATTERN.search(content))
    )
    if has_specificity:
        points += 1

    # A8 Phase 3: Structure markers (+1 for arrows, decision markers)
    if _STRUCTURE_PATTERN.search(content):
        points += 1

    # A8 Phase 3: Brevity bonus (+1 for 50-300 chars — optimal recall range)
    if 50 <= content_len <= 300:
        points += 1

    # A8 Phase 3: Wall-of-text penalty (-2 for >500 chars)
    if content_len > 500:
        points -= 2
        hints.append(
            f"Memory is {content_len} chars — split into focused pieces (<300 chars) for better recall"
        )

    # Cap at 10
    points = max(0, min(points, 10))

    # Determine quality label
    if points >= _HIGH_THRESHOLD:
        quality = "high"
    elif points >= _LOW_THRESHOLD:
        quality = "medium"
    else:
        quality = "low"

    # Don't return hints for high quality
    if quality == "high":
        hints = []

    return QualityResult(
        score=points,
        quality=quality,
        hints=tuple(hints),
    )


def check_write_gate(
    content: str,
    *,
    gate_config: WriteGateConfig,
    is_auto_capture: bool = False,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    context: dict[str, object] | None = None,
) -> QualityResult:
    """Check content against write gate thresholds.

    Runs quality scoring + gate-specific checks (length, filler, wall-of-text).
    Returns a QualityResult with ``rejected=True`` if content fails the gate.

    Args:
        content: The memory content text.
        gate_config: WriteGateConfig with thresholds.
        is_auto_capture: If True, uses stricter ``auto_capture_min_score``.
        memory_type: Memory type string.
        tags: List of tags.
        context: Structured context dict.

    Returns:
        QualityResult — check ``rejected`` field.
    """
    stripped = content.strip()

    # Gate 1: Generic filler check (before scoring)
    if gate_config.reject_generic_filler and _is_generic_filler(stripped):
        return QualityResult(
            score=0,
            quality="low",
            hints=("Provide substantive content instead of generic filler",),
            rejected=True,
            rejection_reason=f"Generic filler rejected: '{stripped[:50]}'",
        )

    # Gate 2: Minimum length
    if len(stripped) < gate_config.min_length:
        return QualityResult(
            score=0,
            quality="low",
            hints=(
                f"Content too short ({len(stripped)} chars). "
                f"Minimum: {gate_config.min_length} chars.",
            ),
            rejected=True,
            rejection_reason=f"Content too short ({len(stripped)} < {gate_config.min_length} chars)",
        )

    # Gate 3: Wall-of-text cap
    if len(stripped) > gate_config.max_content_length:
        return QualityResult(
            score=1,
            quality="low",
            hints=(
                f"Content too long ({len(stripped)} chars). "
                f"Max: {gate_config.max_content_length} chars. "
                "Split into smaller, focused memories.",
            ),
            rejected=True,
            rejection_reason=(
                f"Content too long ({len(stripped)} > {gate_config.max_content_length} chars). "
                "Split into smaller memories."
            ),
        )

    # Gate 4: Quality score check
    quality_result = score_memory(
        content,
        memory_type=memory_type,
        tags=tags,
        context=context,
    )

    min_score = (
        gate_config.auto_capture_min_score if is_auto_capture else gate_config.min_quality_score
    )

    if quality_result.score < min_score:
        return QualityResult(
            score=quality_result.score,
            quality=quality_result.quality,
            hints=quality_result.hints,
            rejected=True,
            rejection_reason=(
                f"Quality score {quality_result.score} below "
                f"minimum {min_score}. Improve content and retry."
            ),
        )

    return quality_result
