"""Unified metacognitive confidence scoring.

Aggregates multiple quality signals (retrieval strength, content quality,
fidelity layer, freshness, familiarity) into a single 0-1 confidence score
that callers can use to gauge how much to trust a recall result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from neural_memory.utils.timeutils import utcnow

# Fidelity layer to score mapping
_FIDELITY_SCORES: dict[str, float] = {
    "verbatim": 1.0,
    "detail": 0.7,
    "summary": 0.5,
    "gist": 0.4,
    "essence": 0.3,
}


@dataclass(frozen=True)
class ConfidenceScore:
    """Unified confidence assessment for a recall result.

    Attributes:
        overall: Single 0-1 score callers want.
        retrieval: Retrieval pipeline strength (sufficiency + activation).
        content_quality: Normalized quality score (0-1).
        fidelity: Encoding fidelity (verbatim=1.0 down to essence=0.3).
        freshness: Age-based decay (1.0 for new, decays over time).
        familiarity_penalty: 0.0 for real match, negative for familiarity fallback.
        components: All raw signal values for transparency.
    """

    overall: float
    retrieval: float
    content_quality: float
    fidelity: float
    freshness: float
    familiarity_penalty: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfidenceWeights:
    """Configurable weights for confidence aggregation."""

    retrieval: float = 0.35
    content_quality: float = 0.25
    fidelity: float = 0.20
    freshness: float = 0.20


def compute_confidence(
    retrieval_score: float = 0.5,
    sufficiency_confidence: float = 0.5,
    quality_score: float = 5.0,
    fidelity_layer: str = "detail",
    created_at: datetime | None = None,
    is_familiarity_fallback: bool = False,
    weights: ConfidenceWeights | None = None,
    extra_signals: dict[str, Any] | None = None,
) -> ConfidenceScore:
    """Compute unified confidence from multiple quality signals.

    Args:
        retrieval_score: Fiber score from retrieval pipeline (0-1 typical).
        sufficiency_confidence: Sufficiency gate confidence (0-1).
        quality_score: Content quality score (0-10 scale).
        fidelity_layer: Encoding fidelity ('verbatim', 'detail', 'summary', 'gist', 'essence').
        created_at: When the memory was created (for freshness).
        is_familiarity_fallback: Whether this is a familiarity guess, not real recall.
        weights: Optional custom weights.
        extra_signals: Additional raw signals to include in components.

    Returns:
        ConfidenceScore with overall and per-dimension scores.
    """
    w = weights or ConfidenceWeights()

    # 1. Retrieval dimension: blend pipeline score with sufficiency
    retrieval = min(1.0, max(0.0, retrieval_score * 0.6 + sufficiency_confidence * 0.4))

    # 2. Content quality: normalize 0-10 to 0-1
    content_quality = min(1.0, max(0.0, quality_score / 10.0))

    # 3. Fidelity: map layer name to score
    fidelity = _FIDELITY_SCORES.get(fidelity_layer, 0.5)

    # 4. Freshness: sigmoid decay based on age in days
    now = utcnow()
    if created_at is not None:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        # Half-life ~30 days: freshness = 1 / (1 + age/30)
        freshness = 1.0 / (1.0 + age_days / 30.0)
    else:
        freshness = 0.5  # Unknown age, neutral

    # 5. Familiarity penalty
    familiarity_penalty = -0.3 if is_familiarity_fallback else 0.0

    # Weighted sum
    overall = (
        w.retrieval * retrieval
        + w.content_quality * content_quality
        + w.fidelity * fidelity
        + w.freshness * freshness
        + familiarity_penalty
    )
    overall = min(1.0, max(0.0, overall))

    # Collect all components
    components: dict[str, float] = {
        "retrieval_score": round(retrieval_score, 4),
        "sufficiency_confidence": round(sufficiency_confidence, 4),
        "quality_score": round(quality_score, 2),
        "fidelity_layer": fidelity,
        "age_days": round((now - created_at).total_seconds() / 86400.0, 1) if created_at else -1,
        "is_familiarity_fallback": 1.0 if is_familiarity_fallback else 0.0,
    }
    if extra_signals:
        for k, v in extra_signals.items():
            if isinstance(v, (int, float)):
                components[k] = round(float(v), 4)

    return ConfidenceScore(
        overall=round(overall, 4),
        retrieval=round(retrieval, 4),
        content_quality=round(content_quality, 4),
        fidelity=round(fidelity, 4),
        freshness=round(freshness, 4),
        familiarity_penalty=round(familiarity_penalty, 4),
        components=components,
    )
