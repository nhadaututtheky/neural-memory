"""Token budget management — cost-aware retrieval and context allocation.

Inspired by TEMM1E's resource budgeting: treats the context window as a
finite resource and allocates tokens based on value-per-token ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.engine.activation import ActivationResult

logger = logging.getLogger(__name__)

# Average tokens per whitespace-separated word (subword tokenization overhead)
TOKEN_RATIO = 1.3


@dataclass(frozen=True)
class TokenCost:
    """Token cost estimate for a single fiber/neuron."""

    fiber_id: str
    content_tokens: int  # Estimated tokens for fiber content
    metadata_tokens: int  # Estimated tokens for metadata/formatting
    total_tokens: int  # content + metadata
    value_score: float  # Activation/relevance score (higher = more valuable)
    value_per_token: float  # value_score / total_tokens (efficiency metric)


@dataclass(frozen=True)
class BudgetAllocation:
    """Result of budget-aware selection."""

    selected: list[TokenCost]  # Fibers selected within budget
    total_tokens_used: int
    total_tokens_budget: int
    tokens_remaining: int
    fibers_dropped: int  # How many fibers were dropped due to budget
    budget_utilization: float  # 0.0-1.0


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for token budget allocation."""

    # Reserved tokens for system overhead (headers, formatting)
    system_overhead_tokens: int = 50
    # Per-fiber formatting overhead (bullet points, tags, etc.)
    per_fiber_overhead: int = 15
    # Minimum tokens per fiber to be worth including
    min_fiber_tokens: int = 10
    # Maximum fibers to consider (pre-filter)
    max_fibers_considered: int = 50
    # Enable cross-fiber dedup/merge/rescore via context compiler
    enable_compiler: bool = True
    # SimHash hamming distance threshold for near-duplicate detection
    compiler_dedup_threshold: int = 10
    # Enable age-compression + SimHash dedup + content cap for individual
    # neurons in the "## Related Information" section of recall output.
    # Measurement showed this section consumes ~86% of context tokens while
    # bypassing all compression. See plan-related-info-compression.md.
    enable_related_compression: bool = True
    # Hard cap on per-neuron content tokens when related compression is on.
    # Neurons exceeding this are truncated via _truncate_to_sentences then
    # word-clipped. Default 150 tokens (neurons should be atomic facts).
    related_neuron_max_tokens: int = 150


def estimate_fiber_tokens(
    content: str,
    anchor_content: str = "",
    summary: str | None = None,
) -> int:
    """Estimate token count for a single fiber based on content.

    Uses the summary if available, otherwise falls back to anchor_content
    or the raw content parameter.

    Args:
        content: Primary content string (used if no summary/anchor).
        anchor_content: Anchor neuron content (fallback when no summary).
        summary: Fiber summary text (preferred — most compressed form).

    Returns:
        Estimated token count as int.
    """
    effective = summary or anchor_content or content
    if not effective:
        return 0
    word_count = len(effective.split())
    return max(1, int(word_count * TOKEN_RATIO))


def compute_token_costs(
    fibers: list[Fiber],
    activations: dict[str, ActivationResult],
    config: BudgetConfig | None = None,
) -> list[TokenCost]:
    """Batch compute token costs for all candidate fibers.

    Args:
        fibers: Candidate Fiber instances.
        activations: Map of neuron_id -> ActivationResult for scoring.
        config: Budget configuration (uses defaults if None).

    Returns:
        List of TokenCost, one per fiber, sorted by value_per_token descending.
    """
    if config is None:
        config = BudgetConfig()

    costs: list[TokenCost] = []

    for fiber in fibers[: config.max_fibers_considered]:
        # Estimate content tokens from summary or anchor content
        content_text = fiber.summary or ""
        if not content_text and hasattr(fiber, "metadata"):
            # Fall back to metadata hint if available
            content_text = fiber.metadata.get("_content_preview", "")
        content_tokens = estimate_fiber_tokens(content_text)

        # Metadata overhead: per-fiber formatting (bullet, tags, timestamps)
        metadata_tokens = config.per_fiber_overhead

        total_tokens = max(config.min_fiber_tokens, content_tokens + metadata_tokens)

        # Compute activation/value score for this fiber
        value_score: float = 0.0
        if fiber.anchor_neuron_id in activations:
            value_score = activations[fiber.anchor_neuron_id].activation_level
        elif fiber.neuron_ids:
            for nid in fiber.neuron_ids:
                ar = activations.get(nid)
                if ar is not None:
                    value_score = max(value_score, ar.activation_level)

        # Also factor in fiber quality signals
        value_score = value_score + fiber.salience * 0.1 + fiber.conductivity * 0.05

        # Efficiency metric: value per token spent
        value_per_token = value_score / total_tokens if total_tokens > 0 else 0.0

        costs.append(
            TokenCost(
                fiber_id=fiber.id,
                content_tokens=content_tokens,
                metadata_tokens=metadata_tokens,
                total_tokens=total_tokens,
                value_score=value_score,
                value_per_token=value_per_token,
            )
        )

    return costs


def allocate_budget(
    costs: list[TokenCost],
    max_tokens: int,
    config: BudgetConfig | None = None,
) -> BudgetAllocation:
    """Greedy value-per-token selection within token budget.

    Sorts costs by value_per_token descending and greedily selects fibers
    as long as they fit within the remaining budget.

    Args:
        costs: TokenCost list (from compute_token_costs).
        max_tokens: Total token budget available.
        config: Budget configuration (uses defaults if None).

    Returns:
        BudgetAllocation with selected fibers and stats.
    """
    if config is None:
        config = BudgetConfig()

    effective_budget = max(0, max_tokens - config.system_overhead_tokens)

    if not costs or effective_budget <= 0:
        return BudgetAllocation(
            selected=[],
            total_tokens_used=0,
            total_tokens_budget=max_tokens,
            tokens_remaining=max_tokens,
            fibers_dropped=len(costs),
            budget_utilization=0.0,
        )

    # Sort by value_per_token descending (greedy knapsack approximation)
    ranked = sorted(costs, key=lambda c: c.value_per_token, reverse=True)

    selected: list[TokenCost] = []
    tokens_used = 0

    for cost in ranked:
        if tokens_used + cost.total_tokens <= effective_budget:
            selected.append(cost)
            tokens_used += cost.total_tokens

    fibers_dropped = len(costs) - len(selected)
    utilization = tokens_used / effective_budget if effective_budget > 0 else 0.0

    return BudgetAllocation(
        selected=selected,
        total_tokens_used=tokens_used,
        total_tokens_budget=max_tokens,
        tokens_remaining=max_tokens - tokens_used - config.system_overhead_tokens,
        fibers_dropped=fibers_dropped,
        budget_utilization=min(1.0, utilization),
    )


def format_budget_report(allocation: BudgetAllocation) -> dict[str, Any]:
    """Format allocation stats for MCP response inclusion.

    Args:
        allocation: Result from allocate_budget().

    Returns:
        Plain dict suitable for JSON serialization in MCP response.
    """
    return {
        "fibers_selected": len(allocation.selected),
        "fibers_dropped": allocation.fibers_dropped,
        "total_tokens_used": allocation.total_tokens_used,
        "total_tokens_budget": allocation.total_tokens_budget,
        "tokens_remaining": allocation.tokens_remaining,
        "budget_utilization": round(allocation.budget_utilization, 3),
        "top_costs": [
            {
                "fiber_id": c.fiber_id,
                "total_tokens": c.total_tokens,
                "value_score": round(c.value_score, 4),
                "value_per_token": round(c.value_per_token, 6),
            }
            for c in allocation.selected[:5]
        ],
    }
