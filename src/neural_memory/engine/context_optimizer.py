"""Smart context optimizer — prioritize and budget context tokens.

Ranks context items by composite score (activation, priority, frequency,
conductivity, freshness) instead of pure recency.  Allocates token budget
proportionally and deduplicates near-identical content via SimHash.

Zero LLM dependency — pure graph metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.safety.freshness import evaluate_freshness
from neural_memory.utils.simhash import is_near_duplicate

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.storage.base import NeuralStorage

# Token estimation ratio (words -> tokens, ~1.3 tokens/word)
_TOKEN_RATIO = 1.3


def _estimate_tokens(text: str) -> int:
    """Estimate LLM token count from text using word-based heuristic."""
    return int(len(text.split()) * _TOKEN_RATIO)


@dataclass(frozen=True)
class ContextItem:
    """A single context entry with computed score and budget.

    Attributes:
        fiber_id: Source fiber ID
        content: Text content to include
        score: Composite relevance score (0.0 - 1.0)
        token_count: Estimated token count of content
        truncated: Whether content was truncated to fit budget
    """

    fiber_id: str
    content: str
    score: float
    token_count: int
    truncated: bool = False


@dataclass(frozen=True)
class ContextPlan:
    """Result of context optimization.

    Attributes:
        items: Ordered context items (highest score first)
        total_tokens: Total estimated tokens used
        dropped_count: Items dropped due to budget or dedup
    """

    items: list[ContextItem]
    total_tokens: int
    dropped_count: int


def compute_composite_score(
    *,
    activation: float = 0.0,
    priority: float = 0.5,
    frequency: float = 0.0,
    conductivity: float = 0.5,
    freshness: float = 0.5,
) -> float:
    """Compute composite relevance score for a context item.

    Args:
        activation: NeuronState activation level (0-1)
        priority: Normalized priority (TypedMemory.priority / 10, default 0.5)
        frequency: Normalized access frequency (min(fiber.frequency / 20, 1.0))
        conductivity: Fiber conductivity (0-1)
        freshness: Freshness score from evaluate_freshness (0-1)

    Returns:
        Composite score in range [0.0, 1.0]
    """
    return (
        0.30 * min(activation, 1.0)
        + 0.25 * min(priority, 1.0)
        + 0.20 * min(frequency, 1.0)
        + 0.15 * min(conductivity, 1.0)
        + 0.10 * min(freshness, 1.0)
    )


def deduplicate_by_simhash(items: list[ContextItem], hashes: dict[str, int]) -> list[ContextItem]:
    """Remove near-duplicate context items, keeping the higher-scoring one.

    Args:
        items: Context items sorted by score descending
        hashes: Mapping of fiber_id -> content_hash (SimHash fingerprint)

    Returns:
        Deduplicated list preserving score order
    """
    kept: list[ContextItem] = []
    kept_hashes: list[int] = []

    for item in items:
        h = hashes.get(item.fiber_id, 0)
        if h == 0:
            # No hash available — keep the item
            kept.append(item)
            continue

        is_dup = False
        for existing_hash in kept_hashes:
            if is_near_duplicate(h, existing_hash):
                is_dup = True
                break

        if not is_dup:
            kept.append(item)
            kept_hashes.append(h)

    return kept


def allocate_token_budgets(
    items: list[ContextItem],
    max_tokens: int,
    min_budget: int = 20,
) -> list[ContextItem]:
    """Allocate token budgets proportionally to composite scores.

    Items whose allocation falls below min_budget are dropped.
    Items exceeding their budget are truncated.

    Args:
        items: Context items sorted by score descending
        max_tokens: Total token budget
        min_budget: Minimum tokens per item (below = dropped)

    Returns:
        Budget-constrained context items
    """
    if not items:
        return []

    total_score = sum(item.score for item in items)
    if total_score <= 0:
        total_score = len(items)  # Equal distribution fallback

    result: list[ContextItem] = []
    tokens_used = 0

    for item in items:
        # Proportional budget
        budget = (
            int((item.score / total_score) * max_tokens)
            if total_score > 0
            else (max_tokens // len(items))
        )
        budget = max(budget, min_budget)

        if tokens_used + min_budget > max_tokens:
            break  # No room left

        if item.token_count <= budget:
            # Fits within budget
            result.append(item)
            tokens_used += item.token_count
        else:
            # Truncate content to fit budget
            words = item.content.split()
            target_words = int(budget / _TOKEN_RATIO)
            if target_words < 5:
                continue  # Too short to be useful
            truncated_content = " ".join(words[:target_words]) + "..."
            truncated_tokens = _estimate_tokens(truncated_content)
            result.append(
                ContextItem(
                    fiber_id=item.fiber_id,
                    content=truncated_content,
                    score=item.score,
                    token_count=truncated_tokens,
                    truncated=True,
                )
            )
            tokens_used += truncated_tokens

    return result


async def optimize_context(
    storage: NeuralStorage,
    fibers: list[Fiber],
    max_tokens: int,
    reference_time: datetime | None = None,
) -> ContextPlan:
    """Optimize context selection and token allocation.

    Scores each fiber by composite relevance, deduplicates by SimHash,
    allocates token budget proportionally, and truncates low-priority items.

    Args:
        storage: Storage backend for neuron state and typed memory lookups
        fibers: Candidate fibers (already filtered by freshness if needed)
        max_tokens: Maximum total token budget
        reference_time: Reference time for freshness (default: now)

    Returns:
        ContextPlan with ordered, budget-constrained items
    """
    if not fibers:
        return ContextPlan(items=[], total_tokens=0, dropped_count=0)

    if reference_time is None:
        from neural_memory.utils.timeutils import utcnow

        reference_time = utcnow()

    # Phase 1: Build scored items
    scored_items: list[ContextItem] = []
    content_hashes: dict[str, int] = {}

    for fiber in fibers:
        # Get content
        content = fiber.summary
        if not content and fiber.anchor_neuron_id:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            if anchor:
                content = anchor.content
                content_hashes[fiber.id] = anchor.content_hash
        if not content:
            continue

        # Get activation level from neuron state
        activation = 0.0
        try:
            if fiber.anchor_neuron_id:
                state = await storage.get_neuron_state(fiber.anchor_neuron_id)
                if state and hasattr(state, "activation_level"):
                    activation = float(state.activation_level)
        except (TypeError, ValueError, AttributeError):
            pass

        # Get priority from typed memory
        priority_norm = 0.5
        try:
            typed_mem = await storage.get_typed_memory(fiber.id)
            if (
                typed_mem
                and hasattr(typed_mem, "priority")
                and isinstance(typed_mem.priority, (int, float))
            ):
                priority_norm = typed_mem.priority / 10.0
        except (TypeError, ValueError, AttributeError):
            pass

        # Frequency (cap at 20)
        freq = getattr(fiber, "frequency", 0) or 0
        frequency_norm = (
            min(freq / 20.0, 1.0) if isinstance(freq, (int, float)) and freq > 0 else 0.0
        )

        # Freshness
        created_at = getattr(fiber, "created_at", None)
        if not isinstance(created_at, datetime):
            created_at = reference_time
        freshness_result = evaluate_freshness(created_at, reference_time)

        # Conductivity
        conductivity = getattr(fiber, "conductivity", 0.5)
        if not isinstance(conductivity, (int, float)):
            conductivity = 0.5

        # Composite score
        score = compute_composite_score(
            activation=activation,
            priority=priority_norm,
            frequency=frequency_norm,
            conductivity=conductivity,
            freshness=freshness_result.score,
        )

        scored_items.append(
            ContextItem(
                fiber_id=fiber.id,
                content=content,
                score=score,
                token_count=_estimate_tokens(content),
            )
        )

    # Phase 2: Sort by score descending
    scored_items.sort(key=lambda x: x.score, reverse=True)

    initial_count = len(scored_items)

    # Phase 3: Deduplicate
    scored_items = deduplicate_by_simhash(scored_items, content_hashes)
    dedup_dropped = initial_count - len(scored_items)

    # Phase 4: Allocate token budgets
    budgeted = allocate_token_budgets(scored_items, max_tokens)
    budget_dropped = len(scored_items) - len(budgeted)

    total_tokens = sum(item.token_count for item in budgeted)

    return ContextPlan(
        items=budgeted,
        total_tokens=total_tokens,
        dropped_count=dedup_dropped + budget_dropped,
    )
