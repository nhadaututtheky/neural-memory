"""Smart context optimizer — prioritize and budget context tokens.

Ranks context items by composite score (activation, priority, frequency,
conductivity, freshness) instead of pure recency.  Allocates token budget
proportionally and deduplicates near-identical content via SimHash.

Zero LLM dependency — pure graph metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
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
    fidelity_level: str = "full"


@dataclass(frozen=True)
class FidelityStats:
    """Counts of items at each fidelity level."""

    full: int = 0
    summary: int = 0
    essence: int = 0
    ghost: int = 0


@dataclass(frozen=True)
class ContextPlan:
    """Result of context optimization.

    Attributes:
        items: Ordered context items (highest score first)
        total_tokens: Total estimated tokens used
        dropped_count: Items dropped due to budget or dedup
        fidelity_stats: Count of items at each fidelity level
        ghost_fiber_ids: Fiber IDs rendered as ghosts (for tracking)
    """

    items: list[ContextItem]
    total_tokens: int
    dropped_count: int
    fidelity_stats: FidelityStats = FidelityStats()
    ghost_fiber_ids: list[str] = field(default_factory=list)


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
                    fidelity_level=item.fidelity_level,
                )
            )
            tokens_used += truncated_tokens

    return result


async def optimize_context(
    storage: NeuralStorage,
    fibers: list[Fiber],
    max_tokens: int,
    reference_time: datetime | None = None,
    fidelity_enabled: bool = True,
    fidelity_full_threshold: float = 0.6,
    fidelity_summary_threshold: float = 0.3,
    fidelity_essence_threshold: float = 0.1,
    decay_rate: float = 0.1,
    decay_floor: float = 0.05,
    embed_fn: Callable[[str], Coroutine[None, None, list[float]]] | None = None,
    exclude_cold: bool = True,
) -> ContextPlan:
    """Optimize context selection and token allocation.

    Scores each fiber by composite relevance, deduplicates by SimHash,
    selects fidelity level per item based on score + budget pressure,
    allocates token budget proportionally, and truncates low-priority items.

    Args:
        storage: Storage backend for neuron state and typed memory lookups
        fibers: Candidate fibers (already filtered by freshness if needed)
        max_tokens: Maximum total token budget
        reference_time: Reference time for freshness (default: now)
        fidelity_enabled: Whether to use fidelity-aware content selection
        fidelity_full_threshold: Score threshold for FULL fidelity
        fidelity_summary_threshold: Score threshold for SUMMARY fidelity
        fidelity_essence_threshold: Score threshold for ESSENCE fidelity
        decay_rate: Lambda for fidelity decay formula
        decay_floor: Minimum fidelity score (ghost floor)
        embed_fn: Optional embedding function for semantic deduplication
        exclude_cold: Whether to exclude COLD-tier memories (default: True).
            HOT-tier memories get a +0.3 score boost.

    Returns:
        ContextPlan with ordered, budget-constrained items + fidelity stats
    """
    from neural_memory.engine.fidelity import (
        FidelityLevel,
        anisotropic_compress,
        compute_fidelity_score,
        render_at_fidelity,
        select_fidelity,
    )

    if not fibers:
        return ContextPlan(items=[], total_tokens=0, dropped_count=0)

    if reference_time is None:
        from neural_memory.utils.timeutils import utcnow

        reference_time = utcnow()

    # Phase 1: Build scored items with full content + metadata
    scored_items: list[ContextItem] = []
    content_hashes: dict[str, int] = {}
    # Keep fiber refs and anchor content for fidelity rendering
    fiber_map: dict[str, Fiber] = {}
    anchor_content_map: dict[str, str] = {}
    activation_map: dict[str, float] = {}
    priority_map: dict[str, float] = {}

    for fiber in fibers:
        # Get content
        anchor_content: str | None = None
        if fiber.anchor_neuron_id:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            if anchor:
                anchor_content = anchor.content
                content_hashes[fiber.id] = anchor.content_hash

        content = fiber.summary or anchor_content
        if not content:
            continue

        fiber_map[fiber.id] = fiber
        anchor_content_map[fiber.id] = anchor_content or ""

        # Get activation level from neuron state
        activation = 0.0
        try:
            if fiber.anchor_neuron_id:
                state = await storage.get_neuron_state(fiber.anchor_neuron_id)
                if state and hasattr(state, "activation_level"):
                    activation = float(state.activation_level)
        except (TypeError, ValueError, AttributeError):
            pass
        activation_map[fiber.id] = activation

        # Get priority + tier from typed memory
        priority_norm = 0.5
        fiber_tier = "warm"
        try:
            typed_mem = await storage.get_typed_memory(fiber.id)
            if typed_mem:
                if hasattr(typed_mem, "priority") and isinstance(typed_mem.priority, (int, float)):
                    priority_norm = typed_mem.priority / 10.0
                raw_tier = getattr(typed_mem, "tier", "warm") or "warm"
                fiber_tier = str(raw_tier).lower()
        except (TypeError, ValueError, AttributeError):
            pass
        priority_map[fiber.id] = priority_norm

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

        # Skip COLD memories in context (unless explicitly requested)
        if fiber_tier == "cold" and exclude_cold:
            continue

        # Composite score
        score = compute_composite_score(
            activation=activation,
            priority=priority_norm,
            frequency=frequency_norm,
            conductivity=conductivity,
            freshness=freshness_result.score,
        )

        # Tier-aware score boost: HOT gets +0.3 to ensure top placement
        if fiber_tier == "hot":
            score = min(1.0, score + 0.3)

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

    # Phase 3.5: Fidelity-aware content selection
    if fidelity_enabled and scored_items:
        # Estimate budget pressure from raw token demand vs budget
        raw_tokens = sum(item.token_count for item in scored_items)
        budget_pressure = min(1.0, max(0.0, (raw_tokens - max_tokens) / max(1, raw_tokens)))

        fidelity_items: list[ContextItem] = []
        for item in scored_items:
            fiber_or_none = fiber_map.get(item.fiber_id)
            if not fiber_or_none:
                fidelity_items.append(item)
                continue
            fiber = fiber_or_none

            # Compute per-fiber fidelity score using decay formula
            last_access = fiber.last_conducted or fiber.created_at
            # No timestamp → treat as old (30 days) so score degrades gracefully
            hours = (reference_time - last_access).total_seconds() / 3600 if last_access else 720.0

            fidelity_score = compute_fidelity_score(
                activation=activation_map.get(fiber.id, 0.0),
                importance=priority_map.get(fiber.id, 0.5),
                hours_since_access=hours,
                decay_rate=decay_rate,
                decay_floor=decay_floor,
            )

            # Ghost visibility boost: recently-shown ghosts are contextually relevant
            if fiber.last_ghost_shown_at:
                ghost_age_hours = (
                    reference_time - fiber.last_ghost_shown_at
                ).total_seconds() / 3600
                if ghost_age_hours < 24:
                    fidelity_score = min(1.0, fidelity_score + 0.1)

            level = select_fidelity(
                fidelity_score,
                budget_pressure,
                full_threshold=fidelity_full_threshold,
                summary_threshold=fidelity_summary_threshold,
                essence_threshold=fidelity_essence_threshold,
            )

            # Use anisotropic compression when embed_fn available and level
            # is ESSENCE or SUMMARY (direction-preserving compression)
            rendered = ""
            anchor_text = anchor_content_map.get(fiber.id, "")
            source_content = anchor_text or fiber.summary or ""
            if (
                embed_fn
                and source_content
                and level in (FidelityLevel.ESSENCE, FidelityLevel.SUMMARY)
            ):
                try:
                    rendered = await anisotropic_compress(source_content, level, embed_fn)
                except Exception:
                    rendered = ""

            if not rendered:
                rendered = render_at_fidelity(fiber, level, anchor_content=anchor_text)
            if not rendered:
                rendered = item.content  # Fallback to original

            fidelity_items.append(
                ContextItem(
                    fiber_id=item.fiber_id,
                    content=rendered,
                    score=item.score,
                    token_count=_estimate_tokens(rendered),
                    fidelity_level=level.value,
                )
            )

        scored_items = fidelity_items

    # Phase 4: Allocate token budgets
    budgeted = allocate_token_budgets(scored_items, max_tokens)
    budget_dropped = len(scored_items) - len(budgeted)

    # Recount fidelity stats and collect ghost fiber IDs from budgeted items only
    final_counts = {"full": 0, "summary": 0, "essence": 0, "ghost": 0}
    ghost_ids: list[str] = []
    for item in budgeted:
        final_counts[item.fidelity_level] = final_counts.get(item.fidelity_level, 0) + 1
        if item.fidelity_level == "ghost":
            ghost_ids.append(item.fiber_id)

    total_tokens = sum(item.token_count for item in budgeted)

    return ContextPlan(
        items=budgeted,
        total_tokens=total_tokens,
        dropped_count=dedup_dropped + budget_dropped,
        fidelity_stats=FidelityStats(
            full=final_counts["full"],
            summary=final_counts["summary"],
            essence=final_counts["essence"],
            ghost=final_counts["ghost"],
        ),
        ghost_fiber_ids=ghost_ids,
    )
