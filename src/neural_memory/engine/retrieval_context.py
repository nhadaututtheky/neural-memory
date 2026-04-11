"""Context formatting for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.activation import ActivationResult
from neural_memory.utils.timeutils import utcnow

# Average tokens per whitespace-separated word (accounts for subword tokenization)
_TOKEN_RATIO = 1.3

# Compression tier thresholds (days)
_FULL_CONTENT_DAYS = 7
_SUMMARY_DAYS = 30
_MINIMAL_DAYS = 90

# Sentence splitting regex — handles ". ", "! ", "? " followed by uppercase or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def compress_for_recall(
    content: str,
    summary: str | None,
    created_at: datetime | None,
    max_sentences_medium: int = 3,
    max_sentences_old: int = 2,
) -> str:
    """Compress memory content based on age for context-efficient recall.

    Tiers:
        < 7 days: full content
        7-30 days: summary (if available) or first N sentences
        30-90 days: summary (if available) or first 2 sentences
        90+ days: summary (if available) or first sentence only

    Args:
        content: Raw memory content.
        summary: Fiber summary (from consolidation), may be None.
        created_at: When the memory was created.
        max_sentences_medium: Max sentences for 7-30 day tier.
        max_sentences_old: Max sentences for 30-90 day tier.

    Returns:
        Compressed content string.
    """
    if not content:
        return ""

    # Safe fallback: no timestamp = treat as recent
    if created_at is None:
        return content

    now = utcnow()
    age_days = (now - created_at).total_seconds() / 86400

    # Tier 1: Recent — full content
    if age_days < _FULL_CONTENT_DAYS:
        return content

    # Tier 2: Medium age — summary or truncated sentences
    if age_days < _SUMMARY_DAYS:
        if summary:
            return summary
        return _truncate_to_sentences(content, max_sentences_medium)

    # Tier 3: Old — summary or key sentences
    if age_days < _MINIMAL_DAYS:
        if summary:
            return summary
        return _truncate_to_sentences(content, max_sentences_old)

    # Tier 4: Very old — summary or first sentence only
    if summary:
        return summary
    return _truncate_to_sentences(content, 1)


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    """Extract first N sentences from text."""
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) <= max_sentences:
        return text
    return ". ".join(s.rstrip(".") for s in sentences[:max_sentences]) + "."


def _estimate_tokens(text: str) -> int:
    """Estimate LLM token count from text using word-based heuristic."""
    return int(len(text.split()) * _TOKEN_RATIO)


# Default token cost assumed for fibers that have no estimated_tokens in metadata.
_DEFAULT_FIBER_TOKENS = 50


@dataclass(frozen=True)
class BudgetResult:
    """Result of budget-aware fiber selection (greedy knapsack).

    Attributes:
        fibers_selected: Number of fibers selected within budget.
        fibers_skipped: Number of fibers excluded due to budget exhaustion.
        tokens_budget: The total token budget given.
        tokens_used: Estimated tokens used by selected fibers.
        tokens_remaining: Budget remaining after selection.
        selection_strategy: "optimal" (knapsack) or "sequential".
        skipped_summary: Up to 5 skipped fibers with cost/activation info.
    """

    fibers_selected: int
    fibers_skipped: int
    tokens_budget: int
    tokens_used: int
    tokens_remaining: int
    selection_strategy: str
    skipped_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for MCP response inclusion."""
        return {
            "fibers_selected": self.fibers_selected,
            "fibers_skipped": self.fibers_skipped,
            "tokens_budget": self.tokens_budget,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "selection_strategy": self.selection_strategy,
            "skipped_summary": list(self.skipped_summary),
        }


def select_within_budget(
    fibers: list[Any],  # list[Fiber]
    activations: dict[str, ActivationResult],
    budget: int,
) -> tuple[list[Any], BudgetResult]:
    """Select fibers using greedy knapsack (value-density ranking) within budget.

    Each fiber's value-density = activation_score / estimated_tokens_cost.
    Fibers are sorted by value-density descending; greedily selected while
    they fit within the remaining budget.

    Args:
        fibers: Candidate fibers (Fiber instances).
        activations: Map of neuron_id → ActivationResult (for scoring).
        budget: Total token budget available.

    Returns:
        Tuple of (selected_fibers, BudgetResult).
    """
    if not fibers or budget <= 0:
        return [], BudgetResult(
            fibers_selected=0,
            fibers_skipped=len(fibers),
            tokens_budget=budget,
            tokens_used=0,
            tokens_remaining=budget,
            selection_strategy="optimal",
            skipped_summary=[],
        )

    # Compute cost and activation score for each fiber.
    fiber_costs: list[tuple[Any, int, float]] = []  # (fiber, cost, activation)
    for fiber in fibers:
        cost = int(fiber.metadata.get("estimated_tokens", _DEFAULT_FIBER_TOKENS))
        cost = max(cost, 1)  # Guard against zero-cost fibers
        # Activation score: use anchor neuron if available, else max over neuron_ids
        activation: float = 0.0
        if fiber.anchor_neuron_id in activations:
            activation = activations[fiber.anchor_neuron_id].activation_level
        elif fiber.neuron_ids:
            for nid in fiber.neuron_ids:
                ar = activations.get(nid)
                if ar is not None:
                    activation = max(activation, ar.activation_level)
        fiber_costs.append((fiber, cost, activation))

    # Sort by value-density descending: higher activation per token wins.
    ranked = sorted(
        fiber_costs,
        key=lambda t: t[2] / t[1],  # activation / cost
        reverse=True,
    )

    selected: list[Any] = []
    skipped: list[Any] = []
    tokens_used = 0

    for fiber, cost, activation in ranked:
        if tokens_used + cost <= budget:
            selected.append(fiber)
            tokens_used += cost
        else:
            skipped.append((fiber, cost, activation))

    # Build skipped summary (top 5 skipped by activation)
    skipped_by_activation = sorted(skipped, key=lambda t: t[2], reverse=True)[:5]
    skipped_summary = [
        {
            "fiber_id": f.id,
            "estimated_tokens": c,
            "activation": round(a, 4),
        }
        for f, c, a in skipped_by_activation
    ]

    result = BudgetResult(
        fibers_selected=len(selected),
        fibers_skipped=len(skipped),
        tokens_budget=budget,
        tokens_used=tokens_used,
        tokens_remaining=budget - tokens_used,
        selection_strategy="optimal",
        skipped_summary=skipped_summary,
    )
    return selected, result


if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.engine.token_budget import BudgetAllocation, BudgetConfig
    from neural_memory.safety.encryption import MemoryEncryptor
    from neural_memory.storage.base import NeuralStorage


async def format_context_budgeted(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    fibers: list[Fiber],
    max_tokens: int,
    encryptor: MemoryEncryptor | None = None,
    brain_id: str = "",
    budget_config: BudgetConfig | None = None,
    clean_for_prompt: bool = False,
    query_terms: list[str] | None = None,
    compile: bool = True,
) -> tuple[str, int, BudgetAllocation]:
    """Format memories with budget-aware fiber selection.

    Unlike format_context() which processes fibers in order and truncates,
    this function selects the highest value-per-token fibers first, ensuring
    the most valuable memories fit within the token budget.

    When query_terms is provided and compile=True, runs cross-fiber
    deduplication, sentence-level merging, and query-relevance re-scoring
    via compile_context() before formatting.

    Args:
        storage: Neural storage backend.
        activations: Map of neuron_id -> ActivationResult for scoring.
        fibers: Candidate fibers from retrieval pipeline.
        max_tokens: Maximum tokens allowed in the formatted output.
        encryptor: Optional memory encryptor for decryption.
        brain_id: Active brain ID (for decryption).
        budget_config: Optional budget configuration overrides.
        clean_for_prompt: If True, emit clean bullet text without headers.
        query_terms: Query terms for relevance re-scoring and dedup.
            When None or empty, compilation is skipped (backward compat).
        compile: Master toggle for the compile step. Set False to disable
            compilation even when query_terms is provided.

    Returns:
        Tuple of (formatted_context, token_estimate, budget_allocation).
    """
    from neural_memory.engine.token_budget import (
        BudgetConfig,
        allocate_budget,
    )

    cfg = budget_config or BudgetConfig()

    # Pre-fetch anchor neurons to get real content for cost estimation
    anchor_ids = list({f.anchor_neuron_id for f in fibers if not f.summary})
    anchor_map = await storage.get_neurons_batch(anchor_ids) if anchor_ids else {}

    # Build costs using actual content (summary or anchor text)
    from neural_memory.engine.token_budget import TokenCost

    costs: list[TokenCost] = []
    for fiber in fibers[: cfg.max_fibers_considered]:
        if fiber.summary:
            content_text = fiber.summary
        else:
            anchor = anchor_map.get(fiber.anchor_neuron_id)
            content_text = anchor.content if anchor else ""

        from neural_memory.engine.token_budget import estimate_fiber_tokens

        content_tokens = estimate_fiber_tokens(content_text)
        metadata_tokens = cfg.per_fiber_overhead
        total_tokens = max(cfg.min_fiber_tokens, content_tokens + metadata_tokens)

        value_score: float = 0.0
        if fiber.anchor_neuron_id in activations:
            value_score = activations[fiber.anchor_neuron_id].activation_level
        elif fiber.neuron_ids:
            for nid in fiber.neuron_ids:
                ar = activations.get(nid)
                if ar is not None:
                    value_score = max(value_score, ar.activation_level)
        value_score = value_score + fiber.salience * 0.1 + fiber.conductivity * 0.05
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

    allocation = allocate_budget(costs, max_tokens, cfg)

    # Build set of selected fiber IDs for fast lookup
    selected_ids = {c.fiber_id for c in allocation.selected}

    # Filter fibers to only the budget-selected ones, preserving value order
    # Sort selected fibers by value_score so highest-value appear first
    cost_by_id = {c.fiber_id: c for c in allocation.selected}
    selected_fibers = sorted(
        [f for f in fibers if f.id in selected_ids],
        key=lambda f: cost_by_id[f.id].value_score,
        reverse=True,
    )

    # --- Compile step (dedup + merge + rescore) ---
    # Only runs when query_terms is provided AND compile=True AND cfg.enable_compiler.
    # Produces a content-override map {fiber_id: compiled_content} used during
    # formatting. Does NOT mutate fiber objects or add new storage calls.
    compiled_content: dict[str, str] = {}
    _should_compile = (
        compile and cfg.enable_compiler and bool(query_terms) and bool(selected_fibers)
    )
    if _should_compile:
        from neural_memory.engine.context_compiler import CompiledChunk, compile_context

        def _maybe_decrypt_inline(text: str, fiber_meta: dict[str, Any]) -> str:
            if encryptor and brain_id and fiber_meta.get("encrypted"):
                return encryptor.decrypt(text, brain_id)
            return text

        raw_chunks: list[CompiledChunk] = []
        for fiber in selected_fibers:
            if fiber.summary:
                raw_text = _maybe_decrypt_inline(fiber.summary, fiber.metadata)
            else:
                anchor = anchor_map.get(fiber.anchor_neuron_id)
                raw_text = _maybe_decrypt_inline(anchor.content, fiber.metadata) if anchor else ""
            if not raw_text:
                continue

            act_score: float = 0.0
            if fiber.anchor_neuron_id in activations:
                act_score = activations[fiber.anchor_neuron_id].activation_level
            elif fiber.neuron_ids:
                for nid in fiber.neuron_ids:
                    ar = activations.get(nid)
                    if ar is not None:
                        act_score = max(act_score, ar.activation_level)

            raw_chunks.append(
                CompiledChunk(
                    fiber_id=fiber.id,
                    content=raw_text,
                    activation_score=act_score,
                    created_at=fiber.created_at,
                    summary=fiber.summary,
                )
            )

        compiled = compile_context(
            raw_chunks,
            query_terms=list(query_terms or []),
            dedup_threshold=cfg.compiler_dedup_threshold,
        )
        # Build override map — only for fibers that survived compilation.
        # Fibers that were merged into another will simply be absent (format_context
        # will re-fetch them normally, but we reorder selected_fibers by compiled order).
        for chunk in compiled:
            compiled_content[chunk.fiber_id] = chunk.content

        # Re-order selected_fibers to match compiled ranking (highest final_score first).
        # Fibers not in compiled_content (merged away) are dropped.
        compiled_fiber_ids_ordered = [c.fiber_id for c in compiled]
        fiber_by_id = {f.id: f for f in selected_fibers}
        selected_fibers = [
            fiber_by_id[fid] for fid in compiled_fiber_ids_ordered if fid in fiber_by_id
        ]

    # Reuse existing format_context logic on the budget-selected subset.
    # When compiled_content is populated, format_context still fetches anchors but
    # the compiled text overrides raw anchor content for the affected fibers.
    formatted, token_estimate = await format_context(
        storage=storage,
        activations=activations,
        fibers=selected_fibers,
        max_tokens=max_tokens,
        encryptor=encryptor,
        brain_id=brain_id,
        clean_for_prompt=clean_for_prompt,
        _compiled_content=compiled_content if compiled_content else None,
    )

    return formatted, token_estimate, allocation


async def format_context(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    fibers: list[Fiber],
    max_tokens: int,
    encryptor: MemoryEncryptor | None = None,
    brain_id: str = "",
    clean_for_prompt: bool = False,
    _compiled_content: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Format activated memories into context for agent injection.

    Args:
        clean_for_prompt: If True, emit clean bullet-point text without
            section headers or neuron-type tags. Prevents self-referential
            noise when output is re-ingested by auto-capture.
        _compiled_content: Internal override map {fiber_id: content} produced
            by the compile step in format_context_budgeted. When a fiber_id
            is present here, its compiled content is used instead of the raw
            anchor/summary content. Not part of the public API.

    Returns:
        Tuple of (formatted_context, token_estimate).
    """

    def _maybe_decrypt(text: str, fiber_meta: dict[str, Any]) -> str:
        """Decrypt content if fiber is encrypted and encryptor is available."""
        if encryptor and brain_id and fiber_meta.get("encrypted"):
            return encryptor.decrypt(text, brain_id)
        return text

    lines: list[str] = []
    token_estimate = 0

    # Add fiber summaries first (batch fetch anchors)
    if fibers:
        if not clean_for_prompt:
            lines.append("## Relevant Memories\n")

        # Only fetch anchors for fibers that are NOT already covered by compiled_content
        anchor_ids = list(
            {
                f.anchor_neuron_id
                for f in fibers[:5]
                if not f.summary and (_compiled_content is None or f.id not in _compiled_content)
            }
        )
        anchor_map = await storage.get_neurons_batch(anchor_ids) if anchor_ids else {}

        for fiber in fibers[:5]:
            # Compiled content takes priority over raw summary/anchor
            if _compiled_content and fiber.id in _compiled_content:
                content = _compiled_content[fiber.id]
            elif fiber.summary:
                content = fiber.summary
            else:
                anchor = anchor_map.get(fiber.anchor_neuron_id)
                if anchor:
                    content = _maybe_decrypt(anchor.content, fiber.metadata)
                else:
                    continue

            # Age-based compression: older memories get compressed
            content = compress_for_recall(
                content,
                summary=fiber.summary,
                created_at=fiber.created_at,
            )

            # Format structured content if metadata has _structure
            content = _format_if_structured(content, fiber.metadata)

            # Truncate long content to fit within token budget
            remaining_budget = max_tokens - token_estimate
            if remaining_budget <= 0:
                break

            content_tokens = _estimate_tokens(content)
            if content_tokens > remaining_budget:
                # Truncate to fit: estimate words from remaining budget
                max_words = int(remaining_budget / _TOKEN_RATIO)
                if max_words < 10:
                    break
                words = content.split()
                content = " ".join(words[:max_words]) + "..."

            line = f"- {content}"
            token_estimate += _estimate_tokens(line)
            lines.append(line)

    # Add individual activated neurons (batch fetch)
    if token_estimate < max_tokens:
        if not clean_for_prompt:
            lines.append("\n## Related Information\n")

        sorted_activations = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )

        top_ids = [r.neuron_id for r in sorted_activations[:20]]
        neuron_map = await storage.get_neurons_batch(top_ids)

        for result in sorted_activations[:20]:
            neuron = neuron_map.get(result.neuron_id)
            if neuron is None:
                continue

            # Skip time neurons in context (they're implicit)
            if neuron.type == NeuronType.TIME:
                continue

            if clean_for_prompt:
                line = f"- {neuron.content}"
            else:
                line = f"- [{neuron.type.value}] {neuron.content}"
            token_estimate += _estimate_tokens(line)

            if token_estimate > max_tokens:
                break

            lines.append(line)

    return "\n".join(lines), token_estimate


def _format_if_structured(content: str, metadata: dict[str, Any]) -> str:
    """Format content using structure metadata if available.

    If the neuron/fiber has _structure metadata (set by StructureDetectionStep),
    re-format the content for readable output. Otherwise return as-is.
    """
    structure = metadata.get("_structure")
    if not structure or not isinstance(structure, dict):
        return content

    fmt = structure.get("format", "plain")
    if fmt == "plain":
        return content

    fields = structure.get("fields", [])
    if not fields:
        return content

    # Rebuild StructuredContent from stored metadata for formatting
    from neural_memory.extraction.structure_detector import (
        ContentFormat,
        StructuredContent,
        StructuredField,
        format_structured_output,
    )

    sc = StructuredContent(
        format=ContentFormat(fmt),
        fields=tuple(
            StructuredField(
                name=f.get("name", ""),
                value=f.get("value", ""),
                field_type=f.get("type", "text"),
            )
            for f in fields
        ),
        raw=content,
    )
    return format_structured_output(sc)
