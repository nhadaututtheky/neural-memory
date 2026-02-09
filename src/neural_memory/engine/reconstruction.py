"""Multi-neuron answer reconstruction.

Replaces the simple top-1 neuron extraction with strategy-based
answer synthesis from the activated subgraph.

Strategy selection based on activation pattern:
1. Single mode: dominant neuron with high confidence → return its content
2. Fiber-summary mode: best fiber has a summary → return summary
3. Multi-neuron mode: top-N neurons sorted by pathway position

This mimics pattern completion in biological memory — the brain
doesn't retrieve a single memory cell, it reconstructs from
multiple activated neurons forming a coherent pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.retrieval_types import ScoreBreakdown
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.engine.causal_traversal import CausalChain, EventSequence
    from neural_memory.storage.base import NeuralStorage


class SynthesisMethod(StrEnum):
    """How the answer was reconstructed."""

    SINGLE = "single"  # Top neuron content
    FIBER_SUMMARY = "fiber_summary"  # Fiber summary text
    MULTI_NEURON = "multi_neuron"  # Multiple neurons combined
    CAUSAL_CHAIN = "causal_chain"  # Causal traversal chain
    TEMPORAL_SEQUENCE = "temporal_sequence"  # Temporal range or event sequence
    NONE = "none"  # No answer found


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of answer reconstruction.

    Attributes:
        answer: Reconstructed answer text
        confidence: Confidence score [0, 1]
        method: Which synthesis strategy was used
        contributing_neuron_ids: IDs of neurons that contributed to the answer
        score_breakdown: Detailed score components
    """

    answer: str | None
    confidence: float
    method: SynthesisMethod
    contributing_neuron_ids: list[str]
    score_breakdown: ScoreBreakdown | None


async def reconstruct_answer(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    intersections: list[str],
    fibers: list[Fiber],
    max_contributing: int = 5,
) -> ReconstructionResult:
    """Reconstruct an answer from the activated subgraph.

    Strategy selection:
    1. If top intersection neuron has confidence > 0.8 → single mode
    2. If best fiber has a summary → fiber-summary mode
    3. Otherwise → multi-neuron mode (top-N non-TIME neurons)

    Args:
        storage: Storage backend for neuron content lookup
        activations: Stabilized activation results
        intersections: Intersection neuron IDs (from multiple anchor sets)
        fibers: Matched fibers from retrieval
        max_contributing: Max neurons for multi-neuron synthesis

    Returns:
        ReconstructionResult with answer and metadata
    """
    if not activations:
        return ReconstructionResult(
            answer=None,
            confidence=0.0,
            method=SynthesisMethod.NONE,
            contributing_neuron_ids=[],
            score_breakdown=None,
        )

    # Build scored candidates (intersection neurons get 1.5x boost)
    candidates = _score_candidates(activations, intersections)

    if not candidates:
        return ReconstructionResult(
            answer=None,
            confidence=0.0,
            method=SynthesisMethod.NONE,
            contributing_neuron_ids=[],
            score_breakdown=None,
        )

    top_id, top_score = candidates[0]
    breakdown = await _compute_score_breakdown(
        storage,
        top_id,
        top_score,
        intersections,
    )

    # Strategy 1: Single mode — dominant neuron
    if top_score > 0.8:
        neuron = await storage.get_neuron(top_id)
        if neuron is not None:
            return ReconstructionResult(
                answer=neuron.content,
                confidence=min(1.0, breakdown.raw_total),
                method=SynthesisMethod.SINGLE,
                contributing_neuron_ids=[top_id],
                score_breakdown=breakdown,
            )

    # Strategy 2: Fiber-summary mode — best fiber has a summary
    for fiber in fibers:
        if fiber.summary:
            return ReconstructionResult(
                answer=fiber.summary,
                confidence=min(1.0, breakdown.raw_total),
                method=SynthesisMethod.FIBER_SUMMARY,
                contributing_neuron_ids=[top_id],
                score_breakdown=breakdown,
            )

    # Strategy 3: Multi-neuron mode — combine top-N neurons
    return await _multi_neuron_reconstruct(
        storage,
        candidates,
        fibers,
        max_contributing,
        breakdown,
    )


def _score_candidates(
    activations: dict[str, ActivationResult],
    intersections: list[str],
) -> list[tuple[str, float]]:
    """Score and rank candidate neurons for reconstruction.

    Intersection neurons get a 1.5x score boost since they were
    reached from multiple anchor sets (higher relevance).
    """
    candidates: list[tuple[str, float]] = []

    for neuron_id in intersections:
        if neuron_id in activations:
            candidates.append(
                (
                    neuron_id,
                    activations[neuron_id].activation_level * 1.5,
                )
            )

    intersection_set = set(intersections)
    for neuron_id, result in activations.items():
        if neuron_id not in intersection_set:
            candidates.append((neuron_id, result.activation_level))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


async def _compute_score_breakdown(
    storage: NeuralStorage,
    top_id: str,
    top_score: float,
    intersections: list[str],
) -> ScoreBreakdown:
    """Compute multi-factor score breakdown for the top neuron."""
    base_confidence = min(1.0, top_score)
    intersection_boost = 0.05 * len(intersections) if intersections else 0.0

    freshness_boost = 0.0
    frequency_boost = 0.0
    top_state = await storage.get_neuron_state(top_id)
    if top_state:
        if top_state.last_activated:
            hours_since = (utcnow() - top_state.last_activated).total_seconds() / 3600
            freshness_boost = 0.15 / (1.0 + math.exp((hours_since - 72) / 36))
        if top_state.access_frequency > 0:
            frequency_boost = min(0.1, 0.03 * math.log1p(top_state.access_frequency))

    # Emotional resonance: check for FELT synapses on the top neuron
    emotional_resonance = 0.0
    felt_synapses = await storage.get_synapses(source_id=top_id, type=SynapseType.FELT)
    if felt_synapses:
        best_intensity = max(s.metadata.get("_intensity", 0.5) for s in felt_synapses)
        emotional_resonance = min(0.1, 0.05 * best_intensity)

    raw_total = (
        base_confidence
        + intersection_boost
        + freshness_boost
        + frequency_boost
        + emotional_resonance
    )
    return ScoreBreakdown(
        base_activation=base_confidence,
        intersection_boost=intersection_boost,
        freshness_boost=freshness_boost,
        frequency_boost=frequency_boost,
        emotional_resonance=emotional_resonance,
        raw_total=raw_total,
    )


async def _multi_neuron_reconstruct(
    storage: NeuralStorage,
    candidates: list[tuple[str, float]],
    fibers: list[Fiber],
    max_contributing: int,
    breakdown: ScoreBreakdown,
) -> ReconstructionResult:
    """Reconstruct answer from multiple neurons.

    Takes top-N non-TIME neurons, orders them by pathway position
    in the best fiber (if available), then concatenates content.
    """
    # Collect top non-TIME neuron IDs
    top_ids = [nid for nid, _ in candidates[: max_contributing * 2]]
    neuron_map = await storage.get_neurons_batch(top_ids)

    # Filter out TIME neurons and missing neurons
    content_neurons = [
        (nid, neuron_map[nid])
        for nid in top_ids
        if nid in neuron_map and neuron_map[nid].type != NeuronType.TIME
    ][:max_contributing]

    if not content_neurons:
        return ReconstructionResult(
            answer=None,
            confidence=0.0,
            method=SynthesisMethod.NONE,
            contributing_neuron_ids=[],
            score_breakdown=breakdown,
        )

    # If we have a fiber with a pathway, sort neurons by pathway order
    if fibers:
        best_fiber = fibers[0]
        if best_fiber.pathway:
            pathway_index = {nid: idx for idx, nid in enumerate(best_fiber.pathway)}
            content_neurons.sort(
                key=lambda pair: pathway_index.get(pair[0], 999),
            )

    contributing_ids = [nid for nid, _ in content_neurons]
    parts = [neuron.content for _, neuron in content_neurons]
    answer = "; ".join(parts)

    return ReconstructionResult(
        answer=answer,
        confidence=min(1.0, breakdown.raw_total),
        method=SynthesisMethod.MULTI_NEURON,
        contributing_neuron_ids=contributing_ids,
        score_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Temporal reasoning formatters (v0.19.0)
# ---------------------------------------------------------------------------


def format_causal_chain(chain: CausalChain) -> str:
    """Format a causal chain as human-readable text.

    For "causes" direction: "A because B because C"
    For "effects" direction: "A leads to B leads to C"

    Args:
        chain: The causal chain to format

    Returns:
        Formatted string, or empty string if chain has no steps
    """
    if not chain.steps:
        return ""

    connector = " because " if chain.direction == "causes" else " leads to "
    return connector.join(step.content for step in chain.steps)


def format_event_sequence(sequence: EventSequence) -> str:
    """Format an event sequence as chronological text.

    Output: "First, A; then B; then C"
    Timestamps included when available.

    Args:
        sequence: The event sequence to format

    Returns:
        Formatted string, or empty string if sequence has no events
    """
    if not sequence.events:
        return ""

    parts: list[str] = []
    for i, event in enumerate(sequence.events):
        prefix = "First, " if i == 0 else "then "
        time_suffix = ""
        if event.timestamp:
            time_suffix = f" ({event.timestamp.strftime('%Y-%m-%d %H:%M')})"
        parts.append(f"{prefix}{event.content}{time_suffix}")

    return "; ".join(parts)


def format_temporal_range(fibers: list[Fiber]) -> str:
    """Format temporally-ranged fibers as a chronological summary.

    Output: one line per fiber with timestamp and summary.

    Args:
        fibers: Fibers sorted chronologically by time_start

    Returns:
        Formatted string, or empty string if no fibers
    """
    if not fibers:
        return ""

    parts: list[str] = []
    for fiber in fibers:
        time_label = ""
        if fiber.time_start:
            time_label = f"[{fiber.time_start.strftime('%Y-%m-%d %H:%M')}] "
        summary = fiber.summary or f"Memory (fiber {fiber.id[:8]})"
        parts.append(f"- {time_label}{summary}")

    return "\n".join(parts)
