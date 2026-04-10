"""Data types for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from neural_memory.engine.reflex_activation import CoActivation

if TYPE_CHECKING:
    from neural_memory.engine.confidence import ConfidenceScore


class DepthLevel(IntEnum):
    """
    Depth levels for retrieval queries.

    Higher depth = more exploration but slower retrieval.
    """

    INSTANT = 0  # Who, where, what (1 hop) - Simple fact retrieval
    CONTEXT = 1  # Before/after (2-3 hops) - Contextual information
    HABIT = 2  # Patterns (cross-time) - Recurring patterns
    DEEP = 3  # Emotions, causality (full) - Deep analysis


@dataclass
class Subgraph:
    """
    Extracted subgraph from activation.

    Attributes:
        neuron_ids: IDs of neurons in the subgraph
        synapse_ids: IDs of synapses connecting neurons
        anchor_ids: IDs of the anchor neurons that started activation
    """

    neuron_ids: list[str]
    synapse_ids: list[str]
    anchor_ids: list[str]


@dataclass(frozen=True)
class ScoreBreakdown:
    """Breakdown of confidence score components.

    Exposes why a memory ranked high so users can understand retrieval decisions.

    Attributes:
        base_activation: Raw activation level from spreading activation
        intersection_boost: Bonus from neurons reached via multiple anchor sets
        freshness_boost: Bonus for recently accessed memories
        frequency_boost: Bonus for frequently accessed memories
        raw_total: Sum of all components before clamping to [0, 1]
    """

    base_activation: float
    intersection_boost: float
    freshness_boost: float
    frequency_boost: float
    emotional_resonance: float
    decision_domain_boost: float = 0.0
    raw_total: float = 0.0


@dataclass
class RetrievalResult:
    """
    Result of a retrieval query.

    Attributes:
        answer: Reconstructed answer text (if determinable)
        confidence: Confidence in the answer (0.0 - 1.0)
        depth_used: Which depth level was used
        neurons_activated: Number of neurons that were activated
        fibers_matched: IDs of fibers that matched the query
        subgraph: The extracted relevant subgraph
        context: Formatted context for injection into agent prompts
        latency_ms: Time taken for retrieval in milliseconds
        co_activations: Neurons that co-activated (Hebbian binding)
        metadata: Additional retrieval metadata
    """

    answer: str | None
    confidence: float
    depth_used: DepthLevel
    neurons_activated: int
    fibers_matched: list[str]
    subgraph: Subgraph
    context: str
    latency_ms: float
    tokens_used: int = 0
    co_activations: list[CoActivation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score_breakdown: ScoreBreakdown | None = None
    contributing_neurons: list[str] = field(default_factory=list)
    synthesis_method: str = "single"
    confidence_score: ConfidenceScore | None = None
