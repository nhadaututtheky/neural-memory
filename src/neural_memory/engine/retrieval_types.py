"""Data types for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from neural_memory.engine.reflex_activation import CoActivation


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
