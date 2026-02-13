"""Composable async pipeline framework for memory encoding.

Extracts the monolithic encode() method into discrete, testable steps
that flow through a shared PipelineContext. Each step implements the
PipelineStep protocol and can be composed/reordered/skipped.

The pipeline is backward-compatible: MemoryEncoder.encode() delegates
to Pipeline.run() internally, preserving the same public API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Synapse

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state flowing through pipeline steps.

    Each step reads from and returns a *new* PipelineContext (immutability
    is maintained by returning updated copies from steps).  In practice,
    steps mutate lists in-place for performance since the context is
    short-lived and not shared across threads.
    """

    # ── Input ──
    content: str
    timestamp: datetime
    metadata: dict[str, Any]
    tags: set[str]
    language: str

    # ── Step flags (set by caller) ──
    skip_conflicts: bool = False
    skip_time_neurons: bool = False
    initial_stage: str = ""
    salience_ceiling: float = 0.0

    # ── Accumulated results (mutated by steps) ──
    neurons_created: list[Neuron] = field(default_factory=list)
    neurons_linked: list[str] = field(default_factory=list)
    synapses_created: list[Synapse] = field(default_factory=list)
    conflicts_detected: int = 0

    # ── Inter-step state ──
    anchor_neuron: Neuron | None = None
    content_hash: int = 0
    time_neurons: list[Neuron] = field(default_factory=list)
    entity_neurons: list[Neuron] = field(default_factory=list)
    concept_neurons: list[Neuron] = field(default_factory=list)
    auto_tags: set[str] = field(default_factory=set)
    agent_tags: set[str] = field(default_factory=set)
    merged_tags: set[str] = field(default_factory=set)
    effective_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PipelineStep(Protocol):
    """Protocol for a single pipeline step."""

    @property
    def name(self) -> str:
        """Human-readable step name for logging."""
        ...

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        """Execute this step, returning the (possibly mutated) context."""
        ...


class Pipeline:
    """Composable async pipeline that runs steps sequentially.

    Usage::

        pipeline = Pipeline([
            ExtractTimeNeuronsStep(...),
            ExtractEntityNeuronsStep(...),
            ...
        ])
        result_ctx = await pipeline.run(ctx, storage, config)
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self._steps = list(steps)

    @property
    def steps(self) -> list[PipelineStep]:
        """Read-only view of registered steps."""
        return list(self._steps)

    async def run(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        """Run all steps sequentially, passing context through each."""
        for step in self._steps:
            logger.debug("Pipeline step: %s", step.name)
            ctx = await step.execute(ctx, storage, config)
        return ctx
