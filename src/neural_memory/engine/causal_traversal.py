"""Causal and temporal traversal engine.

Provides three traversal capabilities for temporal reasoning:
1. Causal chain tracing — follow CAUSED_BY/LEADS_TO synapses
2. Temporal range queries — retrieve fibers within a time window
3. Event sequence tracing — follow BEFORE/AFTER synapses

All traversals use BFS with visited sets for cycle prevention.
No LLM dependency — pure graph traversal.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from neural_memory.core.synapse import SynapseType

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.storage.base import NeuralStorage


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalStep:
    """A single step in a causal chain.

    Attributes:
        neuron_id: ID of the neuron at this step
        content: Text content of the neuron
        synapse_type: The synapse type traversed to reach this step
        weight: Weight of the synapse connecting to this step
        depth: Number of hops from seed (0 = immediate neighbor)
    """

    neuron_id: str
    content: str
    synapse_type: SynapseType
    weight: float
    depth: int


@dataclass(frozen=True)
class CausalChain:
    """Result of tracing a causal chain through the graph.

    Attributes:
        seed_neuron_id: The starting neuron ID
        direction: "causes" (tracing what caused the seed)
                   or "effects" (tracing what the seed caused)
        steps: Ordered tuple of causal steps from seed outward
        total_weight: Product of all step weights (chain confidence)
    """

    seed_neuron_id: str
    direction: str
    steps: tuple[CausalStep, ...]
    total_weight: float


@dataclass(frozen=True)
class EventStep:
    """A single event in a temporal sequence.

    Attributes:
        neuron_id: ID of the neuron representing this event
        content: Text content of the event
        fiber_id: ID of the fiber containing this event (if found)
        timestamp: When this event occurred (from fiber, if available)
        position: 0-indexed position in the sequence
    """

    neuron_id: str
    content: str
    fiber_id: str | None
    timestamp: datetime | None
    position: int


@dataclass(frozen=True)
class EventSequence:
    """Result of tracing a temporal event sequence.

    Attributes:
        seed_neuron_id: The starting neuron ID
        direction: "forward" (what happened next)
                   or "backward" (what happened before)
        events: Ordered tuple of events
    """

    seed_neuron_id: str
    direction: str
    events: tuple[EventStep, ...]


# ---------------------------------------------------------------------------
# Traversal functions
# ---------------------------------------------------------------------------

_CAUSE_TYPES = frozenset({SynapseType.CAUSED_BY})
_EFFECT_TYPES = frozenset({SynapseType.LEADS_TO})
_FORWARD_TYPES = frozenset({SynapseType.BEFORE})
_BACKWARD_TYPES = frozenset({SynapseType.AFTER})


async def trace_causal_chain(
    storage: NeuralStorage,
    seed_neuron_id: str,
    direction: Literal["causes", "effects"],
    max_depth: int = 5,
    min_weight: float = 0.1,
) -> CausalChain:
    """Trace a causal chain through CAUSED_BY/LEADS_TO synapses.

    For "causes": follows CAUSED_BY outward from seed — the seed is the
    effect, and we discover what caused it. Each step moves to the cause.

    For "effects": follows LEADS_TO outward from seed — the seed is the
    cause, and we discover what it led to.

    Uses BFS to find all reachable causal neighbors up to max_depth.
    Cycle detection via visited set prevents infinite loops.

    Args:
        storage: Storage backend (brain context must be set)
        seed_neuron_id: Starting neuron ID
        direction: "causes" to trace what caused the seed,
                   "effects" to trace what the seed caused
        max_depth: Maximum traversal depth (default 5)
        min_weight: Minimum synapse weight to follow (default 0.1)

    Returns:
        CausalChain with ordered steps and aggregate confidence
    """
    synapse_types = list(_CAUSE_TYPES if direction == "causes" else _EFFECT_TYPES)

    steps: list[CausalStep] = []
    visited: set[str] = {seed_neuron_id}
    queue: deque[tuple[str, int]] = deque([(seed_neuron_id, 0)])

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= max_depth:
            continue

        neighbors = await storage.get_neighbors(
            current_id,
            direction="out",
            synapse_types=synapse_types,
            min_weight=min_weight,
        )

        for neuron, synapse in neighbors:
            if neuron.id in visited:
                continue
            visited.add(neuron.id)

            step = CausalStep(
                neuron_id=neuron.id,
                content=neuron.content,
                synapse_type=synapse.type,
                weight=synapse.weight,
                depth=current_depth,
            )
            steps.append(step)
            queue.append((neuron.id, current_depth + 1))

    total_weight = math.prod(s.weight for s in steps) if steps else 0.0

    return CausalChain(
        seed_neuron_id=seed_neuron_id,
        direction=direction,
        steps=tuple(steps),
        total_weight=total_weight,
    )


async def query_temporal_range(
    storage: NeuralStorage,
    start: datetime,
    end: datetime,
    limit: int = 50,
) -> list[Fiber]:
    """Retrieve fibers within a temporal range, ordered chronologically.

    Uses the storage's time overlap query to find fibers whose time window
    intersects [start, end]. Results are sorted by time_start ascending.

    Args:
        storage: Storage backend (brain context must be set)
        start: Start of the time range
        end: End of the time range
        limit: Maximum fibers to return (default 50)

    Returns:
        List of Fiber objects sorted chronologically by time_start
    """
    fibers = await storage.find_fibers(time_overlaps=(start, end), limit=limit)

    # Filter out fibers with no temporal bounds and sort chronologically
    temporal_fibers = [f for f in fibers if f.time_start is not None]
    temporal_fibers.sort(key=lambda f: f.time_start)  # type: ignore[arg-type, return-value]

    return temporal_fibers


async def trace_event_sequence(
    storage: NeuralStorage,
    seed_neuron_id: str,
    direction: Literal["forward", "backward"],
    max_steps: int = 10,
    min_weight: float = 0.1,
) -> EventSequence:
    """Trace an event sequence through BEFORE/AFTER synapses.

    For "forward": follows BEFORE outward — X BEFORE Y means X happened
    first, so following BEFORE from seed finds what happened next.

    For "backward": follows AFTER outward — finds what happened before.

    Each discovered event is enriched with fiber membership and timestamp
    when available.

    Args:
        storage: Storage backend (brain context must be set)
        seed_neuron_id: Starting neuron ID
        direction: "forward" to find subsequent events,
                   "backward" to find preceding events
        max_steps: Maximum events to discover (default 10)
        min_weight: Minimum synapse weight to follow (default 0.1)

    Returns:
        EventSequence with ordered events
    """
    synapse_types = list(_FORWARD_TYPES if direction == "forward" else _BACKWARD_TYPES)

    events: list[EventStep] = []
    visited: set[str] = {seed_neuron_id}
    queue: deque[str] = deque([seed_neuron_id])
    position = 0

    while queue and position < max_steps:
        current_id = queue.popleft()

        neighbors = await storage.get_neighbors(
            current_id,
            direction="out",
            synapse_types=synapse_types,
            min_weight=min_weight,
        )

        for neuron, _synapse in neighbors:
            if neuron.id in visited:
                continue
            if position >= max_steps:
                break

            visited.add(neuron.id)

            # Try to find fiber membership for timestamp
            fiber_id, timestamp = await _get_fiber_info(storage, neuron.id)

            event = EventStep(
                neuron_id=neuron.id,
                content=neuron.content,
                fiber_id=fiber_id,
                timestamp=timestamp,
                position=position,
            )
            events.append(event)
            queue.append(neuron.id)
            position += 1

    return EventSequence(
        seed_neuron_id=seed_neuron_id,
        direction=direction,
        events=tuple(events),
    )


async def _get_fiber_info(
    storage: NeuralStorage,
    neuron_id: str,
) -> tuple[str | None, datetime | None]:
    """Look up fiber membership and timestamp for a neuron.

    Returns:
        Tuple of (fiber_id, timestamp) where timestamp is the fiber's
        time_start. Both may be None if no fiber contains this neuron.
    """
    fibers = await storage.find_fibers(contains_neuron=neuron_id, limit=1)
    if not fibers:
        return None, None

    fiber = fibers[0]
    return fiber.id, fiber.time_start
