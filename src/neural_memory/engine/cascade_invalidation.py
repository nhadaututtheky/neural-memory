"""Cascade staleness propagation through causal graphs.

When a neuron is superseded (marked _superseded=True), downstream neurons
connected via LEADS_TO/CAUSED_BY synapses may contain stale information.
This module propagates _stale markers through the causal graph using BFS.

Design:
- BFS from superseded neuron, following LEADS_TO outward (effects)
- Weight-gated: only follow synapses with weight >= min_weight
- Depth-limited: max 3 hops by default
- Marks both neurons and their containing fibers as stale
- Async-safe: designed to run via _write_queue (no recall latency impact)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.synapse import SynapseType

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeReport:
    """Result of a cascade staleness propagation.

    Attributes:
        neurons_marked: Number of neurons marked stale
        fibers_marked: Number of fibers marked stale
        depth_reached: Maximum depth traversed
    """

    neurons_marked: int
    fibers_marked: int
    depth_reached: int


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

_EFFECT_TYPES = [SynapseType.LEADS_TO]


async def cascade_staleness(
    storage: NeuralStorage,
    neuron_id: str,
    max_depth: int = 3,
    min_weight: float = 0.3,
) -> CascadeReport:
    """Propagate staleness from a superseded neuron through causal graph.

    Follows LEADS_TO synapses outward (effects of the superseded neuron)
    and marks downstream neurons and their fibers as stale.

    Args:
        storage: Storage backend (brain context must be set)
        neuron_id: The superseded neuron ID (starting point)
        max_depth: Maximum BFS depth (default 3)
        min_weight: Minimum synapse weight to follow (default 0.3)

    Returns:
        CascadeReport with counts of marked entities
    """
    neurons_marked = 0
    fibers_marked = 0
    max_depth_reached = 0

    visited: set[str] = {neuron_id}
    queue: deque[tuple[str, int]] = deque([(neuron_id, 0)])
    stale_fiber_ids: set[str] = set()

    while queue:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        try:
            neighbors = await storage.get_neighbors(
                current_id,
                direction="out",
                synapse_types=_EFFECT_TYPES,
                min_weight=min_weight,
            )
        except Exception:
            logger.debug(
                "Cascade: get_neighbors failed for %s (non-critical)",
                current_id,
                exc_info=True,
            )
            continue

        for neuron, _synapse in neighbors:
            if neuron.id in visited:
                continue
            visited.add(neuron.id)

            # Grounded neurons are truth anchors — never cascade stale
            if neuron.metadata.get("_grounded"):
                continue

            next_depth = depth + 1
            max_depth_reached = max(max_depth_reached, next_depth)

            # Mark neuron as stale
            try:
                updated = neuron.with_metadata(_stale=True)
                await storage.update_neuron(updated)
                neurons_marked += 1
            except Exception:
                logger.debug(
                    "Cascade: update_neuron failed for %s",
                    neuron.id,
                    exc_info=True,
                )

            # Find and mark containing fibers
            try:
                fibers = await storage.find_fibers(contains_neuron=neuron.id, limit=5)
                for fiber in fibers:
                    if fiber.id not in stale_fiber_ids:
                        stale_fiber_ids.add(fiber.id)
                        meta = dict(fiber.metadata or {})
                        meta["_stale"] = True
                        await storage.update_fiber_metadata(fiber.id, meta)
                        fibers_marked += 1
            except Exception:
                logger.debug(
                    "Cascade: fiber marking failed for neuron %s",
                    neuron.id,
                    exc_info=True,
                )

            queue.append((neuron.id, next_depth))

    if neurons_marked > 0 or fibers_marked > 0:
        logger.info(
            "Cascade staleness from %s: %d neurons, %d fibers marked (depth %d)",
            neuron_id,
            neurons_marked,
            fibers_marked,
            max_depth_reached,
        )

    return CascadeReport(
        neurons_marked=neurons_marked,
        fibers_marked=fibers_marked,
        depth_reached=max_depth_reached,
    )
