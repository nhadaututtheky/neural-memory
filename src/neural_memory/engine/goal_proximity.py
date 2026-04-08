"""Goal-directed recall: BFS proximity scoring from active goal neurons.

Implements prefrontal cortex-style top-down attention modulation.
Memories topologically close to active goals get a scoring boost
during retrieval, making recall relevance-based instead of just
similarity-based.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType

if TYPE_CHECKING:
    from neural_memory.core.neuron import Neuron
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


async def find_active_goals(storage: NeuralStorage) -> list[Neuron]:
    """Find all neurons that are active goals.

    Queries INTENT neurons and filters by _goal_state == 'active'.

    Args:
        storage: Storage backend

    Returns:
        List of active goal neurons
    """
    try:
        intent_neurons = await storage.find_neurons(
            type=NeuronType.INTENT,
            limit=100,
        )
        return [n for n in intent_neurons if n.is_active_goal]
    except Exception:
        logger.debug("Failed to find active goals", exc_info=True)
        return []


async def compute_goal_proximity(
    storage: NeuralStorage,
    goal_neuron_ids: list[str],
    max_hops: int = 3,
) -> dict[str, float]:
    """Compute proximity scores from active goals via BFS.

    Runs BFS from each goal neuron up to max_hops. Returns a mapping
    of neuron_id -> proximity score where score = 1.0 / (min_hop + 1).

    Hop 0 (goal itself) = 1.0, hop 1 = 0.5, hop 2 = 0.33, hop 3 = 0.25.

    Args:
        storage: Storage backend for graph traversal
        goal_neuron_ids: IDs of active goal neurons
        max_hops: Maximum BFS depth (default 3)

    Returns:
        Dict mapping neuron_id to proximity score (0.0-1.0)
    """
    if not goal_neuron_ids:
        return {}

    # Cap to prevent unbounded BFS in brains with many active goals
    goal_neuron_ids = goal_neuron_ids[:10]

    # Track minimum hop distance per neuron across all goals
    min_distance: dict[str, int] = {}

    for goal_id in goal_neuron_ids:
        # BFS from this goal neuron
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((goal_id, 0))
        visited.add(goal_id)

        # Goal neuron itself is hop 0
        if goal_id not in min_distance or min_distance[goal_id] > 0:
            min_distance[goal_id] = 0

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_hops:
                continue

            try:
                neighbors = await storage.get_neighbors(
                    current_id,
                    direction="both",
                )
                for neighbor_neuron, _synapse in neighbors:
                    nid = neighbor_neuron.id
                    new_depth = depth + 1
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, new_depth))
                    # Update min distance
                    if nid not in min_distance or new_depth < min_distance[nid]:
                        min_distance[nid] = new_depth
            except Exception:
                logger.debug("BFS neighbor lookup failed for %s", current_id, exc_info=True)
                continue

    # Convert distances to proximity scores: 1.0 / (hop + 1)
    return {nid: 1.0 / (dist + 1) for nid, dist in min_distance.items()}
