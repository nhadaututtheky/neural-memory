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


def _effective_priority(
    goal_id: str,
    priorities: dict[str, int],
    parent_map: dict[str, str | None],
) -> float:
    """Compute effective priority with one-level parent inheritance.

    If a subgoal's own priority is lower than parent * 0.8, inherit
    the boosted value. Propagates one level only (no recursion).

    Returns priority weight as float (priority / 10.0).
    """
    own = priorities.get(goal_id, 5)
    parent_id = parent_map.get(goal_id)
    if parent_id and parent_id in priorities:
        inherited = priorities[parent_id] * 0.8
        effective = max(own, inherited)
    else:
        effective = float(own)
    return min(effective, 10.0) / 10.0


async def compute_goal_proximity(
    storage: NeuralStorage,
    goal_neuron_ids: list[str],
    max_hops: int = 3,
    goal_priorities: dict[str, int] | None = None,
    parent_map: dict[str, str | None] | None = None,
) -> dict[str, float]:
    """Compute proximity scores from active goals via BFS.

    Runs BFS from each goal neuron up to max_hops. Returns a mapping
    of neuron_id -> proximity score. When multiple goals compete for the
    same neuron, the highest priority-weighted score wins (conflict resolution).

    Base proximity: 1.0 / (min_hop + 1).
    Priority weight: effective_priority / 10.0, with one-level parent inheritance.
    Final score: base_proximity * priority_weight.

    Args:
        storage: Storage backend for graph traversal
        goal_neuron_ids: IDs of active goal neurons
        max_hops: Maximum BFS depth (default 3)
        goal_priorities: Optional mapping of goal_id -> priority (1-10).
            When provided, higher-priority goals produce stronger proximity.
        parent_map: Optional mapping of goal_id -> parent_goal_id for
            priority inheritance (subgoal inherits parent * 0.8 if higher).

    Returns:
        Dict mapping neuron_id to proximity score (0.0-1.0)
    """
    if not goal_neuron_ids:
        return {}

    # Cap to prevent unbounded BFS in brains with many active goals
    goal_neuron_ids = goal_neuron_ids[:10]
    _priorities = goal_priorities or {}
    _parents = parent_map or {}

    # Track best weighted score per neuron across all goals
    best_score: dict[str, float] = {}

    for goal_id in goal_neuron_ids:
        priority_weight = _effective_priority(goal_id, _priorities, _parents)

        # BFS from this goal neuron
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((goal_id, 0))
        visited.add(goal_id)

        while queue:
            current_id, depth = queue.popleft()

            # Score for current node
            base_proximity = 1.0 / (depth + 1)
            weighted = base_proximity * priority_weight
            if current_id not in best_score or weighted > best_score[current_id]:
                best_score[current_id] = weighted

            if depth >= max_hops:
                continue

            try:
                neighbors = await storage.get_neighbors(
                    current_id,
                    direction="both",
                )
                for neighbor_neuron, _synapse in neighbors:
                    nid = neighbor_neuron.id
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, depth + 1))
            except Exception:
                logger.debug("BFS neighbor lookup failed for %s", current_id, exc_info=True)
                continue

    return {nid: round(score, 4) for nid, score in best_score.items()}
