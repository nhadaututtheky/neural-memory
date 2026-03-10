"""Graph-based query expansion via 1-hop neighbor traversal.

After finding entity/concept anchors, fetches their immediate graph
neighbors as soft anchors. This exploits the knowledge graph structure
to expand recall — e.g., querying "auth" finds ENTITY "OAuth2" which
expands to "JWT", "refresh_token", "session" via synapses.

This mimics associative priming in the human brain: activating a concept
pre-activates related concepts, making them easier to retrieve.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.score_fusion import RankedAnchor

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Neuron types worth expanding from (not TIME, not generic KEYWORD)
_EXPANDABLE_TYPES = frozenset({NeuronType.ENTITY, NeuronType.CONCEPT})


async def expand_via_graph(
    storage: NeuralStorage,
    seed_neuron_ids: list[str],
    max_expansions: int = 10,
    min_synapse_weight: float = 0.3,
) -> tuple[list[str], list[RankedAnchor]]:
    """Fetch 1-hop neighbors of seed neurons as expansion candidates.

    Only expands from ENTITY and CONCEPT neurons to avoid noisy
    expansion from TIME or generic content neurons.

    Args:
        storage: Storage backend for graph queries.
        seed_neuron_ids: Neuron IDs to expand from (typically entity anchors).
        max_expansions: Maximum number of expansion neurons to return.
        min_synapse_weight: Minimum synapse weight to follow.

    Returns:
        Tuple of (expansion neuron IDs, ranked anchors for RRF).
    """
    if not seed_neuron_ids:
        return [], []

    # Filter to expandable neuron types
    seed_neurons = await storage.get_neurons_batch(seed_neuron_ids)
    expandable_ids = [
        nid for nid, neuron in seed_neurons.items() if neuron.type in _EXPANDABLE_TYPES
    ]

    if not expandable_ids:
        return [], []

    # Batch-fetch outgoing synapses for all expandable seeds
    synapses_map = await storage.get_synapses_for_neurons(expandable_ids, direction="out")

    # Collect candidate neighbors with their best synapse weight
    seen_seeds = set(seed_neuron_ids)
    candidates: dict[str, float] = {}  # neuron_id -> best synapse weight

    for synapses in synapses_map.values():
        for synapse in synapses:
            target_id = synapse.target_id
            if target_id in seen_seeds:
                continue
            if synapse.weight < min_synapse_weight:
                continue
            current_best = candidates.get(target_id, 0.0)
            if synapse.weight > current_best:
                candidates[target_id] = synapse.weight

    if not candidates:
        return [], []

    # Filter out TIME neurons from expansion targets
    candidate_ids = list(candidates.keys())
    candidate_neurons = await storage.get_neurons_batch(candidate_ids)
    filtered: list[tuple[str, float]] = []
    for nid, weight in candidates.items():
        neuron = candidate_neurons.get(nid)
        if neuron is not None and neuron.type != NeuronType.TIME:
            filtered.append((nid, weight))

    # Sort by synapse weight (strongest connections first), cap at max
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:max_expansions]

    expansion_ids = [nid for nid, _ in filtered]
    ranked = [
        RankedAnchor(
            neuron_id=nid,
            rank=i + 1,
            retriever="graph_expansion",
            score=weight,
        )
        for i, (nid, weight) in enumerate(filtered)
    ]

    return expansion_ids, ranked
