"""Dream engine — random exploration for hidden connections.

Simulates dream-like exploration by selecting random neurons,
running spreading activation, and creating weak RELATED_TO
synapses between co-activated pairs that lack direct connections.

Dream synapses decay Nx faster than normal during pruning,
so only repeatedly reinforced connections survive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neural_memory.core.synapse import Synapse, SynapseType

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.neuron import Neuron
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class DreamResult:
    """Result of a dream exploration session.

    Attributes:
        synapses_created: New RELATED_TO synapses discovered
        pairs_explored: Number of neuron pairs evaluated
    """

    synapses_created: list[Synapse] = field(default_factory=list)
    pairs_explored: int = 0


async def dream(
    storage: NeuralStorage,
    config: BrainConfig,
    seed: int | None = None,
) -> DreamResult:
    """Run dream exploration to discover hidden connections.

    Selects random neurons, runs spreading activation from each,
    and creates weak RELATED_TO synapses between co-activated
    pairs that have no existing synapse.

    Args:
        storage: Storage backend
        config: Brain configuration
        seed: Optional random seed for deterministic tests

    Returns:
        DreamResult with discovered synapses
    """
    rng = random.Random(seed)

    # Get a pool of neurons to sample from
    all_neurons: list[Neuron] = await storage.find_neurons(limit=10000)
    if len(all_neurons) < 2:
        return DreamResult()

    # Select random seed neurons
    count = min(config.dream_neuron_count, len(all_neurons))
    seed_neurons = rng.sample(all_neurons, count)

    # Build set of existing synapse pairs for fast lookup
    all_synapses = await storage.get_synapses()
    existing_pairs: set[tuple[str, str]] = set()
    for syn in all_synapses:
        existing_pairs.add((syn.source_id, syn.target_id))
        existing_pairs.add((syn.target_id, syn.source_id))

    # Run spreading activation from each seed neuron
    from neural_memory.engine.activation import SpreadingActivation

    activation_engine = SpreadingActivation(storage, config)

    activated_ids: set[str] = set()
    for neuron in seed_neurons:
        try:
            results = await activation_engine.activate(
                anchor_neurons=[neuron.id],
                max_hops=2,
            )
            for result in results.values():
                activated_ids.add(result.neuron_id)
        except Exception:
            continue

    # Create pairs from activated neurons
    activated_list = list(activated_ids)
    pairs_explored = 0
    new_synapses: list[Synapse] = []

    for i in range(len(activated_list)):
        for j in range(i + 1, len(activated_list)):
            pairs_explored += 1
            a_id = activated_list[i]
            b_id = activated_list[j]

            if (a_id, b_id) in existing_pairs:
                continue

            synapse = Synapse.create(
                source_id=a_id,
                target_id=b_id,
                type=SynapseType.RELATED_TO,
                weight=0.1,
                metadata={"_dream": True},
            )
            new_synapses.append(synapse)
            existing_pairs.add((a_id, b_id))
            existing_pairs.add((b_id, a_id))

    return DreamResult(
        synapses_created=new_synapses,
        pairs_explored=pairs_explored,
    )
