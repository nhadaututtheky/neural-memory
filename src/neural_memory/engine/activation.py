"""Spreading activation algorithm for memory retrieval."""

from __future__ import annotations

import asyncio
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

# Safety cap: maximum queue entries to prevent memory exhaustion on dense graphs
_MAX_QUEUE_SIZE = 50_000


@dataclass
class ActivationResult:
    """
    Result of activating a neuron through spreading activation.

    Attributes:
        neuron_id: The activated neuron's ID
        activation_level: Final activation level (0.0 - 1.0)
        hop_distance: Number of hops from the nearest anchor
        path: List of neuron IDs showing how we reached this neuron
        source_anchor: The anchor neuron that led to this activation
    """

    neuron_id: str
    activation_level: float
    hop_distance: int
    path: list[str]
    source_anchor: str


@dataclass
class ActivationState:
    """Internal state during activation spreading."""

    neuron_id: str
    level: float
    hops: int
    path: list[str]
    source: str

    def __lt__(self, other: ActivationState) -> bool:
        """For heap ordering (higher activation = higher priority)."""
        return self.level > other.level


class SpreadingActivation:
    """
    Spreading activation algorithm for neural memory retrieval.

    This implements the core retrieval mechanism: starting from
    anchor neurons and spreading activation through synapses,
    decaying with distance, to find related memories.
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> None:
        """
        Initialize the activation system.

        Args:
            storage: Storage backend to read graph from
            config: Brain configuration for parameters
        """
        self._storage = storage
        self._config = config

    async def activate(
        self,
        anchor_neurons: list[str],
        max_hops: int | None = None,
        decay_factor: float = 0.5,
        min_activation: float | None = None,
    ) -> dict[str, ActivationResult]:
        """
        Spread activation from anchor neurons through the graph.

        The activation spreads through synapses, with the level
        decaying at each hop:
            activation(hop) = initial * decay_factor^hop * synapse_weight

        Args:
            anchor_neurons: Starting neurons with activation = 1.0
            max_hops: Maximum number of hops (default: from config)
            decay_factor: How much activation decays per hop
            min_activation: Minimum activation to continue spreading

        Returns:
            Dict mapping neuron_id to ActivationResult
        """
        if max_hops is None:
            max_hops = self._config.max_spread_hops

        if min_activation is None:
            min_activation = self._config.activation_threshold

        # Track best activation for each neuron
        results: dict[str, ActivationResult] = {}

        # Frequency cache: neuron_id -> access_frequency (myelination boost)
        freq_cache: dict[str, int] = {}

        # Priority queue for BFS with activation ordering
        queue: list[ActivationState] = []

        # Initialize with anchor neurons
        for anchor_id in anchor_neurons:
            neuron = await self._storage.get_neuron(anchor_id)
            if neuron is None:
                continue

            state = ActivationState(
                neuron_id=anchor_id,
                level=1.0,
                hops=0,
                path=[anchor_id],
                source=anchor_id,
            )
            heapq.heappush(queue, state)

            # Record anchor activation
            results[anchor_id] = ActivationResult(
                neuron_id=anchor_id,
                activation_level=1.0,
                hop_distance=0,
                path=[anchor_id],
                source_anchor=anchor_id,
            )

        # Visited tracking (neuron_id, source) to allow multiple paths
        visited: set[tuple[str, str]] = set()

        # Spread activation (capped to prevent memory exhaustion)
        while queue:
            if len(queue) > _MAX_QUEUE_SIZE:
                break
            current = heapq.heappop(queue)

            # Skip if we've visited this neuron from this source
            visit_key = (current.neuron_id, current.source)
            if visit_key in visited:
                continue
            visited.add(visit_key)

            # Skip if we've exceeded max hops
            if current.hops >= max_hops:
                continue

            # Get neighbors
            neighbors = await self._storage.get_neighbors(
                current.neuron_id,
                direction="both",
                min_weight=0.1,
            )

            # Batch-prefetch neuron states for uncached neighbors
            uncached_ids = [n.id for n, _ in neighbors if n.id not in freq_cache]
            if uncached_ids:
                batch_states = await self._storage.get_neuron_states_batch(uncached_ids)
                for nid in uncached_ids:
                    neuron_state = batch_states.get(nid)
                    freq_cache[nid] = neuron_state.access_frequency if neuron_state else 0

            # Build set of refractory neuron IDs for quick lookup
            refractory_ids: set[str] = set()
            if uncached_ids:
                for nid in uncached_ids:
                    neuron_state = batch_states.get(nid)
                    if neuron_state and neuron_state.in_refractory:
                        refractory_ids.add(nid)

            for neighbor_neuron, synapse in neighbors:
                # Skip neurons in refractory cooldown
                if neighbor_neuron.id in refractory_ids:
                    continue
                # Frequency boost: frequently accessed neurons conduct stronger
                # (myelination metaphor — well-used pathways transmit faster)
                freq = freq_cache.get(neighbor_neuron.id, 0)
                freq_factor = 1.0 + min(0.15, 0.05 * math.log1p(freq))

                # Calculate new activation with frequency boost
                new_level = current.level * decay_factor * synapse.weight * freq_factor

                # Skip if below threshold
                if new_level < min_activation:
                    continue

                new_path = [*current.path, neighbor_neuron.id]

                # Update result if this is better activation
                existing = results.get(neighbor_neuron.id)
                if existing is None or new_level > existing.activation_level:
                    results[neighbor_neuron.id] = ActivationResult(
                        neuron_id=neighbor_neuron.id,
                        activation_level=new_level,
                        hop_distance=current.hops + 1,
                        path=new_path,
                        source_anchor=current.source,
                    )

                # Add to queue for further spreading
                new_state = ActivationState(
                    neuron_id=neighbor_neuron.id,
                    level=new_level,
                    hops=current.hops + 1,
                    path=new_path,
                    source=current.source,
                )
                heapq.heappush(queue, new_state)

        return results

    async def activate_from_multiple(
        self,
        anchor_sets: list[list[str]],
        max_hops: int | None = None,
    ) -> tuple[dict[str, ActivationResult], list[str]]:
        """
        Activate from multiple anchor sets and find intersections.

        This is useful when a query has multiple constraints (e.g.,
        time + entity). Neurons activated by multiple anchor sets
        are likely to be more relevant.

        Args:
            anchor_sets: List of anchor neuron lists
            max_hops: Maximum hops for each activation

        Returns:
            Tuple of (combined activations, intersection neuron IDs)
        """
        if not anchor_sets:
            return {}, []

        # Activate from each set in parallel
        tasks = [self.activate(anchors, max_hops) for anchors in anchor_sets if anchors]
        activation_results = list(await asyncio.gather(*tasks)) if tasks else []

        if not activation_results:
            return {}, []

        if len(activation_results) == 1:
            return activation_results[0], list(activation_results[0].keys())

        # Find intersection
        intersection = self._find_intersection(activation_results)

        # Combine results with boosted activation for intersections
        combined: dict[str, ActivationResult] = {}

        for result_set in activation_results:
            for neuron_id, activation in result_set.items():
                existing = combined.get(neuron_id)

                if existing is None:
                    combined[neuron_id] = activation
                else:
                    # Combine activations (take max, but boost if in intersection)
                    if neuron_id in intersection:
                        # Boost: multiply activations
                        new_level = min(
                            1.0, existing.activation_level + activation.activation_level * 0.5
                        )
                    else:
                        new_level = max(existing.activation_level, activation.activation_level)

                    combined[neuron_id] = ActivationResult(
                        neuron_id=neuron_id,
                        activation_level=new_level,
                        hop_distance=min(existing.hop_distance, activation.hop_distance),
                        path=existing.path
                        if existing.hop_distance <= activation.hop_distance
                        else activation.path,
                        source_anchor=existing.source_anchor,
                    )

        return combined, intersection

    def _find_intersection(
        self,
        activation_sets: list[dict[str, ActivationResult]],
    ) -> list[str]:
        """
        Find neurons activated by multiple anchor sets.

        Args:
            activation_sets: List of activation results from different anchor sets

        Returns:
            List of neuron IDs appearing in multiple sets, sorted by
            combined activation level
        """
        if not activation_sets:
            return []

        # Count appearances and sum activations
        appearances: dict[str, int] = defaultdict(int)
        total_activation: dict[str, float] = defaultdict(float)

        for result_set in activation_sets:
            for neuron_id, activation in result_set.items():
                appearances[neuron_id] += 1
                total_activation[neuron_id] += activation.activation_level

        # Find neurons in multiple sets
        multi_set_neurons = [
            (neuron_id, total_activation[neuron_id], count)
            for neuron_id, count in appearances.items()
            if count > 1
        ]

        # Sort by count (descending) then activation (descending)
        multi_set_neurons.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return [n[0] for n in multi_set_neurons]

    async def get_activated_subgraph(
        self,
        activations: dict[str, ActivationResult],
        min_activation: float = 0.2,
        max_neurons: int = 50,
    ) -> tuple[list[str], list[str]]:
        """
        Get the subgraph of activated neurons and their connections.

        Args:
            activations: Activation results
            min_activation: Minimum activation to include
            max_neurons: Maximum neurons to include

        Returns:
            Tuple of (neuron_ids, synapse_ids) in the subgraph
        """
        # Filter and sort by activation
        filtered = [
            (neuron_id, result)
            for neuron_id, result in activations.items()
            if result.activation_level >= min_activation
        ]
        filtered.sort(key=lambda x: x[1].activation_level, reverse=True)

        # Take top neurons
        selected_neurons = [n[0] for n in filtered[:max_neurons]]
        selected_set = set(selected_neurons)

        # Find synapses connecting selected neurons (batch query)
        synapse_ids: list[str] = []

        all_synapses = await self._storage.get_synapses_for_neurons(
            selected_neurons, direction="out"
        )
        for synapses in all_synapses.values():
            for synapse in synapses:
                if synapse.target_id in selected_set:
                    synapse_ids.append(synapse.id)

        return selected_neurons, synapse_ids
