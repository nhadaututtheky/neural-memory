"""Spreading activation algorithm for memory retrieval."""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.neuron import Neuron
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Safety cap: maximum queue entries to prevent memory exhaustion on dense graphs
_MAX_QUEUE_SIZE = 50_000


@dataclass
class ActivationTrace:
    """Per-hop metrics for diminishing returns detection.

    Tracks how many new neurons and how much activation gain each
    hop produces, enabling early termination when spreading adds
    diminishing signal.

    Attributes:
        new_neurons_per_hop: Count of newly discovered neurons at each hop level
        activation_gain_per_hop: Sum of activation added at each hop level
        max_hop_used: Highest hop level that actually produced results
        max_hop_allowed: Maximum hops that were permitted
        stopped_early: Whether activation was terminated early
        stop_reason: Human-readable reason for early stop
    """

    new_neurons_per_hop: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    activation_gain_per_hop: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    max_hop_used: int = 0
    max_hop_allowed: int = 0
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def total_neurons_activated(self) -> int:
        """Total neurons discovered across all hops."""
        return sum(self.new_neurons_per_hop.values())


def should_stop_spreading(
    trace: ActivationTrace,
    current_hop: int,
    threshold: float = 0.15,
    min_new_neurons: int = 2,
    grace_hops: int = 1,
) -> tuple[bool, str]:
    """Check if spreading activation should stop due to diminishing returns.

    Evaluates whether the most recently completed hop produced enough
    new signal to justify continuing. Two criteria:
    1. Absolute: if a hop added fewer than min_new_neurons, stop.
    2. Relative: if gain_ratio (hop[i] / hop[i-1]) < threshold, stop.

    Grace hops are always allowed regardless of signal.

    Args:
        trace: Current activation trace with per-hop metrics.
        current_hop: The hop level we're about to explore.
        threshold: Minimum gain ratio to continue (default 0.15).
        min_new_neurons: Minimum new neurons per hop (default 2).
        grace_hops: Number of initial hops exempt from gating (default 1).

    Returns:
        Tuple of (should_stop, reason_string).
    """
    if current_hop <= grace_hops:
        return False, ""

    prev_hop = current_hop - 1
    prev_new = trace.new_neurons_per_hop.get(prev_hop, 0)

    # Absolute check: too few new neurons from previous hop
    if prev_new < min_new_neurons:
        return True, f"hop {prev_hop} added only {prev_new} neurons (min={min_new_neurons})"

    # Relative check: gain ratio vs the hop before
    if current_hop >= 2:
        prev_prev_new = trace.new_neurons_per_hop.get(prev_hop - 1, 0)
        if prev_prev_new > 0:
            gain_ratio = prev_new / prev_prev_new
            if gain_ratio < threshold:
                return True, (
                    f"gain ratio {gain_ratio:.2f} < {threshold} "
                    f"(hop {prev_hop}: {prev_new}, hop {prev_hop - 1}: {prev_prev_new})"
                )

    return False, ""


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

    Uses generation-based visited tracking: instead of clearing a visited
    set between searches, a generation counter is incremented. Nodes with
    a stale generation number are considered unvisited. This avoids O(N)
    set re-allocation on repeated searches (e.g. during consolidation).
    Inspired by HyperspaceDB's generation-based bitmap approach.
    """

    # Trim stale entries from visited dict every N generations
    _TRIM_INTERVAL = 100
    _TRIM_KEEP_GENERATIONS = 50

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
        self._generation: int = 0
        self._visited_gen: dict[tuple[str, str], int] = {}

    async def activate(
        self,
        anchor_neurons: list[str],
        max_hops: int | None = None,
        decay_factor: float = 0.5,
        min_activation: float | None = None,
        anchor_activations: dict[str, float] | None = None,
    ) -> tuple[dict[str, ActivationResult], ActivationTrace]:
        """
        Spread activation from anchor neurons through the graph.

        The activation spreads through synapses, with the level
        decaying at each hop:
            activation(hop) = initial * decay_factor^hop * synapse_weight

        Includes a diminishing returns gate that stops spreading early
        when new hops produce insufficient new signal.

        Args:
            anchor_neurons: Starting neurons (initial level from anchor_activations or 1.0)
            max_hops: Maximum number of hops (default: from config)
            decay_factor: How much activation decays per hop
            min_activation: Minimum activation to continue spreading
            anchor_activations: Optional per-anchor initial activation levels (from RRF fusion).
                               If None, all anchors start at 1.0.

        Returns:
            Tuple of (dict mapping neuron_id to ActivationResult, ActivationTrace)
        """
        if max_hops is None:
            max_hops = self._config.max_spread_hops

        if min_activation is None:
            min_activation = self._config.activation_threshold

        # Diminishing returns config
        dr_enabled = self._config.diminishing_returns_enabled
        dr_threshold = self._config.diminishing_returns_threshold
        dr_min_neurons = self._config.diminishing_returns_min_neurons
        dr_grace_hops = self._config.diminishing_returns_grace_hops

        # Activation trace for metrics
        trace = ActivationTrace(max_hop_allowed=max_hops)

        # Track best activation for each neuron
        results: dict[str, ActivationResult] = {}

        # Frequency cache: neuron_id -> access_frequency (myelination boost)
        freq_cache: dict[str, int] = {}

        # Neighbor cache: avoid re-fetching neighbors for the same neuron
        neighbor_cache: dict[str, list[tuple[Neuron, Synapse]]] = {}

        # Priority queue for BFS with activation ordering
        queue: list[ActivationState] = []

        # Initialize with anchor neurons (batch fetch)
        anchor_neurons_map = await self._storage.get_neurons_batch(list(anchor_neurons))
        for anchor_id in anchor_neurons:
            if anchor_id not in anchor_neurons_map:
                continue

            initial_level = (
                anchor_activations.get(anchor_id, 1.0) if anchor_activations is not None else 1.0
            )

            state = ActivationState(
                neuron_id=anchor_id,
                level=initial_level,
                hops=0,
                path=[anchor_id],
                source=anchor_id,
            )
            heapq.heappush(queue, state)

            # Record anchor activation
            results[anchor_id] = ActivationResult(
                neuron_id=anchor_id,
                activation_level=initial_level,
                hop_distance=0,
                path=[anchor_id],
                source_anchor=anchor_id,
            )
            trace.new_neurons_per_hop[0] += 1
            trace.activation_gain_per_hop[0] += initial_level

        # Generation-based visited tracking: increment generation per search,
        # nodes with stale generation are considered unvisited (O(1) "clear").
        self._generation += 1
        current_gen = self._generation

        # Periodic trim to prevent unbounded dict growth
        if current_gen % self._TRIM_INTERVAL == 0:
            cutoff = current_gen - self._TRIM_KEEP_GENERATIONS
            self._visited_gen = {k: g for k, g in self._visited_gen.items() if g >= cutoff}

        # Track which hop levels have been checked for diminishing returns
        dr_checked_hops: set[int] = set()

        # Spread activation (capped to prevent memory exhaustion)
        while queue:
            if len(queue) > _MAX_QUEUE_SIZE:
                break
            current = heapq.heappop(queue)

            # Skip if we've visited this neuron from this source in this generation
            visit_key = (current.neuron_id, current.source)
            if self._visited_gen.get(visit_key, -1) == current_gen:
                continue
            self._visited_gen[visit_key] = current_gen

            # Skip if we've exceeded max hops
            if current.hops >= max_hops:
                continue

            # Diminishing returns gate: check at hop transitions
            next_hop = current.hops + 1
            if dr_enabled and next_hop not in dr_checked_hops and next_hop >= 2:
                dr_checked_hops.add(next_hop)
                stop, reason = should_stop_spreading(
                    trace, next_hop, dr_threshold, dr_min_neurons, dr_grace_hops
                )
                if stop:
                    trace.stopped_early = True
                    trace.stop_reason = reason
                    logger.debug(
                        "Diminishing returns gate: stopping at hop %d — %s", next_hop, reason
                    )
                    break

            # Get neighbors (with cache to avoid N+1 re-fetching)
            if current.neuron_id in neighbor_cache:
                neighbors = neighbor_cache[current.neuron_id]
            else:
                neighbors = await self._storage.get_neighbors(
                    current.neuron_id,
                    direction="both",
                    min_weight=0.1,
                )
                neighbor_cache[current.neuron_id] = neighbors

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
                hop = current.hops + 1

                # Update result if this is better activation
                existing = results.get(neighbor_neuron.id)
                if existing is None or new_level > existing.activation_level:
                    # Track new neuron discovery for diminishing returns
                    if existing is None:
                        trace.new_neurons_per_hop[hop] += 1
                    trace.activation_gain_per_hop[hop] += new_level

                    results[neighbor_neuron.id] = ActivationResult(
                        neuron_id=neighbor_neuron.id,
                        activation_level=new_level,
                        hop_distance=hop,
                        path=new_path,
                        source_anchor=current.source,
                    )
                    trace.max_hop_used = max(trace.max_hop_used, hop)

                # Add to queue for further spreading
                new_state = ActivationState(
                    neuron_id=neighbor_neuron.id,
                    level=new_level,
                    hops=hop,
                    path=new_path,
                    source=current.source,
                )
                heapq.heappush(queue, new_state)

        return results, trace

    async def activate_from_multiple(
        self,
        anchor_sets: list[list[str]],
        max_hops: int | None = None,
        anchor_activations: dict[str, float] | None = None,
    ) -> tuple[dict[str, ActivationResult], list[str]]:
        """
        Activate from multiple anchor sets and find intersections.

        This is useful when a query has multiple constraints (e.g.,
        time + entity). Neurons activated by multiple anchor sets
        are likely to be more relevant.

        Args:
            anchor_sets: List of anchor neuron lists
            max_hops: Maximum hops for each activation
            anchor_activations: Optional per-anchor initial activation levels (from RRF).

        Returns:
            Tuple of (combined activations, intersection neuron IDs)
        """
        if not anchor_sets:
            return {}, []

        # Activate from each set in parallel
        tasks = [
            self.activate(anchors, max_hops, anchor_activations=anchor_activations)
            for anchors in anchor_sets
            if anchors
        ]
        raw_results = list(await asyncio.gather(*tasks)) if tasks else []

        if not raw_results:
            return {}, []

        # Unpack (results, trace) tuples — traces logged but not returned
        activation_results = [r[0] for r in raw_results]

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
