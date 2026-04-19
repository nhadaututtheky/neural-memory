"""Personalized PageRank activation for memory retrieval.

Replaces the BFS spreading activation with PPR which:
- Distributes activation proportional to edge weights / out-degree
- Has damping (teleport back to seed set) preventing distant drift
- Naturally handles hub dampening (high-degree nodes don't dominate)
- Converges to a stationary distribution

This is the same algorithm used by Google's Knowledge Graph, Neo4j,
and HippoRAG — a neurobiologically-inspired retrieval system
that models how the hippocampus indexes and retrieves memories.

Uses push-based PPR (more efficient for sparse seed sets):
only process nodes with residual above epsilon.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from neural_memory.engine.activation import ActivationResult, ActivationTrace

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Safety cap: maximum nodes to track during PPR iteration
_MAX_ACTIVE_NODES = 10_000


class PPRActivation:
    """Personalized PageRank activation for memory retrieval.

    Unlike BFS spreading which decays uniformly per hop, PPR:
    - Distributes activation proportional to edge weights / out-degree
    - Has damping (teleport back to seed set) preventing distant drift
    - Naturally handles hub dampening (high-degree nodes don't dominate)
    - Converges to stationary distribution
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> None:
        self._storage = storage
        self._config = config

    async def activate(
        self,
        anchor_neurons: list[str],
        anchor_activations: dict[str, float] | None = None,
        damping: float | None = None,
        max_iterations: int | None = None,
        epsilon: float | None = None,
    ) -> tuple[dict[str, ActivationResult], ActivationTrace]:
        """Run Personalized PageRank from anchor seed set.

        Algorithm (push-based power iteration):
        1. Initialize: residual[seed] = seed_weight for each seed
        2. For each iteration:
           a. For each node u with residual[u] > epsilon:
              - Push (1-damping) * residual[u] to neighbors (weighted by edge weight / out-degree)
              - Add damping * residual[u] to rank[u] (teleport to seed)
              - Zero out residual[u]
           b. Check convergence: sum(residual) < epsilon * |seeds|
        3. Return activated neurons above activation_threshold

        Args:
            anchor_neurons: Starting neurons (seed set).
            anchor_activations: Per-anchor initial weights (from RRF). Default: uniform.
            damping: Teleport probability (default: from config.ppr_damping).
            max_iterations: Max iterations (default: from config.ppr_iterations).
            epsilon: Convergence threshold (default: from config.ppr_epsilon).

        Returns:
            Dict mapping neuron_id to ActivationResult.
        """
        if not anchor_neurons:
            return {}, ActivationTrace()

        if damping is None:
            damping = self._config.ppr_damping
        if max_iterations is None:
            max_iterations = self._config.ppr_iterations
        if epsilon is None:
            epsilon = self._config.ppr_epsilon

        # Initialize seed weights
        if anchor_activations is not None:
            seed_weights = {nid: anchor_activations.get(nid, 1.0) for nid in anchor_neurons}
        else:
            seed_weights = dict.fromkeys(anchor_neurons, 1.0)

        # Normalize seed weights to sum to 1.0
        total_weight = sum(seed_weights.values())
        if total_weight > 0:
            seed_weights = {nid: w / total_weight for nid, w in seed_weights.items()}

        # PPR state: rank (converged score) and residual (pending score)
        rank: dict[str, float] = defaultdict(float)
        residual: dict[str, float] = defaultdict(float, seed_weights)

        # Neighbor cache to avoid re-fetching
        neighbor_cache: dict[str, list[tuple[str, float]]] = {}
        # DREAM hub edges (`_hub=True` metadata) are synthesized by consolidation;
        # dampening their effective weight prevents artefacts from hijacking walks.
        hub_dampening = getattr(self._config, "hub_edge_dampening", 1.0)
        # Track hop distance for each neuron (best path)
        hop_distance: dict[str, int] = dict.fromkeys(anchor_neurons, 0)
        # Track path from nearest seed
        source_anchor: dict[str, str] = {nid: nid for nid in anchor_neurons}

        min_activation = self._config.activation_threshold

        # Diminishing returns config
        dr_enabled = self._config.diminishing_returns_enabled
        dr_min_neurons = self._config.diminishing_returns_min_neurons

        # Activation trace
        trace = ActivationTrace(max_hop_allowed=max_iterations)
        # Track nodes that crossed threshold at each iteration
        above_threshold: set[str] = set()

        for iteration in range(max_iterations):
            # Find active nodes (residual above epsilon)
            active_nodes = [(nid, res) for nid, res in residual.items() if res > epsilon]

            if not active_nodes:
                logger.debug("PPR converged at iteration %d (no active nodes)", iteration)
                break

            # Cap active nodes to prevent memory exhaustion
            if len(active_nodes) > _MAX_ACTIVE_NODES:
                active_nodes.sort(key=lambda x: x[1], reverse=True)
                active_nodes = active_nodes[:_MAX_ACTIVE_NODES]

            # Batch-fetch neighbors for uncached nodes
            uncached = [nid for nid, _ in active_nodes if nid not in neighbor_cache]
            if uncached:
                synapses_map = await self._storage.get_synapses_for_neurons(
                    uncached, direction="out"
                )
                for nid in uncached:
                    synapses = synapses_map.get(nid, [])
                    neighbors = []
                    for s in synapses:
                        if s.weight < 0.1:  # skip near-zero edges
                            continue
                        effective = s.weight
                        meta = getattr(s, "metadata", None) or {}
                        if meta.get("_hub"):
                            effective *= hub_dampening
                        if effective < 1e-9:
                            continue
                        neighbors.append((s.target_id, effective))
                    neighbor_cache[nid] = neighbors

            new_residual: dict[str, float] = defaultdict(float)

            for nid, res in active_nodes:
                neighbors = neighbor_cache.get(nid, [])

                # Absorb: damping fraction stays at this node
                rank[nid] += damping * res

                # Push: distribute (1-damping) to neighbors
                if neighbors:
                    total_out_weight = sum(w for _, w in neighbors)
                    push_amount = (1 - damping) * res

                    for target_id, edge_weight in neighbors:
                        contribution = push_amount * edge_weight / total_out_weight
                        new_residual[target_id] += contribution

                        # Track hop distance (approximation: current + 1)
                        current_hops = hop_distance.get(nid, 0) + 1
                        if target_id not in hop_distance or current_hops < hop_distance[target_id]:
                            hop_distance[target_id] = current_hops
                            source_anchor[target_id] = source_anchor.get(nid, nid)
                else:
                    # Dead end: residual teleports back to seeds
                    for seed_nid, seed_w in seed_weights.items():
                        new_residual[seed_nid] += (1 - damping) * res * seed_w

                # Clear processed residual
                residual[nid] = 0.0

            # Accumulate new residuals
            for nid, new_res in new_residual.items():
                residual[nid] += new_res

            # Track new nodes that crossed activation threshold this iteration
            new_this_iteration = 0
            for nid, score in rank.items():
                max_rank_so_far = max(rank.values()) if rank else 1.0
                if max_rank_so_far < 1e-12:
                    max_rank_so_far = 1.0
                normalized = score / max_rank_so_far
                if normalized >= min_activation and nid not in above_threshold:
                    above_threshold.add(nid)
                    new_this_iteration += 1
            trace.new_neurons_per_hop[iteration] = new_this_iteration
            trace.max_hop_used = iteration

            # Diminishing returns: stop if iteration added too few new nodes
            if dr_enabled and iteration >= 2 and new_this_iteration < dr_min_neurons:
                trace.stopped_early = True
                trace.stop_reason = (
                    f"PPR iteration {iteration} added only {new_this_iteration} "
                    f"new nodes (min={dr_min_neurons})"
                )
                logger.debug("Diminishing returns gate: %s", trace.stop_reason)
                break

            # Check convergence
            total_residual = sum(abs(v) for v in residual.values())
            if total_residual < epsilon * len(seed_weights):
                logger.debug(
                    "PPR converged at iteration %d (residual=%.6f)", iteration, total_residual
                )
                break

        # Convert rank scores to ActivationResult
        results: dict[str, ActivationResult] = {}

        # Normalize ranks so max = 1.0
        max_rank = max(rank.values()) if rank else 1.0
        if max_rank < 1e-12:
            max_rank = 1.0

        for nid, score in rank.items():
            normalized = score / max_rank
            if normalized < min_activation:
                continue

            results[nid] = ActivationResult(
                neuron_id=nid,
                activation_level=normalized,
                hop_distance=hop_distance.get(nid, 0),
                path=[source_anchor.get(nid, nid), nid] if nid not in seed_weights else [nid],
                source_anchor=source_anchor.get(nid, nid),
            )

        return results, trace

    async def activate_from_multiple(
        self,
        anchor_sets: list[list[str]],
        anchor_activations: dict[str, float] | None = None,
    ) -> tuple[dict[str, ActivationResult], list[str]]:
        """PPR with multiple anchor sets, find intersections.

        Runs a single PPR with all anchors as seeds (weighted by RRF
        if available), then identifies intersection neurons that were
        reached from multiple anchor sets.

        Args:
            anchor_sets: List of anchor neuron lists.
            anchor_activations: Per-anchor initial weights (from RRF).

        Returns:
            Tuple of (activation results, intersection neuron IDs).
        """
        if not anchor_sets:
            return {}, []

        # Flatten all anchors into one seed set
        all_anchors = [a for anchors in anchor_sets for a in anchors]
        if not all_anchors:
            return {}, []

        # Run single PPR with all anchors
        results, _trace = await self.activate(
            anchor_neurons=all_anchors,
            anchor_activations=anchor_activations,
        )

        # Find intersections: neurons reached from multiple anchor sets
        if len(anchor_sets) > 1:
            set_membership: dict[str, set[int]] = defaultdict(set)
            for set_idx, anchor_list in enumerate(anchor_sets):
                anchor_set = set(anchor_list)
                for nid, result in results.items():
                    if result.source_anchor in anchor_set:
                        set_membership[nid].add(set_idx)

            intersections = sorted(
                [nid for nid, sets in set_membership.items() if len(sets) > 1],
                key=lambda nid: results[nid].activation_level,
                reverse=True,
            )
        else:
            intersections = list(results.keys())

        return results, intersections
