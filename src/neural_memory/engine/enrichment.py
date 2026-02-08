"""Enrichment engine — transitive closure and cross-cluster linking.

Creates new knowledge from existing knowledge:
- Transitive closure: If A→B→C (CAUSED_BY), infer A→C
- Cross-cluster linking: Connect clusters sharing entity neurons
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neural_memory.core.synapse import Synapse, SynapseType

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of enrichment operations.

    Attributes:
        transitive_synapses: New synapses from transitive closure
        cross_cluster_synapses: New synapses from cross-cluster linking
    """

    transitive_synapses: list[Synapse] = field(default_factory=list)
    cross_cluster_synapses: list[Synapse] = field(default_factory=list)

    @property
    def total_synapses(self) -> int:
        return len(self.transitive_synapses) + len(self.cross_cluster_synapses)


async def find_transitive_closures(
    storage: NeuralStorage,
    max_depth: int = 2,
    max_synapses: int = 50,
) -> list[Synapse]:
    """Find transitive closure opportunities in CAUSED_BY chains.

    For each A→B→C chain where no A→C exists, create a new
    CAUSED_BY synapse with weight = 0.5 * min(w_AB, w_BC).

    Args:
        storage: Storage backend
        max_depth: Maximum chain depth to traverse (default: 2 = A→B→C)
        max_synapses: Maximum new synapses to create

    Returns:
        List of new Synapse objects (not yet persisted)
    """
    causal_synapses = await storage.get_synapses(type=SynapseType.CAUSED_BY)
    if not causal_synapses:
        return []

    # Build adjacency: source_id -> [(target_id, weight)]
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for syn in causal_synapses:
        adjacency[syn.source_id].append((syn.target_id, syn.weight))

    # Build existing pairs for duplicate checking
    existing_pairs: set[tuple[str, str]] = set()
    for syn in causal_synapses:
        existing_pairs.add((syn.source_id, syn.target_id))

    new_synapses: list[Synapse] = []

    # Find A→B→C chains (depth=2)
    for a_id, a_targets in adjacency.items():
        if len(new_synapses) >= max_synapses:
            break
        for b_id, w_ab in a_targets:
            if len(new_synapses) >= max_synapses:
                break
            for c_id, w_bc in adjacency.get(b_id, []):
                if len(new_synapses) >= max_synapses:
                    break
                if a_id == c_id:
                    continue
                if (a_id, c_id) in existing_pairs:
                    continue

                weight = 0.5 * min(w_ab, w_bc)
                synapse = Synapse.create(
                    source_id=a_id,
                    target_id=c_id,
                    type=SynapseType.CAUSED_BY,
                    weight=weight,
                    metadata={"_enriched": True, "_chain": [a_id, b_id, c_id]},
                )
                new_synapses.append(synapse)
                existing_pairs.add((a_id, c_id))

    return new_synapses


async def find_cross_cluster_links(
    storage: NeuralStorage,
    tag_overlap_threshold: float = 0.4,
    max_synapses: int = 50,
) -> list[Synapse]:
    """Find cross-cluster linking opportunities.

    Clusters fibers by tag Jaccard similarity. For pairs of clusters
    sharing entity neurons, creates RELATED_TO between cluster anchors.

    Args:
        storage: Storage backend
        tag_overlap_threshold: Minimum Jaccard overlap for same cluster
        max_synapses: Maximum new synapses to create

    Returns:
        List of new Synapse objects (not yet persisted)
    """
    fibers = await storage.get_fibers(limit=10000)
    tagged_fibers = [f for f in fibers if f.tags]
    if len(tagged_fibers) < 2:
        return []

    n = len(tagged_fibers)

    # Union-Find for clustering
    parent: dict[int, int] = {i: i for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            tags_a = tagged_fibers[i].tags
            tags_b = tagged_fibers[j].tags
            intersection = len(tags_a & tags_b)
            union_size = len(tags_a | tags_b)
            if union_size > 0 and intersection / union_size >= tag_overlap_threshold:
                union(i, j)

    # Group fibers by cluster root
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    cluster_list = [indices for indices in clusters.values() if len(indices) >= 1]
    if len(cluster_list) < 2:
        return []

    # Build cluster entity sets and anchor IDs
    cluster_entities: list[set[str]] = []
    cluster_anchors: list[str] = []
    for indices in cluster_list:
        entity_ids: set[str] = set()
        for idx in indices:
            entity_ids |= tagged_fibers[idx].neuron_ids
        cluster_entities.append(entity_ids)
        # Use the highest-salience fiber's anchor as cluster anchor
        best_fiber = max(
            (tagged_fibers[i] for i in indices),
            key=lambda f: f.salience,
        )
        cluster_anchors.append(best_fiber.anchor_neuron_id)

    # Check existing synapses between cluster anchors
    existing_synapses = await storage.get_synapses(type=SynapseType.RELATED_TO)
    existing_pairs: set[tuple[str, str]] = set()
    for syn in existing_synapses:
        existing_pairs.add((syn.source_id, syn.target_id))
        existing_pairs.add((syn.target_id, syn.source_id))

    new_synapses: list[Synapse] = []
    num_clusters = len(cluster_list)
    for i in range(num_clusters):
        if len(new_synapses) >= max_synapses:
            break
        for j in range(i + 1, num_clusters):
            if len(new_synapses) >= max_synapses:
                break
            shared = cluster_entities[i] & cluster_entities[j]
            if not shared:
                continue

            anchor_a = cluster_anchors[i]
            anchor_b = cluster_anchors[j]
            if anchor_a == anchor_b:
                continue
            if (anchor_a, anchor_b) in existing_pairs:
                continue

            synapse = Synapse.create(
                source_id=anchor_a,
                target_id=anchor_b,
                type=SynapseType.RELATED_TO,
                weight=0.3,
                metadata={
                    "_enriched": True,
                    "_cross_cluster": True,
                    "_shared_neurons": len(shared),
                },
            )
            new_synapses.append(synapse)
            existing_pairs.add((anchor_a, anchor_b))

    return new_synapses


async def enrich(
    storage: NeuralStorage,
    max_depth: int = 2,
    max_synapses: int = 50,
) -> EnrichmentResult:
    """Run full enrichment: transitive closure + cross-cluster linking.

    Args:
        storage: Storage backend
        max_depth: Maximum transitive chain depth
        max_synapses: Maximum new synapses per strategy

    Returns:
        EnrichmentResult with created synapses
    """
    transitive = await find_transitive_closures(storage, max_depth, max_synapses)
    cross_cluster = await find_cross_cluster_links(storage, max_synapses=max_synapses)

    return EnrichmentResult(
        transitive_synapses=transitive,
        cross_cluster_synapses=cross_cluster,
    )
