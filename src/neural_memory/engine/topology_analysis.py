"""Graph topology analysis for neural memory brains.

Computes structural metrics from the neuron-synapse graph:
clustering coefficient, connected component ratio, density,
knowledge density, and enrichment coverage. All metrics use
existing storage API — no new tables or infrastructure needed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


@runtime_checkable
class SynapseLike(Protocol):
    """Minimal synapse interface for topology analysis."""

    source_id: str
    target_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TopologyMetrics:
    """Graph topology metrics for a brain.

    All values normalized to 0.0-1.0 unless noted.

    Attributes:
        clustering_coefficient: How interconnected each neuron's
            neighbors are (0 = no triangles, 1 = fully meshed).
        largest_component_ratio: Fraction of neurons in the largest
            connected component (1.0 = fully connected graph).
        density: Edge count / max possible edges (undirected).
        knowledge_density: Synapses per neuron (NOT 0-1, raw ratio).
        enriched_synapse_ratio: Fraction of synapses created by
            ENRICH consolidation (have ``_enriched`` metadata).
    """

    clustering_coefficient: float
    largest_component_ratio: float
    density: float
    knowledge_density: float
    enriched_synapse_ratio: float


async def compute_topology(
    storage: NeuralStorage,
    brain_id: str,
    *,
    _preloaded_synapses: Sequence[SynapseLike] | None = None,
) -> TopologyMetrics:
    """Compute graph topology metrics for a brain.

    Uses only existing storage methods — no new infrastructure.
    Samples neurons for clustering coefficient to keep cost O(n)
    for large brains.

    Args:
        storage: Neural storage instance.
        brain_id: Brain identifier.
        _preloaded_synapses: Optional pre-fetched synapse list to avoid
            redundant storage calls when called from EvolutionEngine.
    """
    stats = await storage.get_stats(brain_id)
    neuron_count = stats.get("neuron_count", 0)
    synapse_count = stats.get("synapse_count", 0)

    if neuron_count == 0:
        return TopologyMetrics(
            clustering_coefficient=0.0,
            largest_component_ratio=0.0,
            density=0.0,
            knowledge_density=0.0,
            enriched_synapse_ratio=0.0,
        )

    all_synapses: Sequence[SynapseLike] = (
        _preloaded_synapses if _preloaded_synapses is not None else await storage.get_all_synapses()  # type: ignore[assignment]
    )

    # ── Density (undirected: n*(n-1)/2) ────────────────────────
    max_edges = neuron_count * (neuron_count - 1) // 2
    density = synapse_count / max_edges if max_edges > 0 else 0.0

    # ── Knowledge density ────────────────────────────────────
    knowledge_density = synapse_count / max(1, neuron_count)

    # ── Enriched synapse ratio ───────────────────────────────
    enriched_count = sum(
        1 for s in all_synapses if getattr(s, "metadata", None) and s.metadata.get("_enriched")
    )
    enriched_ratio = enriched_count / max(1, len(all_synapses))

    # ── Largest connected component ──────────────────────────
    lcc_ratio = _largest_component_ratio(all_synapses, neuron_count)

    # ── Clustering coefficient ───────────────────────────────
    clustering = _clustering_coefficient(all_synapses)

    return TopologyMetrics(
        clustering_coefficient=clustering,
        largest_component_ratio=lcc_ratio,
        density=min(1.0, density),
        knowledge_density=knowledge_density,
        enriched_synapse_ratio=enriched_ratio,
    )


# ── Internal helpers ─────────────────────────────────────────────

# Max nodes/neighbors sampled for performance-bounded computation
_MAX_SAMPLE_NODES = 200
_MAX_SAMPLE_NEIGHBORS = 200


def _largest_component_ratio(
    synapses: Sequence[SynapseLike],
    neuron_count: int,
) -> float:
    """Compute ratio of neurons in the largest connected component.

    Treats the graph as undirected for connectivity analysis.
    Uses union-find for O(n + e) performance.
    """
    if neuron_count == 0:
        return 0.0

    # Collect all neuron IDs from synapses
    all_nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    for s in synapses:
        all_nodes.add(s.source_id)
        all_nodes.add(s.target_id)
        edges.append((s.source_id, s.target_id))

    if not all_nodes:
        return 0.0

    # Union-Find (path compression + union-by-rank mutates local dicts
    # intentionally for near-constant amortized performance)
    parent: dict[str, str] = {n: n for n in all_nodes}
    rank: dict[str, int] = dict.fromkeys(all_nodes, 0)

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    # Count component sizes
    component_sizes: dict[str, int] = defaultdict(int)
    for n in all_nodes:
        component_sizes[find(n)] += 1

    largest = max(component_sizes.values()) if component_sizes else 0
    # Use neuron_count from stats (includes isolated neurons)
    total = max(neuron_count, len(all_nodes))
    return largest / total


def _clustering_coefficient(synapses: Sequence[SynapseLike]) -> float:
    """Compute global clustering coefficient.

    For each node, checks how many of its neighbor pairs are
    also connected. Averages across all nodes with 2+ neighbors.

    Treats graph as undirected for triangle detection.
    Samples max 200 nodes and caps neighbors at 200 per node
    for bounded O(n) performance.
    """
    # Build undirected adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for s in synapses:
        adj[s.source_id].add(s.target_id)
        adj[s.target_id].add(s.source_id)

    if not adj:
        return 0.0

    # Build edge set for O(1) lookup
    edge_set: set[frozenset[str]] = set()
    for s in synapses:
        edge_set.add(frozenset((s.source_id, s.target_id)))

    # Sample nodes if too many
    nodes = list(adj.keys())
    if len(nodes) > _MAX_SAMPLE_NODES:
        rng = random.Random(42)  # deterministic sampling
        nodes = rng.sample(nodes, _MAX_SAMPLE_NODES)

    coefficients: list[float] = []
    rng_neighbors = random.Random(42)
    for node in nodes:
        neighbors = list(adj[node])
        # Cap neighbors for hub nodes to avoid O(k²) blowup
        if len(neighbors) > _MAX_SAMPLE_NEIGHBORS:
            neighbors = rng_neighbors.sample(neighbors, _MAX_SAMPLE_NEIGHBORS)
        k = len(neighbors)
        if k < 2:
            continue

        # Count triangles (connected neighbor pairs)
        triangles = 0
        possible = k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                if frozenset((neighbors[i], neighbors[j])) in edge_set:
                    triangles += 1

        coefficients.append(triangles / possible if possible > 0 else 0.0)

    return sum(coefficients) / len(coefficients) if coefficients else 0.0
