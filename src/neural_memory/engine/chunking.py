"""Working memory chunking — group retrieval results into cognitive units.

Miller's Law: working memory holds 7±2 items. When retrieval returns
many results, chunking groups them into coherent clusters based on
synapse connectivity, making LLM consumption more efficient.

Neuroscience basis: chunking in working memory — grouping related items
into larger units increases effective capacity (Miller, 1956).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CognitiveChunk:
    """A group of related neurons forming a cognitive unit."""

    label: str
    neuron_ids: tuple[str, ...]
    coherence: float  # avg synapse weight within chunk
    relevance: float  # avg activation of chunk members


def chunk_retrieval_results(
    neuron_ids: list[str],
    activation_levels: dict[str, float],
    synapse_pairs: list[tuple[str, str, float]],  # (source, target, weight)
    neuron_tags: dict[str, list[str]] | None = None,
    max_chunks: int = 5,
    max_per_chunk: int = 4,
) -> list[CognitiveChunk]:
    """Group retrieval results into cognitive chunks.

    Uses greedy clustering: seed from highest activation neuron,
    absorb connected neighbors up to max_per_chunk.

    Args:
        neuron_ids: All activated neuron IDs
        activation_levels: Neuron ID → activation score
        synapse_pairs: List of (source_id, target_id, weight)
        neuron_tags: Optional neuron ID → tags for labeling
        max_chunks: Maximum number of chunks to return
        max_per_chunk: Maximum neurons per chunk

    Returns:
        List of CognitiveChunk, sorted by relevance (descending)
    """
    if not neuron_ids:
        return []

    # Build adjacency: neuron_id → [(neighbor_id, weight)]
    adjacency: dict[str, list[tuple[str, float]]] = {nid: [] for nid in neuron_ids}
    neuron_set = set(neuron_ids)

    for src, tgt, weight in synapse_pairs:
        if src in neuron_set and tgt in neuron_set:
            adjacency.setdefault(src, []).append((tgt, weight))
            adjacency.setdefault(tgt, []).append((src, weight))

    # Sort neurons by activation (descending) — greedy seeding
    sorted_neurons = sorted(neuron_ids, key=lambda n: activation_levels.get(n, 0.0), reverse=True)

    assigned: set[str] = set()
    chunks: list[CognitiveChunk] = []

    for seed in sorted_neurons:
        if seed in assigned or len(chunks) >= max_chunks:
            break

        # Grow cluster from seed
        cluster = [seed]
        assigned.add(seed)

        # Add connected neighbors sorted by weight
        neighbors = sorted(adjacency.get(seed, []), key=lambda x: x[1], reverse=True)
        for neighbor_id, _weight in neighbors:
            if neighbor_id not in assigned and len(cluster) < max_per_chunk:
                cluster.append(neighbor_id)
                assigned.add(neighbor_id)

        # Compute coherence (avg internal edge weight)
        internal_weights = []
        cluster_set = set(cluster)
        for src, tgt, w in synapse_pairs:
            if src in cluster_set and tgt in cluster_set:
                internal_weights.append(w)
        coherence = sum(internal_weights) / len(internal_weights) if internal_weights else 0.0

        # Compute relevance (avg activation)
        activations = [activation_levels.get(n, 0.0) for n in cluster]
        relevance = sum(activations) / len(activations) if activations else 0.0

        # Auto-label from shared tags
        label = _auto_label(cluster, neuron_tags) if neuron_tags else f"cluster-{len(chunks) + 1}"

        chunks.append(
            CognitiveChunk(
                label=label,
                neuron_ids=tuple(cluster),
                coherence=round(coherence, 4),
                relevance=round(relevance, 4),
            )
        )

    # Add ungrouped high-activation neurons as singletons
    for nid in sorted_neurons:
        if nid not in assigned and len(chunks) < max_chunks:
            activation = activation_levels.get(nid, 0.0)
            if activation > 0.0:
                label = _auto_label([nid], neuron_tags) if neuron_tags else f"singleton-{nid[:8]}"
                chunks.append(
                    CognitiveChunk(
                        label=label,
                        neuron_ids=(nid,),
                        coherence=0.0,
                        relevance=round(activation, 4),
                    )
                )
                assigned.add(nid)

    # Sort by relevance
    chunks.sort(key=lambda c: c.relevance, reverse=True)

    return chunks


def _auto_label(neuron_ids: list[str], neuron_tags: dict[str, list[str]] | None) -> str:
    """Generate a label from shared tags across neurons."""
    if not neuron_tags:
        return f"group-{len(neuron_ids)}"

    # Collect all tags, find most common
    tag_counts: dict[str, int] = {}
    for nid in neuron_ids:
        for tag in neuron_tags.get(nid, []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        return f"group-{len(neuron_ids)}"

    # Use the most shared tag
    top_tag = max(tag_counts, key=lambda t: tag_counts[t])
    return top_tag
