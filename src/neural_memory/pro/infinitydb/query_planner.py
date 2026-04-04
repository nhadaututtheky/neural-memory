"""Query planner for InfinityDB.

Plans and executes multi-dimensional queries by fusing vector similarity,
metadata filters, graph proximity, and temporal recency into a single
ranked result set.

Query flow:
1. Parse query dimensions (semantic, type, temporal, graph, priority)
2. For each active dimension, generate candidate sets
3. Fuse candidates using Reciprocal Rank Fusion (RRF)
4. Apply post-filters and limits
5. Return ranked results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# RRF constant (standard value from literature)
RRF_K = 60


@dataclass(frozen=True)
class QueryPlan:
    """A planned multi-dimensional query."""

    # Vector similarity
    query_vector: NDArray[np.float32] | None = None
    vector_k: int = 50
    vector_weight: float = 1.0

    # Metadata filters
    neuron_type: str | None = None
    content_contains: str | None = None
    tags: list[str] = field(default_factory=list)
    ephemeral: bool | None = None

    # Temporal
    min_created: str | None = None
    max_created: str | None = None
    recency_weight: float = 0.0  # 0 = ignore, 1.0 = heavily favor recent

    # Graph proximity
    graph_seed_ids: list[str] = field(default_factory=list)
    graph_max_depth: int = 2
    graph_weight: float = 0.0  # 0 = ignore

    # Priority
    priority_weight: float = 0.0  # 0 = ignore, 1.0 = heavily favor high priority

    # Output
    limit: int = 20
    offset: int = 0
    min_score: float = 0.0


def rrf_fuse(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Args:
        ranked_lists: each list is neuron IDs in rank order (best first)
        weights: per-list importance weights (default: equal)
        k: RRF constant (higher = less aggressive rank emphasis)

    Returns:
        Fused list of (neuron_id, score) sorted by score descending.
    """
    if not ranked_lists:
        return []

    n = len(ranked_lists)
    w = weights if weights and len(weights) == n else [1.0] * n

    scores: dict[str, float] = {}
    for i, rlist in enumerate(ranked_lists):
        for rank, nid in enumerate(rlist):
            rrf_score = w[i] / (k + rank + 1)
            scores[nid] = scores.get(nid, 0.0) + rrf_score

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


class QueryExecutor:
    """Executes multi-dimensional queries against InfinityDB stores."""

    def __init__(
        self,
        metadata_store: Any,
        index: Any,
        graph_store: Any,
    ) -> None:
        self._metadata = metadata_store
        self._index = index
        self._graph = graph_store

    def execute(self, plan: QueryPlan) -> list[dict[str, Any]]:
        """Execute a query plan synchronously.

        Returns ranked list of neuron dicts with _score field.
        """
        ranked_lists: list[list[str]] = []
        weights: list[float] = []

        # Dimension 1: Vector similarity
        if plan.query_vector is not None and plan.vector_weight > 0:
            vec_ranked = self._vector_search(plan.query_vector, plan.vector_k)
            if vec_ranked:
                ranked_lists.append(vec_ranked)
                weights.append(plan.vector_weight)

        # Dimension 2: Graph proximity
        if plan.graph_seed_ids and plan.graph_weight > 0:
            graph_ranked = self._graph_proximity(plan.graph_seed_ids, plan.graph_max_depth)
            if graph_ranked:
                ranked_lists.append(graph_ranked)
                weights.append(plan.graph_weight)

        # Dimension 3: Recency
        if plan.recency_weight > 0:
            recency_ranked = self._recency_rank(limit=plan.vector_k * 2)
            if recency_ranked:
                ranked_lists.append(recency_ranked)
                weights.append(plan.recency_weight)

        # Dimension 4: Priority
        if plan.priority_weight > 0:
            priority_ranked = self._priority_rank(limit=plan.vector_k * 2)
            if priority_ranked:
                ranked_lists.append(priority_ranked)
                weights.append(plan.priority_weight)

        # Fuse with RRF
        if not ranked_lists:
            # Fallback: just use metadata filters
            return self._metadata_only(plan)

        fused = rrf_fuse(ranked_lists, weights)

        # Apply metadata filters and build results
        results: list[dict[str, Any]] = []
        for nid, score in fused:
            if score < plan.min_score:
                continue

            result = self._metadata.get_by_id(nid)
            if result is None:
                continue
            _, meta = result

            # Apply filters
            if not self._passes_filters(meta, plan):
                continue

            results.append({**meta, "_score": round(score, 6)})

        # Apply offset and limit
        return results[plan.offset : plan.offset + plan.limit]

    def _vector_search(self, query_vector: NDArray[np.float32], k: int) -> list[str]:
        """Search by vector similarity, return ranked neuron IDs."""
        if self._index.count == 0:
            return []
        slot_ids, distances = self._index.search(query_vector, k)
        result: list[str] = []
        for slot_id in slot_ids:
            meta = self._metadata.get_by_slot(slot_id)
            if meta is not None:
                nid = meta.get("id", "")
                if nid:
                    result.append(nid)
        return result

    def _graph_proximity(self, seed_ids: list[str], max_depth: int) -> list[str]:
        """Rank by graph distance from seed neurons."""
        all_nodes: list[tuple[str, int]] = []
        seen: set[str] = set()

        for seed in seed_ids:
            traversal = self._graph.bfs(seed, max_depth=max_depth, direction="both", max_nodes=200)
            for nid, depth in traversal:
                if nid not in seen:
                    seen.add(nid)
                    all_nodes.append((nid, depth))

        # Sort by depth ascending (closer = better rank)
        all_nodes.sort(key=lambda x: x[1])
        return [nid for nid, _ in all_nodes]

    def _recency_rank(self, limit: int = 100) -> list[str]:
        """Rank by created_at descending (most recent first)."""
        results = self._metadata.find(limit=limit)
        return [meta.get("id", "") for _, meta in results if meta.get("id")]

    def _priority_rank(self, limit: int = 100) -> list[str]:
        """Rank by priority descending."""
        all_records = self._metadata.iter_all()
        sorted_records = sorted(
            all_records,
            key=lambda x: x[1].get("priority", 0),
            reverse=True,
        )
        return [meta.get("id", "") for _, meta in sorted_records[:limit] if meta.get("id")]

    def _metadata_only(self, plan: QueryPlan) -> list[dict[str, Any]]:
        """Fallback: pure metadata filter query."""
        time_range = None
        if plan.min_created and plan.max_created:
            time_range = (plan.min_created, plan.max_created)

        results = self._metadata.find(
            neuron_type=plan.neuron_type,
            content_contains=plan.content_contains,
            time_range=time_range,
            limit=plan.limit + plan.offset,
            ephemeral=plan.ephemeral,
        )

        filtered = []
        for _, meta in results:
            if self._passes_filters(meta, plan):
                filtered.append({**meta, "_score": 0.0})

        return filtered[plan.offset : plan.offset + plan.limit]

    def _passes_filters(self, meta: dict[str, Any], plan: QueryPlan) -> bool:
        """Check if a neuron passes all metadata filters."""
        if plan.neuron_type and meta.get("type") != plan.neuron_type:
            return False
        if plan.content_contains:
            content = meta.get("content", "")
            if plan.content_contains.lower() not in content.lower():
                return False
        if plan.tags:
            neuron_tags = set(meta.get("tags", []))
            if not neuron_tags.intersection(plan.tags):
                return False
        if plan.ephemeral is not None and meta.get("ephemeral") != plan.ephemeral:
            return False
        if plan.min_created:
            created = meta.get("created_at", "")
            if created and created < plan.min_created:
                return False
        if plan.max_created:
            created = meta.get("created_at", "")
            if created and created > plan.max_created:
                return False
        return True
