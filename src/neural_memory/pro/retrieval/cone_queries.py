"""Cone queries — exhaustive recall via HNSW + threshold filtering.

Pro cone queries use HNSW approximate nearest neighbor search instead of
brute-force O(N) scanning. Retrieves a generous k candidates from HNSW,
then filters by cosine similarity threshold to form the cone.

Algorithm:
1. Embed the query
2. HNSW search for top-k candidates (k = max_results * 2 for safety margin)
3. Filter by cone threshold (cosine similarity >= threshold)
4. Rank by combined score: similarity * 0.7 + activation * 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neural_memory.pro.infinitydb.engine import InfinityDB

logger = logging.getLogger(__name__)

# Default cone threshold (cosine similarity)
DEFAULT_CONE_THRESHOLD = 0.65


@dataclass(frozen=True)
class ConeResult:
    """A single memory within the recall cone."""

    neuron_id: str
    content: str
    similarity: float
    activation: float
    combined_score: float
    neuron_type: str


async def cone_recall(
    query_embedding: list[float] | NDArray[np.float32],
    db: InfinityDB,
    *,
    threshold: float = DEFAULT_CONE_THRESHOLD,
    max_results: int = 500,
) -> list[ConeResult]:
    """HNSW-accelerated cone recall — return all memories within similarity cone.

    Args:
        query_embedding: Pre-computed query embedding vector.
        db: InfinityDB instance (must be open).
        threshold: Minimum cosine similarity (0-1). Lower = wider cone.
        max_results: Safety cap for results.

    Returns:
        List of ConeResult sorted by combined_score descending.
    """
    # Search with generous k to capture the full cone
    search_k = min(max_results * 3, db.neuron_count)
    if search_k <= 0:
        return []

    candidates = await db.search_similar(query_embedding, k=search_k)

    results: list[ConeResult] = []
    for candidate in candidates:
        similarity = candidate.get("similarity", 0.0)
        if similarity < threshold:
            continue  # Outside the cone

        activation = candidate.get("activation_level", 0.5)
        if not isinstance(activation, (int, float)):
            activation = 0.5

        combined = similarity * 0.7 + float(activation) * 0.3

        results.append(
            ConeResult(
                neuron_id=candidate.get("id", ""),
                content=candidate.get("content", ""),
                similarity=round(similarity, 4),
                activation=round(float(activation), 4),
                combined_score=round(combined, 4),
                neuron_type=candidate.get("type", "unknown"),
            )
        )

    results.sort(key=lambda r: r.combined_score, reverse=True)
    return results[:max_results]
