"""Reciprocal Rank Fusion for multi-retriever score blending.

Combines ranked lists from multiple retrievers (FTS5/BM25, embedding
similarity, graph expansion) into a unified score without needing
score normalization.

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"

Formula: score(d) = Σ  weight_i / (k + rank_i(d))
         for each retriever i where document d appears.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default RRF constant — higher k reduces impact of top-ranked items.
# k=60 is the standard from the RRF paper.
DEFAULT_RRF_K = 60

# Default weights per retriever type.
DEFAULT_RETRIEVER_WEIGHTS: dict[str, float] = {
    "time": 1.0,
    "entity": 0.9,
    "keyword": 0.7,
    "embedding": 1.0,
    "graph_expansion": 0.5,
}


@dataclass(frozen=True)
class RankedAnchor:
    """A neuron with its rank position within a specific retriever.

    Attributes:
        neuron_id: The neuron this anchor refers to.
        rank: 1-indexed position within the retriever's result list.
        retriever: Which retriever produced this anchor
                   ("time", "entity", "keyword", "embedding", "graph_expansion").
        score: Optional raw score from the retriever (BM25 rank, cosine sim, etc.).
    """

    neuron_id: str
    rank: int
    retriever: str
    score: float = 0.0


def rrf_fuse(
    ranked_lists: list[list[RankedAnchor]],
    k: int = DEFAULT_RRF_K,
    retriever_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple retriever ranked lists.

    Args:
        ranked_lists: Each inner list is a ranked result from one retriever,
                      ordered by relevance (best first).
        k: RRF constant (default 60). Higher = smoother score distribution.
        retriever_weights: Optional weight per retriever type.
                          Defaults to DEFAULT_RETRIEVER_WEIGHTS.

    Returns:
        Dict mapping neuron_id -> fused RRF score. Scores are NOT
        normalized to [0,1] — they are relative within a query.
    """
    if not ranked_lists:
        return {}

    weights = retriever_weights or DEFAULT_RETRIEVER_WEIGHTS
    fused: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for anchor in ranked_list:
            w = weights.get(anchor.retriever, 1.0)
            contribution = w / (k + anchor.rank)
            fused[anchor.neuron_id] = fused.get(anchor.neuron_id, 0.0) + contribution

    return fused


def rrf_to_activation_levels(
    fused_scores: dict[str, float],
    min_level: float = 0.1,
    max_level: float = 1.0,
) -> dict[str, float]:
    """Convert RRF fused scores to initial activation levels in [min_level, max_level].

    Normalizes RRF scores linearly so the best-scored neuron gets max_level
    and the worst gets min_level. If only one neuron, it gets max_level.

    Args:
        fused_scores: Output of rrf_fuse().
        min_level: Floor activation level for lowest-scored anchor.
        max_level: Ceiling activation level for highest-scored anchor.

    Returns:
        Dict mapping neuron_id -> initial activation level.
    """
    if not fused_scores:
        return {}

    scores = list(fused_scores.values())
    best = max(scores)
    worst = min(scores)
    spread = best - worst

    if spread < 1e-12:
        # All scores equal — give everyone max_level
        return dict.fromkeys(fused_scores, max_level)

    return {
        nid: min_level + (score - worst) / spread * (max_level - min_level)
        for nid, score in fused_scores.items()
    }
