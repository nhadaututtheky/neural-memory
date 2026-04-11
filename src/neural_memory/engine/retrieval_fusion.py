"""Tri-modal retrieval fusion: graph + semantic + lexical.

Normalizes per-channel scores to [0,1] and applies configurable weights
to produce a unified fiber ranking. Channels that have no scores are
gracefully skipped (weight redistributed to active channels).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionWeights:
    """Per-channel weights for tri-modal fusion.

    Attributes:
        graph: Weight for graph activation channel (spreading activation).
        semantic: Weight for semantic channel (embedding cosine similarity).
        lexical: Weight for lexical channel (BM25 / keyword IDF).
    """

    graph: float = 0.5
    semantic: float = 0.3
    lexical: float = 0.2


@dataclass(frozen=True)
class FusionResult:
    """Fused score for a single fiber across all retrieval channels.

    Attributes:
        fiber_id: The fiber this result refers to.
        graph_score: Normalized graph activation score [0,1].
        semantic_score: Normalized semantic similarity score [0,1].
        lexical_score: Normalized lexical relevance score [0,1].
        fused_score: Final weighted sum.
    """

    fiber_id: str
    graph_score: float
    semantic_score: float
    lexical_score: float
    fused_score: float


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1].

    Returns empty dict if input is empty. Returns all-1.0 if all values equal.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    spread = hi - lo

    if spread < 1e-12:
        return dict.fromkeys(scores, 1.0)

    return {k: (v - lo) / spread for k, v in scores.items()}


def fuse_scores(
    graph_scores: dict[str, float],
    semantic_scores: dict[str, float],
    lexical_scores: dict[str, float],
    weights: FusionWeights,
) -> list[FusionResult]:
    """Fuse tri-modal scores into a unified ranking.

    Each channel is normalized independently to [0,1] before weighting.
    If a channel is entirely empty, its weight is redistributed proportionally
    to the active channels.

    Args:
        graph_scores: fiber_id -> raw graph activation score.
        semantic_scores: fiber_id -> raw semantic similarity score.
        lexical_scores: fiber_id -> raw lexical relevance score.
        weights: Per-channel fusion weights.

    Returns:
        List of FusionResult sorted by fused_score descending.
    """
    # Collect all fiber IDs across channels
    all_ids: set[str] = set(graph_scores) | set(semantic_scores) | set(lexical_scores)
    if not all_ids:
        return []

    # Normalize each channel independently
    norm_graph = _normalize(graph_scores)
    norm_semantic = _normalize(semantic_scores)
    norm_lexical = _normalize(lexical_scores)

    # Compute effective weights — redistribute if a channel is empty
    raw_weights = {
        "graph": weights.graph if graph_scores else 0.0,
        "semantic": weights.semantic if semantic_scores else 0.0,
        "lexical": weights.lexical if lexical_scores else 0.0,
    }
    total_weight = sum(raw_weights.values())
    if total_weight < 1e-12:
        return []

    w_graph = raw_weights["graph"] / total_weight
    w_semantic = raw_weights["semantic"] / total_weight
    w_lexical = raw_weights["lexical"] / total_weight

    results: list[FusionResult] = []
    for fid in all_ids:
        g = norm_graph.get(fid, 0.0)
        s = norm_semantic.get(fid, 0.0)
        lx = norm_lexical.get(fid, 0.0)
        fused = g * w_graph + s * w_semantic + lx * w_lexical
        results.append(
            FusionResult(
                fiber_id=fid,
                graph_score=g,
                semantic_score=s,
                lexical_score=lx,
                fused_score=fused,
            )
        )

    results.sort(key=lambda r: r.fused_score, reverse=True)
    return results


# -- Query-type weight presets ------------------------------------------------

_INTENT_WEIGHTS: dict[str, FusionWeights] = {
    # Factual intents: lexical keywords matter most (exact term matching)
    "ask_what": FusionWeights(graph=0.3, semantic=0.2, lexical=0.5),
    "ask_where": FusionWeights(graph=0.3, semantic=0.2, lexical=0.5),
    "ask_who": FusionWeights(graph=0.3, semantic=0.2, lexical=0.5),
    "confirm": FusionWeights(graph=0.3, semantic=0.2, lexical=0.5),
    # Semantic intents: embedding similarity dominates
    "ask_why": FusionWeights(graph=0.2, semantic=0.6, lexical=0.2),
    "ask_how": FusionWeights(graph=0.2, semantic=0.6, lexical=0.2),
    "ask_feeling": FusionWeights(graph=0.2, semantic=0.6, lexical=0.2),
    "compare": FusionWeights(graph=0.25, semantic=0.5, lexical=0.25),
    # Temporal intents: graph structure (time neurons, sequential synapses) is key
    "ask_when": FusionWeights(graph=0.6, semantic=0.2, lexical=0.2),
    "ask_pattern": FusionWeights(graph=0.6, semantic=0.2, lexical=0.2),
    "recall": FusionWeights(graph=0.4, semantic=0.35, lexical=0.25),
    "unknown": FusionWeights(graph=0.5, semantic=0.3, lexical=0.2),
    # Legacy aliases for direct use
    "factual": FusionWeights(graph=0.3, semantic=0.2, lexical=0.5),
    "semantic": FusionWeights(graph=0.2, semantic=0.6, lexical=0.2),
    "temporal": FusionWeights(graph=0.6, semantic=0.2, lexical=0.2),
    "causal": FusionWeights(graph=0.6, semantic=0.25, lexical=0.15),
    "exploratory": FusionWeights(graph=0.4, semantic=0.35, lexical=0.25),
}

_DEFAULT_WEIGHTS = FusionWeights()


def select_weights(query_intent: str) -> FusionWeights:
    """Select fusion weights based on query intent type.

    Args:
        query_intent: A QueryIntent value string (e.g. "factual", "temporal").

    Returns:
        FusionWeights tuned for the query type, or default balanced weights.
    """
    return _INTENT_WEIGHTS.get(query_intent, _DEFAULT_WEIGHTS)
