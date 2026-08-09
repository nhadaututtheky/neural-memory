"""Cascaded cognitive recall — route, candidate evidence, pre-graph gate.

Query-aware cascade:
  route → FTS/vector RRF candidates → sufficiency gate
       → sufficient: early lexical/vector result
       → insufficient/hard: bounded induced-graph activation

Preserves causal/temporal cognition: those routes never early-exit on
lexical score alone. Public ``ReflexPipeline.query()`` stays source-compatible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from neural_memory.engine.score_fusion import (
    DEFAULT_RETRIEVER_WEIGHTS,
    DEFAULT_RRF_K,
    RankedAnchor,
    rrf_fuse,
)
from neural_memory.extraction.parser import QueryIntent, Stimulus

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig

# Candidate sources used for pre-graph evidence (not graph_expansion).
_CANDIDATE_RETRIEVERS: frozenset[str] = frozenset(
    {
        "keyword",
        "bm25",
        "embedding",
        "text_relevance",
        "entity",
        "time",
        "fuzzy",
    }
)

# Mixed-intent priority: more cognitive wins.
_COGNITIVE_RANK: dict[str, int] = {
    "causal": 3,
    "temporal": 2,
    "semantic": 1,
    "exact": 0,
}

# Exact/code/path heuristics (local, deterministic).
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:[/\\][\w.\-]+)+\.\w{1,8}\b|"  # path with extension
    r"\b[\w\-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|md|toml|json|ya?ml|sql)\b",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?\b|"
    r"\b(?:version|ver|v)\s*[:=]?\s*\d+",
    re.IGNORECASE,
)
_CODE_ID_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")

_CAUSAL_TOKENS: frozenset[str] = frozenset(
    {
        "why",
        "because",
        "cause",
        "caused",
        "reason",
        "result",
        "effect",
        "consequence",
        "led",
        "leads",
        "how",
        "tại",
        "sao",
        "vì",
        "lý",
        "do",
        "nguyên",
        "nhân",
        "kết",
        "quả",
        "dẫn",
    }
)
_TEMPORAL_TOKENS: frozenset[str] = frozenset(
    {
        "when",
        "yesterday",
        "today",
        "tomorrow",
        "ago",
        "before",
        "after",
        "recently",
        "earlier",
        "morning",
        "afternoon",
        "evening",
        "week",
        "month",
        "year",
        "khi",
        "hôm",
        "qua",
        "nay",
        "tuần",
        "trước",
        "sau",
        "gần",
        "đây",
        "lúc",
    }
)
_EXACT_PHRASES: frozenset[str] = frozenset(
    {
        "what is",
        "what's",
        "exact",
        "specific",
        "precisely",
        "give me",
        "show me",
        "find",
        "là gì",
        "chính xác",
        "cụ thể",
    }
)

# Defaults when BrainConfig fields are absent (tests / older configs).
DEFAULT_CANDIDATE_CAP = 300
DEFAULT_EXACT_MARGIN = 0.30
DEFAULT_MIN_CANDIDATES = 1
DEFAULT_NEIGHBOR_HOPS = 1
DEFAULT_GRAPH_NODE_BUDGET = 500


class RecallRoute(StrEnum):
    """Deterministic cascade route for a query."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass(frozen=True)
class CandidateEvidence:
    """Pre-graph candidate set from FTS/vector (and related) RRF fusion.

    Attributes:
        route: Selected cascade route.
        ranked_ids: Neuron IDs ordered by fused score (best first).
        scores: Fused RRF scores keyed by neuron ID.
        source_counts: How many hits each retriever contributed.
    """

    route: RecallRoute
    ranked_ids: tuple[str, ...]
    scores: dict[str, float]
    source_counts: dict[str, int]


@dataclass(frozen=True)
class CandidateGateDecision:
    """Pre-graph sufficiency decision.

    Attributes:
        sufficient: True when cheap evidence is enough to skip full graph.
        confidence: Gate confidence in [0, 1].
        reason: Machine-readable reason code.
        graph_scope: Neuron IDs allowed for bounded graph activation.
    """

    sufficient: bool
    confidence: float
    reason: str
    graph_scope: frozenset[str]


@dataclass(frozen=True)
class TraversalBudget:
    """Budgets for bounded graph cognition."""

    max_nodes: int = DEFAULT_GRAPH_NODE_BUDGET
    max_hops: int = 2
    max_time_ms: float = 100.0
    neighbor_hops: int = DEFAULT_NEIGHBOR_HOPS


@dataclass(frozen=True)
class TraversalReport:
    """Report from a bounded graph traversal attempt."""

    scope_size: int
    hops_used: int
    nodes_visited: int
    budget_exhausted: bool
    fallback_reason: str | None = None


def route_query(stimulus: Stimulus) -> RecallRoute:
    """Map a parsed stimulus to a deterministic cascade route.

    Mixed intent prefers the more cognitive route (causal > temporal >
    semantic > exact). Empty / whitespace queries fall through to SEMANTIC
    so existing pipeline validation can reject them.

    Args:
        stimulus: Parsed query signals.

    Returns:
        Selected ``RecallRoute``.
    """
    raw = (stimulus.raw_query or "").strip()
    if not raw:
        return RecallRoute.SEMANTIC

    scores: dict[RecallRoute, float] = {
        RecallRoute.EXACT: 0.0,
        RecallRoute.SEMANTIC: 0.1,  # weak default prior
        RecallRoute.TEMPORAL: 0.0,
        RecallRoute.CAUSAL: 0.0,
    }

    query_lower = raw.lower()
    words = frozenset(query_lower.split())

    # Intent boosts from parser
    intent_map: dict[QueryIntent, RecallRoute] = {
        QueryIntent.ASK_WHY: RecallRoute.CAUSAL,
        QueryIntent.ASK_HOW: RecallRoute.CAUSAL,
        QueryIntent.ASK_WHEN: RecallRoute.TEMPORAL,
        QueryIntent.ASK_WHO: RecallRoute.EXACT,
        QueryIntent.ASK_WHERE: RecallRoute.EXACT,
        QueryIntent.CONFIRM: RecallRoute.EXACT,
        QueryIntent.COMPARE: RecallRoute.SEMANTIC,
        QueryIntent.ASK_WHAT: RecallRoute.SEMANTIC,
        QueryIntent.ASK_PATTERN: RecallRoute.SEMANTIC,
        QueryIntent.ASK_FEELING: RecallRoute.SEMANTIC,
        QueryIntent.RECALL: RecallRoute.SEMANTIC,
    }
    mapped = intent_map.get(stimulus.intent)
    if mapped is not None:
        scores[mapped] += 2.0

    # Token signals
    if words & _CAUSAL_TOKENS or any(t in query_lower for t in ("how come", "led to", "tại sao")):
        scores[RecallRoute.CAUSAL] += 2.5
    if words & _TEMPORAL_TOKENS or stimulus.has_time_context:
        scores[RecallRoute.TEMPORAL] += 2.5
        if stimulus.has_time_context:
            scores[RecallRoute.TEMPORAL] += 1.5
    if any(p in query_lower for p in _EXACT_PHRASES):
        scores[RecallRoute.EXACT] += 1.5

    # Code / path / version → exact
    if _PATH_RE.search(raw) or _VERSION_RE.search(raw) or _CODE_ID_RE.search(raw):
        scores[RecallRoute.EXACT] += 3.0

    # Entities without time lean exact/direct
    if stimulus.has_entities and not stimulus.has_time_context:
        scores[RecallRoute.EXACT] += 1.0

    # Pure conceptual keywords lean semantic
    if stimulus.keywords and not stimulus.has_time_context:
        scores[RecallRoute.SEMANTIC] += 0.5

    # Sort by score then cognitive rank for ties / near-ties
    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1], _COGNITIVE_RANK[item[0].value]),
        reverse=True,
    )
    primary, primary_score = ranked[0]
    secondary, secondary_score = ranked[1]

    # Mixed intent: if secondary is close and more cognitive, prefer it
    if (
        secondary_score > 0
        and primary_score - secondary_score < 1.5
        and _COGNITIVE_RANK[secondary.value] > _COGNITIVE_RANK[primary.value]
    ):
        return secondary

    return primary


def redistribute_retriever_weights(
    present_retrievers: set[str] | frozenset[str],
    base_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Redistribute weights of missing candidate sources onto present ones.

    When vector/embedding is unavailable, its weight is redistributed so
    FTS/keyword still carries full evidence mass.
    """
    weights = dict(base_weights or DEFAULT_RETRIEVER_WEIGHTS)
    present = {r for r in present_retrievers if r in _CANDIDATE_RETRIEVERS}
    if not present:
        return {}

    candidate_base = {
        r: weights.get(r, 1.0) for r in _CANDIDATE_RETRIEVERS if r in weights or r in present
    }
    for r in present:
        candidate_base.setdefault(r, weights.get(r, 1.0))

    present_sum = sum(candidate_base.get(r, 1.0) for r in present)
    total_sum = sum(candidate_base.values()) if candidate_base else present_sum
    if present_sum <= 0:
        return dict.fromkeys(present, 1.0)

    scale = total_sum / present_sum
    return {r: candidate_base.get(r, 1.0) * scale for r in present}


def build_candidate_evidence(
    route: RecallRoute,
    ranked_lists: list[list[RankedAnchor]],
    *,
    k: int = DEFAULT_RRF_K,
    retriever_weights: Mapping[str, float] | None = None,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    live_ids: frozenset[str] | set[str] | None = None,
) -> CandidateEvidence:
    """Fuse FTS/vector ranked lists into capped candidate evidence.

    Drops non-candidate retrievers (e.g. graph_expansion), filters stale
    IDs when ``live_ids`` is provided, and redistributes missing-source
    weights before RRF.
    """
    cap = max(1, min(int(candidate_cap), 10_000))
    source_counts: dict[str, int] = {}
    filtered_lists: list[list[RankedAnchor]] = []

    for ranked in ranked_lists:
        if not ranked:
            continue
        retriever = ranked[0].retriever
        if retriever not in _CANDIDATE_RETRIEVERS:
            continue
        kept: list[RankedAnchor] = []
        for anchor in ranked:
            if live_ids is not None and anchor.neuron_id not in live_ids:
                continue
            kept.append(anchor)
        if not kept:
            continue
        # Re-rank after stale drops so ranks stay 1-indexed contiguous
        renumbered = [
            RankedAnchor(
                neuron_id=a.neuron_id,
                rank=i + 1,
                retriever=a.retriever,
                score=a.score,
            )
            for i, a in enumerate(kept)
        ]
        filtered_lists.append(renumbered)
        source_counts[retriever] = source_counts.get(retriever, 0) + len(renumbered)

    if not filtered_lists:
        return CandidateEvidence(
            route=route,
            ranked_ids=(),
            scores={},
            source_counts={},
        )

    present = set(source_counts)
    weights = redistribute_retriever_weights(present, retriever_weights)
    fused = rrf_fuse(filtered_lists, k=k, retriever_weights=weights)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:cap]
    ranked_ids = tuple(nid for nid, _ in ordered)
    scores = dict(ordered)
    return CandidateEvidence(
        route=route,
        ranked_ids=ranked_ids,
        scores=scores,
        source_counts=dict(source_counts),
    )


def evaluate_candidates(
    evidence: CandidateEvidence,
    config: BrainConfig | None = None,
    *,
    exact_margin: float | None = None,
    min_candidates: int | None = None,
    candidate_cap: int | None = None,
) -> CandidateGateDecision:
    """Pre-graph sufficiency gate.

    Rules:
    - Causal/temporal never exit on lexical/vector score alone.
    - Exact may exit on high top-1 margin and non-empty candidates.
    - Semantic may exit on multi-source agreement + strong top score.
    - Empty / too-few candidates → insufficient.
    """
    margin = (
        exact_margin
        if exact_margin is not None
        else float(getattr(config, "cascade_exact_margin", DEFAULT_EXACT_MARGIN))
    )
    min_n = (
        min_candidates
        if min_candidates is not None
        else int(getattr(config, "cascade_min_candidates", DEFAULT_MIN_CANDIDATES))
    )
    cap = (
        candidate_cap
        if candidate_cap is not None
        else int(getattr(config, "cascade_candidate_cap", DEFAULT_CANDIDATE_CAP))
    )
    scope = frozenset(evidence.ranked_ids[: max(1, min(cap, len(evidence.ranked_ids) or 1))])

    if not evidence.ranked_ids or not evidence.scores:
        return CandidateGateDecision(
            sufficient=False,
            confidence=0.0,
            reason="empty_candidates",
            graph_scope=frozenset(),
        )

    if len(evidence.ranked_ids) < min_n:
        return CandidateGateDecision(
            sufficient=False,
            confidence=0.1,
            reason="low_candidate_count",
            graph_scope=scope,
        )

    # Causal / temporal always need graph cognition
    if evidence.route in (RecallRoute.CAUSAL, RecallRoute.TEMPORAL):
        return CandidateGateDecision(
            sufficient=False,
            confidence=0.0,
            reason="cognitive_route_requires_graph",
            graph_scope=scope,
        )

    top_id = evidence.ranked_ids[0]
    top_score = evidence.scores[top_id]
    second_score = evidence.scores[evidence.ranked_ids[1]] if len(evidence.ranked_ids) > 1 else 0.0
    # Relative margin: (top - second) / top (useful when multi-source boosts top)
    rel_margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    # Mass share: RRF rank-1 vs rank-2 gaps are tiny under k=60; concentration
    # of fused mass is a more stable exact-match signal.
    mass = sum(evidence.scores.values()) or 1.0
    top_share = top_score / mass
    multi_source = len(evidence.source_counts) >= 2
    cap_hit = len(evidence.ranked_ids) >= cap

    if evidence.route == RecallRoute.EXACT:
        sole_hit = len(evidence.ranked_ids) == 1
        # Dominant: sole hit, high mass concentration, multi-source head, or
        # large relative margin when fusion boosts the winner.
        dominant = (
            sole_hit
            or top_share >= (0.5 + margin * 0.5)
            or (multi_source and rel_margin >= margin * 0.5)
        )
        if dominant and top_score > 0:
            conf = min(
                1.0,
                0.55
                + top_share * 0.3
                + (0.1 if multi_source else 0.0)
                + (0.05 if sole_hit else 0.0),
            )
            return CandidateGateDecision(
                sufficient=True,
                confidence=conf,
                reason="exact_high_margin",
                graph_scope=scope,
            )
        return CandidateGateDecision(
            sufficient=False,
            confidence=max(0.15, top_share * 0.5, rel_margin),
            reason="exact_low_margin",
            graph_scope=scope,
        )

    # SEMANTIC
    if multi_source and (rel_margin >= margin * 0.75 or top_share >= 0.55) and top_score > 0:
        conf = min(1.0, 0.5 + max(rel_margin, top_share) * 0.35 + 0.1)
        return CandidateGateDecision(
            sufficient=True,
            confidence=conf,
            reason="semantic_multi_source_agreement",
            graph_scope=scope,
        )

    if cap_hit and not multi_source:
        return CandidateGateDecision(
            sufficient=False,
            confidence=0.2,
            reason="candidate_cap_uncertain",
            graph_scope=scope,
        )

    return CandidateGateDecision(
        sufficient=False,
        confidence=max(0.1, rel_margin * 0.5),
        reason="semantic_weak_signal",
        graph_scope=scope,
    )


def expand_scope_with_neighbors(
    seed_ids: frozenset[str] | set[str],
    adjacency: Mapping[str, list[str] | tuple[str, ...] | set[str]],
    *,
    neighbor_hops: int = DEFAULT_NEIGHBOR_HOPS,
    max_nodes: int = DEFAULT_GRAPH_NODE_BUDGET,
) -> frozenset[str]:
    """Expand candidate seeds by bounded neighbor hops (induced subgraph).

    Pure function — caller supplies adjacency. Empty adjacency returns seeds.
    """
    if not seed_ids:
        return frozenset()

    max_nodes = max(1, int(max_nodes))
    hops = max(0, int(neighbor_hops))
    scope: set[str] = set(seed_ids)
    frontier: set[str] = set(seed_ids)

    for _ in range(hops):
        if len(scope) >= max_nodes:
            break
        nxt: set[str] = set()
        for nid in frontier:
            for neighbor in adjacency.get(nid, ()):
                if neighbor not in scope:
                    nxt.add(neighbor)
                    if len(scope) + len(nxt) >= max_nodes:
                        break
            if len(scope) + len(nxt) >= max_nodes:
                break
        if not nxt:
            break
        # Trim if over budget
        remaining = max_nodes - len(scope)
        if remaining <= 0:
            break
        if len(nxt) > remaining:
            # Deterministic trim for stability
            nxt = set(sorted(nxt)[:remaining])
        scope.update(nxt)
        frontier = nxt

    return frozenset(scope)


async def build_neighbor_adjacency(
    storage: Any,
    seed_ids: frozenset[str] | set[str],
    *,
    neighbor_hops: int = DEFAULT_NEIGHBOR_HOPS,
    max_nodes: int = DEFAULT_GRAPH_NODE_BUDGET,
) -> dict[str, list[str]]:
    """Fetch neighbor adjacency for seed expansion via storage.get_neighbors."""
    if not seed_ids or not hasattr(storage, "get_neighbors"):
        return {}

    adjacency: dict[str, list[str]] = {}
    frontier: set[str] = set(seed_ids)
    seen: set[str] = set(seed_ids)
    hops = max(0, int(neighbor_hops))

    for _ in range(max(hops, 1)):
        if len(seen) >= max_nodes:
            break
        next_frontier: set[str] = set()
        for nid in frontier:
            try:
                neighbors = await storage.get_neighbors(nid, direction="both")
            except Exception:
                continue
            ids: list[str] = []
            for item in neighbors or []:
                # get_neighbors may return (Neuron, Synapse) or Neuron
                if isinstance(item, tuple):
                    neuron = item[0]
                    nid_n = getattr(neuron, "id", None)
                else:
                    nid_n = getattr(item, "id", item if isinstance(item, str) else None)
                if isinstance(nid_n, str) and nid_n:
                    ids.append(nid_n)
                    if nid_n not in seen:
                        next_frontier.add(nid_n)
            adjacency[nid] = ids
            seen.update(ids)
            if len(seen) >= max_nodes:
                break
        frontier = next_frontier
        if not frontier:
            break

    return adjacency


def scores_to_activation_levels(
    scores: Mapping[str, float],
    *,
    min_level: float = 0.1,
    max_level: float = 1.0,
) -> dict[str, float]:
    """Normalize candidate scores to activation levels for early-exit path."""
    if not scores:
        return {}
    vals = list(scores.values())
    best = max(vals)
    worst = min(vals)
    spread = best - worst
    if spread < 1e-12:
        return dict.fromkeys(scores, max_level)
    return {
        nid: min_level + (score - worst) / spread * (max_level - min_level)
        for nid, score in scores.items()
    }


def delta_phase_timings(cumulative_ms: Mapping[str, float]) -> dict[str, float]:
    """Convert cumulative phase timestamps into per-phase deltas.

    Expects ordered insertion of cumulative checkpoints (each value is
    elapsed ms from query start). Returns non-negative deltas.
    """
    if not cumulative_ms:
        return {}
    items = list(cumulative_ms.items())
    deltas: dict[str, float] = {}
    prev = 0.0
    for name, cum in items:
        raw = float(cum) - prev
        deltas[name] = max(0.0, raw)
        prev = float(cum)
    return deltas
