"""Unit tests for cascaded cognitive recall routing and candidate gate."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.engine.retrieval_cascade import (
    DEFAULT_CANDIDATE_CAP,
    CandidateEvidence,
    CandidateGateDecision,
    RecallRoute,
    TraversalBudget,
    build_candidate_evidence,
    delta_phase_timings,
    evaluate_candidates,
    expand_scope_with_neighbors,
    redistribute_retriever_weights,
    route_query,
    scores_to_activation_levels,
)
from neural_memory.engine.score_fusion import RankedAnchor
from neural_memory.extraction.parser import Perspective, QueryIntent, Stimulus


def _stimulus(
    query: str,
    *,
    intent: QueryIntent = QueryIntent.RECALL,
    keywords: list[str] | None = None,
    has_time: bool = False,
) -> Stimulus:
    from neural_memory.extraction.temporal import TimeHint
    from neural_memory.utils.timeutils import utcnow

    time_hints: list[TimeHint] = []
    if has_time:
        from neural_memory.extraction.temporal import TimeGranularity

        now = utcnow()
        time_hints = [
            TimeHint(
                original="yesterday",
                absolute_start=now,
                absolute_end=now,
                granularity=TimeGranularity.DAY,
                is_fuzzy=True,
            )
        ]
    return Stimulus(
        time_hints=time_hints,
        keywords=keywords or [],
        entities=[],
        intent=intent,
        perspective=Perspective.RECALL,
        raw_query=query,
        language="en",
    )


class TestRecallRoute:
    def test_enum_values(self) -> None:
        assert RecallRoute.EXACT == "exact"
        assert RecallRoute.SEMANTIC == "semantic"
        assert RecallRoute.TEMPORAL == "temporal"
        assert RecallRoute.CAUSAL == "causal"


class TestRouteQuery:
    def test_empty_query_semantic(self) -> None:
        assert route_query(_stimulus("")) is RecallRoute.SEMANTIC
        assert route_query(_stimulus("   ")) is RecallRoute.SEMANTIC

    def test_why_is_causal(self) -> None:
        s = _stimulus("Why did the build fail?", intent=QueryIntent.ASK_WHY)
        assert route_query(s) is RecallRoute.CAUSAL

    def test_when_is_temporal(self) -> None:
        s = _stimulus(
            "When did we ship version 2?",
            intent=QueryIntent.ASK_WHEN,
            has_time=True,
        )
        assert route_query(s) is RecallRoute.TEMPORAL

    def test_mixed_why_yesterday_prefers_causal(self) -> None:
        """Mixed intent chooses the more cognitive route (causal > temporal)."""
        s = _stimulus(
            "Why did the build fail yesterday?",
            intent=QueryIntent.ASK_WHY,
            has_time=True,
            keywords=["build", "fail", "yesterday"],
        )
        assert route_query(s) is RecallRoute.CAUSAL

    def test_path_query_is_exact(self) -> None:
        s = _stimulus(
            "What is in src/neural_memory/engine/retrieval.py?",
            intent=QueryIntent.ASK_WHAT,
        )
        assert route_query(s) is RecallRoute.EXACT

    def test_version_query_leans_exact(self) -> None:
        s = _stimulus("What is the current version 4.55.1?", intent=QueryIntent.ASK_WHAT)
        assert route_query(s) is RecallRoute.EXACT

    def test_conceptual_is_semantic(self) -> None:
        s = _stimulus(
            "Tell me about spreading activation",
            intent=QueryIntent.ASK_WHAT,
            keywords=["spreading", "activation"],
        )
        assert route_query(s) is RecallRoute.SEMANTIC

    def test_deterministic(self) -> None:
        s = _stimulus("Why did auth break?", intent=QueryIntent.ASK_WHY)
        assert route_query(s) is route_query(s)


class TestCandidateEvidence:
    def test_empty_lists(self) -> None:
        ev = build_candidate_evidence(RecallRoute.SEMANTIC, [])
        assert ev.ranked_ids == ()
        assert ev.scores == {}
        assert ev.source_counts == {}

    def test_rrf_orders_by_score(self) -> None:
        fts = [
            RankedAnchor(neuron_id="a", rank=1, retriever="keyword", score=10.0),
            RankedAnchor(neuron_id="b", rank=2, retriever="keyword", score=5.0),
        ]
        emb = [
            RankedAnchor(neuron_id="a", rank=1, retriever="embedding", score=0.9),
            RankedAnchor(neuron_id="c", rank=2, retriever="embedding", score=0.5),
        ]
        ev = build_candidate_evidence(RecallRoute.SEMANTIC, [fts, emb])
        assert ev.ranked_ids[0] == "a"
        assert "a" in ev.scores and "b" in ev.scores and "c" in ev.scores
        assert ev.source_counts["keyword"] == 2
        assert ev.source_counts["embedding"] == 2

    def test_stale_ids_filtered(self) -> None:
        fts = [
            RankedAnchor(neuron_id="live", rank=1, retriever="bm25"),
            RankedAnchor(neuron_id="ghost", rank=2, retriever="bm25"),
        ]
        ev = build_candidate_evidence(
            RecallRoute.EXACT,
            [fts],
            live_ids=frozenset({"live"}),
        )
        assert ev.ranked_ids == ("live",)
        assert "ghost" not in ev.scores

    def test_candidate_cap(self) -> None:
        fts = [RankedAnchor(neuron_id=f"n{i}", rank=i + 1, retriever="keyword") for i in range(50)]
        ev = build_candidate_evidence(RecallRoute.SEMANTIC, [fts], candidate_cap=10)
        assert len(ev.ranked_ids) == 10

    def test_graph_expansion_excluded(self) -> None:
        graph = [RankedAnchor(neuron_id="g1", rank=1, retriever="graph_expansion")]
        fts = [RankedAnchor(neuron_id="f1", rank=1, retriever="keyword")]
        ev = build_candidate_evidence(RecallRoute.SEMANTIC, [graph, fts])
        assert "g1" not in ev.scores
        assert "f1" in ev.scores

    def test_frozen(self) -> None:
        ev = CandidateEvidence(
            route=RecallRoute.EXACT,
            ranked_ids=("a",),
            scores={"a": 1.0},
            source_counts={"keyword": 1},
        )
        with pytest.raises(FrozenInstanceError):
            ev.route = RecallRoute.SEMANTIC  # type: ignore[misc]


class TestRedistributeWeights:
    def test_embedding_missing_boosts_keyword(self) -> None:
        base = {"keyword": 0.7, "embedding": 1.0, "bm25": 0.7}
        out = redistribute_retriever_weights({"keyword", "bm25"}, base)
        assert set(out) == {"keyword", "bm25"}
        # Total mass preserved among present candidate sources
        present_base = 0.7 + 0.7
        total = 0.7 + 1.0 + 0.7
        assert abs(sum(out.values()) - total) < 1e-9
        assert abs(out["keyword"] - 0.7 * (total / present_base)) < 1e-9

    def test_empty_present(self) -> None:
        assert redistribute_retriever_weights(set()) == {}


class TestEvaluateCandidates:
    def _ev(
        self,
        route: RecallRoute,
        scores: dict[str, float],
        sources: dict[str, int] | None = None,
    ) -> CandidateEvidence:
        ranked = tuple(sorted(scores, key=lambda n: scores[n], reverse=True))
        return CandidateEvidence(
            route=route,
            ranked_ids=ranked,
            scores=scores,
            source_counts=sources or {"keyword": len(scores)},
        )

    def test_empty_insufficient(self) -> None:
        d = evaluate_candidates(self._ev(RecallRoute.EXACT, {}))
        assert d.sufficient is False
        assert d.reason == "empty_candidates"

    def test_causal_never_early_exit(self) -> None:
        d = evaluate_candidates(
            self._ev(RecallRoute.CAUSAL, {"a": 10.0, "b": 0.1}, {"keyword": 2, "embedding": 1})
        )
        assert d.sufficient is False
        assert d.reason == "cognitive_route_requires_graph"
        assert "a" in d.graph_scope

    def test_temporal_never_early_exit(self) -> None:
        d = evaluate_candidates(self._ev(RecallRoute.TEMPORAL, {"a": 9.0, "b": 0.1}))
        assert d.sufficient is False
        assert d.reason == "cognitive_route_requires_graph"

    def test_exact_high_margin_sufficient(self) -> None:
        # Mass concentration: a holds 1.0/1.2 ≈ 83% > 0.65 threshold
        d = evaluate_candidates(
            self._ev(RecallRoute.EXACT, {"a": 1.0, "b": 0.2}),
            exact_margin=0.3,
        )
        assert d.sufficient is True
        assert d.reason == "exact_high_margin"
        assert d.confidence > 0.5

    def test_exact_sole_hit_sufficient(self) -> None:
        d = evaluate_candidates(self._ev(RecallRoute.EXACT, {"a": 0.5}), exact_margin=0.3)
        assert d.sufficient is True
        assert d.reason == "exact_high_margin"

    def test_exact_tied_scores_insufficient(self) -> None:
        d = evaluate_candidates(
            self._ev(RecallRoute.EXACT, {"a": 1.0, "b": 1.0}),
            exact_margin=0.3,
        )
        assert d.sufficient is False
        assert d.reason == "exact_low_margin"

    def test_semantic_multi_source_agreement(self) -> None:
        d = evaluate_candidates(
            self._ev(
                RecallRoute.SEMANTIC,
                {"a": 1.0, "b": 0.2},
                {"keyword": 2, "embedding": 2},
            ),
            exact_margin=0.3,
        )
        assert d.sufficient is True
        assert d.reason == "semantic_multi_source_agreement"

    def test_semantic_weak_single_source(self) -> None:
        d = evaluate_candidates(
            self._ev(RecallRoute.SEMANTIC, {"a": 1.0, "b": 0.9}, {"keyword": 2}),
            exact_margin=0.3,
        )
        assert d.sufficient is False

    def test_graph_scope_capped(self) -> None:
        scores = {f"n{i}": float(100 - i) for i in range(20)}
        d = evaluate_candidates(
            self._ev(RecallRoute.CAUSAL, scores),
            candidate_cap=5,
        )
        assert len(d.graph_scope) == 5

    def test_decision_frozen(self) -> None:
        d = CandidateGateDecision(
            sufficient=False,
            confidence=0.0,
            reason="x",
            graph_scope=frozenset(),
        )
        with pytest.raises(FrozenInstanceError):
            d.sufficient = True  # type: ignore[misc]


class TestExpandScope:
    def test_empty_seeds(self) -> None:
        assert expand_scope_with_neighbors(frozenset(), {"a": ["b"]}) == frozenset()

    def test_one_hop(self) -> None:
        adj = {"a": ["b", "c"], "b": ["d"], "c": []}
        scope = expand_scope_with_neighbors({"a"}, adj, neighbor_hops=1)
        assert scope == frozenset({"a", "b", "c"})

    def test_two_hops(self) -> None:
        adj = {"a": ["b"], "b": ["c"], "c": ["d"]}
        scope = expand_scope_with_neighbors({"a"}, adj, neighbor_hops=2)
        assert "c" in scope
        assert "d" not in scope  # hop 3

    def test_max_nodes_budget(self) -> None:
        adj = {"a": [f"n{i}" for i in range(20)]}
        scope = expand_scope_with_neighbors({"a"}, adj, neighbor_hops=1, max_nodes=5)
        assert len(scope) == 5
        assert "a" in scope


class TestHelpers:
    def test_scores_to_activation_levels(self) -> None:
        levels = scores_to_activation_levels({"a": 10.0, "b": 0.0})
        assert levels["a"] == pytest.approx(1.0)
        assert levels["b"] == pytest.approx(0.1)

    def test_delta_phase_timings(self) -> None:
        cum = {"route": 2.0, "candidates": 10.0, "gate": 12.0, "graph": 40.0}
        deltas = delta_phase_timings(cum)
        assert deltas["route"] == pytest.approx(2.0)
        assert deltas["candidates"] == pytest.approx(8.0)
        assert deltas["gate"] == pytest.approx(2.0)
        assert deltas["graph"] == pytest.approx(28.0)

    def test_traversal_budget_defaults(self) -> None:
        b = TraversalBudget()
        assert b.max_nodes == 500
        assert DEFAULT_CANDIDATE_CAP == 300
