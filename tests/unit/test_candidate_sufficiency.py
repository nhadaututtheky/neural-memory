"""Pre-graph candidate sufficiency gate tests (Phase 4 Wave 2)."""

from __future__ import annotations

from neural_memory.core.brain import BrainConfig
from neural_memory.engine.retrieval_cascade import (
    CandidateEvidence,
    RecallRoute,
    build_candidate_evidence,
    evaluate_candidates,
)
from neural_memory.engine.score_fusion import RankedAnchor


def _evidence(
    route: RecallRoute,
    ranked: list[tuple[str, float]],
    sources: dict[str, int] | None = None,
) -> CandidateEvidence:
    ranked_ids = tuple(nid for nid, _ in ranked)
    scores = dict(ranked)
    return CandidateEvidence(
        route=route,
        ranked_ids=ranked_ids,
        scores=scores,
        source_counts=sources or {"keyword": len(ranked)},
    )


class TestCandidateSufficiency:
    def test_config_defaults(self) -> None:
        cfg = BrainConfig()
        assert cfg.cascade_recall_enabled is True
        assert cfg.cascade_candidate_cap == 300
        assert cfg.cascade_exact_margin == 0.3
        assert cfg.cascade_min_candidates == 1
        assert cfg.cascade_neighbor_hops == 1
        assert cfg.cascade_graph_node_budget == 500

    def test_unavailable_vector_still_builds_fts_evidence(self) -> None:
        fts = [
            RankedAnchor(neuron_id="a", rank=1, retriever="keyword"),
            RankedAnchor(neuron_id="b", rank=2, retriever="keyword"),
        ]
        # Only FTS list — vector missing
        ev = build_candidate_evidence(RecallRoute.EXACT, [fts])
        assert ev.ranked_ids[0] == "a"
        assert "embedding" not in ev.source_counts
        assert "keyword" in ev.source_counts

    def test_causal_strong_lexical_still_requires_graph(self) -> None:
        ev = _evidence(
            RecallRoute.CAUSAL,
            [("a", 10.0), ("b", 0.01)],
            {"keyword": 5, "embedding": 5},
        )
        decision = evaluate_candidates(ev, BrainConfig())
        assert decision.sufficient is False
        assert decision.reason == "cognitive_route_requires_graph"

    def test_exact_margin_from_config(self) -> None:
        cfg = BrainConfig(cascade_exact_margin=0.5)
        # top_share 1/1.6=0.625 < 0.75 threshold → insufficient
        ev = _evidence(RecallRoute.EXACT, [("a", 1.0), ("b", 0.6)])
        d = evaluate_candidates(ev, cfg)
        assert d.sufficient is False

        # top_share 1/1.15≈0.87 > 0.75 → sufficient
        ev2 = _evidence(RecallRoute.EXACT, [("a", 1.0), ("b", 0.15)])
        d2 = evaluate_candidates(ev2, cfg)
        assert d2.sufficient is True

    def test_one_retriever_degradation(self) -> None:
        """Single retriever sole hit still allows exact early-exit (vector absent)."""
        fts = [
            RankedAnchor(neuron_id="hit", rank=1, retriever="bm25", score=12.0),
        ]
        ev = build_candidate_evidence(RecallRoute.EXACT, [fts])
        d = evaluate_candidates(ev, BrainConfig(cascade_exact_margin=0.3))
        assert d.sufficient is True
        assert d.reason == "exact_high_margin"
        assert "embedding" not in ev.source_counts
