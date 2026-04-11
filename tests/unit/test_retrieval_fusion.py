"""Tests for the hybrid retrieval fusion module."""

from __future__ import annotations

import pytest

from neural_memory.engine.retrieval_fusion import (
    FusionResult,
    FusionWeights,
    _normalize,
    fuse_scores,
    select_weights,
)


class TestNormalize:
    """Tests for min-max normalization."""

    def test_empty_input(self) -> None:
        assert _normalize({}) == {}

    def test_single_value(self) -> None:
        result = _normalize({"a": 5.0})
        assert result == {"a": 1.0}

    def test_equal_values(self) -> None:
        result = _normalize({"a": 3.0, "b": 3.0})
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_different_values(self) -> None:
        result = _normalize({"a": 0.0, "b": 10.0, "c": 5.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(1.0)
        assert result["c"] == pytest.approx(0.5)

    def test_negative_values(self) -> None:
        result = _normalize({"a": -10.0, "b": 10.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(1.0)


class TestFuseScores:
    """Tests for tri-modal score fusion."""

    def test_empty_all_channels(self) -> None:
        results = fuse_scores({}, {}, {}, FusionWeights())
        assert results == []

    def test_single_channel_graph_only(self) -> None:
        graph = {"f1": 1.0, "f2": 0.5}
        results = fuse_scores(graph, {}, {}, FusionWeights())
        # When only graph channel active, its weight is redistributed to 1.0
        assert len(results) == 2
        # f1 has higher graph score -> higher fused score
        assert results[0].fiber_id == "f1"
        assert results[0].fused_score > results[1].fused_score
        # Graph score should be normalized to 1.0 for f1
        assert results[0].graph_score == pytest.approx(1.0)

    def test_single_channel_semantic_only(self) -> None:
        semantic = {"f1": 0.9, "f2": 0.3}
        results = fuse_scores({}, semantic, {}, FusionWeights())
        assert len(results) == 2
        assert results[0].fiber_id == "f1"
        # When only semantic channel, weight redistributed to 1.0
        assert results[0].fused_score == pytest.approx(1.0)

    def test_balanced_two_channels(self) -> None:
        graph = {"f1": 1.0, "f2": 0.0}
        semantic = {"f1": 0.0, "f2": 1.0}
        weights = FusionWeights(graph=0.5, semantic=0.5, lexical=0.0)
        results = fuse_scores(graph, semantic, {}, weights)
        # Both fibers should have equal fused scores (0.5 each)
        scores = {r.fiber_id: r.fused_score for r in results}
        assert scores["f1"] == pytest.approx(scores["f2"])

    def test_three_channels_weighted(self) -> None:
        graph = {"f1": 10.0}  # Only f1 in graph
        semantic = {"f1": 0.8, "f2": 0.9}
        lexical = {"f2": 5.0}  # Only f2 in lexical
        weights = FusionWeights(graph=0.5, semantic=0.3, lexical=0.2)
        results = fuse_scores(graph, semantic, lexical, weights)
        assert len(results) == 2
        # f1: graph=1.0, semantic=0.0 (normalized min), lexical=0.0 (absent)
        # f2: graph=0.0 (absent), semantic=1.0 (normalized max), lexical=1.0
        r_map = {r.fiber_id: r for r in results}
        assert r_map["f1"].graph_score == pytest.approx(1.0)
        assert r_map["f2"].lexical_score == pytest.approx(1.0)

    def test_weight_redistribution_empty_channel(self) -> None:
        """When lexical channel is empty, its weight is redistributed."""
        graph = {"f1": 1.0}
        semantic = {"f1": 1.0}
        weights = FusionWeights(graph=0.4, semantic=0.4, lexical=0.2)
        results = fuse_scores(graph, semantic, {}, weights)
        # Effective weights: graph=0.4/0.8=0.5, semantic=0.4/0.8=0.5
        assert len(results) == 1
        # Both channels normalized to 1.0, each weighted 0.5 -> fused = 1.0
        assert results[0].fused_score == pytest.approx(1.0)

    def test_different_fiber_sets_across_channels(self) -> None:
        """Fibers can appear in different subsets of channels."""
        graph = {"f1": 1.0, "f2": 0.5}
        semantic = {"f2": 0.9, "f3": 0.7}
        lexical = {"f1": 0.3, "f3": 0.8}
        weights = FusionWeights(graph=0.34, semantic=0.33, lexical=0.33)
        results = fuse_scores(graph, semantic, lexical, weights)
        assert len(results) == 3
        fiber_ids = {r.fiber_id for r in results}
        assert fiber_ids == {"f1", "f2", "f3"}

    def test_all_zero_weights(self) -> None:
        """All-zero weights returns empty (total weight < epsilon)."""
        graph = {"f1": 1.0}
        weights = FusionWeights(graph=0.0, semantic=0.0, lexical=0.0)
        results = fuse_scores(graph, {}, {}, weights)
        assert results == []

    def test_sorted_by_fused_score_descending(self) -> None:
        graph = {"f1": 0.1, "f2": 0.5, "f3": 1.0}
        results = fuse_scores(graph, {}, {}, FusionWeights())
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_does_not_mutate_inputs(self) -> None:
        """Fusion must not mutate input dicts."""
        graph = {"f1": 1.0, "f2": 0.5}
        semantic = {"f1": 0.8}
        lexical = {"f2": 0.3}
        graph_copy = dict(graph)
        semantic_copy = dict(semantic)
        lexical_copy = dict(lexical)
        fuse_scores(graph, semantic, lexical, FusionWeights())
        assert graph == graph_copy
        assert semantic == semantic_copy
        assert lexical == lexical_copy


class TestSelectWeights:
    """Tests for query-intent-based weight selection."""

    def test_factual_intents_favor_lexical(self) -> None:
        for intent in ("ask_what", "ask_where", "ask_who", "confirm", "factual"):
            w = select_weights(intent)
            assert w.lexical >= w.graph
            assert w.lexical >= w.semantic

    def test_semantic_intents_favor_embedding(self) -> None:
        for intent in ("ask_why", "ask_how", "ask_feeling", "semantic"):
            w = select_weights(intent)
            assert w.semantic >= w.graph
            assert w.semantic >= w.lexical

    def test_temporal_intents_favor_graph(self) -> None:
        for intent in ("ask_when", "ask_pattern", "temporal"):
            w = select_weights(intent)
            assert w.graph >= w.semantic
            assert w.graph >= w.lexical

    def test_unknown_intent_returns_default(self) -> None:
        w = select_weights("nonexistent_intent")
        default = FusionWeights()
        assert w.graph == default.graph
        assert w.semantic == default.semantic
        assert w.lexical == default.lexical

    def test_weights_sum_approximately_one(self) -> None:
        """All presets should have weights summing to ~1.0."""
        for intent in (
            "ask_what",
            "ask_when",
            "ask_why",
            "recall",
            "unknown",
            "factual",
            "semantic",
            "temporal",
            "causal",
            "exploratory",
        ):
            w = select_weights(intent)
            assert w.graph + w.semantic + w.lexical == pytest.approx(1.0, abs=0.01)


class TestFusionWeightsDataclass:
    """Tests for FusionWeights frozen dataclass."""

    def test_defaults(self) -> None:
        w = FusionWeights()
        assert w.graph == 0.5
        assert w.semantic == 0.3
        assert w.lexical == 0.2

    def test_frozen(self) -> None:
        w = FusionWeights()
        with pytest.raises(AttributeError):
            w.graph = 0.9  # type: ignore[misc]

    def test_custom_values(self) -> None:
        w = FusionWeights(graph=0.1, semantic=0.8, lexical=0.1)
        assert w.semantic == 0.8


class TestFusionResultDataclass:
    """Tests for FusionResult frozen dataclass."""

    def test_creation(self) -> None:
        r = FusionResult(
            fiber_id="f1",
            graph_score=0.8,
            semantic_score=0.5,
            lexical_score=0.3,
            fused_score=0.6,
        )
        assert r.fiber_id == "f1"
        assert r.fused_score == 0.6

    def test_frozen(self) -> None:
        r = FusionResult(
            fiber_id="f1",
            graph_score=0.8,
            semantic_score=0.5,
            lexical_score=0.3,
            fused_score=0.6,
        )
        with pytest.raises(AttributeError):
            r.fused_score = 0.9  # type: ignore[misc]
