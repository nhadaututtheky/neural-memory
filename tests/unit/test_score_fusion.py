"""Tests for Reciprocal Rank Fusion score blending."""

from __future__ import annotations

import pytest

from neural_memory.engine.score_fusion import (
    DEFAULT_RRF_K,
    RankedAnchor,
    rrf_fuse,
    rrf_to_activation_levels,
)


class TestRankedAnchor:
    """Test RankedAnchor frozen dataclass."""

    def test_create(self) -> None:
        anchor = RankedAnchor(neuron_id="n1", rank=1, retriever="fts")
        assert anchor.neuron_id == "n1"
        assert anchor.rank == 1
        assert anchor.retriever == "fts"
        assert anchor.score == 0.0

    def test_immutable(self) -> None:
        anchor = RankedAnchor(neuron_id="n1", rank=1, retriever="fts")
        try:
            anchor.rank = 2  # type: ignore[misc]
            pytest.fail("Should have raised AttributeError")
        except AttributeError:
            pass


class TestRRFFuse:
    """Test rrf_fuse with various inputs."""

    def test_empty_input(self) -> None:
        assert rrf_fuse([]) == {}

    def test_single_retriever(self) -> None:
        ranked = [
            RankedAnchor(neuron_id="a", rank=1, retriever="entity"),
            RankedAnchor(neuron_id="b", rank=2, retriever="entity"),
        ]
        scores = rrf_fuse([ranked])
        assert scores["a"] > scores["b"]

    def test_single_retriever_formula(self) -> None:
        """Verify exact RRF formula: weight / (k + rank)."""
        ranked = [RankedAnchor(neuron_id="a", rank=1, retriever="entity")]
        scores = rrf_fuse([ranked], k=60)
        # entity weight = 0.9 (from DEFAULT_RETRIEVER_WEIGHTS)
        expected = 0.9 / (60 + 1)
        assert abs(scores["a"] - expected) < 1e-10

    def test_two_retrievers_boost_intersection(self) -> None:
        """Neuron appearing in both retrievers gets higher score."""
        fts_list = [
            RankedAnchor(neuron_id="common", rank=1, retriever="keyword"),
            RankedAnchor(neuron_id="fts_only", rank=2, retriever="keyword"),
        ]
        embed_list = [
            RankedAnchor(neuron_id="common", rank=1, retriever="embedding"),
            RankedAnchor(neuron_id="embed_only", rank=2, retriever="embedding"),
        ]
        scores = rrf_fuse([fts_list, embed_list])

        # "common" appears in both → highest score
        assert scores["common"] > scores["fts_only"]
        assert scores["common"] > scores["embed_only"]

    def test_three_retrievers(self) -> None:
        time_list = [RankedAnchor(neuron_id="n1", rank=1, retriever="time")]
        entity_list = [RankedAnchor(neuron_id="n1", rank=1, retriever="entity")]
        keyword_list = [RankedAnchor(neuron_id="n1", rank=1, retriever="keyword")]
        scores = rrf_fuse([time_list, entity_list, keyword_list])

        # n1 appears in all 3 → high score
        expected = 1.0 / 61 + 0.9 / 61 + 0.7 / 61
        assert abs(scores["n1"] - expected) < 1e-10

    def test_custom_weights(self) -> None:
        ranked = [RankedAnchor(neuron_id="a", rank=1, retriever="custom")]
        scores = rrf_fuse([ranked], retriever_weights={"custom": 2.0})
        expected = 2.0 / (DEFAULT_RRF_K + 1)
        assert abs(scores["a"] - expected) < 1e-10

    def test_unknown_retriever_defaults_to_1(self) -> None:
        ranked = [RankedAnchor(neuron_id="a", rank=1, retriever="unknown_source")]
        scores = rrf_fuse([ranked])
        expected = 1.0 / (DEFAULT_RRF_K + 1)
        assert abs(scores["a"] - expected) < 1e-10

    def test_rank_ordering_matters(self) -> None:
        """Lower rank number (better position) = higher RRF contribution."""
        ranked = [
            RankedAnchor(neuron_id="first", rank=1, retriever="entity"),
            RankedAnchor(neuron_id="tenth", rank=10, retriever="entity"),
        ]
        scores = rrf_fuse([ranked])
        assert scores["first"] > scores["tenth"]

    def test_disjoint_lists(self) -> None:
        list_a = [RankedAnchor(neuron_id="a1", rank=1, retriever="time")]
        list_b = [RankedAnchor(neuron_id="b1", rank=1, retriever="entity")]
        scores = rrf_fuse([list_a, list_b])
        assert "a1" in scores
        assert "b1" in scores

    def test_empty_inner_list(self) -> None:
        scores = rrf_fuse([[]])
        assert scores == {}


class TestRRFToActivationLevels:
    """Test normalization of RRF scores to activation levels."""

    def test_empty_input(self) -> None:
        assert rrf_to_activation_levels({}) == {}

    def test_single_neuron_gets_max(self) -> None:
        levels = rrf_to_activation_levels({"n1": 0.5})
        assert levels["n1"] == 1.0

    def test_two_neurons_spread(self) -> None:
        levels = rrf_to_activation_levels({"high": 1.0, "low": 0.0})
        assert levels["high"] == 1.0
        assert levels["low"] == 0.1  # default min_level

    def test_equal_scores_all_max(self) -> None:
        levels = rrf_to_activation_levels({"a": 0.5, "b": 0.5, "c": 0.5})
        assert all(v == 1.0 for v in levels.values())

    def test_custom_min_max(self) -> None:
        levels = rrf_to_activation_levels(
            {"high": 1.0, "low": 0.0},
            min_level=0.5,
            max_level=0.9,
        )
        assert levels["high"] == 0.9
        assert levels["low"] == 0.5

    def test_three_neurons_linear_interpolation(self) -> None:
        levels = rrf_to_activation_levels({"a": 0.0, "b": 0.5, "c": 1.0})
        assert levels["a"] == 0.1  # min
        assert abs(levels["b"] - 0.55) < 1e-10  # midpoint
        assert levels["c"] == 1.0  # max
