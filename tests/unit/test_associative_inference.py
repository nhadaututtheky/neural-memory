"""Tests for the associative inference engine — pure-logic module."""

from __future__ import annotations

import pytest

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.associative_inference import (
    InferenceCandidate,
    InferenceConfig,
    compute_inferred_weight,
    create_inferred_synapse,
    generate_associative_tags,
    identify_candidates,
)

# ── compute_inferred_weight ──────────────────────────────────────


class TestComputeInferredWeight:
    def test_at_threshold_returns_initial(self) -> None:
        config = InferenceConfig(co_activation_threshold=3, inferred_initial_weight=0.3)
        weight = compute_inferred_weight(count=3, avg_strength=0.8, config=config)
        assert weight == pytest.approx(0.3)

    def test_above_threshold_scales_linearly(self) -> None:
        config = InferenceConfig(
            co_activation_threshold=3,
            inferred_initial_weight=0.3,
            weight_scale_factor=0.1,
        )
        # excess = 5 - 3 = 2, weight = 0.3 + 0.1 * 2 * 0.8 = 0.46
        weight = compute_inferred_weight(count=5, avg_strength=0.8, config=config)
        assert weight == pytest.approx(0.46)

    def test_capped_at_max_weight(self) -> None:
        config = InferenceConfig(
            co_activation_threshold=3,
            inferred_initial_weight=0.3,
            inferred_max_weight=0.8,
            weight_scale_factor=0.1,
        )
        # excess = 100, would be 0.3 + 0.1 * 97 * 1.0 = 10.0 -> capped at 0.8
        weight = compute_inferred_weight(count=100, avg_strength=1.0, config=config)
        assert weight == pytest.approx(0.8)

    def test_below_threshold_returns_initial(self) -> None:
        config = InferenceConfig(co_activation_threshold=5, inferred_initial_weight=0.3)
        weight = compute_inferred_weight(count=2, avg_strength=0.9, config=config)
        assert weight == pytest.approx(0.3)

    def test_zero_avg_strength(self) -> None:
        config = InferenceConfig(co_activation_threshold=3, inferred_initial_weight=0.3)
        weight = compute_inferred_weight(count=10, avg_strength=0.0, config=config)
        assert weight == pytest.approx(0.3)


# ── identify_candidates ─────────────────────────────────────────


class TestIdentifyCandidates:
    def test_filters_below_threshold(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [("a", "b", 2, 0.8)]
        new, reinforce = identify_candidates(counts, set(), config)
        assert new == []
        assert reinforce == []

    def test_new_candidates_when_no_existing(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [("a", "b", 5, 0.7)]
        new, reinforce = identify_candidates(counts, set(), config)
        assert len(new) == 1
        assert new[0].neuron_a == "a"
        assert new[0].neuron_b == "b"
        assert reinforce == []

    def test_reinforce_when_pair_exists(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [("a", "b", 5, 0.7)]
        existing = {("a", "b")}
        new, reinforce = identify_candidates(counts, existing, config)
        assert new == []
        assert len(reinforce) == 1

    def test_reinforce_when_reverse_pair_exists(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [("a", "b", 5, 0.7)]
        existing = {("b", "a")}
        new, reinforce = identify_candidates(counts, existing, config)
        assert new == []
        assert len(reinforce) == 1

    def test_sorted_by_count_descending(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [
            ("a", "b", 3, 0.5),
            ("c", "d", 10, 0.9),
            ("e", "f", 5, 0.7),
        ]
        new, _ = identify_candidates(counts, set(), config)
        assert [c.co_activation_count for c in new] == [10, 5, 3]

    def test_capped_at_max_inferences(self) -> None:
        config = InferenceConfig(co_activation_threshold=1, max_inferences_per_run=2)
        counts = [
            ("a", "b", 5, 0.5),
            ("c", "d", 4, 0.5),
            ("e", "f", 3, 0.5),
        ]
        new, _ = identify_candidates(counts, set(), config)
        assert len(new) == 2

    def test_mixed_new_and_reinforce(self) -> None:
        config = InferenceConfig(co_activation_threshold=3)
        counts = [
            ("a", "b", 5, 0.7),
            ("c", "d", 4, 0.6),
        ]
        existing = {("a", "b")}
        new, reinforce = identify_candidates(counts, existing, config)
        assert len(new) == 1
        assert new[0].neuron_a == "c"
        assert len(reinforce) == 1
        assert reinforce[0].neuron_a == "a"


# ── create_inferred_synapse ──────────────────────────────────────


class TestCreateInferredSynapse:
    def test_creates_co_occurs_synapse(self) -> None:
        candidate = InferenceCandidate(
            neuron_a="n1",
            neuron_b="n2",
            co_activation_count=5,
            avg_binding_strength=0.8,
            inferred_weight=0.45,
        )
        synapse = create_inferred_synapse(candidate)
        assert synapse.source_id == "n1"
        assert synapse.target_id == "n2"
        assert synapse.type == SynapseType.CO_OCCURS
        assert synapse.weight == pytest.approx(0.45)
        assert synapse.is_bidirectional
        assert synapse.metadata["_inferred"] is True
        assert synapse.metadata["co_activation_count"] == 5

    def test_has_unique_id(self) -> None:
        candidate = InferenceCandidate("a", "b", 3, 0.5, 0.3)
        s1 = create_inferred_synapse(candidate)
        s2 = create_inferred_synapse(candidate)
        assert s1.id != s2.id


# ── generate_associative_tags ────────────────────────────────────


class TestGenerateAssociativeTags:
    def test_empty_candidates_returns_empty(self) -> None:
        tags = generate_associative_tags([], {}, set())
        assert tags == []

    def test_generates_tag_from_cluster(self) -> None:
        candidates = [
            InferenceCandidate("n1", "n2", 5, 0.8, 0.4),
            InferenceCandidate("n2", "n3", 4, 0.7, 0.35),
        ]
        content_map = {
            "n1": "python programming",
            "n2": "python testing",
            "n3": "python debugging",
        }
        tags = generate_associative_tags(candidates, content_map, set())
        assert len(tags) >= 1
        # "python" should be the most frequent word
        assert tags[0].tag == "python"
        assert len(tags[0].source_neuron_ids) >= 2

    def test_skips_existing_tags(self) -> None:
        candidates = [
            InferenceCandidate("n1", "n2", 5, 0.8, 0.4),
        ]
        content_map = {
            "n1": "python code",
            "n2": "python test",
        }
        tags = generate_associative_tags(candidates, content_map, {"python"})
        assert all(t.tag != "python" for t in tags)

    def test_single_pair_forms_cluster(self) -> None:
        candidates = [InferenceCandidate("n1", "n2", 3, 0.6, 0.3)]
        content_map = {"n1": "react components", "n2": "react hooks"}
        tags = generate_associative_tags(candidates, content_map, set())
        assert len(tags) == 1
        assert tags[0].tag == "react"

    def test_disconnected_pairs_form_separate_clusters(self) -> None:
        candidates = [
            InferenceCandidate("n1", "n2", 3, 0.6, 0.3),
            InferenceCandidate("n3", "n4", 3, 0.6, 0.3),
        ]
        content_map = {
            "n1": "react components",
            "n2": "react hooks",
            "n3": "django views",
            "n4": "django models",
        }
        tags = generate_associative_tags(candidates, content_map, set())
        assert len(tags) == 2
        tag_names = {t.tag for t in tags}
        assert "react" in tag_names
        assert "django" in tag_names

    def test_confidence_scales_with_cluster_size(self) -> None:
        candidates = [
            InferenceCandidate("n1", "n2", 3, 0.6, 0.3),
            InferenceCandidate("n2", "n3", 3, 0.6, 0.3),
            InferenceCandidate("n3", "n4", 3, 0.6, 0.3),
            InferenceCandidate("n4", "n5", 3, 0.6, 0.3),
        ]
        content_map = {f"n{i}": f"topic word{i}" for i in range(1, 6)}
        tags = generate_associative_tags(candidates, content_map, set())
        assert len(tags) >= 1
        # Cluster of 5 -> confidence = min(0.9, 0.3 + 0.1*5) = 0.8
        assert tags[0].confidence == pytest.approx(0.8)

    def test_origin_is_associative(self) -> None:
        candidates = [InferenceCandidate("n1", "n2", 3, 0.6, 0.3)]
        content_map = {"n1": "alpha", "n2": "alpha"}
        tags = generate_associative_tags(candidates, content_map, set())
        if tags:
            assert tags[0].origin == "associative"

    def test_ignores_short_words(self) -> None:
        candidates = [InferenceCandidate("n1", "n2", 3, 0.6, 0.3)]
        content_map = {"n1": "a b c", "n2": "a b c"}
        tags = generate_associative_tags(candidates, content_map, set())
        # All words are < 3 chars, no tags generated
        assert tags == []
