"""Unit tests for Storage Evolution Phase 3.

Covers:
- T1: Causal synapse prune guard (CAUSED_BY/LEADS_TO/ENABLES/PREVENTS never pruned
  unless _inferred=True).
- T2: induce_abstraction() template and metadata.
- T3: Merge strategy persists abstract neuron + IS_A links for large clusters.
"""

from __future__ import annotations

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.abstraction import (
    _extract_top_terms,
    induce_abstraction,
)
from neural_memory.engine.consolidation import (
    _CAUSAL_SYNAPSE_TYPES,
    ConsolidationConfig,
    ConsolidationReport,
)

# ---------------------------------------------------------------------------
# T1 — causal prune guard


class TestCausalPruneGuard:
    """Causal synapses must never be pruned unless explicitly inferred."""

    def test_causal_types_set_contents(self) -> None:
        assert SynapseType.CAUSED_BY in _CAUSAL_SYNAPSE_TYPES
        assert SynapseType.LEADS_TO in _CAUSAL_SYNAPSE_TYPES
        assert SynapseType.ENABLES in _CAUSAL_SYNAPSE_TYPES
        assert SynapseType.PREVENTS in _CAUSAL_SYNAPSE_TYPES
        # Associative types are NOT protected
        assert SynapseType.CO_OCCURS not in _CAUSAL_SYNAPSE_TYPES
        assert SynapseType.RELATED_TO not in _CAUSAL_SYNAPSE_TYPES

    def test_inferred_causal_not_in_guard(self) -> None:
        """Inferred causal synapses should still be prunable (noise filter)."""
        inferred = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.CAUSED_BY,
            metadata={"_inferred": True},
        )
        # The guard predicate mirrors the prune-loop logic.
        is_protected = inferred.type in _CAUSAL_SYNAPSE_TYPES and not inferred.metadata.get(
            "_inferred", False
        )
        assert is_protected is False

    def test_manual_causal_protected(self) -> None:
        manual = Synapse.create(source_id="a", target_id="b", type=SynapseType.CAUSED_BY)
        is_protected = manual.type in _CAUSAL_SYNAPSE_TYPES and not manual.metadata.get(
            "_inferred", False
        )
        assert is_protected is True


# ---------------------------------------------------------------------------
# T2 — induce_abstraction


def _make_cluster() -> list[Neuron]:
    return [
        Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis connection pool timeout set to 30 seconds",
        ),
        Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis retry policy uses exponential backoff",
        ),
        Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis memory limit of 512MB reached during peak",
        ),
        Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis cluster rebalance took 5 minutes",
        ),
        Neuron.create(
            type=NeuronType.CONCEPT,
            content="Redis sentinel provides high availability",
        ),
    ]


class TestInduceAbstraction:
    def test_empty_cluster_raises(self) -> None:
        with pytest.raises(ValueError):
            induce_abstraction([])

    def test_produces_concept_neuron(self) -> None:
        cluster = _make_cluster()
        abstract = induce_abstraction(cluster)
        assert abstract.type is NeuronType.CONCEPT
        assert abstract.abstraction_level == 2

    def test_content_follows_template(self) -> None:
        cluster = _make_cluster()
        abstract = induce_abstraction(cluster)
        # "[N] memories about [TOPIC]: [TERMS]. Key: [content]"
        assert abstract.content.startswith("5 memories about entity:")
        assert "redis" in abstract.content.lower()
        assert "Key:" in abstract.content

    def test_metadata_traces_source(self) -> None:
        cluster = _make_cluster()
        abstract = induce_abstraction(cluster)
        meta = abstract.metadata
        assert meta["_abstraction_induced"] is True
        assert len(meta["_abstraction_source_ids"]) == 5
        assert meta["_abstraction_exemplar_id"] in [n.id for n in cluster]
        assert "redis" in meta["_abstraction_terms"]

    def test_stopwords_filtered(self) -> None:
        contents = [
            "the quick brown fox",
            "the the the the",
            "a an and or but",
        ]
        terms = _extract_top_terms(contents, top_n=5)
        assert "the" not in terms
        assert "and" not in terms
        assert "quick" in terms or "brown" in terms or "fox" in terms

    def test_short_tokens_filtered(self) -> None:
        terms = _extract_top_terms(["ok ok a b c quick"], top_n=5)
        assert "a" not in terms
        assert "b" not in terms
        assert "ok" not in terms  # len < 3

    def test_exemplar_is_highest_priority(self) -> None:
        low = Neuron.create(
            type=NeuronType.ENTITY,
            content="low priority memory",
        ).with_metadata(_goal_priority=2)
        high = Neuron.create(
            type=NeuronType.ENTITY,
            content="important strategic decision",
        ).with_metadata(_goal_priority=9)
        abstract = induce_abstraction([low, high])
        assert abstract.metadata["_abstraction_exemplar_id"] == high.id
        assert "important strategic decision" in abstract.content

    def test_long_content_truncated(self) -> None:
        long_content = "Redis " * 100
        n = Neuron.create(type=NeuronType.ENTITY, content=long_content)
        abstract = induce_abstraction([n, n])
        assert len(abstract.content) < 400  # truncated


# ---------------------------------------------------------------------------
# T3 — merge wiring config


class TestMergeConfig:
    def test_dynamic_abstraction_enabled_by_default(self) -> None:
        cfg = ConsolidationConfig()
        assert cfg.enable_dynamic_abstraction is True
        assert cfg.abstraction_cluster_min_size == 5

    def test_can_disable_dynamic_abstraction(self) -> None:
        cfg = ConsolidationConfig(enable_dynamic_abstraction=False)
        assert cfg.enable_dynamic_abstraction is False

    def test_concepts_created_field_exists(self) -> None:
        report = ConsolidationReport()
        assert report.concepts_created == 0
        report.concepts_created = 3
        assert "Concepts created: 3" in report.summary()
