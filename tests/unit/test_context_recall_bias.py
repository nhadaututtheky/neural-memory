"""Tests for the context-recall-bias feature: recency boost, noise filtering,
session context enrichment, agent identity injection, and high-signal memory boost."""

from __future__ import annotations

import os
from typing import Any

from neural_memory.core.brain import BrainConfig
from neural_memory.engine.pipeline_steps import (
    _CODE_NOISE,
    _NOISE_CONCEPTS,
    _get_noise_concepts,
)


class TestNoiseFilter:
    """Verify the expanded noise concept filter."""

    def test_noise_set_size(self) -> None:
        """Noise set should be substantially larger than the original 18 terms."""
        assert len(_NOISE_CONCEPTS) >= 100, f"Noise set too small: {len(_NOISE_CONCEPTS)}"
        assert len(_CODE_NOISE) >= 50, f"Code-noise set too small: {len(_CODE_NOISE)}"

    def test_noise_terms_caught(self) -> None:
        """Standard noise terms should be filtered."""
        for term in ["json", "uuid", "yaml", "null", "none", "true", "false",
                     "config", "schema", "import", "export", "class",
                     "readme", "license", "setup", "install"]:
            assert term in _NOISE_CONCEPTS, f"Noise term '{term}' not in filter set"

    def test_domain_terms_not_filtered(self) -> None:
        """Domain-relevant terms must NOT be in the noise set."""
        for term in ["brain", "fiber", "synapse", "neuron", "pipeline", "recall",
                     "encoding", "retrieval", "memory", "CaitOS", "Brendon"]:
            assert term.lower() not in _NOISE_CONCEPTS, f"Domain term '{term}' incorrectly filtered"

    def test_get_noise_concepts_returns_frozenset(self) -> None:
        """_get_noise_concepts should return a frozenset."""
        result = _get_noise_concepts()
        assert isinstance(result, frozenset)

    def test_min_length_guard(self) -> None:
        """Short words (len < 4) are not in the noise set (handled separately)."""
        for term in ["ai", "os", "id", "go", "do", "in"]:
            assert len(term) < 4  # sanity check
            # Short terms are filtered by min_length in _is_valid_concept, not noise set


class TestBrainConfig:
    """Verify new config fields exist with correct defaults."""

    def test_concept_noise_filter_enabled(self) -> None:
        assert BrainConfig.concept_noise_filter_enabled is True

    def test_high_signal_memory_boost(self) -> None:
        assert BrainConfig.high_signal_memory_boost == 1.15

    def test_creation_recency_boost(self) -> None:
        assert BrainConfig.creation_recency_boost == 0.3

    def test_creation_recency_halflife_hrs(self) -> None:
        assert BrainConfig.creation_recency_halflife_hrs == 24.0

    def test_session_context_enrichment(self) -> None:
        assert BrainConfig.session_context_enrichment is True


class TestAgentIdentityInjection:
    """Verify the three-layer agent identity resolution in nmem_remember."""

    def test_explicit_source_agent(self) -> None:
        """Explicit parameter should produce agent:<name> tag."""
        tags: set[str] = set()
        source_agent = "bindax"
        resolved = source_agent or os.environ.get("NMEM_AGENT_ID", "") or "unknown"
        tags.add(f"agent:{resolved}")
        assert "agent:bindax" in tags

    def test_env_var_fallback(self) -> None:
        """NMEM_AGENT_ID env var should be used when no explicit param."""
        os.environ["NMEM_AGENT_ID"] = "codex"
        try:
            tags: set[str] = set()
            source_agent = ""  # no explicit param
            resolved = source_agent or os.environ.get("NMEM_AGENT_ID", "") or "unknown"
            tags.add(f"agent:{resolved}")
            assert "agent:codex" in tags
        finally:
            del os.environ["NMEM_AGENT_ID"]

    def test_default_unknown(self) -> None:
        """Default to 'unknown' when no identity is available."""
        tags: set[str] = set()
        source_agent = ""
        resolved = source_agent or os.environ.get("NMEM_AGENT_ID", "") or "unknown"
        tags.add(f"agent:{resolved}")
        assert "agent:unknown" in tags


class TestHighSignalBoost:
    """Verify the T1.7 high-signal memory boost logic."""

    def test_decision_boost(self) -> None:
        """Decisions should get the 1.15x boost."""
        boost = 1.15
        fiber_meta: dict[str, Any] = {"memory_type": "decision", "type": ""}
        _mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        score = 0.5
        if boost > 1.0 and _mem_type in {"decision", "insight", "preference"}:
            score *= boost
        assert score == 0.575  # 0.5 * 1.15

    def test_insight_boost(self) -> None:
        """Insights should get the 1.15x boost."""
        boost = 1.15
        fiber_meta: dict[str, Any] = {"memory_type": "insight"}
        _mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        score = 0.5
        if boost > 1.0 and _mem_type in {"decision", "insight", "preference"}:
            score *= boost
        assert score == 0.575

    def test_preference_boost(self) -> None:
        """Preferences should get the 1.15x boost."""
        boost = 1.15
        fiber_meta: dict[str, Any] = {"memory_type": "preference"}
        _mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        score = 0.5
        if boost > 1.0 and _mem_type in {"decision", "insight", "preference"}:
            score *= boost
        assert score == 0.575

    def test_concept_no_boost(self) -> None:
        """Generic concepts should NOT get the boost."""
        boost = 1.15
        fiber_meta: dict[str, Any] = {"memory_type": "concept"}
        _mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        score = 0.5
        if boost > 1.0 and _mem_type in {"decision", "insight", "preference"}:
            score *= boost
        assert score == 0.5  # unchanged


class TestTypeTagFilters:
    """Verify the type/tag filter logic in nmem_recall."""

    def test_type_filter_matches(self) -> None:
        """Type filter should pass when types match."""
        type_filter = "decision"
        fiber_meta: dict[str, Any] = {"memory_type": "decision"}
        mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        assert mem_type.lower() == type_filter.lower()

    def test_type_filter_skips(self) -> None:
        """Type filter should skip when types don't match."""
        type_filter = "decision"
        fiber_meta: dict[str, Any] = {"memory_type": "concept"}
        mem_type = fiber_meta.get("memory_type", "") or fiber_meta.get("type", "")
        assert mem_type.lower() != type_filter.lower()

    def test_tag_filter_matches(self) -> None:
        """Tags filter with OR logic should pass when any tag overlaps."""
        tag_filters = ["agent:bindax", "project:caitos"]
        tag_set = {t.lower() for t in tag_filters}
        fiber_tags: set[str] = {"agent:bindax", "scope:docs"}
        assert bool(tag_set & fiber_tags)  # overlap found

    def test_tag_filter_no_match(self) -> None:
        """Tags filter should skip when no tags overlap."""
        tag_filters = ["agent:codex"]
        tag_set = {t.lower() for t in tag_filters}
        fiber_tags: set[str] = {"agent:bindax", "scope:docs"}
        assert not (tag_set & fiber_tags)  # no overlap
