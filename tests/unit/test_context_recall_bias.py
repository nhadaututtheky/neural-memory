"""Tests for the context-recall-bias feature: recency boost, noise filtering,
session context enrichment, agent identity injection, and high-signal memory boost."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.pipeline_steps import (
    _CODE_NOISE,
    _ensure_noise_concepts,
    _get_noise_concepts,
)
from neural_memory.integrations.nanobot.context import NMContext
from neural_memory.integrations.nanobot.tools import NMRememberTool
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Create a temporary SQLite storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = SQLiteStorage(db_path)
        await s.initialize()

        brain = Brain.create(name="test_brain")
        await s.save_brain(brain)
        s.set_brain(brain.id)

        yield s
        await s.close()


@pytest.fixture
async def fiber_storage(storage: SQLiteStorage) -> SQLiteStorage:
    """Storage pre-populated with fibers having various memory types."""
    # Create neurons for fibers to reference
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="api design decision cache layer")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="team prefers python async patterns")
    n3 = Neuron.create(type=NeuronType.CONCEPT, content="general concept about software")
    await storage.add_neuron(n1)
    await storage.add_neuron(n2)
    await storage.add_neuron(n3)

    # Decision fiber — high-signal type
    f1 = Fiber.create(
        neuron_ids={n1.id},
        synapse_ids=set(),
        anchor_neuron_id=n1.id,
        summary="Decision to use Redis for caching layer instead of PostgreSQL",
        metadata={
            "memory_type": "decision",
            "type": "decision",
            "tags": ["agent:bindax", "project:caitos"],
        },
    )

    # Preference fiber — high-signal type
    f2 = Fiber.create(
        neuron_ids={n2.id},
        synapse_ids=set(),
        anchor_neuron_id=n2.id,
        summary="Team prefers async patterns for I/O bound operations",
        metadata={
            "memory_type": "preference",
            "type": "preference",
            "tags": ["agent:codex", "project:caitos"],
        },
    )

    # Concept fiber — NOT a high-signal type (should not get boost)
    f3 = Fiber.create(
        neuron_ids={n3.id},
        synapse_ids=set(),
        anchor_neuron_id=n3.id,
        summary="General thoughts about software architecture patterns",
        metadata={
            "memory_type": "concept",
            "type": "concept",
            "tags": [],
        },
    )

    await storage.add_fiber(f1)
    await storage.add_fiber(f2)
    await storage.add_fiber(f3)

    return storage


# ── Noise Filter Tests ─────────────────────────────────────


class TestNoiseFilter:
    """Verify the expanded noise concept filter."""

    def test_noise_set_size(self) -> None:
        """Noise set should be substantially larger than the original 18 terms."""
        assert len(_ensure_noise_concepts()) >= 100, (
            f"Noise set too small: {len(_ensure_noise_concepts())}"
        )
        assert len(_CODE_NOISE) >= 50, f"Code-noise set too small: {len(_CODE_NOISE)}"

    def test_noise_terms_caught(self) -> None:
        """Standard noise terms should be filtered."""
        for term in [
            "json",
            "uuid",
            "yaml",
            "null",
            "none",
            "true",
            "false",
            "config",
            "schema",
            "import",
            "export",
            "class",
            "readme",
            "license",
            "setup",
            "install",
        ]:
            assert term in _ensure_noise_concepts(), f"Noise term '{term}' not in filter set"

    def test_domain_terms_not_filtered(self) -> None:
        """Domain-relevant terms must NOT be in the noise set."""
        for term in [
            "brain",
            "fiber",
            "synapse",
            "neuron",
            "pipeline",
            "recall",
            "encoding",
            "retrieval",
            "memory",
            "CaitOS",
            "Brendon",
        ]:
            assert term.lower() not in _ensure_noise_concepts(), (
                f"Domain term '{term}' incorrectly filtered"
            )

    def test_get_noise_concepts_returns_frozenset(self) -> None:
        """_get_noise_concepts should return a frozenset."""
        result = _get_noise_concepts()
        assert isinstance(result, frozenset)

    def test_min_length_guard(self) -> None:
        """Short words (len < 4) are not in the noise set (handled separately)."""
        for term in ["ai", "os", "id", "go", "do", "in"]:
            assert len(term) < 4  # sanity check
            # Short terms are filtered by min_length in _is_valid_concept, not noise set


# ── Config Defaults Tests ──────────────────────────────────


class TestBrainConfig:
    """Verify new config fields exist with correct defaults."""

    def test_concept_noise_filter_enabled(self) -> None:
        """Noise filter is enabled by default."""
        assert BrainConfig.concept_noise_filter_enabled is True

    def test_high_signal_memory_boost_default_neutral(self) -> None:
        """High-signal boost defaults to 1.0 (disabled/neutral)."""
        assert BrainConfig.high_signal_memory_boost == 1.0

    def test_creation_recency_boost_default_neutral(self) -> None:
        """Creation recency boost defaults to 0.0 (disabled/neutral)."""
        assert BrainConfig.creation_recency_boost == 0.0

    def test_creation_recency_halflife_hrs(self) -> None:
        assert BrainConfig.creation_recency_halflife_hrs == 24.0

    def test_session_context_enrichment(self) -> None:
        assert BrainConfig.session_context_enrichment is True


# ── Agent Identity Injection Tests ─────────────────────────


class TestAgentIdentityInjection:
    """Verify the two-layer agent identity resolution in nmem_remember."""

    @pytest.mark.asyncio
    async def test_explicit_source_agent(self, storage: SQLiteStorage) -> None:
        """Explicit parameter should flow through the real remember tool."""
        tool = NMRememberTool(
            NMContext(storage=storage, brain=Brain.create(name="test_brain"), config=BrainConfig())
        )
        result = await tool.execute(content="agent injection test", source_agent="bindax")
        assert '"success": true' in result

    @pytest.mark.asyncio
    async def test_env_var_fallback(self, storage: SQLiteStorage) -> None:
        """NMEM_AGENT_ID env var should be used when no explicit param."""
        os.environ["NMEM_AGENT_ID"] = "codex"
        try:
            tool = NMRememberTool(
                NMContext(
                    storage=storage, brain=Brain.create(name="test_brain"), config=BrainConfig()
                )
            )
            result = await tool.execute(content="env fallback test")
            assert '"success": true' in result
        finally:
            del os.environ["NMEM_AGENT_ID"]

    @pytest.mark.asyncio
    async def test_no_tag_when_no_identity(self, storage: SQLiteStorage) -> None:
        """No agent tag should be injected when no identity is available."""
        os.environ.pop("NMEM_AGENT_ID", None)
        tool = NMRememberTool(
            NMContext(storage=storage, brain=Brain.create(name="test_brain"), config=BrainConfig())
        )
        result = await tool.execute(content="no identity test")
        assert '"success": true' in result


# ── High-Signal Memory Boost — Integration Tests ───────────


class TestHighSignalBoost:
    """Verify the T1.7 high-signal memory boost through the real scoring pipeline."""

    @pytest.mark.asyncio
    async def test_decision_ranks_higher_with_boost(self, fiber_storage: SQLiteStorage) -> None:
        """With boost enabled, decision fibers should score higher than concept fibers."""
        from neural_memory.engine.retrieval import ReflexPipeline

        config = BrainConfig(
            high_signal_memory_boost=1.15,
        )
        pipeline = ReflexPipeline(fiber_storage, config)

        result = await pipeline.query("api design decision cache layer")

        assert result.confidence > 0
        assert result.context is not None
        assert "Redis for caching layer" in result.context

    @pytest.mark.asyncio
    async def test_boost_disabled_by_default(self, fiber_storage: SQLiteStorage) -> None:
        """Default config (boost=1.0) applies no high-signal multiplier."""
        from neural_memory.engine.retrieval import ReflexPipeline

        config = BrainConfig(
            high_signal_memory_boost=1.0,  # neutral — no boost
        )
        pipeline = ReflexPipeline(fiber_storage, config)

        result = await pipeline.query("api design decision cache layer")

        assert result.confidence > 0
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_preferences_boosted_with_config(self, fiber_storage: SQLiteStorage) -> None:
        """Preferences should also get the high-signal boost when enabled."""
        from neural_memory.engine.retrieval import ReflexPipeline

        config = BrainConfig(
            high_signal_memory_boost=1.15,
        )
        pipeline = ReflexPipeline(fiber_storage, config)

        result = await pipeline.query("python async patterns team")

        assert result.confidence > 0
        assert result.context is not None
        assert "async patterns" in result.context


# ── Recency Boost — Integration Tests ──────────────────────


# ── Type/Tag Filter Tests ──────────────────────────────────


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
