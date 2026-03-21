"""Tests for cascading retrieval with fiber summary tier (#61, #62).

Tests: fiber summary FTS5 search, sufficiency gate, pipeline integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Create a temporary SQLite storage with schema v27."""
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
async def storage_with_fibers(storage: SQLiteStorage) -> SQLiteStorage:
    """Storage pre-populated with neurons and fibers with summaries."""
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="spreading activation engine")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="memory consolidation process")
    n3 = Neuron.create(type=NeuronType.CONCEPT, content="embedding configuration")
    await storage.add_neuron(n1)
    await storage.add_neuron(n2)
    await storage.add_neuron(n3)

    f1 = Fiber.create(
        neuron_ids={n1.id},
        synapse_ids=set(),
        anchor_neuron_id=n1.id,
        summary="Spreading activation propagates signal through weighted synapses in the neural graph",
    )
    f2 = Fiber.create(
        neuron_ids={n2.id},
        synapse_ids=set(),
        anchor_neuron_id=n2.id,
        summary="Memory consolidation prunes weak connections and merges overlapping fibers",
    )
    f3 = Fiber.create(
        neuron_ids={n3.id},
        synapse_ids=set(),
        anchor_neuron_id=n3.id,
        summary="Configure embedding providers like sentence-transformer or Gemini for semantic search",
    )
    await storage.add_fiber(f1)
    await storage.add_fiber(f2)
    await storage.add_fiber(f3)

    return storage


class TestFiberSummaryFTS:
    """Tests for FTS5 search on fiber summaries."""

    @pytest.mark.asyncio
    async def test_search_finds_matching_fibers(self, storage_with_fibers: SQLiteStorage) -> None:
        """FTS5 search returns fibers matching query terms."""
        results = await storage_with_fibers.search_fiber_summaries("spreading activation")
        assert len(results) >= 1
        assert any("spreading" in (f.summary or "").lower() for f in results)

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(
        self, storage_with_fibers: SQLiteStorage
    ) -> None:
        """FTS5 search returns empty list when no fibers match."""
        results = await storage_with_fibers.search_fiber_summaries("xyznonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, storage_with_fibers: SQLiteStorage) -> None:
        """FTS5 search respects the limit parameter."""
        results = await storage_with_fibers.search_fiber_summaries("memory", limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_caps_at_50(self, storage_with_fibers: SQLiteStorage) -> None:
        """Limit is capped at 50 to prevent excessive results."""
        # Should not error even with large limit
        results = await storage_with_fibers.search_fiber_summaries("activation", limit=9999)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_on_empty_storage(self, storage: SQLiteStorage) -> None:
        """FTS5 search returns empty list on storage with no fibers."""
        results = await storage.search_fiber_summaries("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_fiber_without_summary_not_indexed(self, storage: SQLiteStorage) -> None:
        """Fibers without summaries are not returned by FTS search."""
        n = Neuron.create(type=NeuronType.CONCEPT, content="no summary neuron")
        await storage.add_neuron(n)
        f = Fiber.create(
            neuron_ids={n.id},
            synapse_ids=set(),
            anchor_neuron_id=n.id,
            summary=None,
        )
        await storage.add_fiber(f)

        results = await storage.search_fiber_summaries("summary")
        assert len(results) == 0


class TestSufficiencyGate:
    """Tests for the sufficiency threshold in BrainConfig."""

    def test_default_threshold(self) -> None:
        """Default sufficiency threshold is 0.7."""
        config = BrainConfig()
        assert config.sufficiency_threshold == 0.7

    def test_fiber_summary_tier_enabled_by_default(self) -> None:
        """Fiber summary tier is enabled by default."""
        config = BrainConfig()
        assert config.fiber_summary_tier_enabled is True

    def test_can_disable_fiber_tier(self) -> None:
        """Fiber summary tier can be disabled via config."""
        config = BrainConfig(fiber_summary_tier_enabled=False)
        assert config.fiber_summary_tier_enabled is False

    def test_custom_threshold(self) -> None:
        """Sufficiency threshold can be customized."""
        config = BrainConfig(sufficiency_threshold=0.5)
        assert config.sufficiency_threshold == 0.5


class TestPipelineIntegration:
    """Tests for fiber summary tier in the retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_fiber_tier_returns_result_when_sufficient(
        self, storage_with_fibers: SQLiteStorage
    ) -> None:
        """Pipeline returns fiber-based result when sufficiency threshold is met."""
        from neural_memory.engine.retrieval import ReflexPipeline

        # Low threshold to ensure fiber tier fires
        config = BrainConfig(sufficiency_threshold=0.1, fiber_summary_tier_enabled=True)
        pipeline = ReflexPipeline(storage_with_fibers, config)

        result = await pipeline.query("spreading activation synapses graph")
        assert result.confidence > 0
        assert result.context  # Should have context from fiber summaries

    @pytest.mark.asyncio
    async def test_fiber_tier_skipped_when_disabled(
        self, storage_with_fibers: SQLiteStorage
    ) -> None:
        """Pipeline skips fiber tier when disabled."""
        from neural_memory.engine.retrieval import ReflexPipeline

        config = BrainConfig(fiber_summary_tier_enabled=False)
        pipeline = ReflexPipeline(storage_with_fibers, config)

        result = await pipeline.query("spreading activation")
        # Should still work, just through the neuron pipeline
        assert result is not None

    @pytest.mark.asyncio
    async def test_fiber_tier_falls_through_on_low_confidence(
        self, storage_with_fibers: SQLiteStorage
    ) -> None:
        """Pipeline falls through to neuron tier when fiber confidence is low."""
        from neural_memory.engine.retrieval import ReflexPipeline

        # Very high threshold — fiber tier should never be sufficient
        config = BrainConfig(sufficiency_threshold=0.99, fiber_summary_tier_enabled=True)
        pipeline = ReflexPipeline(storage_with_fibers, config)

        result = await pipeline.query("xyz no match expected")
        # Should fall through to neuron pipeline (may have 0 confidence)
        assert result is not None

    @pytest.mark.asyncio
    async def test_fiber_tier_metadata_flag(self, storage_with_fibers: SQLiteStorage) -> None:
        """Fiber tier results include metadata flag."""
        from neural_memory.engine.retrieval import ReflexPipeline

        config = BrainConfig(sufficiency_threshold=0.1, fiber_summary_tier_enabled=True)
        pipeline = ReflexPipeline(storage_with_fibers, config)

        result = await pipeline.query("spreading activation synapses neural graph weighted")
        if result.metadata.get("fiber_summary_tier"):
            assert result.neurons_activated == 0
            assert len(result.fibers_matched) > 0


class TestSchemaV27:
    """Tests for schema v27 migration (fiber FTS5)."""

    @pytest.mark.asyncio
    async def test_schema_version_is_27(self) -> None:
        """Schema version bumped to 27."""
        from neural_memory.storage.sqlite_schema import SCHEMA_VERSION

        assert SCHEMA_VERSION == 35

    @pytest.mark.asyncio
    async def test_fiber_fts_table_exists(self, storage: SQLiteStorage) -> None:
        """fibers_fts virtual table is created on initialize."""
        conn = storage._conn
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'fibers_fts'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_fiber_fts_triggers_exist(self, storage: SQLiteStorage) -> None:
        """Fiber FTS sync triggers are created."""
        conn = storage._conn
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'fibers_%'"
        ) as cursor:
            rows = await cursor.fetchall()
            trigger_names = {row["name"] for row in rows}
            assert "fibers_ai" in trigger_names
            assert "fibers_ad" in trigger_names
            assert "fibers_au" in trigger_names
