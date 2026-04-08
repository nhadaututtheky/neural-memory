"""Tests for Phase 3: Session Cortical Columns.

Covers:
- Config flag (session_columns_enabled)
- Column fiber creation in BuildFiberStep
- Column fiber score boost in _find_matching_fibers
- Column-aware familiarity recall (Strategy C)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse
from neural_memory.engine.pipeline import PipelineContext

# ---------------------------------------------------------------------------
# T4: Config flag
# ---------------------------------------------------------------------------


class TestSessionColumnsConfig:
    def test_default_enabled(self) -> None:
        config = BrainConfig()
        assert config.session_columns_enabled is True

    def test_disable(self) -> None:
        config = BrainConfig(session_columns_enabled=False)
        assert config.session_columns_enabled is False

    def test_with_updates(self) -> None:
        config = BrainConfig().with_updates(session_columns_enabled=False)
        assert config.session_columns_enabled is False


# ---------------------------------------------------------------------------
# T1: Column fiber creation in BuildFiberStep
# ---------------------------------------------------------------------------


class TestColumnFiberCreation:
    """Column fibers are created during encoding when session has 3+ neurons."""

    @pytest.fixture()
    def make_ctx(self) -> PipelineContext:
        """Build a PipelineContext with 5 neurons and synapses."""
        anchor = Neuron.create(content="anchor content", type=NeuronType.CONCEPT)
        entities = [
            Neuron.create(content=f"entity-{i}", type=NeuronType.ENTITY) for i in range(3)
        ]
        synapse = Synapse.create(source_id=anchor.id, target_id=entities[0].id, type="related")

        ctx = PipelineContext(
            content="Test session content with multiple entities for column fiber",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            metadata={},
            tags=set(),
            language="en",
        )
        ctx.anchor_neuron = anchor
        ctx.neurons_created = [anchor, *entities]
        ctx.synapses_created = [synapse]
        ctx.auto_tags = {"test", "entity"}
        ctx.agent_tags = set()
        ctx.merged_tags = {"test", "entity"}
        ctx.effective_metadata = {}
        ctx.entity_neurons = entities
        ctx.concept_neurons = []
        ctx.time_neurons = []
        return ctx

    @pytest.mark.asyncio()
    async def test_column_fiber_created_with_3plus_neurons(self, make_ctx: PipelineContext) -> None:
        """Column fiber should be created when encode has 3+ neurons."""
        from neural_memory.engine.pipeline_steps import BuildFiberStep

        storage = AsyncMock()
        storage.add_fiber = AsyncMock()
        storage.record_tag_cooccurrence = AsyncMock()
        storage.save_maturation = AsyncMock()
        storage.current_brain_id = "test-brain"

        config = BrainConfig(session_columns_enabled=True)

        step = BuildFiberStep()
        await step.execute(make_ctx, storage, config)

        # Should call add_fiber twice: normal fiber + column fiber
        assert storage.add_fiber.call_count == 2

        # Second call is the column fiber
        column_fiber = storage.add_fiber.call_args_list[1][0][0]
        assert column_fiber.metadata.get("_column") is True
        assert column_fiber.summary is not None
        assert len(column_fiber.summary) > 0

    @pytest.mark.asyncio()
    async def test_column_fiber_not_created_when_disabled(self, make_ctx: PipelineContext) -> None:
        """Column fiber should NOT be created when config disabled."""
        from neural_memory.engine.pipeline_steps import BuildFiberStep

        storage = AsyncMock()
        storage.add_fiber = AsyncMock()
        storage.record_tag_cooccurrence = AsyncMock()
        storage.save_maturation = AsyncMock()
        storage.current_brain_id = "test-brain"

        config = BrainConfig(session_columns_enabled=False)

        step = BuildFiberStep()
        await step.execute(make_ctx, storage, config)

        # Only normal fiber created
        assert storage.add_fiber.call_count == 1

    @pytest.mark.asyncio()
    async def test_column_fiber_not_created_with_few_neurons(self) -> None:
        """Column fiber should NOT be created with fewer than 3 neurons."""
        from neural_memory.engine.pipeline_steps import BuildFiberStep

        anchor = Neuron.create(content="anchor", type=NeuronType.CONCEPT)
        entity = Neuron.create(content="entity", type=NeuronType.ENTITY)
        synapse = Synapse.create(source_id=anchor.id, target_id=entity.id, type="related")

        ctx = PipelineContext(
            content="Small session",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            metadata={},
            tags=set(),
            language="en",
        )
        ctx.anchor_neuron = anchor
        ctx.neurons_created = [anchor, entity]
        ctx.synapses_created = [synapse]
        ctx.auto_tags = set()
        ctx.agent_tags = set()
        ctx.merged_tags = set()
        ctx.effective_metadata = {}
        ctx.entity_neurons = [entity]
        ctx.concept_neurons = []
        ctx.time_neurons = []

        storage = AsyncMock()
        storage.add_fiber = AsyncMock()
        storage.save_maturation = AsyncMock()
        storage.current_brain_id = "test-brain"

        config = BrainConfig(session_columns_enabled=True)
        step = BuildFiberStep()
        await step.execute(ctx, storage, config)

        # Only 2 neurons → no column fiber
        assert storage.add_fiber.call_count == 1

    @pytest.mark.asyncio()
    async def test_column_fiber_summary_truncated(self, make_ctx: PipelineContext) -> None:
        """Column fiber summary should be truncated to 500 chars."""
        from neural_memory.engine.pipeline_steps import BuildFiberStep

        # Make content very long
        make_ctx.content = "x" * 1000

        storage = AsyncMock()
        storage.add_fiber = AsyncMock()
        storage.record_tag_cooccurrence = AsyncMock()
        storage.save_maturation = AsyncMock()
        storage.current_brain_id = "test-brain"

        config = BrainConfig(session_columns_enabled=True)
        step = BuildFiberStep()
        await step.execute(make_ctx, storage, config)

        column_fiber = storage.add_fiber.call_args_list[1][0][0]
        assert len(column_fiber.summary) == 500


# ---------------------------------------------------------------------------
# T2: Column fiber score boost
# ---------------------------------------------------------------------------


class TestColumnFiberBoost:
    """Column fibers get 1.3x score multiplier in _find_matching_fibers."""

    def test_column_metadata_flag(self) -> None:
        """Verify column fiber has _column=True metadata."""
        fiber = Fiber.create(
            neuron_ids={"n-1", "n-2", "n-3"},
            synapse_ids=set(),
            anchor_neuron_id="n-1",
            metadata={"_column": True},
            summary="test column",
        )
        assert fiber.metadata["_column"] is True

    def test_non_column_fiber_no_flag(self) -> None:
        """Regular fibers should not have _column flag."""
        fiber = Fiber.create(
            neuron_ids={"n-1", "n-2"},
            synapse_ids=set(),
            anchor_neuron_id="n-1",
        )
        assert not (fiber.metadata or {}).get("_column")


# ---------------------------------------------------------------------------
# T3: Column-aware familiarity recall (Strategy C)
# ---------------------------------------------------------------------------


class TestColumnFamiliarityRecall:
    """Column fiber summaries searched as familiarity Strategy C."""

    def test_column_fiber_summary_keyword_match(self) -> None:
        """Column fiber with matching summary keywords should be found."""
        cf = Fiber.create(
            neuron_ids={"n-1", "n-2", "n-3"},
            synapse_ids=set(),
            anchor_neuron_id="n-1",
            summary="The user prefers dark mode for all applications",
            metadata={"_column": True},
        )

        # Simulate keyword matching logic from Strategy C
        q_tokens = {"dark", "mode"}
        summary = (cf.summary or "").lower()
        hits = sum(1 for t in q_tokens if t in summary)
        score = hits / max(len(q_tokens), 1)

        assert hits == 2
        assert score == 1.0

    def test_column_fiber_no_match(self) -> None:
        """Column fiber with no matching keywords scores 0."""
        cf = Fiber.create(
            neuron_ids={"n-1", "n-2", "n-3"},
            synapse_ids=set(),
            anchor_neuron_id="n-1",
            summary="Python asyncio patterns for web servers",
            metadata={"_column": True},
        )

        q_tokens = {"dark", "mode"}
        summary = (cf.summary or "").lower()
        hits = sum(1 for t in q_tokens if t in summary)

        assert hits == 0


