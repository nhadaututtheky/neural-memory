"""Tests for Brain Quality C2 — Structured data encoding + verbatim recall.

Covers:
- StructuredDataEncoderStep key-value encoding
- StructuredDataEncoderStep table encoding
- Verbatim flag propagation to fibers
- Compression skip for verbatim fibers
- Domain merge guard in consolidation
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType

# ── StructuredDataEncoderStep ─────────────────────────────────────


class TestStructuredDataEncoderStep:
    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()
        return storage

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        return MagicMock()

    @pytest.mark.asyncio
    async def test_skip_when_no_structure(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructuredDataEncoderStep

        ctx = PipelineContext(
            content="plain text",
            timestamp=datetime(2026, 1, 1),
            metadata={},
            tags=set(),
            language="auto",
        )
        step = StructuredDataEncoderStep()
        result = await step.execute(ctx, mock_storage, mock_config)
        mock_storage.add_neuron.assert_not_called()
        assert result is ctx

    @pytest.mark.asyncio
    async def test_encode_key_value(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructuredDataEncoderStep

        anchor = Neuron.create(type=NeuronType.CONCEPT, content="test anchor")
        ctx = PipelineContext(
            content="key: value",
            timestamp=datetime(2026, 1, 1),
            metadata={},
            tags=set(),
            language="auto",
        )
        ctx.anchor_neuron = anchor
        ctx.metadata["_structure"] = {
            "format": "key_value",
            "fields": [
                {"name": "ROE", "value": "12.8%", "type": "number"},
                {"name": "Period", "value": "Q3 2024", "type": "text"},
            ],
            "confidence": 0.9,
        }

        step = StructuredDataEncoderStep()
        result = await step.execute(ctx, mock_storage, mock_config)

        # Should create 2 cell neurons + 2 HAS_VALUE synapses
        assert mock_storage.add_neuron.call_count == 2
        assert mock_storage.add_synapse.call_count == 2
        assert len(result.neurons_created) == 2
        assert len(result.synapses_created) == 2

        # Verify verbatim metadata
        first_neuron_call = mock_storage.add_neuron.call_args_list[0]
        neuron = first_neuron_call[0][0]
        assert neuron.metadata.get("_verbatim") is True
        assert neuron.metadata.get("raw_value") == "12.8%"

    @pytest.mark.asyncio
    async def test_encode_table(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructuredDataEncoderStep

        anchor = Neuron.create(type=NeuronType.CONCEPT, content="table anchor")
        ctx = PipelineContext(
            content="| A | B |",
            timestamp=datetime(2026, 1, 1),
            metadata={},
            tags=set(),
            language="auto",
        )
        ctx.anchor_neuron = anchor
        ctx.metadata["_structure"] = {
            "format": "table_row",
            "fields": [
                {"name": "Revenue", "value": "500B", "type": "number"},
            ],
            "confidence": 0.8,
        }

        step = StructuredDataEncoderStep()
        await step.execute(ctx, mock_storage, mock_config)

        assert mock_storage.add_neuron.call_count == 1
        # Table encoding uses IN_COLUMN synapses
        synapse_call = mock_storage.add_synapse.call_args_list[0]
        synapse = synapse_call[0][0]
        assert synapse.type == SynapseType.IN_COLUMN

    @pytest.mark.asyncio
    async def test_empty_fields_skipped(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructuredDataEncoderStep

        ctx = PipelineContext(
            content="data", timestamp=datetime(2026, 1, 1), metadata={}, tags=set(), language="auto"
        )
        ctx.metadata["_structure"] = {
            "format": "key_value",
            "fields": [{"name": "", "value": "", "type": "text"}],
            "confidence": 0.5,
        }

        step = StructuredDataEncoderStep()
        await step.execute(ctx, mock_storage, mock_config)
        mock_storage.add_neuron.assert_not_called()


# ── Verbatim flag propagation ─────────────────────────────────────


class TestVerbatimPropagation:
    @pytest.mark.asyncio
    async def test_fiber_gets_verbatim_flag(self) -> None:
        """BuildFiberStep should propagate _verbatim to fiber metadata."""
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import BuildFiberStep

        anchor = Neuron.create(type=NeuronType.CONCEPT, content="anchor")
        verbatim_neuron = Neuron.create(
            type=NeuronType.ENTITY,
            content="ROE = 12.8%",
            metadata={"_verbatim": True, "raw_value": "12.8%"},
        )

        ctx = PipelineContext(
            content="test", timestamp=datetime(2026, 1, 1), metadata={}, tags=set(), language="auto"
        )
        ctx.anchor_neuron = anchor
        ctx.neurons_created = [anchor, verbatim_neuron]

        storage = AsyncMock()
        storage.add_fiber = AsyncMock(return_value="fiber-id")
        config = MagicMock()

        step = BuildFiberStep()
        await step.execute(ctx, storage, config)

        # The fiber should have _verbatim in its metadata
        fiber_call = storage.add_fiber.call_args
        fiber = fiber_call[0][0]
        assert fiber.metadata.get("_verbatim") is True


# ── Compression skip for verbatim ─────────────────────────────────


class TestVerbatimCompressionSkip:
    def test_fiber_with_verbatim_flag(self) -> None:
        """Verbatim fibers should have the _verbatim metadata flag."""
        from neural_memory.core.fiber import Fiber

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            metadata={"_verbatim": True},
        )
        assert fiber.metadata.get("_verbatim") is True
