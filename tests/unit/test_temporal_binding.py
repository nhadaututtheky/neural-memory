"""Tests for temporal binding — session-level auto-linking of close memories."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.pipeline import PipelineContext
from neural_memory.engine.temporal_binding import TemporalBindingStep


def _make_config(*, enabled: bool = True, window: float = 300.0) -> MagicMock:
    config = MagicMock()
    config.temporal_binding_enabled = enabled
    config.temporal_binding_window_seconds = window
    return config


def _make_ctx(*, ts: datetime | None = None) -> PipelineContext:
    ts = ts or datetime(2026, 3, 26, 12, 0, 0)
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="test anchor")
    ctx = PipelineContext(
        content="test content",
        timestamp=ts,
        metadata={},
        tags=set(),
        language="en",
    )
    ctx.anchor_neuron = anchor
    return ctx


def _make_fiber(
    *,
    anchor_id: str = "other-anchor",
    time_start: datetime | None = None,
) -> Fiber:
    return Fiber.create(
        neuron_ids={anchor_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        time_start=time_start or datetime(2026, 3, 26, 11, 58, 0),
        time_end=time_start or datetime(2026, 3, 26, 11, 58, 0),
    )


class TestTemporalBindingStep:
    """TemporalBindingStep creates CO_OCCURS synapses for close-in-time memories."""

    @pytest.mark.asyncio
    async def test_creates_synapse_within_window(self) -> None:
        """Two memories within 5 min → CO_OCCURS synapse created."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()

        nearby_fiber = _make_fiber(
            anchor_id="nearby-anchor",
            time_start=datetime(2026, 3, 26, 11, 58, 0),  # 2 min before
        )

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[nearby_fiber])
        storage.add_synapse = AsyncMock(return_value="syn-id")

        result = await step.execute(ctx, storage, config)

        storage.add_synapse.assert_called_once()
        synapse = storage.add_synapse.call_args[0][0]
        assert synapse.type == SynapseType.CO_OCCURS
        assert synapse.source_id == ctx.anchor_neuron.id
        assert synapse.target_id == "nearby-anchor"
        assert synapse.metadata["temporal_binding"] is True
        assert synapse.weight > 0.1  # proximity-weighted
        assert len(result.synapses_created) == 1

    @pytest.mark.asyncio
    async def test_skips_self(self) -> None:
        """Fiber matching own anchor is skipped."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()

        self_fiber = _make_fiber(anchor_id=ctx.anchor_neuron.id)

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[self_fiber])
        storage.add_synapse = AsyncMock()

        await step.execute(ctx, storage, config)
        storage.add_synapse.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_bindings_enforced(self) -> None:
        """At most 3 bindings created per encoding."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()

        fibers = [
            _make_fiber(
                anchor_id=f"anchor-{i}",
                time_start=datetime(2026, 3, 26, 11, 59, 0) - timedelta(seconds=i * 10),
            )
            for i in range(5)
        ]

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=fibers)
        storage.add_synapse = AsyncMock(return_value="syn-id")

        result = await step.execute(ctx, storage, config)
        assert storage.add_synapse.call_count == 3
        assert len(result.synapses_created) == 3

    @pytest.mark.asyncio
    async def test_weight_decreases_with_distance(self) -> None:
        """Closer memories get higher weight."""
        step = TemporalBindingStep()
        config = _make_config(window=300.0)
        ctx = _make_ctx(ts=datetime(2026, 3, 26, 12, 0, 0))

        close_fiber = _make_fiber(
            anchor_id="close",
            time_start=datetime(2026, 3, 26, 11, 59, 30),  # 30s ago
        )
        far_fiber = _make_fiber(
            anchor_id="far",
            time_start=datetime(2026, 3, 26, 11, 55, 0),  # 5 min ago
        )

        weights: list[float] = []

        async def capture_synapse(synapse: object) -> str:
            weights.append(synapse.weight)  # type: ignore[attr-defined]
            return "syn-id"

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[close_fiber, far_fiber])
        storage.add_synapse = AsyncMock(side_effect=capture_synapse)

        await step.execute(ctx, storage, config)
        assert len(weights) == 2
        assert weights[0] > weights[1]  # closer → higher weight

    @pytest.mark.asyncio
    async def test_disabled_via_config(self) -> None:
        """When disabled, no synapses created."""
        step = TemporalBindingStep()
        config = _make_config(enabled=False)
        ctx = _make_ctx()

        storage = AsyncMock()
        result = await step.execute(ctx, storage, config)
        storage.find_fibers.assert_not_called()
        assert result is ctx

    @pytest.mark.asyncio
    async def test_no_anchor_neuron(self) -> None:
        """No anchor → no crash, passthrough."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()
        ctx.anchor_neuron = None

        storage = AsyncMock()
        result = await step.execute(ctx, storage, config)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_no_nearby_fibers(self) -> None:
        """No fibers found → no crash, no synapses."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[])

        result = await step.execute(ctx, storage, config)
        assert len(result.synapses_created) == 0

    @pytest.mark.asyncio
    async def test_duplicate_synapse_handled(self) -> None:
        """ValueError from add_synapse (duplicate) is caught gracefully."""
        step = TemporalBindingStep()
        config = _make_config()
        ctx = _make_ctx()

        nearby_fiber = _make_fiber(anchor_id="dup-anchor")

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[nearby_fiber])
        storage.add_synapse = AsyncMock(side_effect=ValueError("duplicate"))

        result = await step.execute(ctx, storage, config)
        assert len(result.synapses_created) == 0  # not added due to dupe

    @pytest.mark.asyncio
    async def test_step_name(self) -> None:
        """Step name is 'temporal_binding'."""
        assert TemporalBindingStep().name == "temporal_binding"
