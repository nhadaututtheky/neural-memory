"""Tests for hippocampal replay consolidation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.hippocampal_replay import ReplayResult, hippocampal_replay


class TestReplayResult:
    def test_defaults(self) -> None:
        r = ReplayResult(episodes_replayed=0, synapses_strengthened=0, synapses_weakened=0)
        assert r.episodes_replayed == 0

    def test_frozen(self) -> None:
        r = ReplayResult(episodes_replayed=1, synapses_strengthened=2, synapses_weakened=3)
        with pytest.raises(AttributeError):
            r.episodes_replayed = 5  # type: ignore[misc]


class TestHippocampalReplay:
    @pytest.mark.asyncio
    async def test_disabled_returns_zero(self) -> None:
        """When replay_enabled=False, hippocampal_replay returns immediately."""
        config = MagicMock()
        config.replay_enabled = False
        storage = AsyncMock()

        result = await hippocampal_replay(storage, config)
        assert result.episodes_replayed == 0
        storage.find_fibers.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_fibers_returns_zero(self) -> None:
        config = MagicMock()
        config.replay_enabled = True
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98
        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[])

        result = await hippocampal_replay(storage, config)
        assert result.episodes_replayed == 0

    @pytest.mark.asyncio
    async def test_no_recent_fibers(self) -> None:
        config = MagicMock()
        config.replay_enabled = True
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[])

        result = await hippocampal_replay(storage, config)
        assert result.episodes_replayed == 0

    @pytest.mark.asyncio
    async def test_replays_fibers(self) -> None:
        config = MagicMock()
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98

        fiber = MagicMock()
        fiber.id = "fiber-1"
        fiber.anchor_neuron_id = "neuron-1"
        fiber.neuron_ids = ["neuron-1", "neuron-2"]
        fiber.synapse_ids = ["syn-1"]
        fiber.salience = 0.8
        fiber.metadata = {}

        syn = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-2",
            type=SynapseType.CO_OCCURS,
            weight=0.5,
            synapse_id="syn-1",
        )

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[fiber])
        storage.get_synapses = AsyncMock(return_value=[syn])
        storage.update_synapse = AsyncMock()

        result = await hippocampal_replay(storage, config)
        assert result.episodes_replayed >= 1

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self) -> None:
        config = MagicMock()
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98

        fiber = MagicMock()
        fiber.id = "fiber-1"
        fiber.anchor_neuron_id = "neuron-1"
        fiber.neuron_ids = ["neuron-1", "neuron-2"]
        fiber.synapse_ids = ["syn-1"]
        fiber.salience = 0.8
        fiber.metadata = {}

        syn = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-2",
            type=SynapseType.CO_OCCURS,
            weight=0.5,
            synapse_id="syn-1",
        )

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[fiber])
        storage.get_synapses = AsyncMock(return_value=[syn])

        result = await hippocampal_replay(storage, config, dry_run=True)
        storage.update_synapse.assert_not_called()
        assert result.episodes_replayed >= 1
