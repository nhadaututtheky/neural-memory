"""Wave 3 acceptance: batch decay writes + paged fiber scan truncation."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import NeuronState
from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from neural_memory.engine.lifecycle import DecayManager
from neural_memory.utils.timeutils import utcnow


@pytest.mark.asyncio
async def test_apply_decay_batches_writes_and_dry_run_is_zero_write() -> None:
    manager = DecayManager(decay_rate=0.5, prune_threshold=0.01)
    ref = utcnow()
    state = NeuronState(
        neuron_id="n1",
        activation_level=1.0,
        decay_rate=0.5,
        last_activated=ref - timedelta(days=30),
    )

    storage = AsyncMock()
    storage.get_pinned_neuron_ids = AsyncMock(return_value=set())
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_typed_memories = AsyncMock(return_value=[])
    storage.get_fibers_by_ids = AsyncMock(return_value={})
    storage.get_all_neuron_states = AsyncMock(return_value=[state])
    storage.get_all_synapses = AsyncMock(return_value=[])
    storage.update_neuron_states_batch = AsyncMock()
    storage.update_synapses_batch = AsyncMock()

    wet = await manager.apply_decay(storage, reference_time=ref, dry_run=False)
    assert wet.neurons_decayed == 1
    storage.update_neuron_states_batch.assert_awaited_once()
    storage.update_synapses_batch.assert_not_awaited()

    storage.update_neuron_states_batch.reset_mock()
    dry = await manager.apply_decay(storage, reference_time=ref, dry_run=True)
    assert dry.neurons_decayed == 1
    storage.update_neuron_states_batch.assert_not_awaited()
    storage.update_synapses_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_fibers_paged_sets_truncation_flag() -> None:
    engine = ConsolidationEngine(storage=AsyncMock())
    page = [
        Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            summary=f"f{i}",
        )
        for i in range(2)
    ]

    async def fake_get_fibers(*, limit: int = 10, offset: int = 0, **kwargs):
        # Always return a full page for the first max_pages, then extra row.
        if offset >= 4:  # after 2 pages of size 2
            return page[:1]  # more data remains
        return page

    engine._storage.get_fibers = AsyncMock(side_effect=fake_get_fibers)
    report = ConsolidationReport()
    fibers = await engine._load_fibers_paged(report, page_size=2, max_pages=2)

    assert len(fibers) == 4
    assert report.fiber_scan_truncated is True
    assert report.fiber_scan_warnings
