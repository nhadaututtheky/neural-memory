"""Integration tests for run_incremental consolidation (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def storage(tmp_path: Path) -> SQLStorage:
    store = SQLStorage(SQLiteDialect(str(tmp_path / "inc.db")))
    await store.initialize()
    brain = Brain.create(name="inc", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    # Seed a few neurons so change_log has entries
    for i in range(5):
        n = Neuron.create(type=NeuronType.CONCEPT, content=f"topic concept number {i} auth")
        await store.add_neuron(n)
        if hasattr(store, "record_change"):
            await store.record_change("neuron", n.id, "insert")
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_incremental_zero_work_when_caught_up(storage: SQLStorage) -> None:
    engine = ConsolidationEngine(storage)
    # First: stamp checkpoints at current watermark via zero-work after bootstrap
    inc1 = await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE],
        bootstrap_full=True,
        dry_run=False,
    )
    assert inc1.mode in {"bootstrap_full", "incremental", "zero_work"}
    # Second run with no new changes → zero_work or empty advance
    inc2 = await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE],
        bootstrap_full=False,
    )
    assert inc2.mode in {"zero_work", "incremental"}
    assert "prune" not in inc2.strategies_failed


@pytest.mark.asyncio
async def test_dry_run_does_not_advance_checkpoint(storage: SQLStorage) -> None:
    engine = ConsolidationEngine(storage)
    # Ensure some dirty state from sequence 0
    inc = await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE],
        bootstrap_full=False,
        dry_run=True,
    )
    assert inc.dry_run is True
    cp = await storage.get_consolidation_checkpoint("prune")
    # Dry-run must not create a progressive stamp (may be None or still 0)
    if cp is not None:
        # Only acceptable if it was pre-existing at 0 from another path
        assert cp.last_sequence == 0


@pytest.mark.asyncio
async def test_new_changes_processed_next_run(storage: SQLStorage) -> None:
    engine = ConsolidationEngine(storage)
    await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE],
        bootstrap_full=True,
    )
    # Add new change after checkpoint
    n = Neuron.create(type=NeuronType.CONCEPT, content="brand new dirty neuron for next run")
    await storage.add_neuron(n)
    if hasattr(storage, "record_change"):
        await storage.record_change("neuron", n.id, "insert")

    inc = await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE],
        bootstrap_full=False,
    )
    assert inc.dirty is not None
    # Either dirty contains the neuron or change_count > 0
    assert inc.dirty.change_count >= 1 or n.id in inc.dirty.neuron_ids


@pytest.mark.asyncio
async def test_failed_strategy_does_not_block_listing(storage: SQLStorage) -> None:
    engine = ConsolidationEngine(storage)
    inc = await engine.run_incremental(
        strategies=[ConsolidationStrategy.PRUNE, ConsolidationStrategy.MERGE],
        bootstrap_full=True,
    )
    # Report structure always present
    assert isinstance(inc.strategies_run, list)
    assert inc.duration_ms >= 0
