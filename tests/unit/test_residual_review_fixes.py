"""Regression tests for Cognitive Efficiency residual review fixes."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.brain_mode import BrainModeConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.encoding_profiles import resolve_profile
from neural_memory.storage.factory import open_sqlite_storage
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.mark.asyncio
async def test_open_sqlite_storage_unified(tmp_path: Path) -> None:
    db = tmp_path / "u.db"
    store = await open_sqlite_storage(db, storage_adapter="unified", brain_id="b1")
    try:
        assert isinstance(store, SQLStorage)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_open_sqlite_storage_legacy(tmp_path: Path) -> None:
    db = tmp_path / "l.db"
    store = await open_sqlite_storage(db, storage_adapter="legacy", brain_id="b1")
    try:
        assert isinstance(store, SQLiteStorage)
    finally:
        await store.close()


def test_brain_mode_missing_adapter_legacy() -> None:
    cfg = BrainModeConfig.from_dict({"mode": "local"})
    assert cfg.storage_adapter == "legacy"


def test_resolve_profile_forces_cognitive_for_decision() -> None:
    res = resolve_profile(
        configured="lean",
        async_enrichment=True,
        memory_type="decision",
        priority=5,
    )
    assert res.profile.value == "cognitive"
    assert res.async_enrichment is False


@pytest.mark.asyncio
async def test_add_neuron_records_change_log(tmp_path: Path) -> None:
    store = SQLStorage(SQLiteDialect(str(tmp_path / "cl.db")))
    await store.initialize()
    brain = Brain.create(name="cl", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)

    from neural_memory.core.neuron import NeuronType

    n = Neuron.create(type=NeuronType.CONCEPT, content="dirty feed test")
    await store.add_neuron(n)
    changes = await store.get_changes_since(0, limit=10)
    assert any(c.entity_id == n.id and c.operation == "insert" for c in changes)
    await store.close()


@pytest.mark.asyncio
async def test_get_fibers_for_neurons(tmp_path: Path) -> None:
    from neural_memory.core.neuron import NeuronType

    store = SQLStorage(SQLiteDialect(str(tmp_path / "fib.db")))
    await store.initialize()
    brain = Brain.create(name="fib", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)

    n1 = Neuron.create(type=NeuronType.CONCEPT, content="anchor neuron for fiber lookup")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="member neuron for fiber lookup")
    await store.add_neuron(n1)
    await store.add_neuron(n2)
    fiber = Fiber.create(
        neuron_ids={n1.id, n2.id},
        synapse_ids=set(),
        anchor_neuron_id=n1.id,
        summary="fiber via junction",
    )
    await store.add_fiber(fiber)

    found = await store.get_fibers_for_neurons([n2.id])
    assert any(f.id == fiber.id for f in found)
    await store.close()


@pytest.mark.asyncio
async def test_lean_decision_forces_cognitive_profile(tmp_path: Path) -> None:
    store = SQLStorage(SQLiteDialect(str(tmp_path / "enc.db")))
    await store.initialize()
    brain = Brain.create(
        name="enc",
        config=BrainConfig(
            encoding_profile="lean",
            async_enrichment_enabled=True,
            embedding_enabled=False,
            fiber_summary_tier_enabled=False,
        ),
    )
    await store.save_brain(brain)
    store.set_brain(brain.id)
    encoder = MemoryEncoder(store, brain.config)
    result = await encoder.encode(
        "Chose unified adapter over legacy because outbox requires SQLStorage.",
        metadata={"type": "decision", "priority": 8},
    )
    assert result.encoding_profile == "cognitive"
    assert result.enrichment_status == "done"
    # Force-cognitive path must not leave pending outbox jobs for decision writes
    assert await store.count_enrichment_jobs() == 0
    await store.close()


@pytest.mark.asyncio
async def test_async_without_outbox_fails_closed(tmp_path: Path) -> None:
    """Legacy storage lacks outbox — async enrichment must raise, not silent skip."""
    store = SQLiteStorage(str(tmp_path / "legacy.db"))
    await store.initialize()
    brain = Brain.create(
        name="leg",
        config=BrainConfig(
            encoding_profile="lean",
            async_enrichment_enabled=True,
            embedding_enabled=False,
            fiber_summary_tier_enabled=False,
        ),
    )
    await store.save_brain(brain)
    store.set_brain(brain.id)
    encoder = MemoryEncoder(store, brain.config)
    with pytest.raises(RuntimeError, match="async_enrichment"):
        await encoder.encode("This write should fail closed without outbox APIs.")
    await store.close()
