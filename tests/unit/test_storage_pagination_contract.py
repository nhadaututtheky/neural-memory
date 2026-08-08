"""Pagination and batch-read contracts for storage adapters (P2-T3)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.brain_mode import BrainModeConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.factory import create_storage
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


async def _seed_neurons(storage: NeuralStorage, count: int) -> list[Neuron]:
    neurons: list[Neuron] = []
    for i in range(count):
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content=f"pagination-neuron-{i:03d}",
        )
        await storage.add_neuron(neuron)
        neurons.append(neuron)
    return neurons


async def _seed_fibers(storage: NeuralStorage, neurons: list[Neuron]) -> list[Fiber]:
    fibers: list[Fiber] = []
    for i, neuron in enumerate(neurons):
        fiber = replace(
            Fiber.create(
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                summary=f"fiber-{i:03d}",
            ),
            salience=float(i),
        )
        await storage.add_fiber(fiber)
        fibers.append(fiber)
    return fibers


@pytest.fixture
async def unified(tmp_path: Path) -> NeuralStorage:
    storage = SQLStorage(SQLiteDialect(tmp_path / "unified.db"))
    await storage.initialize()
    brain = Brain.create(name="page")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    await storage.close()


@pytest.fixture
async def legacy(tmp_path: Path) -> NeuralStorage:
    storage = SQLiteStorage(tmp_path / "legacy.db")
    await storage.initialize()
    brain = Brain.create(name="page")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["unified", "legacy"])
async def test_get_fibers_offset_pagination(
    backend: str,
    unified: NeuralStorage,
    legacy: NeuralStorage,
) -> None:
    storage = unified if backend == "unified" else legacy
    neurons = await _seed_neurons(storage, 5)
    await _seed_fibers(storage, neurons)

    page1 = await storage.get_fibers(limit=2, order_by="salience", descending=False, offset=0)
    page2 = await storage.get_fibers(limit=2, order_by="salience", descending=False, offset=2)
    page3 = await storage.get_fibers(limit=2, order_by="salience", descending=False, offset=4)

    assert [f.summary for f in page1] == ["fiber-000", "fiber-001"]
    assert [f.summary for f in page2] == ["fiber-002", "fiber-003"]
    assert [f.summary for f in page3] == ["fiber-004"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["unified", "legacy"])
async def test_get_fibers_by_ids_batch(
    backend: str,
    unified: NeuralStorage,
    legacy: NeuralStorage,
) -> None:
    storage = unified if backend == "unified" else legacy
    neurons = await _seed_neurons(storage, 3)
    fibers = await _seed_fibers(storage, neurons)

    wanted = [fibers[0].id, fibers[2].id, "missing-fiber"]
    found = await storage.get_fibers_by_ids(wanted)

    assert set(found) == {fibers[0].id, fibers[2].id}
    assert found[fibers[0].id].summary == "fiber-000"
    assert found[fibers[2].id].summary == "fiber-002"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["unified", "legacy"])
async def test_find_neurons_created_before_is_inclusive(
    backend: str,
    unified: NeuralStorage,
    legacy: NeuralStorage,
) -> None:
    storage = unified if backend == "unified" else legacy
    now = utcnow()
    older = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="older-concept"),
        created_at=now - timedelta(hours=2),
    )
    boundary = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="boundary-concept"),
        created_at=now - timedelta(hours=1),
    )
    newer = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="newer-concept"),
        created_at=now,
    )
    await storage.add_neuron(older)
    await storage.add_neuron(boundary)
    await storage.add_neuron(newer)

    cutoff = boundary.created_at
    exact = await storage.find_neurons(content_exact="boundary-concept", created_before=cutoff)
    like = await storage.find_neurons(content_contains="concept", created_before=cutoff, limit=10)
    full = await storage.find_neurons(created_before=cutoff, limit=10)

    assert [n.content for n in exact] == ["boundary-concept"]
    assert {n.content for n in like} == {"older-concept", "boundary-concept"}
    assert {n.content for n in full} == {"older-concept", "boundary-concept"}


@pytest.mark.asyncio
async def test_unified_get_neuron_hashes_and_invalidation(unified: NeuralStorage) -> None:
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="hash-me",
        content_hash=0xABCDEF01,
    )
    await unified.add_neuron(neuron)

    first = await unified.get_neuron_hashes()
    assert (neuron.id, neuron.content_hash) in first

    # Cache hit path must still return same snapshot.
    second = await unified.get_neuron_hashes()
    assert second == first

    # Writes invalidate snapshot.
    other = Neuron.create(
        type=NeuronType.CONCEPT,
        content="hash-me-too",
        content_hash=0x12345678,
    )
    await unified.add_neuron(other)
    third = await unified.get_neuron_hashes()
    assert (other.id, other.content_hash) in third
    assert len(third) >= 2

    # Brain switch must not leak prior brain snapshot.
    other_brain = Brain.create(name="other")
    await unified.save_brain(other_brain)
    unified.set_brain(other_brain.id)
    empty = await unified.get_neuron_hashes()
    assert empty == []


@pytest.mark.asyncio
async def test_factory_unified_exposes_new_contracts(tmp_path: Path) -> None:
    storage = await create_storage(
        BrainModeConfig.local(storage_adapter="unified"),
        "pilot",
        local_path=str(tmp_path / "pilot.db"),
    )
    try:
        brain = Brain.create(name="pilot", brain_id="pilot")
        await storage.save_brain(brain)
        storage.set_brain("pilot")
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="factory-neuron",
            content_hash=0x55AA55AA,
        )
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="f",
        )
        await storage.add_fiber(fiber)

        hashes = await storage.get_neuron_hashes()
        by_ids = await storage.get_fibers_by_ids([fiber.id])
        page = await storage.get_fibers(limit=1, offset=0)

        assert (neuron.id, neuron.content_hash) in hashes
        assert fiber.id in by_ids
        assert len(page) == 1
    finally:
        await storage.close()
