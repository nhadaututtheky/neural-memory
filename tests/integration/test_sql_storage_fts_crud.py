"""P3-T3: transactional SQLite + FTS coherence for neurons and fibers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def store(tmp_path: Path):
    s = SQLStorage(SQLiteDialect(tmp_path / "fts.db"))
    await s.initialize()
    brain = Brain.create(name="fts")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_neuron_insert_searchable_in_fts(store: SQLStorage) -> None:
    await store.add_neuron(
        Neuron.create(type=NeuronType.CONCEPT, content="quantum entanglement protocol")
    )
    hits = await store.find_neurons(content_contains="entanglement", limit=10)
    assert any("entanglement" in n.content for n in hits)


@pytest.mark.asyncio
async def test_neuron_update_replaces_fts_terms(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="alpha term one")
    await store.add_neuron(n)
    await store.update_neuron(replace(n, content="beta term two"))
    old = await store.find_neurons(content_contains="alpha", limit=10)
    new = await store.find_neurons(content_contains="beta", limit=10)
    assert all(h.id != n.id for h in old)
    assert any(h.id == n.id for h in new)


@pytest.mark.asyncio
async def test_neuron_delete_removes_fts(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="delete me from fts")
    await store.add_neuron(n)
    assert await store.delete_neuron(n.id)
    hits = await store.find_neurons(content_contains="delete", limit=10)
    assert all(h.id != n.id for h in hits)


@pytest.mark.asyncio
async def test_fiber_summary_null_to_text_indexes_fts(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="anchor fiber fts")
    await store.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={n.id},
        synapse_ids=set(),
        anchor_neuron_id=n.id,
        summary=None,
    )
    await store.add_fiber(fiber)
    # NULL summary should not match
    empty = await store.search_fiber_summaries("uniquephrase", limit=10)
    assert all(f.id != fiber.id for f in empty)

    await store.update_fiber(replace(fiber, summary="uniquephrase fiber summary"))
    hits = await store.search_fiber_summaries("uniquephrase", limit=10)
    assert any(f.id == fiber.id for f in hits)


@pytest.mark.asyncio
async def test_fiber_summary_text_to_null_removes_fts(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="anchor drop fts")
    await store.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={n.id},
        synapse_ids=set(),
        anchor_neuron_id=n.id,
        summary="dropphrase fiber summary",
    )
    await store.add_fiber(fiber)
    assert any(f.id == fiber.id for f in await store.search_fiber_summaries("dropphrase", limit=10))

    await store.update_fiber(replace(fiber, summary=None))
    hits = await store.search_fiber_summaries("dropphrase", limit=10)
    assert all(f.id != fiber.id for f in hits)


@pytest.mark.asyncio
async def test_neuron_add_rollback_leaves_no_row(store: SQLStorage) -> None:
    """If state insert fails inside the transaction, neuron row must not commit."""
    n = Neuron.create(type=NeuronType.CONCEPT, content="rollback neuron")
    # Force second statement failure by poisoning after first check: monkeypatch execute
    calls = {"n": 0}
    orig = store._dialect.execute

    async def flaky(sql: str, params=()):
        calls["n"] += 1
        if calls["n"] == 2 and "neuron_states" in sql:
            raise RuntimeError("forced state insert failure")
        return await orig(sql, params)

    store._dialect.execute = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="forced state"):
        await store.add_neuron(n)
    store._dialect.execute = orig  # type: ignore[method-assign]

    found = await store.find_neurons(content_exact="rollback neuron")
    assert found == []
