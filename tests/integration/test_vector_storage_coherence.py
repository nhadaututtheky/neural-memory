"""P3-T4: vector index add/remove/knn stale-safe coherence (SQLite sidecar)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

hnswlib = pytest.importorskip("hnswlib")

# Default SQLiteVectorIndex dimension for the project.
_DIM = 384


@pytest.fixture
async def store(tmp_path: Path):
    s = SQLStorage(SQLiteDialect(tmp_path / "vec.db"))
    await s.initialize()
    brain = Brain.create(name="vec", brain_id="vec")
    await s.save_brain(brain)
    s.set_brain("vec")
    yield s
    await s.close()


def _unit(hot: int = 0) -> list[float]:
    v = [0.0] * _DIM
    v[hot % _DIM] = 1.0
    return v


@pytest.mark.asyncio
async def test_add_then_knn_hits(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="vector target")
    await store.add_neuron(n)
    vec = _unit(hot=0)
    await store.vector_index_add(n.id, vec)
    hits = await store.knn_search(vec, k=5)
    assert any(nid == n.id for nid, _ in hits)


@pytest.mark.asyncio
async def test_delete_neuron_removes_from_knn(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="vector delete me")
    await store.add_neuron(n)
    vec = _unit(hot=1)
    await store.vector_index_add(n.id, vec)
    assert await store.delete_neuron(n.id)
    hits = await store.knn_search(vec, k=5)
    assert all(nid != n.id for nid, _ in hits)


@pytest.mark.asyncio
async def test_knn_filters_stale_sidecar_ids(store: SQLStorage) -> None:
    """Live SQL validation drops IDs that only exist in the HNSW map."""
    n = Neuron.create(type=NeuronType.CONCEPT, content="live vector")
    await store.add_neuron(n)
    vec = _unit(hot=2)
    await store.vector_index_add(n.id, vec)

    index = store._ensure_vector_index()
    assert index is not None
    # Inject a ghost id into the sidecar without a neurons row
    index.add("ghost-missing-id", _unit(hot=2))

    hits = await store.knn_search(vec, k=10)
    assert all(nid != "ghost-missing-id" for nid, _ in hits)
    assert any(nid == n.id for nid, _ in hits)


@pytest.mark.asyncio
async def test_brain_switch_closes_sidecar(store: SQLStorage) -> None:
    other = Brain.create(name="other", brain_id="other")
    await store.save_brain(other)
    n = Neuron.create(type=NeuronType.CONCEPT, content="brain a only")
    await store.add_neuron(n)
    await store.vector_index_add(n.id, _unit(hot=3))
    assert store._vector_index is not None
    store.set_brain("other")
    assert store._vector_index is None
