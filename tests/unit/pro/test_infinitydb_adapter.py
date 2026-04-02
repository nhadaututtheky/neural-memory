"""Tests for InfinityDB Storage Adapter — Phase 1 integration smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.pro.storage_adapter import InfinityDBStorage


@pytest.fixture
async def storage(tmp_path: Path) -> InfinityDBStorage:
    """Create a fresh InfinityDBStorage instance for each test."""
    s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="test")
    await s.open()
    yield s
    await s.close()


class TestCompatMethods:
    """Test initialize() and list_brains() compatibility methods."""

    async def test_initialize_is_alias_for_open(self, tmp_path: Path) -> None:
        s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="init_test")
        await s.initialize()
        assert s._db is not None
        await s.close()

    async def test_initialize_idempotent(self, storage: InfinityDBStorage) -> None:
        db_before = storage._db
        await storage.initialize()
        assert storage._db is db_before

    async def test_list_brains_returns_current(self, storage: InfinityDBStorage) -> None:
        brains = await storage.list_brains()
        assert len(brains) == 1
        assert brains[0]["id"] == "test"
        assert brains[0]["name"] == "test"

    async def test_list_brains_before_open(self, tmp_path: Path) -> None:
        s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="pre_open")
        brains = await s.list_brains()
        assert brains[0]["id"] == "pre_open"


class TestBasicCRUD:
    """Test basic neuron/synapse/fiber CRUD via the adapter."""

    async def test_add_get_neuron(self, storage: InfinityDBStorage) -> None:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Hello InfinityDB")
        nid = await storage.add_neuron(neuron)
        assert nid

        got = await storage.get_neuron(nid)
        assert got is not None
        assert got.content == "Hello InfinityDB"
        assert got.type == NeuronType.CONCEPT

    async def test_find_neurons(self, storage: InfinityDBStorage) -> None:
        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Alpha fact"))
        await storage.add_neuron(Neuron.create(type=NeuronType.ENTITY, content="Beta entity"))

        facts = await storage.find_neurons(type=NeuronType.CONCEPT)
        assert len(facts) >= 1
        assert all(n.type == NeuronType.CONCEPT for n in facts)

    async def test_delete_neuron(self, storage: InfinityDBStorage) -> None:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="To be deleted")
        nid = await storage.add_neuron(neuron)
        assert await storage.delete_neuron(nid)
        assert await storage.get_neuron(nid) is None

    async def test_add_get_synapse(self, storage: InfinityDBStorage) -> None:
        n1 = await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Source"))
        n2 = await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Target"))

        synapse = Synapse.create(source_id=n1, target_id=n2, type=SynapseType.RELATED_TO)
        sid = await storage.add_synapse(synapse)
        assert sid

        synapses = await storage.get_synapses(source_id=n1)
        assert len(synapses) >= 1
        assert any(s.target_id == n2 for s in synapses)

    async def test_add_get_fiber(self, storage: InfinityDBStorage) -> None:
        n1 = await storage.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content="Fiber neuron")
        )
        fiber = Fiber(
            id="",
            summary="Test fiber",
            neuron_ids={n1},
            synapse_ids=set(),
            anchor_neuron_id=n1,
        )
        fid = await storage.add_fiber(fiber)
        assert fid

        got = await storage.get_fiber(fid)
        assert got is not None
        assert got.summary == "Test fiber"

    async def test_get_stats(self, storage: InfinityDBStorage) -> None:
        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Stats test"))
        stats = await storage.get_stats("test")
        assert stats["neuron_count"] >= 1

    async def test_set_brain(self, storage: InfinityDBStorage) -> None:
        storage.set_brain("other_brain")
        assert storage._current_brain_id == "other_brain"


class TestExportImport:
    """Test brain export/import roundtrip."""

    async def test_export_import_roundtrip(self, storage: InfinityDBStorage) -> None:
        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Export me"))
        snapshot = await storage.export_brain("test")
        assert len(snapshot.neurons) >= 1
        assert snapshot.brain_id == "test"
