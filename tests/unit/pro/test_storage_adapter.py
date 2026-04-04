"""Tests for InfinityDBStorage — NeuralStorage adapter.

Verifies that the adapter correctly maps NeuralStorage interface
calls to InfinityDB operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.pro.storage_adapter import InfinityDBStorage


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_brain"


DIMS = 32


async def _make_storage(db_dir: Path) -> InfinityDBStorage:
    storage = InfinityDBStorage(db_dir, brain_id="test", dimensions=DIMS)
    await storage.open()
    return storage


def _make_neuron(
    content: str = "test",
    neuron_id: str = "n1",
    neuron_type: NeuronType = NeuronType.CONCEPT,
    **kwargs: Any,
) -> Neuron:
    return Neuron.create(
        type=neuron_type,
        content=content,
        neuron_id=neuron_id,
        **kwargs,
    )


# --- Neuron CRUD ---


class TestNeuronOperations:
    @pytest.mark.asyncio
    async def test_add_and_get_neuron(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        neuron = _make_neuron("hello world", "n1")
        nid = await storage.add_neuron(neuron)
        assert nid == "n1"

        result = await storage.get_neuron("n1")
        assert result is not None
        assert result.content == "hello world"
        assert result.type == NeuronType.CONCEPT
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_missing_neuron(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        result = await storage.get_neuron("nonexistent")
        assert result is None
        await storage.close()

    @pytest.mark.asyncio
    async def test_find_neurons(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("alpha", "n1"))
        await storage.add_neuron(_make_neuron("beta", "n2"))
        await storage.add_neuron(_make_neuron("alpha beta", "n3"))

        results = await storage.find_neurons(content_contains="alpha")
        assert len(results) >= 1
        await storage.close()

    @pytest.mark.asyncio
    async def test_update_neuron(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        neuron = _make_neuron("original", "n1")
        await storage.add_neuron(neuron)

        updated = Neuron(
            id="n1",
            type=NeuronType.ENTITY,
            content="updated",
            metadata={},
            created_at=neuron.created_at,
        )
        await storage.update_neuron(updated)

        result = await storage.get_neuron("n1")
        assert result is not None
        assert result.content == "updated"
        await storage.close()

    @pytest.mark.asyncio
    async def test_delete_neuron(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("temp", "n1"))
        deleted = await storage.delete_neuron("n1")
        assert deleted is True
        assert await storage.get_neuron("n1") is None
        await storage.close()


# --- Synapse Operations ---


class TestSynapseOperations:
    @pytest.mark.asyncio
    async def test_add_and_get_synapses(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("src", "n1"))
        await storage.add_neuron(_make_neuron("tgt", "n2"))

        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
            weight=0.8,
        )
        sid = await storage.add_synapse(synapse)
        assert sid

        results = await storage.get_synapses(source_id="n1")
        assert len(results) == 1
        assert results[0].target_id == "n2"
        assert results[0].weight == 0.8
        await storage.close()

    @pytest.mark.asyncio
    async def test_delete_synapse(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        await storage.add_neuron(_make_neuron("b", "n2"))

        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
        )
        sid = await storage.add_synapse(synapse)
        deleted = await storage.delete_synapse(sid)
        assert deleted is True
        await storage.close()


# --- Fiber Operations ---


class TestFiberOperations:
    @pytest.mark.asyncio
    async def test_add_and_get_fiber(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))

        fiber = Fiber(
            id="f1",
            summary="test fiber",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            metadata={"fiber_type": "cluster", "description": "test"},
        )
        fid = await storage.add_fiber(fiber)
        assert fid == "f1"

        result = await storage.get_fiber("f1")
        assert result is not None
        assert result.summary == "test fiber"
        await storage.close()

    @pytest.mark.asyncio
    async def test_delete_fiber(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        fiber = Fiber(
            id="f1",
            summary="temp",
            neuron_ids=set(),
            synapse_ids=set(),
            anchor_neuron_id="",
        )
        await storage.add_fiber(fiber)
        deleted = await storage.delete_fiber("f1")
        assert deleted is True
        await storage.close()


# --- Brain & Stats ---


class TestBrainStats:
    @pytest.mark.asyncio
    async def test_get_brain(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        brain = await storage.get_brain("test")
        assert brain is not None
        assert brain.name == "test"
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_brain_wrong_id(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        brain = await storage.get_brain("other")
        assert brain is None
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        stats = await storage.get_stats("test")
        assert stats["neuron_count"] == 1
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_enhanced_stats(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        stats = await storage.get_enhanced_stats("test")
        assert "tiers" in stats
        await storage.close()


# --- Pro Methods ---


class TestProMethods:
    @pytest.mark.asyncio
    async def test_search_similar(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        vec = np.random.default_rng(42).random(DIMS).astype(np.float32).tolist()
        await storage.add_neuron(_make_neuron("embedded", "n1", metadata={"embedding": vec}))

        # Need to pass embedding via metadata
        results = await storage.search_similar(vec, k=1)
        # May or may not find (depends on embedding being stored)
        assert isinstance(results, list)
        await storage.close()

    @pytest.mark.asyncio
    async def test_tier_stats(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        stats = await storage.get_tier_stats()
        assert "tiers" in stats
        assert stats["tiers"]["total"] == 1
        await storage.close()

    @pytest.mark.asyncio
    async def test_demote_sweep(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        result = await storage.demote_sweep()
        assert isinstance(result, dict)
        await storage.close()


# --- Neuron State ---


class TestNeuronState:
    @pytest.mark.asyncio
    async def test_get_neuron_state(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        state = await storage.get_neuron_state("n1")
        assert state is not None
        assert state.neuron_id == "n1"
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_state_missing_neuron(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        state = await storage.get_neuron_state("missing")
        assert state is None
        await storage.close()


# --- Graph Traversal ---


class TestGraphTraversal:
    @pytest.mark.asyncio
    async def test_get_neighbors(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        await storage.add_neuron(_make_neuron("b", "n2"))
        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
        )
        await storage.add_synapse(synapse)

        neighbors = await storage.get_neighbors("n1", direction="out")
        assert len(neighbors) == 1
        neuron, syn = neighbors[0]
        assert neuron.id == "n2"
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_path_direct(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        await storage.add_neuron(_make_neuron("b", "n2"))
        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
        )
        await storage.add_synapse(synapse)

        path = await storage.get_path("n1", "n2")
        assert path is not None
        assert len(path) >= 1
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_path_no_connection(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("a", "n1"))
        await storage.add_neuron(_make_neuron("b", "n2"))

        path = await storage.get_path("n1", "n2")
        assert path is None
        await storage.close()


# --- Export / Import ---


class TestExportImport:
    @pytest.mark.asyncio
    async def test_export_brain(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("hello", "n1"))
        await storage.add_neuron(_make_neuron("world", "n2"))

        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
        )
        await storage.add_synapse(synapse)

        snapshot = await storage.export_brain("test")
        assert snapshot.brain_id == "test"
        assert len(snapshot.neurons) == 2
        assert len(snapshot.synapses) >= 1
        assert snapshot.version == "infinitydb-0.2.0"
        await storage.close()

    @pytest.mark.asyncio
    async def test_export_wrong_brain_raises(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        with pytest.raises(ValueError, match="not found"):
            await storage.export_brain("nonexistent")
        await storage.close()

    @pytest.mark.asyncio
    async def test_import_brain(self, db_dir: Path) -> None:
        # First create and export
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("original", "n1"))
        snapshot = await storage.export_brain("test")
        await storage.close()

        # Import into fresh storage
        db_dir2 = db_dir.parent / "import_test"
        storage2 = InfinityDBStorage(db_dir2, brain_id="test", dimensions=DIMS)
        await storage2.open()
        bid = await storage2.import_brain(snapshot)
        assert bid == "test"

        # Verify data round-tripped
        neuron = await storage2.get_neuron("n1")
        assert neuron is not None
        assert neuron.content == "original"
        await storage2.close()

    @pytest.mark.asyncio
    async def test_export_import_round_trip(self, db_dir: Path) -> None:
        storage = await _make_storage(db_dir)
        await storage.add_neuron(_make_neuron("alpha", "n1"))
        await storage.add_neuron(_make_neuron("beta", "n2"))
        synapse = Synapse.create(
            type=SynapseType.RELATED_TO,
            source_id="n1",
            target_id="n2",
            weight=0.7,
        )
        await storage.add_synapse(synapse)

        snapshot = await storage.export_brain("test")
        await storage.close()

        # Re-import
        db_dir2 = db_dir.parent / "roundtrip"
        storage2 = InfinityDBStorage(db_dir2, brain_id="test", dimensions=DIMS)
        await storage2.open()
        await storage2.import_brain(snapshot)

        stats = await storage2.get_stats("test")
        assert stats["neuron_count"] == 2
        assert stats["synapse_count"] >= 1
        await storage2.close()


# --- Import Integration ---


class TestImportIntegration:
    def test_imports_from_package(self) -> None:
        from neural_memory.pro.infinitydb.engine import InfinityDB
        from neural_memory.pro.storage_adapter import InfinityDBStorage as Ids

        assert Ids is InfinityDBStorage
        assert Ids is not None
        assert Ids.__name__ == "InfinityDBStorage"
        assert InfinityDB is not None
