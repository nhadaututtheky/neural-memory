"""Tests for InfinityDB change log, device registry, and Merkle mixin."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.brain_versioning import BrainVersion
from neural_memory.pro.storage_adapter import InfinityDBStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage(tmp_path: Path) -> InfinityDBStorage:
    s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="test")
    await s.open()
    yield s
    await s.close()


class TestChangeLog:
    async def test_record_and_get(self, storage: InfinityDBStorage) -> None:
        seq = await storage.record_change("neuron", "n1", "insert")
        assert seq >= 1
        changes = await storage.get_changes_since(0)
        assert len(changes) == 1
        assert changes[0]["entity_id"] == "n1"

    async def test_get_changes_since_filters(self, storage: InfinityDBStorage) -> None:
        s1 = await storage.record_change("neuron", "n1", "insert")
        await storage.record_change("neuron", "n2", "insert")
        changes = await storage.get_changes_since(s1)
        assert len(changes) == 1
        assert changes[0]["entity_id"] == "n2"

    async def test_mark_synced(self, storage: InfinityDBStorage) -> None:
        s1 = await storage.record_change("neuron", "n1", "insert")
        await storage.record_change("neuron", "n2", "insert")
        count = await storage.mark_synced(s1)
        assert count == 1
        unsynced = await storage.get_unsynced_changes()
        assert len(unsynced) == 1

    async def test_prune_synced(self, storage: InfinityDBStorage) -> None:
        s1 = await storage.record_change("neuron", "n1", "insert")
        await storage.mark_synced(s1)
        pruned = await storage.prune_synced_changes(older_than_days=0)
        assert pruned == 1

    async def test_stats(self, storage: InfinityDBStorage) -> None:
        await storage.record_change("neuron", "n1", "insert")
        await storage.record_change("neuron", "n2", "insert")
        await storage.mark_synced(1)
        stats = await storage.get_change_log_stats()
        assert stats["total"] == 2
        assert stats["synced"] == 1
        assert stats["pending"] == 1

    async def test_seed_change_log(self, storage: InfinityDBStorage) -> None:
        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Seed"))
        counts = await storage.seed_change_log("dev1")
        assert counts["neurons"] >= 1


class TestDeviceRegistry:
    async def test_register_and_get(self, storage: InfinityDBStorage) -> None:
        dev = await storage.register_device("d1", "My Device")
        assert dev["device_id"] == "d1"
        got = await storage.get_device("d1")
        assert got is not None

    async def test_list_devices(self, storage: InfinityDBStorage) -> None:
        await storage.register_device("d1", "A")
        await storage.register_device("d2", "B")
        devices = await storage.list_devices()
        assert len(devices) == 2

    async def test_update_sync(self, storage: InfinityDBStorage) -> None:
        await storage.register_device("d1")
        await storage.update_device_sync("d1", 42)
        got = await storage.get_device("d1")
        assert got["last_sync_sequence"] == 42

    async def test_remove(self, storage: InfinityDBStorage) -> None:
        await storage.register_device("d1")
        assert await storage.remove_device("d1")
        assert await storage.get_device("d1") is None
        assert not await storage.remove_device("d1")

    async def test_upsert(self, storage: InfinityDBStorage) -> None:
        await storage.register_device("d1", "Old")
        await storage.register_device("d1", "New")
        devices = await storage.list_devices()
        assert len(devices) == 1
        assert devices[0]["device_name"] == "New"


class TestMerkle:
    async def test_compute_root_empty(self, storage: InfinityDBStorage) -> None:
        root = await storage.compute_merkle_root("neuron")
        assert root is None

    async def test_get_merkle_tree_empty(self, storage: InfinityDBStorage) -> None:
        tree = await storage.get_merkle_tree("neuron")
        assert tree == {}

    async def test_get_merkle_root_empty(self, storage: InfinityDBStorage) -> None:
        root = await storage.get_merkle_root()
        assert root is None

    async def test_bucket_entity_ids(self, storage: InfinityDBStorage) -> None:
        nid = await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="Bucket"))
        ids = await storage.get_bucket_entity_ids("neuron", nid[:2])
        assert nid in ids


class TestVersions:
    async def test_save_get_roundtrip(self, storage: InfinityDBStorage) -> None:
        ver = BrainVersion(
            id="v1",
            brain_id="test",
            version_name="v1",
            version_number=1,
            description="First",
            neuron_count=1,
            synapse_count=0,
            fiber_count=0,
            snapshot_hash="abc123",
            created_at=utcnow(),
        )
        await storage.save_version("test", ver, '{"data": 1}')
        got = await storage.get_version("test", "v1")
        assert got is not None
        assert got[0].version_name == "v1"
        assert got[1] == '{"data": 1}'

    async def test_list_versions(self, storage: InfinityDBStorage) -> None:
        for i in range(3):
            ver = BrainVersion(
                id=f"v{i}",
                brain_id="test",
                version_name=f"v{i}",
                version_number=i + 1,
                description=f"Version {i}",
                neuron_count=0,
                synapse_count=0,
                fiber_count=0,
                snapshot_hash="h",
                created_at=utcnow(),
            )
            await storage.save_version("test", ver, "{}")
        versions = await storage.list_versions("test")
        assert len(versions) == 3
        assert versions[0].version_number > versions[-1].version_number

    async def test_next_version_number(self, storage: InfinityDBStorage) -> None:
        assert await storage.get_next_version_number("test") == 1
        ver = BrainVersion(
            id="v1",
            brain_id="test",
            version_name="v1",
            version_number=1,
            description="",
            neuron_count=0,
            synapse_count=0,
            fiber_count=0,
            snapshot_hash="h",
            created_at=utcnow(),
        )
        await storage.save_version("test", ver, "{}")
        assert await storage.get_next_version_number("test") == 2

    async def test_delete_version(self, storage: InfinityDBStorage) -> None:
        ver = BrainVersion(
            id="v1",
            brain_id="test",
            version_name="v1",
            version_number=1,
            description="",
            neuron_count=0,
            synapse_count=0,
            fiber_count=0,
            snapshot_hash="h",
            created_at=utcnow(),
        )
        await storage.save_version("test", ver, "{}")
        assert await storage.delete_version("test", "v1")
        assert not await storage.delete_version("test", "v1")
