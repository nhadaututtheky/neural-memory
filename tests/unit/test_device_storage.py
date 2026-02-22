"""Tests for SQLiteDevicesMixin CRUD operations."""

from __future__ import annotations

import pathlib

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.storage.sqlite_devices import DeviceRecord
from neural_memory.storage.sqlite_store import SQLiteStorage


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage_with_brain(tmp_path: pathlib.Path) -> SQLiteStorage:
    """SQLiteStorage with one initialized brain, ready for device tests."""
    db_path = tmp_path / "test_devices.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    brain = Brain.create(name="device-test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    yield storage

    await storage.close()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRegisterDevice:
    """Test register_device creates and returns a DeviceRecord."""

    async def test_register_device(self, storage_with_brain: SQLiteStorage) -> None:
        """Register a device — verify all returned fields."""
        record = await storage_with_brain.register_device(
            device_id="dev-001", device_name="my-laptop"
        )

        assert isinstance(record, DeviceRecord)
        assert record.device_id == "dev-001"
        assert record.device_name == "my-laptop"
        assert record.last_sync_at is None
        assert record.last_sync_sequence == 0
        # registered_at is populated
        assert record.registered_at is not None

    async def test_register_device_upsert(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Registering the same device_id twice updates device_name."""
        await storage_with_brain.register_device("dev-001", "old-name")
        await storage_with_brain.register_device("dev-001", "new-name")

        fetched = await storage_with_brain.get_device("dev-001")
        assert fetched is not None
        assert fetched.device_name == "new-name"

    async def test_register_device_without_name(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """register_device with no name uses empty string."""
        record = await storage_with_brain.register_device("dev-no-name")
        assert record.device_name == ""

    async def test_register_device_stores_brain_id(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Registered DeviceRecord carries the current brain_id."""
        record = await storage_with_brain.register_device("dev-002", "desktop")
        expected_brain_id = storage_with_brain._get_brain_id()
        assert record.brain_id == expected_brain_id


class TestGetDevice:
    """Test get_device retrieves or returns None."""

    async def test_get_device_not_found(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_device returns None when device_id is not registered."""
        result = await storage_with_brain.get_device("nonexistent-dev")
        assert result is None

    async def test_get_device_returns_record(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_device returns the correct DeviceRecord after registration."""
        await storage_with_brain.register_device("dev-abc", "work-machine")

        record = await storage_with_brain.get_device("dev-abc")
        assert record is not None
        assert record.device_id == "dev-abc"
        assert record.device_name == "work-machine"

    async def test_get_device_returns_device_record_type(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_device returns a DeviceRecord instance."""
        await storage_with_brain.register_device("dev-typed", "typed-machine")
        record = await storage_with_brain.get_device("dev-typed")
        assert isinstance(record, DeviceRecord)


class TestListDevices:
    """Test list_devices returns all devices sorted by registered_at."""

    async def test_list_devices_empty(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """list_devices returns empty list when no devices registered."""
        devices = await storage_with_brain.list_devices()
        assert devices == []

    async def test_list_devices_two_devices(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """register 2 devices, list returns 2 sorted by registered_at ASC."""
        await storage_with_brain.register_device("dev-first", "machine-a")
        await storage_with_brain.register_device("dev-second", "machine-b")

        devices = await storage_with_brain.list_devices()
        assert len(devices) == 2
        # Both device IDs present
        ids = {d.device_id for d in devices}
        assert "dev-first" in ids
        assert "dev-second" in ids

    async def test_list_devices_sorted_by_registered_at(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Devices come back in ascending registered_at order."""
        await storage_with_brain.register_device("dev-a", "alpha")
        await storage_with_brain.register_device("dev-b", "beta")
        await storage_with_brain.register_device("dev-c", "gamma")

        devices = await storage_with_brain.list_devices()
        assert len(devices) == 3
        for i in range(len(devices) - 1):
            assert devices[i].registered_at <= devices[i + 1].registered_at

    async def test_list_devices_returns_device_record_instances(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """All items returned by list_devices are DeviceRecord instances."""
        await storage_with_brain.register_device("dev-x", "x")
        devices = await storage_with_brain.list_devices()
        assert all(isinstance(d, DeviceRecord) for d in devices)


class TestUpdateDeviceSync:
    """Test update_device_sync updates last_sync_at and last_sync_sequence."""

    async def test_update_device_sync(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """After update_device_sync, fetched record reflects new values."""
        await storage_with_brain.register_device("dev-sync", "sync-machine")

        await storage_with_brain.update_device_sync("dev-sync", last_sync_sequence=42)

        record = await storage_with_brain.get_device("dev-sync")
        assert record is not None
        assert record.last_sync_sequence == 42
        assert record.last_sync_at is not None

    async def test_update_device_sync_increases_sequence(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Updating sync twice carries the latest sequence."""
        await storage_with_brain.register_device("dev-seq", "seq-machine")
        await storage_with_brain.update_device_sync("dev-seq", 10)
        await storage_with_brain.update_device_sync("dev-seq", 99)

        record = await storage_with_brain.get_device("dev-seq")
        assert record is not None
        assert record.last_sync_sequence == 99

    async def test_update_device_sync_sets_last_sync_at(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """update_device_sync sets last_sync_at to a recent timestamp."""
        from neural_memory.utils.timeutils import utcnow

        await storage_with_brain.register_device("dev-ts", "ts-machine")
        before = utcnow()
        await storage_with_brain.update_device_sync("dev-ts", 1)
        after = utcnow()

        record = await storage_with_brain.get_device("dev-ts")
        assert record is not None
        assert record.last_sync_at is not None
        assert before <= record.last_sync_at <= after


class TestRemoveDevice:
    """Test remove_device deletes a registered device."""

    async def test_remove_device(self, storage_with_brain: SQLiteStorage) -> None:
        """remove_device returns True and the device is no longer found."""
        await storage_with_brain.register_device("dev-remove", "to-remove")

        removed = await storage_with_brain.remove_device("dev-remove")
        assert removed is True

        fetched = await storage_with_brain.get_device("dev-remove")
        assert fetched is None

    async def test_remove_device_not_found_returns_false(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """remove_device returns False if device_id does not exist."""
        result = await storage_with_brain.remove_device("ghost-device")
        assert result is False

    async def test_remove_device_does_not_affect_others(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Removing one device does not remove others."""
        await storage_with_brain.register_device("dev-keep", "keeper")
        await storage_with_brain.register_device("dev-gone", "gonner")

        await storage_with_brain.remove_device("dev-gone")

        assert await storage_with_brain.get_device("dev-keep") is not None
        assert await storage_with_brain.get_device("dev-gone") is None


class TestBrainIsolation:
    """Devices registered in brain A must not be visible from brain B."""

    async def test_brain_isolation(self, tmp_path: pathlib.Path) -> None:
        """Devices in brain A are invisible when brain B is the active context."""
        db_path = tmp_path / "isolation_devices.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()

        brain_a = Brain.create(name="brain-a", config=BrainConfig())
        brain_b = Brain.create(name="brain-b", config=BrainConfig())
        await storage.save_brain(brain_a)
        await storage.save_brain(brain_b)

        # Register a device under brain A
        storage.set_brain(brain_a.id)
        await storage.register_device("dev-in-a", "a-machine")

        # Switch to brain B — should have no devices
        storage.set_brain(brain_b.id)
        devices_b = await storage.list_devices()
        assert devices_b == []

        # Brain B get_device returns None for brain A's device
        assert await storage.get_device("dev-in-a") is None

        # Switch back to brain A — device still there
        storage.set_brain(brain_a.id)
        devices_a = await storage.list_devices()
        assert len(devices_a) == 1
        assert devices_a[0].device_id == "dev-in-a"

        await storage.close()
