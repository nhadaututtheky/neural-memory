"""Tests for SQLite change log operations (Multi-Device Sync Part B)."""

from __future__ import annotations

import pathlib
from dataclasses import FrozenInstanceError

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.storage.sqlite_change_log import ChangeEntry
from neural_memory.storage.sqlite_store import SQLiteStorage


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage_with_brain(tmp_path: pathlib.Path) -> SQLiteStorage:
    """Create SQLiteStorage with an initialized brain context."""
    db_path = tmp_path / "test_change_log.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    brain = Brain.create(name="change-log-test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    yield storage

    await storage.close()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRecordChange:
    """Test record_change returns a positive sequence number."""

    async def test_record_change(self, storage_with_brain: SQLiteStorage) -> None:
        """Insert a change, verify returned sequence > 0."""
        seq = await storage_with_brain.record_change(
            entity_type="neuron",
            entity_id="n-abc",
            operation="insert",
            device_id="device-1",
            payload={"content": "hello"},
        )
        assert seq > 0

    async def test_record_change_increments_sequence(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Each record_change call returns a strictly increasing sequence."""
        seq1 = await storage_with_brain.record_change(
            entity_type="neuron", entity_id="n-1", operation="insert"
        )
        seq2 = await storage_with_brain.record_change(
            entity_type="neuron", entity_id="n-2", operation="insert"
        )
        assert seq2 > seq1


class TestGetChangesSince:
    """Test get_changes_since returns changes in order after a given sequence."""

    async def test_get_changes_since_zero_returns_all(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_changes_since(0) returns all 3 inserted changes in order."""
        await storage_with_brain.record_change("neuron", "n-1", "insert", device_id="dev-a")
        await storage_with_brain.record_change("synapse", "s-1", "update", device_id="dev-a")
        await storage_with_brain.record_change("fiber", "f-1", "delete", device_id="dev-a")

        changes = await storage_with_brain.get_changes_since(0)
        assert len(changes) == 3
        assert changes[0].entity_type == "neuron"
        assert changes[1].entity_type == "synapse"
        assert changes[2].entity_type == "fiber"

    async def test_get_changes_since_filters_by_sequence(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_changes_since(seq1) returns only changes after seq1."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        await storage_with_brain.record_change("neuron", "n-2", "insert")
        await storage_with_brain.record_change("neuron", "n-3", "insert")

        changes = await storage_with_brain.get_changes_since(seq1)
        # Only the 2nd and 3rd changes come back (id > seq1)
        assert len(changes) == 2

    async def test_get_changes_since_returns_ordered_asc(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Changes are returned in ascending id order."""
        for i in range(5):
            await storage_with_brain.record_change("neuron", f"n-{i}", "insert")

        changes = await storage_with_brain.get_changes_since(0)
        assert len(changes) == 5
        for i in range(len(changes) - 1):
            assert changes[i].id < changes[i + 1].id

    async def test_get_changes_since_returns_change_entry_instances(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """get_changes_since returns ChangeEntry instances."""
        await storage_with_brain.record_change("neuron", "n-x", "insert")
        changes = await storage_with_brain.get_changes_since(0)
        assert len(changes) >= 1
        assert all(isinstance(c, ChangeEntry) for c in changes)


class TestGetUnsyncedChanges:
    """Test get_unsynced_changes returns only un-marked changes."""

    async def test_get_unsynced_changes_returns_all_when_none_synced(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """All new changes are unsynced initially."""
        await storage_with_brain.record_change("neuron", "n-1", "insert")
        await storage_with_brain.record_change("neuron", "n-2", "insert")

        unsynced = await storage_with_brain.get_unsynced_changes()
        assert len(unsynced) == 2

    async def test_get_unsynced_changes_excludes_synced(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """After mark_synced(seq1), get_unsynced_changes returns only seq2+."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        await storage_with_brain.record_change("neuron", "n-2", "insert")

        await storage_with_brain.mark_synced(seq1)

        unsynced = await storage_with_brain.get_unsynced_changes()
        assert len(unsynced) == 1
        assert unsynced[0].entity_id == "n-2"

    async def test_get_unsynced_changes_empty_after_all_synced(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """After marking all synced, get_unsynced_changes returns empty list."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        seq2 = await storage_with_brain.record_change("neuron", "n-2", "insert")

        await storage_with_brain.mark_synced(max(seq1, seq2))

        unsynced = await storage_with_brain.get_unsynced_changes()
        assert unsynced == []


class TestMarkSynced:
    """Test mark_synced marks changes and returns correct count."""

    async def test_mark_synced_returns_count(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """mark_synced returns the number of rows marked."""
        await storage_with_brain.record_change("neuron", "n-1", "insert")
        seq2 = await storage_with_brain.record_change("neuron", "n-2", "insert")

        count = await storage_with_brain.mark_synced(seq2)
        assert count == 2

    async def test_mark_synced_only_marks_unsynced(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """mark_synced does not count already-synced rows."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        seq2 = await storage_with_brain.record_change("neuron", "n-2", "insert")

        # Mark first batch
        count1 = await storage_with_brain.mark_synced(seq1)
        assert count1 == 1

        # Mark second batch — only the second row should be newly marked
        count2 = await storage_with_brain.mark_synced(seq2)
        assert count2 == 1

    async def test_mark_synced_persists_synced_flag(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """After mark_synced, get_changes_since still returns those changes
        (synced flag is separate from existence)."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        await storage_with_brain.mark_synced(seq1)

        changes = await storage_with_brain.get_changes_since(0)
        assert len(changes) == 1
        assert changes[0].synced is True


class TestPruneSyncedChanges:
    """Test prune_synced_changes removes old synced entries."""

    async def test_prune_synced_changes_removes_synced_old(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Synced changes older than older_than_days=0 are pruned."""
        seq = await storage_with_brain.record_change("neuron", "n-old", "insert")
        await storage_with_brain.mark_synced(seq)

        # older_than_days=0 means everything older than now → prunes immediately
        pruned = await storage_with_brain.prune_synced_changes(older_than_days=0)
        assert pruned >= 1

        # Verify gone
        changes = await storage_with_brain.get_changes_since(0)
        assert all(c.entity_id != "n-old" for c in changes)

    async def test_prune_synced_changes_keeps_unsynced(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Unsynced changes are never pruned."""
        await storage_with_brain.record_change("neuron", "n-keep", "insert")
        # Do NOT mark synced

        pruned = await storage_with_brain.prune_synced_changes(older_than_days=0)
        assert pruned == 0

        unsynced = await storage_with_brain.get_unsynced_changes()
        assert any(c.entity_id == "n-keep" for c in unsynced)

    async def test_prune_synced_changes_returns_count(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """prune_synced_changes returns an integer count."""
        result = await storage_with_brain.prune_synced_changes(older_than_days=365)
        assert isinstance(result, int)
        assert result >= 0


class TestGetChangeLogStats:
    """Test get_change_log_stats returns accurate aggregates."""

    async def test_get_change_log_stats_empty(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Stats on empty change log return all zeros."""
        stats = await storage_with_brain.get_change_log_stats()
        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["synced"] == 0
        assert stats["last_sequence"] == 0

    async def test_get_change_log_stats_with_data(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """Insert 3 changes, mark 1 synced, verify stats totals."""
        seq1 = await storage_with_brain.record_change("neuron", "n-1", "insert")
        await storage_with_brain.record_change("neuron", "n-2", "insert")
        await storage_with_brain.record_change("neuron", "n-3", "insert")

        await storage_with_brain.mark_synced(seq1)

        stats = await storage_with_brain.get_change_log_stats()
        assert stats["total"] == 3
        assert stats["synced"] == 1
        assert stats["pending"] == 2
        assert stats["last_sequence"] > 0

    async def test_get_change_log_stats_last_sequence_is_highest_id(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """last_sequence equals the highest change id inserted."""
        await storage_with_brain.record_change("neuron", "n-1", "insert")
        seq_last = await storage_with_brain.record_change("neuron", "n-2", "insert")

        stats = await storage_with_brain.get_change_log_stats()
        assert stats["last_sequence"] == seq_last


class TestChangeEntryFields:
    """Test ChangeEntry frozen dataclass immutability and field types."""

    async def test_change_entry_fields_populated(
        self, storage_with_brain: SQLiteStorage
    ) -> None:
        """ChangeEntry returned from storage has all expected fields."""
        await storage_with_brain.record_change(
            entity_type="fiber",
            entity_id="f-xyz",
            operation="update",
            device_id="dev-42",
            payload={"salience": 0.9},
        )

        changes = await storage_with_brain.get_changes_since(0)
        assert len(changes) == 1
        entry = changes[0]

        assert entry.entity_type == "fiber"
        assert entry.entity_id == "f-xyz"
        assert entry.operation == "update"
        assert entry.device_id == "dev-42"
        assert entry.payload == {"salience": 0.9}
        assert entry.synced is False
        assert entry.id > 0

    def test_change_entry_immutable(self) -> None:
        """ChangeEntry is a frozen dataclass — mutation raises FrozenInstanceError."""
        from datetime import datetime

        entry = ChangeEntry(
            id=1,
            brain_id="b",
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at=datetime(2026, 1, 1),
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            entry.entity_id = "n-2"  # type: ignore[misc]

    def test_change_entry_default_payload_empty(self) -> None:
        """ChangeEntry payload defaults to an empty dict."""
        from datetime import datetime

        entry = ChangeEntry(
            id=1,
            brain_id="b",
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at=datetime(2026, 1, 1),
        )
        assert entry.payload == {}

    def test_change_entry_default_synced_false(self) -> None:
        """ChangeEntry synced defaults to False."""
        from datetime import datetime

        entry = ChangeEntry(
            id=1,
            brain_id="b",
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at=datetime(2026, 1, 1),
        )
        assert entry.synced is False


class TestBrainIsolation:
    """Changes in brain A must not be visible from brain B."""

    async def test_brain_isolation(self, tmp_path: pathlib.Path) -> None:
        """Changes in brain A are not visible when brain B is active."""
        db_path = tmp_path / "isolation_test.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()

        brain_a = Brain.create(name="brain-a", config=BrainConfig())
        brain_b = Brain.create(name="brain-b", config=BrainConfig())
        await storage.save_brain(brain_a)
        await storage.save_brain(brain_b)

        # Write change in brain A
        storage.set_brain(brain_a.id)
        await storage.record_change("neuron", "n-in-a", "insert", device_id="dev")

        # Switch to brain B — should see no changes
        storage.set_brain(brain_b.id)
        changes_b = await storage.get_changes_since(0)
        assert changes_b == []

        # Switch back to brain A — should see 1 change
        storage.set_brain(brain_a.id)
        changes_a = await storage.get_changes_since(0)
        assert len(changes_a) == 1
        assert changes_a[0].entity_id == "n-in-a"

        await storage.close()
