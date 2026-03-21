"""Tests for Phase 5: Sync Safety.

Covers:
- Neuron dedup on sync import (content_hash match)
- Fiber dedup on sync import (anchor match + tag merge)
- Skips for non-duplicate content
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncResponse,
    SyncStatus,
)
from neural_memory.sync.sync_engine import SyncEngine
from neural_memory.utils.timeutils import utcnow


def _make_storage() -> AsyncMock:
    """Create a mock storage with standard methods."""
    storage = AsyncMock()
    storage.current_brain_id = "test"
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers_batch = AsyncMock(return_value=[])
    storage.add_neuron = AsyncMock()
    storage.add_synapse = AsyncMock()
    storage.add_fiber = AsyncMock()
    storage.update_neuron = AsyncMock()
    storage.update_synapse = AsyncMock()
    storage.update_fiber = AsyncMock()
    storage.mark_synced = AsyncMock()
    storage.update_device_sync = AsyncMock()
    return storage


# ---------------------------------------------------------------------------
# Neuron dedup on import
# ---------------------------------------------------------------------------


class TestNeuronDedupOnImport:
    """Test content_hash based dedup when importing neurons."""

    @pytest.mark.asyncio
    async def test_skip_neuron_with_matching_hash(self) -> None:
        """Neuron with same content_hash as existing should be skipped."""
        storage = _make_storage()

        # Mock storage method to report content_hash match
        storage.has_neuron_by_content_hash = AsyncMock(return_value=True)

        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="new-1",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "new-1",
                "type": "concept",
                "content": "test duplicate",
                "content_hash": 12345,
            },
        )

        await engine._apply_remote_change(change)
        # Should NOT have called add_neuron
        storage.add_neuron.assert_not_called()

    @pytest.mark.asyncio
    async def test_insert_neuron_without_hash_match(self) -> None:
        """Neuron with no matching content_hash should be inserted."""
        storage = _make_storage()
        storage.find_neurons = AsyncMock(return_value=[])  # No match
        storage.has_neuron_by_content_hash = AsyncMock(return_value=False)

        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="new-1",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "new-1",
                "type": "concept",
                "content": "unique content",
                "content_hash": 99999,
            },
        )

        await engine._apply_remote_change(change)
        storage.add_neuron.assert_called_once()

    @pytest.mark.asyncio
    async def test_neuron_without_content_hash_always_inserted(self) -> None:
        """Neuron with content_hash=0 should skip dedup and insert."""
        storage = _make_storage()
        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="new-1",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "new-1",
                "type": "concept",
                "content": "no hash",
                "content_hash": 0,
            },
        )

        await engine._apply_remote_change(change)
        storage.add_neuron.assert_called_once()


# ---------------------------------------------------------------------------
# Fiber dedup on import (anchor match + tag merge)
# ---------------------------------------------------------------------------


class TestFiberDedupOnImport:
    """Test fiber dedup via anchor neuron matching."""

    @pytest.mark.asyncio
    async def test_merge_fiber_with_same_anchor(self) -> None:
        """Fiber with same anchor_neuron_id should merge tags, not insert."""
        storage = _make_storage()

        existing_fiber = Fiber(
            id="existing-fiber",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="anchor-1",
            auto_tags={"python"},
            agent_tags={"agent:claude-code"},
            metadata={"project": "nm"},
            frequency=1,
        )
        storage.find_fibers_batch = AsyncMock(return_value=[existing_fiber])

        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="fiber",
            entity_id="incoming-fiber",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "incoming-fiber",
                "anchor_neuron_id": "anchor-1",
                "neuron_ids": ["n1", "n3"],
                "synapse_ids": [],
                "auto_tags": ["auth"],
                "agent_tags": ["agent:cursor"],
                "metadata": {"source": "cursor"},
                "frequency": 1,
            },
        )

        await engine._apply_remote_change(change)

        # Should have updated existing fiber, not inserted new one
        storage.add_fiber.assert_not_called()
        storage.update_fiber.assert_called_once()

        # Check merged tags
        updated_fiber = storage.update_fiber.call_args[0][0]
        assert "python" in updated_fiber.auto_tags
        assert "auth" in updated_fiber.auto_tags
        assert "agent:claude-code" in updated_fiber.agent_tags
        assert "agent:cursor" in updated_fiber.agent_tags
        assert updated_fiber.frequency == 2  # incremented

    @pytest.mark.asyncio
    async def test_insert_fiber_without_anchor_match(self) -> None:
        """Fiber with no matching anchor should be inserted normally."""
        storage = _make_storage()
        storage.find_fibers_batch = AsyncMock(return_value=[])

        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="fiber",
            entity_id="new-fiber",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "new-fiber",
                "anchor_neuron_id": "unique-anchor",
                "neuron_ids": ["n5"],
                "synapse_ids": [],
                "auto_tags": [],
                "agent_tags": [],
                "metadata": {},
                "frequency": 0,
            },
        )

        await engine._apply_remote_change(change)
        storage.add_fiber.assert_called_once()

    @pytest.mark.asyncio
    async def test_fiber_without_anchor_always_inserted(self) -> None:
        """Fiber with no anchor_neuron_id should skip dedup."""
        storage = _make_storage()
        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="fiber",
            entity_id="no-anchor",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "no-anchor",
                "anchor_neuron_id": "",
                "neuron_ids": ["n6"],
                "synapse_ids": [],
                "auto_tags": [],
                "agent_tags": [],
                "metadata": {},
                "frequency": 0,
            },
        )

        await engine._apply_remote_change(change)
        storage.add_fiber.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_fiber_id_not_merged(self) -> None:
        """If existing fiber has same ID as incoming, let normal update handle it."""
        storage = _make_storage()

        existing = Fiber(
            id="same-id",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="anchor-1",
        )
        storage.find_fibers_batch = AsyncMock(return_value=[existing])

        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        change = SyncChange(
            sequence=1,
            entity_type="fiber",
            entity_id="same-id",
            operation="insert",
            device_id="device-2",
            changed_at=utcnow().isoformat(),
            payload={
                "id": "same-id",
                "anchor_neuron_id": "anchor-1",
                "neuron_ids": ["n1"],
                "synapse_ids": [],
                "auto_tags": [],
                "agent_tags": [],
                "metadata": {},
                "frequency": 0,
            },
        )

        await engine._apply_remote_change(change)
        # Should insert (or update via ValueError catch), not merge
        storage.add_fiber.assert_called_once()


# ---------------------------------------------------------------------------
# Process sync response integration
# ---------------------------------------------------------------------------


class TestSyncResponseDedup:
    """Integration test: process_sync_response with dedup changes."""

    @pytest.mark.asyncio
    async def test_skips_own_device_changes(self) -> None:
        """Changes from own device should be skipped."""
        storage = _make_storage()
        storage.get_change_log_stats = AsyncMock(return_value={"last_sequence": 5})
        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        response = SyncResponse(
            hub_sequence=5,
            changes=[
                SyncChange(
                    sequence=1,
                    entity_type="neuron",
                    entity_id="n1",
                    operation="insert",
                    device_id="device-1",  # own device
                    changed_at=utcnow().isoformat(),
                    payload={"id": "n1", "type": "concept", "content": "test"},
                ),
            ],
            conflicts=[],
            status=SyncStatus.SUCCESS,
        )

        result = await engine.process_sync_response(response)
        assert result["skipped"] == 1
        assert result["applied"] == 0
        storage.add_neuron.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_remote_changes(self) -> None:
        """Changes from other devices should be applied."""
        storage = _make_storage()
        storage.find_neurons = AsyncMock(return_value=[])  # No dedup match
        engine = SyncEngine(storage, "device-1", ConflictStrategy.PREFER_RECENT)

        response = SyncResponse(
            hub_sequence=5,
            changes=[
                SyncChange(
                    sequence=1,
                    entity_type="neuron",
                    entity_id="n1",
                    operation="insert",
                    device_id="device-2",
                    changed_at=utcnow().isoformat(),
                    payload={
                        "id": "n1",
                        "type": "concept",
                        "content": "from device 2",
                        "content_hash": 0,
                    },
                ),
            ],
            conflicts=[],
            status=SyncStatus.SUCCESS,
        )

        result = await engine.process_sync_response(response)
        assert result["applied"] == 1
        storage.add_neuron.assert_called_once()
