"""Tests for SyncEngine orchestrator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.storage.sqlite_change_log import ChangeEntry
from neural_memory.storage.sqlite_devices import DeviceRecord
from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)
from neural_memory.sync.sync_engine import SyncEngine


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_storage(
    *,
    device: DeviceRecord | None = None,
    unsynced_changes: list[ChangeEntry] | None = None,
    changes_since: list[ChangeEntry] | None = None,
    mark_synced_count: int = 0,
    last_sequence: int = 10,
) -> MagicMock:
    """Build an AsyncMock storage with sensible defaults."""
    mock = AsyncMock()
    mock.get_device = AsyncMock(return_value=device)
    mock.get_unsynced_changes = AsyncMock(return_value=unsynced_changes or [])
    mock.get_changes_since = AsyncMock(return_value=changes_since or [])
    mock.mark_synced = AsyncMock(return_value=mark_synced_count)
    mock.update_device_sync = AsyncMock()
    mock.record_change = AsyncMock(return_value=1)
    mock.get_change_log_stats = AsyncMock(return_value={"last_sequence": last_sequence})
    return mock


def _make_change_entry(
    entity_id: str = "n-1",
    entity_type: str = "neuron",
    operation: str = "insert",
    device_id: str = "dev-remote",
    seq: int = 1,
) -> ChangeEntry:
    return ChangeEntry(
        id=seq,
        brain_id="brain-test",
        entity_type=entity_type,
        entity_id=entity_id,
        operation=operation,
        device_id=device_id,
        changed_at=datetime(2026, 1, 15, 10, 0, 0),
        payload={"content": "test"},
    )


def _make_device_record(
    device_id: str = "dev-001",
    last_sync_sequence: int = 0,
) -> DeviceRecord:
    return DeviceRecord(
        device_id=device_id,
        brain_id="brain-test",
        device_name="test-machine",
        last_sync_at=None,
        last_sync_sequence=last_sync_sequence,
        registered_at=datetime(2026, 1, 1),
    )


# ── prepare_sync_request ──────────────────────────────────────────────────────


class TestPrepareSyncRequest:
    """Test prepare_sync_request builds a correct SyncRequest."""

    async def test_prepare_sync_request_no_device(self) -> None:
        """When device is not registered, last_sequence defaults to 0."""
        storage = _make_mock_storage(device=None, unsynced_changes=[])
        engine = SyncEngine(storage, device_id="dev-001")

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert isinstance(request, SyncRequest)
        assert request.device_id == "dev-001"
        assert request.brain_id == "brain-test"
        assert request.last_sequence == 0
        assert request.changes == []

    async def test_prepare_sync_request_with_device(self) -> None:
        """When device is registered with last_sync_sequence=5, last_sequence=5."""
        device = _make_device_record(device_id="dev-001", last_sync_sequence=5)
        storage = _make_mock_storage(device=device, unsynced_changes=[])
        engine = SyncEngine(storage, device_id="dev-001")

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert request.last_sequence == 5

    async def test_prepare_sync_request_includes_unsynced_changes(self) -> None:
        """Unsynced local changes are packed into SyncChange objects."""
        entries = [
            _make_change_entry("n-1", seq=1, device_id="dev-001"),
            _make_change_entry("n-2", seq=2, device_id="dev-001"),
        ]
        storage = _make_mock_storage(unsynced_changes=entries)
        engine = SyncEngine(storage, device_id="dev-001")

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert len(request.changes) == 2
        entity_ids = {c.entity_id for c in request.changes}
        assert "n-1" in entity_ids
        assert "n-2" in entity_ids

    async def test_prepare_sync_request_change_maps_fields(self) -> None:
        """ChangeEntry fields are correctly mapped to SyncChange fields."""
        entry = _make_change_entry("n-abc", seq=7, device_id="dev-001", operation="update")
        storage = _make_mock_storage(unsynced_changes=[entry])
        engine = SyncEngine(storage, device_id="dev-001")

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert len(request.changes) == 1
        change = request.changes[0]
        assert change.sequence == 7
        assert change.entity_id == "n-abc"
        assert change.operation == "update"
        assert change.device_id == "dev-001"

    async def test_prepare_sync_request_uses_default_strategy(self) -> None:
        """Default strategy on SyncRequest is PREFER_RECENT."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-001")

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert request.strategy == ConflictStrategy.PREFER_RECENT

    async def test_prepare_sync_request_uses_configured_strategy(self) -> None:
        """SyncEngine strategy propagates to the SyncRequest."""
        storage = _make_mock_storage()
        engine = SyncEngine(
            storage, device_id="dev-001", strategy=ConflictStrategy.PREFER_LOCAL
        )

        request = await engine.prepare_sync_request(brain_id="brain-test")

        assert request.strategy == ConflictStrategy.PREFER_LOCAL


# ── process_sync_response ─────────────────────────────────────────────────────


class TestProcessSyncResponse:
    """Test process_sync_response applies remote changes correctly."""

    async def test_process_sync_response_applies_remote(self) -> None:
        """Non-self changes are applied (counted in 'applied')."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        # Changes from a different device
        changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-remote",
                operation="insert",
                device_id="dev-remote",  # Different from dev-local
                changed_at="2026-01-15T10:00:00",
            )
        ]
        response = SyncResponse(hub_sequence=5, changes=changes)

        result = await engine.process_sync_response(response)

        assert result["applied"] == 1
        assert result["skipped"] == 0

    async def test_process_sync_response_skips_self(self) -> None:
        """Changes from own device_id are skipped."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-self")

        # Change originating from self
        changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-self",
                operation="update",
                device_id="dev-self",  # Same as engine's device_id
                changed_at="2026-01-15T10:00:00",
            )
        ]
        response = SyncResponse(hub_sequence=3, changes=changes)

        result = await engine.process_sync_response(response)

        assert result["skipped"] == 1
        assert result["applied"] == 0

    async def test_process_sync_response_marks_synced(self) -> None:
        """process_sync_response calls mark_synced with hub_sequence."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        response = SyncResponse(hub_sequence=42, changes=[])
        await engine.process_sync_response(response)

        storage.mark_synced.assert_called_once_with(42)

    async def test_process_sync_response_calls_update_device_sync(self) -> None:
        """process_sync_response calls update_device_sync with hub_sequence."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        response = SyncResponse(hub_sequence=99, changes=[])
        await engine.process_sync_response(response)

        storage.update_device_sync.assert_called_once_with("dev-local", 99)

    async def test_process_sync_response_skips_mark_synced_on_zero(self) -> None:
        """If hub_sequence is 0, mark_synced is not called."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        response = SyncResponse(hub_sequence=0, changes=[])
        await engine.process_sync_response(response)

        storage.mark_synced.assert_not_called()

    async def test_process_sync_response_returns_conflict_count(self) -> None:
        """conflicts count in result matches response.conflicts length."""
        from neural_memory.sync.protocol import SyncConflict

        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        conflicts = [
            SyncConflict(
                entity_type="neuron",
                entity_id="n-1",
                local_device="dev-local",
                remote_device="dev-remote",
                resolution="local_prefer_recent",
            )
        ]
        response = SyncResponse(hub_sequence=5, changes=[], conflicts=conflicts)
        result = await engine.process_sync_response(response)

        assert result["conflicts"] == 1

    async def test_process_sync_response_returns_hub_sequence(self) -> None:
        """hub_sequence in result matches response.hub_sequence."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-local")

        response = SyncResponse(hub_sequence=77, changes=[])
        result = await engine.process_sync_response(response)

        assert result["hub_sequence"] == 77

    async def test_process_sync_response_mixed_changes(self) -> None:
        """Mix of self and remote changes: self skipped, remote applied."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-self")

        changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-self",
                operation="update",
                device_id="dev-self",
                changed_at="2026-01-15T10:00:00",
            ),
            SyncChange(
                sequence=2,
                entity_type="neuron",
                entity_id="n-remote",
                operation="insert",
                device_id="dev-other",
                changed_at="2026-01-15T10:00:00",
            ),
        ]
        response = SyncResponse(hub_sequence=10, changes=changes)
        result = await engine.process_sync_response(response)

        assert result["applied"] == 1
        assert result["skipped"] == 1


# ── handle_hub_sync ───────────────────────────────────────────────────────────


class TestHandleHubSync:
    """Test handle_hub_sync filters and records changes correctly."""

    async def test_handle_hub_sync_returns_unseen(self) -> None:
        """Hub returns changes from other devices that requester hasn't seen."""
        # Hub has two stored changes: one from requester, one from another device
        hub_changes = [
            _make_change_entry("n-1", device_id="dev-requester", seq=1),
            _make_change_entry("n-2", device_id="dev-other", seq=2),
        ]
        storage = _make_mock_storage(
            changes_since=hub_changes, last_sequence=5
        )
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-requester",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        response = await engine.handle_hub_sync(request)

        # Should only return the change NOT from dev-requester
        assert all(c.device_id != "dev-requester" for c in response.changes)
        assert len(response.changes) == 1
        assert response.changes[0].entity_id == "n-2"

    async def test_handle_hub_sync_records_incoming(self) -> None:
        """Hub records each incoming change from the requesting device."""
        storage = _make_mock_storage(changes_since=[], last_sequence=0)
        engine = SyncEngine(storage, device_id="dev-hub")

        incoming_changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-new",
                operation="insert",
                device_id="dev-requester",
                changed_at="2026-01-15T10:00:00",
                payload={"content": "hello"},
            )
        ]
        request = SyncRequest(
            device_id="dev-requester",
            brain_id="brain-test",
            last_sequence=0,
            changes=incoming_changes,
        )

        await engine.handle_hub_sync(request)

        # record_change must have been called for the incoming change
        storage.record_change.assert_called()

    async def test_handle_hub_sync_updates_device_watermark(self) -> None:
        """Hub calls update_device_sync to advance requester's watermark."""
        storage = _make_mock_storage(changes_since=[], last_sequence=15)
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-requester",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        await engine.handle_hub_sync(request)

        storage.update_device_sync.assert_called_once_with("dev-requester", 15)

    async def test_handle_hub_sync_returns_sync_response(self) -> None:
        """handle_hub_sync returns a SyncResponse instance."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-requester",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        response = await engine.handle_hub_sync(request)
        assert isinstance(response, SyncResponse)

    async def test_handle_hub_sync_success_status(self) -> None:
        """handle_hub_sync response has SUCCESS status on clean sync."""
        storage = _make_mock_storage()
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-requester",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        response = await engine.handle_hub_sync(request)
        assert response.status == SyncStatus.SUCCESS

    async def test_handle_hub_sync_hub_sequence_from_stats(self) -> None:
        """hub_sequence in response comes from get_change_log_stats."""
        storage = _make_mock_storage(changes_since=[], last_sequence=99)
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-req",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        response = await engine.handle_hub_sync(request)
        assert response.hub_sequence == 99

    async def test_handle_hub_sync_empty_request_no_records(self) -> None:
        """Hub with no incoming changes does not call record_change."""
        storage = _make_mock_storage(changes_since=[], last_sequence=0)
        engine = SyncEngine(storage, device_id="dev-hub")

        request = SyncRequest(
            device_id="dev-req",
            brain_id="brain-test",
            last_sequence=0,
            changes=[],
        )

        await engine.handle_hub_sync(request)

        storage.record_change.assert_not_called()
