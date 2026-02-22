"""Tests for sync protocol dataclasses and enums."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncConflict,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)


# ── SyncChange ────────────────────────────────────────────────────────────────


class TestSyncChange:
    """Tests for the SyncChange frozen dataclass."""

    def test_sync_change_frozen(self) -> None:
        """SyncChange is immutable — mutation raises an error."""
        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev-a",
            changed_at="2026-01-01T00:00:00",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            change.entity_id = "n-mutated"  # type: ignore[misc]

    def test_sync_change_default_payload_empty(self) -> None:
        """payload defaults to an empty dict."""
        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev-a",
            changed_at="2026-01-01T00:00:00",
        )
        assert change.payload == {}

    def test_sync_change_custom_payload(self) -> None:
        """payload is preserved when explicitly set."""
        payload = {"weight": 0.9, "content": "hello"}
        change = SyncChange(
            sequence=5,
            entity_type="fiber",
            entity_id="f-99",
            operation="update",
            device_id="dev-b",
            changed_at="2026-02-01T12:00:00",
            payload=payload,
        )
        assert change.payload == payload

    def test_sync_change_all_fields(self) -> None:
        """All required fields are accessible."""
        change = SyncChange(
            sequence=42,
            entity_type="synapse",
            entity_id="s-7",
            operation="delete",
            device_id="dev-c",
            changed_at="2026-03-01T08:30:00",
        )
        assert change.sequence == 42
        assert change.entity_type == "synapse"
        assert change.entity_id == "s-7"
        assert change.operation == "delete"
        assert change.device_id == "dev-c"
        assert change.changed_at == "2026-03-01T08:30:00"

    def test_sync_change_equality(self) -> None:
        """Two SyncChange instances with identical fields are equal."""
        change1 = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at="2026-01-01T00:00:00",
        )
        change2 = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at="2026-01-01T00:00:00",
        )
        assert change1 == change2


# ── SyncRequest ───────────────────────────────────────────────────────────────


class TestSyncRequest:
    """Tests for the SyncRequest frozen dataclass."""

    def test_sync_request_defaults(self) -> None:
        """Default strategy is PREFER_RECENT and changes is empty list."""
        request = SyncRequest(
            device_id="dev-x",
            brain_id="brain-1",
            last_sequence=0,
        )
        assert request.strategy == ConflictStrategy.PREFER_RECENT
        assert request.changes == []

    def test_sync_request_frozen(self) -> None:
        """SyncRequest is immutable."""
        request = SyncRequest(
            device_id="dev-x",
            brain_id="brain-1",
            last_sequence=0,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            request.brain_id = "mutated"  # type: ignore[misc]

    def test_sync_request_custom_strategy(self) -> None:
        """Custom strategy is preserved."""
        request = SyncRequest(
            device_id="dev-y",
            brain_id="brain-2",
            last_sequence=10,
            strategy=ConflictStrategy.PREFER_LOCAL,
        )
        assert request.strategy == ConflictStrategy.PREFER_LOCAL

    def test_sync_request_with_changes(self) -> None:
        """Changes list is stored on the request."""
        changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-1",
                operation="insert",
                device_id="dev",
                changed_at="2026-01-01T00:00:00",
            )
        ]
        request = SyncRequest(
            device_id="dev",
            brain_id="brain-1",
            last_sequence=0,
            changes=changes,
        )
        assert len(request.changes) == 1

    def test_sync_request_all_fields(self) -> None:
        """All fields are accessible."""
        request = SyncRequest(
            device_id="my-dev",
            brain_id="my-brain",
            last_sequence=99,
        )
        assert request.device_id == "my-dev"
        assert request.brain_id == "my-brain"
        assert request.last_sequence == 99


# ── SyncResponse ──────────────────────────────────────────────────────────────


class TestSyncResponse:
    """Tests for the SyncResponse frozen dataclass."""

    def test_sync_response_defaults(self) -> None:
        """Default status is SUCCESS, changes and conflicts are empty lists."""
        response = SyncResponse(hub_sequence=5)
        assert response.status == SyncStatus.SUCCESS
        assert response.changes == []
        assert response.conflicts == []
        assert response.message == ""

    def test_sync_response_frozen(self) -> None:
        """SyncResponse is immutable."""
        response = SyncResponse(hub_sequence=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            response.hub_sequence = 999  # type: ignore[misc]

    def test_sync_response_custom_status(self) -> None:
        """Custom status is preserved."""
        response = SyncResponse(hub_sequence=0, status=SyncStatus.ERROR)
        assert response.status == SyncStatus.ERROR

    def test_sync_response_with_changes(self) -> None:
        """Changes list is stored correctly."""
        changes = [
            SyncChange(
                sequence=1,
                entity_type="neuron",
                entity_id="n-remote",
                operation="insert",
                device_id="dev-other",
                changed_at="2026-01-01T00:00:00",
            )
        ]
        response = SyncResponse(hub_sequence=1, changes=changes)
        assert len(response.changes) == 1

    def test_sync_response_with_message(self) -> None:
        """Message field is preserved."""
        response = SyncResponse(hub_sequence=0, message="sync complete")
        assert response.message == "sync complete"


# ── SyncConflict ──────────────────────────────────────────────────────────────


class TestSyncConflict:
    """Tests for the SyncConflict frozen dataclass."""

    def test_sync_conflict_frozen(self) -> None:
        """SyncConflict is immutable."""
        conflict = SyncConflict(
            entity_type="neuron",
            entity_id="n-1",
            local_device="dev-a",
            remote_device="dev-b",
            resolution="local_prefer_recent",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            conflict.resolution = "mutated"  # type: ignore[misc]

    def test_sync_conflict_default_details(self) -> None:
        """details defaults to empty string."""
        conflict = SyncConflict(
            entity_type="neuron",
            entity_id="n-1",
            local_device="dev-a",
            remote_device="dev-b",
            resolution="local_prefer_recent",
        )
        assert conflict.details == ""

    def test_sync_conflict_all_fields(self) -> None:
        """All fields are accessible."""
        conflict = SyncConflict(
            entity_type="fiber",
            entity_id="f-99",
            local_device="laptop",
            remote_device="server",
            resolution="remote_delete_wins",
            details="tombstone applied",
        )
        assert conflict.entity_type == "fiber"
        assert conflict.entity_id == "f-99"
        assert conflict.local_device == "laptop"
        assert conflict.remote_device == "server"
        assert conflict.resolution == "remote_delete_wins"
        assert conflict.details == "tombstone applied"


# ── ConflictStrategy ──────────────────────────────────────────────────────────


class TestConflictStrategy:
    """Tests for ConflictStrategy enum values."""

    def test_conflict_strategy_values(self) -> None:
        """All 4 strategies have valid string values."""
        assert ConflictStrategy.PREFER_RECENT == "prefer_recent"
        assert ConflictStrategy.PREFER_LOCAL == "prefer_local"
        assert ConflictStrategy.PREFER_REMOTE == "prefer_remote"
        assert ConflictStrategy.PREFER_STRONGER == "prefer_stronger"

    def test_conflict_strategy_count(self) -> None:
        """There are exactly 4 ConflictStrategy variants."""
        assert len(ConflictStrategy) == 4

    def test_conflict_strategy_is_str(self) -> None:
        """ConflictStrategy values are strings (StrEnum)."""
        for strategy in ConflictStrategy:
            assert isinstance(strategy, str)

    def test_conflict_strategy_from_string(self) -> None:
        """ConflictStrategy can be constructed from its string value."""
        assert ConflictStrategy("prefer_recent") == ConflictStrategy.PREFER_RECENT
        assert ConflictStrategy("prefer_local") == ConflictStrategy.PREFER_LOCAL
        assert ConflictStrategy("prefer_remote") == ConflictStrategy.PREFER_REMOTE
        assert ConflictStrategy("prefer_stronger") == ConflictStrategy.PREFER_STRONGER

    def test_conflict_strategy_invalid_raises(self) -> None:
        """Unknown string raises ValueError."""
        with pytest.raises(ValueError):
            ConflictStrategy("invalid_strategy")


# ── SyncStatus ────────────────────────────────────────────────────────────────


class TestSyncStatus:
    """Tests for SyncStatus enum values."""

    def test_sync_status_values(self) -> None:
        """All 4 statuses have valid string values."""
        assert SyncStatus.SUCCESS == "success"
        assert SyncStatus.PARTIAL == "partial"
        assert SyncStatus.CONFLICT == "conflict"
        assert SyncStatus.ERROR == "error"

    def test_sync_status_count(self) -> None:
        """There are exactly 4 SyncStatus variants."""
        assert len(SyncStatus) == 4

    def test_sync_status_is_str(self) -> None:
        """SyncStatus values are strings (StrEnum)."""
        for status in SyncStatus:
            assert isinstance(status, str)

    def test_sync_status_from_string(self) -> None:
        """SyncStatus can be constructed from its string value."""
        assert SyncStatus("success") == SyncStatus.SUCCESS
        assert SyncStatus("partial") == SyncStatus.PARTIAL
        assert SyncStatus("conflict") == SyncStatus.CONFLICT
        assert SyncStatus("error") == SyncStatus.ERROR

    def test_sync_status_invalid_raises(self) -> None:
        """Unknown string raises ValueError."""
        with pytest.raises(ValueError):
            SyncStatus("unknown_status")
