"""Tests for incremental sync merge logic."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.sync.incremental_merge import (
    _compute_strength,
    _merge_payloads,
    _pick_winner,
    merge_change_lists,
    resolve_entity_conflict,
)
from neural_memory.sync.protocol import ConflictStrategy, SyncChange, SyncConflict


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_change(
    entity_id: str = "n-1",
    entity_type: str = "neuron",
    operation: str = "update",
    device_id: str = "dev-local",
    changed_at: str = "2026-01-15T10:00:00",
    payload: dict | None = None,
    sequence: int = 1,
) -> SyncChange:
    return SyncChange(
        sequence=sequence,
        entity_type=entity_type,
        entity_id=entity_id,
        operation=operation,
        device_id=device_id,
        changed_at=changed_at,
        payload=payload or {},
    )


# ── Delete semantics ──────────────────────────────────────────────────────────


class TestDeleteWins:
    """Delete operations win regardless of strategy."""

    def test_delete_wins_local(self) -> None:
        """Local delete beats remote update."""
        local = _make_change(operation="delete", device_id="dev-local")
        remote = _make_change(operation="update", device_id="dev-remote")

        winner, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_RECENT)

        assert winner is local
        assert conflict.resolution == "local_delete_wins"

    def test_delete_wins_remote(self) -> None:
        """Remote delete beats local update."""
        local = _make_change(operation="update", device_id="dev-local")
        remote = _make_change(operation="delete", device_id="dev-remote")

        winner, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_RECENT)

        assert winner is remote
        assert conflict.resolution == "remote_delete_wins"

    def test_delete_wins_over_insert(self) -> None:
        """Delete beats insert regardless of side."""
        local = _make_change(operation="delete", device_id="dev-local")
        remote = _make_change(operation="insert", device_id="dev-remote")

        winner, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_REMOTE)

        assert winner is local
        assert conflict.resolution == "local_delete_wins"

    def test_conflict_record_has_correct_devices(self) -> None:
        """SyncConflict records local and remote device IDs."""
        local = _make_change(operation="delete", device_id="laptop")
        remote = _make_change(operation="update", device_id="desktop")

        _, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_LOCAL)

        assert conflict.local_device == "laptop"
        assert conflict.remote_device == "desktop"


# ── Strategy: PREFER_RECENT ───────────────────────────────────────────────────


class TestPreferRecent:
    """PREFER_RECENT picks the change with the later timestamp."""

    def test_prefer_recent_picks_local_when_newer(self) -> None:
        """Local wins when local timestamp > remote timestamp."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            changed_at="2026-01-15T12:00:00",
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            changed_at="2026-01-15T10:00:00",
        )

        winner, conflict = resolve_entity_conflict(
            local, remote, ConflictStrategy.PREFER_RECENT
        )

        assert winner.device_id == "dev-local"
        assert "prefer_recent" in conflict.resolution

    def test_prefer_recent_picks_remote_when_newer(self) -> None:
        """Remote wins when remote timestamp > local timestamp."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            changed_at="2026-01-01T00:00:00",
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            changed_at="2026-01-15T23:59:59",
        )

        winner, conflict = resolve_entity_conflict(
            local, remote, ConflictStrategy.PREFER_RECENT
        )

        assert winner.device_id == "dev-remote"
        assert "prefer_recent" in conflict.resolution

    def test_prefer_recent_local_wins_on_tie(self) -> None:
        """When timestamps are equal, local wins (>= comparison)."""
        ts = "2026-01-15T10:00:00"
        local = _make_change(operation="update", device_id="local", changed_at=ts)
        remote = _make_change(operation="update", device_id="remote", changed_at=ts)

        winner, _ = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_RECENT)
        assert winner.device_id == "local"


# ── Strategy: PREFER_LOCAL ────────────────────────────────────────────────────


class TestPreferLocal:
    """PREFER_LOCAL always picks the local change."""

    def test_prefer_local_always_wins(self) -> None:
        """Local change wins regardless of timestamp."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            changed_at="2026-01-01T00:00:00",  # older
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            changed_at="2026-01-15T12:00:00",  # newer
        )

        winner, conflict = resolve_entity_conflict(
            local, remote, ConflictStrategy.PREFER_LOCAL
        )

        assert winner.device_id == "dev-local"
        assert "prefer_local" in conflict.resolution

    def test_prefer_local_resolution_label(self) -> None:
        """Resolution string starts with 'local_'."""
        local = _make_change(operation="update", device_id="l")
        remote = _make_change(operation="update", device_id="r")

        _, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_LOCAL)
        assert conflict.resolution.startswith("local_")


# ── Strategy: PREFER_REMOTE ───────────────────────────────────────────────────


class TestPreferRemote:
    """PREFER_REMOTE always picks the remote change."""

    def test_prefer_remote_always_wins(self) -> None:
        """Remote change wins regardless of timestamp."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            changed_at="2026-01-15T12:00:00",  # newer
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            changed_at="2026-01-01T00:00:00",  # older
        )

        winner, conflict = resolve_entity_conflict(
            local, remote, ConflictStrategy.PREFER_REMOTE
        )

        assert winner.device_id == "dev-remote"
        assert "prefer_remote" in conflict.resolution

    def test_prefer_remote_resolution_label(self) -> None:
        """Resolution string starts with 'remote_'."""
        local = _make_change(operation="update", device_id="l")
        remote = _make_change(operation="update", device_id="r")

        _, conflict = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_REMOTE)
        assert conflict.resolution.startswith("remote_")


# ── Strategy: PREFER_STRONGER ─────────────────────────────────────────────────


class TestPreferStronger:
    """PREFER_STRONGER picks the change with higher neural strength score."""

    def test_prefer_stronger_local_wins(self) -> None:
        """Local wins when it has higher combined neural metrics."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            payload={"weight": 0.9, "salience": 0.8, "conductivity": 0.7, "activation_level": 0.6},
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            payload={"weight": 0.1, "salience": 0.1, "conductivity": 0.1, "activation_level": 0.1},
        )

        winner, conflict = resolve_entity_conflict(
            local, remote, ConflictStrategy.PREFER_STRONGER
        )

        assert winner.device_id == "dev-local"
        assert "prefer_stronger" in conflict.resolution

    def test_prefer_stronger_remote_wins(self) -> None:
        """Remote wins when it has higher combined neural metrics."""
        local = _make_change(
            operation="update",
            device_id="dev-local",
            payload={"weight": 0.1, "salience": 0.2},
        )
        remote = _make_change(
            operation="update",
            device_id="dev-remote",
            payload={"weight": 0.9, "salience": 0.8, "conductivity": 0.5},
        )

        winner, _ = resolve_entity_conflict(local, remote, ConflictStrategy.PREFER_STRONGER)

        assert winner.device_id == "dev-remote"


# ── _merge_payloads ───────────────────────────────────────────────────────────


class TestMergePayloads:
    """Test _merge_payloads applies neural-aware merge rules."""

    def test_merge_payloads_max_fields(self) -> None:
        """weight, salience, conductivity, activation_level take the max."""
        local = {"weight": 0.3, "salience": 0.7, "conductivity": 0.5, "activation_level": 0.2}
        remote = {"weight": 0.8, "salience": 0.2, "conductivity": 0.9, "activation_level": 0.6}

        merged = _merge_payloads(local, remote, "neuron")

        assert merged["weight"] == 0.8          # max(0.3, 0.8)
        assert merged["salience"] == 0.7        # max(0.7, 0.2)
        assert merged["conductivity"] == 0.9    # max(0.5, 0.9)
        assert merged["activation_level"] == 0.6  # max(0.2, 0.6)

    def test_merge_payloads_sum_fields(self) -> None:
        """access_frequency and reinforced_count are summed."""
        local = {"access_frequency": 5, "reinforced_count": 3, "frequency": 10}
        remote = {"access_frequency": 7, "reinforced_count": 2, "frequency": 4}

        merged = _merge_payloads(local, remote, "neuron")

        assert merged["access_frequency"] == 12  # 5 + 7
        assert merged["reinforced_count"] == 5   # 3 + 2
        assert merged["frequency"] == 14          # 10 + 4

    def test_merge_payloads_union_tags(self) -> None:
        """tags, auto_tags, agent_tags are unioned and sorted."""
        local = {"tags": ["b", "a"], "auto_tags": ["x"], "agent_tags": ["m"]}
        remote = {"tags": ["c", "a"], "auto_tags": ["y"], "agent_tags": ["n"]}

        merged = _merge_payloads(local, remote, "fiber")

        assert merged["tags"] == ["a", "b", "c"]
        assert merged["auto_tags"] == ["x", "y"]
        assert merged["agent_tags"] == ["m", "n"]

    def test_merge_payloads_union_tags_deduplicates(self) -> None:
        """Tags appearing on both sides appear only once in the union."""
        local = {"tags": ["alpha", "shared"]}
        remote = {"tags": ["shared", "beta"]}

        merged = _merge_payloads(local, remote, "fiber")

        assert merged["tags"].count("shared") == 1
        assert set(merged["tags"]) == {"alpha", "shared", "beta"}

    def test_merge_payloads_fill_missing(self) -> None:
        """Fields present only in remote are filled in."""
        local = {"weight": 0.5}
        remote = {"weight": 0.3, "content": "hello", "extra_field": 42}

        merged = _merge_payloads(local, remote, "neuron")

        # weight takes max → local wins (0.5 > 0.3)
        assert merged["weight"] == 0.5
        # extra fields from remote fill in
        assert merged["content"] == "hello"
        assert merged["extra_field"] == 42

    def test_merge_payloads_local_wins_for_other_fields(self) -> None:
        """For fields not in any special category, local value is preserved."""
        local = {"description": "local desc"}
        remote = {"description": "remote desc"}

        merged = _merge_payloads(local, remote, "neuron")

        # Local is the base — local wins for unspecialized fields
        assert merged["description"] == "local desc"

    def test_merge_payloads_remote_only_max_field(self) -> None:
        """If local doesn't have the field but remote does, remote fills in."""
        local: dict = {}
        remote = {"weight": 0.9}

        merged = _merge_payloads(local, remote, "neuron")

        assert merged["weight"] == 0.9


# ── merge_change_lists ────────────────────────────────────────────────────────


class TestMergeChangeLists:
    """Test merge_change_lists produces correct merged output and conflicts."""

    def test_merge_change_lists_no_conflict(self) -> None:
        """Disjoint local and remote changes merge without conflicts."""
        local = [
            _make_change(entity_id="n-1", device_id="dev-local", sequence=1),
        ]
        remote = [
            _make_change(entity_id="n-2", device_id="dev-remote", sequence=2),
        ]

        merged, conflicts = merge_change_lists(local, remote, ConflictStrategy.PREFER_RECENT)

        entity_ids = {c.entity_id for c in merged}
        assert "n-1" in entity_ids
        assert "n-2" in entity_ids
        assert conflicts == []

    def test_merge_change_lists_with_conflict(self) -> None:
        """Same entity on both sides produces one conflict record."""
        local = [_make_change(entity_id="n-shared", device_id="dev-local", sequence=1)]
        remote = [_make_change(entity_id="n-shared", device_id="dev-remote", sequence=2)]

        merged, conflicts = merge_change_lists(local, remote, ConflictStrategy.PREFER_LOCAL)

        assert len(merged) == 1
        assert len(conflicts) == 1
        assert conflicts[0].entity_id == "n-shared"

    def test_merge_change_lists_multiple_conflicts(self) -> None:
        """Multiple overlapping entities produce multiple conflict records."""
        local = [
            _make_change(entity_id="n-1", device_id="local", sequence=1),
            _make_change(entity_id="n-2", device_id="local", sequence=2),
        ]
        remote = [
            _make_change(entity_id="n-1", device_id="remote", sequence=3),
            _make_change(entity_id="n-2", device_id="remote", sequence=4),
        ]

        _, conflicts = merge_change_lists(local, remote, ConflictStrategy.PREFER_RECENT)

        assert len(conflicts) == 2

    def test_merge_change_lists_empty_inputs(self) -> None:
        """Both empty inputs → empty merged, no conflicts."""
        merged, conflicts = merge_change_lists([], [], ConflictStrategy.PREFER_RECENT)
        assert merged == []
        assert conflicts == []

    def test_merge_change_lists_only_local(self) -> None:
        """Only local changes, no remote — all come through as-is."""
        local = [_make_change(entity_id=f"n-{i}", sequence=i) for i in range(3)]
        merged, conflicts = merge_change_lists(local, [], ConflictStrategy.PREFER_RECENT)
        assert len(merged) == 3
        assert conflicts == []

    def test_merge_change_lists_only_remote(self) -> None:
        """Only remote changes, no local — all come through as-is."""
        remote = [_make_change(entity_id=f"n-{i}", sequence=i) for i in range(3)]
        merged, conflicts = merge_change_lists([], remote, ConflictStrategy.PREFER_RECENT)
        assert len(merged) == 3
        assert conflicts == []

    def test_merge_change_lists_returns_tuples(self) -> None:
        """Return type is a 2-tuple of (list[SyncChange], list[SyncConflict])."""
        result = merge_change_lists([], [], ConflictStrategy.PREFER_RECENT)
        merged, conflicts = result
        assert isinstance(merged, list)
        assert isinstance(conflicts, list)


# ── _compute_strength ─────────────────────────────────────────────────────────


class TestComputeStrength:
    """Test _compute_strength sums the four neural metrics."""

    def test_compute_strength_all_fields(self) -> None:
        """Sum of weight + salience + conductivity + activation_level."""
        payload = {
            "weight": 0.4,
            "salience": 0.3,
            "conductivity": 0.2,
            "activation_level": 0.1,
        }
        assert _compute_strength(payload) == pytest.approx(1.0)

    def test_compute_strength_empty_payload(self) -> None:
        """Empty payload → strength is 0.0."""
        assert _compute_strength({}) == pytest.approx(0.0)

    def test_compute_strength_partial_fields(self) -> None:
        """Missing fields default to 0.0."""
        payload = {"weight": 0.5, "salience": 0.5}
        assert _compute_strength(payload) == pytest.approx(1.0)

    def test_compute_strength_returns_float(self) -> None:
        """Return type is float."""
        result = _compute_strength({"weight": 1.0})
        assert isinstance(result, float)


# ── SyncChange immutability ───────────────────────────────────────────────────


class TestSyncChangeImmutable:
    """Test SyncChange is a frozen dataclass."""

    def test_sync_change_frozen(self) -> None:
        """SyncChange fields cannot be mutated after creation."""
        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at="2026-01-01T00:00:00",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            change.entity_id = "n-2"  # type: ignore[misc]

    def test_sync_change_default_payload_empty(self) -> None:
        """SyncChange payload defaults to empty dict."""
        change = SyncChange(
            sequence=1,
            entity_type="neuron",
            entity_id="n-1",
            operation="insert",
            device_id="dev",
            changed_at="2026-01-01T00:00:00",
        )
        assert change.payload == {}
