"""Incremental merge for multi-device sync with neural-aware conflict resolution."""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.sync.protocol import ConflictStrategy, SyncChange, SyncConflict

logger = logging.getLogger(__name__)


def resolve_entity_conflict(
    local_change: SyncChange,
    remote_change: SyncChange,
    strategy: ConflictStrategy,
) -> tuple[SyncChange, SyncConflict]:
    """Resolve a conflict between local and remote changes to the same entity.

    Neural-aware merge rules:
    - Delete wins (tombstone semantics) regardless of strategy
    - weight = max(local, remote)
    - access_frequency = sum(local, remote)
    - tags = union(local, remote)
    - conductivity = max(local, remote)
    - For other fields: use strategy to pick winner

    Returns:
        Tuple of (winning_change, conflict_record)
    """
    # Delete always wins (tombstone semantics)
    if local_change.operation == "delete":
        return local_change, SyncConflict(
            entity_type=local_change.entity_type,
            entity_id=local_change.entity_id,
            local_device=local_change.device_id,
            remote_device=remote_change.device_id,
            resolution="local_delete_wins",
        )
    if remote_change.operation == "delete":
        return remote_change, SyncConflict(
            entity_type=remote_change.entity_type,
            entity_id=remote_change.entity_id,
            local_device=local_change.device_id,
            remote_device=remote_change.device_id,
            resolution="remote_delete_wins",
        )

    # Both are insert or update — merge payloads with neural rules
    merged_payload = _merge_payloads(
        local_change.payload, remote_change.payload, local_change.entity_type
    )

    # Determine winner for base fields
    winner = _pick_winner(local_change, remote_change, strategy)
    resolution = f"{'local' if winner is local_change else 'remote'}_{strategy.value}"

    from dataclasses import replace

    merged = replace(winner, payload=merged_payload)

    conflict = SyncConflict(
        entity_type=local_change.entity_type,
        entity_id=local_change.entity_id,
        local_device=local_change.device_id,
        remote_device=remote_change.device_id,
        resolution=resolution,
    )

    return merged, conflict


def _pick_winner(
    local: SyncChange,
    remote: SyncChange,
    strategy: ConflictStrategy,
) -> SyncChange:
    """Pick the winning change based on strategy."""
    if strategy == ConflictStrategy.PREFER_LOCAL:
        return local
    if strategy == ConflictStrategy.PREFER_REMOTE:
        return remote
    if strategy == ConflictStrategy.PREFER_RECENT:
        # Compare timestamps — lexicographic ISO comparison is safe for naive UTC
        if local.changed_at >= remote.changed_at:
            return local
        return remote
    if strategy == ConflictStrategy.PREFER_STRONGER:
        # Compare neural metrics (weight, salience, conductivity)
        local_strength = _compute_strength(local.payload)
        remote_strength = _compute_strength(remote.payload)
        if local_strength >= remote_strength:
            return local
        return remote
    return local  # fallback


def _compute_strength(payload: dict[str, Any]) -> float:
    """Compute a strength score from neural metrics in payload."""
    score = 0.0
    score += float(payload.get("weight", 0.0))
    score += float(payload.get("salience", 0.0))
    score += float(payload.get("conductivity", 0.0))
    score += float(payload.get("activation_level", 0.0))
    return score


def _merge_payloads(
    local: dict[str, Any],
    remote: dict[str, Any],
    entity_type: str,
) -> dict[str, Any]:
    """Merge two payloads using neural-aware rules.

    Rules:
    - weight: max
    - access_frequency: sum
    - reinforced_count: sum
    - tags: union
    - auto_tags: union
    - agent_tags: union
    - salience: max
    - conductivity: max
    - activation_level: max
    - frequency: sum
    - Other fields: use the non-empty value, prefer local if both exist
    """
    merged: dict[str, Any] = {**local}  # Start with local as base

    # Numeric max fields
    for key in ("weight", "salience", "conductivity", "activation_level"):
        if key in local and key in remote:
            merged[key] = max(float(local[key]), float(remote[key]))
        elif key in remote:
            merged[key] = remote[key]

    # Numeric sum fields
    for key in ("access_frequency", "reinforced_count", "frequency"):
        if key in local and key in remote:
            merged[key] = int(local[key]) + int(remote[key])
        elif key in remote:
            merged[key] = remote[key]

    # Set union fields (stored as JSON arrays)
    for key in ("tags", "auto_tags", "agent_tags"):
        local_val = local.get(key)
        remote_val = remote.get(key)
        local_set = set(local_val if isinstance(local_val, list) else [])
        remote_set = set(remote_val if isinstance(remote_val, list) else [])
        if local_set or remote_set:
            merged[key] = sorted(local_set | remote_set)

    # Fill missing fields from remote
    for key, value in remote.items():
        if key not in merged:
            merged[key] = value

    return merged


def merge_change_lists(
    local_changes: list[SyncChange],
    remote_changes: list[SyncChange],
    strategy: ConflictStrategy,
) -> tuple[list[SyncChange], list[SyncConflict]]:
    """Merge two lists of changes, resolving conflicts.

    Returns:
        Tuple of (merged_changes, conflicts)
    """
    # Index remote changes by (entity_type, entity_id) for conflict detection
    remote_by_entity: dict[tuple[str, str], SyncChange] = {}
    for change in remote_changes:
        key = (change.entity_type, change.entity_id)
        # Keep latest remote change per entity
        if key not in remote_by_entity or change.changed_at > remote_by_entity[key].changed_at:
            remote_by_entity[key] = change

    merged: list[SyncChange] = []
    conflicts: list[SyncConflict] = []
    seen_entities: set[tuple[str, str]] = set()

    for local_change in local_changes:
        key = (local_change.entity_type, local_change.entity_id)
        seen_entities.add(key)

        if key in remote_by_entity:
            # Conflict — same entity modified on both sides
            winner, conflict = resolve_entity_conflict(
                local_change, remote_by_entity[key], strategy
            )
            merged.append(winner)
            conflicts.append(conflict)
        else:
            # No conflict — local change only
            merged.append(local_change)

    # Add remote-only changes (no conflict)
    for key, remote_change in remote_by_entity.items():
        if key not in seen_entities:
            merged.append(remote_change)

    return merged, conflicts
