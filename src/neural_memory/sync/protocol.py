"""Sync protocol data structures for incremental multi-device sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class SyncStatus(StrEnum):
    """Sync operation status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ERROR = "error"


class ConflictStrategy(StrEnum):
    """How to resolve sync conflicts."""

    PREFER_RECENT = "prefer_recent"
    PREFER_LOCAL = "prefer_local"
    PREFER_REMOTE = "prefer_remote"
    PREFER_STRONGER = "prefer_stronger"


@dataclass(frozen=True)
class SyncChange:
    """A single change to be synced."""

    sequence: int
    entity_type: str  # "neuron", "synapse", "fiber"
    entity_id: str
    operation: str  # "insert", "update", "delete"
    device_id: str
    changed_at: str  # ISO format
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncRequest:
    """Request from a device to sync changes."""

    device_id: str
    brain_id: str
    last_sequence: int  # Last known sequence from hub
    changes: list[SyncChange] = field(default_factory=list)
    strategy: ConflictStrategy = ConflictStrategy.PREFER_RECENT


@dataclass(frozen=True)
class SyncConflict:
    """A conflict detected during sync."""

    entity_type: str
    entity_id: str
    local_device: str
    remote_device: str
    resolution: str  # which side won
    details: str = ""


@dataclass(frozen=True)
class SyncResponse:
    """Response from hub after sync."""

    hub_sequence: int  # Current hub sequence after sync
    changes: list[SyncChange] = field(default_factory=list)  # Changes for the requesting device
    conflicts: list[SyncConflict] = field(default_factory=list)
    status: SyncStatus = SyncStatus.SUCCESS
    message: str = ""
