"""Brain versioning engine — snapshot-based version control for brains.

Creates point-in-time snapshots of brain state, supports rollback
and diffing between versions. Snapshots are JSON blobs stored in
SQLite, reusing the existing export_brain()/import_brain() machinery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainSnapshot
    from neural_memory.storage.base import NeuralStorage


# ── Data structures ──────────────────────────────────────────────


@dataclass(frozen=True)
class BrainVersion:
    """A point-in-time snapshot reference for a brain."""

    id: str
    brain_id: str
    version_name: str
    version_number: int
    description: str
    neuron_count: int
    synapse_count: int
    fiber_count: int
    snapshot_hash: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionDiff:
    """Diff between two brain versions."""

    from_version: str
    to_version: str
    neurons_added: tuple[str, ...]
    neurons_removed: tuple[str, ...]
    neurons_modified: tuple[str, ...]
    synapses_added: tuple[str, ...]
    synapses_removed: tuple[str, ...]
    synapses_weight_changed: tuple[tuple[str, float, float], ...]
    fibers_added: tuple[str, ...]
    fibers_removed: tuple[str, ...]
    summary: str


# ── Helpers ──────────────────────────────────────────────────────


def _snapshot_to_json(snapshot: BrainSnapshot) -> str:
    """Serialize a BrainSnapshot to JSON string."""
    data = {
        "brain_id": snapshot.brain_id,
        "brain_name": snapshot.brain_name,
        "exported_at": snapshot.exported_at.isoformat(),
        "version": snapshot.version,
        "neurons": snapshot.neurons,
        "synapses": snapshot.synapses,
        "fibers": snapshot.fibers,
        "config": snapshot.config,
        "metadata": snapshot.metadata,
    }
    return json.dumps(data, sort_keys=True, default=str)


def _json_to_snapshot(json_str: str) -> BrainSnapshot:
    """Deserialize a JSON string to BrainSnapshot.

    Raises:
        ValueError: If required fields are missing from the JSON data.
    """
    from neural_memory.core.brain import BrainSnapshot

    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Invalid snapshot: expected a JSON object")

    # Validate required fields
    required_fields = (
        "brain_id",
        "brain_name",
        "exported_at",
        "version",
        "neurons",
        "synapses",
        "fibers",
        "config",
    )
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Invalid snapshot: missing fields {missing}")

    # Validate types of list fields
    for list_field in ("neurons", "synapses", "fibers"):
        if not isinstance(data[list_field], list):
            raise ValueError(f"Invalid snapshot: '{list_field}' must be a list")

    return BrainSnapshot(
        brain_id=data["brain_id"],
        brain_name=data["brain_name"],
        exported_at=datetime.fromisoformat(data["exported_at"]),
        version=data["version"],
        neurons=data["neurons"],
        synapses=data["synapses"],
        fibers=data["fibers"],
        config=data["config"],
        metadata=data.get("metadata", {}),
    )


def _compute_hash(snapshot_json: str) -> str:
    """Compute SHA-256 hash of serialized snapshot."""
    return hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()


def _compute_diff(
    from_snapshot: BrainSnapshot,
    to_snapshot: BrainSnapshot,
    from_version_id: str = "",
    to_version_id: str = "",
) -> VersionDiff:
    """Compute diff between two snapshots.

    Compares neurons by ID, synapses by ID, fibers by ID.

    Args:
        from_snapshot: Source snapshot to compare from.
        to_snapshot: Target snapshot to compare to.
        from_version_id: Version ID for the source snapshot.
        to_version_id: Version ID for the target snapshot.
    """
    # Neuron diff
    from_neurons = {n["id"]: n for n in from_snapshot.neurons}
    to_neurons = {n["id"]: n for n in to_snapshot.neurons}

    from_neuron_ids = set(from_neurons.keys())
    to_neuron_ids = set(to_neurons.keys())

    neurons_added = tuple(sorted(to_neuron_ids - from_neuron_ids))
    neurons_removed = tuple(sorted(from_neuron_ids - to_neuron_ids))

    neurons_modified: list[str] = []
    for nid in from_neuron_ids & to_neuron_ids:
        if from_neurons[nid].get("content") != to_neurons[nid].get("content") or from_neurons[
            nid
        ].get("type") != to_neurons[nid].get("type"):
            neurons_modified.append(nid)
    neurons_modified_tuple = tuple(sorted(neurons_modified))

    # Synapse diff
    from_synapses = {s["id"]: s for s in from_snapshot.synapses}
    to_synapses = {s["id"]: s for s in to_snapshot.synapses}

    from_synapse_ids = set(from_synapses.keys())
    to_synapse_ids = set(to_synapses.keys())

    synapses_added = tuple(sorted(to_synapse_ids - from_synapse_ids))
    synapses_removed = tuple(sorted(from_synapse_ids - to_synapse_ids))

    weight_changes: list[tuple[str, float, float]] = []
    for sid in from_synapse_ids & to_synapse_ids:
        old_w = from_synapses[sid].get("weight", 0.0)
        new_w = to_synapses[sid].get("weight", 0.0)
        if abs(old_w - new_w) > 1e-6:
            weight_changes.append((sid, old_w, new_w))
    synapses_weight_changed = tuple(sorted(weight_changes, key=lambda x: x[0]))

    # Fiber diff
    from_fibers = {f["id"] for f in from_snapshot.fibers}
    to_fibers = {f["id"] for f in to_snapshot.fibers}

    fibers_added = tuple(sorted(to_fibers - from_fibers))
    fibers_removed = tuple(sorted(from_fibers - to_fibers))

    # Summary
    parts: list[str] = []
    if neurons_added:
        parts.append(f"+{len(neurons_added)} neurons")
    if neurons_removed:
        parts.append(f"-{len(neurons_removed)} neurons")
    if neurons_modified_tuple:
        parts.append(f"~{len(neurons_modified_tuple)} neurons modified")
    if synapses_added:
        parts.append(f"+{len(synapses_added)} synapses")
    if synapses_removed:
        parts.append(f"-{len(synapses_removed)} synapses")
    if synapses_weight_changed:
        parts.append(f"~{len(synapses_weight_changed)} synapse weights changed")
    if fibers_added:
        parts.append(f"+{len(fibers_added)} fibers")
    if fibers_removed:
        parts.append(f"-{len(fibers_removed)} fibers")
    summary = ", ".join(parts) if parts else "No changes"

    return VersionDiff(
        from_version=from_version_id,
        to_version=to_version_id,
        neurons_added=neurons_added,
        neurons_removed=neurons_removed,
        neurons_modified=neurons_modified_tuple,
        synapses_added=synapses_added,
        synapses_removed=synapses_removed,
        synapses_weight_changed=synapses_weight_changed,
        fibers_added=fibers_added,
        fibers_removed=fibers_removed,
        summary=summary,
    )


# ── Versioning engine ────────────────────────────────────────────


class VersioningEngine:
    """Brain version control engine.

    Creates snapshots of brain state, supports rollback and diff.
    Delegates storage to the underlying NeuralStorage implementation.
    """

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def create_version(
        self,
        brain_id: str,
        version_name: str,
        description: str = "",
    ) -> BrainVersion:
        """Create a new version snapshot of the current brain state.

        Args:
            brain_id: ID of the brain to snapshot
            version_name: User-provided name (must be unique per brain)
            description: Optional description

        Returns:
            The created BrainVersion

        Raises:
            ValueError: If version_name already exists for this brain
        """
        # Check name uniqueness — fetch ALL versions with a reasonable cap
        existing = await self._storage.list_versions(brain_id, limit=10000)
        for v in existing:
            if v.version_name == version_name:
                raise ValueError(
                    f"Version name '{version_name}' already exists for brain {brain_id}"
                )

        # Export current state
        snapshot = await self._storage.export_brain(brain_id)
        snapshot_json = _snapshot_to_json(snapshot)
        snapshot_hash = _compute_hash(snapshot_json)

        # Get next version number
        version_number = await self._storage.get_next_version_number(brain_id)

        version = BrainVersion(
            id=str(uuid4()),
            brain_id=brain_id,
            version_name=version_name,
            version_number=version_number,
            description=description,
            neuron_count=len(snapshot.neurons),
            synapse_count=len(snapshot.synapses),
            fiber_count=len(snapshot.fibers),
            snapshot_hash=snapshot_hash,
            created_at=utcnow(),
        )

        await self._storage.save_version(brain_id, version, snapshot_json)
        return version

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        """List versions for a brain, most recent first.

        Args:
            brain_id: Brain ID
            limit: Maximum versions to return

        Returns:
            List of BrainVersion, newest first
        """
        return await self._storage.list_versions(brain_id, limit=limit)

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> BrainVersion | None:
        """Get a specific version by ID.

        Args:
            brain_id: Brain ID
            version_id: Version ID

        Returns:
            BrainVersion if found, None otherwise
        """
        result = await self._storage.get_version(brain_id, version_id)
        if result is None:
            return None
        return result[0]

    async def rollback(
        self,
        brain_id: str,
        version_id: str,
    ) -> BrainVersion:
        """Rollback brain to a previous version.

        Creates a new version entry named 'rollback-to-{original_name}'
        before restoring the old state.

        Args:
            brain_id: Brain ID
            version_id: Version to rollback to

        Returns:
            The new BrainVersion created for the rollback

        Raises:
            ValueError: If version_id not found
        """
        result = await self._storage.get_version(brain_id, version_id)
        if result is None:
            raise ValueError(f"Version {version_id} not found for brain {brain_id}")

        target_version, snapshot_json = result
        snapshot = _json_to_snapshot(snapshot_json)

        # Create a rollback version entry for the current state first
        rollback_name = f"rollback-to-{target_version.version_name}"
        # Ensure unique name — fetch ALL versions with a reasonable cap
        existing_names = {
            v.version_name for v in await self._storage.list_versions(brain_id, limit=10000)
        }
        if rollback_name in existing_names:
            suffix = 1
            while f"{rollback_name}-{suffix}" in existing_names:
                suffix += 1
            rollback_name = f"{rollback_name}-{suffix}"

        # Clear and reimport the target state
        await self._storage.clear(brain_id)
        # Re-create the brain entry
        brain_data = snapshot.config
        from neural_memory.core.brain import Brain, BrainConfig

        config = BrainConfig(**brain_data) if brain_data else BrainConfig()
        brain = Brain.create(
            name=snapshot.brain_name,
            config=config,
            brain_id=brain_id,
        )
        await self._storage.save_brain(brain)
        await self._storage.import_brain(snapshot, brain_id)

        # Now create the rollback version
        version_number = await self._storage.get_next_version_number(brain_id)
        snapshot_hash = _compute_hash(snapshot_json)

        rollback_version = BrainVersion(
            id=str(uuid4()),
            brain_id=brain_id,
            version_name=rollback_name,
            version_number=version_number,
            description=f"Rollback to version '{target_version.version_name}'",
            neuron_count=len(snapshot.neurons),
            synapse_count=len(snapshot.synapses),
            fiber_count=len(snapshot.fibers),
            snapshot_hash=snapshot_hash,
            created_at=utcnow(),
            metadata={"rollback_from": version_id},
        )

        await self._storage.save_version(brain_id, rollback_version, snapshot_json)
        return rollback_version

    async def diff(
        self,
        brain_id: str,
        from_version_id: str,
        to_version_id: str,
    ) -> VersionDiff:
        """Compute diff between two versions.

        Args:
            brain_id: Brain ID
            from_version_id: Source version ID
            to_version_id: Target version ID

        Returns:
            VersionDiff with changes between versions

        Raises:
            ValueError: If either version not found
        """
        from_result = await self._storage.get_version(brain_id, from_version_id)
        if from_result is None:
            raise ValueError(f"Version {from_version_id} not found")

        to_result = await self._storage.get_version(brain_id, to_version_id)
        if to_result is None:
            raise ValueError(f"Version {to_version_id} not found")

        from_version, from_json = from_result
        to_version, to_json = to_result

        from_snapshot = _json_to_snapshot(from_json)
        to_snapshot = _json_to_snapshot(to_json)

        return _compute_diff(
            from_snapshot,
            to_snapshot,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
        )
