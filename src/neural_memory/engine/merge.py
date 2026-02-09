"""Brain merge engine — conflict resolution for brain snapshots.

Provides pure-function merge of two BrainSnapshots with configurable
conflict resolution strategies and provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from neural_memory.utils.timeutils import utcnow


class ConflictStrategy(StrEnum):
    """Strategy for resolving conflicts during merge."""

    PREFER_LOCAL = "prefer_local"
    PREFER_REMOTE = "prefer_remote"
    PREFER_RECENT = "prefer_recent"
    PREFER_STRONGER = "prefer_stronger"


@dataclass(frozen=True)
class ConflictItem:
    """Record of a single conflict resolution."""

    entity_type: str  # "neuron", "synapse", "fiber"
    local_id: str
    incoming_id: str
    resolution: str  # "kept_local", "kept_incoming"
    reason: str


@dataclass
class MergeReport:
    """Report of merge operation results."""

    neurons_added: int = 0
    neurons_updated: int = 0
    neurons_skipped: int = 0
    synapses_added: int = 0
    synapses_updated: int = 0
    fibers_added: int = 0
    fibers_updated: int = 0
    fibers_skipped: int = 0
    conflicts: list[ConflictItem] = field(default_factory=list)
    id_remap: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Merge Report:",
            f"  Neurons: {self.neurons_added} added, {self.neurons_updated} updated, {self.neurons_skipped} skipped",
            f"  Synapses: {self.synapses_added} added, {self.synapses_updated} updated",
            f"  Fibers: {self.fibers_added} added, {self.fibers_updated} updated, {self.fibers_skipped} skipped",
            f"  Conflicts resolved: {len(self.conflicts)}",
            f"  ID remaps: {len(self.id_remap)}",
        ]
        if self.conflicts:
            lines.append("  Conflict details:")
            for c in self.conflicts[:10]:
                lines.append(f"    [{c.entity_type}] {c.resolution}: {c.reason}")
            if len(self.conflicts) > 10:
                lines.append(f"    ... and {len(self.conflicts) - 10} more")
        return "\n".join(lines)


def _neuron_fingerprint(neuron_data: dict[str, Any]) -> str:
    """Create fingerprint for neuron deduplication."""
    ntype = neuron_data.get("type", "")
    content = neuron_data.get("content", "").strip().lower()
    return f"{ntype}:{content}"


def _synapse_triple(synapse_data: dict[str, Any], id_remap: dict[str, str]) -> str:
    """Create semantic triple for synapse deduplication."""
    source = id_remap.get(synapse_data["source_id"], synapse_data["source_id"])
    target = id_remap.get(synapse_data["target_id"], synapse_data["target_id"])
    stype = synapse_data.get("type", "")
    return f"{source}:{target}:{stype}"


def _fiber_neuron_set(fiber_data: dict[str, Any], id_remap: dict[str, str]) -> frozenset[str]:
    """Create remapped neuron set for fiber deduplication."""
    neuron_ids = fiber_data.get("neuron_ids", [])
    return frozenset(id_remap.get(nid, nid) for nid in neuron_ids)


def _resolve_neuron_conflict(
    local: dict[str, Any],
    incoming: dict[str, Any],
    strategy: ConflictStrategy,
) -> str:
    """Resolve neuron conflict, return 'kept_local' or 'kept_incoming'."""
    if strategy == ConflictStrategy.PREFER_LOCAL:
        return "kept_local"
    elif strategy == ConflictStrategy.PREFER_REMOTE:
        return "kept_incoming"
    elif strategy == ConflictStrategy.PREFER_RECENT:
        local_time = local.get("created_at", "")
        incoming_time = incoming.get("created_at", "")
        return "kept_incoming" if incoming_time > local_time else "kept_local"
    elif strategy == ConflictStrategy.PREFER_STRONGER:
        # For neurons, prefer the one with more metadata (richer)
        local_meta = len(local.get("metadata", {}))
        incoming_meta = len(incoming.get("metadata", {}))
        return "kept_incoming" if incoming_meta > local_meta else "kept_local"
    return "kept_local"


def _resolve_synapse_conflict(
    local: dict[str, Any],
    incoming: dict[str, Any],
    strategy: ConflictStrategy,
) -> str:
    """Resolve synapse conflict."""
    if strategy == ConflictStrategy.PREFER_LOCAL:
        return "kept_local"
    elif strategy == ConflictStrategy.PREFER_REMOTE:
        return "kept_incoming"
    elif strategy == ConflictStrategy.PREFER_RECENT:
        local_time = local.get("created_at", "")
        incoming_time = incoming.get("created_at", "")
        return "kept_incoming" if incoming_time > local_time else "kept_local"
    elif strategy == ConflictStrategy.PREFER_STRONGER:
        local_weight = local.get("weight", 0.0)
        incoming_weight = incoming.get("weight", 0.0)
        return "kept_incoming" if incoming_weight > local_weight else "kept_local"
    return "kept_local"


def _remap_ids_in_list(ids: list[str], id_remap: dict[str, str]) -> list[str]:
    """Remap a list of IDs through the id_remap dict."""
    return [id_remap.get(i, i) for i in ids]


def _add_provenance(
    data: dict[str, Any],
    source_brain_id: str,
    resolution: str,
) -> dict[str, Any]:
    """Add merge provenance to entity metadata."""
    metadata = dict(data.get("metadata", {}))
    metadata["_merge_source_brain"] = source_brain_id
    metadata["_merge_timestamp"] = utcnow().isoformat()
    metadata["_merge_resolution"] = resolution
    return {**data, "metadata": metadata}


def merge_snapshots(
    local: Any,
    incoming: Any,
    strategy: ConflictStrategy = ConflictStrategy.PREFER_LOCAL,
) -> tuple[Any, MergeReport]:
    """Merge two BrainSnapshots with conflict resolution.

    Pure function — no side effects. Returns a new merged snapshot
    and a report of all operations performed.

    Algorithm phases:
    1. Neuron fingerprinting — detect duplicates via (type, content)
    2. Synapse merge — remap IDs, detect duplicates via semantic triple
    3. Fiber merge — remap neuron/synapse IDs, detect duplicates by neuron set
    4. Metadata merge — typed_memories, projects with fiber ID remapping

    Args:
        local: The local BrainSnapshot
        incoming: The incoming BrainSnapshot to merge
        strategy: How to resolve conflicts

    Returns:
        Tuple of (merged_snapshot, merge_report)
    """
    from neural_memory.core.brain import BrainSnapshot

    report = MergeReport()

    # Phase 1: Neuron merge with fingerprinting
    local_neurons_by_fp: dict[str, dict[str, Any]] = {}
    for neuron in local.neurons:
        fp = _neuron_fingerprint(neuron)
        local_neurons_by_fp[fp] = neuron

    merged_neurons: list[dict[str, Any]] = list(local.neurons)
    id_remap: dict[str, str] = {}

    for incoming_neuron in incoming.neurons:
        fp = _neuron_fingerprint(incoming_neuron)

        if fp in local_neurons_by_fp:
            # Duplicate found — resolve conflict
            local_neuron = local_neurons_by_fp[fp]
            resolution = _resolve_neuron_conflict(local_neuron, incoming_neuron, strategy)

            id_remap[incoming_neuron["id"]] = local_neuron["id"]

            if resolution == "kept_incoming":
                # Replace local with incoming (preserve local ID)
                updated = {**incoming_neuron, "id": local_neuron["id"]}
                updated = _add_provenance(updated, incoming.brain_id, resolution)
                merged_neurons = [
                    updated if n["id"] == local_neuron["id"] else n for n in merged_neurons
                ]
                report.neurons_updated += 1
            else:
                report.neurons_skipped += 1

            report.conflicts.append(
                ConflictItem(
                    entity_type="neuron",
                    local_id=local_neuron["id"],
                    incoming_id=incoming_neuron["id"],
                    resolution=resolution,
                    reason=f"fingerprint match: {fp[:50]}",
                )
            )
        else:
            # New neuron — add it
            provenance_neuron = _add_provenance(incoming_neuron, incoming.brain_id, "added")
            merged_neurons.append(provenance_neuron)
            report.neurons_added += 1

    report.id_remap = dict(id_remap)

    # Phase 2: Synapse merge
    local_synapses_by_triple: dict[str, dict[str, Any]] = {}
    for synapse in local.synapses:
        triple = _synapse_triple(synapse, {})
        local_synapses_by_triple[triple] = synapse

    merged_synapses: list[dict[str, Any]] = list(local.synapses)

    for incoming_synapse in incoming.synapses:
        # Remap source/target IDs
        remapped_synapse = dict(incoming_synapse)
        remapped_synapse["source_id"] = id_remap.get(
            incoming_synapse["source_id"], incoming_synapse["source_id"]
        )
        remapped_synapse["target_id"] = id_remap.get(
            incoming_synapse["target_id"], incoming_synapse["target_id"]
        )

        triple = _synapse_triple(remapped_synapse, {})

        if triple in local_synapses_by_triple:
            # Duplicate synapse
            local_synapse = local_synapses_by_triple[triple]
            resolution = _resolve_synapse_conflict(local_synapse, remapped_synapse, strategy)

            id_remap[incoming_synapse["id"]] = local_synapse["id"]

            if resolution == "kept_incoming":
                updated = {**remapped_synapse, "id": local_synapse["id"]}
                updated = _add_provenance(updated, incoming.brain_id, resolution)
                merged_synapses = [
                    updated if s["id"] == local_synapse["id"] else s for s in merged_synapses
                ]
                report.synapses_updated += 1

            report.conflicts.append(
                ConflictItem(
                    entity_type="synapse",
                    local_id=local_synapse["id"],
                    incoming_id=incoming_synapse["id"],
                    resolution=resolution,
                    reason=f"triple match: {triple[:60]}",
                )
            )
        else:
            # New synapse — remap and add
            remapped_synapse = _add_provenance(remapped_synapse, incoming.brain_id, "added")
            merged_synapses.append(remapped_synapse)
            report.synapses_added += 1

    # Phase 3: Fiber merge
    local_fibers_by_neurons: dict[frozenset[str], dict[str, Any]] = {}
    for fiber in local.fibers:
        neuron_set = _fiber_neuron_set(fiber, {})
        local_fibers_by_neurons[neuron_set] = fiber

    merged_fibers: list[dict[str, Any]] = list(local.fibers)

    for incoming_fiber in incoming.fibers:
        # Remap all IDs
        remapped_fiber = dict(incoming_fiber)
        remapped_fiber["neuron_ids"] = _remap_ids_in_list(
            incoming_fiber.get("neuron_ids", []), id_remap
        )
        remapped_fiber["synapse_ids"] = _remap_ids_in_list(
            incoming_fiber.get("synapse_ids", []), id_remap
        )
        remapped_fiber["anchor_neuron_id"] = id_remap.get(
            incoming_fiber.get("anchor_neuron_id", ""),
            incoming_fiber.get("anchor_neuron_id", ""),
        )
        if "pathway" in incoming_fiber:
            remapped_fiber["pathway"] = _remap_ids_in_list(incoming_fiber["pathway"], id_remap)

        neuron_set = _fiber_neuron_set(remapped_fiber, {})

        if neuron_set in local_fibers_by_neurons:
            # Duplicate fiber by neuron set
            local_fiber = local_fibers_by_neurons[neuron_set]

            # Use strategy to pick winner
            resolution = _resolve_neuron_conflict(local_fiber, remapped_fiber, strategy)

            if resolution == "kept_incoming":
                updated = {**remapped_fiber, "id": local_fiber["id"]}
                updated = _add_provenance(updated, incoming.brain_id, resolution)
                merged_fibers = [
                    updated if f["id"] == local_fiber["id"] else f for f in merged_fibers
                ]
                report.fibers_updated += 1
            else:
                report.fibers_skipped += 1

            report.conflicts.append(
                ConflictItem(
                    entity_type="fiber",
                    local_id=local_fiber["id"],
                    incoming_id=incoming_fiber["id"],
                    resolution=resolution,
                    reason="neuron set match",
                )
            )
        else:
            remapped_fiber = _add_provenance(remapped_fiber, incoming.brain_id, "added")
            merged_fibers.append(remapped_fiber)
            report.fibers_added += 1

    # Phase 4: Metadata merge (typed_memories, projects)
    local_meta = dict(local.metadata)
    incoming_meta = incoming.metadata

    # Merge typed_memories
    local_typed = local_meta.get("typed_memories", [])
    incoming_typed = incoming_meta.get("typed_memories", [])
    local_fiber_ids = {tm.get("fiber_id") for tm in local_typed}

    for tm in incoming_typed:
        remapped_fiber_id = id_remap.get(tm.get("fiber_id", ""), tm.get("fiber_id", ""))
        if remapped_fiber_id not in local_fiber_ids:
            local_typed.append({**tm, "fiber_id": remapped_fiber_id})

    # Merge projects
    local_projects = local_meta.get("projects", [])
    incoming_projects = incoming_meta.get("projects", [])
    local_project_ids = {p.get("id") for p in local_projects}

    for proj in incoming_projects:
        if proj.get("id") not in local_project_ids:
            local_projects.append(proj)

    merged_metadata = {
        **local_meta,
        "typed_memories": local_typed,
        "projects": local_projects,
    }

    # Build merged snapshot
    merged_snapshot = BrainSnapshot(
        brain_id=local.brain_id,
        brain_name=local.brain_name,
        exported_at=utcnow(),
        version=local.version,
        neurons=merged_neurons,
        synapses=merged_synapses,
        fibers=merged_fibers,
        config=local.config,
        metadata=merged_metadata,
    )

    return merged_snapshot, report
