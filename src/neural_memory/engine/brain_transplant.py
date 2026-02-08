"""Brain transplant engine — partial subgraph extraction and cross-brain merge.

Extracts a filtered subgraph from a BrainSnapshot based on tag/type/salience
criteria, then merges it into a target brain using the existing merge_snapshots()
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neural_memory.engine.merge import ConflictStrategy, MergeReport, merge_snapshots

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainSnapshot
    from neural_memory.storage.base import NeuralStorage


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransplantFilter:
    """Criteria for selecting a subgraph from a brain snapshot.

    All criteria are combined with AND logic (every non-None field must match).
    Within a single criterion, items are combined with OR logic — e.g. a fiber
    matches if it has *any* of the listed tags.

    Attributes:
        tags: Match fibers that carry ANY of these tags.
        memory_types: Match typed_memories whose memory_type is in this set.
        neuron_types: Include only neurons whose type is in this set.
        min_salience: Minimum salience a fiber must have to be included.
        include_orphan_neurons: When True, include neurons that are not
            referenced by any matched fiber but satisfy neuron_types.
    """

    tags: frozenset[str] | None = None
    memory_types: frozenset[str] | None = None
    neuron_types: frozenset[str] | None = None
    min_salience: float = 0.0
    include_orphan_neurons: bool = False


@dataclass(frozen=True)
class TransplantResult:
    """Outcome of a transplant operation.

    Attributes:
        merge_report: Full merge report produced by merge_snapshots().
        fibers_transplanted: Number of fibers in the extracted subgraph.
        neurons_transplanted: Number of neurons in the extracted subgraph.
        synapses_transplanted: Number of synapses in the extracted subgraph.
        filter_used: The TransplantFilter that governed extraction.
    """

    merge_report: MergeReport
    fibers_transplanted: int
    neurons_transplanted: int
    synapses_transplanted: int
    filter_used: TransplantFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fiber_matches_tags(
    fiber: dict[str, Any],
    required_tags: frozenset[str],
) -> bool:
    """Return True if the fiber carries ANY of the required tags."""
    fiber_tags = set(fiber.get("tags", []))
    return bool(fiber_tags & required_tags)


def _fiber_matches_salience(
    fiber: dict[str, Any],
    min_salience: float,
) -> bool:
    """Return True if the fiber's salience meets the minimum threshold."""
    return float(fiber.get("salience", 0.0)) >= min_salience


def _collect_typed_memory_fiber_ids(
    metadata: dict[str, Any],
    memory_types: frozenset[str],
) -> set[str]:
    """Return fiber IDs linked to typed_memories that match the filter."""
    typed_memories: list[dict[str, Any]] = metadata.get("typed_memories", [])
    return {
        tm.get("fiber_id", "") for tm in typed_memories if tm.get("memory_type", "") in memory_types
    }


def _filter_fibers(
    fibers: list[dict[str, Any]],
    filt: TransplantFilter,
    typed_memory_fiber_ids: set[str],
) -> list[dict[str, Any]]:
    """Return the subset of fibers that satisfy every active filter criterion."""
    matched: list[dict[str, Any]] = []

    for fiber in fibers:
        # Salience gate — always checked (defaults to 0.0 so passes by default)
        if not _fiber_matches_salience(fiber, filt.min_salience):
            continue

        # Tag matching (if requested)
        passes_tags = filt.tags is None or _fiber_matches_tags(fiber, filt.tags)

        # Memory-type matching (if requested)
        passes_type = filt.memory_types is None or fiber.get("id", "") in typed_memory_fiber_ids

        if passes_tags and passes_type:
            matched.append(fiber)

    return matched


def _collect_ids_from_fibers(
    fibers: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Collect all neuron_ids and synapse_ids referenced by the given fibers."""
    neuron_ids: set[str] = set()
    synapse_ids: set[str] = set()

    for fiber in fibers:
        neuron_ids.update(fiber.get("neuron_ids", []))
        synapse_ids.update(fiber.get("synapse_ids", []))

        # Include the anchor neuron as well
        anchor = fiber.get("anchor_neuron_id")
        if anchor:
            neuron_ids.add(anchor)

        # Include pathway neurons
        neuron_ids.update(fiber.get("pathway", []))

    return neuron_ids, synapse_ids


def _filter_neurons(
    neurons: list[dict[str, Any]],
    allowed_ids: set[str],
    neuron_types: frozenset[str] | None,
    include_orphans: bool,
) -> list[dict[str, Any]]:
    """Return neurons that are in the allowed set and pass the type filter.

    When *include_orphans* is True and *neuron_types* is set, neurons that
    match the type filter are included even if not in *allowed_ids*.
    """
    result: list[dict[str, Any]] = []

    for neuron in neurons:
        nid = neuron.get("id", "")
        ntype = neuron.get("type", "")
        in_allowed = nid in allowed_ids
        passes_type = neuron_types is None or ntype in neuron_types

        if (in_allowed and passes_type) or (
            include_orphans and neuron_types is not None and passes_type
        ):
            result.append(neuron)

    return result


def _filter_synapses(
    synapses: list[dict[str, Any]],
    allowed_neuron_ids: set[str],
) -> list[dict[str, Any]]:
    """Return synapses whose *both* endpoints are in the allowed neuron set."""
    return [
        s
        for s in synapses
        if s.get("source_id", "") in allowed_neuron_ids
        and s.get("target_id", "") in allowed_neuron_ids
    ]


def _filter_metadata(
    metadata: dict[str, Any],
    matched_fiber_ids: set[str],
) -> dict[str, Any]:
    """Return metadata trimmed to only reference matched fibers."""
    typed_memories = [
        tm
        for tm in metadata.get("typed_memories", [])
        if tm.get("fiber_id", "") in matched_fiber_ids
    ]
    # Keep projects intact — they are brain-level, not fiber-level
    return {
        **metadata,
        "typed_memories": typed_memories,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_subgraph(
    snapshot: BrainSnapshot,
    filt: TransplantFilter,
) -> BrainSnapshot:
    """Extract a filtered subgraph from a brain snapshot.

    Pure function with no side effects. Applies the filter criteria to select
    fibers, then collects all neurons and synapses referenced by those fibers.
    Synapses are only included when *both* of their endpoint neurons are in
    the extracted set.

    Args:
        snapshot: The source brain snapshot.
        filt: The filter criteria to apply.

    Returns:
        A new BrainSnapshot containing only the matched subgraph.
    """
    from neural_memory.core.brain import BrainSnapshot as _BrainSnapshot

    # Determine fiber IDs linked via typed_memories (if memory_types filter active)
    typed_memory_fiber_ids: set[str] = set()
    if filt.memory_types is not None:
        typed_memory_fiber_ids = _collect_typed_memory_fiber_ids(
            snapshot.metadata, filt.memory_types
        )

    # Phase 1: Filter fibers
    matched_fibers = _filter_fibers(snapshot.fibers, filt, typed_memory_fiber_ids)
    matched_fiber_ids = {f.get("id", "") for f in matched_fibers}

    # Phase 2: Collect neuron + synapse IDs from matched fibers
    fiber_neuron_ids, fiber_synapse_ids = _collect_ids_from_fibers(matched_fibers)

    # Phase 3: Filter neurons (respects neuron_types and orphan setting)
    matched_neurons = _filter_neurons(
        snapshot.neurons,
        fiber_neuron_ids,
        filt.neuron_types,
        filt.include_orphan_neurons,
    )
    final_neuron_ids = {n.get("id", "") for n in matched_neurons}

    # Phase 4: Filter synapses — both endpoints must be in final neuron set
    matched_synapses = _filter_synapses(snapshot.synapses, final_neuron_ids)

    # Phase 5: Trim metadata
    filtered_metadata = _filter_metadata(snapshot.metadata, matched_fiber_ids)

    return _BrainSnapshot(
        brain_id=snapshot.brain_id,
        brain_name=snapshot.brain_name,
        exported_at=snapshot.exported_at,
        version=snapshot.version,
        neurons=matched_neurons,
        synapses=matched_synapses,
        fibers=matched_fibers,
        config=snapshot.config,
        metadata=filtered_metadata,
    )


async def transplant(
    source_storage: NeuralStorage,
    target_storage: NeuralStorage,
    source_brain_id: str,
    target_brain_id: str,
    filt: TransplantFilter,
    strategy: ConflictStrategy = ConflictStrategy.PREFER_LOCAL,
) -> TransplantResult:
    """Extract a subgraph from one brain and merge it into another.

    Workflow:
        1. Export the source brain as a snapshot.
        2. Extract a subgraph using *filt*.
        3. Export the target brain as a snapshot.
        4. Merge the subgraph into the target using *strategy*.
        5. Clear the target brain and re-import the merged snapshot.

    Args:
        source_storage: Storage backend holding the source brain.
        target_storage: Storage backend holding the target brain.
        source_brain_id: ID of the brain to extract from.
        target_brain_id: ID of the brain to merge into.
        filt: Criteria governing which parts of the source to transplant.
        strategy: Conflict resolution strategy passed to merge_snapshots().

    Returns:
        A TransplantResult summarising what was transplanted.

    Raises:
        ValueError: If either brain does not exist in its storage.
    """
    from neural_memory.core.brain import Brain, BrainConfig

    # Step 1: Export source and extract subgraph
    source_snapshot = await source_storage.export_brain(source_brain_id)
    subgraph = extract_subgraph(source_snapshot, filt)

    # Step 2: Export target
    target_snapshot = await target_storage.export_brain(target_brain_id)

    # Step 3: Merge extracted subgraph into target
    merged, merge_report = merge_snapshots(target_snapshot, subgraph, strategy)

    # Step 4: Clear and reimport
    await target_storage.clear(target_brain_id)

    config = BrainConfig(**merged.config) if merged.config else BrainConfig()
    brain = Brain.create(
        name=merged.brain_name,
        config=config,
        brain_id=target_brain_id,
    )
    await target_storage.save_brain(brain)
    await target_storage.import_brain(merged, target_brain_id)

    return TransplantResult(
        merge_report=merge_report,
        fibers_transplanted=len(subgraph.fibers),
        neurons_transplanted=len(subgraph.neurons),
        synapses_transplanted=len(subgraph.synapses),
        filter_used=filt,
    )
