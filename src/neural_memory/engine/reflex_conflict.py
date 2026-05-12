"""Reflex Arc conflict detection and pinning logic.

Handles:
- Detecting SimHash-based conflicts between a new reflex and existing ones
- Pinning/unpinning neurons as reflexes with max cap enforcement
- Auto-superseding conflicting reflexes via SUPERSEDES synapses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronStatus
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.simhash import hamming_distance, simhash

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.neuron import Neuron
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# SimHash hamming distance threshold for conflict detection.
# 10 bits out of 64 ≈ ~85% similarity.
REFLEX_CONFLICT_THRESHOLD = 10


@dataclass(frozen=True)
class ReflexConflict:
    """A detected conflict between a new reflex and an existing one.

    Attributes:
        existing_id: ID of the existing reflex neuron
        existing_content: Content of the existing reflex neuron
        hamming_distance: SimHash hamming distance (lower = more similar)
        action: Recommended action ("supersede")
    """

    existing_id: str
    existing_content: str
    hamming_distance: int
    action: str = "supersede"


@dataclass(frozen=True)
class ReflexPinResult:
    """Result of a pin_as_reflex operation.

    Attributes:
        pinned: Whether the neuron was successfully pinned
        conflicts_resolved: List of conflicts that were auto-resolved
        error: Error message if pinning failed, None on success
    """

    pinned: bool
    conflicts_resolved: list[ReflexConflict]
    error: str | None = None


async def check_conflicts(
    neuron: Neuron,
    storage: NeuralStorage,
    threshold: int = REFLEX_CONFLICT_THRESHOLD,
) -> list[ReflexConflict]:
    """Check if a neuron conflicts with existing reflexes via SimHash similarity.

    Args:
        neuron: The neuron to check (will compute SimHash from content)
        storage: Storage backend to query existing reflexes
        threshold: Max hamming distance to consider a conflict (default 10)

    Returns:
        List of detected conflicts, empty if none found
    """
    new_hash = neuron.content_hash or simhash(neuron.content)
    if new_hash == 0:
        return []

    existing_reflexes = await storage.find_reflex_neurons(limit=50)

    conflicts: list[ReflexConflict] = []
    for existing in existing_reflexes:
        # Skip self-comparison
        if existing.id == neuron.id:
            continue

        existing_hash = existing.content_hash or simhash(existing.content)
        if existing_hash == 0:
            continue

        dist = hamming_distance(new_hash, existing_hash)
        if dist <= threshold:
            conflicts.append(
                ReflexConflict(
                    existing_id=existing.id,
                    existing_content=existing.content,
                    hamming_distance=dist,
                )
            )

    return conflicts


async def pin_as_reflex(
    neuron_id: str,
    storage: NeuralStorage,
    config: BrainConfig,
    threshold: int = REFLEX_CONFLICT_THRESHOLD,
) -> ReflexPinResult:
    """Pin a neuron as a reflex (always-on in recall).

    Enforces max cap, detects SimHash conflicts, auto-supersedes old reflexes.

    Args:
        neuron_id: ID of the neuron to pin
        storage: Storage backend
        config: Brain config (for max_reflexes)
        threshold: SimHash conflict threshold

    Returns:
        ReflexPinResult with success/failure and resolved conflicts
    """
    neuron = await storage.get_neuron(neuron_id)
    if neuron is None:
        return ReflexPinResult(pinned=False, conflicts_resolved=[], error="Neuron not found")

    if neuron.reflex:
        return ReflexPinResult(pinned=True, conflicts_resolved=[])

    # Check conflicts first — resolving them may free up slots
    conflicts = await check_conflicts(neuron, storage, threshold=threshold)

    # Resolve conflicts: unpin old reflexes + create supersedes synapses
    resolved: list[ReflexConflict] = []
    for conflict in conflicts:
        old_neuron = await storage.get_neuron(conflict.existing_id)
        if old_neuron is not None and old_neuron.reflex:
            unpinned = old_neuron.with_reflex(pinned=False).with_status(
                NeuronStatus.SUPERSEDED, superseded_by=neuron_id
            )
            await storage.update_neuron(unpinned)

            synapse = Synapse.create(
                source_id=neuron_id,
                target_id=conflict.existing_id,
                type=SynapseType.SUPERSEDES,
                weight=1.0,
                metadata={"_reflex_supersede": True, "hamming_distance": conflict.hamming_distance},
            )
            await storage.add_synapse(synapse)
            resolved.append(conflict)
            logger.info(
                "Reflex conflict resolved: %s supersedes %s (hamming=%d)",
                neuron_id,
                conflict.existing_id,
                conflict.hamming_distance,
            )

    # Check max cap (after resolving conflicts, since unpins free slots)
    existing_reflexes = await storage.find_reflex_neurons(limit=50)
    current_count = len(existing_reflexes)
    max_reflexes = getattr(config, "max_reflexes", 20)

    if current_count >= max_reflexes:
        return ReflexPinResult(
            pinned=False,
            conflicts_resolved=resolved,
            error=f"Max reflexes reached ({max_reflexes}). Unpin one first.",
        )

    # Pin the neuron — also revive to ACTIVE so a formerly-superseded
    # neuron does not get hard-dropped by `_filter_by_status` after pin.
    pinned_neuron = neuron.with_reflex(pinned=True).with_status(NeuronStatus.ACTIVE)
    await storage.update_neuron(pinned_neuron)
    logger.info("Neuron %s pinned as reflex", neuron_id)

    return ReflexPinResult(pinned=True, conflicts_resolved=resolved)


async def unpin_reflex(
    neuron_id: str,
    storage: NeuralStorage,
) -> bool:
    """Unpin a neuron from reflex status.

    Args:
        neuron_id: ID of the neuron to unpin
        storage: Storage backend

    Returns:
        True if unpinned, False if neuron not found or not a reflex
    """
    neuron = await storage.get_neuron(neuron_id)
    if neuron is None or not neuron.reflex:
        return False

    unpinned = neuron.with_reflex(pinned=False)
    await storage.update_neuron(unpinned)
    logger.info("Neuron %s unpinned from reflex", neuron_id)
    return True
