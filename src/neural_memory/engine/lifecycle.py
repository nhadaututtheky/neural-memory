"""Memory lifecycle management - decay, reinforcement, compression.

Implements the Ebbinghaus forgetting curve for natural memory decay
and reinforcement for frequently accessed memories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import NeuronState
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


@dataclass
class DecayReport:
    """Report of decay operation results."""

    neurons_processed: int = 0
    neurons_decayed: int = 0
    neurons_pruned: int = 0
    synapses_processed: int = 0
    synapses_decayed: int = 0
    synapses_pruned: int = 0
    duration_ms: float = 0.0
    reference_time: datetime = field(default_factory=utcnow)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Decay Report ({self.reference_time.strftime('%Y-%m-%d %H:%M')})",
            f"  Neurons: {self.neurons_decayed}/{self.neurons_processed} decayed, {self.neurons_pruned} pruned",
            f"  Synapses: {self.synapses_decayed}/{self.synapses_processed} decayed, {self.synapses_pruned} pruned",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        return "\n".join(lines)


class DecayManager:
    """Manage memory decay using Ebbinghaus forgetting curve.

    Decay formula: retention = e^(-decay_rate * days_since_access)

    Memories that haven't been accessed recently will have their
    activation levels reduced. Memories below the prune threshold
    can be marked as dormant or removed.
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        prune_threshold: float = 0.01,
        min_age_days: float = 1.0,
    ):
        """Initialize decay manager.

        Args:
            decay_rate: Rate of decay per day (0.1 = 10% per day)
            prune_threshold: Activation level below which to prune
            min_age_days: Minimum age before applying decay
        """
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.min_age_days = min_age_days

    async def apply_decay(
        self,
        storage: NeuralStorage,
        reference_time: datetime | None = None,
        dry_run: bool = False,
    ) -> DecayReport:
        """Apply decay to all neurons and synapses in storage.

        Args:
            storage: Storage instance to apply decay to
            reference_time: Reference time for decay calculation (default: now)
            dry_run: If True, calculate but don't save changes

        Returns:
            DecayReport with statistics
        """
        import time

        start_time = time.perf_counter()
        reference_time = reference_time or utcnow()
        report = DecayReport(reference_time=reference_time)

        # Preload pinned neuron IDs to skip during decay
        pinned_neuron_ids: set[str] = set()
        if hasattr(storage, "get_pinned_neuron_ids"):
            pinned_neuron_ids = await storage.get_pinned_neuron_ids()

        # Preload grounded neuron IDs — grounded neurons resist all decay
        grounded_neuron_ids: set[str] = set()
        try:
            all_neurons = await storage.find_neurons(limit=5000)
            grounded_neuron_ids = {n.id for n in all_neurons if n.grounded}
        except Exception:
            logger.debug("Grounded neuron preload failed (non-critical)", exc_info=True)

        # Preload neuron→tier mapping for tier-aware decay
        from neural_memory.core.memory_types import TIER_DECAY_FLOORS, TIER_DECAY_MULTIPLIERS

        neuron_tier_map: dict[str, str] = {}
        try:
            if hasattr(storage, "find_typed_memories"):
                # Collect fiber_id → tier mapping, then batch-resolve fibers
                fiber_tier_pairs: list[tuple[str, str]] = []
                for tier_val in ("hot", "cold"):
                    tier_mems = await storage.find_typed_memories(tier=tier_val, limit=1000)
                    for tm in tier_mems:
                        fiber_tier_pairs.append((tm.fiber_id, tier_val))

                # Enforce BOUNDARY invariant: boundary memories always get HOT floor
                # even if stored pre-Phase3 with default tier="warm"
                from neural_memory.core.memory_types import MemoryType

                boundary_mems = await storage.find_typed_memories(
                    memory_type=MemoryType.BOUNDARY, limit=1000
                )
                for tm in boundary_mems:
                    fiber_tier_pairs.append((tm.fiber_id, "hot"))

                # Deduplicate fiber IDs and resolve fibers
                unique_fids = {fid for fid, _ in fiber_tier_pairs}
                fiber_cache: dict[str, Any] = {}
                for fid in unique_fids:
                    fiber_cache[fid] = await storage.get_fiber(fid)

                # Build neuron→tier map (boundary "hot" entries added last → override)
                for fid, tier_val in fiber_tier_pairs:
                    fiber = fiber_cache.get(fid)
                    if fiber:
                        for nid in fiber.neuron_ids:
                            neuron_tier_map[nid] = tier_val
        except (TypeError, AttributeError):
            logger.debug("Tier map build failed (non-critical)", exc_info=True)

        # Get all neuron states
        states = await storage.get_all_neuron_states()
        report.neurons_processed = len(states)

        for state in states:
            # Skip neurons belonging to pinned (KB) fibers or grounded neurons
            if state.neuron_id in pinned_neuron_ids or state.neuron_id in grounded_neuron_ids:
                continue
            # Use last_activated if available, otherwise fall back to created_at
            if state.last_activated is None:
                reference_activated = (
                    state.created_at if hasattr(state, "created_at") else reference_time
                )
            else:
                reference_activated = state.last_activated

            # Calculate time since last activation (or creation)
            time_diff = reference_time - reference_activated
            days_elapsed = time_diff.total_seconds() / 86400

            # Skip if too recent
            if days_elapsed < self.min_age_days:
                continue

            # Calculate decay using per-neuron rate (type-aware + tier-aware)
            neuron_tier = neuron_tier_map.get(state.neuron_id, "warm")
            tier_multiplier = TIER_DECAY_MULTIPLIERS.get(neuron_tier, 1.0)
            tier_floor = TIER_DECAY_FLOORS.get(neuron_tier, 0.0)
            effective_rate = state.decay_rate * tier_multiplier
            decay_factor = math.exp(-effective_rate * days_elapsed)
            new_level = max(tier_floor, state.activation_level * decay_factor)

            if new_level < state.activation_level:
                report.neurons_decayed += 1

                pruned = new_level < self.prune_threshold
                if pruned:
                    report.neurons_pruned += 1

                if not dry_run:
                    # Apply tier-aware level directly (skip state.decay() to avoid
                    # double-computing — we already have the correct new_level)
                    final_level = 0.0 if (pruned and tier_floor == 0.0) else new_level
                    decayed_state = dc_replace(
                        state,
                        activation_level=final_level,
                        last_activated=reference_time,
                    )
                    await storage.update_neuron_state(decayed_state)

        # Get all synapses and apply decay
        synapses = await storage.get_all_synapses()
        report.synapses_processed = len(synapses)

        for synapse in synapses:
            # Skip synapses connected to pinned neurons
            if synapse.source_id in pinned_neuron_ids or synapse.target_id in pinned_neuron_ids:
                continue

            # Use last_activated if available, otherwise fall back to created_at
            if synapse.last_activated is None:
                synapse_reference = (
                    synapse.created_at if hasattr(synapse, "created_at") else reference_time
                )
            else:
                synapse_reference = synapse.last_activated

            time_diff = reference_time - synapse_reference
            days_elapsed = time_diff.total_seconds() / 86400

            if days_elapsed < self.min_age_days:
                continue

            # Decay synapse weight
            decay_factor = math.exp(-self.decay_rate * days_elapsed)

            # Emotional synapses decay slower (emotional persistence)
            if synapse.type in (SynapseType.FELT, SynapseType.EVOKES):
                intensity = synapse.metadata.get("_intensity", 0.5)
                # High-intensity: decay^0.5 (much slower), low: decay^0.8 (slightly slower)
                emotional_factor = 0.5 + 0.3 * (1.0 - intensity)
                decay_factor = decay_factor**emotional_factor

            new_weight = synapse.weight * decay_factor

            if new_weight < synapse.weight:
                report.synapses_decayed += 1

                if new_weight < self.prune_threshold:
                    report.synapses_pruned += 1
                    if not dry_run:
                        # Zero out weight for pruned synapses
                        pruned_synapse = synapse.decay(0.0)
                        await storage.update_synapse(pruned_synapse)

                elif not dry_run:
                    decayed_synapse = synapse.decay(decay_factor)
                    await storage.update_synapse(decayed_synapse)

        report.duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    async def consolidate(
        self,
        storage: NeuralStorage,
        frequency_threshold: int = 5,
        boost_delta: float = 0.03,
    ) -> int:
        """Consolidate frequently-accessed memory paths.

        Boosts synapse weights for fibers that have been accessed
        at least `frequency_threshold` times, reinforcing well-trodden
        memory pathways into long-term structures.

        Args:
            storage: Storage instance containing fibers and synapses
            frequency_threshold: Minimum fiber frequency to consolidate
            boost_delta: Amount to boost each synapse weight

        Returns:
            Number of synapses consolidated (weight-boosted)
        """
        fibers = await storage.get_fibers(
            limit=100,
            order_by="frequency",
            descending=True,
        )

        consolidated = 0

        # Filter eligible fibers first
        eligible_fibers = [f for f in fibers if f.frequency >= frequency_threshold]

        # Collect ALL synapse IDs from ALL eligible fibers into one list
        all_synapse_ids: list[str] = []
        for fiber in eligible_fibers:
            all_synapse_ids.extend(fiber.synapse_ids)

        if not all_synapse_ids:
            return consolidated

        # Batch fetch: get all synapses for eligible fibers' neuron IDs
        # Since there's no get_synapses_batch(ids), use get_synapses_for_neurons
        # to fetch synapses connected to fiber neurons, then index by synapse ID.
        all_neuron_ids: list[str] = list(
            {nid for fiber in eligible_fibers for nid in fiber.neuron_ids}
        )
        outgoing = await storage.get_synapses_for_neurons(all_neuron_ids, direction="out")
        incoming = await storage.get_synapses_for_neurons(all_neuron_ids, direction="in")

        # Build synapse lookup by ID
        synapse_map: dict[str, Synapse] = {}
        for synapses_list in outgoing.values():
            for syn in synapses_list:
                synapse_map[syn.id] = syn
        for synapses_list in incoming.values():
            for syn in synapses_list:
                synapse_map[syn.id] = syn

        to_update: list[Synapse] = []
        for fiber in eligible_fibers:
            for synapse_id in fiber.synapse_ids:
                synapse = synapse_map.get(synapse_id)
                if synapse is None:
                    continue

                reinforced = synapse.reinforce(boost_delta)
                to_update.append(reinforced)
                consolidated += 1

        if to_update:
            await storage.update_synapses_batch(to_update)
        return consolidated


class ReinforcementManager:
    """Strengthen frequently accessed memory paths.

    When memories are accessed, their activation levels and
    synapse weights are increased (reinforced).
    """

    def __init__(
        self,
        reinforcement_delta: float = 0.05,
        max_activation: float = 1.0,
        max_weight: float = 1.0,
    ):
        """Initialize reinforcement manager.

        Args:
            reinforcement_delta: Amount to increase on each access
            max_activation: Maximum activation level
            max_weight: Maximum synapse weight
        """
        self.reinforcement_delta = reinforcement_delta
        self.max_activation = max_activation
        self.max_weight = max_weight

    async def reinforce(
        self,
        storage: NeuralStorage,
        neuron_ids: list[str],
        synapse_ids: list[str] | None = None,
    ) -> int:
        """Reinforce accessed neurons and synapses.

        Args:
            storage: Storage instance
            neuron_ids: List of accessed neuron IDs
            synapse_ids: Optional list of accessed synapse IDs

        Returns:
            Number of items reinforced
        """
        reinforced = 0

        # Batch fetch all neuron states at once
        states_map = await storage.get_neuron_states_batch(neuron_ids)
        now = utcnow()

        to_update: list[NeuronState] = []
        for neuron_id in neuron_ids:
            state = states_map.get(neuron_id)
            if state:
                new_level = min(
                    state.activation_level + self.reinforcement_delta,
                    self.max_activation,
                )
                # Directly set activation level (bypass sigmoid for reinforcement)
                reinforced_state = dc_replace(
                    state,
                    activation_level=new_level,
                    access_frequency=state.access_frequency + 1,
                    last_activated=now,
                )
                to_update.append(reinforced_state)
                reinforced += 1

        if to_update:
            await storage.update_neuron_states_batch(to_update)

        # Rehearse maturation records for fibers connected to reinforced neurons.
        # This is required for EPISODIC → SEMANTIC transition (needs 3+ distinct days).
        if neuron_ids:
            try:
                fibers = await storage.find_fibers_batch(neuron_ids[:10], limit_per_neuron=3)
                seen_fiber_ids: set[str] = set()
                for fiber in fibers[:10]:
                    if fiber.id in seen_fiber_ids:
                        continue
                    seen_fiber_ids.add(fiber.id)
                    record = await storage.get_maturation(fiber.id)
                    if record is not None:
                        updated = record.rehearse(now)
                        await storage.save_maturation(updated)
            except Exception:
                logger.debug("Maturation rehearsal skipped during reinforce", exc_info=True)

        if synapse_ids:
            # Batch fetch synapses via neuron-based lookup
            # Collect neuron IDs involved in synapse reinforcement from states
            all_neuron_ids = list(states_map.keys())
            if all_neuron_ids:
                outgoing = await storage.get_synapses_for_neurons(all_neuron_ids, direction="out")
                incoming = await storage.get_synapses_for_neurons(all_neuron_ids, direction="in")
                synapse_map_2: dict[str, Synapse] = {}
                for synapses_out in outgoing.values():
                    for syn in synapses_out:
                        synapse_map_2[syn.id] = syn
                for synapses_in in incoming.values():
                    for syn in synapses_in:
                        synapse_map_2[syn.id] = syn
            else:
                synapse_map_2 = {}

            synapse_updates: list[Synapse] = []
            for synapse_id in synapse_ids:
                synapse = synapse_map_2.get(synapse_id)
                if synapse is None:
                    # Fallback for synapses not connected to reinforced neurons
                    synapse = await storage.get_synapse(synapse_id)
                if synapse:
                    new_weight = min(
                        synapse.weight + self.reinforcement_delta,
                        self.max_weight,
                    )
                    reinforced_synapse = synapse.reinforce(new_weight - synapse.weight)
                    synapse_updates.append(reinforced_synapse)
                    reinforced += 1

            if synapse_updates:
                await storage.update_synapses_batch(synapse_updates)

        return reinforced
