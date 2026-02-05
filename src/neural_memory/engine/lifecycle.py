"""Memory lifecycle management - decay, reinforcement, compression.

Implements the Ebbinghaus forgetting curve for natural memory decay
and reinforcement for frequently accessed memories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

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
    reference_time: datetime = field(default_factory=datetime.now)

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
        reference_time = reference_time or datetime.now()
        report = DecayReport(reference_time=reference_time)

        # Get all neuron states
        states = await storage.get_all_neuron_states()
        report.neurons_processed = len(states)

        for state in states:
            if state.last_activated is None:
                continue

            # Calculate time since last activation
            time_diff = reference_time - state.last_activated
            days_elapsed = time_diff.total_seconds() / 86400

            # Skip if too recent
            if days_elapsed < self.min_age_days:
                continue

            # Calculate decay
            decay_factor = math.exp(-self.decay_rate * days_elapsed)
            new_level = state.activation_level * decay_factor

            if new_level < state.activation_level:
                report.neurons_decayed += 1

                if new_level < self.prune_threshold:
                    report.neurons_pruned += 1
                    new_level = 0.0

                if not dry_run:
                    # Update the neuron state
                    updated_state = state.with_activation(new_level)
                    updated_state = type(state)(
                        neuron_id=state.neuron_id,
                        activation_level=new_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=state.decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

        # Get all synapses and apply decay
        synapses = await storage.get_all_synapses()
        report.synapses_processed = len(synapses)

        for synapse in synapses:
            if synapse.last_activated is None:
                continue

            time_diff = reference_time - synapse.last_activated
            days_elapsed = time_diff.total_seconds() / 86400

            if days_elapsed < self.min_age_days:
                continue

            # Decay synapse weight
            decay_factor = math.exp(-self.decay_rate * days_elapsed)
            new_weight = synapse.weight * decay_factor

            if new_weight < synapse.weight:
                report.synapses_decayed += 1

                if new_weight < self.prune_threshold:
                    report.synapses_pruned += 1
                    if not dry_run:
                        # Could delete synapse here, but just set weight to 0 for now
                        pass

                if not dry_run and new_weight >= self.prune_threshold:
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

        for fiber in fibers:
            if fiber.frequency < frequency_threshold:
                break

            for synapse_id in fiber.synapse_ids:
                synapse = await storage.get_synapse(synapse_id)
                if synapse is None:
                    continue

                reinforced = synapse.reinforce(boost_delta)
                await storage.update_synapse(reinforced)
                consolidated += 1

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

        for neuron_id in neuron_ids:
            state = await storage.get_neuron_state(neuron_id)
            if state:
                new_level = min(
                    state.activation_level + self.reinforcement_delta,
                    self.max_activation,
                )
                activated_state = state.activate(new_level - state.activation_level)
                await storage.update_neuron_state(activated_state)
                reinforced += 1

        if synapse_ids:
            for synapse_id in synapse_ids:
                synapse = await storage.get_synapse(synapse_id)
                if synapse:
                    new_weight = min(
                        synapse.weight + self.reinforcement_delta,
                        self.max_weight,
                    )
                    reinforced_synapse = synapse.reinforce(new_weight - synapse.weight)
                    await storage.update_synapse(reinforced_synapse)
                    reinforced += 1

        return reinforced
