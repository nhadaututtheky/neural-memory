"""Trail-based reflex activation through fiber pathways."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.engine.activation import ActivationResult
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.fiber import Fiber
    from neural_memory.storage.base import NeuralStorage


@dataclass
class CoActivation:
    """
    Neurons that fired together within a temporal window.

    Implements Hebbian principle: "Neurons that fire together wire together"
    Co-activation tracking enables binding between simultaneously active neurons.

    Attributes:
        neuron_ids: The neurons that co-activated
        temporal_window_ms: How close in time they fired
        co_fire_count: Number of times they co-activated
        binding_strength: Strength of the co-activation binding (0.0 - 1.0)
        source_anchors: Which anchor sets activated these neurons
    """

    neuron_ids: frozenset[str]
    temporal_window_ms: int
    co_fire_count: int = 1
    binding_strength: float = 0.5
    source_anchors: list[str] = field(default_factory=list)


class ReflexActivation:
    """
    Trail-based activation through fiber pathways.

    Unlike SpreadingActivation which uses distance-based decay,
    ReflexActivation conducts signals along fiber pathways with
    trail decay that considers:
    - Fiber conductivity
    - Synapse weight
    - Time factor (recent fibers conduct better)
    - Position in pathway
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> None:
        """
        Initialize the reflex activation system.

        Args:
            storage: Storage backend
            config: Brain configuration
        """
        self._storage = storage
        self._config = config

    async def activate_trail(
        self,
        anchor_neurons: list[str],
        fibers: list[Fiber],
        reference_time: datetime | None = None,
        decay_rate: float = 0.15,
    ) -> dict[str, ActivationResult]:
        """
        Spread activation along fiber pathways with trail decay.

        Trail decay formula:
            new_level = level * (1 - decay) * synapse.weight * fiber.conductivity * time_factor

        Args:
            anchor_neurons: Starting neurons with activation = 1.0
            fibers: Fibers to conduct through
            reference_time: Reference time for time factor calculation
            decay_rate: Base decay rate per hop

        Returns:
            Dict mapping neuron_id to ActivationResult
        """
        if reference_time is None:
            reference_time = utcnow()

        results: dict[str, ActivationResult] = {}

        # Initialize anchor neurons
        for anchor_id in anchor_neurons:
            results[anchor_id] = ActivationResult(
                neuron_id=anchor_id,
                activation_level=1.0,
                hop_distance=0,
                path=[anchor_id],
                source_anchor=anchor_id,
            )

        # Conduct through each fiber that contains anchor neurons
        for fiber in fibers:
            # Find anchor neurons in this fiber's pathway
            fiber_anchors = [a for a in anchor_neurons if fiber.is_in_pathway(a)]
            if not fiber_anchors:
                continue

            # Calculate time factor for this fiber
            time_factor = self._compute_time_factor(fiber, reference_time)

            # Conduct from each anchor along the pathway
            for anchor_id in fiber_anchors:
                start_pos = fiber.pathway_position(anchor_id)
                if start_pos is None:
                    continue

                # Spread forward in pathway
                self._conduct_along_pathway(
                    results=results,
                    fiber=fiber,
                    start_pos=start_pos,
                    direction=1,
                    anchor_id=anchor_id,
                    decay_rate=decay_rate,
                    time_factor=time_factor,
                )

                # Spread backward in pathway
                self._conduct_along_pathway(
                    results=results,
                    fiber=fiber,
                    start_pos=start_pos,
                    direction=-1,
                    anchor_id=anchor_id,
                    decay_rate=decay_rate,
                    time_factor=time_factor,
                )

        return results

    def _conduct_along_pathway(
        self,
        results: dict[str, ActivationResult],
        fiber: Fiber,
        start_pos: int,
        direction: int,
        anchor_id: str,
        decay_rate: float,
        time_factor: float,
    ) -> None:
        """
        Conduct activation along a fiber pathway in one direction.

        Args:
            results: Results dict to update
            fiber: The fiber to conduct through
            start_pos: Starting position in pathway
            direction: 1 for forward, -1 for backward
            anchor_id: The source anchor neuron
            decay_rate: Decay rate per hop
            time_factor: Time-based conductivity factor
        """
        current_level = 1.0
        path = [fiber.pathway[start_pos]]
        pos = start_pos + direction
        hops = 0

        while 0 <= pos < len(fiber.pathway):
            hops += 1
            neuron_id = fiber.pathway[pos]

            # Trail decay formula
            # level * (1 - decay) * fiber.conductivity * time_factor
            current_level = current_level * (1 - decay_rate) * fiber.conductivity * time_factor

            # Stop if below threshold
            if current_level < self._config.activation_threshold:
                break

            path = [*path, neuron_id]

            # Update if better than existing
            existing = results.get(neuron_id)
            if existing is None or current_level > existing.activation_level:
                results[neuron_id] = ActivationResult(
                    neuron_id=neuron_id,
                    activation_level=current_level,
                    hop_distance=hops,
                    path=path,
                    source_anchor=anchor_id,
                )

            pos += direction

    def _compute_time_factor(
        self,
        fiber: Fiber,
        reference_time: datetime,
    ) -> float:
        """
        Compute time-based conductivity factor.

        Recent fibers conduct better. Decay over 7 days.

        Args:
            fiber: The fiber
            reference_time: Reference time

        Returns:
            Time factor between 0.1 and 1.0
        """
        if fiber.last_conducted is None:
            # Use fiber salience as proxy for importance
            return 0.3 + 0.4 * fiber.salience

        age_hours = (reference_time - fiber.last_conducted).total_seconds() / 3600
        # Sigmoid decay: ~1.0 at <1 day, ~0.5 at 3 days, ~0.15 at 7 days
        return max(0.1, 1.0 / (1.0 + math.exp((age_hours - 72) / 36)))

    def find_co_activated(
        self,
        activation_sets: list[dict[str, ActivationResult]],
        temporal_window_ms: int = 500,
    ) -> list[CoActivation]:
        """
        Find neurons that co-activated within a temporal window.

        Implements Hebbian principle: "Neurons that fire together wire together"

        Args:
            activation_sets: Activation results from different anchor sets
            temporal_window_ms: How close in time neurons must fire

        Returns:
            List of CoActivation objects sorted by binding strength
        """
        if not activation_sets:
            return []

        # Track which anchors activated each neuron
        neuron_sources: dict[str, list[int]] = defaultdict(list)

        for i, activation_set in enumerate(activation_sets):
            for neuron_id in activation_set:
                neuron_sources[neuron_id].append(i)

        # Find neurons activated by multiple sources and group them
        co_activated_neurons: list[str] = []
        max_source_count = 0

        for neuron_id, sources in neuron_sources.items():
            if len(sources) < 2:
                continue
            co_activated_neurons.append(neuron_id)
            max_source_count = max(max_source_count, len(sources))

        if not co_activated_neurons:
            return []

        # Group all co-activated neurons into a single CoActivation
        # (they all fired within the same query/context temporal window)
        source_activations: list[float] = []
        for neuron_id in co_activated_neurons:
            neuron_total = 0.0
            neuron_count = 0
            for i in neuron_sources[neuron_id]:
                if neuron_id in activation_sets[i]:
                    neuron_total += activation_sets[i][neuron_id].activation_level
                    neuron_count += 1
            if neuron_count > 0:
                # Average activation per neuron across its sources
                source_activations.append(neuron_total / len(activation_sets))

        binding_strength = (
            sum(source_activations) / len(co_activated_neurons) if source_activations else 0.0
        )

        co_activations: list[CoActivation] = [
            CoActivation(
                neuron_ids=frozenset(co_activated_neurons),
                temporal_window_ms=temporal_window_ms,
                co_fire_count=max_source_count,
                binding_strength=min(1.0, binding_strength),
                source_anchors=[],
            )
        ]

        return co_activations

    async def activate_with_co_binding(
        self,
        anchor_sets: list[list[str]],
        fibers: list[Fiber],
        reference_time: datetime | None = None,
    ) -> tuple[dict[str, ActivationResult], list[CoActivation]]:
        """
        Activate from multiple anchor sets with co-activation binding.

        Combines trail activation with co-activation detection.

        Args:
            anchor_sets: List of anchor neuron lists (e.g., [time_anchors, entity_anchors])
            fibers: Fibers to conduct through
            reference_time: Reference time for time factor

        Returns:
            Tuple of (combined activations, co-activations)
        """
        if reference_time is None:
            reference_time = utcnow()

        # Activate from each anchor set in parallel
        tasks = [
            self.activate_trail(
                anchor_neurons=anchors,
                fibers=fibers,
                reference_time=reference_time,
            )
            for anchors in anchor_sets
            if anchors
        ]
        activation_results = list(await asyncio.gather(*tasks)) if tasks else []

        if not activation_results:
            return {}, []

        # Find co-activations
        co_activations = self.find_co_activated(activation_results)

        # Combine results with boosted activation for co-activated neurons
        combined: dict[str, ActivationResult] = {}
        co_activated_ids = {neuron_id for co in co_activations for neuron_id in co.neuron_ids}

        for result_set in activation_results:
            for neuron_id, activation in result_set.items():
                existing = combined.get(neuron_id)

                if existing is None:
                    combined[neuron_id] = activation
                else:
                    # Boost co-activated neurons
                    if neuron_id in co_activated_ids:
                        new_level = min(
                            1.0,
                            existing.activation_level + activation.activation_level * 0.5,
                        )
                    else:
                        new_level = max(existing.activation_level, activation.activation_level)

                    combined[neuron_id] = ActivationResult(
                        neuron_id=neuron_id,
                        activation_level=new_level,
                        hop_distance=min(existing.hop_distance, activation.hop_distance),
                        path=existing.path
                        if existing.hop_distance <= activation.hop_distance
                        else activation.path,
                        source_anchor=existing.source_anchor,
                    )

        return combined, co_activations
