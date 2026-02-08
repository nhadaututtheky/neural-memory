"""Workflow suggestion engine — proactive next-action recommendations.

Uses spreading activation through BEFORE synapses to suggest
what action the user is likely to do next, based on learned habits.

Dual threshold prevents premature suggestions:
- Synapse weight must exceed habit_suggestion_min_weight
- Sequential count must exceed habit_suggestion_min_count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowSuggestion:
    """A suggested next action based on learned habits.

    Attributes:
        action_type: The suggested action type
        confidence: Confidence score (synapse weight)
        source_habit: Name of the habit that triggered this suggestion
        sequential_count: How many times this sequence was observed
    """

    action_type: str
    confidence: float
    source_habit: str | None = None
    sequential_count: int = 0


async def suggest_next_action(
    storage: NeuralStorage,
    current_action: str,
    config: BrainConfig,
) -> list[WorkflowSuggestion]:
    """Suggest next actions based on learned sequential patterns.

    Finds the ACTION neuron for current_action, gets outgoing BEFORE
    neighbors, filters by dual threshold (weight + sequential_count),
    and returns sorted suggestions.

    Args:
        storage: Storage backend
        current_action: The action just performed
        config: Brain configuration

    Returns:
        List of WorkflowSuggestion sorted by confidence descending
    """
    # Find ACTION neuron for current action
    neurons = await storage.find_neurons(
        content_exact=current_action,
        type=NeuronType.ACTION,
    )
    if not neurons:
        return []

    neuron = neurons[0]

    # Get outgoing BEFORE neighbors
    neighbors = await storage.get_neighbors(
        neuron.id,
        direction="out",
        synapse_types=[SynapseType.BEFORE],
        min_weight=config.habit_suggestion_min_weight,
    )

    suggestions: list[WorkflowSuggestion] = []
    for neighbor_neuron, synapse in neighbors:
        sequential_count = synapse.metadata.get("sequential_count", 0)
        if sequential_count < config.habit_suggestion_min_count:
            continue

        # Look for a workflow fiber containing both neurons
        source_habit = None
        try:
            fibers = await storage.find_fibers(contains_neuron=neuron.id, limit=20)
            for fiber in fibers:
                if neighbor_neuron.id in fiber.neuron_ids and fiber.metadata.get("_habit_pattern"):
                    source_habit = fiber.summary
                    break
        except Exception:
            logger.debug("Workflow fiber lookup failed", exc_info=True)

        suggestions.append(
            WorkflowSuggestion(
                action_type=neighbor_neuron.content,
                confidence=synapse.weight,
                source_habit=source_habit,
                sequential_count=sequential_count,
            )
        )

    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions
