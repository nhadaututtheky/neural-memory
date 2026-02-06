"""Context formatting and answer reconstitution for retrieval."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.activation import ActivationResult

# Average tokens per whitespace-separated word (accounts for subword tokenization)
_TOKEN_RATIO = 1.3


def _estimate_tokens(text: str) -> int:
    """Estimate LLM token count from text using word-based heuristic."""
    return int(len(text.split()) * _TOKEN_RATIO)


if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.extraction.parser import Stimulus
    from neural_memory.storage.base import NeuralStorage


async def reconstitute_answer(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    intersections: list[str],
    stimulus: Stimulus,
) -> tuple[str | None, float]:
    """
    Attempt to reconstitute an answer from activated neurons.

    Returns (answer_text, confidence)
    """
    if not activations:
        return None, 0.0

    # Find the most relevant neurons
    candidates: list[tuple[str, float]] = []

    # Prioritize intersection neurons
    for neuron_id in intersections:
        if neuron_id in activations:
            candidates.append((neuron_id, activations[neuron_id].activation_level * 1.5))

    # Add highly activated neurons
    for neuron_id, result in activations.items():
        if neuron_id not in intersections:
            candidates.append((neuron_id, result.activation_level))

    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        return None, 0.0

    # Get the top neuron's content as answer
    top_neuron_id = candidates[0][0]
    top_neuron = await storage.get_neuron(top_neuron_id)

    if top_neuron is None:
        return None, 0.0

    # Multi-factor confidence scoring
    base_confidence = min(1.0, candidates[0][1])

    # Intersection boost (neurons reached from multiple anchor sets)
    intersection_boost = 0.05 * len(intersections) if intersections else 0.0

    # Freshness + frequency boosts from neuron state
    freshness_boost = 0.0
    frequency_boost = 0.0
    top_state = await storage.get_neuron_state(top_neuron_id)
    if top_state:
        # Freshness: sigmoid decay — ~0.15 at <1h, ~0.08 at 3d, ~0.02 at 7d
        if top_state.last_activated:
            hours_since = (datetime.now() - top_state.last_activated).total_seconds() / 3600
            freshness_boost = 0.15 / (1.0 + math.exp((hours_since - 72) / 36))

        # Frequency: logarithmic — diminishing returns past ~10 accesses
        if top_state.access_frequency > 0:
            frequency_boost = min(0.1, 0.03 * math.log1p(top_state.access_frequency))

    confidence = min(1.0, base_confidence + intersection_boost + freshness_boost + frequency_boost)

    return top_neuron.content, confidence


async def format_context(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    fibers: list[Fiber],
    max_tokens: int,
) -> tuple[str, int]:
    """Format activated memories into context for agent injection.

    Returns:
        Tuple of (formatted_context, token_estimate).
    """
    lines: list[str] = []
    token_estimate = 0

    # Add fiber summaries first (batch fetch anchors)
    if fibers:
        lines.append("## Relevant Memories\n")

        anchor_ids = list({f.anchor_neuron_id for f in fibers[:5] if not f.summary})
        anchor_map = await storage.get_neurons_batch(anchor_ids) if anchor_ids else {}

        for fiber in fibers[:5]:
            if fiber.summary:
                line = f"- {fiber.summary}"
            else:
                anchor = anchor_map.get(fiber.anchor_neuron_id)
                if anchor:
                    line = f"- {anchor.content}"
                else:
                    continue

            token_estimate += _estimate_tokens(line)
            if token_estimate > max_tokens:
                break

            lines.append(line)

    # Add individual activated neurons (batch fetch)
    if token_estimate < max_tokens:
        lines.append("\n## Related Information\n")

        sorted_activations = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )

        top_ids = [r.neuron_id for r in sorted_activations[:20]]
        neuron_map = await storage.get_neurons_batch(top_ids)

        for result in sorted_activations[:20]:
            neuron = neuron_map.get(result.neuron_id)
            if neuron is None:
                continue

            # Skip time neurons in context (they're implicit)
            if neuron.type == NeuronType.TIME:
                continue

            line = f"- [{neuron.type.value}] {neuron.content}"
            token_estimate += _estimate_tokens(line)

            if token_estimate > max_tokens:
                break

            lines.append(line)

    return "\n".join(lines), token_estimate
