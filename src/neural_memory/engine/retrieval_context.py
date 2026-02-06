"""Context formatting and answer reconstitution for retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.activation import ActivationResult

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

    # Confidence based on activation and intersection count
    confidence = min(1.0, candidates[0][1])
    if intersections:
        confidence = min(1.0, confidence + 0.1 * len(intersections))

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

    # Add fiber summaries first
    if fibers:
        lines.append("## Relevant Memories\n")

        for fiber in fibers[:5]:
            if fiber.summary:
                line = f"- {fiber.summary}"
            else:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    line = f"- {anchor.content}"
                else:
                    continue

            token_estimate += len(line.split())
            if token_estimate > max_tokens:
                break

            lines.append(line)

    # Add individual activated neurons
    if token_estimate < max_tokens:
        lines.append("\n## Related Information\n")

        sorted_activations = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )

        for result in sorted_activations[:20]:
            neuron = await storage.get_neuron(result.neuron_id)
            if neuron is None:
                continue

            # Skip time neurons in context (they're implicit)
            if neuron.type == NeuronType.TIME:
                continue

            line = f"- [{neuron.type.value}] {neuron.content}"
            token_estimate += len(line.split())

            if token_estimate > max_tokens:
                break

            lines.append(line)

    return "\n".join(lines), token_estimate
