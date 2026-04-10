"""Abstraction-level constraint for spreading activation.

Neurons are assigned an abstraction level based on their type:
  Level 1 (concrete): TIME, SPATIAL, ENTITY, ACTION, STATE, SENSORY
  Level 2 (abstract): CONCEPT, INTENT
  Level 3 (meta):     HYPOTHESIS, PREDICTION, SCHEMA

Level 0 means unassigned — unconstrained by default.

The constraint gate in SpreadingActivation uses can_activate() to skip
neighbors whose level distance from the current node exceeds the configured
max_distance. Both nodes must have a non-zero level for the check to fire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType

if TYPE_CHECKING:
    from neural_memory.core.neuron import Neuron

# Canonical abstraction level per NeuronType.
# Level 0 = unassigned (not listed here; caller falls back to 0 via .get()).
DEFAULT_ABSTRACTION_LEVELS: dict[NeuronType, int] = {
    # Level 1 — concrete, grounded in experience
    NeuronType.TIME: 1,
    NeuronType.SPATIAL: 1,
    NeuronType.ENTITY: 1,
    NeuronType.ACTION: 1,
    NeuronType.STATE: 1,
    NeuronType.SENSORY: 1,
    # Level 2 — abstract / intentional
    NeuronType.CONCEPT: 2,
    NeuronType.INTENT: 2,
    # Level 3 — meta / epistemic
    NeuronType.HYPOTHESIS: 3,
    NeuronType.PREDICTION: 3,
    NeuronType.SCHEMA: 3,
}


def assign_abstraction_level(neuron: Neuron) -> Neuron:
    """Return a neuron with its abstraction level set from DEFAULT_ABSTRACTION_LEVELS.

    If the neuron already has a non-zero abstraction level in its metadata,
    it is returned unchanged (preserves explicit overrides).

    Args:
        neuron: The source neuron (never mutated).

    Returns:
        New Neuron with _abstraction_level set, or the same neuron if already set.
    """
    if neuron.abstraction_level != 0:
        return neuron

    level = DEFAULT_ABSTRACTION_LEVELS.get(neuron.type, 0)
    if level == 0:
        return neuron

    return neuron.with_abstraction_level(level)


def can_activate(source: Neuron, target: Neuron, max_distance: int = 1) -> bool:
    """Return True if spreading activation may cross from source to target.

    The constraint only fires when *both* neurons have a non-zero abstraction
    level. If either is 0 (unassigned), activation is always allowed.

    Args:
        source: The neuron currently being spread from.
        target: The candidate neighbor neuron.
        max_distance: Maximum allowed difference in abstraction levels (default 1).

    Returns:
        True if activation may flow from source to target.
    """
    src_level = source.abstraction_level
    dst_level = target.abstraction_level

    if src_level == 0 or dst_level == 0:
        return True

    return abs(src_level - dst_level) <= max_distance
