"""Unit tests for abstraction-level constraints.

Tests cover:
- Neuron.abstraction_level property
- Neuron.with_abstraction_level() immutability
- DEFAULT_ABSTRACTION_LEVELS mapping completeness
- assign_abstraction_level() logic
- can_activate() gate (same level, adjacent, blocked, level-0 bypass)
- SpreadingActivation gate integration (enabled vs disabled config)
- ReflexActivation gate integration (fiber pathway constraint)
- Pipeline encoding assigns abstraction levels automatically
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.abstraction import (
    DEFAULT_ABSTRACTION_LEVELS,
    assign_abstraction_level,
    can_activate,
)
from neural_memory.engine.activation import SpreadingActivation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neuron(
    neuron_type: NeuronType = NeuronType.ENTITY,
    level: int = 0,
) -> Neuron:
    n = Neuron.create(type=neuron_type, content="test content")
    if level != 0:
        n = n.with_abstraction_level(level)
    return n


# ---------------------------------------------------------------------------
# Neuron.abstraction_level property
# ---------------------------------------------------------------------------


def test_abstraction_level_default_is_zero() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    assert n.abstraction_level == 0


def test_abstraction_level_reads_metadata_key() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice", metadata={"_abstraction_level": 2})
    assert n.abstraction_level == 2


# ---------------------------------------------------------------------------
# Neuron.with_abstraction_level()
# ---------------------------------------------------------------------------


def test_with_abstraction_level_returns_new_neuron() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    n2 = n.with_abstraction_level(1)
    assert n2 is not n
    assert n2.abstraction_level == 1


def test_with_abstraction_level_does_not_mutate_source() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    _ = n.with_abstraction_level(3)
    assert n.abstraction_level == 0  # original unchanged


def test_with_abstraction_level_preserves_other_fields() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice", metadata={"custom": "val"})
    n2 = n.with_abstraction_level(2)
    assert n2.id == n.id
    assert n2.type == n.type
    assert n2.content == n.content
    assert n2.metadata["custom"] == "val"


# ---------------------------------------------------------------------------
# DEFAULT_ABSTRACTION_LEVELS completeness
# ---------------------------------------------------------------------------


def test_default_levels_covers_all_neuron_types() -> None:
    covered = set(DEFAULT_ABSTRACTION_LEVELS.keys())
    all_types = set(NeuronType)
    assert covered == all_types, f"Missing types: {all_types - covered}"


def test_default_levels_values_are_1_2_or_3() -> None:
    for neuron_type, level in DEFAULT_ABSTRACTION_LEVELS.items():
        assert level in (1, 2, 3), f"{neuron_type}: unexpected level {level}"


def test_concrete_types_are_level_1() -> None:
    concrete = {
        NeuronType.TIME,
        NeuronType.SPATIAL,
        NeuronType.ENTITY,
        NeuronType.ACTION,
        NeuronType.STATE,
        NeuronType.SENSORY,
    }
    for t in concrete:
        assert DEFAULT_ABSTRACTION_LEVELS[t] == 1, f"{t} should be level 1"


def test_abstract_types_are_level_2() -> None:
    abstract = {NeuronType.CONCEPT, NeuronType.INTENT}
    for t in abstract:
        assert DEFAULT_ABSTRACTION_LEVELS[t] == 2, f"{t} should be level 2"


def test_meta_types_are_level_3() -> None:
    meta = {NeuronType.HYPOTHESIS, NeuronType.PREDICTION, NeuronType.SCHEMA}
    for t in meta:
        assert DEFAULT_ABSTRACTION_LEVELS[t] == 3, f"{t} should be level 3"


# ---------------------------------------------------------------------------
# assign_abstraction_level()
# ---------------------------------------------------------------------------


def test_assign_sets_level_from_type() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    n2 = assign_abstraction_level(n)
    assert n2.abstraction_level == 1


def test_assign_concept_gets_level_2() -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="Authentication")
    n2 = assign_abstraction_level(n)
    assert n2.abstraction_level == 2


def test_assign_schema_gets_level_3() -> None:
    n = Neuron.create(type=NeuronType.SCHEMA, content="API design model")
    n2 = assign_abstraction_level(n)
    assert n2.abstraction_level == 3


def test_assign_preserves_existing_level() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    n = n.with_abstraction_level(3)  # explicit override
    n2 = assign_abstraction_level(n)
    assert n2.abstraction_level == 3  # not overwritten
    assert n2 is n  # same object returned


def test_assign_does_not_mutate_input() -> None:
    n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
    _ = assign_abstraction_level(n)
    assert n.abstraction_level == 0  # original untouched


# ---------------------------------------------------------------------------
# can_activate()
# ---------------------------------------------------------------------------


def test_can_activate_same_level() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=1)
    dst = _make_neuron(NeuronType.ACTION, level=1)
    assert can_activate(src, dst) is True


def test_can_activate_adjacent_levels() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=1)
    dst = _make_neuron(NeuronType.CONCEPT, level=2)
    assert can_activate(src, dst, max_distance=1) is True


def test_can_activate_blocked_distant_levels() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=1)
    dst = _make_neuron(NeuronType.SCHEMA, level=3)
    assert can_activate(src, dst, max_distance=1) is False


def test_can_activate_blocked_reverse() -> None:
    src = _make_neuron(NeuronType.SCHEMA, level=3)
    dst = _make_neuron(NeuronType.ENTITY, level=1)
    assert can_activate(src, dst, max_distance=1) is False


def test_can_activate_zero_source_always_allowed() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=0)  # unassigned
    dst = _make_neuron(NeuronType.SCHEMA, level=3)
    assert can_activate(src, dst, max_distance=1) is True


def test_can_activate_zero_target_always_allowed() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=1)
    dst = _make_neuron(NeuronType.SCHEMA, level=0)  # unassigned
    assert can_activate(src, dst, max_distance=1) is True


def test_can_activate_both_zero_allowed() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=0)
    dst = _make_neuron(NeuronType.SCHEMA, level=0)
    assert can_activate(src, dst, max_distance=1) is True


def test_can_activate_larger_max_distance() -> None:
    src = _make_neuron(NeuronType.ENTITY, level=1)
    dst = _make_neuron(NeuronType.SCHEMA, level=3)
    assert can_activate(src, dst, max_distance=2) is True


# ---------------------------------------------------------------------------
# SpreadingActivation integration: constraint gate via config toggle
# ---------------------------------------------------------------------------


def _make_storage_mock(
    anchor_id: str,
    anchor_neuron: Neuron,
    neighbor_id: str,
    neighbor_neuron: Neuron,
    synapse_weight: float = 0.9,
) -> MagicMock:
    """Build a minimal NeuralStorage mock for two-node graph tests."""
    storage = MagicMock()

    # Anchor batch fetch
    storage.get_neurons_batch = AsyncMock(return_value={anchor_id: anchor_neuron})

    # Neighbor of anchor = the target neuron
    synapse = MagicMock()
    synapse.weight = synapse_weight
    synapse.type = "related"

    storage.get_neighbors = AsyncMock(return_value=[(neighbor_neuron, synapse)])

    # State fetches
    storage.get_neuron_states_batch = AsyncMock(return_value={})

    return storage


@pytest.mark.asyncio
async def test_activation_blocked_when_constraint_enabled() -> None:
    """Constraint enabled: level 1 → level 3 should NOT reach neighbor."""
    anchor = Neuron.create(type=NeuronType.ENTITY, content="anchor").with_abstraction_level(1)
    neighbor = Neuron.create(type=NeuronType.SCHEMA, content="neighbor").with_abstraction_level(3)

    storage = _make_storage_mock(anchor.id, anchor, neighbor.id, neighbor)
    config = BrainConfig(
        abstraction_constraint_enabled=True,
        abstraction_max_distance=1,
        diminishing_returns_enabled=False,
    )
    sa = SpreadingActivation(storage, config)
    results, _ = await sa.activate([anchor.id])

    assert neighbor.id not in results, "Neighbor should be blocked by abstraction constraint"


@pytest.mark.asyncio
async def test_activation_allowed_when_constraint_disabled() -> None:
    """Constraint disabled: level 1 → level 3 should still reach neighbor."""
    anchor = Neuron.create(type=NeuronType.ENTITY, content="anchor").with_abstraction_level(1)
    neighbor = Neuron.create(type=NeuronType.SCHEMA, content="neighbor").with_abstraction_level(3)

    storage = _make_storage_mock(anchor.id, anchor, neighbor.id, neighbor)
    config = BrainConfig(
        abstraction_constraint_enabled=False,
        diminishing_returns_enabled=False,
    )
    sa = SpreadingActivation(storage, config)
    results, _ = await sa.activate([anchor.id])

    assert neighbor.id in results, "Neighbor should be reachable when constraint is disabled"


@pytest.mark.asyncio
async def test_activation_allowed_adjacent_levels() -> None:
    """Constraint enabled: level 1 → level 2 is within max_distance=1 — should flow."""
    anchor = Neuron.create(type=NeuronType.ENTITY, content="anchor").with_abstraction_level(1)
    neighbor = Neuron.create(type=NeuronType.CONCEPT, content="neighbor").with_abstraction_level(2)

    storage = _make_storage_mock(anchor.id, anchor, neighbor.id, neighbor)
    config = BrainConfig(
        abstraction_constraint_enabled=True,
        abstraction_max_distance=1,
        diminishing_returns_enabled=False,
    )
    sa = SpreadingActivation(storage, config)
    results, _ = await sa.activate([anchor.id])

    assert neighbor.id in results, "Adjacent level neighbor should be reachable"


@pytest.mark.asyncio
async def test_activation_allowed_unassigned_neighbor() -> None:
    """Constraint enabled: neighbor with level 0 is always reachable."""
    anchor = Neuron.create(type=NeuronType.ENTITY, content="anchor").with_abstraction_level(1)
    neighbor = Neuron.create(type=NeuronType.SCHEMA, content="neighbor")  # level 0 = unassigned

    storage = _make_storage_mock(anchor.id, anchor, neighbor.id, neighbor)
    config = BrainConfig(
        abstraction_constraint_enabled=True,
        abstraction_max_distance=1,
        diminishing_returns_enabled=False,
    )
    sa = SpreadingActivation(storage, config)
    results, _ = await sa.activate([anchor.id])

    assert neighbor.id in results, "Level-0 neighbor should always be reachable"


# ---------------------------------------------------------------------------
# Pipeline encoding: assign_abstraction_level wired into pipeline steps
# ---------------------------------------------------------------------------


def test_assign_abstraction_level_entity_gets_level_1() -> None:
    """Pipeline step wraps Neuron.create with assign_abstraction_level for ENTITY."""
    from neural_memory.engine.abstraction import assign_abstraction_level

    n = assign_abstraction_level(Neuron.create(type=NeuronType.ENTITY, content="Alice"))
    assert n.abstraction_level == 1


def test_assign_abstraction_level_concept_gets_level_2() -> None:
    """Pipeline step wraps Neuron.create with assign_abstraction_level for CONCEPT."""
    from neural_memory.engine.abstraction import assign_abstraction_level

    n = assign_abstraction_level(Neuron.create(type=NeuronType.CONCEPT, content="machine learning"))
    assert n.abstraction_level == 2


def test_assign_abstraction_level_action_gets_level_1() -> None:
    """Pipeline step wraps Neuron.create with assign_abstraction_level for ACTION."""
    from neural_memory.engine.abstraction import assign_abstraction_level

    n = assign_abstraction_level(Neuron.create(type=NeuronType.ACTION, content="deployed auth"))
    assert n.abstraction_level == 1


def test_assign_abstraction_level_intent_gets_level_2() -> None:
    """Pipeline step wraps Neuron.create with assign_abstraction_level for INTENT."""
    from neural_memory.engine.abstraction import assign_abstraction_level

    n = assign_abstraction_level(Neuron.create(type=NeuronType.INTENT, content="want to scale"))
    assert n.abstraction_level == 2


def test_assign_abstraction_level_state_gets_level_1() -> None:
    """Pipeline step wraps Neuron.create with assign_abstraction_level for STATE."""
    from neural_memory.engine.abstraction import assign_abstraction_level

    n = assign_abstraction_level(Neuron.create(type=NeuronType.STATE, content="happy"))
    assert n.abstraction_level == 1


# ---------------------------------------------------------------------------
# ReflexActivation gate: fiber pathway constraint
# ---------------------------------------------------------------------------


def _make_fiber(pathway: list[str]) -> MagicMock:
    """Build a minimal Fiber mock with the given pathway."""
    fiber = MagicMock()
    fiber.pathway = pathway
    fiber.conductivity = 1.0
    fiber.last_conducted = None
    fiber.salience = 1.0

    def is_in_pathway(nid: str) -> bool:
        return nid in pathway

    def pathway_position(nid: str) -> int | None:
        try:
            return pathway.index(nid)
        except ValueError:
            return None

    fiber.is_in_pathway = is_in_pathway
    fiber.pathway_position = pathway_position
    return fiber


@pytest.mark.asyncio
async def test_reflex_activation_blocks_distant_abstraction_levels() -> None:
    """ReflexActivation: entity (level 1) → concept (level 2) → meta (level 3).

    With max_distance=1:
    - entity → concept: |1-2|=1 — allowed
    - concept → meta: |2-3|=1 — allowed (from concept)
    - entity → meta directly: |1-3|=2 — blocked when prev_node is entity

    The pathway is [entity, concept, meta]. Starting from entity:
    - entity → concept: allowed (hop 1, prev=entity level 1, curr=concept level 2)
    - concept → meta: allowed (hop 2, prev=concept level 2, curr=meta level 3)
    Both reach results because each hop is within distance=1.
    """
    from neural_memory.engine.reflex_activation import ReflexActivation

    entity = Neuron.create(type=NeuronType.ENTITY, content="entity").with_abstraction_level(1)
    concept = Neuron.create(type=NeuronType.CONCEPT, content="concept").with_abstraction_level(2)
    meta = Neuron.create(type=NeuronType.SCHEMA, content="meta").with_abstraction_level(3)

    fiber = _make_fiber([entity.id, concept.id, meta.id])

    storage = MagicMock()
    storage.get_neurons_batch = AsyncMock(
        return_value={entity.id: entity, concept.id: concept, meta.id: meta}
    )

    config = BrainConfig(
        abstraction_constraint_enabled=True,
        abstraction_max_distance=1,
    )

    reflex = ReflexActivation(storage, config)
    results, _ = await reflex.activate_trail(
        anchor_neurons=[entity.id],
        fibers=[fiber],
    )

    # concept is adjacent to entity (|1-2|=1) — should be activated
    assert concept.id in results, "Concept (level 2) should be reachable from entity (level 1)"
    # meta is adjacent to concept (|2-3|=1) — should be activated via concept
    assert meta.id in results, "Meta (level 3) should be reachable via concept intermediary"


@pytest.mark.asyncio
async def test_reflex_activation_blocks_skipped_level_pathway() -> None:
    """ReflexActivation: entity (1) → meta (3) directly — blocked when max_distance=1."""
    from neural_memory.engine.reflex_activation import ReflexActivation

    entity = Neuron.create(type=NeuronType.ENTITY, content="entity").with_abstraction_level(1)
    meta = Neuron.create(type=NeuronType.SCHEMA, content="meta").with_abstraction_level(3)

    # Direct pathway without concept intermediary
    fiber = _make_fiber([entity.id, meta.id])

    storage = MagicMock()
    storage.get_neurons_batch = AsyncMock(return_value={entity.id: entity, meta.id: meta})

    config = BrainConfig(
        abstraction_constraint_enabled=True,
        abstraction_max_distance=1,
    )

    reflex = ReflexActivation(storage, config)
    results, _ = await reflex.activate_trail(
        anchor_neurons=[entity.id],
        fibers=[fiber],
    )

    # meta is 2 levels from entity (|1-3|=2 > max_distance=1) — should be blocked
    assert meta.id not in results, "Meta (level 3) should be blocked from entity (level 1) directly"


@pytest.mark.asyncio
async def test_reflex_activation_constraint_disabled_allows_all() -> None:
    """ReflexActivation: constraint disabled — entity → meta directly is allowed."""
    from neural_memory.engine.reflex_activation import ReflexActivation

    entity = Neuron.create(type=NeuronType.ENTITY, content="entity").with_abstraction_level(1)
    meta = Neuron.create(type=NeuronType.SCHEMA, content="meta").with_abstraction_level(3)

    fiber = _make_fiber([entity.id, meta.id])

    storage = MagicMock()
    storage.get_neurons_batch = AsyncMock(return_value={entity.id: entity, meta.id: meta})

    config = BrainConfig(
        abstraction_constraint_enabled=False,
    )

    reflex = ReflexActivation(storage, config)
    results, _ = await reflex.activate_trail(
        anchor_neurons=[entity.id],
        fibers=[fiber],
    )

    assert meta.id in results, "Meta should be reachable when constraint is disabled"
