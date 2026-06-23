"""Tests for causal and temporal traversal engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.causal_traversal import (
    CausalChain,
    CausalStep,
    EventSequence,
    EventStep,
    query_temporal_range,
    trace_causal_chain,
    trace_event_sequence,
)
from neural_memory.engine.reconstruction import (
    format_causal_chain,
    format_event_sequence,
    format_temporal_range,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def causal_storage() -> InMemoryStorage:
    """Storage with a causal/temporal graph.

    Graph:
        A("deploy failed") --CAUSED_BY--> B("config error") w=0.8
        B("config error")  --CAUSED_BY--> C("missing env var") w=0.9
        C("missing env var") --LEADS_TO--> B("config error") w=0.9
        B("config error")  --LEADS_TO--> A("deploy failed") w=0.8
        D("service down")  --BEFORE--> A("deploy failed") w=0.7
        A("deploy failed") --BEFORE--> E("user reports") w=0.6
    """
    store = InMemoryStorage()
    brain = Brain.create(name="causal", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    neurons = [
        Neuron.create(type=NeuronType.ACTION, content="deploy failed", neuron_id="n-a"),
        Neuron.create(type=NeuronType.CONCEPT, content="config error", neuron_id="n-b"),
        Neuron.create(type=NeuronType.CONCEPT, content="missing env var", neuron_id="n-c"),
        Neuron.create(type=NeuronType.ACTION, content="service down", neuron_id="n-d"),
        Neuron.create(type=NeuronType.ACTION, content="user reports", neuron_id="n-e"),
    ]
    for n in neurons:
        await store.add_neuron(n)

    synapses = [
        # Causal: A caused by B caused by C
        Synapse.create(source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.8),
        Synapse.create(source_id="n-b", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.9),
        # Effects: C leads to B leads to A
        Synapse.create(source_id="n-c", target_id="n-b", type=SynapseType.LEADS_TO, weight=0.9),
        Synapse.create(source_id="n-b", target_id="n-a", type=SynapseType.LEADS_TO, weight=0.8),
        # Temporal: D before A before E
        Synapse.create(source_id="n-d", target_id="n-a", type=SynapseType.BEFORE, weight=0.7),
        Synapse.create(source_id="n-a", target_id="n-e", type=SynapseType.BEFORE, weight=0.6),
    ]
    for s in synapses:
        await store.add_synapse(s)

    return store


@pytest_asyncio.fixture
async def temporal_storage() -> InMemoryStorage:
    """Storage with fibers spanning different time ranges."""
    store = InMemoryStorage()
    brain = Brain.create(name="temporal", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    base = datetime(2026, 2, 1, 10, 0)
    neurons = [
        Neuron.create(type=NeuronType.ENTITY, content=f"event-{i}", neuron_id=f"tn-{i}")
        for i in range(3)
    ]
    for n in neurons:
        await store.add_neuron(n)

    fibers = [
        Fiber.create(
            neuron_ids={neurons[0].id},
            synapse_ids=set(),
            anchor_neuron_id=neurons[0].id,
            time_start=base,
            time_end=base + timedelta(hours=1),
            summary="Morning standup",
            fiber_id="f-0",
        ),
        Fiber.create(
            neuron_ids={neurons[1].id},
            synapse_ids=set(),
            anchor_neuron_id=neurons[1].id,
            time_start=base + timedelta(days=1),
            time_end=base + timedelta(days=1, hours=2),
            summary="Afternoon deploy",
            fiber_id="f-1",
        ),
        Fiber.create(
            neuron_ids={neurons[2].id},
            synapse_ids=set(),
            anchor_neuron_id=neurons[2].id,
            time_start=base + timedelta(days=5),
            time_end=base + timedelta(days=5, hours=1),
            summary="Postmortem review",
            fiber_id="f-2",
        ),
    ]
    for f in fibers:
        await store.add_fiber(f)

    return store


# ── Data Structure Tests ─────────────────────────────────────────


class TestCausalStep:
    """CausalStep data structure tests."""

    def test_frozen(self) -> None:
        step = CausalStep(
            neuron_id="n-1",
            content="test",
            synapse_type=SynapseType.CAUSED_BY,
            weight=0.8,
            depth=0,
        )
        with pytest.raises(AttributeError):
            step.weight = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        step = CausalStep(
            neuron_id="n-1",
            content="error",
            synapse_type=SynapseType.LEADS_TO,
            weight=0.7,
            depth=2,
        )
        assert step.neuron_id == "n-1"
        assert step.content == "error"
        assert step.synapse_type == SynapseType.LEADS_TO
        assert step.weight == 0.7
        assert step.depth == 2


class TestCausalChain:
    """CausalChain data structure tests."""

    def test_frozen(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-a",
            direction="causes",
            steps=(),
            total_weight=0.0,
        )
        with pytest.raises(AttributeError):
            chain.direction = "effects"  # type: ignore[misc]

    def test_empty_chain(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-a",
            direction="causes",
            steps=(),
            total_weight=0.0,
        )
        assert len(chain.steps) == 0
        assert chain.total_weight == 0.0


class TestEventStep:
    """EventStep data structure tests."""

    def test_frozen(self) -> None:
        event = EventStep(
            neuron_id="n-1",
            content="event",
            fiber_id=None,
            timestamp=None,
            position=0,
        )
        with pytest.raises(AttributeError):
            event.position = 1  # type: ignore[misc]


class TestEventSequence:
    """EventSequence data structure tests."""

    def test_empty_sequence(self) -> None:
        seq = EventSequence(
            seed_neuron_id="n-a",
            direction="forward",
            events=(),
        )
        assert len(seq.events) == 0


# ── Causal Chain Traversal Tests ─────────────────────────────────


class TestTraceCausalChain:
    """Tests for trace_causal_chain function."""

    async def test_linear_chain_causes(self, causal_storage: InMemoryStorage) -> None:
        """A --CAUSED_BY--> B --CAUSED_BY--> C should produce 2-step chain."""
        chain = await trace_causal_chain(causal_storage, "n-a", "causes")
        assert len(chain.steps) == 2
        assert chain.steps[0].content == "config error"
        assert chain.steps[1].content == "missing env var"
        assert chain.direction == "causes"

    async def test_linear_chain_effects(self, causal_storage: InMemoryStorage) -> None:
        """C --LEADS_TO--> B --LEADS_TO--> A should produce 2-step chain."""
        chain = await trace_causal_chain(causal_storage, "n-c", "effects")
        assert len(chain.steps) == 2
        assert chain.steps[0].content == "config error"
        assert chain.steps[1].content == "deploy failed"
        assert chain.direction == "effects"

    async def test_total_weight_is_geometric_mean(self, causal_storage: InMemoryStorage) -> None:
        """Total weight is the geometric mean of step weights (#34).

        The raw product penalized breadth (n weights all <=1.0 collapse toward 0);
        the geometric mean normalizes over the number of steps.
        """
        chain = await trace_causal_chain(causal_storage, "n-a", "causes")
        expected = (0.8 * 0.9) ** 0.5  # geometric mean of the two step weights
        assert abs(chain.total_weight - expected) < 1e-9

    async def test_max_depth_limit(self, causal_storage: InMemoryStorage) -> None:
        """Traversal should stop at max_depth."""
        chain = await trace_causal_chain(causal_storage, "n-a", "causes", max_depth=1)
        assert len(chain.steps) == 1
        assert chain.steps[0].content == "config error"

    async def test_cycle_detection(self, causal_storage: InMemoryStorage) -> None:
        """Cycles should be handled by visited set (not loop forever)."""
        # Add cycle: C --CAUSED_BY--> A (creates A->B->C->A cycle)
        cycle_synapse = Synapse.create(
            source_id="n-c", target_id="n-a", type=SynapseType.CAUSED_BY, weight=0.5
        )
        await causal_storage.add_synapse(cycle_synapse)

        chain = await trace_causal_chain(causal_storage, "n-a", "causes")
        # Should still terminate; visited set prevents revisiting n-a
        neuron_ids = {s.neuron_id for s in chain.steps}
        assert "n-a" not in neuron_ids  # Seed should never appear in steps

    async def test_no_causal_synapses(self, causal_storage: InMemoryStorage) -> None:
        """Neuron with no causal outbound synapses returns empty chain."""
        chain = await trace_causal_chain(causal_storage, "n-c", "causes")
        assert len(chain.steps) == 0
        assert chain.total_weight == 0.0

    async def test_min_weight_filter(self, causal_storage: InMemoryStorage) -> None:
        """Synapses below min_weight should be skipped."""
        chain = await trace_causal_chain(causal_storage, "n-a", "causes", min_weight=0.85)
        # Only B->C (w=0.9) passes, A->B (w=0.8) does not
        assert len(chain.steps) == 0


# ── Temporal Range Tests ─────────────────────────────────────────


class TestQueryTemporalRange:
    """Tests for query_temporal_range function."""

    async def test_fibers_within_range(self, temporal_storage: InMemoryStorage) -> None:
        """Should return fibers overlapping the time range."""
        start = datetime(2026, 2, 1, 0, 0)
        end = datetime(2026, 2, 3, 0, 0)
        fibers = await query_temporal_range(temporal_storage, start, end)
        assert len(fibers) == 2
        assert fibers[0].summary == "Morning standup"
        assert fibers[1].summary == "Afternoon deploy"

    async def test_fibers_outside_range_excluded(self, temporal_storage: InMemoryStorage) -> None:
        """Fibers outside the range should not be returned."""
        start = datetime(2026, 2, 10, 0, 0)
        end = datetime(2026, 2, 11, 0, 0)
        fibers = await query_temporal_range(temporal_storage, start, end)
        assert len(fibers) == 0

    async def test_chronological_ordering(self, temporal_storage: InMemoryStorage) -> None:
        """Results should be sorted by time_start ascending."""
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 12, 31, 0, 0)
        fibers = await query_temporal_range(temporal_storage, start, end)
        assert len(fibers) == 3
        for i in range(len(fibers) - 1):
            assert fibers[i].time_start <= fibers[i + 1].time_start  # type: ignore[operator]

    async def test_empty_storage(self) -> None:
        """Empty brain returns no fibers."""
        store = InMemoryStorage()
        brain = Brain.create(name="empty", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        fibers = await query_temporal_range(store, datetime(2026, 1, 1), datetime(2026, 12, 31))
        assert len(fibers) == 0


# ── Event Sequence Tests ─────────────────────────────────────────


class TestTraceEventSequence:
    """Tests for trace_event_sequence function."""

    async def test_forward_sequence(self, causal_storage: InMemoryStorage) -> None:
        """D --BEFORE--> A --BEFORE--> E should produce forward sequence from D."""
        seq = await trace_event_sequence(causal_storage, "n-d", "forward")
        assert len(seq.events) == 2
        assert seq.events[0].content == "deploy failed"
        assert seq.events[1].content == "user reports"
        assert seq.direction == "forward"

    async def test_forward_from_middle(self, causal_storage: InMemoryStorage) -> None:
        """Starting from A, forward should find E."""
        seq = await trace_event_sequence(causal_storage, "n-a", "forward")
        assert len(seq.events) == 1
        assert seq.events[0].content == "user reports"

    async def test_max_steps_limit(self, causal_storage: InMemoryStorage) -> None:
        """Should stop after max_steps events."""
        seq = await trace_event_sequence(causal_storage, "n-d", "forward", max_steps=1)
        assert len(seq.events) == 1

    async def test_empty_sequence(self, causal_storage: InMemoryStorage) -> None:
        """Neuron with no temporal outbound synapses returns empty sequence."""
        seq = await trace_event_sequence(causal_storage, "n-e", "forward")
        assert len(seq.events) == 0

    async def test_positions_are_sequential(self, causal_storage: InMemoryStorage) -> None:
        """Event positions should be 0-indexed and sequential."""
        seq = await trace_event_sequence(causal_storage, "n-d", "forward")
        for i, event in enumerate(seq.events):
            assert event.position == i


# ── Formatter Tests ──────────────────────────────────────────────


class TestFormatCausalChain:
    """Tests for format_causal_chain."""

    def test_causes_direction(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-a",
            direction="causes",
            steps=(
                CausalStep("n-b", "config error", SynapseType.CAUSED_BY, 0.8, 0),
                CausalStep("n-c", "missing env var", SynapseType.CAUSED_BY, 0.9, 1),
            ),
            total_weight=0.72,
        )
        result = format_causal_chain(chain)
        assert result == "config error because missing env var"

    def test_effects_direction(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-c",
            direction="effects",
            steps=(
                CausalStep("n-b", "config error", SynapseType.LEADS_TO, 0.9, 0),
                CausalStep("n-a", "deploy failed", SynapseType.LEADS_TO, 0.8, 1),
            ),
            total_weight=0.72,
        )
        result = format_causal_chain(chain)
        assert result == "config error leads to deploy failed"

    def test_empty_chain(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-a",
            direction="causes",
            steps=(),
            total_weight=0.0,
        )
        assert format_causal_chain(chain) == ""

    def test_single_step(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n-a",
            direction="causes",
            steps=(CausalStep("n-b", "config error", SynapseType.CAUSED_BY, 0.8, 0),),
            total_weight=0.8,
        )
        assert format_causal_chain(chain) == "config error"


class TestFormatEventSequence:
    """Tests for format_event_sequence."""

    def test_ordered_text(self) -> None:
        seq = EventSequence(
            seed_neuron_id="n-d",
            direction="forward",
            events=(
                EventStep("n-a", "deploy failed", None, None, 0),
                EventStep("n-e", "user reports", None, None, 1),
            ),
        )
        result = format_event_sequence(seq)
        assert result == "First, deploy failed; then user reports"

    def test_with_timestamps(self) -> None:
        ts = datetime(2026, 2, 1, 10, 30)
        seq = EventSequence(
            seed_neuron_id="n-d",
            direction="forward",
            events=(EventStep("n-a", "deploy failed", "f-1", ts, 0),),
        )
        result = format_event_sequence(seq)
        assert "2026-02-01 10:30" in result
        assert "First, deploy failed" in result

    def test_empty_sequence(self) -> None:
        seq = EventSequence(seed_neuron_id="n-a", direction="forward", events=())
        assert format_event_sequence(seq) == ""


class TestFormatTemporalRange:
    """Tests for format_temporal_range."""

    def test_chronological_list(self) -> None:
        fibers = [
            Fiber.create(
                neuron_ids={"n-1"},
                synapse_ids=set(),
                anchor_neuron_id="n-1",
                time_start=datetime(2026, 2, 1, 10, 0),
                summary="Morning standup",
                fiber_id="f-0",
            ),
            Fiber.create(
                neuron_ids={"n-2"},
                synapse_ids=set(),
                anchor_neuron_id="n-2",
                time_start=datetime(2026, 2, 2, 14, 0),
                summary="Afternoon deploy",
                fiber_id="f-1",
            ),
        ]
        result = format_temporal_range(fibers)
        assert "Morning standup" in result
        assert "Afternoon deploy" in result
        assert result.index("Morning") < result.index("Afternoon")

    def test_empty_list(self) -> None:
        assert format_temporal_range([]) == ""

    def test_fiber_without_summary(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n-1"},
            synapse_ids=set(),
            anchor_neuron_id="n-1",
            time_start=datetime(2026, 2, 1, 10, 0),
            fiber_id="f-nosumm",
        )
        result = format_temporal_range([fiber])
        assert "f-nosumm" in result  # Falls back to fiber ID prefix
