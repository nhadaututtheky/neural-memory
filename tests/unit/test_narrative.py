"""Tests for the memory narrative engine."""

from __future__ import annotations

from datetime import timedelta

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.narrative import (
    NarrativeItem,
    generate_causal_narrative,
    generate_timeline_narrative,
    generate_topic_narrative,
)
from neural_memory.utils.timeutils import utcnow


class TestTimelineNarrative:
    """Tests for timeline narrative generation."""

    async def test_empty_timeline(self, storage) -> None:
        """Test timeline with no fibers in range."""
        start = utcnow() - timedelta(days=30)
        end = utcnow() - timedelta(days=20)
        narrative = await generate_timeline_narrative(storage, start, end)
        assert narrative.mode == "timeline"
        assert narrative.items == []
        assert "No memories found" in narrative.markdown

    async def test_timeline_with_fibers(self, storage) -> None:
        """Test timeline returns fibers in time order."""
        now = utcnow()
        # Create two neurons
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="first event", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="second event", neuron_id="n2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Create fibers with time ranges
        f1 = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            time_start=now - timedelta(days=5),
            time_end=now - timedelta(days=4),
            summary="First event happened",
        )
        f2 = Fiber.create(
            neuron_ids={"n2"},
            synapse_ids=set(),
            anchor_neuron_id="n2",
            time_start=now - timedelta(days=2),
            time_end=now - timedelta(days=1),
            summary="Second event happened",
        )
        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        narrative = await generate_timeline_narrative(
            storage,
            start_date=now - timedelta(days=10),
            end_date=now,
        )
        assert narrative.mode == "timeline"
        assert len(narrative.items) == 2
        assert narrative.items[0].summary == "First event happened"
        assert narrative.items[1].summary == "Second event happened"
        assert "Timeline:" in narrative.markdown

    async def test_timeline_respects_max_fibers(self, storage) -> None:
        """Test max_fibers limit is respected."""
        now = utcnow()
        for i in range(5):
            n = Neuron.create(type=NeuronType.CONCEPT, content=f"event {i}", neuron_id=f"n{i}")
            await storage.add_neuron(n)
            f = Fiber.create(
                neuron_ids={f"n{i}"},
                synapse_ids=set(),
                anchor_neuron_id=f"n{i}",
                time_start=now - timedelta(days=5 - i),
                time_end=now - timedelta(days=4 - i),
                summary=f"Event {i}",
            )
            await storage.add_fiber(f)

        narrative = await generate_timeline_narrative(
            storage,
            start_date=now - timedelta(days=10),
            end_date=now,
            max_fibers=3,
        )
        assert len(narrative.items) <= 3


class TestTopicNarrative:
    """Tests for topic narrative generation."""

    async def test_topic_no_results(self, storage, brain_config) -> None:
        """Test topic with no matching memories."""
        narrative = await generate_topic_narrative(storage, brain_config, "nonexistent topic")
        assert narrative.mode == "topic"
        assert narrative.items == []
        assert "No memories found" in narrative.markdown

    async def test_topic_with_results(self, storage, brain_config) -> None:
        """Test topic narrative finds related fibers."""
        # Create neurons and fibers about auth
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="authentication", neuron_id="auth")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="JWT tokens", neuron_id="jwt")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s = Synapse.create(
            source_id="auth", target_id="jwt", type=SynapseType.RELATED_TO, weight=0.8
        )
        await storage.add_synapse(s)

        f = Fiber.create(
            neuron_ids={"auth", "jwt"},
            synapse_ids={s.id},
            anchor_neuron_id="auth",
            summary="Authentication uses JWT tokens",
        )
        await storage.add_fiber(f)

        narrative = await generate_topic_narrative(storage, brain_config, "authentication")
        assert narrative.mode == "topic"
        assert "Topic:" in narrative.markdown


class TestCausalNarrative:
    """Tests for causal narrative generation."""

    async def test_causal_no_seed(self, storage) -> None:
        """Test causal with no matching neurons."""
        narrative = await generate_causal_narrative(storage, "nonexistent")
        assert narrative.mode == "causal"
        assert narrative.items == []
        assert "No causal chain found" in narrative.markdown

    async def test_causal_with_chain(self, storage) -> None:
        """Test causal chain traversal."""
        # Create: outage CAUSED_BY jwt_bug
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="outage", neuron_id="outage")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="JWT bug", neuron_id="jwt_bug")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s = Synapse.create(
            source_id="outage",
            target_id="jwt_bug",
            type=SynapseType.CAUSED_BY,
            weight=0.9,
        )
        await storage.add_synapse(s)

        narrative = await generate_causal_narrative(storage, "outage")
        assert narrative.mode == "causal"
        assert len(narrative.items) >= 1  # At least the seed
        assert "outage" in narrative.items[0].summary
        assert "Causal Chain:" in narrative.markdown


class TestNarrativeItem:
    """Tests for NarrativeItem dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        item = NarrativeItem(fiber_id="f1", timestamp="2026-02-01", summary="test")
        assert item.tags == []
        assert item.relevance == 0.0

    def test_frozen(self) -> None:
        """Test immutability."""
        item = NarrativeItem(fiber_id="f1", timestamp="2026-02-01", summary="test")
        with pytest.raises(AttributeError):
            item.summary = "changed"  # type: ignore[misc]
