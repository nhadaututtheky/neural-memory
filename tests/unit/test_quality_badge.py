"""Tests for quality badge in diagnostics engine."""

from __future__ import annotations

from datetime import datetime

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.diagnostics import DiagnosticsEngine, QualityBadge
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def empty_storage() -> InMemoryStorage:
    """Storage with a brain but no data."""
    store = InMemoryStorage()
    brain = Brain.create(name="empty", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest_asyncio.fixture
async def rich_storage() -> InMemoryStorage:
    """Storage with diverse data for high quality score."""
    store = InMemoryStorage()
    brain = Brain.create(name="rich", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    neurons = [
        Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1"),
        Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-2"),
        Neuron.create(type=NeuronType.ACTION, content="deployed", neuron_id="n-3"),
        Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-4"),
        Neuron.create(type=NeuronType.TIME, content="monday", neuron_id="n-5"),
        Neuron.create(type=NeuronType.SPATIAL, content="production", neuron_id="n-6"),
    ]
    for n in neurons:
        await store.add_neuron(n)

    synapses = [
        Synapse.create(
            source_id="n-3",
            target_id="n-1",
            type=SynapseType.INVOLVES,
            weight=0.8,
            synapse_id="s-1",
        ),
        Synapse.create(
            source_id="n-3",
            target_id="n-5",
            type=SynapseType.HAPPENED_AT,
            weight=0.7,
            synapse_id="s-2",
        ),
        Synapse.create(
            source_id="n-3",
            target_id="n-6",
            type=SynapseType.AT_LOCATION,
            weight=0.6,
            synapse_id="s-3",
        ),
        Synapse.create(
            source_id="n-1",
            target_id="n-4",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="s-4",
        ),
        Synapse.create(
            source_id="n-1",
            target_id="n-2",
            type=SynapseType.SIMILAR_TO,
            weight=0.4,
            synapse_id="s-5",
        ),
        Synapse.create(
            source_id="n-2",
            target_id="n-4",
            type=SynapseType.CO_OCCURS,
            weight=0.3,
            synapse_id="s-6",
        ),
    ]
    for s in synapses:
        await store.add_synapse(s)

    fiber = Fiber.create(
        neuron_ids={"n-1", "n-2", "n-3", "n-4", "n-5", "n-6"},
        synapse_ids={"s-1", "s-2", "s-3", "s-4", "s-5", "s-6"},
        anchor_neuron_id="n-3",
        fiber_id="f-1",
    )
    await store.add_fiber(fiber)

    return store


class TestQualityBadge:
    """Test QualityBadge computation."""

    @pytest.mark.asyncio
    async def test_badge_frozen(self) -> None:
        """QualityBadge should be immutable."""
        badge = QualityBadge(
            grade="A",
            purity_score=95.0,
            marketplace_eligible=True,
            badge_label="A - Excellent",
            computed_at=datetime.now(),
            component_summary={},
        )
        with pytest.raises(AttributeError):
            badge.grade = "F"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_empty_brain_badge(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain should get F grade and not be marketplace eligible."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        assert badge.grade == "F"
        assert badge.purity_score == 0.0
        assert badge.marketplace_eligible is False
        assert badge.badge_label == "F - Failing"

    @pytest.mark.asyncio
    async def test_rich_brain_badge(self, rich_storage: InMemoryStorage) -> None:
        """Rich brain should get a non-F grade."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        assert badge.grade != "F"
        assert badge.purity_score > 0.0

    @pytest.mark.asyncio
    async def test_marketplace_eligible_threshold(self, rich_storage: InMemoryStorage) -> None:
        """Marketplace eligibility requires grade A or B."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        if badge.grade in ("A", "B"):
            assert badge.marketplace_eligible is True
        else:
            assert badge.marketplace_eligible is False

    @pytest.mark.asyncio
    async def test_component_summary_present(self, rich_storage: InMemoryStorage) -> None:
        """Badge should include component summary with all metrics."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        expected_keys = {
            "connectivity",
            "diversity",
            "freshness",
            "consolidation_ratio",
            "orphan_rate",
            "activation_efficiency",
            "recall_confidence",
        }
        assert set(badge.component_summary.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_computed_at_set(self, empty_storage: InMemoryStorage) -> None:
        """Badge should have computed_at timestamp."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        assert isinstance(badge.computed_at, datetime)

    @pytest.mark.asyncio
    async def test_badge_label_format(self, rich_storage: InMemoryStorage) -> None:
        """Badge label should follow 'X - Description' format."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        assert " - " in badge.badge_label
        assert badge.badge_label.startswith(badge.grade)

    @pytest.mark.asyncio
    async def test_purity_score_range(self, rich_storage: InMemoryStorage) -> None:
        """Purity score should be in [0, 100]."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        badge = await engine.compute_quality_badge(brain_id)

        assert 0.0 <= badge.purity_score <= 100.0
