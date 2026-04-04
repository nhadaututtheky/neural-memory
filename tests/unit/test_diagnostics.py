"""Tests for brain health diagnostics engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.diagnostics import (
    BrainHealthReport,
    DiagnosticsEngine,
    DiagnosticWarning,
    WarningSeverity,
    _score_to_grade,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


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
    """Storage with diverse neurons, synapses, and fresh fibers."""
    store = InMemoryStorage()
    brain = Brain.create(name="rich", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Create neurons of various types
    neurons = [
        Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1"),
        Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-2"),
        Neuron.create(type=NeuronType.ACTION, content="deployed", neuron_id="n-3"),
        Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-4"),
        Neuron.create(type=NeuronType.TIME, content="monday", neuron_id="n-5"),
        Neuron.create(type=NeuronType.SPATIAL, content="production", neuron_id="n-6"),
        Neuron.create(type=NeuronType.STATE, content="frustration", neuron_id="n-7"),
        Neuron.create(type=NeuronType.INTENT, content="optimize", neuron_id="n-8"),
    ]
    for n in neurons:
        await store.add_neuron(n)

    # Create synapses of diverse types (>= 5 types)
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
            source_id="n-4", target_id="n-8", type=SynapseType.ENABLES, weight=0.6, synapse_id="s-6"
        ),
        Synapse.create(
            source_id="n-3", target_id="n-7", type=SynapseType.FELT, weight=0.5, synapse_id="s-7"
        ),
        Synapse.create(
            source_id="n-2",
            target_id="n-4",
            type=SynapseType.CO_OCCURS,
            weight=0.3,
            synapse_id="s-8",
        ),
    ]
    for s in synapses:
        await store.add_synapse(s)

    # Activate some neurons (for activation_efficiency)
    for nid in ["n-1", "n-3", "n-4", "n-5", "n-6"]:
        state = await store.get_neuron_state(nid)
        if state:
            activated = state.activate(0.5)
            await store.update_neuron_state(activated)

    # Create fresh fibers
    fiber1 = Fiber.create(
        neuron_ids={"n-1", "n-3", "n-4", "n-5"},
        synapse_ids={"s-1", "s-2", "s-4"},
        anchor_neuron_id="n-3",
        fiber_id="f-1",
        tags={"redis", "caching", "deployment"},
    )
    await store.add_fiber(fiber1)

    fiber2 = Fiber.create(
        neuron_ids={"n-2", "n-3", "n-6", "n-8"},
        synapse_ids={"s-3", "s-5", "s-6"},
        anchor_neuron_id="n-3",
        fiber_id="f-2",
        tags={"postgresql", "database", "optimization"},
    )
    await store.add_fiber(fiber2)

    return store


# ── Grade mapping tests ──────────────────────────────────────────


class TestGradeMapping:
    """Test score-to-grade conversion."""

    def test_grade_a(self) -> None:
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(90) == "A"

    def test_grade_b(self) -> None:
        assert _score_to_grade(85) == "B"
        assert _score_to_grade(75) == "B"

    def test_grade_c(self) -> None:
        assert _score_to_grade(70) == "C"
        assert _score_to_grade(60) == "C"

    def test_grade_d(self) -> None:
        assert _score_to_grade(55) == "D"
        assert _score_to_grade(40) == "D"

    def test_grade_f(self) -> None:
        assert _score_to_grade(30) == "F"
        assert _score_to_grade(0) == "F"

    def test_boundary_values(self) -> None:
        assert _score_to_grade(89.9) == "B"
        assert _score_to_grade(90.0) == "A"
        assert _score_to_grade(74.9) == "C"
        assert _score_to_grade(75.0) == "B"


class TestBrainHealthReport:
    """Test report dataclass properties."""

    def test_frozen(self) -> None:
        """BrainHealthReport should be immutable."""
        report = BrainHealthReport(
            purity_score=50.0,
            grade="D",
            connectivity=0.5,
            diversity=0.3,
            freshness=0.8,
            consolidation_ratio=0.0,
            orphan_rate=0.1,
            activation_efficiency=0.6,
            recall_confidence=0.4,
            neuron_count=10,
            synapse_count=15,
            fiber_count=3,
            warnings=(),
            recommendations=(),
        )
        with pytest.raises(AttributeError):
            report.grade = "A"  # type: ignore[misc]

    def test_warning_frozen(self) -> None:
        """DiagnosticWarning should be immutable."""
        warning = DiagnosticWarning(
            severity=WarningSeverity.WARNING,
            code="TEST",
            message="Test warning",
        )
        with pytest.raises(AttributeError):
            warning.code = "CHANGED"  # type: ignore[misc]


# ── Engine tests ─────────────────────────────────────────────────


class TestDiagnosticsEmpty:
    """Test diagnostics on empty brain."""

    @pytest.mark.asyncio
    async def test_empty_brain_grade_f(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain should get grade F."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert report.grade == "F"
        assert report.purity_score == 0.0

    @pytest.mark.asyncio
    async def test_empty_brain_warning(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain should have EMPTY_BRAIN warning."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        report = await engine.analyze(brain_id)

        codes = {w.code for w in report.warnings}
        assert "EMPTY_BRAIN" in codes

    @pytest.mark.asyncio
    async def test_empty_brain_recommendation(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain should have a recommendation to start storing memories."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert len(report.recommendations) >= 1
        assert any("nmem_remember" in r for r in report.recommendations)

    @pytest.mark.asyncio
    async def test_empty_brain_zero_metrics(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain metrics should all be zero."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert report.connectivity == 0.0
        assert report.diversity == 0.0
        assert report.freshness == 0.0
        assert report.orphan_rate == 0.0


class TestDiagnosticsPopulated:
    """Test diagnostics on populated brain."""

    @pytest.mark.asyncio
    async def test_metrics_valid_ranges(self, rich_storage: InMemoryStorage) -> None:
        """All metrics should be in [0.0, 1.0]."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert 0.0 <= report.connectivity <= 1.0
        assert 0.0 <= report.diversity <= 1.0
        assert 0.0 <= report.freshness <= 1.0
        assert 0.0 <= report.consolidation_ratio <= 1.0
        assert 0.0 <= report.orphan_rate <= 1.0
        assert 0.0 <= report.activation_efficiency <= 1.0
        assert 0.0 <= report.recall_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_purity_in_range(self, rich_storage: InMemoryStorage) -> None:
        """Purity score should be in [0, 100]."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert 0.0 <= report.purity_score <= 100.0

    @pytest.mark.asyncio
    async def test_grade_not_f(self, rich_storage: InMemoryStorage) -> None:
        """Rich storage should not get grade F."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert report.grade != "F"

    @pytest.mark.asyncio
    async def test_counts_match(self, rich_storage: InMemoryStorage) -> None:
        """Report counts should match storage."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert report.neuron_count == 8
        assert report.synapse_count == 8
        assert report.fiber_count == 2

    @pytest.mark.asyncio
    async def test_freshness_all_recent(self, rich_storage: InMemoryStorage) -> None:
        """All fibers created now should give freshness 1.0."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert report.freshness == 1.0

    @pytest.mark.asyncio
    async def test_activation_efficiency_partial(self, rich_storage: InMemoryStorage) -> None:
        """5 of 8 neurons activated should give ~0.625."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        # 5 activated out of 8 total
        assert 0.5 <= report.activation_efficiency <= 0.75

    @pytest.mark.asyncio
    async def test_diversity_multiple_types(self, rich_storage: InMemoryStorage) -> None:
        """8 synapse types should give high diversity."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        # 8 distinct types out of 29 — moderate diversity
        assert report.diversity > 0.3


class TestConnectivity:
    """Test connectivity metric calculation."""

    def test_zero_neurons(self) -> None:
        assert DiagnosticsEngine._compute_connectivity(10, 0) == 0.0

    def test_low_connectivity(self) -> None:
        """1 synapse per neuron should give low score."""
        score = DiagnosticsEngine._compute_connectivity(10, 10)
        assert score < 0.2

    def test_ideal_connectivity(self) -> None:
        """5 synapses per neuron should give medium-high score."""
        score = DiagnosticsEngine._compute_connectivity(50, 10)
        assert 0.7 < score < 1.0

    def test_high_connectivity(self) -> None:
        """10 synapses per neuron should give very high score."""
        score = DiagnosticsEngine._compute_connectivity(100, 10)
        assert score > 0.95


class TestDiversity:
    """Test diversity metric calculation."""

    def test_no_synapses(self) -> None:
        assert DiagnosticsEngine._compute_diversity({}) == 0.0

    def test_single_type(self) -> None:
        """All synapses of one type should give diversity 0."""
        stats = {"by_type": {"RELATED_TO": {"count": 100}}}
        score = DiagnosticsEngine._compute_diversity(stats)
        assert score == 0.0

    def test_two_types_equal(self) -> None:
        """Two equally distributed types should give moderate diversity."""
        stats = {"by_type": {"RELATED_TO": {"count": 50}, "CO_OCCURS": {"count": 50}}}
        score = DiagnosticsEngine._compute_diversity(stats)
        assert 0.1 < score < 0.5

    def test_many_types(self) -> None:
        """Many types should give higher diversity."""
        stats = {
            "by_type": {
                "RELATED_TO": {"count": 20},
                "CO_OCCURS": {"count": 15},
                "CAUSED_BY": {"count": 10},
                "LEADS_TO": {"count": 10},
                "INVOLVES": {"count": 15},
                "HAPPENED_AT": {"count": 10},
                "AT_LOCATION": {"count": 10},
                "FELT": {"count": 5},
                "SIMILAR_TO": {"count": 5},
            }
        }
        score = DiagnosticsEngine._compute_diversity(stats)
        assert score > 0.5


class TestPurityFormula:
    """Test that purity formula produces expected values."""

    @pytest.mark.asyncio
    async def test_purity_components(self, rich_storage: InMemoryStorage) -> None:
        """Verify purity equals weighted sum of components."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        expected = (
            report.connectivity * 0.25
            + report.diversity * 0.20
            + report.freshness * 0.15
            + report.consolidation_ratio * 0.15
            + (1.0 - report.orphan_rate) * 0.10
            + report.activation_efficiency * 0.10
            + report.recall_confidence * 0.05
        ) * 100

        assert report.purity_score == pytest.approx(expected, abs=0.2)


class TestWarnings:
    """Test warning generation."""

    @pytest.mark.asyncio
    async def test_stale_brain_warning(self) -> None:
        """Brain with only old fibers should get STALE_BRAIN warning."""
        store = InMemoryStorage()
        brain = Brain.create(name="stale", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        # Add a neuron and synapse so it's not "empty"
        n1 = Neuron.create(type=NeuronType.ENTITY, content="old", neuron_id="n-old")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="thing", neuron_id="n-thing")
        await store.add_neuron(n1)
        await store.add_neuron(n2)
        s = Synapse.create(
            source_id="n-old",
            target_id="n-thing",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="s-old",
        )
        await store.add_synapse(s)

        # Create a fiber with old timestamp
        old_time = datetime.now() - timedelta(days=30)
        fiber = Fiber.create(
            neuron_ids={"n-old", "n-thing"},
            synapse_ids={"s-old"},
            anchor_neuron_id="n-old",
            fiber_id="f-old",
        )
        # Override created_at to be old
        fiber = Fiber(
            id=fiber.id,
            neuron_ids=fiber.neuron_ids,
            synapse_ids=fiber.synapse_ids,
            anchor_neuron_id=fiber.anchor_neuron_id,
            pathway=fiber.pathway,
            conductivity=fiber.conductivity,
            last_conducted=None,
            time_start=None,
            time_end=None,
            frequency=fiber.frequency,
            salience=fiber.salience,
            coherence=fiber.coherence,
            summary=None,
            auto_tags=frozenset(),
            agent_tags=frozenset(),
            metadata={},
            created_at=old_time,
        )
        await store.add_fiber(fiber)

        engine = DiagnosticsEngine(store)
        report = await engine.analyze(brain.id)
        codes = {w.code for w in report.warnings}
        assert "STALE_BRAIN" in codes

    @pytest.mark.asyncio
    async def test_low_connectivity_warning(self) -> None:
        """Brain with few synapses per neuron should get LOW_CONNECTIVITY."""
        store = InMemoryStorage()
        brain = Brain.create(name="sparse", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        # 10 neurons, 5 synapses (0.5 per neuron)
        for i in range(10):
            n = Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
            await store.add_neuron(n)
        for i in range(5):
            s = Synapse.create(
                source_id=f"n-{i}",
                target_id=f"n-{i + 1}",
                type=SynapseType.RELATED_TO,
                weight=0.5,
                synapse_id=f"s-{i}",
            )
            await store.add_synapse(s)

        fiber = Fiber.create(
            neuron_ids={f"n-{i}" for i in range(10)},
            synapse_ids={f"s-{i}" for i in range(5)},
            anchor_neuron_id="n-0",
            fiber_id="f-sparse",
        )
        await store.add_fiber(fiber)

        engine = DiagnosticsEngine(store)
        report = await engine.analyze(brain.id)
        codes = {w.code for w in report.warnings}
        assert "LOW_CONNECTIVITY" in codes

    @pytest.mark.asyncio
    async def test_low_diversity_warning(self) -> None:
        """Brain with only one synapse type should get LOW_DIVERSITY."""
        store = InMemoryStorage()
        brain = Brain.create(name="mono", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        n1 = Neuron.create(type=NeuronType.ENTITY, content="a", neuron_id="n-a")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="b", neuron_id="n-b")
        n3 = Neuron.create(type=NeuronType.ENTITY, content="c", neuron_id="n-c")
        await store.add_neuron(n1)
        await store.add_neuron(n2)
        await store.add_neuron(n3)

        # All RELATED_TO synapses (1 type)
        for i, (src, tgt) in enumerate([("n-a", "n-b"), ("n-b", "n-c"), ("n-a", "n-c")]):
            s = Synapse.create(
                source_id=src,
                target_id=tgt,
                type=SynapseType.RELATED_TO,
                weight=0.5,
                synapse_id=f"s-{i}",
            )
            await store.add_synapse(s)

        fiber = Fiber.create(
            neuron_ids={"n-a", "n-b", "n-c"},
            synapse_ids={"s-0", "s-1", "s-2"},
            anchor_neuron_id="n-a",
            fiber_id="f-mono",
        )
        await store.add_fiber(fiber)

        engine = DiagnosticsEngine(store)
        report = await engine.analyze(brain.id)
        codes = {w.code for w in report.warnings}
        assert "LOW_DIVERSITY" in codes

    @pytest.mark.asyncio
    async def test_high_orphan_rate_warning(self) -> None:
        """Brain with neurons not in any synapse or fiber should get HIGH_ORPHAN_RATE."""
        store = InMemoryStorage()
        brain = Brain.create(name="orphans", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        # 10 neurons, only 2 connected via synapse, only 3 in fiber
        # Remaining 7 are truly orphaned (no synapse, no fiber)
        for i in range(10):
            n = Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
            await store.add_neuron(n)

        s = Synapse.create(
            source_id="n-0",
            target_id="n-1",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="s-only",
        )
        await store.add_synapse(s)

        # Fiber only covers n-0, n-1, n-2 — leaves n-3..n-9 truly orphaned
        fiber = Fiber.create(
            neuron_ids={"n-0", "n-1", "n-2"},
            synapse_ids={"s-only"},
            anchor_neuron_id="n-0",
            fiber_id="f-orphans",
        )
        await store.add_fiber(fiber)

        engine = DiagnosticsEngine(store)
        report = await engine.analyze(brain.id)
        codes = {w.code for w in report.warnings}
        assert "HIGH_ORPHAN_RATE" in codes
        # 7 out of 10 neurons are truly orphaned = 70%
        assert report.orphan_rate > 0.20

    @pytest.mark.asyncio
    async def test_fiber_membership_reduces_orphan_rate(self) -> None:
        """Neurons in fibers should not count as orphans even without synapses."""
        store = InMemoryStorage()
        brain = Brain.create(name="fiber_linked", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        # 10 neurons, only 2 connected via synapse, but ALL in fiber
        for i in range(10):
            n = Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
            await store.add_neuron(n)

        s = Synapse.create(
            source_id="n-0",
            target_id="n-1",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="s-only",
        )
        await store.add_synapse(s)

        # Fiber covers all 10 neurons
        fiber = Fiber.create(
            neuron_ids={f"n-{i}" for i in range(10)},
            synapse_ids={"s-only"},
            anchor_neuron_id="n-0",
            fiber_id="f-all",
        )
        await store.add_fiber(fiber)

        engine = DiagnosticsEngine(store)
        report = await engine.analyze(brain.id)
        # All neurons are in the fiber, so orphan rate should be 0
        assert report.orphan_rate == 0.0
        codes = {w.code for w in report.warnings}
        assert "HIGH_ORPHAN_RATE" not in codes

    @pytest.mark.asyncio
    async def test_no_consolidation_warning(self, rich_storage: InMemoryStorage) -> None:
        """Brain with no SEMANTIC fibers should get LOW_CONSOLIDATION."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        codes = {w.code for w in report.warnings}
        assert "LOW_CONSOLIDATION" in codes

    @pytest.mark.asyncio
    async def test_no_stale_warning_when_fresh(self, rich_storage: InMemoryStorage) -> None:
        """Fresh brain should not have STALE_BRAIN warning."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        codes = {w.code for w in report.warnings}
        assert "STALE_BRAIN" not in codes

    @pytest.mark.asyncio
    async def test_no_empty_warning_when_populated(self, rich_storage: InMemoryStorage) -> None:
        """Populated brain should not have EMPTY_BRAIN warning."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        codes = {w.code for w in report.warnings}
        assert "EMPTY_BRAIN" not in codes


class TestRecommendations:
    """Test that warnings produce matching recommendations."""

    @pytest.mark.asyncio
    async def test_empty_brain_has_recommendation(self, empty_storage: InMemoryStorage) -> None:
        """Empty brain should suggest storing memories."""
        engine = DiagnosticsEngine(empty_storage)
        brain_id = empty_storage._current_brain_id
        report = await engine.analyze(brain_id)

        assert len(report.recommendations) >= 1

    @pytest.mark.asyncio
    async def test_each_warning_has_recommendation(self, rich_storage: InMemoryStorage) -> None:
        """Each non-info warning should produce at least one recommendation."""
        engine = DiagnosticsEngine(rich_storage)
        brain_id = rich_storage._current_brain_id
        report = await engine.analyze(brain_id)

        non_info_warnings = [w for w in report.warnings if w.severity != WarningSeverity.INFO]
        if non_info_warnings:
            assert len(report.recommendations) >= 1
