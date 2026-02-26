"""Tests for v1.0.2 health and correctness fixes.

Covers:
1. Context truncation (retrieval_context.py) — long fibers truncated instead of skipped
2. Diversity metric (diagnostics.py) — normalized against 8 expected types, threshold 3
3. Temporal neighbor synapses (encoder.py) — BEFORE/AFTER instead of RELATED_TO
4. Version bump — __version__ == "1.0.2"
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.diagnostics import DiagnosticsEngine
from neural_memory.engine.pipeline import PipelineContext
from neural_memory.engine.pipeline_steps import TemporalLinkingStep
from neural_memory.engine.retrieval_context import _TOKEN_RATIO, _estimate_tokens, format_context
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow

# ── Helpers ──────────────────────────────────────────────────────


def _make_neuron(content: str, neuron_id: str | None = None) -> Neuron:
    """Create a CONCEPT neuron with given content."""
    return Neuron.create(type=NeuronType.CONCEPT, content=content, neuron_id=neuron_id)


def _make_fiber(
    anchor: Neuron,
    *,
    summary: str | None = None,
    fiber_id: str | None = None,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
) -> Fiber:
    """Create a minimal fiber around an anchor neuron."""
    ts = time_start or utcnow()
    te = time_end or ts
    return Fiber.create(
        neuron_ids={anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=anchor.id,
        time_start=ts,
        time_end=te,
        summary=summary,
        fiber_id=fiber_id,
    )


def _make_activation(neuron_id: str, level: float = 0.8) -> ActivationResult:
    """Create a basic ActivationResult."""
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=1,
        path=[],
        source_anchor=None,
    )


@pytest_asyncio.fixture
async def storage_with_brain() -> InMemoryStorage:
    """InMemoryStorage with a brain context set, ready for operations."""
    store = InMemoryStorage()
    brain = Brain.create(name="test-brain", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


# ── Fix 1: Context truncation ────────────────────────────────────


class TestContextTruncation:
    """Verify format_context truncates long fiber content instead of skipping."""

    @pytest.mark.asyncio
    async def test_context_truncates_long_content(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """A single fiber with 1000-word anchor content and max_tokens=100
        should produce truncated content ending with '...'."""
        store = storage_with_brain
        long_content = " ".join(f"word{i}" for i in range(1000))
        anchor = _make_neuron(long_content, neuron_id="n-long")
        await store.add_neuron(anchor)

        fiber = _make_fiber(anchor, fiber_id="f-long")
        await store.add_fiber(fiber)

        activations = {anchor.id: _make_activation(anchor.id)}
        context, token_est = await format_context(
            storage=store,
            activations=activations,
            fibers=[fiber],
            max_tokens=100,
        )

        # Content should be present (not skipped)
        assert len(context) > 0
        # Should end with truncation marker
        assert "..." in context
        # Should NOT contain all 1000 words
        assert "word999" not in context
        # Token estimate should respect the budget
        assert token_est <= 110  # small overshoot from header is acceptable

    @pytest.mark.asyncio
    async def test_context_fits_short_content(self, storage_with_brain: InMemoryStorage) -> None:
        """A fiber with 10-word anchor content and max_tokens=500
        should include the full content without truncation."""
        store = storage_with_brain
        short_content = " ".join(f"word{i}" for i in range(10))
        anchor = _make_neuron(short_content, neuron_id="n-short")
        await store.add_neuron(anchor)

        fiber = _make_fiber(anchor, fiber_id="f-short")
        await store.add_fiber(fiber)

        activations = {anchor.id: _make_activation(anchor.id)}
        context, token_est = await format_context(
            storage=store,
            activations=activations,
            fibers=[fiber],
            max_tokens=500,
        )

        # Full content should be present
        assert "word0" in context
        assert "word9" in context
        # Should not have truncation marker
        assert "..." not in context

    @pytest.mark.asyncio
    async def test_context_multiple_fibers_budget(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """Three fibers: first moderate, second fits partially, third cut off."""
        store = storage_with_brain

        # Fiber 1: moderate (50 words)
        content1 = " ".join(f"alpha{i}" for i in range(50))
        n1 = _make_neuron(content1, neuron_id="n-1")
        await store.add_neuron(n1)
        f1 = _make_fiber(n1, fiber_id="f-1")
        await store.add_fiber(f1)

        # Fiber 2: moderate (50 words)
        content2 = " ".join(f"beta{i}" for i in range(50))
        n2 = _make_neuron(content2, neuron_id="n-2")
        await store.add_neuron(n2)
        f2 = _make_fiber(n2, fiber_id="f-2")
        await store.add_fiber(f2)

        # Fiber 3: moderate (50 words)
        content3 = " ".join(f"gamma{i}" for i in range(50))
        n3 = _make_neuron(content3, neuron_id="n-3")
        await store.add_neuron(n3)
        f3 = _make_fiber(n3, fiber_id="f-3")
        await store.add_fiber(f3)

        activations = {
            n1.id: _make_activation(n1.id, 0.9),
            n2.id: _make_activation(n2.id, 0.8),
            n3.id: _make_activation(n3.id, 0.7),
        }

        # Budget fits ~first fiber fully, second partially, third cut off
        # 50 words * 1.3 = ~65 tokens per fiber line (plus "- " prefix)
        # Set budget so second fiber must be truncated
        context, token_est = await format_context(
            storage=store,
            activations=activations,
            fibers=[f1, f2, f3],
            max_tokens=120,
        )

        # First fiber should be present
        assert "alpha0" in context
        # Second fiber should be at least partially present (truncated)
        assert "beta0" in context
        # Third fiber content should be absent or mostly absent
        # (budget exhausted after first two)
        # The key invariant: token estimate stays within budget
        assert token_est <= 130  # small overshoot tolerance

    @pytest.mark.asyncio
    async def test_context_empty_fibers(self, storage_with_brain: InMemoryStorage) -> None:
        """Empty fibers list produces empty context."""
        store = storage_with_brain

        context, token_est = await format_context(
            storage=store,
            activations={},
            fibers=[],
            max_tokens=500,
        )

        # No fibers section at all, only possibly Related Information header
        assert "## Relevant Memories" not in context
        assert token_est == 0 or token_est < 50  # header-only at most


# ── Fix 2: Diversity metric ──────────────────────────────────────


class TestDiversityMetric:
    """Verify diversity normalizes against _EXPECTED_SYNAPSE_TYPES = 8."""

    def test_diversity_four_types_reasonable_score(self) -> None:
        """4 synapse types with even distribution should give > 0.5 diversity.

        Previously normalized against 20, giving ~0.33; now normalized
        against 8, giving ~0.67.
        """
        stats: dict[str, Any] = {
            "by_type": {
                "RELATED_TO": {"count": 25},
                "CO_OCCURS": {"count": 25},
                "CAUSED_BY": {"count": 25},
                "LEADS_TO": {"count": 25},
            }
        }
        score = DiagnosticsEngine._compute_diversity(stats)
        # Shannon entropy of uniform(4) = log(4) ~ 1.386
        # Normalized: log(4) / log(8) ~ 1.386 / 2.079 ~ 0.667
        assert score > 0.5
        assert score == pytest.approx(math.log(4) / math.log(8), abs=0.01)

    def test_diversity_warning_threshold_three_types_no_warning(self) -> None:
        """3+ synapse types should NOT trigger LOW_DIVERSITY warning.

        Uses _generate_diagnostics to verify the actual warning logic.
        """
        synapse_stats: dict[str, Any] = {
            "by_type": {
                "RELATED_TO": {"count": 10},
                "CO_OCCURS": {"count": 10},
                "CAUSED_BY": {"count": 10},
            }
        }
        engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
        warnings, _ = engine._generate_diagnostics(
            neuron_count=10,
            synapse_count=30,
            fiber_count=1,
            raw_connectivity=3.0,
            synapse_stats=synapse_stats,
            orphan_rate=0.0,
            consolidation_ratio=1.0,
            freshness=1.0,
            fibers=[],
        )
        codes = {w.code for w in warnings}
        assert "LOW_DIVERSITY" not in codes

    def test_diversity_warning_threshold_two_types_triggers(self) -> None:
        """2 synapse types SHOULD trigger LOW_DIVERSITY warning (types_used < 3).

        Uses _generate_diagnostics to verify the actual warning logic.
        """
        synapse_stats: dict[str, Any] = {
            "by_type": {
                "RELATED_TO": {"count": 50},
                "CO_OCCURS": {"count": 50},
            }
        }
        engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
        warnings, recommendations = engine._generate_diagnostics(
            neuron_count=10,
            synapse_count=100,
            fiber_count=1,
            raw_connectivity=10.0,
            synapse_stats=synapse_stats,
            orphan_rate=0.0,
            consolidation_ratio=1.0,
            freshness=1.0,
            fibers=[],
        )
        codes = {w.code for w in warnings}
        assert "LOW_DIVERSITY" in codes
        # Verify the warning message references the expected count (8)
        low_div_warning = next(w for w in warnings if w.code == "LOW_DIVERSITY")
        assert "8" in low_div_warning.message
        assert low_div_warning.details["types_expected"] == 8

    def test_diversity_expected_types_constant(self) -> None:
        """Verify _EXPECTED_SYNAPSE_TYPES is 8."""
        assert DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES == 8

    def test_diversity_total_synapse_types_matches_enum(self) -> None:
        """_TOTAL_SYNAPSE_TYPES should match the actual enum member count."""
        assert len(SynapseType) == DiagnosticsEngine._TOTAL_SYNAPSE_TYPES
        assert len(SynapseType) == 22


class TestTokenEstimation:
    """Verify token estimation constants and helpers."""

    def test_token_ratio_constant(self) -> None:
        """_TOKEN_RATIO should be 1.3."""
        assert _TOKEN_RATIO == 1.3

    def test_estimate_tokens_basic(self) -> None:
        """_estimate_tokens for 10 words should return 13."""
        text = " ".join(f"word{i}" for i in range(10))
        assert _estimate_tokens(text) == 13  # 10 * 1.3 = 13


# ── Fix 3: Temporal neighbor synapses ────────────────────────────


class TestTemporalNeighborSynapses:
    """Verify TemporalLinkingStep creates BEFORE/AFTER synapses."""

    @pytest_asyncio.fixture
    async def step_and_storage(self) -> tuple[TemporalLinkingStep, InMemoryStorage, BrainConfig]:
        """Set up a TemporalLinkingStep with real InMemoryStorage."""
        store = InMemoryStorage()
        brain = Brain.create(name="temporal-test", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        config = BrainConfig()
        step = TemporalLinkingStep()
        return step, store, config

    def _make_ctx(self, anchor: Neuron, timestamp: datetime) -> PipelineContext:
        return PipelineContext(
            content="",
            timestamp=timestamp,
            metadata={},
            tags=set(),
            language="auto",
            anchor_neuron=anchor,
        )

    @pytest.mark.asyncio
    async def test_temporal_neighbor_before_synapse(
        self, step_and_storage: tuple[TemporalLinkingStep, InMemoryStorage, BrainConfig]
    ) -> None:
        """A fiber created BEFORE the current memory gets AFTER synapse type.

        If older_fiber.time_start < current timestamp, then
        current anchor -> older_fiber.anchor gets synapse type AFTER
        (meaning: current happened AFTER older).
        """
        step, store, config = step_and_storage
        now = utcnow()

        # Create an older fiber (1 hour before 'now')
        older_anchor = _make_neuron("older memory content", neuron_id="n-older")
        await store.add_neuron(older_anchor)
        older_time = now - timedelta(hours=1)
        older_fiber = _make_fiber(
            older_anchor,
            fiber_id="f-older",
            time_start=older_time,
            time_end=older_time,
        )
        await store.add_fiber(older_fiber)

        # Create the current anchor neuron
        current_anchor = _make_neuron("current memory content", neuron_id="n-current")
        await store.add_neuron(current_anchor)

        # Link temporal neighbors via TemporalLinkingStep
        ctx = self._make_ctx(current_anchor, now)
        result_ctx = await step.execute(ctx, store, config)

        # Should have linked to the older fiber
        assert older_anchor.id in result_ctx.neurons_linked

        # Check the created synapse type
        synapses = await store.get_synapses(
            source_id=current_anchor.id,
            target_id=older_anchor.id,
        )
        assert len(synapses) == 1
        assert synapses[0].type == SynapseType.AFTER

    @pytest.mark.asyncio
    async def test_temporal_neighbor_after_synapse(
        self, step_and_storage: tuple[TemporalLinkingStep, InMemoryStorage, BrainConfig]
    ) -> None:
        """A fiber created AFTER the current memory gets BEFORE synapse type.

        If newer_fiber.time_start > current timestamp, then
        current anchor -> newer_fiber.anchor gets synapse type BEFORE
        (meaning: current happened BEFORE newer).
        """
        step, store, config = step_and_storage
        now = utcnow()

        # Create a newer fiber (1 hour after 'now')
        newer_anchor = _make_neuron("newer memory content", neuron_id="n-newer")
        await store.add_neuron(newer_anchor)
        newer_time = now + timedelta(hours=1)
        newer_fiber = _make_fiber(
            newer_anchor,
            fiber_id="f-newer",
            time_start=newer_time,
            time_end=newer_time,
        )
        await store.add_fiber(newer_fiber)

        # Create the current anchor neuron
        current_anchor = _make_neuron("current memory content", neuron_id="n-current")
        await store.add_neuron(current_anchor)

        # Link temporal neighbors via TemporalLinkingStep
        ctx = self._make_ctx(current_anchor, now)
        result_ctx = await step.execute(ctx, store, config)

        # Should have linked to the newer fiber
        assert newer_anchor.id in result_ctx.neurons_linked

        # Check the created synapse type
        synapses = await store.get_synapses(
            source_id=current_anchor.id,
            target_id=newer_anchor.id,
        )
        assert len(synapses) == 1
        assert synapses[0].type == SynapseType.BEFORE

    @pytest.mark.asyncio
    async def test_temporal_neighbor_same_time_related(
        self, step_and_storage: tuple[TemporalLinkingStep, InMemoryStorage, BrainConfig]
    ) -> None:
        """Same timestamp defaults to RELATED_TO.

        When fiber.time_start == current timestamp (neither before nor after),
        the synapse type should be RELATED_TO.
        """
        step, store, config = step_and_storage
        now = utcnow()

        # Create a fiber with the exact same timestamp
        same_anchor = _make_neuron("same-time memory content", neuron_id="n-same")
        await store.add_neuron(same_anchor)
        same_fiber = _make_fiber(
            same_anchor,
            fiber_id="f-same",
            time_start=now,
            time_end=now,
        )
        await store.add_fiber(same_fiber)

        # Create the current anchor neuron
        current_anchor = _make_neuron("current memory content", neuron_id="n-current")
        await store.add_neuron(current_anchor)

        # Link temporal neighbors via TemporalLinkingStep
        ctx = self._make_ctx(current_anchor, now)
        result_ctx = await step.execute(ctx, store, config)

        # Should have linked
        assert same_anchor.id in result_ctx.neurons_linked

        # Check synapse type is RELATED_TO (same timestamp)
        synapses = await store.get_synapses(
            source_id=current_anchor.id,
            target_id=same_anchor.id,
        )
        assert len(synapses) == 1
        assert synapses[0].type == SynapseType.RELATED_TO


# ── Fix 4: Version bump ──────────────────────────────────────────


class TestVersionBump:
    """Verify the package version is current."""

    def test_version_is_current(self) -> None:
        import neural_memory

        assert neural_memory.__version__ == "2.12.0"
