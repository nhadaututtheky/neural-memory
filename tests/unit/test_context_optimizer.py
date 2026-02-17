"""Tests for Smart Context Optimizer (Feature A)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.engine.context_optimizer import (
    ContextItem,
    ContextPlan,
    allocate_token_budgets,
    compute_composite_score,
    deduplicate_by_simhash,
    optimize_context,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


# ── Helpers ──────────────────────────────────────────────────────


def _make_neuron(neuron_id: str, content: str, content_hash: int = 0) -> Neuron:
    """Create a simple neuron for testing."""
    return Neuron(
        id=neuron_id,
        type=NeuronType.SENSORY,
        content=content,
        content_hash=content_hash,
        created_at=utcnow(),
    )


def _make_fiber(
    anchor_neuron_id: str,
    summary: str | None = None,
    frequency: int = 0,
    conductivity: float = 0.5,
    fiber_id: str | None = None,
) -> Fiber:
    """Create a fiber with minimal boilerplate."""
    fiber = Fiber.create(
        neuron_ids={anchor_neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_neuron_id,
        summary=summary,
        fiber_id=fiber_id,
    )
    if frequency != 0 or conductivity != 0.5:
        fiber = replace(fiber, frequency=frequency, conductivity=conductivity)
    return fiber


# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """Storage with a brain context, ready for test data."""
    storage = InMemoryStorage()
    brain = Brain.create(name="ctx-opt-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# ── Data Structure Tests ─────────────────────────────────────────


class TestContextItem:
    """ContextItem frozen dataclass tests."""

    def test_frozen(self) -> None:
        item = ContextItem(fiber_id="f1", content="hello", score=0.5, token_count=10)
        with pytest.raises(AttributeError):
            item.score = 0.9  # type: ignore[misc]

    def test_defaults(self) -> None:
        item = ContextItem(fiber_id="f1", content="hello", score=0.5, token_count=10)
        assert item.truncated is False

    def test_fields(self) -> None:
        item = ContextItem(
            fiber_id="f1", content="hello world", score=0.85, token_count=3, truncated=True
        )
        assert item.fiber_id == "f1"
        assert item.content == "hello world"
        assert item.score == 0.85
        assert item.token_count == 3
        assert item.truncated is True


class TestContextPlan:
    """ContextPlan frozen dataclass tests."""

    def test_frozen(self) -> None:
        plan = ContextPlan(items=[], total_tokens=0, dropped_count=0)
        with pytest.raises(AttributeError):
            plan.total_tokens = 100  # type: ignore[misc]

    def test_empty(self) -> None:
        plan = ContextPlan(items=[], total_tokens=0, dropped_count=0)
        assert len(plan.items) == 0
        assert plan.dropped_count == 0


# ── Composite Score Tests ────────────────────────────────────────


class TestCompositeScore:
    """Tests for compute_composite_score."""

    def test_all_zeros(self) -> None:
        score = compute_composite_score(
            activation=0.0, priority=0.0, frequency=0.0, conductivity=0.0, freshness=0.0
        )
        assert score == 0.0

    def test_all_ones(self) -> None:
        score = compute_composite_score(
            activation=1.0, priority=1.0, frequency=1.0, conductivity=1.0, freshness=1.0
        )
        assert score == pytest.approx(1.0)

    def test_weight_distribution(self) -> None:
        # Activation has highest weight (0.30)
        score_high_activation = compute_composite_score(activation=1.0)
        score_high_freshness = compute_composite_score(freshness=1.0)
        assert score_high_activation > score_high_freshness

    def test_clamped_above_one(self) -> None:
        score = compute_composite_score(
            activation=2.0, priority=2.0, frequency=2.0, conductivity=2.0, freshness=2.0
        )
        assert score == pytest.approx(1.0)

    def test_defaults(self) -> None:
        score = compute_composite_score()
        expected = 0.30 * 0.0 + 0.25 * 0.5 + 0.20 * 0.0 + 0.15 * 0.5 + 0.10 * 0.5
        assert score == pytest.approx(expected)


# ── Deduplication Tests ──────────────────────────────────────────


class TestDeduplication:
    """Tests for deduplicate_by_simhash."""

    def test_no_hashes_keeps_all(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="hello", score=0.9, token_count=5),
            ContextItem(fiber_id="f2", content="hello", score=0.8, token_count=5),
        ]
        result = deduplicate_by_simhash(items, {})
        assert len(result) == 2

    def test_identical_hashes_dedup(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="hello world test", score=0.9, token_count=5),
            ContextItem(fiber_id="f2", content="hello world test", score=0.8, token_count=5),
        ]
        hashes = {"f1": 12345, "f2": 12345}
        result = deduplicate_by_simhash(items, hashes)
        assert len(result) == 1
        assert result[0].fiber_id == "f1"

    def test_different_hashes_keep_all(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="alpha", score=0.9, token_count=3),
            ContextItem(fiber_id="f2", content="beta", score=0.8, token_count=3),
        ]
        hashes = {"f1": 0, "f2": 0xFFFFFFFFFFFFFFFF}
        result = deduplicate_by_simhash(items, hashes)
        assert len(result) == 2

    def test_zero_hash_kept(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="test", score=0.9, token_count=3),
        ]
        hashes = {"f1": 0}
        result = deduplicate_by_simhash(items, hashes)
        assert len(result) == 1


# ── Token Budget Allocation Tests ────────────────────────────────


class TestAllocateTokenBudgets:
    """Tests for allocate_token_budgets."""

    def test_empty_items(self) -> None:
        result = allocate_token_budgets([], max_tokens=1000)
        assert result == []

    def test_all_fit(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="hello world", score=0.9, token_count=10),
            ContextItem(fiber_id="f2", content="test data", score=0.5, token_count=8),
        ]
        result = allocate_token_budgets(items, max_tokens=1000)
        assert len(result) == 2
        assert not result[0].truncated
        assert not result[1].truncated

    def test_budget_overflow_truncates(self) -> None:
        long_content = " ".join(["word"] * 200)
        items = [
            ContextItem(fiber_id="f1", content=long_content, score=0.9, token_count=260),
        ]
        result = allocate_token_budgets(items, max_tokens=50)
        assert len(result) == 1
        assert result[0].truncated is True
        assert result[0].token_count <= 260

    def test_drops_when_no_room(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="first item", score=0.9, token_count=90),
            ContextItem(fiber_id="f2", content="second item", score=0.1, token_count=90),
        ]
        result = allocate_token_budgets(items, max_tokens=100)
        assert len(result) == 1
        assert result[0].fiber_id == "f1"

    def test_min_budget_enforced(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="a b c d e", score=0.9, token_count=7),
        ]
        result = allocate_token_budgets(items, max_tokens=100, min_budget=20)
        assert len(result) == 1


# ── Integration: optimize_context ────────────────────────────────


class TestOptimizeContext:
    """Integration tests for the full optimize_context pipeline."""

    async def test_empty_fibers(self, store: InMemoryStorage) -> None:
        plan = await optimize_context(store, [], max_tokens=500)
        assert plan.items == []
        assert plan.total_tokens == 0
        assert plan.dropped_count == 0

    async def test_single_fiber_with_summary(self, store: InMemoryStorage) -> None:
        now = utcnow()
        neuron = _make_neuron("n1", "anchor content")
        await store.add_neuron(neuron)

        fiber = _make_fiber("n1", summary="This is a test memory")
        await store.add_fiber(fiber)

        plan = await optimize_context(store, [fiber], max_tokens=500, reference_time=now)
        assert len(plan.items) == 1
        assert plan.items[0].content == "This is a test memory"
        assert plan.items[0].score > 0

    async def test_fiber_without_summary_uses_anchor(self, store: InMemoryStorage) -> None:
        now = utcnow()
        neuron = _make_neuron("n1", "Anchor neuron content", content_hash=42)
        await store.add_neuron(neuron)

        fiber = _make_fiber("n1")
        await store.add_fiber(fiber)

        plan = await optimize_context(store, [fiber], max_tokens=500, reference_time=now)
        assert len(plan.items) == 1
        assert plan.items[0].content == "Anchor neuron content"

    async def test_scoring_uses_activation(self, store: InMemoryStorage) -> None:
        now = utcnow()

        n1 = _make_neuron("n1", "high activation")
        n2 = _make_neuron("n2", "low activation")
        await store.add_neuron(n1)
        await store.add_neuron(n2)

        state_high = NeuronState(neuron_id="n1", activation_level=0.9)
        state_low = NeuronState(neuron_id="n2", activation_level=0.1)
        await store.update_neuron_state(state_high)
        await store.update_neuron_state(state_low)

        f1 = _make_fiber("n1")
        f2 = _make_fiber("n2")
        await store.add_fiber(f1)
        await store.add_fiber(f2)

        plan = await optimize_context(store, [f1, f2], max_tokens=5000, reference_time=now)
        assert len(plan.items) == 2
        assert plan.items[0].content == "high activation"
        assert plan.items[0].score > plan.items[1].score

    async def test_scoring_uses_priority(self, store: InMemoryStorage) -> None:
        now = utcnow()

        n1 = _make_neuron("n-crit", "critical memory")
        n2 = _make_neuron("n-low", "normal memory")
        await store.add_neuron(n1)
        await store.add_neuron(n2)

        f1 = _make_fiber("n-crit", summary="critical memory")
        f2 = _make_fiber("n-low", summary="normal memory")
        await store.add_fiber(f1)
        await store.add_fiber(f2)

        tm1 = TypedMemory(fiber_id=f1.id, memory_type=MemoryType.FACT, priority=Priority.CRITICAL)
        tm2 = TypedMemory(fiber_id=f2.id, memory_type=MemoryType.FACT, priority=Priority.LOW)
        await store.add_typed_memory(tm1)
        await store.add_typed_memory(tm2)

        plan = await optimize_context(store, [f1, f2], max_tokens=500, reference_time=now)
        assert len(plan.items) == 2
        assert plan.items[0].content == "critical memory"

    async def test_scoring_uses_frequency(self, store: InMemoryStorage) -> None:
        now = utcnow()

        n1 = _make_neuron("n-freq", "frequent memory")
        n2 = _make_neuron("n-rare", "rare memory")
        await store.add_neuron(n1)
        await store.add_neuron(n2)

        f1 = _make_fiber("n-freq", summary="frequent memory", frequency=20)
        f2 = _make_fiber("n-rare", summary="rare memory", frequency=1)
        await store.add_fiber(f1)
        await store.add_fiber(f2)

        plan = await optimize_context(store, [f1, f2], max_tokens=500, reference_time=now)
        assert len(plan.items) == 2
        assert plan.items[0].content == "frequent memory"

    async def test_dropped_count(self, store: InMemoryStorage) -> None:
        now = utcnow()

        fibers = []
        for i in range(10):
            content = " ".join(["word"] * 50)  # ~65 tokens each
            nid = f"n-drop-{i}"
            neuron = _make_neuron(nid, f"neuron content {i}")
            await store.add_neuron(neuron)
            f = _make_fiber(nid, summary=content)
            await store.add_fiber(f)
            fibers.append(f)

        plan = await optimize_context(store, fibers, max_tokens=100, reference_time=now)
        assert plan.dropped_count > 0
        assert plan.total_tokens <= 100

    async def test_fibers_without_content_skipped(self, store: InMemoryStorage) -> None:
        now = utcnow()

        # Neuron for fiber without summary — anchor has no content in fiber but neuron exists
        n1 = _make_neuron("n-empty", "")
        n2 = _make_neuron("n-content", "has content")
        await store.add_neuron(n1)
        await store.add_neuron(n2)

        f1 = _make_fiber("n-empty")  # No summary, anchor neuron content is empty
        f2 = _make_fiber("n-content", summary="has content")
        await store.add_fiber(f1)
        await store.add_fiber(f2)

        plan = await optimize_context(store, [f1, f2], max_tokens=500, reference_time=now)
        assert len(plan.items) == 1
        assert plan.items[0].content == "has content"
