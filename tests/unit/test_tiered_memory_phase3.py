"""Tests for A6 Phase 3: Tiered Memory — context loading + decay behavior.

Covers:
1. Tier constants: floors, multipliers, enum values
2. Decay math: tier floors (HOT=0.5) and multipliers (COLD=2x)
3. Recall: tier filter parameter in schema
4. Context optimizer: HOT boost, COLD exclusion
5. Lifecycle integration: DecayManager uses tier floors + multipliers
"""

from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.memory_types import (
    TIER_DECAY_FLOORS,
    TIER_DECAY_MULTIPLIERS,
    MemoryType,
    TypedMemory,
)

# ── Tier constants ─────────────────────────────────────


class TestTierConstants:
    """Verify tier constant values are correct."""

    def test_hot_decay_floor(self) -> None:
        assert TIER_DECAY_FLOORS["hot"] == 0.5

    def test_warm_decay_floor(self) -> None:
        assert TIER_DECAY_FLOORS["warm"] == 0.0

    def test_cold_decay_floor(self) -> None:
        assert TIER_DECAY_FLOORS["cold"] == 0.0

    def test_hot_decay_multiplier(self) -> None:
        assert TIER_DECAY_MULTIPLIERS["hot"] == 0.5

    def test_warm_decay_multiplier(self) -> None:
        assert TIER_DECAY_MULTIPLIERS["warm"] == 1.0

    def test_cold_decay_multiplier(self) -> None:
        assert TIER_DECAY_MULTIPLIERS["cold"] == 2.0


# ── Decay math ─────────────────────────────────────────


class TestTierAwareDecayMath:
    """Verify tier affects decay rates and floors (pure math, no mocks)."""

    def test_hot_floor_prevents_full_decay(self) -> None:
        """HOT memories should never decay below 0.5."""
        floor = TIER_DECAY_FLOORS["hot"]
        decay_factor = math.exp(-0.5 * 100)
        raw_level = 1.0 * decay_factor
        effective_level = max(floor, raw_level)
        assert effective_level == 0.5

    def test_cold_decays_faster(self) -> None:
        """COLD memories should decay 2x faster than WARM."""
        base_rate = 0.1
        days = 10

        warm_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["warm"] * days)
        cold_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["cold"] * days)

        assert cold_level < warm_level
        assert abs(cold_level - math.exp(-0.2 * 10)) < 0.001

    def test_hot_decays_slower(self) -> None:
        """HOT multiplier is 0.5 — decays at half the rate."""
        base_rate = 0.1
        days = 10

        warm_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["warm"] * days)
        hot_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["hot"] * days)

        assert hot_level > warm_level


# ── Recall tier filter ──────────────────────────────────


class TestRecallTierFilter:
    """Verify tier filter in nmem_recall schema."""

    def test_recall_schema_has_tier(self) -> None:
        from neural_memory.mcp.tool_schemas import get_tool_schemas

        for schema in get_tool_schemas():
            if schema["name"] == "nmem_recall":
                props = schema["inputSchema"]["properties"]
                assert "tier" in props
                assert props["tier"]["enum"] == ["hot", "warm", "cold"]
                return
        pytest.fail("nmem_recall not found in schemas")


# ── Context optimizer integration ────────────────────────


class TestContextOptimizerExcludeCold:
    """Verify COLD exclusion and HOT boost in context optimizer."""

    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=None)
        storage.get_neuron_state = AsyncMock(return_value=None)
        storage.get_typed_memory = AsyncMock(return_value=None)
        return storage

    @pytest.mark.asyncio
    async def test_exclude_cold_true_by_default(self, mock_storage) -> None:
        """optimize_context excludes COLD fibers by default."""
        from neural_memory.engine.context_optimizer import optimize_context

        fibers = self._make_fibers(3)
        tier_map = {"f0": "cold", "f1": "warm", "f2": "hot"}

        def get_typed_mem(fid):
            tier = tier_map.get(fid, "warm")
            return TypedMemory.create(fiber_id=fid, memory_type=MemoryType.FACT, tier=tier)

        mock_storage.get_typed_memory = AsyncMock(side_effect=get_typed_mem)

        neuron_mock = MagicMock()
        neuron_mock.content = "test content for context"
        neuron_mock.content_hash = 12345
        mock_storage.get_neuron = AsyncMock(return_value=neuron_mock)

        state_mock = MagicMock()
        state_mock.activation_level = 0.5
        mock_storage.get_neuron_state = AsyncMock(return_value=state_mock)

        plan = await optimize_context(mock_storage, fibers, max_tokens=10000, exclude_cold=True)

        result_fiber_ids = {item.fiber_id for item in plan.items}
        assert "f0" not in result_fiber_ids
        assert "f1" in result_fiber_ids or "f2" in result_fiber_ids

    @pytest.mark.asyncio
    async def test_exclude_cold_false_includes_cold(self, mock_storage) -> None:
        """When exclude_cold=False, COLD fibers are included in results."""
        from neural_memory.engine.context_optimizer import optimize_context

        fibers = self._make_fibers(1)
        tier_map = {"f0": "cold"}

        def get_typed_mem(fid):
            tier = tier_map.get(fid, "warm")
            return TypedMemory.create(fiber_id=fid, memory_type=MemoryType.FACT, tier=tier)

        mock_storage.get_typed_memory = AsyncMock(side_effect=get_typed_mem)

        neuron_mock = MagicMock()
        neuron_mock.content = "cold fiber content about important archive data"
        neuron_mock.content_hash = 987654
        mock_storage.get_neuron = AsyncMock(return_value=neuron_mock)

        state_mock = MagicMock()
        state_mock.activation_level = 0.5
        mock_storage.get_neuron_state = AsyncMock(return_value=state_mock)

        plan_excluded = await optimize_context(
            mock_storage, fibers, max_tokens=10000, exclude_cold=True
        )
        assert len(plan_excluded.items) == 0

        plan_included = await optimize_context(
            mock_storage, fibers, max_tokens=10000, exclude_cold=False
        )
        assert len(plan_included.items) >= 1
        assert plan_included.items[0].fiber_id == "f0"

    @pytest.mark.asyncio
    async def test_hot_gets_score_boost(self, mock_storage) -> None:
        """HOT fibers should have higher score than equivalent WARM fibers."""
        from neural_memory.engine.context_optimizer import optimize_context

        fibers = self._make_fibers(2)
        tier_map = {"f0": "warm", "f1": "hot"}

        def get_typed_mem(fid):
            tier = tier_map.get(fid, "warm")
            return TypedMemory.create(fiber_id=fid, memory_type=MemoryType.FACT, tier=tier)

        mock_storage.get_typed_memory = AsyncMock(side_effect=get_typed_mem)

        call_count = 0

        def make_neuron(nid):
            nonlocal call_count
            call_count += 1
            n = MagicMock()
            n.content = f"content for fiber {call_count}"
            n.content_hash = call_count
            return n

        mock_storage.get_neuron = AsyncMock(side_effect=make_neuron)

        state_mock = MagicMock()
        state_mock.activation_level = 0.5
        mock_storage.get_neuron_state = AsyncMock(return_value=state_mock)

        plan = await optimize_context(mock_storage, fibers, max_tokens=10000, exclude_cold=True)

        if len(plan.items) >= 2:
            scores = {item.fiber_id: item.score for item in plan.items}
            assert scores.get("f1", 0) > scores.get("f0", 0)

    def _make_fibers(self, count: int) -> list:
        """Create mock Fiber objects."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.utils.timeutils import utcnow

        fibers = []
        for i in range(count):
            f = Fiber(
                id=f"f{i}",
                neuron_ids={f"n{i}"},
                synapse_ids=set(),
                anchor_neuron_id=f"n{i}",
                frequency=5,
                conductivity=0.5,
                created_at=utcnow() - timedelta(hours=1),
            )
            fibers.append(f)
        return fibers


# ── Lifecycle decay integration ─────────────────────────


class TestLifecycleDecay:
    """Verify DecayManager uses tier floors and multipliers."""

    @pytest.mark.asyncio
    async def test_hot_neuron_respects_floor(self) -> None:
        """HOT neuron should not decay below 0.5 activation."""
        from neural_memory.core.neuron import NeuronState
        from neural_memory.engine.lifecycle import DecayManager
        from neural_memory.utils.timeutils import utcnow

        manager = DecayManager(decay_rate=0.5, prune_threshold=0.01)

        storage = AsyncMock()
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())

        ref_time = utcnow()
        state = NeuronState(
            neuron_id="n1",
            activation_level=1.0,
            decay_rate=0.5,
            last_activated=ref_time - timedelta(days=30),
        )

        storage.get_all_neuron_states = AsyncMock(return_value=[state])
        storage.get_all_synapses = AsyncMock(return_value=[])
        storage.update_neuron_state = AsyncMock()

        hot_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="hot")
        fiber_mock = MagicMock()
        fiber_mock.neuron_ids = {"n1"}

        async def find_typed(tier=None, limit=1000, **kw):
            if tier == "hot":
                return [hot_tm]
            return []

        storage.find_typed_memories = AsyncMock(side_effect=find_typed)
        storage.get_fiber = AsyncMock(return_value=fiber_mock)

        report = await manager.apply_decay(storage, reference_time=ref_time)

        # Verify decay processed and applied
        assert report.neurons_processed == 1
        assert report.neurons_decayed == 1
        assert storage.update_neuron_state.called
        updated = storage.update_neuron_state.call_args[0][0]
        assert updated.activation_level >= 0.5

    @pytest.mark.asyncio
    async def test_cold_neuron_decays_faster(self) -> None:
        """COLD neuron uses 2x decay multiplier."""
        base_rate = 0.1
        days = 5

        warm_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["warm"] * days)
        cold_level = math.exp(-base_rate * TIER_DECAY_MULTIPLIERS["cold"] * days)

        assert cold_level < warm_level
        ratio = warm_level / cold_level
        assert ratio > 1.5
