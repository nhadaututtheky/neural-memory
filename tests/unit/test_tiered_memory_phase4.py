"""Tests for A6 Phase 4: Tiered Memory — stats, dashboard API, polish.

Covers:
1. nmem_stats includes tier_distribution
2. Dashboard /api/dashboard/tier-stats endpoint
3. Tier distribution counts accuracy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    TypedMemory,
)

# ── Stats tier distribution ───────────────────────────


class TestStatsTierDistribution:
    """Verify nmem_stats response includes tier_distribution."""

    @pytest.mark.asyncio
    async def test_stats_includes_tier_distribution(self) -> None:
        """_stats() response should contain tier_distribution dict."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = ToolHandler.__new__(ToolHandler)
        handler.config = MagicMock()
        handler.config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
        handler.config.safety = MagicMock(auto_redact_min_severity=3)
        handler.config.auto = MagicMock(enabled=False)
        handler.hooks = MagicMock()
        handler.hooks.emit = AsyncMock()

        # Mock storage
        storage = AsyncMock()
        handler.get_storage = AsyncMock(return_value=storage)

        brain = MagicMock()
        brain.id = "test-brain"
        brain.name = "test"
        brain.config = MagicMock()
        storage.get_brain = AsyncMock(return_value=brain)
        storage.brain_id = "test-brain"

        storage.get_enhanced_stats = AsyncMock(
            return_value={
                "neuron_count": 100,
                "synapse_count": 200,
                "fiber_count": 50,
                "db_size_bytes": 1024,
                "today_fibers_count": 5,
                "hot_neurons": [],
                "newest_memory": None,
            }
        )
        storage.get_synapses = AsyncMock(return_value=[])

        # Mock find_typed_memories for tier counts
        hot_mems = [
            TypedMemory.create(fiber_id=f"h{i}", memory_type=MemoryType.FACT, tier="hot")
            for i in range(3)
        ]
        warm_mems = [
            TypedMemory.create(fiber_id=f"w{i}", memory_type=MemoryType.FACT, tier="warm")
            for i in range(10)
        ]
        cold_mems = [
            TypedMemory.create(fiber_id=f"c{i}", memory_type=MemoryType.FACT, tier="cold")
            for i in range(2)
        ]

        async def find_by_tier(
            tier: str | None = None, limit: int = 1000, **kw: object
        ) -> list[TypedMemory]:
            if tier == "hot":
                return hot_mems
            if tier == "warm":
                return warm_mems
            if tier == "cold":
                return cold_mems
            return []

        storage.find_typed_memories = AsyncMock(side_effect=find_by_tier)

        # Mock onboarding/hints
        handler._check_onboarding = AsyncMock(return_value=None)
        handler.get_update_hint = MagicMock(return_value=None)
        handler._generate_stats_hints = AsyncMock(return_value=[])

        result = await handler._stats({})

        assert "tier_distribution" in result
        assert result["tier_distribution"]["hot"] == 3
        assert result["tier_distribution"]["warm"] == 10
        assert result["tier_distribution"]["cold"] == 2

    @pytest.mark.asyncio
    async def test_stats_tier_distribution_graceful_on_error(self) -> None:
        """tier_distribution should default to zeros if find_typed_memories fails."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = ToolHandler.__new__(ToolHandler)
        handler.config = MagicMock()
        handler.config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
        handler.config.safety = MagicMock(auto_redact_min_severity=3)
        handler.config.auto = MagicMock(enabled=False)
        handler.hooks = MagicMock()
        handler.hooks.emit = AsyncMock()

        storage = AsyncMock()
        handler.get_storage = AsyncMock(return_value=storage)

        brain = MagicMock()
        brain.id = "test-brain"
        brain.name = "test"
        brain.config = MagicMock()
        storage.get_brain = AsyncMock(return_value=brain)
        storage.brain_id = "test-brain"

        storage.get_enhanced_stats = AsyncMock(
            return_value={
                "neuron_count": 0,
                "synapse_count": 0,
                "fiber_count": 0,
            }
        )
        storage.get_synapses = AsyncMock(return_value=[])
        storage.find_typed_memories = AsyncMock(side_effect=Exception("db error"))

        handler._check_onboarding = AsyncMock(return_value=None)
        handler.get_update_hint = MagicMock(return_value=None)
        handler._generate_stats_hints = AsyncMock(return_value=[])

        result = await handler._stats({})

        assert result["tier_distribution"] == {"hot": 0, "warm": 0, "cold": 0}


# ── Dashboard API tier-stats endpoint ─────────────────


class TestDashboardTierStatsEndpoint:
    """Verify /api/dashboard/tier-stats returns correct tier counts."""

    @pytest.mark.asyncio
    async def test_tier_stats_endpoint_returns_distribution(self) -> None:
        """get_tier_stats should return TierDistribution model."""
        from neural_memory.server.routes.dashboard_api import TierDistribution, get_tier_stats

        storage = AsyncMock()

        hot_mems = [
            TypedMemory.create(fiber_id="h1", memory_type=MemoryType.FACT, tier="hot"),
        ]
        warm_mems = [
            TypedMemory.create(fiber_id=f"w{i}", memory_type=MemoryType.FACT, tier="warm")
            for i in range(5)
        ]
        cold_mems = []

        async def find_by_tier(
            tier: str | None = None, limit: int = 1000, **kw: object
        ) -> list[TypedMemory]:
            if tier == "hot":
                return hot_mems
            if tier == "warm":
                return warm_mems
            if tier == "cold":
                return cold_mems
            return []

        storage.find_typed_memories = AsyncMock(side_effect=find_by_tier)

        result = await get_tier_stats(storage)

        assert isinstance(result, TierDistribution)
        assert result.hot == 1
        assert result.warm == 5
        assert result.cold == 0
        assert result.total == 6

    @pytest.mark.asyncio
    async def test_tier_stats_empty_brain(self) -> None:
        """Empty brain should return all zeros."""
        from neural_memory.server.routes.dashboard_api import get_tier_stats

        storage = AsyncMock()
        storage.find_typed_memories = AsyncMock(return_value=[])

        result = await get_tier_stats(storage)

        assert result.hot == 0
        assert result.warm == 0
        assert result.cold == 0
        assert result.total == 0


# ── Tier distribution accuracy ────────────────────────


class TestTierDistributionAccuracy:
    """Verify tier counts match actual typed_memory tiers."""

    def test_all_tiers_represented(self) -> None:
        """MemoryTier enum should have exactly 3 values."""
        assert set(MemoryTier) == {MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD}

    def test_typed_memory_default_tier_is_warm(self) -> None:
        """Default tier for new TypedMemory should be warm."""
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT)
        assert tm.tier == "warm"

    def test_boundary_type_always_hot(self) -> None:
        """Boundary memories should always have HOT tier."""
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.BOUNDARY, tier="cold")
        assert tm.tier == MemoryTier.HOT
