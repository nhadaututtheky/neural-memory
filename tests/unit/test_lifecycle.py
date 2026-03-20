"""Unit tests for Phase 1: Memory Lifecycle Engine.

Covers:
- calculate_heat_score() — various combinations
- determine_lifecycle_state() — state transitions
- Heat resistance: hot memory resists compression
- Frozen memory: never compresses
- Snapshot creation and recovery
- batch_update_last_accessed()
- Lifecycle state transitions in storage
- MCP tool actions: status, freeze, thaw, recover
- LIFECYCLE consolidation strategy
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.compression import (
    CompressionConfig,
    CompressionEngine,
    CompressionTier,
    LifecycleState,
    calculate_heat_score,
    determine_lifecycle_state,
)
from neural_memory.engine.consolidation import ConsolidationStrategy
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fiber(
    *,
    days_old: float = 0.0,
    compression_tier: int = 0,
    pinned: bool = False,
) -> Fiber:
    """Create a minimal Fiber with a specific age and compression tier."""
    anchor_id = "anchor-lc-1"
    created = utcnow() - timedelta(days=days_old)
    return Fiber(
        id="fiber-lc-1",
        neuron_ids={anchor_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        compression_tier=compression_tier,
        created_at=created,
        pinned=pinned,
    )


def _make_neuron(
    *,
    content: str = "test content",
    neuron_id: str = "neuron-lc-1",
    days_old: float = 0.0,
) -> Neuron:
    """Create a minimal Neuron."""
    return Neuron(
        id=neuron_id,
        type=NeuronType.ENTITY,
        content=content,
        metadata={},
        created_at=utcnow() - timedelta(days=days_old),
    )


# ---------------------------------------------------------------------------
# calculate_heat_score tests
# ---------------------------------------------------------------------------


class TestCalculateHeatScore:
    """Tests for heat score calculation."""

    def test_never_accessed_returns_low_heat(self) -> None:
        """A never-accessed neuron has no recency contribution."""
        config = CompressionConfig()
        heat = calculate_heat_score(
            last_accessed_at=None,
            access_count=0,
            priority=5,
            reference_time=utcnow(),
            config=config,
        )
        # Only priority contributes: 5/10 * 0.2 = 0.1
        assert heat == pytest.approx(0.1, abs=0.01)

    def test_recently_accessed_high_heat(self) -> None:
        """A recently-accessed neuron should have high heat."""
        config = CompressionConfig()
        now = utcnow()
        heat = calculate_heat_score(
            last_accessed_at=now,
            access_count=10,
            priority=8,
            reference_time=now,
            config=config,
        )
        # recency=1.0*0.4, access=0.5*0.4, priority=0.8*0.2 = 0.4+0.2+0.16 = 0.76
        assert heat > 0.5

    def test_stale_access_reduces_recency(self) -> None:
        """A neuron accessed 30 days ago has low recency."""
        config = CompressionConfig()
        now = utcnow()
        accessed_30d_ago = now - timedelta(days=30)
        heat = calculate_heat_score(
            last_accessed_at=accessed_30d_ago,
            access_count=0,
            priority=0,
            reference_time=now,
            config=config,
        )
        # recency = exp(-30/7) ≈ 0.013
        expected_recency = math.exp(-30 / 7.0) * config.heat_recency_weight
        assert heat == pytest.approx(expected_recency, abs=0.01)

    def test_high_access_count_saturates(self) -> None:
        """Access count saturates at 20 (score = 1.0)."""
        config = CompressionConfig()
        heat_20 = calculate_heat_score(
            last_accessed_at=None,
            access_count=20,
            priority=0,
            reference_time=utcnow(),
            config=config,
        )
        heat_100 = calculate_heat_score(
            last_accessed_at=None,
            access_count=100,
            priority=0,
            reference_time=utcnow(),
            config=config,
        )
        assert heat_20 == heat_100

    def test_priority_10_max_contribution(self) -> None:
        """Priority 10 contributes maximum priority weight."""
        config = CompressionConfig()
        heat = calculate_heat_score(
            last_accessed_at=None,
            access_count=0,
            priority=10,
            reference_time=utcnow(),
            config=config,
        )
        assert heat == pytest.approx(config.heat_priority_weight, abs=0.01)

    def test_heat_clamped_to_0_1(self) -> None:
        """Heat score is always in [0.0, 1.0]."""
        config = CompressionConfig()
        now = utcnow()
        heat = calculate_heat_score(
            last_accessed_at=now,
            access_count=1000,
            priority=10,
            reference_time=now,
            config=config,
        )
        assert 0.0 <= heat <= 1.0

    def test_zero_access_zero_priority_zero_recency(self) -> None:
        """All-zero inputs give near-zero heat."""
        config = CompressionConfig()
        heat = calculate_heat_score(
            last_accessed_at=None,
            access_count=0,
            priority=0,
            reference_time=utcnow(),
            config=config,
        )
        assert heat == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# determine_lifecycle_state tests
# ---------------------------------------------------------------------------


class TestDetermineLifecycleState:
    """Tests for lifecycle state determination."""

    def test_fresh_memory_is_active(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=1.0, heat_score=0.0, config=config)
        assert state == LifecycleState.ACTIVE

    def test_7_to_30_days_is_warm(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=15.0, heat_score=0.0, config=config)
        assert state == LifecycleState.WARM

    def test_30_to_90_days_is_cool(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=60.0, heat_score=0.0, config=config)
        assert state == LifecycleState.COOL

    def test_90_to_180_days_is_compressed(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=120.0, heat_score=0.0, config=config)
        assert state == LifecycleState.COMPRESSED

    def test_over_180_days_is_archived(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=200.0, heat_score=0.0, config=config)
        assert state == LifecycleState.ARCHIVED

    def test_hot_memory_overrides_age(self) -> None:
        """A 60-day-old memory with high heat should stay WARM, not COOL."""
        config = CompressionConfig()
        # Heat above threshold (0.5 default)
        state = determine_lifecycle_state(age_days=60.0, heat_score=0.7, config=config)
        assert state == LifecycleState.WARM

    def test_hot_young_memory_stays_active(self) -> None:
        config = CompressionConfig()
        state = determine_lifecycle_state(age_days=2.0, heat_score=0.8, config=config)
        assert state == LifecycleState.ACTIVE

    def test_threshold_boundary(self) -> None:
        """Heat at or above threshold triggers resistance (>=)."""
        config = CompressionConfig(heat_resistance_threshold=0.5)
        # heat=0.5 IS >= threshold, so resistance applies → WARM not COOL
        state_at_threshold = determine_lifecycle_state(age_days=60.0, heat_score=0.5, config=config)
        assert state_at_threshold == LifecycleState.WARM
        # heat=0.49 is below threshold, no resistance → COOL
        state_below = determine_lifecycle_state(age_days=60.0, heat_score=0.49, config=config)
        assert state_below == LifecycleState.COOL


# ---------------------------------------------------------------------------
# CompressionEngine.determine_target_tier with heat/frozen tests
# ---------------------------------------------------------------------------


class TestDetermineTargetTierWithHeat:
    """Tests for heat-aware tier determination in CompressionEngine."""

    def _make_engine(self) -> CompressionEngine:
        mock_storage = MagicMock()
        return CompressionEngine(mock_storage)

    def test_frozen_always_returns_full(self) -> None:
        engine = self._make_engine()
        fiber = _make_fiber(days_old=200.0)
        tier = engine.determine_target_tier(fiber, utcnow(), heat_score=0.0, frozen=True)
        assert tier == CompressionTier.FULL

    def test_frozen_beats_any_age(self) -> None:
        engine = self._make_engine()
        for days in [0, 30, 90, 200]:
            fiber = _make_fiber(days_old=float(days))
            tier = engine.determine_target_tier(fiber, utcnow(), frozen=True)
            assert tier == CompressionTier.FULL

    def test_hot_memory_resists_by_one_tier(self) -> None:
        engine = self._make_engine()
        # 60-day fiber normally → ENTITY_ONLY (tier 2), hot → EXTRACTIVE (tier 1)
        fiber = _make_fiber(days_old=60.0)
        cold_tier = engine.determine_target_tier(fiber, utcnow(), heat_score=0.0)
        hot_tier = engine.determine_target_tier(fiber, utcnow(), heat_score=0.7)
        assert cold_tier == CompressionTier.ENTITY_ONLY
        assert hot_tier == CompressionTier.EXTRACTIVE

    def test_very_hot_memory_never_below_extractive(self) -> None:
        """Heat > 0.8 caps at EXTRACTIVE (not FULL)."""
        engine = self._make_engine()
        fiber = _make_fiber(days_old=120.0)  # normally TEMPLATE
        tier = engine.determine_target_tier(fiber, utcnow(), heat_score=0.9)
        assert tier == CompressionTier.EXTRACTIVE

    def test_no_heat_backward_compat(self) -> None:
        """Default heat=0.0 frozen=False → exact same behavior as before."""
        config = CompressionConfig()
        engine = CompressionEngine(MagicMock(), config=config)
        now = utcnow()
        cases = [
            (3.0, CompressionTier.FULL),
            (15.0, CompressionTier.EXTRACTIVE),
            (60.0, CompressionTier.ENTITY_ONLY),
            (120.0, CompressionTier.TEMPLATE),
            (200.0, CompressionTier.GRAPH_ONLY),
        ]
        for days, expected_tier in cases:
            fiber = _make_fiber(days_old=days)
            assert engine.determine_target_tier(fiber, now) == expected_tier

    def test_fresh_memory_heat_no_change(self) -> None:
        """Fresh memory stays FULL regardless of heat."""
        engine = self._make_engine()
        fiber = _make_fiber(days_old=2.0)
        tier = engine.determine_target_tier(fiber, utcnow(), heat_score=0.9)
        assert tier == CompressionTier.FULL


# ---------------------------------------------------------------------------
# SQLite storage: batch_update_last_accessed, lifecycle state, snapshots
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create a fresh SQLiteStorage with a default brain."""
    s = SQLiteStorage(db_path=str(tmp_path / "test_lifecycle.db"))
    await s.initialize()
    brain = Brain.create(name="lc-test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


@pytest.mark.asyncio
async def test_batch_update_last_accessed(storage: SQLiteStorage) -> None:
    """batch_update_last_accessed sets last_accessed_at for all given neurons."""
    neuron = _make_neuron(neuron_id="n-batch-1")
    await storage.add_neuron(neuron)

    before = utcnow()
    await storage.batch_update_last_accessed(["n-batch-1"])

    # Verify via direct DB query
    brain_id = storage.brain_id
    conn = storage._ensure_read_conn()
    async with conn.execute(
        "SELECT last_accessed_at FROM neurons WHERE id = ? AND brain_id = ?",
        ("n-batch-1", brain_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    from datetime import datetime

    accessed_at = datetime.fromisoformat(row[0])
    assert accessed_at >= before


@pytest.mark.asyncio
async def test_batch_update_last_accessed_empty(storage: SQLiteStorage) -> None:
    """batch_update_last_accessed with empty list is a no-op."""
    await storage.batch_update_last_accessed([])


@pytest.mark.asyncio
async def test_update_neuron_lifecycle_state(storage: SQLiteStorage) -> None:
    """update_neuron_lifecycle changes lifecycle_state column."""
    neuron = _make_neuron(neuron_id="n-lc-state-1")
    await storage.add_neuron(neuron)

    await storage.update_neuron_lifecycle("n-lc-state-1", "warm")

    brain_id = storage.brain_id
    conn = storage._ensure_read_conn()
    async with conn.execute(
        "SELECT lifecycle_state FROM neurons WHERE id = ? AND brain_id = ?",
        ("n-lc-state-1", brain_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "warm"


@pytest.mark.asyncio
async def test_update_neuron_frozen(storage: SQLiteStorage) -> None:
    """update_neuron_frozen sets/clears the frozen flag."""
    neuron = _make_neuron(neuron_id="n-frozen-1")
    await storage.add_neuron(neuron)

    brain_id = storage.brain_id

    # Freeze
    await storage.update_neuron_frozen("n-frozen-1", frozen=True)
    conn = storage._ensure_read_conn()
    async with conn.execute(
        "SELECT frozen FROM neurons WHERE id = ? AND brain_id = ?",
        ("n-frozen-1", brain_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1

    # Thaw
    await storage.update_neuron_frozen("n-frozen-1", frozen=False)
    async with conn.execute(
        "SELECT frozen FROM neurons WHERE id = ? AND brain_id = ?",
        ("n-frozen-1", brain_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 0


@pytest.mark.asyncio
async def test_get_lifecycle_distribution(storage: SQLiteStorage) -> None:
    """get_lifecycle_distribution returns count per state."""
    for i in range(3):
        neuron = _make_neuron(neuron_id=f"n-dist-{i}")
        await storage.add_neuron(neuron)

    # By default all are 'active'
    dist = await storage.get_lifecycle_distribution()
    assert dist.get("active", 0) == 3

    # Update one to 'warm'
    await storage.update_neuron_lifecycle("n-dist-0", "warm")
    dist2 = await storage.get_lifecycle_distribution()
    assert dist2.get("warm", 0) == 1
    assert dist2.get("active", 0) == 2


# ---------------------------------------------------------------------------
# Neuron snapshot storage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_neuron_snapshot(storage: SQLiteStorage) -> None:
    """Can save and retrieve a neuron snapshot."""
    neuron = _make_neuron(neuron_id="n-snap-1", content="original text")
    await storage.add_neuron(neuron)

    brain_id = storage.brain_id
    assert brain_id is not None
    now_iso = utcnow().isoformat()
    await storage.save_neuron_snapshot(
        neuron_id="n-snap-1",
        brain_id=brain_id,
        original_content="original text",
        compressed_at=now_iso,
        tier=3,
    )

    snapshot = await storage.get_neuron_snapshot("n-snap-1")
    assert snapshot is not None
    assert snapshot["original_content"] == "original text"
    assert snapshot["tier"] == 3
    assert snapshot["brain_id"] == brain_id


@pytest.mark.asyncio
async def test_get_neuron_snapshot_not_found(storage: SQLiteStorage) -> None:
    """Returns None for non-existent snapshot."""
    result = await storage.get_neuron_snapshot("nonexistent-neuron")
    assert result is None


@pytest.mark.asyncio
async def test_delete_neuron_snapshot(storage: SQLiteStorage) -> None:
    """Deleting a snapshot removes it."""
    neuron = _make_neuron(neuron_id="n-snap-del-1")
    await storage.add_neuron(neuron)

    brain_id = storage.brain_id
    assert brain_id is not None
    await storage.save_neuron_snapshot(
        neuron_id="n-snap-del-1",
        brain_id=brain_id,
        original_content="something",
        compressed_at=utcnow().isoformat(),
        tier=3,
    )

    deleted = await storage.delete_neuron_snapshot("n-snap-del-1")
    assert deleted is True

    after = await storage.get_neuron_snapshot("n-snap-del-1")
    assert after is None


@pytest.mark.asyncio
async def test_delete_nonexistent_snapshot_returns_false(storage: SQLiteStorage) -> None:
    """Deleting a non-existent snapshot returns False."""
    result = await storage.delete_neuron_snapshot("ghost-neuron")
    assert result is False


@pytest.mark.asyncio
async def test_snapshot_upsert(storage: SQLiteStorage) -> None:
    """Saving a snapshot twice upserts (overwrites) the existing one."""
    neuron = _make_neuron(neuron_id="n-snap-upsert")
    await storage.add_neuron(neuron)

    brain_id = storage.brain_id
    assert brain_id is not None
    await storage.save_neuron_snapshot(
        neuron_id="n-snap-upsert",
        brain_id=brain_id,
        original_content="v1",
        compressed_at=utcnow().isoformat(),
        tier=3,
    )
    await storage.save_neuron_snapshot(
        neuron_id="n-snap-upsert",
        brain_id=brain_id,
        original_content="v2",
        compressed_at=utcnow().isoformat(),
        tier=4,
    )

    snapshot = await storage.get_neuron_snapshot("n-snap-upsert")
    assert snapshot is not None
    assert snapshot["original_content"] == "v2"
    assert snapshot["tier"] == 4


# ---------------------------------------------------------------------------
# CompressionEngine.recover_fiber tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_fiber_from_snapshot(storage: SQLiteStorage) -> None:
    """recover_fiber restores neuron content from neuron_snapshots table."""
    brain_id = storage.brain_id
    assert brain_id is not None
    # Setup: neuron + fiber
    neuron = _make_neuron(neuron_id="n-recover-1", content="original content for recovery")
    await storage.add_neuron(neuron)
    fiber = Fiber(
        id="f-recover-1",
        neuron_ids={"n-recover-1"},
        synapse_ids=set(),
        anchor_neuron_id="n-recover-1",
        compression_tier=3,
        created_at=utcnow() - timedelta(days=100),
    )
    await storage.add_fiber(fiber)

    # Save snapshot
    await storage.save_neuron_snapshot(
        neuron_id="n-recover-1",
        brain_id=brain_id,
        original_content="original content for recovery",
        compressed_at=utcnow().isoformat(),
        tier=3,
    )

    # Compress neuron content (simulate Tier 3 effect)
    from dataclasses import replace as dc_replace

    compressed = dc_replace(neuron, content="entity1 related_to entity2")
    await storage.update_neuron(compressed)

    # Recover
    engine = CompressionEngine(storage)
    success = await engine.recover_fiber("f-recover-1")

    assert success is True

    # Verify neuron restored
    restored = await storage.get_neuron("n-recover-1")
    assert restored is not None
    assert restored.content == "original content for recovery"

    # Verify snapshot deleted
    snapshot = await storage.get_neuron_snapshot("n-recover-1")
    assert snapshot is None

    # Verify fiber compression_tier reset
    updated_fiber = await storage.get_fiber("f-recover-1")
    assert updated_fiber is not None
    assert updated_fiber.compression_tier == 0


@pytest.mark.asyncio
async def test_recover_fiber_fallback_to_backup(storage: SQLiteStorage) -> None:
    """recover_fiber falls back to decompress_fiber for Tier 1-2 fibers."""
    neuron = _make_neuron(neuron_id="n-fallback-1", content="compressed tier 1 content")
    await storage.add_neuron(neuron)
    fiber = Fiber(
        id="f-fallback-1",
        neuron_ids={"n-fallback-1"},
        synapse_ids=set(),
        anchor_neuron_id="n-fallback-1",
        compression_tier=1,
        created_at=utcnow() - timedelta(days=10),
    )
    await storage.add_fiber(fiber)

    # Save backup (tier 1-2 style)
    await storage.save_compression_backup(
        fiber_id="f-fallback-1",
        original_content="original content before extractive",
        compression_tier=1,
        original_token_count=6,
        compressed_token_count=3,
    )

    engine = CompressionEngine(storage)
    success = await engine.recover_fiber("f-fallback-1")

    assert success is True

    restored = await storage.get_neuron("n-fallback-1")
    assert restored is not None
    assert restored.content == "original content before extractive"


@pytest.mark.asyncio
async def test_recover_fiber_not_found(storage: SQLiteStorage) -> None:
    """recover_fiber returns False for unknown fiber."""
    engine = CompressionEngine(storage)
    success = await engine.recover_fiber("nonexistent-fiber")
    assert success is False


# ---------------------------------------------------------------------------
# MCP tool handler tests (nmem_lifecycle)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock ToolHandler-like object."""
    handler = MagicMock()
    return handler


@pytest.mark.asyncio
async def test_lifecycle_status_action() -> None:
    """nmem_lifecycle status returns lifecycle distribution."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    # Build a minimal mock for the handler
    storage = AsyncMock()
    brain = MagicMock()
    brain.id = "brain-test"
    storage.brain_id = "brain-test"
    storage.get_brain = AsyncMock(return_value=brain)
    storage.get_lifecycle_distribution = AsyncMock(
        return_value={"active": 10, "warm": 3, "cool": 1}
    )

    handler = ToolHandler()

    async def mock_get_storage() -> AsyncMock:
        return storage

    handler.get_storage = mock_get_storage  # type: ignore[method-assign]

    result = await handler._lifecycle({"action": "status"})
    assert "distribution" in result
    assert result["distribution"]["active"] == 10
    assert result["total_neurons"] == 14


@pytest.mark.asyncio
async def test_lifecycle_freeze_action() -> None:
    """nmem_lifecycle freeze sets frozen flag on neuron."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    storage = AsyncMock()
    brain = MagicMock()
    brain.id = "brain-test"
    storage.brain_id = "brain-test"
    storage.get_brain = AsyncMock(return_value=brain)
    storage.update_neuron_frozen = AsyncMock()

    handler = ToolHandler()

    async def mock_get_storage() -> AsyncMock:
        return storage

    handler.get_storage = mock_get_storage  # type: ignore[method-assign]

    result = await handler._lifecycle({"action": "freeze", "id": "n-freeze-1"})
    assert result["frozen"] is True
    assert result["neuron_id"] == "n-freeze-1"
    storage.update_neuron_frozen.assert_called_once_with("n-freeze-1", frozen=True)


@pytest.mark.asyncio
async def test_lifecycle_thaw_action() -> None:
    """nmem_lifecycle thaw clears frozen flag on neuron."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    storage = AsyncMock()
    brain = MagicMock()
    brain.id = "brain-test"
    storage.brain_id = "brain-test"
    storage.get_brain = AsyncMock(return_value=brain)
    storage.update_neuron_frozen = AsyncMock()

    handler = ToolHandler()

    async def mock_get_storage() -> AsyncMock:
        return storage

    handler.get_storage = mock_get_storage  # type: ignore[method-assign]

    result = await handler._lifecycle({"action": "thaw", "id": "n-thaw-1"})
    assert result["frozen"] is False
    assert result["neuron_id"] == "n-thaw-1"
    storage.update_neuron_frozen.assert_called_once_with("n-thaw-1", frozen=False)


@pytest.mark.asyncio
async def test_lifecycle_missing_id_error() -> None:
    """nmem_lifecycle recover/freeze/thaw without id returns error."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    storage = AsyncMock()
    brain = MagicMock()
    brain.id = "brain-test"
    storage.brain_id = "brain-test"
    storage.get_brain = AsyncMock(return_value=brain)

    handler = ToolHandler()

    async def mock_get_storage() -> AsyncMock:
        return storage

    handler.get_storage = mock_get_storage  # type: ignore[method-assign]

    for action in ("recover", "freeze", "thaw"):
        result = await handler._lifecycle({"action": action})
        assert "error" in result


@pytest.mark.asyncio
async def test_lifecycle_unknown_action() -> None:
    """nmem_lifecycle with unknown action returns error."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    storage = AsyncMock()
    brain = MagicMock()
    brain.id = "brain-test"
    storage.brain_id = "brain-test"
    storage.get_brain = AsyncMock(return_value=brain)

    handler = ToolHandler()

    async def mock_get_storage() -> AsyncMock:
        return storage

    handler.get_storage = mock_get_storage  # type: ignore[method-assign]

    result = await handler._lifecycle({"action": "explode"})
    assert "error" in result


# ---------------------------------------------------------------------------
# ConsolidationStrategy.LIFECYCLE exists
# ---------------------------------------------------------------------------


def test_lifecycle_strategy_in_enum() -> None:
    """LIFECYCLE should be a valid ConsolidationStrategy."""
    assert ConsolidationStrategy.LIFECYCLE == "lifecycle"


def test_lifecycle_strategy_in_strategy_tiers() -> None:
    """LIFECYCLE should be in STRATEGY_TIERS alongside COMPRESS."""
    from neural_memory.engine.consolidation import ConsolidationEngine

    all_strategies_in_tiers: set[ConsolidationStrategy] = set()
    for tier_set in ConsolidationEngine.STRATEGY_TIERS:
        all_strategies_in_tiers.update(tier_set)
    assert ConsolidationStrategy.LIFECYCLE in all_strategies_in_tiers
