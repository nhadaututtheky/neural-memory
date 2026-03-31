"""Tests for B5 Phase 1: Auto-Tier Engine — promotion, demotion, protection."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.engine.tier_engine import TierChange, TierEngine, TierReport
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.unified_config import TierConfig
from neural_memory.utils.timeutils import utcnow


# ── Fixtures ─────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create an initialized SQLiteStorage with a brain."""
    db_path = tmp_path / "test_tier.db"
    store = SQLiteStorage(db_path=str(db_path))
    await store.initialize()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=4,
        max_context_tokens=1500,
    )
    brain = Brain.create(name="tier-test", config=config)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store  # type: ignore[misc]
    await store.close()


@pytest.fixture
def tier_config() -> TierConfig:
    """Default tier config for testing."""
    return TierConfig(
        auto_enabled=True,
        promote_threshold=3,
        demote_inactive_days=7,
        cold_archive_days=30,
        max_hot_memories=50,
    )


async def _create_memory(
    storage: SQLiteStorage,
    fiber_id: str,
    memory_type: MemoryType = MemoryType.FACT,
    tier: str = MemoryTier.WARM,
    access_frequency: int = 0,
    last_activated_days_ago: int | None = None,
    pinned: bool = False,
) -> TypedMemory:
    """Helper to create a fiber + neuron + state + typed_memory for testing."""
    now = utcnow()
    neuron_id = f"n-{fiber_id}"

    # Create neuron
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content=f"Content for {fiber_id}",
    )
    neuron = Neuron(
        id=neuron_id,
        type=neuron.type,
        content=neuron.content,
        metadata=neuron.metadata,
        created_at=neuron.created_at,
    )
    await storage.add_neuron(neuron)

    # Create neuron state
    last_activated = None
    if last_activated_days_ago is not None:
        last_activated = now - timedelta(days=last_activated_days_ago)

    state = NeuronState(
        neuron_id=neuron_id,
        activation_level=0.5,
        access_frequency=access_frequency,
        last_activated=last_activated,
        created_at=now,
    )
    await storage.update_neuron_state(state)

    # Create fiber
    fiber = Fiber(
        id=fiber_id,
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        pathway=[neuron_id],
        pinned=pinned,
    )
    await storage.add_fiber(fiber)

    # Create typed memory
    tm = TypedMemory(
        fiber_id=fiber_id,
        memory_type=memory_type,
        priority=Priority.NORMAL,
        provenance=Provenance(source="test"),
        tier=tier,
        created_at=now,
    )
    await storage.add_typed_memory(tm)
    return tm


# ── TierConfig ───────────────────────────────────────────


class TestTierConfig:
    def test_defaults(self) -> None:
        cfg = TierConfig()
        assert cfg.auto_enabled is False
        assert cfg.promote_threshold == 5
        assert cfg.demote_inactive_days == 30
        assert cfg.cold_archive_days == 90
        assert cfg.max_hot_memories == 100

    def test_from_dict(self) -> None:
        cfg = TierConfig.from_dict({
            "auto_enabled": True,
            "promote_threshold": 10,
            "demote_inactive_days": 14,
            "cold_archive_days": 60,
            "max_hot_memories": 200,
        })
        assert cfg.auto_enabled is True
        assert cfg.promote_threshold == 10
        assert cfg.demote_inactive_days == 14
        assert cfg.cold_archive_days == 60
        assert cfg.max_hot_memories == 200

    def test_from_dict_defaults(self) -> None:
        cfg = TierConfig.from_dict({})
        assert cfg.auto_enabled is False
        assert cfg.promote_threshold == 5

    def test_to_dict_roundtrip(self) -> None:
        cfg = TierConfig(auto_enabled=True, promote_threshold=8)
        restored = TierConfig.from_dict(cfg.to_dict())
        assert restored.auto_enabled is True
        assert restored.promote_threshold == 8


# ── TierReport ───────────────────────────────────────────


class TestTierReport:
    def test_empty_report(self) -> None:
        report = TierReport()
        assert report.total_changes == 0
        assert report.dry_run is True

    def test_total_changes(self) -> None:
        change = TierChange(
            fiber_id="f1",
            memory_type="fact",
            from_tier="warm",
            to_tier="hot",
            reason="test",
        )
        report = TierReport(promoted=[change], demoted=[change])
        assert report.total_changes == 2

    def test_to_dict(self) -> None:
        report = TierReport(dry_run=False, skipped_boundary=2, skipped_pinned=1)
        d = report.to_dict()
        assert d["dry_run"] is False
        assert d["skipped_boundary"] == 2
        assert d["skipped_pinned"] == 1
        assert d["total_changes"] == 0


# ── Promotion: WARM → HOT ───────────────────────────────


class TestPromotion:
    @pytest.mark.asyncio
    async def test_promotes_warm_with_high_access(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """WARM memories with access_frequency >= threshold get promoted to HOT."""
        await _create_memory(
            storage, "f1", access_frequency=5, last_activated_days_ago=1
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.promoted) == 1
        assert report.promoted[0].fiber_id == "f1"
        assert report.promoted[0].to_tier == "hot"

        # Verify the memory was actually updated
        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "hot"

    @pytest.mark.asyncio
    async def test_does_not_promote_below_threshold(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """WARM memories below promote_threshold stay WARM."""
        await _create_memory(
            storage, "f1", access_frequency=2, last_activated_days_ago=1
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.promoted) == 0

    @pytest.mark.asyncio
    async def test_respects_max_hot_cap(
        self, storage: SQLiteStorage
    ) -> None:
        """Promotion stops when max_hot_memories cap is reached."""
        config = TierConfig(
            auto_enabled=True,
            promote_threshold=1,
            max_hot_memories=2,
        )
        # Create 1 existing HOT + 3 WARM eligible for promotion
        await _create_memory(
            storage, "existing-hot", tier=MemoryTier.HOT,
            access_frequency=10, last_activated_days_ago=1,
        )
        for i in range(3):
            await _create_memory(
                storage, f"warm-{i}", access_frequency=5, last_activated_days_ago=1,
            )

        engine = TierEngine(storage, config)
        report = await engine.evaluate(storage.brain_id)

        # Only 1 should be promoted (cap=2, already 1 HOT)
        assert len(report.promoted) == 1
        assert report.skipped_at_cap >= 1

    @pytest.mark.asyncio
    async def test_dry_run_does_not_mutate(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Dry run calculates changes but doesn't apply them."""
        await _create_memory(
            storage, "f1", access_frequency=5, last_activated_days_ago=1
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.promoted) == 1
        assert report.dry_run is True

        # Memory should still be WARM
        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "warm"


# ── Demotion: HOT → WARM ────────────────────────────────


class TestDemotion:
    @pytest.mark.asyncio
    async def test_demotes_inactive_hot(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """HOT memories inactive for > demote_inactive_days get demoted to WARM."""
        await _create_memory(
            storage, "f1", tier=MemoryTier.HOT,
            access_frequency=10, last_activated_days_ago=10,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.demoted) == 1
        assert report.demoted[0].fiber_id == "f1"

        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "warm"

    @pytest.mark.asyncio
    async def test_does_not_demote_recently_active(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """HOT memories accessed recently stay HOT."""
        await _create_memory(
            storage, "f1", tier=MemoryTier.HOT,
            access_frequency=10, last_activated_days_ago=2,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.demoted) == 0


# ── Archive: WARM → COLD ────────────────────────────────


class TestArchive:
    @pytest.mark.asyncio
    async def test_archives_long_inactive_warm(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """WARM memories inactive for > cold_archive_days get archived to COLD."""
        await _create_memory(
            storage, "f1", access_frequency=1, last_activated_days_ago=60,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.archived) == 1

        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "cold"

    @pytest.mark.asyncio
    async def test_does_not_archive_recent_warm(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """WARM memories with recent access stay WARM."""
        await _create_memory(
            storage, "f1", access_frequency=1, last_activated_days_ago=10,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.archived) == 0


# ── Boundary Protection ──────────────────────────────────


class TestBoundaryProtection:
    @pytest.mark.asyncio
    async def test_boundary_never_demoted(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """BOUNDARY type memories are never demoted from HOT."""
        await _create_memory(
            storage, "f-boundary", memory_type=MemoryType.BOUNDARY,
            tier=MemoryTier.HOT, access_frequency=0, last_activated_days_ago=100,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.demoted) == 0
        assert report.skipped_boundary >= 1

    @pytest.mark.asyncio
    async def test_boundary_never_archived(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """BOUNDARY type WARM memories are never archived to COLD."""
        await _create_memory(
            storage, "f-boundary", memory_type=MemoryType.BOUNDARY,
            tier=MemoryTier.WARM, access_frequency=0, last_activated_days_ago=100,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.archived) == 0
        assert report.skipped_boundary >= 1


# ── Pinned Protection ────────────────────────────────────


class TestPinnedProtection:
    @pytest.mark.asyncio
    async def test_pinned_never_demoted(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Pinned fibers are never demoted from HOT."""
        await _create_memory(
            storage, "f-pinned", tier=MemoryTier.HOT,
            access_frequency=0, last_activated_days_ago=100, pinned=True,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.demoted) == 0
        assert report.skipped_pinned >= 1

    @pytest.mark.asyncio
    async def test_pinned_never_archived(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Pinned fibers are never archived from WARM."""
        await _create_memory(
            storage, "f-pinned", access_frequency=0,
            last_activated_days_ago=100, pinned=True,
        )
        engine = TierEngine(storage, tier_config)
        report = await engine.evaluate(storage.brain_id)

        assert len(report.archived) == 0
        assert report.skipped_pinned >= 1


# ── Oscillation Prevention ───────────────────────────────


class TestOscillation:
    @pytest.mark.asyncio
    async def test_no_promote_then_demote_same_cycle(
        self, storage: SQLiteStorage
    ) -> None:
        """A memory promoted in this cycle cannot be demoted in the same cycle."""
        config = TierConfig(
            auto_enabled=True,
            promote_threshold=3,
            demote_inactive_days=1,  # very aggressive demotion
        )
        # This memory is WARM with high access but also "inactive" (>1 day)
        # It should be promoted but NOT demoted in the same cycle
        await _create_memory(
            storage, "f1", access_frequency=5, last_activated_days_ago=5,
        )
        engine = TierEngine(storage, config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.promoted) == 1
        assert len(report.demoted) == 0  # not demoted back

        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "hot"


# ── Promotion History ────────────────────────────────────


class TestPromotionHistory:
    @pytest.mark.asyncio
    async def test_history_recorded_on_promotion(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Promotion adds an entry to promotion_history in metadata."""
        await _create_memory(
            storage, "f1", access_frequency=5, last_activated_days_ago=1,
        )
        engine = TierEngine(storage, tier_config)
        await engine.apply(storage.brain_id, dry_run=False)

        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        history = tm.metadata.get("promotion_history", [])
        assert len(history) == 1
        assert history[0]["from"] == "warm"
        assert history[0]["to"] == "hot"

    @pytest.mark.asyncio
    async def test_history_recorded_on_demotion(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Demotion adds an entry to promotion_history in metadata."""
        await _create_memory(
            storage, "f1", tier=MemoryTier.HOT,
            access_frequency=10, last_activated_days_ago=10,
        )
        engine = TierEngine(storage, tier_config)
        await engine.apply(storage.brain_id, dry_run=False)

        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        history = tm.metadata.get("promotion_history", [])
        assert len(history) == 1
        assert history[0]["from"] == "hot"
        assert history[0]["to"] == "warm"


# ── TierChange ───────────────────────────────────────────


class TestTierChange:
    def test_to_dict(self) -> None:
        change = TierChange(
            fiber_id="f1",
            memory_type="fact",
            from_tier="warm",
            to_tier="hot",
            reason="access_frequency=5 >= 3",
        )
        d = change.to_dict()
        assert d["fiber_id"] == "f1"
        assert d["from_tier"] == "warm"
        assert d["to_tier"] == "hot"
        assert d["reason"] == "access_frequency=5 >= 3"


# ── Never-activated fallback ─────────────────────────────


class TestNeverActivatedFallback:
    @pytest.mark.asyncio
    async def test_demotes_hot_never_activated(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """HOT memories that were never activated get demoted using created_at fallback."""
        # last_activated_days_ago=None → last_activated=None, but created 100 days ago
        await _create_memory(
            storage, "f-never", tier=MemoryTier.HOT,
            access_frequency=0, last_activated_days_ago=None,
        )
        # Manually backdate the neuron state created_at so it's older than demote threshold
        state = await storage.get_neuron_state("n-f-never")
        assert state is not None
        from dataclasses import replace
        old_state = replace(state, created_at=utcnow() - timedelta(days=100))
        await storage.update_neuron_state(old_state)

        engine = TierEngine(storage, tier_config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.demoted) == 1
        tm = await storage.get_typed_memory("f-never")
        assert tm is not None
        assert tm.tier == "warm"

    @pytest.mark.asyncio
    async def test_archives_warm_never_activated(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """WARM memories never activated get archived using created_at fallback."""
        await _create_memory(
            storage, "f-never", access_frequency=0, last_activated_days_ago=None,
        )
        state = await storage.get_neuron_state("n-f-never")
        assert state is not None
        from dataclasses import replace
        old_state = replace(state, created_at=utcnow() - timedelta(days=100))
        await storage.update_neuron_state(old_state)

        engine = TierEngine(storage, tier_config)
        report = await engine.apply(storage.brain_id, dry_run=False)

        assert len(report.archived) == 1
        tm = await storage.get_typed_memory("f-never")
        assert tm is not None
        assert tm.tier == "cold"


# ── TierConfig validation ────────────────────────────────


class TestTierConfigValidation:
    def test_negative_promote_threshold_clamped(self) -> None:
        cfg = TierConfig(promote_threshold=-5)
        assert cfg.promote_threshold == 1

    def test_zero_max_hot_clamped(self) -> None:
        cfg = TierConfig(max_hot_memories=0)
        assert cfg.max_hot_memories == 1

    def test_negative_demote_days_clamped(self) -> None:
        cfg = TierConfig(demote_inactive_days=-10)
        assert cfg.demote_inactive_days == 1

    def test_valid_values_unchanged(self) -> None:
        cfg = TierConfig(promote_threshold=10, max_hot_memories=50)
        assert cfg.promote_threshold == 10
        assert cfg.max_hot_memories == 50

    def test_cold_archive_days_clamped_to_demote(self) -> None:
        """cold_archive_days is clamped to >= demote_inactive_days."""
        cfg = TierConfig(demote_inactive_days=60, cold_archive_days=30)
        assert cfg.cold_archive_days == 60  # clamped up to match demote

    def test_cold_archive_days_ok_when_greater(self) -> None:
        """cold_archive_days >= demote_inactive_days passes through unchanged."""
        cfg = TierConfig(demote_inactive_days=7, cold_archive_days=90)
        assert cfg.cold_archive_days == 90


# ── History cap ─────────────────────────────────────────


class TestHistoryCap:
    @pytest.mark.asyncio
    async def test_history_capped_at_20(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """promotion_history is capped at 20 entries."""
        # Create a memory with 19 existing history entries
        await _create_memory(
            storage, "f-cap", access_frequency=5, last_activated_days_ago=1,
        )
        tm = await storage.get_typed_memory("f-cap")
        assert tm is not None
        existing_history = [{"from": "warm", "to": "hot", "reason": f"round-{i}", "at": "2026-01-01"} for i in range(19)]
        from dataclasses import replace
        updated = replace(tm, metadata={**tm.metadata, "promotion_history": existing_history})
        await storage.update_typed_memory(updated)

        engine = TierEngine(storage, tier_config)
        await engine.apply(storage.brain_id, dry_run=False)

        tm2 = await storage.get_typed_memory("f-cap")
        assert tm2 is not None
        history = tm2.metadata.get("promotion_history", [])
        assert len(history) == 20  # 19 + 1 new, exactly at cap


# ── Double-report dry-run ───────────────────────────────


class TestDoubleReportDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_idempotent(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """Running evaluate twice returns the same report (no side effects)."""
        await _create_memory(
            storage, "f1", access_frequency=5, last_activated_days_ago=1,
        )
        engine = TierEngine(storage, tier_config)
        report1 = await engine.evaluate(storage.brain_id)
        report2 = await engine.evaluate(storage.brain_id)

        assert len(report1.promoted) == len(report2.promoted)
        assert len(report1.demoted) == len(report2.demoted)
        assert len(report1.archived) == len(report2.archived)
        assert report1.dry_run is True
        assert report2.dry_run is True

        # Memory still WARM after two dry runs
        tm = await storage.get_typed_memory("f1")
        assert tm is not None
        assert tm.tier == "warm"


# ── Brain ID mismatch ──────────────────────────────────


class TestBrainIdMismatch:
    @pytest.mark.asyncio
    async def test_warns_on_brain_id_mismatch(
        self, storage: SQLiteStorage, tier_config: TierConfig
    ) -> None:
        """TierEngine logs warning when brain_id doesn't match storage."""
        await _create_memory(storage, "f1", access_frequency=5)
        engine = TierEngine(storage, tier_config)

        # Should complete without exception (just logs a warning)
        report = await engine.evaluate("nonexistent-brain-id")
        assert isinstance(report, TierReport)
