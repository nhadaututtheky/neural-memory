"""Tests for compression/tier integration into InfinityDB engine.

Verifies that:
1. Neurons get initial tier classification on add
2. Auto-promote on access (get_neuron)
3. Demote sweep reclassifies stale neurons
4. Tier stats reflect correct distribution
5. TierConfig is customizable
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from neural_memory.pro.infinitydb.compressor import CompressionTier
from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.tier_manager import TierConfig


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_brain"


@pytest.fixture
def dims() -> int:
    return 32


async def _make_db(
    db_dir: Path, dims: int = 32, tier_config: TierConfig | None = None
) -> InfinityDB:
    db = InfinityDB(db_dir, brain_id="test", dimensions=dims, tier_config=tier_config)
    await db.open()
    return db


def _old_timestamp(days_ago: int) -> str:
    """Create an ISO timestamp N days in the past."""
    dt = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days_ago)
    return dt.isoformat()


# --- Initial Tier Classification ---


class TestInitialTier:
    """New neurons get correct initial tier."""

    @pytest.mark.asyncio
    async def test_new_neuron_gets_tier(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("test", neuron_id="n1")

        neuron = await db.get_neuron("n1")
        assert neuron is not None
        assert "tier" in neuron
        # New neuron with default priority=5, just created → WARM (access_count=0 < 5)
        assert neuron["tier"] == int(CompressionTier.WARM)
        await db.close()

    @pytest.mark.asyncio
    async def test_high_priority_neuron_is_active(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("important", neuron_id="n1", priority=9)

        neuron = await db.get_neuron("n1")
        assert neuron is not None
        assert neuron["tier"] == int(CompressionTier.ACTIVE)
        await db.close()

    @pytest.mark.asyncio
    async def test_zero_priority_is_crystal(self, db_dir: Path, dims: int) -> None:
        """Priority 0 neurons go straight to CRYSTAL."""
        config = TierConfig()
        db = await _make_db(db_dir, dims, tier_config=config)
        await db.add_neuron("archived", neuron_id="n1", priority=0)

        neuron = await db.get_neuron("n1")
        assert neuron is not None
        assert neuron["tier"] == int(CompressionTier.CRYSTAL)
        await db.close()


# --- Auto-Promote on Access ---


class TestAutoPromote:
    """get_neuron auto-promotes tier when access pattern warrants it."""

    @pytest.mark.asyncio
    async def test_promote_on_access_with_high_priority(self, db_dir: Path, dims: int) -> None:
        """Neuron with priority=9 should be ACTIVE regardless of tier."""
        db = await _make_db(db_dir, dims)
        await db.add_neuron("test", neuron_id="n1", priority=9)
        await db.flush()

        # Force tier to WARM manually
        result = db._metadata.get_by_id("n1")
        assert result is not None
        slot, _ = result
        db._metadata.update(slot, {"tier": int(CompressionTier.WARM)})

        # Access should promote back to ACTIVE
        neuron = await db.get_neuron("n1")
        assert neuron is not None
        assert neuron["tier"] == int(CompressionTier.ACTIVE)
        await db.close()


# --- Demote Sweep ---


class TestDemoteSweep:
    """demote_sweep reclassifies neurons based on access patterns."""

    @pytest.mark.asyncio
    async def test_demote_stale_neurons(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(warm_after_days=7, cool_after_days=30, frozen_after_days=90)
        db = await _make_db(db_dir, dims, tier_config=config)
        await db.add_neuron("old", neuron_id="n1", priority=5)
        await db.flush()

        # Manually backdate accessed_at to 40 days ago
        result = db._metadata.get_by_id("n1")
        assert result is not None
        slot, _ = result
        db._metadata.update(
            slot,
            {
                "accessed_at": _old_timestamp(40),
                "tier": int(CompressionTier.ACTIVE),
            },
        )

        demoted = await db.demote_sweep()
        assert "COOL" in demoted
        assert demoted["COOL"] == 1

        # Check tier directly (get_neuron would auto-promote on access)
        result = db._metadata.get_by_id("n1")
        assert result is not None
        _, meta = result
        assert meta["tier"] == int(CompressionTier.COOL)
        await db.close()

    @pytest.mark.asyncio
    async def test_demote_to_frozen(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(frozen_after_days=90)
        db = await _make_db(db_dir, dims, tier_config=config)
        await db.add_neuron("ancient", neuron_id="n1", priority=3)
        await db.flush()

        result = db._metadata.get_by_id("n1")
        assert result is not None
        slot, _ = result
        db._metadata.update(
            slot,
            {
                "accessed_at": _old_timestamp(100),
                "tier": int(CompressionTier.ACTIVE),
            },
        )

        demoted = await db.demote_sweep()
        assert "FROZEN" in demoted
        await db.close()

    @pytest.mark.asyncio
    async def test_no_demotion_for_fresh_neurons(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("fresh", neuron_id="n1", priority=5)

        demoted = await db.demote_sweep()
        assert len(demoted) == 0
        await db.close()

    @pytest.mark.asyncio
    async def test_high_priority_never_demoted(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(high_priority_threshold=8)
        db = await _make_db(db_dir, dims, tier_config=config)
        await db.add_neuron("critical", neuron_id="n1", priority=9)
        await db.flush()

        # Even with old access time, high priority stays ACTIVE
        result = db._metadata.get_by_id("n1")
        assert result is not None
        slot, _ = result
        db._metadata.update(slot, {"accessed_at": _old_timestamp(200)})

        demoted = await db.demote_sweep()
        assert len(demoted) == 0
        await db.close()


# --- Tier Stats ---


class TestTierStats:
    """get_tier_stats returns correct distribution."""

    @pytest.mark.asyncio
    async def test_stats_reflect_tier_distribution(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        # High priority → ACTIVE
        await db.add_neuron("active", neuron_id="n1", priority=9)
        # Normal priority → WARM
        await db.add_neuron("warm", neuron_id="n2", priority=5)
        await db.add_neuron("warm2", neuron_id="n3", priority=5)

        stats = await db.get_tier_stats()
        tiers = stats["tiers"]
        assert tiers["total"] == 3
        assert tiers["active"] == 1
        assert tiers["warm"] == 2
        assert "savings" in stats
        await db.close()

    @pytest.mark.asyncio
    async def test_stats_with_mixed_tiers(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(warm_after_days=7, cool_after_days=30)
        db = await _make_db(db_dir, dims, tier_config=config)

        await db.add_neuron("n", neuron_id="n1", priority=9)  # ACTIVE
        await db.add_neuron("n", neuron_id="n2", priority=5)  # WARM
        await db.add_neuron("n", neuron_id="n3", priority=0)  # CRYSTAL
        await db.flush()

        # Backdate n2 to force COOL
        result = db._metadata.get_by_id("n2")
        assert result is not None
        slot, _ = result
        db._metadata.update(
            slot,
            {
                "accessed_at": _old_timestamp(40),
                "tier": int(CompressionTier.COOL),
            },
        )

        stats = await db.get_tier_stats()
        tiers = stats["tiers"]
        assert tiers["active"] == 1
        assert tiers["cool"] == 1
        assert tiers["crystal"] == 1
        await db.close()

    @pytest.mark.asyncio
    async def test_savings_estimate(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        for i in range(10):
            await db.add_neuron(f"n{i}", neuron_id=f"n{i}", priority=5)

        stats = await db.get_tier_stats()
        savings = stats["savings"]
        assert "compression_ratio" in savings
        assert "savings_percent" in savings
        assert savings["all_active_bytes"] > 0
        await db.close()


# --- TierConfig Customization ---


class TestTierConfig:
    """Custom TierConfig is respected."""

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(
            warm_after_days=3,
            cool_after_days=10,
            frozen_after_days=30,
            min_access_for_active=2,
            high_priority_threshold=7,
        )
        db = await _make_db(db_dir, dims, tier_config=config)

        # priority=7 >= threshold → ACTIVE
        await db.add_neuron("p7", neuron_id="n1", priority=7)
        neuron = await db.get_neuron("n1")
        assert neuron is not None
        assert neuron["tier"] == int(CompressionTier.ACTIVE)
        await db.close()

    @pytest.mark.asyncio
    async def test_disabled_auto_demote(self, db_dir: Path, dims: int) -> None:
        config = TierConfig(auto_demote_enabled=False)
        db = await _make_db(db_dir, dims, tier_config=config)
        await db.add_neuron("test", neuron_id="n1", priority=5)
        await db.flush()

        result = db._metadata.get_by_id("n1")
        assert result is not None
        slot, _ = result
        db._metadata.update(slot, {"accessed_at": _old_timestamp(200)})

        demoted = await db.demote_sweep()
        assert len(demoted) == 0  # auto-demote disabled
        await db.close()
