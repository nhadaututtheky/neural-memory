"""Tests for A6 Phase 1: Tiered Memory Loading — schema, types, CRUD."""

from __future__ import annotations

import pytest

from neural_memory.core.memory_types import (
    DEFAULT_DECAY_RATES,
    DEFAULT_EXPIRY_DAYS,
    TIER_DECAY_FLOORS,
    TIER_DECAY_MULTIPLIERS,
    MemoryTier,
    MemoryType,
    Priority,
    TypedMemory,
)

# ── MemoryTier enum ──────────────────────────────────────


class TestMemoryTier:
    def test_tier_values(self) -> None:
        assert MemoryTier.HOT == "hot"
        assert MemoryTier.WARM == "warm"
        assert MemoryTier.COLD == "cold"

    def test_tier_from_string(self) -> None:
        assert MemoryTier("hot") == MemoryTier.HOT
        assert MemoryTier("warm") == MemoryTier.WARM
        assert MemoryTier("cold") == MemoryTier.COLD

    def test_invalid_tier(self) -> None:
        with pytest.raises(ValueError):
            MemoryTier("invalid")


# ── Tier constants ────────────────────────────────────────


class TestTierConstants:
    def test_decay_floors(self) -> None:
        assert TIER_DECAY_FLOORS["hot"] == 0.5
        assert TIER_DECAY_FLOORS["warm"] == 0.0
        assert TIER_DECAY_FLOORS["cold"] == 0.0

    def test_decay_multipliers(self) -> None:
        assert TIER_DECAY_MULTIPLIERS["hot"] == 0.5
        assert TIER_DECAY_MULTIPLIERS["warm"] == 1.0
        assert TIER_DECAY_MULTIPLIERS["cold"] == 2.0


# ── Boundary type ─────────────────────────────────────────


class TestBoundaryType:
    def test_boundary_in_memory_type(self) -> None:
        assert MemoryType.BOUNDARY == "boundary"
        assert MemoryType("boundary") == MemoryType.BOUNDARY

    def test_boundary_never_expires(self) -> None:
        assert DEFAULT_EXPIRY_DAYS[MemoryType.BOUNDARY] is None

    def test_boundary_slowest_decay(self) -> None:
        assert DEFAULT_DECAY_RATES[MemoryType.BOUNDARY] == 0.01


# ── TypedMemory tier field ────────────────────────────────


class TestTypedMemoryTier:
    def test_default_tier_is_warm(self) -> None:
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT)
        assert tm.tier == "warm"

    def test_create_with_hot_tier(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.PREFERENCE,
            tier=MemoryTier.HOT,
        )
        assert tm.tier == "hot"

    def test_create_with_cold_tier(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.CONTEXT,
            tier=MemoryTier.COLD,
        )
        assert tm.tier == "cold"

    def test_boundary_auto_promotes_to_hot(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.BOUNDARY,
        )
        assert tm.tier == "hot"

    def test_boundary_ignores_explicit_warm(self) -> None:
        """Boundary type always gets HOT tier, even if warm is specified."""
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.BOUNDARY,
            tier=MemoryTier.WARM,
        )
        assert tm.tier == "hot"

    def test_with_tier(self) -> None:
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT)
        assert tm.tier == "warm"
        hot = tm.with_tier(MemoryTier.HOT)
        assert hot.tier == "hot"
        assert hot.fiber_id == tm.fiber_id
        assert hot.memory_type == tm.memory_type

    def test_with_tier_boundary_enforces_hot(self) -> None:
        """Boundary memories stay HOT even via with_tier('cold')."""
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.BOUNDARY)
        assert tm.tier == "hot"
        demoted = tm.with_tier(MemoryTier.COLD)
        assert demoted.tier == "hot"  # invariant enforced
        warm = tm.with_tier(MemoryTier.WARM)
        assert warm.tier == "hot"  # invariant enforced

    def test_with_tier_immutability(self) -> None:
        """with_tier returns new object, doesn't mutate original."""
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT)
        hot = tm.with_tier(MemoryTier.HOT)
        assert tm.tier == "warm"
        assert hot.tier == "hot"
        assert tm is not hot

    def test_with_priority_preserves_tier(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier=MemoryTier.HOT,
        )
        updated = tm.with_priority(Priority.CRITICAL)
        assert updated.tier == "hot"

    def test_verify_preserves_tier(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier=MemoryTier.COLD,
        )
        verified = tm.verify()
        assert verified.tier == "cold"

    def test_extend_expiry_preserves_tier(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier=MemoryTier.HOT,
        )
        extended = tm.extend_expiry(days=30)
        assert extended.tier == "hot"


# ── Schema migration (v36 → v37) ─────────────────────────


class TestSchemaMigration:
    def test_migration_exists(self) -> None:
        from neural_memory.storage.sqlite_schema import MIGRATIONS, SCHEMA_VERSION

        assert SCHEMA_VERSION == 37
        assert (36, 37) in MIGRATIONS
        stmts = MIGRATIONS[(36, 37)]
        assert any("tier" in s for s in stmts)
        assert any("idx_typed_memories_tier" in s for s in stmts)


# ── SQLite CRUD with tier ────────────────────────────────


class TestSqliteTierCrud:
    @pytest.fixture
    async def storage(self, tmp_path):
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.sqlite_store import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        brain = Brain.create(name="tier-test", config=BrainConfig())
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        yield storage
        await storage.close()

    async def _add_fiber(self, storage, fiber_id: str = "f1") -> str:
        from neural_memory.core.fiber import Fiber

        fiber = Fiber(
            id=fiber_id,
            neuron_ids=set(),
            synapse_ids=set(),
            anchor_neuron_id="n1",
            essence="test fiber",
            salience=0.5,
        )
        await storage.add_fiber(fiber)
        return fiber_id

    async def test_add_and_get_with_tier(self, storage) -> None:
        await self._add_fiber(storage)
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.PREFERENCE,
            tier=MemoryTier.HOT,
        )
        await storage.add_typed_memory(tm)
        result = await storage.get_typed_memory("f1")
        assert result is not None
        assert result.tier == "hot"

    async def test_default_tier_is_warm(self, storage) -> None:
        await self._add_fiber(storage)
        tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT)
        await storage.add_typed_memory(tm)
        result = await storage.get_typed_memory("f1")
        assert result is not None
        assert result.tier == "warm"

    async def test_find_by_tier(self, storage) -> None:
        for i, tier in enumerate(["hot", "warm", "cold", "hot"]):
            fid = f"f{i}"
            await self._add_fiber(storage, fid)
            tm = TypedMemory.create(
                fiber_id=fid,
                memory_type=MemoryType.FACT,
                tier=tier,
            )
            await storage.add_typed_memory(tm)

        hot = await storage.find_typed_memories(tier="hot")
        assert len(hot) == 2
        assert all(m.tier == "hot" for m in hot)

        warm = await storage.find_typed_memories(tier="warm")
        assert len(warm) == 1

        cold = await storage.find_typed_memories(tier="cold")
        assert len(cold) == 1

    async def test_find_without_tier_returns_all(self, storage) -> None:
        for i, tier in enumerate(["hot", "warm", "cold"]):
            fid = f"f{i}"
            await self._add_fiber(storage, fid)
            tm = TypedMemory.create(
                fiber_id=fid,
                memory_type=MemoryType.FACT,
                tier=tier,
            )
            await storage.add_typed_memory(tm)

        all_memories = await storage.find_typed_memories()
        assert len(all_memories) == 3

    async def test_update_tier(self, storage) -> None:
        await self._add_fiber(storage)
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier=MemoryTier.WARM,
        )
        await storage.add_typed_memory(tm)

        updated = tm.with_tier(MemoryTier.HOT)
        await storage.update_typed_memory(updated)

        result = await storage.get_typed_memory("f1")
        assert result is not None
        assert result.tier == "hot"

    async def test_boundary_stored_as_hot(self, storage) -> None:
        await self._add_fiber(storage)
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.BOUNDARY,
        )
        assert tm.tier == "hot"
        await storage.add_typed_memory(tm)
        result = await storage.get_typed_memory("f1")
        assert result is not None
        assert result.tier == "hot"
        assert result.memory_type == MemoryType.BOUNDARY

    async def test_migration_creates_tier_column(self, storage) -> None:
        """Integration: verify tier column exists + default after migration."""
        conn = storage._conn
        async with conn.execute("PRAGMA table_info(typed_memories)") as cursor:
            columns = await cursor.fetchall()
            col_names = [c[1] for c in columns]
            assert "tier" in col_names

    async def test_migration_default_tier_warm(self, storage) -> None:
        """Rows inserted without explicit tier get 'warm' from DB default."""
        conn = storage._conn
        await self._add_fiber(storage)
        # Insert directly via SQL without tier to test DB default
        await conn.execute(
            "INSERT INTO typed_memories "
            "(fiber_id, brain_id, memory_type, priority, provenance, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("f1", storage.brain_id, "fact", 5, "test"),
        )
        await conn.commit()
        async with conn.execute(
            "SELECT tier FROM typed_memories WHERE fiber_id = ?", ("f1",)
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "warm"
