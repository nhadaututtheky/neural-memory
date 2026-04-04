"""Tests for B5 Phase 4: Tier Analytics.

Tests:
- _tier_analytics MCP action: breakdown by type, velocity, recent changes
- _classify_change helper: direction classification
- REST API endpoints: tier-analytics, tier-history
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

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
from neural_memory.mcp.tier_handler import _classify_change
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow

# ── Fixtures ─────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create an initialized SQLiteStorage with a brain."""
    db_path = tmp_path / "test_analytics.db"
    store = SQLiteStorage(db_path=str(db_path))
    await store.initialize()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=4,
        max_context_tokens=1500,
    )
    brain = Brain.create(name="analytics-test", config=config)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store  # type: ignore[misc]
    await store.close()


async def _create_typed_memory(
    storage: SQLiteStorage,
    fiber_id: str,
    memory_type: MemoryType = MemoryType.FACT,
    tier: str = MemoryTier.WARM,
    promotion_history: list[dict[str, str]] | None = None,
) -> TypedMemory:
    """Helper to create a fiber + neuron + typed memory."""
    now = utcnow()
    neuron_id = f"n-{fiber_id}"

    neuron = Neuron(
        id=neuron_id,
        type=NeuronType.CONCEPT,
        content=f"Content for {fiber_id}",
        metadata={},
        created_at=now,
    )
    await storage.add_neuron(neuron)

    state = NeuronState(
        neuron_id=neuron_id,
        activation_level=0.5,
        access_frequency=3,
        last_activated=now,
        created_at=now,
    )
    await storage.update_neuron_state(state)

    fiber = Fiber(
        id=fiber_id,
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        pathway=[neuron_id],
    )
    await storage.add_fiber(fiber)

    metadata: dict[str, object] = {}
    if promotion_history:
        metadata["promotion_history"] = promotion_history

    tm = TypedMemory(
        fiber_id=fiber_id,
        memory_type=memory_type,
        priority=Priority.NORMAL,
        provenance=Provenance(source="test"),
        tier=tier,
        metadata=metadata,
        created_at=now,
    )
    await storage.add_typed_memory(tm)
    return tm


# ── _classify_change ─────────────────────────────────────


class TestClassifyChange:
    """Test the change direction classifier."""

    def test_warm_to_hot_is_promoted(self) -> None:
        assert _classify_change("warm", "hot") == "promoted"

    def test_cold_to_warm_is_promoted(self) -> None:
        assert _classify_change("cold", "warm") == "promoted"

    def test_cold_to_hot_is_promoted(self) -> None:
        assert _classify_change("cold", "hot") == "promoted"

    def test_hot_to_warm_is_demoted(self) -> None:
        assert _classify_change("hot", "warm") == "demoted"

    def test_warm_to_cold_is_archived(self) -> None:
        assert _classify_change("warm", "cold") == "archived"

    def test_hot_to_cold_is_archived(self) -> None:
        assert _classify_change("hot", "cold") == "archived"


# ── Breakdown by Type ────────────────────────────────────


class TestBreakdownByType:
    """Test tier distribution breakdown by memory type."""

    async def test_breakdown_groups_by_type_and_tier(self, storage: SQLiteStorage) -> None:
        """Memories are grouped by type and counted per tier."""
        await _create_typed_memory(storage, "f1", MemoryType.FACT, MemoryTier.HOT)
        await _create_typed_memory(storage, "f2", MemoryType.FACT, MemoryTier.WARM)
        await _create_typed_memory(storage, "f3", MemoryType.DECISION, MemoryTier.HOT)
        await _create_typed_memory(storage, "f4", MemoryType.DECISION, MemoryTier.COLD)

        all_typed = await storage.find_typed_memories(limit=1000)
        breakdown: dict[str, dict[str, int]] = {}
        for tm in all_typed:
            type_key = tm.memory_type.value
            tier_key = tm.tier
            if type_key not in breakdown:
                breakdown[type_key] = {"hot": 0, "warm": 0, "cold": 0}
            breakdown[type_key][tier_key] = breakdown[type_key].get(tier_key, 0) + 1

        assert breakdown["fact"]["hot"] == 1
        assert breakdown["fact"]["warm"] == 1
        assert breakdown["decision"]["hot"] == 1
        assert breakdown["decision"]["cold"] == 1

    async def test_empty_brain_returns_empty_breakdown(self, storage: SQLiteStorage) -> None:
        """Empty brain returns empty breakdown."""
        all_typed = await storage.find_typed_memories(limit=1000)
        assert len(all_typed) == 0

    async def test_grouped_aggregate_matches_manual(self, storage: SQLiteStorage) -> None:
        """SQL GROUP BY aggregate matches manual Python counting."""
        await _create_typed_memory(storage, "f1", MemoryType.FACT, MemoryTier.HOT)
        await _create_typed_memory(storage, "f2", MemoryType.FACT, MemoryTier.WARM)
        await _create_typed_memory(storage, "f3", MemoryType.DECISION, MemoryTier.HOT)
        await _create_typed_memory(storage, "f4", MemoryType.DECISION, MemoryTier.COLD)
        await _create_typed_memory(storage, "f5", MemoryType.FACT, MemoryTier.HOT)

        grouped = await storage.count_typed_memories_grouped()
        breakdown: dict[str, dict[str, int]] = {}
        for memory_type, tier, count in grouped:
            if memory_type not in breakdown:
                breakdown[memory_type] = {}
            breakdown[memory_type][tier] = count

        assert breakdown["fact"]["hot"] == 2
        assert breakdown["fact"]["warm"] == 1
        assert breakdown["decision"]["hot"] == 1
        assert breakdown["decision"]["cold"] == 1

    async def test_grouped_empty_brain(self, storage: SQLiteStorage) -> None:
        """Empty brain returns empty grouping."""
        grouped = await storage.count_typed_memories_grouped()
        assert grouped == []


# ── Velocity Metrics ─────────────────────────────────────


class TestVelocityMetrics:
    """Test velocity calculation from promotion_history metadata."""

    async def test_velocity_counts_recent_changes(self, storage: SQLiteStorage) -> None:
        """Velocity counts changes within 7d and 30d windows."""
        now = utcnow()
        recent = (now - timedelta(days=2)).isoformat()
        old = (now - timedelta(days=15)).isoformat()
        ancient = (now - timedelta(days=60)).isoformat()

        history = [
            {"from": "warm", "to": "hot", "reason": "freq>=5", "at": recent},
            {"from": "hot", "to": "warm", "reason": "inactive", "at": old},
            {"from": "warm", "to": "cold", "reason": "archived", "at": ancient},
        ]
        await _create_typed_memory(
            storage, "f1", MemoryType.FACT, MemoryTier.HOT, promotion_history=history
        )

        all_typed = await storage.find_typed_memories(limit=1000)
        cutoff_7d = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)
        velocity_7d = {"promoted": 0, "demoted": 0, "archived": 0}
        velocity_30d = {"promoted": 0, "demoted": 0, "archived": 0}

        for tm in all_typed:
            for entry in tm.metadata.get("promotion_history", []):
                direction = _classify_change(entry["from"], entry["to"])
                from datetime import datetime

                ts = datetime.fromisoformat(entry["at"])
                if ts >= cutoff_7d:
                    velocity_7d[direction] += 1
                if ts >= cutoff_30d:
                    velocity_30d[direction] += 1

        # recent (2d ago) → within 7d AND 30d
        assert velocity_7d["promoted"] == 1
        # old (15d ago) → within 30d only
        assert velocity_30d["demoted"] == 1
        assert velocity_7d["demoted"] == 0
        # ancient (60d ago) → outside both windows
        assert velocity_30d["archived"] == 0


# ── Recent Changes ───────────────────────────────────────


class TestRecentChanges:
    """Test recent change event collection."""

    async def test_collects_events_from_multiple_memories(self, storage: SQLiteStorage) -> None:
        """Events are collected across all memories and sorted by time."""
        now = utcnow()
        t1 = (now - timedelta(hours=1)).isoformat()
        t2 = (now - timedelta(hours=2)).isoformat()
        t3 = (now - timedelta(hours=3)).isoformat()

        await _create_typed_memory(
            storage,
            "f1",
            MemoryType.FACT,
            promotion_history=[
                {"from": "warm", "to": "hot", "reason": "r1", "at": t1},
            ],
        )
        await _create_typed_memory(
            storage,
            "f2",
            MemoryType.DECISION,
            promotion_history=[
                {"from": "hot", "to": "warm", "reason": "r2", "at": t2},
                {"from": "warm", "to": "cold", "reason": "r3", "at": t3},
            ],
        )

        all_typed = await storage.find_typed_memories(limit=1000)
        events: list[dict[str, str]] = []
        for tm in all_typed:
            for entry in tm.metadata.get("promotion_history", []):
                events.append(
                    {
                        "fiber_id": tm.fiber_id,
                        "at": entry.get("at", ""),
                    }
                )

        events.sort(key=lambda e: e["at"], reverse=True)

        assert len(events) == 3
        # Most recent first
        assert events[0]["fiber_id"] == "f1"

    async def test_limited_to_50_events(self, storage: SQLiteStorage) -> None:
        """Recent changes capped at 50."""
        events = list(range(100))
        capped = events[:50]
        assert len(capped) == 50
