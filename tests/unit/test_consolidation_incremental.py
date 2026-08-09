"""Unit tests for dirty-set incremental consolidation (Phase 6)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
from neural_memory.engine.consolidation_candidates import build_merge_candidates
from neural_memory.engine.consolidation_incremental import DirtySet, build_dirty_set
from neural_memory.utils.timeutils import utcnow


class TestCheckpoint:
    def test_rejects_negative_sequence(self) -> None:
        with pytest.raises(ValueError):
            ConsolidationCheckpoint.create(brain_id="b", strategy="prune", last_sequence=-1)

    def test_normalizes_strategy(self) -> None:
        cp = ConsolidationCheckpoint.create(brain_id="b", strategy=" MERGE ", last_sequence=3)
        assert cp.strategy == "merge"
        assert cp.last_sequence == 3

    def test_frozen(self) -> None:
        cp = ConsolidationCheckpoint.create(brain_id="b", strategy="prune", last_sequence=0)
        with pytest.raises(FrozenInstanceError):
            cp.last_sequence = 5  # type: ignore[misc]


class TestDirtySet:
    def test_empty_is_empty(self) -> None:
        d = DirtySet(
            neuron_ids=frozenset(),
            synapse_ids=frozenset(),
            fiber_ids=frozenset(),
            high_watermark=0,
            truncated=False,
        )
        assert d.is_empty
        assert d.total_entities == 0


class _FakeChange:
    def __init__(
        self,
        id: int,
        entity_type: str,
        entity_id: str,
        payload: dict | None = None,
    ) -> None:
        self.id = id
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.payload = payload or {}
        self.operation = "insert"
        self.brain_id = "b"
        self.device_id = ""
        self.changed_at = utcnow()
        self.synced = False


class _FakeStorage:
    def __init__(self, changes: list[_FakeChange]) -> None:
        self._changes = changes
        self.brain_id = "b"
        self._current_brain_id = "b"

    async def get_change_log_max_sequence(self) -> int:
        return max((c.id for c in self._changes), default=0)

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list:
        rows = [c for c in self._changes if c.id > sequence]
        return rows[:limit]


@pytest.mark.asyncio
async def test_build_dirty_set_classifies_entities() -> None:
    changes = [
        _FakeChange(1, "neuron", "n1"),
        _FakeChange(2, "synapse", "s1", {"source_id": "n1", "target_id": "n2"}),
        _FakeChange(3, "fiber", "f1", {"neuron_ids": ["n3"], "anchor_neuron_id": "n3"}),
    ]
    storage = _FakeStorage(changes)
    dirty = await build_dirty_set(storage, "prune", from_sequence=0)  # type: ignore[arg-type]
    assert "n1" in dirty.neuron_ids
    assert "n2" in dirty.neuron_ids  # from synapse endpoints
    assert "n3" in dirty.neuron_ids
    assert "s1" in dirty.synapse_ids
    assert "f1" in dirty.fiber_ids
    assert dirty.high_watermark == 3
    assert dirty.truncated is False


@pytest.mark.asyncio
async def test_build_dirty_set_truncates() -> None:
    changes = [_FakeChange(i, "neuron", f"n{i}") for i in range(1, 21)]
    storage = _FakeStorage(changes)
    dirty = await build_dirty_set(
        storage,  # type: ignore[arg-type]
        "merge",
        max_changes=5,
        from_sequence=0,
    )
    assert dirty.truncated is True
    assert dirty.change_count == 5
    assert dirty.high_watermark == 5  # cursor when truncated


@pytest.mark.asyncio
async def test_build_dirty_set_defers_past_watermark() -> None:
    storage = _FakeStorage([_FakeChange(10, "neuron", "n10")])
    dirty = await build_dirty_set(storage, "prune", from_sequence=10)  # type: ignore[arg-type]
    assert dirty.is_empty
    assert dirty.high_watermark == 10


def test_build_merge_candidates_bounded() -> None:
    class F:
        def __init__(self, id: str, tags: set[str], nids: set[str]) -> None:
            self.id = id
            self.tags = tags
            self.neuron_ids = nids
            self.metadata = {"content_hash": hash(id) & 0xFFFFFFFF, "type": "fact"}

    fibers = [F(f"f{i}", {"auth", f"t{i % 3}"}, {f"n{i}", "shared"}) for i in range(20)]
    dirty = DirtySet(
        neuron_ids=frozenset({"n0"}),
        synapse_ids=frozenset(),
        fiber_ids=frozenset({"f0", "f1"}),
        high_watermark=2,
        truncated=False,
    )
    buckets = build_merge_candidates(dirty, fibers, max_candidates=50, max_bucket=10)
    assert buckets.pair_count >= 1
    # All pairs involve at least one seed eventually via buckets
    assert isinstance(buckets.truncated, bool)
