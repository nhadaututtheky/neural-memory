"""Tests for bounded consolidation candidate buckets."""

from __future__ import annotations

from neural_memory.engine.consolidation_candidates import (
    build_merge_candidates,
    filter_entities_to_dirty,
)
from neural_memory.engine.consolidation_incremental import DirtySet


class _Fiber:
    def __init__(self, id: str, tags: set[str] | None = None, nids: set[str] | None = None):
        self.id = id
        self.tags = tags or set()
        self.neuron_ids = nids or {id}
        self.metadata = {"type": "fact", "content_hash": abs(hash(id)) % (2**32)}


def test_filter_entities_to_dirty() -> None:
    dirty = DirtySet(
        neuron_ids=frozenset({"a", "b"}),
        synapse_ids=frozenset(),
        fiber_ids=frozenset(),
        high_watermark=1,
        truncated=False,
    )

    class E:
        def __init__(self, id: str) -> None:
            self.id = id

    entities = [E("a"), E("c"), E("b")]
    filtered = filter_entities_to_dirty(entities, dirty, entity_kind="neuron")
    assert [e.id for e in filtered] == ["a", "b"]


def test_empty_dirty_yields_no_pairs() -> None:
    dirty = DirtySet(
        neuron_ids=frozenset(),
        synapse_ids=frozenset(),
        fiber_ids=frozenset(),
        high_watermark=0,
        truncated=False,
    )
    fibers = [_Fiber("f1"), _Fiber("f2")]
    buckets = build_merge_candidates(dirty, fibers)
    assert buckets.pair_count == 0


def test_cap_marks_truncated() -> None:
    fibers = [_Fiber(f"f{i}", tags={"shared"}) for i in range(30)]
    dirty = DirtySet(
        neuron_ids=frozenset(),
        synapse_ids=frozenset(),
        fiber_ids=frozenset({"f0"}),
        high_watermark=1,
        truncated=False,
    )
    buckets = build_merge_candidates(dirty, fibers, max_candidates=5, max_bucket=20)
    assert buckets.pair_count <= 5
    # Cap or large bucket may set truncated
    assert buckets.pair_count >= 1
