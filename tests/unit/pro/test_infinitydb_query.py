"""Tests for InfinityDB Phase 5 — Query Planner + Multi-dimensional queries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.graph_store import GraphStore
from neural_memory.pro.infinitydb.hnsw_index import HNSWIndex
from neural_memory.pro.infinitydb.metadata_store import MetadataStore
from neural_memory.pro.infinitydb.query_planner import (
    QueryExecutor,
    QueryPlan,
    rrf_fuse,
)

# ── RRF Unit Tests ──


class TestRRFFusion:
    def test_single_list(self) -> None:
        result = rrf_fuse([["a", "b", "c"]])
        ids = [nid for nid, _ in result]
        assert ids == ["a", "b", "c"]

    def test_two_lists_agreement(self) -> None:
        """Same order in both lists → same order in result."""
        result = rrf_fuse([["a", "b", "c"], ["a", "b", "c"]])
        ids = [nid for nid, _ in result]
        assert ids == ["a", "b", "c"]

    def test_two_lists_disagreement(self) -> None:
        """Items in both lists always outscore items in only one."""
        result = rrf_fuse([["a", "b"], ["c", "b"]])
        ids = [nid for nid, _ in result]
        # "b" is in both lists → highest combined RRF
        assert ids[0] == "b"

    def test_weighted_lists(self) -> None:
        """Higher weight = more influence on final rank."""
        result = rrf_fuse(
            [["a", "b", "c"], ["c", "b", "a"]],
            weights=[10.0, 1.0],
        )
        ids = [nid for nid, _ in result]
        assert ids[0] == "a"  # "a" is rank 0 in the heavier list

    def test_empty_lists(self) -> None:
        assert rrf_fuse([]) == []

    def test_disjoint_lists(self) -> None:
        """Items only in one list still get ranked."""
        result = rrf_fuse([["a", "b"], ["c", "d"]])
        ids = [nid for nid, _ in result]
        assert set(ids) == {"a", "b", "c", "d"}

    def test_scores_decrease(self) -> None:
        result = rrf_fuse([["a", "b", "c", "d"]])
        scores = [s for _, s in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ── QueryExecutor Unit Tests ──


@pytest.fixture
def stores(tmp_path: Path) -> tuple[MetadataStore, HNSWIndex, GraphStore]:
    ms = MetadataStore(tmp_path / "test.meta")
    ms.open()
    idx = HNSWIndex(tmp_path / "test.idx", dimensions=8)
    idx.open(max_elements=1024)
    gs = GraphStore(tmp_path / "test.graph")
    gs.open()
    return ms, idx, gs


def _add_neuron(
    ms: MetadataStore,
    idx: HNSWIndex,
    nid: str,
    content: str,
    embedding: list[float] | None = None,
    slot: int = -1,
    priority: int = 5,
    neuron_type: str = "fact",
    tags: list[str] | None = None,
) -> None:
    vec_slot = slot
    if embedding is not None:
        vec = np.array(embedding, dtype=np.float32)
        idx.add(slot, vec)
        vec_slot = slot

    meta = {
        "id": nid,
        "type": neuron_type,
        "content": content,
        "priority": priority,
        "activation_level": 1.0,
        "created_at": "2026-03-20T12:00:00",
        "updated_at": "2026-03-20T12:00:00",
        "accessed_at": "2026-03-20T12:00:00",
        "access_count": 0,
        "ephemeral": False,
        "tags": tags or [],
        "vec_slot": vec_slot,
    }
    ms.add(slot if slot >= 0 else ms.next_free_slot(), meta)


class TestQueryExecutorVector:
    def test_vector_only_query(self, stores: tuple) -> None:
        ms, idx, gs = stores
        # Add neurons with similar embeddings
        _add_neuron(ms, idx, "n1", "hello", [1, 0, 0, 0, 0, 0, 0, 0], slot=0)
        _add_neuron(ms, idx, "n2", "world", [0.9, 0.1, 0, 0, 0, 0, 0, 0], slot=1)
        _add_neuron(ms, idx, "n3", "far", [0, 0, 0, 0, 0, 0, 0, 1], slot=2)

        plan = QueryPlan(
            query_vector=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            limit=10,
        )
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 3
        # n1 should be most similar
        assert results[0]["id"] == "n1"


class TestQueryExecutorFilters:
    def test_type_filter(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "fact 1", neuron_type="fact", slot=-1)
        _add_neuron(ms, idx, "n2", "decision 1", neuron_type="decision", slot=-2)
        _add_neuron(ms, idx, "n3", "fact 2", neuron_type="fact", slot=-3)

        plan = QueryPlan(neuron_type="fact", limit=10)
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 2
        assert all(r["type"] == "fact" for r in results)

    def test_content_filter(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "Python is great", slot=-1)
        _add_neuron(ms, idx, "n2", "JavaScript rocks", slot=-2)
        _add_neuron(ms, idx, "n3", "Python rules", slot=-3)

        plan = QueryPlan(content_contains="Python", limit=10)
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 2

    def test_tag_filter(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "a", tags=["python", "web"], slot=-1)
        _add_neuron(ms, idx, "n2", "b", tags=["rust", "cli"], slot=-2)
        _add_neuron(ms, idx, "n3", "c", tags=["python", "ml"], slot=-3)

        plan = QueryPlan(tags=["python"], limit=10)
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 2

    def test_ephemeral_filter(self, stores: tuple) -> None:
        ms, idx, gs = stores
        meta1 = {
            "id": "n1",
            "type": "fact",
            "content": "perm",
            "priority": 5,
            "ephemeral": False,
            "tags": [],
            "vec_slot": -1,
            "created_at": "2026-03-20T12:00:00",
            "updated_at": "2026-03-20T12:00:00",
            "accessed_at": "2026-03-20T12:00:00",
            "access_count": 0,
            "activation_level": 1.0,
        }
        meta2 = {**meta1, "id": "n2", "content": "temp", "ephemeral": True}
        ms.add(-1, meta1)
        ms.add(-2, meta2)

        plan = QueryPlan(ephemeral=False, limit=10)
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)
        assert len(results) == 1
        assert results[0]["id"] == "n1"


class TestQueryExecutorMultiDim:
    def test_vector_plus_priority(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "close", [1, 0, 0, 0, 0, 0, 0, 0], slot=0, priority=3)
        _add_neuron(ms, idx, "n2", "far", [0, 0, 0, 0, 0, 0, 0, 1], slot=1, priority=9)
        _add_neuron(ms, idx, "n3", "medium", [0.5, 0.5, 0, 0, 0, 0, 0, 0], slot=2, priority=7)

        plan = QueryPlan(
            query_vector=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            priority_weight=0.5,
            limit=10,
        )
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 3
        # n1 is closest by vector but low priority
        # n3 is medium vector + high priority
        # The exact ordering depends on RRF fusion

    def test_vector_plus_graph(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "seed", [1, 0, 0, 0, 0, 0, 0, 0], slot=0)
        _add_neuron(ms, idx, "n2", "neighbor", [0, 0, 0, 0, 0, 0, 0, 1], slot=1)
        _add_neuron(ms, idx, "n3", "distant", [0.5, 0.5, 0, 0, 0, 0, 0, 0], slot=2)

        gs.add_edge("n1", "n2", edge_type="related")

        plan = QueryPlan(
            query_vector=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            graph_seed_ids=["n1"],
            graph_weight=0.5,
            limit=10,
        )
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)

        assert len(results) == 3

    def test_pagination(self, stores: tuple) -> None:
        ms, idx, gs = stores
        for i in range(10):
            _add_neuron(ms, idx, f"n{i}", f"content-{i}", slot=-(i + 1))

        plan = QueryPlan(limit=3, offset=2)
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)
        assert len(results) == 3

    def test_min_score_filter(self, stores: tuple) -> None:
        ms, idx, gs = stores
        _add_neuron(ms, idx, "n1", "a", [1, 0, 0, 0, 0, 0, 0, 0], slot=0)
        _add_neuron(ms, idx, "n2", "b", [0, 1, 0, 0, 0, 0, 0, 0], slot=1)

        plan = QueryPlan(
            query_vector=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            min_score=0.02,  # very high threshold
            limit=10,
        )
        executor = QueryExecutor(ms, idx, gs)
        results = executor.execute(plan)
        # Some results may be filtered out by min_score
        assert all(r["_score"] >= 0.02 for r in results)


# ── InfinityDB Integration Tests ──


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "brains"


class TestInfinityDBQuery:
    async def test_query_with_vector(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec3 = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        await db.add_neuron("alpha", neuron_id="n1", embedding=vec1)
        await db.add_neuron("beta", neuron_id="n2", embedding=vec2)
        await db.add_neuron("gamma", neuron_id="n3", embedding=vec3)

        plan = QueryPlan(
            query_vector=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            limit=3,
        )
        results = await db.query(plan)

        assert len(results) == 3
        assert results[0]["id"] == "n1"  # exact match

        await db.close()

    async def test_query_with_type_filter(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("fact1", neuron_id="n1", neuron_type="fact")
        await db.add_neuron("decision1", neuron_id="n2", neuron_type="decision")
        await db.add_neuron("fact2", neuron_id="n3", neuron_type="fact")

        plan = QueryPlan(neuron_type="fact", limit=10)
        results = await db.query(plan)

        assert len(results) == 2
        assert all(r["type"] == "fact" for r in results)

        await db.close()

    async def test_query_multidim(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        await db.add_neuron("seed", neuron_id="n1", embedding=vec1, priority=9)
        await db.add_neuron("linked", neuron_id="n2", embedding=vec2, priority=5)
        await db.add_synapse("n1", "n2")

        plan = QueryPlan(
            query_vector=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            graph_seed_ids=["n1"],
            graph_weight=0.5,
            priority_weight=0.3,
            limit=10,
        )
        results = await db.query(plan)
        assert len(results) == 2

        await db.close()

    async def test_query_empty_db(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        plan = QueryPlan(
            query_vector=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            vector_weight=1.0,
            limit=10,
        )
        results = await db.query(plan)
        assert results == []

        await db.close()
