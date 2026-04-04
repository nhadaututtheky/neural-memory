"""Tests for InfinityDB Phase 2 — Graph Layer (synapses, traversal, fibers)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.fiber_store import FiberStore
from neural_memory.pro.infinitydb.graph_store import GraphStore

# ── GraphStore unit tests ──


@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    gs = GraphStore(tmp_path / "test.graph")
    gs.open()
    return gs


class TestGraphStoreBasic:
    def test_add_edge(self, graph_store: GraphStore) -> None:
        eid = graph_store.add_edge("n1", "n2", edge_type="related")
        assert eid
        assert graph_store.edge_count == 1

    def test_get_outgoing(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2", edge_type="causes")
        graph_store.add_edge("n1", "n3", edge_type="related")
        edges = graph_store.get_outgoing("n1")
        assert len(edges) == 2
        targets = {e["target_id"] for e in edges}
        assert targets == {"n2", "n3"}

    def test_get_incoming(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n3")
        graph_store.add_edge("n2", "n3")
        edges = graph_store.get_incoming("n3")
        assert len(edges) == 2
        sources = {e["source_id"] for e in edges}
        assert sources == {"n1", "n2"}

    def test_get_edges_between(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2", edge_type="a")
        graph_store.add_edge("n1", "n2", edge_type="b")
        graph_store.add_edge("n1", "n3", edge_type="c")
        edges = graph_store.get_edges_between("n1", "n2")
        assert len(edges) == 2
        types = {e["type"] for e in edges}
        assert types == {"a", "b"}

    def test_get_neighbors_outgoing(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n1", "n3")
        nb = graph_store.get_neighbors("n1", direction="outgoing")
        assert set(nb) == {"n2", "n3"}

    def test_get_neighbors_incoming(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n3")
        graph_store.add_edge("n2", "n3")
        nb = graph_store.get_neighbors("n3", direction="incoming")
        assert set(nb) == {"n1", "n2"}

    def test_get_neighbors_both(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n3", "n2")
        nb = graph_store.get_neighbors("n2", direction="both")
        assert set(nb) == {"n1", "n3"}

    def test_get_neighbors_with_type_filter(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2", edge_type="causes")
        graph_store.add_edge("n1", "n3", edge_type="related")
        nb = graph_store.get_neighbors("n1", direction="outgoing", edge_type="causes")
        assert nb == ["n2"]

    def test_get_neighbors_invalid_direction(self, graph_store: GraphStore) -> None:
        with pytest.raises(ValueError, match="Invalid direction"):
            graph_store.get_neighbors("n1", direction="sideways")

    def test_delete_edge(self, graph_store: GraphStore) -> None:
        eid = graph_store.add_edge("n1", "n2")
        assert graph_store.edge_count == 1
        assert graph_store.delete_edge(eid)
        assert graph_store.edge_count == 0
        assert graph_store.get_outgoing("n1") == []

    def test_delete_edge_not_found(self, graph_store: GraphStore) -> None:
        assert not graph_store.delete_edge("nonexistent")

    def test_delete_neuron_edges(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n1", "n3")
        graph_store.add_edge("n4", "n1")
        assert graph_store.edge_count == 3
        deleted = graph_store.delete_neuron_edges("n1")
        assert deleted == 3
        assert graph_store.edge_count == 0

    def test_update_edge(self, graph_store: GraphStore) -> None:
        eid = graph_store.add_edge("n1", "n2", weight=1.0)
        assert graph_store.update_edge(eid, {"weight": 0.5, "type": "strong"})
        edge = graph_store.get_edge_by_id(eid)
        assert edge is not None
        assert edge["weight"] == 0.5
        assert edge["type"] == "strong"

    def test_update_edge_rejects_dangerous_fields(self, graph_store: GraphStore) -> None:
        eid = graph_store.add_edge("n1", "n2")
        assert not graph_store.update_edge(eid, {"source_id": "evil", "id": "hack"})

    def test_get_subgraph(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n2", "n3")
        graph_store.add_edge("n3", "n4")
        edges = graph_store.get_subgraph(["n1", "n2", "n3"])
        assert len(edges) == 2  # n1->n2, n2->n3 (n3->n4 excluded)

    def test_edge_metadata(self, graph_store: GraphStore) -> None:
        eid = graph_store.add_edge("n1", "n2", metadata={"reason": "test"})
        edge = graph_store.get_edge_by_id(eid)
        assert edge is not None
        assert edge["metadata"]["reason"] == "test"

    def test_iter_all_edges(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n2", "n3")
        all_edges = graph_store.iter_all_edges()
        assert len(all_edges) == 2


class TestGraphStoreBFS:
    def test_bfs_linear(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n2", "n3")
        graph_store.add_edge("n3", "n4")
        result = graph_store.bfs("n1", max_depth=3)
        ids = [nid for nid, _ in result]
        assert ids == ["n1", "n2", "n3", "n4"]

    def test_bfs_depth_limit(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n2", "n3")
        graph_store.add_edge("n3", "n4")
        result = graph_store.bfs("n1", max_depth=1)
        ids = [nid for nid, _ in result]
        assert ids == ["n1", "n2"]

    def test_bfs_branching(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n1", "n3")
        graph_store.add_edge("n2", "n4")
        result = graph_store.bfs("n1", max_depth=2)
        assert len(result) == 4
        depths = dict(result)
        assert depths["n1"] == 0
        assert depths["n2"] == 1
        assert depths["n3"] == 1
        assert depths["n4"] == 2

    def test_bfs_cycle(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n2", "n3")
        graph_store.add_edge("n3", "n1")
        result = graph_store.bfs("n1", max_depth=10)
        assert len(result) == 3

    def test_bfs_max_nodes(self, graph_store: GraphStore) -> None:
        for i in range(100):
            graph_store.add_edge("root", f"n{i}")
        result = graph_store.bfs("root", max_depth=1, max_nodes=10)
        assert len(result) <= 10

    def test_bfs_incoming(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n2", "n1")
        graph_store.add_edge("n3", "n2")
        result = graph_store.bfs("n1", max_depth=2, direction="incoming")
        ids = {nid for nid, _ in result}
        assert "n2" in ids
        assert "n3" in ids

    def test_bfs_both_directions(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        graph_store.add_edge("n3", "n1")
        result = graph_store.bfs("n1", max_depth=1, direction="both")
        ids = {nid for nid, _ in result}
        assert ids == {"n1", "n2", "n3"}

    def test_bfs_with_type_filter(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2", edge_type="causes")
        graph_store.add_edge("n1", "n3", edge_type="related")
        result = graph_store.bfs("n1", max_depth=1, edge_type="causes")
        ids = [nid for nid, _ in result]
        assert "n2" in ids
        assert "n3" not in ids

    def test_bfs_zero_depth(self, graph_store: GraphStore) -> None:
        graph_store.add_edge("n1", "n2")
        result = graph_store.bfs("n1", max_depth=0)
        assert result == []


class TestGraphStorePersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "test.graph"

        gs1 = GraphStore(path)
        gs1.open()
        gs1.add_edge("n1", "n2", edge_type="causes", weight=0.8)
        gs1.add_edge("n2", "n3")
        gs1.close()

        gs2 = GraphStore(path)
        gs2.open()
        assert gs2.edge_count == 2
        edges = gs2.get_outgoing("n1")
        assert len(edges) == 1
        assert edges[0]["target_id"] == "n2"
        assert edges[0]["weight"] == 0.8
        gs2.close()

    def test_corrupted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.graph"
        path.write_bytes(b"garbage data here")
        gs = GraphStore(path)
        gs.open()
        assert gs.edge_count == 0


# ── FiberStore unit tests ──


@pytest.fixture
def fiber_store(tmp_path: Path) -> FiberStore:
    fs = FiberStore(tmp_path / "test.fibers")
    fs.open()
    return fs


class TestFiberStoreBasic:
    def test_add_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("test-cluster")
        assert fid
        assert fiber_store.count == 1

    def test_get_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("memories", description="Important ones")
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert fiber["name"] == "memories"
        assert fiber["description"] == "Important ones"

    def test_get_fiber_returns_copy(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("test", neuron_ids=["n1"])
        f1 = fiber_store.get_fiber(fid)
        f2 = fiber_store.get_fiber(fid)
        assert f1 is not f2
        assert f1["neuron_ids"] is not f2["neuron_ids"]

    def test_get_fiber_not_found(self, fiber_store: FiberStore) -> None:
        assert fiber_store.get_fiber("nonexistent") is None

    def test_add_neuron_to_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("cluster1")
        assert fiber_store.add_neuron_to_fiber(fid, "n1")
        assert fiber_store.add_neuron_to_fiber(fid, "n2")
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert set(fiber["neuron_ids"]) == {"n1", "n2"}

    def test_add_neuron_idempotent(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("cluster1")
        fiber_store.add_neuron_to_fiber(fid, "n1")
        fiber_store.add_neuron_to_fiber(fid, "n1")
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert fiber["neuron_ids"] == ["n1"]

    def test_remove_neuron_from_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("cluster1", neuron_ids=["n1", "n2"])
        assert fiber_store.remove_neuron_from_fiber(fid, "n1")
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert fiber["neuron_ids"] == ["n2"]

    def test_remove_neuron_not_in_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("cluster1", neuron_ids=["n1"])
        assert not fiber_store.remove_neuron_from_fiber(fid, "n99")

    def test_get_fibers_for_neuron(self, fiber_store: FiberStore) -> None:
        f1 = fiber_store.add_fiber("a", neuron_ids=["n1"])
        f2 = fiber_store.add_fiber("b", neuron_ids=["n1", "n2"])
        fibers = fiber_store.get_fibers_for_neuron("n1")
        assert set(fibers) == {f1, f2}

    def test_delete_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("doomed", neuron_ids=["n1"])
        assert fiber_store.delete_fiber(fid)
        assert fiber_store.count == 0
        assert fiber_store.get_fibers_for_neuron("n1") == []

    def test_delete_fiber_not_found(self, fiber_store: FiberStore) -> None:
        assert not fiber_store.delete_fiber("nope")

    def test_remove_neuron_from_all(self, fiber_store: FiberStore) -> None:
        fiber_store.add_fiber("a", neuron_ids=["n1"])
        fiber_store.add_fiber("b", neuron_ids=["n1", "n2"])
        removed = fiber_store.remove_neuron_from_all("n1")
        assert removed == 2
        assert fiber_store.get_fibers_for_neuron("n1") == []

    def test_find_fibers(self, fiber_store: FiberStore) -> None:
        fiber_store.add_fiber("work memories", fiber_type="cluster")
        fiber_store.add_fiber("habits", fiber_type="habit")
        fiber_store.add_fiber("work decisions", fiber_type="cluster")

        results = fiber_store.find_fibers(name_contains="work")
        assert len(results) == 2

        results = fiber_store.find_fibers(fiber_type="habit")
        assert len(results) == 1
        assert results[0]["name"] == "habits"

    def test_update_fiber(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("old name")
        assert fiber_store.update_fiber(fid, {"name": "new name", "description": "updated"})
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert fiber["name"] == "new name"
        assert fiber["description"] == "updated"

    def test_update_fiber_rejects_neuron_ids(self, fiber_store: FiberStore) -> None:
        fid = fiber_store.add_fiber("test", neuron_ids=["n1"])
        fiber_store.update_fiber(fid, {"neuron_ids": ["evil"]})
        fiber = fiber_store.get_fiber(fid)
        assert fiber is not None
        assert fiber["neuron_ids"] == ["n1"]


class TestFiberStorePersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "test.fibers"

        fs1 = FiberStore(path)
        fs1.open()
        fid = fs1.add_fiber("cluster", neuron_ids=["n1", "n2"])
        fs1.close()

        fs2 = FiberStore(path)
        fs2.open()
        assert fs2.count == 1
        fiber = fs2.get_fiber(fid)
        assert fiber is not None
        assert set(fiber["neuron_ids"]) == {"n1", "n2"}
        fs2.close()

    def test_corrupted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.fibers"
        path.write_bytes(b"corrupted")
        fs = FiberStore(path)
        fs.open()
        assert fs.count == 0


# ── InfinityDB integration tests for graph + fiber (async) ──


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "brains"


class TestInfinityDBGraph:
    async def test_add_and_get_synapse(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("source", neuron_id="n1")
        await db.add_neuron("target", neuron_id="n2")
        await db.add_synapse("n1", "n2", edge_type="causes", weight=0.9)

        edges = await db.get_synapses("n1", direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["target_id"] == "n2"
        assert edges[0]["weight"] == 0.9

        await db.close()

    async def test_synapse_count_in_stats(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_synapse("n1", "n2")
        await db.add_synapse("n2", "n1")

        stats = await db.get_stats()
        assert stats["synapse_count"] == 2
        assert db.synapse_count == 2

        await db.close()

    async def test_delete_synapse(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        eid = await db.add_synapse("n1", "n2")
        assert db.synapse_count == 1

        assert await db.delete_synapse(eid)
        assert db.synapse_count == 0

        await db.close()

    async def test_update_synapse(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        eid = await db.add_synapse("n1", "n2", weight=1.0)

        assert await db.update_synapse(eid, {"weight": 0.3})
        edges = await db.get_synapses("n1")
        assert edges[0]["weight"] == 0.3

        await db.close()

    async def test_get_neighbors(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        for i in range(4):
            await db.add_neuron(f"node{i}", neuron_id=f"n{i}")
        await db.add_synapse("n0", "n1")
        await db.add_synapse("n0", "n2")
        await db.add_synapse("n3", "n0")

        nb = await db.get_neighbors("n0", direction="outgoing")
        assert set(nb) == {"n1", "n2"}

        nb = await db.get_neighbors("n0", direction="incoming")
        assert set(nb) == {"n3"}

        nb = await db.get_neighbors("n0", direction="both")
        assert set(nb) == {"n1", "n2", "n3"}

        await db.close()

    async def test_bfs_traversal(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        for i in range(5):
            await db.add_neuron(f"node{i}", neuron_id=f"n{i}")
        await db.add_synapse("n0", "n1")
        await db.add_synapse("n1", "n2")
        await db.add_synapse("n2", "n3")
        await db.add_synapse("n3", "n4")

        result = await db.bfs_traverse("n0", max_depth=2)
        ids = [nid for nid, _ in result]
        assert ids == ["n0", "n1", "n2"]

        await db.close()

    async def test_get_subgraph(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        for i in range(4):
            await db.add_neuron(f"node{i}", neuron_id=f"n{i}")
        await db.add_synapse("n0", "n1")
        await db.add_synapse("n1", "n2")
        await db.add_synapse("n2", "n3")

        edges = await db.get_subgraph(["n0", "n1", "n2"])
        assert len(edges) == 2

        await db.close()

    async def test_delete_neuron_cleans_edges(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_neuron("c", neuron_id="n3")
        await db.add_synapse("n1", "n2")
        await db.add_synapse("n3", "n1")

        await db.delete_neuron("n1")
        assert db.synapse_count == 0

        await db.close()

    async def test_graph_persistence(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()
        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_synapse("n1", "n2", edge_type="causes")
        await db.close()

        db2 = InfinityDB(db_dir, dimensions=8)
        await db2.open()
        assert db2.synapse_count == 1
        edges = await db2.get_synapses("n1")
        assert len(edges) == 1
        assert edges[0]["type"] == "causes"
        await db2.close()

    async def test_get_synapses_both(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()
        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_neuron("c", neuron_id="n3")
        await db.add_synapse("n1", "n2")
        await db.add_synapse("n3", "n1")

        edges = await db.get_synapses("n1", direction="both")
        assert len(edges) == 2

        await db.close()

    async def test_get_synapses_both_dedup_self_loop(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()
        await db.add_neuron("a", neuron_id="n1")
        await db.add_synapse("n1", "n1")  # self-loop

        edges = await db.get_synapses("n1", direction="both")
        assert len(edges) == 1  # no duplicate

        await db.close()

    async def test_add_synapse_validates_neurons(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()
        await db.add_neuron("a", neuron_id="n1")

        with pytest.raises(ValueError, match="Source neuron not found"):
            await db.add_synapse("nonexistent", "n1")

        with pytest.raises(ValueError, match="Target neuron not found"):
            await db.add_synapse("n1", "nonexistent")

        await db.close()


class TestInfinityDBFiber:
    async def test_add_and_get_fiber(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        fid = await db.add_fiber("work-memories", description="Work stuff")
        fiber = await db.get_fiber(fid)
        assert fiber is not None
        assert fiber["name"] == "work-memories"
        assert db.fiber_count == 1

        await db.close()

    async def test_add_neuron_to_fiber(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("mem1", neuron_id="n1")
        fid = await db.add_fiber("cluster")
        await db.add_neuron_to_fiber(fid, "n1")

        fiber = await db.get_fiber(fid)
        assert fiber is not None
        assert "n1" in fiber["neuron_ids"]

        fibers = await db.get_fibers_for_neuron("n1")
        assert fid in fibers

        await db.close()

    async def test_remove_neuron_from_fiber(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        fid = await db.add_fiber("cluster", neuron_ids=["n1", "n2"])
        await db.remove_neuron_from_fiber(fid, "n1")
        fiber = await db.get_fiber(fid)
        assert fiber is not None
        assert fiber["neuron_ids"] == ["n2"]

        await db.close()

    async def test_find_fibers(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_fiber("work memories")
        await db.add_fiber("habits", fiber_type="habit")
        await db.add_fiber("work decisions")

        results = await db.find_fibers(name_contains="work")
        assert len(results) == 2

        await db.close()

    async def test_delete_fiber(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        fid = await db.add_fiber("temp")
        assert await db.delete_fiber(fid)
        assert db.fiber_count == 0

        await db.close()

    async def test_delete_neuron_cleans_fiber(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_neuron("mem", neuron_id="n1")
        fid = await db.add_fiber("cluster", neuron_ids=["n1"])
        await db.delete_neuron("n1")

        fiber = await db.get_fiber(fid)
        assert fiber is not None
        assert fiber["neuron_ids"] == []

        await db.close()

    async def test_fiber_persistence(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()
        fid = await db.add_fiber("persistent", neuron_ids=["n1", "n2"])
        await db.close()

        db2 = InfinityDB(db_dir, dimensions=8)
        await db2.open()
        assert db2.fiber_count == 1
        fiber = await db2.get_fiber(fid)
        assert fiber is not None
        assert set(fiber["neuron_ids"]) == {"n1", "n2"}
        await db2.close()

    async def test_fiber_count_in_stats(self, db_dir: Path) -> None:
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        await db.add_fiber("a")
        await db.add_fiber("b")

        stats = await db.get_stats()
        assert stats["fiber_count"] == 2

        await db.close()


# ── Benchmark ──


class TestGraphBenchmark:
    async def test_10k_edges(self, db_dir: Path) -> None:
        """Benchmark: 10K edges insert + BFS."""
        db = InfinityDB(db_dir, dimensions=8)
        await db.open()

        # Create nodes
        for i in range(1000):
            await db.add_neuron(f"node-{i}", neuron_id=f"n{i}")

        # Create 10K edges (chain + random)
        t0 = time.perf_counter()
        for i in range(999):
            await db.add_synapse(f"n{i}", f"n{i + 1}")
        rng = np.random.default_rng(42)
        for _ in range(9001):
            a = int(rng.integers(0, 1000))
            b = int(rng.integers(0, 1000))
            if a != b:
                await db.add_synapse(f"n{a}", f"n{b}")
        t1 = time.perf_counter()

        assert db.synapse_count >= 9000

        # BFS from node 0
        t2 = time.perf_counter()
        result = await db.bfs_traverse("n0", max_depth=3, max_nodes=500)
        t3 = time.perf_counter()

        assert len(result) > 0

        print("\n--- Phase 2 Benchmark ---")
        print(f"10K edges insert: {t1 - t0:.2f}s")
        print(f"BFS depth=3 (max 500 nodes): {(t3 - t2) * 1000:.2f}ms, found {len(result)} nodes")
        print(f"Synapse count: {db.synapse_count}")

        await db.close()
