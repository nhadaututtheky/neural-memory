"""Tests for InfinityDB core engine — Phase 1."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.file_format import BrainPaths, InfinityHeader
from neural_memory.pro.infinitydb.hnsw_index import HNSWIndex
from neural_memory.pro.infinitydb.metadata_store import MetadataStore
from neural_memory.pro.infinitydb.vector_store import VectorStore

# ============================================================
# File Format Tests
# ============================================================


class TestInfinityHeader:
    def test_roundtrip(self) -> None:
        header = InfinityHeader(version=1, dimensions=384, neuron_count=1000)
        data = header.to_bytes()
        restored = InfinityHeader.from_bytes(data)
        assert restored.version == 1
        assert restored.dimensions == 384
        assert restored.neuron_count == 1000

    def test_invalid_magic(self) -> None:
        with pytest.raises(ValueError, match="Invalid magic"):
            InfinityHeader.from_bytes(b"\x00" * 50)

    def test_short_data(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            InfinityHeader.from_bytes(b"INF")

    def test_brain_paths(self, tmp_path: Path) -> None:
        paths = BrainPaths(tmp_path, "test-brain")
        assert paths.brain_dir == tmp_path / "test-brain"
        assert paths.header.suffix == ".inf"
        assert paths.vectors.suffix == ".vec"
        paths.ensure_dirs()
        assert paths.brain_dir.exists()


# ============================================================
# Vector Store Tests
# ============================================================


class TestVectorStore:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        slot = store.add(vec)
        assert slot == 0
        result = store.get(slot)
        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)
        store.close()

    def test_multiple_adds(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        for i in range(10):
            vec = np.full(4, float(i), dtype=np.float32)
            slot = store.add(vec)
            assert slot == i
        assert store.count == 10
        store.close()

    def test_delete_and_reuse(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        slot0 = store.add(np.ones(4, dtype=np.float32))
        store.add(np.ones(4, dtype=np.float32) * 2)
        store.delete(slot0)
        assert store.get(slot0) is None
        assert store.count == 1
        # Reuse freed slot
        slot2 = store.add(np.ones(4, dtype=np.float32) * 3)
        assert slot2 == slot0  # Reused!
        assert store.count == 2
        store.close()

    def test_auto_grow(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        # Default initial capacity is 1024, add more
        for i in range(1100):
            store.add(np.full(4, float(i), dtype=np.float32))
        assert store.count == 1100
        # Verify last element
        result = store.get(1099)
        assert result is not None
        np.testing.assert_array_almost_equal(result, np.full(4, 1099.0))
        store.close()

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "persist.vec"
        # Write
        store = VectorStore(path, dimensions=4)
        store.open()
        store.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        store.flush()
        store.close()
        # Read back
        store2 = VectorStore(path, dimensions=4)
        store2.open()
        result = store2.get(0)
        assert result is not None
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])
        store2.close()

    def test_wrong_dimensions(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        with pytest.raises(ValueError, match="Expected shape"):
            store.add(np.ones(8, dtype=np.float32))
        store.close()

    def test_get_all_vectors(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "test.vec", dimensions=4)
        store.open()
        store.add(np.ones(4, dtype=np.float32))
        store.add(np.ones(4, dtype=np.float32) * 2)
        store.add(np.ones(4, dtype=np.float32) * 3)
        store.delete(1)  # Delete middle
        slots, vecs = store.get_all_vectors()
        assert len(slots) == 2
        assert 1 not in slots
        store.close()


# ============================================================
# Metadata Store Tests
# ============================================================


class TestMetadataStore:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "type": "fact", "content": "hello"})
        result = store.get_by_id("n1")
        assert result is not None
        slot, meta = result
        assert slot == 0
        assert meta["content"] == "hello"
        store.close()

    def test_duplicate_id(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "content": "a"})
        with pytest.raises(ValueError, match="already exists"):
            store.add(1, {"id": "n1", "content": "b"})
        store.close()

    def test_update(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "content": "old"})
        store.update(0, {"content": "new"})
        result = store.get_by_id("n1")
        assert result is not None
        _, meta = result
        assert meta["content"] == "new"
        store.close()

    def test_delete(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "content": "test"})
        store.delete(0)
        assert store.get_by_id("n1") is None
        assert store.count == 0
        store.close()

    def test_find_by_type(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "type": "fact", "content": "a", "created_at": "2026-01-01"})
        store.add(1, {"id": "n2", "type": "insight", "content": "b", "created_at": "2026-01-02"})
        store.add(2, {"id": "n3", "type": "fact", "content": "c", "created_at": "2026-01-03"})
        results = store.find(neuron_type="fact")
        assert len(results) == 2
        store.close()

    def test_find_by_content(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "content": "Hello World", "created_at": "2026-01-01"})
        store.add(1, {"id": "n2", "content": "Goodbye", "created_at": "2026-01-02"})
        results = store.find(content_contains="hello")
        assert len(results) == 1
        assert results[0][1]["id"] == "n1"
        store.close()

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "persist.meta"
        store = MetadataStore(path)
        store.open()
        store.add(0, {"id": "n1", "content": "persisted"})
        store.flush()
        store.close()
        # Reload
        store2 = MetadataStore(path)
        store2.open()
        result = store2.get_by_id("n1")
        assert result is not None
        assert result[1]["content"] == "persisted"
        store2.close()

    def test_suggest(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.meta")
        store.open()
        store.add(0, {"id": "n1", "type": "fact", "content": "Python is great"})
        store.add(1, {"id": "n2", "type": "fact", "content": "Python 3.11 features"})
        store.add(2, {"id": "n3", "type": "fact", "content": "JavaScript is ok"})
        results = store.suggest("Python")
        assert len(results) == 2
        store.close()


# ============================================================
# HNSW Index Tests
# ============================================================


class TestHNSWIndex:
    def test_add_and_search(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=100)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        index.add(0, vec)
        slots, dists = index.search(vec, k=1)
        assert slots == [0]
        assert dists[0] < 0.01  # Near-zero distance for same vector
        index.close()

    def test_search_nearest(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=100)
        # Add 3 orthogonal vectors
        index.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        index.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        index.add(2, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        # Search for something close to [1,0,0,0]
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        slots, _ = index.search(query, k=1)
        assert slots[0] == 0  # Closest to first vector
        index.close()

    def test_delete(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=100)
        index.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        index.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        index.delete(0)
        # Search should not return deleted element
        slots, _ = index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert slots[0] == 1  # Only remaining element
        index.close()

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "persist.idx"
        index = HNSWIndex(path, dimensions=4)
        index.open(max_elements=100)
        index.add(0, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        index.close()
        # Reload
        index2 = HNSWIndex(path, dimensions=4)
        index2.open(max_elements=100)
        assert index2.count == 1
        slots, _ = index2.search(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), k=1)
        assert slots[0] == 0
        index2.close()

    def test_auto_resize(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=10)
        # Add more than initial capacity
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            index.add(i, vec)
        assert index.count == 50
        index.close()

    def test_batch_add(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=1000)
        vecs = np.random.randn(100, 4).astype(np.float32)
        slots = list(range(100))
        index.add_batch(slots, vecs)
        assert index.count == 100
        index.close()

    def test_empty_search(self, tmp_path: Path) -> None:
        index = HNSWIndex(tmp_path / "test.idx", dimensions=4)
        index.open(max_elements=10)
        slots, dists = index.search(np.ones(4, dtype=np.float32), k=5)
        assert slots == []
        assert dists == []
        index.close()


# ============================================================
# InfinityDB Engine Tests
# ============================================================


class TestInfinityDB:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "infinity_test"

    async def test_open_close(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        assert db.is_open
        assert db.neuron_count == 0
        await db.close()
        assert not db.is_open

    async def test_add_neuron_no_vector(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        nid = await db.add_neuron("hello world", neuron_type="fact")
        assert nid is not None
        assert db.neuron_count == 1
        neuron = await db.get_neuron(nid)
        assert neuron is not None
        assert neuron["content"] == "hello world"
        assert neuron["type"] == "fact"
        await db.close()

    async def test_add_neuron_with_vector(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        vec = [1.0, 0.0, 0.0, 0.0]
        nid = await db.add_neuron("test", embedding=vec)
        assert db.neuron_count == 1
        # Search should find it
        results = await db.search_similar(vec, k=1)
        assert len(results) == 1
        assert results[0]["id"] == nid
        assert results[0]["similarity"] > 0.99
        await db.close()

    async def test_find_neurons(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        await db.add_neuron("fact one", neuron_type="fact")
        await db.add_neuron("insight one", neuron_type="insight")
        await db.add_neuron("fact two", neuron_type="fact")
        results = await db.find_neurons(neuron_type="fact")
        assert len(results) == 2
        results = await db.find_neurons(content_contains="one")
        assert len(results) == 2
        await db.close()

    async def test_update_neuron(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        nid = await db.add_neuron("old content")
        await db.update_neuron(nid, content="new content", priority=9)
        neuron = await db.get_neuron(nid)
        assert neuron is not None
        assert neuron["content"] == "new content"
        assert neuron["priority"] == 9
        await db.close()

    async def test_delete_neuron(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        nid = await db.add_neuron("to delete", embedding=[1.0, 0.0, 0.0, 0.0])
        assert db.neuron_count == 1
        deleted = await db.delete_neuron(nid)
        assert deleted
        assert db.neuron_count == 0
        assert await db.get_neuron(nid) is None
        await db.close()

    async def test_search_similar(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        # Add 3 neurons with different vectors
        await db.add_neuron("north", embedding=[1.0, 0.0, 0.0, 0.0])
        await db.add_neuron("east", embedding=[0.0, 1.0, 0.0, 0.0])
        await db.add_neuron("up", embedding=[0.0, 0.0, 1.0, 0.0])
        # Search close to "north"
        results = await db.search_similar([0.9, 0.1, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0]["content"] == "north"  # Most similar
        await db.close()

    async def test_persistence(self, db_path: Path) -> None:
        # Write
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        nid = await db.add_neuron("persist me", embedding=[1.0, 2.0, 3.0, 4.0])
        await db.close()
        # Read back
        db2 = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db2.open()
        assert db2.neuron_count == 1
        neuron = await db2.get_neuron(nid)
        assert neuron is not None
        assert neuron["content"] == "persist me"
        await db2.close()

    async def test_stats(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        await db.add_neuron("a", embedding=[1.0, 0.0, 0.0, 0.0])
        await db.add_neuron("b")  # No vector
        stats = await db.get_stats()
        assert stats["neuron_count"] == 2
        assert stats["vector_count"] == 1
        assert stats["dimensions"] == 4
        await db.close()

    async def test_suggest(self, db_path: Path) -> None:
        db = InfinityDB(db_path, brain_id="test", dimensions=4)
        await db.open()
        await db.add_neuron("Python rocks")
        await db.add_neuron("Python 3.11")
        await db.add_neuron("JavaScript")
        results = await db.suggest_neurons("Pyth")
        assert len(results) == 2
        await db.close()


# ============================================================
# Quick Benchmark (100K neurons)
# ============================================================


class TestBenchmark:
    """Performance benchmarks. Run with: pytest -k bench -s --timeout=600"""

    @pytest.mark.timeout(600)
    async def test_100k_insert_and_search(self, tmp_path: Path) -> None:
        """Insert 100K neurons with 384D vectors, search top-100."""
        dims = 384
        n = 100_000
        db = InfinityDB(tmp_path / "bench", brain_id="bench", dimensions=dims)
        await db.open()

        # Generate random normalized vectors
        vectors = np.random.randn(n, dims).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        vectors = vectors / norms

        # Batch insert using fast sync path
        neurons = [
            {"neuron_id": f"n_{i}", "content": f"neuron_{i}", "embedding": vectors[i]}
            for i in range(n)
        ]

        t0 = time.perf_counter()
        ids = await db.add_neurons_batch(neurons)
        insert_time = time.perf_counter() - t0
        print(f"\n[BENCH] Batch insert {n:,} neurons: {insert_time:.2f}s ({n / insert_time:.0f}/s)")

        assert len(ids) == n

        # Search
        query = np.random.randn(dims).astype(np.float32)
        query = query / np.linalg.norm(query)

        t0 = time.perf_counter()
        results = await db.search_similar(query, k=100)
        search_time = (time.perf_counter() - t0) * 1000
        print(f"[BENCH] Search top-100 in {n:,}: {search_time:.2f}ms")

        assert len(results) == 100
        assert search_time < 100  # Target: <100ms

        stats = await db.get_stats()
        print(f"[BENCH] Stats: {stats}")

        await db.close()
