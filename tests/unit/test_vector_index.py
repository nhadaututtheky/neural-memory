"""Tests for SQLiteVectorIndex — HNSW sidecar for SQLite brains."""

from __future__ import annotations

import pytest

from neural_memory.engine.embedding.vector_index import SQLiteVectorIndex, is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="hnswlib not installed")


@pytest.fixture
def index(tmp_path):
    """Create a fresh vector index."""
    idx = SQLiteVectorIndex(base_path=tmp_path / "test_brain", dimensions=4)
    idx.open()
    yield idx
    idx.close()


class TestBasicOperations:
    def test_add_and_search(self, index: SQLiteVectorIndex):
        index.add("n1", [1.0, 0.0, 0.0, 0.0])
        index.add("n2", [0.0, 1.0, 0.0, 0.0])
        index.add("n3", [0.9, 0.1, 0.0, 0.0])

        results = index.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        # n1 should be most similar to query
        assert results[0][0] == "n1"
        assert results[0][1] > 0.9

    def test_search_empty_index(self, index: SQLiteVectorIndex):
        results = index.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_count(self, index: SQLiteVectorIndex):
        assert index.count == 0
        index.add("n1", [1.0, 0.0, 0.0, 0.0])
        assert index.count == 1
        index.add("n2", [0.0, 1.0, 0.0, 0.0])
        assert index.count == 2

    def test_remove(self, index: SQLiteVectorIndex):
        index.add("n1", [1.0, 0.0, 0.0, 0.0])
        index.add("n2", [0.0, 1.0, 0.0, 0.0])
        assert index.count == 2

        index.remove("n1")
        assert index.count == 1

        results = index.search([1.0, 0.0, 0.0, 0.0], k=5)
        # n1 should not appear
        ids = [r[0] for r in results]
        assert "n1" not in ids

    def test_remove_nonexistent(self, index: SQLiteVectorIndex):
        # Should not raise
        index.remove("nonexistent")

    def test_add_updates_existing(self, index: SQLiteVectorIndex):
        index.add("n1", [1.0, 0.0, 0.0, 0.0])
        index.add("n1", [0.0, 1.0, 0.0, 0.0])  # Update vector
        assert index.count == 1

        results = index.search([0.0, 1.0, 0.0, 0.0], k=1)
        assert results[0][0] == "n1"
        assert results[0][1] > 0.9

    def test_dim_mismatch_skipped(self, index: SQLiteVectorIndex):
        index.add("n1", [1.0, 0.0])  # Wrong dim
        assert index.count == 0


class TestBatchOperations:
    def test_add_batch(self, index: SQLiteVectorIndex):
        ids = [f"n{i}" for i in range(10)]
        vectors = [[float(i == j) for j in range(4)] for i in range(4)]
        # Pad to 10
        vectors.extend([[0.25, 0.25, 0.25, 0.25]] * 6)

        index.add_batch(ids, vectors)
        assert index.count == 10

        results = index.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0][0] == "n0"


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        base = tmp_path / "brain"

        # Create and populate
        idx1 = SQLiteVectorIndex(base_path=base, dimensions=4)
        idx1.open()
        idx1.add("n1", [1.0, 0.0, 0.0, 0.0])
        idx1.add("n2", [0.0, 1.0, 0.0, 0.0])
        idx1.close()

        # Reopen
        idx2 = SQLiteVectorIndex(base_path=base, dimensions=4)
        idx2.open()
        assert idx2.count == 2

        results = idx2.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0][0] == "n1"
        idx2.close()

    def test_mapping_round_trip(self, tmp_path):
        base = tmp_path / "brain"

        idx = SQLiteVectorIndex(base_path=base, dimensions=4)
        idx.open()
        for i in range(50):
            idx.add(f"neuron-{i}", [float(i % 4 == j) for j in range(4)])
        idx.close()

        idx2 = SQLiteVectorIndex(base_path=base, dimensions=4)
        idx2.open()
        assert idx2.count == 50
        # Search should return valid neuron IDs
        results = idx2.search([1.0, 0.0, 0.0, 0.0], k=5)
        for nid, _sim in results:
            assert nid.startswith("neuron-")
        idx2.close()


class TestRebuild:
    def test_rebuild_from_dict(self, tmp_path):
        base = tmp_path / "brain"
        idx = SQLiteVectorIndex(base_path=base, dimensions=4)

        neuron_vectors = {
            "n1": [1.0, 0.0, 0.0, 0.0],
            "n2": [0.0, 1.0, 0.0, 0.0],
            "n3": [0.0, 0.0, 1.0, 0.0],
        }
        idx.rebuild(neuron_vectors)
        assert idx.count == 3

        results = idx.search([0.0, 1.0, 0.0, 0.0], k=1)
        assert results[0][0] == "n2"
        idx.close()
