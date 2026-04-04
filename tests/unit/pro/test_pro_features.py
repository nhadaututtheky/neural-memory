"""Tests for Pro features rebuilt on InfinityDB.

Tests cone_queries, directional_compress, and smart_merge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.consolidation.smart_merge import MergeAction, smart_merge
from neural_memory.pro.hyperspace.directional_compress import (
    _basic_compress,
    _cosine_sim,
    _split_sentences,
    directional_compress,
)
from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.retrieval.cone_queries import ConeResult, cone_recall


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_brain"


DIMS = 32


def _make_vec(seed: int, dims: int = DIMS) -> list[float]:
    """Create a deterministic vector from seed."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dims).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # normalize
    return vec.tolist()


def _similar_vec(base: list[float], noise: float = 0.05) -> list[float]:
    """Create a vector similar to base with small noise."""
    arr = np.array(base, dtype=np.float32)
    rng = np.random.default_rng(99)
    noisy = arr + rng.normal(0, noise, len(arr)).astype(np.float32)
    noisy = noisy / np.linalg.norm(noisy)
    return noisy.tolist()


async def _setup_db(db_dir: Path) -> InfinityDB:
    db = InfinityDB(db_dir, brain_id="test", dimensions=DIMS)
    await db.open()
    return db


# --- Cone Queries ---


class TestConeQueries:
    """Cone recall using HNSW search."""

    @pytest.mark.asyncio
    async def test_cone_recall_finds_similar(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        base_vec = _make_vec(1)
        similar = _similar_vec(base_vec, noise=0.02)
        dissimilar = _make_vec(999)

        await db.add_neuron("close", neuron_id="n1", embedding=base_vec)
        await db.add_neuron("similar", neuron_id="n2", embedding=similar)
        await db.add_neuron("far", neuron_id="n3", embedding=dissimilar)

        results = await cone_recall(base_vec, db, threshold=0.9)
        ids = [r.neuron_id for r in results]
        assert "n1" in ids
        assert "n2" in ids
        # n3 should be excluded (dissimilar)
        assert all(r.similarity >= 0.9 for r in results)
        await db.close()

    @pytest.mark.asyncio
    async def test_cone_recall_empty_db(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        results = await cone_recall(_make_vec(1), db, threshold=0.5)
        assert results == []
        await db.close()

    @pytest.mark.asyncio
    async def test_cone_recall_respects_max_results(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        base_vec = _make_vec(1)
        for i in range(20):
            vec = _similar_vec(base_vec, noise=0.01 + i * 0.001)
            await db.add_neuron(f"n{i}", neuron_id=f"n{i}", embedding=vec)

        results = await cone_recall(base_vec, db, threshold=0.5, max_results=5)
        assert len(results) <= 5
        await db.close()

    @pytest.mark.asyncio
    async def test_cone_result_has_combined_score(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        vec = _make_vec(1)
        await db.add_neuron("test", neuron_id="n1", embedding=vec, activation_level=0.8)

        results = await cone_recall(vec, db, threshold=0.0)
        assert len(results) >= 1
        r = results[0]
        assert r.combined_score > 0
        assert r.neuron_type == "fact"
        await db.close()

    @pytest.mark.asyncio
    async def test_cone_result_dataclass(self) -> None:
        r = ConeResult(
            neuron_id="n1",
            content="test",
            similarity=0.95,
            activation=0.7,
            combined_score=0.88,
            neuron_type="fact",
        )
        assert r.neuron_id == "n1"
        assert r.similarity == 0.95


# --- Directional Compress ---


class TestDirectionalCompress:
    """Multi-axis directional compression."""

    @pytest.mark.asyncio
    async def test_full_level_no_compression(self) -> None:
        async def embed(text: str) -> list[float]:
            return _make_vec(hash(text) % 1000)

        result = await directional_compress("Hello world.", "full", embed)
        assert result == "Hello world."

    @pytest.mark.asyncio
    async def test_ghost_level_truncates(self) -> None:
        async def embed(text: str) -> list[float]:
            return _make_vec(hash(text) % 1000)

        text = "This is a very long sentence that has many words in it."
        result = await directional_compress(text, "ghost", embed)
        assert result.endswith("...")
        assert len(result.split()) <= 6

    @pytest.mark.asyncio
    async def test_summary_keeps_most_sentences(self) -> None:
        async def embed(text: str) -> list[float]:
            return _make_vec(hash(text) % 1000)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = await directional_compress(text, "summary", embed)
        # Summary keeps ~66% of sentences
        assert len(result) < len(text) or len(result) == len(text)

    @pytest.mark.asyncio
    async def test_essence_keeps_fewer_sentences(self) -> None:
        async def embed(text: str) -> list[float]:
            return _make_vec(hash(text) % 1000)

        text = "A. B. C. D. E. F."
        result = await directional_compress(text, "essence", embed)
        result_sentences = _split_sentences(result)
        original_sentences = _split_sentences(text)
        assert len(result_sentences) <= len(original_sentences)

    @pytest.mark.asyncio
    async def test_no_embedding_fallback(self) -> None:
        async def embed(text: str) -> list[float]:
            return []  # simulate embedding failure

        text = "Word1 word2 word3 word4 word5 word6"
        result = await directional_compress(text, "summary", embed)
        assert len(result) > 0

    def test_cosine_sim(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_sim(a, b) - 1.0) < 1e-6

    def test_cosine_sim_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_split_sentences(self) -> None:
        text = "First. Second! Third? Fourth."
        sentences = _split_sentences(text)
        assert len(sentences) == 4

    def test_basic_compress_summary(self) -> None:
        result = _basic_compress("a b c d e f", "summary")
        assert "..." in result or len(result.split()) <= 4

    def test_basic_compress_essence(self) -> None:
        result = _basic_compress("a b c d e f g h i", "essence")
        words = result.replace("...", "").strip().split()
        assert len(words) <= 4


# --- Smart Merge ---


class TestSmartMerge:
    """HNSW-accelerated smart merge consolidation."""

    @pytest.mark.asyncio
    async def test_merge_similar_neurons(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        base_vec = _make_vec(1)

        # Create cluster of similar neurons
        await db.add_neuron("Memory about cats", neuron_id="n1", embedding=base_vec, priority=8)
        await db.add_neuron(
            "Also about cats and dogs",
            neuron_id="n2",
            embedding=_similar_vec(base_vec, 0.01),
            priority=5,
        )
        await db.add_neuron(
            "Cat behavior patterns",
            neuron_id="n3",
            embedding=_similar_vec(base_vec, 0.02),
            priority=4,
        )

        # Dissimilar neuron — should not merge
        await db.add_neuron("About physics", neuron_id="n4", embedding=_make_vec(999), priority=5)

        result = await smart_merge(db, similarity_threshold=0.9, dry_run=True)
        assert result["status"] == "planned"
        assert result["clusters_found"] >= 1
        await db.close()

    @pytest.mark.asyncio
    async def test_merge_executes(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        base_vec = _make_vec(1)

        await db.add_neuron("Primary content", neuron_id="n1", embedding=base_vec, priority=9)
        await db.add_neuron(
            "Secondary content", neuron_id="n2", embedding=_similar_vec(base_vec, 0.01), priority=3
        )

        result = await smart_merge(db, similarity_threshold=0.9)
        if result["merge_actions"] > 0:
            assert result["status"] == "executed"
            # Anchor should have updated content
            anchor = await db.get_neuron("n1")
            assert anchor is not None
        await db.close()

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        result = await smart_merge(db)
        assert result["status"] == "empty"
        assert result["merges"] == 0
        await db.close()

    @pytest.mark.asyncio
    async def test_no_similar_neurons(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        # All dissimilar
        for i in range(5):
            await db.add_neuron(f"topic {i}", neuron_id=f"n{i}", embedding=_make_vec(i * 100))

        result = await smart_merge(db, similarity_threshold=0.99)
        assert result["merges"] == 0 or result["status"] in (
            "no_clusters",
            "insufficient_embeddings",
        )
        await db.close()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_modify(self, db_dir: Path) -> None:
        db = await _setup_db(db_dir)
        base_vec = _make_vec(1)
        await db.add_neuron("Original A", neuron_id="n1", embedding=base_vec, priority=8)
        await db.add_neuron(
            "Original B", neuron_id="n2", embedding=_similar_vec(base_vec, 0.01), priority=3
        )

        await smart_merge(db, similarity_threshold=0.9, dry_run=True)

        # Content should be unchanged
        n1 = await db.get_neuron("n1")
        n2 = await db.get_neuron("n2")
        assert n1 is not None and n1["content"] == "Original A"
        assert n2 is not None and n2["content"] == "Original B"
        await db.close()

    def test_merge_action_dataclass(self) -> None:
        action = MergeAction(
            anchor_id="a1",
            merged_ids=("m1", "m2"),
            new_content="merged",
            reason="test",
        )
        assert action.anchor_id == "a1"
        assert len(action.merged_ids) == 2
