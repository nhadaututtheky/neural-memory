"""Tests for WAL integration into InfinityDB engine.

Verifies that:
1. WAL is opened/closed with engine lifecycle
2. Write operations log to WAL before applying
3. Crash recovery replays WAL entries correctly
4. Flush checkpoints the WAL
5. Replay is idempotent (no duplicates)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.wal import WALOp


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_brain"


@pytest.fixture
def dims() -> int:
    return 32


async def _make_db(db_dir: Path, dims: int = 32) -> InfinityDB:
    db = InfinityDB(db_dir, brain_id="test", dimensions=dims)
    await db.open()
    return db


def _random_vec(dims: int = 32) -> list[float]:
    return np.random.default_rng(42).random(dims).astype(np.float32).tolist()


# --- Lifecycle Tests ---


class TestWALLifecycle:
    """WAL opens and closes with engine."""

    @pytest.mark.asyncio
    async def test_wal_opens_with_engine(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        assert db._wal.is_open
        wal_path = db._paths.wal
        assert wal_path.exists()
        await db.close()

    @pytest.mark.asyncio
    async def test_wal_closes_with_engine(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.close()
        assert not db._wal.is_open

    @pytest.mark.asyncio
    async def test_wal_checkpointed_on_flush(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("test", neuron_id="n1")
        assert db._wal.entry_count > 0
        await db.flush()
        # After flush, WAL is checkpointed (truncated)
        assert db._wal.entry_count == 0
        await db.close()


# --- WAL Logging Tests ---


class TestWALLogging:
    """Write operations log to WAL before applying."""

    @pytest.mark.asyncio
    async def test_add_neuron_logs_to_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("hello world", neuron_id="n1", neuron_type="fact")

        entries = db._wal.get_pending_entries()
        assert len(entries) == 1
        assert entries[0].op == WALOp.ADD_NEURON
        assert entries[0].payload["id"] == "n1"
        assert entries[0].payload["content"] == "hello world"
        await db.close()

    @pytest.mark.asyncio
    async def test_add_neuron_with_embedding_logs_to_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        vec = _random_vec(dims)
        await db.add_neuron("embedded", neuron_id="n1", embedding=vec)

        entries = db._wal.get_pending_entries()
        assert len(entries) == 1
        assert "embedding" in entries[0].payload
        assert len(entries[0].payload["embedding"]) == dims
        await db.close()

    @pytest.mark.asyncio
    async def test_delete_neuron_logs_to_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("to delete", neuron_id="n1")
        await db.flush()  # checkpoint add

        await db.delete_neuron("n1")
        entries = db._wal.get_pending_entries()
        assert len(entries) == 1
        assert entries[0].op == WALOp.DELETE_NEURON
        assert entries[0].payload["id"] == "n1"
        await db.close()

    @pytest.mark.asyncio
    async def test_update_neuron_logs_to_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("original", neuron_id="n1")
        await db.flush()

        await db.update_neuron("n1", content="updated")
        entries = db._wal.get_pending_entries()
        assert len(entries) == 1
        assert entries[0].op == WALOp.UPDATE_NEURON
        assert entries[0].payload["id"] == "n1"
        assert entries[0].payload["updates"]["content"] == "updated"
        await db.close()

    @pytest.mark.asyncio
    async def test_add_synapse_logs_to_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("source", neuron_id="n1")
        await db.add_neuron("target", neuron_id="n2")
        await db.flush()

        await db.add_synapse("n1", "n2", edge_type="related", edge_id="e1")
        entries = db._wal.get_pending_entries()
        assert len(entries) == 1
        assert entries[0].op == WALOp.ADD_SYNAPSE
        assert entries[0].payload["source_id"] == "n1"
        assert entries[0].payload["target_id"] == "n2"
        assert entries[0].payload["edge_id"] == "e1"
        await db.close()

    @pytest.mark.asyncio
    async def test_multiple_ops_accumulate_in_wal(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_synapse("n1", "n2")

        entries = db._wal.get_pending_entries()
        assert len(entries) == 3
        assert entries[0].op == WALOp.ADD_NEURON
        assert entries[1].op == WALOp.ADD_NEURON
        assert entries[2].op == WALOp.ADD_SYNAPSE
        await db.close()


# --- Crash Recovery Tests ---


class TestCrashRecovery:
    """WAL replay recovers data after simulated crash."""

    @pytest.mark.asyncio
    async def test_recover_add_neuron(self, db_dir: Path, dims: int) -> None:
        """Simulate crash after WAL write but before flush."""
        db = await _make_db(db_dir, dims)
        await db.add_neuron("crash test", neuron_id="n1", neuron_type="insight")

        # Simulate crash: close WAL without checkpoint, then corrupt metadata
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        # Delete metadata to simulate partial write
        if db._paths.meta.exists():
            db._paths.meta.unlink()

        # Reopen — WAL replay should recover the neuron
        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        neuron = await db2.get_neuron("n1")
        assert neuron is not None
        assert neuron["content"] == "crash test"
        assert neuron["type"] == "insight"
        await db2.close()

    @pytest.mark.asyncio
    async def test_recover_add_neuron_with_embedding(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        vec = _random_vec(dims)
        await db.add_neuron("vec crash", neuron_id="n1", embedding=vec)

        # Simulate crash
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        if db._paths.meta.exists():
            db._paths.meta.unlink()

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        neuron = await db2.get_neuron("n1")
        assert neuron is not None
        assert neuron["vec_slot"] >= 0

        # Verify vector is searchable
        results = await db2.search_similar(vec, k=1)
        assert len(results) >= 1
        assert results[0]["id"] == "n1"
        await db2.close()

    @pytest.mark.asyncio
    async def test_recover_delete_neuron(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("will delete", neuron_id="n1")
        await db.flush()  # persist the add

        await db.delete_neuron("n1")

        # Simulate crash before flush
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        neuron = await db2.get_neuron("n1")
        assert neuron is None  # should be deleted by replay
        await db2.close()

    @pytest.mark.asyncio
    async def test_recover_update_neuron(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("original", neuron_id="n1")
        await db.flush()

        await db.update_neuron("n1", content="updated")

        # Simulate crash
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        neuron = await db2.get_neuron("n1")
        assert neuron is not None
        assert neuron["content"] == "updated"
        await db2.close()

    @pytest.mark.asyncio
    async def test_recover_add_synapse(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("src", neuron_id="n1")
        await db.add_neuron("tgt", neuron_id="n2")
        await db.flush()

        await db.add_synapse("n1", "n2", edge_type="caused", edge_id="e1")

        # Simulate crash
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        synapses = await db2.get_synapses("n1", direction="outgoing")
        assert len(synapses) == 1
        assert synapses[0]["target_id"] == "n2"
        assert synapses[0]["type"] == "caused"
        await db2.close()


# --- Idempotency Tests ---


class TestReplayIdempotency:
    """WAL replay must be idempotent — no duplicates on double-replay."""

    @pytest.mark.asyncio
    async def test_replay_add_neuron_idempotent(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("idempotent", neuron_id="n1")

        # Force replay — entries are processed but skipped (idempotent)
        db._replay_wal()

        # Key assertion: no duplicate neurons created
        assert db.neuron_count == 1
        await db.close()

    @pytest.mark.asyncio
    async def test_replay_add_synapse_idempotent(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("a", neuron_id="n1")
        await db.add_neuron("b", neuron_id="n2")
        await db.add_synapse("n1", "n2", edge_id="e1")

        # Force replay
        db._replay_wal()
        # Synapse should be skipped (already exists)
        assert db.synapse_count == 1
        await db.close()

    @pytest.mark.asyncio
    async def test_replay_delete_idempotent(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("temp", neuron_id="n1")
        await db.flush()
        await db.delete_neuron("n1")

        # Force replay — delete on missing neuron should be no-op
        db._replay_wal()
        assert db.neuron_count == 0
        await db.close()

    @pytest.mark.asyncio
    async def test_replay_update_missing_neuron_skips(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("temp", neuron_id="n1")
        await db.flush()
        await db.update_neuron("n1", content="new")
        await db.delete_neuron("n1")

        # WAL has update + delete for n1. Replay: update skips (deleted), delete skips
        db._replay_wal()
        assert db.neuron_count == 0
        await db.close()


# --- Edge Cases ---


class TestWALEdgeCases:
    """Edge cases for WAL integration."""

    @pytest.mark.asyncio
    async def test_empty_wal_on_clean_open(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        entries = db._wal.get_pending_entries()
        assert len(entries) == 0
        await db.close()

    @pytest.mark.asyncio
    async def test_wal_survives_open_close_cycle(self, db_dir: Path, dims: int) -> None:
        db = await _make_db(db_dir, dims)
        await db.add_neuron("persist", neuron_id="n1")
        await db.close()  # close flushes + checkpoints

        db2 = await _make_db(db_dir, dims)
        neuron = await db2.get_neuron("n1")
        assert neuron is not None
        assert neuron["content"] == "persist"
        await db2.close()

    @pytest.mark.asyncio
    async def test_corrupt_wal_entry_stops_replay(self, db_dir: Path, dims: int) -> None:
        """If WAL has corrupt entry, replay stops gracefully and prior data survives."""
        db = await _make_db(db_dir, dims)
        await db.add_neuron("good", neuron_id="n1")
        await db.close()

        # Append garbage to WAL file
        wal_path = db._paths.wal
        with open(wal_path, "ab") as f:
            f.write(struct.pack("<I", 100))  # length prefix
            f.write(b"garbage" * 15)  # invalid msgpack

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()  # should not crash

        # Pre-crash data must survive
        neuron = await db2.get_neuron("n1")
        assert neuron is not None
        assert neuron["content"] == "good"
        await db2.close()

    @pytest.mark.asyncio
    async def test_many_operations_then_recover(self, db_dir: Path, dims: int) -> None:
        """Stress: many ops, crash, recover all."""
        db = await _make_db(db_dir, dims)
        for i in range(50):
            await db.add_neuron(f"neuron {i}", neuron_id=f"n{i}")

        # Add some synapses
        for i in range(0, 48, 2):
            await db.add_synapse(f"n{i}", f"n{i + 1}", edge_id=f"e{i}")

        # Simulate crash
        db._wal.close()
        db._metadata.close()
        db._vectors.close()
        db._index.close()
        db._graph.close()
        db._fibers.close()
        db._is_open = False

        if db._paths.meta.exists():
            db._paths.meta.unlink()

        db2 = InfinityDB(db_dir, brain_id="test", dimensions=dims)
        await db2.open()

        assert db2.neuron_count == 50
        assert db2.synapse_count == 24
        await db2.close()
