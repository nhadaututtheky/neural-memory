"""Tests for InfinityDB Phase 4 — WAL + Crash Safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.pro.infinitydb.wal import (
    WALEntry,
    WALOp,
    WriteAheadLog,
)


@pytest.fixture
def wal_path(tmp_path: Path) -> Path:
    return tmp_path / "test.wal"


@pytest.fixture
def wal(wal_path: Path) -> WriteAheadLog:
    w = WriteAheadLog(wal_path)
    w.open()
    return w


class TestWALBasic:
    def test_open_creates_file(self, wal_path: Path) -> None:
        w = WriteAheadLog(wal_path)
        w.open()
        assert wal_path.exists()
        assert wal_path.stat().st_size >= 5  # magic header
        w.close()

    def test_append_entry(self, wal: WriteAheadLog) -> None:
        seq = wal.append(WALOp.ADD_NEURON, {"id": "n1", "content": "hello"})
        assert seq == 1
        assert wal.entry_count == 1

    def test_append_multiple(self, wal: WriteAheadLog) -> None:
        wal.append(WALOp.ADD_NEURON, {"id": "n1"})
        wal.append(WALOp.ADD_SYNAPSE, {"source": "n1", "target": "n2"})
        wal.append(WALOp.DELETE_NEURON, {"id": "n1"})
        assert wal.entry_count == 3

    def test_sequence_increments(self, wal: WriteAheadLog) -> None:
        s1 = wal.append(WALOp.ADD_NEURON, {"id": "n1"})
        s2 = wal.append(WALOp.ADD_NEURON, {"id": "n2"})
        assert s2 == s1 + 1


class TestWALPersistence:
    def test_reopen_preserves_entries(self, wal_path: Path) -> None:
        w1 = WriteAheadLog(wal_path)
        w1.open()
        w1.append(WALOp.ADD_NEURON, {"id": "n1", "content": "test"})
        w1.append(WALOp.ADD_SYNAPSE, {"source": "n1", "target": "n2"})
        w1.close()

        w2 = WriteAheadLog(wal_path)
        w2.open()
        assert w2.entry_count == 2
        entries = w2.get_pending_entries()
        assert len(entries) == 2
        assert entries[0].op == WALOp.ADD_NEURON
        assert entries[0].payload["id"] == "n1"
        assert entries[1].op == WALOp.ADD_SYNAPSE
        w2.close()

    def test_sequence_continues_after_reopen(self, wal_path: Path) -> None:
        w1 = WriteAheadLog(wal_path)
        w1.open()
        w1.append(WALOp.ADD_NEURON, {"id": "n1"})
        w1.append(WALOp.ADD_NEURON, {"id": "n2"})
        w1.close()

        w2 = WriteAheadLog(wal_path)
        w2.open()
        s3 = w2.append(WALOp.ADD_NEURON, {"id": "n3"})
        assert s3 == 3
        w2.close()


class TestWALCheckpoint:
    def test_checkpoint_truncates(self, wal: WriteAheadLog) -> None:
        wal.append(WALOp.ADD_NEURON, {"id": "n1"})
        wal.append(WALOp.ADD_NEURON, {"id": "n2"})
        assert wal.entry_count == 2

        wal.checkpoint()
        assert wal.entry_count == 0
        assert wal.get_pending_entries() == []

    def test_checkpoint_allows_new_entries(self, wal: WriteAheadLog) -> None:
        wal.append(WALOp.ADD_NEURON, {"id": "n1"})
        wal.checkpoint()

        seq = wal.append(WALOp.ADD_NEURON, {"id": "n2"})
        assert seq == 1  # resets after checkpoint
        assert wal.entry_count == 1

    def test_needs_checkpoint(self, wal: WriteAheadLog, wal_path: Path) -> None:
        assert not wal.needs_checkpoint()
        # Write enough data to exceed threshold (normally 50MB, we just check the flag)


class TestWALEntry:
    def test_serialize_deserialize(self) -> None:
        entry = WALEntry(
            seq=42,
            op=WALOp.ADD_NEURON,
            timestamp="2026-03-23T12:00:00",
            payload={"id": "n1", "content": "hello world"},
        )
        data = entry.to_bytes()
        # Skip the 4-byte length prefix
        restored = WALEntry.from_bytes(data[4:])
        assert restored.seq == 42
        assert restored.op == WALOp.ADD_NEURON
        assert restored.payload["id"] == "n1"

    def test_all_op_types(self) -> None:
        for op in WALOp:
            entry = WALEntry(seq=1, op=op, timestamp="now", payload={})
            data = entry.to_bytes()
            restored = WALEntry.from_bytes(data[4:])
            assert restored.op == op


class TestWALRecovery:
    def test_corrupted_file_handled(self, wal_path: Path) -> None:
        """Corrupted WAL should not crash — just start fresh."""
        wal_path.write_bytes(b"garbage data")
        w = WriteAheadLog(wal_path)
        w.open()
        assert w.entry_count == 0
        w.close()

    def test_truncated_entry_handled(self, wal_path: Path) -> None:
        """WAL with truncated last entry should load all complete entries."""
        w = WriteAheadLog(wal_path)
        w.open()
        w.append(WALOp.ADD_NEURON, {"id": "n1"})
        w.append(WALOp.ADD_NEURON, {"id": "n2"})
        w.close()

        # Truncate last few bytes to simulate crash during write
        data = wal_path.read_bytes()
        wal_path.write_bytes(data[:-5])

        w2 = WriteAheadLog(wal_path)
        w2.open()
        entries = w2.get_pending_entries()
        assert len(entries) == 1  # only first complete entry
        assert entries[0].payload["id"] == "n1"
        w2.close()

    def test_invalid_magic_resets(self, wal_path: Path) -> None:
        """WAL with wrong magic should be reset."""
        wal_path.write_bytes(b"WRONG" + b"\x00" * 100)
        w = WriteAheadLog(wal_path)
        w.open()
        assert w.entry_count == 0
        # Should be able to write new entries
        w.append(WALOp.ADD_NEURON, {"id": "n1"})
        assert w.entry_count == 1
        w.close()

    def test_empty_file_handled(self, wal_path: Path) -> None:
        wal_path.write_bytes(b"")
        w = WriteAheadLog(wal_path)
        w.open()
        assert w.entry_count == 0
        w.close()

    def test_replay_after_crash(self, wal_path: Path) -> None:
        """Simulate crash: write entries but no checkpoint, then recover."""
        w1 = WriteAheadLog(wal_path)
        w1.open()
        w1.append(WALOp.ADD_NEURON, {"id": "n1", "content": "memory 1"})
        w1.append(WALOp.ADD_SYNAPSE, {"source": "n1", "target": "n2"})
        w1.append(WALOp.UPDATE_NEURON, {"id": "n1", "content": "updated"})
        # Simulate crash — don't close cleanly, just close file handle
        w1._file.close()
        w1._file = None

        # Recovery: read pending entries
        w2 = WriteAheadLog(wal_path)
        w2.open()
        pending = w2.get_pending_entries()
        assert len(pending) == 3
        assert pending[0].op == WALOp.ADD_NEURON
        assert pending[1].op == WALOp.ADD_SYNAPSE
        assert pending[2].op == WALOp.UPDATE_NEURON
        assert pending[2].payload["content"] == "updated"

        # After successful replay, checkpoint
        w2.checkpoint()
        assert w2.entry_count == 0
        w2.close()

    def test_append_after_recovery(self, wal_path: Path) -> None:
        """After recovering from crash, new entries should work."""
        w1 = WriteAheadLog(wal_path)
        w1.open()
        w1.append(WALOp.ADD_NEURON, {"id": "n1"})
        w1._file.close()
        w1._file = None

        w2 = WriteAheadLog(wal_path)
        w2.open()
        w2.append(WALOp.ADD_NEURON, {"id": "n2"})
        assert w2.entry_count == 2
        entries = w2.get_pending_entries()
        ids = [e.payload["id"] for e in entries]
        assert ids == ["n1", "n2"]
        w2.close()


class TestWALOps:
    def test_op_values(self) -> None:
        assert WALOp.ADD_NEURON == 1
        assert WALOp.DELETE_NEURON == 2
        assert WALOp.UPDATE_NEURON == 3
        assert WALOp.ADD_SYNAPSE == 4
        assert WALOp.DELETE_SYNAPSE == 5
        assert WALOp.ADD_FIBER == 6
        assert WALOp.DELETE_FIBER == 7
        assert WALOp.CHECKPOINT == 100

    def test_not_opened_raises(self, wal_path: Path) -> None:
        w = WriteAheadLog(wal_path)
        with pytest.raises(RuntimeError, match="WAL not opened"):
            w.append(WALOp.ADD_NEURON, {})
