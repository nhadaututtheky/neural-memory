"""Write-Ahead Log for InfinityDB.

Provides crash safety by logging operations BEFORE applying them.
On crash recovery, the WAL is replayed to restore consistent state.

WAL entry format (msgpack):
  [sequence_number, operation, timestamp, payload]

Operations: ADD_NEURON, DELETE_NEURON, UPDATE_NEURON, ADD_SYNAPSE,
DELETE_SYNAPSE, ADD_FIBER, DELETE_FIBER, CHECKPOINT

Recovery protocol:
1. On open, check for WAL file
2. If WAL exists and no checkpoint marker, replay from last checkpoint
3. After successful flush, write CHECKPOINT and truncate WAL
"""

from __future__ import annotations

import io
import logging
import struct
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import msgpack

logger = logging.getLogger(__name__)

# WAL file header magic
WAL_MAGIC = b"IWAL\x01"
WAL_HEADER_SIZE = 5

# Maximum WAL size before forced checkpoint (50MB)
MAX_WAL_SIZE = 50 * 1024 * 1024


class WALOp(IntEnum):
    """WAL operation types."""

    ADD_NEURON = 1
    DELETE_NEURON = 2
    UPDATE_NEURON = 3
    ADD_SYNAPSE = 4
    DELETE_SYNAPSE = 5
    ADD_FIBER = 6
    DELETE_FIBER = 7
    CHECKPOINT = 100


class WALEntry:
    """A single WAL entry."""

    __slots__ = ("op", "payload", "seq", "timestamp")

    def __init__(
        self,
        seq: int,
        op: WALOp,
        timestamp: str,
        payload: dict[str, Any],
    ) -> None:
        self.seq = seq
        self.op = op
        self.timestamp = timestamp
        self.payload = payload

    def to_bytes(self) -> bytes:
        """Serialize entry to bytes with length prefix."""
        data: bytes = msgpack.packb(
            [self.seq, int(self.op), self.timestamp, self.payload],
            use_bin_type=True,
        )
        # Length-prefix for framing: [4-byte length][msgpack data]
        return struct.pack("<I", len(data)) + data

    @classmethod
    def from_bytes(cls, data: bytes) -> WALEntry:
        """Deserialize entry from msgpack bytes (without length prefix)."""
        arr = msgpack.unpackb(data, raw=False)
        return cls(
            seq=arr[0],
            op=WALOp(arr[1]),
            timestamp=arr[2],
            payload=arr[3],
        )


def _utcnow() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


class WriteAheadLog:
    """Append-only write-ahead log for crash recovery."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: io.BufferedWriter | None = None
        self._seq = 0
        self._entry_count = 0
        self._is_open = False

    @property
    def entry_count(self) -> int:
        return self._entry_count

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def current_size(self) -> int:
        """Current WAL file size in bytes."""
        if self._path.exists():
            return self._path.stat().st_size
        return 0

    def open(self) -> None:
        """Open or create the WAL file."""
        if self._path.exists() and self._path.stat().st_size > 0:
            # Validate header
            with open(self._path, "rb") as f:
                magic = f.read(WAL_HEADER_SIZE)
                if magic != WAL_MAGIC:
                    logger.warning("Invalid WAL magic, resetting: %s", self._path)
                    self._create_new()
                    return
            # Count existing entries and find last seq
            self._scan_existing()
            # Open for append
            self._file = open(self._path, "ab")
        else:
            self._create_new()

        self._is_open = True
        logger.debug("WAL opened: %d entries, seq=%d", self._entry_count, self._seq)

    def _create_new(self) -> None:
        """Create a fresh WAL file with header."""
        self._file = open(self._path, "wb")
        self._file.write(WAL_MAGIC)
        self._file.flush()
        self._seq = 0
        self._entry_count = 0

    def _scan_existing(self) -> None:
        """Scan WAL to count entries and find last sequence number."""
        self._entry_count = 0
        self._seq = 0
        entries = self._read_all_entries()
        if entries:
            self._entry_count = len(entries)
            self._seq = entries[-1].seq

    def append(self, op: WALOp, payload: dict[str, Any]) -> int:
        """Append an entry to the WAL. Returns the sequence number."""
        if self._file is None:
            msg = "WAL not opened"
            raise RuntimeError(msg)

        self._seq += 1
        entry = WALEntry(
            seq=self._seq,
            op=op,
            timestamp=_utcnow(),
            payload=payload,
        )

        data = entry.to_bytes()
        self._file.write(data)
        self._file.flush()
        self._entry_count += 1

        return self._seq

    def checkpoint(self) -> None:
        """Write a checkpoint marker and truncate the WAL.

        Called after successful flush of all stores to disk.
        """
        if self._file is None:
            return

        # Close current file
        self._file.close()

        # Truncate by recreating
        self._create_new()
        self._is_open = True

        logger.debug("WAL checkpointed — truncated")

    def needs_checkpoint(self) -> bool:
        """Check if WAL should be checkpointed (size threshold)."""
        return self.current_size > MAX_WAL_SIZE

    def get_pending_entries(self) -> list[WALEntry]:
        """Get all entries since last checkpoint (for replay)."""
        return self._read_all_entries()

    def _read_all_entries(self) -> list[WALEntry]:
        """Read all entries from the WAL file."""
        entries: list[WALEntry] = []
        try:
            with open(self._path, "rb") as f:
                magic = f.read(WAL_HEADER_SIZE)
                if magic != WAL_MAGIC:
                    return []

                while True:
                    len_bytes = f.read(4)
                    if len(len_bytes) < 4:
                        break  # EOF or truncated
                    length = struct.unpack("<I", len_bytes)[0]
                    if length > 10 * 1024 * 1024:  # >10MB single entry = corrupt
                        logger.warning("WAL entry too large (%d bytes), stopping", length)
                        break
                    data = f.read(length)
                    if len(data) < length:
                        logger.warning("Truncated WAL entry at seq %d", len(entries))
                        break  # Truncated entry — stop here
                    try:
                        entry = WALEntry.from_bytes(data)
                        entries.append(entry)
                    except (msgpack.UnpackException, ValueError, IndexError) as e:
                        logger.warning("Corrupt WAL entry at offset %d: %s", f.tell(), e)
                        break

        except OSError as e:
            logger.error("Cannot read WAL: %s", e)

        return entries

    def close(self) -> None:
        """Close the WAL file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._is_open = False
