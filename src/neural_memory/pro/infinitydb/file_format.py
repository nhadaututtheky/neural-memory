"""InfinityDB file format — header spec for .inf files.

File layout per brain directory:
  brain.inf   — Header (this module)
  brain.vec   — Memory-mapped float32 vectors (N x D)
  brain.idx   — HNSW index (hnswlib binary)
  brain.graph — Synapse adjacency (msgpack)
  brain.meta  — Neuron metadata (msgpack)
  brain.wal   — Write-ahead log
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# Magic bytes: "INFDB" + version
MAGIC = b"INFDB\x01"
HEADER_FORMAT = "<6sIIIIQQ"  # magic(6)+ver(4)+dims(4)+tier(4)+flags(4)+neurons(8)+synapses(8)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# File extensions
EXT_HEADER = ".inf"
EXT_VECTORS = ".vec"
EXT_INDEX = ".idx"
EXT_GRAPH = ".graph"
EXT_META = ".meta"
EXT_WAL = ".wal"
EXT_FIBERS = ".fibers"


@dataclass(frozen=True)
class InfinityHeader:
    """Header stored in brain.inf file."""

    version: int = 1
    dimensions: int = 384
    tier_config: int = 0  # bitmask for tier settings
    flags: int = 0
    neuron_count: int = 0
    synapse_count: int = 0

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            HEADER_FORMAT,
            MAGIC,
            self.version,
            self.dimensions,
            self.tier_config,
            self.flags,
            self.neuron_count,
            self.synapse_count,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> InfinityHeader:
        """Deserialize header from bytes."""
        if len(data) < HEADER_SIZE:
            msg = f"Header too short: {len(data)} < {HEADER_SIZE}"
            raise ValueError(msg)
        magic, version, dims, tier_config, flags, n_count, s_count = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        if magic != MAGIC:
            msg = f"Invalid magic bytes: {magic!r}"
            raise ValueError(msg)
        return cls(
            version=version,
            dimensions=dims,
            tier_config=tier_config,
            flags=flags,
            neuron_count=n_count,
            synapse_count=s_count,
        )


# Safe brain_id pattern: alphanumeric, dash, underscore, dot
_SAFE_BRAIN_ID = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


@dataclass
class BrainPaths:
    """All file paths for a brain directory."""

    base_dir: Path
    brain_id: str
    header: Path = field(init=False)
    vectors: Path = field(init=False)
    index: Path = field(init=False)
    graph: Path = field(init=False)
    meta: Path = field(init=False)
    wal: Path = field(init=False)
    fibers: Path = field(init=False)

    def __post_init__(self) -> None:
        # SECURITY: validate brain_id to prevent path traversal
        if not _SAFE_BRAIN_ID.match(self.brain_id):
            msg = f"Invalid brain_id: {self.brain_id!r} — must match [a-zA-Z0-9_\\-\\.]+"
            raise ValueError(msg)
        d = (self.base_dir / self.brain_id).resolve()
        if not d.is_relative_to(self.base_dir.resolve()):
            msg = f"brain_id resolves outside base_dir: {d}"
            raise ValueError(msg)
        self.header = d / f"brain{EXT_HEADER}"
        self.vectors = d / f"brain{EXT_VECTORS}"
        self.index = d / f"brain{EXT_INDEX}"
        self.graph = d / f"brain{EXT_GRAPH}"
        self.meta = d / f"brain{EXT_META}"
        self.wal = d / f"brain{EXT_WAL}"
        self.fibers = d / f"brain{EXT_FIBERS}"

    @property
    def brain_dir(self) -> Path:
        return self.base_dir / self.brain_id

    def ensure_dirs(self) -> None:
        """Create brain directory if it doesn't exist."""
        self.brain_dir.mkdir(parents=True, exist_ok=True)
