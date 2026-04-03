"""Memory-mapped vector storage for InfinityDB.

Stores N-dimensional float32 vectors in a numpy mmap file.
Supports append, read, delete (mark-as-deleted), and auto-grow.

Layout: flat float32 array of shape (capacity, dimensions).
Deleted vectors are zeroed out and tracked in a free set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Initial capacity and growth factor
INITIAL_CAPACITY = 1024
GROWTH_FACTOR = 2


class VectorStore:
    """Memory-mapped vector storage with auto-grow."""

    def __init__(self, path: Path, dimensions: int) -> None:
        self._path = path
        self._dimensions = dimensions
        self._capacity = 0
        self._count = 0
        self._mmap: NDArray[np.float32] | None = None
        self._free_slots: set[int] = set()
        self._next_pos = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def open(self) -> None:
        """Open or create the vector file."""
        if self._path.exists() and self._path.stat().st_size > 0:
            self._mmap = np.memmap(
                str(self._path),
                dtype=np.float32,
                mode="r+",
            )
            total_elements = self._mmap.shape[0]
            self._capacity = total_elements // self._dimensions
            self._mmap = self._mmap.reshape((self._capacity, self._dimensions))
            self._restore_state()
            logger.debug(
                "Opened vector store: capacity=%d, dims=%d, count=%d, next_pos=%d",
                self._capacity,
                self._dimensions,
                self._count,
                self._next_pos,
            )
        else:
            self._create_new(INITIAL_CAPACITY)

    def _restore_state(self) -> None:
        """Restore _next_pos, _count, and _free_slots from mmap contents."""
        if self._mmap is None:
            return
        row_norms = np.linalg.norm(self._mmap, axis=1)
        occupied = row_norms > 0
        occupied_indices = np.where(occupied)[0]
        if len(occupied_indices) > 0:
            self._next_pos = int(occupied_indices[-1]) + 1
            self._count = int(np.sum(occupied[: self._next_pos]))
            free_indices = np.where(~occupied[: self._next_pos])[0]
            self._free_slots = set(free_indices.tolist())
        else:
            self._next_pos = 0
            self._count = 0
            self._free_slots = set()

    def _create_new(self, capacity: int) -> None:
        """Create a new mmap file with given capacity."""
        self._capacity = capacity
        self._mmap = np.memmap(
            str(self._path),
            dtype=np.float32,
            mode="w+",
            shape=(capacity, self._dimensions),
        )
        self._mmap[:] = 0
        self._mmap.flush()
        logger.debug("Created vector store: capacity=%d, dims=%d", capacity, self._dimensions)

    def _grow(self) -> None:
        """Double the capacity of the vector store."""
        if self._mmap is None:
            msg = "Vector store not opened"
            raise RuntimeError(msg)

        old_capacity = self._capacity
        new_capacity = old_capacity * GROWTH_FACTOR

        # CRITICAL: flush before copying to ensure data is on disk
        self._mmap.flush()  # type: ignore[attr-defined]
        old_data = np.array(self._mmap[:old_capacity], copy=True)

        # Release old mmap
        del self._mmap

        # Create new larger file
        self._mmap = np.memmap(
            str(self._path),
            dtype=np.float32,
            mode="w+",
            shape=(new_capacity, self._dimensions),
        )
        self._mmap[:old_capacity] = old_data
        self._mmap[old_capacity:] = 0
        self._mmap.flush()
        self._capacity = new_capacity
        logger.debug("Grew vector store: %d -> %d", old_capacity, new_capacity)

    def add(self, vector: NDArray[np.float32]) -> int:
        """Add a vector, return its slot index."""
        if self._mmap is None:
            msg = "Vector store not opened"
            raise RuntimeError(msg)

        if vector.shape != (self._dimensions,):
            msg = f"Expected shape ({self._dimensions},), got {vector.shape}"
            raise ValueError(msg)

        # Reuse a free slot if available
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._next_pos
            # Grow BEFORE incrementing _next_pos to avoid state corruption on failure
            if slot >= self._capacity:
                self._grow()
            self._next_pos += 1

        self._mmap[slot] = vector
        self._count += 1
        return slot

    def get(self, slot: int) -> NDArray[np.float32] | None:
        """Get vector at slot. Returns None if deleted/empty."""
        if self._mmap is None or slot < 0 or slot >= self._next_pos:
            return None
        if slot in self._free_slots:
            return None
        vec = np.array(self._mmap[slot], copy=True)
        # Check if zeroed (deleted) — fallback for restored state
        if np.all(vec == 0):
            return None
        return vec

    def get_batch(self, slots: list[int]) -> NDArray[np.float32]:
        """Get multiple vectors at once. Returns (len(slots), dims) array."""
        if self._mmap is None:
            msg = "Vector store not opened"
            raise RuntimeError(msg)
        valid = [s for s in slots if 0 <= s < self._next_pos and s not in self._free_slots]
        if not valid:
            return np.empty((0, self._dimensions), dtype=np.float32)
        return np.array(self._mmap[valid], copy=True)

    def delete(self, slot: int) -> bool:
        """Mark a vector slot as deleted (zero it out)."""
        if self._mmap is None or slot < 0 or slot >= self._next_pos:
            return False
        self._mmap[slot] = 0
        self._free_slots.add(slot)
        self._count = max(0, self._count - 1)
        return True

    def update(self, slot: int, vector: NDArray[np.float32]) -> bool:
        """Update vector at slot."""
        if self._mmap is None or slot < 0 or slot >= self._next_pos:
            return False
        if vector.shape != (self._dimensions,):
            msg = f"Expected shape ({self._dimensions},), got {vector.shape}"
            raise ValueError(msg)
        self._mmap[slot] = vector
        # Remove from free slots if it was there
        self._free_slots.discard(slot)
        return True

    def get_all_vectors(self) -> tuple[list[int], NDArray[np.float32]]:
        """Get all non-deleted vectors. Returns (slot_indices, vectors_array)."""
        if self._mmap is None:
            return [], np.empty((0, self._dimensions), dtype=np.float32)

        slots = [
            i
            for i in range(self._next_pos)
            if i not in self._free_slots and not np.all(self._mmap[i] == 0)
        ]

        if not slots:
            return [], np.empty((0, self._dimensions), dtype=np.float32)

        vectors = np.array(self._mmap[slots], copy=True)
        return slots, vectors

    def flush(self) -> None:
        """Flush mmap to disk."""
        if self._mmap is not None:
            self._mmap.flush()  # type: ignore[attr-defined]

    def close(self) -> None:
        """Close the vector store."""
        if self._mmap is not None:
            self._mmap.flush()  # type: ignore[attr-defined]
            del self._mmap
            self._mmap = None
