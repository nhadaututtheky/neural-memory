"""HNSW approximate nearest neighbor index for InfinityDB.

Wraps hnswlib for fast vector similarity search.
Supports add, search, delete (mark), save/load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hnswlib
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# HNSW tuning parameters
DEFAULT_M = 16  # Number of bi-directional links per element
DEFAULT_EF_CONSTRUCTION = 200  # Construction time/accuracy tradeoff
DEFAULT_EF_SEARCH = 100  # Search time/accuracy tradeoff
DEFAULT_SPACE = "cosine"  # Distance metric


class HNSWIndex:
    """HNSW approximate nearest neighbor index."""

    def __init__(
        self,
        path: Path,
        dimensions: int,
        *,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        space: str = DEFAULT_SPACE,
    ) -> None:
        self._path = path
        self._dimensions = dimensions
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._space = space
        self._index: hnswlib.Index | None = None
        self._max_elements = 0
        self._current_count = 0

    @property
    def count(self) -> int:
        return self._current_count

    def open(self, max_elements: int = 1024) -> None:
        """Open or create the HNSW index."""
        self._index = hnswlib.Index(space=self._space, dim=self._dimensions)

        if self._path.exists() and self._path.stat().st_size > 0:
            self._index.load_index(str(self._path), max_elements=max_elements)
            self._max_elements = max_elements
            self._current_count = self._index.get_current_count()
            self._index.set_ef(self._ef_search)
            logger.debug(
                "Loaded HNSW index: %d elements, max=%d",
                self._current_count,
                self._max_elements,
            )
        else:
            self._max_elements = max_elements
            self._index.init_index(
                max_elements=max_elements,
                M=self._m,
                ef_construction=self._ef_construction,
            )
            self._index.set_ef(self._ef_search)
            logger.debug("Created new HNSW index: max=%d", max_elements)

    def _ensure_capacity(self, needed: int) -> None:
        """Resize index if needed."""
        if self._index is None:
            msg = "Index not opened"
            raise RuntimeError(msg)
        if needed > self._max_elements:
            new_max = max(needed, self._max_elements * 2)
            self._index.resize_index(new_max)
            self._max_elements = new_max
            logger.debug("Resized HNSW index to %d", new_max)

    def add(self, slot: int, vector: NDArray[np.float32]) -> None:
        """Add a vector with its slot ID to the index."""
        if self._index is None:
            msg = "Index not opened"
            raise RuntimeError(msg)
        self._ensure_capacity(self._current_count + 1)
        self._index.add_items(
            vector.reshape(1, -1).astype(np.float32),
            np.array([slot], dtype=np.int64),
        )
        self._current_count += 1

    def add_batch(self, slots: list[int], vectors: NDArray[np.float32]) -> None:
        """Add multiple vectors at once."""
        if self._index is None:
            msg = "Index not opened"
            raise RuntimeError(msg)
        n = len(slots)
        if n == 0:
            return
        self._ensure_capacity(self._current_count + n)
        self._index.add_items(
            vectors.astype(np.float32),
            np.array(slots, dtype=np.int64),
        )
        self._current_count += n

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> tuple[list[int], list[float]]:
        """Search for k nearest neighbors.

        Returns:
            (slot_ids, distances) — slots sorted by similarity (best first).
            For cosine space, distance = 1 - cosine_similarity.
        """
        if self._index is None or self._current_count == 0:
            return [], []

        k = min(k, self._current_count)
        labels, distances = self._index.knn_query(query.reshape(1, -1).astype(np.float32), k=k)

        slot_ids = labels[0].tolist()
        dists = distances[0].tolist()
        return slot_ids, dists

    def search_batch(
        self,
        queries: NDArray[np.float32],
        k: int = 10,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Batch search for k nearest neighbors per query."""
        if self._index is None or self._current_count == 0:
            n = queries.shape[0]
            return np.empty((n, 0), dtype=np.int64), np.empty((n, 0), dtype=np.float32)

        k = min(k, self._current_count)
        labels, distances = self._index.knn_query(queries.astype(np.float32), k=k)
        return labels, distances

    def delete(self, slot: int) -> None:
        """Mark an element as deleted in the index."""
        if self._index is None:
            return
        try:
            self._index.mark_deleted(slot)
            self._current_count = max(0, self._current_count - 1)
        except RuntimeError as e:
            # Only suppress "not found" errors, re-raise others
            if "not found" in str(e).lower() or "label" in str(e).lower():
                logger.debug("HNSW delete: slot %d not in index", slot)
            else:
                raise

    def save(self) -> None:
        """Save index to disk."""
        if self._index is not None:
            self._index.save_index(str(self._path))
            logger.debug("Saved HNSW index: %d elements", self._current_count)

    def close(self) -> None:
        """Save and release the index."""
        self.save()
        self._index = None
