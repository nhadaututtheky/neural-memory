"""VectorSearchMixin — adds knn_search to SQLStorage.

Lazy-loads a SQLiteVectorIndex sidecar on first call. Returns empty
results gracefully when hnswlib is not installed or no index exists.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.engine.embedding.vector_index import SQLiteVectorIndex

logger = logging.getLogger(__name__)


class VectorSearchMixin:
    """Mixin that adds vector search capabilities to SQL-backed storage."""

    _vector_index: SQLiteVectorIndex | None
    _vector_available: bool
    _vector_cold_start_warned: bool

    def _init_vector_search(self) -> None:
        """Initialize vector search state. Call from storage __init__."""
        self._vector_index = None
        self._vector_available = False
        self._vector_cold_start_warned = False

    def _ensure_vector_index(self) -> SQLiteVectorIndex | None:
        """Lazy-load the vector index on first access."""
        if self._vector_index is not None:
            return self._vector_index

        from neural_memory.engine.embedding.vector_index import (
            SQLiteVectorIndex,
            is_available,
        )

        if not is_available():
            if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
                raise RuntimeError("hnswlib is required for benchmark vector search")
            if not self._vector_cold_start_warned:
                logger.debug("hnswlib not installed — knn_search disabled")
                self._vector_cold_start_warned = True
            return None

        # Determine sidecar path from the dialect's db_path
        db_path = getattr(self, "_dialect", None)
        if db_path is not None:
            db_path = getattr(db_path, "_db_path", None)
        if db_path is None:
            if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
                raise RuntimeError("benchmark vector search requires a SQLite database path")
            return None

        from pathlib import Path

        db_path = Path(db_path)
        brain_id = getattr(self, "_current_brain_id", None) or "default"
        base_path = db_path.parent / brain_id

        try:
            index = SQLiteVectorIndex(base_path=base_path)
            index.open()
            self._vector_index = index
            self._vector_available = True
            return index
        except Exception:
            if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
                raise
            logger.debug("Failed to open vector index", exc_info=True)
            return None

    async def knn_search(
        self,
        query_vector: list[float],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """K-nearest-neighbor search — backend-aware.

        - SQLite: HNSW sidecar (hnswlib), live-ID validated.
        - Postgres (dialect.supports_vector): pgvector via
          ``find_neurons_by_embedding`` — never the SQLite HNSW path.

        Returns:
            [(neuron_id, similarity)] sorted by similarity descending.
            Empty list if index unavailable or empty.
        """
        dialect = getattr(self, "_dialect", None)
        # Postgres / vector-capable SQL: do NOT use SQLite HNSW sidecar
        if dialect is not None and getattr(dialect, "supports_vector", False):
            find_emb = getattr(self, "find_neurons_by_embedding", None)
            if find_emb is None:
                return []
            try:
                hits = await find_emb(query_vector, limit=k)
            except Exception:
                logger.debug("pgvector knn_search failed", exc_info=True)
                return []
            # Normalize to (id, score) — may be Neuron objects or tuples
            out: list[tuple[str, float]] = []
            for item in hits or []:
                if isinstance(item, tuple) and len(item) >= 2:
                    out.append((str(item[0]), float(item[1])))
                else:
                    nid = getattr(item, "id", None)
                    score = float(getattr(item, "score", getattr(item, "similarity", 0.0)))
                    if nid:
                        out.append((str(nid), score))
                if len(out) >= k:
                    break
            return out

        index = self._ensure_vector_index()
        if index is None or index.count == 0:
            if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
                raise RuntimeError("benchmark vector index is unavailable or empty")
            if not self._vector_cold_start_warned:
                logger.info("No vector index found. Run `nmem embed` to enable semantic search.")
                self._vector_cold_start_warned = True
            return []
        # Over-fetch then filter so k live hits remain when some IDs are stale.
        raw = index.search(query_vector, k=max(k * 3, k))
        if not raw:
            return []
        live = await self._filter_live_neuron_ids([nid for nid, _ in raw])
        validated: list[tuple[str, float]] = []
        for nid, score in raw:
            if nid in live:
                validated.append((nid, score))
            if len(validated) >= k:
                break
        return validated

    async def _filter_live_neuron_ids(self, neuron_ids: list[str]) -> set[str]:
        """Return the subset of IDs that still exist in the active brain."""
        if not neuron_ids:
            return set()
        get_batch = getattr(self, "get_neurons_batch", None)
        if get_batch is None:
            return set(neuron_ids)
        found = await get_batch(neuron_ids)
        return set(found)

    def _persist_vector_index(self) -> None:
        """Best-effort save so crash before close() does not lose mutations."""
        index = self._vector_index
        if index is None:
            return
        save = getattr(index, "save", None)
        if callable(save):
            try:
                save()
            except Exception:
                logger.debug("vector index save failed (non-critical)", exc_info=True)

    async def vector_index_add(self, neuron_id: str, vector: list[float]) -> None:
        """Add or replace a neuron's embedding in the vector index."""
        dialect = getattr(self, "_dialect", None)
        if dialect is not None and getattr(dialect, "supports_vector", False):
            # Postgres: embeddings live in SQL; sidecar not used
            return
        index = self._ensure_vector_index()
        if index is not None:
            # Replace semantics: remove prior vector if present, then add.
            try:
                index.remove(neuron_id)
            except Exception:
                logger.debug("vector replace: prior id absent for %s", neuron_id, exc_info=True)
            index.add(neuron_id, vector)
            self._persist_vector_index()

    async def vector_index_remove(self, neuron_id: str) -> None:
        """Remove/tombstone a neuron from the vector index."""
        dialect = getattr(self, "_dialect", None)
        if dialect is not None and getattr(dialect, "supports_vector", False):
            return
        if self._vector_index is not None:
            self._vector_index.remove(neuron_id)
            self._persist_vector_index()
        else:
            # Lazy-open so delete after brain switch still hits the right sidecar.
            index = self._ensure_vector_index()
            if index is not None:
                index.remove(neuron_id)
                self._persist_vector_index()

    def _close_vector_index(self) -> None:
        """Close and save vector index. Call from storage close()."""
        if self._vector_index is not None:
            try:
                self._vector_index.close()
            except Exception:
                logger.debug("Failed to close vector index", exc_info=True)
            self._vector_index = None
