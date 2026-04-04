"""Msgpack-based metadata storage for InfinityDB.

Stores neuron metadata (id, type, content, timestamps, priority, etc.)
in a compact msgpack format with in-memory indexes for fast lookup.

On-disk: single msgpack file with array of dicts.
In-memory: dict[neuron_id -> slot_index] for O(1) ID lookup,
           and the full metadata list for iteration/search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import msgpack

logger = logging.getLogger(__name__)


class MetadataStore:
    """Neuron metadata storage with msgpack serialization."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # slot_index -> metadata dict
        self._records: dict[int, dict[str, Any]] = {}
        # neuron_id -> slot_index
        self._id_index: dict[str, int] = {}
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._records)

    def open(self) -> None:
        """Load metadata from disk."""
        # Recover from interrupted flush
        tmp = self._path.with_suffix(".meta.tmp")
        if tmp.exists():
            logger.warning("Recovering from interrupted flush: %s", tmp)
            tmp.replace(self._path)

        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                with open(self._path, "rb") as f:
                    raw = msgpack.unpack(f, raw=False)
                if isinstance(raw, dict):
                    records = raw.get("records", {})
                    for slot_str, meta in records.items():
                        slot = int(slot_str)
                        self._records[slot] = meta
                        if "id" in meta:
                            self._id_index[meta["id"]] = slot
                logger.debug("Loaded %d metadata records", len(self._records))
            except (msgpack.UnpackException, ValueError, TypeError) as e:
                logger.error("Corrupted metadata file %s: %s — starting fresh", self._path, e)
                self._records.clear()
                self._id_index.clear()
        else:
            logger.debug("No existing metadata, starting fresh")

    def add(self, slot: int, metadata: dict[str, Any]) -> None:
        """Add metadata for a vector slot."""
        neuron_id = metadata.get("id", "")
        if neuron_id and neuron_id in self._id_index:
            msg = f"Neuron ID already exists: {neuron_id}"
            raise ValueError(msg)
        self._records[slot] = dict(metadata)  # Defensive copy
        if neuron_id:
            self._id_index[neuron_id] = slot
        self._dirty = True

    def get_by_slot(self, slot: int) -> dict[str, Any] | None:
        """Get metadata by vector slot index. Returns a copy."""
        meta = self._records.get(slot)
        return dict(meta) if meta is not None else None

    def get_by_id(self, neuron_id: str) -> tuple[int, dict[str, Any]] | None:
        """Get (slot, metadata) by neuron ID. Returns a copy of metadata."""
        slot = self._id_index.get(neuron_id)
        if slot is None:
            return None
        meta = self._records.get(slot)
        if meta is None:
            return None
        return slot, dict(meta)

    def update(self, slot: int, updates: dict[str, Any]) -> bool:
        """Update metadata fields for a slot."""
        meta = self._records.get(slot)
        if meta is None:
            return False
        # If ID is changing, update index
        old_id = meta.get("id", "")
        new_id = updates.get("id", old_id)
        if new_id != old_id:
            if old_id in self._id_index:
                del self._id_index[old_id]
            self._id_index[new_id] = slot
        self._records[slot] = {**meta, **updates}
        self._dirty = True
        return True

    def delete(self, slot: int) -> bool:
        """Delete metadata for a slot."""
        meta = self._records.get(slot)
        if meta is None:
            return False
        neuron_id = meta.get("id", "")
        if neuron_id in self._id_index:
            del self._id_index[neuron_id]
        del self._records[slot]
        self._dirty = True
        return True

    def find(
        self,
        *,
        neuron_type: str | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[str, str] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Find metadata matching filters. Returns list of (slot, metadata)."""
        results: list[tuple[int, dict[str, Any]]] = []

        for slot, meta in self._records.items():
            if neuron_type is not None and meta.get("type") != neuron_type:
                continue
            if content_exact is not None and meta.get("content") != content_exact:
                continue
            if content_contains is not None:
                content = meta.get("content", "")
                if content_contains.lower() not in content.lower():
                    continue
            if time_range is not None:
                created = meta.get("created_at", "")
                if created and (created < time_range[0] or created > time_range[1]):
                    continue
            if ephemeral is not None and meta.get("ephemeral") != ephemeral:
                continue
            results.append((slot, meta))

        # Sort by created_at descending (newest first)
        results.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)

        # Apply offset and limit
        return results[offset : offset + limit]

    def suggest(
        self,
        prefix: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons by content prefix match."""
        results: list[dict[str, Any]] = []
        prefix_lower = prefix.lower()
        for meta in self._records.values():
            if type_filter is not None and meta.get("type") != type_filter:
                continue
            content = meta.get("content", "")
            if content.lower().startswith(prefix_lower):
                results.append(meta)
                if len(results) >= limit:
                    break
        return results

    def next_free_slot(self) -> int:
        """Get next available negative slot for neurons without vectors."""
        existing = self._records
        slot = -1
        while slot in existing:
            slot -= 1
        return slot

    def get_all_ids(self) -> list[str]:
        """Get all neuron IDs."""
        return list(self._id_index.keys())

    def get_slot_for_id(self, neuron_id: str) -> int | None:
        """Get the vector slot for a neuron ID."""
        return self._id_index.get(neuron_id)

    def iter_all(self) -> list[tuple[int, dict[str, Any]]]:
        """Iterate all records as (slot, metadata) pairs."""
        return list(self._records.items())

    def flush(self) -> None:
        """Save metadata to disk if dirty. Uses atomic write (tmp + rename)."""
        if not self._dirty:
            return
        data = {
            "version": 1,
            "records": {str(k): v for k, v in self._records.items()},
        }
        # Atomic write: write to temp file, then rename
        tmp_path = self._path.with_suffix(".meta.tmp")
        with open(tmp_path, "wb") as f:
            msgpack.pack(data, f, use_bin_type=True)
        tmp_path.replace(self._path)
        self._dirty = False
        logger.debug("Flushed %d metadata records to disk", len(self._records))

    def close(self) -> None:
        """Flush and close."""
        self.flush()
        self._records.clear()
        self._id_index.clear()
