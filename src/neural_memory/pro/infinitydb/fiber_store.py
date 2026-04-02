"""Fiber storage for InfinityDB.

Fibers are named collections of neurons — analogous to brain regions
or memory clusters. Each fiber has metadata (name, type, description)
and a list of neuron IDs it contains.

On-disk: msgpack file with fiber dict.
In-memory: dict[fiber_id -> fiber_dict] + reverse index [neuron_id -> set[fiber_id]].
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import msgpack

logger = logging.getLogger(__name__)


class FiberStore:
    """Named neuron collections (fibers/clusters)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # fiber_id -> fiber dict (id, name, type, description, neuron_ids, metadata)
        self._fibers: dict[str, dict[str, Any]] = {}
        # neuron_id -> set of fiber_ids (reverse lookup)
        self._neuron_fibers: dict[str, set[str]] = {}
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._fibers)

    def open(self) -> None:
        """Load fibers from disk."""
        # Recover from interrupted flush
        tmp = self._path.with_suffix(".fibers.tmp")
        if tmp.exists():
            logger.warning("Recovering from interrupted flush: %s", tmp)
            tmp.replace(self._path)

        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                with open(self._path, "rb") as f:
                    raw = msgpack.unpack(f, raw=False)
                if isinstance(raw, dict):
                    fibers = raw.get("fibers", {})
                    for fid, fiber in fibers.items():
                        self._fibers[fid] = fiber
                        for nid in fiber.get("neuron_ids", []):
                            self._neuron_fibers.setdefault(nid, set()).add(fid)
                logger.debug("Loaded %d fibers", len(self._fibers))
            except (msgpack.UnpackException, ValueError, TypeError) as e:
                logger.error("Corrupted fiber file %s: %s — starting fresh", self._path, e)
                self._fibers.clear()
                self._neuron_fibers.clear()
        else:
            logger.debug("No existing fibers, starting fresh")

    def add_fiber(
        self,
        name: str,
        *,
        fiber_id: str | None = None,
        fiber_type: str = "cluster",
        description: str = "",
        neuron_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new fiber. Returns fiber ID."""
        fid = fiber_id or str(uuid.uuid4())
        if fid in self._fibers:
            msg = f"Fiber ID already exists: {fid}"
            raise ValueError(msg)
        nids = list(neuron_ids) if neuron_ids else []

        fiber: dict[str, Any] = {
            "id": fid,
            "name": name,
            "type": fiber_type,
            "description": description,
            "neuron_ids": nids,
        }
        if metadata:
            fiber["metadata"] = dict(metadata)

        self._fibers[fid] = fiber
        for nid in nids:
            self._neuron_fibers.setdefault(nid, set()).add(fid)

        self._dirty = True
        return fid

    def get_fiber(self, fiber_id: str) -> dict[str, Any] | None:
        """Get fiber by ID. Returns a copy."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return None
        return {**fiber, "neuron_ids": list(fiber.get("neuron_ids", []))}

    def find_fibers(
        self,
        *,
        name_contains: str | None = None,
        fiber_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find fibers matching filters."""
        results: list[dict[str, Any]] = []
        for fiber in self._fibers.values():
            if fiber_type is not None and fiber.get("type") != fiber_type:
                continue
            if name_contains is not None:
                name = fiber.get("name", "")
                if name_contains.lower() not in name.lower():
                    continue
            results.append({**fiber, "neuron_ids": list(fiber.get("neuron_ids", []))})
            if len(results) >= limit:
                break
        return results

    def add_neuron_to_fiber(self, fiber_id: str, neuron_id: str) -> bool:
        """Add a neuron to a fiber."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return False
        nids = fiber.get("neuron_ids", [])
        if neuron_id not in nids:
            nids.append(neuron_id)
            fiber["neuron_ids"] = nids
            self._neuron_fibers.setdefault(neuron_id, set()).add(fiber_id)
            self._dirty = True
        return True

    def remove_neuron_from_fiber(self, fiber_id: str, neuron_id: str) -> bool:
        """Remove a neuron from a fiber."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return False
        nids = fiber.get("neuron_ids", [])
        if neuron_id in nids:
            nids.remove(neuron_id)
            fiber["neuron_ids"] = nids
            if neuron_id in self._neuron_fibers:
                self._neuron_fibers[neuron_id].discard(fiber_id)
                if not self._neuron_fibers[neuron_id]:
                    del self._neuron_fibers[neuron_id]
            self._dirty = True
            return True
        return False

    def get_fibers_for_neuron(self, neuron_id: str) -> list[str]:
        """Get all fiber IDs that contain a neuron."""
        return list(self._neuron_fibers.get(neuron_id, set()))

    def delete_fiber(self, fiber_id: str) -> bool:
        """Delete a fiber."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return False
        # Clean up reverse index
        for nid in fiber.get("neuron_ids", []):
            if nid in self._neuron_fibers:
                self._neuron_fibers[nid].discard(fiber_id)
                if not self._neuron_fibers[nid]:
                    del self._neuron_fibers[nid]
        del self._fibers[fiber_id]
        self._dirty = True
        return True

    def remove_neuron_from_all(self, neuron_id: str) -> int:
        """Remove a neuron from all fibers it belongs to. Returns count of fibers affected."""
        fiber_ids = list(self._neuron_fibers.get(neuron_id, set()))
        removed = 0
        for fid in fiber_ids:
            fiber = self._fibers.get(fid)
            if fiber is not None:
                nids = fiber.get("neuron_ids", [])
                if neuron_id in nids:
                    nids.remove(neuron_id)
                    fiber["neuron_ids"] = nids
                    removed += 1
        if neuron_id in self._neuron_fibers:
            del self._neuron_fibers[neuron_id]
        if removed > 0:
            self._dirty = True
        return removed

    def update_fiber(self, fiber_id: str, updates: dict[str, Any]) -> bool:
        """Update fiber fields (name, type, description, metadata)."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return False
        updatable = {"name", "type", "description", "metadata"}
        for key, val in updates.items():
            if key in updatable:
                fiber[key] = val
        self._dirty = True
        return True

    def flush(self) -> None:
        """Save fibers to disk if dirty. Atomic write."""
        if not self._dirty:
            return
        # Snapshot to prevent concurrent mutation during serialization
        fibers_snapshot = {k: dict(v) for k, v in self._fibers.items()}
        data = {
            "version": 1,
            "fibers": fibers_snapshot,
        }
        # msgpack can't serialize sets, convert neuron_fibers not needed (not stored)
        tmp_path = self._path.with_suffix(".fibers.tmp")
        with open(tmp_path, "wb") as f:
            msgpack.pack(data, f, use_bin_type=True)
        tmp_path.replace(self._path)
        self._dirty = False
        logger.debug("Flushed %d fibers to disk", len(self._fibers))

    def close(self) -> None:
        """Flush and close."""
        self.flush()
        self._fibers.clear()
        self._neuron_fibers.clear()
