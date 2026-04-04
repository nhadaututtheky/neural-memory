"""InfinityDB Change Log + Device Registry + Merkle mixin.

In-memory implementations for sync operations.
All volatile (session-scoped, not persisted).
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.brain_versioning import BrainVersion


class InfinityDBSyncMixin:
    """Mixin providing change log, device registry, and Merkle ops.

    Composing class must call ``_init_sync_stores()`` in __init__ and
    provide ``db``, ``find_neurons``, ``get_synapses``, ``find_fibers``.
    """

    if TYPE_CHECKING:

        @property
        def db(self) -> Any:
            raise NotImplementedError

        _current_brain_id: str | None

    def _init_sync_stores(self) -> None:
        self._change_log: list[dict[str, Any]] = []
        self._change_sequence: int = 0
        self._devices: dict[str, dict[str, Any]] = {}
        self._merkle_cache: dict[str, dict[str, str]] = {}
        self._versions: dict[str, list[tuple[Any, str]]] = {}  # brain_id -> [(BrainVersion, json)]

    # ========== Change Log ==========

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        self._change_sequence += 1
        entry = {
            "id": self._change_sequence,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "device_id": device_id,
            "payload": payload,
            "changed_at": utcnow(),
            "synced": False,
        }
        self._change_log.append(entry)
        return self._change_sequence

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[Any]:
        results = [e for e in self._change_log if e["id"] > sequence]
        results.sort(key=lambda e: e["id"])
        return results[:limit]

    async def get_unsynced_changes(self, limit: int = 1000) -> list[Any]:
        results = [e for e in self._change_log if not e["synced"]]
        results.sort(key=lambda e: e["id"])
        return results[:limit]

    async def mark_synced(self, up_to_sequence: int) -> int:
        count = 0
        for entry in self._change_log:
            if entry["id"] <= up_to_sequence and not entry["synced"]:
                entry["synced"] = True
                count += 1
        return count

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        cutoff = utcnow() - timedelta(days=older_than_days)
        before = len(self._change_log)
        self._change_log = [
            e for e in self._change_log if not (e["synced"] and e["changed_at"] < cutoff)
        ]
        return before - len(self._change_log)

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        existing_ids = {e["entity_id"] for e in self._change_log}
        counts = {"neurons": 0, "synapses": 0, "fibers": 0}

        neurons = await self.find_neurons(limit=50000)  # type: ignore[attr-defined]
        for n in neurons:
            if n.id not in existing_ids:
                await self.record_change("neuron", n.id, "insert", device_id)
                counts["neurons"] += 1

        fibers = await self.find_fibers(limit=50000)  # type: ignore[attr-defined]
        for f in fibers:
            if f.id not in existing_ids:
                await self.record_change("fiber", f.id, "insert", device_id)
                counts["fibers"] += 1

        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        total = len(self._change_log)
        synced = sum(1 for e in self._change_log if e["synced"])
        return {
            "total": total,
            "pending": total - synced,
            "synced": synced,
            "last_sequence": self._change_sequence,
        }

    # ========== Device Registry ==========

    async def register_device(self, device_id: str, device_name: str = "") -> Any:
        now = utcnow()
        if device_id in self._devices:
            self._devices[device_id]["device_name"] = (
                device_name or self._devices[device_id]["device_name"]
            )
            return self._devices[device_id]
        record = {
            "device_id": device_id,
            "device_name": device_name,
            "brain_id": self._current_brain_id or "default",
            "registered_at": now,
            "last_sync_at": None,
            "last_sync_sequence": 0,
        }
        self._devices[device_id] = record
        return record

    async def get_device(self, device_id: str) -> Any | None:
        return self._devices.get(device_id)

    async def list_devices(self) -> list[Any]:
        devices = list(self._devices.values())
        devices.sort(key=lambda d: d.get("registered_at", utcnow()))
        return devices

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        device = self._devices.get(device_id)
        if device is not None:
            device["last_sync_at"] = utcnow()
            device["last_sync_sequence"] = last_sync_sequence

    async def remove_device(self, device_id: str) -> bool:
        return self._devices.pop(device_id, None) is not None

    # ========== Merkle Hash ==========

    async def compute_merkle_root(self, entity_type: str, *, is_pro: bool = False) -> str | None:
        tree = await self.get_merkle_tree(entity_type, is_pro=is_pro)
        if not tree:
            return None
        combined = "".join(sorted(tree.values()))
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get_merkle_tree(self, entity_type: str, *, is_pro: bool = False) -> dict[str, str]:
        return dict(self._merkle_cache.get(entity_type, {}))

    async def invalidate_merkle_prefix(
        self,
        entity_type: str,
        entity_id: str,
        *,
        is_pro: bool = False,
    ) -> None:
        prefix = entity_id[:2] if entity_id else ""
        cache = self._merkle_cache.get(entity_type, {})
        cache.pop(prefix, None)

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        all_roots: list[str] = []
        for et in ("neuron", "synapse", "fiber"):
            root = await self.compute_merkle_root(et, is_pro=is_pro)
            if root:
                all_roots.append(root)
        if not all_roots:
            return None
        return hashlib.sha256("".join(sorted(all_roots)).encode()).hexdigest()

    async def get_bucket_entity_ids(
        self,
        entity_type: str,
        prefix: str,
        *,
        is_pro: bool = False,
    ) -> list[str]:
        if entity_type == "neuron":
            neurons = await self.find_neurons(limit=50000)  # type: ignore[attr-defined]
            return [n.id for n in neurons if n.id.startswith(prefix)]
        if entity_type == "fiber":
            fibers = await self.find_fibers(limit=50000)  # type: ignore[attr-defined]
            return [f.id for f in fibers if f.id.startswith(prefix)]
        return []

    # ========== Version Operations ==========

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        if brain_id not in self._versions:
            self._versions[brain_id] = []
        # Replace if same version_id exists
        self._versions[brain_id] = [
            (v, s) for v, s in self._versions[brain_id] if v.id != version.id
        ]
        self._versions[brain_id].append((version, snapshot_json))

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        for v, snap in self._versions.get(brain_id, []):
            if v.id == version_id:
                return (v, snap)
        return None

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        entries = self._versions.get(brain_id, [])
        # Most recent first
        sorted_entries = sorted(entries, key=lambda e: e[0].version_number, reverse=True)
        return [v for v, _ in sorted_entries[:limit]]

    async def get_next_version_number(self, brain_id: str) -> int:
        entries = self._versions.get(brain_id, [])
        if not entries:
            return 1
        return int(max(v.version_number for v, _ in entries)) + 1

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        entries = self._versions.get(brain_id, [])
        before = len(entries)
        self._versions[brain_id] = [(v, s) for v, s in entries if v.id != version_id]
        return len(self._versions[brain_id]) < before
