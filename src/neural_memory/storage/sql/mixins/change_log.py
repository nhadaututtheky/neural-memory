"""Change log operations mixin for multi-device sync — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangeEntry:
    """A single change log entry."""

    id: int  # Auto-incremented sequence number
    brain_id: str
    entity_type: str  # "neuron", "synapse", "fiber"
    entity_id: str
    operation: str  # "insert", "update", "delete"
    device_id: str
    changed_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    synced: bool = False


class ChangeLogMixin:
    """Mixin providing change log CRUD operations for incremental sync."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a change to the log. Returns the sequence number (id)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        await d.execute(
            f"""INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               VALUES ({d.phs(8)})""",
            [
                brain_id,
                entity_type,
                entity_id,
                operation,
                device_id,
                now,
                json.dumps(payload or {}),
                0,
            ],
        )
        # Get the inserted id using a portable query
        row = await d.fetch_one(
            f"""SELECT MAX(id) as id FROM change_log
               WHERE brain_id = {d.ph(1)} AND entity_type = {d.ph(2)}
                 AND entity_id = {d.ph(3)}""",
            [brain_id, entity_type, entity_id],
        )
        return int(row.get("id", 0)) if row else 0

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[ChangeEntry]:
        """Get changes after a given sequence number, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT * FROM change_log WHERE brain_id = {d.ph(1)} AND id > {d.ph(2)} ORDER BY id ASC LIMIT {d.ph(3)}",
            [brain_id, sequence, safe_limit],
        )
        return [_row_to_change_entry(r) for r in rows]

    async def get_unsynced_changes(self, limit: int = 1000) -> list[ChangeEntry]:
        """Get all unsynced changes, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT * FROM change_log WHERE brain_id = {d.ph(1)} AND synced = 0 ORDER BY id ASC LIMIT {d.ph(2)}",
            [brain_id, safe_limit],
        )
        return [_row_to_change_entry(r) for r in rows]

    async def mark_synced(self, up_to_sequence: int) -> int:
        """Mark all changes up to a sequence number as synced. Returns count marked."""
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"UPDATE change_log SET synced = 1 WHERE brain_id = {d.ph(1)} AND id <= {d.ph(2)} AND synced = 0",
            [brain_id, up_to_sequence],
        )

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        """Delete synced changes older than N days. Returns count pruned."""
        d = self._dialect
        brain_id = self._get_brain_id()
        cutoff = d.serialize_dt(utcnow() - timedelta(days=older_than_days))

        return await d.execute_count(
            f"DELETE FROM change_log WHERE brain_id = {d.ph(1)} AND synced = 1 AND changed_at < {d.ph(2)}",
            [brain_id, cutoff],
        )

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        """Seed the change log with all existing entities as 'insert' entries.

        This enables initial sync for brains created before sync was enabled.
        Existing change_log entries are preserved — only entities NOT already
        tracked in the log are added.

        Returns:
            Dict with counts: neurons, synapses, fibers seeded.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())
        counts: dict[str, int] = {"neurons": 0, "synapses": 0, "fibers": 0}

        # Seed neurons not already in change_log
        # Build payload in Python per-row instead of json_object() (not portable)
        neuron_rows = await d.fetch_all(
            f"""SELECT n.id, n.type, n.content, n.metadata, n.content_hash, n.created_at
               FROM neurons n
               WHERE n.brain_id = {d.ph(1)}
                 AND COALESCE(n.ephemeral, 0) = 0
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = {d.ph(2)} AND cl.entity_type = 'neuron' AND cl.entity_id = n.id
                 )""",
            [brain_id, brain_id],
        )
        for nr in neuron_rows:
            payload_json = json.dumps(
                {
                    "id": nr["id"],
                    "type": nr["type"],
                    "content": nr["content"],
                    "metadata": nr["metadata"],
                    "content_hash": nr["content_hash"],
                    "created_at": str(nr["created_at"]),
                }
            )
            await d.execute(
                f"""INSERT INTO change_log
                   (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
                   VALUES ({d.phs(8)})""",
                [brain_id, "neuron", nr["id"], "insert", device_id, now, payload_json, 0],
            )
            counts["neurons"] += 1

        # Seed synapses not already in change_log
        synapse_rows = await d.fetch_all(
            f"""SELECT s.id, s.source_id, s.target_id, s.type, s.weight, s.direction,
                      s.metadata, s.reinforced_count, s.last_activated, s.created_at
               FROM synapses s
               WHERE s.brain_id = {d.ph(1)}
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = {d.ph(2)} AND cl.entity_type = 'synapse' AND cl.entity_id = s.id
                 )""",
            [brain_id, brain_id],
        )
        for sr in synapse_rows:
            payload_json = json.dumps(
                {
                    "id": sr["id"],
                    "source_id": sr["source_id"],
                    "target_id": sr["target_id"],
                    "type": sr["type"],
                    "weight": sr["weight"],
                    "direction": sr["direction"],
                    "metadata": sr["metadata"],
                    "reinforced_count": sr["reinforced_count"],
                    "last_activated": str(sr["last_activated"]) if sr["last_activated"] else None,
                    "created_at": str(sr["created_at"]),
                }
            )
            await d.execute(
                f"""INSERT INTO change_log
                   (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
                   VALUES ({d.phs(8)})""",
                [brain_id, "synapse", sr["id"], "insert", device_id, now, payload_json, 0],
            )
            counts["synapses"] += 1

        # Seed fibers not already in change_log
        fiber_rows = await d.fetch_all(
            f"""SELECT f.id, f.neuron_ids, f.synapse_ids, f.anchor_neuron_id, f.pathway,
                      f.conductivity, f.last_conducted, f.time_start, f.time_end,
                      f.coherence, f.salience, f.frequency, f.summary,
                      f.auto_tags, f.agent_tags, f.metadata, f.compression_tier, f.created_at
               FROM fibers f
               WHERE f.brain_id = {d.ph(1)}
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = {d.ph(2)} AND cl.entity_type = 'fiber' AND cl.entity_id = f.id
                 )""",
            [brain_id, brain_id],
        )
        for fr in fiber_rows:
            payload_json = json.dumps(
                {
                    "id": fr["id"],
                    "neuron_ids": fr["neuron_ids"],
                    "synapse_ids": fr["synapse_ids"],
                    "anchor_neuron_id": fr["anchor_neuron_id"],
                    "pathway": fr["pathway"],
                    "conductivity": fr["conductivity"],
                    "last_conducted": str(fr["last_conducted"]) if fr["last_conducted"] else None,
                    "time_start": str(fr["time_start"]) if fr["time_start"] else None,
                    "time_end": str(fr["time_end"]) if fr["time_end"] else None,
                    "coherence": fr["coherence"],
                    "salience": fr["salience"],
                    "frequency": fr["frequency"],
                    "summary": fr["summary"],
                    "auto_tags": fr["auto_tags"],
                    "agent_tags": fr["agent_tags"],
                    "metadata": fr["metadata"],
                    "compression_tier": fr["compression_tier"],
                    "created_at": str(fr["created_at"]),
                }
            )
            await d.execute(
                f"""INSERT INTO change_log
                   (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
                   VALUES ({d.phs(8)})""",
                [brain_id, "fiber", fr["id"], "insert", device_id, now, payload_json, 0],
            )
            counts["fibers"] += 1

        total = counts["neurons"] + counts["synapses"] + counts["fibers"]
        logger.info(
            "Seeded change log for brain %s: %d neurons, %d synapses, %d fibers (%d total)",
            brain_id,
            counts["neurons"],
            counts["synapses"],
            counts["fibers"],
            total,
        )
        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        """Get change log statistics."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT
                COUNT(*) as total,
                SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN synced = 1 THEN 1 ELSE 0 END) as synced,
                MAX(id) as last_sequence
               FROM change_log WHERE brain_id = {d.ph(1)}""",
            [brain_id],
        )
        if row is None:
            return {"total": 0, "pending": 0, "synced": 0, "last_sequence": 0}
        return {
            "total": row.get("total") or 0,
            "pending": row.get("pending") or 0,
            "synced": row.get("synced") or 0,
            "last_sequence": row.get("last_sequence") or 0,
        }


def _row_to_change_entry(row: dict[str, Any]) -> ChangeEntry:
    """Convert a database row dict to a ChangeEntry."""
    changed_at_val = row["changed_at"]
    if isinstance(changed_at_val, datetime):
        changed_at = changed_at_val
    else:
        changed_at = datetime.fromisoformat(str(changed_at_val))

    return ChangeEntry(
        id=int(row["id"]),
        brain_id=str(row["brain_id"]),
        entity_type=str(row["entity_type"]),
        entity_id=str(row["entity_id"]),
        operation=str(row["operation"]),
        device_id=str(row["device_id"] or ""),
        changed_at=changed_at,
        payload=json.loads(str(row["payload"])) if row["payload"] else {},
        synced=bool(row["synced"]),
    )
