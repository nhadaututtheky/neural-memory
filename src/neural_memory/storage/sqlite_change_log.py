"""SQLite change log operations mixin for multi-device sync."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

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


class SQLiteChangeLogMixin:
    """Mixin providing change log CRUD operations for incremental sync."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SQLiteStorage at runtime.
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

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
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                brain_id,
                entity_type,
                entity_id,
                operation,
                device_id,
                now,
                json.dumps(payload or {}),
            ),
        )
        await conn.commit()
        return cursor.lastrowid or 0

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[ChangeEntry]:
        """Get changes after a given sequence number, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM change_log WHERE brain_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (brain_id, sequence, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_change_entry(dict(zip(col_names, r, strict=False))) for r in rows]

    async def get_unsynced_changes(self, limit: int = 1000) -> list[ChangeEntry]:
        """Get all unsynced changes, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM change_log WHERE brain_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
            (brain_id, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_change_entry(dict(zip(col_names, r, strict=False))) for r in rows]

    async def mark_synced(self, up_to_sequence: int) -> int:
        """Mark all changes up to a sequence number as synced. Returns count marked."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "UPDATE change_log SET synced = 1 WHERE brain_id = ? AND id <= ? AND synced = 0",
            (brain_id, up_to_sequence),
        )
        await conn.commit()
        return cursor.rowcount

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        """Delete synced changes older than N days. Returns count pruned."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cutoff = (utcnow() - timedelta(days=older_than_days)).isoformat()

        cursor = await conn.execute(
            "DELETE FROM change_log WHERE brain_id = ? AND synced = 1 AND changed_at < ?",
            (brain_id, cutoff),
        )
        await conn.commit()
        return cursor.rowcount

    async def get_change_log_stats(self) -> dict[str, Any]:
        """Get change log statistics."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN synced = 1 THEN 1 ELSE 0 END) as synced,
                MAX(id) as last_sequence
               FROM change_log WHERE brain_id = ?""",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return {"total": 0, "pending": 0, "synced": 0, "last_sequence": 0}
            col_names = [d[0] for d in (cursor.description or [])]
            data = dict(zip(col_names, row, strict=False))
        return {
            "total": data["total"] or 0,
            "pending": data["pending"] or 0,
            "synced": data["synced"] or 0,
            "last_sequence": data["last_sequence"] or 0,
        }


def _row_to_change_entry(row: dict[str, Any]) -> ChangeEntry:
    """Convert a database row dict to a ChangeEntry."""
    return ChangeEntry(
        id=int(row["id"]),
        brain_id=str(row["brain_id"]),
        entity_type=str(row["entity_type"]),
        entity_id=str(row["entity_id"]),
        operation=str(row["operation"]),
        device_id=str(row["device_id"] or ""),
        changed_at=datetime.fromisoformat(str(row["changed_at"])),
        payload=json.loads(str(row["payload"])) if row["payload"] else {},
        synced=bool(row["synced"]),
    )
