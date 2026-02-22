"""SQLite device registry operations mixin for multi-device sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceRecord:
    """A registered device for a brain."""

    device_id: str
    brain_id: str
    device_name: str
    last_sync_at: datetime | None
    last_sync_sequence: int
    registered_at: datetime


class SQLiteDevicesMixin:
    """Mixin providing device registry operations for multi-device sync."""

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

    async def register_device(self, device_id: str, device_name: str = "") -> DeviceRecord:
        """Register a device for the current brain (upsert)."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow()

        await conn.execute(
            """INSERT INTO devices (device_id, brain_id, device_name, last_sync_sequence, registered_at)
               VALUES (?, ?, ?, 0, ?)
               ON CONFLICT(brain_id, device_id) DO UPDATE SET device_name = ?""",
            (device_id, brain_id, device_name, now.isoformat(), device_name),
        )
        await conn.commit()

        return DeviceRecord(
            device_id=device_id,
            brain_id=brain_id,
            device_name=device_name,
            last_sync_at=None,
            last_sync_sequence=0,
            registered_at=now,
        )

    async def get_device(self, device_id: str) -> DeviceRecord | None:
        """Get device info for a specific device."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM devices WHERE brain_id = ? AND device_id = ?",
            (brain_id, device_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        col_names = [d[0] for d in (cursor.description or [])]
        return _row_to_device(dict(zip(col_names, row, strict=False)))

    async def list_devices(self) -> list[DeviceRecord]:
        """List all registered devices for the current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM devices WHERE brain_id = ? ORDER BY registered_at ASC",
            (brain_id,),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_device(dict(zip(col_names, r, strict=False))) for r in rows]

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        """Update the last sync timestamp and sequence for a device."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        await conn.execute(
            """UPDATE devices SET last_sync_at = ?, last_sync_sequence = ?
               WHERE brain_id = ? AND device_id = ?""",
            (now, last_sync_sequence, brain_id, device_id),
        )
        await conn.commit()

    async def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry. Returns True if deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM devices WHERE brain_id = ? AND device_id = ?",
            (brain_id, device_id),
        )
        await conn.commit()
        return cursor.rowcount > 0


def _row_to_device(row: dict[str, Any]) -> DeviceRecord:
    """Convert a database row dict to a DeviceRecord."""
    return DeviceRecord(
        device_id=str(row["device_id"]),
        brain_id=str(row["brain_id"]),
        device_name=str(row["device_name"] or ""),
        last_sync_at=datetime.fromisoformat(str(row["last_sync_at"])) if row["last_sync_at"] else None,
        last_sync_sequence=int(row["last_sync_sequence"] or 0),
        registered_at=datetime.fromisoformat(str(row["registered_at"])),
    )
