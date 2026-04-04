"""Device registry operations mixin for multi-device sync — dialect-agnostic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

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


class DevicesMixin:
    """Mixin providing device registry operations for multi-device sync."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_device(self, device_id: str, device_name: str = "") -> DeviceRecord:
        """Register a device for the current brain (upsert)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = utcnow()

        upsert_sql = d.upsert_sql(
            "devices",
            ["device_id", "brain_id", "device_name", "last_sync_sequence", "registered_at"],
            ["brain_id", "device_id"],
            ["device_name"],
        )
        await d.execute(
            upsert_sql,
            [device_id, brain_id, device_name, 0, d.serialize_dt(now)],
        )

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
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM devices WHERE brain_id = {d.ph(1)} AND device_id = {d.ph(2)}",
            [brain_id, device_id],
        )
        if row is None:
            return None
        return _row_to_device(row)

    async def list_devices(self) -> list[DeviceRecord]:
        """List all registered devices for the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT * FROM devices WHERE brain_id = {d.ph(1)} ORDER BY registered_at ASC",
            [brain_id],
        )
        return [_row_to_device(r) for r in rows]

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        """Update the last sync timestamp and sequence for a device."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        await d.execute(
            f"""UPDATE devices SET last_sync_at = {d.ph(1)}, last_sync_sequence = {d.ph(2)}
               WHERE brain_id = {d.ph(3)} AND device_id = {d.ph(4)}""",
            [now, last_sync_sequence, brain_id, device_id],
        )

    async def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry. Returns True if deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM devices WHERE brain_id = {d.ph(1)} AND device_id = {d.ph(2)}",
            [brain_id, device_id],
        )
        return count > 0


def _row_to_device(row: dict[str, Any]) -> DeviceRecord:
    """Convert a database row dict to a DeviceRecord."""

    def _safe_dt(val: object) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    return DeviceRecord(
        device_id=str(row["device_id"]),
        brain_id=str(row["brain_id"]),
        device_name=str(row["device_name"] or ""),
        last_sync_at=_safe_dt(row["last_sync_at"]),
        last_sync_sequence=int(row["last_sync_sequence"] or 0),
        registered_at=_safe_dt(row["registered_at"]) or utcnow(),
    )
