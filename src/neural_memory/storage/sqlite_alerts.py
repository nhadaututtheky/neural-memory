"""SQLite alerts storage mixin."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from neural_memory.core.alert import Alert, AlertStatus, AlertType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cooldown: suppress duplicate alerts of the same type within this window
_DEDUP_COOLDOWN = timedelta(hours=6)


def _row_to_alert(row: dict[str, object]) -> Alert:
    """Convert a sqlite Row to an Alert dataclass."""
    from datetime import datetime

    def _parse_dt(val: object) -> datetime | None:
        if val is None:
            return None
        return datetime.fromisoformat(str(val))

    raw_metadata = row.get("metadata", "{}")
    metadata = json.loads(str(raw_metadata)) if raw_metadata else {}

    return Alert(
        id=str(row["id"]),
        brain_id=str(row["brain_id"]),
        alert_type=AlertType(str(row["alert_type"])),
        severity=str(row.get("severity", "low")),
        message=str(row.get("message", "")),
        recommended_action=str(row.get("recommended_action", "")),
        status=AlertStatus(str(row["status"])),
        created_at=_parse_dt(row["created_at"]) or utcnow(),
        seen_at=_parse_dt(row.get("seen_at")),
        acknowledged_at=_parse_dt(row.get("acknowledged_at")),
        resolved_at=_parse_dt(row.get("resolved_at")),
        metadata=metadata,
    )


class SQLiteAlertsMixin:
    """Mixin providing alert CRUD operations for SQLiteStorage."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def record_alert(self, alert: Alert) -> str:
        """Insert a new alert, respecting dedup cooldown.

        Returns the alert ID if inserted, empty string if suppressed.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Dedup: check for recent alert of same type
        cutoff = (utcnow() - _DEDUP_COOLDOWN).isoformat()
        cursor = await conn.execute(
            """SELECT COUNT(*) FROM alerts
               WHERE brain_id = ? AND alert_type = ? AND created_at > ?
                 AND status IN ('active', 'seen')""",
            (brain_id, alert.alert_type.value, cutoff),
        )
        row = await cursor.fetchone()
        if row and row[0] > 0:
            return ""  # Suppressed by cooldown

        await conn.execute(
            """INSERT INTO alerts
               (id, brain_id, alert_type, severity, message,
                recommended_action, status, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                alert.id,
                brain_id,
                alert.alert_type.value,
                alert.severity,
                alert.message,
                alert.recommended_action,
                alert.status.value,
                alert.created_at.isoformat(),
                json.dumps(alert.metadata),
            ),
        )
        await conn.commit()
        return alert.id

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        """Get active/seen/acknowledged alerts (not resolved)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)

        cursor = await conn.execute(
            """SELECT * FROM alerts
               WHERE brain_id = ? AND status IN ('active', 'seen', 'acknowledged')
               ORDER BY
                 CASE severity
                   WHEN 'critical' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   ELSE 3
                 END,
                 created_at DESC
               LIMIT ?""",
            (brain_id, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_alert(dict(zip(col_names, r, strict=False))) for r in rows]

    async def count_pending_alerts(self) -> int:
        """Count active + seen alerts (not acknowledged or resolved)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT COUNT(*) FROM alerts
               WHERE brain_id = ? AND status IN ('active', 'seen')""",
            (brain_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        """Mark alerts as seen. Returns count of updated rows."""
        if not alert_ids:
            return 0
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        placeholders = ", ".join("?" for _ in alert_ids)
        cursor = await conn.execute(
            f"""UPDATE alerts SET status = 'seen', seen_at = ?
                WHERE brain_id = ? AND id IN ({placeholders})
                  AND status = 'active'""",
            (now, brain_id, *alert_ids),
        )
        await conn.commit()
        return cursor.rowcount

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        """Mark a single alert as acknowledged. Returns True if updated."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """UPDATE alerts SET status = 'acknowledged', acknowledged_at = ?
               WHERE brain_id = ? AND id = ?
                 AND status IN ('active', 'seen')""",
            (now, brain_id, alert_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        """Resolve all active/seen alerts of given types. Returns count."""
        if not alert_types:
            return 0
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        placeholders = ", ".join("?" for _ in alert_types)
        cursor = await conn.execute(
            f"""UPDATE alerts SET status = 'resolved', resolved_at = ?
                WHERE brain_id = ? AND alert_type IN ({placeholders})
                  AND status IN ('active', 'seen')""",
            (now, brain_id, *alert_types),
        )
        await conn.commit()
        return cursor.rowcount

    async def get_alert(self, alert_id: str) -> Alert | None:
        """Get a single alert by ID."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM alerts WHERE brain_id = ? AND id = ?",
            (brain_id, alert_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        col_names = [d[0] for d in (cursor.description or [])]
        return _row_to_alert(dict(zip(col_names, row, strict=False)))
