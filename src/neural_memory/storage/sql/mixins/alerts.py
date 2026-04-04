"""Alert storage mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from neural_memory.core.alert import Alert, AlertStatus, AlertType
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Cooldown: suppress duplicate alerts of the same type within this window
_DEDUP_COOLDOWN = timedelta(hours=6)


def _safe_parse_dt(val: object) -> datetime | None:
    """Parse a datetime value safely (works with both str and native datetime)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val))


def _row_to_alert(row: dict[str, object]) -> Alert:
    """Convert a row dict to an Alert dataclass."""
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
        created_at=_safe_parse_dt(row["created_at"]) or utcnow(),
        seen_at=_safe_parse_dt(row.get("seen_at")),
        acknowledged_at=_safe_parse_dt(row.get("acknowledged_at")),
        resolved_at=_safe_parse_dt(row.get("resolved_at")),
        metadata=metadata,
    )


class AlertsMixin:
    """Mixin providing alert CRUD operations for SQLStorage."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def record_alert(self, alert: Alert) -> str:
        """Insert a new alert, respecting dedup cooldown.

        Returns the alert ID if inserted, empty string if suppressed.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        # Dedup: check for recent alert of same type
        cutoff = d.serialize_dt(utcnow() - _DEDUP_COOLDOWN)
        row = await d.fetch_one(
            f"""SELECT COUNT(*) as cnt FROM alerts
                WHERE brain_id = {d.ph(1)} AND alert_type = {d.ph(2)}
                  AND created_at > {d.ph(3)}
                  AND status IN ('active', 'seen')""",
            [brain_id, alert.alert_type.value, cutoff],
        )
        if row and row.get("cnt", 0) > 0:
            return ""  # Suppressed by cooldown

        await d.execute(
            f"""INSERT INTO alerts
                (id, brain_id, alert_type, severity, message,
                 recommended_action, status, created_at, metadata)
                VALUES ({d.phs(9)})""",
            [
                alert.id,
                brain_id,
                alert.alert_type.value,
                alert.severity,
                alert.message,
                alert.recommended_action,
                alert.status.value,
                d.serialize_dt(alert.created_at),
                json.dumps(alert.metadata),
            ],
        )
        return alert.id

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        """Get active/seen/acknowledged alerts (not resolved)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)

        rows = await d.fetch_all(
            f"""SELECT * FROM alerts
                WHERE brain_id = {d.ph(1)} AND status IN ('active', 'seen', 'acknowledged')
                ORDER BY
                  CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                  END,
                  created_at DESC
                LIMIT {d.ph(2)}""",
            [brain_id, safe_limit],
        )
        return [_row_to_alert(r) for r in rows]

    async def count_pending_alerts(self) -> int:
        """Count active + seen alerts (not acknowledged or resolved)."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT COUNT(*) as cnt FROM alerts
                WHERE brain_id = {d.ph(1)} AND status IN ('active', 'seen')""",
            [brain_id],
        )
        return row.get("cnt", 0) if row else 0

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        """Mark alerts as seen. Returns count of updated rows."""
        if not alert_ids:
            return 0
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        in_fragment, in_params = d.in_clause(3, alert_ids)
        return await d.execute_count(
            f"""UPDATE alerts SET status = 'seen', seen_at = {d.ph(1)}
                WHERE brain_id = {d.ph(2)} AND id {in_fragment}
                  AND status = 'active'""",
            [now, brain_id, *in_params],
        )

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        """Mark a single alert as acknowledged. Returns True if updated."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        count = await d.execute_count(
            f"""UPDATE alerts SET status = 'acknowledged', acknowledged_at = {d.ph(1)}
                WHERE brain_id = {d.ph(2)} AND id = {d.ph(3)}
                  AND status IN ('active', 'seen')""",
            [now, brain_id, alert_id],
        )
        return count > 0

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        """Resolve all active/seen alerts of given types. Returns count."""
        if not alert_types:
            return 0
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        in_fragment, in_params = d.in_clause(3, alert_types)
        return await d.execute_count(
            f"""UPDATE alerts SET status = 'resolved', resolved_at = {d.ph(1)}
                WHERE brain_id = {d.ph(2)} AND alert_type {in_fragment}
                  AND status IN ('active', 'seen')""",
            [now, brain_id, *in_params],
        )

    async def get_alert(self, alert_id: str) -> Alert | None:
        """Get a single alert by ID."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM alerts WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}",
            [brain_id, alert_id],
        )
        if not row:
            return None
        return _row_to_alert(row)
