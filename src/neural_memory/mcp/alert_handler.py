"""MCP alert handler mixin — persistent alert lifecycle management.

Creates alerts from HealthPulse hints, auto-resolves cleared conditions,
surfaces pending counts, and provides list/acknowledge actions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neural_memory.core.alert import Alert, AlertStatus, AlertType
from neural_memory.mcp.maintenance_handler import HealthHint, HealthPulse
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)

# Map HealthHint message prefix → AlertType
_HINT_TO_ALERT_TYPE: dict[str, AlertType] = {
    "High neuron count": AlertType.HIGH_NEURON_COUNT,
    "High fiber count": AlertType.HIGH_FIBER_COUNT,
    "High synapse count": AlertType.HIGH_SYNAPSE_COUNT,
    "Low connectivity": AlertType.LOW_CONNECTIVITY,
    "High orphan ratio": AlertType.HIGH_ORPHAN_RATIO,
    "expired memories": AlertType.EXPIRED_MEMORIES,
    "fibers are stale": AlertType.STALE_FIBERS,
}


def _hint_to_alert_type(hint: HealthHint) -> AlertType | None:
    """Classify a HealthHint into an AlertType by message pattern."""
    msg = hint.message
    for pattern, alert_type in _HINT_TO_ALERT_TYPE.items():
        if pattern in msg:
            return alert_type
    return None


class AlertHandler:
    """MCP mixin for proactive alerts."""

    # Protocol stubs for composed class
    if TYPE_CHECKING:
        from neural_memory.unified_config import UnifiedConfig

        config: UnifiedConfig

        async def get_storage(self) -> SQLiteStorage:
            raise NotImplementedError

    async def _alerts(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_alerts tool calls: list / acknowledge."""
        action = args.get("action", "list")

        if action == "list":
            return await self._alerts_list(args)
        elif action == "acknowledge":
            return await self._alerts_acknowledge(args)
        else:
            return {"error": f"Unknown alerts action: {action}"}

    async def _alerts_list(self, args: dict[str, Any]) -> dict[str, Any]:
        """List active/seen/acknowledged alerts."""
        try:
            storage = await self.get_storage()
            limit = min(args.get("limit", 50), 200)
            alerts = await storage.get_active_alerts(limit=limit)

            # Auto-mark active alerts as seen
            active_ids = [a.id for a in alerts if a.status == AlertStatus.ACTIVE]
            if active_ids:
                await storage.mark_alerts_seen(active_ids)

            return {
                "alerts": [
                    {
                        "id": a.id,
                        "type": a.alert_type.value,
                        "severity": a.severity,
                        "message": a.message,
                        "recommended_action": a.recommended_action,
                        "status": a.status.value,
                        "created_at": a.created_at.isoformat(),
                    }
                    for a in alerts
                ],
                "count": len(alerts),
            }
        except Exception:
            logger.error("Failed to list alerts", exc_info=True)
            return {"error": "Failed to retrieve alerts."}

    async def _alerts_acknowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        """Acknowledge a specific alert by ID."""
        alert_id = args.get("alert_id")
        if not alert_id:
            return {"error": "alert_id is required for acknowledge action."}

        try:
            storage = await self.get_storage()
            updated = await storage.mark_alert_acknowledged(str(alert_id))
            if updated:
                return {"acknowledged": True, "alert_id": str(alert_id)}
            return {"acknowledged": False, "message": "Alert not found or already acknowledged."}
        except Exception:
            logger.error("Failed to acknowledge alert %s", alert_id, exc_info=True)
            return {"error": "Failed to acknowledge alert."}

    async def _surface_pending_alerts(self) -> dict[str, int] | None:
        """Return pending alert count for surfacing in responses.

        Returns None if no pending alerts or on error.
        """
        try:
            storage = await self.get_storage()
            count = await storage.count_pending_alerts()
            if count > 0:
                return {"pending_alerts": count}
        except Exception:
            logger.debug("Alert count check failed", exc_info=True)
        return None

    async def _create_alerts_from_pulse(self, pulse: HealthPulse) -> int:
        """Create alerts from HealthPulse hints. Returns count of new alerts."""
        if not pulse.hints:
            return 0

        storage = await self.get_storage()
        brain_id = storage._get_brain_id()
        created = 0

        for hint in pulse.hints:
            alert_type = _hint_to_alert_type(hint)
            if alert_type is None:
                continue

            alert = Alert(
                id=uuid4().hex[:16],
                brain_id=brain_id,
                alert_type=alert_type,
                severity=hint.severity.value,
                message=hint.message,
                recommended_action=f"Run consolidation with '{hint.recommended_strategy}' strategy."
                if hint.recommended_strategy != "none"
                else "Review and take action as needed.",
                status=AlertStatus.ACTIVE,
                created_at=utcnow(),
            )

            result = await storage.record_alert(alert)
            if result:
                created += 1

        if created:
            logger.info("Created %d alert(s) from health pulse", created)
        return created

    async def _auto_resolve_cleared(self, pulse: HealthPulse) -> int:
        """Resolve alerts whose conditions are no longer present.

        If a HealthPulse has no hints for a type that has active alerts,
        those alerts are auto-resolved.
        """
        storage = await self.get_storage()
        active_alerts = await storage.get_active_alerts(limit=200)
        if not active_alerts:
            return 0

        # Determine which alert types are still flagged
        active_types_in_pulse: set[AlertType] = set()
        for hint in pulse.hints:
            alert_type = _hint_to_alert_type(hint)
            if alert_type:
                active_types_in_pulse.add(alert_type)

        # Find types with active alerts but NOT in current pulse
        types_to_resolve: list[str] = []
        for alert in active_alerts:
            if alert.alert_type not in active_types_in_pulse and alert.status != AlertStatus.ACKNOWLEDGED:
                if alert.alert_type.value not in types_to_resolve:
                    types_to_resolve.append(alert.alert_type.value)

        if not types_to_resolve:
            return 0

        resolved = await storage.resolve_alerts_by_type(types_to_resolve)
        if resolved:
            logger.info("Auto-resolved %d alert(s) for cleared conditions", resolved)
        return resolved
