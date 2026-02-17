"""Proactive alert model for brain health monitoring.

Alerts persist across sessions, track lifecycle state, and support
acknowledgment workflows. Generated from HealthPulse hints during
maintenance checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from neural_memory.utils.timeutils import utcnow


class AlertStatus(StrEnum):
    """Alert lifecycle states."""

    ACTIVE = "active"
    SEEN = "seen"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertType(StrEnum):
    """Categories of brain health alerts."""

    HIGH_NEURON_COUNT = "high_neuron_count"
    HIGH_FIBER_COUNT = "high_fiber_count"
    HIGH_SYNAPSE_COUNT = "high_synapse_count"
    LOW_CONNECTIVITY = "low_connectivity"
    HIGH_ORPHAN_RATIO = "high_orphan_ratio"
    EXPIRED_MEMORIES = "expired_memories"
    STALE_FIBERS = "stale_fibers"


@dataclass(frozen=True)
class Alert:
    """A persistent brain health alert with lifecycle tracking.

    Attributes:
        id: Unique alert ID
        brain_id: Brain this alert belongs to
        alert_type: Category of health issue
        severity: low / medium / high / critical
        message: Human-readable description
        recommended_action: Suggested remediation
        status: Lifecycle state
        created_at: When the alert was first created
        seen_at: When the alert was first included in a response
        acknowledged_at: When the user explicitly acknowledged
        resolved_at: When the underlying issue was cleared
        metadata: Extra context (thresholds, counts, etc.)
    """

    id: str
    brain_id: str
    alert_type: AlertType
    severity: str = "low"
    message: str = ""
    recommended_action: str = ""
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=utcnow)
    seen_at: datetime | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
