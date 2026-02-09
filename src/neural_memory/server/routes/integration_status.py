"""Integration status API — activity metrics and log for the dashboard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from neural_memory.server.dependencies import get_storage
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ── Models ────────────────────────────────────────────────


class IntegrationMetrics(BaseModel):
    """Per-source daily metrics."""

    integration_id: str
    memories_today: int = 0
    recalls_today: int = 0
    contexts_today: int = 0
    total_today: int = 0
    last_call_at: str | None = None
    error_count: int = 0


class ActivityLogEntry(BaseModel):
    """Single activity log entry."""

    id: str
    action_type: str
    action_context: str = ""
    source: str = "mcp"
    tags: list[str] = Field(default_factory=list)
    created_at: str
    fiber_id: str | None = None


class ActivityResponse(BaseModel):
    """Response for the activity endpoint."""

    metrics: list[IntegrationMetrics] = Field(default_factory=list)
    activity: list[ActivityLogEntry] = Field(default_factory=list)
    total_events: int = 0


# ── Helpers ───────────────────────────────────────────────


def _extract_source(session_id: str | None) -> str:
    """Extract source name from session_id prefix."""
    if not session_id:
        return "mcp"
    for prefix in ("openclaw-", "nanobot-", "mcp-"):
        if session_id.startswith(prefix):
            return prefix.rstrip("-")
    return "mcp"


def _compute_metrics(
    events: list,
) -> list[IntegrationMetrics]:
    """Group action events by source and compute daily metrics."""
    from collections import defaultdict

    by_source: dict[str, dict] = defaultdict(
        lambda: {
            "memories": 0,
            "recalls": 0,
            "contexts": 0,
            "total": 0,
            "last_at": None,
            "errors": 0,
        }
    )

    for ev in events:
        source = _extract_source(ev.session_id)
        bucket = by_source[source]
        bucket["total"] += 1

        if ev.action_type == "remember":
            bucket["memories"] += 1
        elif ev.action_type == "recall":
            bucket["recalls"] += 1
        elif ev.action_type == "context":
            bucket["contexts"] += 1

        ts = ev.created_at.isoformat() if ev.created_at else None
        if ts and (bucket["last_at"] is None or ts > bucket["last_at"]):
            bucket["last_at"] = ts

    return [
        IntegrationMetrics(
            integration_id=source,
            memories_today=data["memories"],
            recalls_today=data["recalls"],
            contexts_today=data["contexts"],
            total_today=data["total"],
            last_call_at=data["last_at"],
            error_count=data["errors"],
        )
        for source, data in by_source.items()
    ]


# ── Endpoint ──────────────────────────────────────────────


@router.get(
    "/activity",
    response_model=ActivityResponse,
    summary="Get integration activity metrics and log",
)
async def get_activity(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: int = Query(default=50, ge=1, le=500),
    since: str | None = Query(default=None),
) -> ActivityResponse:
    """Get daily integration metrics and recent activity log."""
    today_midnight = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    try:
        # All events today for metrics
        today_events = await storage.get_action_sequences(
            since=today_midnight, limit=1000
        )
    except Exception:
        logger.debug("Failed to fetch today's action events", exc_info=True)
        today_events = []

    metrics = _compute_metrics(today_events)

    # Activity log (recent events, optionally filtered by since)
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            pass

    try:
        log_events = await storage.get_action_sequences(
            since=since_dt, limit=limit
        )
        # Reverse to get newest first
        log_events = list(reversed(log_events))
    except Exception:
        logger.debug("Failed to fetch activity log", exc_info=True)
        log_events = []

    activity = [
        ActivityLogEntry(
            id=ev.id,
            action_type=ev.action_type,
            action_context=ev.action_context,
            source=_extract_source(ev.session_id),
            tags=list(ev.tags),
            created_at=ev.created_at.isoformat() if ev.created_at else "",
            fiber_id=ev.fiber_id,
        )
        for ev in log_events
    ]

    return ActivityResponse(
        metrics=metrics,
        activity=activity,
        total_events=len(today_events),
    )
