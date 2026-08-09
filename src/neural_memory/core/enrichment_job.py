"""Enrichment outbox job model — durable async work after lean write ack."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


class EnrichmentKind(StrEnum):
    """Kinds of post-ack enrichment work."""

    EMBED = "embed"
    LINK = "link"
    ENRICH = "enrich"
    ANALYTICS = "analytics"


class EnrichmentStatus(StrEnum):
    """Lifecycle of an enrichment outbox job."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    DEAD = "dead"


@dataclass(frozen=True)
class EnrichmentJob:
    """One durable enrichment unit of work.

    Unique on ``(brain_id, idempotency_key)`` so retries and restarts
    never double-apply effects.
    """

    id: str
    brain_id: str
    kind: EnrichmentKind
    entity_id: str
    idempotency_key: str
    payload: dict[str, Any]
    status: EnrichmentStatus
    attempts: int
    available_at: datetime
    created_at: datetime
    updated_at: datetime
    last_error: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    @classmethod
    def create(
        cls,
        *,
        brain_id: str,
        kind: EnrichmentKind,
        entity_id: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        available_at: datetime | None = None,
    ) -> EnrichmentJob:
        """Create a new pending enrichment job."""
        now = utcnow()
        return cls(
            id=str(uuid4()),
            brain_id=brain_id,
            kind=kind,
            entity_id=entity_id,
            idempotency_key=idempotency_key,
            payload=dict(payload or {}),
            status=EnrichmentStatus.PENDING,
            attempts=0,
            available_at=available_at or now,
            created_at=now,
            updated_at=now,
        )


@dataclass(frozen=True)
class EnrichmentReport:
    """Result of a worker batch pass."""

    claimed: int = 0
    completed: int = 0
    failed: int = 0
    dead_lettered: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def processed(self) -> int:
        return self.completed + self.failed + self.dead_lettered
