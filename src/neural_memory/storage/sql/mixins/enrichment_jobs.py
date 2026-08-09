"""Enrichment outbox mixin — durable async jobs after lean write ack.

Separate from ``change_log.synced`` (sync hub). Jobs are claimed with a
lease, retried with exponential backoff, and dead-lettered after max attempts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from neural_memory.core.enrichment_job import (
    EnrichmentJob,
    EnrichmentKind,
    EnrichmentStatus,
)
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_MAX_CLAIM = 50
_MAX_ATTEMPTS_DEFAULT = 5
_BACKOFF_BASE_SEC = 2.0
_BACKOFF_CAP_SEC = 300.0


def _safe_parse_dt(val: object) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        # Normalize to naive UTC for SQLite storage convention
        if val.tzinfo is not None:
            return val.replace(tzinfo=None)
        return val
    try:
        parsed = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return None


def _row_to_job(row: dict[str, Any]) -> EnrichmentJob:
    raw_payload = row.get("payload", "{}")
    try:
        payload = json.loads(str(raw_payload)) if raw_payload else {}
    except (TypeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    return EnrichmentJob(
        id=str(row["id"]),
        brain_id=str(row["brain_id"]),
        kind=EnrichmentKind(str(row["kind"])),
        entity_id=str(row["entity_id"]),
        idempotency_key=str(row["idempotency_key"]),
        payload=payload,
        status=EnrichmentStatus(str(row["status"])),
        attempts=int(row.get("attempts") or 0),
        available_at=_safe_parse_dt(row.get("available_at")) or utcnow(),
        created_at=_safe_parse_dt(row.get("created_at")) or utcnow(),
        updated_at=_safe_parse_dt(row.get("updated_at")) or utcnow(),
        last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
        lease_owner=str(row["lease_owner"]) if row.get("lease_owner") is not None else None,
        lease_expires_at=_safe_parse_dt(row.get("lease_expires_at")),
    )


class EnrichmentJobsMixin:
    """Mixin providing enrichment outbox CRUD and claim lifecycle."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def enqueue_enrichment_job(
        self,
        job: EnrichmentJob,
        *,
        in_transaction: bool = False,
    ) -> EnrichmentJob:
        """Insert a job; on idempotency conflict return the existing row.

        ``in_transaction`` is reserved for callers that already hold a write
        txn; the dialect commit behavior is unchanged (dialect auto-commit
        unless nested).
        """
        del in_transaction  # documented for API symmetry with plan contract
        d = self._dialect
        brain_id = job.brain_id or self._get_brain_id()
        now = utcnow()

        try:
            await d.execute(
                f"""INSERT INTO enrichment_jobs
                    (id, brain_id, kind, entity_id, idempotency_key, payload,
                     status, attempts, available_at, created_at, updated_at,
                     last_error, lease_owner, lease_expires_at)
                    VALUES ({d.phs(14)})""",
                [
                    job.id,
                    brain_id,
                    job.kind.value,
                    job.entity_id,
                    job.idempotency_key,
                    json.dumps(job.payload),
                    job.status.value,
                    job.attempts,
                    d.serialize_dt(job.available_at),
                    d.serialize_dt(job.created_at or now),
                    d.serialize_dt(job.updated_at or now),
                    job.last_error,
                    job.lease_owner,
                    d.serialize_dt(job.lease_expires_at) if job.lease_expires_at else None,
                ],
            )
            return job
        except Exception as exc:
            # Unique conflict → return existing
            msg = str(exc).lower()
            if "unique" not in msg and "duplicate" not in msg and "constraint" not in msg:
                raise
            existing = await self.get_enrichment_job_by_key(job.idempotency_key, brain_id=brain_id)
            if existing is not None:
                return existing
            raise

    async def get_enrichment_job(self, job_id: str) -> EnrichmentJob | None:
        d = self._dialect
        brain_id = self._get_brain_id()
        row = await d.fetch_one(
            f"""SELECT * FROM enrichment_jobs
                WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}""",
            [brain_id, job_id],
        )
        return _row_to_job(row) if row else None

    async def get_enrichment_job_by_key(
        self,
        idempotency_key: str,
        *,
        brain_id: str | None = None,
    ) -> EnrichmentJob | None:
        d = self._dialect
        bid = brain_id or self._get_brain_id()
        row = await d.fetch_one(
            f"""SELECT * FROM enrichment_jobs
                WHERE brain_id = {d.ph(1)} AND idempotency_key = {d.ph(2)}""",
            [bid, idempotency_key],
        )
        return _row_to_job(row) if row else None

    async def claim_enrichment_jobs(
        self,
        *,
        limit: int = 50,
        lease_seconds: int = 30,
        worker_id: str | None = None,
        kinds: list[EnrichmentKind] | None = None,
    ) -> list[EnrichmentJob]:
        """Atomically claim up to ``limit`` available pending/expired-lease jobs."""
        d = self._dialect
        brain_id = self._get_brain_id()
        safe_limit = max(1, min(int(limit), _MAX_CLAIM))
        lease_seconds = max(1, min(int(lease_seconds), 600))
        owner = worker_id or str(uuid4())
        now = utcnow()
        now_s = d.serialize_dt(now)
        lease_exp = d.serialize_dt(now + timedelta(seconds=lease_seconds))

        # Candidate selection: pending & available, or running with expired lease
        kind_clause = ""
        params: list[Any] = [brain_id, now_s, now_s]
        if kinds:
            placeholders = ", ".join(d.ph(i) for i in range(4, 4 + len(kinds)))
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend(k.value for k in kinds)
        params.append(safe_limit)

        candidates = await d.fetch_all(
            f"""SELECT id FROM enrichment_jobs
                WHERE brain_id = {d.ph(1)}
                  AND (
                    (status = 'pending' AND available_at <= {d.ph(2)})
                    OR (status = 'running' AND lease_expires_at IS NOT NULL
                        AND lease_expires_at < {d.ph(3)})
                  )
                  {kind_clause}
                ORDER BY available_at ASC
                LIMIT {d.ph(len(params))}""",
            params,
        )
        if not candidates:
            return []

        claimed: list[EnrichmentJob] = []
        for cand in candidates:
            job_id = str(cand["id"])
            # Optimistic claim: only succeed if still claimable
            result = await d.execute(
                f"""UPDATE enrichment_jobs
                    SET status = 'running',
                        lease_owner = {d.ph(1)},
                        lease_expires_at = {d.ph(2)},
                        updated_at = {d.ph(3)},
                        attempts = attempts + 1
                    WHERE brain_id = {d.ph(4)} AND id = {d.ph(5)}
                      AND (
                        (status = 'pending' AND available_at <= {d.ph(6)})
                        OR (status = 'running' AND lease_expires_at IS NOT NULL
                            AND lease_expires_at < {d.ph(7)})
                      )""",
                [owner, lease_exp, now_s, brain_id, job_id, now_s, now_s],
            )
            # Some dialects return rowcount; if unavailable re-fetch
            row = await d.fetch_one(
                f"""SELECT * FROM enrichment_jobs
                    WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}
                      AND status = 'running' AND lease_owner = {d.ph(3)}""",
                [brain_id, job_id, owner],
            )
            if row:
                claimed.append(_row_to_job(row))
            del result

        return claimed

    async def complete_enrichment_job(self, job_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()
        now_s = d.serialize_dt(utcnow())
        await d.execute(
            f"""UPDATE enrichment_jobs
                SET status = 'done',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error = NULL,
                    updated_at = {d.ph(1)}
                WHERE brain_id = {d.ph(2)} AND id = {d.ph(3)}
                  AND status = 'running'""",
            [now_s, brain_id, job_id],
        )
        row = await d.fetch_one(
            f"""SELECT status FROM enrichment_jobs
                WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}""",
            [brain_id, job_id],
        )
        return bool(row and str(row.get("status")) == "done")

    async def fail_enrichment_job(
        self,
        job_id: str,
        *,
        error: str,
        max_attempts: int = _MAX_ATTEMPTS_DEFAULT,
        backoff_base: float = _BACKOFF_BASE_SEC,
        backoff_cap: float = _BACKOFF_CAP_SEC,
    ) -> EnrichmentJob | None:
        """Mark failed; requeue with backoff or dead-letter when exhausted."""
        d = self._dialect
        brain_id = self._get_brain_id()
        job = await self.get_enrichment_job(job_id)
        if job is None:
            return None

        now = utcnow()
        max_attempts = max(1, min(int(max_attempts), 20))
        error_text = (error or "unknown")[:500]

        if job.attempts >= max_attempts:
            await d.execute(
                f"""UPDATE enrichment_jobs
                    SET status = 'dead',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = {d.ph(1)},
                        updated_at = {d.ph(2)}
                    WHERE brain_id = {d.ph(3)} AND id = {d.ph(4)}""",
                [error_text, d.serialize_dt(now), brain_id, job_id],
            )
        else:
            delay = min(backoff_cap, backoff_base * (2 ** max(0, job.attempts - 1)))
            available = now + timedelta(seconds=delay)
            await d.execute(
                f"""UPDATE enrichment_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = {d.ph(1)},
                        available_at = {d.ph(2)},
                        updated_at = {d.ph(3)}
                    WHERE brain_id = {d.ph(4)} AND id = {d.ph(5)}""",
                [
                    error_text,
                    d.serialize_dt(available),
                    d.serialize_dt(now),
                    brain_id,
                    job_id,
                ],
            )
        return await self.get_enrichment_job(job_id)

    async def requeue_enrichment_job(
        self,
        job_id: str,
        *,
        available_at: datetime | None = None,
    ) -> bool:
        """Force requeue (e.g. shutdown release)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = utcnow()
        avail = available_at or now
        await d.execute(
            f"""UPDATE enrichment_jobs
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    available_at = {d.ph(1)},
                    updated_at = {d.ph(2)}
                WHERE brain_id = {d.ph(3)} AND id = {d.ph(4)}
                  AND status IN ('running', 'pending')""",
            [d.serialize_dt(avail), d.serialize_dt(now), brain_id, job_id],
        )
        row = await self.get_enrichment_job(job_id)
        return bool(row and row.status is EnrichmentStatus.PENDING)

    async def release_enrichment_leases(self, worker_id: str) -> int:
        """Release all running leases owned by worker (shutdown)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now_s = d.serialize_dt(utcnow())
        rows = await d.fetch_all(
            f"""SELECT id FROM enrichment_jobs
                WHERE brain_id = {d.ph(1)}
                  AND status = 'running'
                  AND lease_owner = {d.ph(2)}""",
            [brain_id, worker_id],
        )
        count = 0
        for row in rows:
            ok = await self.requeue_enrichment_job(str(row["id"]))
            if ok:
                count += 1
        del now_s
        return count

    async def list_enrichment_jobs(
        self,
        *,
        status: EnrichmentStatus | None = None,
        limit: int = 100,
    ) -> list[EnrichmentJob]:
        d = self._dialect
        brain_id = self._get_brain_id()
        safe_limit = max(1, min(int(limit), 500))
        if status is not None:
            rows = await d.fetch_all(
                f"""SELECT * FROM enrichment_jobs
                    WHERE brain_id = {d.ph(1)} AND status = {d.ph(2)}
                    ORDER BY created_at DESC
                    LIMIT {d.ph(3)}""",
                [brain_id, status.value, safe_limit],
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT * FROM enrichment_jobs
                    WHERE brain_id = {d.ph(1)}
                    ORDER BY created_at DESC
                    LIMIT {d.ph(2)}""",
                [brain_id, safe_limit],
            )
        return [_row_to_job(r) for r in rows]

    async def count_enrichment_jobs(
        self,
        *,
        status: EnrichmentStatus | None = None,
    ) -> int:
        d = self._dialect
        brain_id = self._get_brain_id()
        if status is not None:
            row = await d.fetch_one(
                f"""SELECT COUNT(*) as cnt FROM enrichment_jobs
                    WHERE brain_id = {d.ph(1)} AND status = {d.ph(2)}""",
                [brain_id, status.value],
            )
        else:
            row = await d.fetch_one(
                f"""SELECT COUNT(*) as cnt FROM enrichment_jobs
                    WHERE brain_id = {d.ph(1)}""",
                [brain_id],
            )
        return int(row["cnt"]) if row else 0
