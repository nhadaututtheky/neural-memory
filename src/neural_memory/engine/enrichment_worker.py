"""Bounded idempotent worker for durable enrichment outbox jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from neural_memory.core.enrichment_job import (
    EnrichmentJob,
    EnrichmentKind,
    EnrichmentReport,
    EnrichmentStatus,
)

logger = logging.getLogger(__name__)

_DEFAULT_BATCH = 50
_DEFAULT_CONCURRENCY = 4
_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_LEASE_SEC = 30


async def process_enrichment_batch(
    storage: Any,
    *,
    limit: int = _DEFAULT_BATCH,
    concurrency: int = _DEFAULT_CONCURRENCY,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    lease_seconds: int = _DEFAULT_LEASE_SEC,
    worker_id: str | None = None,
    embedding_provider: Any | None = None,
) -> EnrichmentReport:
    """Claim and process a bounded batch of enrichment jobs.

    Completes a job only after its side effect succeeds. Failures requeue
    with backoff; exhausted attempts go to dead-letter.

    Args:
        storage: Storage with enrichment outbox APIs.
        limit: Max jobs to claim (capped at 50).
        concurrency: Max concurrent job handlers.
        max_attempts: Attempts before dead-letter.
        lease_seconds: Claim lease duration.
        worker_id: Stable worker identity for lease ownership.
        embedding_provider: Optional provider for EMBED jobs.
    """
    if not hasattr(storage, "claim_enrichment_jobs"):
        return EnrichmentReport(errors=("outbox_unavailable",))

    owner = worker_id or f"worker-{uuid4().hex[:8]}"
    safe_limit = max(1, min(int(limit), _DEFAULT_BATCH))
    safe_conc = max(1, min(int(concurrency), 16))

    try:
        jobs = await storage.claim_enrichment_jobs(
            limit=safe_limit,
            lease_seconds=lease_seconds,
            worker_id=owner,
        )
    except Exception as exc:
        logger.error("Failed to claim enrichment jobs: %s", exc)
        return EnrichmentReport(errors=(f"claim_failed:{type(exc).__name__}",))

    if not jobs:
        return EnrichmentReport(claimed=0)

    sem = asyncio.Semaphore(safe_conc)
    completed = 0
    failed = 0
    dead = 0
    skipped = 0
    errors: list[str] = []

    async def _run_one(job: EnrichmentJob) -> None:
        nonlocal completed, failed, dead, skipped
        async with sem:
            try:
                ok = await _apply_job(
                    storage,
                    job,
                    embedding_provider=embedding_provider,
                )
                if ok is None:
                    skipped += 1
                    await storage.complete_enrichment_job(job.id, worker_id=owner)
                    return
                if ok:
                    await storage.complete_enrichment_job(job.id, worker_id=owner)
                    completed += 1
                else:
                    updated = await storage.fail_enrichment_job(
                        job.id,
                        error="handler_returned_false",
                        max_attempts=max_attempts,
                        worker_id=owner,
                    )
                    if updated and updated.status is EnrichmentStatus.DEAD:
                        dead += 1
                    else:
                        failed += 1
            except Exception as exc:
                logger.debug(
                    "Enrichment job %s failed: %s",
                    job.id,
                    exc,
                    exc_info=True,
                )
                errors.append(f"{job.kind.value}:{type(exc).__name__}")
                try:
                    updated = await storage.fail_enrichment_job(
                        job.id,
                        error=str(exc)[:500],
                        max_attempts=max_attempts,
                        worker_id=owner,
                    )
                    if updated and updated.status is EnrichmentStatus.DEAD:
                        dead += 1
                    else:
                        failed += 1
                except Exception as fail_exc:
                    logger.error(
                        "Failed to record enrichment failure for %s: %s",
                        job.id,
                        fail_exc,
                    )
                    failed += 1

    await asyncio.gather(*[_run_one(j) for j in jobs])

    return EnrichmentReport(
        claimed=len(jobs),
        completed=completed,
        failed=failed,
        dead_lettered=dead,
        skipped=skipped,
        errors=tuple(errors[:20]),
    )


async def _apply_job(
    storage: Any,
    job: EnrichmentJob,
    *,
    embedding_provider: Any | None,
) -> bool | None:
    """Apply one job. Returns True ok, False fail, None skip (deleted entity)."""
    if job.kind is EnrichmentKind.EMBED:
        return await _apply_embed(storage, job, embedding_provider)
    if job.kind is EnrichmentKind.LINK:
        return await _apply_link(storage, job)
    if job.kind is EnrichmentKind.ENRICH:
        return await _apply_enrich(storage, job)
    if job.kind is EnrichmentKind.ANALYTICS:
        # Analytics is best-effort bookkeeping — always succeed
        return True
    logger.warning("Unknown enrichment kind %s", job.kind)
    return False


async def _apply_embed(
    storage: Any,
    job: EnrichmentJob,
    provider: Any | None,
) -> bool | None:
    if provider is None:
        raise RuntimeError("embedding_provider_unavailable")

    neuron_id = job.entity_id
    batch = await storage.get_neurons_batch([neuron_id])
    neuron = batch.get(neuron_id)
    if neuron is None:
        return None  # deleted entity — complete as skip

    content = getattr(neuron, "content", None) or job.payload.get("content")
    if not content or not isinstance(content, str):
        return None

    vector = await provider.embed(content)
    if not hasattr(storage, "vector_index_add"):
        raise RuntimeError("vector_index_unavailable")
    # Ensure index can accept writes (hnswlib etc.)
    index = getattr(storage, "_vector_index", None)
    ensure = getattr(storage, "_ensure_vector_index", None)
    if ensure is not None:
        try:
            index = ensure()
        except Exception:
            index = None
    if index is None and ensure is not None:
        raise RuntimeError("vector_index_unavailable")
    await storage.vector_index_add(neuron_id, vector)
    return True


async def _apply_link(storage: Any, job: EnrichmentJob) -> bool | None:
    """Deferred semantic/cross-memory linking (best-effort)."""
    entity_id = job.entity_id
    batch = await storage.get_neurons_batch([entity_id])
    if entity_id not in batch:
        return None
    # Lightweight: record that link pass ran; full graph linking remains
    # available via consolidation / semantic_discovery paths.
    return True


async def _apply_enrich(storage: Any, job: EnrichmentJob) -> bool | None:
    """Deferred enrichment (transitive closure is batch-oriented)."""
    entity_id = job.entity_id
    batch = await storage.get_neurons_batch([entity_id])
    if entity_id not in batch:
        return None
    return True


async def enqueue_post_encode_jobs(
    storage: Any,
    *,
    brain_id: str,
    entity_id: str,
    content: str,
    kinds: list[EnrichmentKind] | None = None,
    embedding_enabled: bool = False,
) -> list[Any]:
    """Enqueue standard post-encode enrichment jobs (idempotent)."""
    if not hasattr(storage, "enqueue_enrichment_job"):
        return []

    from neural_memory.core.enrichment_job import EnrichmentJob

    if kinds is not None:
        selected = list(kinds)
    else:
        selected = [EnrichmentKind.LINK, EnrichmentKind.ENRICH]
        if embedding_enabled:
            selected.insert(0, EnrichmentKind.EMBED)
    enqueued: list[Any] = []
    for kind in selected:
        job = EnrichmentJob.create(
            brain_id=brain_id,
            kind=kind,
            entity_id=entity_id,
            idempotency_key=f"{kind.value}:{entity_id}",
            payload={"content": content[:2000]} if kind is EnrichmentKind.EMBED else {},
        )
        try:
            stored = await storage.enqueue_enrichment_job(job)
            enqueued.append(stored)
        except Exception:
            logger.debug("Failed to enqueue %s for %s", kind.value, entity_id, exc_info=True)
    return enqueued
