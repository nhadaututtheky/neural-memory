"""Tests for enrichment outbox worker (Phase 5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.enrichment_job import (
    EnrichmentJob,
    EnrichmentKind,
    EnrichmentStatus,
)
from neural_memory.engine.enrichment_worker import (
    enqueue_post_encode_jobs,
    process_enrichment_batch,
)
from neural_memory.utils.timeutils import utcnow


def _job(
    kind: EnrichmentKind = EnrichmentKind.EMBED,
    entity_id: str = "n1",
    attempts: int = 1,
) -> EnrichmentJob:
    now = utcnow()
    return EnrichmentJob(
        id="job-1",
        brain_id="b1",
        kind=kind,
        entity_id=entity_id,
        idempotency_key=f"{kind.value}:{entity_id}",
        payload={"content": "hello vector world"},
        status=EnrichmentStatus.RUNNING,
        attempts=attempts,
        available_at=now,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_process_batch_no_outbox() -> None:
    storage = MagicMock(spec=[])
    report = await process_enrichment_batch(storage)
    assert report.claimed == 0
    assert "outbox_unavailable" in report.errors


@pytest.mark.asyncio
async def test_process_embed_job_completes() -> None:
    job = _job()
    storage = AsyncMock()
    storage.claim_enrichment_jobs = AsyncMock(return_value=[job])
    storage.complete_enrichment_job = AsyncMock(return_value=True)
    storage.fail_enrichment_job = AsyncMock()
    neuron = MagicMock()
    neuron.content = "hello vector world"
    storage.get_neurons_batch = AsyncMock(return_value={"n1": neuron})
    storage.vector_index_add = AsyncMock()

    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    report = await process_enrichment_batch(storage, embedding_provider=provider)
    assert report.claimed == 1
    assert report.completed == 1
    storage.vector_index_add.assert_awaited_once()
    storage.complete_enrichment_job.assert_awaited()
    call_kwargs = storage.complete_enrichment_job.await_args
    assert call_kwargs.args[0] == "job-1"


@pytest.mark.asyncio
async def test_deleted_entity_skipped() -> None:
    job = _job()
    storage = AsyncMock()
    storage.claim_enrichment_jobs = AsyncMock(return_value=[job])
    storage.complete_enrichment_job = AsyncMock(return_value=True)
    storage.get_neurons_batch = AsyncMock(return_value={})
    provider = AsyncMock()
    provider.embed = AsyncMock()

    report = await process_enrichment_batch(storage, embedding_provider=provider)
    assert report.skipped == 1
    provider.embed.assert_not_awaited()
    storage.complete_enrichment_job.assert_awaited()


@pytest.mark.asyncio
async def test_provider_missing_fails() -> None:
    job = _job()
    storage = AsyncMock()
    storage.claim_enrichment_jobs = AsyncMock(return_value=[job])
    storage.fail_enrichment_job = AsyncMock(
        return_value=EnrichmentJob(
            id=job.id,
            brain_id=job.brain_id,
            kind=job.kind,
            entity_id=job.entity_id,
            idempotency_key=job.idempotency_key,
            payload=job.payload,
            status=EnrichmentStatus.PENDING,
            attempts=job.attempts,
            available_at=job.available_at,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
    )
    neuron = MagicMock()
    neuron.content = "x"
    storage.get_neurons_batch = AsyncMock(return_value={"n1": neuron})

    report = await process_enrichment_batch(storage, embedding_provider=None)
    assert report.failed == 1
    storage.fail_enrichment_job.assert_awaited()


@pytest.mark.asyncio
async def test_enqueue_post_encode_idempotent_keys() -> None:
    storage = AsyncMock()
    stored = []

    async def _enqueue(job: EnrichmentJob, **_kwargs: object) -> EnrichmentJob:
        stored.append(job)
        return job

    storage.enqueue_enrichment_job = _enqueue
    jobs = await enqueue_post_encode_jobs(
        storage,
        brain_id="b1",
        entity_id="n9",
        content="test content for embedding",
        embedding_enabled=True,
    )
    assert len(jobs) == 3
    keys = {j.idempotency_key for j in stored}
    assert "embed:n9" in keys
    assert "link:n9" in keys
    assert "enrich:n9" in keys

    # Without embeddings: no EMBED job
    stored.clear()
    jobs2 = await enqueue_post_encode_jobs(
        storage,
        brain_id="b1",
        entity_id="n9",
        content="test",
        embedding_enabled=False,
    )
    assert len(jobs2) == 2
    assert all(j.kind.value != "embed" for j in stored)
