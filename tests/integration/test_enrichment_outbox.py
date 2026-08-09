"""Integration tests for durable enrichment outbox (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.enrichment_job import EnrichmentJob, EnrichmentKind, EnrichmentStatus
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.enrichment_worker import process_enrichment_batch
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION


@pytest.fixture
async def storage(tmp_path: Path) -> SQLStorage:
    dialect = SQLiteDialect(str(tmp_path / "outbox.db"))
    store = SQLStorage(dialect)
    await store.initialize()
    brain = Brain.create(name="outbox", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_schema_version_is_40(storage: SQLStorage) -> None:
    row = await storage._dialect.fetch_one("SELECT version FROM schema_version")
    assert row is not None
    assert int(row["version"]) == SCHEMA_VERSION == 41


@pytest.mark.asyncio
async def test_enqueue_claim_complete(storage: SQLStorage) -> None:
    job = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.LINK,
        entity_id="n1",
        idempotency_key="link:n1",
    )
    stored = await storage.enqueue_enrichment_job(job)
    assert stored.id == job.id
    assert await storage.count_enrichment_jobs(status=EnrichmentStatus.PENDING) == 1

    claimed = await storage.claim_enrichment_jobs(limit=10, worker_id="w1")
    assert len(claimed) == 1
    assert claimed[0].status is EnrichmentStatus.RUNNING
    assert claimed[0].attempts == 1

    ok = await storage.complete_enrichment_job(claimed[0].id)
    assert ok is True
    assert await storage.count_enrichment_jobs(status=EnrichmentStatus.DONE) == 1


@pytest.mark.asyncio
async def test_duplicate_enqueue_idempotent(storage: SQLStorage) -> None:
    j1 = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.EMBED,
        entity_id="n2",
        idempotency_key="embed:n2",
    )
    j2 = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.EMBED,
        entity_id="n2",
        idempotency_key="embed:n2",
    )
    a = await storage.enqueue_enrichment_job(j1)
    b = await storage.enqueue_enrichment_job(j2)
    assert a.id == b.id
    assert await storage.count_enrichment_jobs() == 1


@pytest.mark.asyncio
async def test_fail_then_dead_letter(storage: SQLStorage) -> None:
    job = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.ENRICH,
        entity_id="n3",
        idempotency_key="enrich:n3",
    )
    await storage.enqueue_enrichment_job(job)
    claimed = await storage.claim_enrichment_jobs(limit=1, worker_id="w")
    assert claimed
    # attempts is already 1 after claim; force max_attempts=1 → dead
    updated = await storage.fail_enrichment_job(
        claimed[0].id,
        error="boom",
        max_attempts=1,
    )
    assert updated is not None
    assert updated.status is EnrichmentStatus.DEAD
    assert updated.last_error == "boom"


@pytest.mark.asyncio
async def test_lease_release_on_shutdown(storage: SQLStorage) -> None:
    job = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.ANALYTICS,
        entity_id="n4",
        idempotency_key="analytics:n4",
    )
    await storage.enqueue_enrichment_job(job)
    claimed = await storage.claim_enrichment_jobs(limit=1, worker_id="shutdown-w")
    assert claimed and claimed[0].status is EnrichmentStatus.RUNNING
    released = await storage.release_enrichment_leases("shutdown-w")
    assert released == 1
    pending = await storage.list_enrichment_jobs(status=EnrichmentStatus.PENDING)
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_worker_processes_analytics(storage: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="analytics entity")
    await storage.add_neuron(n)
    job = EnrichmentJob.create(
        brain_id=storage.brain_id or "",
        kind=EnrichmentKind.ANALYTICS,
        entity_id=n.id,
        idempotency_key=f"analytics:{n.id}",
    )
    await storage.enqueue_enrichment_job(job)
    report = await process_enrichment_batch(storage, worker_id="test-w")
    assert report.claimed == 1
    assert report.completed == 1
