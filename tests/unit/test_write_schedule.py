"""Postgres overlaps independent writes. SQLite runs them one at a time."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from neural_memory.core.enrichment_job import EnrichmentJob, EnrichmentKind, EnrichmentStatus
from neural_memory.engine import enrichment_worker as worker_mod
from neural_memory.engine.enrichment_worker import process_enrichment_batch
from neural_memory.engine.write_queue import _gather_count
from neural_memory.storage.write_schedule import run_writes, writes_run_concurrently
from neural_memory.utils.timeutils import utcnow


def _backend(name: str | None) -> SimpleNamespace:
    if name is None:
        return SimpleNamespace()
    return SimpleNamespace(_dialect=SimpleNamespace(name=name))


def _job(job_id: str, entity_id: str) -> EnrichmentJob:
    now = utcnow()
    return EnrichmentJob(
        id=job_id,
        brain_id="b1",
        kind=EnrichmentKind.EMBED,
        entity_id=entity_id,
        idempotency_key=f"embed:{entity_id}",
        payload={"content": "hello"},
        status=EnrichmentStatus.RUNNING,
        attempts=1,
        available_at=now,
        created_at=now,
        updated_at=now,
    )


def _async_return(value: object):
    async def _call(*_args: object, **_kwargs: object) -> object:
        return value

    return _call


def test_only_postgres_overlaps() -> None:
    assert writes_run_concurrently(_backend("postgres"))
    assert not writes_run_concurrently(_backend("sqlite"))
    assert not writes_run_concurrently(_backend(None))


async def test_sqlite_writes_do_not_overlap() -> None:
    current = 0
    peak = 0

    async def _one(value: int) -> int:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0)
        current -= 1
        return value

    assert await run_writes(_backend("sqlite"), [_one(1), _one(2), _one(3)]) == [1, 2, 3]
    assert peak == 1


async def test_postgres_writes_overlap_and_keep_order() -> None:
    current = 0
    peak = 0
    started = 0
    release = asyncio.Event()

    async def _one(value: int) -> int:
        nonlocal current, peak, started
        current += 1
        peak = max(peak, current)
        started += 1
        if started == 2:
            release.set()
        await release.wait()
        current -= 1
        return value

    found = await asyncio.wait_for(
        run_writes(_backend("postgres"), [_one(1), _one(2)]),
        timeout=2,
    )
    assert found == [1, 2]
    assert peak == 2


async def test_gather_count_keeps_the_successes() -> None:
    async def _ok() -> None:
        return None

    async def _bad() -> None:
        raise RuntimeError("nope")

    count = await _gather_count(_backend("postgres"), [_ok(), _bad(), _ok()], "write failed")
    assert count == 2
    serial = await _gather_count(_backend("sqlite"), [_ok(), _bad(), _ok()], "write failed")
    assert serial == 2


async def _run_enrichment(name: str, apply) -> tuple[int, object]:
    current = 0
    peak = 0

    async def _apply(storage: object, job: object, *, embedding_provider: object) -> bool:
        del storage, job, embedding_provider
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await apply()
        current -= 1
        return True

    storage = _backend(name)
    storage.claim_enrichment_jobs = _async_return(  # type: ignore[attr-defined]
        [_job("job-a", "n1"), _job("job-b", "n2")]
    )
    storage.complete_enrichment_job = _async_return(True)  # type: ignore[attr-defined]
    original = worker_mod._apply_job
    worker_mod._apply_job = _apply
    try:
        report = await asyncio.wait_for(process_enrichment_batch(storage, concurrency=4), timeout=2)
    finally:
        worker_mod._apply_job = original
    return peak, report


async def test_postgres_enrichment_jobs_overlap() -> None:
    started = 0
    release = asyncio.Event()

    async def _body() -> None:
        nonlocal started
        started += 1
        if started == 2:
            release.set()
        await release.wait()

    peak, report = await _run_enrichment("postgres", _body)
    assert peak == 2
    assert report.completed == 2
    assert report.claimed == 2


async def test_sqlite_enrichment_jobs_stay_serial() -> None:
    async def _body() -> None:
        await asyncio.sleep(0)

    peak, report = await _run_enrichment("sqlite", _body)
    assert peak == 1
    assert report.completed == 2
