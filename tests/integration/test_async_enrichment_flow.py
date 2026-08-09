"""End-to-end lean write + durable enrichment flow (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.enrichment_job import EnrichmentStatus
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.enrichment_worker import process_enrichment_batch
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def lean_storage(tmp_path: Path) -> SQLStorage:
    dialect = SQLiteDialect(str(tmp_path / "lean.db"))
    store = SQLStorage(dialect)
    await store.initialize()
    brain = Brain.create(
        name="lean",
        config=BrainConfig(
            encoding_profile="lean",
            async_enrichment_enabled=True,
            embedding_enabled=False,
            fiber_summary_tier_enabled=False,
        ),
    )
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_lean_encode_enqueues_jobs(lean_storage: SQLStorage) -> None:
    brain = await lean_storage.get_brain(lean_storage.brain_id or "")
    assert brain is not None
    encoder = MemoryEncoder(lean_storage, brain.config)
    result = await encoder.encode(
        "Chose lean writes over sync embedding because ack latency matters."
    )
    assert result.fiber is not None
    assert result.encoding_profile == "lean"
    assert result.enrichment_status == "pending"
    pending = await lean_storage.count_enrichment_jobs(status=EnrichmentStatus.PENDING)
    assert pending >= 1


@pytest.mark.asyncio
async def test_cognitive_default_no_outbox() -> None:
    # Compatibility: missing/default profile does not enqueue
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        store = SQLStorage(SQLiteDialect(f"{tmp}/cog.db"))
        await store.initialize()
        brain = Brain.create(name="cog", config=BrainConfig())  # cognitive + no async
        await store.save_brain(brain)
        store.set_brain(brain.id)
        encoder = MemoryEncoder(store, brain.config)
        result = await encoder.encode("Compatibility path still uses cognitive encoding.")
        assert result.encoding_profile == "cognitive"
        assert result.enrichment_status == "done"
        assert await store.count_enrichment_jobs() == 0
        await store.close()


@pytest.mark.asyncio
async def test_worker_drains_pending(lean_storage: SQLStorage) -> None:
    brain = await lean_storage.get_brain(lean_storage.brain_id or "")
    assert brain is not None
    encoder = MemoryEncoder(lean_storage, brain.config)
    await encoder.encode("Root cause was missing outbox which led to lost embeddings.")
    report = await process_enrichment_batch(lean_storage, worker_id="flow")
    assert report.claimed >= 1
    # LINK/ENRICH/EMBED without provider: embed fails, others may complete
    done = await lean_storage.count_enrichment_jobs(status=EnrichmentStatus.DONE)
    pending = await lean_storage.count_enrichment_jobs(status=EnrichmentStatus.PENDING)
    running = await lean_storage.count_enrichment_jobs(status=EnrichmentStatus.RUNNING)
    # Some work progressed; nothing stuck running
    assert running == 0
    assert done + pending >= 1
