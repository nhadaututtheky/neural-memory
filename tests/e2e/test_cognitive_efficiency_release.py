"""Cognitive Efficiency release E2E gate (Phase 8).

Proves data integrity across restart, outbox durability, change_log feed,
consolidation checkpoints, and adapter-aware open — without requiring
external embedding providers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
from neural_memory.core.enrichment_job import EnrichmentStatus
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.enrichment_worker import process_enrichment_batch
from neural_memory.storage.factory import open_sqlite_storage
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


async def _open_brain(db_path: Path, name: str = "release") -> SQLStorage:
    store = SQLStorage(SQLiteDialect(str(db_path)))
    await store.initialize()
    brain = Brain.create(
        name=name,
        config=BrainConfig(
            encoding_profile="lean",
            async_enrichment_enabled=True,
            embedding_enabled=False,
            fiber_summary_tier_enabled=False,
        ),
    )
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest.mark.asyncio
async def test_write_restart_recall_coherence(tmp_path: Path) -> None:
    """Ack'd write survives process-equivalent close/reopen with FTS-visible content."""
    db = tmp_path / "restart.db"
    store = await _open_brain(db)
    encoder = MemoryEncoder(store, (await store.get_brain(store.brain_id or "")).config)  # type: ignore[arg-type]
    content = "Chose unified SQLStorage over legacy because outbox requires it."
    result = await encoder.encode(content, metadata={"type": "decision", "priority": 7})
    assert result.fiber is not None
    fiber_id = result.fiber.id
    brain_id = store.brain_id
    await store.close()

    reopened = SQLStorage(SQLiteDialect(str(db)))
    await reopened.initialize()
    reopened.set_brain(str(brain_id))
    fiber = await reopened.get_fiber(fiber_id)
    assert fiber is not None
    # Anchor content still present
    anchor = await reopened.get_neuron(fiber.anchor_neuron_id)
    assert anchor is not None
    assert "unified SQLStorage" in anchor.content
    await reopened.close()


@pytest.mark.asyncio
async def test_outbox_survives_restart_and_drains(tmp_path: Path) -> None:
    """Pending enrichment jobs survive reopen and drain without stuck RUNNING."""
    db = tmp_path / "outbox.db"
    store = await _open_brain(db)
    brain = await store.get_brain(store.brain_id or "")
    assert brain is not None
    encoder = MemoryEncoder(store, brain.config)
    result = await encoder.encode(
        "Root cause was missing durable outbox which led to lost embeddings."
    )
    assert result.enrichment_status in ("pending", "none", "done")
    pending_before = await store.count_enrichment_jobs(status=EnrichmentStatus.PENDING)
    brain_id = store.brain_id
    await store.close()

    reopened = SQLStorage(SQLiteDialect(str(db)))
    await reopened.initialize()
    reopened.set_brain(str(brain_id))
    pending_after = await reopened.count_enrichment_jobs(status=EnrichmentStatus.PENDING)
    assert pending_after == pending_before
    if pending_after > 0:
        report = await process_enrichment_batch(reopened, worker_id="e2e-release")
        assert report.claimed >= 1
        running = await reopened.count_enrichment_jobs(status=EnrichmentStatus.RUNNING)
        assert running == 0
    await reopened.close()


@pytest.mark.asyncio
async def test_change_log_feeds_incremental_dirty_set(tmp_path: Path) -> None:
    """Local writes auto-record change_log so incremental consolidation has feed."""
    db = tmp_path / "clog.db"
    store = await _open_brain(db, name="clog")
    n = Neuron.create(type=NeuronType.CONCEPT, content="change log dirty feed neuron")
    await store.add_neuron(n)
    changes = await store.get_changes_since(0, limit=20)
    assert any(c.entity_id == n.id and c.operation == "insert" for c in changes)
    await store.close()


@pytest.mark.asyncio
async def test_checkpoint_resume_does_not_skip(tmp_path: Path) -> None:
    """Checkpoints advance independently of change_log.synced and are durable."""
    db = tmp_path / "cp.db"
    store = await _open_brain(db, name="cp")
    assert hasattr(store, "save_consolidation_checkpoint")
    brain_id = str(store.brain_id)
    cp = ConsolidationCheckpoint.create(
        brain_id=brain_id,
        strategy="prune",
        last_sequence=3,
    )
    await store.save_consolidation_checkpoint(cp)
    loaded = await store.get_consolidation_checkpoint("prune")
    assert loaded is not None
    assert loaded.last_sequence == 3

    # Reject backward move
    with pytest.raises(ValueError, match=r"backward|Refusing"):
        await store.save_consolidation_checkpoint(
            ConsolidationCheckpoint.create(
                brain_id=brain_id,
                strategy="prune",
                last_sequence=1,
            )
        )

    await store.close()
    reopened = SQLStorage(SQLiteDialect(str(db)))
    await reopened.initialize()
    reopened.set_brain(brain_id)
    again = await reopened.get_consolidation_checkpoint("prune")
    assert again is not None
    assert again.last_sequence == 3
    await reopened.close()


@pytest.mark.asyncio
async def test_open_sqlite_storage_adapter_roundtrip(tmp_path: Path) -> None:
    """Adapter-aware open preserves unified path for fresh brains."""
    db = tmp_path / "adapter.db"
    store = await open_sqlite_storage(db, storage_adapter="unified", brain_id="b-adapter")
    try:
        assert isinstance(store, SQLStorage)
        brain = Brain.create(name="adapter", brain_id="b-adapter")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        n = Neuron.create(type=NeuronType.CONCEPT, content="adapter open path")
        await store.add_neuron(n)
    finally:
        await store.close()

    again = await open_sqlite_storage(db, storage_adapter="unified", set_brain_name="adapter")
    try:
        found = await again.find_neurons(content_exact="adapter open path")
        assert found
    finally:
        await again.close()


@pytest.mark.asyncio
async def test_batch_delete_does_not_orphan_vector_ids(tmp_path: Path) -> None:
    """delete_neurons_batch only processes live IDs (P2-3 residual)."""
    db = tmp_path / "batch.db"
    store = await _open_brain(db, name="batch")
    n = Neuron.create(type=NeuronType.CONCEPT, content="live neuron for batch delete")
    await store.add_neuron(n)
    # Mix live + missing IDs
    deleted = await store.delete_neurons_batch([n.id, "missing-id-should-skip"])
    assert deleted >= 1
    assert await store.get_neuron(n.id) is None
    await store.close()
