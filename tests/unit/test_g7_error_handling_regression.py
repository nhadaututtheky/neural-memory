"""Regression tests for G7-errors cluster (audit 2026-06).

Covers:
  * #22 — save_maturation must UPSERT (advance stage on second save) instead of
          issuing a plain INSERT whose IntegrityError is silently swallowed,
          which blocked the EPISODIC->SEMANTIC transition.
  * #27 — list_pinned_fibers must not SELECT non-existent type/priority columns
          on `fibers` (they live on typed_memories) — used to raise
          OperationalError: no such column: type.
  * #28 — get_fiber_stage_counts must read stage from memory_maturations, not
          from the non-existent fibers.stage column (which pinned
          consolidation_ratio at 0.0).
  * #59 — add_project must only raise "already exists" on a genuine UNIQUE/PK
          integrity violation, not on every failure.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


# --------------------------------------------------------------------------- #
# Unified SQLStorage fixture (exercises sql/mixins/* — where #22 lived)
# --------------------------------------------------------------------------- #
@pytest.fixture
async def sql_storage(tmp_path):
    dialect = SQLiteDialect(tmp_path / "unified.db")
    store = SQLStorage(dialect)
    await store.initialize()
    brain = Brain.create(name="g7-brain", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


# --------------------------------------------------------------------------- #
# Legacy SQLiteStorage fixture (exercises sqlite_fibers.py — #27/#28)
# --------------------------------------------------------------------------- #
@pytest.fixture
async def legacy_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStorage(Path(tmpdir) / "legacy.db")
        await store.initialize()
        brain = Brain.create(name="g7-legacy-brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        yield store
        await store.close()


async def _make_fiber(store, summary: str = "fiber") -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"anchor for {summary}")
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=summary,
    )
    await store.add_fiber(fiber)
    return fiber


# --------------------------------------------------------------------------- #
# #22 — maturation upsert (CRITICAL: unblocks EPISODIC->SEMANTIC)
# --------------------------------------------------------------------------- #
class TestMaturationUpsert:
    @pytest.mark.asyncio
    async def test_second_save_updates_stage(self, sql_storage):
        """save_maturation twice for the same fiber must persist the new stage,
        not silently swallow an IntegrityError and keep the old one."""
        fiber = await _make_fiber(sql_storage, "matures")

        rec = MaturationRecord(
            fiber_id=fiber.id,
            brain_id=sql_storage._get_brain_id(),
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow() - timedelta(days=10),
            rehearsal_count=1,
            reinforcement_timestamps=(1.0,),
        )
        await sql_storage.save_maturation(rec)

        advanced = MaturationRecord(
            fiber_id=fiber.id,
            brain_id=sql_storage._get_brain_id(),
            stage=MemoryStage.SEMANTIC,
            stage_entered_at=utcnow(),
            rehearsal_count=5,
            reinforcement_timestamps=(1.0, 2.0, 3.0),
        )
        await sql_storage.save_maturation(advanced)

        loaded = await sql_storage.get_maturation(fiber.id)
        assert loaded is not None
        assert loaded.stage == MemoryStage.SEMANTIC
        assert loaded.rehearsal_count == 5
        assert loaded.reinforcement_timestamps == (1.0, 2.0, 3.0)

    @pytest.mark.asyncio
    async def test_deleted_fiber_fk_violation_swallowed(self, sql_storage):
        """A maturation save for a fiber that never existed (FK violation) is
        the one expected case that stays swallowed."""
        rec = MaturationRecord(
            fiber_id="does-not-exist",
            brain_id=sql_storage._get_brain_id(),
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow(),
            rehearsal_count=0,
            reinforcement_timestamps=(),
        )
        # Must not raise.
        await sql_storage.save_maturation(rec)
        assert await sql_storage.get_maturation("does-not-exist") is None


# --------------------------------------------------------------------------- #
# #27 — list_pinned_fibers schema fix
# --------------------------------------------------------------------------- #
class TestListPinnedFibers:
    @pytest.mark.asyncio
    async def test_list_pinned_no_operational_error(self, legacy_storage):
        """Must not raise 'no such column: type'; returns the pinned fiber."""
        fiber = await _make_fiber(legacy_storage, "pinned one")
        await legacy_storage.pin_fibers([fiber.id], pinned=True)

        result = await legacy_storage.list_pinned_fibers()
        assert len(result) == 1
        row = result[0]
        assert row["fiber_id"] == fiber.id
        assert row["summary"] == "pinned one"
        # Untyped fiber falls back to defaults via the LEFT JOIN.
        assert row["type"] == "unknown"
        assert row["priority"] == 5

    @pytest.mark.asyncio
    async def test_list_pinned_reads_typed_metadata(self, legacy_storage):
        """type/priority are surfaced from typed_memories when present."""
        fiber = await _make_fiber(legacy_storage, "typed pinned")
        await legacy_storage.pin_fibers([fiber.id], pinned=True)
        tm = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
        )
        await legacy_storage.add_typed_memory(tm)

        result = await legacy_storage.list_pinned_fibers()
        assert len(result) == 1
        assert result[0]["type"] == MemoryType.DECISION.value
        assert result[0]["priority"] == int(Priority.HIGH)

    @pytest.mark.asyncio
    async def test_list_pinned_empty(self, legacy_storage):
        """Zero pinned fibers resolves at prepare time without raising."""
        assert await legacy_storage.list_pinned_fibers() == []


# --------------------------------------------------------------------------- #
# #28 — get_fiber_stage_counts schema fix
# --------------------------------------------------------------------------- #
class TestFiberStageCounts:
    @pytest.mark.asyncio
    async def test_stage_counts_from_maturations_legacy(self, legacy_storage):
        brain_id = legacy_storage._get_brain_id()
        f1 = await _make_fiber(legacy_storage, "f1")
        f2 = await _make_fiber(legacy_storage, "f2")
        await legacy_storage.save_maturation(
            MaturationRecord(
                fiber_id=f1.id,
                brain_id=brain_id,
                stage=MemoryStage.SEMANTIC,
                stage_entered_at=utcnow(),
                rehearsal_count=3,
                reinforcement_timestamps=(),
            )
        )
        await legacy_storage.save_maturation(
            MaturationRecord(
                fiber_id=f2.id,
                brain_id=brain_id,
                stage=MemoryStage.SEMANTIC,
                stage_entered_at=utcnow(),
                rehearsal_count=3,
                reinforcement_timestamps=(),
            )
        )

        counts = await legacy_storage.get_fiber_stage_counts(brain_id)
        assert counts.get(MemoryStage.SEMANTIC.value) == 2

    @pytest.mark.asyncio
    async def test_stage_counts_from_maturations_unified(self, sql_storage):
        brain_id = sql_storage._get_brain_id()
        fiber = await _make_fiber(sql_storage, "uf1")
        await sql_storage.save_maturation(
            MaturationRecord(
                fiber_id=fiber.id,
                brain_id=brain_id,
                stage=MemoryStage.EPISODIC,
                stage_entered_at=utcnow(),
                rehearsal_count=1,
                reinforcement_timestamps=(),
            )
        )
        counts = await sql_storage.get_fiber_stage_counts(brain_id)
        assert counts.get(MemoryStage.EPISODIC.value) == 1


# --------------------------------------------------------------------------- #
# #59 — add_project narrow catch
# --------------------------------------------------------------------------- #
class TestAddProjectNarrowCatch:
    @pytest.mark.asyncio
    async def test_duplicate_raises_already_exists(self, sql_storage):
        from neural_memory.core.project import Project

        proj = Project.create(name="proj-a")
        await sql_storage.add_project(proj)
        with pytest.raises(ValueError, match="already exists"):
            await sql_storage.add_project(proj)

    @pytest.mark.asyncio
    async def test_non_integrity_error_propagates(self, sql_storage):
        """A non-integrity failure (e.g. a transient driver error) must NOT be
        masked as 'already exists'."""
        from neural_memory.core.project import Project

        proj = Project.create(name="proj-b")

        class _BoomError(RuntimeError):
            pass

        async def _boom(*_a, **_k):
            raise _BoomError("connection reset by peer")

        original = sql_storage._dialect.execute
        sql_storage._dialect.execute = _boom  # type: ignore[method-assign]
        try:
            with pytest.raises(_BoomError):
                await sql_storage.add_project(proj)
        finally:
            sql_storage._dialect.execute = original  # type: ignore[method-assign]
