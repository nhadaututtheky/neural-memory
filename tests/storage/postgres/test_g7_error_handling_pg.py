"""Postgres regression tests for the G7-errors cluster on the unified
``SQLStorage(PostgresDialect(...))`` backend.

Guards the dialect-portable fixes:

* #22 — ``save_maturation`` issues a real UPSERT keyed on
        ``(brain_id, fiber_id)``, so a second save advances the stage instead
        of swallowing a UNIQUE violation (which blocked EPISODIC->SEMANTIC).
* #28 — ``get_fiber_stage_counts`` reads ``stage`` from ``memory_maturations``
        (the column does not exist on ``fibers``).
* #59 — ``add_project`` raises ``ValueError('already exists')`` only on a
        genuine integrity violation, and is actually reachable on the unified
        backend (the BrainOpsMixin protocol stub no longer shadows it via MRO).
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import Any

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.project import Project
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.utils.timeutils import utcnow

POSTGRES_HOST = os.environ.get("POSTGRES_TEST_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_TEST_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_TEST_DB", "neuralmemory_test")
POSTGRES_USER = os.environ.get("POSTGRES_TEST_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_TEST_PASSWORD", "")


async def _reset_public_schema() -> None:
    import asyncpg

    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD or None,
    )
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def unified_pg() -> Any:
    from neural_memory.storage.sql import SQLStorage
    from neural_memory.storage.sql.postgres_dialect import PostgresDialect

    await _reset_public_schema()

    dialect = PostgresDialect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    store = SQLStorage(dialect)
    await store.initialize()

    brain = Brain.create(name=f"g7_{uuid.uuid4().hex[:12]}")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    try:
        yield store
    finally:
        await store.close()
        try:
            await _reset_public_schema()
        except Exception:
            pass


async def _make_fiber(store: Any, summary: str = "fiber") -> Fiber:
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


@pytest.mark.asyncio
async def test_save_maturation_upserts_on_postgres(unified_pg: Any) -> None:
    """#22: second save advances the stage rather than swallowing a UNIQUE
    violation."""
    fiber = await _make_fiber(unified_pg, "matures-pg")
    brain_id = unified_pg._get_brain_id()

    await unified_pg.save_maturation(
        MaturationRecord(
            fiber_id=fiber.id,
            brain_id=brain_id,
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow() - timedelta(days=10),
            rehearsal_count=1,
            reinforcement_timestamps=(1.0,),
        )
    )
    await unified_pg.save_maturation(
        MaturationRecord(
            fiber_id=fiber.id,
            brain_id=brain_id,
            stage=MemoryStage.SEMANTIC,
            stage_entered_at=utcnow(),
            rehearsal_count=5,
            reinforcement_timestamps=(1.0, 2.0, 3.0),
        )
    )

    loaded = await unified_pg.get_maturation(fiber.id)
    assert loaded is not None
    assert loaded.stage == MemoryStage.SEMANTIC
    assert loaded.rehearsal_count == 5


@pytest.mark.asyncio
async def test_get_fiber_stage_counts_on_postgres(unified_pg: Any) -> None:
    """#28: stage counts come from memory_maturations, not fibers.stage."""
    brain_id = unified_pg._get_brain_id()
    f1 = await _make_fiber(unified_pg, "pg-f1")
    f2 = await _make_fiber(unified_pg, "pg-f2")
    for fid in (f1.id, f2.id):
        await unified_pg.save_maturation(
            MaturationRecord(
                fiber_id=fid,
                brain_id=brain_id,
                stage=MemoryStage.SEMANTIC,
                stage_entered_at=utcnow(),
                rehearsal_count=3,
                reinforcement_timestamps=(),
            )
        )

    counts = await unified_pg.get_fiber_stage_counts(brain_id)
    assert counts.get(MemoryStage.SEMANTIC.value) == 2


@pytest.mark.asyncio
async def test_add_project_reachable_and_narrow_on_postgres(unified_pg: Any) -> None:
    """#59: add_project works on the unified backend (no MRO stub shadowing)
    and only the genuine duplicate raises 'already exists'."""
    proj = Project.create(name="pg-proj")
    pid = await unified_pg.add_project(proj)
    assert pid == proj.id

    with pytest.raises(ValueError, match="already exists"):
        await unified_pg.add_project(proj)
