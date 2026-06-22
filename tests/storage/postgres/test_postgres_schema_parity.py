"""Regression tests for the PostgreSQL schema/runtime TYPE ALIGNMENT cluster.

Covers the G2-pg-schema audit findings:

* #2/#3/#4/#5 — ``PostgresDialect.get_schema_ddl()`` must emit a physically
  Postgres-correct schema (BIGINT content_hash, TIMESTAMPTZ datetimes,
  ``content_tsv``/``summary_tsv`` generated columns) rather than a
  text-mangled SQLite schema.
* #14 — ``Fiber.essence`` persists on the native Postgres backend.
* #48 — ``Fiber.last_ghost_shown_at`` persists on the native backend.
* #25 — ``Neuron.ephemeral`` persists and ``find_neurons`` honors the filter.
* #49 — neurons lifecycle columns exist on the native backend.
* #73 — ``typed_memories.project_id`` FK nulls on project delete.
* #24 — list[float] embeddings encode via the registered pgvector codec.
* #61 — ``promote_memory_type`` accepts an ISO ``new_expires_at`` string.

The native-backend tests reuse the ``storage`` fixture from ``conftest.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

# ── #2/#3/#4/#5: dialect get_schema_ddl() type alignment ──────────────────


def test_get_schema_ddl_aligns_types() -> None:
    """The dialect DDL must use Postgres-correct physical types.

    No DB required — this is a pure string check on the generated DDL.
    """
    from neural_memory.storage.sql.postgres_dialect import PostgresDialect

    ddl = PostgresDialect().get_schema_ddl()

    # #4: 64-bit SimHash needs BIGINT, not 32-bit INTEGER.
    assert "content_hash BIGINT" in ddl
    assert "content_hash INTEGER" not in ddl

    # #5: datetime columns bound via serialize_dt must be TIMESTAMPTZ.
    assert "created_at TIMESTAMPTZ" in ddl
    assert "created_at TEXT" not in ddl
    assert "last_ghost_shown_at TIMESTAMPTZ" in ddl

    # ISO-string columns MUST stay TEXT (cognitive predicted_at, sync updated_at).
    assert "predicted_at TEXT" in ddl
    assert "updated_at TEXT DEFAULT ''" in ddl

    # #3: FTS generated columns + GIN indexes referenced by fts_*_query.
    assert "content_tsv" in ddl
    assert "summary_tsv" in ddl
    assert "idx_neurons_content_tsv" in ddl
    assert "idx_fibers_summary_tsv" in ddl


# ── #25: ephemeral neuron flag ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ephemeral_neuron_persists_and_filters(storage: Any, make_neuron: Any) -> None:
    """ephemeral round-trips and find_neurons(ephemeral=...) filters correctly."""
    from neural_memory.core.neuron import Neuron, NeuronType

    eph = Neuron.create(type=NeuronType.CONCEPT, content="scratch note", ephemeral=True)
    perm = Neuron.create(type=NeuronType.CONCEPT, content="permanent note", ephemeral=False)
    await storage.add_neuron(eph)
    await storage.add_neuron(perm)

    got_eph = await storage.get_neuron(eph.id)
    got_perm = await storage.get_neuron(perm.id)
    assert got_eph is not None and got_eph.ephemeral is True
    assert got_perm is not None and got_perm.ephemeral is False

    only_perm = await storage.find_neurons(ephemeral=False)
    perm_ids = {n.id for n in only_perm}
    assert perm.id in perm_ids
    assert eph.id not in perm_ids

    only_eph = await storage.find_neurons(ephemeral=True)
    eph_ids = {n.id for n in only_eph}
    assert eph.id in eph_ids
    assert perm.id not in eph_ids


# ── #14 / #48: fiber essence + ghost throttle ─────────────────────────────


@pytest.mark.asyncio
async def test_fiber_essence_and_ghost_persist(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """essence (#14) and last_ghost_shown_at (#48) survive a write/read round-trip."""
    import dataclasses

    from neural_memory.core.fiber import Fiber
    from neural_memory.utils.timeutils import utcnow

    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = Fiber.create(
        neuron_ids={n.id for n in sample_neurons[:3]},
        synapse_ids={s.id for s in sample_synapses[:2]},
        anchor_neuron_id=sample_neurons[0].id,
        summary="Full fiber summary",
        essence="One-sentence distillation.",
    )
    await storage.add_fiber(fiber)

    got = await storage.get_fiber(fiber.id)
    assert got is not None
    assert got.essence == "One-sentence distillation."
    assert got.last_ghost_shown_at is None

    # Persist a ghost-shown timestamp via update_fiber.
    ts = utcnow()
    updated = dataclasses.replace(got, last_ghost_shown_at=ts)
    await storage.update_fiber(updated)

    re_got = await storage.get_fiber(fiber.id)
    assert re_got is not None
    assert re_got.last_ghost_shown_at is not None
    assert abs((re_got.last_ghost_shown_at - ts).total_seconds()) < 1.0
    assert re_got.essence == "One-sentence distillation."


# ── #49: neuron lifecycle columns exist ───────────────────────────────────


@pytest.mark.asyncio
async def test_neuron_lifecycle_columns_exist(storage: Any) -> None:
    """last_accessed_at / lifecycle_state / frozen must exist on native Postgres."""
    rows = await storage._query_ro(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'neurons'"
    )
    cols = {r["column_name"] for r in rows}
    assert {"last_accessed_at", "lifecycle_state", "frozen", "ephemeral"} <= cols


# ── #73: typed_memories project_id composite FK to projects ───────────────


@pytest.mark.asyncio
async def test_typed_memory_project_fk_exists(storage: Any) -> None:
    """The composite (brain_id, project_id) -> projects FK exists on Postgres (#73).

    SQLite declares ``FOREIGN KEY (brain_id, project_id) REFERENCES
    projects(brain_id, id) ON DELETE SET NULL``; the native Postgres schema
    previously omitted it entirely (deleted projects left dangling
    project_id values). This verifies the constraint now exists with the
    same SET NULL action, restoring schema parity.

    Note: because ``brain_id`` is part of the composite FK AND is NOT NULL,
    a direct project delete raises a NOT NULL violation on BOTH backends
    (SET NULL would null brain_id too) — so we assert the constraint's
    existence/shape rather than exercise the unreachable cascade.
    """
    row = await storage._query_one(
        """
        SELECT confdeltype
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_class fref ON fref.oid = con.confrelid
        WHERE rel.relname = 'typed_memories'
          AND fref.relname = 'projects'
          AND con.contype = 'f'
        """
    )
    assert row is not None, "typed_memories -> projects FK is missing"
    # confdeltype 'n' == ON DELETE SET NULL (pg returns a single-char "char"
    # type, surfaced by asyncpg as bytes).
    confdeltype = row["confdeltype"]
    if isinstance(confdeltype, bytes):
        confdeltype = confdeltype.decode()
    assert confdeltype == "n"


# ── #24: pgvector embedding codec ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_embedding_codec_round_trip(storage: Any) -> None:
    """A list[float] embedding encodes to the vector column and search works (#24)."""
    from neural_memory.core.neuron import Neuron, NeuronType

    vec = [0.1] * 384
    n = Neuron.create(
        type=NeuronType.CONCEPT,
        content="vector neuron",
        metadata={"_embedding": vec},
    )
    # Must not raise on encode.
    await storage.add_neuron(n)

    results = await storage.find_neurons_by_embedding(vec, limit=5)
    ids = {neuron.id for neuron, _score in results}
    assert n.id in ids


# ── #61: promote_memory_type with ISO string expires_at ───────────────────


@pytest.mark.asyncio
async def test_promote_memory_type_accepts_iso_string(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """promote_memory_type must parse an ISO new_expires_at str, not bind it raw (#61)."""
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.memory_types import (
        Confidence,
        MemoryType,
        Priority,
        Provenance,
        TypedMemory,
    )

    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = Fiber.create(
        neuron_ids={n.id for n in sample_neurons[:2]},
        synapse_ids={s.id for s in sample_synapses[:1]},
        anchor_neuron_id=sample_neurons[0].id,
        summary="promote me",
    )
    await storage.add_fiber(fiber)

    tm = TypedMemory(
        fiber_id=fiber.id,
        memory_type=MemoryType.FACT,
        priority=Priority.NORMAL,
        provenance=Provenance(source="test", confidence=Confidence.MEDIUM),
    )
    await storage.add_typed_memory(tm)

    ok = await storage.promote_memory_type(
        fiber.id,
        MemoryType.INSIGHT,
        new_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert ok is True

    got = await storage.get_typed_memory(fiber.id)
    assert got is not None
    assert got.memory_type == MemoryType.INSIGHT
    assert got.expires_at is not None
    assert got.expires_at.year == 2099
