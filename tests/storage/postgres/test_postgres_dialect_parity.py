"""Regression tests for the G3-dialect-parity cluster on the unified
``SQLStorage(PostgresDialect(...))`` backend.

These guard query-level behaviors that previously crashed or mis-filtered on
Postgres while working on SQLite:

* #6  — JSON tag filter binds a dialect-correct param shape (``'["work"]'``)
        and casts the TEXT tag column to jsonb, instead of binding a bare tag
        into ``::jsonb`` (which raised ``invalid input syntax for type json``).
        Covers both ``find_fibers`` and ``find_fibers_batch``.
* #18 — ``get_synapses_for_neurons(direction='both')`` reuses the single
        ``= ANY($2)`` array param instead of passing too many args (which
        raised "the server expects 2 arguments, 3 were passed").
* #19 — ``promote_memory_type`` parses ``new_expires_at`` to a datetime and
        serializes via the dialect, instead of binding a bare ISO string into
        the TIMESTAMPTZ ``expires_at`` column (DataError).
* #20 — ``suggest_neurons`` escapes LIKE wildcards in the Postgres ILIKE
        branch so ``user_`` matches literally rather than as a wildcard.

The fixture builds a real ``SQLStorage(PostgresDialect()).initialize()`` (the
schema is now physically Postgres-correct after the G2 cluster) bound to a
unique brain, and tears the brain down afterward.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

POSTGRES_HOST = os.environ.get("POSTGRES_TEST_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_TEST_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_TEST_DB", "neuralmemory_test")
POSTGRES_USER = os.environ.get("POSTGRES_TEST_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_TEST_PASSWORD", "")


async def _reset_public_schema() -> None:
    """Drop + recreate the ``public`` schema for clean test isolation.

    The unified dialect path and the legacy native backend share the same DB
    but define *physically different* tables (Theme B in the audit). Resetting
    the schema before and after this module keeps the dialect-created tables
    from poisoning the native ``ensure_schema`` migrations in sibling tests.
    """
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
    """A real ``SQLStorage(PostgresDialect())`` bound to a unique brain."""
    from neural_memory.core.brain import Brain
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

    brain = Brain.create(name=f"parity_{uuid.uuid4().hex[:12]}")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    try:
        yield store
    finally:
        await store.close()
        # Leave a clean schema so sibling native-backend tests can rebuild.
        try:
            await _reset_public_schema()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_find_fibers_tag_filter_does_not_crash(unified_pg: Any) -> None:
    """#6: ``find_fibers(tags=...)`` returns the tagged fiber on Postgres."""
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronType

    s = unified_pg
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="Python")
    await s.add_neuron(n1)
    tagged = Fiber.create(
        neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id, agent_tags={"work"}
    )
    untagged = Fiber.create(
        neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id, agent_tags={"home"}
    )
    await s.add_fiber(tagged)
    await s.add_fiber(untagged)

    matches = await s.find_fibers(tags={"work"})
    ids = {f.id for f in matches}
    assert tagged.id in ids
    assert untagged.id not in ids

    # No-match tag must not crash and must return nothing.
    assert await s.find_fibers(tags={"does-not-exist"}) == []


@pytest.mark.asyncio
async def test_find_fibers_batch_tag_filter_does_not_crash(unified_pg: Any) -> None:
    """#6: ``find_fibers_batch`` had NO post-filter, so the SQL must be valid."""
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronType

    s = unified_pg
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="Python")
    await s.add_neuron(n1)
    f = Fiber.create(
        neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id, agent_tags={"work"}
    )
    await s.add_fiber(f)

    res = await s.find_fibers_batch([n1.id], tags={"work"})
    assert f.id in {x.id for x in res}


@pytest.mark.asyncio
async def test_get_synapses_for_neurons_direction_both(unified_pg: Any) -> None:
    """#18: ``direction='both'`` must not raise an arg-count error on Postgres."""
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.core.synapse import Synapse, SynapseType

    s = unified_pg
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="A")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="B")
    await s.add_neuron(n1)
    await s.add_neuron(n2)
    syn = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
    await s.add_synapse(syn)

    result = await s.get_synapses_for_neurons([n1.id, n2.id], direction="both")
    # The synapse appears under both its source and target.
    assert any(x.id == syn.id for x in result[n1.id])
    assert any(x.id == syn.id for x in result[n2.id])


@pytest.mark.asyncio
async def test_promote_memory_type_accepts_iso_expiry(unified_pg: Any) -> None:
    """#19: a bare ISO ``new_expires_at`` string must bind into TIMESTAMPTZ."""
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.memory_types import MemoryType, TypedMemory
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.storage.sql.mixins.typed_memory import TypedMemoryMixin

    s = unified_pg
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="Python")
    await s.add_neuron(n1)
    f = Fiber.create(neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id)
    await s.add_fiber(f)

    tm = TypedMemory.create(fiber_id=f.id, memory_type=MemoryType.CONTEXT)
    # Call the real mixin method directly: the public SQLStorage.add_typed_memory
    # resolves to a BrainOpsMixin protocol stub (an unrelated MRO quirk), but
    # promote_memory_type — the method under test — resolves to TypedMemoryMixin.
    await TypedMemoryMixin.add_typed_memory(s, tm)

    promoted = await s.promote_memory_type(
        f.id, MemoryType.FACT, new_expires_at="2030-01-01T00:00:00+00:00"
    )
    assert promoted is True

    got = await s.get_typed_memory(f.id)
    assert got is not None
    assert got.expires_at is not None
    assert got.expires_at.year == 2030


@pytest.mark.asyncio
async def test_suggest_neurons_escapes_like_wildcards(unified_pg: Any) -> None:
    """#20: ``user_`` must match literally, not as an underscore wildcard."""
    from neural_memory.core.neuron import Neuron, NeuronType

    s = unified_pg
    literal = Neuron.create(type=NeuronType.CONCEPT, content="user_admin")
    wildcard = Neuron.create(type=NeuronType.CONCEPT, content="userXadmin")
    await s.add_neuron(literal)
    await s.add_neuron(wildcard)

    suggestions = await s.suggest_neurons("user_", limit=10)
    contents = {r["content"] for r in suggestions}
    assert "user_admin" in contents
    # The underscore must be literal, so userXadmin must NOT match.
    assert "userXadmin" not in contents
