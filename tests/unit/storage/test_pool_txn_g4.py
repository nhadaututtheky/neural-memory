"""Regression tests for the G4-pool-txn cluster.

Connection-pool lifecycle + atomic (transactional) writes. All findings here
have a *visible* failure on the SQLite path or in pure Python (no Postgres
server required); the Postgres-only findings (#17 ContextVar txn, #60 native
import_brain atomicity) are exercised by tests/storage/postgres when a server
is available.

* #9/#10 — ``ReadPool.acquire()`` returned a pooled connection that callers
           closed via ``async with`` (its ``__aexit__`` calls ``close()``),
           permanently disabling pooled readers after <= pool_size calls.
           ``get_neuron_hashes`` / ``has_neuron_by_content_hash`` must NOT
           close the pooled connection, and the new ``connection()`` CM must
           release (not close) on exit.
* #64    — ``ReadPool`` must not alias the same connection to concurrent
           readers; the least-busy ``connection()`` CM spreads load.
* #54    — ``refresh_hot_index`` must be atomic: a bad item rolls the whole
           DELETE+INSERT back instead of wiping the index.
* #55    — ``update_fiber`` junction refresh must be atomic: a failure must not
           leave ``fiber_neurons`` empty while the fibers row claims members.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.read_pool import ReadPool
from neural_memory.storage.sql import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── ReadPool contract (#9/#10/#64) ──────────────────────────────────


@pytest.fixture
async def read_pool() -> Any:
    td = tempfile.mkdtemp()
    db_path = Path(td) / "pool.db"
    # Seed a real table so reader connections have something to query.
    import aiosqlite

    conn = await aiosqlite.connect(db_path)
    await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    await conn.execute("INSERT INTO t (v) VALUES ('a'), ('b')")
    await conn.commit()
    await conn.close()

    pool = ReadPool(db_path, pool_size=3)
    await pool.initialize()
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_connection_cm_does_not_close_pooled_conn(read_pool: ReadPool) -> None:
    """#9/#10: the ``connection()`` CM releases — never closes — on exit."""
    # Use every connection many more times than the pool size; before the fix
    # this closed pooled readers one-by-one and the 4th+ use raised
    # "Cannot operate on a closed database".
    for _ in range(10):
        async with read_pool.connection() as db:
            async with db.execute("SELECT COUNT(*) FROM t") as cur:
                row = await cur.fetchone()
                assert row[0] == 2

    # All three connections are still open and usable.
    assert read_pool.size == 3
    conn = read_pool.acquire()
    async with conn.execute("SELECT 1") as cur:
        assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_acquire_returns_open_connection_reusable(read_pool: ReadPool) -> None:
    """#9/#10: bare ``acquire()`` (fire-and-forget) returns a live connection
    each time; using ``conn.execute`` as a CM (cursor, not connection) keeps
    the pooled connection open."""
    for _ in range(12):  # > pool_size * any rotation
        conn = read_pool.acquire()
        async with conn.execute("SELECT v FROM t ORDER BY id LIMIT 1") as cur:
            assert (await cur.fetchone())[0] == "a"
    assert read_pool.size == 3


@pytest.mark.asyncio
async def test_concurrent_connections_spread_across_pool(read_pool: ReadPool) -> None:
    """#64: concurrent ``connection()`` holders get distinct connections."""
    import asyncio

    seen: list[int] = []

    async def hold() -> None:
        async with read_pool.connection() as db:
            seen.append(id(db))
            # Hold the connection while the siblings also acquire.
            await asyncio.sleep(0.05)

    await asyncio.gather(hold(), hold(), hold())
    # Three concurrent holders, pool_size 3 → three distinct connections.
    assert len(set(seen)) == 3


@pytest.mark.asyncio
async def test_get_neuron_hashes_does_not_break_pool() -> None:
    """#9/#10 end-to-end: repeated get_neuron_hashes on real SQLiteStorage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(Path(tmpdir) / "t.db")
        await storage.initialize()
        brain = Brain.create(name="pool-test")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="python"))

        # Many more calls than pool_size (3). Pre-fix, pooled readers closed
        # one-by-one and a later read raised on a closed connection.
        for _ in range(10):
            hashes = await storage.get_neuron_hashes()
            assert isinstance(hashes, list)
            await storage.has_neuron_by_content_hash(12345)

        # A subsequent normal read still works (the writer/readers are intact).
        neurons = await storage.find_neurons(content_exact="python")
        assert len(neurons) == 1
        await storage.close()


# ── Atomic writes (#54/#55) ─────────────────────────────────────────


@pytest.fixture
async def sql_storage() -> Any:
    td = tempfile.mkdtemp()
    s = SQLStorage(SQLiteDialect(db_path=os.path.join(td, "t.db")))
    await s.initialize()
    brain = Brain.create(name="g4-txn")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_refresh_hot_index_atomic_on_bad_item(sql_storage: Any) -> None:
    """#54: a malformed item rolls back the whole DELETE+INSERT."""
    good = [
        {
            "slot": i,
            "category": "c",
            "neuron_id": f"n{i}",
            "summary": "s",
            "confidence": 0.5,
            "score": 1.0,
        }
        for i in range(3)
    ]
    await sql_storage.refresh_hot_index(good)
    assert len(await sql_storage.get_hot_index(limit=10)) == 3

    # Second refresh contains a bad item (missing 'slot' → KeyError mid-loop).
    bad = [
        {
            "slot": 0,
            "category": "c",
            "neuron_id": "x",
            "summary": "s",
            "confidence": 0.5,
            "score": 1.0,
        },
        {"category": "c", "neuron_id": "y", "summary": "s", "score": 1.0},  # no slot
    ]
    with pytest.raises(KeyError):
        await sql_storage.refresh_hot_index(bad)

    # The DELETE must have been rolled back together with the partial INSERT:
    # the original 3 rows survive (index NOT wiped/partial).
    surviving = await sql_storage.get_hot_index(limit=10)
    assert len(surviving) == 3


async def _junction_neuron_ids(storage: Any, fiber_id: str) -> set[str]:
    """Read the fiber_neurons junction rows directly (bypasses find_fibers)."""
    d = storage._dialect
    rows = await d.fetch_all(
        f"SELECT neuron_id FROM fiber_neurons WHERE fiber_id = {d.ph(1)}",  # noqa: S608
        (fiber_id,),
    )
    return {r["neuron_id"] for r in rows}


@pytest.mark.asyncio
async def test_update_fiber_junction_atomic(sql_storage: Any) -> None:
    """#55: update_fiber's UPDATE + junction DELETE + re-INSERT is one txn.

    A successful update leaves the junction consistent with the fibers row's
    neuron_ids (the junction is fully swapped, never left empty/partial).
    """
    import dataclasses

    n1 = Neuron.create(type=NeuronType.CONCEPT, content="alpha")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="beta")
    n3 = Neuron.create(type=NeuronType.CONCEPT, content="gamma")
    for n in (n1, n2, n3):
        await sql_storage.add_neuron(n)

    fiber = Fiber.create(
        neuron_ids={n1.id, n2.id},
        synapse_ids=set(),
        anchor_neuron_id=n1.id,
        summary="link",
    )
    await sql_storage.add_fiber(fiber)
    assert await _junction_neuron_ids(sql_storage, fiber.id) == {n1.id, n2.id}

    # Replace members entirely. The DELETE + re-INSERT must commit together so
    # the junction ends up as exactly {n2, n3} — never empty, never partial.
    fiber = dataclasses.replace(fiber, neuron_ids={n2.id, n3.id})
    await sql_storage.update_fiber(fiber)
    assert await _junction_neuron_ids(sql_storage, fiber.id) == {n2.id, n3.id}

    # And the persisted fiber row agrees with the junction.
    reloaded = await sql_storage.get_fiber(fiber.id)
    assert reloaded is not None
    assert reloaded.neuron_ids == {n2.id, n3.id}
