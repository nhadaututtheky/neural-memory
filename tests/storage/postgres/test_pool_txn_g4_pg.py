"""Postgres regression tests for the G4-pool-txn cluster.

Requires a running pgvector Postgres (auto-skipped otherwise via conftest).

* #60 — native ``import_brain`` must be all-or-nothing. A failure midway
        (e.g. a neuron whose ``type`` is not a valid ``NeuronType``) must roll
        back the brains row + every prior insert, leaving NO partially-imported
        brain.
* #17 — ``PostgresDialect.transaction()`` must isolate the active transaction
        connection per task (ContextVar), so a concurrent task running its own
        transaction does not see / clobber it.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from neural_memory.core.brain import BrainSnapshot
from neural_memory.utils.timeutils import utcnow


@pytest.mark.asyncio
async def test_import_brain_atomic_on_bad_neuron(storage: Any, brain_id: str) -> None:
    """#60: a bad neuron type rolls the whole import back (no partial brain)."""
    target = f"{brain_id}_atomic_{uuid.uuid4().hex[:6]}"

    good_id = uuid.uuid4().hex
    snapshot = BrainSnapshot(
        brain_id=target,
        brain_name=target,
        exported_at=utcnow(),
        version="0.1.0",
        neurons=[
            {
                "id": good_id,
                "type": "concept",
                "content": "valid neuron",
                "metadata": {},
                "created_at": utcnow().isoformat(),
            },
            {
                # Invalid enum value → NeuronType(...) raises mid-loop, AFTER
                # the brains row + the first neuron have been written.
                "id": uuid.uuid4().hex,
                "type": "NOT_A_REAL_TYPE",
                "content": "boom",
                "metadata": {},
                "created_at": utcnow().isoformat(),
            },
        ],
        synapses=[],
        fibers=[],
        config={},
        metadata={"typed_memories": [], "projects": []},
    )

    with pytest.raises(ValueError):
        await storage.import_brain(snapshot, target_brain_id=target)

    # The whole import must have rolled back: no brain row, no leaked neuron.
    assert await storage.get_brain(target) is None

    storage.set_brain(target)
    try:
        leaked = await storage.find_neurons(content_exact="valid neuron")
        assert leaked == []
    finally:
        storage.set_brain(brain_id)


@pytest.mark.asyncio
async def test_dialect_transaction_isolated_across_tasks() -> None:
    """#17: concurrent transaction() blocks see independent txn connections."""
    import asyncio
    import os

    from neural_memory.storage.sql import SQLStorage
    from neural_memory.storage.sql.postgres_dialect import PostgresDialect

    host = os.environ.get("POSTGRES_TEST_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_TEST_PORT", "5432"))
    db = os.environ.get("POSTGRES_TEST_DB", "neuralmemory_test")
    user = os.environ.get("POSTGRES_TEST_USER", "postgres")
    pw = os.environ.get("POSTGRES_TEST_PASSWORD", "")

    dialect = PostgresDialect(host=host, port=port, database=db, user=user, password=pw)
    storage = SQLStorage(dialect)
    await storage.initialize()
    try:
        observed: dict[str, Any] = {}

        async def worker(tag: str) -> None:
            async with dialect.transaction():
                # Inside the block, _get_conn_or_none must return THIS task's
                # connection, not another concurrent worker's.
                conn = dialect._get_conn_or_none()
                assert conn is not None
                observed[tag] = id(conn)
                # Yield control so the sibling task interleaves while our txn
                # is open. With an instance attribute this would clobber.
                await asyncio.sleep(0.05)
                # Still our own connection after the await.
                assert dialect._get_conn_or_none() is conn
                # A trivial query routes to our txn connection.
                row = await dialect.fetch_one("SELECT 1 AS x")
                assert row["x"] == 1

        await asyncio.gather(worker("a"), worker("b"))

        # Each concurrent transaction used a DISTINCT pooled connection.
        assert observed["a"] != observed["b"]
        # Outside any transaction, the ContextVar is clear.
        assert dialect._get_conn_or_none() is None
    finally:
        await storage.close()
