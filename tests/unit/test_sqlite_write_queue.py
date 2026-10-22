"""SQLite writes on the real path share one worker and one connection."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator

import pytest

from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def dialect(tmp_path) -> AsyncIterator[SQLiteDialect]:
    write_dialect = SQLiteDialect(tmp_path / "writes.db")
    await write_dialect.initialize()
    yield write_dialect
    await write_dialect.close()


def _worker_alive(dialect: SQLiteDialect) -> bool:
    return dialect._write_task is not None and not dialect._write_task.done()


async def _values(dialect: SQLiteDialect) -> list[int]:
    rows = await dialect.fetch_all("SELECT x FROM t ORDER BY x")
    return [int(row["x"]) for row in rows]


async def test_concurrent_executes_all_commit(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    count = 40
    await asyncio.gather(*(dialect.execute("INSERT INTO t VALUES (?)", (i,)) for i in range(count)))
    assert await _values(dialect) == list(range(count))
    assert _worker_alive(dialect)


async def test_cancelled_execute_does_not_kill_worker(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    task = asyncio.create_task(dialect.execute("INSERT INTO t VALUES (1)"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(dialect.execute("INSERT INTO t VALUES (2)"), timeout=5)
    found = await _values(dialect)
    assert found == [2] or found == [1, 2]
    assert _worker_alive(dialect)


async def test_failed_execute_does_not_kill_worker(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    with pytest.raises(sqlite3.Error, match="no such table"):
        await dialect.execute("INSERT INTO missing VALUES (1)")

    await dialect.execute("INSERT INTO t VALUES (4)")
    assert await _values(dialect) == [4]
    assert _worker_alive(dialect)


async def test_execute_many_and_script_persist(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    params = [1]
    await dialect.execute("INSERT INTO t VALUES (?)", params)
    assert params == [1]

    rows = [[2], [3]]
    await dialect.execute_many("INSERT INTO t VALUES (?)", rows)
    assert rows == [[2], [3]]

    await dialect.execute_script("INSERT INTO t VALUES (4); INSERT INTO t VALUES (5);")
    assert await _values(dialect) == [1, 2, 3, 4, 5]
    assert _worker_alive(dialect)


async def test_execute_count_and_insert_returning_id(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, x INTEGER)")
    inserted = await dialect.insert_returning_id(
        "INSERT INTO t (x) VALUES (?) RETURNING id",
        (7,),
    )
    assert inserted == 1
    deleted = await dialect.execute_count("DELETE FROM t WHERE x = ?", (7,))
    assert deleted == 1
    changed = await dialect.execute_returning_count("INSERT INTO t (x) VALUES (?)", (8,))
    assert changed == 1
    rows = await dialect.fetch_all("SELECT x FROM t")
    assert [int(row["x"]) for row in rows] == [8]


async def test_open_transaction_excludes_queued_write(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    started = asyncio.Event()
    release = asyncio.Event()

    async def txn() -> None:
        async with dialect.transaction():
            await dialect.execute("INSERT INTO t VALUES (1)")
            await dialect.execute_many("INSERT INTO t VALUES (?)", [(2,)])
            started.set()
            await release.wait()
            visible = await _values(dialect)
            assert visible == [1, 2]

    txn_task = asyncio.create_task(txn())
    await started.wait()
    queued = asyncio.create_task(dialect.execute("INSERT INTO t VALUES (3)"))
    queued_many = asyncio.create_task(dialect.execute_many("INSERT INTO t VALUES (?)", [(4,)]))
    queued_script = asyncio.create_task(dialect.execute_script("INSERT INTO t VALUES (5);"))
    await asyncio.sleep(0.05)
    assert not queued.done()
    assert not queued_many.done()
    assert not queued_script.done()

    release.set()
    await asyncio.wait_for(txn_task, timeout=5)
    await asyncio.wait_for(queued, timeout=5)
    await asyncio.wait_for(queued_many, timeout=5)
    await asyncio.wait_for(queued_script, timeout=5)
    assert await _values(dialect) == [1, 2, 3, 4, 5]
    assert _worker_alive(dialect)


async def test_rollback_does_not_drop_a_queued_write(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    started = asyncio.Event()
    release = asyncio.Event()

    async def txn() -> None:
        async with dialect.transaction():
            await dialect.execute("INSERT INTO t VALUES (1)")
            await dialect.execute_many("INSERT INTO t VALUES (?)", [(2,)])
            started.set()
            await release.wait()
            raise RuntimeError("boom")

    txn_task = asyncio.create_task(txn())
    await started.wait()
    queued = asyncio.create_task(dialect.execute("INSERT INTO t VALUES (3)"))
    await asyncio.sleep(0.05)
    assert not queued.done()
    release.set()

    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(txn_task, timeout=5)
    await asyncio.wait_for(queued, timeout=5)
    assert await _values(dialect) == [3]
    assert _worker_alive(dialect)


async def test_cancelled_transaction_rolls_back_and_unblocks_writer(
    dialect: SQLiteDialect,
) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    started = asyncio.Event()

    async def txn() -> None:
        async with dialect.transaction():
            await dialect.execute("INSERT INTO t VALUES (1)")
            started.set()
            await asyncio.sleep(3600)

    txn_task = asyncio.create_task(txn())
    await started.wait()
    queued = asyncio.create_task(dialect.execute("INSERT INTO t VALUES (2)"))
    await asyncio.sleep(0.05)
    assert not queued.done()
    txn_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await txn_task

    await asyncio.wait_for(queued, timeout=5)
    assert await _values(dialect) == [2]
    assert _worker_alive(dialect)


async def test_nested_transaction_is_rejected(dialect: SQLiteDialect) -> None:
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    async with dialect.transaction():
        await dialect.execute("INSERT INTO t VALUES (1)")
        with pytest.raises(RuntimeError, match="cannot be nested"):
            async with dialect.transaction():
                await dialect.execute("INSERT INTO t VALUES (2)")
        assert await _values(dialect) == [1]

    assert await _values(dialect) == [1]
    assert _worker_alive(dialect)
