"""Regression: a cancelled caller must not kill the SQLite write worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def dialect(tmp_path) -> AsyncIterator[SQLiteDialect]:
    write_dialect = SQLiteDialect(tmp_path / "wq.db")
    await write_dialect.initialize()
    await write_dialect.execute_write("CREATE TABLE t (x INTEGER)")
    yield write_dialect
    await write_dialect.close()


async def test_cancelled_writer_does_not_kill_worker(dialect: SQLiteDialect) -> None:
    task = asyncio.create_task(dialect.execute_write("INSERT INTO t VALUES (1)"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(dialect.execute_write("INSERT INTO t VALUES (2)"), timeout=5)
    assert dialect._write_task is not None and not dialect._write_task.done()
