"""Read-only connection pool for parallel SQLite reads under WAL mode.

WAL (Write-Ahead Logging) allows multiple concurrent readers alongside
a single writer. This pool provides dedicated read connections so that
asyncio.gather() calls actually execute in parallel (each connection
runs in its own aiosqlite thread).

Writer connection remains the main ``_conn`` on SQLiteStorage.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# Default pool size — 3 readers is a good balance for typical workloads.
DEFAULT_POOL_SIZE = 3


class ReadPool:
    """Pool of read-only SQLite connections for parallel query execution.

    Each connection runs in its own thread (via aiosqlite), enabling
    genuine parallel reads under WAL mode. Connections are created on
    ``initialize()`` and reused via round-robin acquisition.

    Attributes:
        _db_path: Path to the SQLite database file.
        _pool_size: Number of reader connections.
        _connections: List of open reader connections.
        _index: Round-robin counter for connection selection.
    """

    def __init__(self, db_path: Path, pool_size: int = DEFAULT_POOL_SIZE) -> None:
        self._db_path = db_path
        self._pool_size = max(1, pool_size)
        self._connections: list[aiosqlite.Connection] = []
        self._index = 0
        # Per-connection in-use depth. Used by acquire() to prefer the
        # least-busy reader so concurrent asyncio.gather() readers spread
        # across distinct connections instead of aliasing one (fix for #64).
        # This is best-effort accounting: the cursor context managers that
        # consume the connection increment/decrement via connection().
        self._inuse: list[int] = []

    async def initialize(self) -> None:
        """Create and configure all reader connections."""
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=30000")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-8000")
            await conn.execute("PRAGMA query_only=ON")
            self._connections.append(conn)
            self._inuse.append(0)

        logger.debug("ReadPool: initialized %d reader connections", self._pool_size)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a pooled reader for the duration of the ``async with`` block.

        This is the ONLY safe way to use a pooled connection as an
        ``async with`` target: on exit the connection is RELEASED back to
        the pool, NEVER closed. Using the bare ``aiosqlite.Connection`` from
        :meth:`acquire` as an ``async with`` target would call its
        ``__aexit__`` → ``close()`` and permanently destroy a pooled reader
        (the #9/#10 bug). Always use this CM (or ``async with conn.execute``)
        instead.

        While the block is active the connection's in-use count is raised so
        :meth:`acquire` steers concurrent readers to other connections,
        preserving parallelism (#64).

        Usage::

            async with pool.connection() as db:
                async with db.execute(sql, params) as cursor:
                    ...

        Raises:
            RuntimeError: If the pool has not been initialized.
        """
        conn, idx = self._acquire_indexed()
        self._inuse[idx] += 1
        try:
            yield conn
        finally:
            if self._inuse[idx] > 0:
                self._inuse[idx] -= 1

    def _acquire_indexed(self) -> tuple[aiosqlite.Connection, int]:
        if not self._connections:
            raise RuntimeError("ReadPool not initialized. Call initialize() first.")
        # Prefer the least-busy connection; break ties round-robin so idle
        # pools still rotate evenly.
        min_load = min(self._inuse)
        for offset in range(self._pool_size):
            idx = (self._index + offset) % self._pool_size
            if self._inuse[idx] == min_load:
                self._index = (idx + 1) % self._pool_size
                return self._connections[idx], idx
        # Unreachable: min_load is always present.
        idx = self._index % self._pool_size
        self._index += 1
        return self._connections[idx], idx

    def acquire(self) -> aiosqlite.Connection:
        """Acquire a reader connection (least-busy, round-robin tie-break).

        Synchronous fire-and-forget contract: the returned connection is a
        long-lived pooled connection that the caller must NOT close. Wrap
        the query in ``async with conn.execute(...) as cursor`` (the cursor
        CM is safe). Do NOT use the returned connection itself as an
        ``async with`` target — use :meth:`connection` for that (fix #9/#10).

        Returns:
            An open read-only aiosqlite connection.

        Raises:
            RuntimeError: If the pool has not been initialized.
        """
        conn, _ = self._acquire_indexed()
        return conn

    @property
    def size(self) -> int:
        """Number of active reader connections."""
        return len(self._connections)

    async def close(self) -> None:
        """Close all reader connections."""
        for conn in self._connections:
            try:
                await conn.close()
            except Exception:
                logger.debug("ReadPool: error closing connection", exc_info=True)
        self._connections.clear()
        self._inuse.clear()
        self._index = 0
        logger.debug("ReadPool: all reader connections closed")
