"""Read-only connection pool for parallel SQLite reads under WAL mode.

WAL (Write-Ahead Logging) allows multiple concurrent readers alongside
a single writer. This pool provides dedicated read connections so that
asyncio.gather() calls actually execute in parallel (each connection
runs in its own aiosqlite thread).

Writer connection remains the main ``_conn`` on SQLiteStorage.
"""

from __future__ import annotations

import logging
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

    async def initialize(self) -> None:
        """Create and configure all reader connections."""
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-8000")
            await conn.execute("PRAGMA query_only=ON")
            self._connections.append(conn)

        logger.debug("ReadPool: initialized %d reader connections", self._pool_size)

    def acquire(self) -> aiosqlite.Connection:
        """Acquire a reader connection via round-robin.

        Returns:
            An open read-only aiosqlite connection.

        Raises:
            RuntimeError: If the pool has not been initialized.
        """
        if not self._connections:
            raise RuntimeError("ReadPool not initialized. Call initialize() first.")
        conn = self._connections[self._index % self._pool_size]
        self._index += 1
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
        self._index = 0
        logger.debug("ReadPool: all reader connections closed")
