"""PostgreSQL base mixin — connection pool, query helpers, pgvector setup."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PostgresBaseMixin:
    """Base mixin for PostgreSQL storage operations.

    Requires: _pool, _current_brain_id, set_brain().
    """

    _pool: Any  # asyncpg.Pool
    _current_brain_id: str | None

    def _get_brain_id(self) -> str:
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    async def _query(self, sql: str, *args: Any, timeout: float = 30.0) -> Any:
        """Execute a write query. Returns last result."""
        import asyncpg

        conn: asyncpg.Connection = await self._pool.acquire()
        try:
            return await conn.execute(sql, *args, timeout=timeout)
        finally:
            await self._pool.release(conn)

    async def _query_ro(self, sql: str, *args: Any, timeout: float = 30.0) -> list[Any]:
        """Execute a read query. Returns list of records."""
        import asyncpg

        conn: asyncpg.Connection = await self._pool.acquire()
        try:
            return list(await conn.fetch(sql, *args, timeout=timeout))
        finally:
            await self._pool.release(conn)

    async def _query_one(self, sql: str, *args: Any, timeout: float = 30.0) -> Any | None:
        """Execute a read query expecting at most one row."""
        import asyncpg

        conn: asyncpg.Connection = await self._pool.acquire()
        try:
            return await conn.fetchrow(sql, *args, timeout=timeout)
        finally:
            await self._pool.release(conn)

    async def _executemany(
        self, sql: str, args_list: list[tuple[Any, ...]], timeout: float = 30.0
    ) -> None:
        """Execute a parameterized query for each args tuple in one connection."""
        import asyncpg

        conn: asyncpg.Connection = await self._pool.acquire()
        try:
            await conn.executemany(sql, args_list, timeout=timeout)
        finally:
            await self._pool.release(conn)

    @staticmethod
    def _serialize_metadata(meta: dict[str, Any]) -> str:
        return json.dumps(meta)

    @staticmethod
    def _dt_to_str(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None
