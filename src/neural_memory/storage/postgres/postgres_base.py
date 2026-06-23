"""PostgreSQL base mixin — connection pool, query helpers, pgvector setup."""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Per-task transaction connection for the native Postgres backend. When set,
# the _query/_query_*/_executemany helpers route to THIS connection (already
# inside an open transaction) instead of acquiring a fresh pooled connection
# that auto-commits per statement. Lets multi-statement operations such as
# import_brain run all-or-nothing (#60). ContextVar (not an instance attribute)
# so concurrent tasks each see their own transaction connection.
_txn_conn_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "postgres_base_txn_conn", default=None
)


def _as_aware_utc(value: Any) -> Any:
    """Pin a naive datetime to UTC before it is bound to a TIMESTAMPTZ column.

    The project stores all instants as *naive* UTC (see ``utils/timeutils``).
    asyncpg encodes a naive ``datetime`` into ``TIMESTAMPTZ`` using the
    **Python process** local timezone, not the session ``TimeZone`` — so a
    naive UTC value silently shifts by the host's UTC offset (#15). Binding a
    tz-aware UTC datetime makes the encoded instant correct regardless of the
    process or server timezone. Non-datetime values pass through untouched.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _utc_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_as_aware_utc(a) for a in args)


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

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Any]:
        """Run a block of _query/_executemany calls atomically.

        Acquires ONE pooled connection, opens an asyncpg transaction, and
        binds it to the current task via a ContextVar so every _query* /
        _executemany call inside the block runs on that connection and
        commits/rolls back together (#60). Nested calls reuse the same
        connection. Yields the connection for callers that need it directly.
        """
        existing = _txn_conn_var.get()
        if existing is not None:
            # Already inside a transaction in this task — reuse it.
            yield existing
            return

        conn: Any = await self._pool.acquire()
        token = _txn_conn_var.set(conn)
        try:
            async with conn.transaction():
                yield conn
        finally:
            _txn_conn_var.reset(token)
            await self._pool.release(conn)

    async def _query(self, sql: str, *args: Any, timeout: float = 30.0) -> Any:
        """Execute a write query. Returns last result."""
        txn_conn = _txn_conn_var.get()
        if txn_conn is not None:
            return await txn_conn.execute(sql, *_utc_args(args), timeout=timeout)

        conn: Any = await self._pool.acquire()
        try:
            return await conn.execute(sql, *_utc_args(args), timeout=timeout)
        finally:
            await self._pool.release(conn)

    async def _query_ro(self, sql: str, *args: Any, timeout: float = 30.0) -> list[Any]:
        """Execute a read query. Returns list of records."""
        txn_conn = _txn_conn_var.get()
        if txn_conn is not None:
            return list(await txn_conn.fetch(sql, *_utc_args(args), timeout=timeout))

        conn: Any = await self._pool.acquire()
        try:
            return list(await conn.fetch(sql, *_utc_args(args), timeout=timeout))
        finally:
            await self._pool.release(conn)

    async def _query_one(self, sql: str, *args: Any, timeout: float = 30.0) -> Any | None:
        """Execute a read query expecting at most one row."""
        txn_conn = _txn_conn_var.get()
        if txn_conn is not None:
            return await txn_conn.fetchrow(sql, *_utc_args(args), timeout=timeout)

        conn: Any = await self._pool.acquire()
        try:
            return await conn.fetchrow(sql, *_utc_args(args), timeout=timeout)
        finally:
            await self._pool.release(conn)

    async def _executemany(
        self, sql: str, args_list: list[tuple[Any, ...]], timeout: float = 30.0
    ) -> None:
        """Execute a parameterized query for each args tuple in one connection."""
        prepared = [_utc_args(tuple(a)) for a in args_list]
        txn_conn = _txn_conn_var.get()
        if txn_conn is not None:
            await txn_conn.executemany(sql, prepared, timeout=timeout)
            return

        conn: Any = await self._pool.acquire()
        try:
            await conn.executemany(sql, prepared, timeout=timeout)
        finally:
            await self._pool.release(conn)

    @staticmethod
    def _serialize_metadata(meta: dict[str, Any]) -> str:
        return json.dumps(meta)

    @staticmethod
    def _dt_to_str(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None
