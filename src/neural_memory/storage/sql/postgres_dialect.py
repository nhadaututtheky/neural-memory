"""PostgreSQL dialect — asyncpg connection pool with pgvector and tsvector support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


class PostgresDialect(Dialect):
    """PostgreSQL dialect using asyncpg.

    Manages a connection pool (asyncpg.Pool). Supports pgvector for
    embedding search and tsvector for full-text search.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "neuralmemory",
        user: str = "postgres",
        password: str = "",
        pool_min: int = 1,
        pool_max: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any = None  # asyncpg.Pool
        self._has_vector: bool = False
        self._has_fts: bool = True  # PostgreSQL always has tsvector

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    @property
    def supports_vector(self) -> bool:
        return self._has_vector

    @property
    def supports_fts(self) -> bool:
        return self._has_fts

    @property
    def supports_ilike(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "postgres"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._user,
            password=self._password or None,
            min_size=self._pool_min,
            max_size=self._pool_max,
            command_timeout=60,
        )

        # Check pgvector extension
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            self._has_vector = row is not None

        logger.info(
            "PostgreSQL dialect initialized: %s:%d/%s (vector=%s)",
            self._host,
            self._port,
            self._database,
            self._has_vector,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgresDialect not initialized — call initialize() first")
        return self._pool

    # ------------------------------------------------------------------
    # Query execution (transaction-aware: uses _txn_conn when inside
    # a transaction() block, otherwise acquires from pool per call)
    # ------------------------------------------------------------------

    def _get_conn_or_none(self) -> Any:
        """Return the transaction connection if inside a transaction, else None."""
        return getattr(self, "_txn_conn", None)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> str:
        conn = self._get_conn_or_none()
        if conn is not None:
            result = await conn.execute(sql, *params)
            return result or ""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)
            return result or ""

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._get_conn_or_none()
        if conn is not None:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        conn = self._get_conn_or_none()
        if conn is not None:
            row = await conn.fetchrow(sql, *params)
            return dict(row) if row else None
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)
            return dict(row) if row else None

    async def execute_many(self, sql: str, args_list: Sequence[Sequence[Any]]) -> None:
        conn = self._get_conn_or_none()
        if conn is not None:
            await conn.executemany(sql, [tuple(a) for a in args_list])
            return
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(sql, [tuple(a) for a in args_list])

    async def execute_script(self, sql: str) -> None:
        conn = self._get_conn_or_none()
        if conn is not None:
            await conn.execute(sql)
            return
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql)

    async def execute_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._get_conn_or_none()
        if conn is not None:
            result = await conn.execute(sql, *params)
        else:
            pool = self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(sql, *params)
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def execute_returning_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._get_conn_or_none()
        if conn is not None:
            result = await conn.execute(sql, *params)
        else:
            pool = self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(sql, *params)
        # asyncpg returns status strings like "INSERT 0 1", "UPDATE 3",
        # "DELETE 5". The last token is the affected row count.
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # Placeholder generation
    # ------------------------------------------------------------------

    def ph(self, index: int) -> str:
        return f"${index}"

    def phs(self, count: int, start: int = 1) -> str:
        return ", ".join(f"${i}" for i in range(start, start + count))

    def in_clause(self, param_start: int, values: Sequence[Any]) -> tuple[str, list[Any]]:
        # PostgreSQL uses ANY with an array parameter.
        # Let asyncpg infer the array type instead of hardcoding ::text[].
        return f"= ANY(${param_start})", [list(values)]

    # ------------------------------------------------------------------
    # SQL generation helpers
    # ------------------------------------------------------------------

    def upsert_sql(
        self,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
        update_columns: Sequence[str],
    ) -> str:
        cols = ", ".join(columns)
        phs = self.phs(len(columns))
        conflict = ", ".join(conflict_columns)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
        return (
            f"INSERT INTO {table} ({cols}) VALUES ({phs}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        )

    def insert_or_ignore_sql(
        self,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
    ) -> str:
        cols = ", ".join(columns)
        phs = self.phs(len(columns))
        conflict = ", ".join(conflict_columns)
        return f"INSERT INTO {table} ({cols}) VALUES ({phs}) ON CONFLICT ({conflict}) DO NOTHING"

    # ------------------------------------------------------------------

    def fts_neuron_query(self, term_param: int, brain_id_param: int) -> tuple[str, str]:
        from_clause = "neurons n"
        where_clause = (
            f"n.content_tsv @@ plainto_tsquery('english', ${term_param}) "
            f"AND n.brain_id = ${brain_id_param}"
        )
        return from_clause, where_clause

    def fts_fiber_query(self, term_param: int, brain_id_param: int) -> tuple[str, str]:
        from_clause = "fibers f"
        where_clause = (
            f"f.summary_tsv @@ plainto_tsquery('english', ${term_param}) "
            f"AND f.brain_id = ${brain_id_param}"
        )
        return from_clause, where_clause

    # ------------------------------------------------------------------
    # JSON operators (override for JSONB)
    # ------------------------------------------------------------------

    def json_extract(self, column: str, key: str) -> str:
        return f"{column}->>'{key}'"

    def json_contains_key(self, column: str, key_param: int) -> str:
        return f"{column} ? ${key_param}"

    def json_array_contains(self, column: str, value_param: int) -> str:
        return f"{column} @> ${value_param}::jsonb"

    # ------------------------------------------------------------------
    # Date/time (native TIMESTAMPTZ)
    # ------------------------------------------------------------------

    def serialize_dt(self, dt: datetime | None) -> Any:
        if dt is None:
            return None
        # asyncpg accepts datetime objects directly
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    def normalize_dt(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value
        return datetime.fromisoformat(str(value))

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Atomic transaction using a dedicated connection from the pool."""
        pool = self._ensure_pool()
        conn = await pool.acquire()
        txn = conn.transaction()
        await txn.start()
        # Temporarily stash the transaction connection so that execute
        # methods called inside the block can use it.  Since neural-memory
        # is single-writer async, this is safe.
        prev_txn_conn = getattr(self, "_txn_conn", None)
        self._txn_conn = conn
        try:
            yield
            await txn.commit()
        except Exception:
            await txn.rollback()
            raise
        finally:
            self._txn_conn = prev_txn_conn
            await pool.release(conn)

    # ------------------------------------------------------------------
    # Schema DDL
    # ------------------------------------------------------------------

    def get_schema_ddl(self) -> str:
        """Return PostgreSQL-compatible DDL for all schema tables.

        Converts the SQLite SCHEMA by replacing SQLite-specific syntax
        with PostgreSQL equivalents.
        """
        from neural_memory.storage.sqlite_schema import SCHEMA

        ddl = SCHEMA
        # Replace SQLite AUTOINCREMENT with PostgreSQL SERIAL
        ddl = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        # Remove SQLite-specific pragmas (they appear as comments or
        # standalone statements that PG would reject).  The SCHEMA
        # constant does not contain pragmas (those are in initialize()),
        # but guard against future additions.
        import re

        ddl = re.sub(r"PRAGMA\s+[^;]+;", "", ddl)
        return ddl

    # ------------------------------------------------------------------
    # Schema helpers (override for PostgreSQL types)
    # ------------------------------------------------------------------

    def auto_increment_pk(self) -> str:
        return "SERIAL PRIMARY KEY"

    def timestamp_type(self) -> str:
        return "TIMESTAMPTZ"

    def jsonb_type(self) -> str:
        return "JSONB"
