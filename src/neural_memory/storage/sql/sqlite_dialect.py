"""SQLite dialect — aiosqlite connection with ReadPool and FTS5 support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from neural_memory.storage.read_pool import DEFAULT_POOL_SIZE, ReadPool
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.cjk import cjk_spaced

logger = logging.getLogger(__name__)


class SQLiteDialect(Dialect):
    """SQLite dialect using aiosqlite.

    Manages a single write connection + optional ReadPool for parallel reads.
    Handles commit after each write, FTS5 virtual tables, and WAL mode.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self._db_path = Path(db_path).resolve()
        self._conn: aiosqlite.Connection | None = None
        self._read_pool: ReadPool | None = None
        self._pool_size = pool_size
        self._has_fts: bool = False
        self._in_transaction: bool = False

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    @property
    def supports_vector(self) -> bool:
        return False

    @property
    def supports_fts(self) -> bool:
        return self._has_fts

    @property
    def supports_ilike(self) -> bool:
        return False  # SQLite LIKE is case-insensitive for ASCII by default

    @property
    def name(self) -> str:
        return "sqlite"

    def is_integrity_error(self, exc: BaseException) -> bool:
        import sqlite3

        if isinstance(exc, sqlite3.IntegrityError):
            return True
        return super().is_integrity_error(exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open connection, set pragmas, create schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # FTS sync triggers call cjk_spaced() so the index stores
        # CJK-spaced text (see utils/cjk.py). Must be registered before
        # any write or migration touches the FTS tables.
        await self._conn.create_function("cjk_spaced", 1, cjk_spaced)

        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=30000")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=-8000")
        await self._conn.commit()

        # Reader pool (WAL): parallel reads outside explicit transactions.
        # Inside transaction(), reads stay on the writer for read-your-writes.
        self._read_pool = ReadPool(self._db_path, pool_size=self._pool_size)
        await self._read_pool.initialize()

        logger.info("SQLite dialect initialized: %s", self._db_path)

    async def close(self) -> None:
        if self._read_pool is not None:
            await self._read_pool.close()
            self._read_pool = None
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteDialect not initialized — call initialize() first")
        return self._conn

    def _read_conn(self) -> aiosqlite.Connection:
        """Prefer a pooled reader; fall back to writer if pool unavailable."""
        if self._read_pool is not None and not self._in_transaction:
            return self._read_pool.acquire()
        return self._ensure_conn()

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Txn-aware read connection CM (pool outside txn, writer inside)."""
        if self._read_pool is not None and not self._in_transaction:
            async with self._read_pool.connection() as conn:
                yield conn
            return
        yield self._ensure_conn()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> str:
        conn = self._ensure_conn()
        await conn.execute(sql, tuple(params))
        if not self._in_transaction:
            await conn.commit()
        return ""

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self._read_connection() as conn:
            async with conn.execute(sql, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    return []
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=False)) for row in rows]

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        async with self._read_connection() as conn:
            async with conn.execute(sql, tuple(params)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row, strict=False))

    async def execute_many(self, sql: str, args_list: Sequence[Sequence[Any]]) -> None:
        conn = self._ensure_conn()
        await conn.executemany(sql, [tuple(a) for a in args_list])
        if not self._in_transaction:
            await conn.commit()

    async def execute_script(self, sql: str) -> None:
        conn = self._ensure_conn()
        await conn.executescript(sql)

    async def execute_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._ensure_conn()
        cursor = await conn.execute(sql, tuple(params))
        if not self._in_transaction:
            await conn.commit()
        return cursor.rowcount

    async def execute_returning_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._ensure_conn()
        await conn.execute(sql, tuple(params))
        if not self._in_transaction:
            await conn.commit()
        cursor = await conn.execute("SELECT changes() as cnt", ())
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def insert_returning_id(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._ensure_conn()
        async with conn.execute(sql, tuple(params)) as cursor:
            row = await cursor.fetchone()
            inserted_id = int(row[0]) if row and row[0] is not None else 0
        if not self._in_transaction:
            await conn.commit()
        return inserted_id

    # ------------------------------------------------------------------
    # Placeholder generation
    # ------------------------------------------------------------------

    def ph(self, index: int) -> str:
        return "?"

    def phs(self, count: int, start: int = 1) -> str:
        return ", ".join("?" for _ in range(count))

    def in_clause(self, param_start: int, values: Sequence[Any]) -> tuple[str, list[Any]]:
        placeholders = ", ".join("?" for _ in values)
        return f"IN ({placeholders})", list(values)

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
        updates = ", ".join(f"{c} = excluded.{c}" for c in update_columns)
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
        if not self._has_fts:
            raise NotImplementedError("FTS5 not available on this SQLite database")
        from_clause = "neurons n JOIN neurons_fts fts ON n.rowid = fts.rowid"
        where_clause = "fts.neurons_fts MATCH ? AND fts.brain_id = ?"
        return from_clause, where_clause

    def fts_fiber_query(self, term_param: int, brain_id_param: int) -> tuple[str, str]:
        if not self._has_fts:
            raise NotImplementedError("FTS5 not available on this SQLite database")
        from_clause = "fibers f JOIN fibers_fts fts ON f.rowid = fts.rowid"
        where_clause = "fts.fibers_fts MATCH ? AND fts.brain_id = ?"
        return from_clause, where_clause

    def fts_neuron_rank_order(self, term_param: int) -> str:
        if not self._has_fts:
            raise NotImplementedError("FTS5 not available on this SQLite database")
        # FTS5 'rank' column: lower = better, so plain ASC.
        return "fts.rank"

    def fts_neuron_score_expr(self, term_param: int) -> str:
        if not self._has_fts:
            raise NotImplementedError("FTS5 not available on this SQLite database")
        # Negate so higher = better when combined with other positive signals.
        return "(-fts.rank)"

    # ------------------------------------------------------------------
    # Schema helpers (override defaults for SQLite)
    # ------------------------------------------------------------------

    def get_schema_ddl(self) -> str:
        from neural_memory.storage.sqlite_schema import SCHEMA

        return SCHEMA

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Atomic transaction for SQLite.

        Sets ``_in_transaction`` so that individual execute/execute_many
        calls skip their per-statement ``conn.commit()``. The commit (or
        rollback) happens here at the end of the block.
        """
        conn = self._ensure_conn()
        await conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            self._in_transaction = False

    def auto_increment_pk(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def timestamp_type(self) -> str:
        return "TEXT"

    def jsonb_type(self) -> str:
        return "TEXT"
