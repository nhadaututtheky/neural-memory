"""SQLite dialect — aiosqlite connection with ReadPool and FTS5 support."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

import aiosqlite

from neural_memory.storage.read_pool import DEFAULT_POOL_SIZE, ReadPool
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.cjk import cjk_spaced

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
# One queued write: the operation plus the caller's future.
# None on the queue is the sentinel that stops the worker.
_WriteOp = Callable[[aiosqlite.Connection], Awaitable[Any]]
_WriteItem = tuple[_WriteOp, asyncio.Future[Any]]


class SQLiteDialect(Dialect):
    """SQLite dialect using aiosqlite.

    Manages a single write connection + optional ReadPool for parallel reads.
    Handles commit after each write, FTS5 virtual tables, and WAL mode.

    Writes go through one worker so two commits cannot overlap on that
    connection. ``transaction()`` holds the connection until it commits
    or rolls back, and a queued write cannot join that transaction.
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
        # Held for the whole transaction() body. The worker takes the same
        # lock per item, so a queued write waits out an open transaction.
        self._write_lock = asyncio.Lock()
        self._txn_task: asyncio.Task[Any] | None = None

        # Write serialization queue (single writer for SQLite)
        self._write_queue: asyncio.Queue[_WriteItem | None] | None = None
        self._write_task: asyncio.Task[None] | None = None

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

        # Start single-writer worker (serializes all writes on SQLite)
        self._write_queue = asyncio.Queue()
        self._write_task = asyncio.create_task(self._write_worker())

        logger.info("SQLite dialect initialized: %s", self._db_path)

    async def close(self) -> None:
        # Stop write worker
        if self._write_task is not None and self._write_queue is not None:
            await self._write_queue.put(None)
            await self._write_task
            self._write_task = None
            self._write_queue = None

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
    # Write serialization (single writer for SQLite)
    # ------------------------------------------------------------------

    async def _write_worker(self) -> None:
        """Background task that serializes all writes to the single SQLite connection."""
        assert self._write_queue is not None
        assert self._conn is not None

        while True:
            item = await self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break

            op, fut = item
            try:
                async with self._write_lock:
                    try:
                        result = await op(self._conn)
                    except Exception as exc:
                        await self._rollback_if_open(self._conn)
                        self._reject_write(fut, exc)
                    else:
                        self._resolve_write(fut, result)
            finally:
                self._write_queue.task_done()

    @staticmethod
    def _resolve_write(fut: asyncio.Future[Any], result: Any) -> None:
        """Deliver a write result unless the caller already gave up.

        Cancelling the caller cancels this future. ``set_result`` would then
        raise ``InvalidStateError`` and kill the worker (#216).
        """
        if fut.done():
            return
        try:
            fut.set_result(result)
        except asyncio.InvalidStateError:
            return

    @staticmethod
    def _reject_write(fut: asyncio.Future[Any], exc: BaseException) -> None:
        """Deliver a write failure. Log it when nobody is waiting anymore."""
        if not fut.done():
            try:
                fut.set_exception(exc)
            except asyncio.InvalidStateError:
                logger.warning("SQLite write failed after the caller went away: %s", exc)
            return
        logger.warning("SQLite write failed after the caller went away: %s", exc)

    async def _rollback_if_open(self, conn: aiosqlite.Connection) -> None:
        """Roll back a failed write so the next statement sees a clean connection."""
        if not conn.in_transaction:
            return
        try:
            await conn.rollback()
        except Exception:
            logger.warning("SQLite rollback failed", exc_info=True)

    async def _rollback_despite_cancel(self, conn: aiosqlite.Connection) -> None:
        """Roll back even when this task is already cancelled.

        A cancelled task raises ``CancelledError`` at the next await, so the
        rollback itself would be skipped and the connection left inside the
        transaction. Clear the pending cancel for that await, then put it back.
        """
        current = asyncio.current_task()
        pending = 0
        if current is not None:
            while current.cancelling():
                current.uncancel()
                pending += 1
        try:
            await self._rollback_if_open(conn)
        finally:
            if current is not None:
                for _ in range(pending):
                    current.cancel()

    async def _dispatch(
        self,
        op: Callable[[aiosqlite.Connection], Awaitable[_T]],
    ) -> _T:
        """Run ``op`` on the writer connection, serialized with every other write.

        Inside ``transaction()`` the caller already holds the write lock, so
        the statement runs here directly. Queueing it would wait for the
        worker, and the worker would wait for the lock.
        """
        # ``current_task()`` is None outside a Task. Treating that as the
        # transaction owner would skip the queue whenever ``_txn_task`` is
        # also None.
        current = asyncio.current_task()
        owns_transaction = current is not None and self._txn_task is current
        if owns_transaction or self._write_queue is None:
            return await op(self._ensure_conn())
        return await self._submit(op)

    async def _submit(
        self,
        op: Callable[[aiosqlite.Connection], Awaitable[_T]],
    ) -> _T:
        assert self._write_queue is not None
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._write_queue.put((op, fut))
        return cast("_T", await fut)

    async def _commit_unless_txn(self, conn: aiosqlite.Connection) -> None:
        if not self._in_transaction:
            await conn.commit()

    async def execute_write(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Execute one write through the same queue as ``execute``.

        Kept for callers that only need a statement and a commit. Reads stay
        on ``fetch_all`` / ``fetch_one``. Explicit transactions use
        ``transaction()``.
        """
        await self.execute(sql, params)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> str:
        bound = tuple(params)

        async def op(conn: aiosqlite.Connection) -> str:
            await conn.execute(sql, bound)
            await self._commit_unless_txn(conn)
            return ""

        return await self._dispatch(op)

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
        # Snapshot the rows so a caller mutating the list cannot change a
        # write that is already queued.
        bound_rows = tuple(tuple(row) for row in args_list)

        async def op(conn: aiosqlite.Connection) -> None:
            await conn.executemany(sql, bound_rows)
            await self._commit_unless_txn(conn)

        await self._dispatch(op)

    async def execute_script(self, sql: str) -> None:
        async def op(conn: aiosqlite.Connection) -> None:
            # executescript commits any pending transaction itself.
            await conn.executescript(sql)

        await self._dispatch(op)

    async def execute_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        bound = tuple(params)

        async def op(conn: aiosqlite.Connection) -> int:
            cursor = await conn.execute(sql, bound)
            await self._commit_unless_txn(conn)
            return cursor.rowcount

        return await self._dispatch(op)

    async def execute_returning_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        bound = tuple(params)

        async def op(conn: aiosqlite.Connection) -> int:
            await conn.execute(sql, bound)
            await self._commit_unless_txn(conn)
            # changes() must run before another writer resets it.
            cursor = await conn.execute("SELECT changes() as cnt", ())
            row = await cursor.fetchone()
            return row[0] if row else 0

        return await self._dispatch(op)

    async def insert_returning_id(self, sql: str, params: Sequence[Any] = ()) -> int:
        bound = tuple(params)

        async def op(conn: aiosqlite.Connection) -> int:
            async with conn.execute(sql, bound) as cursor:
                row = await cursor.fetchone()
                inserted_id = int(row[0]) if row and row[0] is not None else 0
            await self._commit_unless_txn(conn)
            return inserted_id

        return await self._dispatch(op)

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

        Holds the write lock for the whole block so queued writes wait
        outside it. Statements issued on this task skip their own commit.
        Do not wait, inside the block, on another task that writes: that
        task cannot start until this block releases the lock.

        Nested transactions on the same task are rejected. A second
        ``BEGIN`` would deadlock on the lock.
        """
        current = asyncio.current_task()
        if current is not None and self._txn_task is current:
            raise RuntimeError("SQLiteDialect transactions cannot be nested")

        conn = self._ensure_conn()
        await self._write_lock.acquire()
        self._txn_task = asyncio.current_task()
        try:
            await conn.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            try:
                yield
                await conn.commit()
            except BaseException:
                await self._rollback_despite_cancel(conn)
                raise
            finally:
                self._in_transaction = False
        finally:
            self._txn_task = None
            self._write_lock.release()

    def auto_increment_pk(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def timestamp_type(self) -> str:
        return "TEXT"

    def jsonb_type(self) -> str:
        return "TEXT"
