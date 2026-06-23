"""PostgreSQL dialect — asyncpg connection pool with pgvector and tsvector support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


# Idempotent DDL that creates the tsvector columns + GIN indexes used by
# ``fts_neuron_query`` / ``fts_fiber_query``. Appended to the schema DDL
# returned by :meth:`PostgresDialect.get_schema_ddl`. Generated columns
# (PG 12+) keep the tsvector in sync with ``content`` / ``summary``
# automatically, so no insert-path changes are needed.
#
# The DO blocks below also upgrade pre-existing *non-generated* tsvector
# columns left over from older code paths: if the column exists but is
# plain (``attgenerated = ''``), we drop and re-add it as a STORED
# generated column so FTS lookups actually return matches.
_PG_FTS_DDL = """
-- Full-text search support (idempotent; safe to re-run).
DO $nm_fts_neurons$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'neurons'
          AND a.attname = 'content_tsv'
          AND a.attgenerated = ''
    ) THEN
        EXECUTE 'ALTER TABLE neurons DROP COLUMN content_tsv';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'neurons' AND a.attname = 'content_tsv'
    ) THEN
        EXECUTE 'ALTER TABLE neurons ADD COLUMN content_tsv tsvector '
             || 'GENERATED ALWAYS AS '
             || '(to_tsvector(''english'', coalesce(content, ''''))) STORED';
    END IF;
END
$nm_fts_neurons$;
CREATE INDEX IF NOT EXISTS idx_neurons_content_tsv
    ON neurons USING GIN (content_tsv);

DO $nm_fts_fibers$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'fibers'
          AND a.attname = 'summary_tsv'
          AND a.attgenerated = ''
    ) THEN
        EXECUTE 'ALTER TABLE fibers DROP COLUMN summary_tsv';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'fibers' AND a.attname = 'summary_tsv'
    ) THEN
        EXECUTE 'ALTER TABLE fibers ADD COLUMN summary_tsv tsvector '
             || 'GENERATED ALWAYS AS '
             || '(to_tsvector(''english'', coalesce(summary, ''''))) STORED';
    END IF;
END
$nm_fts_fibers$;
CREATE INDEX IF NOT EXISTS idx_fibers_summary_tsv
    ON fibers USING GIN (summary_tsv);
"""


# Per-table set of columns the dialect mixins bind via ``serialize_dt`` —
# i.e. real ``datetime`` objects on the PG path — which therefore MUST be
# ``TIMESTAMPTZ`` rather than the SQLite ``TEXT``. Columns NOT listed here
# (notably the ``updated_at TEXT DEFAULT ''`` sync strings on
# neurons/synapses/fibers, and ``cognitive_state.predicted_at`` /
# ``resolved_at`` / ``last_evidence_at`` which are bound as ISO strings)
# stay ``TEXT`` so asyncpg's binding remains type-correct.
_PG_TIMESTAMP_COLUMNS: dict[str, frozenset[str]] = {
    "brains": frozenset({"created_at", "updated_at"}),
    "neurons": frozenset({"created_at", "last_accessed_at"}),
    "neuron_states": frozenset({"last_activated", "refractory_until", "created_at"}),
    "synapses": frozenset({"last_activated", "created_at"}),
    "fibers": frozenset(
        {"last_conducted", "time_start", "time_end", "last_ghost_shown_at", "created_at"}
    ),
    "typed_memories": frozenset({"expires_at", "created_at"}),
    "projects": frozenset({"start_date", "end_date", "created_at"}),
    "memory_maturations": frozenset({"stage_entered_at"}),
    "co_activation_events": frozenset({"created_at"}),
    "action_events": frozenset({"created_at"}),
    "brain_versions": frozenset({"created_at"}),
    "sync_states": frozenset({"last_sync_at"}),
    "alerts": frozenset({"created_at", "seen_at", "acknowledged_at", "resolved_at"}),
    "review_schedules": frozenset({"next_review", "last_reviewed", "created_at"}),
    "depth_priors": frozenset({"last_updated", "created_at"}),
    "compression_backups": frozenset({"compressed_at"}),
    "neuron_snapshots": frozenset({"compressed_at"}),
    "change_log": frozenset({"changed_at"}),
    "devices": frozenset({"last_sync_at", "registered_at"}),
    "retrieval_calibration": frozenset({"created_at"}),
    "tool_events": frozenset({"created_at"}),
    "training_files": frozenset({"trained_at", "created_at"}),
    # cognitive_state.created_at is serialize_dt; predicted_at/resolved_at/
    # last_evidence_at are bound as ISO strings → stay TEXT.
    "cognitive_state": frozenset({"created_at"}),
    "hot_index": frozenset({"updated_at"}),
    "knowledge_gaps": frozenset({"detected_at", "resolved_at"}),
    "sources": frozenset({"effective_date", "expires_at", "created_at", "updated_at"}),
    "session_summaries": frozenset({"started_at", "ended_at"}),
    "retriever_calibration": frozenset({"created_at"}),
    "tag_cooccurrence": frozenset({"last_seen"}),
    "drift_clusters": frozenset({"created_at", "resolved_at"}),
    "merkle_hashes": frozenset({"updated_at"}),
}


def _convert_timestamp_columns(ddl: str) -> str:
    """Rewrite per-table TEXT timestamp columns to TIMESTAMPTZ.

    Operates on each ``CREATE TABLE IF NOT EXISTS <name> ( ... );`` block
    independently, converting only the columns listed in
    ``_PG_TIMESTAMP_COLUMNS`` for that table. This avoids mis-converting
    same-named columns that are bound as ISO strings in other tables.
    """
    import re

    def _convert_block(match: re.Match[str]) -> str:
        table = match.group("table")
        block = match.group(0)
        cols = _PG_TIMESTAMP_COLUMNS.get(table)
        if not cols:
            return block
        for col in cols:
            # Match "<col> TEXT" at a column-definition boundary (preceded
            # by whitespace, followed by a word boundary) and swap the type.
            # ``\g<indent>`` preserves the leading newline/indentation.
            block = re.sub(
                rf"(?P<indent>\n\s*){re.escape(col)}\s+TEXT\b",
                rf"\g<indent>{col} TIMESTAMPTZ",
                block,
            )
        return block

    return re.sub(
        r"CREATE TABLE IF NOT EXISTS (?P<table>\w+)\s*\(.*?\);",
        _convert_block,
        ddl,
        flags=re.DOTALL,
    )


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

    def fts_neuron_rank_order(self, term_param: int) -> str:
        # ts_rank returns float where higher = more relevant; use DESC.
        return f"ts_rank(n.content_tsv, plainto_tsquery('english', ${term_param})) DESC"

    def fts_neuron_score_expr(self, term_param: int) -> str:
        # Higher = better — directly usable in composite scoring.
        return f"ts_rank(n.content_tsv, plainto_tsquery('english', ${term_param}))"

    # ------------------------------------------------------------------
    # JSON operators (override for JSONB)
    # ------------------------------------------------------------------

    def json_extract(self, column: str, key: str) -> str:
        return f"{column}->>'{key}'"

    def json_contains_key(self, column: str, key_param: int) -> str:
        return f"{column} ? ${key_param}"

    def json_array_contains(self, column: str, value_param: int) -> str:
        # JSON tag columns (tags/auto_tags/agent_tags) are stored as TEXT on the
        # unified schema (SQLite-derived), so the column must be cast to jsonb
        # before the containment operator — ``text @> jsonb`` has no operator.
        return f"{column}::jsonb @> ${value_param}::jsonb"

    def json_array_contains_param(self, value: Any) -> Any:
        # ``tags @> $N::jsonb`` needs a JSON document. A bare tag like ``work``
        # is not valid JSON, so wrap it as a one-element JSON array.
        import json

        return json.dumps([value])

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

        Translates the canonical SQLite ``SCHEMA`` into a *physically*
        Postgres-correct schema that matches what the dialect mixins
        bind at runtime:

        * SQLite ``INTEGER PRIMARY KEY AUTOINCREMENT`` → ``SERIAL PRIMARY KEY``.
        * ``content_hash INTEGER`` → ``content_hash BIGINT`` so 64-bit
          SimHash fingerprints do not overflow PG's signed 32-bit
          ``INTEGER`` (closes #4).
        * Every column the mixins bind via ``serialize_dt`` (a real
          ``datetime`` on the PG path) is emitted as ``TIMESTAMPTZ``
          instead of SQLite ``TEXT`` — applied **per table** because some
          ``*_at`` columns (e.g. ``cognitive_state.predicted_at``,
          ``*.updated_at`` sync strings) are bound as plain ISO strings
          and MUST stay ``TEXT`` (closes #5).
        * The ``projects`` table is hoisted ahead of ``typed_memories``,
          which forward-references it (PostgreSQL enforces FKs at
          CREATE TABLE time, unlike SQLite).
        * Appends the idempotent ``content_tsv`` / ``summary_tsv``
          generated columns + GIN indexes required by FTS, so
          ``fts_neuron_query`` / ``fts_fiber_query`` resolve (closes #3).
        """
        import re

        from neural_memory.storage.sqlite_schema import SCHEMA

        ddl = SCHEMA
        # Replace SQLite AUTOINCREMENT with PostgreSQL SERIAL
        ddl = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        # Remove SQLite-specific pragmas (defensive — SCHEMA has none today).
        ddl = re.sub(r"PRAGMA\s+[^;]+;", "", ddl)

        # 64-bit SimHash fingerprint: PG INTEGER is signed 32-bit (#4).
        ddl = ddl.replace("content_hash INTEGER", "content_hash BIGINT")

        # Per-table TIMESTAMPTZ conversion. Only columns that the dialect
        # mixins bind through ``serialize_dt`` (i.e. as datetime objects)
        # are converted; ISO-string columns (sync ``updated_at`` defaults,
        # cognitive ``predicted_at``/``resolved_at``/``last_evidence_at``)
        # stay TEXT so asyncpg's str binding stays type-correct.
        ddl = _convert_timestamp_columns(ddl)

        # `typed_memories` declares a FOREIGN KEY to `projects`, but the
        # SCHEMA defines `projects` AFTER `typed_memories`. SQLite is
        # forgiving (FKs checked on DML); PostgreSQL rejects the forward
        # reference. Hoist copies of `brains` and `projects` to the top in
        # dependency order — the originals later become no-ops via
        # IF NOT EXISTS.
        prelude = ""
        for tbl in ("brains", "projects"):
            m = re.search(
                rf"CREATE TABLE IF NOT EXISTS {tbl}\s*\([^;]*?\);",
                ddl,
                re.DOTALL,
            )
            if m:
                prelude += m.group(0) + "\n"
        ddl = prelude + ddl
        # Append tsvector generated columns + GIN indexes for FTS.
        # All idempotent — re-running this DDL on an existing DB is a no-op.
        ddl += _PG_FTS_DDL
        return ddl

    # ------------------------------------------------------------------
    # Schema helpers (override for PostgreSQL types)
    # ------------------------------------------------------------------

    def date_trunc_day(self, column: str) -> str:
        # TIMESTAMPTZ → DATE truncation is native and correct on Postgres.
        return f"{column}::date"

    def auto_increment_pk(self) -> str:
        return "SERIAL PRIMARY KEY"

    def timestamp_type(self) -> str:
        return "TIMESTAMPTZ"

    def jsonb_type(self) -> str:
        return "JSONB"
