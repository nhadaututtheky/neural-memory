"""SQL Dialect abstraction — isolates database engine differences.

Each dialect implements the same interface so that storage mixins can
execute queries without knowing whether they're talking to SQLite,
PostgreSQL, MySQL, or any other SQL database.

The dialect handles:
- Connection management (aiosqlite vs asyncpg vs aiomysql)
- Parameter placeholder syntax (? vs $N vs %s)
- Upsert / INSERT OR IGNORE syntax
- Full-text search (FTS5 vs tsvector vs FULLTEXT)
- JSON operators (json_extract vs ->> vs @>)
- Date/time serialization
- Array handling (IN-clause expansion vs ANY())
- Feature flags (vector search, ILIKE, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any


class Dialect(ABC):
    """Abstract SQL dialect — one implementation per database engine."""

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    @property
    def supports_vector(self) -> bool:
        """Whether this dialect supports vector similarity search (e.g., pgvector)."""
        return False

    @property
    def supports_fts(self) -> bool:
        """Whether full-text search is available."""
        return False

    @property
    def supports_ilike(self) -> bool:
        """Whether ILIKE (case-insensitive LIKE) is supported."""
        return False

    @property
    def name(self) -> str:
        """Dialect name for logging and diagnostics."""
        return "unknown"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Open connection(s) and create schema if needed."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close all connections and release resources."""
        ...

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> str:
        """Execute a write query (INSERT/UPDATE/DELETE).

        Returns a status string (e.g., "INSERT 1") or empty string.
        Handles commit internally for SQLite.
        """
        ...

    @abstractmethod
    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute a read query and return all rows as dicts."""
        ...

    @abstractmethod
    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        """Execute a read query and return the first row as dict, or None."""
        ...

    @abstractmethod
    async def execute_many(self, sql: str, args_list: Sequence[Sequence[Any]]) -> None:
        """Execute a parameterized query for each set of arguments."""
        ...

    @abstractmethod
    async def execute_script(self, sql: str) -> None:
        """Execute a multi-statement SQL script (DDL, migrations)."""
        ...

    @abstractmethod
    async def execute_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a write query and return the number of affected rows."""
        ...

    async def execute_returning_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a write query that returns a row-count expression and return the count.

        Used for INSERT...SELECT statements that use expressions like
        ``SELECT changes()`` (SQLite) or ``SELECT COUNT(*)`` to get the number
        of rows affected by the preceding DML statement.
        """
        await self.execute(sql, params)
        row = await self.fetch_one("SELECT 0 as cnt", ())
        return row.get("cnt", 0) if row else 0

    # ------------------------------------------------------------------
    # Placeholder generation
    # ------------------------------------------------------------------

    @abstractmethod
    def ph(self, index: int) -> str:
        """Return a single parameter placeholder.

        SQLite: ``ph(1)`` → ``?``
        PostgreSQL: ``ph(1)`` → ``$1``
        """
        ...

    def phs(self, count: int, start: int = 1) -> str:
        """Return comma-separated placeholders.

        ``phs(3)`` → ``?, ?, ?`` (SQLite) or ``$1, $2, $3`` (PostgreSQL)
        """
        return ", ".join(self.ph(i) for i in range(start, start + count))

    @abstractmethod
    def in_clause(self, param_start: int, values: Sequence[Any]) -> tuple[str, list[Any]]:
        """Build an IN clause for the given values.

        SQLite: returns ``"IN (?, ?, ?)", [v1, v2, v3]``
        PostgreSQL: returns ``"= ANY($N::text[])", [list_of_values]``

        Args:
            param_start: The starting parameter index.
            values: The values to match against.

        Returns:
            Tuple of (SQL fragment, parameter list).
        """
        ...

    # ------------------------------------------------------------------
    # SQL generation helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_sql(
        self,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
        update_columns: Sequence[str],
    ) -> str:
        """Generate an UPSERT statement.

        SQLite: ``INSERT OR REPLACE INTO ...`` or ``INSERT ... ON CONFLICT DO UPDATE``
        PostgreSQL: ``INSERT ... ON CONFLICT (...) DO UPDATE SET ...``
        """
        ...

    @abstractmethod
    def insert_or_ignore_sql(
        self,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
    ) -> str:
        """Generate an INSERT-or-skip statement.

        SQLite: ``INSERT OR IGNORE INTO ...``
        PostgreSQL: ``INSERT ... ON CONFLICT DO NOTHING``
        """
        ...

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def fts_neuron_query(self, term_param: int, brain_id_param: int) -> tuple[str, str]:
        """Return (FROM clause, WHERE clause) for neuron FTS.

        Raises NotImplementedError if FTS is not supported.
        """
        raise NotImplementedError(f"{self.name} dialect does not support FTS")

    def fts_fiber_query(self, term_param: int, brain_id_param: int) -> tuple[str, str]:
        """Return (FROM clause, WHERE clause) for fiber FTS."""
        raise NotImplementedError(f"{self.name} dialect does not support FTS")

    # ------------------------------------------------------------------
    # JSON operators
    # ------------------------------------------------------------------

    def json_extract(self, column: str, key: str) -> str:
        """SQL expression to extract a JSON key as text.

        SQLite: ``json_extract(column, '$.key')``
        PostgreSQL: ``column->>'key'``
        """
        return f"json_extract({column}, '$.{key}')"

    def json_contains_key(self, column: str, key_param: int) -> str:
        """SQL expression: does JSON object contain key?

        SQLite: ``json_extract(column, '$.' || ?) IS NOT NULL``
        PostgreSQL: ``column ? $N``
        """
        return f"json_extract({column}, '$.' || {self.ph(key_param)}) IS NOT NULL"

    def json_array_contains(self, column: str, value_param: int) -> str:
        """SQL expression: does JSON array contain value?

        SQLite: ``EXISTS (SELECT 1 FROM json_each(column) WHERE value = ?)``
        PostgreSQL: ``column @> $N::jsonb``
        """
        return f"EXISTS (SELECT 1 FROM json_each({column}) WHERE value = {self.ph(value_param)})"

    # ------------------------------------------------------------------
    # Date/time handling
    # ------------------------------------------------------------------

    def serialize_dt(self, dt: datetime | None) -> Any:
        """Serialize a datetime for storage.

        SQLite: returns ISO format string.
        PostgreSQL: returns the datetime object directly.
        """
        if dt is None:
            return None
        return dt.isoformat()

    def normalize_dt(self, value: Any) -> datetime | None:
        """Normalize a datetime value from a row.

        SQLite: parses ISO format string.
        PostgreSQL: returns the datetime object directly.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def auto_increment_pk(self) -> str:
        """SQL type for an auto-incrementing integer primary key.

        SQLite: ``INTEGER PRIMARY KEY AUTOINCREMENT``
        PostgreSQL: ``SERIAL PRIMARY KEY``
        """
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def timestamp_type(self) -> str:
        """SQL type for timestamp columns.

        SQLite: ``TEXT`` (ISO format)
        PostgreSQL: ``TIMESTAMPTZ``
        """
        return "TEXT"

    def jsonb_type(self) -> str:
        """SQL type for JSON columns.

        SQLite: ``TEXT``
        PostgreSQL: ``JSONB``
        """
        return "TEXT"

    def get_schema_ddl(self) -> str:
        """Return the full DDL script for creating all schema tables.

        Override in subclasses that need dialect-specific DDL (e.g.,
        PostgreSQL uses SERIAL, TIMESTAMPTZ, JSONB instead of SQLite types).
        Default implementation returns the SQLite SCHEMA.
        """
        from neural_memory.storage.sqlite_schema import SCHEMA

        return SCHEMA

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for atomic transactions.

        Default implementation is a no-op (each statement auto-commits).
        Override in subclasses to provide real transaction semantics.
        """
        yield
