"""Database schema introspection for DB-to-Brain training pipeline.

Extracts structural metadata (tables, columns, foreign keys, indexes)
from relational databases. Only schema knowledge — no raw data rows.

SQLite dialect implemented for v1. PostgreSQL/MySQL deferred as optional deps.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


# ── Frozen dataclasses ──────────────────────────────────────────


@dataclass(frozen=True)
class ColumnInfo:
    """Column metadata from a database table.

    Attributes:
        name: Column name.
        data_type: SQL data type (e.g., "INTEGER", "TEXT", "VARCHAR(255)").
        nullable: Whether the column allows NULL values.
        primary_key: Whether the column is part of the primary key.
        default_value: Default value expression, if any.
        max_length: Maximum length for string types, if known.
        comment: Column comment/description from DB metadata, if available.
    """

    name: str
    data_type: str
    nullable: bool
    primary_key: bool
    default_value: str | None = None
    max_length: int | None = None
    comment: str | None = None


@dataclass(frozen=True)
class ForeignKeyInfo:
    """Foreign key constraint metadata.

    Attributes:
        column_name: Column in the referencing (child) table.
        referenced_table: Name of the referenced (parent) table.
        referenced_column: Column in the referenced table.
        on_delete: ON DELETE action (CASCADE, SET NULL, etc.).
        on_update: ON UPDATE action.
    """

    column_name: str
    referenced_table: str
    referenced_column: str
    on_delete: str | None = None
    on_update: str | None = None


@dataclass(frozen=True)
class IndexInfo:
    """Index metadata.

    Attributes:
        name: Index name.
        columns: Ordered column names in the index.
        unique: Whether the index enforces uniqueness.
    """

    name: str
    columns: tuple[str, ...]
    unique: bool


@dataclass(frozen=True)
class TableInfo:
    """Complete metadata for a single database table.

    Attributes:
        name: Table name.
        schema: Schema/namespace (None for SQLite).
        columns: Column metadata, ordered by position.
        foreign_keys: Foreign key constraints.
        indexes: Index metadata.
        row_count_estimate: Approximate row count (0 if unknown).
        comment: Table comment/description from DB metadata, if available.
    """

    name: str
    schema: str | None
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKeyInfo, ...]
    indexes: tuple[IndexInfo, ...]
    row_count_estimate: int = 0
    comment: str | None = None


@dataclass(frozen=True)
class SchemaSnapshot:
    """Immutable snapshot of a database schema.

    Captures all structural metadata at a point in time.
    The schema_fingerprint is a SHA256 hash of sorted table+column names,
    used to detect whether a schema has already been trained.

    Attributes:
        database_name: Database name or file path.
        dialect: Database dialect identifier ("sqlite", "postgresql", etc.).
        tables: All table metadata.
        introspected_at: Timestamp of introspection.
        schema_fingerprint: SHA256 hash for change detection.
    """

    database_name: str
    dialect: str
    tables: tuple[TableInfo, ...]
    introspected_at: datetime = field(default_factory=utcnow)
    schema_fingerprint: str = ""


# ── Dialect protocol ────────────────────────────────────────────


@runtime_checkable
class SchemaDialect(Protocol):
    """Protocol for database-specific schema introspection."""

    async def get_tables(self, connection: Any) -> list[str]:
        """Return list of user table names (excluding system tables)."""
        ...

    async def get_columns(
        self, connection: Any, table: str
    ) -> list[ColumnInfo]:
        """Return column metadata for a table."""
        ...

    async def get_foreign_keys(
        self, connection: Any, table: str
    ) -> list[ForeignKeyInfo]:
        """Return foreign key constraints for a table."""
        ...

    async def get_indexes(
        self, connection: Any, table: str
    ) -> list[IndexInfo]:
        """Return index metadata for a table."""
        ...

    async def get_row_count_estimate(
        self, connection: Any, table: str
    ) -> int:
        """Return approximate row count for a table."""
        ...


# ── SQLite dialect ──────────────────────────────────────────────


class SQLiteDialect:
    """SQLite schema introspection using PRAGMA statements."""

    async def get_tables(self, connection: Any) -> list[str]:
        """Get user table names, excluding sqlite_ system tables."""
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_columns(
        self, connection: Any, table: str
    ) -> list[ColumnInfo]:
        """Get column metadata via PRAGMA table_info."""
        cursor = await connection.execute(f"PRAGMA table_info('{table}')")
        rows = await cursor.fetchall()

        columns: list[ColumnInfo] = []
        for row in rows:
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            columns.append(
                ColumnInfo(
                    name=row[1],
                    data_type=row[2] or "BLOB",
                    nullable=not bool(row[3]),
                    primary_key=bool(row[5]),
                    default_value=row[4],
                    max_length=None,
                    comment=None,
                )
            )
        return columns

    async def get_foreign_keys(
        self, connection: Any, table: str
    ) -> list[ForeignKeyInfo]:
        """Get FK constraints via PRAGMA foreign_key_list."""
        cursor = await connection.execute(
            f"PRAGMA foreign_key_list('{table}')"
        )
        rows = await cursor.fetchall()

        fks: list[ForeignKeyInfo] = []
        for row in rows:
            # Columns: id, seq, table, from, to, on_update, on_delete, match
            fks.append(
                ForeignKeyInfo(
                    column_name=row[3],
                    referenced_table=row[2],
                    referenced_column=row[4],
                    on_delete=row[6] if row[6] != "NO ACTION" else None,
                    on_update=row[5] if row[5] != "NO ACTION" else None,
                )
            )
        return fks

    async def get_indexes(
        self, connection: Any, table: str
    ) -> list[IndexInfo]:
        """Get index metadata via PRAGMA index_list + index_info."""
        cursor = await connection.execute(f"PRAGMA index_list('{table}')")
        index_rows = await cursor.fetchall()

        indexes: list[IndexInfo] = []
        for idx_row in index_rows:
            # Columns: seq, name, unique, origin, partial
            idx_name = idx_row[1]
            is_unique = bool(idx_row[2])

            # Skip auto-generated indexes for PKs
            if idx_name.startswith("sqlite_autoindex_"):
                continue

            col_cursor = await connection.execute(
                f"PRAGMA index_info('{idx_name}')"
            )
            col_rows = await col_cursor.fetchall()
            col_names = tuple(r[2] for r in col_rows if r[2])

            if col_names:
                indexes.append(
                    IndexInfo(
                        name=idx_name,
                        columns=col_names,
                        unique=is_unique,
                    )
                )
        return indexes

    async def get_row_count_estimate(
        self, connection: Any, table: str
    ) -> int:
        """Get exact row count (SQLite has no estimate mechanism)."""
        cursor = await connection.execute(
            f"SELECT COUNT(*) FROM '{table}'"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0


# ── Schema introspector ─────────────────────────────────────────


class SchemaIntrospector:
    """Main entry point for database schema introspection.

    Auto-detects dialect from connection string prefix.
    Currently supports SQLite only (v1).
    """

    _DIALECTS: dict[str, type] = {
        "sqlite": SQLiteDialect,
    }

    async def introspect(self, connection_string: str) -> SchemaSnapshot:
        """Introspect database schema and return frozen snapshot.

        Args:
            connection_string: Database connection string.
                SQLite: ``sqlite:///path/to/db.db``

        Returns:
            Frozen SchemaSnapshot with all table metadata.

        Raises:
            ValueError: If dialect is unsupported or connection fails.
        """
        dialect_name = self._detect_dialect(connection_string)
        dialect = self._DIALECTS[dialect_name]()

        db_path = self._extract_path(connection_string, dialect_name)
        connection = await self._create_connection(db_path, dialect_name)

        try:
            table_names = await dialect.get_tables(connection)

            tables: list[TableInfo] = []
            for name in table_names:
                columns = await dialect.get_columns(connection, name)
                foreign_keys = await dialect.get_foreign_keys(connection, name)
                indexes = await dialect.get_indexes(connection, name)
                row_count = await dialect.get_row_count_estimate(
                    connection, name
                )

                tables.append(
                    TableInfo(
                        name=name,
                        schema=None,
                        columns=tuple(columns),
                        foreign_keys=tuple(foreign_keys),
                        indexes=tuple(indexes),
                        row_count_estimate=row_count,
                        comment=None,
                    )
                )

            fingerprint = self._compute_fingerprint(tables)

            return SchemaSnapshot(
                database_name=db_path,
                dialect=dialect_name,
                tables=tuple(tables),
                introspected_at=utcnow(),
                schema_fingerprint=fingerprint,
            )
        finally:
            await connection.close()

    def _detect_dialect(self, connection_string: str) -> str:
        """Auto-detect dialect from connection string prefix.

        Raises:
            ValueError: If the dialect is unsupported.
        """
        lower = connection_string.lower()
        if lower.startswith("sqlite:///"):
            return "sqlite"
        msg = (
            f"Unsupported dialect in connection string: {connection_string!r}. "
            "Supported: sqlite:///path"
        )
        raise ValueError(msg)

    def _extract_path(
        self, connection_string: str, dialect_name: str
    ) -> str:
        """Extract database path from connection string.

        Raises:
            ValueError: If path contains traversal attempts.
        """
        if dialect_name == "sqlite":
            path = connection_string[len("sqlite:///"):]
            if not path:
                msg = "Empty path in SQLite connection string"
                raise ValueError(msg)
            # Security: reject path traversal
            if ".." in path:
                msg = "Path traversal (..) not allowed in connection string"
                raise ValueError(msg)
            return path
        msg = f"Path extraction not implemented for {dialect_name}"
        raise ValueError(msg)

    async def _create_connection(
        self, db_path: str, dialect_name: str
    ) -> Any:
        """Create async database connection.

        Raises:
            ValueError: If connection cannot be established.
        """
        if dialect_name == "sqlite":
            try:
                import aiosqlite
            except ImportError as exc:
                msg = "aiosqlite is required for SQLite introspection: pip install aiosqlite"
                raise ValueError(msg) from exc

            try:
                conn = await aiosqlite.connect(db_path)
                # Enable FK support for accurate introspection
                await conn.execute("PRAGMA foreign_keys = ON")
                return conn
            except Exception as exc:
                msg = f"Failed to connect to SQLite database: {db_path!r}"
                raise ValueError(msg) from exc

        msg = f"Connection creation not implemented for {dialect_name}"
        raise ValueError(msg)

    def _compute_fingerprint(self, tables: list[TableInfo]) -> str:
        """Compute SHA256 fingerprint of schema structure.

        Hash is based on sorted table names and their column names,
        so the same schema always produces the same fingerprint.
        """
        parts: list[str] = []
        for table in sorted(tables, key=lambda t: t.name):
            col_names = ",".join(c.name for c in table.columns)
            parts.append(f"{table.name}:{col_names}")

        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
