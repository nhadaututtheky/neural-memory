"""Tests for db_introspector: database schema extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_memory.engine.db_introspector import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    SchemaIntrospector,
    SchemaSnapshot,
    SQLiteDialect,
    TableInfo,
)


# ── Frozen dataclass tests ──────────────────────────────────────


class TestFrozenDataclasses:
    """All schema dataclasses are immutable."""

    def test_column_info_frozen(self) -> None:
        col = ColumnInfo("id", "INTEGER", False, True, None, None, None)
        with pytest.raises(AttributeError):
            col.name = "changed"  # type: ignore[misc]

    def test_table_info_frozen(self) -> None:
        table = TableInfo("users", None, (), (), (), 0, None)
        with pytest.raises(AttributeError):
            table.name = "changed"  # type: ignore[misc]

    def test_schema_snapshot_frozen(self) -> None:
        snap = SchemaSnapshot("test.db", "sqlite", ())
        with pytest.raises(AttributeError):
            snap.dialect = "changed"  # type: ignore[misc]

    def test_fk_info_frozen(self) -> None:
        fk = ForeignKeyInfo("user_id", "users", "id", None, None)
        with pytest.raises(AttributeError):
            fk.column_name = "changed"  # type: ignore[misc]

    def test_index_info_frozen(self) -> None:
        idx = IndexInfo("idx_email", ("email",), True)
        with pytest.raises(AttributeError):
            idx.unique = False  # type: ignore[misc]


# ── Dialect detection ───────────────────────────────────────────


class TestDialectDetection:
    """SchemaIntrospector auto-detects dialect from connection string."""

    def test_sqlite_detected(self) -> None:
        introspector = SchemaIntrospector()
        assert introspector._detect_dialect("sqlite:///test.db") == "sqlite"

    def test_sqlite_case_insensitive(self) -> None:
        introspector = SchemaIntrospector()
        assert introspector._detect_dialect("SQLite:///test.db") == "sqlite"

    def test_unknown_dialect_raises(self) -> None:
        introspector = SchemaIntrospector()
        with pytest.raises(ValueError, match="Unsupported dialect"):
            introspector._detect_dialect("mongodb://localhost/db")

    def test_empty_string_raises(self) -> None:
        introspector = SchemaIntrospector()
        with pytest.raises(ValueError, match="Unsupported dialect"):
            introspector._detect_dialect("")


# ── Path validation ─────────────────────────────────────────────


class TestPathValidation:
    """Security: path traversal rejected."""

    def test_path_traversal_rejected(self) -> None:
        introspector = SchemaIntrospector()
        with pytest.raises(ValueError, match="Path traversal"):
            introspector._extract_path("sqlite:///../../etc/passwd", "sqlite")

    def test_empty_path_rejected(self) -> None:
        introspector = SchemaIntrospector()
        with pytest.raises(ValueError, match="Empty path"):
            introspector._extract_path("sqlite:///", "sqlite")

    def test_valid_path_extracted(self) -> None:
        introspector = SchemaIntrospector()
        path = introspector._extract_path("sqlite:///data/test.db", "sqlite")
        assert path == "data/test.db"


# ── SQLite dialect ──────────────────────────────────────────────


class TestSQLiteDialect:
    """Tests using real aiosqlite with in-memory databases."""

    @pytest.mark.asyncio
    async def test_get_tables(self) -> None:
        """Discovers user tables, excludes sqlite_ system tables."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            await conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY)")

            dialect = SQLiteDialect()
            tables = await dialect.get_tables(conn)

            assert "users" in tables
            assert "posts" in tables
            assert not any(t.startswith("sqlite_") for t in tables)

    @pytest.mark.asyncio
    async def test_get_columns(self) -> None:
        """Parses column metadata: types, nullability, PK."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE users ("
                "  id INTEGER PRIMARY KEY,"
                "  email TEXT NOT NULL,"
                "  name TEXT,"
                "  age INTEGER DEFAULT 0"
                ")"
            )

            dialect = SQLiteDialect()
            columns = await dialect.get_columns(conn, "users")

            assert len(columns) == 4

            id_col = next(c for c in columns if c.name == "id")
            assert id_col.data_type == "INTEGER"
            assert id_col.primary_key is True

            email_col = next(c for c in columns if c.name == "email")
            assert email_col.nullable is False
            assert email_col.data_type == "TEXT"

            name_col = next(c for c in columns if c.name == "name")
            assert name_col.nullable is True

            age_col = next(c for c in columns if c.name == "age")
            assert age_col.default_value == "0"

    @pytest.mark.asyncio
    async def test_get_foreign_keys(self) -> None:
        """Parses FK constraints including cascade actions."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            await conn.execute(
                "CREATE TABLE posts ("
                "  id INTEGER PRIMARY KEY,"
                "  user_id INTEGER NOT NULL,"
                "  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"
                ")"
            )

            dialect = SQLiteDialect()
            fks = await dialect.get_foreign_keys(conn, "posts")

            assert len(fks) == 1
            fk = fks[0]
            assert fk.column_name == "user_id"
            assert fk.referenced_table == "users"
            assert fk.referenced_column == "id"
            assert fk.on_delete == "CASCADE"

    @pytest.mark.asyncio
    async def test_get_foreign_keys_no_action(self) -> None:
        """NO ACTION is normalized to None."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            await conn.execute(
                "CREATE TABLE posts ("
                "  id INTEGER PRIMARY KEY,"
                "  user_id INTEGER,"
                "  FOREIGN KEY (user_id) REFERENCES users(id)"
                ")"
            )

            dialect = SQLiteDialect()
            fks = await dialect.get_foreign_keys(conn, "posts")

            assert len(fks) == 1
            assert fks[0].on_delete is None
            assert fks[0].on_update is None

    @pytest.mark.asyncio
    async def test_get_indexes(self) -> None:
        """Parses user-created indexes (excludes auto-indexes)."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE users ("
                "  id INTEGER PRIMARY KEY,"
                "  email TEXT"
                ")"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX idx_email ON users(email)"
            )

            dialect = SQLiteDialect()
            indexes = await dialect.get_indexes(conn, "users")

            assert len(indexes) == 1
            idx = indexes[0]
            assert idx.name == "idx_email"
            assert idx.columns == ("email",)
            assert idx.unique is True

    @pytest.mark.asyncio
    async def test_get_row_count(self) -> None:
        """Returns exact row count for SQLite."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
            )
            await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
            await conn.execute("INSERT INTO users VALUES (2, 'Bob')")
            await conn.commit()

            dialect = SQLiteDialect()
            count = await dialect.get_row_count_estimate(conn, "users")
            assert count == 2

    @pytest.mark.asyncio
    async def test_empty_table(self) -> None:
        """Empty table returns 0 rows and empty lists."""
        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE empty_table (id INTEGER PRIMARY KEY)"
            )

            dialect = SQLiteDialect()
            count = await dialect.get_row_count_estimate(conn, "empty_table")
            assert count == 0

            fks = await dialect.get_foreign_keys(conn, "empty_table")
            assert fks == []

            indexes = await dialect.get_indexes(conn, "empty_table")
            assert indexes == []


# ── Full introspection integration ──────────────────────────────


class TestFullIntrospection:
    """End-to-end introspection with temp database files."""

    @pytest.mark.asyncio
    async def test_full_schema_snapshot(self) -> None:
        """Full introspection produces correct SchemaSnapshot."""
        import aiosqlite

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "CREATE TABLE categories ("
                "  id INTEGER PRIMARY KEY,"
                "  name TEXT NOT NULL"
                ")"
            )
            await conn.execute(
                "CREATE TABLE products ("
                "  id INTEGER PRIMARY KEY,"
                "  name TEXT NOT NULL,"
                "  category_id INTEGER,"
                "  FOREIGN KEY (category_id) REFERENCES categories(id)"
                ")"
            )
            await conn.commit()

        try:
            introspector = SchemaIntrospector()
            snapshot = await introspector.introspect(f"sqlite:///{db_path}")

            assert snapshot.dialect == "sqlite"
            assert len(snapshot.tables) == 2
            assert snapshot.schema_fingerprint != ""

            table_names = {t.name for t in snapshot.tables}
            assert "categories" in table_names
            assert "products" in table_names

            products = next(
                t for t in snapshot.tables if t.name == "products"
            )
            assert len(products.columns) == 3
            assert len(products.foreign_keys) == 1
            assert products.foreign_keys[0].referenced_table == "categories"
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_nonexistent_db_raises(self) -> None:
        """Missing database directory raises ValueError."""
        introspector = SchemaIntrospector()
        with pytest.raises(ValueError, match="Failed to connect"):
            await introspector.introspect(
                "sqlite:///nonexistent_dir_12345/test.db"
            )


# ── Schema fingerprint ──────────────────────────────────────────


class TestSchemaFingerprint:
    """Fingerprint for change detection."""

    def test_same_schema_same_fingerprint(self) -> None:
        introspector = SchemaIntrospector()
        tables = [
            TableInfo(
                "users",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
            TableInfo(
                "posts",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        fp1 = introspector._compute_fingerprint(tables)
        fp2 = introspector._compute_fingerprint(tables)
        assert fp1 == fp2

    def test_different_schema_different_fingerprint(self) -> None:
        introspector = SchemaIntrospector()
        tables1 = [
            TableInfo(
                "users",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        tables2 = [
            TableInfo(
                "orders",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        fp1 = introspector._compute_fingerprint(tables1)
        fp2 = introspector._compute_fingerprint(tables2)
        assert fp1 != fp2

    def test_column_change_changes_fingerprint(self) -> None:
        introspector = SchemaIntrospector()
        tables1 = [
            TableInfo(
                "users",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        tables2 = [
            TableInfo(
                "users",
                None,
                (
                    ColumnInfo("id", "INT", False, True),
                    ColumnInfo("email", "TEXT", True, False),
                ),
                (),
                (),
                0,
            ),
        ]
        fp1 = introspector._compute_fingerprint(tables1)
        fp2 = introspector._compute_fingerprint(tables2)
        assert fp1 != fp2

    def test_order_independent(self) -> None:
        """Table order doesn't affect fingerprint (sorted internally)."""
        introspector = SchemaIntrospector()
        tables_ab = [
            TableInfo(
                "a",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
            TableInfo(
                "b",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        tables_ba = [
            TableInfo(
                "b",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
            TableInfo(
                "a",
                None,
                (ColumnInfo("id", "INT", False, True),),
                (),
                (),
                0,
            ),
        ]
        fp1 = introspector._compute_fingerprint(tables_ab)
        fp2 = introspector._compute_fingerprint(tables_ba)
        assert fp1 == fp2
