"""Tests for SQL Dialect abstraction layer."""

from __future__ import annotations

from datetime import UTC

import pytest

from neural_memory.storage.sql.postgres_dialect import PostgresDialect
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

# ── SQLiteDialect ───────────────────────────────────────────────────


class TestSQLiteDialectPlaceholders:
    def test_ph_always_question_mark(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        assert d.ph(1) == "?"
        assert d.ph(5) == "?"
        assert d.ph(100) == "?"

    def test_phs(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        assert d.phs(3) == "?, ?, ?"
        assert d.phs(1) == "?"
        assert d.phs(0) == ""

    def test_in_clause(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        sql, params = d.in_clause(1, ["a", "b", "c"])
        assert sql == "IN (?, ?, ?)"
        assert params == ["a", "b", "c"]

    def test_in_clause_empty(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        sql, params = d.in_clause(1, [])
        assert sql == "IN ()"
        assert params == []


class TestSQLiteDialectSQL:
    def test_upsert(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        sql = d.upsert_sql("neurons", ["id", "content"], ["id"], ["content"])
        assert "INSERT INTO neurons" in sql
        assert "ON CONFLICT (id)" in sql
        assert "DO UPDATE SET content = excluded.content" in sql

    def test_insert_or_ignore(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        sql = d.insert_or_ignore_sql(
            "fiber_neurons",
            ["brain_id", "fiber_id", "neuron_id"],
            ["brain_id", "fiber_id", "neuron_id"],
        )
        assert "DO NOTHING" in sql

    def test_json_extract(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        assert d.json_extract("metadata", "type") == "json_extract(metadata, '$.type')"

    def test_auto_increment_pk(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        assert d.auto_increment_pk() == "INTEGER PRIMARY KEY AUTOINCREMENT"

    def test_timestamp_type(self):
        d = SQLiteDialect(db_path="/tmp/test.db")
        assert d.timestamp_type() == "TEXT"


class TestSQLiteDialectFeatures:
    def test_name(self):
        assert SQLiteDialect(db_path="/tmp/test.db").name == "sqlite"

    def test_no_vector(self):
        assert not SQLiteDialect(db_path="/tmp/test.db").supports_vector

    def test_no_ilike(self):
        assert not SQLiteDialect(db_path="/tmp/test.db").supports_ilike


class TestSQLiteDialectLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self, tmp_path):
        d = SQLiteDialect(db_path=tmp_path / "test.db")
        await d.initialize()
        assert d._conn is not None
        await d.close()
        assert d._conn is None

    @pytest.mark.asyncio
    async def test_execute_and_fetch(self, tmp_path):
        d = SQLiteDialect(db_path=tmp_path / "test.db")
        await d.initialize()
        try:
            await d.execute("CREATE TABLE test (id TEXT, value TEXT)")
            await d.execute("INSERT INTO test (id, value) VALUES (?, ?)", ["1", "hello"])

            rows = await d.fetch_all("SELECT * FROM test")
            assert len(rows) == 1
            assert rows[0]["id"] == "1"
            assert rows[0]["value"] == "hello"

            row = await d.fetch_one("SELECT * FROM test WHERE id = ?", ["1"])
            assert row is not None
            assert row["value"] == "hello"

            row = await d.fetch_one("SELECT * FROM test WHERE id = ?", ["999"])
            assert row is None
        finally:
            await d.close()

    @pytest.mark.asyncio
    async def test_execute_many(self, tmp_path):
        d = SQLiteDialect(db_path=tmp_path / "test.db")
        await d.initialize()
        try:
            await d.execute("CREATE TABLE test (id TEXT, value TEXT)")
            await d.execute_many(
                "INSERT INTO test (id, value) VALUES (?, ?)",
                [["1", "a"], ["2", "b"], ["3", "c"]],
            )
            rows = await d.fetch_all("SELECT * FROM test ORDER BY id")
            assert len(rows) == 3
        finally:
            await d.close()


# ── PostgresDialect ─────────────────────────────────────────────────


class TestPostgresDialectPlaceholders:
    def test_ph_numbered(self):
        d = PostgresDialect()
        assert d.ph(1) == "$1"
        assert d.ph(5) == "$5"
        assert d.ph(100) == "$100"

    def test_phs(self):
        d = PostgresDialect()
        assert d.phs(3) == "$1, $2, $3"
        assert d.phs(3, start=5) == "$5, $6, $7"
        assert d.phs(1) == "$1"

    def test_in_clause(self):
        d = PostgresDialect()
        sql, params = d.in_clause(1, ["a", "b", "c"])
        assert sql == "= ANY($1)"
        assert params == [["a", "b", "c"]]


class TestPostgresDialectSQL:
    def test_upsert(self):
        d = PostgresDialect()
        sql = d.upsert_sql("neurons", ["id", "content"], ["id"], ["content"])
        assert "INSERT INTO neurons" in sql
        assert "ON CONFLICT (id)" in sql
        assert "DO UPDATE SET content = EXCLUDED.content" in sql

    def test_json_extract(self):
        d = PostgresDialect()
        assert d.json_extract("metadata", "type") == "metadata->>'type'"

    def test_json_contains_key(self):
        d = PostgresDialect()
        assert d.json_contains_key("metadata", 1) == "metadata ? $1"

    def test_json_array_contains(self):
        d = PostgresDialect()
        assert d.json_array_contains("tags", 2) == "tags @> $2::jsonb"

    def test_auto_increment_pk(self):
        d = PostgresDialect()
        assert d.auto_increment_pk() == "SERIAL PRIMARY KEY"

    def test_timestamp_type(self):
        d = PostgresDialect()
        assert d.timestamp_type() == "TIMESTAMPTZ"

    def test_jsonb_type(self):
        d = PostgresDialect()
        assert d.jsonb_type() == "JSONB"


class TestPostgresDialectFeatures:
    def test_name(self):
        assert PostgresDialect().name == "postgres"

    def test_supports_ilike(self):
        assert PostgresDialect().supports_ilike

    def test_supports_fts(self):
        assert PostgresDialect().supports_fts


class TestPostgresDialectDatetime:
    def test_serialize_naive(self):
        from datetime import datetime

        d = PostgresDialect()
        dt = datetime(2026, 1, 1, 12, 0, 0)
        result = d.serialize_dt(dt)
        assert result.tzinfo == UTC

    def test_serialize_aware(self):
        from datetime import datetime

        d = PostgresDialect()
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = d.serialize_dt(dt)
        assert result is dt

    def test_serialize_none(self):
        d = PostgresDialect()
        assert d.serialize_dt(None) is None

    def test_normalize_string(self):
        d = PostgresDialect()
        result = d.normalize_dt("2026-01-01T12:00:00+00:00")
        assert result is not None
        assert result.year == 2026


# ── Cross-dialect compatibility ─────────────────────────────────────


class TestDialectCompatibility:
    """Verify both dialects produce semantically equivalent SQL for the same operation."""

    def test_upsert_both_use_on_conflict(self):
        sqlite = SQLiteDialect(db_path="/tmp/test.db")
        pg = PostgresDialect()

        for d in (sqlite, pg):
            sql = d.upsert_sql("t", ["a", "b"], ["a"], ["b"])
            assert "ON CONFLICT (a)" in sql
            assert "DO UPDATE SET" in sql

    def test_insert_or_ignore_both_use_do_nothing(self):
        sqlite = SQLiteDialect(db_path="/tmp/test.db")
        pg = PostgresDialect()

        for d in (sqlite, pg):
            sql = d.insert_or_ignore_sql("t", ["a", "b"], ["a"])
            assert "DO NOTHING" in sql
