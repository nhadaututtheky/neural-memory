"""Migration tests for enrichment_jobs schema (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from neural_memory.storage.sqlite_schema import SCHEMA_VERSION, run_migrations


@pytest.mark.asyncio
async def test_migrate_39_to_40(tmp_path: Path) -> None:
    db = tmp_path / "mig.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        await conn.execute("INSERT INTO schema_version (version) VALUES (39)")
        # Minimal brains table for FK
        await conn.execute(
            """CREATE TABLE brains (
                id TEXT PRIMARY KEY, name TEXT, config TEXT,
                created_at TEXT, updated_at TEXT
            )"""
        )
        await conn.commit()

        final = await run_migrations(conn, 39)
        assert final == SCHEMA_VERSION == 42

        row = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='enrichment_jobs'"
        )
        assert await row.fetchone() is not None

        # Unique index on idempotency
        cols = await conn.execute("PRAGMA table_info(enrichment_jobs)")
        col_names = {r[1] for r in await cols.fetchall()}
        assert "idempotency_key" in col_names
        assert "lease_owner" in col_names


@pytest.mark.asyncio
async def test_fresh_install_has_enrichment_jobs(tmp_path: Path) -> None:
    from neural_memory.storage.sql.sql_storage import SQLStorage
    from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

    store = SQLStorage(SQLiteDialect(str(tmp_path / "fresh.db")))
    await store.initialize()
    row = await store._dialect.fetch_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='enrichment_jobs'"
    )
    assert row is not None
    ver = await store._dialect.fetch_one("SELECT version FROM schema_version")
    assert int(ver["version"]) == 42
    await store.close()


@pytest.mark.asyncio
async def test_already_migrated_noop(tmp_path: Path) -> None:
    from neural_memory.storage.sql.sql_storage import SQLStorage
    from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

    path = str(tmp_path / "twice.db")
    store = SQLStorage(SQLiteDialect(path))
    await store.initialize()
    await store.close()

    store2 = SQLStorage(SQLiteDialect(path))
    await store2.initialize()  # must not raise
    ver = await store2._dialect.fetch_one("SELECT version FROM schema_version")
    assert int(ver["version"]) == 42
    await store2.close()
