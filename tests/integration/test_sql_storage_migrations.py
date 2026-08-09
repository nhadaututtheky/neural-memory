"""P3-T1: migration-safe SQLStorage initialization matrix."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION


async def _open_unified(path: Path) -> SQLStorage:
    store = SQLStorage(SQLiteDialect(path))
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_fresh_db_stamps_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    store = await _open_unified(db)
    try:
        row = await store._dialect.fetch_one("SELECT version FROM schema_version")
        assert row is not None
        assert int(row["version"]) == SCHEMA_VERSION
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_already_current_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "current.db"
    first = await _open_unified(db)
    await first.close()
    second = await _open_unified(db)
    try:
        row = await second._dialect.fetch_one("SELECT version FROM schema_version")
        assert int(row["version"]) == SCHEMA_VERSION
        brain = Brain.create(name="mig")
        await second.save_brain(brain)
        second.set_brain(brain.id)
        nid = await second.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="reinit-ok"))
        assert nid
        found = await second.find_neurons(content_exact="reinit-ok")
        assert len(found) == 1
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_old_schema_migrates_to_current(tmp_path: Path) -> None:
    """A DB stamped one version behind is upgraded via run_migrations."""
    db = tmp_path / "old.db"
    # Build full current schema first, rewind stamp so init re-runs migrations.
    store = await _open_unified(db)
    await store.close()
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            "UPDATE schema_version SET version = ?",
            (max(1, SCHEMA_VERSION - 1),),
        )
        await conn.commit()

    store = await _open_unified(db)
    try:
        row = await store._dialect.fetch_one("SELECT version FROM schema_version")
        assert int(row["version"]) == SCHEMA_VERSION
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_future_version_rejected_without_stamp_change(tmp_path: Path) -> None:
    db = tmp_path / "future.db"
    future = SCHEMA_VERSION + 99
    async with aiosqlite.connect(db) as conn:
        await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        await conn.execute("INSERT INTO schema_version (version) VALUES (?)", (future,))
        await conn.commit()

    store = SQLStorage(SQLiteDialect(db))
    with pytest.raises(RuntimeError, match="newer than this package"):
        await store.initialize()
    await store.close()

    async with aiosqlite.connect(db) as conn:
        async with conn.execute("SELECT version FROM schema_version") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == future


@pytest.mark.asyncio
async def test_corrupt_version_rejected(tmp_path: Path) -> None:
    db = tmp_path / "corrupt.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute("CREATE TABLE schema_version (version TEXT PRIMARY KEY)")
        await conn.execute("INSERT INTO schema_version (version) VALUES ('not-a-number')")
        await conn.commit()

    store = SQLStorage(SQLiteDialect(db))
    with pytest.raises(RuntimeError, match="Corrupt schema_version"):
        await store.initialize()
    await store.close()
