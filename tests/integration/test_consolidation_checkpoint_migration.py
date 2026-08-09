"""Migration + API tests for consolidation_checkpoints (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION, run_migrations


@pytest.fixture
async def storage(tmp_path: Path) -> SQLStorage:
    store = SQLStorage(SQLiteDialect(str(tmp_path / "cp.db")))
    await store.initialize()
    brain = Brain.create(name="cp", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_schema_version_41(storage: SQLStorage) -> None:
    row = await storage._dialect.fetch_one("SELECT version FROM schema_version")
    assert int(row["version"]) == SCHEMA_VERSION == 41


@pytest.mark.asyncio
async def test_migrate_40_to_41(tmp_path: Path) -> None:
    db = tmp_path / "mig41.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        await conn.execute("INSERT INTO schema_version (version) VALUES (40)")
        await conn.execute(
            """CREATE TABLE brains (
                id TEXT PRIMARY KEY, name TEXT, config TEXT,
                created_at TEXT, updated_at TEXT
            )"""
        )
        await conn.commit()
        final = await run_migrations(conn, 40)
        assert final == 41
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidation_checkpoints'"
        )
        assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_save_get_checkpoint_monotonic(storage: SQLStorage) -> None:
    cp = ConsolidationCheckpoint.create(
        brain_id=storage.brain_id or "",
        strategy="prune",
        last_sequence=10,
    )
    saved = await storage.save_consolidation_checkpoint(cp)
    assert saved.last_sequence == 10

    got = await storage.get_consolidation_checkpoint("prune")
    assert got is not None
    assert got.last_sequence == 10

    # Forward ok
    await storage.save_consolidation_checkpoint(
        ConsolidationCheckpoint.create(
            brain_id=storage.brain_id or "",
            strategy="prune",
            last_sequence=20,
        )
    )
    # Backward rejected
    with pytest.raises(ValueError, match="backward"):
        await storage.save_consolidation_checkpoint(
            ConsolidationCheckpoint.create(
                brain_id=storage.brain_id or "",
                strategy="prune",
                last_sequence=5,
            )
        )


@pytest.mark.asyncio
async def test_strategies_independent(storage: SQLStorage) -> None:
    bid = storage.brain_id or ""
    await storage.save_consolidation_checkpoint(
        ConsolidationCheckpoint.create(brain_id=bid, strategy="prune", last_sequence=5)
    )
    await storage.save_consolidation_checkpoint(
        ConsolidationCheckpoint.create(brain_id=bid, strategy="merge", last_sequence=9)
    )
    prune = await storage.get_consolidation_checkpoint("prune")
    merge = await storage.get_consolidation_checkpoint("merge")
    assert prune and prune.last_sequence == 5
    assert merge and merge.last_sequence == 9


@pytest.mark.asyncio
async def test_reset_checkpoint(storage: SQLStorage) -> None:
    bid = storage.brain_id or ""
    await storage.save_consolidation_checkpoint(
        ConsolidationCheckpoint.create(brain_id=bid, strategy="mature", last_sequence=7)
    )
    n = await storage.reset_consolidation_checkpoint("mature", audit_reason="test")
    assert n >= 1
    assert await storage.get_consolidation_checkpoint("mature") is None
