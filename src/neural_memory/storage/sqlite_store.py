"""SQLite storage backend for persistent neural memory."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.sqlite_brain_ops import SQLiteBrainMixin
from neural_memory.storage.sqlite_fibers import SQLiteFiberMixin
from neural_memory.storage.sqlite_neurons import SQLiteNeuronMixin
from neural_memory.storage.sqlite_projects import SQLiteProjectMixin
from neural_memory.storage.sqlite_schema import SCHEMA, SCHEMA_VERSION, run_migrations
from neural_memory.storage.sqlite_synapses import SQLiteSynapseMixin
from neural_memory.storage.sqlite_typed import SQLiteTypedMemoryMixin


class SQLiteStorage(
    SQLiteNeuronMixin,
    SQLiteSynapseMixin,
    SQLiteFiberMixin,
    SQLiteTypedMemoryMixin,
    SQLiteProjectMixin,
    SQLiteBrainMixin,
    NeuralStorage,
):
    """SQLite-based storage for persistent neural memory.

    Good for single-instance deployment and local development.
    Data persists to disk and survives restarts.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._current_brain_id: str | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema.

        For new databases, creates all tables at the latest schema version.
        For existing databases, runs pending migrations first (e.g. adding
        missing columns like conductivity) then applies the full schema
        so that indexes on new columns can be created safely.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute("PRAGMA foreign_keys = ON")

        # Ensure version table exists so we can read the current version
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )
        await self._conn.commit()

        # Check stored version and migrate if needed BEFORE full schema
        async with self._conn.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()

        if row is not None and row["version"] < SCHEMA_VERSION:
            await run_migrations(self._conn, row["version"])

        # Full schema: CREATE TABLE/INDEX IF NOT EXISTS (safe after migration)
        await self._conn.executescript(SCHEMA)

        # Stamp version for brand-new databases
        async with self._conn.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()
            if row is None:
                await self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
                await self._conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context for operations."""
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        """Get current brain ID or raise error."""
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    def _ensure_conn(self) -> aiosqlite.Connection:
        """Ensure connection is available."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        conn = self._ensure_conn()
        stats: dict[str, int] = {}

        for table, key in [
            ("neurons", "neuron_count"),
            ("synapses", "synapse_count"),
            ("fibers", "fiber_count"),
            ("projects", "project_count"),
        ]:
            async with conn.execute(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE brain_id = ?", (brain_id,)
            ) as cursor:
                row = await cursor.fetchone()
                stats[key] = row["cnt"] if row else 0

        return stats

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        conn = self._ensure_conn()

        for table in [
            "typed_memories",
            "projects",
            "fibers",
            "synapses",
            "neuron_states",
            "neurons",
        ]:
            await conn.execute(f"DELETE FROM {table} WHERE brain_id = ?", (brain_id,))

        await conn.execute("DELETE FROM brains WHERE id = ?", (brain_id,))
        await conn.commit()

    # ========== Compatibility with PersistentStorage ==========

    def disable_auto_save(self) -> None:
        """No-op for SQLite (transactions handle this)."""

    def enable_auto_save(self) -> None:
        """No-op for SQLite (transactions handle this)."""

    async def batch_save(self) -> None:
        """Commit any pending transactions."""
        conn = self._ensure_conn()
        await conn.commit()

    async def _save_to_file(self) -> None:
        """No-op for SQLite (auto-persisted)."""
