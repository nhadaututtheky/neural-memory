"""SQLite storage backend for persistent neural memory."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.neuron_cache import NeuronLookupCache
from neural_memory.storage.read_pool import ReadPool
from neural_memory.storage.sqlite_action_log import SQLiteActionLogMixin
from neural_memory.storage.sqlite_brain_ops import SQLiteBrainMixin
from neural_memory.storage.sqlite_coactivation import SQLiteCoActivationMixin
from neural_memory.storage.sqlite_fibers import SQLiteFiberMixin
from neural_memory.storage.sqlite_maturation import SQLiteMaturationMixin
from neural_memory.storage.sqlite_neurons import SQLiteNeuronMixin
from neural_memory.storage.sqlite_projects import SQLiteProjectMixin
from neural_memory.storage.sqlite_schema import (
    SCHEMA,
    SCHEMA_VERSION,
    ensure_fts_tables,
    run_migrations,
)
from neural_memory.storage.sqlite_synapses import SQLiteSynapseMixin
from neural_memory.storage.sqlite_sync_state import SQLiteSyncStateMixin
from neural_memory.storage.sqlite_typed import SQLiteTypedMemoryMixin
from neural_memory.storage.sqlite_versioning import SQLiteVersioningMixin

logger = logging.getLogger(__name__)


class SQLiteStorage(
    SQLiteNeuronMixin,
    SQLiteSynapseMixin,
    SQLiteFiberMixin,
    SQLiteTypedMemoryMixin,
    SQLiteProjectMixin,
    SQLiteMaturationMixin,
    SQLiteActionLogMixin,
    SQLiteCoActivationMixin,
    SQLiteVersioningMixin,
    SQLiteSyncStateMixin,
    SQLiteBrainMixin,
    NeuralStorage,
):
    """SQLite-based storage for persistent neural memory.

    Good for single-instance deployment and local development.
    Data persists to disk and survives restarts.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).resolve()
        self._conn: aiosqlite.Connection | None = None
        self._current_brain_id: str | None = None
        self._has_fts: bool = False
        self._neuron_cache = NeuronLookupCache(ttl_seconds=30.0, max_entries=500)
        self._read_pool: ReadPool | None = None

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
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=-8000")

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

        # FTS5 virtual table + sync triggers (individual execute, not executescript)
        await ensure_fts_tables(self._conn)
        self._has_fts = await self._check_fts_available()

        # Stamp version for brand-new databases
        async with self._conn.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()
            if row is None:
                await self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
                await self._conn.commit()

        # Initialize read-only connection pool for parallel reads
        self._read_pool = ReadPool(self._db_path)
        await self._read_pool.initialize()

    async def close(self) -> None:
        """Close database connection and reader pool."""
        if self._read_pool:
            await self._read_pool.close()
            self._read_pool = None
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def current_brain_id(self) -> str | None:
        """The active brain ID, or None if not set."""
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context for operations."""
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        """Get current brain ID or raise error."""
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    def _ensure_conn(self) -> aiosqlite.Connection:
        """Ensure writer connection is available."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        """Get a read-only connection from the pool (falls back to writer)."""
        if self._read_pool is not None:
            return self._read_pool.acquire()
        return self._ensure_conn()

    async def _check_fts_available(self) -> bool:
        """Check whether the neurons_fts table is usable.

        Returns False if FTS5 is not compiled into the SQLite build
        (rare, but possible on some minimal distributions).
        """
        conn = self._ensure_conn()
        try:
            await conn.execute("SELECT * FROM neurons_fts LIMIT 0")
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            logger.debug("FTS5 table not available", exc_info=True)
            return False

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        conn = self._ensure_read_conn()

        async with conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM neurons WHERE brain_id = ?) as neuron_count,
                (SELECT COUNT(*) FROM synapses WHERE brain_id = ?) as synapse_count,
                (SELECT COUNT(*) FROM fibers WHERE brain_id = ?) as fiber_count,
                (SELECT COUNT(*) FROM projects WHERE brain_id = ?) as project_count
            """,
            (brain_id, brain_id, brain_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            return {
                "neuron_count": row["neuron_count"] if row else 0,
                "synapse_count": row["synapse_count"] if row else 0,
                "fiber_count": row["fiber_count"] if row else 0,
                "project_count": row["project_count"] if row else 0,
            }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        conn = self._ensure_read_conn()
        basic_stats = await self.get_stats(brain_id)

        # DB file size
        db_size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0

        # Hot neurons (most frequently accessed)
        hot_neurons: list[dict[str, Any]] = []
        async with conn.execute(
            """SELECT ns.neuron_id, n.content, n.type,
                      ns.activation_level, ns.access_frequency
               FROM neuron_states ns
               JOIN neurons n ON n.brain_id = ns.brain_id AND n.id = ns.neuron_id
               WHERE ns.brain_id = ?
               ORDER BY ns.access_frequency DESC
               LIMIT 10""",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                hot_neurons.append(
                    {
                        "neuron_id": row["neuron_id"],
                        "content": row["content"],
                        "type": row["type"],
                        "activation_level": row["activation_level"],
                        "access_frequency": row["access_frequency"],
                    }
                )

        # Today's fibers
        from neural_memory.utils.timeutils import utcnow

        today_midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM fibers WHERE brain_id = ? AND created_at >= ?",
            (brain_id, today_midnight.isoformat()),
        ) as cursor:
            today_row = await cursor.fetchone()
            today_fibers_count = today_row["cnt"] if today_row else 0

        # Synapse stats by type
        synapse_stats: dict[str, Any] = {
            "avg_weight": 0.0,
            "total_reinforcements": 0,
            "by_type": {},
        }
        async with conn.execute(
            """SELECT type, AVG(weight) as avg_w, SUM(reinforced_count) as total_r, COUNT(*) as cnt
               FROM synapses WHERE brain_id = ?
               GROUP BY type""",
            (brain_id,),
        ) as cursor:
            total_weight = 0.0
            total_count = 0
            total_reinforcements = 0
            async for row in cursor:
                synapse_stats["by_type"][row["type"]] = {
                    "count": row["cnt"],
                    "avg_weight": round(row["avg_w"], 4),
                    "total_reinforcements": row["total_r"] or 0,
                }
                total_weight += (row["avg_w"] or 0.0) * row["cnt"]
                total_count += row["cnt"]
                total_reinforcements += row["total_r"] or 0

        if total_count > 0:
            synapse_stats["avg_weight"] = round(total_weight / total_count, 4)
        synapse_stats["total_reinforcements"] = total_reinforcements

        # Neuron type breakdown
        neuron_type_breakdown: dict[str, int] = {}
        async with conn.execute(
            "SELECT type, COUNT(*) as cnt FROM neurons WHERE brain_id = ? GROUP BY type",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                neuron_type_breakdown[row["type"]] = row["cnt"]

        # Memory time range
        oldest_memory: str | None = None
        newest_memory: str | None = None
        async with conn.execute(
            "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM fibers WHERE brain_id = ?",
            (brain_id,),
        ) as cursor:
            time_row = await cursor.fetchone()
            if time_row:
                oldest_memory = time_row["oldest"]
                newest_memory = time_row["newest"]

        return {
            **basic_stats,
            "db_size_bytes": db_size_bytes,
            "hot_neurons": hot_neurons,
            "today_fibers_count": today_fibers_count,
            "synapse_stats": synapse_stats,
            "neuron_type_breakdown": neuron_type_breakdown,
            "oldest_memory": oldest_memory,
            "newest_memory": newest_memory,
        }

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        conn = self._ensure_conn()

        brain_tables = (
            "sync_states",
            "action_events",
            "brain_versions",
            "memory_maturations",
            "co_activation_events",
            "typed_memories",
            "projects",
            "fiber_neurons",
            "fibers",
            "synapses",
            "neuron_states",
            "neurons",
        )
        for table in brain_tables:
            # Table name is from a hardcoded tuple — safe to interpolate.
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
