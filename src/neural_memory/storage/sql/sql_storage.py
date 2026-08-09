"""Unified SQL storage — composes all dialect-agnostic mixins into one class.

SQLStorage works with any SQL backend by delegating engine-specific behaviour
to a :class:`Dialect` instance (SQLiteDialect, PostgresDialect, etc.).

Usage::

    from neural_memory.storage.sql.sql_storage import SQLStorage
    from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

    dialect = SQLiteDialect("/path/to/brain.db")
    store = SQLStorage(dialect)
    await store.initialize()
"""

from __future__ import annotations

import asyncio
import logging

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.neuron_cache import NeuronLookupCache
from neural_memory.storage.sql.dialect import Dialect

# -- Domain mixins (19 simple mixins) --
from neural_memory.storage.sql.mixins.action_log import ActionLogMixin
from neural_memory.storage.sql.mixins.alerts import AlertsMixin
from neural_memory.storage.sql.mixins.brain_ops import BrainOpsMixin
from neural_memory.storage.sql.mixins.calibration import CalibrationMixin
from neural_memory.storage.sql.mixins.change_log import ChangeLogMixin
from neural_memory.storage.sql.mixins.coactivation import CoActivationMixin
from neural_memory.storage.sql.mixins.cognitive import CognitiveMixin
from neural_memory.storage.sql.mixins.compression import CompressionMixin
from neural_memory.storage.sql.mixins.depth_priors import DepthPriorsMixin
from neural_memory.storage.sql.mixins.devices import DevicesMixin
from neural_memory.storage.sql.mixins.drift import DriftMixin
from neural_memory.storage.sql.mixins.enrichment_jobs import EnrichmentJobsMixin
from neural_memory.storage.sql.mixins.entity_refs import EntityRefsMixin
from neural_memory.storage.sql.mixins.fibers import FiberMixin
from neural_memory.storage.sql.mixins.maturation import MaturationMixin
from neural_memory.storage.sql.mixins.merkle import MerkleMixin

# -- Core mixins (neurons, synapses, fibers, brain ops, typed memory, cognitive) --
from neural_memory.storage.sql.mixins.neurons import NeuronMixin
from neural_memory.storage.sql.mixins.projects import ProjectsMixin
from neural_memory.storage.sql.mixins.reviews import ReviewsMixin
from neural_memory.storage.sql.mixins.sessions import SessionsMixin
from neural_memory.storage.sql.mixins.sources import SourcesMixin
from neural_memory.storage.sql.mixins.synapses import SynapseMixin
from neural_memory.storage.sql.mixins.sync_state import SyncStateMixin
from neural_memory.storage.sql.mixins.tool_events import ToolEventsMixin
from neural_memory.storage.sql.mixins.training_files import TrainingFilesMixin
from neural_memory.storage.sql.mixins.typed_memory import TypedMemoryMixin
from neural_memory.storage.sql.mixins.vector_search import VectorSearchMixin
from neural_memory.storage.sql.mixins.versioning import VersioningMixin

logger = logging.getLogger(__name__)


class SQLStorage(
    # Core mixins
    NeuronMixin,
    SynapseMixin,
    FiberMixin,
    BrainOpsMixin,
    TypedMemoryMixin,
    CognitiveMixin,
    # Domain mixins
    ActionLogMixin,
    AlertsMixin,
    CalibrationMixin,
    ChangeLogMixin,
    CoActivationMixin,
    CompressionMixin,
    DepthPriorsMixin,
    DevicesMixin,
    DriftMixin,
    EnrichmentJobsMixin,
    EntityRefsMixin,
    MaturationMixin,
    MerkleMixin,
    ProjectsMixin,
    ReviewsMixin,
    SessionsMixin,
    SourcesMixin,
    SyncStateMixin,
    ToolEventsMixin,
    TrainingFilesMixin,
    VectorSearchMixin,
    VersioningMixin,
    # ABC last — mixins satisfy abstract methods
    NeuralStorage,
):
    """Unified SQL storage for neural memory.

    Composes 26 dialect-agnostic mixins that implement all abstract methods
    defined in :class:`NeuralStorage`. Each mixin accesses the database
    exclusively through ``self._dialect``, making SQLStorage work with
    SQLite, PostgreSQL, or any future SQL backend.
    """

    def __init__(self, dialect: Dialect) -> None:
        self._dialect = dialect
        self._current_brain_id: str | None = None
        self._neuron_cache = NeuronLookupCache(ttl_seconds=30.0, max_entries=500)
        # SimHash snapshot: (brain_id, data_version|None, monotonic_ts, hashes)
        self._hash_snapshot: tuple[str, int | None, float, list[tuple[str, int]]] | None = None
        self._init_vector_search()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the dialect and create schema tables.

        SQLite path is migration-safe (mirrors legacy SQLiteStorage):
        - Fresh DB: apply SCHEMA, FTS, stamp SCHEMA_VERSION
        - Old DB: run_migrations then SCHEMA IF NOT EXISTS
        - Current: no-op migrations, re-apply IF NOT EXISTS DDL
        - Future / corrupt version: raise without mutating user data mid-flight
        """
        await self._dialect.initialize()

        if self._dialect.name == "sqlite":
            await self._initialize_sqlite_schema()
        elif self._dialect.name == "postgres":
            # PostgreSQL needs a compatible DDL variant — the dialect
            # provides it via get_schema_ddl().
            ddl = self._dialect.get_schema_ddl()
            await self._dialect.execute_script(ddl)
        else:
            # Fallback for unknown dialects: try the SQLite schema (best effort)
            from neural_memory.storage.sqlite_schema import SCHEMA

            await self._dialect.execute_script(SCHEMA)

        logger.info("SQLStorage initialized with %s dialect", self._dialect.name)

    async def _initialize_sqlite_schema(self) -> None:
        """Migration-safe SQLite schema bootstrap for the unified adapter."""
        from neural_memory.storage.sqlite_schema import (
            SCHEMA,
            SCHEMA_VERSION,
            ensure_fiber_fts_tables,
            ensure_fts_tables,
            run_migrations,
        )

        conn = self._dialect._ensure_conn()  # type: ignore[attr-defined]

        # Version table must exist before we read the stamp.
        await self._dialect.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )

        row = await self._dialect.fetch_one("SELECT version FROM schema_version")
        current: int | None = None
        if row is not None:
            raw = row.get("version") if isinstance(row, dict) else None
            if raw is None and row:
                raw = next(iter(row.values()), None)
            try:
                current = int(raw) if raw is not None else None
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Corrupt schema_version value {raw!r}; refusing to migrate. "
                    "Restore from backup or delete the corrupt DB after export."
                ) from exc

        if current is not None and current > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current} is newer than this package "
                f"({SCHEMA_VERSION}). Upgrade neural-memory or restore an older DB."
            )

        if current is not None and current < 0:
            raise RuntimeError(f"Invalid schema_version {current}; refusing to migrate.")

        if current is not None and current < SCHEMA_VERSION:
            logger.info(
                "SQLStorage migrating schema %d → %d",
                current,
                SCHEMA_VERSION,
            )
            try:
                await run_migrations(conn, current)
            except Exception:
                logger.error(
                    "SQLStorage migration failed at version %s; DB left for recovery",
                    current,
                    exc_info=True,
                )
                raise

        # Full schema: CREATE TABLE/INDEX IF NOT EXISTS (safe after migration)
        await self._dialect.execute_script(SCHEMA)

        # FTS5 virtual tables + sync triggers (individual execute, not script)
        await ensure_fts_tables(conn)
        await ensure_fiber_fts_tables(conn)
        self._dialect._has_fts = True  # type: ignore[attr-defined]

        # Stamp version for brand-new databases (no prior row)
        row_after = await self._dialect.fetch_one("SELECT version FROM schema_version")
        if row_after is None:
            await self._dialect.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            logger.info("SQLStorage stamped schema_version=%d (fresh DB)", SCHEMA_VERSION)
        else:
            # Ensure stamp matches current package after successful migrate/init
            stamped = row_after.get("version") if isinstance(row_after, dict) else None
            try:
                stamped_i = int(stamped) if stamped is not None else None
            except (TypeError, ValueError):
                stamped_i = None
            if stamped_i is not None and stamped_i < SCHEMA_VERSION:
                await self._dialect.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )

    async def close(self) -> None:
        """Close the dialect connection(s).

        Drains pending pipeline background tasks first so writes don't
        race the connection teardown (matters on Windows + aiosqlite).
        """
        tasks = getattr(self, "_pipeline_bg_tasks", None)
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)

        self._close_vector_index()
        await self._dialect.close()
        logger.debug("SQLStorage closed")

    # ------------------------------------------------------------------
    # Brain context
    # ------------------------------------------------------------------

    @property
    def brain_id(self) -> str | None:
        """The active brain ID, or None if not set."""
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context for operations."""
        if brain_id != self._current_brain_id:
            self._close_vector_index()  # Each brain has its own sidecar
            self._hash_snapshot = None  # Never leak hashes across brains
            self._neuron_cache.invalidate()
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        """Get current brain ID or raise if not set."""
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    # ------------------------------------------------------------------
    # Batch / auto-save compatibility
    # ------------------------------------------------------------------

    def disable_auto_save(self) -> None:
        """No-op — SQL backends commit per-statement or per-transaction."""

    def enable_auto_save(self) -> None:
        """No-op — SQL backends commit per-statement or per-transaction."""

    async def batch_save(self) -> None:
        """No-op — writes are committed by the dialect immediately."""

    async def _save_to_file(self) -> None:
        """No-op — SQL backends auto-persist."""
