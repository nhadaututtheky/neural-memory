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

import logging

from neural_memory.storage.base import NeuralStorage
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
from neural_memory.storage.sql.mixins.versioning import VersioningMixin

# -- Schema DDL --
from neural_memory.storage.sqlite_schema import SCHEMA

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the dialect and create schema tables."""
        await self._dialect.initialize()
        # Create all tables and indexes via the shared DDL script.
        # The SCHEMA constant uses CREATE TABLE/INDEX IF NOT EXISTS,
        # so it is safe to run repeatedly and works for both SQLite
        # and PostgreSQL for the core table definitions.
        await self._dialect.execute_script(SCHEMA)
        logger.info("SQLStorage initialized with %s dialect", self._dialect.name)

    async def close(self) -> None:
        """Close the dialect connection(s)."""
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
