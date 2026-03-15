"""PostgreSQL + pgvector composite storage backend."""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_brains import PostgresBrainMixin
from neural_memory.storage.postgres.postgres_fibers import PostgresFiberMixin
from neural_memory.storage.postgres.postgres_neurons import PostgresNeuronMixin
from neural_memory.storage.postgres.postgres_schema import ensure_schema
from neural_memory.storage.postgres.postgres_synapses import PostgresSynapseMixin
from neural_memory.storage.postgres.postgres_typed import PostgresTypedMemoryMixin

logger = logging.getLogger(__name__)


class PostgreSQLStorage(
    PostgresTypedMemoryMixin,
    PostgresNeuronMixin,
    PostgresSynapseMixin,
    PostgresFiberMixin,
    PostgresBrainMixin,
    PostgresBaseMixin,
    NeuralStorage,
):
    """PostgreSQL-backed storage with pgvector for embeddings and tsvector for FTS.

    Supported operations:
        Core CRUD — neurons, synapses, fibers, brains, typed memories
        Vector search — pgvector cosine similarity via find_neurons_by_embedding()
        Full-text search — tsvector via find_neurons(content_contains=...)
        Graph traversal — get_neighbors(), get_path()
        Import/export — export_brain(), import_brain()

    Not yet implemented (requires SQLite backend):
        Cognitive tools (nmem_hypothesize, nmem_schema, nmem_reflect)
        Sync engine (nmem_sync, incremental merge)
        Review & source registry (nmem_review, nmem_source)
        Hooks (pre_compact, post_tool_use)
        Calibration, drift detection, priming

    Usage:
        storage = PostgreSQLStorage(
            host="localhost", port=5432, database="neuralmemory",
            user="nm", password="..."
        )
        await storage.initialize()
        storage.set_brain("my-brain")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "neuralmemory",
        user: str = "postgres",
        password: str = "",
        embedding_dim: int = 384,
    ) -> None:
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._embedding_dim = embedding_dim
        self._pool: Any = None
        self._current_brain_id: str | None = None

    async def initialize(self) -> None:
        """Create connection pool and schema."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._user,
            password=self._password or None,
            min_size=1,
            max_size=10,
            command_timeout=60,
        )
        await ensure_schema(self._pool, embedding_dim=self._embedding_dim)
        logger.info("PostgreSQL connected: %s:%d/%s", self._host, self._port, self._database)

    @property
    def brain_id(self) -> str | None:
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        self._current_brain_id = brain_id

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def disable_auto_save(self) -> None:
        pass

    def enable_auto_save(self) -> None:
        pass

    async def batch_save(self) -> None:
        pass
