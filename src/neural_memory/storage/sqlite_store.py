"""SQLite storage backend for persistent neural memory."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.base import NeuralStorage

# Schema version for migrations
SCHEMA_VERSION = 1

SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Brains table
CREATE TABLE IF NOT EXISTS brains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config TEXT NOT NULL,  -- JSON
    owner_id TEXT,
    is_public INTEGER DEFAULT 0,
    shared_with TEXT DEFAULT '[]',  -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Neurons table (composite key: brain_id + id for brain isolation)
CREATE TABLE IF NOT EXISTS neurons (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_neurons_type ON neurons(brain_id, type);
CREATE INDEX IF NOT EXISTS idx_neurons_created ON neurons(brain_id, created_at);

-- Neuron states table
CREATE TABLE IF NOT EXISTS neuron_states (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    activation_level REAL DEFAULT 0.0,
    access_frequency INTEGER DEFAULT 0,
    last_activated TEXT,
    decay_rate REAL DEFAULT 0.1,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id, neuron_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);

-- Synapses table
CREATE TABLE IF NOT EXISTS synapses (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL DEFAULT 0.5,
    direction TEXT DEFAULT 'uni',
    metadata TEXT DEFAULT '{}',  -- JSON
    reinforced_count INTEGER DEFAULT 0,
    last_activated TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, source_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, target_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(brain_id, source_id);
CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(brain_id, target_id);

-- Fibers table
CREATE TABLE IF NOT EXISTS fibers (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    neuron_ids TEXT NOT NULL,  -- JSON array
    synapse_ids TEXT NOT NULL,  -- JSON array
    anchor_neuron_id TEXT NOT NULL,
    time_start TEXT,
    time_end TEXT,
    coherence REAL DEFAULT 0.0,
    salience REAL DEFAULT 0.0,
    frequency INTEGER DEFAULT 0,
    summary TEXT,
    tags TEXT DEFAULT '[]',  -- JSON array
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fibers_created ON fibers(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fibers_salience ON fibers(brain_id, salience);

-- Typed memories table
CREATE TABLE IF NOT EXISTS typed_memories (
    fiber_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    provenance TEXT NOT NULL,  -- JSON
    expires_at TEXT,
    project_id TEXT,
    tags TEXT DEFAULT '[]',  -- JSON array
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, fiber_id),
    FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, project_id) REFERENCES projects(brain_id, id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_typed_memories_type ON typed_memories(brain_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_typed_memories_project ON typed_memories(brain_id, project_id);
CREATE INDEX IF NOT EXISTS idx_typed_memories_expires ON typed_memories(brain_id, expires_at);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    start_date TEXT NOT NULL,
    end_date TEXT,
    tags TEXT DEFAULT '[]',  -- JSON array
    priority REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(brain_id, name);
"""


class SQLiteStorage(NeuralStorage):
    """
    SQLite-based storage for persistent neural memory.

    Good for single-instance deployment and local development.
    Data persists to disk and survives restarts.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._current_brain_id: str | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._conn.execute("PRAGMA foreign_keys = ON")

        # Create schema
        await self._conn.executescript(SCHEMA)

        # Check/set schema version
        async with self._conn.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()
            if row is None:
                await self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
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

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO neurons (id, brain_id, type, content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    neuron.id,
                    brain_id,
                    neuron.type.value,
                    neuron.content,
                    json.dumps(neuron.metadata),
                    neuron.created_at.isoformat(),
                ),
            )

            # Initialize state
            await conn.execute(
                """INSERT INTO neuron_states (neuron_id, brain_id, created_at)
                   VALUES (?, ?, ?)""",
                (neuron.id, brain_id, datetime.utcnow().isoformat()),
            )

            await conn.commit()
            return neuron.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Neuron {neuron.id} already exists")

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neurons WHERE id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_neuron(row)

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[Neuron]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM neurons WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if type is not None:
            query += " AND type = ?"
            params.append(type.value)

        if content_contains is not None:
            query += " AND content LIKE ?"
            params.append(f"%{content_contains}%")

        if content_exact is not None:
            query += " AND content = ?"
            params.append(content_exact)

        if time_range is not None:
            start, end = time_range
            query += " AND created_at >= ? AND created_at <= ?"
            params.append(start.isoformat())
            params.append(end.isoformat())

        query += " LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_neuron(row) for row in rows]

    async def update_neuron(self, neuron: Neuron) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE neurons SET type = ?, content = ?, metadata = ?
               WHERE id = ? AND brain_id = ?""",
            (
                neuron.type.value,
                neuron.content,
                json.dumps(neuron.metadata),
                neuron.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Neuron {neuron.id} does not exist")

        await conn.commit()

    async def delete_neuron(self, neuron_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Delete neuron (cascade will handle synapses and state)
        cursor = await conn.execute(
            "DELETE FROM neurons WHERE id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    def _row_to_neuron(self, row: aiosqlite.Row) -> Neuron:
        """Convert database row to Neuron."""
        return Neuron(
            id=row["id"],
            type=NeuronType(row["type"]),
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neuron_states WHERE neuron_id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_neuron_state(row)

    async def update_neuron_state(self, state: NeuronState) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT OR REPLACE INTO neuron_states
               (neuron_id, brain_id, activation_level, access_frequency,
                last_activated, decay_rate, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                state.neuron_id,
                brain_id,
                state.activation_level,
                state.access_frequency,
                state.last_activated.isoformat() if state.last_activated else None,
                state.decay_rate,
                state.created_at.isoformat(),
            ),
        )
        await conn.commit()

    def _row_to_neuron_state(self, row: aiosqlite.Row) -> NeuronState:
        """Convert database row to NeuronState."""
        return NeuronState(
            neuron_id=row["neuron_id"],
            activation_level=row["activation_level"],
            access_frequency=row["access_frequency"],
            last_activated=(
                datetime.fromisoformat(row["last_activated"])
                if row["last_activated"]
                else None
            ),
            decay_rate=row["decay_rate"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Verify neurons exist
        async with conn.execute(
            "SELECT id FROM neurons WHERE id IN (?, ?) AND brain_id = ?",
            (synapse.source_id, synapse.target_id, brain_id),
        ) as cursor:
            rows = await cursor.fetchall()
            found_ids = {row["id"] for row in rows}

        if synapse.source_id not in found_ids:
            raise ValueError(f"Source neuron {synapse.source_id} does not exist")
        if synapse.target_id not in found_ids:
            raise ValueError(f"Target neuron {synapse.target_id} does not exist")

        try:
            await conn.execute(
                """INSERT INTO synapses
                   (id, brain_id, source_id, target_id, type, weight, direction,
                    metadata, reinforced_count, last_activated, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    synapse.id,
                    brain_id,
                    synapse.source_id,
                    synapse.target_id,
                    synapse.type.value,
                    synapse.weight,
                    synapse.direction.value,
                    json.dumps(synapse.metadata),
                    synapse.reinforced_count,
                    synapse.last_activated.isoformat() if synapse.last_activated else None,
                    synapse.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return synapse.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Synapse {synapse.id} already exists")

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM synapses WHERE id = ? AND brain_id = ?",
            (synapse_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_synapse(row)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM synapses WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if source_id is not None:
            query += " AND source_id = ?"
            params.append(source_id)

        if target_id is not None:
            query += " AND target_id = ?"
            params.append(target_id)

        if type is not None:
            query += " AND type = ?"
            params.append(type.value)

        if min_weight is not None:
            query += " AND weight >= ?"
            params.append(min_weight)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_synapse(row) for row in rows]

    async def update_synapse(self, synapse: Synapse) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE synapses SET type = ?, weight = ?, direction = ?,
               metadata = ?, reinforced_count = ?, last_activated = ?
               WHERE id = ? AND brain_id = ?""",
            (
                synapse.type.value,
                synapse.weight,
                synapse.direction.value,
                json.dumps(synapse.metadata),
                synapse.reinforced_count,
                synapse.last_activated.isoformat() if synapse.last_activated else None,
                synapse.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Synapse {synapse.id} does not exist")

        await conn.commit()

    async def delete_synapse(self, synapse_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM synapses WHERE id = ? AND brain_id = ?",
            (synapse_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    def _row_to_synapse(self, row: aiosqlite.Row) -> Synapse:
        """Convert database row to Synapse."""
        return Synapse(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=SynapseType(row["type"]),
            weight=row["weight"],
            direction=Direction(row["direction"]),
            metadata=json.loads(row["metadata"]),
            reinforced_count=row["reinforced_count"],
            last_activated=(
                datetime.fromisoformat(row["last_activated"])
                if row["last_activated"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        results: list[tuple[Neuron, Synapse]] = []

        # Build type filter
        type_filter = ""
        if synapse_types:
            types_str = ",".join(f"'{t.value}'" for t in synapse_types)
            type_filter = f" AND s.type IN ({types_str})"

        weight_filter = ""
        if min_weight is not None:
            weight_filter = f" AND s.weight >= {min_weight}"

        # Outgoing connections
        if direction in ("out", "both"):
            query = f"""
                SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type,
                       s.weight, s.direction, s.metadata as s_metadata,
                       s.reinforced_count, s.last_activated as s_last_activated,
                       s.created_at as s_created_at
                FROM synapses s
                JOIN neurons n ON s.target_id = n.id
                WHERE s.source_id = ? AND s.brain_id = ?{type_filter}{weight_filter}
            """
            async with conn.execute(query, (neuron_id, brain_id)) as cursor:
                async for row in cursor:
                    neuron = self._row_to_neuron(row)
                    synapse = Synapse(
                        id=row["s_id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        type=SynapseType(row["s_type"]),
                        weight=row["weight"],
                        direction=Direction(row["direction"]),
                        metadata=json.loads(row["s_metadata"]),
                        reinforced_count=row["reinforced_count"],
                        last_activated=(
                            datetime.fromisoformat(row["s_last_activated"])
                            if row["s_last_activated"]
                            else None
                        ),
                        created_at=datetime.fromisoformat(row["s_created_at"]),
                    )
                    results.append((neuron, synapse))

        # Incoming connections
        if direction in ("in", "both"):
            query = f"""
                SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type,
                       s.weight, s.direction, s.metadata as s_metadata,
                       s.reinforced_count, s.last_activated as s_last_activated,
                       s.created_at as s_created_at
                FROM synapses s
                JOIN neurons n ON s.source_id = n.id
                WHERE s.target_id = ? AND s.brain_id = ?{type_filter}{weight_filter}
            """
            async with conn.execute(query, (neuron_id, brain_id)) as cursor:
                async for row in cursor:
                    synapse = Synapse(
                        id=row["s_id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        type=SynapseType(row["s_type"]),
                        weight=row["weight"],
                        direction=Direction(row["direction"]),
                        metadata=json.loads(row["s_metadata"]),
                        reinforced_count=row["reinforced_count"],
                        last_activated=(
                            datetime.fromisoformat(row["s_last_activated"])
                            if row["s_last_activated"]
                            else None
                        ),
                        created_at=datetime.fromisoformat(row["s_created_at"]),
                    )

                    # For incoming, only include if bidirectional when direction is "in"
                    if direction == "in" and not synapse.is_bidirectional:
                        continue

                    neuron = self._row_to_neuron(row)
                    if (neuron, synapse) not in results:
                        results.append((neuron, synapse))

        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Find shortest path using BFS."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Verify both neurons exist
        async with conn.execute(
            "SELECT id FROM neurons WHERE id IN (?, ?) AND brain_id = ?",
            (source_id, target_id, brain_id),
        ) as cursor:
            rows = await cursor.fetchall()
            if len(rows) < 2:
                return None

        # BFS for shortest path
        from collections import deque

        visited = {source_id}
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque(
            [(source_id, [])]
        )  # (current_id, path of (neuron_id, synapse_id))

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_hops:
                continue

            # Get outgoing synapses
            async with conn.execute(
                """SELECT id, target_id FROM synapses
                   WHERE source_id = ? AND brain_id = ?""",
                (current_id, brain_id),
            ) as cursor:
                async for row in cursor:
                    next_id = row["target_id"]
                    synapse_id = row["id"]

                    if next_id == target_id:
                        # Found path
                        full_path = path + [(next_id, synapse_id)]
                        return await self._build_path_result(full_path)

                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, path + [(next_id, synapse_id)]))

        return None

    async def _build_path_result(
        self, path: list[tuple[str, str]]
    ) -> list[tuple[Neuron, Synapse]]:
        """Build path result from neuron/synapse IDs."""
        result: list[tuple[Neuron, Synapse]] = []
        for neuron_id, synapse_id in path:
            neuron = await self.get_neuron(neuron_id)
            synapse = await self.get_synapse(synapse_id)
            if neuron and synapse:
                result.append((neuron, synapse))
        return result

    # ========== Fiber Operations ==========

    async def add_fiber(self, fiber: Fiber) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO fibers
                   (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                    time_start, time_end, coherence, salience, frequency,
                    summary, tags, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fiber.id,
                    brain_id,
                    json.dumps(list(fiber.neuron_ids)),
                    json.dumps(list(fiber.synapse_ids)),
                    fiber.anchor_neuron_id,
                    fiber.time_start.isoformat() if fiber.time_start else None,
                    fiber.time_end.isoformat() if fiber.time_end else None,
                    fiber.coherence,
                    fiber.salience,
                    fiber.frequency,
                    fiber.summary,
                    json.dumps(list(fiber.tags)),
                    json.dumps(fiber.metadata),
                    fiber.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return fiber.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Fiber {fiber.id} already exists")

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM fibers WHERE id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_fiber(row)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM fibers WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if contains_neuron is not None:
            query += " AND neuron_ids LIKE ?"
            params.append(f'%"{contains_neuron}"%')

        if time_overlaps is not None:
            start, end = time_overlaps
            # Fiber overlaps if: fiber_start <= query_end AND fiber_end >= query_start
            query += " AND (time_start IS NULL OR time_start <= ?)"
            query += " AND (time_end IS NULL OR time_end >= ?)"
            params.append(end.isoformat())
            params.append(start.isoformat())

        if min_salience is not None:
            query += " AND salience >= ?"
            params.append(min_salience)

        query += " ORDER BY salience DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            fibers = [self._row_to_fiber(row) for row in rows]

        # Filter by tags in Python (JSON array doesn't support efficient set operations)
        if tags is not None:
            fibers = [f for f in fibers if tags.issubset(f.tags)]

        return fibers

    async def update_fiber(self, fiber: Fiber) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE fibers SET neuron_ids = ?, synapse_ids = ?,
               anchor_neuron_id = ?, time_start = ?, time_end = ?,
               coherence = ?, salience = ?, frequency = ?,
               summary = ?, tags = ?, metadata = ?
               WHERE id = ? AND brain_id = ?""",
            (
                json.dumps(list(fiber.neuron_ids)),
                json.dumps(list(fiber.synapse_ids)),
                fiber.anchor_neuron_id,
                fiber.time_start.isoformat() if fiber.time_start else None,
                fiber.time_end.isoformat() if fiber.time_end else None,
                fiber.coherence,
                fiber.salience,
                fiber.frequency,
                fiber.summary,
                json.dumps(list(fiber.tags)),
                json.dumps(fiber.metadata),
                fiber.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Fiber {fiber.id} does not exist")

        await conn.commit()

    async def delete_fiber(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM fibers WHERE id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        order_dir = "DESC" if descending else "ASC"
        query = f"SELECT * FROM fibers WHERE brain_id = ? ORDER BY {order_by} {order_dir} LIMIT ?"

        async with conn.execute(query, (brain_id, limit)) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_fiber(row) for row in rows]

    def _row_to_fiber(self, row: aiosqlite.Row) -> Fiber:
        """Convert database row to Fiber."""
        return Fiber(
            id=row["id"],
            neuron_ids=set(json.loads(row["neuron_ids"])),
            synapse_ids=set(json.loads(row["synapse_ids"])),
            anchor_neuron_id=row["anchor_neuron_id"],
            time_start=(
                datetime.fromisoformat(row["time_start"])
                if row["time_start"]
                else None
            ),
            time_end=(
                datetime.fromisoformat(row["time_end"]) if row["time_end"] else None
            ),
            coherence=row["coherence"],
            salience=row["salience"],
            frequency=row["frequency"],
            summary=row["summary"],
            tags=set(json.loads(row["tags"])),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== TypedMemory Operations ==========

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Verify fiber exists
        async with conn.execute(
            "SELECT id FROM fibers WHERE id = ? AND brain_id = ?",
            (typed_memory.fiber_id, brain_id),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError(f"Fiber {typed_memory.fiber_id} does not exist")

        provenance_dict = {
            "source": typed_memory.provenance.source,
            "confidence": typed_memory.provenance.confidence.value,
            "verified": typed_memory.provenance.verified,
            "verified_at": (
                typed_memory.provenance.verified_at.isoformat()
                if typed_memory.provenance.verified_at
                else None
            ),
            "created_by": typed_memory.provenance.created_by,
            "last_confirmed": (
                typed_memory.provenance.last_confirmed.isoformat()
                if typed_memory.provenance.last_confirmed
                else None
            ),
        }

        await conn.execute(
            """INSERT OR REPLACE INTO typed_memories
               (fiber_id, brain_id, memory_type, priority, provenance,
                expires_at, project_id, tags, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                typed_memory.fiber_id,
                brain_id,
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_dict),
                typed_memory.expires_at.isoformat() if typed_memory.expires_at else None,
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                typed_memory.created_at.isoformat(),
            ),
        )
        await conn.commit()
        return typed_memory.fiber_id

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM typed_memories WHERE fiber_id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_typed_memory(row)

    async def find_typed_memories(
        self,
        memory_type: MemoryType | None = None,
        min_priority: Priority | None = None,
        include_expired: bool = False,
        project_id: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[TypedMemory]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM typed_memories WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        if min_priority is not None:
            query += " AND priority >= ?"
            params.append(min_priority.value)

        if not include_expired:
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.utcnow().isoformat())

        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)

        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            memories = [self._row_to_typed_memory(row) for row in rows]

        # Filter by tags in Python
        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        provenance_dict = {
            "source": typed_memory.provenance.source,
            "confidence": typed_memory.provenance.confidence.value,
            "verified": typed_memory.provenance.verified,
            "verified_at": (
                typed_memory.provenance.verified_at.isoformat()
                if typed_memory.provenance.verified_at
                else None
            ),
            "created_by": typed_memory.provenance.created_by,
            "last_confirmed": (
                typed_memory.provenance.last_confirmed.isoformat()
                if typed_memory.provenance.last_confirmed
                else None
            ),
        }

        cursor = await conn.execute(
            """UPDATE typed_memories SET memory_type = ?, priority = ?,
               provenance = ?, expires_at = ?, project_id = ?,
               tags = ?, metadata = ?
               WHERE fiber_id = ? AND brain_id = ?""",
            (
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_dict),
                typed_memory.expires_at.isoformat() if typed_memory.expires_at else None,
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                typed_memory.fiber_id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"TypedMemory for fiber {typed_memory.fiber_id} does not exist"
            )

        await conn.commit()

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM typed_memories WHERE fiber_id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_expired_memories(self) -> list[TypedMemory]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT * FROM typed_memories
               WHERE brain_id = ? AND expires_at IS NOT NULL AND expires_at <= ?""",
            (brain_id, datetime.utcnow().isoformat()),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_typed_memory(row) for row in rows]

    def _row_to_typed_memory(self, row: aiosqlite.Row) -> TypedMemory:
        """Convert database row to TypedMemory."""
        prov_data = json.loads(row["provenance"])
        provenance = Provenance(
            source=prov_data.get("source", "unknown"),
            confidence=Confidence(prov_data.get("confidence", "medium")),
            verified=prov_data.get("verified", False),
            verified_at=(
                datetime.fromisoformat(prov_data["verified_at"])
                if prov_data.get("verified_at")
                else None
            ),
            created_by=prov_data.get("created_by", "unknown"),
            last_confirmed=(
                datetime.fromisoformat(prov_data["last_confirmed"])
                if prov_data.get("last_confirmed")
                else None
            ),
        )

        return TypedMemory(
            fiber_id=row["fiber_id"],
            memory_type=MemoryType(row["memory_type"]),
            priority=Priority(row["priority"]),
            provenance=provenance,
            expires_at=(
                datetime.fromisoformat(row["expires_at"])
                if row["expires_at"]
                else None
            ),
            project_id=row["project_id"],
            tags=frozenset(json.loads(row["tags"])),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Project Operations ==========

    async def add_project(self, project: Project) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO projects
                   (id, brain_id, name, description, start_date, end_date,
                    tags, priority, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project.id,
                    brain_id,
                    project.name,
                    project.description,
                    project.start_date.isoformat(),
                    project.end_date.isoformat() if project.end_date else None,
                    json.dumps(list(project.tags)),
                    project.priority,
                    json.dumps(project.metadata),
                    project.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return project.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Project {project.id} already exists")

    async def get_project(self, project_id: str) -> Project | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM projects WHERE id = ? AND brain_id = ?",
            (project_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_project(row)

    async def get_project_by_name(self, name: str) -> Project | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM projects WHERE brain_id = ? AND LOWER(name) = LOWER(?)",
            (brain_id, name),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_project(row)

    async def list_projects(
        self,
        active_only: bool = False,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[Project]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM projects WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if active_only:
            now = datetime.utcnow().isoformat()
            query += " AND start_date <= ? AND (end_date IS NULL OR end_date > ?)"
            params.extend([now, now])

        query += " ORDER BY priority DESC, start_date DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            projects = [self._row_to_project(row) for row in rows]

        # Filter by tags in Python
        if tags is not None:
            projects = [p for p in projects if tags.intersection(p.tags)]

        return projects

    async def update_project(self, project: Project) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE projects SET name = ?, description = ?,
               start_date = ?, end_date = ?, tags = ?,
               priority = ?, metadata = ?
               WHERE id = ? AND brain_id = ?""",
            (
                project.name,
                project.description,
                project.start_date.isoformat(),
                project.end_date.isoformat() if project.end_date else None,
                json.dumps(list(project.tags)),
                project.priority,
                json.dumps(project.metadata),
                project.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Project {project.id} does not exist")

        await conn.commit()

    async def delete_project(self, project_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM projects WHERE id = ? AND brain_id = ?",
            (project_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_project_memories(
        self,
        project_id: str,
        include_expired: bool = False,
    ) -> list[TypedMemory]:
        return await self.find_typed_memories(
            project_id=project_id,
            include_expired=include_expired,
        )

    def _row_to_project(self, row: aiosqlite.Row) -> Project:
        """Convert database row to Project."""
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            start_date=datetime.fromisoformat(row["start_date"]),
            end_date=(
                datetime.fromisoformat(row["end_date"]) if row["end_date"] else None
            ),
            tags=frozenset(json.loads(row["tags"])),
            priority=row["priority"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Brain Operations ==========

    async def save_brain(self, brain: Brain) -> None:
        conn = self._ensure_conn()

        await conn.execute(
            """INSERT OR REPLACE INTO brains
               (id, name, config, owner_id, is_public, shared_with, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                brain.id,
                brain.name,
                json.dumps({
                    "decay_rate": brain.config.decay_rate,
                    "reinforcement_delta": brain.config.reinforcement_delta,
                    "activation_threshold": brain.config.activation_threshold,
                    "max_spread_hops": brain.config.max_spread_hops,
                    "max_context_tokens": brain.config.max_context_tokens,
                }),
                brain.owner_id,
                1 if brain.is_public else 0,
                json.dumps(brain.shared_with),
                brain.created_at.isoformat(),
                brain.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_brain(self, brain_id: str) -> Brain | None:
        conn = self._ensure_conn()

        async with conn.execute(
            "SELECT * FROM brains WHERE id = ?", (brain_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_brain(row)

    def _row_to_brain(self, row: aiosqlite.Row) -> Brain:
        """Convert database row to Brain."""
        config_data = json.loads(row["config"])
        config = BrainConfig(
            decay_rate=config_data.get("decay_rate", 0.1),
            reinforcement_delta=config_data.get("reinforcement_delta", 0.05),
            activation_threshold=config_data.get("activation_threshold", 0.2),
            max_spread_hops=config_data.get("max_spread_hops", 4),
            max_context_tokens=config_data.get("max_context_tokens", 1500),
        )

        return Brain(
            id=row["id"],
            name=row["name"],
            config=config,
            owner_id=row["owner_id"],
            is_public=bool(row["is_public"]),
            shared_with=json.loads(row["shared_with"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        conn = self._ensure_conn()

        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        # Export neurons
        neurons = []
        async with conn.execute(
            "SELECT * FROM neurons WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            async for row in cursor:
                neurons.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                })

        # Export synapses
        synapses = []
        async with conn.execute(
            "SELECT * FROM synapses WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            async for row in cursor:
                synapses.append({
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": row["type"],
                    "weight": row["weight"],
                    "direction": row["direction"],
                    "metadata": json.loads(row["metadata"]),
                    "reinforced_count": row["reinforced_count"],
                    "created_at": row["created_at"],
                })

        # Export fibers
        fibers = []
        async with conn.execute(
            "SELECT * FROM fibers WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            async for row in cursor:
                fibers.append({
                    "id": row["id"],
                    "neuron_ids": json.loads(row["neuron_ids"]),
                    "synapse_ids": json.loads(row["synapse_ids"]),
                    "anchor_neuron_id": row["anchor_neuron_id"],
                    "time_start": row["time_start"],
                    "time_end": row["time_end"],
                    "coherence": row["coherence"],
                    "salience": row["salience"],
                    "frequency": row["frequency"],
                    "summary": row["summary"],
                    "tags": json.loads(row["tags"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                })

        # Export typed memories
        typed_memories = []
        async with conn.execute(
            "SELECT * FROM typed_memories WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            async for row in cursor:
                typed_memories.append({
                    "fiber_id": row["fiber_id"],
                    "memory_type": row["memory_type"],
                    "priority": row["priority"],
                    "provenance": json.loads(row["provenance"]),
                    "expires_at": row["expires_at"],
                    "project_id": row["project_id"],
                    "tags": json.loads(row["tags"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                })

        # Export projects
        projects = []
        async with conn.execute(
            "SELECT * FROM projects WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            async for row in cursor:
                projects.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "tags": json.loads(row["tags"]),
                    "priority": row["priority"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                })

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=datetime.utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config={
                "decay_rate": brain.config.decay_rate,
                "reinforcement_delta": brain.config.reinforcement_delta,
                "activation_threshold": brain.config.activation_threshold,
                "max_spread_hops": brain.config.max_spread_hops,
                "max_context_tokens": brain.config.max_context_tokens,
            },
            metadata={
                "typed_memories": typed_memories,
                "projects": projects,
            },
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        brain_id = target_brain_id or snapshot.brain_id

        # Create brain
        config = BrainConfig(**snapshot.config)
        brain = Brain.create(
            name=snapshot.brain_name,
            config=config,
            brain_id=brain_id,
        )
        await self.save_brain(brain)

        # Set context
        old_brain_id = self._current_brain_id
        self.set_brain(brain_id)

        try:
            # Import neurons
            for n_data in snapshot.neurons:
                neuron = Neuron(
                    id=n_data["id"],
                    type=NeuronType(n_data["type"]),
                    content=n_data["content"],
                    metadata=n_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(n_data["created_at"]),
                )
                await self.add_neuron(neuron)

            # Import synapses
            for s_data in snapshot.synapses:
                synapse = Synapse(
                    id=s_data["id"],
                    source_id=s_data["source_id"],
                    target_id=s_data["target_id"],
                    type=SynapseType(s_data["type"]),
                    weight=s_data["weight"],
                    direction=Direction(s_data["direction"]),
                    metadata=s_data.get("metadata", {}),
                    reinforced_count=s_data.get("reinforced_count", 0),
                    created_at=datetime.fromisoformat(s_data["created_at"]),
                )
                await self.add_synapse(synapse)

            # Import fibers
            for f_data in snapshot.fibers:
                fiber = Fiber(
                    id=f_data["id"],
                    neuron_ids=set(f_data["neuron_ids"]),
                    synapse_ids=set(f_data["synapse_ids"]),
                    anchor_neuron_id=f_data["anchor_neuron_id"],
                    time_start=(
                        datetime.fromisoformat(f_data["time_start"])
                        if f_data.get("time_start")
                        else None
                    ),
                    time_end=(
                        datetime.fromisoformat(f_data["time_end"])
                        if f_data.get("time_end")
                        else None
                    ),
                    coherence=f_data.get("coherence", 0.0),
                    salience=f_data.get("salience", 0.0),
                    frequency=f_data.get("frequency", 0),
                    summary=f_data.get("summary"),
                    tags=set(f_data.get("tags", [])),
                    metadata=f_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(f_data["created_at"]),
                )
                await self.add_fiber(fiber)

            # Import projects first (typed_memories may reference them)
            projects_data = snapshot.metadata.get("projects", [])
            for p_data in projects_data:
                project = Project(
                    id=p_data["id"],
                    name=p_data["name"],
                    description=p_data.get("description", ""),
                    start_date=datetime.fromisoformat(p_data["start_date"]),
                    end_date=(
                        datetime.fromisoformat(p_data["end_date"])
                        if p_data.get("end_date")
                        else None
                    ),
                    tags=frozenset(p_data.get("tags", [])),
                    priority=p_data.get("priority", 1.0),
                    metadata=p_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(p_data["created_at"]),
                )
                await self.add_project(project)

            # Import typed memories
            typed_memories_data = snapshot.metadata.get("typed_memories", [])
            for tm_data in typed_memories_data:
                prov_data = tm_data.get("provenance", {})
                provenance = Provenance(
                    source=prov_data.get("source", "import"),
                    confidence=Confidence(prov_data.get("confidence", "medium")),
                    verified=prov_data.get("verified", False),
                    verified_at=(
                        datetime.fromisoformat(prov_data["verified_at"])
                        if prov_data.get("verified_at")
                        else None
                    ),
                    created_by=prov_data.get("created_by", "import"),
                    last_confirmed=(
                        datetime.fromisoformat(prov_data["last_confirmed"])
                        if prov_data.get("last_confirmed")
                        else None
                    ),
                )

                typed_memory = TypedMemory(
                    fiber_id=tm_data["fiber_id"],
                    memory_type=MemoryType(tm_data["memory_type"]),
                    priority=Priority(tm_data["priority"]),
                    provenance=provenance,
                    expires_at=(
                        datetime.fromisoformat(tm_data["expires_at"])
                        if tm_data.get("expires_at")
                        else None
                    ),
                    project_id=tm_data.get("project_id"),
                    tags=frozenset(tm_data.get("tags", [])),
                    metadata=tm_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(tm_data["created_at"]),
                )
                await self.add_typed_memory(typed_memory)

        finally:
            self._current_brain_id = old_brain_id

        return brain_id

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        conn = self._ensure_conn()

        stats = {}

        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM neurons WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["neuron_count"] = row["cnt"] if row else 0

        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM synapses WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["synapse_count"] = row["cnt"] if row else 0

        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM fibers WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["fiber_count"] = row["cnt"] if row else 0

        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM projects WHERE brain_id = ?", (brain_id,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["project_count"] = row["cnt"] if row else 0

        return stats

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        conn = self._ensure_conn()

        # Delete in order to respect foreign keys
        await conn.execute(
            "DELETE FROM typed_memories WHERE brain_id = ?", (brain_id,)
        )
        await conn.execute("DELETE FROM projects WHERE brain_id = ?", (brain_id,))
        await conn.execute("DELETE FROM fibers WHERE brain_id = ?", (brain_id,))
        await conn.execute("DELETE FROM synapses WHERE brain_id = ?", (brain_id,))
        await conn.execute(
            "DELETE FROM neuron_states WHERE brain_id = ?", (brain_id,)
        )
        await conn.execute("DELETE FROM neurons WHERE brain_id = ?", (brain_id,))
        await conn.execute("DELETE FROM brains WHERE id = ?", (brain_id,))

        await conn.commit()
