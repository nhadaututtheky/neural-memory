"""SQLite schema definition for neural memory storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

# Schema version for migrations
SCHEMA_VERSION = 2

# ── Migrations ──────────────────────────────────────────────────────
# Each entry maps (from_version -> to_version) with a list of SQL statements.
# Migrations run sequentially in initialize() when db version < SCHEMA_VERSION.

MIGRATIONS: dict[tuple[int, int], list[str]] = {
    (1, 2): [
        "ALTER TABLE fibers ADD COLUMN pathway TEXT DEFAULT '[]'",
        "ALTER TABLE fibers ADD COLUMN conductivity REAL DEFAULT 1.0",
        "ALTER TABLE fibers ADD COLUMN last_conducted TEXT",
        "CREATE INDEX IF NOT EXISTS idx_fibers_conductivity ON fibers(brain_id, conductivity)",
    ],
}


async def run_migrations(conn: aiosqlite.Connection, current_version: int) -> int:
    """Apply all pending migrations from current_version to SCHEMA_VERSION.

    Returns the final schema version after all migrations.
    """
    version = current_version

    while version < SCHEMA_VERSION:
        next_version = version + 1
        key = (version, next_version)
        statements = MIGRATIONS.get(key, [])

        for sql in statements:
            try:
                await conn.execute(sql)
            except Exception:
                # Column/index may already exist (partial migration or manual fix).
                # ALTER TABLE ADD COLUMN raises OperationalError if column exists.
                pass

        version = next_version

    # Update stored version
    await conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    await conn.commit()

    return version


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
    pathway TEXT DEFAULT '[]',  -- JSON array: ordered neuron sequence
    conductivity REAL DEFAULT 1.0,  -- Signal transmission quality (0.0-1.0)
    last_conducted TEXT,  -- When fiber last conducted a signal
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
CREATE INDEX IF NOT EXISTS idx_fibers_conductivity ON fibers(brain_id, conductivity);

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
