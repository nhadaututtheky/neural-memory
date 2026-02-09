"""SQLite schema definition for neural memory storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 12

# â”€â”€ Migrations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each entry maps (from_version -> to_version) with a list of SQL statements.
# Migrations run sequentially in initialize() when db version < SCHEMA_VERSION.

# FTS5 setup statements â€” must be executed via individual execute() calls,
# NOT executescript(), because trigger bodies contain semicolons inside
# BEGIN...END blocks that executescript() would incorrectly split on.
FTS_SETUP_STATEMENTS: list[str] = [
    # FTS5 virtual table (external content -> neurons table).
    # Only index 'content' for searching; 'brain_id' is UNINDEXED (filter only).
    # We join on rowid to retrieve the full neuron row, so no neuron_id column needed.
    """CREATE VIRTUAL TABLE IF NOT EXISTS neurons_fts USING fts5(
        content,
        brain_id UNINDEXED,
        content='neurons',
        content_rowid='rowid',
        tokenize='porter unicode61 remove_diacritics 0'
    )""",
    # Auto-sync: insert
    """CREATE TRIGGER IF NOT EXISTS neurons_ai AFTER INSERT ON neurons BEGIN
        INSERT INTO neurons_fts(rowid, content, brain_id)
        VALUES (new.rowid, new.content, new.brain_id);
    END""",
    # Auto-sync: delete
    """CREATE TRIGGER IF NOT EXISTS neurons_ad AFTER DELETE ON neurons BEGIN
        INSERT INTO neurons_fts(neurons_fts, rowid, content, brain_id)
        VALUES ('delete', old.rowid, old.content, old.brain_id);
    END""",
    # Auto-sync: update
    """CREATE TRIGGER IF NOT EXISTS neurons_au AFTER UPDATE ON neurons BEGIN
        INSERT INTO neurons_fts(neurons_fts, rowid, content, brain_id)
        VALUES ('delete', old.rowid, old.content, old.brain_id);
        INSERT INTO neurons_fts(rowid, content, brain_id)
        VALUES (new.rowid, new.content, new.brain_id);
    END""",
]

MIGRATIONS: dict[tuple[int, int], list[str]] = {
    (1, 2): [
        "ALTER TABLE fibers ADD COLUMN pathway TEXT DEFAULT '[]'",
        "ALTER TABLE fibers ADD COLUMN conductivity REAL DEFAULT 1.0",
        "ALTER TABLE fibers ADD COLUMN last_conducted TEXT",
        "CREATE INDEX IF NOT EXISTS idx_fibers_conductivity ON fibers(brain_id, conductivity)",
    ],
    (2, 3): [
        # FTS table + triggers are created by ensure_fts_tables() in run_migrations.
        # Backfill FTS index from existing neurons.
        (
            "INSERT OR IGNORE INTO neurons_fts(rowid, content, brain_id) "
            "SELECT rowid, content, brain_id FROM neurons"
        ),
    ],
    (3, 4): [
        # Junction table for fast fiberâ†”neuron lookups (replaces LIKE on JSON)
        """CREATE TABLE IF NOT EXISTS fiber_neurons (
            brain_id TEXT NOT NULL,
            fiber_id TEXT NOT NULL,
            neuron_id TEXT NOT NULL,
            PRIMARY KEY (brain_id, fiber_id, neuron_id),
            FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE,
            FOREIGN KEY (brain_id, neuron_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_fiber_neurons_neuron ON fiber_neurons(brain_id, neuron_id)",
        # Backfill from existing fibers JSON arrays
        (
            "INSERT OR IGNORE INTO fiber_neurons (brain_id, fiber_id, neuron_id) "
            "SELECT f.brain_id, f.id, json_each.value "
            "FROM fibers f, json_each(f.neuron_ids)"
        ),
        # Index for hot neurons query in get_enhanced_stats
        "CREATE INDEX IF NOT EXISTS idx_neuron_states_freq ON neuron_states(brain_id, access_frequency DESC)",
    ],
    (4, 5): [
        # SimHash content fingerprint for near-duplicate detection
        "ALTER TABLE neurons ADD COLUMN content_hash INTEGER DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_neurons_hash ON neurons(brain_id, content_hash)",
    ],
    (5, 6): [
        # NeuronSpec v1: activation units with sigmoid, firing threshold, refractory period
        "ALTER TABLE neuron_states ADD COLUMN firing_threshold REAL DEFAULT 0.3",
        "ALTER TABLE neuron_states ADD COLUMN refractory_until TEXT",
        "ALTER TABLE neuron_states ADD COLUMN refractory_period_ms REAL DEFAULT 500.0",
        "ALTER TABLE neuron_states ADD COLUMN homeostatic_target REAL DEFAULT 0.5",
    ],
    (6, 7): [
        # Memory maturation lifecycle: STM -> Working -> Episodic -> Semantic
        """CREATE TABLE IF NOT EXISTS memory_maturations (
            fiber_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT 'stm',
            stage_entered_at TEXT NOT NULL,
            rehearsal_count INTEGER DEFAULT 0,
            reinforcement_timestamps TEXT DEFAULT '[]',
            PRIMARY KEY (brain_id, fiber_id),
            FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_maturations_stage ON memory_maturations(brain_id, stage)",
    ],
    (7, 8): [
        # Tag origin tracking: separate auto-generated tags from agent-provided tags
        "ALTER TABLE fibers ADD COLUMN auto_tags TEXT DEFAULT '[]'",
        "ALTER TABLE fibers ADD COLUMN agent_tags TEXT DEFAULT '[]'",
        # Backfill: existing tags -> agent_tags (conservative -- cannot determine origin retroactively)
        "UPDATE fibers SET agent_tags = tags WHERE tags != '[]'",
    ],
    (8, 9): [
        # Co-activation event persistence for associative inference
        """CREATE TABLE IF NOT EXISTS co_activation_events (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            neuron_a TEXT NOT NULL,
            neuron_b TEXT NOT NULL,
            binding_strength REAL NOT NULL,
            source_anchor TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_co_activation_pair ON co_activation_events(brain_id, neuron_a, neuron_b)",
        "CREATE INDEX IF NOT EXISTS idx_co_activation_created ON co_activation_events(brain_id, created_at)",
    ],
    (9, 10): [
        # Action event log â€” hippocampal buffer for habit learning
        """CREATE TABLE IF NOT EXISTS action_events (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            session_id TEXT,
            action_type TEXT NOT NULL,
            action_context TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            fiber_id TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_action_events_type ON action_events(brain_id, action_type)",
        "CREATE INDEX IF NOT EXISTS idx_action_events_session ON action_events(brain_id, session_id)",
        "CREATE INDEX IF NOT EXISTS idx_action_events_created ON action_events(brain_id, created_at)",
    ],
    (10, 11): [
        # Brain versioning â€” point-in-time snapshots
        """CREATE TABLE IF NOT EXISTS brain_versions (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            version_name TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            description TEXT DEFAULT '',
            neuron_count INTEGER DEFAULT 0,
            synapse_count INTEGER DEFAULT 0,
            fiber_count INTEGER DEFAULT 0,
            snapshot_hash TEXT NOT NULL,
            snapshot_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            PRIMARY KEY (brain_id, id),
            UNIQUE (brain_id, version_name)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_brain_versions_number ON brain_versions(brain_id, version_number DESC)",
        "CREATE INDEX IF NOT EXISTS idx_brain_versions_created ON brain_versions(brain_id, created_at DESC)",
    ],
    (11, 12): [
        # Composite index for synapse pair lookups (source_id + target_id)
        "CREATE INDEX IF NOT EXISTS idx_synapses_pair ON synapses(brain_id, source_id, target_id)",
        # Composite index for fiber tag searches
        "CREATE INDEX IF NOT EXISTS idx_fibers_tags ON fibers(brain_id, tags)",
    ],
}


async def ensure_fts_tables(conn: aiosqlite.Connection) -> None:
    """Create FTS5 virtual table and sync triggers if they don't exist.

    Uses individual execute() calls (not executescript) because trigger
    bodies contain semicolons inside BEGIN...END blocks.
    """
    for sql in FTS_SETUP_STATEMENTS:
        await conn.execute(sql)
    await conn.commit()


async def run_migrations(conn: aiosqlite.Connection, current_version: int) -> int:
    """Apply all pending migrations from current_version to SCHEMA_VERSION.

    Returns the final schema version after all migrations.
    """
    version = current_version

    while version < SCHEMA_VERSION:
        next_version = version + 1
        key = (version, next_version)

        # FTS tables must exist before the v2->v3 backfill INSERT runs
        if key == (2, 3):
            await ensure_fts_tables(conn)

        statements = MIGRATIONS.get(key, [])

        for sql in statements:
            try:
                await conn.execute(sql)
            except Exception as e:
                # Column/index may already exist (partial migration or manual fix).
                # ALTER TABLE ADD COLUMN raises OperationalError if column exists.
                import sqlite3

                if isinstance(e, sqlite3.OperationalError) and (
                    "duplicate column" in str(e).lower() or "already exists" in str(e).lower()
                ):
                    logger.debug("Migration already applied: %s", e)
                else:
                    logger.warning("Migration statement failed: %s â€” %s", sql[:80], e)

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
    content_hash INTEGER DEFAULT 0,  -- SimHash fingerprint for near-duplicate detection
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_neurons_type ON neurons(brain_id, type);
CREATE INDEX IF NOT EXISTS idx_neurons_created ON neurons(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_neurons_hash ON neurons(brain_id, content_hash);

-- Neuron states table
CREATE TABLE IF NOT EXISTS neuron_states (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    activation_level REAL DEFAULT 0.0,
    access_frequency INTEGER DEFAULT 0,
    last_activated TEXT,
    decay_rate REAL DEFAULT 0.1,
    firing_threshold REAL DEFAULT 0.3,
    refractory_until TEXT,
    refractory_period_ms REAL DEFAULT 500.0,
    homeostatic_target REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id, neuron_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_neuron_states_freq ON neuron_states(brain_id, access_frequency DESC);

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
CREATE INDEX IF NOT EXISTS idx_synapses_pair ON synapses(brain_id, source_id, target_id);

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
    tags TEXT DEFAULT '[]',  -- JSON array (union of auto_tags + agent_tags)
    auto_tags TEXT DEFAULT '[]',  -- JSON array: tags from auto-extraction
    agent_tags TEXT DEFAULT '[]',  -- JSON array: tags from calling agent
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fibers_created ON fibers(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fibers_salience ON fibers(brain_id, salience);
CREATE INDEX IF NOT EXISTS idx_fibers_conductivity ON fibers(brain_id, conductivity);

-- Fiber-neuron junction table (fast lookups)
CREATE TABLE IF NOT EXISTS fiber_neurons (
    brain_id TEXT NOT NULL,
    fiber_id TEXT NOT NULL,
    neuron_id TEXT NOT NULL,
    PRIMARY KEY (brain_id, fiber_id, neuron_id),
    FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, neuron_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fiber_neurons_neuron ON fiber_neurons(brain_id, neuron_id);

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
CREATE INDEX IF NOT EXISTS idx_typed_memories_expiry ON typed_memories(brain_id, expires_at);

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

-- Memory maturation lifecycle tracking
CREATE TABLE IF NOT EXISTS memory_maturations (
    fiber_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'stm',
    stage_entered_at TEXT NOT NULL,
    rehearsal_count INTEGER DEFAULT 0,
    reinforcement_timestamps TEXT DEFAULT '[]',  -- JSON array of ISO timestamps
    PRIMARY KEY (brain_id, fiber_id),
    FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_maturations_stage ON memory_maturations(brain_id, stage);

-- Co-activation events for associative inference
CREATE TABLE IF NOT EXISTS co_activation_events (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    neuron_a TEXT NOT NULL,  -- canonical: a < b
    neuron_b TEXT NOT NULL,
    binding_strength REAL NOT NULL,
    source_anchor TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_co_activation_pair ON co_activation_events(brain_id, neuron_a, neuron_b);
CREATE INDEX IF NOT EXISTS idx_co_activation_created ON co_activation_events(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_co_activation_time ON co_activation_events(brain_id, created_at, neuron_a, neuron_b);

-- Action event log for habit learning
CREATE TABLE IF NOT EXISTS action_events (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    session_id TEXT,
    action_type TEXT NOT NULL,
    action_context TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',  -- JSON array
    fiber_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_action_events_type ON action_events(brain_id, action_type);
CREATE INDEX IF NOT EXISTS idx_action_events_session ON action_events(brain_id, session_id);
CREATE INDEX IF NOT EXISTS idx_action_events_created ON action_events(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_action_events_sequence ON action_events(brain_id, session_id, created_at);

-- Brain versioning snapshots
CREATE TABLE IF NOT EXISTS brain_versions (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    version_name TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    description TEXT DEFAULT '',
    neuron_count INTEGER DEFAULT 0,
    synapse_count INTEGER DEFAULT 0,
    fiber_count INTEGER DEFAULT 0,
    snapshot_hash TEXT NOT NULL,
    snapshot_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (brain_id, id),
    UNIQUE (brain_id, version_name)
);
CREATE INDEX IF NOT EXISTS idx_brain_versions_number ON brain_versions(brain_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_brain_versions_created ON brain_versions(brain_id, created_at DESC);
"""
