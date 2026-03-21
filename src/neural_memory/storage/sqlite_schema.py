"""SQLite schema definition for neural memory storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 35

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
        tokenize='porter unicode61 remove_diacritics 2'
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

FIBER_FTS_SETUP_STATEMENTS: list[str] = [
    # FTS5 virtual table for fiber summaries (external content -> fibers table).
    """CREATE VIRTUAL TABLE IF NOT EXISTS fibers_fts USING fts5(
        summary,
        brain_id UNINDEXED,
        content='fibers',
        content_rowid='rowid',
        tokenize='porter unicode61 remove_diacritics 2'
    )""",
    # Auto-sync: insert
    """CREATE TRIGGER IF NOT EXISTS fibers_ai AFTER INSERT ON fibers
    WHEN new.summary IS NOT NULL AND new.summary != '' BEGIN
        INSERT INTO fibers_fts(rowid, summary, brain_id)
        VALUES (new.rowid, new.summary, new.brain_id);
    END""",
    # Auto-sync: delete
    """CREATE TRIGGER IF NOT EXISTS fibers_ad AFTER DELETE ON fibers
    WHEN old.summary IS NOT NULL AND old.summary != '' BEGIN
        INSERT INTO fibers_fts(fibers_fts, rowid, summary, brain_id)
        VALUES ('delete', old.rowid, old.summary, old.brain_id);
    END""",
    # Auto-sync: update
    """CREATE TRIGGER IF NOT EXISTS fibers_au AFTER UPDATE ON fibers
    WHEN old.summary IS NOT NULL AND old.summary != '' BEGIN
        INSERT INTO fibers_fts(fibers_fts, rowid, summary, brain_id)
        VALUES ('delete', old.rowid, old.summary, old.brain_id);
        INSERT INTO fibers_fts(rowid, summary, brain_id)
        VALUES (new.rowid, COALESCE(new.summary, ''), new.brain_id);
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
    (12, 13): [
        # Sync state persistence for external source auto-sync (e.g. Mem0)
        """CREATE TABLE IF NOT EXISTS sync_states (
            source_system TEXT NOT NULL,
            source_collection TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            last_sync_at TEXT,
            records_imported INTEGER DEFAULT 0,
            last_record_id TEXT,
            metadata TEXT DEFAULT '{}',
            PRIMARY KEY (brain_id, source_system, source_collection)
        )""",
    ],
    (13, 14): [
        # Proactive alerts queue for brain health monitoring
        """CREATE TABLE IF NOT EXISTS alerts (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'low',
            message TEXT NOT NULL,
            recommended_action TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            seen_at TEXT,
            acknowledged_at TEXT,
            resolved_at TEXT,
            metadata TEXT DEFAULT '{}',
            PRIMARY KEY (brain_id, id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(brain_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(brain_id, alert_type, status)",
    ],
    (14, 15): [
        # Spaced repetition review schedules (Leitner box system)
        """CREATE TABLE IF NOT EXISTS review_schedules (
            fiber_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            box INTEGER NOT NULL DEFAULT 1,
            next_review TEXT,
            last_reviewed TEXT,
            review_count INTEGER DEFAULT 0,
            streak INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (fiber_id, brain_id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_review_next ON review_schedules(brain_id, next_review)",
        "CREATE INDEX IF NOT EXISTS idx_review_box ON review_schedules(brain_id, box)",
    ],
    (15, 16): [
        # Bayesian depth priors for adaptive recall
        """CREATE TABLE IF NOT EXISTS depth_priors (
            brain_id TEXT NOT NULL,
            entity_text TEXT NOT NULL,
            depth_level INTEGER NOT NULL,
            alpha REAL NOT NULL DEFAULT 1.0,
            beta REAL NOT NULL DEFAULT 1.0,
            total_queries INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, entity_text, depth_level),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_depth_priors_entity ON depth_priors(brain_id, entity_text)",
        # Compression backups for reversible compression (tiers 1-2)
        """CREATE TABLE IF NOT EXISTS compression_backups (
            fiber_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            original_content TEXT NOT NULL,
            compression_tier INTEGER NOT NULL,
            compressed_at TEXT NOT NULL,
            original_token_count INTEGER DEFAULT 0,
            compressed_token_count INTEGER DEFAULT 0,
            PRIMARY KEY (brain_id, fiber_id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_compression_tier ON compression_backups(brain_id, compression_tier)",
        # Fiber compression tier tracking
        "ALTER TABLE fibers ADD COLUMN compression_tier INTEGER DEFAULT 0",
    ],
    (16, 17): [
        # Multi-device sync: device tracking columns on core tables
        "ALTER TABLE neurons ADD COLUMN device_id TEXT DEFAULT ''",
        "ALTER TABLE neurons ADD COLUMN device_origin TEXT DEFAULT ''",
        "ALTER TABLE neurons ADD COLUMN updated_at TEXT DEFAULT ''",
        "ALTER TABLE synapses ADD COLUMN device_id TEXT DEFAULT ''",
        "ALTER TABLE synapses ADD COLUMN device_origin TEXT DEFAULT ''",
        "ALTER TABLE synapses ADD COLUMN updated_at TEXT DEFAULT ''",
        "ALTER TABLE fibers ADD COLUMN device_id TEXT DEFAULT ''",
        "ALTER TABLE fibers ADD COLUMN device_origin TEXT DEFAULT ''",
        "ALTER TABLE fibers ADD COLUMN updated_at TEXT DEFAULT ''",
        # Change log (append-only journal for incremental sync)
        """CREATE TABLE IF NOT EXISTS change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brain_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            device_id TEXT NOT NULL DEFAULT '',
            changed_at TEXT NOT NULL,
            payload TEXT DEFAULT '{}',
            synced INTEGER DEFAULT 0
        )""",
        "CREATE INDEX IF NOT EXISTS idx_change_log_brain_synced ON change_log(brain_id, synced, changed_at)",
        # Device registry for multi-device sync
        """CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            device_name TEXT DEFAULT '',
            last_sync_at TEXT,
            last_sync_sequence INTEGER DEFAULT 0,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, device_id)
        )""",
        # Backfill updated_at from created_at for existing rows
        "UPDATE neurons SET updated_at = created_at WHERE updated_at = '' OR updated_at IS NULL",
        "UPDATE synapses SET updated_at = created_at WHERE updated_at = '' OR updated_at IS NULL",
        "UPDATE fibers SET updated_at = created_at WHERE updated_at = '' OR updated_at IS NULL",
        # Indexes for incremental sync queries
        "CREATE INDEX IF NOT EXISTS idx_neurons_updated ON neurons(brain_id, updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_synapses_updated ON synapses(brain_id, updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_fibers_updated ON fibers(brain_id, updated_at)",
    ],
    (17, 18): [
        """CREATE TABLE IF NOT EXISTS retrieval_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brain_id TEXT NOT NULL,
            gate TEXT NOT NULL,
            predicted_sufficient INTEGER NOT NULL,
            actual_confidence REAL NOT NULL DEFAULT 0.0,
            actual_fibers INTEGER NOT NULL DEFAULT 0,
            query_intent TEXT NOT NULL DEFAULT '',
            metrics_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_calibration_brain_gate ON retrieval_calibration(brain_id, gate)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_created ON retrieval_calibration(brain_id, created_at)",
    ],
    (18, 19): [
        """CREATE TABLE IF NOT EXISTS tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brain_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            server_name TEXT NOT NULL DEFAULT '',
            args_summary TEXT NOT NULL DEFAULT '',
            success INTEGER NOT NULL DEFAULT 1,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            session_id TEXT NOT NULL DEFAULT '',
            task_context TEXT NOT NULL DEFAULT '',
            processed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_tool_events_tool ON tool_events(brain_id, tool_name)",
        "CREATE INDEX IF NOT EXISTS idx_tool_events_processed ON tool_events(brain_id, processed)",
        "CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(brain_id, session_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_tool_events_created ON tool_events(brain_id, created_at)",
    ],
    (19, 20): [
        # Pinned flag for KB (knowledge base) memories — skip decay/prune/compress
        "ALTER TABLE fibers ADD COLUMN pinned INTEGER DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_fibers_pinned ON fibers(brain_id, pinned)",
        # Training file tracking for dedup and resume
        """CREATE TABLE IF NOT EXISTS training_files (
            id TEXT PRIMARY KEY,
            brain_id TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            chunks_total INTEGER NOT NULL DEFAULT 0,
            chunks_completed INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            domain_tag TEXT DEFAULT '',
            trained_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_training_files_hash ON training_files(brain_id, file_hash)",
    ],
    (20, 21): [
        # Cognitive layer: hypothesis/prediction confidence tracking
        """CREATE TABLE IF NOT EXISTS cognitive_state (
            neuron_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_for_count INTEGER NOT NULL DEFAULT 0,
            evidence_against_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'confirmed', 'refuted', 'superseded', 'pending', 'expired')),
            predicted_at TEXT,
            resolved_at TEXT,
            schema_version INTEGER DEFAULT 1,
            parent_schema_id TEXT,
            last_evidence_at TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, neuron_id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_confidence ON cognitive_state(brain_id, confidence DESC)",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_status ON cognitive_state(brain_id, status)",
        # Cognitive layer: pre-computed hot index (max 20 entries per brain)
        """CREATE TABLE IF NOT EXISTS hot_index (
            brain_id TEXT NOT NULL,
            slot INTEGER NOT NULL,
            category TEXT NOT NULL,
            neuron_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            confidence REAL,
            score REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, slot),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        # Cognitive layer: metacognition — knowledge gaps
        """CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id TEXT PRIMARY KEY,
            brain_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            detection_source TEXT NOT NULL,
            related_neuron_ids TEXT DEFAULT '[]',
            resolved_at TEXT,
            resolved_by_neuron_id TEXT,
            priority REAL DEFAULT 0.5,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_gaps_brain ON knowledge_gaps(brain_id, resolved_at)",
        "CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(brain_id, priority DESC)",
    ],
    (21, 22): [
        # Trust score: queryable trust level + source on typed memories
        "ALTER TABLE typed_memories ADD COLUMN trust_score REAL DEFAULT NULL",
        "ALTER TABLE typed_memories ADD COLUMN source TEXT DEFAULT NULL",
        "CREATE INDEX IF NOT EXISTS idx_typed_memories_trust ON typed_memories(brain_id, trust_score)",
    ],
    (22, 23): [
        # Source registry: first-class provenance tracking
        """CREATE TABLE IF NOT EXISTS sources (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'document',
            version TEXT DEFAULT '',
            effective_date TEXT,
            expires_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            file_hash TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(brain_id, source_type)",
        "CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(brain_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(brain_id, name)",
    ],
    (23, 24): [
        # Session summaries: persist session intelligence for drift detection
        """CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            topics_json TEXT NOT NULL DEFAULT '[]',
            topic_weights_json TEXT NOT NULL DEFAULT '{}',
            top_entities_json TEXT NOT NULL DEFAULT '[]',
            query_count INTEGER NOT NULL DEFAULT 0,
            avg_confidence REAL NOT NULL DEFAULT 0.0,
            avg_depth REAL NOT NULL DEFAULT 0.0,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_session_summaries_brain ON session_summaries(brain_id, ended_at)",
        "CREATE INDEX IF NOT EXISTS idx_session_summaries_session ON session_summaries(session_id)",
    ],
    (24, 25): [
        # Retriever calibration: per-brain EMA weights for dynamic RRF
        """CREATE TABLE IF NOT EXISTS retriever_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brain_id TEXT NOT NULL,
            retriever_type TEXT NOT NULL,
            contributed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_retriever_cal_brain ON retriever_calibration(brain_id, retriever_type, created_at)",
        # Graph density metric in brain metadata
        """ALTER TABLE brains ADD COLUMN graph_density REAL NOT NULL DEFAULT 0.0""",
    ],
    (25, 26): [
        # Tag co-occurrence matrix for semantic drift detection
        """CREATE TABLE IF NOT EXISTS tag_cooccurrence (
            brain_id TEXT NOT NULL,
            tag_a TEXT NOT NULL,
            tag_b TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            last_seen TEXT NOT NULL,
            PRIMARY KEY (brain_id, tag_a, tag_b),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_tag_cooccurrence_brain ON tag_cooccurrence(brain_id, count DESC)",
        # Drift detection results (persisted for review/dismiss)
        """CREATE TABLE IF NOT EXISTS drift_clusters (
            id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            canonical TEXT NOT NULL,
            members TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'detected',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            PRIMARY KEY (brain_id, id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_drift_clusters_status ON drift_clusters(brain_id, status)",
    ],
    (26, 27): [
        # Fiber summary FTS5 is created by ensure_fiber_fts_tables() in run_migrations.
        # Backfill FTS index from existing fiber summaries.
        (
            "INSERT OR IGNORE INTO fibers_fts(rowid, summary, brain_id) "
            "SELECT rowid, summary, brain_id FROM fibers WHERE summary IS NOT NULL AND summary != ''"
        ),
    ],
    (27, 28): [
        # Keyword document frequency for IDF-weighted synapse creation
        """CREATE TABLE IF NOT EXISTS keyword_document_frequency (
            brain_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            fiber_count INTEGER NOT NULL DEFAULT 1,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (brain_id, keyword),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_keyword_df_brain ON keyword_document_frequency(brain_id)",
    ],
    (28, 29): [
        # Entity reference tracking for lazy entity promotion (B7)
        # Entities need 2+ mentions before being promoted to full neurons
        """CREATE TABLE IF NOT EXISTS entity_refs (
            brain_id TEXT NOT NULL,
            entity_text TEXT NOT NULL,
            fiber_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (brain_id, entity_text, fiber_id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_entity_refs_text ON entity_refs(brain_id, entity_text)",
    ],
    (29, 30): [
        # Performance: index on neurons(brain_id, content) for exact-match lookups
        "CREATE INDEX IF NOT EXISTS idx_neurons_content ON neurons(brain_id, content)",
    ],
    (30, 31): [
        # FTS5: enable diacritics removal for Vietnamese/multilingual recall.
        # remove_diacritics=2 strips ALL diacritics (Vietnamese, Chinese pinyin, etc.)
        # so "Tài khoản" matches queries "tai khoan" or "Tài khoản".
        #
        # Rebuild neurons_fts: drop triggers → drop table → recreate → reindex
        "DROP TRIGGER IF EXISTS neurons_au",
        "DROP TRIGGER IF EXISTS neurons_ad",
        "DROP TRIGGER IF EXISTS neurons_ai",
        "DROP TABLE IF EXISTS neurons_fts",
        # Rebuild fibers_fts
        "DROP TRIGGER IF EXISTS fibers_fts_au",
        "DROP TRIGGER IF EXISTS fibers_fts_ad",
        "DROP TRIGGER IF EXISTS fibers_fts_ai",
        "DROP TABLE IF EXISTS fibers_fts",
    ],
    (31, 32): [
        # Memory Lifecycle Engine: access tracking, lifecycle states, freeze, and snapshots.
        "ALTER TABLE neurons ADD COLUMN last_accessed_at TEXT",
        "ALTER TABLE neurons ADD COLUMN lifecycle_state TEXT DEFAULT 'active'",
        "ALTER TABLE neurons ADD COLUMN frozen INTEGER DEFAULT 0",
        # neuron_snapshots: stores original content before destructive Tier 3/4 compression
        """CREATE TABLE IF NOT EXISTS neuron_snapshots (
            neuron_id TEXT NOT NULL,
            brain_id TEXT NOT NULL,
            original_content TEXT NOT NULL,
            compressed_at TEXT NOT NULL,
            tier INTEGER NOT NULL,
            PRIMARY KEY (brain_id, neuron_id),
            FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
        )""",
        # Indexes for lifecycle queries
        "CREATE INDEX IF NOT EXISTS idx_neurons_lifecycle ON neurons(brain_id, lifecycle_state)",
        "CREATE INDEX IF NOT EXISTS idx_neurons_last_accessed ON neurons(brain_id, last_accessed_at)",
    ],
    (32, 33): [
        # Ephemeral memories: session-scoped neurons that auto-expire and never sync.
        "ALTER TABLE neurons ADD COLUMN ephemeral INTEGER DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_neurons_ephemeral ON neurons(brain_id, ephemeral)",
    ],
    (33, 34): [
        # Fidelity layers Phase 1: essence field on fibers for single-sentence distillation
        "ALTER TABLE fibers ADD COLUMN essence TEXT",
    ],
    (34, 35): [
        # Fidelity layers Phase 3: ghost recall tracking
        "ALTER TABLE fibers ADD COLUMN last_ghost_shown_at TEXT",
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


async def ensure_fiber_fts_tables(conn: aiosqlite.Connection) -> None:
    """Create FTS5 virtual table and sync triggers for fiber summaries."""
    for sql in FIBER_FTS_SETUP_STATEMENTS:
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

        # Fiber FTS tables must exist before the v26->v27 backfill INSERT runs
        if key == (26, 27):
            await ensure_fiber_fts_tables(conn)

        # v30->v31 drops FTS tables; recreate with new tokenizer + reindex
        if key == (30, 31):
            # Statements drop the tables first, then we recreate + backfill
            statements_31 = MIGRATIONS.get(key, [])
            for sql in statements_31:
                try:
                    await conn.execute(sql)
                except Exception:
                    pass  # triggers/tables may not exist
            await ensure_fts_tables(conn)
            await ensure_fiber_fts_tables(conn)
            # Backfill neurons_fts from existing data
            await conn.execute(
                "INSERT OR IGNORE INTO neurons_fts(rowid, content, brain_id) "
                "SELECT rowid, content, brain_id FROM neurons"
            )
            # Backfill fibers_fts from existing data
            await conn.execute(
                "INSERT OR IGNORE INTO fibers_fts(rowid, summary, brain_id) "
                "SELECT rowid, summary, brain_id FROM fibers WHERE summary IS NOT NULL"
            )
            await conn.commit()
            version = next_version
            continue

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
                    logger.error(
                        "Migration %d->%d failed: %s -- %s",
                        version,
                        next_version,
                        sql[:80],
                        e,
                    )
                    raise

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
    graph_density REAL NOT NULL DEFAULT 0.0,
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
    device_id TEXT DEFAULT '',  -- Device that last modified this neuron
    device_origin TEXT DEFAULT '',  -- Device that originally created this neuron
    updated_at TEXT DEFAULT '',  -- Last modification timestamp
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,  -- Updated on every recall hit (batch)
    lifecycle_state TEXT DEFAULT 'active',  -- ACTIVE | WARM | COOL | COMPRESSED | ARCHIVED
    frozen INTEGER DEFAULT 0,  -- 1 = never compress
    ephemeral INTEGER DEFAULT 0,  -- 1 = session-scoped, auto-expires, never synced
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_neurons_type ON neurons(brain_id, type);
CREATE INDEX IF NOT EXISTS idx_neurons_created ON neurons(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_neurons_hash ON neurons(brain_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_neurons_updated ON neurons(brain_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_neurons_content ON neurons(brain_id, content);
CREATE INDEX IF NOT EXISTS idx_neurons_lifecycle ON neurons(brain_id, lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_neurons_last_accessed ON neurons(brain_id, last_accessed_at);

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
    device_id TEXT DEFAULT '',  -- Device that last modified this synapse
    device_origin TEXT DEFAULT '',  -- Device that originally created this synapse
    updated_at TEXT DEFAULT '',  -- Last modification timestamp
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, source_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, target_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(brain_id, source_id);
CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(brain_id, target_id);
CREATE INDEX IF NOT EXISTS idx_synapses_pair ON synapses(brain_id, source_id, target_id);
CREATE INDEX IF NOT EXISTS idx_synapses_updated ON synapses(brain_id, updated_at);

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
    essence TEXT,  -- Single-sentence distillation (max 150 chars) for fidelity layers
    last_ghost_shown_at TEXT,  -- When fiber was last rendered as ghost in context
    tags TEXT DEFAULT '[]',  -- JSON array (union of auto_tags + agent_tags)
    auto_tags TEXT DEFAULT '[]',  -- JSON array: tags from auto-extraction
    agent_tags TEXT DEFAULT '[]',  -- JSON array: tags from calling agent
    metadata TEXT DEFAULT '{}',  -- JSON
    compression_tier INTEGER DEFAULT 0,  -- 0=full, 1=extractive, 2=entity, 3=template, 4=graph-only
    pinned INTEGER DEFAULT 0,  -- 1=KB memory, skip decay/prune/compress
    device_id TEXT DEFAULT '',  -- Device that last modified this fiber
    device_origin TEXT DEFAULT '',  -- Device that originally created this fiber
    updated_at TEXT DEFAULT '',  -- Last modification timestamp
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fibers_created ON fibers(brain_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fibers_salience ON fibers(brain_id, salience);
CREATE INDEX IF NOT EXISTS idx_fibers_conductivity ON fibers(brain_id, conductivity);
CREATE INDEX IF NOT EXISTS idx_fibers_updated ON fibers(brain_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_fibers_pinned ON fibers(brain_id, pinned);
CREATE INDEX IF NOT EXISTS idx_fibers_time_range ON fibers(brain_id, created_at, time_start);

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
    trust_score REAL DEFAULT NULL,
    source TEXT DEFAULT NULL,
    PRIMARY KEY (brain_id, fiber_id),
    FOREIGN KEY (brain_id, fiber_id) REFERENCES fibers(brain_id, id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE,
    FOREIGN KEY (brain_id, project_id) REFERENCES projects(brain_id, id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_typed_memories_type ON typed_memories(brain_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_typed_memories_project ON typed_memories(brain_id, project_id);
CREATE INDEX IF NOT EXISTS idx_typed_memories_expires ON typed_memories(brain_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_typed_memories_trust ON typed_memories(brain_id, trust_score);

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

-- Sync state persistence for external source auto-sync
CREATE TABLE IF NOT EXISTS sync_states (
    source_system TEXT NOT NULL,
    source_collection TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    last_sync_at TEXT,
    records_imported INTEGER DEFAULT 0,
    last_record_id TEXT,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (brain_id, source_system, source_collection)
);

-- Proactive alerts queue for brain health monitoring
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'low',
    message TEXT NOT NULL,
    recommended_action TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    seen_at TEXT,
    acknowledged_at TEXT,
    resolved_at TEXT,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (brain_id, id)
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(brain_id, status);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(brain_id, alert_type, status);

-- Spaced repetition review schedules (Leitner box system)
CREATE TABLE IF NOT EXISTS review_schedules (
    fiber_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    box INTEGER NOT NULL DEFAULT 1,
    next_review TEXT,
    last_reviewed TEXT,
    review_count INTEGER DEFAULT 0,
    streak INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (fiber_id, brain_id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_review_next ON review_schedules(brain_id, next_review);
CREATE INDEX IF NOT EXISTS idx_review_box ON review_schedules(brain_id, box);

-- Bayesian depth priors for adaptive recall
CREATE TABLE IF NOT EXISTS depth_priors (
    brain_id TEXT NOT NULL,
    entity_text TEXT NOT NULL,
    depth_level INTEGER NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    total_queries INTEGER DEFAULT 0,
    last_updated TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, entity_text, depth_level),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_depth_priors_entity ON depth_priors(brain_id, entity_text);

-- Compression backups for reversible compression (tiers 1-2)
CREATE TABLE IF NOT EXISTS compression_backups (
    fiber_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    original_content TEXT NOT NULL,
    compression_tier INTEGER NOT NULL,
    compressed_at TEXT NOT NULL,
    original_token_count INTEGER DEFAULT 0,
    compressed_token_count INTEGER DEFAULT 0,
    PRIMARY KEY (brain_id, fiber_id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_compression_tier ON compression_backups(brain_id, compression_tier);

-- Neuron snapshots for recovery of Tier 3/4 (destructive) compression
CREATE TABLE IF NOT EXISTS neuron_snapshots (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    original_content TEXT NOT NULL,
    compressed_at TEXT NOT NULL,
    tier INTEGER NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);

-- Change log (append-only journal for incremental sync)
CREATE TABLE IF NOT EXISTS change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brain_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT '',
    changed_at TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    synced INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_change_log_brain_synced ON change_log(brain_id, synced, changed_at);

-- Device registry for multi-device sync
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    device_name TEXT DEFAULT '',
    last_sync_at TEXT,
    last_sync_sequence INTEGER DEFAULT 0,
    registered_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, device_id)
);

-- Retrieval sufficiency calibration (v18)
CREATE TABLE IF NOT EXISTS retrieval_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brain_id TEXT NOT NULL,
    gate TEXT NOT NULL,
    predicted_sufficient INTEGER NOT NULL,
    actual_confidence REAL NOT NULL DEFAULT 0.0,
    actual_fibers INTEGER NOT NULL DEFAULT 0,
    query_intent TEXT NOT NULL DEFAULT '',
    metrics_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_calibration_brain_gate ON retrieval_calibration(brain_id, gate);
CREATE INDEX IF NOT EXISTS idx_calibration_created ON retrieval_calibration(brain_id, created_at);

-- Tool events (v19)
CREATE TABLE IF NOT EXISTS tool_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brain_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    server_name TEXT NOT NULL DEFAULT '',
    args_summary TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 1,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    session_id TEXT NOT NULL DEFAULT '',
    task_context TEXT NOT NULL DEFAULT '',
    processed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tool_events_tool ON tool_events(brain_id, tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_events_processed ON tool_events(brain_id, processed);
CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(brain_id, session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_events_created ON tool_events(brain_id, created_at);

-- Training file tracking (doc-to-brain dedup & resume)
CREATE TABLE IF NOT EXISTS training_files (
    id TEXT PRIMARY KEY,
    brain_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    chunks_total INTEGER NOT NULL DEFAULT 0,
    chunks_completed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    domain_tag TEXT DEFAULT '',
    trained_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_training_files_hash ON training_files(brain_id, file_hash);

-- Cognitive layer: hypothesis/prediction confidence tracking
CREATE TABLE IF NOT EXISTS cognitive_state (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_for_count INTEGER NOT NULL DEFAULT 0,
    evidence_against_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'confirmed', 'refuted', 'superseded', 'pending', 'expired')),
    predicted_at TEXT,
    resolved_at TEXT,
    schema_version INTEGER DEFAULT 1,
    parent_schema_id TEXT,
    last_evidence_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cognitive_confidence ON cognitive_state(brain_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_cognitive_status ON cognitive_state(brain_id, status);

-- Cognitive layer: pre-computed hot index (max 20 entries per brain)
CREATE TABLE IF NOT EXISTS hot_index (
    brain_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    category TEXT NOT NULL,
    neuron_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence REAL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, slot),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);

-- Cognitive layer: metacognition — knowledge gaps
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id TEXT PRIMARY KEY,
    brain_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    detection_source TEXT NOT NULL,
    related_neuron_ids TEXT DEFAULT '[]',
    resolved_at TEXT,
    resolved_by_neuron_id TEXT,
    priority REAL DEFAULT 0.5,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_gaps_brain ON knowledge_gaps(brain_id, resolved_at);
CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(brain_id, priority DESC);

-- Source registry for provenance tracking
CREATE TABLE IF NOT EXISTS sources (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'document',
    version TEXT DEFAULT '',
    effective_date TEXT,
    expires_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    file_hash TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(brain_id, source_type);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(brain_id, status);
CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(brain_id, name);

-- Session summaries for session intelligence
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    topics_json TEXT NOT NULL DEFAULT '[]',
    topic_weights_json TEXT NOT NULL DEFAULT '{}',
    top_entities_json TEXT NOT NULL DEFAULT '[]',
    query_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL NOT NULL DEFAULT 0.0,
    avg_depth REAL NOT NULL DEFAULT 0.0,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_session_summaries_brain ON session_summaries(brain_id, ended_at);
CREATE INDEX IF NOT EXISTS idx_session_summaries_session ON session_summaries(session_id);

-- Retriever calibration: per-brain EMA weights for RRF (v25)
CREATE TABLE IF NOT EXISTS retriever_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brain_id TEXT NOT NULL,
    retriever_type TEXT NOT NULL,
    contributed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_retriever_cal_brain ON retriever_calibration(brain_id, retriever_type, created_at);

-- Tag co-occurrence matrix for semantic drift detection (v26)
CREATE TABLE IF NOT EXISTS tag_cooccurrence (
    brain_id TEXT NOT NULL,
    tag_a TEXT NOT NULL,
    tag_b TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (brain_id, tag_a, tag_b),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tag_cooccurrence_brain ON tag_cooccurrence(brain_id, count DESC);

-- Drift detection results (v26)
CREATE TABLE IF NOT EXISTS drift_clusters (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    canonical TEXT NOT NULL,
    members TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'detected',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_drift_clusters_status ON drift_clusters(brain_id, status);
"""
