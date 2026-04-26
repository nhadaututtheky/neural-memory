"""Tests for InfinityDB Phase 6 — SQLite → InfinityDB Migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.migrator import (
    _MAX_ERRORS,
    MIGRATION_BATCH_SIZE,
    MigrationStats,
    SQLiteToInfinityMigrator,
    estimate_migration,
)

# ── Helpers ──


def _create_sqlite_db(
    path: Path,
    *,
    neurons: int = 0,
    synapses: int = 0,
    fibers: int = 0,
    embeddings: bool = False,
    dimensions: int = 8,
    with_fiber_neurons: bool = True,
) -> Path:
    """Create a minimal SQLite database mimicking Neural Memory schema."""
    conn = sqlite3.connect(str(path))
    c = conn.cursor()

    # Create neurons table
    c.execute("""
        CREATE TABLE neurons (
            id TEXT PRIMARY KEY,
            content TEXT,
            type TEXT DEFAULT 'fact',
            priority INTEGER DEFAULT 5,
            activation_level REAL DEFAULT 1.0,
            ephemeral INTEGER DEFAULT 0,
            tags TEXT DEFAULT '',
            embedding BLOB
        )
    """)

    # Create synapses table
    c.execute("""
        CREATE TABLE synapses (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            target_id TEXT,
            type TEXT DEFAULT 'related',
            weight REAL DEFAULT 1.0
        )
    """)

    # Create fibers table
    c.execute("""
        CREATE TABLE fibers (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT DEFAULT 'cluster',
            description TEXT DEFAULT ''
        )
    """)

    if with_fiber_neurons:
        c.execute("""
            CREATE TABLE fiber_neurons (
                fiber_id TEXT,
                neuron_id TEXT,
                PRIMARY KEY (fiber_id, neuron_id)
            )
        """)

    # Insert neurons
    for i in range(neurons):
        emb_blob = None
        if embeddings:
            vec = np.random.default_rng(i).standard_normal(dimensions).astype(np.float32)
            emb_blob = vec.tobytes()
        tags = f"tag-{i % 3},tag-{i % 5}" if i % 2 == 0 else ""
        c.execute(
            "INSERT INTO neurons (id, content, type, priority, activation_level, ephemeral, tags, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"n{i}",
                f"neuron content {i}",
                "fact" if i % 2 == 0 else "decision",
                5 + i % 5,
                1.0,
                0,
                tags,
                emb_blob,
            ),
        )

    # Insert synapses (ensure source/target exist)
    for i in range(min(synapses, neurons - 1)):
        c.execute(
            "INSERT INTO synapses (id, source_id, target_id, type, weight) VALUES (?, ?, ?, ?, ?)",
            (f"s{i}", f"n{i}", f"n{i + 1}", "related", 0.8),
        )

    # Insert fibers
    for i in range(fibers):
        c.execute(
            "INSERT INTO fibers (id, name, type, description) VALUES (?, ?, ?, ?)",
            (f"f{i}", f"fiber-{i}", "cluster", f"Description {i}"),
        )
        if with_fiber_neurons and neurons > 0:
            # Associate first 2 neurons to each fiber
            for j in range(min(2, neurons)):
                c.execute(
                    "INSERT OR IGNORE INTO fiber_neurons (fiber_id, neuron_id) VALUES (?, ?)",
                    (f"f{i}", f"n{j}"),
                )

    conn.commit()
    conn.close()
    return path


# ── MigrationStats Tests ──


class TestMigrationStats:
    def test_defaults(self) -> None:
        stats = MigrationStats()
        assert stats.total_migrated == 0
        assert not stats.has_errors

    def test_total_migrated(self) -> None:
        stats = MigrationStats(neurons_migrated=10, synapses_migrated=5, fibers_migrated=3)
        assert stats.total_migrated == 18

    def test_has_errors(self) -> None:
        stats = MigrationStats(errors=["something went wrong"])
        assert stats.has_errors

    def test_as_dict(self) -> None:
        stats = MigrationStats(neurons_migrated=3, elapsed_seconds=1.234)
        d = stats.as_dict()
        assert d["neurons_migrated"] == 3
        assert d["elapsed_seconds"] == 1.23
        assert d["total_migrated"] == 3
        assert "errors" in d

    def test_add_error_respects_cap(self) -> None:
        stats = MigrationStats()
        for i in range(30):
            stats.add_error(f"err-{i}")
        assert len(stats.errors) == _MAX_ERRORS

    def test_fibers_skipped_in_dict(self) -> None:
        stats = MigrationStats(fibers_skipped=3)
        d = stats.as_dict()
        assert d["fibers_skipped"] == 3


# ── Estimate Migration Tests ──


class TestEstimateMigration:
    async def test_estimate_counts(self, tmp_path: Path) -> None:
        db_path = _create_sqlite_db(tmp_path / "test.db", neurons=10, synapses=5, fibers=2)
        result = await estimate_migration(db_path)
        assert result["neurons_count"] == 10
        assert result["synapses_count"] == 5
        assert result["fibers_count"] == 2
        assert result["file_size_bytes"] > 0

    async def test_estimate_missing_file(self, tmp_path: Path) -> None:
        result = await estimate_migration(tmp_path / "nonexistent.db")
        assert "error" in result

    async def test_estimate_empty_db(self, tmp_path: Path) -> None:
        db_path = _create_sqlite_db(tmp_path / "empty.db")
        result = await estimate_migration(db_path)
        assert result["neurons_count"] == 0
        assert result["synapses_count"] == 0
        assert result["fibers_count"] == 0


# ── Migrator Unit Tests ──


@pytest.fixture
def infinity_db(tmp_path: Path) -> InfinityDB:
    return InfinityDB(tmp_path / "infinity", dimensions=8)


class TestMigratorBasic:
    async def test_migrate_missing_sqlite(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(tmp_path / "nonexistent.db", infinity_db)
        stats = await migrator.migrate()
        assert stats.has_errors
        assert "not found" in stats.errors[0]
        await infinity_db.close()

    async def test_migrate_neurons_only(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=5)
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 5
        assert stats.synapses_migrated == 0
        assert stats.fibers_migrated == 0
        assert not stats.has_errors
        assert stats.elapsed_seconds > 0
        assert infinity_db.neuron_count == 5
        await infinity_db.close()

    async def test_migrate_with_embeddings(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=3, embeddings=True, dimensions=8)
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 3
        assert not stats.has_errors

        # Verify embeddings are searchable
        query = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        results = await infinity_db.search_similar(query, k=3)
        assert len(results) == 3
        await infinity_db.close()

    async def test_migrate_synapses(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=5, synapses=4)
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 5
        assert stats.synapses_migrated == 4
        assert not stats.has_errors
        assert infinity_db.synapse_count == 4
        await infinity_db.close()

    async def test_migrate_fibers(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        db_path = _create_sqlite_db(
            tmp_path / "src.db", neurons=3, fibers=2, with_fiber_neurons=True
        )
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 3
        assert stats.fibers_migrated == 2
        assert not stats.has_errors
        assert infinity_db.fiber_count == 2

        # Verify fiber-neuron associations
        fiber = await infinity_db.get_fiber("f0")
        assert fiber is not None
        assert "n0" in fiber.get("neuron_ids", [])
        await infinity_db.close()

    async def test_migrate_full(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Full migration: neurons + embeddings + synapses + fibers."""
        db_path = _create_sqlite_db(
            tmp_path / "src.db",
            neurons=10,
            synapses=8,
            fibers=3,
            embeddings=True,
            dimensions=8,
        )
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 10
        assert stats.synapses_migrated == 8
        assert stats.fibers_migrated == 3
        assert stats.total_migrated == 21
        assert not stats.has_errors
        await infinity_db.close()


class TestMigratorEdgeCases:
    async def test_empty_tables(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        db_path = _create_sqlite_db(tmp_path / "empty.db")
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.total_migrated == 0
        assert not stats.has_errors
        await infinity_db.close()

    async def test_missing_tables(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """SQLite with no neurons/synapses/fibers tables should not crash."""
        db_path = tmp_path / "bare.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (id TEXT)")
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.total_migrated == 0
        assert not stats.has_errors
        await infinity_db.close()

    async def test_neuron_without_id(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Neurons with NULL id should be skipped."""
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE neurons (id TEXT, content TEXT, type TEXT DEFAULT 'fact', priority INTEGER DEFAULT 5)"
        )
        conn.execute("INSERT INTO neurons (id, content) VALUES (NULL, 'no id')")
        conn.execute("INSERT INTO neurons (id, content) VALUES ('n1', 'has id')")
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 1  # only the one with id
        await infinity_db.close()

    async def test_synapse_without_source_or_target(
        self, tmp_path: Path, infinity_db: InfinityDB
    ) -> None:
        """Synapses with empty source/target should be skipped."""
        db_path = tmp_path / "bad_syn.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE neurons (id TEXT)")
        conn.execute(
            "CREATE TABLE synapses (id TEXT, source_id TEXT, target_id TEXT, type TEXT, weight REAL)"
        )
        conn.execute("INSERT INTO synapses VALUES ('s1', '', 'n2', 'related', 1.0)")
        conn.execute("INSERT INTO synapses VALUES ('s2', 'n1', '', 'related', 1.0)")
        conn.execute("INSERT INTO synapses VALUES ('s3', 'n1', 'n2', 'related', 1.0)")
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.synapses_migrated == 1
        assert stats.synapses_skipped == 2
        await infinity_db.close()

    async def test_fiber_without_fiber_neurons_table(
        self, tmp_path: Path, infinity_db: InfinityDB
    ) -> None:
        """Fibers should migrate even without fiber_neurons join table."""
        db_path = _create_sqlite_db(
            tmp_path / "no_join.db", neurons=2, fibers=2, with_fiber_neurons=False
        )
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.fibers_migrated == 2
        await infinity_db.close()

    async def test_wrong_embedding_dimensions(
        self, tmp_path: Path, infinity_db: InfinityDB
    ) -> None:
        """Embeddings with wrong dimensions should be skipped (neuron still migrated)."""
        db_path = tmp_path / "wrong_dim.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE neurons (
                id TEXT PRIMARY KEY, content TEXT, type TEXT DEFAULT 'fact',
                priority INTEGER DEFAULT 5, embedding BLOB
            )
        """)
        # Insert neuron with 16-dim embedding (db expects 8)
        wrong_vec = np.ones(16, dtype=np.float32)
        conn.execute(
            "INSERT INTO neurons (id, content, embedding) VALUES (?, ?, ?)",
            ("n1", "wrong dims", wrong_vec.tobytes()),
        )
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 1  # neuron still migrated
        assert not stats.has_errors

        # Neuron exists but without vector
        neuron = await infinity_db.get_neuron("n1")
        assert neuron is not None
        assert neuron["content"] == "wrong dims"
        await infinity_db.close()

    async def test_tags_parsing(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Tags stored as comma-separated string should be parsed into list."""
        db_path = tmp_path / "tags.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE neurons (
                id TEXT PRIMARY KEY, content TEXT, type TEXT DEFAULT 'fact',
                priority INTEGER DEFAULT 5, tags TEXT
            )
        """)
        conn.execute(
            "INSERT INTO neurons (id, content, tags) VALUES (?, ?, ?)",
            ("n1", "tagged", "python, ai, ml"),
        )
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 1
        neuron = await infinity_db.get_neuron("n1")
        assert neuron is not None
        assert set(neuron.get("tags", [])) == {"python", "ai", "ml"}
        await infinity_db.close()

    async def test_batch_size_respected(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Verify migration works with custom batch size."""
        db_path = _create_sqlite_db(tmp_path / "batch.db", neurons=10)
        await infinity_db.open()

        migrator = SQLiteToInfinityMigrator(db_path, infinity_db, batch_size=3)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 10
        assert not stats.has_errors
        await infinity_db.close()


class TestMigratorAlternateSchema:
    async def test_neuron_id_column_name(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Handle 'neuron_id' column instead of 'id'."""
        db_path = tmp_path / "alt.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE neurons (
                neuron_id TEXT PRIMARY KEY, content TEXT,
                neuron_type TEXT DEFAULT 'fact', priority INTEGER DEFAULT 5
            )
        """)
        conn.execute("INSERT INTO neurons (neuron_id, content) VALUES ('n1', 'alt schema')")
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.neurons_migrated == 1
        neuron = await infinity_db.get_neuron("n1")
        assert neuron is not None
        await infinity_db.close()

    async def test_synapse_alt_columns(self, tmp_path: Path, infinity_db: InfinityDB) -> None:
        """Handle 'synapse_id' and 'synapse_type' column names."""
        db_path = tmp_path / "alt_syn.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE neurons (id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("""
            CREATE TABLE synapses (
                synapse_id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
                synapse_type TEXT DEFAULT 'related', weight REAL DEFAULT 1.0
            )
        """)
        conn.execute("INSERT INTO neurons VALUES ('n1', 'a')")
        conn.execute("INSERT INTO neurons VALUES ('n2', 'b')")
        conn.execute(
            "INSERT INTO synapses (synapse_id, source_id, target_id, synapse_type) VALUES ('s1', 'n1', 'n2', 'causal')"
        )
        conn.commit()
        conn.close()

        await infinity_db.open()
        migrator = SQLiteToInfinityMigrator(db_path, infinity_db)
        stats = await migrator.migrate()

        assert stats.synapses_migrated == 1
        await infinity_db.close()


# ── Integration: Migrate then Query ──


class TestMigrationIntegration:
    async def test_migrate_then_search(self, tmp_path: Path) -> None:
        """After migration, vector search should work."""
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=20, embeddings=True, dimensions=8)
        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        assert stats.neurons_migrated == 20

        # Search by vector
        query = np.random.default_rng(42).standard_normal(8).astype(np.float32)
        results = await db.search_similar(query, k=5)
        assert len(results) == 5

        await db.close()

    async def test_migrate_then_graph_traverse(self, tmp_path: Path) -> None:
        """After migration, graph traversal should work."""
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=10, synapses=9)
        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        assert stats.synapses_migrated == 9

        # BFS from n0 should reach connected neurons
        neighbors = await db.get_neighbors("n0", direction="outgoing")
        assert "n1" in neighbors

        await db.close()

    async def test_migrate_then_fiber_lookup(self, tmp_path: Path) -> None:
        """After migration, fiber lookups should work."""
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=5, fibers=3)
        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        assert stats.fibers_migrated == 3

        fibers = await db.find_fibers()
        assert len(fibers) == 3

        # Fiber 0 should have neuron associations
        fiber = await db.get_fiber("f0")
        assert fiber is not None
        assert len(fiber.get("neuron_ids", [])) > 0

        await db.close()

    async def test_batch_size_constant(self) -> None:
        assert MIGRATION_BATCH_SIZE == 1000

    async def test_idempotent_migration(self, tmp_path: Path) -> None:
        """M5: Running migration twice should not duplicate neurons."""
        db_path = _create_sqlite_db(tmp_path / "src.db", neurons=5, synapses=3, fibers=2)
        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats1 = await migrator.migrate()
        assert stats1.neurons_migrated == 5

        # Second run — neurons already exist, batch will fail, fallback also fails
        await migrator.migrate()
        # Neurons should be skipped (not duplicated)
        assert db.neuron_count == 5  # still 5, not 10

        await db.close()

    async def test_priority_clamped(self, tmp_path: Path) -> None:
        """M4: Priority values outside [1, 10] should be clamped."""
        db_path = tmp_path / "prio.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE neurons (
                id TEXT PRIMARY KEY, content TEXT, type TEXT DEFAULT 'fact',
                priority INTEGER
            )
        """)
        conn.execute("INSERT INTO neurons VALUES ('n1', 'low', 'fact', -5)")
        conn.execute("INSERT INTO neurons VALUES ('n2', 'high', 'fact', 999)")
        conn.execute("INSERT INTO neurons VALUES ('n3', 'normal', 'fact', 7)")
        conn.commit()
        conn.close()

        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        assert stats.neurons_migrated == 3

        n1 = await db.get_neuron("n1")
        n2 = await db.get_neuron("n2")
        n3 = await db.get_neuron("n3")
        assert n1 is not None and n1["priority"] == 1  # clamped up
        assert n2 is not None and n2["priority"] == 10  # clamped down
        assert n3 is not None and n3["priority"] == 7  # unchanged

        await db.close()

    async def test_fibers_modern_schema_no_name_column(self, tmp_path: Path) -> None:
        """Issue #147: production SQLite fibers schema has no `name` column —
        the legacy migrator skipped 100% of fibers because it required both
        `id` AND `name`. Modern fibers carry `summary` instead, and the
        migrator must fall back gracefully."""
        db_path = tmp_path / "modern.db"
        conn = sqlite3.connect(str(db_path))
        # Mirror sqlite_schema.py:832 — no `name` column
        conn.execute(
            """CREATE TABLE fibers (
                id TEXT PRIMARY KEY,
                summary TEXT,
                description TEXT DEFAULT ''
            )"""
        )
        # Fiber 1: summary only — pre-fix this would skip
        conn.execute("INSERT INTO fibers (id, summary) VALUES ('f1', 'cluster about Python')")
        # Fiber 2: empty summary — should still migrate, name falls back to id
        conn.execute("INSERT INTO fibers (id, summary) VALUES ('f2', '')")
        # Fiber 3: NULL summary — same fallback path
        conn.execute("INSERT INTO fibers (id) VALUES ('f3')")
        conn.commit()
        conn.close()

        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()

        # All three fibers must migrate — none are skipped just because
        # they lack a legacy `name` column.
        assert stats.fibers_migrated == 3
        assert stats.fibers_skipped == 0

        f1 = await db.get_fiber("f1")
        assert f1 is not None
        assert f1.get("name") == "cluster about Python"

        await db.close()

    async def test_fiber_summary_preserved_in_metadata(self, tmp_path: Path) -> None:
        """Issue #147 audit Angle 7: `summary` is read out of SQLite to derive
        the InfinityDB `name` and `description`, which loses the original
        field-name semantic. The migrator now ALSO preserves it under
        `metadata["summary"]` so downstream readers using that key still
        see the value."""
        db_path = tmp_path / "summary.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE fibers (
                id TEXT PRIMARY KEY,
                summary TEXT,
                essence TEXT,
                pathway TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO fibers (id, summary, essence, pathway) "
            "VALUES ('f1', 'Big-picture summary', 'one-line essence', '[\"n1\",\"n2\"]')"
        )
        conn.commit()
        conn.close()

        db = InfinityDB(tmp_path / "inf", dimensions=8)
        await db.open()
        try:
            stats = await SQLiteToInfinityMigrator(db_path, db).migrate()
            assert stats.fibers_migrated == 1

            f = await db.get_fiber("f1")
            assert f is not None
            meta = f.get("metadata") or {}
            # The key the original SQLite row used must still be retrievable.
            assert meta.get("summary") == "Big-picture summary"
            assert meta.get("essence") == "one-line essence"
            assert meta.get("pathway") == '["n1","n2"]'
        finally:
            await db.close()

    async def test_runtime_path_round_trip(self, tmp_path: Path) -> None:
        """Issue #147: data migrated to InfinityDB(brains_dir, brain_id=name)
        MUST be readable by reopening at the SAME (brains_dir, brain_id) —
        because that is exactly how the runtime opens the backend after
        `nmem storage switch infinitydb`. Pre-fix the CLI passed
        `InfinityDBStorage(brain_dir)` which placed data one level too deep
        and the runtime read 0 memories."""
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir()
        brain_name = "default"
        sqlite_path = brains_dir / f"{brain_name}.db"
        _create_sqlite_db(sqlite_path, neurons=7, synapses=4, fibers=2)

        # Write through the same call shape the CLI now uses.
        writer = InfinityDB(brains_dir, brain_id=brain_name)
        await writer.open()
        try:
            stats = await SQLiteToInfinityMigrator(sqlite_path, writer).migrate()
            await writer.flush()
        finally:
            await writer.close()

        assert stats.neurons_migrated == 7

        # Now reopen with EXACTLY the runtime invocation pattern.
        reader = InfinityDB(brains_dir, brain_id=brain_name)
        await reader.open()
        try:
            verify = await reader.get_stats()
        finally:
            await reader.close()

        # If the path/brain_id pair was wrong on either side, this would be 0.
        assert verify["neuron_count"] == 7
        assert verify["synapse_count"] == 4
        assert verify["fiber_count"] == 2

    async def test_invalid_neuron_type_defaults_to_fact(self, tmp_path: Path) -> None:
        """M3: Unknown neuron types should default to 'fact'."""
        db_path = tmp_path / "bad_type.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE neurons (
                id TEXT PRIMARY KEY, content TEXT, type TEXT, priority INTEGER DEFAULT 5
            )
        """)
        conn.execute("INSERT INTO neurons VALUES ('n1', 'ok', 'fact', 5)")
        conn.execute("INSERT INTO neurons VALUES ('n2', 'bad', 'GARBAGE_TYPE', 5)")
        conn.execute("INSERT INTO neurons VALUES ('n3', 'empty', '', 5)")
        conn.commit()
        conn.close()

        db = InfinityDB(tmp_path / "infinity", dimensions=8)
        await db.open()

        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        assert stats.neurons_migrated == 3

        n1 = await db.get_neuron("n1")
        n2 = await db.get_neuron("n2")
        n3 = await db.get_neuron("n3")
        assert n1 is not None and n1["type"] == "fact"
        assert n2 is not None and n2["type"] == "fact"  # GARBAGE_TYPE → fact
        assert n3 is not None and n3["type"] == "fact"  # empty → fact

        await db.close()
