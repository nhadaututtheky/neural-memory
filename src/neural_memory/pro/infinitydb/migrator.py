"""SQLite → InfinityDB migration engine.

Migrates neurons, synapses, and fibers from Neural Memory's SQLite
storage to InfinityDB format. Supports incremental migration with
progress tracking.

Usage:
    migrator = SQLiteToInfinityMigrator(sqlite_path, infinity_db)
    stats = await migrator.migrate()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neural_memory.pro.infinitydb.engine import InfinityDB

logger = logging.getLogger(__name__)

# Batch size for migration operations
MIGRATION_BATCH_SIZE = 1000

# Maximum errors to track before silently discarding
_MAX_ERRORS = 20

# Valid neuron types — unknown values fall back to "fact"
_VALID_NEURON_TYPES = frozenset(
    {
        "fact",
        "decision",
        "error",
        "insight",
        "preference",
        "workflow",
        "instruction",
        "concept",
        "entity",
        "pattern",
    }
)

# Priority bounds
_MIN_PRIORITY = 1
_MAX_PRIORITY = 10

# Whitelist for estimate_migration table queries
_COUNTABLE_TABLES = frozenset({"neurons", "synapses", "fibers"})

# Pre-built count queries (avoids f-string SQL — H1 fix)
_COUNT_QUERIES: dict[str, str] = {
    table: f"SELECT COUNT(*) FROM {table}" for table in _COUNTABLE_TABLES
}


@dataclass
class MigrationStats:
    """Statistics from a migration run."""

    neurons_migrated: int = 0
    synapses_migrated: int = 0
    fibers_migrated: int = 0
    neurons_skipped: int = 0
    synapses_skipped: int = 0
    fibers_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_migrated(self) -> int:
        return self.neurons_migrated + self.synapses_migrated + self.fibers_migrated

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, msg: str) -> None:
        """Add an error message, respecting the global cap."""
        if len(self.errors) < _MAX_ERRORS:
            self.errors.append(msg)

    def as_dict(self) -> dict[str, Any]:
        return {
            "neurons_migrated": self.neurons_migrated,
            "synapses_migrated": self.synapses_migrated,
            "fibers_migrated": self.fibers_migrated,
            "neurons_skipped": self.neurons_skipped,
            "synapses_skipped": self.synapses_skipped,
            "fibers_skipped": self.fibers_skipped,
            "errors": list(self.errors),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_migrated": self.total_migrated,
        }


class SQLiteToInfinityMigrator:
    """Migrates data from Neural Memory's SQLite database to InfinityDB."""

    def __init__(
        self,
        sqlite_path: str | Path,
        target_db: InfinityDB,
        *,
        batch_size: int = MIGRATION_BATCH_SIZE,
        skip_existing: bool = True,
    ) -> None:
        self._sqlite_path = Path(sqlite_path)
        self._target = target_db
        self._batch_size = batch_size
        self._skip_existing = skip_existing

    async def migrate(self) -> MigrationStats:
        """Run full migration. Returns stats."""
        import time

        t0 = time.perf_counter()
        stats = MigrationStats()

        if not self._sqlite_path.exists():
            stats.errors.append(f"SQLite file not found: {self._sqlite_path}")
            return stats

        try:
            # Run migration in thread to not block event loop
            await asyncio.to_thread(self._migrate_sync, stats)
        except Exception as e:
            stats.add_error(f"Migration failed: {e}")
            logger.error("Migration failed: %s", e)

        stats.elapsed_seconds = time.perf_counter() - t0
        logger.info(
            "Migration complete: %d neurons, %d synapses, %d fibers in %.2fs",
            stats.neurons_migrated,
            stats.synapses_migrated,
            stats.fibers_migrated,
            stats.elapsed_seconds,
        )
        return stats

    def _migrate_sync(self, stats: MigrationStats) -> None:
        """Synchronous migration — runs in a thread."""
        # H2 fix: contextlib.closing ensures connection closes on all exit paths
        with contextlib.closing(sqlite3.connect(str(self._sqlite_path))) as conn:
            conn.row_factory = sqlite3.Row
            # C3 fix: pin a consistent read snapshot for the entire migration
            conn.execute("BEGIN DEFERRED")
            try:
                self._migrate_neurons(conn, stats)
                self._migrate_synapses(conn, stats)
                self._migrate_fibers(conn, stats)
            finally:
                conn.execute("ROLLBACK")  # read-only, just release the snapshot

    def _migrate_neurons(self, conn: sqlite3.Connection, stats: MigrationStats) -> None:
        """Migrate neurons table using keyset pagination."""
        cursor = conn.cursor()

        # Check if neurons table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'")
        if cursor.fetchone() is None:
            logger.info("No neurons table found, skipping neuron migration")
            return

        # Get column names to handle different schema versions
        cursor.execute("PRAGMA table_info(neurons)")
        columns = {row["name"] for row in cursor.fetchall()}

        # Determine the ID column name for keyset pagination
        id_col = "id" if "id" in columns else "neuron_id" if "neuron_id" in columns else None
        if id_col is None:
            stats.add_error("Neurons table has no 'id' or 'neuron_id' column")
            return

        cursor.execute("SELECT COUNT(*) FROM neurons")
        total = cursor.fetchone()[0]
        logger.info("Migrating %d neurons...", total)

        # H4 fix: keyset pagination (O(N) vs O(N²) with OFFSET)
        last_id = ""
        migrated_in_phase = 0
        while migrated_in_phase < total:
            cursor.execute(
                f"SELECT * FROM neurons WHERE {id_col} > ? ORDER BY {id_col} LIMIT ?",
                (last_id, self._batch_size),
            )
            rows = cursor.fetchall()
            if not rows:
                break

            batch: list[dict[str, Any]] = []
            for row in rows:
                row_dict = dict(row)
                last_id = row_dict.get(id_col, last_id)
                neuron = self._row_to_neuron(row_dict, columns)
                if neuron:
                    batch.append(neuron)

            if batch:
                # C1 fix: _add_neurons_batch_sync is atomic (has rollback),
                # so on exception none were committed — fallback retries all
                try:
                    self._target._add_neurons_batch_sync(batch)
                    stats.neurons_migrated += len(batch)
                except Exception as e:
                    stats.add_error(f"Neuron batch after '{last_id}': {e}")
                    # Fallback: individual inserts (batch was rolled back, no dupes)
                    for neuron in batch:
                        try:
                            self._target._add_neurons_batch_sync([neuron])
                            stats.neurons_migrated += 1
                        except Exception as ie:
                            stats.neurons_skipped += 1
                            stats.add_error(f"Neuron {neuron.get('neuron_id', '?')}: {ie}")

            migrated_in_phase += len(rows)

    def _row_to_neuron(self, row: dict[str, Any], columns: set[str]) -> dict[str, Any] | None:
        """Convert a SQLite row to a neuron dict for InfinityDB."""
        nid = row.get("id") or row.get("neuron_id")
        if not nid:
            return None

        content = row.get("content", "")

        # M3 fix: validate neuron_type against known set
        raw_type = row.get("type", row.get("neuron_type", "fact"))
        neuron_type = raw_type if raw_type in _VALID_NEURON_TYPES else "fact"
        if raw_type != neuron_type:
            logger.warning("Unknown neuron type %r for %s, defaulting to 'fact'", raw_type, nid)

        # M4 fix: clamp priority to valid range
        raw_priority = row.get("priority", 5)
        priority = (
            max(_MIN_PRIORITY, min(_MAX_PRIORITY, int(raw_priority)))
            if raw_priority is not None
            else 5
        )

        result: dict[str, Any] = {
            "neuron_id": str(nid),
            "content": content or "",
            "neuron_type": neuron_type,
            "priority": priority,
        }

        # Optional fields
        if "activation_level" in columns and row.get("activation_level") is not None:
            result["activation_level"] = float(row["activation_level"])
        if "ephemeral" in columns:
            result["ephemeral"] = bool(row.get("ephemeral", False))
        if "tags" in columns and row.get("tags"):
            tags = row["tags"]
            if isinstance(tags, str):
                result["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        # Preserve original creation timestamp
        if "created_at" in columns and row.get("created_at") is not None:
            result["created_at"] = str(row["created_at"])

        # Embedding (stored as blob in some NM versions)
        if "embedding" in columns and row.get("embedding") is not None:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                if emb.shape[0] == self._target.dimensions:
                    result["embedding"] = emb
            except (ValueError, TypeError):
                pass  # Skip invalid embeddings

        return result

    def _migrate_synapses(self, conn: sqlite3.Connection, stats: MigrationStats) -> None:
        """Migrate synapses table using keyset pagination."""
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synapses'")
        if cursor.fetchone() is None:
            logger.info("No synapses table found, skipping synapse migration")
            return

        # Determine ID column for keyset pagination
        cursor.execute("PRAGMA table_info(synapses)")
        syn_columns = {row["name"] for row in cursor.fetchall()}
        id_col = (
            "id" if "id" in syn_columns else "synapse_id" if "synapse_id" in syn_columns else None
        )

        cursor.execute("SELECT COUNT(*) FROM synapses")
        total = cursor.fetchone()[0]
        logger.info("Migrating %d synapses...", total)

        # H4 fix: keyset pagination for synapses too
        last_id = ""
        migrated_in_phase = 0
        while migrated_in_phase < total:
            if id_col:
                cursor.execute(
                    f"SELECT * FROM synapses WHERE {id_col} > ? ORDER BY {id_col} LIMIT ?",
                    (last_id, self._batch_size),
                )
            else:
                # Fallback to OFFSET if no ID column (rare, but safe)
                cursor.execute(
                    "SELECT * FROM synapses LIMIT ? OFFSET ?",
                    (self._batch_size, migrated_in_phase),
                )
            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                row_dict = dict(row)
                if id_col:
                    last_id = str(row_dict.get(id_col, last_id))
                source = row_dict.get("source_id", "")
                target = row_dict.get("target_id", "")
                syn_type = (
                    row_dict.get("type", row_dict.get("synapse_type", "related")) or "related"
                )
                weight = float(row_dict.get("weight", 1.0))
                edge_id = row_dict.get("id") or row_dict.get("synapse_id")

                if not source or not target:
                    stats.synapses_skipped += 1
                    continue

                try:
                    # Note: we call graph store directly (not engine) to skip
                    # neuron existence validation during migration (H3 acknowledged)
                    self._target._graph.add_edge(
                        str(source),
                        str(target),
                        edge_type=syn_type,
                        weight=weight,
                        edge_id=str(edge_id) if edge_id else None,
                    )
                    stats.synapses_migrated += 1
                except Exception as e:
                    stats.synapses_skipped += 1
                    stats.add_error(f"Synapse {source}->{target}: {e}")

            migrated_in_phase += len(rows)

    def _migrate_fibers(self, conn: sqlite3.Connection, stats: MigrationStats) -> None:
        """Migrate fibers table in batches."""
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fibers'")
        if cursor.fetchone() is None:
            logger.info("No fibers table found, skipping fiber migration")
            return

        # Check if fiber_neurons join table exists (do this once, not per-fiber)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fiber_neurons'")
        has_fiber_neurons = cursor.fetchone() is not None

        # Determine ID column for keyset pagination
        cursor.execute("PRAGMA table_info(fibers)")
        fiber_columns = {row["name"] for row in cursor.fetchall()}
        id_col = (
            "id" if "id" in fiber_columns else "fiber_id" if "fiber_id" in fiber_columns else None
        )

        cursor.execute("SELECT COUNT(*) FROM fibers")
        total = cursor.fetchone()[0]
        logger.info("Migrating %d fibers...", total)

        # C2 fix: batch fibers like neurons and synapses
        last_id = ""
        migrated_in_phase = 0
        while migrated_in_phase < total:
            if id_col:
                cursor.execute(
                    f"SELECT * FROM fibers WHERE {id_col} > ? ORDER BY {id_col} LIMIT ?",
                    (last_id, self._batch_size),
                )
            else:
                cursor.execute(
                    "SELECT * FROM fibers LIMIT ? OFFSET ?",
                    (self._batch_size, migrated_in_phase),
                )
            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                row_dict = dict(row)
                fid = row_dict.get("id") or row_dict.get("fiber_id")
                if id_col:
                    last_id = str(row_dict.get(id_col, last_id))
                name = row_dict.get("name", "")
                fiber_type = (
                    row_dict.get("type", row_dict.get("fiber_type", "cluster")) or "cluster"
                )
                description = row_dict.get("description", "")

                if not fid or not name:
                    stats.fibers_skipped += 1
                    continue

                try:
                    # Get neuron-fiber associations if join table exists
                    neuron_ids: list[str] = []
                    if has_fiber_neurons:
                        cursor.execute(
                            "SELECT neuron_id FROM fiber_neurons WHERE fiber_id = ?",
                            (fid,),
                        )
                        neuron_ids = [str(r["neuron_id"]) for r in cursor.fetchall()]

                    # Preserve fiber metadata for round-trip fidelity
                    fiber_meta: dict[str, Any] = {}
                    for key in (
                        "salience", "conductivity", "coherence", "frequency",
                        "compression_tier", "pinned",
                    ):
                        val = row_dict.get(key)
                        if val is not None:
                            fiber_meta[key] = val
                    for dt_key in ("time_start", "time_end", "created_at", "last_conducted"):
                        val = row_dict.get(dt_key)
                        if val is not None:
                            fiber_meta[dt_key] = str(val)
                    # Tags (stored as comma-separated in SQLite)
                    for tag_key in ("auto_tags", "agent_tags"):
                        raw = row_dict.get(tag_key)
                        if raw:
                            fiber_meta[tag_key] = [
                                t.strip() for t in str(raw).split(",") if t.strip()
                            ]
                    for str_key in ("pathway", "anchor_neuron_id", "essence"):
                        val = row_dict.get(str_key)
                        if val is not None:
                            fiber_meta[str_key] = val
                    # Synapse IDs from join table
                    synapse_ids: list[str] = []
                    if has_fiber_neurons:
                        try:
                            cursor.execute(
                                "SELECT synapse_id FROM fiber_neurons WHERE fiber_id = ? AND synapse_id IS NOT NULL",
                                (fid,),
                            )
                            synapse_ids = [str(r[0]) for r in cursor.fetchall() if r[0]]
                        except Exception:
                            pass  # Column may not exist
                    if synapse_ids:
                        fiber_meta["synapse_ids"] = synapse_ids

                    self._target._fibers.add_fiber(
                        name=name,
                        fiber_id=str(fid),
                        fiber_type=fiber_type,
                        description=description,
                        neuron_ids=neuron_ids,
                        metadata=fiber_meta if fiber_meta else None,
                    )
                    stats.fibers_migrated += 1
                except Exception as e:
                    stats.fibers_skipped += 1
                    stats.add_error(f"Fiber {fid}: {e}")

            migrated_in_phase += len(rows)


async def estimate_migration(sqlite_path: str | Path) -> dict[str, Any]:
    """Estimate migration size without actually migrating."""
    path = Path(sqlite_path)
    if not path.exists():
        return {"error": "File not found", "path": str(path)}

    def _estimate() -> dict[str, Any]:
        # H2 fix: use contextlib.closing for safe connection lifecycle
        with contextlib.closing(sqlite3.connect(str(path))) as conn:
            result: dict[str, Any] = {"path": str(path)}
            cursor = conn.cursor()
            for table in ("neurons", "synapses", "fibers"):
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                if cursor.fetchone():
                    # H1 fix: use pre-built queries from whitelist, no f-string
                    cursor.execute(_COUNT_QUERIES[table])
                    result[f"{table}_count"] = cursor.fetchone()[0]
                else:
                    result[f"{table}_count"] = 0
            result["file_size_bytes"] = path.stat().st_size
            return result

    return await asyncio.to_thread(_estimate)
