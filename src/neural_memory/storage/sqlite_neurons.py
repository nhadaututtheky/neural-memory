"""SQLite neuron and neuron state operations mixin."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.sqlite_row_mappers import row_to_neuron, row_to_neuron_state
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite


def _build_fts_query(search_term: str) -> str:
    """Build an FTS5 MATCH expression from a user search string.

    Splits on whitespace, quotes each token to escape FTS5 operators
    (AND, OR, NOT, NEAR, *, etc.), and joins with implicit AND.
    Double quotes within tokens are escaped by doubling them.
    Example: 'API design' → '"API" "design"'
    """
    tokens = search_term.split()
    if not tokens:
        return '""'
    return " ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens)


def _build_fts_prefix_query(prefix: str) -> str:
    """Build FTS5 MATCH with prefix on last token.

    All tokens except the last are quoted (exact match).
    The last token is sanitized and gets a ``*`` suffix (prefix match).
    Example: ``'API des'`` → ``'"API" des*'``
    """
    tokens = prefix.split()
    if not tokens:
        return '""'
    parts: list[str] = []
    for token in tokens[:-1]:
        escaped = token.replace(chr(34), chr(34) + chr(34))
        parts.append(f'"{escaped}"')
    last = re.sub(r"[^\w]", "", tokens[-1], flags=re.UNICODE)
    if last:
        parts.append(f"{last}*")
    return " ".join(parts) if parts else '""'


class SQLiteNeuronMixin:
    """Mixin providing neuron and neuron state CRUD operations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    _has_fts: bool

    if TYPE_CHECKING:
        from neural_memory.storage.neuron_cache import NeuronLookupCache

        _neuron_cache: NeuronLookupCache

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO neurons (id, brain_id, type, content, metadata, content_hash, created_at, ephemeral)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    neuron.id,
                    brain_id,
                    neuron.type.value,
                    neuron.content,
                    json.dumps(neuron.metadata),
                    neuron.content_hash,
                    neuron.created_at.isoformat(),
                    1 if neuron.ephemeral else 0,
                ),
            )

            # Initialize state
            await conn.execute(
                """INSERT INTO neuron_states
                   (neuron_id, brain_id, firing_threshold, refractory_period_ms,
                    homeostatic_target, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (neuron.id, brain_id, 0.3, 500.0, 0.5, utcnow().isoformat()),
            )

            await conn.commit()
            # Surgical invalidation: only evict the key that this neuron matches
            self._neuron_cache.invalidate_key(neuron.content, neuron.type.value)
            return neuron.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Neuron {neuron.id} already exists")

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neurons WHERE id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_neuron(row)

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Fetch multiple neurons in a single SQL query."""
        if not neuron_ids:
            return {}

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in neuron_ids)
        query = f"SELECT * FROM neurons WHERE brain_id = ? AND id IN ({placeholders})"
        params: list[Any] = [brain_id, *neuron_ids]

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return {row["id"]: row_to_neuron(row) for row in rows}

    async def has_neuron_by_content_hash(self, content_hash: int) -> bool:
        """Check if a neuron with this content hash exists (fast indexed lookup)."""
        async with self._read_pool.acquire() as db:  # type: ignore[attr-defined]
            cursor = await db.execute(
                "SELECT 1 FROM neurons WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            row = await cursor.fetchone()
            return row is not None

    async def find_neurons_exact_batch(
        self,
        contents: list[str],
        type: NeuronType | None = None,
        ephemeral: bool | None = None,
    ) -> dict[str, Neuron]:
        """Find neurons by exact content for multiple contents in one query."""
        if not contents:
            return {}

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in contents)
        query = f"SELECT * FROM neurons WHERE brain_id = ? AND content IN ({placeholders})"
        params: list[Any] = [brain_id, *contents]

        if type is not None:
            query += " AND type = ?"
            params.append(type.value)

        if ephemeral is not None:
            query += " AND ephemeral = ?"
            params.append(1 if ephemeral else 0)

        results: dict[str, Neuron] = {}
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                neuron = row_to_neuron(row)
                # First match per content wins
                if neuron.content not in results:
                    results[neuron.content] = neuron
        return results

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[Neuron]:
        # Cache shortcut for exact-match lookups (most repeated pattern)
        if content_exact is not None and content_contains is None and time_range is None:
            type_val = type.value if type is not None else None
            cached = self._neuron_cache.get(content_exact, type_val)
            if cached is not None:
                return cached[offset : offset + limit]

        # Full-scan path (no content filter): allow larger batches for pagination
        full_scan = content_contains is None and content_exact is None
        limit = min(limit, 10000 if full_scan else 1000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        use_fts = self._has_fts and content_contains is not None and content_exact is None

        if use_fts:
            # FTS5 path: JOIN for ranked full-text search with BM25
            fts_terms = _build_fts_query(content_contains)  # type: ignore[arg-type]
            query = (
                "SELECT n.* FROM neurons n "
                "JOIN neurons_fts fts ON n.rowid = fts.rowid "
                "WHERE fts.neurons_fts MATCH ? AND fts.brain_id = ?"
            )
            params: list[Any] = [fts_terms, brain_id]

            if type is not None:
                query += " AND n.type = ?"
                params.append(type.value)

            if time_range is not None:
                start, end = time_range
                query += " AND n.created_at >= ? AND n.created_at <= ?"
                params.append(start.isoformat())
                params.append(end.isoformat())

            if ephemeral is not None:
                query += " AND n.ephemeral = ?"
                params.append(1 if ephemeral else 0)

            query += " ORDER BY fts.rank LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        else:
            # Fallback: original LIKE query (or exact match / no content filter)
            query = "SELECT * FROM neurons WHERE brain_id = ?"
            params = [brain_id]

            if type is not None:
                query += " AND type = ?"
                params.append(type.value)

            if content_contains is not None:
                # Escape LIKE wildcards in user input
                escaped = (
                    content_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                query += " AND content LIKE ? ESCAPE '\\'"
                params.append(f"%{escaped}%")

            if content_exact is not None:
                query += " AND content = ?"
                params.append(content_exact)

            if time_range is not None:
                start, end = time_range
                query += " AND created_at >= ? AND created_at <= ?"
                params.append(start.isoformat())
                params.append(end.isoformat())

            if ephemeral is not None:
                query += " AND ephemeral = ?"
                params.append(1 if ephemeral else 0)

            query += " ORDER BY id LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            result = [row_to_neuron(row) for row in rows]

        # Populate cache for exact-match queries
        if content_exact is not None and content_contains is None and time_range is None:
            type_val = type.value if type is not None else None
            self._neuron_cache.put(content_exact, type_val, result)

        return result

    async def update_neuron(self, neuron: Neuron) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE neurons SET type = ?, content = ?, metadata = ?, content_hash = ?
               WHERE id = ? AND brain_id = ?""",
            (
                neuron.type.value,
                neuron.content,
                json.dumps(neuron.metadata),
                neuron.content_hash,
                neuron.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Neuron {neuron.id} does not exist")

        await conn.commit()
        self._neuron_cache.invalidate()

    async def delete_neuron(self, neuron_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM neurons WHERE id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        )
        await conn.commit()
        self._neuron_cache.invalidate()

        return cursor.rowcount > 0

    async def delete_neurons_batch(self, neuron_ids: list[str]) -> int:
        """Delete multiple neurons in batched SQL statements.

        Uses chunked DELETE ... WHERE id IN (...) for efficiency.
        Returns total number of deleted rows.
        """
        if not neuron_ids:
            return 0

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        deleted = 0
        chunk_size = 500

        for start in range(0, len(neuron_ids), chunk_size):
            chunk = neuron_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cursor = await conn.execute(
                f"DELETE FROM neurons WHERE brain_id = ? AND id IN ({placeholders})",
                [brain_id, *chunk],
            )
            deleted += cursor.rowcount

        await conn.commit()
        self._neuron_cache.invalidate()
        return deleted

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neuron_states WHERE neuron_id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_neuron_state(row)

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Batch fetch neuron states in a single SQL query."""
        if not neuron_ids:
            return {}

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in neuron_ids)
        query = f"SELECT * FROM neuron_states WHERE brain_id = ? AND neuron_id IN ({placeholders})"
        params: list[Any] = [brain_id, *neuron_ids]

        result: dict[str, NeuronState] = {}
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                state = row_to_neuron_state(row)
                result[state.neuron_id] = state

        return result

    async def update_neuron_state(self, state: NeuronState) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT OR REPLACE INTO neuron_states
                   (neuron_id, brain_id, activation_level, access_frequency,
                    last_activated, decay_rate, firing_threshold, refractory_until,
                    refractory_period_ms, homeostatic_target, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.neuron_id,
                    brain_id,
                    state.activation_level,
                    state.access_frequency,
                    state.last_activated.isoformat() if state.last_activated else None,
                    state.decay_rate,
                    state.firing_threshold,
                    state.refractory_until.isoformat() if state.refractory_until else None,
                    state.refractory_period_ms,
                    state.homeostatic_target,
                    state.created_at.isoformat(),
                ),
            )
            await conn.commit()
        except sqlite3.IntegrityError:
            # Neuron was deleted (e.g., by consolidation pruning) between
            # state read and state write — skip silently.
            import logging

            logging.getLogger(__name__).debug(
                "Skipping state update for deleted neuron %s", state.neuron_id
            )

    async def update_neuron_states_batch(self, states: list[NeuronState]) -> None:
        """Update multiple neuron states in one batch."""
        if not states:
            return
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        rows = [
            (
                s.neuron_id,
                brain_id,
                s.activation_level,
                s.access_frequency,
                s.last_activated.isoformat() if s.last_activated else None,
                s.decay_rate,
                s.firing_threshold,
                s.refractory_until.isoformat() if s.refractory_until else None,
                s.refractory_period_ms,
                s.homeostatic_target,
                s.created_at.isoformat(),
            )
            for s in states
        ]
        try:
            await conn.executemany(
                """INSERT OR REPLACE INTO neuron_states
                   (neuron_id, brain_id, activation_level, access_frequency,
                    last_activated, decay_rate, firing_threshold, refractory_until,
                    refractory_period_ms, homeostatic_target, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await conn.commit()
        except sqlite3.IntegrityError:
            pass  # Neurons may have been pruned; skip silently as in update_neuron_state

    async def get_all_neuron_states(self) -> list[NeuronState]:
        """Get all neuron states for current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neuron_states WHERE brain_id = ? LIMIT 10000",
            (brain_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row_to_neuron_state(row) for row in rows]

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons matching a prefix, ranked by relevance + frequency."""
        limit = min(limit, 100)
        if not prefix.strip():
            return []

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        if self._has_fts:
            fts_expr = _build_fts_prefix_query(prefix)
            query = (
                "SELECT n.id AS neuron_id, n.content, n.type,"
                " COALESCE(ns.access_frequency, 0) AS access_frequency,"
                " COALESCE(ns.activation_level, 0.0) AS activation_level,"
                " ("
                "   -fts.rank"
                "   + COALESCE(ns.access_frequency, 0) * 0.1"
                "   + COALESCE(ns.activation_level, 0.0) * 0.5"
                " ) AS score"
                " FROM neurons n"
                " JOIN neurons_fts fts ON n.rowid = fts.rowid"
                " LEFT JOIN neuron_states ns"
                "   ON ns.brain_id = n.brain_id AND ns.neuron_id = n.id"
                " WHERE fts.neurons_fts MATCH ? AND fts.brain_id = ?"
            )
            params: list[Any] = [fts_expr, brain_id]

            if type_filter is not None:
                query += " AND n.type = ?"
                params.append(type_filter.value)

            query += " ORDER BY score DESC LIMIT ?"
            params.append(limit)
        else:
            query = (
                "SELECT n.id AS neuron_id, n.content, n.type,"
                " COALESCE(ns.access_frequency, 0) AS access_frequency,"
                " COALESCE(ns.activation_level, 0.0) AS activation_level,"
                " ("
                "   COALESCE(ns.access_frequency, 0) * 0.1"
                "   + COALESCE(ns.activation_level, 0.0) * 0.5"
                " ) AS score"
                " FROM neurons n"
                " LEFT JOIN neuron_states ns"
                "   ON ns.brain_id = n.brain_id AND ns.neuron_id = n.id"
                " WHERE n.brain_id = ? AND n.content LIKE ? ESCAPE '\\'"
            )
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params = [brain_id, f"{escaped}%"]

            if type_filter is not None:
                query += " AND n.type = ?"
                params.append(type_filter.value)

            query += " ORDER BY COALESCE(ns.access_frequency, 0) DESC LIMIT ?"
            params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "neuron_id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "access_frequency": row[3],
                    "activation_level": row[4],
                    "score": row[5],
                }
                for row in rows
            ]

    # ========== Access Tracking ==========

    async def batch_update_last_accessed(self, neuron_ids: list[str]) -> None:
        """Update last_accessed_at for neurons in batch using a single SQL UPDATE.

        Uses placeholders to build ``UPDATE ... WHERE id IN (...)`` safely.

        Args:
            neuron_ids: List of neuron IDs to update.
        """
        if not neuron_ids:
            return

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now_iso = utcnow().isoformat()

        placeholders = ",".join("?" for _ in neuron_ids)
        params: list[Any] = [now_iso, brain_id, *neuron_ids]
        await conn.execute(
            f"UPDATE neurons SET last_accessed_at = ? WHERE brain_id = ? AND id IN ({placeholders})",
            params,
        )
        await conn.commit()

    # ========== Lifecycle State ==========

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        """Update the lifecycle_state column for a neuron.

        Args:
            neuron_id: The neuron ID to update.
            lifecycle_state: New lifecycle state string.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        await conn.execute(
            "UPDATE neurons SET lifecycle_state = ? WHERE id = ? AND brain_id = ?",
            (lifecycle_state, neuron_id, brain_id),
        )
        await conn.commit()

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        """Set or clear the frozen flag for a neuron.

        Args:
            neuron_id: The neuron ID to update.
            frozen: True to prevent compression, False to resume normal lifecycle.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        await conn.execute(
            "UPDATE neurons SET frozen = ? WHERE id = ? AND brain_id = ?",
            (1 if frozen else 0, neuron_id, brain_id),
        )
        await conn.commit()

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        """Set or clear the ephemeral flag for a neuron.

        Args:
            neuron_id: The neuron ID to update.
            ephemeral: True for session-scoped, False for permanent.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        await conn.execute(
            "UPDATE neurons SET ephemeral = ? WHERE id = ? AND brain_id = ?",
            (1 if ephemeral else 0, neuron_id, brain_id),
        )
        await conn.commit()
        self._neuron_cache.invalidate()

    async def update_neurons_ephemeral_batch(self, neuron_ids: list[str], ephemeral: bool) -> None:
        """Batch-set ephemeral flag for multiple neurons.

        Args:
            neuron_ids: Neuron IDs to update.
            ephemeral: True for session-scoped, False for permanent.
        """
        if not neuron_ids:
            return
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        chunk_size = 500
        for start in range(0, len(neuron_ids), chunk_size):
            chunk = neuron_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            await conn.execute(
                f"UPDATE neurons SET ephemeral = ? WHERE brain_id = ? AND id IN ({placeholders})",
                [1 if ephemeral else 0, brain_id, *chunk],
            )
        await conn.commit()
        self._neuron_cache.invalidate()

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        """Return count of neurons by lifecycle_state for the current brain.

        Returns:
            Dict mapping state name to count.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT COALESCE(lifecycle_state, 'active'), COUNT(*) "
            "FROM neurons WHERE brain_id = ? "
            "GROUP BY lifecycle_state",
            (brain_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return {str(row[0]): int(row[1]) for row in rows}

    # ========== Neuron Snapshots (Tier 3-4 recovery) ==========

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        """Save (upsert) a pre-compression content snapshot for a neuron.

        Args:
            neuron_id: The neuron whose content is being snapshotted.
            brain_id: Brain that owns the neuron.
            original_content: Full original text before compression.
            compressed_at: ISO timestamp of when compression occurred.
            tier: The compression tier being applied (3 or 4).
        """
        conn = self._ensure_conn()
        await conn.execute(
            """INSERT INTO neuron_snapshots (neuron_id, brain_id, original_content, compressed_at, tier)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(brain_id, neuron_id) DO UPDATE SET
                   original_content = excluded.original_content,
                   compressed_at = excluded.compressed_at,
                   tier = excluded.tier""",
            (neuron_id, brain_id, original_content, compressed_at, tier),
        )
        await conn.commit()

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        """Retrieve the snapshot for a neuron, if any.

        Args:
            neuron_id: The neuron ID to look up.

        Returns:
            Dict with snapshot fields or None if not found.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT neuron_id, brain_id, original_content, compressed_at, tier "
            "FROM neuron_snapshots WHERE neuron_id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "neuron_id": row[0],
                "brain_id": row[1],
                "original_content": row[2],
                "compressed_at": row[3],
                "tier": row[4],
            }

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        """Delete the snapshot for a neuron.

        Args:
            neuron_id: The neuron ID whose snapshot should be removed.

        Returns:
            True if a row was deleted, False if no snapshot existed.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cursor = await conn.execute(
            "DELETE FROM neuron_snapshots WHERE neuron_id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        """Delete ephemeral neurons older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before ephemeral neurons are deleted.

        Returns:
            Number of deleted neurons.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cutoff = (utcnow() - timedelta(hours=max_age_hours)).isoformat()

        cursor = await conn.execute(
            "DELETE FROM neurons WHERE brain_id = ? AND ephemeral = 1 AND created_at < ?",
            (brain_id, cutoff),
        )
        await conn.commit()
        if cursor.rowcount > 0:
            self._neuron_cache.invalidate()
        return cursor.rowcount
