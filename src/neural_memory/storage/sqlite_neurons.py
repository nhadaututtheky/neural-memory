"""SQLite neuron and neuron state operations mixin."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
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
                """INSERT INTO neurons (id, brain_id, type, content, metadata, content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    neuron.id,
                    brain_id,
                    neuron.type.value,
                    neuron.content,
                    json.dumps(neuron.metadata),
                    neuron.content_hash,
                    neuron.created_at.isoformat(),
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
            self._neuron_cache.invalidate()
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

    async def find_neurons_exact_batch(
        self,
        contents: list[str],
        type: NeuronType | None = None,
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
    ) -> list[Neuron]:
        # Cache shortcut for exact-match lookups (most repeated pattern)
        if content_exact is not None and content_contains is None and time_range is None:
            type_val = type.value if type is not None else None
            cached = self._neuron_cache.get(content_exact, type_val)
            if cached is not None:
                return cached[:limit]

        limit = min(limit, 1000)
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

            query += " ORDER BY fts.rank LIMIT ?"
            params.append(limit)
        else:
            # Fallback: original LIKE query (or exact match / no content filter)
            query = "SELECT * FROM neurons WHERE brain_id = ?"
            params = [brain_id]

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
                " WHERE n.brain_id = ? AND n.content LIKE ?"
            )
            params = [brain_id, f"{prefix}%"]

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
