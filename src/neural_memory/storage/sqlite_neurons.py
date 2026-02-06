"""SQLite neuron and neuron state operations mixin."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.sqlite_row_mappers import row_to_neuron, row_to_neuron_state

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

    def _ensure_conn(self) -> aiosqlite.Connection: ...
    def _get_brain_id(self) -> str: ...

    _has_fts: bool

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
            return row_to_neuron(row)

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
            return [row_to_neuron(row) for row in rows]

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

        cursor = await conn.execute(
            "DELETE FROM neurons WHERE id = ? AND brain_id = ?",
            (neuron_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

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
            return row_to_neuron_state(row)

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

    async def get_all_neuron_states(self) -> list[NeuronState]:
        """Get all neuron states for current brain."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM neuron_states WHERE brain_id = ?",
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
        if not prefix.strip():
            return []

        conn = self._ensure_conn()
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
