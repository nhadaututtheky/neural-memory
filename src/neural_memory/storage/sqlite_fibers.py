"""SQLite fiber operations mixin."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.sqlite_row_mappers import row_to_fiber

if TYPE_CHECKING:
    import aiosqlite


class SQLiteFiberMixin:
    """Mixin providing fiber CRUD operations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_fiber(self, fiber: Fiber) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO fibers
                   (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                    pathway, conductivity, last_conducted,
                    time_start, time_end, coherence, salience, frequency,
                    summary, tags, auto_tags, agent_tags, metadata,
                    compression_tier, pinned, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fiber.id,
                    brain_id,
                    json.dumps(list(fiber.neuron_ids)),
                    json.dumps(list(fiber.synapse_ids)),
                    fiber.anchor_neuron_id,
                    json.dumps(fiber.pathway),
                    fiber.conductivity,
                    fiber.last_conducted.isoformat() if fiber.last_conducted else None,
                    fiber.time_start.isoformat() if fiber.time_start else None,
                    fiber.time_end.isoformat() if fiber.time_end else None,
                    fiber.coherence,
                    fiber.salience,
                    fiber.frequency,
                    fiber.summary,
                    json.dumps(list(fiber.tags)),
                    json.dumps(list(fiber.auto_tags)),
                    json.dumps(list(fiber.agent_tags)),
                    json.dumps(fiber.metadata),
                    fiber.compression_tier,
                    1 if fiber.pinned else 0,
                    fiber.created_at.isoformat(),
                ),
            )

            # Populate junction table for fast lookups
            if fiber.neuron_ids:
                await conn.executemany(
                    "INSERT OR IGNORE INTO fiber_neurons (brain_id, fiber_id, neuron_id) VALUES (?, ?, ?)",
                    [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
                )

            await conn.commit()
            return fiber.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Fiber {fiber.id} already exists")

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM fibers WHERE id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_fiber(row)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        limit = min(limit, 1000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM fibers WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if contains_neuron is not None:
            query += " AND id IN (SELECT fiber_id FROM fiber_neurons WHERE brain_id = ? AND neuron_id = ?)"
            params.extend([brain_id, contains_neuron])

        if time_overlaps is not None:
            start, end = time_overlaps
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
            fibers = [row_to_fiber(row) for row in rows]

        # Filter by tags in Python (JSON array doesn't support efficient set operations)
        if tags is not None:
            fibers = [f for f in fibers if tags.issubset(f.tags)]

        return fibers

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
    ) -> list[Fiber]:
        """Find fibers containing any of the given neurons in a single SQL query."""
        if not neuron_ids:
            return []

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in neuron_ids)
        # Use junction table for efficient lookup, limit total results
        total_limit = limit_per_neuron * len(neuron_ids)
        query = f"""
            SELECT DISTINCT f.* FROM fibers f
            JOIN fiber_neurons fn ON f.brain_id = fn.brain_id AND f.id = fn.fiber_id
            WHERE fn.brain_id = ? AND fn.neuron_id IN ({placeholders})
            ORDER BY f.salience DESC
            LIMIT ?
        """
        params: list[Any] = [brain_id, *neuron_ids, total_limit]

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row_to_fiber(row) for row in rows]

    async def update_fiber(self, fiber: Fiber) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE fibers SET neuron_ids = ?, synapse_ids = ?,
               anchor_neuron_id = ?, pathway = ?, conductivity = ?,
               last_conducted = ?, time_start = ?, time_end = ?,
               coherence = ?, salience = ?, frequency = ?,
               summary = ?, tags = ?, auto_tags = ?, agent_tags = ?,
               metadata = ?, compression_tier = ?, pinned = ?
               WHERE id = ? AND brain_id = ?""",
            (
                json.dumps(list(fiber.neuron_ids)),
                json.dumps(list(fiber.synapse_ids)),
                fiber.anchor_neuron_id,
                json.dumps(fiber.pathway),
                fiber.conductivity,
                fiber.last_conducted.isoformat() if fiber.last_conducted else None,
                fiber.time_start.isoformat() if fiber.time_start else None,
                fiber.time_end.isoformat() if fiber.time_end else None,
                fiber.coherence,
                fiber.salience,
                fiber.frequency,
                fiber.summary,
                json.dumps(list(fiber.tags)),
                json.dumps(list(fiber.auto_tags)),
                json.dumps(list(fiber.agent_tags)),
                json.dumps(fiber.metadata),
                fiber.compression_tier,
                1 if fiber.pinned else 0,
                fiber.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Fiber {fiber.id} does not exist")

        # Refresh junction table
        await conn.execute(
            "DELETE FROM fiber_neurons WHERE brain_id = ? AND fiber_id = ?",
            (brain_id, fiber.id),
        )
        if fiber.neuron_ids:
            await conn.executemany(
                "INSERT OR IGNORE INTO fiber_neurons (brain_id, fiber_id, neuron_id) VALUES (?, ?, ?)",
                [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
            )

        await conn.commit()

    async def delete_fiber(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Delete junction entries first
        await conn.execute(
            "DELETE FROM fiber_neurons WHERE brain_id = ? AND fiber_id = ?",
            (brain_id, fiber_id),
        )

        cursor = await conn.execute(
            "DELETE FROM fibers WHERE id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_pinned_neuron_ids(self) -> set[str]:
        """Get all neuron IDs that belong to pinned fibers.

        Used by lifecycle systems (decay, prune) to skip pinned neurons.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT neuron_ids FROM fibers WHERE brain_id = ? AND pinned = 1",
            (brain_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        result: set[str] = set()
        for row in rows:
            neuron_ids_raw = row[0]
            if neuron_ids_raw:
                result.update(json.loads(neuron_ids_raw))
        return result

    async def pin_fibers(self, fiber_ids: list[str], pinned: bool = True) -> int:
        """Pin or unpin fibers by ID.

        Returns:
            Number of fibers updated.
        """
        if not fiber_ids:
            return 0

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        pin_val = 1 if pinned else 0
        placeholders = ",".join("?" for _ in fiber_ids)
        cursor = await conn.execute(
            f"UPDATE fibers SET pinned = ? WHERE brain_id = ? AND id IN ({placeholders})",  # noqa: S608
            [pin_val, brain_id, *fiber_ids],
        )
        await conn.commit()
        return cursor.rowcount

    async def get_stale_fiber_count(self, brain_id: str, stale_days: int = 90) -> int:
        conn = self._ensure_read_conn()
        from neural_memory.utils.timeutils import utcnow

        cutoff = (utcnow() - timedelta(days=stale_days)).isoformat()

        async with conn.execute(
            """SELECT COUNT(*) FROM fibers
               WHERE brain_id = ?
                 AND (
                   (last_conducted IS NULL AND created_at <= ?)
                   OR (last_conducted IS NOT NULL AND last_conducted <= ?)
                 )""",
            (brain_id, cutoff, cutoff),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        limit = min(limit, 1000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        order_dir = "DESC" if descending else "ASC"
        _allowed_order = {"created_at", "salience", "frequency"}
        if order_by not in _allowed_order:
            order_by = "created_at"
        query = f"SELECT * FROM fibers WHERE brain_id = ? ORDER BY {order_by} {order_dir} LIMIT ?"

        async with conn.execute(query, (brain_id, limit)) as cursor:
            rows = await cursor.fetchall()
            return [row_to_fiber(row) for row in rows]
