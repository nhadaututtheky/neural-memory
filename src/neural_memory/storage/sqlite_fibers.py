"""SQLite fiber operations mixin."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.sqlite_row_mappers import row_to_fiber
from neural_memory.utils.timeutils import utcnow


def _build_fts_query(search_term: str) -> str:
    """Build an FTS5 MATCH expression from a user search string."""
    tokens = search_term.split()
    if not tokens:
        return '""'
    return " ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens)


if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


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
                    summary, essence, last_ghost_shown_at,
                    tags, auto_tags, agent_tags, metadata,
                    compression_tier, pinned, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    fiber.essence,
                    fiber.last_ghost_shown_at.isoformat() if fiber.last_ghost_shown_at else None,
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
            # Invalidate Merkle hash cache for the affected bucket
            await self.invalidate_merkle_prefix("fiber", fiber.id, is_pro=True)  # type: ignore[attr-defined]
            return fiber.id
        except sqlite3.IntegrityError as exc:
            msg = str(exc).lower()
            if "foreign key" in msg:
                raise ValueError(
                    f"Fiber {fiber.id} references non-existent neurons or brain"
                ) from exc
            raise ValueError(f"Fiber {fiber.id} already exists") from exc

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

    async def search_fiber_summaries(self, query: str, *, limit: int = 10) -> list[Fiber]:
        """Search fiber summaries using FTS5 full-text search.

        Returns fibers whose summary matches the query, ranked by BM25.
        Falls back to LIKE search if FTS5 is not available.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 50)

        # Try FTS5 path first
        try:
            fts_terms = _build_fts_query(query)
            sql = (
                "SELECT f.* FROM fibers f "
                "JOIN fibers_fts fts ON f.rowid = fts.rowid "
                "WHERE fts.fibers_fts MATCH ? AND fts.brain_id = ? "
                "ORDER BY fts.rank LIMIT ?"
            )
            async with conn.execute(sql, (fts_terms, brain_id, capped_limit)) as cursor:
                rows = await cursor.fetchall()
                return [row_to_fiber(row) for row in rows]
        except Exception:
            logger.debug("Fiber FTS5 unavailable, falling back to LIKE search")

        # Fallback: LIKE search on summary
        like_terms = [f"%{token}%" for token in query.split() if token]
        if not like_terms:
            return []

        where_clauses = " AND ".join("f.summary LIKE ?" for _ in like_terms)
        sql = (
            f"SELECT f.* FROM fibers f "
            f"WHERE f.brain_id = ? AND f.summary IS NOT NULL AND {where_clauses} "
            f"ORDER BY f.salience DESC LIMIT ?"
        )
        params: list[Any] = [brain_id, *like_terms, capped_limit]
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [row_to_fiber(row) for row in rows]

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
        tag_mode: str = "and",
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

        if metadata_key is not None:
            # Use double-quoted member syntax to treat dots as literal characters
            query += " AND json_extract(metadata, ?) IS NOT NULL"
            params.append(f'$."{metadata_key}"')

        # When tags filter is needed, fetch more rows to compensate for
        # post-SQL filtering (tags are stored as JSON arrays)
        fetch_limit = min(limit * 3, 3000) if tags else limit
        query += " ORDER BY salience DESC LIMIT ?"
        params.append(fetch_limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            fibers = [row_to_fiber(row) for row in rows]

        # Filter by tags in Python (JSON array doesn't support efficient set operations)
        if tags is not None:
            if tag_mode == "or":
                fibers = [f for f in fibers if tags & f.tags]
            else:
                fibers = [f for f in fibers if tags.issubset(f.tags)]

        return fibers[:limit]

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
        tags: set[str] | None = None,
        tag_mode: str = "and",
        created_before: datetime | None = None,
    ) -> list[Fiber]:
        """Find fibers containing any of the given neurons in a single SQL query."""
        if not neuron_ids:
            return []

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in neuron_ids)
        # Use junction table for efficient lookup, limit total results
        total_limit = limit_per_neuron * len(neuron_ids)
        sql = (
            f"SELECT DISTINCT f.* FROM fibers f"
            f" JOIN fiber_neurons fn ON f.brain_id = fn.brain_id AND f.id = fn.fiber_id"
            f" WHERE fn.brain_id = ? AND fn.neuron_id IN ({placeholders})"
        )
        params: list[Any] = [brain_id, *neuron_ids]

        # Tag filter: f.tags column stores the union of auto_tags + agent_tags
        if tags:
            tag_clauses = ["EXISTS (SELECT 1 FROM json_each(f.tags) WHERE value = ?)" for _ in tags]
            joiner = " OR " if tag_mode == "or" else " AND "
            sql += f" AND ({joiner.join(tag_clauses)})"
            params.extend(tags)

        if created_before is not None:
            sql += " AND f.created_at <= ?"
            params.append(created_before.isoformat())

        sql += " ORDER BY f.salience DESC LIMIT ?"
        params.append(total_limit)

        async with conn.execute(sql, params) as cursor:
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
               summary = ?, essence = ?, last_ghost_shown_at = ?,
               tags = ?, auto_tags = ?, agent_tags = ?,
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
                fiber.essence,
                fiber.last_ghost_shown_at.isoformat() if fiber.last_ghost_shown_at else None,
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
            # Fiber was deleted (e.g. by consolidation prune) between
            # deferred queue enqueue and flush — skip gracefully.
            logger.debug("Skipping update for deleted fiber %s", fiber.id)
            return

        # Refresh junction table
        try:
            await conn.execute(
                "DELETE FROM fiber_neurons WHERE brain_id = ? AND fiber_id = ?",
                (brain_id, fiber.id),
            )
            if fiber.neuron_ids:
                await conn.executemany(
                    "INSERT OR IGNORE INTO fiber_neurons (brain_id, fiber_id, neuron_id) VALUES (?, ?, ?)",
                    [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
                )
        except sqlite3.IntegrityError:
            logger.debug(
                "FK constraint on fiber_neurons for fiber %s — fiber or neuron deleted concurrently",
                fiber.id,
            )
            return

        await conn.commit()

    async def update_fiber_metadata(self, fiber_id: str, metadata: dict[str, Any]) -> None:
        """Update only the metadata JSON column for a fiber (lightweight patch).

        Fetches the current fiber, merges the provided metadata on top, then
        persists via update_fiber.  No-op if the fiber does not exist.

        Args:
            fiber_id: The fiber to update.
            metadata: New metadata values to merge (existing keys are overwritten).
        """
        fiber = await self.get_fiber(fiber_id)
        if fiber is None:
            return

        import dataclasses

        updated_meta = {**fiber.metadata, **metadata}
        updated_fiber = dataclasses.replace(fiber, metadata=updated_meta)
        await self.update_fiber(updated_fiber)

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
        if cursor.rowcount > 0:
            await self.invalidate_merkle_prefix("fiber", fiber_id, is_pro=True)  # type: ignore[attr-defined]

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

    async def list_pinned_fibers(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all pinned fibers for the current brain.

        Returns:
            List of dicts with fiber details.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)

        async with conn.execute(
            "SELECT id, summary, type, priority, tags, created_at "
            "FROM fibers WHERE brain_id = ? AND pinned = 1 "
            "ORDER BY created_at DESC LIMIT ?",
            (brain_id, safe_limit),
        ) as cursor:
            rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "fiber_id": row[0],
                    "summary": row[1] or "",
                    "type": row[2] or "unknown",
                    "priority": row[3] or 5,
                    "tags": json.loads(row[4]) if row[4] else [],
                    "created_at": row[5] or "",
                }
            )
        return results

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
            f"UPDATE fibers SET pinned = ? WHERE brain_id = ? AND id IN ({placeholders})",
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

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        conn = self._ensure_read_conn()
        async with conn.execute(
            "SELECT stage, COUNT(*) FROM fibers WHERE brain_id = ? GROUP BY stage",
            (brain_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def get_total_fiber_count(self) -> int:
        """Get total number of fibers for the current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        async with conn.execute(
            "SELECT COUNT(*) FROM fibers WHERE brain_id = ?",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def batch_update_ghost_shown(self, fiber_ids: list[str], timestamp: datetime) -> int:
        """Batch update last_ghost_shown_at for multiple fibers in one query."""
        if not fiber_ids:
            return 0
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        placeholders = ",".join("?" for _ in fiber_ids)
        await conn.execute(
            f"UPDATE fibers SET last_ghost_shown_at = ? WHERE brain_id = ? AND id IN ({placeholders})",
            [timestamp.isoformat(), brain_id, *fiber_ids],
        )
        await conn.commit()
        return len(fiber_ids)

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        """Get document frequency for a batch of keywords."""
        if not keywords:
            return {}
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        placeholders = ",".join("?" for _ in keywords)
        async with conn.execute(
            f"SELECT keyword, fiber_count FROM keyword_document_frequency"
            f" WHERE brain_id = ? AND keyword IN ({placeholders})",
            [brain_id, *keywords],
        ) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        """Increment document frequency for each keyword (UPSERT)."""
        if not keywords:
            return
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now_iso = utcnow().isoformat()
        # Deduplicate keywords (one fiber = one count per keyword)
        unique_keywords = set(keywords)
        await conn.executemany(
            "INSERT INTO keyword_document_frequency (brain_id, keyword, fiber_count, last_updated)"
            " VALUES (?, ?, 1, ?)"
            " ON CONFLICT(brain_id, keyword) DO UPDATE SET"
            " fiber_count = fiber_count + 1, last_updated = ?",
            [(brain_id, kw, now_iso, now_iso) for kw in unique_keywords],
        )
        await conn.commit()

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
