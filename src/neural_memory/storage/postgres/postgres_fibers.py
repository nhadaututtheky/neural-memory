"""PostgreSQL fiber operations."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import row_to_fiber
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class PostgresFiberMixin(PostgresBaseMixin):
    """PostgreSQL fiber CRUD."""

    async def add_fiber(self, fiber: Fiber) -> str:
        brain_id = self._get_brain_id()
        try:
            await self._query(
                """INSERT INTO fibers
                   (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                    pathway, conductivity, last_conducted, time_start, time_end,
                    coherence, salience, frequency, summary, tags, auto_tags,
                    agent_tags, metadata, compression_tier, pinned, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                           $15, $16, $17, $18, $19, $20, $21)""",
                fiber.id,
                brain_id,
                json.dumps(list(fiber.neuron_ids)),
                json.dumps(list(fiber.synapse_ids)),
                fiber.anchor_neuron_id,
                json.dumps(fiber.pathway),
                fiber.conductivity,
                fiber.last_conducted,
                fiber.time_start,
                fiber.time_end,
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
                fiber.created_at,
            )
            if fiber.neuron_ids:
                args_list = [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids]
                await self._executemany(
                    """INSERT INTO fiber_neurons (brain_id, fiber_id, neuron_id)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (brain_id, fiber_id, neuron_id) DO NOTHING""",
                    args_list,
                )
            return fiber.id
        except Exception as e:
            from asyncpg.exceptions import UniqueViolationError

            if isinstance(e, UniqueViolationError):
                raise ValueError(f"Fiber {fiber.id} already exists") from e
            raise

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM fibers WHERE brain_id = $1 AND id = $2",
            brain_id,
            fiber_id,
        )
        if row is None:
            return None
        return row_to_fiber(row)

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
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        query = "SELECT * FROM fibers WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if contains_neuron is not None:
            params.append(contains_neuron)
            query += f" AND id IN (SELECT fiber_id FROM fiber_neurons WHERE brain_id = $1 AND neuron_id = ${len(params)})"

        if time_overlaps is not None:
            start, end = time_overlaps
            idx = len(params) + 1
            query += f" AND (time_start IS NULL OR time_start <= ${idx + 1})"
            query += f" AND (time_end IS NULL OR time_end >= ${idx})"
            params.extend([end, start])

        if min_salience is not None:
            params.append(min_salience)
            query += f" AND salience >= ${len(params)}"

        if metadata_key is not None:
            params.append(metadata_key)
            query += f" AND metadata ? ${len(params)}"

        if tags is not None and tags:
            if tag_mode == "or":
                # ?| checks if ANY key in the array exists in the jsonb
                # asyncpg auto-coerces Python list to PostgreSQL text[]
                params.append(list(tags))
                query += f" AND tags ?| ${len(params)}::text[]"
            else:
                # @> checks containment (all must match)
                params.append(json.dumps(list(tags)))
                query += f" AND tags @> ${len(params)}::jsonb"

        params.append(limit)
        query += f" ORDER BY salience DESC LIMIT ${len(params)}"

        rows = await self._query_ro(query, *params)
        return [row_to_fiber(r) for r in rows]

    async def update_fiber(self, fiber: Fiber) -> None:
        brain_id = self._get_brain_id()
        r = await self._query(
            """UPDATE fibers SET
               neuron_ids = $1, synapse_ids = $2, anchor_neuron_id = $3,
               pathway = $4, conductivity = $5, last_conducted = $6,
               time_start = $7, time_end = $8, coherence = $9, salience = $10,
               frequency = $11, summary = $12, tags = $13, auto_tags = $14,
               agent_tags = $15, metadata = $16, compression_tier = $17,
               pinned = $18
               WHERE brain_id = $19 AND id = $20""",
            json.dumps(list(fiber.neuron_ids)),
            json.dumps(list(fiber.synapse_ids)),
            fiber.anchor_neuron_id,
            json.dumps(fiber.pathway),
            fiber.conductivity,
            fiber.last_conducted,
            fiber.time_start,
            fiber.time_end,
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
            brain_id,
            fiber.id,
        )
        if r == "UPDATE 0":
            raise ValueError(f"Fiber {fiber.id} does not exist")

    async def delete_fiber(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()
        r = await self._query(
            "DELETE FROM fibers WHERE brain_id = $1 AND id = $2",
            brain_id,
            fiber_id,
        )
        return bool(r != "DELETE 0")

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        """Get fibers with ordering."""
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        _allowed_order = {"created_at", "salience", "frequency"}
        if order_by not in _allowed_order:
            order_by = "created_at"
        order_dir = "DESC" if descending else "ASC"
        query = f"SELECT * FROM fibers WHERE brain_id = $1 ORDER BY {order_by} {order_dir} LIMIT $2"
        rows = await self._query_ro(query, brain_id, limit)
        return [row_to_fiber(r) for r in rows]

    # ──────────────────── Pinning ────────────────────

    async def pin_fibers(self, fiber_ids: list[str], pinned: bool = True) -> int:
        """Pin or unpin fibers by ID."""
        if not fiber_ids:
            return 0
        brain_id = self._get_brain_id()
        pin_val = 1 if pinned else 0
        result = await self._query(
            "UPDATE fibers SET pinned = $1 WHERE brain_id = $2 AND id = ANY($3::text[])",
            pin_val,
            brain_id,
            fiber_ids,
        )
        # asyncpg returns "UPDATE N"
        if result and "UPDATE" in str(result):
            try:
                return int(str(result).split()[-1])
            except (ValueError, IndexError):
                pass
        return 0

    async def get_pinned_neuron_ids(self) -> set[str]:
        """Get all neuron IDs that belong to pinned fibers."""
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            "SELECT neuron_ids FROM fibers WHERE brain_id = $1 AND pinned = 1",
            brain_id,
        )
        result: set[str] = set()
        for row in rows:
            neuron_ids_raw = row["neuron_ids"]
            if neuron_ids_raw:
                parsed = (
                    json.loads(neuron_ids_raw)
                    if isinstance(neuron_ids_raw, str)
                    else neuron_ids_raw
                )
                result.update(parsed)
        return result

    async def list_pinned_fibers(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all pinned fibers for the current brain."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)
        rows = await self._query_ro(
            """SELECT f.id, f.summary, f.tags, f.created_at,
                      tm.memory_type, tm.priority
               FROM fibers f
               LEFT JOIN typed_memories tm ON tm.fiber_id = f.id AND tm.brain_id = f.brain_id
               WHERE f.brain_id = $1 AND f.pinned = 1
               ORDER BY f.created_at DESC LIMIT $2""",
            brain_id,
            safe_limit,
        )
        results: list[dict[str, Any]] = []
        for r in rows:
            tags_raw = r["tags"]
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
            results.append(
                {
                    "fiber_id": str(r["id"]),
                    "summary": r["summary"] or "",
                    "type": r["memory_type"] or "unknown",
                    "priority": r["priority"] or 5,
                    "tags": tags,
                    "created_at": str(r["created_at"] or ""),
                }
            )
        return results

    # ──────────────────── Batch & Search (Phase 3 parity) ────────────────────

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
        tags: set[str] | None = None,
        tag_mode: str = "and",
        created_before: datetime | None = None,
    ) -> list[Fiber]:
        """Find fibers connected to any of the given neuron IDs.

        Uses a window function to enforce per-neuron limit, then deduplicates.
        """
        if not neuron_ids:
            return []
        brain_id = self._get_brain_id()
        total_limit = limit_per_neuron * len(neuron_ids)

        # Build base query with ROW_NUMBER for per-neuron limit
        query = """
            SELECT DISTINCT ON (f.id) f.*
            FROM (
                SELECT f2.*, fn.neuron_id AS _matched_neuron,
                       ROW_NUMBER() OVER (
                           PARTITION BY fn.neuron_id ORDER BY f2.salience DESC
                       ) AS _rn
                FROM fibers f2
                JOIN fiber_neurons fn
                    ON f2.brain_id = fn.brain_id AND f2.id = fn.fiber_id
                WHERE fn.brain_id = $1 AND fn.neuron_id = ANY($2::text[])
            ) f
            WHERE f._rn <= $3
        """
        params: list[Any] = [brain_id, neuron_ids, limit_per_neuron]

        if created_before is not None:
            params.append(created_before)
            query += f" AND f.created_at <= ${len(params)}"

        if tags:
            if tag_mode == "or":
                params.append(list(tags))
                query += f" AND f.tags ?| ${len(params)}::text[]"
            else:
                params.append(json.dumps(list(tags)))
                query += f" AND f.tags @> ${len(params)}::jsonb"

        params.append(total_limit)
        query += f" ORDER BY f.id, f.salience DESC LIMIT ${len(params)}"

        rows = await self._query_ro(query, *params)
        return [row_to_fiber(r) for r in rows]

    async def search_fiber_summaries(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Fiber]:
        """Full-text search on fiber summaries using PostgreSQL tsvector.

        Falls back to ILIKE if tsvector column is not populated.
        """
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 50)

        # Build tsquery from search terms
        tokens = query.strip().split()
        if not tokens:
            return []

        # Try tsvector first
        try:
            ts_query = " & ".join(t for t in tokens if t)
            rows = await self._query_ro(
                """SELECT f.* FROM fibers f
                   WHERE f.brain_id = $1
                     AND f.summary_tsv @@ to_tsquery('english', $2)
                   ORDER BY ts_rank(f.summary_tsv, to_tsquery('english', $2)) DESC,
                            f.salience DESC
                   LIMIT $3""",
                brain_id,
                ts_query,
                capped_limit,
            )
            return [row_to_fiber(r) for r in rows]
        except Exception:
            logger.debug("FTS failed, falling back to ILIKE", exc_info=True)

        # ILIKE fallback
        like_conditions = []
        params: list[Any] = [brain_id]
        for token in tokens:
            params.append(f"%{token}%")
            like_conditions.append(f"f.summary ILIKE ${len(params)}")
        where_clause = " AND ".join(like_conditions)
        params.append(capped_limit)
        sql = f"""SELECT f.* FROM fibers f
                  WHERE f.brain_id = $1
                    AND f.summary IS NOT NULL
                    AND {where_clause}
                  ORDER BY f.salience DESC LIMIT ${len(params)}"""
        rows = await self._query_ro(sql, *params)
        return [row_to_fiber(r) for r in rows]

    async def update_fiber_metadata(
        self,
        fiber_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Merge new metadata into existing fiber metadata (JSONB ||)."""
        self._get_brain_id()  # Validate brain context
        # Fetch, merge in Python, write back (safe for complex nested merges)
        fiber = await self.get_fiber(fiber_id)
        if fiber is None:
            return
        updated_meta = {**fiber.metadata, **metadata}
        updated_fiber = dataclasses.replace(fiber, metadata=updated_meta)
        await self.update_fiber(updated_fiber)

    # ──────────────────── Stats Methods ────────────────────

    async def get_stale_fiber_count(self, brain_id: str, stale_days: int = 90) -> int:
        """Count fibers not accessed within the given number of days."""
        brain_id = self._get_brain_id()
        cutoff = utcnow() - timedelta(days=stale_days)
        row = await self._query_one(
            """SELECT COUNT(*) AS cnt FROM fibers
               WHERE brain_id = $1
                 AND (
                   (last_conducted IS NULL AND created_at <= $2)
                   OR (last_conducted IS NOT NULL AND last_conducted <= $2)
                 )""",
            brain_id,
            cutoff,
        )
        return int(row["cnt"]) if row else 0

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        """Count fibers by maturation stage (from metadata _stage)."""
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            """SELECT COALESCE(metadata->>'_stage', 'episodic') AS stage,
                      COUNT(*) AS cnt
               FROM fibers
               WHERE brain_id = $1
               GROUP BY stage""",
            brain_id,
        )
        return {str(r["stage"]): int(r["cnt"]) for r in rows}

    async def get_total_fiber_count(self) -> int:
        """Count total fibers in the current brain."""
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT COUNT(*) AS cnt FROM fibers WHERE brain_id = $1",
            brain_id,
        )
        return int(row["cnt"]) if row else 0

    async def batch_update_ghost_shown(self, fiber_ids: list[str], timestamp: datetime) -> int:
        """Update last_ghost_shown_at for the given fiber IDs."""
        if not fiber_ids:
            return 0
        brain_id = self._get_brain_id()
        result = await self._query(
            """UPDATE fibers SET last_ghost_shown_at = $1
               WHERE brain_id = $2 AND id = ANY($3::text[])""",
            timestamp,
            brain_id,
            fiber_ids,
        )
        # asyncpg returns status string like "UPDATE N"
        if isinstance(result, str) and result.startswith("UPDATE"):
            parts = result.split()
            return int(parts[1]) if len(parts) > 1 else 0
        return len(fiber_ids)

    # ──────────────────── Keyword Document Frequency ────────────────────

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        """Get document frequency for a batch of keywords."""
        if not keywords:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            """SELECT keyword, fiber_count
               FROM keyword_document_frequency
               WHERE brain_id = $1 AND keyword = ANY($2::text[])""",
            brain_id,
            keywords,
        )
        return {str(r["keyword"]): int(r["fiber_count"]) for r in rows}

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        """Increment document frequency for keywords (upsert)."""
        if not keywords:
            return
        brain_id = self._get_brain_id()
        now = utcnow()
        unique_keywords = set(keywords)
        args_list = [(brain_id, kw, 1, now) for kw in unique_keywords]
        await self._executemany(
            """INSERT INTO keyword_document_frequency
                   (brain_id, keyword, fiber_count, last_updated)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (brain_id, keyword)
               DO UPDATE SET fiber_count = keyword_document_frequency.fiber_count + 1,
                             last_updated = EXCLUDED.last_updated""",
            args_list,
        )
