"""Dialect-agnostic fiber operations mixin.

Merges all methods from ``SQLiteFiberMixin`` and ``PostgresFiberMixin``
into a single ``FiberMixin`` that delegates every SQL detail to
``self._dialect`` (a ``Dialect`` instance injected at construction time).

Rules enforced:
  - ZERO bare ``?`` or ``$N`` -- every placeholder is ``d.ph(N)``
  - ZERO ``.isoformat()`` -- datetimes via ``d.serialize_dt(dt)``
  - ZERO ``fromisoformat`` -- datetimes via ``d.normalize_dt(val)``
  - ZERO ``conn.execute`` / ``conn.commit`` / ``cursor``
  - ``row_to_fiber(d, row)`` with dialect as the first argument everywhere
  - IN clauses via ``d.in_clause()``
  - FTS via ``d.fts_fiber_query()`` with ``d.supports_fts`` guard
  - Tag membership via ``d.json_array_contains()``
  - Metadata key check via ``d.json_contains_key()``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.sql.row_mappers import row_to_fiber
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FTS query builder (SQLite FTS5 syntax -- dialect decides when to use it)
# ---------------------------------------------------------------------------

def _build_fts_query(search_term: str) -> str:
    """Build an FTS5 MATCH expression from a user search string."""
    tokens = search_term.split()
    if not tokens:
        return '""'
    return " ".join(
        f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens
    )


# ---------------------------------------------------------------------------
# FiberMixin
# ---------------------------------------------------------------------------

class FiberMixin:
    """Dialect-agnostic fiber CRUD.

    Requires the host class to provide::

        self._dialect: Dialect
        self._get_brain_id() -> str
        self.invalidate_merkle_prefix(record_type, record_id, is_pro=False)
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ add
    # ------------------------------------------------------------------

    async def add_fiber(self, fiber: Fiber) -> str:
        d = self._dialect
        brain_id = self._get_brain_id()

        params: list[Any] = [
            fiber.id,                                   # 1
            brain_id,                                   # 2
            json.dumps(list(fiber.neuron_ids)),          # 3
            json.dumps(list(fiber.synapse_ids)),         # 4
            fiber.anchor_neuron_id,                      # 5
            json.dumps(fiber.pathway),                   # 6
            fiber.conductivity,                          # 7
            d.serialize_dt(fiber.last_conducted),        # 8
            d.serialize_dt(fiber.time_start),            # 9
            d.serialize_dt(fiber.time_end),              # 10
            fiber.coherence,                             # 11
            fiber.salience,                              # 12
            fiber.frequency,                             # 13
            fiber.summary,                               # 14
            fiber.essence,                               # 15
            d.serialize_dt(fiber.last_ghost_shown_at),   # 16
            json.dumps(list(fiber.tags)),                # 17
            json.dumps(list(fiber.auto_tags)),           # 18
            json.dumps(list(fiber.agent_tags)),          # 19
            json.dumps(fiber.metadata),                  # 20
            fiber.compression_tier,                      # 21
            1 if fiber.pinned else 0,                    # 22
            d.serialize_dt(fiber.created_at),            # 23
        ]

        await d.execute(
            f"""INSERT INTO fibers
               (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                pathway, conductivity, last_conducted,
                time_start, time_end, coherence, salience, frequency,
                summary, essence, last_ghost_shown_at,
                tags, auto_tags, agent_tags, metadata,
                compression_tier, pinned, created_at)
               VALUES ({d.phs(23, 1)})""",
            params,
        )

        # Populate junction table for fast lookups
        if fiber.neuron_ids:
            insert_sql = d.insert_or_ignore_sql(
                "fiber_neurons",
                ["brain_id", "fiber_id", "neuron_id"],
                ["brain_id", "fiber_id", "neuron_id"],
            )
            await d.execute_many(
                insert_sql,
                [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
            )

        # Invalidate Merkle hash cache for the affected bucket
        await self.invalidate_merkle_prefix("fiber", fiber.id, is_pro=True)  # type: ignore[attr-defined]
        return fiber.id

    # ------------------------------------------------------------------ get
    # ------------------------------------------------------------------

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM fibers WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (fiber_id, brain_id),
        )
        if row is None:
            return None
        return row_to_fiber(d, row)

    # ------------------------------------------------------------------ search
    # ------------------------------------------------------------------

    async def search_fiber_summaries(
        self, query: str, *, limit: int = 10
    ) -> list[Fiber]:
        """Search fiber summaries using full-text search.

        Returns fibers whose summary matches the query.
        Falls back to LIKE search when FTS is unavailable.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 50)

        # FTS path
        if d.supports_fts:
            try:
                fts_terms = _build_fts_query(query)
                from_clause, where_clause = d.fts_fiber_query(1, 2)
                sql = (
                    f"SELECT f.* FROM {from_clause} "
                    f"WHERE {where_clause} "
                    f"ORDER BY f.salience DESC LIMIT {d.ph(3)}"
                )
                rows = await d.fetch_all(sql, (fts_terms, brain_id, capped_limit))
                return [row_to_fiber(d, row) for row in rows]
            except NotImplementedError:
                logger.debug("FTS not supported, falling back to LIKE search")

        # Fallback: LIKE search on summary
        tokens = [t for t in query.split() if t]
        if not tokens:
            return []

        # Build params: brain_id first, then one LIKE term per token, then limit
        n = 1
        params: list[Any] = [brain_id]
        n += 1  # n=2

        where_parts: list[str] = []
        for token in tokens:
            params.append(f"%{token}%")
            where_parts.append(f"f.summary LIKE {d.ph(n)}")
            n += 1

        params.append(capped_limit)
        like_where = " AND ".join(where_parts)

        sql = (
            f"SELECT f.* FROM fibers f "
            f"WHERE f.brain_id = {d.ph(1)} "
            f"AND f.summary IS NOT NULL AND {like_where} "
            f"ORDER BY f.salience DESC LIMIT {d.ph(n)}"
        )
        rows = await d.fetch_all(sql, params)
        return [row_to_fiber(d, row) for row in rows]

    # ------------------------------------------------------------------ find
    # ------------------------------------------------------------------

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        d = self._dialect
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        n = 1
        params: list[Any] = [brain_id]
        where_parts: list[str] = [f"brain_id = {d.ph(n)}"]
        n += 1

        if contains_neuron is not None:
            params.append(contains_neuron)
            where_parts.append(
                f"id IN (SELECT fiber_id FROM fiber_neurons "
                f"WHERE brain_id = {d.ph(1)} AND neuron_id = {d.ph(n)})"
            )
            n += 1

        if time_overlaps is not None:
            start, end = time_overlaps
            params.append(d.serialize_dt(end))
            where_parts.append(f"(time_start IS NULL OR time_start <= {d.ph(n)})")
            n += 1
            params.append(d.serialize_dt(start))
            where_parts.append(f"(time_end IS NULL OR time_end >= {d.ph(n)})")
            n += 1

        if min_salience is not None:
            params.append(min_salience)
            where_parts.append(f"salience >= {d.ph(n)}")
            n += 1

        if metadata_key is not None:
            params.append(metadata_key)
            where_parts.append(d.json_contains_key("metadata", n))
            n += 1

        if tags is not None and tags:
            for tag in tags:
                params.append(tag)
                where_parts.append(d.json_array_contains("tags", n))
                n += 1

        # When tags filter is present, fetch more rows to compensate for
        # possible post-SQL filtering
        fetch_limit = min(limit * 3, 3000) if tags else limit
        params.append(fetch_limit)

        sql = (
            "SELECT * FROM fibers WHERE "
            + " AND ".join(where_parts)
            + f" ORDER BY salience DESC LIMIT {d.ph(n)}"
        )
        rows = await d.fetch_all(sql, params)
        fibers = [row_to_fiber(d, row) for row in rows]

        # Post-filter by tags for safety (JSON array set operations are imprecise)
        if tags is not None:
            fibers = [f for f in fibers if tags.issubset(f.tags)]

        return fibers[:limit]

    # ------------------------------------------------------------------ find batch
    # ------------------------------------------------------------------

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
        tags: set[str] | None = None,
    ) -> list[Fiber]:
        """Find fibers containing any of the given neurons in a single SQL query."""
        if not neuron_ids:
            return []

        d = self._dialect
        brain_id = self._get_brain_id()
        total_limit = limit_per_neuron * len(neuron_ids)

        # param 1 = brain_id
        n = 2  # next available param index
        params: list[Any] = [brain_id]

        # IN clause for neuron_ids
        in_sql, in_params = d.in_clause(n, neuron_ids)
        params.extend(in_params)
        # Advance n past the params consumed by in_clause
        n += len(in_params)

        # Tag filter: all tags must match (AND semantics)
        tag_parts: list[str] = []
        if tags:
            for tag in tags:
                params.append(tag)
                tag_parts.append(d.json_array_contains("f.tags", n))
                n += 1

        tag_sql = ""
        if tag_parts:
            tag_sql = " AND " + " AND ".join(tag_parts)

        params.append(total_limit)
        sql = (
            f"SELECT DISTINCT f.* FROM fibers f "
            f"JOIN fiber_neurons fn ON f.brain_id = fn.brain_id AND f.id = fn.fiber_id "
            f"WHERE fn.brain_id = {d.ph(1)} AND fn.neuron_id {in_sql}"
            f"{tag_sql}"
            f" ORDER BY f.salience DESC LIMIT {d.ph(n)}"
        )

        rows = await d.fetch_all(sql, params)
        return [row_to_fiber(d, row) for row in rows]

    # ------------------------------------------------------------------ update
    # ------------------------------------------------------------------

    async def update_fiber(self, fiber: Fiber) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        params: list[Any] = [
            json.dumps(list(fiber.neuron_ids)),          # 1
            json.dumps(list(fiber.synapse_ids)),         # 2
            fiber.anchor_neuron_id,                      # 3
            json.dumps(fiber.pathway),                   # 4
            fiber.conductivity,                          # 5
            d.serialize_dt(fiber.last_conducted),        # 6
            d.serialize_dt(fiber.time_start),            # 7
            d.serialize_dt(fiber.time_end),              # 8
            fiber.coherence,                             # 9
            fiber.salience,                              # 10
            fiber.frequency,                             # 11
            fiber.summary,                               # 12
            fiber.essence,                               # 13
            d.serialize_dt(fiber.last_ghost_shown_at),   # 14
            json.dumps(list(fiber.tags)),                # 15
            json.dumps(list(fiber.auto_tags)),           # 16
            json.dumps(list(fiber.agent_tags)),          # 17
            json.dumps(fiber.metadata),                  # 18
            fiber.compression_tier,                      # 19
            1 if fiber.pinned else 0,                    # 20
            fiber.id,                                    # 21
            brain_id,                                    # 22
        ]

        await d.execute(
            f"""UPDATE fibers SET
               neuron_ids = {d.ph(1)}, synapse_ids = {d.ph(2)},
               anchor_neuron_id = {d.ph(3)}, pathway = {d.ph(4)},
               conductivity = {d.ph(5)}, last_conducted = {d.ph(6)},
               time_start = {d.ph(7)}, time_end = {d.ph(8)},
               coherence = {d.ph(9)}, salience = {d.ph(10)},
               frequency = {d.ph(11)}, summary = {d.ph(12)},
               essence = {d.ph(13)}, last_ghost_shown_at = {d.ph(14)},
               tags = {d.ph(15)}, auto_tags = {d.ph(16)},
               agent_tags = {d.ph(17)},
               metadata = {d.ph(18)}, compression_tier = {d.ph(19)},
               pinned = {d.ph(20)}
               WHERE id = {d.ph(21)} AND brain_id = {d.ph(22)}""",
            params,
        )

        # Refresh junction table
        await d.execute(
            f"DELETE FROM fiber_neurons WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}",
            (brain_id, fiber.id),
        )
        if fiber.neuron_ids:
            insert_sql = d.insert_or_ignore_sql(
                "fiber_neurons",
                ["brain_id", "fiber_id", "neuron_id"],
                ["brain_id", "fiber_id", "neuron_id"],
            )
            await d.execute_many(
                insert_sql,
                [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
            )

    async def update_fiber_metadata(
        self, fiber_id: str, metadata: dict[str, Any]
    ) -> None:
        """Update only the metadata JSON column for a fiber (lightweight patch).

        Fetches the current fiber, merges the provided metadata on top, then
        persists via update_fiber.  No-op if the fiber does not exist.
        """
        fiber = await self.get_fiber(fiber_id)
        if fiber is None:
            return

        import dataclasses

        updated_meta = {**fiber.metadata, **metadata}
        updated_fiber = dataclasses.replace(fiber, metadata=updated_meta)
        await self.update_fiber(updated_fiber)

    # ------------------------------------------------------------------ delete
    # ------------------------------------------------------------------

    async def delete_fiber(self, fiber_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()

        # Delete junction entries first
        await d.execute(
            f"DELETE FROM fiber_neurons WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}",
            (brain_id, fiber_id),
        )

        count = await d.execute_count(
            f"DELETE FROM fibers WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (fiber_id, brain_id),
        )
        if count > 0:
            await self.invalidate_merkle_prefix("fiber", fiber_id, is_pro=True)  # type: ignore[attr-defined]
        return count > 0

    # ------------------------------------------------------------------ pinned
    # ------------------------------------------------------------------

    async def get_pinned_neuron_ids(self) -> set[str]:
        """Get all neuron IDs that belong to pinned fibers.

        Used by lifecycle systems (decay, prune) to skip pinned neurons.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT neuron_ids FROM fibers WHERE brain_id = {d.ph(1)} AND pinned = 1",
            (brain_id,),
        )

        result: set[str] = set()
        for row in rows:
            neuron_ids_raw = row.get("neuron_ids") if isinstance(row, dict) else row[0]
            if neuron_ids_raw:
                parsed = (
                    json.loads(neuron_ids_raw)
                    if isinstance(neuron_ids_raw, str)
                    else neuron_ids_raw
                )
                result.update(parsed)
        return result

    async def list_pinned_fibers(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all pinned fibers for the current brain.

        Includes memory type and priority from the typed_memories table
        where available.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)

        try:
            rows = await d.fetch_all(
                f"""SELECT f.id, f.summary, f.tags, f.created_at,
                          tm.memory_type, tm.priority
                   FROM fibers f
                   LEFT JOIN typed_memories tm
                     ON tm.fiber_id = f.id AND tm.brain_id = f.brain_id
                   WHERE f.brain_id = {d.ph(1)} AND f.pinned = 1
                   ORDER BY f.created_at DESC LIMIT {d.ph(2)}""",
                (brain_id, safe_limit),
            )
        except Exception:
            # Fallback if typed_memories table is not available
            rows = await d.fetch_all(
                f"SELECT id, summary, summary AS type, 5 AS priority, tags, created_at "
                f"FROM fibers WHERE brain_id = {d.ph(1)} AND pinned = 1 "
                f"ORDER BY created_at DESC LIMIT {d.ph(2)}",
                (brain_id, safe_limit),
            )

        results: list[dict[str, Any]] = []
        for row in rows:
            d_row = dict(row)
            tags_raw = d_row.get("tags")
            tags = (
                json.loads(tags_raw)
                if isinstance(tags_raw, str)
                else (tags_raw or [])
            )
            results.append(
                {
                    "fiber_id": str(d_row.get("id")),
                    "summary": d_row.get("summary") or "",
                    "type": d_row.get("memory_type") or "unknown",
                    "priority": d_row.get("priority") or 5,
                    "tags": tags,
                    "created_at": str(d_row.get("created_at") or ""),
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

        d = self._dialect
        brain_id = self._get_brain_id()
        pin_val = 1 if pinned else 0

        # Build IN clause for fiber_ids starting at param index 3
        in_sql, in_params = d.in_clause(3, fiber_ids)
        params: list[Any] = [pin_val, brain_id, *in_params]

        count = await d.execute_count(
            f"UPDATE fibers SET pinned = {d.ph(1)} "
            f"WHERE brain_id = {d.ph(2)} AND id {in_sql}",
            params,
        )
        return count

    # ------------------------------------------------------------------ stale / stage / count
    # ------------------------------------------------------------------

    async def get_stale_fiber_count(
        self, brain_id: str, stale_days: int = 90
    ) -> int:
        d = self._dialect
        cutoff = d.serialize_dt(utcnow() - timedelta(days=stale_days))

        row = await d.fetch_one(
            f"""SELECT COUNT(*) AS cnt FROM fibers
               WHERE brain_id = {d.ph(1)}
                 AND (
                   (last_conducted IS NULL AND created_at <= {d.ph(2)})
                   OR (last_conducted IS NOT NULL AND last_conducted <= {d.ph(3)})
                 )""",
            (brain_id, cutoff, cutoff),
        )
        return int(row["cnt"]) if row else 0

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        d = self._dialect
        rows = await d.fetch_all(
            f"SELECT stage, COUNT(*) AS cnt FROM fibers "
            f"WHERE brain_id = {d.ph(1)} GROUP BY stage",
            (brain_id,),
        )
        return {row["stage"]: int(row["cnt"]) for row in rows}

    async def get_total_fiber_count(self) -> int:
        """Get total number of fibers for the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()
        row = await d.fetch_one(
            f"SELECT COUNT(*) AS cnt FROM fibers WHERE brain_id = {d.ph(1)}",
            (brain_id,),
        )
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------ ghost / keyword df
    # ------------------------------------------------------------------

    async def batch_update_ghost_shown(
        self, fiber_ids: list[str], timestamp: datetime
    ) -> int:
        """Batch update last_ghost_shown_at for multiple fibers in one query."""
        if not fiber_ids:
            return 0
        d = self._dialect
        brain_id = self._get_brain_id()

        # param 1 = timestamp, param 2 = brain_id, then IN clause from 3
        in_sql, in_params = d.in_clause(3, fiber_ids)
        params: list[Any] = [d.serialize_dt(timestamp), brain_id, *in_params]

        count = await d.execute_count(
            f"UPDATE fibers SET last_ghost_shown_at = {d.ph(1)} "
            f"WHERE brain_id = {d.ph(2)} AND id {in_sql}",
            params,
        )
        return count

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        """Get document frequency for a batch of keywords."""
        if not keywords:
            return {}
        d = self._dialect
        brain_id = self._get_brain_id()

        # param 1 = brain_id, IN clause from 2
        in_sql, in_params = d.in_clause(2, keywords)
        params: list[Any] = [brain_id, *in_params]

        rows = await d.fetch_all(
            f"SELECT keyword, fiber_count "
            f"FROM keyword_document_frequency "
            f"WHERE brain_id = {d.ph(1)} AND keyword {in_sql}",
            params,
        )
        return {row["keyword"]: int(row["fiber_count"]) for row in rows}

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        """Increment document frequency for each keyword (UPSERT)."""
        if not keywords:
            return
        d = self._dialect
        brain_id = self._get_brain_id()
        now_ser = d.serialize_dt(utcnow())

        unique_keywords = set(keywords)
        upsert_sql = d.upsert_sql(
            "keyword_document_frequency",
            ["brain_id", "keyword", "fiber_count", "last_updated"],
            ["brain_id", "keyword"],
            ["fiber_count", "last_updated"],
        )
        await d.execute_many(
            upsert_sql,
            [(brain_id, kw, 1, now_ser) for kw in unique_keywords],
        )

    # ------------------------------------------------------------------ get_fibers
    # ------------------------------------------------------------------

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        """Get fibers with ordering."""
        d = self._dialect
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        _allowed_order = {"created_at", "salience", "frequency"}
        if order_by not in _allowed_order:
            order_by = "created_at"
        order_dir = "DESC" if descending else "ASC"

        sql = (
            f"SELECT * FROM fibers WHERE brain_id = {d.ph(1)} "
            f"ORDER BY {order_by} {order_dir} LIMIT {d.ph(2)}"
        )
        rows = await d.fetch_all(sql, (brain_id, limit))
        return [row_to_fiber(d, row) for row in rows]
