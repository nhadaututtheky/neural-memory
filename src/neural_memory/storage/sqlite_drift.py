"""SQLite mixin for semantic drift detection persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteDriftMixin:
    """Mixin providing CRUD for tag_cooccurrence and drift_clusters tables."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tag co-occurrence
    # ------------------------------------------------------------------

    async def record_tag_cooccurrence(self, tags: set[str]) -> None:
        """Record tag co-occurrence pairs from a single fiber.

        For each pair (a, b) where a < b (canonical order), upsert
        the count and last_seen timestamp.
        """
        if len(tags) < 2:
            return

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        sorted_tags = sorted(tags)
        pairs: list[tuple[str, str, str, str]] = []
        for i in range(len(sorted_tags)):
            for j in range(i + 1, len(sorted_tags)):
                pairs.append((brain_id, sorted_tags[i], sorted_tags[j], now))

        # Cap pair generation to avoid O(n^2) explosion on large tag sets
        pairs = pairs[:100]

        for brain_id_val, tag_a, tag_b, ts in pairs:
            await conn.execute(
                """INSERT INTO tag_cooccurrence (brain_id, tag_a, tag_b, count, last_seen)
                   VALUES (?, ?, ?, 1, ?)
                   ON CONFLICT (brain_id, tag_a, tag_b)
                   DO UPDATE SET count = count + 1, last_seen = ?""",
                (brain_id_val, tag_a, tag_b, ts, ts),
            )
        await conn.commit()

    async def get_tag_cooccurrence(
        self,
        min_count: int = 2,
        limit: int = 500,
    ) -> list[tuple[str, str, int]]:
        """Get tag co-occurrence pairs above threshold.

        Returns list of (tag_a, tag_b, count) sorted by count descending.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 2000)

        cursor = await conn.execute(
            """SELECT tag_a, tag_b, count
               FROM tag_cooccurrence
               WHERE brain_id = ? AND count >= ?
               ORDER BY count DESC
               LIMIT ?""",
            (brain_id, min_count, capped_limit),
        )
        return [(row[0], row[1], row[2]) for row in await cursor.fetchall()]

    async def get_tag_fiber_counts(self) -> dict[str, int]:
        """Get fiber count per tag for Jaccard calculation.

        Returns dict of {tag: fiber_count}.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        # Count fibers per tag via auto_tags + agent_tags JSON arrays
        cursor = await conn.execute(
            """SELECT DISTINCT f.id, f.auto_tags, f.agent_tags
               FROM fibers f
               WHERE f.brain_id = ?
               LIMIT 10000""",
            (brain_id,),
        )
        rows = await cursor.fetchall()

        tag_counts: dict[str, int] = {}
        for _fid, auto_tags_json, agent_tags_json in rows:
            try:
                auto = json.loads(auto_tags_json) if auto_tags_json else []
                agent = json.loads(agent_tags_json) if agent_tags_json else []
                all_tags = set(auto) | set(agent)
                for tag in all_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        return tag_counts

    # ------------------------------------------------------------------
    # Drift clusters
    # ------------------------------------------------------------------

    async def save_drift_cluster(
        self,
        cluster_id: str,
        canonical: str,
        members: list[str],
        confidence: float,
        status: str = "detected",
    ) -> None:
        """Upsert a drift cluster detection result."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        await conn.execute(
            """INSERT INTO drift_clusters
               (id, brain_id, canonical, members, confidence, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (brain_id, id)
               DO UPDATE SET canonical = ?, members = ?, confidence = ?,
                             status = ?, resolved_at = NULL""",
            (
                cluster_id,
                brain_id,
                canonical,
                json.dumps(members),
                confidence,
                status,
                now,
                canonical,
                json.dumps(members),
                confidence,
                status,
            ),
        )
        await conn.commit()

    async def get_drift_clusters(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Get drift clusters, optionally filtered by status."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped = min(limit, 200)

        if status:
            cursor = await conn.execute(
                """SELECT id, canonical, members, confidence, status, created_at, resolved_at
                   FROM drift_clusters
                   WHERE brain_id = ? AND status = ?
                   ORDER BY confidence DESC
                   LIMIT ?""",
                (brain_id, status, capped),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, canonical, members, confidence, status, created_at, resolved_at
                   FROM drift_clusters
                   WHERE brain_id = ?
                   ORDER BY confidence DESC
                   LIMIT ?""",
                (brain_id, capped),
            )

        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "canonical": r[1],
                "members": json.loads(r[2]) if r[2] else [],
                "confidence": r[3],
                "status": r[4],
                "created_at": r[5],
                "resolved_at": r[6],
            }
            for r in rows
        ]

    async def resolve_drift_cluster(
        self,
        cluster_id: str,
        status: str,
    ) -> bool:
        """Update drift cluster status (merged/aliased/dismissed)."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """UPDATE drift_clusters
               SET status = ?, resolved_at = ?
               WHERE brain_id = ? AND id = ?""",
            (status, now, brain_id, cluster_id),
        )
        await conn.commit()
        return cursor.rowcount > 0
