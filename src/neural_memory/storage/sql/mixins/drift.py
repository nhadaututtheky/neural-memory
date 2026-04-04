"""Semantic drift detection persistence mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class DriftMixin:
    """Mixin providing CRUD for tag_cooccurrence and drift_clusters tables."""

    _dialect: Dialect

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

        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        sorted_tags = sorted(tags)
        pairs: list[tuple[str, str, str, str]] = []
        for i in range(len(sorted_tags)):
            for j in range(i + 1, len(sorted_tags)):
                pairs.append((brain_id, sorted_tags[i], sorted_tags[j], now))

        # Cap pair generation to avoid O(n^2) explosion on large tag sets
        pairs = pairs[:100]

        for brain_id_val, tag_a, tag_b, ts in pairs:
            await d.execute(
                f"""INSERT INTO tag_cooccurrence (brain_id, tag_a, tag_b, count, last_seen)
                   VALUES ({d.phs(5)})
                   ON CONFLICT (brain_id, tag_a, tag_b)
                   DO UPDATE SET count = tag_cooccurrence.count + 1, last_seen = EXCLUDED.last_seen""",
                [brain_id_val, tag_a, tag_b, 1, ts],
            )

    async def get_tag_cooccurrence(
        self,
        min_count: int = 2,
        limit: int = 500,
    ) -> list[tuple[str, str, int]]:
        """Get tag co-occurrence pairs above threshold.

        Returns list of (tag_a, tag_b, count) sorted by count descending.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 2000)

        rows = await d.fetch_all(
            f"""SELECT tag_a, tag_b, count
               FROM tag_cooccurrence
               WHERE brain_id = {d.ph(1)} AND count >= {d.ph(2)}
               ORDER BY count DESC
               LIMIT {d.ph(3)}""",
            [brain_id, min_count, capped_limit],
        )
        return [(r["tag_a"], r["tag_b"], r["count"]) for r in rows]

    async def get_tag_fiber_counts(self) -> dict[str, int]:
        """Get fiber count per tag for Jaccard calculation.

        Returns dict of {tag: fiber_count}.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        # Count fibers per tag via auto_tags + agent_tags JSON arrays
        rows = await d.fetch_all(
            f"""SELECT id, auto_tags, agent_tags
               FROM fibers f
               WHERE f.brain_id = {d.ph(1)}
               LIMIT 10000""",
            [brain_id],
        )

        tag_counts: dict[str, int] = {}
        for r in rows:
            try:
                auto = json.loads(r["auto_tags"]) if r["auto_tags"] else []
                agent = json.loads(r["agent_tags"]) if r["agent_tags"] else []
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
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        upsert_sql = d.upsert_sql(
            "drift_clusters",
            ["id", "brain_id", "canonical", "members", "confidence", "status", "created_at"],
            ["brain_id", "id"],
            ["canonical", "members", "confidence", "status"],
        )
        # After upsert, clear resolved_at separately if needed
        await d.execute(
            upsert_sql,
            [
                cluster_id,
                brain_id,
                canonical,
                json.dumps(members),
                confidence,
                status,
                now,
            ],
        )
        # Clear resolved_at on update
        await d.execute(
            f"""UPDATE drift_clusters SET resolved_at = NULL
               WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}""",
            [brain_id, cluster_id],
        )

    async def get_drift_clusters(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Get drift clusters, optionally filtered by status."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped = min(limit, 200)

        if status:
            rows = await d.fetch_all(
                f"""SELECT id, canonical, members, confidence, status, created_at, resolved_at
                   FROM drift_clusters
                   WHERE brain_id = {d.ph(1)} AND status = {d.ph(2)}
                   ORDER BY confidence DESC
                   LIMIT {d.ph(3)}""",
                [brain_id, status, capped],
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT id, canonical, members, confidence, status, created_at, resolved_at
                   FROM drift_clusters
                   WHERE brain_id = {d.ph(1)}
                   ORDER BY confidence DESC
                   LIMIT {d.ph(2)}""",
                [brain_id, capped],
            )

        return [
            {
                "id": r["id"],
                "canonical": r["canonical"],
                "members": json.loads(r["members"]) if r["members"] else [],
                "confidence": r["confidence"],
                "status": r["status"],
                "created_at": r["created_at"],
                "resolved_at": r["resolved_at"],
            }
            for r in rows
        ]

    async def resolve_drift_cluster(
        self,
        cluster_id: str,
        status: str,
    ) -> bool:
        """Update drift cluster status (merged/aliased/dismissed)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        count = await d.execute_count(
            f"""UPDATE drift_clusters
               SET status = {d.ph(1)}, resolved_at = {d.ph(2)}
               WHERE brain_id = {d.ph(3)} AND id = {d.ph(4)}""",
            [status, now, brain_id, cluster_id],
        )
        return count > 0
