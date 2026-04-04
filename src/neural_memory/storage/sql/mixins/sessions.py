"""Session summary persistence mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from typing import Any

from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)

# Bounds
MAX_SUMMARIES_PER_BRAIN = 500
MAX_RECENT_SUMMARIES = 100


class SessionsMixin:
    """Mixin: session summary CRUD for SQLStorage."""

    _dialect: Dialect

    @property
    def brain_id(self) -> str | None:
        raise NotImplementedError

    async def save_session_summary(
        self,
        session_id: str,
        topics: list[str],
        topic_weights: dict[str, float],
        top_entities: list[tuple[str, int]],
        query_count: int,
        avg_confidence: float,
        avg_depth: float,
        started_at: str,
        ended_at: str,
    ) -> None:
        """Persist a session summary snapshot.

        Args:
            session_id: Unique session identifier.
            topics: Top topic strings from session EMA.
            topic_weights: Topic → EMA weight mapping.
            top_entities: Most common entities as (entity, count) pairs.
            query_count: Total queries in this session.
            avg_confidence: Average confidence across queries.
            avg_depth: Average depth used across queries.
            started_at: ISO timestamp of session start.
            ended_at: ISO timestamp of this summary.
        """
        brain_id = self.brain_id
        if not brain_id:
            return

        d = self._dialect
        await d.execute(
            f"""INSERT INTO session_summaries
                (session_id, brain_id, topics_json, topic_weights_json,
                 top_entities_json, query_count, avg_confidence, avg_depth,
                 started_at, ended_at)
                VALUES ({d.phs(10)})""",
            [
                session_id,
                brain_id,
                json.dumps(topics),
                json.dumps(topic_weights),
                json.dumps(top_entities),
                query_count,
                round(avg_confidence, 4),
                round(avg_depth, 2),
                started_at,
                ended_at,
            ],
        )

        # Prune old summaries if over limit
        await self._prune_session_summaries(brain_id)

    async def get_recent_session_summaries(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch recent session summaries for the current brain.

        Args:
            limit: Maximum number of summaries to return.

        Returns:
            List of session summary dicts, most recent first.
        """
        brain_id = self.brain_id
        if not brain_id:
            return []

        capped_limit = min(limit, MAX_RECENT_SUMMARIES)
        d = self._dialect
        rows = await d.fetch_all(
            f"""SELECT session_id, topics_json, topic_weights_json,
                      top_entities_json, query_count, avg_confidence,
                      avg_depth, started_at, ended_at
               FROM session_summaries
               WHERE brain_id = {d.ph(1)}
               ORDER BY ended_at DESC
               LIMIT {d.ph(2)}""",
            [brain_id, capped_limit],
        )

        return [
            {
                "session_id": r["session_id"],
                "topics": json.loads(r["topics_json"]),
                "topic_weights": json.loads(r["topic_weights_json"]),
                "top_entities": json.loads(r["top_entities_json"]),
                "query_count": r["query_count"],
                "avg_confidence": r["avg_confidence"],
                "avg_depth": r["avg_depth"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
            }
            for r in rows
        ]

    async def _prune_session_summaries(self, brain_id: str) -> None:
        """Keep only the most recent MAX_SUMMARIES_PER_BRAIN summaries."""
        d = self._dialect
        await d.execute(
            f"""DELETE FROM session_summaries
                WHERE brain_id = {d.ph(1)} AND id NOT IN (
                    SELECT id FROM session_summaries
                    WHERE brain_id = {d.ph(2)}
                    ORDER BY ended_at DESC
                    LIMIT {d.ph(3)}
                )""",
            [brain_id, brain_id, MAX_SUMMARIES_PER_BRAIN],
        )
