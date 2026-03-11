"""SQLite mixin for session summary persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Bounds
MAX_SUMMARIES_PER_BRAIN = 500
MAX_RECENT_SUMMARIES = 100


class SQLiteSessionsMixin:
    """Mixin: session summary CRUD for SQLiteStorage."""

    if TYPE_CHECKING:

        @property
        def brain_id(self) -> str | None:
            raise NotImplementedError

        def _ensure_conn(self) -> aiosqlite.Connection:
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

        conn = self._ensure_conn()
        await conn.execute(
            """INSERT INTO session_summaries
               (session_id, brain_id, topics_json, topic_weights_json,
                top_entities_json, query_count, avg_confidence, avg_depth,
                started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
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
            ),
        )
        await conn.commit()

        # Prune old summaries if over limit
        await self._prune_session_summaries(conn, brain_id)

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
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """SELECT session_id, topics_json, topic_weights_json,
                      top_entities_json, query_count, avg_confidence,
                      avg_depth, started_at, ended_at
               FROM session_summaries
               WHERE brain_id = ?
               ORDER BY ended_at DESC
               LIMIT ?""",
            (brain_id, capped_limit),
        )
        rows = await cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "topics": json.loads(row[1]),
                "topic_weights": json.loads(row[2]),
                "top_entities": json.loads(row[3]),
                "query_count": row[4],
                "avg_confidence": row[5],
                "avg_depth": row[6],
                "started_at": row[7],
                "ended_at": row[8],
            }
            for row in rows
        ]

    async def _prune_session_summaries(self, conn: aiosqlite.Connection, brain_id: str) -> None:
        """Keep only the most recent MAX_SUMMARIES_PER_BRAIN summaries."""
        await conn.execute(
            """DELETE FROM session_summaries
               WHERE brain_id = ? AND id NOT IN (
                   SELECT id FROM session_summaries
                   WHERE brain_id = ?
                   ORDER BY ended_at DESC
                   LIMIT ?
               )""",
            (brain_id, brain_id, MAX_SUMMARIES_PER_BRAIN),
        )
