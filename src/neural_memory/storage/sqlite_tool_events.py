"""SQLite mixin for tool event storage and statistics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth
_MAX_EVENTS_PER_BRAIN = 100_000


class SQLiteToolEventsMixin:
    """Mixin providing CRUD for the tool_events table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def insert_tool_events(
        self,
        brain_id: str,
        events: list[dict[str, Any]],
    ) -> int:
        """Insert raw tool events into the staging table.

        Args:
            brain_id: Brain context.
            events: List of dicts with keys: tool_name, server_name,
                args_summary, success, duration_ms, session_id,
                task_context, created_at.

        Returns:
            Number of events inserted.
        """
        if not events:
            return 0

        conn = self._ensure_conn()
        inserted = 0
        for ev in events:
            await conn.execute(
                """INSERT INTO tool_events
                   (brain_id, tool_name, server_name, args_summary,
                    success, duration_ms, session_id, task_context,
                    processed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    brain_id,
                    ev.get("tool_name", ""),
                    ev.get("server_name", ""),
                    ev.get("args_summary", "")[:200],
                    1 if ev.get("success", True) else 0,
                    ev.get("duration_ms", 0),
                    ev.get("session_id", ""),
                    ev.get("task_context", ""),
                    ev.get("created_at", utcnow().isoformat()),
                ),
            )
            inserted += 1
        await conn.commit()
        return inserted

    async def get_unprocessed_events(
        self,
        brain_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get unprocessed tool events for pattern detection.

        Returns list of dicts ordered by created_at ASC.
        """
        conn = self._ensure_read_conn()
        safe_limit = min(limit, 10000)
        results: list[dict[str, Any]] = []
        async with conn.execute(
            """SELECT id, tool_name, server_name, args_summary,
                      success, duration_ms, session_id, task_context,
                      created_at
               FROM tool_events
               WHERE brain_id = ? AND processed = 0
               ORDER BY created_at ASC
               LIMIT ?""",
            (brain_id, safe_limit),
        ) as cursor:
            async for row in cursor:
                results.append(
                    {
                        "id": row["id"],
                        "tool_name": row["tool_name"],
                        "server_name": row["server_name"],
                        "args_summary": row["args_summary"],
                        "success": bool(row["success"]),
                        "duration_ms": row["duration_ms"],
                        "session_id": row["session_id"],
                        "task_context": row["task_context"],
                        "created_at": row["created_at"],
                    }
                )
        return results

    async def mark_events_processed(
        self,
        brain_id: str,
        event_ids: list[int],
    ) -> None:
        """Mark tool events as processed."""
        if not event_ids:
            return
        conn = self._ensure_conn()
        placeholders = ",".join("?" for _ in event_ids)
        # Table/column names are hardcoded — safe to interpolate placeholders.
        await conn.execute(
            f"UPDATE tool_events SET processed = 1 WHERE brain_id = ? AND id IN ({placeholders})",
            [brain_id, *event_ids],
        )
        await conn.commit()

    async def prune_old_events(
        self,
        brain_id: str,
        keep_days: int = 90,
    ) -> int:
        """Delete processed events older than keep_days.

        Returns number of rows deleted.
        """
        conn = self._ensure_conn()
        from datetime import timedelta

        cutoff = (utcnow() - timedelta(days=keep_days)).isoformat()
        cursor = await conn.execute(
            "DELETE FROM tool_events WHERE brain_id = ? AND processed = 1 AND created_at < ?",
            (brain_id, cutoff),
        )
        deleted = cursor.rowcount
        await conn.commit()
        return deleted

    async def cap_tool_events(self, brain_id: str) -> int:
        """Enforce max events per brain by deleting oldest processed rows."""
        conn = self._ensure_conn()
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM tool_events WHERE brain_id = ?",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            total = row["cnt"] if row else 0

        if total <= _MAX_EVENTS_PER_BRAIN:
            return 0

        excess = total - _MAX_EVENTS_PER_BRAIN
        cursor = await conn.execute(
            """DELETE FROM tool_events WHERE brain_id = ? AND id IN (
                SELECT id FROM tool_events WHERE brain_id = ? AND processed = 1
                ORDER BY created_at ASC LIMIT ?
            )""",
            (brain_id, brain_id, excess),
        )
        deleted = cursor.rowcount
        await conn.commit()
        return deleted

    async def get_tool_stats(self, brain_id: str) -> dict[str, Any]:
        """Get tool usage statistics for a brain.

        Returns dict with top_tools, total_events, success_rate.
        """
        conn = self._ensure_read_conn()

        # Total counts
        async with conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
               FROM tool_events WHERE brain_id = ?""",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            total = row["total"] if row else 0
            successes = row["successes"] if row else 0

        # Top tools by frequency
        top_tools: list[dict[str, Any]] = []
        async with conn.execute(
            """SELECT tool_name, server_name, COUNT(*) as cnt,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok,
                      AVG(duration_ms) as avg_ms
               FROM tool_events WHERE brain_id = ?
               GROUP BY tool_name, server_name
               ORDER BY cnt DESC
               LIMIT 20""",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                top_tools.append(
                    {
                        "tool_name": row["tool_name"],
                        "server_name": row["server_name"],
                        "count": row["cnt"],
                        "success_rate": round(row["ok"] / row["cnt"], 2) if row["cnt"] > 0 else 0,
                        "avg_duration_ms": round(row["avg_ms"] or 0),
                    }
                )

        return {
            "total_events": total,
            "success_rate": round(successes / total, 2) if total > 0 else 0,
            "top_tools": top_tools,
        }
