"""Tool event storage and statistics mixin — dialect-agnostic."""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth
_MAX_EVENTS_PER_BRAIN = 100_000


class ToolEventsMixin:
    """Mixin providing CRUD for the tool_events table."""

    _dialect: Dialect

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

        d = self._dialect
        inserted = 0
        for ev in events:
            await d.execute(
                f"""INSERT INTO tool_events
                   (brain_id, tool_name, server_name, args_summary,
                    success, duration_ms, session_id, task_context,
                    processed, created_at)
                   VALUES ({d.phs(10)})""",
                [
                    brain_id,
                    ev.get("tool_name", ""),
                    ev.get("server_name", ""),
                    ev.get("args_summary", "")[:200],
                    1 if ev.get("success", True) else 0,
                    ev.get("duration_ms", 0),
                    ev.get("session_id", ""),
                    ev.get("task_context", ""),
                    0,
                    ev.get("created_at", d.serialize_dt(utcnow())),
                ],
            )
            inserted += 1
        return inserted

    async def get_unprocessed_events(
        self,
        brain_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get unprocessed tool events for pattern detection.

        Returns list of dicts ordered by created_at ASC.
        """
        d = self._dialect
        safe_limit = min(limit, 10000)

        rows = await d.fetch_all(
            f"""SELECT id, tool_name, server_name, args_summary,
                      success, duration_ms, session_id, task_context,
                      created_at
               FROM tool_events
               WHERE brain_id = {d.ph(1)} AND processed = 0
               ORDER BY created_at ASC
               LIMIT {d.ph(2)}""",
            [brain_id, safe_limit],
        )
        results: list[dict[str, Any]] = []
        for row in rows:
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
        d = self._dialect

        in_fragment, in_params = d.in_clause(2, [str(eid) for eid in event_ids])
        await d.execute(
            f"UPDATE tool_events SET processed = 1 WHERE brain_id = {d.ph(1)} AND id {in_fragment}",
            [brain_id, *in_params],
        )

    async def prune_old_events(
        self,
        brain_id: str,
        keep_days: int = 90,
    ) -> int:
        """Delete processed events older than keep_days.

        Returns number of rows deleted.
        """
        d = self._dialect
        from datetime import timedelta

        cutoff = d.serialize_dt(utcnow() - timedelta(days=keep_days))
        return await d.execute_count(
            f"DELETE FROM tool_events WHERE brain_id = {d.ph(1)} AND processed = 1 AND created_at < {d.ph(2)}",
            [brain_id, cutoff],
        )

    async def cap_tool_events(self, brain_id: str) -> int:
        """Enforce max events per brain by deleting oldest processed rows."""
        d = self._dialect

        row = await d.fetch_one(
            f"SELECT COUNT(*) as cnt FROM tool_events WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )
        total = row.get("cnt", 0) if row else 0

        if total <= _MAX_EVENTS_PER_BRAIN:
            return 0

        excess = total - _MAX_EVENTS_PER_BRAIN
        return await d.execute_count(
            f"""DELETE FROM tool_events WHERE brain_id = {d.ph(1)} AND id IN (
                SELECT id FROM tool_events WHERE brain_id = {d.ph(2)} AND processed = 1
                ORDER BY created_at ASC LIMIT {d.ph(3)}
            )""",
            [brain_id, brain_id, excess],
        )

    async def get_tool_stats(self, brain_id: str) -> dict[str, Any]:
        """Get tool usage statistics for a brain.

        Returns dict with top_tools, total_events, success_rate.
        """
        d = self._dialect

        # Total counts
        row = await d.fetch_one(
            f"""SELECT COUNT(*) as total,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
               FROM tool_events WHERE brain_id = {d.ph(1)}""",
            [brain_id],
        )
        total = row.get("total", 0) if row else 0
        successes = row.get("successes", 0) if row else 0

        # Top tools by frequency
        top_tools: list[dict[str, Any]] = []
        rows = await d.fetch_all(
            f"""SELECT tool_name, server_name, COUNT(*) as cnt,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok,
                      AVG(duration_ms) as avg_ms
               FROM tool_events WHERE brain_id = {d.ph(1)}
               GROUP BY tool_name, server_name
               ORDER BY cnt DESC
               LIMIT 20""",
            [brain_id],
        )
        for row in rows:
            cnt = row["cnt"]
            top_tools.append(
                {
                    "tool_name": row["tool_name"],
                    "server_name": row["server_name"],
                    "count": cnt,
                    "success_rate": round(row["ok"] / cnt, 2) if cnt > 0 else 0,
                    "avg_duration_ms": round(row["avg_ms"] or 0),
                }
            )

        return {
            "total_events": total,
            "success_rate": round(successes / total, 2) if total > 0 else 0,
            "top_tools": top_tools,
        }

    async def get_tool_stats_by_period(
        self,
        brain_id: str,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get tool usage stats aggregated by day.

        Returns list of dicts with {date, tool_name, count, success_rate, avg_duration_ms}.
        """
        d = self._dialect
        safe_days = min(max(days, 1), 365)
        safe_limit = min(limit, 50)

        from datetime import timedelta

        cutoff = d.serialize_dt(utcnow() - timedelta(days=safe_days))

        rows = await d.fetch_all(
            f"""SELECT CAST(created_at AS DATE) as day, tool_name,
                      COUNT(*) as cnt,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok,
                      AVG(duration_ms) as avg_ms
               FROM tool_events
               WHERE brain_id = {d.ph(1)} AND created_at >= {d.ph(2)}
               GROUP BY day, tool_name
               ORDER BY day DESC, cnt DESC
               LIMIT {d.ph(3)}""",
            [brain_id, cutoff, safe_limit * safe_days],
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            cnt = row["cnt"]
            results.append(
                {
                    "date": row["day"],
                    "tool_name": row["tool_name"],
                    "count": cnt,
                    "success_rate": round(row["ok"] / cnt, 2) if cnt > 0 else 0,
                    "avg_duration_ms": round(row["avg_ms"] or 0),
                }
            )
        return results
