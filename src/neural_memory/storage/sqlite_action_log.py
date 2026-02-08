"""SQLite mixin for action event log storage."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from neural_memory.core.action_event import ActionEvent


class SQLiteActionLogMixin:
    """Action event log persistence for SQLiteStorage.

    Stores lightweight action events for sequence mining and
    habit detection. Events are grouped by session_id and
    ordered by created_at.
    """

    async def record_action(
        self,
        action_type: str,
        action_context: str = "",
        tags: tuple[str, ...] | list[str] = (),
        session_id: str | None = None,
        fiber_id: str | None = None,
    ) -> str:
        """Record an action event in the hippocampal buffer.

        Args:
            action_type: Type of action (e.g., "remember", "recall")
            action_context: Optional context string
            tags: Tags for categorization
            session_id: Optional session grouping
            fiber_id: Optional associated fiber

        Returns:
            The action event ID
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        event_id = str(uuid4())

        await conn.execute(
            """INSERT INTO action_events
               (id, brain_id, session_id, action_type, action_context, tags, fiber_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                brain_id,
                session_id,
                action_type,
                action_context,
                json.dumps(list(tags)),
                fiber_id,
                datetime.utcnow().isoformat(),
            ),
        )
        await conn.commit()
        return event_id

    async def get_action_sequences(
        self,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[ActionEvent]:
        """Get action events ordered by time.

        Args:
            session_id: Filter by session
            since: Only events after this time
            limit: Maximum events to return

        Returns:
            List of ActionEvent objects ordered by created_at
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        conditions = ["brain_id = ?"]
        params: list = [brain_id]

        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)

        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since.isoformat())

        where = " AND ".join(conditions)
        params.append(limit)

        query = f"""
            SELECT id, brain_id, session_id, action_type, action_context,
                   tags, fiber_id, created_at
            FROM action_events
            WHERE {where}
            ORDER BY created_at ASC
            LIMIT ?
        """

        results: list[ActionEvent] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                tags_raw = row["tags"] or "[]"
                tags = tuple(json.loads(tags_raw))
                results.append(
                    ActionEvent(
                        id=row["id"],
                        brain_id=row["brain_id"],
                        session_id=row["session_id"],
                        action_type=row["action_type"],
                        action_context=row["action_context"] or "",
                        tags=tags,
                        fiber_id=row["fiber_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )

        return results

    async def prune_action_events(self, older_than: datetime) -> int:
        """Remove action events older than the given time.

        Args:
            older_than: Remove events created before this time

        Returns:
            Number of events pruned
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM action_events WHERE brain_id = ? AND created_at < ?",
            (brain_id, older_than.isoformat()),
        )
        await conn.commit()
        return cursor.rowcount
