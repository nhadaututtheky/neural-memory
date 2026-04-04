"""Action event log storage mixin — dialect-agnostic."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from neural_memory.core.action_event import ActionEvent
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow


class ActionLogMixin:
    """Action event log persistence for SQLStorage.

    Stores lightweight action events for sequence mining and
    habit detection. Events are grouped by session_id and
    ordered by created_at.
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

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
        d = self._dialect
        brain_id = self._get_brain_id()
        event_id = str(uuid4())

        await d.execute(
            f"""INSERT INTO action_events
               (id, brain_id, session_id, action_type, action_context, tags, fiber_id, created_at)
               VALUES ({d.phs(8)})""",
            [
                event_id,
                brain_id,
                session_id,
                action_type,
                action_context,
                json.dumps(list(tags)),
                fiber_id,
                d.serialize_dt(utcnow()),
            ],
        )
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
        limit = min(limit, 1000)
        d = self._dialect
        brain_id = self._get_brain_id()

        conditions = [f"brain_id = {d.ph(1)}"]
        params: list[Any] = [brain_id]

        if session_id is not None:
            conditions.append(f"session_id = {d.ph(len(params) + 1)}")
            params.append(session_id)

        if since is not None:
            conditions.append(f"created_at >= {d.ph(len(params) + 1)}")
            params.append(d.serialize_dt(since))

        where = " AND ".join(conditions)
        params.append(limit)

        query = f"""
            SELECT id, brain_id, session_id, action_type, action_context,
                   tags, fiber_id, created_at
            FROM action_events
            WHERE {where}
            ORDER BY created_at ASC
            LIMIT {d.ph(len(params))}
        """

        rows = await d.fetch_all(query, params)
        results: list[ActionEvent] = []
        for row in rows:
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
                    created_at=row["created_at"]
                    if isinstance(row["created_at"], datetime)
                    else datetime.fromisoformat(row["created_at"]),
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
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"DELETE FROM action_events WHERE brain_id = {d.ph(1)} AND created_at < {d.ph(2)}",
            [brain_id, d.serialize_dt(older_than)],
        )
