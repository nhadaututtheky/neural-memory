"""Tests for nmem_situation MCP tool (Phase 2 agent ergonomics).

Situation snapshot aggregates active session state, recent decisions,
open blockers, and gap detection in a single response.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.utils.timeutils import utcnow


def _make_handler(brain_id: str = "test-brain"):
    """Build a handler with mock storage exposing find_typed_memories."""
    from neural_memory.mcp.session_handler import SessionHandler

    storage = AsyncMock()
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage._current_brain_id = brain_id

    brain_mock = MagicMock(id=brain_id, config=MagicMock())
    storage.get_brain = AsyncMock(return_value=brain_mock)

    # Default: nothing in brain
    storage.find_typed_memories = AsyncMock(return_value=[])

    class TestHandler(SessionHandler):
        config = MagicMock()

        async def get_storage(self) -> Any:
            return storage

    return TestHandler(), storage


def _mk_typed(
    content: str,
    mem_type: MemoryType,
    *,
    age_minutes: int = 0,
    tags: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TypedMemory:
    """Helper to build a TypedMemory at a relative age.

    Content is stored in metadata['content'] so _resolve_content can read it
    without a fiber fetch (cheap path).
    """
    when = utcnow() - timedelta(minutes=age_minutes)
    md = {"content": content, **(metadata or {})}
    tm = TypedMemory.create(
        fiber_id=f"fiber-{content[:10]}",
        memory_type=mem_type,
        priority=Priority.from_int(5),
        source="test",
        tags=tags or set(),
        metadata=md,
    )
    from dataclasses import replace

    return replace(tm, created_at=when)


class TestSituation:
    @pytest.mark.asyncio
    async def test_empty_brain(self) -> None:
        handler, _storage = _make_handler()
        result = await handler._situation({})
        assert "active_task" in result
        assert result["active_task"] is None
        assert result["recent_decisions"] == []
        assert result["open_blockers"] == []
        assert result["gap_detected"] is False

    @pytest.mark.asyncio
    async def test_recent_decisions_sorted(self) -> None:
        """Recent decisions return top 3 by created_at desc."""
        handler, storage = _make_handler()

        decisions = [
            _mk_typed("old decision", MemoryType.DECISION, age_minutes=600),
            _mk_typed("newest decision", MemoryType.DECISION, age_minutes=5),
            _mk_typed("mid decision", MemoryType.DECISION, age_minutes=60),
            _mk_typed("oldest decision", MemoryType.DECISION, age_minutes=1200),
        ]

        async def _find(memory_type: MemoryType | None = None, **_kw: Any) -> list[TypedMemory]:
            if memory_type == MemoryType.DECISION:
                return decisions
            return []

        storage.find_typed_memories = AsyncMock(side_effect=_find)

        result = await handler._situation({})
        # Top 3 most recent, in order
        assert len(result["recent_decisions"]) == 3
        contents = [d["content"] for d in result["recent_decisions"]]
        # The newest first; oldest excluded
        assert contents[0] == "newest decision"
        assert "oldest decision" not in contents

    @pytest.mark.asyncio
    async def test_open_blockers_filter(self) -> None:
        """Only TODO with blocker tag and without resolved tag are returned."""
        handler, storage = _make_handler()

        memories = [
            _mk_typed("blocker A", MemoryType.TODO, tags={"blocker"}),
            _mk_typed("blocker B done", MemoryType.TODO, tags={"blocker", "resolved"}),
            _mk_typed("plain todo", MemoryType.TODO, tags=set()),
        ]

        async def _find(
            memory_type: MemoryType | None = None,
            tags: set[str] | None = None,
            **_kw: Any,
        ) -> list[TypedMemory]:
            if memory_type == MemoryType.TODO and tags and "blocker" in tags:
                return [m for m in memories if "blocker" in m.tags]
            return []

        storage.find_typed_memories = AsyncMock(side_effect=_find)

        result = await handler._situation({})
        blocker_contents = [b["content"] for b in result["open_blockers"]]
        assert "blocker A" in blocker_contents
        assert "blocker B done" not in blocker_contents

    @pytest.mark.asyncio
    async def test_active_session_extracted(self) -> None:
        """When a session is active, its task surfaces into active_task."""
        handler, storage = _make_handler()

        active_session = _mk_typed(
            "session state",
            MemoryType.CONTEXT,
            tags={"session_state"},
            metadata={
                "active": True,
                "feature": "agent-ergonomics",
                "task": "implement phase 2",
                "started_at": "2026-05-17T10:00:00",
            },
        )

        async def _find(
            memory_type: MemoryType | None = None,
            tags: set[str] | None = None,
            **_kw: Any,
        ) -> list[TypedMemory]:
            if tags and "session_state" in tags:
                return [active_session]
            return []

        storage.find_typed_memories = AsyncMock(side_effect=_find)

        result = await handler._situation({})
        assert result["active_task"] == "implement phase 2"
        assert result["active_feature"] == "agent-ergonomics"

    @pytest.mark.asyncio
    async def test_response_has_files_in_session_key(self) -> None:
        """files_in_session is a contract key — must always be present."""
        handler, _storage = _make_handler()
        result = await handler._situation({})
        assert "files_in_session" in result
        assert result["files_in_session"] == []

    @pytest.mark.asyncio
    async def test_gap_detected_via_timestamp(self) -> None:
        """A summary newer than the fingerprint by ≥60s triggers gap_detected."""
        from datetime import datetime as _dt

        handler, storage = _make_handler()

        fp_saved_at = (utcnow() - timedelta(hours=2)).isoformat()
        summary_created_at = utcnow() - timedelta(minutes=10)

        fp_mem = _mk_typed(
            "session_fingerprint:abc",
            MemoryType.CONTEXT,
            tags={"session_fingerprint"},
            metadata={"fingerprint": "abc", "saved_at": fp_saved_at},
        )
        summary_mem = _mk_typed(
            "session ended summary",
            MemoryType.CONTEXT,
            tags={"session_summary"},
        )
        # Backdate summary's created_at to AFTER fingerprint.saved_at
        from dataclasses import replace as _dc_replace

        summary_mem = _dc_replace(summary_mem, created_at=summary_created_at)

        async def _find(
            memory_type: MemoryType | None = None,
            tags: set[str] | None = None,
            **_kw: Any,
        ) -> list[TypedMemory]:
            if tags and "session_fingerprint" in tags:
                return [fp_mem]
            if tags and "session_summary" in tags:
                return [summary_mem]
            return []

        storage.find_typed_memories = AsyncMock(side_effect=_find)
        # Make sure _dt is referenced so import doesn't get flagged unused.
        assert _dt is not None

        result = await handler._situation({})
        assert result["gap_detected"] is True
