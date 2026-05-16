"""Tests for prefer_recent recall flag (Phase 3 agent ergonomics).

Tests the pure re-rank helper and the schema exposure. End-to-end recall
behavior is covered by existing recall tests; this guards the new flag.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.utils.timeutils import utcnow


class _StubFiber:
    """Minimal fiber stub with the timestamp fields used by the re-rank helper."""

    def __init__(
        self,
        fid: str,
        time_end: Any = None,
        created_at: Any = None,
    ) -> None:
        self.id = fid
        self.time_end = time_end
        self.created_at = created_at or utcnow()


@pytest.mark.asyncio
async def test_rerank_by_recency_orders_newest_first() -> None:
    from neural_memory.mcp.recall_handler import _rerank_by_recency

    now = utcnow()
    fibers = {
        "old": _StubFiber("old", time_end=now - timedelta(days=30)),
        "mid": _StubFiber("mid", time_end=now - timedelta(days=3)),
        "new": _StubFiber("new", time_end=now - timedelta(hours=1)),
    }
    storage = AsyncMock()
    storage.get_fiber = AsyncMock(side_effect=lambda fid: fibers.get(fid))

    out = await _rerank_by_recency(["old", "mid", "new"], storage)
    assert out == ["new", "mid", "old"]


@pytest.mark.asyncio
async def test_rerank_falls_back_to_created_at_when_time_end_missing() -> None:
    from neural_memory.mcp.recall_handler import _rerank_by_recency

    now = utcnow()
    fibers = {
        "a": _StubFiber("a", time_end=None, created_at=now - timedelta(days=10)),
        "b": _StubFiber("b", time_end=None, created_at=now - timedelta(days=1)),
    }
    storage = AsyncMock()
    storage.get_fiber = AsyncMock(side_effect=lambda fid: fibers.get(fid))

    out = await _rerank_by_recency(["a", "b"], storage)
    assert out == ["b", "a"]


@pytest.mark.asyncio
async def test_rerank_keeps_unknown_fibers_at_end() -> None:
    """Fibers that can't be fetched go last (no crash)."""
    from neural_memory.mcp.recall_handler import _rerank_by_recency

    now = utcnow()
    fibers = {"a": _StubFiber("a", time_end=now - timedelta(hours=1))}
    storage = AsyncMock()
    storage.get_fiber = AsyncMock(side_effect=lambda fid: fibers.get(fid))

    out = await _rerank_by_recency(["a", "missing"], storage)
    # Known fiber first, missing one at the end
    assert out[0] == "a"
    assert out[-1] == "missing"


def test_prefer_recent_in_schema() -> None:
    """Schema exposes the prefer_recent flag on nmem_recall."""
    from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS

    recall_schema = next(s for s in _ALL_TOOL_SCHEMAS if s["name"] == "nmem_recall")
    props = recall_schema["inputSchema"]["properties"]
    assert "prefer_recent" in props
    assert props["prefer_recent"]["type"] == "boolean"
