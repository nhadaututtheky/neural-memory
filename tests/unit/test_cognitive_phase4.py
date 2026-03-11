"""Tests for cognitive handler Phase 4: hot index (nmem_cognitive) and knowledge gaps (nmem_gaps)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.mcp.cognitive_handler import CognitiveHandler


def _make_storage(brain_id: str | None = "test-brain") -> AsyncMock:
    """Build a mock NeuralStorage with explicit brain_id."""
    storage = AsyncMock()
    storage._current_brain_id = brain_id
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage.get_brain = AsyncMock(return_value=MagicMock(config=MagicMock()))
    storage.get_hot_index = AsyncMock(return_value=[])
    storage.get_calibration_stats = AsyncMock(
        return_value={"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
    )
    storage.list_knowledge_gaps = AsyncMock(return_value=[])
    storage.list_cognitive_states = AsyncMock(return_value=[])
    storage.list_predictions = AsyncMock(return_value=[])
    storage.get_neuron = AsyncMock(return_value=MagicMock(content="test content"))
    storage.get_neurons_batch = AsyncMock(return_value={})
    storage.refresh_hot_index = AsyncMock(return_value=0)
    storage.add_knowledge_gap = AsyncMock(return_value="gap-1")
    storage.get_knowledge_gap = AsyncMock(return_value=None)
    storage.resolve_knowledge_gap = AsyncMock(return_value=True)
    return storage


def _make_handler(storage: AsyncMock) -> CognitiveHandler:
    handler = CognitiveHandler()
    handler.get_storage = AsyncMock(return_value=storage)  # type: ignore[attr-defined]
    handler.config = MagicMock()  # type: ignore[attr-defined]
    return handler


# ──────────────────── nmem_cognitive tests ────────────────────


class TestCognitiveSummary:
    """Tests for _cognitive action=summary."""

    @pytest.mark.asyncio
    async def test_summary_empty(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "summary"})
        assert result["hot_count"] == 0
        assert result["top_gaps_count"] == 0
        assert result["calibration"]["score"] == 0.5

    @pytest.mark.asyncio
    async def test_summary_with_items(self) -> None:
        storage = _make_storage()
        storage.get_hot_index = AsyncMock(
            return_value=[
                {
                    "slot": 0,
                    "category": "hypothesis",
                    "neuron_id": "n-1",
                    "summary": "test hyp",
                    "confidence": 0.6,
                    "score": 7.5,
                    "updated_at": "2026-03-06",
                },
            ]
        )
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "summary"})
        assert result["hot_count"] == 1
        assert result["hot_items"][0]["category"] == "hypothesis"

    @pytest.mark.asyncio
    async def test_summary_with_gaps(self) -> None:
        storage = _make_storage()
        storage.list_knowledge_gaps = AsyncMock(
            return_value=[
                {"id": "g-1", "topic": "Missing auth docs", "priority": 0.8},
                {"id": "g-2", "topic": "Unclear perf reqs", "priority": 0.6},
            ]
        )
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "summary"})
        assert result["top_gaps_count"] == 2
        assert len(result["top_gaps"]) == 2

    @pytest.mark.asyncio
    async def test_summary_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "summary"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_summary_invalid_action(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "delete"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_summary_bad_limit(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._cognitive({"action": "summary", "limit": "abc"})
        # Should fallback to 10
        storage.get_hot_index.assert_called_once_with(limit=10)


class TestCognitiveRefresh:
    """Tests for _cognitive action=refresh."""

    @pytest.mark.asyncio
    async def test_refresh_empty(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "refresh"})
        assert result["status"] == "refreshed"
        assert result["items_indexed"] == 0

    @pytest.mark.asyncio
    async def test_refresh_with_hypotheses(self) -> None:
        storage = _make_storage()
        storage.list_cognitive_states = AsyncMock(
            return_value=[
                {
                    "neuron_id": "h-1",
                    "confidence": 0.6,
                    "evidence_for_count": 2,
                    "evidence_against_count": 1,
                    "status": "active",
                    "created_at": "2026-03-01",
                },
            ]
        )
        storage.get_neurons_batch = AsyncMock(
            return_value={"h-1": MagicMock(content="hypothesis content")}
        )
        storage.refresh_hot_index = AsyncMock(return_value=1)
        handler = _make_handler(storage)

        result = await handler._cognitive({"action": "refresh"})
        assert result["status"] == "refreshed"
        assert result["hypotheses_scored"] == 1
        storage.refresh_hot_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_with_predictions(self) -> None:
        storage = _make_storage()
        storage.list_predictions = AsyncMock(
            return_value=[
                {
                    "neuron_id": "p-1",
                    "confidence": 0.7,
                    "status": "pending",
                    "predicted_at": "2026-04-01",
                    "created_at": "2026-03-01",
                },
            ]
        )
        storage.get_neurons_batch = AsyncMock(
            return_value={"p-1": MagicMock(content="prediction content")}
        )
        storage.refresh_hot_index = AsyncMock(return_value=1)
        handler = _make_handler(storage)

        result = await handler._cognitive({"action": "refresh"})
        assert result["predictions_scored"] == 1

    @pytest.mark.asyncio
    async def test_refresh_sorts_by_score(self) -> None:
        """Items should be slot-numbered by descending score."""
        storage = _make_storage()
        storage.list_cognitive_states = AsyncMock(
            return_value=[
                {
                    "neuron_id": "h-1",
                    "confidence": 0.5,
                    "evidence_for_count": 5,
                    "evidence_against_count": 0,
                    "status": "active",
                    "created_at": "2026-03-01",
                },
                {
                    "neuron_id": "h-2",
                    "confidence": 0.5,
                    "evidence_for_count": 0,
                    "evidence_against_count": 0,
                    "status": "active",
                    "created_at": "2026-03-01",
                },
            ]
        )
        storage.get_neurons_batch = AsyncMock(
            return_value={
                "h-1": MagicMock(content="hyp 1"),
                "h-2": MagicMock(content="hyp 2"),
            }
        )
        handler = _make_handler(storage)

        captured_items: list[Any] = []

        async def capture_refresh(items: list[dict[str, Any]]) -> int:
            captured_items.extend(items)
            return len(items)

        storage.refresh_hot_index = capture_refresh

        await handler._cognitive({"action": "refresh"})
        assert len(captured_items) == 2
        assert captured_items[0]["slot"] == 0
        assert captured_items[0]["score"] >= captured_items[1]["score"]

    @pytest.mark.asyncio
    async def test_refresh_storage_failure(self) -> None:
        storage = _make_storage()
        storage.list_cognitive_states = AsyncMock(side_effect=RuntimeError("DB error"))
        handler = _make_handler(storage)
        result = await handler._cognitive({"action": "refresh"})
        assert "error" in result


# ──────────────────── nmem_gaps tests ────────────────────


class TestGapsDetect:
    """Tests for _gaps action=detect."""

    @pytest.mark.asyncio
    async def test_detect_basic(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps(
            {
                "action": "detect",
                "topic": "Missing deployment docs",
                "source": "user_flagged",
            }
        )
        assert result["status"] == "detected"
        assert result["gap_id"] == "gap-1"
        assert result["priority"] == 0.6  # user_flagged default

    @pytest.mark.asyncio
    async def test_detect_no_topic(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "detect"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_topic_too_long(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "detect", "topic": "x" * 501})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_invalid_source(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps(
            {
                "action": "detect",
                "topic": "Test",
                "source": "invalid_source",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_custom_priority(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps(
            {
                "action": "detect",
                "topic": "Critical gap",
                "source": "contradicting_evidence",
                "priority": 0.95,
            }
        )
        assert result["priority"] == 0.95

    @pytest.mark.asyncio
    async def test_detect_with_related_neurons(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._gaps(
            {
                "action": "detect",
                "topic": "Gap with context",
                "source": "recall_miss",
                "related_neuron_ids": ["n-1", "n-2"],
            }
        )
        call_kwargs = storage.add_knowledge_gap.call_args[1]
        assert call_kwargs["related_neuron_ids"] == ["n-1", "n-2"]

    @pytest.mark.asyncio
    async def test_detect_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "detect", "topic": "Test"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_default_source(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "detect", "topic": "Test gap"})
        assert result["source"] == "user_flagged"


class TestGapsList:
    """Tests for _gaps action=list."""

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "list"})
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_with_results(self) -> None:
        storage = _make_storage()
        storage.list_knowledge_gaps = AsyncMock(
            return_value=[
                {
                    "id": "g-1",
                    "topic": "Missing docs",
                    "priority": 0.8,
                    "detected_at": "2026-03-01",
                    "detection_source": "user_flagged",
                    "related_neuron_ids": [],
                    "resolved_at": None,
                    "resolved_by_neuron_id": None,
                },
            ]
        )
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "list"})
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_include_resolved(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._gaps({"action": "list", "include_resolved": True})
        storage.list_knowledge_gaps.assert_called_once_with(include_resolved=True, limit=20)

    @pytest.mark.asyncio
    async def test_list_bad_limit(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._gaps({"action": "list", "limit": "abc"})
        storage.list_knowledge_gaps.assert_called_once_with(include_resolved=False, limit=20)


class TestGapsResolve:
    """Tests for _gaps action=resolve."""

    @pytest.mark.asyncio
    async def test_resolve_success(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps(
            {
                "action": "resolve",
                "gap_id": "g-1",
                "resolved_by_neuron_id": "n-42",
            }
        )
        assert result["status"] == "resolved"
        assert result["resolved_by_neuron_id"] == "n-42"

    @pytest.mark.asyncio
    async def test_resolve_no_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "resolve"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_resolve_not_found(self) -> None:
        storage = _make_storage()
        storage.resolve_knowledge_gap = AsyncMock(return_value=False)
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "resolve", "gap_id": "nope"})
        assert "error" in result
        assert "not found" in result["error"]


class TestGapsGet:
    """Tests for _gaps action=get."""

    @pytest.mark.asyncio
    async def test_get_not_found(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "get", "gap_id": "nope"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        storage = _make_storage()
        storage.get_knowledge_gap = AsyncMock(
            return_value={
                "id": "g-1",
                "topic": "Missing docs",
                "priority": 0.8,
                "detected_at": "2026-03-01",
                "detection_source": "user_flagged",
                "related_neuron_ids": ["n-1"],
                "resolved_at": None,
                "resolved_by_neuron_id": None,
            }
        )
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "get", "gap_id": "g-1"})
        assert result["topic"] == "Missing docs"

    @pytest.mark.asyncio
    async def test_get_no_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "get"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_action(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._gaps({"action": "delete"})
        assert "error" in result


# ──────────────────── Storage integration tests ────────────────────


class TestSQLiteHotIndex:
    """Integration tests for hot_index storage methods."""

    @pytest.mark.asyncio
    async def test_refresh_and_get(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE hot_index (
            brain_id TEXT NOT NULL, slot INTEGER NOT NULL,
            category TEXT NOT NULL, neuron_id TEXT NOT NULL,
            summary TEXT NOT NULL, confidence REAL,
            score REAL NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (brain_id, slot)
        )""")

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_conn = lambda: db  # type: ignore[assignment]
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        items = [
            {
                "slot": 0,
                "category": "hypothesis",
                "neuron_id": "n-1",
                "summary": "Test hyp",
                "confidence": 0.6,
                "score": 8.0,
            },
            {
                "slot": 1,
                "category": "prediction",
                "neuron_id": "n-2",
                "summary": "Test pred",
                "confidence": 0.7,
                "score": 6.0,
            },
        ]
        count = await mixin.refresh_hot_index(items)
        assert count == 2

        result = await mixin.get_hot_index(limit=10)
        assert len(result) == 2
        assert result[0]["score"] == 8.0  # Sorted by score DESC
        assert result[1]["score"] == 6.0

        # Refresh replaces old items
        count = await mixin.refresh_hot_index(
            [
                {
                    "slot": 0,
                    "category": "hypothesis",
                    "neuron_id": "n-3",
                    "summary": "New hyp",
                    "confidence": 0.5,
                    "score": 9.0,
                },
            ]
        )
        assert count == 1
        result = await mixin.get_hot_index()
        assert len(result) == 1

        await db.close()


class TestSQLiteKnowledgeGaps:
    """Integration tests for knowledge_gaps storage methods."""

    @pytest.mark.asyncio
    async def test_add_and_list(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE knowledge_gaps (
            id TEXT PRIMARY KEY, brain_id TEXT NOT NULL,
            topic TEXT NOT NULL, detected_at TEXT NOT NULL,
            detection_source TEXT NOT NULL,
            related_neuron_ids TEXT DEFAULT '[]',
            resolved_at TEXT, resolved_by_neuron_id TEXT,
            priority REAL DEFAULT 0.5
        )""")

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_conn = lambda: db  # type: ignore[assignment]
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        gap_id = await mixin.add_knowledge_gap(
            topic="Missing auth docs",
            detection_source="user_flagged",
            priority=0.8,
            related_neuron_ids=["n-1", "n-2"],
        )
        assert gap_id  # Non-empty UUID

        gaps = await mixin.list_knowledge_gaps()
        assert len(gaps) == 1
        assert gaps[0]["topic"] == "Missing auth docs"
        assert gaps[0]["related_neuron_ids"] == ["n-1", "n-2"]
        assert gaps[0]["priority"] == 0.8

        await db.close()

    @pytest.mark.asyncio
    async def test_resolve_gap(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE knowledge_gaps (
            id TEXT PRIMARY KEY, brain_id TEXT NOT NULL,
            topic TEXT NOT NULL, detected_at TEXT NOT NULL,
            detection_source TEXT NOT NULL,
            related_neuron_ids TEXT DEFAULT '[]',
            resolved_at TEXT, resolved_by_neuron_id TEXT,
            priority REAL DEFAULT 0.5
        )""")

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_conn = lambda: db  # type: ignore[assignment]
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        gap_id = await mixin.add_knowledge_gap(topic="Test gap", detection_source="recall_miss")

        # Resolve it
        success = await mixin.resolve_knowledge_gap(gap_id, resolved_by_neuron_id="n-42")
        assert success

        # Should not appear in unresolved list
        gaps = await mixin.list_knowledge_gaps(include_resolved=False)
        assert len(gaps) == 0

        # Should appear with include_resolved
        gaps = await mixin.list_knowledge_gaps(include_resolved=True)
        assert len(gaps) == 1
        assert gaps[0]["resolved_at"] is not None
        assert gaps[0]["resolved_by_neuron_id"] == "n-42"

        # Double resolve should return False
        success = await mixin.resolve_knowledge_gap(gap_id)
        assert not success

        await db.close()

    @pytest.mark.asyncio
    async def test_get_gap(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE knowledge_gaps (
            id TEXT PRIMARY KEY, brain_id TEXT NOT NULL,
            topic TEXT NOT NULL, detected_at TEXT NOT NULL,
            detection_source TEXT NOT NULL,
            related_neuron_ids TEXT DEFAULT '[]',
            resolved_at TEXT, resolved_by_neuron_id TEXT,
            priority REAL DEFAULT 0.5
        )""")

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_conn = lambda: db  # type: ignore[assignment]
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        gap_id = await mixin.add_knowledge_gap(
            topic="Specific gap", detection_source="stale_schema", priority=0.4
        )

        gap = await mixin.get_knowledge_gap(gap_id)
        assert gap is not None
        assert gap["topic"] == "Specific gap"
        assert gap["detection_source"] == "stale_schema"

        # Non-existent
        assert await mixin.get_knowledge_gap("nonexistent") is None

        await db.close()
