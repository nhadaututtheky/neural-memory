"""Tests for nmem_show MCP tool and exact recall mode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.memory_types import MemoryType, Priority

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(brain_id: str = "test-brain"):
    """Build a minimal ToolHandler with mock storage."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    storage = AsyncMock()
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage._current_brain_id = brain_id

    brain_mock = MagicMock(id=brain_id, config=MagicMock())
    storage.get_brain = AsyncMock(return_value=brain_mock)
    storage.disable_auto_save = MagicMock()

    class TestHandler(ToolHandler):
        config = MagicMock()
        config.auto = MagicMock(enabled=False)
        hooks = AsyncMock()
        hooks.emit = AsyncMock(return_value=None)

        async def get_storage(self):
            return storage

        def _fire_eternal_trigger(self, content: str) -> None:
            pass

        async def _check_maintenance(self):
            return None

        def _get_maintenance_hint(self, pulse):
            return None

        async def _passive_capture(self, text: str) -> None:
            pass

        def get_update_hint(self):
            return None

        async def _get_active_session(self, storage):
            return None

        async def _check_onboarding(self):
            return None

        async def _surface_pending_alerts(self):
            return None

        async def _record_tool_action(self, action: str, detail: str) -> None:
            pass

        async def _check_cross_language_hint(self, *args, **kwargs):
            return None

    return TestHandler(), storage


# ---------------------------------------------------------------------------
# nmem_show tests
# ---------------------------------------------------------------------------


class TestShow:
    @pytest.mark.asyncio
    async def test_show_missing_id(self) -> None:
        handler, _ = _make_handler()
        result = await handler._show({})
        assert "error" in result
        assert "memory_id" in result["error"]

    @pytest.mark.asyncio
    async def test_show_not_found(self) -> None:
        handler, storage = _make_handler()
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=None)

        result = await handler._show({"memory_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_show_fiber_path(self) -> None:
        handler, storage = _make_handler()

        typed_mem = MagicMock()
        typed_mem.memory_type = MemoryType.FACT
        typed_mem.priority = Priority.NORMAL
        typed_mem.tags = {"tag1", "tag2"}
        typed_mem.trust_score = 0.9
        typed_mem.expires_at = None

        fiber = MagicMock()
        fiber.anchor_neuron_id = "anchor-1"
        fiber.neuron_count = 3
        fiber.summary = "A summary"
        fiber.metadata = {}
        fiber.created_at = MagicMock(isoformat=MagicMock(return_value="2026-01-01T00:00:00"))

        anchor = MagicMock()
        anchor.content = "Exact verbatim content here"

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=anchor)
        storage.get_synapses = AsyncMock(return_value=[])

        result = await handler._show({"memory_id": "fiber-123"})
        assert result["content"] == "Exact verbatim content here"
        assert result["memory_type"] == "fact"
        assert result["priority"] == 5
        assert result["anchor_neuron_id"] == "anchor-1"
        assert "synapses" in result

    @pytest.mark.asyncio
    async def test_show_neuron_path(self) -> None:
        handler, storage = _make_handler()

        storage.get_typed_memory = AsyncMock(return_value=None)
        neuron = MagicMock()
        neuron.content = "Raw neuron content"
        neuron.type = MagicMock(value="concept")
        neuron.created_at = MagicMock(isoformat=MagicMock(return_value="2026-01-01T00:00:00"))
        neuron.metadata = {"key": "val"}

        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.get_synapses = AsyncMock(return_value=[])

        result = await handler._show({"memory_id": "neuron-456"})
        assert result["content"] == "Raw neuron content"
        assert result["neuron_type"] == "concept"

    @pytest.mark.asyncio
    async def test_show_no_brain(self) -> None:
        handler, storage = _make_handler()
        storage.brain_id = None

        result = await handler._show({"memory_id": "abc"})
        assert "error" in result
        assert "brain" in result["error"].lower()


# ---------------------------------------------------------------------------
# Exact recall mode tests
# ---------------------------------------------------------------------------


class TestExactRecall:
    @pytest.mark.asyncio
    async def test_recall_invalid_mode(self) -> None:
        handler, _ = _make_handler()
        result = await handler._recall({"query": "test", "mode": "invalid"})
        assert "error" in result
        assert "mode" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_recall_exact_returns_raw_content(self) -> None:
        handler, storage = _make_handler()

        # Mock ReflexPipeline
        mock_result = MagicMock()
        mock_result.context = "summarized version"
        mock_result.confidence = 0.8
        mock_result.neurons_activated = 5
        mock_result.fibers_matched = ["fib-1", "fib-2"]
        mock_result.depth_used = MagicMock(value=1)
        mock_result.tokens_used = 100
        mock_result.score_breakdown = None
        mock_result.metadata = {}

        fiber1 = MagicMock()
        fiber1.anchor_neuron_id = "n1"
        fiber1.metadata = {}
        fiber1.created_at = MagicMock(isoformat=MagicMock(return_value="2026-01-01"))
        fiber2 = MagicMock()
        fiber2.anchor_neuron_id = "n2"
        fiber2.metadata = {}
        fiber2.created_at = MagicMock(isoformat=MagicMock(return_value="2026-01-02"))

        neuron1 = MagicMock(content="Raw content A")
        neuron2 = MagicMock(content="Raw content B")

        tm1 = MagicMock()
        tm1.memory_type = MemoryType.FACT
        tm1.priority = Priority.NORMAL
        tm1.tags = {"t1"}
        tm2 = MagicMock()
        tm2.memory_type = MemoryType.DECISION
        tm2.priority = Priority.HIGH
        tm2.tags = {"t2"}

        storage.get_fiber = AsyncMock(side_effect=lambda fid: fiber1 if fid == "fib-1" else fiber2)
        storage.get_neuron = AsyncMock(side_effect=lambda nid: neuron1 if nid == "n1" else neuron2)
        storage.get_typed_memory = AsyncMock(side_effect=lambda fid: tm1 if fid == "fib-1" else tm2)

        with patch("neural_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls:
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            result = await handler._recall({"query": "test query", "mode": "exact"})

        assert result["mode"] == "exact"
        assert len(result["memories"]) == 2
        assert result["memories"][0]["content"] == "Raw content A"
        assert result["memories"][1]["content"] == "Raw content B"
        assert result["memories"][0]["memory_type"] == "fact"
        assert result["memories"][1]["memory_type"] == "decision"

    @pytest.mark.asyncio
    async def test_recall_default_mode_unchanged(self) -> None:
        handler, storage = _make_handler()

        mock_result = MagicMock()
        mock_result.context = "formatted context"
        mock_result.confidence = 0.7
        mock_result.neurons_activated = 3
        mock_result.fibers_matched = ["fib-1"]
        mock_result.depth_used = MagicMock(value=1)
        mock_result.tokens_used = 50
        mock_result.score_breakdown = None
        mock_result.metadata = {}

        with patch("neural_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls:
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            result = await handler._recall({"query": "test query"})

        assert "answer" in result
        assert result["answer"] == "formatted context"
        assert "mode" not in result  # default mode doesn't add mode field
