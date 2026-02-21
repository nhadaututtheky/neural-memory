"""Tests for memory expiry alerts feature.

Tests storage methods (get_expiring_memories_for_fibers, get_expiring_memory_count)
and MCP handler enrichment (warn_expiry_days on recall/context).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from neural_memory.mcp.maintenance_handler import HintSeverity, _evaluate_thresholds
from neural_memory.mcp.server import MCPServer
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


def _make_typed_memory(
    fiber_id: str,
    expires_in_days: int | None = None,
    memory_type: MemoryType = MemoryType.FACT,
    priority: int = 5,
) -> TypedMemory:
    """Create a TypedMemory with optional expiry."""
    now = utcnow()
    return TypedMemory(
        fiber_id=fiber_id,
        memory_type=memory_type,
        priority=Priority.from_int(priority),
        provenance=Provenance(source="test"),
        expires_at=(now + timedelta(days=expires_in_days)) if expires_in_days is not None else None,
        created_at=now,
    )


def _make_storage() -> InMemoryStorage:
    """Create an InMemoryStorage with brain context set."""
    from neural_memory.storage.memory_store import InMemoryStorage

    storage = InMemoryStorage()
    storage.set_brain("test-brain")
    storage._typed_memories.setdefault("test-brain", {})
    return storage


class TestExpiringMemoriesStorage:
    """Tests for get_expiring_memories_for_fibers and get_expiring_memory_count."""

    @pytest.mark.asyncio
    async def test_empty_fiber_ids(self) -> None:
        """Empty fiber_ids returns empty list."""
        storage = _make_storage()
        result = await storage.get_expiring_memories_for_fibers([], within_days=7)
        assert result == []

    @pytest.mark.asyncio
    async def test_within_window(self) -> None:
        """Memory expiring in 3 days found when within_days=7."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        tm = _make_typed_memory("fiber-1", expires_in_days=3)
        storage._typed_memories[brain_id]["fiber-1"] = tm

        result = await storage.get_expiring_memories_for_fibers(["fiber-1"], within_days=7)
        assert len(result) == 1
        assert result[0].fiber_id == "fiber-1"

    @pytest.mark.asyncio
    async def test_outside_window(self) -> None:
        """Memory expiring in 30 days NOT found when within_days=7."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        tm = _make_typed_memory("fiber-1", expires_in_days=30)
        storage._typed_memories[brain_id]["fiber-1"] = tm

        result = await storage.get_expiring_memories_for_fibers(["fiber-1"], within_days=7)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_already_expired_excluded(self) -> None:
        """Already-expired memories are excluded (expires_at < now)."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        now = utcnow()
        tm = TypedMemory(
            fiber_id="fiber-1",
            memory_type=MemoryType.FACT,
            priority=Priority.from_int(5),
            provenance=Provenance(source="test"),
            expires_at=now - timedelta(days=1),
            created_at=now - timedelta(days=10),
        )
        storage._typed_memories[brain_id]["fiber-1"] = tm

        result = await storage.get_expiring_memories_for_fibers(["fiber-1"], within_days=7)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_expiry_excluded(self) -> None:
        """Memories with no expiry are excluded."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        tm = _make_typed_memory("fiber-1", expires_in_days=None)
        storage._typed_memories[brain_id]["fiber-1"] = tm

        result = await storage.get_expiring_memories_for_fibers(["fiber-1"], within_days=7)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fiber_id_filtering(self) -> None:
        """Only returns memories for requested fiber IDs."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        tm1 = _make_typed_memory("fiber-1", expires_in_days=3)
        tm2 = _make_typed_memory("fiber-2", expires_in_days=3)
        storage._typed_memories[brain_id]["fiber-1"] = tm1
        storage._typed_memories[brain_id]["fiber-2"] = tm2

        result = await storage.get_expiring_memories_for_fibers(["fiber-1"], within_days=7)
        assert len(result) == 1
        assert result[0].fiber_id == "fiber-1"

    @pytest.mark.asyncio
    async def test_expiring_memory_count(self) -> None:
        """Count matches number of soon-to-expire memories."""
        storage = _make_storage()
        brain_id = storage._get_brain_id()

        tm1 = _make_typed_memory("fiber-1", expires_in_days=3)
        tm2 = _make_typed_memory("fiber-2", expires_in_days=5)
        tm3 = _make_typed_memory("fiber-3", expires_in_days=30)  # outside window
        storage._typed_memories[brain_id]["fiber-1"] = tm1
        storage._typed_memories[brain_id]["fiber-2"] = tm2
        storage._typed_memories[brain_id]["fiber-3"] = tm3

        count = await storage.get_expiring_memory_count(within_days=7)
        assert count == 2


class TestRecallExpiryWarnings:
    """Tests for warn_expiry_days on nmem_recall."""

    @pytest.fixture
    def server(self) -> MCPServer:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_recall_without_warn_expiry_days(self, server: MCPServer) -> None:
        """No expiry_warnings key when param omitted."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(
            return_value=MagicMock(id="test-brain", config=MagicMock())
        )

        # Mock the recall pipeline
        mock_result = MagicMock()
        mock_result.context = "Some memory context"
        mock_result.confidence = 0.9
        mock_result.neurons_activated = 3
        mock_result.fibers_matched = ["f-1"]
        mock_result.depth_used = MagicMock(value=1)
        mock_result.tokens_used = 10
        mock_result.score_breakdown = None
        mock_result.metadata = {}

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline") as mock_pipeline_cls,
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            result = await server.call_tool("nmem_recall", {"query": "test"})

        assert "expiry_warnings" not in result

    @pytest.mark.asyncio
    async def test_recall_with_warn_expiry_days(self, server: MCPServer) -> None:
        """expiry_warnings present when warn_expiry_days set and memories expiring."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(
            return_value=MagicMock(id="test-brain", config=MagicMock())
        )

        expiring_tm = _make_typed_memory("f-1", expires_in_days=3, memory_type=MemoryType.DECISION)
        mock_storage.get_expiring_memories_for_fibers = AsyncMock(return_value=[expiring_tm])

        mock_result = MagicMock()
        mock_result.context = "Some memory"
        mock_result.confidence = 0.9
        mock_result.neurons_activated = 3
        mock_result.fibers_matched = ["f-1"]
        mock_result.depth_used = MagicMock(value=1)
        mock_result.tokens_used = 10
        mock_result.score_breakdown = None
        mock_result.metadata = {}

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline") as mock_pipeline_cls,
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            result = await server.call_tool("nmem_recall", {"query": "test", "warn_expiry_days": 7})

        assert "expiry_warnings" in result
        assert len(result["expiry_warnings"]) == 1
        warning = result["expiry_warnings"][0]
        assert warning["fiber_id"] == "f-1"
        assert warning["memory_type"] == "decision"
        assert "days_until_expiry" in warning
        assert "priority" in warning
        assert "suggestion" in warning


class TestMaintenanceExpiryHint:
    """Tests for expiring-soon maintenance hint."""

    @staticmethod
    def _make_cfg() -> MagicMock:
        """Create a MaintenanceConfig mock with all thresholds set high."""
        return MagicMock(
            neuron_warn_threshold=999999,
            fiber_warn_threshold=999999,
            synapse_warn_threshold=999999,
            connectivity_warn_threshold=0.0,
            orphan_ratio_threshold=1.0,
            expired_memory_warn_threshold=999999,
            stale_fiber_ratio_threshold=1.0,
        )

    def test_expiring_soon_hint_present(self) -> None:
        """Hint added when expiring_soon_count > 0."""
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=10,
            synapse_count=10,
            connectivity=0.5,
            orphan_ratio=0.1,
            expired_memory_count=0,
            expiring_soon_count=5,
            stale_fiber_ratio=0.0,
            cfg=self._make_cfg(),
        )
        expiry_hints = [h for h in hints if "expiring within 7 days" in h.message]
        assert len(expiry_hints) == 1
        assert expiry_hints[0].severity == HintSeverity.LOW
        assert "5 memories" in expiry_hints[0].message

    def test_no_hint_when_zero_expiring(self) -> None:
        """No hint when expiring_soon_count is 0."""
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=10,
            synapse_count=10,
            connectivity=0.5,
            orphan_ratio=0.1,
            expired_memory_count=0,
            expiring_soon_count=0,
            stale_fiber_ratio=0.0,
            cfg=self._make_cfg(),
        )
        expiry_hints = [h for h in hints if "expiring within 7 days" in h.message]
        assert len(expiry_hints) == 0
