"""Tests for nmem_edit and nmem_forget MCP tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.mcp.server import MCPServer


def _make_server() -> MCPServer:
    """Create a test server with mocked config."""
    server = MCPServer.__new__(MCPServer)
    server._config = MagicMock()
    server._config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    server._config.safety = MagicMock(auto_redact_min_severity=3)
    server._config.auto = MagicMock(enabled=False)
    server._config.dedup = MagicMock(enabled=False)
    server._config.tool_tier = MagicMock(tier="full")
    server._storage = None
    server._hooks = None
    server._eternal_trigger_count = 0
    return server


class TestNmemEdit:
    """Tests for the nmem_edit handler."""

    @pytest.mark.asyncio
    async def test_edit_missing_memory_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_edit", {})
        assert "error" in result
        assert "memory_id" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_no_changes(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_edit", {"memory_id": "abc"})
        assert "error" in result
        assert "At least one" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_invalid_type(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_edit", {"memory_id": "abc", "type": "invalid_type"})
        assert "error" in result
        assert "Invalid memory type" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_content_too_long(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_edit", {"memory_id": "abc", "content": "x" * 200_000})
        assert "error" in result
        assert "too long" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_edit", {"memory_id": "nonexistent", "type": "fact"})
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_typed_memory_success(self) -> None:
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.DECISION,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = Fiber.create(
            neuron_ids={"neuron-1"},
            synapse_ids=set(),
            anchor_neuron_id="neuron-1",
            fiber_id="fiber-1",
        )

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_typed_memory = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_edit", {"memory_id": "fiber-1", "type": "fact", "priority": 8}
        )
        assert result["status"] == "edited"
        assert any("type:" in c for c in result["changes"])
        assert any("priority:" in c for c in result["changes"])
        storage.update_typed_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_content_updates_anchor_neuron(self) -> None:
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
        from neural_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()
        fiber.anchor_neuron_id = "neuron-1"
        anchor = Neuron.create(type=NeuronType.CONCEPT, content="old content")

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=anchor)
        storage.update_neuron = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_edit", {"memory_id": "fiber-1", "content": "new corrected content"}
        )
        assert result["status"] == "edited"
        assert any("content updated" in c for c in result["changes"])
        storage.update_neuron.assert_awaited_once()


class TestNmemForget:
    """Tests for the nmem_forget handler."""

    @pytest.mark.asyncio
    async def test_forget_missing_memory_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_forget", {})
        assert "error" in result
        assert "memory_id" in result["error"]

    @pytest.mark.asyncio
    async def test_forget_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_forget", {"memory_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_soft_delete_success(self) -> None:
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_typed_memory = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_forget", {"memory_id": "fiber-1", "reason": "completed"}
        )
        assert result["status"] == "soft_deleted"
        storage.update_typed_memory.assert_awaited_once()
        # Verify expires_at was set
        updated_tm = storage.update_typed_memory.call_args[0][0]
        assert updated_tm.expires_at is not None

    @pytest.mark.asyncio
    async def test_hard_delete_success(self) -> None:
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.delete_typed_memory = AsyncMock()
        storage.delete_fiber = AsyncMock()
        storage.batch_save = AsyncMock()
        storage.disable_auto_save = MagicMock()
        storage.enable_auto_save = MagicMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_forget", {"memory_id": "fiber-1", "hard": True, "confirm": True}
        )
        assert result["status"] == "hard_deleted"
        storage.delete_typed_memory.assert_awaited_once()
        storage.delete_fiber.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hard_delete_requires_confirm(self) -> None:
        """#79: hard delete without confirm returns a preview, deletes nothing."""
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=MagicMock())
        storage.delete_typed_memory = AsyncMock()
        storage.delete_fiber = AsyncMock()
        storage.disable_auto_save = MagicMock()
        storage.enable_auto_save = MagicMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_forget", {"memory_id": "fiber-1", "hard": True})
        assert result["status"] == "pending_confirmation"
        storage.delete_typed_memory.assert_not_awaited()
        storage.delete_fiber.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hard_delete_neuron_only(self) -> None:
        """Hard delete on a neuron ID (not fiber) should work."""
        from neural_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="orphan")
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.delete_neuron = AsyncMock(return_value=True)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_forget", {"memory_id": neuron.id, "hard": True, "confirm": True}
        )
        assert result["status"] == "hard_deleted"
        storage.delete_neuron.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hard_delete_neuron_only_requires_confirm(self) -> None:
        """#79: neuron-only hard delete also gates on confirm."""
        from neural_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="orphan")
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.delete_neuron = AsyncMock(return_value=True)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_forget", {"memory_id": neuron.id, "hard": True})
        assert result["status"] == "pending_confirmation"
        storage.delete_neuron.assert_not_awaited()
