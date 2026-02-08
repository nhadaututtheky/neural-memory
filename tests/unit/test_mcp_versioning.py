"""Tests for MCP versioning and transplant tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.server import MCPServer


class TestVersionTool:
    """Test nmem_version MCP tool."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_version_create(self, server: MCPServer) -> None:
        """Should create a brain version snapshot."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool(
                "nmem_version",
                {"action": "create", "name": "v1-test", "description": "Test snapshot"},
            )

        assert "error" not in result
        assert result.get("success") is True
        assert result["version_name"] == "v1-test"
        assert result["version_number"] == 1

    @pytest.mark.asyncio
    async def test_version_list_empty(self, server: MCPServer) -> None:
        """Should return empty list when no versions exist."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool("nmem_version", {"action": "list"})

        assert "error" not in result
        assert result["count"] == 0
        assert result["versions"] == []

    @pytest.mark.asyncio
    async def test_version_list_after_create(self, server: MCPServer) -> None:
        """Should list created versions."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            await server.call_tool("nmem_version", {"action": "create", "name": "snap-1"})
            await server.call_tool("nmem_version", {"action": "create", "name": "snap-2"})
            result = await server.call_tool("nmem_version", {"action": "list"})

        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_version_create_missing_name(self, server: MCPServer) -> None:
        """Should error when name is missing for create."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool("nmem_version", {"action": "create"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_version_rollback_nonexistent(self, server: MCPServer) -> None:
        """Should return error for nonexistent version."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool(
                "nmem_version",
                {"action": "rollback", "version_id": "nonexistent-id"},
            )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_version_rollback_missing_id(self, server: MCPServer) -> None:
        """Should error when version_id is missing for rollback."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool("nmem_version", {"action": "rollback"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_version_diff_missing_ids(self, server: MCPServer) -> None:
        """Should error when diff IDs are missing."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool("nmem_version", {"action": "diff"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_version_unknown_action(self, server: MCPServer) -> None:
        """Should error on unknown action."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        with patch.object(server, "get_storage", return_value=storage):
            result = await server.call_tool("nmem_version", {"action": "unknown"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_version_no_brain(self, server: MCPServer) -> None:
        """Should error when no brain is configured."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "nonexistent"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_version", {"action": "list"})

        assert "error" in result


class TestTransplantTool:
    """Test nmem_transplant MCP tool."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_transplant_nonexistent_source(self, server: MCPServer) -> None:
        """Should error when source brain doesn't exist."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain")
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.find_brain_by_name = AsyncMock(return_value=None)

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_transplant", {"source_brain": "nonexistent-brain"}
            )

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_transplant_missing_source(self, server: MCPServer) -> None:
        """Should error when source_brain is missing."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain")
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_transplant", {})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_transplant_no_brain(self, server: MCPServer) -> None:
        """Should error when no brain is configured."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "nonexistent"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_transplant", {"source_brain": "some-brain"})

        assert "error" in result

    def test_transplant_tool_schema(self, server: MCPServer) -> None:
        """Transplant tool should have correct schema fields."""
        tools = server.get_tools()
        transplant_tool = next(t for t in tools if t["name"] == "nmem_transplant")
        props = transplant_tool["inputSchema"]["properties"]
        assert "source_brain" in props
        assert "tags" in props
        assert "memory_types" in props
        assert "strategy" in props

    def test_version_tool_schema(self, server: MCPServer) -> None:
        """Version tool should have correct schema fields."""
        tools = server.get_tools()
        version_tool = next(t for t in tools if t["name"] == "nmem_version")
        props = version_tool["inputSchema"]["properties"]
        assert "action" in props
        assert "name" in props
        assert "version_id" in props
        assert "from_version" in props
        assert "to_version" in props
