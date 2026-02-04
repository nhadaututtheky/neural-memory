"""Tests for MCP server."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.server import MCPServer, create_mcp_server, handle_message


class TestMCPServer:
    """Tests for MCPServer class."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.CLIConfig") as mock_config:
            mock_config.load.return_value = MagicMock(
                get_brain_path=MagicMock(return_value="/tmp/test-brain")
            )
            return MCPServer()

    def test_create_mcp_server(self) -> None:
        """Test server factory function."""
        with patch("neural_memory.mcp.server.CLIConfig") as mock_config:
            mock_config.load.return_value = MagicMock(
                get_brain_path=MagicMock(return_value="/tmp/test-brain")
            )
            server = create_mcp_server()
            assert isinstance(server, MCPServer)

    def test_get_tools(self, server: MCPServer) -> None:
        """Test that get_tools returns all expected tools."""
        tools = server.get_tools()

        assert len(tools) == 5
        tool_names = {tool["name"] for tool in tools}
        assert tool_names == {
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_todo",
            "nmem_stats",
        }

    def test_tool_schemas(self, server: MCPServer) -> None:
        """Test that tool schemas are valid."""
        tools = server.get_tools()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]

    def test_remember_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_remember tool schema."""
        tools = server.get_tools()
        remember_tool = next(t for t in tools if t["name"] == "nmem_remember")

        schema = remember_tool["inputSchema"]
        assert "content" in schema["properties"]
        assert "type" in schema["properties"]
        assert "priority" in schema["properties"]
        assert "tags" in schema["properties"]
        assert "expires_days" in schema["properties"]
        assert schema["required"] == ["content"]

    def test_recall_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_recall tool schema."""
        tools = server.get_tools()
        recall_tool = next(t for t in tools if t["name"] == "nmem_recall")

        schema = recall_tool["inputSchema"]
        assert "query" in schema["properties"]
        assert "depth" in schema["properties"]
        assert "max_tokens" in schema["properties"]
        assert "min_confidence" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_context_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_context tool schema."""
        tools = server.get_tools()
        context_tool = next(t for t in tools if t["name"] == "nmem_context")

        schema = context_tool["inputSchema"]
        assert "limit" in schema["properties"]
        assert "fresh_only" in schema["properties"]

    def test_todo_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_todo tool schema."""
        tools = server.get_tools()
        todo_tool = next(t for t in tools if t["name"] == "nmem_todo")

        schema = todo_tool["inputSchema"]
        assert "task" in schema["properties"]
        assert "priority" in schema["properties"]
        assert schema["required"] == ["task"]

    def test_stats_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_stats tool schema."""
        tools = server.get_tools()
        stats_tool = next(t for t in tools if t["name"] == "nmem_stats")

        schema = stats_tool["inputSchema"]
        assert schema["properties"] == {}


class TestMCPToolCalls:
    """Tests for MCP tool call execution."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance with mocked storage."""
        with patch("neural_memory.mcp.server.CLIConfig") as mock_config:
            mock_config.load.return_value = MagicMock(
                get_brain_path=MagicMock(return_value="/tmp/test-brain")
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server: MCPServer) -> None:
        """Test calling unknown tool returns error."""
        result = await server.call_tool("unknown_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_remember_tool(self, server: MCPServer) -> None:
        """Test nmem_remember tool execution."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(
            id="test-brain",
            name="test",
            config=MagicMock(),
        )
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="fiber-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool(
                "nmem_remember",
                {"content": "Test memory", "type": "fact", "priority": 7},
            )

        assert result["success"] is True
        assert result["fiber_id"] == "fiber-123"
        assert result["memory_type"] == "fact"

    @pytest.mark.asyncio
    async def test_remember_no_brain(self, server: MCPServer) -> None:
        """Test nmem_remember when no brain is configured."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_remember", {"content": "Test"})

        assert "error" in result
        assert "No brain configured" in result["error"]

    @pytest.mark.asyncio
    async def test_recall_tool(self, server: MCPServer) -> None:
        """Test nmem_recall tool execution."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(
            id="test-brain",
            name="test",
            config=MagicMock(),
        )
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=MagicMock(
                context="Test answer",
                confidence=0.85,
                neurons_activated=5,
                fibers_matched=2,
                depth_used=MagicMock(value=1),
            )
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
        ):
            result = await server.call_tool("nmem_recall", {"query": "test query"})

        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.85
        assert result["neurons_activated"] == 5

    @pytest.mark.asyncio
    async def test_recall_low_confidence(self, server: MCPServer) -> None:
        """Test nmem_recall with confidence below threshold."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(
            id="test-brain",
            name="test",
            config=MagicMock(),
        )
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=MagicMock(
                context="Weak answer",
                confidence=0.3,
                neurons_activated=2,
                fibers_matched=1,
                depth_used=MagicMock(value=1),
            )
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
        ):
            result = await server.call_tool(
                "nmem_recall", {"query": "test", "min_confidence": 0.5}
            )

        assert result["answer"] is None
        assert "No memories found" in result["message"]

    @pytest.mark.asyncio
    async def test_context_tool(self, server: MCPServer) -> None:
        """Test nmem_context tool execution."""
        mock_storage = AsyncMock()
        mock_fibers = [
            MagicMock(summary="Memory 1", anchor_neuron_id=None),
            MagicMock(summary="Memory 2", anchor_neuron_id=None),
        ]
        mock_storage.get_fibers = AsyncMock(return_value=mock_fibers)

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_context", {"limit": 5})

        assert result["count"] == 2
        assert "Memory 1" in result["context"]
        assert "Memory 2" in result["context"]

    @pytest.mark.asyncio
    async def test_context_empty(self, server: MCPServer) -> None:
        """Test nmem_context with no memories."""
        mock_storage = AsyncMock()
        mock_storage.get_fibers = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_context", {})

        assert result["count"] == 0
        assert "No memories stored" in result["context"]

    @pytest.mark.asyncio
    async def test_todo_tool(self, server: MCPServer) -> None:
        """Test nmem_todo tool (delegates to remember)."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(
            id="test-brain",
            name="test",
            config=MagicMock(),
        )
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="todo-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool(
                "nmem_todo", {"task": "Review code", "priority": 8}
            )

        assert result["success"] is True
        assert result["memory_type"] == "todo"

    @pytest.mark.asyncio
    async def test_stats_tool(self, server: MCPServer) -> None:
        """Test nmem_stats tool execution."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock()
        mock_brain.id = "test-brain"
        mock_brain.name = "my-brain"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_stats = AsyncMock(
            return_value={
                "neuron_count": 100,
                "synapse_count": 250,
                "fiber_count": 50,
            }
        )

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_stats", {})

        assert result["brain"] == "my-brain"
        assert result["neuron_count"] == 100
        assert result["synapse_count"] == 250
        assert result["fiber_count"] == 50


class TestMCPProtocol:
    """Tests for MCP protocol message handling."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.CLIConfig") as mock_config:
            mock_config.load.return_value = MagicMock(
                get_brain_path=MagicMock(return_value="/tmp/test-brain")
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_initialize_message(self, server: MCPServer) -> None:
        """Test MCP initialize message."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert response["result"]["serverInfo"]["name"] == "neural-memory"
        assert "capabilities" in response["result"]

    @pytest.mark.asyncio
    async def test_tools_list_message(self, server: MCPServer) -> None:
        """Test MCP tools/list message."""
        message = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 5

    @pytest.mark.asyncio
    async def test_tools_call_message(self, server: MCPServer) -> None:
        """Test MCP tools/call message."""
        mock_storage = AsyncMock()
        mock_storage.get_fibers = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            message = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "nmem_context", "arguments": {"limit": 5}},
            }

            response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]
        assert response["result"]["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_tools_call_error(self, server: MCPServer) -> None:
        """Test MCP tools/call error handling."""
        with patch.object(
            server, "call_tool", side_effect=Exception("Test error")
        ):
            message = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "nmem_context", "arguments": {}},
            }

            response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32000
        assert "Test error" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_notifications_initialized(self, server: MCPServer) -> None:
        """Test MCP notifications/initialized message (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }

        response = await handle_message(server, message)

        assert response is None

    @pytest.mark.asyncio
    async def test_unknown_method(self, server: MCPServer) -> None:
        """Test MCP unknown method error."""
        message = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method",
            "params": {},
        }

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]


class TestMCPStorage:
    """Tests for MCP server storage management."""

    @pytest.mark.asyncio
    async def test_get_storage_caches_instance(self) -> None:
        """Test that get_storage caches the storage instance."""
        with patch("neural_memory.mcp.server.CLIConfig") as mock_config:
            mock_config.load.return_value = MagicMock(
                get_brain_path=MagicMock(return_value="/tmp/test-brain")
            )
            server = MCPServer()

        mock_storage = AsyncMock()

        with patch(
            "neural_memory.mcp.server.PersistentStorage.load",
            return_value=mock_storage,
        ) as mock_load:
            storage1 = await server.get_storage()
            storage2 = await server.get_storage()

        # Should only load once
        mock_load.assert_called_once()
        assert storage1 is storage2
