"""Tests for MCP server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.auto_capture import analyze_text_for_memories
from neural_memory.mcp.server import MCPServer, create_mcp_server, handle_message


class TestMCPServer:
    """Tests for MCPServer class."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    def test_create_mcp_server(self) -> None:
        """Test server factory function."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            server = create_mcp_server()
            assert isinstance(server, MCPServer)

    def test_get_tools(self, server: MCPServer) -> None:
        """Test that get_tools returns all expected tools."""
        tools = server.get_tools()

        assert len(tools) == 8
        tool_names = {tool["name"] for tool in tools}
        assert tool_names == {
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_todo",
            "nmem_stats",
            "nmem_auto",
            "nmem_suggest",
            "nmem_session",
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

    def test_suggest_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_suggest tool schema."""
        tools = server.get_tools()
        suggest_tool = next(t for t in tools if t["name"] == "nmem_suggest")

        schema = suggest_tool["inputSchema"]
        assert "prefix" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "type_filter" in schema["properties"]
        assert schema["required"] == ["prefix"]

    def test_session_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_session tool schema."""
        tools = server.get_tools()
        session_tool = next(t for t in tools if t["name"] == "nmem_session")

        schema = session_tool["inputSchema"]
        assert "action" in schema["properties"]
        assert "feature" in schema["properties"]
        assert "task" in schema["properties"]
        assert "progress" in schema["properties"]
        assert "notes" in schema["properties"]
        assert schema["required"] == ["action"]


class TestMCPToolCalls:
    """Tests for MCP tool call execution."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance with mocked storage."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
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
                tokens_used=12,
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
        assert isinstance(result["tokens_used"], int)
        assert result["tokens_used"] >= 0

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
            result = await server.call_tool("nmem_recall", {"query": "test", "min_confidence": 0.5})

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
        assert isinstance(result["tokens_used"], int)
        assert result["tokens_used"] >= 0

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
            result = await server.call_tool("nmem_todo", {"task": "Review code", "priority": 8})

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
        mock_storage.get_enhanced_stats = AsyncMock(
            return_value={
                "neuron_count": 100,
                "synapse_count": 250,
                "fiber_count": 50,
                "db_size_bytes": 12345,
                "today_fibers_count": 3,
                "hot_neurons": [],
                "synapse_stats": {},
                "neuron_type_breakdown": {},
                "oldest_memory": None,
                "newest_memory": None,
            }
        )

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_stats", {})

        assert result["brain"] == "my-brain"
        assert result["neuron_count"] == 100
        assert result["synapse_count"] == 250
        assert result["fiber_count"] == 50
        assert result["db_size_bytes"] == 12345
        assert result["today_fibers_count"] == 3

    @pytest.mark.asyncio
    async def test_auto_tool_status(self, server: MCPServer) -> None:
        """Test nmem_auto status action."""
        result = await server.call_tool("nmem_auto", {"action": "status"})

        assert "enabled" in result
        assert "capture_decisions" in result
        assert "capture_errors" in result

    @pytest.mark.asyncio
    async def test_auto_tool_analyze(self, server: MCPServer) -> None:
        """Test nmem_auto analyze action."""
        text = "We decided to use PostgreSQL for the database. TODO: Set up migrations."
        result = await server.call_tool("nmem_auto", {"action": "analyze", "text": text})

        assert "detected" in result
        assert len(result["detected"]) >= 1  # Should detect at least the TODO

    @pytest.mark.asyncio
    async def test_auto_tool_analyze_errors(self, server: MCPServer) -> None:
        """Test nmem_auto detects error patterns."""
        text = "The error was: connection timeout. The issue is that the server is down."
        result = await server.call_tool("nmem_auto", {"action": "analyze", "text": text})

        assert "detected" in result
        detected_types = [d["type"] for d in result["detected"]]
        assert "error" in detected_types

    @pytest.mark.asyncio
    async def test_auto_tool_analyze_empty(self, server: MCPServer) -> None:
        """Test nmem_auto with no detectable content."""
        result = await server.call_tool("nmem_auto", {"action": "analyze", "text": "Hello world"})

        assert result["detected"] == []

    @pytest.mark.asyncio
    async def test_auto_tool_process(self) -> None:
        """Test nmem_auto process action (analyze + save)."""
        # Create server with proper auto config mocked
        mock_auto_config = MagicMock(
            enabled=True,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            min_confidence=0.7,
        )
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=mock_auto_config,
            )
            server = MCPServer()

        mock_storage = AsyncMock()
        mock_brain = MagicMock(
            id="test-brain",
            name="test",
            config=MagicMock(),
        )
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="auto-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.MemoryEncoder", return_value=mock_encoder),
        ):
            text = "We decided to use Redis for caching. TODO: Set up Redis server."
            result = await server.call_tool("nmem_auto", {"action": "process", "text": text})

        assert "saved" in result
        assert result["saved"] >= 1  # Should save at least the decision or TODO

    @pytest.mark.asyncio
    async def test_auto_tool_process_empty(self, server: MCPServer) -> None:
        """Test nmem_auto process with no detectable content."""
        result = await server.call_tool("nmem_auto", {"action": "process", "text": "Hello world"})

        assert result["saved"] == 0

    @pytest.mark.asyncio
    async def test_suggest_basic(self, server: MCPServer) -> None:
        """Test nmem_suggest returns matching suggestions."""
        mock_storage = AsyncMock()
        mock_storage.suggest_neurons = AsyncMock(
            return_value=[
                {
                    "neuron_id": "n-1",
                    "content": "API design patterns",
                    "type": "concept",
                    "access_frequency": 5,
                    "activation_level": 0.8,
                    "score": 1.5,
                },
            ]
        )

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_suggest", {"prefix": "API"})

        assert result["count"] == 1
        assert result["suggestions"][0]["content"] == "API design patterns"
        assert result["suggestions"][0]["neuron_id"] == "n-1"
        assert result["suggestions"][0]["score"] == 1.5
        assert isinstance(result["tokens_used"], int)
        assert result["tokens_used"] >= 0

    @pytest.mark.asyncio
    async def test_suggest_empty_prefix(self, server: MCPServer) -> None:
        """Test nmem_suggest with empty prefix returns empty."""
        result = await server.call_tool("nmem_suggest", {"prefix": ""})

        assert result["suggestions"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_suggest_with_type_filter(self, server: MCPServer) -> None:
        """Test nmem_suggest with type_filter."""
        mock_storage = AsyncMock()
        mock_storage.suggest_neurons = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_suggest", {"prefix": "auth", "type_filter": "concept"}
            )

        mock_storage.suggest_neurons.assert_called_once()
        call_kwargs = mock_storage.suggest_neurons.call_args
        assert call_kwargs.kwargs["type_filter"].value == "concept"
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_suggest_respects_limit(self, server: MCPServer) -> None:
        """Test nmem_suggest respects limit parameter."""
        mock_storage = AsyncMock()
        mock_storage.suggest_neurons = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            await server.call_tool("nmem_suggest", {"prefix": "test", "limit": 2})

        call_kwargs = mock_storage.suggest_neurons.call_args
        assert call_kwargs.kwargs["limit"] == 2

    @pytest.mark.asyncio
    async def test_session_get_empty(self, server: MCPServer) -> None:
        """Test nmem_session get with no active session."""
        mock_storage = AsyncMock()
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_session", {"action": "get"})

        assert result["active"] is False
        assert "No active session" in result["message"]

    @pytest.mark.asyncio
    async def test_session_set_and_get(self, server: MCPServer) -> None:
        """Test nmem_session set then get roundtrip."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", name="test", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        mock_fiber = MagicMock(id="session-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool(
                "nmem_session",
                {
                    "action": "set",
                    "feature": "auth",
                    "task": "login form",
                    "progress": 0.3,
                    "notes": "Working on OAuth",
                },
            )

        assert result["active"] is True
        assert result["feature"] == "auth"
        assert result["task"] == "login form"
        assert result["progress"] == 0.3
        assert result["notes"] == "Working on OAuth"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_session_end(self, server: MCPServer) -> None:
        """Test nmem_session end creates summary."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", name="test", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_existing = MagicMock(
            metadata={
                "feature": "auth",
                "task": "login form",
                "progress": 0.75,
                "started_at": "2026-02-06T10:00:00",
            }
        )
        mock_storage.find_typed_memories = AsyncMock(return_value=[mock_existing])

        mock_fiber = MagicMock(id="summary-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool("nmem_session", {"action": "end"})

        assert result["active"] is False
        assert "auth" in result["summary"]
        assert "75%" in result["summary"]
        assert "message" in result

    @pytest.mark.asyncio
    async def test_session_end_no_active(self, server: MCPServer) -> None:
        """Test nmem_session end with no active session."""
        mock_storage = AsyncMock()
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_session", {"action": "end"})

        assert result["active"] is False
        assert "No active session" in result["message"]


class TestMCPProtocol:
    """Tests for MCP protocol message handling."""

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
        assert len(response["result"]["tools"]) == 8

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
        with patch.object(server, "call_tool", side_effect=Exception("Test error")):
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


class TestMCPResources:
    """Tests for MCP server resources (system prompts)."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    def test_get_resources(self, server: MCPServer) -> None:
        """Test that get_resources returns available prompts."""
        resources = server.get_resources()

        assert len(resources) == 2
        uris = {r["uri"] for r in resources}
        assert "neuralmemory://prompt/system" in uris
        assert "neuralmemory://prompt/compact" in uris

    def test_get_resource_content_system(self, server: MCPServer) -> None:
        """Test getting system prompt content."""
        content = server.get_resource_content("neuralmemory://prompt/system")

        assert content is not None
        assert "NeuralMemory" in content
        assert "nmem_remember" in content

    def test_get_resource_content_compact(self, server: MCPServer) -> None:
        """Test getting compact prompt content."""
        content = server.get_resource_content("neuralmemory://prompt/compact")

        assert content is not None
        assert len(content) < 1000  # Compact should be shorter

    def test_get_resource_content_unknown(self, server: MCPServer) -> None:
        """Test getting unknown resource returns None."""
        content = server.get_resource_content("neuralmemory://unknown")

        assert content is None

    @pytest.mark.asyncio
    async def test_resources_list_message(self, server: MCPServer) -> None:
        """Test MCP resources/list message."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}}

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "resources" in response["result"]
        assert len(response["result"]["resources"]) == 2

    @pytest.mark.asyncio
    async def test_resources_read_message(self, server: MCPServer) -> None:
        """Test MCP resources/read message."""
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "neuralmemory://prompt/system"},
        }

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "contents" in response["result"]
        assert response["result"]["contents"][0]["uri"] == "neuralmemory://prompt/system"
        assert "NeuralMemory" in response["result"]["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_resources_read_not_found(self, server: MCPServer) -> None:
        """Test MCP resources/read with unknown URI."""
        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "neuralmemory://unknown"},
        }

        response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "error" in response
        assert response["error"]["code"] == -32002


class TestMCPStorage:
    """Tests for MCP server storage management."""

    @pytest.mark.asyncio
    async def test_get_storage_caches_instance(self) -> None:
        """Test that get_storage caches the storage instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            server = MCPServer()

        mock_storage = AsyncMock()

        with patch(
            "neural_memory.mcp.server.get_shared_storage",
            return_value=mock_storage,
        ) as mock_load:
            storage1 = await server.get_storage()
            storage2 = await server.get_storage()

        # Should only load once
        mock_load.assert_called_once()
        assert storage1 is storage2


class TestAutoCapture:
    """Tests for auto-capture pattern detection."""

    def test_insight_pattern_english(self) -> None:
        """Test insight detection for English aha-moments."""
        text = "Turns out the timeout was caused by DNS resolution being slow."
        detected = analyze_text_for_memories(text, capture_insights=True)
        types = [d["type"] for d in detected]
        assert "insight" in types

    def test_insight_pattern_root_cause(self) -> None:
        """Test insight detection for root cause identification."""
        text = "The root cause was a race condition in the connection pool handler."
        detected = analyze_text_for_memories(text, capture_insights=True)
        types = [d["type"] for d in detected]
        assert "insight" in types

    def test_insight_pattern_realized(self) -> None:
        """Test insight detection for realization patterns."""
        text = "I realized that the cache was never being invalidated after writes."
        detected = analyze_text_for_memories(text, capture_insights=True)
        types = [d["type"] for d in detected]
        assert "insight" in types

    def test_vietnamese_decision_pattern(self) -> None:
        """Test Vietnamese decision pattern detection."""
        text = "Quyết định dùng Redis thay vì Memcached cho hệ thống caching."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        types = [d["type"] for d in detected]
        assert "decision" in types

    def test_vietnamese_insight_pattern(self) -> None:
        """Test Vietnamese insight pattern detection."""
        text = "Hóa ra lỗi do DNS resolution bị chậm khi kết nối database."
        detected = analyze_text_for_memories(text, capture_insights=True)
        types = [d["type"] for d in detected]
        assert "insight" in types

    def test_vietnamese_error_pattern(self) -> None:
        """Test Vietnamese error pattern detection."""
        text = "Lỗi do connection pool bị đầy khi có quá nhiều request đồng thời."
        detected = analyze_text_for_memories(text, capture_errors=True)
        types = [d["type"] for d in detected]
        assert "error" in types

    def test_minimum_text_length_guard(self) -> None:
        """Test that very short text returns empty."""
        text = "short text"
        detected = analyze_text_for_memories(text)
        assert detected == []

    def test_dedup_strips_prefix(self) -> None:
        """Test deduplication handles type prefixes correctly."""
        text = "We decided to use Redis. The decision is: use Redis for caching."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        # Both patterns match "use Redis" but dedup should merge
        contents = [d["content"].lower() for d in detected]
        redis_matches = [c for c in contents if "redis" in c]
        # Should have at most 2 (different captures), but dedup limits exact dupes
        assert len(redis_matches) <= 3

    def test_tuple_match_handling(self) -> None:
        """Test patterns with multiple groups (tuple matches)."""
        text = "We chose PostgreSQL over MySQL for better JSONB support."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        assert len(detected) >= 1

    def test_capture_insights_disabled(self) -> None:
        """Test that insights are not captured when disabled."""
        text = "Turns out the timeout was caused by DNS resolution being slow."
        detected = analyze_text_for_memories(text, capture_insights=False)
        types = [d["type"] for d in detected]
        assert "insight" not in types


class TestPassiveCapture:
    """Tests for passive auto-capture on recall."""

    def _make_server(self, *, auto_enabled: bool = True) -> MCPServer:
        """Create a server with controlled auto config."""
        mock_auto_config = MagicMock(
            enabled=auto_enabled,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            min_confidence=0.7,
        )
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=mock_auto_config,
            )
            return MCPServer()

    def _mock_recall_deps(self, server: MCPServer):
        """Set up mocks for recall dependencies."""
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
                tokens_used=12,
            )
        )
        return mock_storage, mock_pipeline

    @pytest.mark.asyncio
    async def test_passive_capture_on_long_recall(self) -> None:
        """Test that recall with long query triggers passive capture."""
        server = self._make_server(auto_enabled=True)
        mock_storage, mock_pipeline = self._mock_recall_deps(server)

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
            patch.object(server, "_passive_capture", new_callable=AsyncMock) as mock_capture,
        ):
            long_query = (
                "We decided to switch from MySQL to PostgreSQL because of better JSONB support"
            )
            await server.call_tool("nmem_recall", {"query": long_query})

        mock_capture.assert_called_once_with(long_query)

    @pytest.mark.asyncio
    async def test_passive_capture_skipped_short_query(self) -> None:
        """Test that short queries do not trigger passive capture."""
        server = self._make_server(auto_enabled=True)
        mock_storage, mock_pipeline = self._mock_recall_deps(server)

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
            patch.object(server, "_passive_capture", new_callable=AsyncMock) as mock_capture,
        ):
            await server.call_tool("nmem_recall", {"query": "auth setup"})

        mock_capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_passive_capture_skipped_when_disabled(self) -> None:
        """Test that passive capture is skipped when auto-capture is disabled."""
        server = self._make_server(auto_enabled=False)
        mock_storage, mock_pipeline = self._mock_recall_deps(server)

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
            patch.object(server, "_passive_capture", new_callable=AsyncMock) as mock_capture,
        ):
            long_query = (
                "We decided to switch from MySQL to PostgreSQL because of better JSONB support"
            )
            await server.call_tool("nmem_recall", {"query": long_query})

        mock_capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_respects_enabled_flag(self) -> None:
        """Test that process action returns early when disabled."""
        server = self._make_server(auto_enabled=False)

        result = await server.call_tool(
            "nmem_auto",
            {"action": "process", "text": "We decided to use Redis for caching."},
        )

        assert result["saved"] == 0
        assert "disabled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_passive_capture_error_swallowed(self) -> None:
        """Test that errors in passive capture don't break recall."""
        server = self._make_server(auto_enabled=True)
        mock_storage, mock_pipeline = self._mock_recall_deps(server)

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.server.ReflexPipeline", return_value=mock_pipeline),
            patch(
                "neural_memory.mcp.server.analyze_text_for_memories",
                side_effect=RuntimeError("boom"),
            ),
        ):
            long_query = (
                "We decided to switch from MySQL to PostgreSQL because of better JSONB support"
            )
            result = await server.call_tool("nmem_recall", {"query": long_query})

        # Recall should still succeed despite passive capture error
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.85
