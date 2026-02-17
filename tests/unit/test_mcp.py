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

        assert len(tools) == 20
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
            "nmem_index",
            "nmem_import",
            "nmem_eternal",
            "nmem_recap",
            "nmem_health",
            "nmem_evolution",
            "nmem_habits",
            "nmem_version",
            "nmem_transplant",
            "nmem_conflicts",
            "nmem_train",
            "nmem_train_db",
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

    def test_index_tool_schema(self, server: MCPServer) -> None:
        """Test nmem_index tool schema."""
        tools = server.get_tools()
        index_tool = next(t for t in tools if t["name"] == "nmem_index")

        schema = index_tool["inputSchema"]
        assert "action" in schema["properties"]
        assert "path" in schema["properties"]
        assert "extensions" in schema["properties"]
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
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
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
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
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
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
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
    async def test_index_scan(self, server: MCPServer) -> None:
        """Test nmem_index scan action returns file/neuron counts."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", name="test", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="idx-123")
        mock_result = MagicMock(
            fiber=mock_fiber,
            neurons_created=[MagicMock(), MagicMock()],
            synapses_created=[MagicMock()],
        )
        mock_encoder = MagicMock()
        mock_encoder.index_directory = AsyncMock(return_value=[mock_result])

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch(
                "neural_memory.engine.codebase_encoder.CodebaseEncoder",
                return_value=mock_encoder,
            ),
        ):
            result = await server.call_tool("nmem_index", {"action": "scan", "path": "."})

        assert result["files_indexed"] == 1
        assert result["neurons_created"] == 2
        assert result["synapses_created"] == 1
        assert "message" in result

    @pytest.mark.asyncio
    async def test_index_status_empty(self, server: MCPServer) -> None:
        """Test nmem_index status when nothing indexed."""
        mock_storage = AsyncMock()
        mock_storage.find_neurons = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_index", {"action": "status"})

        assert result["indexed_files"] == 0
        assert "No codebase indexed" in result["message"]

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
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
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
        """Test nmem_session end creates tombstone and summary."""
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

        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            side_effect=[
                MagicMock(fiber=MagicMock(id="tombstone-123"), neurons_created=[]),
                MagicMock(fiber=MagicMock(id="summary-123"), neurons_created=[]),
                MagicMock(fiber=MagicMock(id="fingerprint-123"), neurons_created=[]),
            ]
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool("nmem_session", {"action": "end"})

        assert result["active"] is False
        assert "auth" in result["summary"]
        assert "75%" in result["summary"]
        assert "message" in result
        # Three typed memories: tombstone + summary + fingerprint
        assert mock_storage.add_typed_memory.call_count == 3

    @pytest.mark.asyncio
    async def test_session_end_no_active(self, server: MCPServer) -> None:
        """Test nmem_session end with no active session."""
        mock_storage = AsyncMock()
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_session", {"action": "end"})

        assert result["active"] is False
        assert "No active session" in result["message"]


class TestMCPErrorPaths:
    """Tests for error paths: no brain, missing storage, etc."""

    def _make_server(self) -> MCPServer:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=MagicMock(enabled=False),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_session_set_no_brain(self) -> None:
        """Test nmem_session set when brain is missing."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_session",
                {"action": "set", "feature": "auth"},
            )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_session_end_no_brain(self) -> None:
        """Test nmem_session end when brain is missing."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        mock_existing = MagicMock(
            metadata={
                "feature": "auth",
                "task": "test",
                "progress": 0.5,
                "started_at": "2026-01-01T00:00:00",
            },
        )
        mock_storage.find_typed_memories = AsyncMock(return_value=[mock_existing])

        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=MagicMock(id="f-1"), neurons_created=[]),
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
        ):
            result = await server.call_tool("nmem_session", {"action": "end"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_eternal_save_no_brain(self) -> None:
        """Test nmem_eternal save when brain is missing returns error."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        ctx = AsyncMock()

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
        ):
            result = await server.call_tool(
                "nmem_eternal",
                {"action": "save", "instruction": "Always use TypeScript"},
            )

        assert "error" in result
        assert "No brain" in result["error"]

    @pytest.mark.asyncio
    async def test_recap_topic_no_brain(self) -> None:
        """Test nmem_recap with topic when brain is missing."""
        server = self._make_server()
        ctx = AsyncMock()
        ctx.get_injection = AsyncMock(return_value="context")

        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
        ):
            result = await server.call_tool("nmem_recap", {"topic": "auth"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_max_tokens_clamped(self) -> None:
        """Test that max_tokens is clamped to 10000."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.find_typed_memories = AsyncMock(return_value=[])

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=MagicMock(
                context="answer",
                confidence=0.9,
                depth_used=MagicMock(value=1),
                neurons_activated=5,
                fibers_matched=[],
                latency_ms=10.0,
                co_activations=[],
                metadata={},
            )
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
        ):
            await server.call_tool(
                "nmem_recall",
                {"query": "test", "max_tokens": 999999},
            )

        call_kwargs = mock_pipeline.query.call_args
        assert call_kwargs.kwargs.get("max_tokens", call_kwargs[1].get("max_tokens", 0)) <= 10000


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
        assert len(response["result"]["tools"]) == 20

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
        assert "failed unexpectedly" in response["error"]["message"]

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
        assert len(content) < 1500  # Compact should be shorter

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

        # get_shared_storage is called each time to detect brain switches,
        # but returns the same cached instance for the same brain/db_path.
        assert mock_load.call_count == 2
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

    def test_preference_pattern_i_prefer(self) -> None:
        """Test preference detection for explicit 'I prefer' statements."""
        text = "I prefer using PostgreSQL over MySQL for all new projects."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_dont_use(self) -> None:
        """Test preference detection for negative preference 'don't use'."""
        text = "Don't use global variables in the codebase, they cause bugs."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_always(self) -> None:
        """Test preference detection for 'always use' patterns."""
        text = "Always use type hints when writing Python functions."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_correction(self) -> None:
        """Test preference detection for correction patterns."""
        text = "Actually, it should be snake_case not camelCase for Python."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_thats_wrong(self) -> None:
        """Test preference detection for 'that's wrong' correction."""
        text = "That's wrong, the timeout should be 30 seconds not 10."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_vietnamese_thich(self) -> None:
        """Test Vietnamese preference pattern 'thích/muốn'."""
        text = "Mình thích dùng dark mode hơn khi code vào ban đêm."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_vietnamese_dung(self) -> None:
        """Test Vietnamese negative preference 'đừng dùng'."""
        text = "Đừng dùng var trong JavaScript, luôn dùng const hoặc let."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_pattern_vietnamese_correction(self) -> None:
        """Test Vietnamese correction pattern 'sai rồi'."""
        text = "Sai rồi, phải dùng async/await thay vì callback ở đây."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        types = [d["type"] for d in detected]
        assert "preference" in types

    def test_preference_disabled(self) -> None:
        """Test that preferences are not captured when disabled."""
        text = "I prefer using PostgreSQL over MySQL for all new projects."
        detected = analyze_text_for_memories(text, capture_preferences=False)
        types = [d["type"] for d in detected]
        assert "preference" not in types

    def test_preference_high_confidence(self) -> None:
        """Test that preference patterns have appropriate confidence."""
        text = "Always use parameterized SQL queries to prevent injection."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        prefs = [d for d in detected if d["type"] == "preference"]
        assert len(prefs) >= 1
        assert prefs[0]["confidence"] >= 0.7

    def test_preference_has_priority_7(self) -> None:
        """Test that preferences get priority 7 (above normal 5)."""
        text = "I prefer tabs over spaces for indentation in this project."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        prefs = [d for d in detected if d["type"] == "preference"]
        assert len(prefs) >= 1
        assert prefs[0]["priority"] == 7

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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
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
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
            patch(
                "neural_memory.mcp.auto_handler.analyze_text_for_memories",
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


class TestMCPEternal:
    """Tests for nmem_eternal and nmem_recap tool calls."""

    def _make_server(self) -> MCPServer:
        """Create a server with eternal config mocked."""
        mock_eternal_config = MagicMock(
            enabled=True,
            notifications=True,
            auto_save_interval=15,
            context_warning_threshold=0.8,
            max_context_tokens=128_000,
        )
        mock_auto_config = MagicMock(
            enabled=False,
            min_confidence=0.7,
        )
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                eternal=mock_eternal_config,
                auto=mock_auto_config,
            )
            return MCPServer()

    def _mock_eternal_context(self) -> AsyncMock:
        """Create a mocked async EternalContext."""
        ctx = AsyncMock()
        ctx.message_count = 10
        ctx.increment_message_count = MagicMock(return_value=11)
        ctx.get_status = AsyncMock(
            return_value={
                "memory_counts": {
                    "fact": 3,
                    "decision": 2,
                    "instruction": 1,
                    "todo": 1,
                    "error": 0,
                    "context": 1,
                    "preference": 0,
                    "insight": 0,
                    "workflow": 0,
                    "reference": 0,
                },
                "session": {
                    "feature": "auth",
                    "task": "login form",
                    "progress": 0.5,
                    "branch": "feat/auth",
                },
                "message_count": 10,
            }
        )
        ctx.estimate_context_usage = AsyncMock(return_value=0.042)
        ctx.get_injection = AsyncMock(
            return_value="## Project Context\n- Project: TestProject\n- Stack: Python, FastAPI"
        )
        return ctx

    @pytest.mark.asyncio
    async def test_eternal_status(self) -> None:
        """Test nmem_eternal status returns memory counts and session."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_eternal", {"action": "status"})

        assert result["enabled"] is True
        assert result["memory_counts"]["fact"] == 3
        assert result["memory_counts"]["decision"] == 2
        assert result["session"]["feature"] == "auth"
        assert result["session"]["task"] == "login form"
        assert result["session"]["progress"] == 0.5
        assert result["message_count"] == 10
        assert result["context_usage"] == 0.042

    @pytest.mark.asyncio
    async def test_eternal_save_empty(self) -> None:
        """Test nmem_eternal save with no fields returns no changes."""
        server = self._make_server()
        ctx = self._mock_eternal_context()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
        ):
            result = await server.call_tool("nmem_eternal", {"action": "save"})

        assert result["saved"] is True
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_eternal_save_project_context(self) -> None:
        """Test nmem_eternal save with project_name and tech_stack."""
        server = self._make_server()
        ctx = self._mock_eternal_context()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        mock_storage.find_typed_memories = AsyncMock(return_value=[])
        mock_remember = AsyncMock(return_value={"stored": True})

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
            patch.object(server, "_remember", mock_remember),
        ):
            result = await server.call_tool(
                "nmem_eternal",
                {
                    "action": "save",
                    "project_name": "NewProject",
                    "tech_stack": ["Go", "gRPC"],
                },
            )

        assert result["saved"] is True
        assert "project_context" in result["items"]
        mock_remember.assert_called_once()
        call_args = mock_remember.call_args[0][0]
        assert "NewProject" in call_args["content"]
        assert "Go" in call_args["content"]
        assert call_args["type"] == "fact"
        assert call_args["priority"] == 10
        assert "project_context" in call_args["tags"]

    @pytest.mark.asyncio
    async def test_eternal_save_decision_and_instruction(self) -> None:
        """Test nmem_eternal save with decision and instruction."""
        server = self._make_server()
        ctx = self._mock_eternal_context()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        remember_calls: list[dict] = []

        async def track_remember(args: dict) -> dict:
            remember_calls.append(args)
            return {"stored": True}

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
            patch.object(server, "_remember", side_effect=track_remember),
        ):
            result = await server.call_tool(
                "nmem_eternal",
                {
                    "action": "save",
                    "decision": "Use gRPC",
                    "reason": "Performance",
                    "instruction": "Use proto3",
                },
            )

        assert result["saved"] is True
        assert "decision" in result["items"]
        assert "instruction" in result["items"]
        assert len(remember_calls) == 2
        # Decision call
        assert "gRPC" in remember_calls[0]["content"]
        assert "Performance" in remember_calls[0]["content"]
        assert remember_calls[0]["type"] == "decision"
        # Instruction call
        assert remember_calls[1]["content"] == "Use proto3"
        assert remember_calls[1]["type"] == "instruction"

    @pytest.mark.asyncio
    async def test_eternal_save_deduplicates_project_context(self) -> None:
        """Test save deletes old project_context facts before encoding new."""
        server = self._make_server()
        ctx = self._mock_eternal_context()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"
        old_mem = MagicMock(fiber_id="old-fiber-1")
        mock_storage.find_typed_memories = AsyncMock(return_value=[old_mem])
        mock_storage.delete_typed_memory = AsyncMock()

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
            patch.object(server, "_remember", AsyncMock(return_value={"stored": True})),
        ):
            await server.call_tool(
                "nmem_eternal",
                {"action": "save", "project_name": "Updated"},
            )

        mock_storage.delete_typed_memory.assert_called_once_with("old-fiber-1")

    @pytest.mark.asyncio
    async def test_eternal_unknown_action(self) -> None:
        """Test nmem_eternal with unknown action."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_eternal", {"action": "bogus"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_recap_level(self) -> None:
        """Test nmem_recap with level."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_recap", {"level": 2})

        assert "context" in result
        assert result["level"] == 2
        assert "tokens_used" in result
        ctx.get_injection.assert_called_once_with(level=2)

    @pytest.mark.asyncio
    async def test_recap_default_level(self) -> None:
        """Test nmem_recap defaults to level 1."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_recap", {})

        assert result["level"] == 1
        ctx.get_injection.assert_called_once_with(level=1)

    @pytest.mark.asyncio
    async def test_recap_level_clamped(self) -> None:
        """Test nmem_recap level is clamped to 1-3."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_recap", {"level": 99})

        assert result["level"] == 3

    @pytest.mark.asyncio
    async def test_recap_with_feature_welcome(self) -> None:
        """Test nmem_recap appends welcome when feature is set."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_recap", {"level": 1})

        assert "Welcome back!" in result["message"]

    @pytest.mark.asyncio
    async def test_recap_no_welcome_without_feature(self) -> None:
        """Test nmem_recap no welcome when no feature set."""
        server = self._make_server()
        ctx = self._mock_eternal_context()
        ctx.get_status = AsyncMock(
            return_value={
                "memory_counts": {},
                "session": {},
                "message_count": 0,
            }
        )

        with patch.object(server, "get_eternal_context", return_value=ctx):
            result = await server.call_tool("nmem_recap", {"level": 1})

        assert "Welcome back!" not in result["message"]

    @pytest.mark.asyncio
    async def test_recap_topic(self) -> None:
        """Test nmem_recap with topic search."""
        server = self._make_server()
        ctx = self._mock_eternal_context()

        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=MagicMock(
                context="Auth uses JWT tokens",
                confidence=0.9,
            )
        )

        with (
            patch.object(server, "get_eternal_context", return_value=ctx),
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.eternal_handler.ReflexPipeline", return_value=mock_pipeline),
        ):
            result = await server.call_tool("nmem_recap", {"topic": "auth"})

        assert result["topic"] == "auth"
        assert result["confidence"] == 0.9
        assert "auth" in result["message"].lower()


class TestMCPImport:
    """Tests for nmem_import tool calls."""

    def _make_server(self) -> MCPServer:
        """Create a server with mocked config."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_import_success(self) -> None:
        """Test nmem_import with successful sync."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_result = MagicMock(
            source_system="chromadb",
            source_collection="default",
            records_fetched=10,
            records_imported=8,
            records_skipped=1,
            records_failed=1,
            duration_seconds=2.5,
            errors=["one error"],
        )
        mock_sync_state = MagicMock()

        mock_adapter = MagicMock()
        mock_engine = MagicMock()
        mock_engine.sync = AsyncMock(return_value=(mock_result, mock_sync_state))

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.integration.adapters.get_adapter", return_value=mock_adapter),
            patch("neural_memory.integration.sync_engine.SyncEngine", return_value=mock_engine),
        ):
            result = await server.call_tool(
                "nmem_import",
                {"source": "chromadb", "connection": "/tmp/chroma", "collection": "test"},
            )

        assert result["success"] is True
        assert result["source"] == "chromadb"
        assert result["records_imported"] == 8
        assert result["records_failed"] == 1

    @pytest.mark.asyncio
    async def test_import_no_brain(self) -> None:
        """Test nmem_import when no brain configured."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_import", {"source": "chromadb"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_no_source(self) -> None:
        """Test nmem_import without source."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_import", {"source": ""})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_adapter_not_found(self) -> None:
        """Test nmem_import with unknown adapter."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch(
                "neural_memory.integration.adapters.get_adapter",
                side_effect=ValueError("Unknown adapter"),
            ),
        ):
            result = await server.call_tool("nmem_import", {"source": "unknown_system"})

        assert "error" in result
        assert "Unsupported or misconfigured source" in result["error"]

    @pytest.mark.asyncio
    async def test_import_sync_failure(self) -> None:
        """Test nmem_import when sync raises."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_adapter = MagicMock()
        mock_engine = MagicMock()
        mock_engine.sync = AsyncMock(side_effect=RuntimeError("Connection refused"))

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.integration.adapters.get_adapter", return_value=mock_adapter),
            patch("neural_memory.integration.sync_engine.SyncEngine", return_value=mock_engine),
        ):
            result = await server.call_tool("nmem_import", {"source": "chromadb"})

        assert "error" in result
        assert "failed unexpectedly" in result["error"]

    @pytest.mark.asyncio
    async def test_import_adapter_kwargs_awf(self, tmp_path: object) -> None:
        """Test that AWF source passes brain_dir kwarg."""
        brain_dir = str(tmp_path)  # Real directory for path validation
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_result = MagicMock(
            source_system="awf",
            source_collection="default",
            records_fetched=5,
            records_imported=5,
            records_skipped=0,
            records_failed=0,
            duration_seconds=1.0,
            errors=[],
        )
        mock_engine = MagicMock()
        mock_engine.sync = AsyncMock(return_value=(mock_result, MagicMock()))

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.integration.adapters.get_adapter") as mock_get_adapter,
            patch("neural_memory.integration.sync_engine.SyncEngine", return_value=mock_engine),
        ):
            mock_get_adapter.return_value = MagicMock()
            await server.call_tool("nmem_import", {"source": "awf", "connection": brain_dir})

        from pathlib import Path as _P

        resolved_brain_dir = str(_P(brain_dir).resolve())
        mock_get_adapter.assert_called_once_with("awf", brain_dir=resolved_brain_dir)


class TestMCPAutoExtended:
    """Extended tests for nmem_auto enable/disable/error paths."""

    def _make_server(self, *, auto_enabled: bool = True) -> MCPServer:
        """Create server with controllable auto config."""
        mock_auto_config = MagicMock(
            enabled=auto_enabled,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            min_confidence=0.7,
        )
        mock_eternal_config = MagicMock(enabled=False)
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=mock_auto_config,
                eternal=mock_eternal_config,
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_auto_enable(self) -> None:
        """Test nmem_auto enable action toggles enabled flag."""
        server = self._make_server(auto_enabled=False)

        def fake_replace(obj, **kwargs):
            new_obj = MagicMock(wraps=obj)
            for k, v in kwargs.items():
                setattr(new_obj, k, v)
            new_obj.save = MagicMock()
            return new_obj

        with patch("dataclasses.replace", side_effect=fake_replace):
            result = await server.call_tool("nmem_auto", {"action": "enable"})

        assert result["enabled"] is True
        assert "enabled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_auto_disable(self) -> None:
        """Test nmem_auto disable action toggles enabled flag."""
        server = self._make_server(auto_enabled=True)

        def fake_replace(obj, **kwargs):
            new_obj = MagicMock(wraps=obj)
            for k, v in kwargs.items():
                setattr(new_obj, k, v)
            new_obj.save = MagicMock()
            return new_obj

        with patch("dataclasses.replace", side_effect=fake_replace):
            result = await server.call_tool("nmem_auto", {"action": "disable"})

        assert result["enabled"] is False
        assert "disabled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_auto_analyze_empty_text(self) -> None:
        """Test nmem_auto analyze with empty text."""
        server = self._make_server()

        result = await server.call_tool("nmem_auto", {"action": "analyze", "text": ""})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_auto_process_empty_text(self) -> None:
        """Test nmem_auto process with empty text."""
        server = self._make_server()

        result = await server.call_tool("nmem_auto", {"action": "process", "text": ""})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_auto_unknown_action(self) -> None:
        """Test nmem_auto with unknown action."""
        server = self._make_server()

        result = await server.call_tool("nmem_auto", {"action": "bogus"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_auto_analyze_with_save(self) -> None:
        """Test nmem_auto analyze with save=True."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="auto-save-123")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
        ):
            text = "We decided to use PostgreSQL. TODO: Set up migrations."
            result = await server.call_tool(
                "nmem_auto", {"action": "analyze", "text": text, "save": True}
            )

        assert "detected" in result
        assert "saved" in result


class TestMCPContextExtended:
    """Extended tests for nmem_context fresh_only and anchor fallback."""

    @pytest.fixture
    def server(self) -> MCPServer:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_context_fresh_only(self, server: MCPServer) -> None:
        """Test nmem_context with fresh_only=True filters old fibers."""
        from datetime import datetime, timedelta

        mock_storage = AsyncMock()
        now = datetime.now()
        fresh_fiber = MagicMock(
            summary="Fresh memory",
            anchor_neuron_id=None,
            created_at=now - timedelta(hours=1),
        )
        old_fiber = MagicMock(
            summary="Old memory",
            anchor_neuron_id=None,
            created_at=now - timedelta(days=90),
        )
        mock_storage.get_fibers = AsyncMock(return_value=[fresh_fiber, old_fiber])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_context", {"fresh_only": True, "limit": 10})

        assert result["count"] >= 1
        assert "Fresh memory" in result["context"]

    @pytest.mark.asyncio
    async def test_context_anchor_fallback(self, server: MCPServer) -> None:
        """Test nmem_context falls back to anchor neuron when fiber has no summary."""
        mock_storage = AsyncMock()
        fiber = MagicMock(summary=None, anchor_neuron_id="anchor-1")
        mock_storage.get_fibers = AsyncMock(return_value=[fiber])
        mock_storage.get_neuron = AsyncMock(return_value=MagicMock(content="Anchor content"))

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_context", {})

        assert result["count"] == 1
        assert "Anchor content" in result["context"]

    @pytest.mark.asyncio
    async def test_context_no_summary_no_anchor(self, server: MCPServer) -> None:
        """Test nmem_context with fiber that has no summary and no anchor."""
        mock_storage = AsyncMock()
        fiber = MagicMock(summary=None, anchor_neuron_id=None)
        mock_storage.get_fibers = AsyncMock(return_value=[fiber])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_context", {})

        assert result["count"] == 0


class TestMCPRecallExtended:
    """Extended tests for recall: no brain, session injection, sensitive content."""

    def _make_server(self) -> MCPServer:
        mock_auto_config = MagicMock(enabled=False, min_confidence=0.7)
        mock_eternal_config = MagicMock(enabled=False)
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=mock_auto_config,
                eternal=mock_eternal_config,
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_recall_no_brain(self) -> None:
        """Test nmem_recall when no brain configured."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_recall", {"query": "test"})

        assert "error" in result
        assert "No brain" in result["error"]

    @pytest.mark.asyncio
    async def test_recall_session_context_injection(self) -> None:
        """Test that short queries get session context injected."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        # Simulate active session
        mock_session = MagicMock(metadata={"feature": "auth", "task": "login", "active": True})
        mock_storage.find_typed_memories = AsyncMock(return_value=[mock_session])

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=MagicMock(
                context="Auth answer",
                confidence=0.9,
                neurons_activated=3,
                fibers_matched=1,
                depth_used=MagicMock(value=1),
                tokens_used=10,
            )
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.ReflexPipeline", return_value=mock_pipeline),
        ):
            result = await server.call_tool("nmem_recall", {"query": "how it works"})

        # Verify query was enriched with session context
        call_args = mock_pipeline.query.call_args
        assert "context:" in call_args.kwargs["query"]
        assert "auth" in call_args.kwargs["query"]
        assert result["answer"] == "Auth answer"

    @pytest.mark.asyncio
    async def test_remember_sensitive_content(self) -> None:
        """Test nmem_remember handles sensitive content.

        With auto-redact (Phase F), severity-3 API keys are auto-redacted
        rather than blocking. The memory is stored with [REDACTED] content.
        """
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_remember",
                {"content": "API_KEY=sk-1234567890abcdef"},
            )

        # Severity-3 API key is auto-redacted and stored successfully
        assert result.get("success") is True
        assert result.get("auto_redacted") is True

    @pytest.mark.asyncio
    async def test_remember_auto_type_detection(self) -> None:
        """Test nmem_remember auto-detects type when not specified."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        mock_fiber = MagicMock(id="fiber-auto")
        mock_encoder = AsyncMock()
        mock_encoder.encode = AsyncMock(
            return_value=MagicMock(fiber=mock_fiber, neurons_created=[])
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder", return_value=mock_encoder),
        ):
            # No "type" in args — should use suggest_memory_type
            result = await server.call_tool(
                "nmem_remember",
                {"content": "TODO: fix the login bug"},
            )

        assert result["success"] is True
        assert "memory_type" in result


class TestMCPMiscErrors:
    """Tests for miscellaneous error paths."""

    @pytest.fixture
    def server(self) -> MCPServer:
        mock_eternal_config = MagicMock(enabled=False)
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                eternal=mock_eternal_config,
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_stats_no_brain(self, server: MCPServer) -> None:
        """Test nmem_stats when no brain configured."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_stats", {})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_session_unknown_action(self, server: MCPServer) -> None:
        """Test nmem_session with unknown action."""
        mock_storage = AsyncMock()

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_session", {"action": "bogus"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_index_unknown_action(self, server: MCPServer) -> None:
        """Test nmem_index with unknown action."""
        mock_storage = AsyncMock()

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_index", {"action": "bogus"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_index_scan_no_brain(self, server: MCPServer) -> None:
        """Test nmem_index scan with no brain."""
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(return_value=None)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_index", {"action": "scan"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_index_scan_not_directory(self, server: MCPServer) -> None:
        """Test nmem_index scan with non-existent path."""
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_index",
                {"action": "scan", "path": "/nonexistent/dir/xyz123"},
            )

        assert "error" in result
        assert "Not a directory" in result["error"]

    @pytest.mark.asyncio
    async def test_session_get_active(self, server: MCPServer) -> None:
        """Test nmem_session get with active session returns data."""
        mock_storage = AsyncMock()
        mock_session = MagicMock(
            metadata={
                "active": True,
                "feature": "deploy",
                "task": "k8s setup",
                "progress": 0.6,
                "started_at": "2026-02-06T10:00:00",
                "notes": "In progress",
                "branch": "feat/deploy",
                "commit": "abc123",
                "repo": "myrepo",
            }
        )
        mock_storage.find_typed_memories = AsyncMock(return_value=[mock_session])

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_session", {"action": "get"})

        assert result["active"] is True
        assert result["feature"] == "deploy"
        assert result["task"] == "k8s setup"
        assert result["progress"] == 0.6
        assert result["branch"] == "feat/deploy"


class TestMCPFireTrigger:
    """Tests for _fire_eternal_trigger."""

    def _make_server(self, *, eternal_enabled: bool = True) -> MCPServer:
        mock_eternal_config = MagicMock(
            enabled=eternal_enabled,
            auto_save_interval=15,
            context_warning_threshold=0.8,
            max_context_tokens=128_000,
        )
        mock_auto_config = MagicMock(enabled=False, min_confidence=0.7)
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                eternal=mock_eternal_config,
                auto=mock_auto_config,
            )
            return MCPServer()

    def test_fire_trigger_disabled(self) -> None:
        """Test _fire_eternal_trigger does nothing when eternal disabled."""
        server = self._make_server(eternal_enabled=False)
        # Should not raise
        server._fire_eternal_trigger("test text")

    def test_fire_trigger_increments_counter(self) -> None:
        """Test _fire_eternal_trigger increments message count and calls check_triggers."""
        server = self._make_server(eternal_enabled=True)
        ctx = MagicMock()
        ctx.increment_message_count = MagicMock(return_value=15)
        server._eternal_ctx = ctx

        with patch("neural_memory.mcp.eternal_handler.check_triggers") as mock_check:
            mock_check.return_value = MagicMock(triggered=False)
            server._fire_eternal_trigger("Some text")

        ctx.increment_message_count.assert_called_once()
        mock_check.assert_called_once()

    def test_fire_trigger_skips_when_ctx_not_initialized(self) -> None:
        """Test _fire_eternal_trigger returns early if ctx is None."""
        server = self._make_server(eternal_enabled=True)
        assert server._eternal_ctx is None

        with patch("neural_memory.mcp.eternal_handler.check_triggers") as mock_check:
            server._fire_eternal_trigger("Some text")

        mock_check.assert_not_called()

    def test_fire_trigger_swallows_errors(self) -> None:
        """Test _fire_eternal_trigger swallows exceptions."""
        server = self._make_server(eternal_enabled=True)
        ctx = MagicMock()
        ctx.increment_message_count = MagicMock(side_effect=RuntimeError("boom"))
        server._eternal_ctx = ctx

        # Should not raise
        server._fire_eternal_trigger("Some text")


class TestMCPInputValidation:
    """Tests for security input validation guards."""

    def _make_server(self) -> MCPServer:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                auto=MagicMock(enabled=True, min_confidence=0.7),
                eternal=MagicMock(enabled=False),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_remember_rejects_oversized_content(self) -> None:
        """Content over 100KB is rejected."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_remember", {"content": "x" * 200_000})

        assert "error" in result
        assert "too long" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_auto_analyze_rejects_oversized_text(self) -> None:
        """Text over 100KB is rejected in analyze action."""
        server = self._make_server()

        result = await server.call_tool("nmem_auto", {"action": "analyze", "text": "x" * 200_000})

        assert "error" in result
        assert "too long" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_auto_process_rejects_oversized_text(self) -> None:
        """Text over 100KB is rejected in process action."""
        server = self._make_server()

        result = await server.call_tool("nmem_auto", {"action": "process", "text": "x" * 200_000})

        assert "error" in result
        assert "too long" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_remember_rejects_invalid_type(self) -> None:
        """Invalid memory type returns error, not crash."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_remember", {"content": "test", "type": "invalid_type"}
            )

        assert "error" in result
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_recall_rejects_invalid_depth(self) -> None:
        """Invalid depth level returns error, not crash."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_brain = MagicMock(id="test-brain", config=MagicMock())
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_recall", {"query": "test", "depth": 99})

        assert "error" in result
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_suggest_rejects_invalid_type_filter(self) -> None:
        """Invalid type_filter returns error, not crash."""
        server = self._make_server()
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_suggest", {"prefix": "test", "type_filter": "bogus"}
            )

        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_auto_capture_truncates_huge_text(self) -> None:
        """Regex processing truncates text over 50KB."""
        huge = "We decided to use Redis. " * 5000  # > 50KB
        result = analyze_text_for_memories(huge)
        # Should complete quickly without ReDoS, and still detect patterns
        assert isinstance(result, list)
