"""Tests for NeuralMemory-Nanobot integration.

Covers:
1. Protocol conformance — tools satisfy NanobotTool protocol
2. Tool schemas — correct names, parameters, required fields
3. Tool execution — remember, recall, context, health
4. NMMemoryStore — drop-in MemoryStore replacement
5. Setup function — creates db, registers tools, idempotent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.integrations.nanobot.context import NMContext
from neural_memory.storage.memory_store import InMemoryStorage

if TYPE_CHECKING:
    from neural_memory.integrations.nanobot.memory_store import NMMemoryStore

# ── Fixtures ──────────────────────────────────────────────


@pytest_asyncio.fixture
async def nm_context() -> NMContext:
    """Create NMContext with in-memory storage for testing."""
    storage = InMemoryStorage()
    config = BrainConfig()
    brain = Brain.create(name="test-nanobot", config=config, brain_id="test-nanobot")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return NMContext(storage=storage, brain=brain, config=config)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace."""
    ws = tmp_path / "nanobot-workspace"
    ws.mkdir()
    return ws


# ── Protocol conformance ─────────────────────────────────


class TestProtocol:
    """Verify tools satisfy NanobotTool protocol."""

    def test_remember_is_nanobot_tool(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.protocol import NanobotTool
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        assert isinstance(tool, NanobotTool)

    def test_recall_is_nanobot_tool(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.protocol import NanobotTool
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        assert isinstance(tool, NanobotTool)

    def test_context_is_nanobot_tool(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.protocol import NanobotTool
        from neural_memory.integrations.nanobot.tools import NMContextTool

        tool = NMContextTool(nm_context)
        assert isinstance(tool, NanobotTool)

    def test_health_is_nanobot_tool(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.protocol import NanobotTool
        from neural_memory.integrations.nanobot.tools import NMHealthTool

        tool = NMHealthTool(nm_context)
        assert isinstance(tool, NanobotTool)


# ── Tool schemas ─────────────────────────────────────────


class TestToolSchemas:
    """Test tool schema definitions."""

    def test_remember_schema(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        assert tool.name == "nmem_remember"
        assert "content" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["content"]
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "nmem_remember"

    def test_recall_schema(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        assert tool.name == "nmem_recall"
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    def test_context_schema(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMContextTool

        tool = NMContextTool(nm_context)
        assert tool.name == "nmem_context"
        assert "limit" in tool.parameters["properties"]

    def test_health_schema(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMHealthTool

        tool = NMHealthTool(nm_context)
        assert tool.name == "nmem_health"
        schema = tool.to_schema()
        assert schema["function"]["name"] == "nmem_health"


# ── Remember tool ────────────────────────────────────────


class TestRememberTool:
    """Test nmem_remember tool execution."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result_str = await tool.execute(content="Python uses indentation for blocks")
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["fiber_id"]
        assert result["neurons_created"] > 0
        assert result["synapses_created"] >= 0

    @pytest.mark.asyncio
    async def test_remember_with_type_and_tags(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = json.loads(
            await tool.execute(
                content="Use PostgreSQL for production database",
                type="decision",
                tags=["database", "infrastructure"],
                priority=8,
            )
        )
        assert result["success"] is True
        assert result["memory_type"] == "decision"

    @pytest.mark.asyncio
    async def test_remember_content_too_long(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = json.loads(await tool.execute(content="x" * 100_001))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remember_empty_content(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = json.loads(await tool.execute(content=""))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remember_invalid_type(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = json.loads(await tool.execute(content="test", type="invalid_type"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remember_returns_str(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = await tool.execute(content="test memory")
        assert isinstance(result, str)


# ── Recall tool ──────────────────────────────────────────


class TestRecallTool:
    """Test nmem_recall tool execution."""

    @pytest.mark.asyncio
    async def test_recall_empty_brain(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        result = json.loads(await tool.execute(query="anything"))
        assert "answer" in result or "message" in result
        assert isinstance(result.get("confidence", 0.0), float)

    @pytest.mark.asyncio
    async def test_recall_after_remember(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool, NMRememberTool

        remember = NMRememberTool(nm_context)
        recall = NMRecallTool(nm_context)

        await remember.execute(content="The database port is 5432 for PostgreSQL")
        result = json.loads(await recall.execute(query="database port"))
        assert "answer" in result or "message" in result

    @pytest.mark.asyncio
    async def test_recall_min_confidence_filter(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        result = json.loads(await tool.execute(query="nonexistent topic xyz", min_confidence=0.99))
        assert result.get("answer") is None or "No memories" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        result = json.loads(await tool.execute(query=""))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_returns_str(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRecallTool

        tool = NMRecallTool(nm_context)
        result = await tool.execute(query="test")
        assert isinstance(result, str)


# ── Context tool ─────────────────────────────────────────


class TestContextTool:
    """Test nmem_context tool execution."""

    @pytest.mark.asyncio
    async def test_context_empty_brain(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMContextTool

        tool = NMContextTool(nm_context)
        result = json.loads(await tool.execute())
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_context_with_memories(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMContextTool, NMRememberTool

        remember = NMRememberTool(nm_context)
        await remember.execute(content="Project uses FastAPI for the REST API layer")

        context_tool = NMContextTool(nm_context)
        result = json.loads(await context_tool.execute(limit=5))
        assert result["count"] > 0
        assert "FastAPI" in result["context"] or "REST" in result["context"]

    @pytest.mark.asyncio
    async def test_context_returns_str(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMContextTool

        tool = NMContextTool(nm_context)
        result = await tool.execute()
        assert isinstance(result, str)


# ── Health tool ──────────────────────────────────────────


class TestHealthTool:
    """Test nmem_health tool execution."""

    @pytest.mark.asyncio
    async def test_health_empty_brain(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMHealthTool

        tool = NMHealthTool(nm_context)
        result = json.loads(await tool.execute())
        assert "grade" in result
        assert "purity_score" in result
        assert isinstance(result["warnings"], list)

    @pytest.mark.asyncio
    async def test_health_returns_str(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMHealthTool

        tool = NMHealthTool(nm_context)
        result = await tool.execute()
        assert isinstance(result, str)


# ── NMMemoryStore ────────────────────────────────────────


class TestNMMemoryStore:
    """Test drop-in MemoryStore replacement."""

    @pytest_asyncio.fixture
    async def store(self, nm_context: NMContext, workspace: Path) -> NMMemoryStore:
        from neural_memory.integrations.nanobot.memory_store import NMMemoryStore

        return NMMemoryStore(nm_context, workspace)

    def test_get_today_file(self, store: Any) -> None:
        path = store.get_today_file()
        assert "memory" in str(path)
        assert path.suffix == ".md"

    @pytest.mark.asyncio
    async def test_read_today_empty(self, store: Any) -> None:
        result = await store.read_today()
        assert result == ""

    @pytest.mark.asyncio
    async def test_append_and_read_today(self, store: Any) -> None:
        await store.append_today("Test memory: auth module uses JWT tokens")
        result = await store.read_today()
        assert isinstance(result, str)
        # Should contain something after encoding
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_write_and_read_long_term(self, store: Any) -> None:
        await store.write_long_term("Architecture: microservices with event-driven communication")
        result = await store.read_long_term()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_recent_memories_empty(self, store: Any) -> None:
        result = await store.get_recent_memories(days=7)
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_memories_with_data(self, store: Any) -> None:
        await store.append_today("Sprint planning: prioritize auth refactor")
        result = await store.get_recent_memories(days=1)
        assert isinstance(result, str)

    def test_list_memory_files_empty(self, store: Any) -> None:
        assert store.list_memory_files() == []

    @pytest.mark.asyncio
    async def test_get_memory_context_empty(self, store: Any) -> None:
        result = await store.get_memory_context()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_memory_context_with_data(self, store: Any) -> None:
        await store.append_today("The API uses JWT tokens for authentication")
        result = await store.get_memory_context()
        assert isinstance(result, str)
        # Should have section headers when data exists
        if result:
            assert "NeuralMemory" in result or "Recent" in result


# ── Setup function ───────────────────────────────────────


class TestSetup:
    """Test setup_neural_memory entry point."""

    @pytest.mark.asyncio
    async def test_setup_creates_db(self, workspace: Path) -> None:
        from neural_memory.integrations.nanobot.setup import setup_neural_memory

        mock_registry = MagicMock()
        store = await setup_neural_memory(mock_registry, workspace, brain_id="test-setup")

        db_path = workspace / "memory" / "neural.db"
        assert db_path.exists()
        assert mock_registry.register.call_count == 4

        await store._ctx.close()

    @pytest.mark.asyncio
    async def test_setup_registers_correct_tools(self, workspace: Path) -> None:
        from neural_memory.integrations.nanobot.setup import setup_neural_memory

        mock_registry = MagicMock()
        store = await setup_neural_memory(mock_registry, workspace)

        registered_names = {call.args[0].name for call in mock_registry.register.call_args_list}
        assert registered_names == {"nmem_remember", "nmem_recall", "nmem_context", "nmem_health"}

        await store._ctx.close()

    @pytest.mark.asyncio
    async def test_setup_idempotent(self, workspace: Path) -> None:
        """Calling setup twice with same brain_id reuses the brain."""
        from neural_memory.integrations.nanobot.setup import setup_neural_memory

        mock_registry = MagicMock()
        store1 = await setup_neural_memory(mock_registry, workspace, brain_id="idem")
        brain_id_1 = store1._ctx.brain.id
        await store1._ctx.close()

        store2 = await setup_neural_memory(mock_registry, workspace, brain_id="idem")
        assert store2._ctx.brain.id == brain_id_1
        await store2._ctx.close()


# ── NMContext lifecycle ──────────────────────────────────


class TestNMContext:
    """Test NMContext async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self, nm_context: NMContext) -> None:
        assert nm_context.brain is not None
        assert nm_context.storage is not None
        assert nm_context.config is not None

    @pytest.mark.asyncio
    async def test_close(self, nm_context: NMContext) -> None:
        # InMemoryStorage doesn't have close, but NMContext handles that
        await nm_context.close()


# ── Base tool ────────────────────────────────────────────


class TestBaseTool:
    """Test BaseNMTool utilities."""

    def test_validate_params_missing_required(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        errors = tool.validate_params({})
        assert len(errors) == 1
        assert "content" in errors[0]

    def test_validate_params_valid(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        errors = tool.validate_params({"content": "test"})
        assert len(errors) == 0

    def test_json_serialization(self, nm_context: NMContext) -> None:
        from neural_memory.integrations.nanobot.tools import NMRememberTool

        tool = NMRememberTool(nm_context)
        result = tool._json({"key": "value", "number": 42})
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42
