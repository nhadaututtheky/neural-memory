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
        mock_brain = MagicMock(id="test-brain", name="test-brain")
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        mock_storage._current_brain_id = "test-brain"

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch(
                "neural_memory.unified_config.get_shared_storage",
                side_effect=FileNotFoundError("Brain not found"),
            ),
        ):
            result = await server.call_tool(
                "nmem_transplant", {"source_brain": "nonexistent-brain"}
            )

        assert "error" in result
        assert "not found" in result["error"].lower()

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


class TestVersionDiffWithData:
    """Test nmem_version diff action with actual data."""

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
    async def test_version_diff_with_data(self, server: MCPServer) -> None:
        """Diff should return correct change counts."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Add initial neurons
        n1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        with patch.object(server, "get_storage", return_value=storage):
            # Create version v1
            v1_result = await server.call_tool(
                "nmem_version",
                {"action": "create", "name": "v1-baseline"},
            )
            assert v1_result.get("success") is True
            v1_id = v1_result["version_id"]

            # Add a new neuron
            n3 = Neuron.create(type=NeuronType.ACTION, content="deploy", neuron_id="n-3")
            await storage.add_neuron(n3)

            # Create version v2
            v2_result = await server.call_tool(
                "nmem_version",
                {"action": "create", "name": "v2-with-deploy"},
            )
            assert v2_result.get("success") is True
            v2_id = v2_result["version_id"]

            # Diff v1 -> v2
            diff_result = await server.call_tool(
                "nmem_version",
                {"action": "diff", "from_version": v1_id, "to_version": v2_id},
            )

        assert "error" not in diff_result
        assert diff_result["neurons_added"] == 1
        assert diff_result["neurons_removed"] == 0
        assert "n-3" in diff_result["summary"] or "+1 neurons" in diff_result["summary"]


class TestVersionRollbackWithData:
    """Test nmem_version rollback action with actual data."""

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
    async def test_version_rollback_with_data(self, server: MCPServer) -> None:
        """Rollback should restore brain state."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create 3 neurons with synapses
        n1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-2")
        n3 = Neuron.create(type=NeuronType.ACTION, content="deploy", neuron_id="n-3")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        s1 = Synapse.create(
            source_id="n-1",
            target_id="n-2",
            type=SynapseType.RELATED_TO,
            weight=0.7,
            synapse_id="s-1",
        )
        await storage.add_synapse(s1)

        with patch.object(server, "get_storage", return_value=storage):
            # Create version v1 with 3 neurons
            v1_result = await server.call_tool(
                "nmem_version",
                {"action": "create", "name": "v1-three-neurons"},
            )
            assert v1_result.get("success") is True
            assert v1_result["neuron_count"] == 3
            v1_id = v1_result["version_id"]

            # Add neuron n-4
            n4 = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-4")
            await storage.add_neuron(n4)

            # Verify 4 neurons now
            stats = await storage.get_stats("test-brain")
            assert stats["neuron_count"] == 4

            # Rollback to v1
            rollback_result = await server.call_tool(
                "nmem_version",
                {"action": "rollback", "version_id": v1_id},
            )

        assert "error" not in rollback_result
        assert rollback_result.get("success") is True
        assert rollback_result["neuron_count"] == 3

        # Verify actual state was restored
        stats = await storage.get_stats("test-brain")
        assert stats["neuron_count"] == 3


class TestTransplantWithData:
    """Test nmem_transplant with actual data between brains."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create an MCP server instance."""
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="target-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/target-brain.db"),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_transplant_with_data(self, server: MCPServer) -> None:
        """Transplant should move data between brains."""
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()

        # Create source brain with data
        source_brain = Brain.create(
            name="source-brain", config=BrainConfig(), brain_id="source-brain"
        )
        await storage.save_brain(source_brain)
        storage.set_brain(source_brain.id)

        n1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-src-1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-src-2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s1 = Synapse.create(
            source_id="n-src-1",
            target_id="n-src-2",
            type=SynapseType.RELATED_TO,
            weight=0.8,
            synapse_id="s-src-1",
        )
        await storage.add_synapse(s1)

        f1 = Fiber.create(
            neuron_ids={"n-src-1", "n-src-2"},
            synapse_ids={"s-src-1"},
            anchor_neuron_id="n-src-1",
            fiber_id="f-src-1",
            tags={"redis", "cache"},
        )
        await storage.add_fiber(f1)

        # Create target brain with existing data
        target_brain = Brain.create(
            name="target-brain", config=BrainConfig(), brain_id="target-brain"
        )
        await storage.save_brain(target_brain)
        storage.set_brain(target_brain.id)

        nt1 = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-tgt-1")
        await storage.add_neuron(nt1)

        # Set context back to target brain for the MCP call
        storage.set_brain("target-brain")

        # Create a separate source storage view backed by the same data
        source_storage = InMemoryStorage()
        source_storage._neurons = storage._neurons
        source_storage._synapses = storage._synapses
        source_storage._fibers = storage._fibers
        source_storage._brains = storage._brains
        source_storage._typed_memories = storage._typed_memories
        source_storage._projects = storage._projects
        source_storage._graph = storage._graph
        source_storage.set_brain("source-brain")

        async def mock_get_shared_storage(brain_name: str | None = None) -> InMemoryStorage:
            return source_storage

        with (
            patch.object(server, "get_storage", return_value=storage),
            patch(
                "neural_memory.unified_config.get_shared_storage",
                side_effect=mock_get_shared_storage,
            ),
        ):
            result = await server.call_tool(
                "nmem_transplant",
                {"source_brain": "source-brain", "tags": ["redis"]},
            )

        assert "error" not in result
        assert result.get("success") is True
        assert result["fibers_transplanted"] == 1
        assert result["neurons_transplanted"] == 2

        # Verify target brain now has both original and transplanted data
        storage.set_brain("target-brain")
        stats = await storage.get_stats("target-brain")
        # Target should have the original neuron plus transplanted neurons
        assert stats["neuron_count"] >= 2
        assert stats["fiber_count"] >= 1
