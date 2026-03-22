"""Tests for nmem_explain — connection explainer feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.connection_explainer import (
    _confidence_to_strength,
    compute_path_confidence,
    explain_connection,
)
from neural_memory.mcp.server import MCPServer, handle_message
from neural_memory.unified_config import ResponseConfig, ToolTierConfig


def _make_neuron(neuron_id: str, content: str, type: NeuronType = NeuronType.CONCEPT) -> Neuron:
    return Neuron.create(neuron_id=neuron_id, content=content, type=type)


def _make_synapse(
    synapse_id: str,
    source_id: str,
    target_id: str,
    type: SynapseType = SynapseType.RELATED_TO,
    weight: float = 0.8,
) -> Synapse:
    return Synapse.create(
        synapse_id=synapse_id, source_id=source_id, target_id=target_id, type=type, weight=weight
    )


def _make_fiber(id: str, neuron_ids: set[str], summary: str = "") -> MagicMock:
    fiber = MagicMock()
    fiber.id = id
    fiber.neuron_ids = neuron_ids
    fiber.summary = summary
    return fiber


class TestExplainConnection:
    """Tests for the explain_connection engine function."""

    @pytest.mark.asyncio
    async def test_path_found(self) -> None:
        """When a path exists between entities, should return explanation."""
        n_a = _make_neuron("n1", "React")
        n_b = _make_neuron("n2", "virtual DOM")
        n_c = _make_neuron("n3", "performance")
        s_ab = _make_synapse("s1", "n1", "n2", SynapseType.RELATED_TO)
        s_bc = _make_synapse("s2", "n2", "n3", SynapseType.LEADS_TO)

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if "React" in (content_contains or "") else [n_c]
            )
        )
        storage.get_path = AsyncMock(return_value=[(n_b, s_ab), (n_c, s_bc)])
        storage.find_fibers_batch = AsyncMock(return_value=[])

        result = await explain_connection(storage, "React", "performance")

        assert result.found is True
        assert result.total_hops == 2
        assert len(result.steps) == 2
        assert result.steps[0].content == "virtual DOM"
        assert result.steps[1].content == "performance"
        assert "React" in result.markdown
        assert "performance" in result.markdown

    @pytest.mark.asyncio
    async def test_no_path_found(self) -> None:
        """When no path exists, should return found=False."""
        n_a = _make_neuron("n1", "React")
        n_b = _make_neuron("n2", "cooking")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if "React" in (content_contains or "") else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=None)

        result = await explain_connection(storage, "React", "cooking")

        assert result.found is False
        assert "no path found" in result.markdown.lower()

    @pytest.mark.asyncio
    async def test_entity_not_found(self) -> None:
        """When entity has no matching neurons, should return found=False."""
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])

        result = await explain_connection(storage, "nonexistent", "also_nonexistent")

        assert result.found is False
        assert "entity not found" in result.markdown.lower()

    @pytest.mark.asyncio
    async def test_max_hops_capped(self) -> None:
        """max_hops should be capped at 10."""
        n_a = _make_neuron("n1", "A")
        n_b = _make_neuron("n2", "B")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if content_contains == "A" else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=None)

        await explain_connection(storage, "A", "B", max_hops=100)

        # Should be capped at 10, not 100
        storage.get_path.assert_called_with("n1", "n2", max_hops=10, bidirectional=True)

    @pytest.mark.asyncio
    async def test_fiber_evidence_hydrated(self) -> None:
        """Fiber summaries should appear as evidence in steps."""
        n_a = _make_neuron("n1", "authentication")
        n_b = _make_neuron("n2", "JWT")
        s_ab = _make_synapse("s1", "n1", "n2", SynapseType.INVOLVES)

        fiber = _make_fiber("f1", {"n1", "n2"}, summary="JWT tokens used for auth")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if "auth" in (content_contains or "") else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=[(n_b, s_ab)])
        storage.find_fibers_batch = AsyncMock(return_value=[fiber])

        result = await explain_connection(storage, "authentication", "JWT")

        assert result.found is True
        assert len(result.steps) == 1
        assert "JWT tokens used for auth" in result.steps[0].evidence

    @pytest.mark.asyncio
    async def test_bidirectional_flag_passed(self) -> None:
        """get_path should be called with bidirectional=True."""
        n_a = _make_neuron("n1", "A")
        n_b = _make_neuron("n2", "B")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if content_contains == "A" else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=None)

        await explain_connection(storage, "A", "B")

        storage.get_path.assert_called_with("n1", "n2", max_hops=6, bidirectional=True)

    @pytest.mark.asyncio
    async def test_same_neuron_skipped(self) -> None:
        """When source and target resolve to same neuron, should try other candidates."""
        n_same = _make_neuron("n1", "React performance")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[n_same])
        storage.get_path = AsyncMock(return_value=None)

        result = await explain_connection(storage, "React", "performance")

        # get_path should never be called since src.id == tgt.id
        storage.get_path.assert_not_called()
        assert result.found is False

    @pytest.mark.asyncio
    async def test_default_max_hops(self) -> None:
        """Default max_hops should be 6."""
        n_a = _make_neuron("n1", "A")
        n_b = _make_neuron("n2", "B")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if content_contains == "A" else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=None)

        await explain_connection(storage, "A", "B")

        storage.get_path.assert_called_with("n1", "n2", max_hops=6, bidirectional=True)

    @pytest.mark.asyncio
    async def test_avg_weight_calculation(self) -> None:
        """avg_weight should be correctly computed from step weights."""
        n_a = _make_neuron("n1", "React")
        n_b = _make_neuron("n2", "virtual DOM")
        n_c = _make_neuron("n3", "performance")
        s_ab = _make_synapse("s1", "n1", "n2", SynapseType.RELATED_TO, weight=0.6)
        s_bc = _make_synapse("s2", "n2", "n3", SynapseType.LEADS_TO, weight=0.8)

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if "React" in (content_contains or "") else [n_c]
            )
        )
        storage.get_path = AsyncMock(return_value=[(n_b, s_ab), (n_c, s_bc)])
        storage.find_fibers_batch = AsyncMock(return_value=[])

        result = await explain_connection(storage, "React", "performance")

        assert result.found is True
        assert result.avg_weight == round((0.6 + 0.8) / 2, 3)


class TestPathConfidence:
    """Tests for Tribunal-style confidence scoring."""

    def test_single_hop_high_weight(self) -> None:
        """1 hop with weight=1.0 → exp(-0.4) ≈ 0.670."""
        conf = compute_path_confidence(1, 1.0)
        assert abs(conf - 0.6703) < 0.001

    def test_two_hops(self) -> None:
        """2 hops with weight=1.0 → exp(-0.8) ≈ 0.449."""
        conf = compute_path_confidence(2, 1.0)
        assert abs(conf - 0.4493) < 0.001

    def test_five_hops(self) -> None:
        """5 hops → exp(-2.0) ≈ 0.135."""
        conf = compute_path_confidence(5, 1.0)
        assert abs(conf - 0.1353) < 0.001

    def test_ten_hops(self) -> None:
        """10 hops → exp(-4.0) ≈ 0.018."""
        conf = compute_path_confidence(10, 1.0)
        assert abs(conf - 0.0183) < 0.002

    def test_weight_scaling(self) -> None:
        """Weight scales confidence linearly."""
        full = compute_path_confidence(2, 1.0)
        half = compute_path_confidence(2, 0.5)
        assert abs(half - full * 0.5) < 0.001

    def test_zero_hops(self) -> None:
        """Zero hops → 0.0 (invalid path)."""
        assert compute_path_confidence(0, 1.0) == 0.0

    def test_zero_weight(self) -> None:
        """Zero weight → 0.0."""
        assert compute_path_confidence(3, 0.0) == 0.0

    def test_weight_capped_at_one(self) -> None:
        """Weight > 1.0 should be capped."""
        conf = compute_path_confidence(1, 2.0)
        assert conf == compute_path_confidence(1, 1.0)

    def test_strength_labels(self) -> None:
        """Strength labels should map correctly."""
        assert _confidence_to_strength(0.8) == "strong"
        assert _confidence_to_strength(0.7) == "strong"
        assert _confidence_to_strength(0.5) == "moderate"
        assert _confidence_to_strength(0.4) == "moderate"
        assert _confidence_to_strength(0.3) == "weak"
        assert _confidence_to_strength(0.2) == "weak"
        assert _confidence_to_strength(0.1) == "tenuous"
        assert _confidence_to_strength(0.0) == "tenuous"


class TestExplainConnectionConfidence:
    """Integration tests: confidence field in explain_connection results."""

    @pytest.mark.asyncio
    async def test_result_includes_confidence(self) -> None:
        """ConnectionExplanation should include confidence and strength."""
        n_a = _make_neuron("n1", "React")
        n_b = _make_neuron("n2", "DOM")
        s_ab = _make_synapse("s1", "n1", "n2", weight=0.8)

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if "React" in (content_contains or "") else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=[(n_b, s_ab)])
        storage.find_fibers_batch = AsyncMock(return_value=[])

        result = await explain_connection(storage, "React", "DOM")

        assert result.found is True
        assert result.confidence > 0.0
        assert result.strength in ("strong", "moderate", "weak", "tenuous")
        # 1 hop, weight 0.8: exp(-0.4) * 0.8 ≈ 0.536
        assert abs(result.confidence - 0.536) < 0.01
        assert result.strength == "moderate"

    @pytest.mark.asyncio
    async def test_markdown_includes_confidence(self) -> None:
        """Markdown output should show confidence score and strength."""
        n_a = _make_neuron("n1", "A")
        n_b = _make_neuron("n2", "B")
        s_ab = _make_synapse("s1", "n1", "n2", weight=1.0)

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=lambda content_contains=None, limit=5: (
                [n_a] if content_contains == "A" else [n_b]
            )
        )
        storage.get_path = AsyncMock(return_value=[(n_b, s_ab)])
        storage.find_fibers_batch = AsyncMock(return_value=[])

        result = await explain_connection(storage, "A", "B")

        assert "Confidence:" in result.markdown
        assert "moderate" in result.markdown

    @pytest.mark.asyncio
    async def test_not_found_has_zero_confidence(self) -> None:
        """Not-found results should have 0 confidence."""
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])

        result = await explain_connection(storage, "X", "Y")

        assert result.confidence == 0.0
        assert result.strength == "none"


class TestConnectionMCPHandler:
    """Tests for nmem_explain MCP tool integration."""

    @pytest.fixture
    def server(self) -> MCPServer:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
                tool_tier=ToolTierConfig(tier="full"),
                response=ResponseConfig(),
                encryption=MagicMock(enabled=False, auto_encrypt_sensitive=False),
                safety=MagicMock(auto_redact_min_severity=3),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_explain_missing_args(self, server: MCPServer) -> None:
        """Should return error when required args missing."""
        result = await server._explain({"from_entity": "React"})
        assert "error" in result

        result = await server._explain({"to_entity": "performance"})
        assert "error" in result

        result = await server._explain({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_explain_tool_in_schemas(self) -> None:
        """nmem_explain should be in tool schemas."""
        from neural_memory.mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "nmem_explain" in names

    @pytest.mark.asyncio
    async def test_explain_tool_dispatch(self, server: MCPServer) -> None:
        """nmem_explain should be in dispatch dict."""
        mock_storage = AsyncMock()
        mock_storage.find_neurons = AsyncMock(return_value=[])
        mock_brain = MagicMock()
        mock_brain.id = "test-brain-id"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool(
                "nmem_explain",
                {"from_entity": "React", "to_entity": "performance"},
            )

        assert "found" in result or "error" in result

    @pytest.mark.asyncio
    async def test_explain_via_handle_message(self, server: MCPServer) -> None:
        """nmem_explain should work through JSON-RPC handle_message."""
        mock_storage = AsyncMock()
        mock_storage.find_neurons = AsyncMock(return_value=[])
        mock_brain = MagicMock()
        mock_brain.id = "test-brain-id"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)

        with patch.object(server, "get_storage", return_value=mock_storage):
            message = {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "nmem_explain",
                    "arguments": {"from_entity": "A", "to_entity": "B"},
                },
            }
            response = await handle_message(server, message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 99
        assert "result" in response
