"""Tests for related memory discovery in _remember() flow (P1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.server import MCPServer


@dataclass
class _FakeNeuron:
    id: str
    content: str
    metadata: dict[str, Any]
    type: MagicMock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.type is None:
            self.type = MagicMock(value="concept")


@dataclass
class _FakeFiber:
    id: str
    anchor_neuron_id: str
    summary: str
    neuron_ids: set[str]
    synapse_ids: set[str] = None  # type: ignore[assignment]
    tags: set[str] = None  # type: ignore[assignment]
    auto_tags: set[str] = None  # type: ignore[assignment]
    agent_tags: set[str] = None  # type: ignore[assignment]
    metadata: dict[str, Any] = None  # type: ignore[assignment]
    salience: float = 0.5
    conductivity: float = 1.0
    last_conducted: Any = None
    frequency: int = 1

    def __post_init__(self) -> None:
        if self.synapse_ids is None:
            self.synapse_ids = set()
        if self.tags is None:
            self.tags = set()
        if self.auto_tags is None:
            self.auto_tags = set()
        if self.agent_tags is None:
            self.agent_tags = set()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class _FakeActivationResult:
    neuron_id: str
    activation_level: float
    hop_distance: int
    path: list[str]
    source_anchor: str


@dataclass
class _FakeEncodingResult:
    fiber: _FakeFiber
    neurons_created: list[_FakeNeuron]
    neurons_linked: list[str]
    synapses_created: list[Any]
    conflicts_detected: int = 0


@dataclass
class _FakeBrain:
    id: str = "test-brain"
    name: str = "test-brain"
    config: Any = None

    def __post_init__(self) -> None:
        if self.config is None:
            from neural_memory.core.brain import BrainConfig

            self.config = BrainConfig()


def _make_server() -> MCPServer:
    """Create a MCPServer with mocked storage."""
    server = MCPServer()
    server.config = MagicMock()
    server.config.maintenance.enabled = False
    server.config.auto.enabled = False
    return server


class TestRelatedMemoryDiscovery:
    """Test the related memory discovery feature in _remember()."""

    @pytest.mark.asyncio
    async def test_no_related_memories_empty_brain(self) -> None:
        """On an empty brain, _remember should succeed with no related_memories."""
        server = _make_server()

        new_fiber = _FakeFiber(
            id="fiber-new",
            anchor_neuron_id="anchor-new",
            summary="Test memory",
            neuron_ids={"anchor-new"},
        )
        encode_result = _FakeEncodingResult(
            fiber=new_fiber,
            neurons_created=[_FakeNeuron("anchor-new", "test content", {"is_anchor": True})],
            neurons_linked=[],
            synapses_created=[],
        )

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_brain = AsyncMock(return_value=_FakeBrain())
        mock_storage.get_neurons_batch = AsyncMock(return_value={})
        mock_storage.find_fibers_batch = AsyncMock(return_value=[])

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        # SpreadingActivation returns empty (no related neurons)
        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_encoder,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_activation,
        ):
            mock_encoder.return_value.encode = AsyncMock(return_value=encode_result)
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(
                return_value={
                    "anchor-new": _FakeActivationResult(
                        neuron_id="anchor-new",
                        activation_level=1.0,
                        hop_distance=0,
                        path=["anchor-new"],
                        source_anchor="anchor-new",
                    ),
                }
            )
            mock_activation.return_value = mock_activator

            response = await server._remember({"content": "test content"})

        assert response["success"] is True
        assert "related_memories" not in response

    @pytest.mark.asyncio
    async def test_related_memories_returned(self) -> None:
        """When related anchor neurons exist, related_memories should be populated."""
        server = _make_server()

        new_fiber = _FakeFiber(
            id="fiber-new",
            anchor_neuron_id="anchor-new",
            summary="New memory",
            neuron_ids={"anchor-new"},
        )
        encode_result = _FakeEncodingResult(
            fiber=new_fiber,
            neurons_created=[_FakeNeuron("anchor-new", "new content", {"is_anchor": True})],
            neurons_linked=[],
            synapses_created=[],
        )

        related_fiber = _FakeFiber(
            id="fiber-related",
            anchor_neuron_id="anchor-related",
            summary="Related memory about similar topic",
            neuron_ids={"anchor-related"},
        )

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_brain = AsyncMock(return_value=_FakeBrain())
        mock_storage.get_neurons_batch = AsyncMock(
            return_value={
                "anchor-related": _FakeNeuron(
                    "anchor-related", "related content", {"is_anchor": True}
                ),
            }
        )
        mock_storage.find_fibers_batch = AsyncMock(return_value=[related_fiber])

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_encoder,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_activation,
        ):
            mock_encoder.return_value.encode = AsyncMock(return_value=encode_result)
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(
                return_value={
                    "anchor-new": _FakeActivationResult(
                        neuron_id="anchor-new",
                        activation_level=1.0,
                        hop_distance=0,
                        path=["anchor-new"],
                        source_anchor="anchor-new",
                    ),
                    "shared-entity": _FakeActivationResult(
                        neuron_id="shared-entity",
                        activation_level=0.5,
                        hop_distance=1,
                        path=["anchor-new", "shared-entity"],
                        source_anchor="anchor-new",
                    ),
                    "anchor-related": _FakeActivationResult(
                        neuron_id="anchor-related",
                        activation_level=0.25,
                        hop_distance=2,
                        path=["anchor-new", "shared-entity", "anchor-related"],
                        source_anchor="anchor-new",
                    ),
                }
            )
            mock_activation.return_value = mock_activator

            response = await server._remember({"content": "new content"})

        assert response["success"] is True
        assert "related_memories" in response
        related = response["related_memories"]
        assert len(related) == 1
        assert related[0]["fiber_id"] == "fiber-related"
        assert related[0]["similarity"] == 0.25
        assert "Related memory" in related[0]["preview"]

    @pytest.mark.asyncio
    async def test_activation_failure_graceful(self) -> None:
        """If spreading activation fails, _remember should still succeed."""
        server = _make_server()

        new_fiber = _FakeFiber(
            id="fiber-new",
            anchor_neuron_id="anchor-new",
            summary="Test memory",
            neuron_ids={"anchor-new"},
        )
        encode_result = _FakeEncodingResult(
            fiber=new_fiber,
            neurons_created=[_FakeNeuron("anchor-new", "test", {"is_anchor": True})],
            neurons_linked=[],
            synapses_created=[],
        )

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_brain = AsyncMock(return_value=_FakeBrain())

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_encoder,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_activation,
        ):
            mock_encoder.return_value.encode = AsyncMock(return_value=encode_result)
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(side_effect=RuntimeError("activation failed"))
            mock_activation.return_value = mock_activator

            response = await server._remember({"content": "test content"})

        assert response["success"] is True
        assert "related_memories" not in response

    @pytest.mark.asyncio
    async def test_max_three_related_memories(self) -> None:
        """Should return at most 3 related memories."""
        server = _make_server()

        new_fiber = _FakeFiber(
            id="fiber-new",
            anchor_neuron_id="anchor-new",
            summary="New memory",
            neuron_ids={"anchor-new"},
        )
        encode_result = _FakeEncodingResult(
            fiber=new_fiber,
            neurons_created=[_FakeNeuron("anchor-new", "content", {"is_anchor": True})],
            neurons_linked=[],
            synapses_created=[],
        )

        # Create 5 related fibers
        related_fibers = []
        activations = {
            "anchor-new": _FakeActivationResult(
                neuron_id="anchor-new",
                activation_level=1.0,
                hop_distance=0,
                path=["anchor-new"],
                source_anchor="anchor-new",
            ),
        }
        neurons_batch = {}
        for i in range(5):
            fid = f"fiber-r{i}"
            aid = f"anchor-r{i}"
            related_fibers.append(
                _FakeFiber(id=fid, anchor_neuron_id=aid, summary=f"Related {i}", neuron_ids={aid})
            )
            activations[aid] = _FakeActivationResult(
                neuron_id=aid,
                activation_level=0.3 - i * 0.05,
                hop_distance=2,
                path=["anchor-new", "shared", aid],
                source_anchor="anchor-new",
            )
            neurons_batch[aid] = _FakeNeuron(aid, f"related content {i}", {"is_anchor": True})

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_brain = AsyncMock(return_value=_FakeBrain())
        mock_storage.get_neurons_batch = AsyncMock(return_value=neurons_batch)
        mock_storage.find_fibers_batch = AsyncMock(return_value=related_fibers)

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_encoder,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_activation,
        ):
            mock_encoder.return_value.encode = AsyncMock(return_value=encode_result)
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(return_value=activations)
            mock_activation.return_value = mock_activator

            response = await server._remember({"content": "content"})

        assert response["success"] is True
        assert "related_memories" in response
        assert len(response["related_memories"]) <= 3
