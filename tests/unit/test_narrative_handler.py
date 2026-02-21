"""Tests for MCP narrative handler."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.mcp.narrative_handler import NarrativeHandler
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


class MockNarrativeServer(NarrativeHandler):
    """Mock server for testing NarrativeHandler mixin."""

    def __init__(self, storage: InMemoryStorage, config: MagicMock) -> None:
        self._storage = storage
        self.config = config

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig()


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain.create(name="test", config=brain_config)


@pytest.fixture
async def storage(brain: Brain) -> InMemoryStorage:
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest.fixture
def server(storage: InMemoryStorage) -> MockNarrativeServer:
    config = MagicMock()
    return MockNarrativeServer(storage, config)


class TestNarrativeHandler:
    """Tests for NarrativeHandler mixin."""

    async def test_timeline_action(self, server: MockNarrativeServer) -> None:
        """Test timeline action with date range."""
        now = utcnow()
        result = await server._narrative(
            {
                "action": "timeline",
                "start_date": (now - timedelta(days=7)).isoformat(),
                "end_date": now.isoformat(),
            }
        )
        assert result["action"] == "timeline"
        assert "markdown" in result

    async def test_timeline_missing_dates(self, server: MockNarrativeServer) -> None:
        """Test timeline without dates returns error."""
        result = await server._narrative({"action": "timeline"})
        assert "error" in result

    async def test_timeline_invalid_dates(self, server: MockNarrativeServer) -> None:
        """Test timeline with bad date format."""
        result = await server._narrative(
            {
                "action": "timeline",
                "start_date": "not-a-date",
                "end_date": "also-not-a-date",
            }
        )
        assert "error" in result

    async def test_topic_action(
        self, server: MockNarrativeServer, storage: InMemoryStorage
    ) -> None:
        """Test topic action."""
        n = Neuron.create(type=NeuronType.CONCEPT, content="test topic", neuron_id="n1")
        await storage.add_neuron(n)
        f = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            summary="About test topic",
        )
        await storage.add_fiber(f)

        result = await server._narrative({"action": "topic", "topic": "test topic"})
        assert result["action"] == "topic"
        assert "markdown" in result

    async def test_topic_missing_topic(self, server: MockNarrativeServer) -> None:
        """Test topic without topic param returns error."""
        result = await server._narrative({"action": "topic"})
        assert "error" in result

    async def test_causal_action(self, server: MockNarrativeServer) -> None:
        """Test causal action with no matching neurons."""
        result = await server._narrative({"action": "causal", "topic": "nonexistent"})
        assert result["action"] == "causal"
        assert result["items"] == 0

    async def test_causal_missing_topic(self, server: MockNarrativeServer) -> None:
        """Test causal without topic returns error."""
        result = await server._narrative({"action": "causal"})
        assert "error" in result

    async def test_unknown_action(self, server: MockNarrativeServer) -> None:
        """Test unknown action returns error."""
        result = await server._narrative({"action": "invalid"})
        assert "error" in result

    async def test_no_brain(self) -> None:
        """Test with no brain configured."""
        store = InMemoryStorage()
        store.set_brain("nonexistent")
        config = MagicMock()
        s = MockNarrativeServer(store, config)
        result = await s._narrative({"action": "topic", "topic": "test"})
        assert "error" in result
