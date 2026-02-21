"""Tests for MCP review handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.mcp.review_handler import ReviewHandler
from neural_memory.storage.memory_store import InMemoryStorage


class MockReviewServer(ReviewHandler):
    """Mock server for testing ReviewHandler mixin."""

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
def server(storage: InMemoryStorage) -> MockReviewServer:
    config = MagicMock()
    return MockReviewServer(storage, config)


@pytest.fixture
async def fiber_with_neuron(storage: InMemoryStorage) -> Fiber:
    """Create a fiber with a neuron for testing."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="test concept", neuron_id="n1")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={"n1"},
        synapse_ids=set(),
        anchor_neuron_id="n1",
        summary="Test fiber for review",
    )
    await storage.add_fiber(fiber)
    return fiber


class TestReviewHandler:
    """Tests for ReviewHandler mixin."""

    async def test_queue_empty(self, server: MockReviewServer) -> None:
        """Test queue action with no reviews."""
        result = await server._review({"action": "queue"})
        assert result["action"] == "queue"
        assert result["count"] == 0
        assert result["items"] == []

    async def test_schedule_fiber(self, server: MockReviewServer, fiber_with_neuron: Fiber) -> None:
        """Test scheduling a fiber for review."""
        result = await server._review(
            {
                "action": "schedule",
                "fiber_id": fiber_with_neuron.id,
            }
        )
        assert result["action"] == "schedule"
        assert result["fiber_id"] == fiber_with_neuron.id
        assert result["box"] == 1

    async def test_schedule_already_scheduled(
        self, server: MockReviewServer, fiber_with_neuron: Fiber
    ) -> None:
        """Test scheduling already-scheduled fiber."""
        await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        result = await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        assert "already scheduled" in result["message"]

    async def test_schedule_nonexistent_fiber(self, server: MockReviewServer) -> None:
        """Test scheduling nonexistent fiber returns error."""
        result = await server._review({"action": "schedule", "fiber_id": "nope"})
        assert "error" in result

    async def test_mark_success(self, server: MockReviewServer, fiber_with_neuron: Fiber) -> None:
        """Test marking a review as successful."""
        await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        result = await server._review(
            {
                "action": "mark",
                "fiber_id": fiber_with_neuron.id,
                "success": True,
            }
        )
        assert result["action"] == "mark"
        assert result["success"] is True
        assert result["new_box"] == 2

    async def test_mark_failure(self, server: MockReviewServer, fiber_with_neuron: Fiber) -> None:
        """Test marking a review as failed."""
        await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        result = await server._review(
            {
                "action": "mark",
                "fiber_id": fiber_with_neuron.id,
                "success": False,
            }
        )
        assert result["new_box"] == 1
        assert result["streak"] == 0

    async def test_mark_missing_fiber_id(self, server: MockReviewServer) -> None:
        """Test mark without fiber_id returns error."""
        result = await server._review({"action": "mark"})
        assert "error" in result

    async def test_stats(
        self,
        server: MockReviewServer,
        storage: InMemoryStorage,
        brain: Brain,
        fiber_with_neuron: Fiber,
    ) -> None:
        """Test stats action."""
        await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        result = await server._review({"action": "stats"})
        assert result["action"] == "stats"
        assert result["total"] == 1

    async def test_queue_after_schedule(
        self, server: MockReviewServer, fiber_with_neuron: Fiber
    ) -> None:
        """Test queue shows scheduled fibers."""
        await server._review({"action": "schedule", "fiber_id": fiber_with_neuron.id})
        result = await server._review({"action": "queue"})
        assert result["count"] == 1
        assert result["items"][0]["fiber_id"] == fiber_with_neuron.id
        assert result["items"][0]["summary"] == "Test fiber for review"

    async def test_unknown_action(self, server: MockReviewServer) -> None:
        """Test unknown action returns error."""
        result = await server._review({"action": "invalid"})
        assert "error" in result

    async def test_no_brain(self) -> None:
        """Test with no brain configured."""
        store = InMemoryStorage()
        store.set_brain("nonexistent")
        config = MagicMock()
        s = MockReviewServer(store, config)
        result = await s._review({"action": "queue"})
        assert "error" in result
