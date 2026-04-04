"""Tests for brain milestone engine and MCP handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.milestone import (
    MILESTONE_META,
    MILESTONES,
    MilestoneEngine,
    MilestoneReport,
    MilestoneSnapshot,
    _get_achieved_milestones,
    _get_next_milestone,
)
from neural_memory.mcp.milestone_handler import MilestoneHandler
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow

# ── Unit tests for helpers ────────────────────────────────────


class TestMilestoneHelpers:
    """Tests for milestone helper functions."""

    def test_get_next_milestone_zero(self) -> None:
        assert _get_next_milestone(0) == 100

    def test_get_next_milestone_50(self) -> None:
        assert _get_next_milestone(50) == 100

    def test_get_next_milestone_100(self) -> None:
        assert _get_next_milestone(100) == 250

    def test_get_next_milestone_9999(self) -> None:
        assert _get_next_milestone(9999) == 10000

    def test_get_next_milestone_all_achieved(self) -> None:
        assert _get_next_milestone(10000) is None
        assert _get_next_milestone(99999) is None

    def test_get_achieved_milestones_zero(self) -> None:
        assert _get_achieved_milestones(0) == []

    def test_get_achieved_milestones_150(self) -> None:
        assert _get_achieved_milestones(150) == [100]

    def test_get_achieved_milestones_1000(self) -> None:
        assert _get_achieved_milestones(1000) == [100, 250, 500, 1000]

    def test_get_achieved_milestones_all(self) -> None:
        assert _get_achieved_milestones(10000) == list(MILESTONES)

    def test_milestone_meta_has_all_thresholds(self) -> None:
        for m in MILESTONES:
            assert m in MILESTONE_META, f"Missing meta for milestone {m}"


# ── MilestoneEngine tests ─────────────────────────────────────


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


async def _populate_neurons(storage: InMemoryStorage, brain_id: str, count: int) -> None:
    """Add `count` neurons to the storage."""
    for i in range(count):
        neuron = Neuron.create(
            content=f"Test neuron {i}",
            type=NeuronType.CONCEPT,
        )
        await storage.add_neuron(neuron)


async def _populate_fibers(storage: InMemoryStorage, brain_id: str, count: int) -> None:
    """Add `count` fibers to the storage."""
    for i in range(count):
        anchor_id = f"anchor-{i}"
        fiber = Fiber.create(
            summary=f"Test fiber {i}",
            neuron_ids={anchor_id},
            synapse_ids=set(),
            anchor_neuron_id=anchor_id,
        )
        await storage.add_fiber(fiber)


class TestMilestoneEngine:
    """Tests for MilestoneEngine."""

    async def test_check_no_milestone_empty_brain(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        engine = MilestoneEngine(storage)
        result = await engine.check_and_record(brain.id)
        assert result is None

    async def test_check_no_milestone_below_threshold(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 50)
        engine = MilestoneEngine(storage)
        result = await engine.check_and_record(brain.id)
        assert result is None

    async def test_check_milestone_reached(self, storage: InMemoryStorage, brain: Brain) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 10)
        engine = MilestoneEngine(storage)
        result = await engine.check_and_record(brain.id)
        assert result is not None
        assert isinstance(result, MilestoneReport)
        assert result.snapshot.threshold == 100
        assert result.title == "First Hundred"
        assert result.snapshot.neuron_count == 100

    async def test_milestone_persisted_in_metadata(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 5)
        engine = MilestoneEngine(storage)
        await engine.check_and_record(brain.id)

        # Check milestone was saved
        updated_brain = await storage.get_brain(brain.id)
        assert updated_brain is not None
        milestones = updated_brain.metadata.get("_milestones", [])
        assert len(milestones) == 1
        assert milestones[0]["threshold"] == 100

    async def test_milestone_not_recorded_twice(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 5)
        engine = MilestoneEngine(storage)

        # First check — records milestone
        result1 = await engine.check_and_record(brain.id)
        assert result1 is not None

        # Second check — no new milestone
        result2 = await engine.check_and_record(brain.id)
        assert result2 is None

    async def test_get_progress_below_first(self, storage: InMemoryStorage, brain: Brain) -> None:
        await _populate_neurons(storage, brain.id, 30)
        engine = MilestoneEngine(storage)
        progress = await engine.get_progress(brain.id)
        assert progress["next_milestone"] == 100
        assert progress["remaining"] == 70
        assert progress["progress_pct"] == 30.0

    async def test_get_progress_all_achieved(self, storage: InMemoryStorage, brain: Brain) -> None:
        await _populate_neurons(storage, brain.id, 10000)
        engine = MilestoneEngine(storage)
        progress = await engine.get_progress(brain.id)
        assert progress["next_milestone"] is None

    async def test_get_history_empty(self, storage: InMemoryStorage, brain: Brain) -> None:
        engine = MilestoneEngine(storage)
        history = await engine.get_history(brain.id)
        assert history == []

    async def test_get_history_after_milestone(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 5)
        engine = MilestoneEngine(storage)
        await engine.check_and_record(brain.id)

        history = await engine.get_history(brain.id)
        assert len(history) == 1
        assert history[0]["threshold"] == 100

    async def test_generate_report_empty_brain(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        engine = MilestoneEngine(storage)
        report = await engine.generate_report(brain.id)
        assert report is None

    async def test_generate_report_nonempty(self, storage: InMemoryStorage, brain: Brain) -> None:
        await _populate_neurons(storage, brain.id, 50)
        await _populate_fibers(storage, brain.id, 5)
        engine = MilestoneEngine(storage)
        report = await engine.generate_report(brain.id)
        assert report is not None
        assert "# Milestone:" in report.markdown
        assert report.snapshot.neuron_count == 50

    async def test_report_markdown_contains_stats(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 10)
        engine = MilestoneEngine(storage)
        result = await engine.check_and_record(brain.id)
        assert result is not None
        assert "Neurons: **100**" in result.markdown
        assert "## Brain Snapshot" in result.markdown

    async def test_multiple_milestones_records_highest(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        """When jumping from 0 to 600 neurons, record milestone 500 as main."""
        await _populate_neurons(storage, brain.id, 600)
        await _populate_fibers(storage, brain.id, 10)
        engine = MilestoneEngine(storage)
        result = await engine.check_and_record(brain.id)
        assert result is not None
        assert result.snapshot.threshold == 500

        # All intermediate milestones should also be recorded
        updated_brain = await storage.get_brain(brain.id)
        assert updated_brain is not None
        milestones = updated_brain.metadata.get("_milestones", [])
        thresholds = {m["threshold"] for m in milestones}
        assert 100 in thresholds
        assert 250 in thresholds
        assert 500 in thresholds


# ── MilestoneHandler MCP tests ────────────────────────────────


class MockMilestoneServer(MilestoneHandler):
    """Mock server for testing MilestoneHandler mixin."""

    def __init__(self, storage: InMemoryStorage, config: MagicMock) -> None:
        self._storage = storage
        self.config = config

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


@pytest.fixture
def server(storage: InMemoryStorage) -> MockMilestoneServer:
    config = MagicMock()
    return MockMilestoneServer(storage, config)


class TestMilestoneHandler:
    """Tests for MilestoneHandler MCP mixin."""

    async def test_check_no_milestone(self, server: MockMilestoneServer) -> None:
        result = await server._milestone({"action": "check"})
        assert result["action"] == "check"
        assert result["new_milestone"] is False

    async def test_check_with_milestone(
        self, server: MockMilestoneServer, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 5)
        result = await server._milestone({"action": "check"})
        assert result["action"] == "check"
        assert result["new_milestone"] is True
        assert result["milestone"] == 100
        assert "markdown" in result

    async def test_progress_action(
        self, server: MockMilestoneServer, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 50)
        result = await server._milestone({"action": "progress"})
        assert result["action"] == "progress"
        assert result["next_milestone"] == 100
        assert result["remaining"] == 50

    async def test_history_empty(self, server: MockMilestoneServer) -> None:
        result = await server._milestone({"action": "history"})
        assert result["action"] == "history"
        assert result["milestones"] == []

    async def test_history_after_check(
        self, server: MockMilestoneServer, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 100)
        await _populate_fibers(storage, brain.id, 5)
        await server._milestone({"action": "check"})
        result = await server._milestone({"action": "history"})
        assert result["action"] == "history"
        assert result["total"] == 1

    async def test_report_action(
        self, server: MockMilestoneServer, storage: InMemoryStorage, brain: Brain
    ) -> None:
        await _populate_neurons(storage, brain.id, 50)
        await _populate_fibers(storage, brain.id, 5)
        result = await server._milestone({"action": "report"})
        assert result["action"] == "report"
        assert "markdown" in result
        assert result["neuron_count"] == 50

    async def test_report_empty_brain(self, server: MockMilestoneServer) -> None:
        result = await server._milestone({"action": "report"})
        assert result["action"] == "report"
        assert "error" in result

    async def test_unknown_action(self, server: MockMilestoneServer) -> None:
        result = await server._milestone({"action": "invalid"})
        assert "error" in result

    async def test_default_action_is_check(self, server: MockMilestoneServer) -> None:
        result = await server._milestone({})
        assert result["action"] == "check"
        assert result["new_milestone"] is False


class TestMilestoneSnapshot:
    """Tests for MilestoneSnapshot immutability."""

    def test_snapshot_is_frozen(self) -> None:
        snapshot = MilestoneSnapshot(
            threshold=100,
            achieved_at=utcnow(),
            neuron_count=100,
            synapse_count=50,
            fiber_count=30,
            purity_score=75.0,
            grade="B",
            days_from_first_memory=10,
            days_from_prev_milestone=None,
            top_types={"RELATED_TO": 20},
        )
        with pytest.raises(AttributeError):
            snapshot.threshold = 200  # type: ignore[misc]
