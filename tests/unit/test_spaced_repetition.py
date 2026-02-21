"""Tests for spaced repetition engine and review schedule model."""

from __future__ import annotations

from datetime import timedelta

import pytest

from neural_memory.core.review_schedule import (
    LEITNER_INTERVALS,
    MAX_BOX,
    MIN_BOX,
    ReviewSchedule,
)
from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine


class TestReviewSchedule:
    """Tests for the ReviewSchedule frozen dataclass."""

    def test_create(self) -> None:
        """Test factory method creates valid schedule."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id="b1")

        assert schedule.fiber_id == "f1"
        assert schedule.brain_id == "b1"
        assert schedule.box == MIN_BOX
        assert schedule.review_count == 0
        assert schedule.streak == 0
        assert schedule.next_review is not None
        assert schedule.created_at is not None

    def test_advance_success(self) -> None:
        """Test successful review advances box."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id="b1")
        advanced = schedule.advance(success=True)

        assert advanced.box == 2
        assert advanced.streak == 1
        assert advanced.review_count == 1
        assert advanced.last_reviewed is not None
        assert advanced.next_review is not None
        # Box 2 interval is 3 days
        expected_interval = timedelta(days=LEITNER_INTERVALS[2])
        actual_delta = advanced.next_review - advanced.last_reviewed
        # Allow 1 second tolerance
        assert abs(actual_delta - expected_interval).total_seconds() < 1

    def test_advance_failure_resets_to_box_1(self) -> None:
        """Test failed review resets to box 1."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id="b1")
        # Advance twice successfully
        s2 = schedule.advance(success=True)
        s3 = s2.advance(success=True)
        assert s3.box == 3
        assert s3.streak == 2

        # Fail
        failed = s3.advance(success=False)
        assert failed.box == MIN_BOX
        assert failed.streak == 0
        assert failed.review_count == 3

    def test_advance_success_caps_at_max_box(self) -> None:
        """Test box doesn't exceed MAX_BOX on success."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id="b1")
        # Advance to max box
        s = schedule
        for _ in range(MAX_BOX):
            s = s.advance(success=True)
        assert s.box == MAX_BOX

        # One more success should stay at MAX_BOX
        s_extra = s.advance(success=True)
        assert s_extra.box == MAX_BOX

    def test_frozen(self) -> None:
        """Test that ReviewSchedule is immutable."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id="b1")
        with pytest.raises(AttributeError):
            schedule.box = 5  # type: ignore[misc]

    def test_leitner_intervals(self) -> None:
        """Test that all box intervals are defined."""
        for box in range(MIN_BOX, MAX_BOX + 1):
            assert box in LEITNER_INTERVALS
            assert LEITNER_INTERVALS[box] > 0


class TestSpacedRepetitionEngine:
    """Tests for SpacedRepetitionEngine."""

    @pytest.fixture
    def engine(self, storage, brain_config):
        return SpacedRepetitionEngine(storage, brain_config)

    async def test_get_review_queue_empty(self, engine) -> None:
        """Test empty queue returns empty list."""
        items = await engine.get_review_queue()
        assert items == []

    async def test_auto_schedule_fiber(self, engine, storage, brain) -> None:
        """Test auto-scheduling a fiber."""
        # Create a fiber first
        from neural_memory.core.fiber import Fiber

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            summary="Test fiber",
        )
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test", neuron_id="n1")
        await storage.add_neuron(neuron)
        await storage.add_fiber(fiber)

        schedule = await engine.auto_schedule_fiber(fiber.id, brain.id)
        assert schedule is not None
        assert schedule.fiber_id == fiber.id
        assert schedule.box == MIN_BOX

    async def test_auto_schedule_idempotent(self, engine, storage, brain) -> None:
        """Test auto-scheduling same fiber twice returns None."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test", neuron_id="n1")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={"n1"}, synapse_ids=set(), anchor_neuron_id="n1")
        await storage.add_fiber(fiber)

        first = await engine.auto_schedule_fiber(fiber.id, brain.id)
        assert first is not None

        second = await engine.auto_schedule_fiber(fiber.id, brain.id)
        assert second is None

    async def test_process_review_success(self, engine, storage, brain) -> None:
        """Test processing a successful review."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test", neuron_id="n1")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={"n1"}, synapse_ids=set(), anchor_neuron_id="n1")
        await storage.add_fiber(fiber)
        await engine.auto_schedule_fiber(fiber.id, brain.id)

        result = await engine.process_review(fiber.id, success=True)
        assert result["success"] is True
        assert result["new_box"] == 2
        assert result["streak"] == 1

    async def test_process_review_failure(self, engine, storage, brain) -> None:
        """Test processing a failed review."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test", neuron_id="n1")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={"n1"}, synapse_ids=set(), anchor_neuron_id="n1")
        await storage.add_fiber(fiber)
        await engine.auto_schedule_fiber(fiber.id, brain.id)

        # First success
        await engine.process_review(fiber.id, success=True)

        # Then fail
        result = await engine.process_review(fiber.id, success=False)
        assert result["success"] is False
        assert result["new_box"] == MIN_BOX
        assert result["streak"] == 0

    async def test_process_review_not_scheduled(self, engine) -> None:
        """Test reviewing unscheduled fiber returns error."""
        result = await engine.process_review("nonexistent", success=True)
        assert "error" in result

    async def test_get_stats(self, engine, storage, brain) -> None:
        """Test review statistics."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test", neuron_id="n1")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={"n1"}, synapse_ids=set(), anchor_neuron_id="n1")
        await storage.add_fiber(fiber)
        await engine.auto_schedule_fiber(fiber.id, brain.id)

        stats = await engine.get_stats()
        assert stats["total"] == 1
        assert stats["due"] == 1  # Just created, immediately due
        assert stats["box_1"] == 1

    async def test_review_queue_with_fiber(self, engine, storage, brain) -> None:
        """Test queue returns fiber summaries."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test concept", neuron_id="n1")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            summary="A test fiber summary",
        )
        await storage.add_fiber(fiber)
        await engine.auto_schedule_fiber(fiber.id, brain.id)

        items = await engine.get_review_queue()
        assert len(items) == 1
        assert items[0]["fiber_id"] == fiber.id
        assert items[0]["summary"] == "A test fiber summary"
        assert items[0]["box"] == 1


class TestStorageReviewMethods:
    """Tests for review schedule storage CRUD."""

    async def test_add_and_get_schedule(self, storage, brain) -> None:
        """Test adding and retrieving a schedule."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id=brain.id)
        await storage.add_review_schedule(schedule)

        retrieved = await storage.get_review_schedule("f1")
        assert retrieved is not None
        assert retrieved.fiber_id == "f1"
        assert retrieved.box == 1

    async def test_upsert_schedule(self, storage, brain) -> None:
        """Test upserting updates existing schedule."""
        s1 = ReviewSchedule.create(fiber_id="f1", brain_id=brain.id)
        await storage.add_review_schedule(s1)

        s2 = s1.advance(success=True)
        await storage.add_review_schedule(s2)

        retrieved = await storage.get_review_schedule("f1")
        assert retrieved is not None
        assert retrieved.box == 2

    async def test_delete_schedule(self, storage, brain) -> None:
        """Test deleting a schedule."""
        schedule = ReviewSchedule.create(fiber_id="f1", brain_id=brain.id)
        await storage.add_review_schedule(schedule)

        deleted = await storage.delete_review_schedule("f1")
        assert deleted is True

        retrieved = await storage.get_review_schedule("f1")
        assert retrieved is None

    async def test_delete_nonexistent(self, storage) -> None:
        """Test deleting nonexistent schedule returns False."""
        deleted = await storage.delete_review_schedule("nope")
        assert deleted is False

    async def test_get_due_reviews(self, storage, brain) -> None:
        """Test getting due reviews."""
        s1 = ReviewSchedule.create(fiber_id="f1", brain_id=brain.id)
        await storage.add_review_schedule(s1)

        # Also create one that's not due yet (advance to box 5)
        s2 = ReviewSchedule.create(fiber_id="f2", brain_id=brain.id)
        advanced = (
            s2.advance(success=True)
            .advance(success=True)
            .advance(success=True)
            .advance(success=True)
        )
        await storage.add_review_schedule(advanced)

        due = await storage.get_due_reviews(limit=10)
        # Only f1 should be due (f2 is in box 5, next_review is far in the future)
        assert len(due) == 1
        assert due[0].fiber_id == "f1"

    async def test_get_review_stats(self, storage, brain) -> None:
        """Test review stats."""
        s1 = ReviewSchedule.create(fiber_id="f1", brain_id=brain.id)
        await storage.add_review_schedule(s1)

        s2 = ReviewSchedule.create(fiber_id="f2", brain_id=brain.id)
        advanced = s2.advance(success=True)  # Box 2
        await storage.add_review_schedule(advanced)

        stats = await storage.get_review_stats()
        assert stats["total"] == 2
        assert stats["box_1"] == 1
        assert stats["box_2"] == 1
