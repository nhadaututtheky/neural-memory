"""Unit tests for sequence mining — habit detection from action events."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.action_event import ActionEvent
from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.sequence_mining import (
    SequencePair,
    extract_habit_candidates,
    heuristic_habit_name,
    learn_habits,
    mine_sequential_pairs,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context configured for habit testing."""
    s = InMemoryStorage()
    brain = Brain.create(
        name="habit-test",
        config=BrainConfig(
            habit_min_frequency=2,
            sequential_window_seconds=60.0,
        ),
        owner_id="test",
    )
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


# ── mine_sequential_pairs ────────────────────────────────────────


class TestMineSequentialPairs:
    """Tests for mine_sequential_pairs pure function."""

    def test_three_events_one_session(self) -> None:
        """Three consecutive events in one session produce two pairs."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base),
            ActionEvent(id="2", brain_id="b", session_id="s1", action_type="remember", created_at=base + timedelta(seconds=5)),
            ActionEvent(id="3", brain_id="b", session_id="s1", action_type="context", created_at=base + timedelta(seconds=10)),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert len(pairs) == 2
        pair_keys = {(p.action_a, p.action_b) for p in pairs}
        assert ("recall", "remember") in pair_keys
        assert ("remember", "context") in pair_keys
        for p in pairs:
            assert p.count == 1

    def test_window_filters_large_gaps(self) -> None:
        """Events separated by more than window_seconds are not counted."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base),
            ActionEvent(id="2", brain_id="b", session_id="s1", action_type="remember", created_at=base + timedelta(seconds=120)),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert len(pairs) == 0

    def test_groups_by_session_id(self) -> None:
        """Events in different sessions are mined independently."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base),
            ActionEvent(id="2", brain_id="b", session_id="s1", action_type="remember", created_at=base + timedelta(seconds=5)),
            ActionEvent(id="3", brain_id="b", session_id="s2", action_type="recall", created_at=base),
            ActionEvent(id="4", brain_id="b", session_id="s2", action_type="remember", created_at=base + timedelta(seconds=5)),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        # Same pair from two sessions should aggregate count
        assert len(pairs) == 1
        assert pairs[0].action_a == "recall"
        assert pairs[0].action_b == "remember"
        assert pairs[0].count == 2

    def test_empty_list_returns_empty(self) -> None:
        """Empty event list produces no pairs."""
        pairs = mine_sequential_pairs([], window_seconds=30.0)
        assert pairs == []

    def test_avg_gap_seconds(self) -> None:
        """Average gap is computed correctly across sessions."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="A", created_at=base),
            ActionEvent(id="2", brain_id="b", session_id="s1", action_type="B", created_at=base + timedelta(seconds=10)),
            ActionEvent(id="3", brain_id="b", session_id="s2", action_type="A", created_at=base),
            ActionEvent(id="4", brain_id="b", session_id="s2", action_type="B", created_at=base + timedelta(seconds=20)),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=60.0)

        assert len(pairs) == 1
        assert pairs[0].avg_gap_seconds == pytest.approx(15.0)

    def test_sorted_by_count_descending(self) -> None:
        """Results are sorted by count descending."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            # recall->remember appears twice
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base),
            ActionEvent(id="2", brain_id="b", session_id="s1", action_type="remember", created_at=base + timedelta(seconds=5)),
            ActionEvent(id="3", brain_id="b", session_id="s2", action_type="recall", created_at=base),
            ActionEvent(id="4", brain_id="b", session_id="s2", action_type="remember", created_at=base + timedelta(seconds=5)),
            # context->recall appears once
            ActionEvent(id="5", brain_id="b", session_id="s3", action_type="context", created_at=base),
            ActionEvent(id="6", brain_id="b", session_id="s3", action_type="recall", created_at=base + timedelta(seconds=5)),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert pairs[0].count >= pairs[-1].count


# ── extract_habit_candidates ─────────────────────────────────────


class TestExtractHabitCandidates:
    """Tests for extract_habit_candidates pure function."""

    def test_bigrams_from_frequent_pairs(self) -> None:
        """Frequent pairs produce bigram candidates."""
        pairs = [
            SequencePair(action_a="recall", action_b="remember", count=5, avg_gap_seconds=3.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3, total_sessions=10)

        assert len(candidates) >= 1
        bigrams = [c for c in candidates if len(c.steps) == 2]
        assert len(bigrams) == 1
        assert bigrams[0].steps == ("recall", "remember")
        assert bigrams[0].frequency == 5
        assert bigrams[0].confidence == pytest.approx(0.5)

    def test_trigrams_from_chained_pairs(self) -> None:
        """A->B and B->C produce trigram A->B->C."""
        pairs = [
            SequencePair(action_a="recall", action_b="edit", count=4, avg_gap_seconds=2.0),
            SequencePair(action_a="edit", action_b="test", count=3, avg_gap_seconds=5.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3, total_sessions=5)

        trigrams = [c for c in candidates if len(c.steps) == 3]
        assert len(trigrams) == 1
        assert trigrams[0].steps == ("recall", "edit", "test")
        assert trigrams[0].frequency == 3  # min(4, 3)
        assert trigrams[0].avg_duration_seconds == pytest.approx(7.0)

    def test_filters_by_min_frequency(self) -> None:
        """Pairs below min_frequency are excluded."""
        pairs = [
            SequencePair(action_a="recall", action_b="remember", count=1, avg_gap_seconds=3.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3)

        assert candidates == []

    def test_skips_cycles(self) -> None:
        """A->B + B->A should not produce trigram A->B->A."""
        pairs = [
            SequencePair(action_a="recall", action_b="edit", count=5, avg_gap_seconds=2.0),
            SequencePair(action_a="edit", action_b="recall", count=5, avg_gap_seconds=2.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3)

        trigrams = [c for c in candidates if len(c.steps) == 3]
        assert len(trigrams) == 0

    def test_empty_pairs_returns_empty(self) -> None:
        """Empty pairs list returns empty candidates."""
        candidates = extract_habit_candidates([], min_frequency=1)
        assert candidates == []

    def test_confidence_calculation(self) -> None:
        """Confidence is frequency / total_sessions."""
        pairs = [
            SequencePair(action_a="A", action_b="B", count=6, avg_gap_seconds=1.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=1, total_sessions=12)

        assert len(candidates) >= 1
        assert candidates[0].confidence == pytest.approx(0.5)


# ── heuristic_habit_name ─────────────────────────────────────────


class TestHeuristicHabitName:
    """Tests for heuristic_habit_name pure function."""

    def test_joins_steps_with_hyphen(self) -> None:
        """Multiple steps are joined with hyphens."""
        name = heuristic_habit_name(("recall", "edit", "test"))
        assert name == "recall-edit-test"

    def test_single_step(self) -> None:
        """Single step returns the step itself."""
        name = heuristic_habit_name(("recall",))
        assert name == "recall"

    def test_two_steps(self) -> None:
        """Two steps produce hyphen-joined name."""
        name = heuristic_habit_name(("remember", "context"))
        assert name == "remember-context"


# ── learn_habits (integration) ───────────────────────────────────


class TestLearnHabits:
    """Integration tests for learn_habits async pipeline."""

    async def test_full_pipeline(self, store: InMemoryStorage) -> None:
        """Record actions, mine habits, verify neurons + synapses + fiber."""
        # Record repeated action sequences across multiple sessions
        # Pattern: recall -> remember (repeated 3 times in 3 sessions)
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        learned, report = await learn_habits(store, config, now)

        assert report.sequences_analyzed == 6
        assert len(learned) >= 1

        habit = learned[0]
        assert habit.steps == ("recall", "remember")
        assert habit.name == "recall-remember"
        assert habit.frequency >= 2

        # Verify workflow fiber was created
        assert habit.workflow_fiber is not None
        assert habit.workflow_fiber.summary == "recall-remember"
        assert habit.workflow_fiber.metadata.get("_habit_pattern") is True
        assert habit.workflow_fiber.metadata.get("_workflow_actions") == ["recall", "remember"]

        # Verify BEFORE synapses were created
        assert len(habit.sequence_synapses) >= 1
        for syn in habit.sequence_synapses:
            assert syn.type == SynapseType.BEFORE

    async def test_insufficient_events_returns_empty(self, store: InMemoryStorage) -> None:
        """Fewer than 2 events produces no habits."""
        await store.record_action("recall", session_id="s0")

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        learned, report = await learn_habits(store, config, now)

        assert learned == []
        assert report.sequences_analyzed == 1
        assert report.habits_learned == 0

    async def test_action_neurons_created(self, store: InMemoryStorage) -> None:
        """Learned habits create ACTION neurons for each step."""
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("edit", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        await learn_habits(store, config, now)

        # Verify ACTION neurons exist
        recall_neurons = await store.find_neurons(content_exact="recall", type=NeuronType.ACTION)
        edit_neurons = await store.find_neurons(content_exact="edit", type=NeuronType.ACTION)
        assert len(recall_neurons) >= 1
        assert len(edit_neurons) >= 1

    async def test_workflow_fiber_metadata(self, store: InMemoryStorage) -> None:
        """Workflow fiber carries correct habit metadata."""
        for session_num in range(4):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)
            await store.record_action("context", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        learned, report = await learn_habits(store, config, now)

        assert report.habits_learned >= 1

        # Find the workflow fiber with the most steps (trigram if available)
        names = {h.name for h in learned}

        # At minimum, bigrams should be present
        assert any("recall" in n for n in names)

        # Check all learned habits have correct metadata
        for habit in learned:
            fiber = habit.workflow_fiber
            assert fiber.metadata.get("_habit_pattern") is True
            assert fiber.metadata.get("_habit_frequency") >= 2
            assert "_habit_confidence" in fiber.metadata
            assert fiber.metadata["_workflow_actions"] == list(habit.steps)

    async def test_below_frequency_threshold_no_habits(self, store: InMemoryStorage) -> None:
        """Events below habit_min_frequency produce no habits."""
        # Only one session -- frequency=1 which is below min_frequency=2
        await store.record_action("recall", session_id="s0")
        await store.record_action("remember", session_id="s0")

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        learned, report = await learn_habits(store, config, now)

        assert learned == []
        assert report.habits_learned == 0

    async def test_report_fields(self, store: InMemoryStorage) -> None:
        """HabitReport fields are populated correctly."""
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = datetime.utcnow()

        _learned, report = await learn_habits(store, config, now)

        assert report.sequences_analyzed == 6
        assert report.habits_learned >= 1
        assert report.pairs_strengthened >= 1
        assert isinstance(report.action_events_pruned, int)
