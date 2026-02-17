"""Tests for Recall Pattern Learning (Feature C)."""

from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio

from neural_memory.core.action_event import ActionEvent
from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.query_pattern_mining import (
    QueryPatternCandidate,
    QueryPatternReport,
    TopicPair,
    extract_pattern_candidates,
    extract_topics,
    learn_query_patterns,
    mine_query_topic_pairs,
    suggest_follow_up_queries,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """Storage with brain context."""
    storage = InMemoryStorage()
    brain = Brain.create(name="qp-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


def _make_recall_event(
    query: str,
    session_id: str = "s1",
    minutes_offset: int = 0,
) -> ActionEvent:
    """Create a recall action event."""
    base_time = utcnow()
    return ActionEvent(
        session_id=session_id,
        action_type="recall",
        action_context=query,
        created_at=base_time + timedelta(minutes=minutes_offset),
    )


# ── Extract Topics Tests ─────────────────────────────────────────


class TestExtractTopics:
    """Tests for extract_topics."""

    def test_basic_extraction(self) -> None:
        topics = extract_topics("authentication middleware setup")
        assert len(topics) > 0
        # All topics should be lowercase
        for t in topics:
            assert t == t.lower()

    def test_empty_string(self) -> None:
        assert extract_topics("") == []

    def test_short_string(self) -> None:
        assert extract_topics("ab") == []

    def test_deduplication(self) -> None:
        topics = extract_topics("auth auth auth setup")
        # "auth" should appear only once
        auth_count = sum(1 for t in topics if t == "auth")
        assert auth_count <= 1

    def test_cap_at_10(self) -> None:
        long_text = " ".join([f"keyword{i}" for i in range(20)])
        topics = extract_topics(long_text)
        assert len(topics) <= 10


# ── Mine Topic Pairs Tests ───────────────────────────────────────


class TestMineTopicPairs:
    """Tests for mine_query_topic_pairs."""

    def test_empty_events(self) -> None:
        assert mine_query_topic_pairs([]) == []

    def test_non_recall_events_ignored(self) -> None:
        events = [
            ActionEvent(
                session_id="s1",
                action_type="remember",
                action_context="database schema",
                created_at=utcnow(),
            ),
        ]
        assert mine_query_topic_pairs(events) == []

    def test_single_event_no_pairs(self) -> None:
        events = [_make_recall_event("authentication setup")]
        assert mine_query_topic_pairs(events) == []

    def test_pair_extraction(self) -> None:
        events = [
            _make_recall_event("authentication jwt tokens", minutes_offset=0),
            _make_recall_event("middleware express routing", minutes_offset=2),
        ]
        pairs = mine_query_topic_pairs(events)
        assert len(pairs) > 0
        # All pairs should have count >= 1
        for pair in pairs:
            assert pair.count >= 1

    def test_window_filtering(self) -> None:
        events = [
            _make_recall_event("auth setup", minutes_offset=0),
            _make_recall_event("database migration", minutes_offset=20),  # 20 min gap
        ]
        # Default window is 600s (10 min), so 20 min gap should be excluded
        pairs = mine_query_topic_pairs(events, window_seconds=600.0)
        assert len(pairs) == 0

    def test_sorted_by_count(self) -> None:
        # Create same pair multiple times across sessions
        events = []
        for session_idx in range(3):
            events.append(
                _make_recall_event(
                    "react components",
                    session_id=f"s{session_idx}",
                    minutes_offset=0,
                )
            )
            events.append(
                _make_recall_event(
                    "state management",
                    session_id=f"s{session_idx}",
                    minutes_offset=2,
                )
            )
        pairs = mine_query_topic_pairs(events)
        if len(pairs) > 1:
            # Should be sorted by count desc
            assert pairs[0].count >= pairs[-1].count


# ── Pattern Candidate Tests ──────────────────────────────────────


class TestExtractPatternCandidates:
    """Tests for extract_pattern_candidates."""

    def test_empty_pairs(self) -> None:
        assert extract_pattern_candidates([]) == []

    def test_below_threshold(self) -> None:
        pairs = [TopicPair(topic_a="auth", topic_b="jwt", count=1, avg_gap_seconds=30)]
        candidates = extract_pattern_candidates(pairs, min_frequency=3)
        assert len(candidates) == 0

    def test_above_threshold(self) -> None:
        pairs = [TopicPair(topic_a="auth", topic_b="jwt", count=5, avg_gap_seconds=30)]
        candidates = extract_pattern_candidates(pairs, min_frequency=3, total_sessions=10)
        assert len(candidates) == 1
        assert candidates[0].topics == ("auth", "jwt")
        assert candidates[0].frequency == 5
        assert candidates[0].confidence == 0.5  # 5/10

    def test_capped_at_20(self) -> None:
        pairs = [
            TopicPair(topic_a=f"topic{i}", topic_b=f"topic{i+1}", count=5, avg_gap_seconds=30)
            for i in range(30)
        ]
        candidates = extract_pattern_candidates(pairs, min_frequency=1)
        assert len(candidates) <= 20


# ── Data Structure Tests ─────────────────────────────────────────


class TestDataStructures:
    """Tests for frozen dataclasses."""

    def test_topic_pair_frozen(self) -> None:
        pair = TopicPair(topic_a="a", topic_b="b", count=1, avg_gap_seconds=0)
        with pytest.raises(AttributeError):
            pair.count = 5  # type: ignore[misc]

    def test_candidate_frozen(self) -> None:
        c = QueryPatternCandidate(topics=("a", "b"), frequency=3, confidence=0.5)
        with pytest.raises(AttributeError):
            c.frequency = 10  # type: ignore[misc]

    def test_report_defaults(self) -> None:
        r = QueryPatternReport()
        assert r.topics_extracted == 0
        assert r.pairs_found == 0
        assert r.patterns_learned == 0


# ── Integration: learn_query_patterns ────────────────────────────


class TestLearnQueryPatterns:
    """Integration tests for learn_query_patterns."""

    async def test_too_few_events(self, store: InMemoryStorage) -> None:
        brain_id = store._current_brain_id
        assert brain_id is not None
        brain = await store.get_brain(brain_id)
        assert brain is not None

        report = await learn_query_patterns(store, brain.config, utcnow())
        assert report.topics_extracted == 0
        assert report.patterns_learned == 0

    async def test_learns_patterns_from_events(self, store: InMemoryStorage) -> None:
        brain_id = store._current_brain_id
        assert brain_id is not None
        brain = await store.get_brain(brain_id)
        assert brain is not None

        # Create recall events with repeated topic pairs
        now = utcnow()
        for session_idx in range(5):
            sid = f"session-{session_idx}"
            await store.record_action(
                action_type="recall",
                action_context="authentication jwt tokens security",
                session_id=sid,
            )
            await store.record_action(
                action_type="recall",
                action_context="middleware express routing handlers",
                session_id=sid,
            )

        report = await learn_query_patterns(store, brain.config, now)
        assert report.topics_extracted > 0


# ── Integration: suggest_follow_up_queries ───────────────────────


class TestSuggestFollowUpQueries:
    """Tests for suggest_follow_up_queries."""

    async def test_no_topics(self, store: InMemoryStorage) -> None:
        brain_id = store._current_brain_id
        assert brain_id is not None
        brain = await store.get_brain(brain_id)
        assert brain is not None

        result = await suggest_follow_up_queries(store, [], brain.config)
        assert result == []

    async def test_no_patterns_returns_empty(self, store: InMemoryStorage) -> None:
        brain_id = store._current_brain_id
        assert brain_id is not None
        brain = await store.get_brain(brain_id)
        assert brain is not None

        result = await suggest_follow_up_queries(store, ["auth", "jwt"], brain.config)
        assert result == []
