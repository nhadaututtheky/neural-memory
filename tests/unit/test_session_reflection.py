"""Tests for session-end reflection engine.

Covers:
- SessionReflection dataclass (empty, fields)
- reflect_on_session() core logic
- Minimum memory threshold
- Pattern detection integration (recurring, temporal, contradiction)
- Pattern neuron mapping (type + priority)
- Session stats and summary generation
- MAX_REFLECTION_NEURONS cap
"""

from __future__ import annotations

import pytest

from neural_memory.engine.session_reflection import (
    MAX_REFLECTION_NEURONS,
    SessionReflection,
    reflect_on_session,
)

# ── SessionReflection dataclass ──────────────────────────────────────


class TestSessionReflection:
    def test_empty(self) -> None:
        r = SessionReflection.empty()
        assert r.summary == ""
        assert r.pattern_neurons == []
        assert r.session_stats == {}
        assert r.patterns_found == 0

    def test_frozen(self) -> None:
        r = SessionReflection.empty()
        with pytest.raises(AttributeError):
            r.summary = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        r = SessionReflection(
            summary="test",
            pattern_neurons=[{"type": "insight", "content": "x", "priority": 5}],
            session_stats={"memories": 3},
            patterns_found=1,
        )
        assert r.summary == "test"
        assert len(r.pattern_neurons) == 1
        assert r.patterns_found == 1


# ── reflect_on_session() ────────────────────────────────────────────


class TestReflectOnSession:
    def test_below_minimum_returns_empty(self) -> None:
        """Fewer than MIN_SESSION_MEMORIES returns empty reflection."""
        memories = [{"content": "fact one", "type": "fact"}]
        result = reflect_on_session(memories)
        assert result == SessionReflection.empty()

    def test_exactly_minimum_with_no_patterns(self) -> None:
        """At threshold but no detectable patterns still produces stats."""
        memories = [
            {"content": "alpha info", "type": "fact"},
            {"content": "beta info", "type": "fact"},
            {"content": "gamma info", "type": "fact"},
        ]
        result = reflect_on_session(memories, query_count=5)
        assert result.patterns_found == 0
        assert result.pattern_neurons == []
        assert result.session_stats["memories"] == 3
        assert result.session_stats["queries"] == 5
        assert "3 memories" in result.summary

    def test_empty_content_filtered(self) -> None:
        """Memories with empty/missing content are filtered out."""
        memories = [
            {"content": "valid one", "type": "fact"},
            {"content": "", "type": "fact"},
            {"content": "valid two", "type": "fact"},
            {"type": "fact"},  # no content key
        ]
        # Only 2 valid → below threshold
        result = reflect_on_session(memories)
        assert result == SessionReflection.empty()

    def test_recurring_entity_pattern(self) -> None:
        """Recurring entity (3+ mentions) generates insight neuron."""
        memories = [
            {"content": "Used FastAPI for the endpoint", "type": "fact"},
            {"content": "FastAPI handles validation well", "type": "insight"},
            {"content": "FastAPI integrates with Pydantic", "type": "fact"},
        ]
        result = reflect_on_session(memories)
        assert result.patterns_found >= 1
        # Should have at least one insight neuron
        insights = [n for n in result.pattern_neurons if n["type"] == "insight"]
        assert len(insights) >= 1
        assert insights[0]["priority"] == 6
        assert "[Session reflection]" in insights[0]["content"]

    def test_temporal_sequence_pattern(self) -> None:
        """Temporal markers (3+ memories) generate workflow neuron."""
        memories = [
            {"content": "First we set up the database", "type": "workflow"},
            {"content": "Then we configured the API", "type": "workflow"},
            {"content": "After that we added auth", "type": "workflow"},
        ]
        result = reflect_on_session(memories)
        workflows = [n for n in result.pattern_neurons if n["type"] == "workflow"]
        assert len(workflows) >= 1
        assert workflows[0]["priority"] == 6

    def test_contradiction_pattern(self) -> None:
        """Contradicting memories generate decision neuron."""
        memories = [
            {"content": "Should use Redis for caching", "type": "decision"},
            {"content": "Should not use Redis due to cost", "type": "decision"},
            {"content": "Memcached is the best alternative", "type": "fact"},
        ]
        result = reflect_on_session(memories)
        decisions = [n for n in result.pattern_neurons if n["type"] == "decision"]
        assert len(decisions) >= 1
        assert decisions[0]["priority"] == 7

    def test_max_reflection_neurons_cap(self) -> None:
        """Pattern neurons are capped at MAX_REFLECTION_NEURONS."""
        # Create enough memories to trigger many patterns
        # 4 different recurring entities (each appearing 3+ times)
        memories = []
        for entity in ["FastAPI", "Django", "Flask", "Express"]:
            for i in range(3):
                memories.append({"content": f"{entity} is great tool {i}", "type": "fact"})

        result = reflect_on_session(memories)
        assert len(result.pattern_neurons) <= MAX_REFLECTION_NEURONS

    def test_session_topics_in_summary(self) -> None:
        """Session topics appear in summary string."""
        memories = [
            {"content": "alpha content here", "type": "fact"},
            {"content": "beta content here", "type": "fact"},
            {"content": "gamma content here", "type": "fact"},
        ]
        result = reflect_on_session(
            memories,
            session_topics=["auth", "database", "testing"],
            query_count=10,
        )
        assert "auth" in result.summary
        assert "database" in result.summary
        assert "10 queries" in result.summary

    def test_no_topics_shows_various(self) -> None:
        """No session topics defaults to 'various'."""
        memories = [
            {"content": "fact one info", "type": "fact"},
            {"content": "fact two info", "type": "fact"},
            {"content": "fact three info", "type": "fact"},
        ]
        result = reflect_on_session(memories)
        assert "various" in result.summary

    def test_stats_structure(self) -> None:
        """session_stats has expected keys."""
        memories = [
            {"content": "alpha data", "type": "fact"},
            {"content": "beta data", "type": "fact"},
            {"content": "gamma data", "type": "fact"},
        ]
        result = reflect_on_session(memories, session_topics=["auth"], query_count=7)
        stats = result.session_stats
        assert stats["memories"] == 3
        assert stats["queries"] == 7
        assert stats["topics"] == ["auth"]
        assert "patterns" in stats

    def test_tags_preserved_in_memory_dicts(self) -> None:
        """Tags from input memories are passed through to detect_patterns."""
        memories = [
            {"content": "fact about FastAPI", "type": "fact", "tags": ["api"]},
            {"content": "FastAPI is fast", "type": "insight", "tags": ["perf"]},
            {"content": "FastAPI uses Starlette", "type": "fact", "tags": ["api"]},
        ]
        # Should not crash; tags are preserved in the dicts passed to detect_patterns
        result = reflect_on_session(memories)
        assert result.session_stats["memories"] == 3
