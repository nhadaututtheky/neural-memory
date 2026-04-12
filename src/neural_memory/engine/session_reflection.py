"""Session-end reflection — synthesize learnings from a session.

When a session ends (via nmem_auto process/flush), this module analyzes
memories saved during the session and generates higher-order insights:
- Recurring themes (entities that appeared across multiple memories)
- Workflow patterns (temporal sequences)
- Contradictions (opposing statements)

All processing is rule-based (zero LLM calls), using the existing
reflection.py pattern detection engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Minimum session memories needed to trigger reflection
MIN_SESSION_MEMORIES = 3

# Maximum reflection neurons to create per session
MAX_REFLECTION_NEURONS = 3


@dataclass(frozen=True)
class SessionReflection:
    """Result of session-end reflection analysis."""

    summary: str
    pattern_neurons: list[dict[str, Any]]  # [{type, content, priority}]
    session_stats: dict[str, Any]  # {queries, topics, memories}
    patterns_found: int = 0

    @staticmethod
    def empty() -> SessionReflection:
        return SessionReflection(
            summary="",
            pattern_neurons=[],
            session_stats={},
            patterns_found=0,
        )


def reflect_on_session(
    memories: list[dict[str, Any]],
    session_topics: list[str] | None = None,
    query_count: int = 0,
) -> SessionReflection:
    """Analyze session memories and generate reflection insights.

    Takes memory dicts (already saved during the session) and runs
    pattern detection to produce higher-order insights.

    Args:
        memories: List of dicts with "content", "type" keys.
            Typically the accumulated auto-captured memories from the session.
        session_topics: Top session topics from SessionState (optional).
        query_count: Number of queries in this session.

    Returns:
        SessionReflection with summary, pattern neurons, and stats.
    """
    if len(memories) < MIN_SESSION_MEMORIES:
        return SessionReflection.empty()

    # Ensure dicts have required keys for detect_patterns
    memory_dicts: list[dict[str, Any]] = [
        {
            "content": m.get("content", ""),
            "type": m.get("type", "fact"),
            "tags": m.get("tags", []),
        }
        for m in memories
        if m.get("content")
    ]

    if len(memory_dicts) < MIN_SESSION_MEMORIES:
        return SessionReflection.empty()

    # Run pattern detection
    from neural_memory.engine.reflection import detect_patterns

    patterns = detect_patterns(memory_dicts)

    # Build pattern neurons (capped)
    pattern_neurons: list[dict[str, Any]] = []
    for pattern in patterns[:MAX_REFLECTION_NEURONS]:
        ptype = pattern["pattern_type"]
        desc = pattern["description"]

        if ptype == "recurring_entity":
            neuron_type = "insight"
            priority = 6
        elif ptype == "temporal_sequence":
            neuron_type = "workflow"
            priority = 6
        elif ptype == "contradiction":
            neuron_type = "decision"
            priority = 7
        else:
            neuron_type = "insight"
            priority = 5

        pattern_neurons.append(
            {
                "type": neuron_type,
                "content": f"[Session reflection] {desc}",
                "priority": priority,
            }
        )

    # Build summary
    topics_str = ", ".join(session_topics[:5]) if session_topics else "various"
    summary = (
        f"Session summary: {len(memory_dicts)} memories across topics: {topics_str}. "
        f"{query_count} queries. {len(patterns)} patterns detected."
    )

    stats: dict[str, Any] = {
        "memories": len(memory_dicts),
        "queries": query_count,
        "topics": session_topics[:5] if session_topics else [],
        "patterns": len(patterns),
    }

    return SessionReflection(
        summary=summary,
        pattern_neurons=pattern_neurons,
        session_stats=stats,
        patterns_found=len(patterns),
    )
