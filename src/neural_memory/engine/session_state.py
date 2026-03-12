"""Session intelligence — in-memory session state for retrieval pipeline.

Tracks topics, entities, and query history across MCP calls within a session.
Enables adaptive depth selection and predictive priming in later phases.

Design:
- In-memory only (not SQLite) — lives in MCP server process lifetime
- Lightweight: <1ms overhead per recall
- SessionManager is a singleton with LRU eviction and auto-expiry
"""

from __future__ import annotations

import logging
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

from neural_memory.extraction.keywords import extract_keywords

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_EMA_ALPHA = 0.3  # Recent queries weighted 3x over history
MAX_RECENT_QUERIES = 20
SESSION_EXPIRY_SECONDS = 2 * 60 * 60  # 2 hours
MAX_CONCURRENT_SESSIONS = 10
SUMMARY_PERSIST_INTERVAL = 10  # Persist every N queries
MAX_TOPICS_RETURNED = 5


# ── Data Models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryRecord:
    """Record of a single query within a session.

    Attributes:
        query: The raw query text.
        depth_used: Depth level used for retrieval (0-3).
        confidence: Confidence of the retrieval result.
        fibers_matched: Number of fibers matched.
        entities: Entities extracted from the query.
        keywords: Keywords extracted from the query.
        timestamp: Unix timestamp of the query.
    """

    query: str
    depth_used: int
    confidence: float
    fibers_matched: int
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class SessionState:
    """In-memory session state for retrieval pipeline.

    Mutable by design — updated on every recall query. Not persisted
    directly; session summaries are periodically saved to SQLite.

    Attributes:
        session_id: Unique session identifier.
        topic_ema: EMA-weighted topic scores (higher = more recent/frequent).
        recent_queries: Circular buffer of recent QueryRecords.
        recent_entities: Frequency counter of entities seen in session.
        query_count: Total queries in this session.
        started_at: Unix timestamp when session was created.
        last_active: Unix timestamp of most recent activity.
        _last_persist_count: Query count at last summary persist.
    """

    session_id: str
    topic_ema: dict[str, float] = field(default_factory=dict)
    recent_queries: list[QueryRecord] = field(default_factory=list)
    recent_entities: Counter[str] = field(default_factory=Counter)
    query_count: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    _last_persist_count: int = 0

    # Priming metrics (populated by retrieval pipeline)
    priming_total: int = 0
    priming_hits: int = 0
    priming_misses: int = 0

    @property
    def priming_hit_rate(self) -> float:
        """Fraction of primed neurons that appeared in final results."""
        if self.priming_total == 0:
            return 0.0
        return self.priming_hits / self.priming_total

    def record_query(
        self,
        query: str,
        depth_used: int,
        confidence: float,
        fibers_matched: int,
        entities: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        """Record a completed query and update topic EMA.

        Args:
            query: Raw query text.
            depth_used: Depth level used.
            confidence: Result confidence (0.0-1.0).
            fibers_matched: Number of fibers matched.
            entities: Pre-extracted entities (extracted if None).
            keywords: Pre-extracted keywords (extracted if None).
        """
        now = time.monotonic()

        # Extract topics if not provided
        if keywords is None:
            keywords = extract_keywords(query)
        if entities is None:
            entities = []

        record = QueryRecord(
            query=query,
            depth_used=depth_used,
            confidence=confidence,
            fibers_matched=fibers_matched,
            entities=list(entities),
            keywords=list(keywords),
            timestamp=now,
        )

        # Append to recent queries (bounded)
        self.recent_queries.append(record)
        if len(self.recent_queries) > MAX_RECENT_QUERIES:
            self.recent_queries = self.recent_queries[-MAX_RECENT_QUERIES:]

        # Update entity frequency
        for entity in entities:
            self.recent_entities[entity] += 1

        # Update topic EMA
        self._update_topic_ema(keywords + entities)

        self.query_count += 1
        self.last_active = now

    def _update_topic_ema(self, topics: list[str]) -> None:
        """Update topic EMA scores with new topics.

        Topics present in the current query get boosted. All existing
        topics decay slightly (multiplied by 1-alpha).
        """
        alpha = DEFAULT_EMA_ALPHA

        # Decay all existing topics
        decayed: dict[str, float] = {}
        for topic, score in self.topic_ema.items():
            new_score = score * (1 - alpha)
            if new_score >= 0.01:  # Prune near-zero topics
                decayed[topic] = new_score

        # Boost current topics
        for topic in topics:
            if topic:
                normalized = topic.lower().strip()
                if normalized:
                    decayed[normalized] = decayed.get(normalized, 0.0) + alpha

        self.topic_ema = decayed

    def get_top_topics(self, limit: int = MAX_TOPICS_RETURNED) -> list[str]:
        """Return top topics sorted by EMA weight."""
        if not self.topic_ema:
            return []
        sorted_topics = sorted(self.topic_ema.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _score in sorted_topics[:limit]]

    def get_topic_weights(self, limit: int = MAX_TOPICS_RETURNED) -> dict[str, float]:
        """Return top topics with their EMA weights."""
        if not self.topic_ema:
            return {}
        sorted_topics = sorted(self.topic_ema.items(), key=lambda x: x[1], reverse=True)
        return {topic: round(score, 4) for topic, score in sorted_topics[:limit]}

    def needs_persist(self) -> bool:
        """Check if session summary should be persisted."""
        return (self.query_count - self._last_persist_count) >= SUMMARY_PERSIST_INTERVAL

    def mark_persisted(self) -> None:
        """Mark that a summary was just persisted."""
        self._last_persist_count = self.query_count

    def is_expired(self, now: float | None = None) -> bool:
        """Check if session has expired due to inactivity."""
        if now is None:
            now = time.monotonic()
        return (now - self.last_active) > SESSION_EXPIRY_SECONDS

    def to_summary_dict(self) -> dict[str, Any]:
        """Export session state as a summary dict for persistence."""
        return {
            "session_id": self.session_id,
            "topics": self.get_top_topics(10),
            "topic_weights": self.get_topic_weights(10),
            "top_entities": self.recent_entities.most_common(10),
            "query_count": self.query_count,
            "avg_confidence": (
                sum(q.confidence for q in self.recent_queries) / len(self.recent_queries)
                if self.recent_queries
                else 0.0
            ),
            "avg_depth": (
                sum(q.depth_used for q in self.recent_queries) / len(self.recent_queries)
                if self.recent_queries
                else 0.0
            ),
            "priming_hit_rate": round(self.priming_hit_rate, 4),
            "priming_total": self.priming_total,
        }


# ── Session Manager ────────────────────────────────────────────────────


class SessionManager:
    """Singleton manager for in-memory session states.

    Features:
    - LRU eviction when max concurrent sessions reached
    - Auto-expiry of inactive sessions (2h default)
    - Thread-safe via simple dict operations (GIL-protected)
    """

    _instance: SessionManager | None = None

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    @classmethod
    def get_instance(cls) -> SessionManager:
        """Get or create the singleton SessionManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def get_or_create(self, session_id: str) -> SessionState:
        """Get existing session or create a new one.

        Moves accessed session to end of OrderedDict (LRU update).
        Auto-expires stale sessions and evicts oldest if at capacity.
        """
        now = time.monotonic()

        # Expire stale sessions
        self._expire_stale(now)

        if session_id in self._sessions:
            # Move to end (most recently used)
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        # Evict oldest if at capacity
        while len(self._sessions) >= MAX_CONCURRENT_SESSIONS:
            evicted_id, evicted = self._sessions.popitem(last=False)
            logger.debug("Session evicted (LRU): %s (%d queries)", evicted_id, evicted.query_count)

        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        """Get session without creating or updating LRU order."""
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> SessionState | None:
        """Remove and return a session."""
        return self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def all_sessions(self) -> list[SessionState]:
        """Return all active sessions (for iteration/persist)."""
        return list(self._sessions.values())

    def _expire_stale(self, now: float) -> None:
        """Remove sessions that have been inactive too long."""
        expired_ids = [sid for sid, state in self._sessions.items() if state.is_expired(now)]
        for sid in expired_ids:
            removed = self._sessions.pop(sid)
            logger.debug(
                "Session expired: %s (%d queries, %d topics)",
                sid,
                removed.query_count,
                len(removed.topic_ema),
            )
