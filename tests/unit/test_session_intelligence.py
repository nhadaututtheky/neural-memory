"""Tests for session intelligence (Phase 1 of v4.0 Brain Intelligence).

Covers:
- SessionState creation and field defaults
- record_query() updates topic EMA correctly
- Topic EMA decay over multiple queries
- get_top_topics() and get_topic_weights() ordering
- SessionManager get_or_create(), LRU eviction, auto-expiry
- needs_persist() trigger at correct intervals
- to_summary_dict() export format
- is_expired() with time thresholds
- SQLiteSessionsMixin CRUD operations
"""

from __future__ import annotations

import time
from collections import Counter

import pytest

from neural_memory.engine.session_state import (
    DEFAULT_EMA_ALPHA,
    MAX_CONCURRENT_SESSIONS,
    MAX_RECENT_QUERIES,
    SESSION_EXPIRY_SECONDS,
    SUMMARY_PERSIST_INTERVAL,
    QueryRecord,
    SessionManager,
    SessionState,
)

# ── QueryRecord ───────────────────────────────────────────────────────


class TestQueryRecord:
    def test_creation(self) -> None:
        qr = QueryRecord(
            query="test query",
            depth_used=2,
            confidence=0.85,
            fibers_matched=5,
            entities=["python"],
            keywords=["test", "query"],
        )
        assert qr.query == "test query"
        assert qr.depth_used == 2
        assert qr.confidence == 0.85
        assert qr.fibers_matched == 5
        assert qr.entities == ["python"]
        assert qr.keywords == ["test", "query"]

    def test_frozen(self) -> None:
        qr = QueryRecord(query="x", depth_used=1, confidence=0.5, fibers_matched=0)
        with pytest.raises(AttributeError):
            qr.query = "y"  # type: ignore[misc]

    def test_defaults(self) -> None:
        qr = QueryRecord(query="x", depth_used=0, confidence=0.0, fibers_matched=0)
        assert qr.entities == []
        assert qr.keywords == []
        assert isinstance(qr.timestamp, float)


# ── SessionState ──────────────────────────────────────────────────────


class TestSessionState:
    def test_creation_defaults(self) -> None:
        s = SessionState(session_id="s1")
        assert s.session_id == "s1"
        assert s.topic_ema == {}
        assert s.recent_queries == []
        assert isinstance(s.recent_entities, Counter)
        assert s.query_count == 0
        assert s._last_persist_count == 0

    def test_record_query_increments_count(self) -> None:
        s = SessionState(session_id="s1")
        s.record_query(
            "auth bug",
            depth_used=1,
            confidence=0.7,
            fibers_matched=3,
            keywords=["auth", "bug"],
            entities=[],
        )
        assert s.query_count == 1
        s.record_query(
            "login issue",
            depth_used=2,
            confidence=0.8,
            fibers_matched=5,
            keywords=["login"],
            entities=[],
        )
        assert s.query_count == 2

    def test_record_query_updates_topic_ema(self) -> None:
        s = SessionState(session_id="s1")
        s.record_query(
            "database migration",
            depth_used=1,
            confidence=0.9,
            fibers_matched=2,
            keywords=["database", "migration"],
            entities=[],
        )
        assert "database" in s.topic_ema
        assert "migration" in s.topic_ema
        assert s.topic_ema["database"] == pytest.approx(DEFAULT_EMA_ALPHA)
        assert s.topic_ema["migration"] == pytest.approx(DEFAULT_EMA_ALPHA)

    def test_topic_ema_decay(self) -> None:
        """Old topics decay when new queries come in with different topics."""
        s = SessionState(session_id="s1")
        s.record_query(
            "auth", depth_used=1, confidence=0.5, fibers_matched=1, keywords=["auth"], entities=[]
        )
        initial_auth = s.topic_ema["auth"]

        # Query with different topic → auth should decay
        s.record_query(
            "database",
            depth_used=1,
            confidence=0.5,
            fibers_matched=1,
            keywords=["database"],
            entities=[],
        )
        assert s.topic_ema["auth"] < initial_auth
        assert s.topic_ema["auth"] == pytest.approx(initial_auth * (1 - DEFAULT_EMA_ALPHA))

    def test_topic_ema_boost_on_repeat(self) -> None:
        """Repeating the same topic boosts its EMA score."""
        s = SessionState(session_id="s1")
        s.record_query(
            "auth", depth_used=1, confidence=0.5, fibers_matched=1, keywords=["auth"], entities=[]
        )
        score_after_1 = s.topic_ema["auth"]

        s.record_query(
            "auth again",
            depth_used=1,
            confidence=0.5,
            fibers_matched=1,
            keywords=["auth"],
            entities=[],
        )
        score_after_2 = s.topic_ema["auth"]
        # decayed + boosted should be higher than just decayed
        assert score_after_2 > score_after_1 * (1 - DEFAULT_EMA_ALPHA)

    def test_topic_ema_prunes_near_zero(self) -> None:
        """Topics below 0.01 are pruned."""
        s = SessionState(session_id="s1")
        s.topic_ema = {"old_topic": 0.01}
        # Decay: 0.01 * 0.7 = 0.007 < 0.01 → pruned
        s.record_query(
            "new topic",
            depth_used=1,
            confidence=0.5,
            fibers_matched=1,
            keywords=["new"],
            entities=[],
        )
        assert "old_topic" not in s.topic_ema

    def test_topic_normalization(self) -> None:
        """Topics are lowercased and stripped."""
        s = SessionState(session_id="s1")
        s.record_query(
            "test",
            depth_used=1,
            confidence=0.5,
            fibers_matched=0,
            keywords=["  Auth  ", "DATABASE"],
            entities=[],
        )
        assert "auth" in s.topic_ema
        assert "database" in s.topic_ema
        assert "  Auth  " not in s.topic_ema

    def test_entities_tracked(self) -> None:
        s = SessionState(session_id="s1")
        s.record_query(
            "q1",
            depth_used=1,
            confidence=0.5,
            fibers_matched=1,
            keywords=[],
            entities=["Python", "FastAPI"],
        )
        s.record_query(
            "q2", depth_used=1, confidence=0.5, fibers_matched=1, keywords=[], entities=["Python"]
        )
        assert s.recent_entities["Python"] == 2
        assert s.recent_entities["FastAPI"] == 1

    def test_entities_added_to_topic_ema(self) -> None:
        """Entities are also tracked in topic EMA."""
        s = SessionState(session_id="s1")
        s.record_query(
            "q1",
            depth_used=1,
            confidence=0.5,
            fibers_matched=1,
            keywords=["auth"],
            entities=["PostgreSQL"],
        )
        assert "postgresql" in s.topic_ema  # lowercased
        assert "auth" in s.topic_ema

    def test_recent_queries_bounded(self) -> None:
        """Recent queries list doesn't exceed MAX_RECENT_QUERIES."""
        s = SessionState(session_id="s1")
        for i in range(MAX_RECENT_QUERIES + 10):
            s.record_query(
                f"query {i}",
                depth_used=1,
                confidence=0.5,
                fibers_matched=0,
                keywords=[f"kw{i}"],
                entities=[],
            )
        assert len(s.recent_queries) == MAX_RECENT_QUERIES
        # Most recent is last
        assert s.recent_queries[-1].query == f"query {MAX_RECENT_QUERIES + 9}"

    def test_auto_extract_keywords(self) -> None:
        """When keywords=None, extract_keywords is called."""
        s = SessionState(session_id="s1")
        s.record_query(
            "database migration strategy",
            depth_used=1,
            confidence=0.5,
            fibers_matched=0,
            keywords=None,
            entities=[],
        )
        # Should have some topics from keyword extraction
        assert len(s.topic_ema) > 0

    def test_get_top_topics_ordered(self) -> None:
        s = SessionState(session_id="s1")
        s.topic_ema = {"auth": 0.9, "database": 0.5, "api": 0.1}
        topics = s.get_top_topics(limit=2)
        assert topics == ["auth", "database"]

    def test_get_top_topics_empty(self) -> None:
        s = SessionState(session_id="s1")
        assert s.get_top_topics() == []

    def test_get_topic_weights(self) -> None:
        s = SessionState(session_id="s1")
        s.topic_ema = {"auth": 0.91234, "db": 0.50001}
        weights = s.get_topic_weights(limit=5)
        assert weights == {"auth": 0.9123, "db": 0.5}

    def test_get_topic_weights_empty(self) -> None:
        s = SessionState(session_id="s1")
        assert s.get_topic_weights() == {}

    def test_needs_persist(self) -> None:
        s = SessionState(session_id="s1")
        assert not s.needs_persist()
        for i in range(SUMMARY_PERSIST_INTERVAL):
            s.record_query(
                f"q{i}",
                depth_used=1,
                confidence=0.5,
                fibers_matched=0,
                keywords=["kw"],
                entities=[],
            )
        assert s.needs_persist()

    def test_mark_persisted(self) -> None:
        s = SessionState(session_id="s1")
        for i in range(SUMMARY_PERSIST_INTERVAL):
            s.record_query(
                f"q{i}",
                depth_used=1,
                confidence=0.5,
                fibers_matched=0,
                keywords=["kw"],
                entities=[],
            )
        assert s.needs_persist()
        s.mark_persisted()
        assert not s.needs_persist()

    def test_is_expired_not_expired(self) -> None:
        s = SessionState(session_id="s1")
        assert not s.is_expired()

    def test_is_expired_after_timeout(self) -> None:
        s = SessionState(session_id="s1")
        future = time.monotonic() + SESSION_EXPIRY_SECONDS + 1
        assert s.is_expired(now=future)

    def test_to_summary_dict(self) -> None:
        s = SessionState(session_id="s1")
        s.record_query(
            "auth login",
            depth_used=2,
            confidence=0.85,
            fibers_matched=3,
            keywords=["auth", "login"],
            entities=["User"],
        )
        summary = s.to_summary_dict()
        assert summary["session_id"] == "s1"
        assert summary["query_count"] == 1
        assert isinstance(summary["topics"], list)
        assert isinstance(summary["topic_weights"], dict)
        assert isinstance(summary["top_entities"], list)
        assert summary["avg_confidence"] == pytest.approx(0.85)
        assert summary["avg_depth"] == pytest.approx(2.0)

    def test_to_summary_dict_empty(self) -> None:
        s = SessionState(session_id="s1")
        summary = s.to_summary_dict()
        assert summary["query_count"] == 0
        assert summary["avg_confidence"] == 0.0
        assert summary["avg_depth"] == 0.0
        assert summary["topics"] == []


# ── SessionManager ────────────────────────────────────────────────────


class TestSessionManager:
    def setup_method(self) -> None:
        SessionManager.reset()

    def test_singleton(self) -> None:
        mgr1 = SessionManager.get_instance()
        mgr2 = SessionManager.get_instance()
        assert mgr1 is mgr2

    def test_reset(self) -> None:
        mgr1 = SessionManager.get_instance()
        SessionManager.reset()
        mgr2 = SessionManager.get_instance()
        assert mgr1 is not mgr2

    def test_get_or_create_new(self) -> None:
        mgr = SessionManager.get_instance()
        session = mgr.get_or_create("s1")
        assert session.session_id == "s1"
        assert mgr.active_count() == 1

    def test_get_or_create_existing(self) -> None:
        mgr = SessionManager.get_instance()
        s1 = mgr.get_or_create("s1")
        s1.record_query(
            "test", depth_used=1, confidence=0.5, fibers_matched=0, keywords=["test"], entities=[]
        )
        s1_again = mgr.get_or_create("s1")
        assert s1_again is s1
        assert s1_again.query_count == 1

    def test_lru_eviction(self) -> None:
        """Oldest sessions evicted when at capacity."""
        mgr = SessionManager.get_instance()
        for i in range(MAX_CONCURRENT_SESSIONS):
            mgr.get_or_create(f"s{i}")
        assert mgr.active_count() == MAX_CONCURRENT_SESSIONS

        # Adding one more should evict s0
        mgr.get_or_create("new_session")
        assert mgr.active_count() == MAX_CONCURRENT_SESSIONS
        assert mgr.get("s0") is None
        assert mgr.get("new_session") is not None

    def test_lru_access_updates_order(self) -> None:
        """Accessing a session moves it to end (most recent)."""
        mgr = SessionManager.get_instance()
        for i in range(MAX_CONCURRENT_SESSIONS):
            mgr.get_or_create(f"s{i}")

        # Access s0 to make it most recently used
        mgr.get_or_create("s0")

        # Now adding new session should evict s1 (oldest), not s0
        mgr.get_or_create("new_session")
        assert mgr.get("s0") is not None  # s0 was refreshed
        assert mgr.get("s1") is None  # s1 was oldest → evicted

    def test_auto_expiry(self) -> None:
        """Expired sessions are removed on next get_or_create."""
        mgr = SessionManager.get_instance()
        session = mgr.get_or_create("s1")
        # Simulate expiry by backdating last_active
        session.last_active = time.monotonic() - SESSION_EXPIRY_SECONDS - 1
        mgr.get_or_create("s2")  # triggers _expire_stale
        assert mgr.get("s1") is None
        assert mgr.get("s2") is not None

    def test_get_without_create(self) -> None:
        mgr = SessionManager.get_instance()
        assert mgr.get("nonexistent") is None
        mgr.get_or_create("s1")
        assert mgr.get("s1") is not None

    def test_remove(self) -> None:
        mgr = SessionManager.get_instance()
        mgr.get_or_create("s1")
        removed = mgr.remove("s1")
        assert removed is not None
        assert removed.session_id == "s1"
        assert mgr.active_count() == 0

    def test_remove_nonexistent(self) -> None:
        mgr = SessionManager.get_instance()
        assert mgr.remove("nonexistent") is None

    def test_all_sessions(self) -> None:
        mgr = SessionManager.get_instance()
        mgr.get_or_create("s1")
        mgr.get_or_create("s2")
        sessions = mgr.all_sessions()
        assert len(sessions) == 2
        assert {s.session_id for s in sessions} == {"s1", "s2"}


# ── SQLite Session Summaries ──────────────────────────────────────────


@pytest.fixture
async def storage(tmp_path):
    """Create a fresh SQLiteStorage for testing."""
    from neural_memory.storage.sqlite_store import SQLiteStorage

    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()

    # Create a brain for FK constraints
    await store._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("test-brain", "test", "{}", "2026-03-11T00:00:00", "2026-03-11T00:00:00"),
    )
    await store._ensure_conn().commit()
    store.set_brain("test-brain")

    yield store
    await store.close()


class TestSQLiteSessionSummaries:
    async def test_save_and_retrieve(self, storage) -> None:
        await storage.save_session_summary(
            session_id="sess-1",
            topics=["auth", "database"],
            topic_weights={"auth": 0.9, "database": 0.5},
            top_entities=[("User", 3), ("Session", 1)],
            query_count=5,
            avg_confidence=0.85,
            avg_depth=1.5,
            started_at="2026-03-11T10:00:00",
            ended_at="2026-03-11T10:30:00",
        )

        summaries = await storage.get_recent_session_summaries(limit=10)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["session_id"] == "sess-1"
        assert s["topics"] == ["auth", "database"]
        assert s["topic_weights"] == {"auth": 0.9, "database": 0.5}
        # top_entities round-trips back to (entity, count) tuples (audit #74):
        # JSON serializes tuples as arrays, but the read coerces them back so
        # consumers relying on tuple semantics (==, hashing) stay correct.
        assert s["top_entities"] == [("User", 3), ("Session", 1)]
        assert all(isinstance(e, tuple) for e in s["top_entities"])
        assert s["query_count"] == 5
        assert s["avg_confidence"] == pytest.approx(0.85)
        assert s["avg_depth"] == pytest.approx(1.5)

    async def test_no_brain_returns_empty(self, storage) -> None:
        storage.set_brain("")  # empty brain_id
        # save_session_summary with no brain_id should no-op (brain_id falsy)
        # But set_brain sets it to "" which is falsy
        await storage.save_session_summary(
            session_id="x",
            topics=[],
            topic_weights={},
            top_entities=[],
            query_count=0,
            avg_confidence=0.0,
            avg_depth=0.0,
            started_at="",
            ended_at="",
        )
        # Reset brain to retrieve
        storage.set_brain("test-brain")
        summaries = await storage.get_recent_session_summaries()
        assert len(summaries) == 0

    async def test_limit_capped(self, storage) -> None:
        for i in range(5):
            await storage.save_session_summary(
                session_id=f"sess-{i}",
                topics=[f"topic{i}"],
                topic_weights={f"topic{i}": 0.5},
                top_entities=[],
                query_count=i,
                avg_confidence=0.5,
                avg_depth=1.0,
                started_at=f"2026-03-11T{10 + i}:00:00",
                ended_at=f"2026-03-11T{10 + i}:30:00",
            )

        summaries = await storage.get_recent_session_summaries(limit=3)
        assert len(summaries) == 3
        # Most recent first
        assert summaries[0]["session_id"] == "sess-4"

    async def test_ordering_by_ended_at(self, storage) -> None:
        await storage.save_session_summary(
            session_id="old",
            topics=[],
            topic_weights={},
            top_entities=[],
            query_count=1,
            avg_confidence=0.5,
            avg_depth=1.0,
            started_at="2026-03-10T08:00:00",
            ended_at="2026-03-10T09:00:00",
        )
        await storage.save_session_summary(
            session_id="new",
            topics=[],
            topic_weights={},
            top_entities=[],
            query_count=2,
            avg_confidence=0.7,
            avg_depth=2.0,
            started_at="2026-03-11T10:00:00",
            ended_at="2026-03-11T11:00:00",
        )

        summaries = await storage.get_recent_session_summaries()
        assert summaries[0]["session_id"] == "new"
        assert summaries[1]["session_id"] == "old"

    async def test_confidence_rounding(self, storage) -> None:
        await storage.save_session_summary(
            session_id="s1",
            topics=[],
            topic_weights={},
            top_entities=[],
            query_count=1,
            avg_confidence=0.123456789,
            avg_depth=1.789,
            started_at="2026-03-11T10:00:00",
            ended_at="2026-03-11T10:30:00",
        )
        summaries = await storage.get_recent_session_summaries()
        assert summaries[0]["avg_confidence"] == pytest.approx(0.1235, abs=0.0001)
        assert summaries[0]["avg_depth"] == pytest.approx(1.79, abs=0.01)
