"""Unit tests for A8 Phase 2: Proactive Context.

Tests surface signal injection, cluster-based topic injection,
session meta-summary, and tool event → topic EMA feed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.session_state import SessionManager, SessionState
from neural_memory.surface.models import (
    Cluster,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session_manager():
    """Reset SessionManager singleton between tests."""
    SessionManager.reset()
    yield
    SessionManager.reset()


def _make_surface(
    signals: list[Signal] | None = None,
    clusters: list[Cluster] | None = None,
    meta: SurfaceMeta | None = None,
    graph: list[GraphEntry] | None = None,
) -> KnowledgeSurface:
    """Build a minimal KnowledgeSurface for testing."""
    fm = SurfaceFrontmatter(brain="test", updated="2026-04-01")
    return KnowledgeSurface(
        frontmatter=fm,
        signals=tuple(signals or []),
        clusters=tuple(clusters or []),
        meta=meta or SurfaceMeta(),
        graph=tuple(graph or []),
    )


def _make_session_state(
    topics: dict[str, float] | None = None,
    query_count: int = 5,
) -> SessionState:
    """Create a detached SessionState (not in manager)."""
    state = SessionState(session_id="test")
    state.query_count = query_count
    if topics:
        state.topic_ema = dict(topics)
    return state


async def _call_proactive(
    surface: KnowledgeSurface | None = None,
    session_state: SessionState | None = None,
    storage: AsyncMock | None = None,
    existing_fiber_ids: set[str] | None = None,
) -> dict[str, str]:
    """Call _build_proactive_context with controlled surface + session."""
    from neural_memory.mcp.recall_handler import RecallHandler

    handler = MagicMock()
    handler._surface_text = "mock-text" if surface is not None else None

    patches = {}
    if surface is not None:
        patches["neural_memory.surface.parser.parse"] = MagicMock(return_value=surface)

    # Patch SessionManager.get_instance().get() to return our session
    mock_mgr = MagicMock()
    mock_mgr.get.return_value = session_state
    patches["neural_memory.engine.session_state.SessionManager.get_instance"] = MagicMock(
        return_value=mock_mgr
    )

    ctx = {}
    for target, mock in patches.items():
        p = patch(target, mock)
        p.start()
        ctx[target] = p

    try:
        result = await RecallHandler._build_proactive_context(
            handler,
            storage or AsyncMock(),
            existing_fiber_ids or set(),
        )
    finally:
        for p in ctx.values():
            p.stop()

    return result


# ---------------------------------------------------------------------------
# T2.2: Surface SIGNALS as proactive alerts
# ---------------------------------------------------------------------------


class TestSurfaceSignals:
    """Surface SIGNALS should appear in context output."""

    async def test_urgent_signals_formatted(self):
        """URGENT signals should be prefixed with '!'."""
        surface = _make_surface(
            signals=[
                Signal(level=SignalLevel.URGENT, text="Redis connection pool exhaustion"),
            ]
        )
        result = await _call_proactive(surface=surface)
        assert "! Redis connection pool exhaustion" in result["signals"]

    async def test_watching_signals_formatted(self):
        """WATCHING signals should be prefixed with '~'."""
        surface = _make_surface(
            signals=[
                Signal(level=SignalLevel.WATCHING, text="Auth token rotation pending"),
            ]
        )
        result = await _call_proactive(surface=surface)
        assert "~ Auth token rotation pending" in result["signals"]

    async def test_mixed_signals_under_header(self):
        """Both URGENT and WATCHING signals should appear under header."""
        surface = _make_surface(
            signals=[
                Signal(level=SignalLevel.URGENT, text="Critical alert"),
                Signal(level=SignalLevel.WATCHING, text="Minor watch"),
            ]
        )
        result = await _call_proactive(surface=surface)
        assert "--- active signals ---" in result["signals"]
        assert "! Critical alert" in result["signals"]
        assert "~ Minor watch" in result["signals"]

    async def test_no_signals_returns_empty(self):
        """No signals → empty string."""
        surface = _make_surface(signals=[])
        result = await _call_proactive(surface=surface)
        assert result["signals"] == ""


# ---------------------------------------------------------------------------
# T2.1: Surface cluster injection
# ---------------------------------------------------------------------------


class TestClusterInjection:
    """Surface CLUSTERS matched to session topics inject memories."""

    async def test_matching_cluster_injects_topic_context(self):
        """Cluster whose name overlaps session topic should inject fibers."""
        node = SurfaceNode(id="d1", content="auth middleware", node_type="decision")
        cluster = Cluster(name="auth", node_ids=("d1",), description="authentication flow")
        surface = _make_surface(
            clusters=[cluster],
            graph=[GraphEntry(node=node)],
        )
        session = _make_session_state(topics={"auth": 0.8, "cache": 0.5})

        mock_storage = AsyncMock()
        # H1 fix: now uses find_neurons + find_fibers instead of find_typed_memories
        mock_neuron = MagicMock()
        mock_neuron.id = "n-auth-1"
        mock_storage.find_neurons = AsyncMock(return_value=[mock_neuron])

        mock_fiber = MagicMock()
        mock_fiber.id = "fiber-auth-1"
        mock_fiber.summary = "Auth middleware uses JWT with rotation"
        mock_fiber.anchor_neuron_id = "n-auth-1"
        mock_storage.find_fibers = AsyncMock(return_value=[mock_fiber])
        mock_storage.get_fiber = AsyncMock(return_value=mock_fiber)

        result = await _call_proactive(
            surface=surface,
            session_state=session,
            storage=mock_storage,
        )
        assert "--- active topic context ---" in result["topic_context"]
        assert "Auth middleware uses JWT" in result["topic_context"]

    async def test_no_matching_cluster_no_injection(self):
        """Cluster name doesn't match session topics → no injection."""
        cluster = Cluster(name="database", node_ids=("d1",))
        surface = _make_surface(clusters=[cluster])
        session = _make_session_state(topics={"auth": 0.8})

        result = await _call_proactive(surface=surface, session_state=session)
        assert result["topic_context"] == ""

    async def test_no_session_state_no_injection(self):
        """Without session state, cluster injection is skipped."""
        cluster = Cluster(name="auth", node_ids=("d1",))
        surface = _make_surface(clusters=[cluster])

        result = await _call_proactive(surface=surface, session_state=None)
        assert result["topic_context"] == ""

    async def test_existing_fiber_ids_deduplicated(self):
        """Already-injected fibers should not be injected again."""
        node = SurfaceNode(id="d1", content="auth", node_type="decision")
        cluster = Cluster(name="auth", node_ids=("d1",))
        surface = _make_surface(
            clusters=[cluster],
            graph=[GraphEntry(node=node)],
        )
        session = _make_session_state(topics={"auth": 0.8})

        mock_storage = AsyncMock()
        mock_neuron = MagicMock()
        mock_neuron.id = "n-auth-1"
        mock_storage.find_neurons = AsyncMock(return_value=[mock_neuron])

        mock_fiber = MagicMock()
        mock_fiber.id = "fiber-already-injected"
        mock_storage.find_fibers = AsyncMock(return_value=[mock_fiber])

        result = await _call_proactive(
            surface=surface,
            session_state=session,
            storage=mock_storage,
            existing_fiber_ids={"fiber-already-injected"},
        )
        # Fiber already in existing_ids → should not appear in topic_context
        assert result["topic_context"] == ""

    async def test_max_9_topic_memories_cap(self):
        """Should inject at most 9 topic memories (3 clusters x 3 each)."""
        nodes = [
            SurfaceNode(id=f"d{i}", content=f"auth item {i}", node_type="decision")
            for i in range(15)
        ]
        cluster = Cluster(
            name="auth",
            node_ids=tuple(f"d{i}" for i in range(15)),
        )
        surface = _make_surface(
            clusters=[cluster],
            graph=[GraphEntry(node=n) for n in nodes],
        )
        session = _make_session_state(topics={"auth": 0.9})

        # Each neuron lookup returns a unique neuron → fiber
        call_count = 0

        async def mock_find_neurons(content_contains=None, limit=None, **kwargs):
            nonlocal call_count
            call_count += 1
            n = MagicMock()
            n.id = f"n-{call_count}"
            return [n]

        fiber_count = 0

        async def mock_find_fibers(contains_neuron=None, limit=None, **kwargs):
            nonlocal fiber_count
            fiber_count += 1
            f = MagicMock()
            f.id = f"fiber-{fiber_count}"
            f.summary = f"Auth content {fiber_count}"
            f.anchor_neuron_id = f"n-{fiber_count}"
            return [f]

        mock_storage = AsyncMock()
        mock_storage.find_neurons = mock_find_neurons
        mock_storage.find_fibers = mock_find_fibers

        mock_fiber = MagicMock()
        mock_fiber.summary = "Some auth content"
        mock_fiber.anchor_neuron_id = "n1"
        mock_storage.get_fiber = AsyncMock(return_value=mock_fiber)

        result = await _call_proactive(
            surface=surface,
            session_state=session,
            storage=mock_storage,
        )
        # M5 fix: assert topic_context is non-empty, then check cap
        assert result["topic_context"] != "", "Expected non-empty topic_context for 15 nodes"
        lines = [line for line in result["topic_context"].split("\n") if line.startswith("- ")]
        assert len(lines) <= 9


# ---------------------------------------------------------------------------
# T2.4: Session meta-summary
# ---------------------------------------------------------------------------


class TestSessionMetaSummary:
    """Session summary line in context header."""

    async def test_summary_includes_query_count_and_topics(self):
        """Summary should show query count and top topics."""
        session = _make_session_state(topics={"auth": 0.8, "cache": 0.6}, query_count=12)
        result = await _call_proactive(session_state=session)
        assert "Session: 12 queries" in result["session_summary"]
        assert "auth" in result["session_summary"]

    async def test_summary_includes_surface_coverage(self):
        """Summary should include surface coverage when available."""
        session = _make_session_state(topics={"auth": 0.7}, query_count=5)
        surface = _make_surface(meta=SurfaceMeta(coverage=0.85))
        result = await _call_proactive(surface=surface, session_state=session)
        assert "85%" in result["session_summary"]

    async def test_staleness_only_shown_when_high(self):
        """Staleness should only appear when >50%."""
        session = _make_session_state(topics={"auth": 0.7}, query_count=3)

        # Low staleness — should NOT appear
        surface_low = _make_surface(meta=SurfaceMeta(coverage=0.9, staleness=0.3))
        result = await _call_proactive(surface=surface_low, session_state=session)
        assert "staleness" not in result["session_summary"]

        # High staleness — should appear
        surface_high = _make_surface(meta=SurfaceMeta(coverage=0.9, staleness=0.7))
        result2 = await _call_proactive(surface=surface_high, session_state=session)
        assert "staleness: 70%" in result2["session_summary"]

    async def test_no_session_no_summary(self):
        """Without session state, summary is empty."""
        result = await _call_proactive(session_state=None)
        assert result["session_summary"] == ""

    async def test_summary_wraps_in_dashes(self):
        """Summary line should be wrapped in --- delimiters."""
        session = _make_session_state(topics={"auth": 0.7}, query_count=3)
        result = await _call_proactive(session_state=session)
        assert result["session_summary"].startswith("--- ")
        assert result["session_summary"].endswith(" ---")


# ---------------------------------------------------------------------------
# T2.3: Tool event → topic EMA feed
# ---------------------------------------------------------------------------


class TestToolTopicEmaFeed:
    """Tool actions should feed topics into session EMA."""

    def test_tool_context_feeds_keywords(self):
        """Keywords from tool context should appear in session EMA."""
        from neural_memory.mcp.evolution_handler import EvolutionHandler

        mgr = SessionManager.get_instance()
        state = mgr.get_or_create("mcp-200")
        state.topic_ema = {}

        EvolutionHandler._feed_tool_topics_to_ema(
            "mcp-200", "auth middleware JWT rotation", "recall"
        )

        weights = state.get_topic_weights(limit=10)
        assert len(weights) > 0

    def test_tool_context_feeds_file_stems(self):
        """File stems from paths should be extracted as topics."""
        from neural_memory.mcp.evolution_handler import EvolutionHandler

        mgr = SessionManager.get_instance()
        state = mgr.get_or_create("mcp-201")
        state.topic_ema = {}

        EvolutionHandler._feed_tool_topics_to_ema(
            "mcp-201", "editing retrieval.py and auth_middleware.ts", "remember"
        )

        weights = state.get_topic_weights(limit=10)
        assert "retrieval" in weights or "auth_middleware" in weights

    def test_half_weight_alpha_is_lower(self):
        """Tool EMA feed should use lower alpha (0.15) than record_query (0.3)."""
        from neural_memory.mcp.evolution_handler import EvolutionHandler

        mgr = SessionManager.get_instance()

        # Feed via tool (half weight alpha=0.15)
        state1 = mgr.get_or_create("mcp-202")
        state1.topic_ema = {"existing": 1.0}
        EvolutionHandler._feed_tool_topics_to_ema("mcp-202", "database", "recall")
        # existing should decay less with half alpha (0.15) vs full (0.3)
        tool_decay = state1.topic_ema.get("existing", 0.0)

        state2 = mgr.get_or_create("mcp-203")
        state2.topic_ema = {"existing": 1.0}
        state2.record_query(
            query="database",
            depth_used=1,
            confidence=0.8,
            fibers_matched=5,
        )
        query_decay = state2.topic_ema.get("existing", 0.0)

        # Tool uses half alpha (0.15) → less decay: 1.0 * (1-0.15) = 0.85
        # Query uses full alpha (0.3) → more decay: 1.0 * (1-0.3) = 0.7
        assert tool_decay > query_decay

    def test_no_session_no_error(self):
        """If session doesn't exist, should silently return."""
        from neural_memory.mcp.evolution_handler import EvolutionHandler

        # No session created for this ID — should not raise
        EvolutionHandler._feed_tool_topics_to_ema("nonexistent-session", "some context", "recall")

    def test_empty_context_no_change(self):
        """Empty context should not modify EMA."""
        from neural_memory.mcp.evolution_handler import EvolutionHandler

        mgr = SessionManager.get_instance()
        state = mgr.get_or_create("mcp-204")
        state.topic_ema = {"existing": 0.5}

        EvolutionHandler._feed_tool_topics_to_ema("mcp-204", "", "recall")

        # extract_keywords("") returns [] → early return, no changes at all
        assert "existing" in state.topic_ema
        assert state.topic_ema["existing"] == 0.5  # M6 fix: verify value unchanged


# ---------------------------------------------------------------------------
# Combined / integration-style tests
# ---------------------------------------------------------------------------


class TestProactiveContextCombined:
    """End-to-end proactive context assembly."""

    async def test_all_sections_populated(self):
        """When all data available, all three sections should be populated."""
        surface = _make_surface(
            signals=[Signal(level=SignalLevel.URGENT, text="High CPU usage")],
            clusters=[Cluster(name="perf", node_ids=("d1",), description="performance")],
            meta=SurfaceMeta(coverage=0.92),
            graph=[
                GraphEntry(node=SurfaceNode(id="d1", content="perf tuning", node_type="insight"))
            ],
        )
        session = _make_session_state(topics={"perf": 0.9}, query_count=8)

        mock_storage = AsyncMock()
        mock_neuron = MagicMock()
        mock_neuron.id = "n-perf-1"
        mock_storage.find_neurons = AsyncMock(return_value=[mock_neuron])

        mock_fiber = MagicMock()
        mock_fiber.id = "fiber-perf-1"
        mock_fiber.summary = "DB query takes 200ms avg"
        mock_fiber.anchor_neuron_id = "n-perf-1"
        mock_storage.find_fibers = AsyncMock(return_value=[mock_fiber])
        mock_storage.get_fiber = AsyncMock(return_value=mock_fiber)

        result = await _call_proactive(surface=surface, session_state=session, storage=mock_storage)

        assert result["signals"] != ""
        assert result["topic_context"] != ""
        assert result["session_summary"] != ""

    async def test_graceful_with_no_surface(self):
        """Without surface, signals and clusters are empty but no error."""
        session = _make_session_state(topics={"auth": 0.7}, query_count=3)
        result = await _call_proactive(session_state=session)
        assert result["signals"] == ""
        assert result["topic_context"] == ""
        assert "Session: 3 queries" in result["session_summary"]

    async def test_graceful_with_no_session_and_no_surface(self):
        """Without session or surface, all sections are empty."""
        result = await _call_proactive()
        assert result["signals"] == ""
        assert result["topic_context"] == ""
        assert result["session_summary"] == ""
