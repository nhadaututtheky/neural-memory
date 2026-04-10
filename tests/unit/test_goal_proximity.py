"""Tests for goal-directed recall: BFS proximity scoring + Neuron goal helpers.

Covers:
- Neuron goal metadata properties (goal_state, goal_priority, goal_keywords)
- with_goal_state immutability
- find_active_goals filtering
- compute_goal_proximity BFS (linear, branching, empty)
- Proximity score decay: hop 0=1.0, hop 1=0.5, hop 2=0.33
- SessionState.seed_intent EMA boosting
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.goal_proximity import compute_goal_proximity, find_active_goals
from neural_memory.engine.session_state import SessionState

# ---------------------------------------------------------------------------
# Neuron goal metadata helpers
# ---------------------------------------------------------------------------


class TestNeuronGoalHelpers:
    def test_default_not_goal(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        assert n.goal_state is None
        assert n.goal_priority == 5
        assert n.goal_keywords == []
        assert n.is_active_goal is False

    def test_create_with_goal_metadata(self) -> None:
        n = Neuron.create(
            type=NeuronType.INTENT,
            content="optimize API",
            metadata={
                "_goal_state": "active",
                "_goal_priority": 8,
                "_goal_keywords": ["api", "optimize"],
            },
        )
        assert n.goal_state == "active"
        assert n.goal_priority == 8
        assert n.goal_keywords == ["api", "optimize"]
        assert n.is_active_goal is True

    def test_with_goal_state_immutable(self) -> None:
        n = Neuron.create(type=NeuronType.INTENT, content="original")
        updated = n.with_goal_state("active", priority=7, keywords=["test"])
        # Original unchanged
        assert n.goal_state is None
        # Updated has new state
        assert updated.goal_state == "active"
        assert updated.goal_priority == 7
        assert updated.goal_keywords == ["test"]
        assert updated.id == n.id
        assert updated.content == n.content

    def test_with_goal_state_clamps_priority(self) -> None:
        n = Neuron.create(type=NeuronType.INTENT, content="test")
        low = n.with_goal_state("active", priority=-5)
        high = n.with_goal_state("active", priority=99)
        assert low.goal_priority == 1
        assert high.goal_priority == 10

    def test_paused_goal_not_active(self) -> None:
        n = Neuron.create(
            type=NeuronType.INTENT,
            content="paused goal",
            metadata={"_goal_state": "paused"},
        )
        assert n.goal_state == "paused"
        assert n.is_active_goal is False

    def test_completed_goal_not_active(self) -> None:
        n = Neuron.create(
            type=NeuronType.INTENT,
            content="done",
            metadata={"_goal_state": "completed"},
        )
        assert n.is_active_goal is False


# ---------------------------------------------------------------------------
# find_active_goals
# ---------------------------------------------------------------------------


class TestFindActiveGoals:
    @pytest.mark.asyncio
    async def test_filters_active_only(self) -> None:
        active = Neuron.create(
            type=NeuronType.INTENT,
            content="active",
            metadata={"_goal_state": "active"},
        )
        paused = Neuron.create(
            type=NeuronType.INTENT,
            content="paused",
            metadata={"_goal_state": "paused"},
        )
        regular_intent = Neuron.create(type=NeuronType.INTENT, content="no goal")

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[active, paused, regular_intent])

        result = await find_active_goals(storage)
        assert len(result) == 1
        assert result[0].id == active.id

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=Exception("db error"))

        result = await find_active_goals(storage)
        assert result == []


# ---------------------------------------------------------------------------
# compute_goal_proximity — BFS scoring
# ---------------------------------------------------------------------------


def _mock_storage_with_graph(
    edges: dict[str, list[str]],
) -> AsyncMock:
    """Create mock storage with a simple undirected graph.

    edges: {neuron_id: [neighbor_id, ...]}
    """
    storage = AsyncMock()

    async def get_neighbors(
        neuron_id: str, direction: str = "both", **kwargs: object
    ) -> list[tuple[MagicMock, MagicMock]]:
        neighbor_ids = edges.get(neuron_id, [])
        result = []
        for nid in neighbor_ids:
            n = MagicMock()
            n.id = nid
            s = MagicMock()
            result.append((n, s))
        return result

    storage.get_neighbors = AsyncMock(side_effect=get_neighbors)
    return storage


class TestComputeGoalProximity:
    @pytest.mark.asyncio
    async def test_empty_goals(self) -> None:
        storage = AsyncMock()
        result = await compute_goal_proximity(storage, [], max_hops=3)
        assert result == {}

    @pytest.mark.asyncio
    async def test_linear_chain_scores(self) -> None:
        """Goal -> A -> B -> C, hop scores decay correctly."""
        edges = {
            "goal": ["A"],
            "A": ["goal", "B"],
            "B": ["A", "C"],
            "C": ["B"],
        }
        storage = _mock_storage_with_graph(edges)

        # With explicit priority 10, weight=1.0 → scores match base proximity
        result = await compute_goal_proximity(
            storage, ["goal"], max_hops=3, goal_priorities={"goal": 10}
        )

        assert result["goal"] == pytest.approx(1.0)  # hop 0
        assert result["A"] == pytest.approx(0.5)  # hop 1
        assert result["B"] == pytest.approx(1 / 3, rel=1e-3)  # hop 2
        assert result["C"] == pytest.approx(0.25)  # hop 3

    @pytest.mark.asyncio
    async def test_max_hops_limit(self) -> None:
        """Nodes beyond max_hops are not included."""
        edges = {
            "goal": ["A"],
            "A": ["goal", "B"],
            "B": ["A", "C"],
            "C": ["B"],
        }
        storage = _mock_storage_with_graph(edges)

        result = await compute_goal_proximity(storage, ["goal"], max_hops=1)

        assert "goal" in result
        assert "A" in result
        assert "B" not in result
        assert "C" not in result

    @pytest.mark.asyncio
    async def test_branching_graph(self) -> None:
        """Goal connects to A and B, both connect to C."""
        edges = {
            "goal": ["A", "B"],
            "A": ["goal", "C"],
            "B": ["goal", "C"],
            "C": ["A", "B"],
        }
        storage = _mock_storage_with_graph(edges)

        result = await compute_goal_proximity(
            storage, ["goal"], max_hops=3, goal_priorities={"goal": 10}
        )

        assert result["goal"] == pytest.approx(1.0)
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)
        # C is hop 2 from goal (via A or B)
        assert result["C"] == pytest.approx(1 / 3, rel=1e-3)

    @pytest.mark.asyncio
    async def test_multiple_goals_take_min_distance(self) -> None:
        """Two goals: g1->A->B, g2->B. B should get hop 1 (from g2), not hop 2."""
        edges = {
            "g1": ["A"],
            "A": ["g1", "B"],
            "g2": ["B"],
            "B": ["A", "g2"],
        }
        storage = _mock_storage_with_graph(edges)

        result = await compute_goal_proximity(
            storage, ["g1", "g2"], max_hops=3, goal_priorities={"g1": 10, "g2": 10}
        )

        assert result["B"] == pytest.approx(0.5)  # hop 1 from g2, not hop 2 from g1

    @pytest.mark.asyncio
    async def test_neighbor_error_graceful(self) -> None:
        """If get_neighbors fails, BFS continues with other branches."""
        storage = AsyncMock()
        storage.get_neighbors = AsyncMock(side_effect=Exception("network error"))

        result = await compute_goal_proximity(
            storage, ["goal"], max_hops=3, goal_priorities={"goal": 10}
        )
        # Goal itself is still scored (hop 0)
        assert result["goal"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SessionState.seed_intent
# ---------------------------------------------------------------------------


class TestSessionSeedIntent:
    def test_seed_intent_boosts_ema(self) -> None:
        ss = SessionState(session_id="test")
        ss.seed_intent("debugging sync-hub teams endpoint")

        # Keywords should be in topic_ema with strong boost
        assert ss.session_intent == "debugging sync-hub teams endpoint"
        assert len(ss.topic_ema) > 0
        # At least some keywords should have high weight
        max_weight = max(ss.topic_ema.values())
        assert max_weight >= 0.5  # intent_alpha = 0.6

    def test_seed_intent_empty(self) -> None:
        ss = SessionState(session_id="test")
        ss.seed_intent("")
        assert ss.session_intent == ""
        assert len(ss.topic_ema) == 0

    def test_seed_intent_preserves_existing_topics(self) -> None:
        ss = SessionState(session_id="test")
        ss.topic_ema = {"python": 0.5, "async": 0.3}
        ss.seed_intent("optimize database queries")
        # Old topics preserved
        assert "python" in ss.topic_ema
        assert "async" in ss.topic_ema
        # New intent keywords added
        assert len(ss.topic_ema) > 2
