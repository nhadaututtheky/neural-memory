"""Tests for Phase 4: Goal Hierarchy + Unified Confidence API.

Covers:
- SUBGOAL_OF synapse type existence and role mapping
- Neuron.parent_goal_id property
- with_goal_state parent_goal_id parameter
- Goal handler: create with parent, subgoals listing, completion hint
- BFS priority inheritance from parent goals
- ConfidenceScore computation (all dimensions)
- Confidence weights configurability
- Familiarity fallback penalty
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SYNAPSE_ROLES, SynapseRole, SynapseType
from neural_memory.engine.confidence import (
    ConfidenceWeights,
    compute_confidence,
)
from neural_memory.engine.goal_proximity import (
    _effective_priority,
    compute_goal_proximity,
)
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Part A: Goal Hierarchy — Synapse & Neuron
# ---------------------------------------------------------------------------


class TestSubgoalOfSynapse:
    def test_subgoal_of_exists(self) -> None:
        assert hasattr(SynapseType, "SUBGOAL_OF")
        assert SynapseType.SUBGOAL_OF == "subgoal_of"

    def test_subgoal_of_role_is_structural(self) -> None:
        assert SYNAPSE_ROLES[SynapseType.SUBGOAL_OF] == SynapseRole.STRUCTURAL


class TestNeuronParentGoalId:
    def test_default_no_parent(self) -> None:
        n = Neuron.create(type=NeuronType.INTENT, content="test")
        assert n.parent_goal_id is None

    def test_parent_goal_id_from_metadata(self) -> None:
        n = Neuron.create(
            type=NeuronType.INTENT,
            content="sub task",
            metadata={"_parent_goal_id": "parent-123", "_goal_state": "active"},
        )
        assert n.parent_goal_id == "parent-123"

    def test_with_goal_state_sets_parent(self) -> None:
        n = Neuron.create(type=NeuronType.INTENT, content="child goal")
        updated = n.with_goal_state("active", priority=6, parent_goal_id="parent-456")
        assert updated.parent_goal_id == "parent-456"
        assert updated.goal_state == "active"
        assert updated.goal_priority == 6
        # Original unchanged
        assert n.parent_goal_id is None

    def test_with_goal_state_no_parent_default(self) -> None:
        n = Neuron.create(type=NeuronType.INTENT, content="standalone")
        updated = n.with_goal_state("active", priority=5)
        assert updated.parent_goal_id is None


# ---------------------------------------------------------------------------
# Part A: Goal Handler — Create with parent + Subgoals listing
# ---------------------------------------------------------------------------


class TestGoalHandlerSubgoals:
    @pytest.mark.asyncio
    async def test_create_with_parent(self) -> None:
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = GoalHandler()

        parent = Neuron.create(
            type=NeuronType.INTENT,
            content="Ship v5.0",
            metadata={"_goal_state": "active", "_goal_priority": 9},
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=parent)
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._goal_create(
            {
                "goal": "Implement scheduler",
                "priority": 7,
                "parent_goal_id": parent.id,
            }
        )

        assert result.get("parent_goal_id") == parent.id
        assert result["state"] == "active"
        assert result["priority"] == 7
        # Synapse should have been created
        storage.add_synapse.assert_called_once()
        synapse_arg = storage.add_synapse.call_args[0][0]
        assert synapse_arg.type == SynapseType.SUBGOAL_OF
        assert synapse_arg.target_id == parent.id

    @pytest.mark.asyncio
    async def test_create_with_invalid_parent(self) -> None:
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = GoalHandler()
        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=None)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._goal_create(
            {
                "goal": "Child goal",
                "parent_goal_id": "nonexistent",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_parent_not_a_goal(self) -> None:
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = GoalHandler()
        non_goal = Neuron.create(type=NeuronType.CONCEPT, content="just a concept")
        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=non_goal)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._goal_create(
            {
                "goal": "Child goal",
                "parent_goal_id": non_goal.id,
            }
        )
        assert "error" in result
        assert "not a goal" in result["error"]

    @pytest.mark.asyncio
    async def test_subgoals_listing(self) -> None:
        from neural_memory.core.synapse import Synapse
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = GoalHandler()

        parent = Neuron.create(
            type=NeuronType.INTENT,
            content="Ship v5.0",
            metadata={"_goal_state": "active", "_goal_priority": 9},
        )
        child1 = Neuron.create(
            type=NeuronType.INTENT,
            content="Implement scheduler",
            metadata={"_goal_state": "completed", "_goal_priority": 7},
        )
        child2 = Neuron.create(
            type=NeuronType.INTENT,
            content="Add ACL",
            metadata={"_goal_state": "completed", "_goal_priority": 8},
        )

        syn1 = Synapse.create(
            source_id=child1.id,
            target_id=parent.id,
            type=SynapseType.SUBGOAL_OF,
            weight=0.8,
        )
        syn2 = Synapse.create(
            source_id=child2.id,
            target_id=parent.id,
            type=SynapseType.SUBGOAL_OF,
            weight=0.8,
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(
            side_effect=lambda nid: {
                parent.id: parent,
                child1.id: child1,
                child2.id: child2,
            }.get(nid)
        )
        storage.get_synapses = AsyncMock(return_value=[syn1, syn2])
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._goal_subgoals({"goal_id": parent.id})

        assert result["total"] == 2
        assert result["all_completed"] is True
        assert "Consider completing parent" in result["hint"]

    @pytest.mark.asyncio
    async def test_subgoals_not_all_completed(self) -> None:
        from neural_memory.core.synapse import Synapse
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = GoalHandler()

        parent = Neuron.create(
            type=NeuronType.INTENT,
            content="Big goal",
            metadata={"_goal_state": "active"},
        )
        child = Neuron.create(
            type=NeuronType.INTENT,
            content="In progress",
            metadata={"_goal_state": "active", "_goal_priority": 5},
        )
        syn = Synapse.create(
            source_id=child.id,
            target_id=parent.id,
            type=SynapseType.SUBGOAL_OF,
            weight=0.8,
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(
            side_effect=lambda nid: {
                parent.id: parent,
                child.id: child,
            }.get(nid)
        )
        storage.get_synapses = AsyncMock(return_value=[syn])
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._goal_subgoals({"goal_id": parent.id})
        assert result["all_completed"] is False
        assert result["hint"] == ""


# ---------------------------------------------------------------------------
# Part A: BFS Priority Inheritance
# ---------------------------------------------------------------------------


class TestEffectivePriority:
    def test_no_parent(self) -> None:
        weight = _effective_priority("g1", {"g1": 6}, {})
        assert weight == pytest.approx(0.6)

    def test_parent_boosts(self) -> None:
        # Child priority 4, parent priority 10 → inherited = 10 * 0.8 = 8
        weight = _effective_priority("child", {"child": 4, "parent": 10}, {"child": "parent"})
        assert weight == pytest.approx(0.8)  # max(4, 8) = 8 → 8/10

    def test_child_higher_than_inherited(self) -> None:
        # Child priority 9, parent priority 6 → inherited = 4.8, child wins
        weight = _effective_priority("child", {"child": 9, "parent": 6}, {"child": "parent"})
        assert weight == pytest.approx(0.9)

    def test_parent_not_in_priorities(self) -> None:
        # Parent exists in map but not in priorities dict → no inheritance
        weight = _effective_priority("child", {"child": 5}, {"child": "parent"})
        assert weight == pytest.approx(0.5)

    def test_caps_at_1(self) -> None:
        weight = _effective_priority("child", {"child": 10, "parent": 10}, {"child": "parent"})
        assert weight == pytest.approx(1.0)


class TestGoalProximityWithInheritance:
    @pytest.mark.asyncio
    async def test_subgoal_inherits_parent_priority(self) -> None:
        """Subgoal (priority 3) with parent (priority 10) should get boosted proximity."""
        edges: dict[str, list[str]] = {
            "sub": ["A"],
            "A": ["sub"],
        }
        storage = AsyncMock()

        async def get_neighbors(
            neuron_id: str, direction: str = "both", **kw: object
        ) -> list[tuple[MagicMock, MagicMock]]:
            nids = edges.get(neuron_id, [])
            result = []
            for nid in nids:
                n = MagicMock()
                n.id = nid
                result.append((n, MagicMock()))
            return result

        storage.get_neighbors = AsyncMock(side_effect=get_neighbors)

        # Without parent: priority 3 → weight 0.3
        result_no_parent = await compute_goal_proximity(
            storage,
            ["sub"],
            max_hops=1,
            goal_priorities={"sub": 3},
        )
        assert result_no_parent["sub"] == pytest.approx(0.3)

        # With parent: inherited = 10 * 0.8 = 8, effective = max(3, 8) = 8 → weight 0.8
        result_with_parent = await compute_goal_proximity(
            storage,
            ["sub"],
            max_hops=1,
            goal_priorities={"sub": 3, "parent": 10},
            parent_map={"sub": "parent"},
        )
        assert result_with_parent["sub"] == pytest.approx(0.8)
        assert result_with_parent["A"] == pytest.approx(0.4)  # hop 1: 0.5 * 0.8


# ---------------------------------------------------------------------------
# Part B: Unified Confidence API
# ---------------------------------------------------------------------------


class TestConfidenceScore:
    def test_high_quality_recent_verbatim(self) -> None:
        """Recent, high-quality, verbatim match should score > 0.8."""
        score = compute_confidence(
            retrieval_score=0.9,
            sufficiency_confidence=0.85,
            quality_score=9.0,
            fidelity_layer="verbatim",
            created_at=utcnow() - timedelta(hours=1),
            is_familiarity_fallback=False,
        )
        assert score.overall > 0.8
        assert score.fidelity == 1.0
        assert score.familiarity_penalty == 0.0

    def test_old_low_quality_gist_familiarity(self) -> None:
        """Old, low-quality, gist familiarity fallback should score < 0.3."""
        score = compute_confidence(
            retrieval_score=0.2,
            sufficiency_confidence=0.1,
            quality_score=2.0,
            fidelity_layer="gist",
            created_at=utcnow() - timedelta(days=180),
            is_familiarity_fallback=True,
        )
        assert score.overall < 0.3
        assert score.familiarity_penalty == -0.3

    def test_familiarity_penalty_applied(self) -> None:
        """Familiarity fallback should reduce overall score."""
        base = compute_confidence(
            retrieval_score=0.5,
            sufficiency_confidence=0.5,
            quality_score=5.0,
            is_familiarity_fallback=False,
        )
        penalized = compute_confidence(
            retrieval_score=0.5,
            sufficiency_confidence=0.5,
            quality_score=5.0,
            is_familiarity_fallback=True,
        )
        assert penalized.overall < base.overall

    def test_custom_weights(self) -> None:
        """Custom weights should change the scoring."""
        # All weight on retrieval
        weights = ConfidenceWeights(
            retrieval=1.0,
            content_quality=0.0,
            fidelity=0.0,
            freshness=0.0,
        )
        score = compute_confidence(
            retrieval_score=0.9,
            sufficiency_confidence=0.8,
            quality_score=1.0,  # low quality shouldn't matter
            fidelity_layer="gist",  # low fidelity shouldn't matter
            weights=weights,
        )
        # Retrieval = 0.9*0.6 + 0.8*0.4 = 0.86, overall ~ 0.86
        assert score.overall > 0.8

    def test_overall_clamped_0_1(self) -> None:
        """Overall should always be in [0, 1]."""
        low = compute_confidence(
            retrieval_score=0.0,
            sufficiency_confidence=0.0,
            quality_score=0.0,
            fidelity_layer="essence",
            created_at=utcnow() - timedelta(days=3650),
            is_familiarity_fallback=True,
        )
        assert low.overall >= 0.0

        high = compute_confidence(
            retrieval_score=1.0,
            sufficiency_confidence=1.0,
            quality_score=10.0,
            fidelity_layer="verbatim",
            created_at=utcnow(),
        )
        assert high.overall <= 1.0

    def test_components_included(self) -> None:
        """Components dict should contain raw signal values."""
        score = compute_confidence(
            retrieval_score=0.7,
            sufficiency_confidence=0.6,
            quality_score=8.0,
            extra_signals={"custom_metric": 0.42},
        )
        assert "retrieval_score" in score.components
        assert "sufficiency_confidence" in score.components
        assert "quality_score" in score.components
        assert score.components["custom_metric"] == 0.42

    def test_freshness_decays_with_age(self) -> None:
        """Older memories should have lower freshness."""
        recent = compute_confidence(created_at=utcnow() - timedelta(hours=1))
        old = compute_confidence(created_at=utcnow() - timedelta(days=90))
        assert recent.freshness > old.freshness

    def test_unknown_fidelity_defaults(self) -> None:
        """Unknown fidelity layer should get 0.5."""
        score = compute_confidence(fidelity_layer="unknown_layer")
        assert score.fidelity == 0.5

    def test_no_created_at_neutral_freshness(self) -> None:
        """No created_at should give neutral freshness (0.5)."""
        score = compute_confidence(created_at=None)
        assert score.freshness == 0.5
