"""Tests for depth gap fixes (v4.43 batch).

Covers:
1. Interference: pinned/grounded neuron exemption
2. Hippocampal replay: configurable window/schedule
3. Emotion recall filter (min_arousal)
4. Forgetting curve at-risk query (lifecycle action)
5. Goal conflict resolution: priority-weighted BFS proximity
6. Temporal neighborhood MCP tool (nmem_causal)
7. Valence recall filter
8. Context-dependent retrieval: session topic in fingerprint
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.interference import (
    InterferenceResult,
    InterferenceType,
    detect_interference,
    resolve_interference,
)
from neural_memory.utils.timeutils import utcnow

# ═══════════════════════ Helpers ═══════════════════════


def _make_neuron(
    content: str,
    tags: list[str] | None = None,
    created_offset_hours: int = 0,
    grounded: bool = False,
    neuron_id: str | None = None,
) -> Neuron:
    meta: dict[str, object] = {"tags": tags or []}
    if grounded:
        meta["_grounded"] = True
    n = Neuron.create(
        content=content,
        type=NeuronType.CONCEPT,
        metadata=meta,
    )
    if neuron_id:
        n = replace(n, id=neuron_id)
    if created_offset_hours:
        n = replace(n, created_at=utcnow() - timedelta(hours=created_offset_hours))
    return n


def _make_interference_config(enabled: bool = True, fan_threshold: int = 15) -> MagicMock:
    config = MagicMock()
    config.interference_detection_enabled = enabled
    config.fan_effect_threshold = fan_threshold
    return config


# ═══════════════════════ 1. Interference — pinned/grounded exemption ═══════════════════════


class TestInterferencePinnedExemption:
    @pytest.mark.asyncio
    async def test_grounded_neuron_skipped_as_new(self) -> None:
        """A grounded neuron should never cause interference as the new neuron."""
        config = _make_interference_config()
        grounded = _make_neuron("Python patterns", ["python"], grounded=True)
        storage = AsyncMock()

        results = await detect_interference(grounded, storage, config)
        assert results == []
        # Storage should never be queried since we return early
        storage.find_neurons.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_candidate_skipped(self) -> None:
        """Pinned neurons should not appear as interference candidates."""
        config = _make_interference_config()
        new_neuron = _make_neuron("Python async patterns", ["python"])
        pinned_neuron = _make_neuron(
            "Python async await fundamentals",
            ["python"],
            created_offset_hours=24,
            neuron_id="pinned-123",
        )

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[pinned_neuron])
        storage.get_pinned_neuron_ids = AsyncMock(return_value={"pinned-123"})

        results = await detect_interference(new_neuron, storage, config)
        # Pinned neuron should be skipped entirely
        non_fan = [r for r in results if r.interference_type != InterferenceType.FAN_EFFECT]
        assert len(non_fan) == 0

    @pytest.mark.asyncio
    async def test_grounded_candidate_skipped(self) -> None:
        """Grounded neurons should not appear as interference candidates."""
        config = _make_interference_config()
        new_neuron = _make_neuron("Python design patterns overview", ["python"])
        grounded_candidate = _make_neuron(
            "Python design patterns reference",
            ["python"],
            created_offset_hours=24,
            grounded=True,
        )

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[grounded_candidate])
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())

        results = await detect_interference(new_neuron, storage, config)
        non_fan = [r for r in results if r.interference_type != InterferenceType.FAN_EFFECT]
        assert len(non_fan) == 0

    @pytest.mark.asyncio
    async def test_resolve_skips_pinned_retroactive(self) -> None:
        """resolve_interference should skip weight reduction for pinned neurons."""
        config = MagicMock()
        new_neuron = _make_neuron("new content", ["python"])
        results = [
            InterferenceResult(
                neuron_id="pinned-456",
                score=0.8,
                interference_type=InterferenceType.RETROACTIVE,
            )
        ]

        storage = AsyncMock()
        storage.get_pinned_neuron_ids = AsyncMock(return_value={"pinned-456"})
        storage.add_synapse = AsyncMock()
        storage.get_synapses = AsyncMock(return_value=[])

        report = await resolve_interference(results, new_neuron, storage, config)
        # Pinned neuron should be skipped, so contradicts_created stays 0
        assert report.contradicts_created == 0
        storage.add_synapse.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_pinned_still_detected(self) -> None:
        """Regular (non-pinned, non-grounded) neurons still detected normally."""
        config = _make_interference_config()
        new_neuron = _make_neuron("Python async await patterns are complex", ["python"])
        old_neuron = _make_neuron(
            "Python async await patterns are complicated",
            ["python"],
            created_offset_hours=24,
            neuron_id="regular-789",
        )

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[old_neuron])
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())

        results = await detect_interference(new_neuron, storage, config)
        retro = [r for r in results if r.interference_type == InterferenceType.RETROACTIVE]
        assert len(retro) >= 1


# ═══════════════════════ 2. Hippocampal replay — configurable window ═══════════════════════


class TestReplayConfigurableWindow:
    @pytest.mark.asyncio
    async def test_custom_window_hours(self) -> None:
        """replay_window_hours config should be passed to find_fibers."""
        from neural_memory.engine.hippocampal_replay import hippocampal_replay

        config = MagicMock()
        config.replay_enabled = True
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98
        config.replay_window_hours = 48.0
        config.replay_max_episodes = 10

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[])

        await hippocampal_replay(storage, config)

        # Verify find_fibers was called with the time window
        call_kwargs = storage.find_fibers.call_args
        assert call_kwargs is not None
        time_overlaps = call_kwargs.kwargs.get("time_overlaps") or call_kwargs[1].get(
            "time_overlaps"
        )
        assert time_overlaps is not None
        start, end = time_overlaps
        # The window should be ~48 hours
        delta = end - start
        assert 47.9 < delta.total_seconds() / 3600 < 48.1

    @pytest.mark.asyncio
    async def test_custom_max_episodes(self) -> None:
        """replay_max_episodes config should cap the number of fibers replayed."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.hippocampal_replay import hippocampal_replay

        config = MagicMock()
        config.replay_enabled = True
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98
        config.replay_window_hours = 24.0
        config.replay_max_episodes = 2  # Only replay 2

        # Create 5 fibers
        fibers = []
        for i in range(5):
            f = MagicMock(spec=Fiber)
            f.neuron_ids = [f"n-{i}-a", f"n-{i}-b"]
            f.salience = 0.5
            fibers.append(f)

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=fibers)
        storage.get_synapses = AsyncMock(return_value=[])

        result = await hippocampal_replay(storage, config, seed=42)
        # Should have replayed at most 2 episodes
        assert result.episodes_replayed <= 2

    @pytest.mark.asyncio
    async def test_default_values_work(self) -> None:
        """When config attrs are missing, defaults should be used."""
        from neural_memory.engine.hippocampal_replay import hippocampal_replay

        config = MagicMock(spec=[])  # Empty spec — no attributes
        config.replay_enabled = True
        config.replay_ltp_factor = 1.1
        config.replay_ltd_factor = 0.98
        # Don't set replay_window_hours or replay_max_episodes

        storage = AsyncMock()
        storage.find_fibers = AsyncMock(return_value=[])

        # Should not raise
        result = await hippocampal_replay(storage, config)
        assert result.episodes_replayed == 0


# ═══════════════════════ 3. Emotion recall filter (min_arousal) ═══════════════════════


class TestMinArousalFilter:
    @pytest.mark.asyncio
    async def test_min_arousal_schema_exists(self) -> None:
        """The nmem_recall tool schema should include min_arousal parameter."""
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS as TOOL_SCHEMAS

        recall_schema = None
        for schema in TOOL_SCHEMAS:
            if schema["name"] == "nmem_recall":
                recall_schema = schema
                break

        assert recall_schema is not None
        props = recall_schema["inputSchema"]["properties"]
        assert "min_arousal" in props
        assert props["min_arousal"]["type"] == "number"
        assert props["min_arousal"]["minimum"] == 0.0
        assert props["min_arousal"]["maximum"] == 1.0


# ═══════════════════════ 4. Forgetting curve at-risk query ═══════════════════════


class TestLifecycleAtRisk:
    @pytest.mark.asyncio
    async def test_at_risk_schema_exists(self) -> None:
        """The nmem_lifecycle tool schema should include at_risk action."""
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS as TOOL_SCHEMAS

        lifecycle_schema = None
        for schema in TOOL_SCHEMAS:
            if schema["name"] == "nmem_lifecycle":
                lifecycle_schema = schema
                break

        assert lifecycle_schema is not None
        action_prop = lifecycle_schema["inputSchema"]["properties"]["action"]
        assert "at_risk" in action_prop["enum"]

        # within_days parameter should exist
        props = lifecycle_schema["inputSchema"]["properties"]
        assert "within_days" in props
        assert props["within_days"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_at_risk_returns_counts(self) -> None:
        """at_risk action should return expiring and expired counts."""
        from neural_memory.mcp.lifecycle_handler import LifecycleHandler

        handler = MagicMock(spec=LifecycleHandler)
        handler._lifecycle = LifecycleHandler._lifecycle.__get__(handler)

        brain = MagicMock()
        brain.id = "test-brain"

        storage = AsyncMock()
        storage.get_brain = AsyncMock(return_value=brain)
        storage.brain_id = "test-brain"
        storage.get_expiring_memory_count = AsyncMock(return_value=5)
        storage.get_expired_memory_count = AsyncMock(return_value=2)
        storage.find_fibers = AsyncMock(return_value=[])

        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._lifecycle({"action": "at_risk", "within_days": 14})
        assert result["expiring_soon"] == 5
        assert result["already_expired"] == 2
        assert result["within_days"] == 14
        assert "at_risk_memories" in result

    @pytest.mark.asyncio
    async def test_at_risk_default_7_days(self) -> None:
        """at_risk without within_days should default to 7."""
        from neural_memory.mcp.lifecycle_handler import LifecycleHandler

        handler = MagicMock(spec=LifecycleHandler)
        handler._lifecycle = LifecycleHandler._lifecycle.__get__(handler)

        brain = MagicMock()
        brain.id = "test-brain"

        storage = AsyncMock()
        storage.get_brain = AsyncMock(return_value=brain)
        storage.brain_id = "test-brain"
        storage.get_expiring_memory_count = AsyncMock(return_value=0)
        storage.get_expired_memory_count = AsyncMock(return_value=0)
        storage.find_fibers = AsyncMock(return_value=[])

        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._lifecycle({"action": "at_risk"})
        assert result["within_days"] == 7
        storage.get_expiring_memory_count.assert_called_once_with(within_days=7)


# ═══════════════════════ 5. Feature Registry updated ═══════════════════════


# ═══════════════════════ 6. Goal conflict resolution ═══════════════════════


class TestGoalPriorityWeightedProximity:
    @pytest.mark.asyncio
    async def test_higher_priority_stronger_score(self) -> None:
        """Goal with priority 10 should produce higher proximity than priority 1."""
        from neural_memory.engine.goal_proximity import compute_goal_proximity

        neighbor = _make_neuron("shared neighbor", neuron_id="neighbor-1")

        storage = AsyncMock()

        # Both goals connect to same neighbor at hop 1
        async def mock_neighbors(
            nid: str, direction: str = "both", **kwargs: object
        ) -> list[tuple[MagicMock, MagicMock]]:
            if nid in ("goal-high", "goal-low"):
                syn = MagicMock()
                syn.weight = 0.5
                return [(neighbor, syn)]
            return []

        storage.get_neighbors = mock_neighbors

        # High-priority goal (10) should give stronger scores
        scores_high = await compute_goal_proximity(
            storage,
            ["goal-high"],
            goal_priorities={"goal-high": 10},
        )

        scores_low = await compute_goal_proximity(
            storage,
            ["goal-low"],
            goal_priorities={"goal-low": 1},
        )

        # neighbor-1 at hop 1: base=0.5, high_priority=0.5*1.0=0.5, low_priority=0.5*0.1=0.05
        assert scores_high.get("neighbor-1", 0) > scores_low.get("neighbor-1", 0)

    @pytest.mark.asyncio
    async def test_competing_goals_best_score_wins(self) -> None:
        """When two goals compete, the best weighted score should win."""
        from neural_memory.engine.goal_proximity import compute_goal_proximity

        neighbor = _make_neuron("shared", neuron_id="shared-1")

        storage = AsyncMock()

        async def mock_neighbors(
            nid: str, direction: str = "both", **kwargs: object
        ) -> list[tuple[MagicMock, MagicMock]]:
            if nid in ("g1", "g2"):
                syn = MagicMock()
                return [(neighbor, syn)]
            return []

        storage.get_neighbors = mock_neighbors

        scores = await compute_goal_proximity(
            storage,
            ["g1", "g2"],
            goal_priorities={"g1": 2, "g2": 8},
        )

        # shared-1 should have score from g2 (higher priority)
        # g2: base=0.5 * 0.8 = 0.4; g1: base=0.5 * 0.2 = 0.1
        assert scores.get("shared-1", 0) == pytest.approx(0.4, abs=0.01)

    @pytest.mark.asyncio
    async def test_default_priority_is_5(self) -> None:
        """Without explicit priorities, default priority 5 should be used."""
        from neural_memory.engine.goal_proximity import compute_goal_proximity

        storage = AsyncMock()
        storage.get_neighbors = AsyncMock(return_value=[])

        scores = await compute_goal_proximity(storage, ["g1"])
        # Goal itself at hop 0: base=1.0 * (5/10) = 0.5
        assert scores.get("g1", 0) == pytest.approx(0.5, abs=0.01)


# ═══════════════════════ 7. Temporal neighborhood MCP tool ═══════════════════════


class TestCausalMCPTool:
    @pytest.mark.asyncio
    async def test_causal_schema_exists(self) -> None:
        """nmem_causal tool schema should exist with trace/sequence actions."""
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS as TOOL_SCHEMAS

        causal_schema = None
        for schema in TOOL_SCHEMAS:
            if schema["name"] == "nmem_causal":
                causal_schema = schema
                break

        assert causal_schema is not None
        action_prop = causal_schema["inputSchema"]["properties"]["action"]
        assert "trace" in action_prop["enum"]
        assert "sequence" in action_prop["enum"]
        assert "neuron_id" in causal_schema["inputSchema"]["required"]

    @pytest.mark.asyncio
    async def test_causal_trace_missing_neuron(self) -> None:
        """trace action with non-existent neuron should return error."""
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = MagicMock(spec=GoalHandler)
        handler._causal = GoalHandler._causal.__get__(handler)

        storage = AsyncMock()
        storage.brain_id = "test"
        storage.get_neuron = AsyncMock(return_value=None)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._causal({"action": "trace", "neuron_id": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_causal_invalid_action(self) -> None:
        """Unknown action should return error."""
        from neural_memory.mcp.goal_handler import GoalHandler

        handler = MagicMock(spec=GoalHandler)
        handler._causal = GoalHandler._causal.__get__(handler)

        storage = AsyncMock()
        storage.brain_id = "test"
        neuron = _make_neuron("test content", neuron_id="n1")
        storage.get_neuron = AsyncMock(return_value=neuron)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._causal({"action": "bogus", "neuron_id": "n1"})
        assert "error" in result


# ═══════════════════════ 8. Valence recall filter ═══════════════════════


class TestValenceFilter:
    def test_valence_schema_exists(self) -> None:
        """nmem_recall schema should include valence parameter."""
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS as TOOL_SCHEMAS

        recall_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "nmem_recall")
        props = recall_schema["inputSchema"]["properties"]
        assert "valence" in props
        assert set(props["valence"]["enum"]) == {"positive", "negative", "neutral"}
