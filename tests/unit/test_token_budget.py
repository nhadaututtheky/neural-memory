"""Tests for token budget management module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.token_budget import (
    BudgetAllocation,
    BudgetConfig,
    TokenCost,
    allocate_budget,
    compute_token_costs,
    estimate_fiber_tokens,
    format_budget_report,
)

# ─────────────────────── Helpers ───────────────────────


def _make_fiber(
    fiber_id: str,
    summary: str | None = None,
    salience: float = 0.0,
    conductivity: float = 1.0,
    neuron_ids: set[str] | None = None,
    anchor_neuron_id: str = "",
) -> Any:
    """Create a minimal mock Fiber object."""
    fiber = MagicMock()
    fiber.id = fiber_id
    fiber.summary = summary
    fiber.salience = salience
    fiber.conductivity = conductivity
    fiber.neuron_ids = neuron_ids or {anchor_neuron_id or fiber_id}
    fiber.anchor_neuron_id = anchor_neuron_id or fiber_id
    fiber.metadata = {}
    return fiber


def _make_activation(neuron_id: str, level: float) -> ActivationResult:
    """Create a minimal ActivationResult."""
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=0,
        path=[neuron_id],
        source_anchor=neuron_id,
    )


# ─────────────────────── TestEstimateFiberTokens ───────────────────────


class TestEstimateFiberTokens:
    """Tests for estimate_fiber_tokens()."""

    def test_basic_content(self) -> None:
        tokens = estimate_fiber_tokens("hello world test")
        # 3 words * 1.3 = 3 (rounded)
        assert tokens >= 3

    def test_empty_content(self) -> None:
        tokens = estimate_fiber_tokens("")
        assert tokens == 0

    def test_prefers_summary_over_anchor(self) -> None:
        tokens_with_summary = estimate_fiber_tokens(
            content="long content with many words here",
            summary="short",
        )
        tokens_without_summary = estimate_fiber_tokens(
            content="long content with many words here",
        )
        # Summary is shorter, so tokens should be fewer
        assert tokens_with_summary < tokens_without_summary

    def test_prefers_anchor_over_content_when_no_summary(self) -> None:
        tokens = estimate_fiber_tokens(
            content="fallback",
            anchor_content="anchor text is used",
        )
        # anchor_content preferred when no summary
        assert tokens >= 3  # at least 3 words worth

    def test_long_content(self) -> None:
        long_text = " ".join(["word"] * 100)
        tokens = estimate_fiber_tokens(long_text)
        assert tokens >= 100  # 100 words * 1.3 ratio

    def test_returns_at_least_one_for_nonempty(self) -> None:
        tokens = estimate_fiber_tokens("x")
        assert tokens >= 1

    def test_with_only_summary(self) -> None:
        tokens = estimate_fiber_tokens("", summary="summary only")
        assert tokens >= 1


# ─────────────────────── TestComputeTokenCosts ───────────────────────


class TestComputeTokenCosts:
    """Tests for compute_token_costs()."""

    def test_basic_batch(self) -> None:
        fibers = [
            _make_fiber("f1", summary="short summary"),
            _make_fiber("f2", summary="a longer summary with more content here"),
        ]
        activations = {
            "f1": _make_activation("f1", 0.8),
            "f2": _make_activation("f2", 0.3),
        }
        costs = compute_token_costs(fibers, activations)
        assert len(costs) == 2
        ids = {c.fiber_id for c in costs}
        assert ids == {"f1", "f2"}

    def test_cost_fields_populated(self) -> None:
        fiber = _make_fiber("f1", summary="test summary content")
        activations = {"f1": _make_activation("f1", 0.5)}
        costs = compute_token_costs([fiber], activations)
        assert len(costs) == 1
        c = costs[0]
        assert c.fiber_id == "f1"
        assert c.content_tokens >= 0
        assert c.metadata_tokens >= 0
        assert c.total_tokens >= 1
        assert c.value_score >= 0.0
        assert c.value_per_token >= 0.0

    def test_value_per_token_calculation(self) -> None:
        fiber = _make_fiber("f1", summary="word " * 10)
        activations = {"f1": _make_activation("f1", 1.0)}
        costs = compute_token_costs([fiber], activations)
        assert len(costs) == 1
        c = costs[0]
        assert c.value_per_token == pytest.approx(c.value_score / c.total_tokens, rel=1e-3)

    def test_empty_fibers(self) -> None:
        costs = compute_token_costs([], {})
        assert costs == []

    def test_no_activation_for_fiber(self) -> None:
        fiber = _make_fiber("f1", summary="test")
        costs = compute_token_costs([fiber], {})
        assert len(costs) == 1
        c = costs[0]
        # Value score comes from salience + conductivity only (both near 0 + 0.05)
        assert c.value_score >= 0.0

    def test_max_fibers_considered(self) -> None:
        fibers = [_make_fiber(f"f{i}") for i in range(100)]
        config = BudgetConfig(max_fibers_considered=10)
        costs = compute_token_costs(fibers, {}, config)
        assert len(costs) == 10

    def test_respects_min_fiber_tokens(self) -> None:
        fiber = _make_fiber("f1", summary="")
        config = BudgetConfig(min_fiber_tokens=20)
        costs = compute_token_costs([fiber], {}, config)
        assert costs[0].total_tokens >= 20

    def test_uses_anchor_activation(self) -> None:
        fiber = _make_fiber("f1", anchor_neuron_id="anchor1")
        activations = {"anchor1": _make_activation("anchor1", 0.9)}
        costs = compute_token_costs([fiber], activations)
        assert costs[0].value_score >= 0.9  # activation + small salience/conductivity bonus


# ─────────────────────── TestAllocateBudget ───────────────────────


class TestAllocateBudget:
    """Tests for allocate_budget()."""

    def _make_cost(self, fiber_id: str, total_tokens: int, value_score: float) -> TokenCost:
        vpt = value_score / total_tokens if total_tokens > 0 else 0.0
        return TokenCost(
            fiber_id=fiber_id,
            content_tokens=total_tokens - 10,
            metadata_tokens=10,
            total_tokens=total_tokens,
            value_score=value_score,
            value_per_token=vpt,
        )

    def test_selects_highest_value_per_token(self) -> None:
        # f1: 0.8 / 100 = 0.008 vpt
        # f2: 0.1 / 10 = 0.01 vpt (better efficiency)
        costs = [
            self._make_cost("f1", 100, 0.8),
            self._make_cost("f2", 10, 0.1),
        ]
        allocation = allocate_budget(costs, max_tokens=200)
        selected_ids = {c.fiber_id for c in allocation.selected}
        assert "f2" in selected_ids  # More efficient

    def test_budget_exhaustion_drops_fibers(self) -> None:
        costs = [self._make_cost(f"f{i}", 200, 0.5) for i in range(10)]
        allocation = allocate_budget(costs, max_tokens=500)
        # Budget = 500 - 50 overhead = 450 → fits 2 fibers (400 tokens)
        assert len(allocation.selected) <= 3
        assert allocation.fibers_dropped > 0

    def test_empty_costs(self) -> None:
        allocation = allocate_budget([], max_tokens=1000)
        assert allocation.selected == []
        assert allocation.fibers_dropped == 0
        assert allocation.total_tokens_used == 0

    def test_zero_budget(self) -> None:
        costs = [self._make_cost("f1", 50, 0.5)]
        allocation = allocate_budget(costs, max_tokens=0)
        assert allocation.selected == []
        assert allocation.fibers_dropped == 1

    def test_single_fiber_fits(self) -> None:
        costs = [self._make_cost("f1", 50, 0.5)]
        allocation = allocate_budget(costs, max_tokens=1000)
        assert len(allocation.selected) == 1
        assert allocation.fibers_dropped == 0

    def test_utilization_calculated(self) -> None:
        costs = [self._make_cost("f1", 100, 0.5)]
        allocation = allocate_budget(costs, max_tokens=200)
        assert 0.0 <= allocation.budget_utilization <= 1.0

    def test_tokens_used_matches_selected(self) -> None:
        costs = [
            self._make_cost("f1", 50, 0.5),
            self._make_cost("f2", 60, 0.4),
        ]
        allocation = allocate_budget(costs, max_tokens=500)
        expected_used = sum(c.total_tokens for c in allocation.selected)
        assert allocation.total_tokens_used == expected_used

    def test_all_fibers_fit(self) -> None:
        costs = [self._make_cost(f"f{i}", 10, 0.5) for i in range(5)]
        allocation = allocate_budget(costs, max_tokens=10000)
        assert len(allocation.selected) == 5
        assert allocation.fibers_dropped == 0

    def test_custom_config_overhead(self) -> None:
        config = BudgetConfig(system_overhead_tokens=100)
        costs = [self._make_cost("f1", 50, 0.5)]
        allocation = allocate_budget(costs, max_tokens=200, config=config)
        # Effective budget = 200 - 100 = 100 → f1 (50 tokens) fits
        assert len(allocation.selected) == 1


# ─────────────────────── TestBudgetAllocation ───────────────────────


class TestBudgetAllocation:
    """Tests for BudgetAllocation dataclass."""

    def test_frozen(self) -> None:
        alloc = BudgetAllocation(
            selected=[],
            total_tokens_used=0,
            total_tokens_budget=1000,
            tokens_remaining=1000,
            fibers_dropped=0,
            budget_utilization=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            alloc.fibers_dropped = 5  # type: ignore[misc]

    def test_utilization_range(self) -> None:
        costs = [
            TokenCost(
                fiber_id="f1",
                content_tokens=40,
                metadata_tokens=10,
                total_tokens=50,
                value_score=0.5,
                value_per_token=0.01,
            )
        ]
        allocation = allocate_budget(costs, max_tokens=1000)
        assert 0.0 <= allocation.budget_utilization <= 1.0

    def test_fibers_dropped_count(self) -> None:
        total = 10
        costs = [
            TokenCost(
                fiber_id=f"f{i}",
                content_tokens=200,
                metadata_tokens=10,
                total_tokens=210,
                value_score=0.5,
                value_per_token=0.5 / 210,
            )
            for i in range(total)
        ]
        allocation = allocate_budget(costs, max_tokens=500)
        assert allocation.fibers_dropped + len(allocation.selected) == total


# ─────────────────────── TestFormatBudgetReport ───────────────────────


class TestFormatBudgetReport:
    """Tests for format_budget_report()."""

    def test_returns_dict(self) -> None:
        allocation = BudgetAllocation(
            selected=[],
            total_tokens_used=0,
            total_tokens_budget=1000,
            tokens_remaining=1000,
            fibers_dropped=0,
            budget_utilization=0.0,
        )
        report = format_budget_report(allocation)
        assert isinstance(report, dict)

    def test_required_keys(self) -> None:
        allocation = BudgetAllocation(
            selected=[],
            total_tokens_used=100,
            total_tokens_budget=1000,
            tokens_remaining=900,
            fibers_dropped=2,
            budget_utilization=0.1,
        )
        report = format_budget_report(allocation)
        assert "fibers_selected" in report
        assert "fibers_dropped" in report
        assert "total_tokens_used" in report
        assert "total_tokens_budget" in report
        assert "tokens_remaining" in report
        assert "budget_utilization" in report
        assert "top_costs" in report

    def test_top_costs_capped_at_5(self) -> None:
        selected = [
            TokenCost(
                fiber_id=f"f{i}",
                content_tokens=10,
                metadata_tokens=5,
                total_tokens=15,
                value_score=float(i),
                value_per_token=float(i) / 15,
            )
            for i in range(10)
        ]
        allocation = BudgetAllocation(
            selected=selected,
            total_tokens_used=150,
            total_tokens_budget=500,
            tokens_remaining=350,
            fibers_dropped=0,
            budget_utilization=0.3,
        )
        report = format_budget_report(allocation)
        assert len(report["top_costs"]) == 5

    def test_values_are_serializable(self) -> None:
        import json

        allocation = BudgetAllocation(
            selected=[],
            total_tokens_used=0,
            total_tokens_budget=1000,
            tokens_remaining=1000,
            fibers_dropped=0,
            budget_utilization=0.0,
        )
        report = format_budget_report(allocation)
        # Should not raise
        json.dumps(report)


# ─────────────────────── TestFormatContextBudgeted ───────────────────────


class TestFormatContextBudgeted:
    """Integration tests for format_context_budgeted()."""

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.get_neurons_batch = AsyncMock(return_value={})
        return storage

    @pytest.mark.asyncio
    async def test_returns_tuple_of_three(self, mock_storage: AsyncMock) -> None:
        from neural_memory.engine.retrieval_context import format_context_budgeted

        result = await format_context_budgeted(
            storage=mock_storage,
            activations={},
            fibers=[],
            max_tokens=1000,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        context, tokens, allocation = result
        assert isinstance(context, str)
        assert isinstance(tokens, int)

    @pytest.mark.asyncio
    async def test_empty_fibers_returns_empty(self, mock_storage: AsyncMock) -> None:
        from neural_memory.engine.retrieval_context import format_context_budgeted

        context, tokens, allocation = await format_context_budgeted(
            storage=mock_storage,
            activations={},
            fibers=[],
            max_tokens=1000,
        )
        assert allocation.fibers_dropped == 0
        assert len(allocation.selected) == 0

    @pytest.mark.asyncio
    async def test_budget_allocation_returned(self, mock_storage: AsyncMock) -> None:
        from neural_memory.engine.retrieval_context import format_context_budgeted
        from neural_memory.engine.token_budget import BudgetAllocation
        from neural_memory.utils.timeutils import utcnow

        fiber = _make_fiber("f1", summary="test memory content here")
        # Ensure created_at is a real datetime so compress_for_recall works
        fiber.created_at = utcnow()
        activations = {"f1": _make_activation("f1", 0.8)}
        mock_storage.get_neurons_batch = AsyncMock(return_value={})

        context, tokens, allocation = await format_context_budgeted(
            storage=mock_storage,
            activations=activations,
            fibers=[fiber],
            max_tokens=1000,
        )
        assert isinstance(allocation, BudgetAllocation)
        assert allocation.total_tokens_budget == 1000


# ─────────────────────── TestNmemBudget ───────────────────────


class TestNmemBudget:
    """Tests for the nmem_budget MCP tool handler."""

    @pytest.fixture
    def server(self) -> Any:
        """Create minimal MCPServer mock."""
        from neural_memory.mcp.server import MCPServer
        from neural_memory.unified_config import BudgetRetrievalConfig, ToolTierConfig

        with patch("neural_memory.mcp.server.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                current_brain="test",
                tool_tier=ToolTierConfig(tier="full"),
                budget=BudgetRetrievalConfig(),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_invalid_action(self, server: Any) -> None:
        result = await server._budget({"action": "invalid"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_action(self, server: Any) -> None:
        result = await server._budget({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_estimate_requires_query(self, server: Any) -> None:
        storage_mock = AsyncMock()
        storage_mock.brain_id = "brain-1"
        storage_mock.get_brain = AsyncMock(
            return_value=MagicMock(
                id="brain-1",
                config=MagicMock(),
            )
        )
        with patch.object(server, "get_storage", return_value=storage_mock):
            result = await server._budget({"action": "estimate"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_action_no_fibers(self, server: Any) -> None:
        storage_mock = AsyncMock()
        storage_mock.brain_id = "brain-1"
        storage_mock.get_brain = AsyncMock(
            return_value=MagicMock(
                id="brain-1",
                config=MagicMock(),
            )
        )
        storage_mock.list_fibers = AsyncMock(return_value=[])
        with patch.object(server, "get_storage", return_value=storage_mock):
            result = await server._budget({"action": "analyze"})
        assert result.get("total_fibers") == 0

    @pytest.mark.asyncio
    async def test_optimize_action_no_fibers(self, server: Any) -> None:
        storage_mock = AsyncMock()
        storage_mock.brain_id = "brain-1"
        storage_mock.get_brain = AsyncMock(
            return_value=MagicMock(
                id="brain-1",
                config=MagicMock(),
            )
        )
        storage_mock.list_fibers = AsyncMock(return_value=[])
        with patch.object(server, "get_storage", return_value=storage_mock):
            result = await server._budget({"action": "optimize"})
        assert result.get("recommendations") == []


# ─────────────────────── TestNmemRecallWithBudget ───────────────────────


class TestNmemRecallWithBudget:
    """Tests for nmem_recall with recall_token_budget parameter."""

    @pytest.fixture
    def server(self) -> Any:
        from neural_memory.mcp.server import MCPServer
        from neural_memory.unified_config import (
            BudgetRetrievalConfig,
            EncryptionConfig,
            ToolTierConfig,
        )

        with patch("neural_memory.mcp.server.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                current_brain="test",
                tool_tier=ToolTierConfig(tier="full"),
                budget=BudgetRetrievalConfig(),
                auto=MagicMock(enabled=False),
                encryption=EncryptionConfig(enabled=False),
                data_dir=MagicMock(),
            )
            return MCPServer()

    @pytest.mark.asyncio
    async def test_recall_without_budget_param(self, server: Any) -> None:
        """Standard recall with no recall_token_budget should work normally."""
        from neural_memory.engine.retrieval_types import DepthLevel

        storage_mock = AsyncMock()
        storage_mock.brain_id = "brain-1"
        storage_mock.get_brain = AsyncMock(
            return_value=MagicMock(
                id="brain-1",
                config=MagicMock(
                    activation_threshold=0.2,
                    max_spread_hops=4,
                ),
            )
        )

        mock_result = MagicMock()
        mock_result.confidence = 0.8
        mock_result.neurons_activated = 3
        mock_result.fibers_matched = []
        mock_result.depth_used = DepthLevel(1)
        mock_result.context = "Test context"
        mock_result.tokens_used = 50
        mock_result.score_breakdown = None
        mock_result.metadata = {}
        mock_result.co_activations = []

        with (
            patch.object(server, "get_storage", return_value=storage_mock),
            patch.object(server, "_get_active_session", return_value=None),
            patch.object(server, "_check_surface_depth", return_value=(None, None)),
            patch.object(server, "_passive_capture", return_value=None),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_check_maintenance", return_value=None),
            patch.object(server, "_get_maintenance_hint", return_value=None),
            patch.object(server, "get_update_hint", return_value=None),
            patch.object(server, "_check_onboarding", return_value=None),
            patch.object(server, "_check_cross_language_hint", return_value=None),
            patch.object(server, "_surface_pending_alerts", return_value={}),
            patch.object(server, "_record_tool_action"),
            patch("neural_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            result = await server._recall({"query": "test query"})

        assert "answer" in result
        assert "budget_stats" not in result  # No budget param = no budget stats


# ─────────────────────── TestBudgetConfig ───────────────────────


class TestBudgetConfig:
    """Tests for BudgetConfig defaults and frozen behavior."""

    def test_defaults(self) -> None:
        cfg = BudgetConfig()
        assert cfg.system_overhead_tokens == 50
        assert cfg.per_fiber_overhead == 15
        assert cfg.min_fiber_tokens == 10
        assert cfg.max_fibers_considered == 50

    def test_frozen(self) -> None:
        cfg = BudgetConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.system_overhead_tokens = 100  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = BudgetConfig(system_overhead_tokens=100, per_fiber_overhead=20)
        assert cfg.system_overhead_tokens == 100
        assert cfg.per_fiber_overhead == 20
