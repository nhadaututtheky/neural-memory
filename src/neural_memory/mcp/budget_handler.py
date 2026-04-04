"""MCP handler mixin for token budget analysis and optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.constants import MAX_TOKEN_BUDGET
from neural_memory.mcp.tool_handler_utils import _require_brain_id
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class BudgetHandler:
    """Mixin providing token budget analysis handler."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _budget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Token budget analysis for recall context allocation."""
        action = args.get("action", "")
        if action not in ("estimate", "analyze", "optimize"):
            return {
                "error": f"Invalid action: {action!r}. Must be 'estimate', 'analyze', or 'optimize'."
            }

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for budget analysis")
            return {"error": "No brain configured"}

        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        max_tokens = min(
            int(args.get("max_tokens", self.config.budget.default_tokens)), MAX_TOKEN_BUDGET
        )

        from neural_memory.engine.token_budget import (
            BudgetConfig,
            allocate_budget,
            compute_token_costs,
            format_budget_report,
        )

        budget_cfg = BudgetConfig(
            system_overhead_tokens=self.config.budget.system_overhead,
            per_fiber_overhead=self.config.budget.per_fiber_overhead,
        )

        if action == "estimate":
            query = args.get("query", "")
            if not query or not isinstance(query, str):
                return {"error": "query is required for action='estimate'"}

            try:
                pipeline = ReflexPipeline(storage, brain.config)
                result = await pipeline.query(
                    query=query,
                    depth=DepthLevel(0),  # Shallow — just activation, no heavy traversal
                    max_tokens=max_tokens,
                    reference_time=utcnow(),
                )
            except Exception:
                logger.error("Budget estimate pipeline failed", exc_info=True)
                return {"error": "Failed to run retrieval pipeline for estimate"}

            # Fetch fiber objects for the matched fibers
            fibers: list[Any] = []
            for fid in result.fibers_matched or []:
                fiber = await storage.get_fiber(fid)
                if fiber:
                    fibers.append(fiber)

            # Build activations map from co_activations (RetrievalResult has no .activations field)
            from neural_memory.engine.activation import ActivationResult

            estimate_activations: dict[str, ActivationResult] = {}
            for co in result.co_activations:
                for nid in co.neuron_ids:
                    if nid not in estimate_activations:
                        estimate_activations[nid] = ActivationResult(
                            neuron_id=nid,
                            activation_level=co.binding_strength,
                            hop_distance=0,
                            path=[nid],
                            source_anchor=nid,
                        )

            costs = compute_token_costs(fibers, estimate_activations, budget_cfg)
            allocation = allocate_budget(costs, max_tokens, budget_cfg)
            report = format_budget_report(allocation)

            return {
                "action": "estimate",
                "query": query,
                "max_tokens": max_tokens,
                "neurons_activated": result.neurons_activated,
                "fibers_found": len(fibers),
                "budget": report,
                "would_drop": allocation.fibers_dropped,
                "confidence": result.confidence,
            }

        elif action == "analyze":
            # Profile the brain's token cost distribution by memory type
            try:
                fibers_list = await storage.get_fibers(limit=min(200, 1000))
            except Exception:
                logger.error("Budget analyze: get_fibers failed", exc_info=True)
                return {"error": "Failed to list fibers for analysis"}

            if not fibers_list:
                return {
                    "action": "analyze",
                    "brain": brain_id,
                    "total_fibers": 0,
                    "message": "No fibers found in brain",
                }

            # Compute costs with uniform activation score (no query context)
            dummy_activations: dict[str, Any] = {}
            costs = compute_token_costs(fibers_list, dummy_activations, budget_cfg)

            if not costs:
                return {
                    "action": "analyze",
                    "brain": brain_id,
                    "total_fibers": len(fibers_list),
                    "message": "No cost data computed",
                }

            total_tokens = sum(c.total_tokens for c in costs)
            avg_tokens = total_tokens / len(costs) if costs else 0
            max_cost = max(c.total_tokens for c in costs)
            min_cost = min(c.total_tokens for c in costs)

            # Top 5 most expensive fibers
            top_expensive = sorted(costs, key=lambda c: c.total_tokens, reverse=True)[:5]

            return {
                "action": "analyze",
                "brain": brain_id,
                "total_fibers": len(fibers_list),
                "total_tokens_all_fibers": total_tokens,
                "avg_tokens_per_fiber": round(avg_tokens, 1),
                "max_fiber_tokens": max_cost,
                "min_fiber_tokens": min_cost,
                "estimated_full_recall_tokens": total_tokens + budget_cfg.system_overhead_tokens,
                "would_fit_in_4k": sum(1 for c in costs if c.total_tokens <= 4000),
                "top_expensive_fibers": [
                    {"fiber_id": c.fiber_id, "total_tokens": c.total_tokens} for c in top_expensive
                ],
            }

        else:  # optimize
            # Find low-value-per-token fibers that are compression candidates
            try:
                fibers_list = await storage.get_fibers(limit=min(200, 1000))
            except Exception:
                logger.error("Budget optimize: get_fibers failed", exc_info=True)
                return {"error": "Failed to list fibers for optimization"}

            if not fibers_list:
                return {
                    "action": "optimize",
                    "recommendations": [],
                    "message": "No fibers found",
                }

            dummy_activations = {}
            costs = compute_token_costs(fibers_list, dummy_activations, budget_cfg)

            # Fibers with zero or very low value_per_token and high cost are candidates
            candidates = [
                c
                for c in costs
                if c.total_tokens > 100  # Only fibers large enough to matter
            ]
            # Sort by worst efficiency (high cost, low value) — cost > 100, no activation context
            candidates_sorted = sorted(candidates, key=lambda c: c.total_tokens, reverse=True)[:10]

            recommendations = []
            for c in candidates_sorted:
                savings_estimate = max(0, c.total_tokens - budget_cfg.per_fiber_overhead - 20)
                recommendations.append(
                    {
                        "fiber_id": c.fiber_id,
                        "current_tokens": c.total_tokens,
                        "estimated_savings_if_compressed": savings_estimate,
                        "suggestion": "Consider running nmem_consolidate to compress this fiber's content into a summary.",
                    }
                )

            return {
                "action": "optimize",
                "fibers_analyzed": len(costs),
                "compression_candidates": len(recommendations),
                "recommendations": recommendations,
                "tip": "Run nmem_consolidate with strategy='mature' to auto-summarize old fibers.",
            }
