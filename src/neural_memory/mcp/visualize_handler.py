"""MCP handler mixin for memory visualization — nmem_visualize tool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class VisualizeHandler:
    """Mixin providing nmem_visualize tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _visualize(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_visualize tool calls.

        Recalls memories matching the query, extracts numeric data,
        auto-detects chart type, and returns a chart specification.
        """
        query = args.get("query", "")
        if not query:
            return {"error": "query parameter is required"}

        chart_type = args.get("chart_type")
        output_format = args.get("format", "vega_lite")
        limit = min(args.get("limit", 20), 50)

        storage = await self.get_storage()
        brain_id = _require_brain_id(storage)
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        try:
            # Step 1: Recall neurons matching query (reuse existing retrieval)
            neurons = await self._visualize_recall(storage, query, limit)
            if not neurons:
                return {
                    "query": query,
                    "chart_type": "table",
                    "message": "No data found for query. Try a broader query.",
                    "data_points": [],
                }

            # Step 2: Extract data points from neurons
            from neural_memory.engine.chart_generator import (
                extract_data_points,
                generate_chart,
            )

            data_points = extract_data_points(neurons, query)
            if not data_points:
                return {
                    "query": query,
                    "chart_type": "table",
                    "message": "Found memories but no numeric data to chart. Returning content.",
                    "memories": [
                        {
                            "id": getattr(n, "id", ""),
                            "content": (getattr(n, "content", "") or "")[:200],
                            "type": getattr(n, "type", ""),
                        }
                        for n in neurons[:10]
                    ],
                }

            # Step 3+4: Detect chart type + generate spec
            chart = generate_chart(
                data_points,
                chart_type=chart_type,
                title=query,
                output_format=output_format,
            )

            result: dict[str, Any] = {
                "query": query,
                "chart_type": chart.chart_type,
                "title": chart.title,
                "data_points_count": len(chart.data_points),
                "provenance": list(chart.provenance),
            }

            if chart.vega_lite:
                result["vega_lite"] = chart.vega_lite
            if chart.markdown:
                result["markdown"] = chart.markdown
            if chart.ascii_chart:
                result["ascii"] = chart.ascii_chart

            return result

        except Exception:
            logger.error("Visualize handler failed for query '%s'", query, exc_info=True)
            return {"error": "Visualization generation failed"}

    async def _visualize_recall(
        self,
        storage: Any,
        query: str,
        limit: int,
    ) -> list[Any]:
        """Recall neurons matching query for visualization.

        Uses find_neurons with content search — simpler than full SA
        pipeline since we need raw neuron data, not ranked fibers.
        """
        try:
            neurons = await storage.find_neurons(
                content_like=query,
                limit=limit,
            )
            return neurons or []
        except Exception:
            logger.debug("Visualize recall failed, trying broader search", exc_info=True)
            try:
                # Fallback: get recent neurons
                neurons = await storage.find_neurons(limit=limit)
                return neurons or []
            except Exception:
                logger.error("Visualize recall fallback failed", exc_info=True)
                return []
