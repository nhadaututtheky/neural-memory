"""MCP handler mixin for connection explanation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class ConnectionHandler:
    """Mixin providing nmem_explain tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _explain(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_explain tool calls.

        Find and explain the shortest path between two entities
        in the neural graph, with supporting memory evidence.
        """
        from neural_memory.engine.connection_explainer import explain_connection

        from_entity = args.get("from_entity", "").strip()
        to_entity = args.get("to_entity", "").strip()

        if not from_entity or not to_entity:
            return {"error": "from_entity and to_entity are required"}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for explain")
            return {"error": "No brain configured"}

        try:
            max_hops = min(int(args.get("max_hops", 6)), 10)
        except (TypeError, ValueError):
            max_hops = 6

        try:
            result = await explain_connection(storage, from_entity, to_entity, max_hops=max_hops)
            return {
                "found": result.found,
                "from_entity": result.from_entity,
                "to_entity": result.to_entity,
                "total_hops": result.total_hops,
                "avg_weight": result.avg_weight,
                "steps": [
                    {
                        "content": s.content,
                        "synapse_type": s.synapse_type,
                        "weight": s.weight,
                        "evidence": list(s.evidence),
                    }
                    for s in result.steps
                ],
                "markdown": result.markdown,
            }
        except Exception:
            logger.error("Connection explain failed", exc_info=True)
            return {"error": "Connection explanation failed"}
