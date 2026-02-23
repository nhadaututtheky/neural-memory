"""MCP handler mixin for memory narrative operations."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class NarrativeHandler:
    """Mixin providing nmem_narrative tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _narrative(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_narrative tool calls.

        Actions:
            timeline — Time-range narrative
            topic    — SA-driven topic narrative
            causal   — Causal chain narrative
        """
        action = args.get("action", "topic")
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        try:
            if action == "timeline":
                return await self._narrative_timeline(storage, args)
            elif action == "topic":
                return await self._narrative_topic(storage, brain, args)
            elif action == "causal":
                return await self._narrative_causal(storage, args)
            else:
                return {"error": f"Unknown narrative action: {action}"}
        except Exception:
            logger.error("Narrative handler failed for action '%s'", action, exc_info=True)
            return {"error": "Narrative generation failed"}

    async def _narrative_timeline(
        self,
        storage: NeuralStorage,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate timeline narrative."""
        from neural_memory.engine.narrative import generate_timeline_narrative

        start_str = args.get("start_date")
        end_str = args.get("end_date")
        if not start_str or not end_str:
            return {"error": "start_date and end_date are required for timeline action"}

        try:
            start_date = datetime.fromisoformat(start_str)
            end_date = datetime.fromisoformat(end_str)
        except ValueError:
            return {"error": "Invalid date format. Use ISO format (e.g., 2026-02-01)"}

        max_fibers = min(args.get("max_fibers", 20), 50)
        narrative = await generate_timeline_narrative(
            storage, start_date, end_date, max_fibers=max_fibers
        )
        return {
            "action": "timeline",
            "title": narrative.title,
            "items": len(narrative.items),
            "markdown": narrative.markdown,
        }

    async def _narrative_topic(
        self,
        storage: NeuralStorage,
        brain: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate topic narrative."""
        from neural_memory.engine.narrative import generate_topic_narrative

        topic = args.get("topic")
        if not topic:
            return {"error": "topic is required for topic action"}

        max_fibers = min(args.get("max_fibers", 20), 50)
        narrative = await generate_topic_narrative(
            storage, brain.config, topic, max_fibers=max_fibers
        )
        return {
            "action": "topic",
            "title": narrative.title,
            "items": len(narrative.items),
            "markdown": narrative.markdown,
        }

    async def _narrative_causal(
        self,
        storage: NeuralStorage,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate causal narrative."""
        from neural_memory.engine.narrative import generate_causal_narrative

        topic = args.get("topic")
        if not topic:
            return {"error": "topic is required for causal action"}

        max_depth = min(args.get("max_depth", 5), 10)
        narrative = await generate_causal_narrative(storage, topic, max_depth=max_depth)
        return {
            "action": "causal",
            "title": narrative.title,
            "items": len(narrative.items),
            "markdown": narrative.markdown,
        }
