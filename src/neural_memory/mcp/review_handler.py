"""MCP handler mixin for spaced repetition review operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class ReviewHandler:
    """Mixin providing nmem_review tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _review(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_review tool calls.

        Actions:
            queue   — Get fibers due for review
            mark    — Mark a fiber review as success/failure
            schedule — Manually schedule a fiber for review
            stats   — Get review statistics
        """
        action = args.get("action", "queue")
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        try:
            if action == "queue":
                return await self._review_queue(storage, brain, args)
            elif action == "mark":
                return await self._review_mark(storage, brain, args)
            elif action == "schedule":
                return await self._review_schedule(storage, brain, args)
            elif action == "stats":
                return await self._review_stats(storage)
            else:
                return {"error": f"Unknown review action: {action}"}
        except Exception:
            logger.error("Review handler failed for action '%s'", action, exc_info=True)
            return {"error": "Review operation failed"}

    async def _review_queue(
        self,
        storage: NeuralStorage,
        brain: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the review queue."""
        from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

        limit = min(args.get("limit", 20), 100)
        engine = SpacedRepetitionEngine(storage, brain.config)
        items = await engine.get_review_queue(limit=limit)
        return {"action": "queue", "items": items, "count": len(items)}

    async def _review_mark(
        self,
        storage: NeuralStorage,
        brain: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Mark a fiber review as success or failure."""
        from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

        fiber_id = args.get("fiber_id")
        if not fiber_id:
            return {"error": "fiber_id is required for mark action"}

        success = args.get("success", True)
        engine = SpacedRepetitionEngine(storage, brain.config)
        result = await engine.process_review(fiber_id, success=success)
        result["action"] = "mark"
        return result

    async def _review_schedule(
        self,
        storage: NeuralStorage,
        brain: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Manually schedule a fiber for review."""
        from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

        fiber_id = args.get("fiber_id")
        if not fiber_id:
            return {"error": "fiber_id is required for schedule action"}

        # Verify fiber exists
        fiber = await storage.get_fiber(fiber_id)
        if not fiber:
            return {"error": f"Fiber {fiber_id} not found"}

        engine = SpacedRepetitionEngine(storage, brain.config)
        schedule = await engine.auto_schedule_fiber(fiber_id, brain.id)
        if schedule is None:
            return {"action": "schedule", "message": "Fiber already scheduled for review"}

        return {
            "action": "schedule",
            "fiber_id": fiber_id,
            "box": schedule.box,
            "next_review": schedule.next_review.isoformat() if schedule.next_review else None,
            "message": "Fiber scheduled for review",
        }

    async def _review_stats(self, storage: NeuralStorage) -> dict[str, Any]:
        """Get review statistics."""
        result: dict[str, Any] = dict(await storage.get_review_stats())
        result["action"] = "stats"
        return result
