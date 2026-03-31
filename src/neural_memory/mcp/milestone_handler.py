"""MCP handler mixin for brain milestone analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class MilestoneHandler:
    """Mixin providing nmem_milestone tool handler for MCPServer.

    Actions:
        check    — Check if a new milestone was reached and auto-record it
        progress — Show progress toward next milestone
        history  — List all recorded milestones
        report   — Force-generate a growth report for current state
    """

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _milestone(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_milestone tool calls."""
        action = args.get("action", "check")
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        try:
            if action == "check":
                return await self._milestone_check(storage, brain)
            elif action == "progress":
                return await self._milestone_progress(storage, brain)
            elif action == "history":
                return await self._milestone_history(storage, brain)
            elif action == "report":
                return await self._milestone_report(storage, brain)
            else:
                return {"error": f"Unknown milestone action: {action}"}
        except Exception:
            logger.error("Milestone handler failed for action '%s'", action, exc_info=True)
            return {"error": "Milestone analysis failed"}

    async def _milestone_check(
        self,
        storage: NeuralStorage,
        brain: Any,
    ) -> dict[str, Any]:
        """Check for new milestones and auto-record."""
        from neural_memory.engine.milestone import MilestoneEngine

        engine = MilestoneEngine(storage)
        report = await engine.check_and_record(brain.id)

        if report is None:
            # No new milestone — return progress instead
            progress = await engine.get_progress(brain.id)
            return {
                "action": "check",
                "new_milestone": False,
                **progress,
            }

        return {
            "action": "check",
            "new_milestone": True,
            "milestone": report.snapshot.threshold,
            "title": report.title,
            "grade": report.snapshot.grade,
            "purity_score": report.snapshot.purity_score,
            "achievements": list(report.achievements),
            "markdown": report.markdown,
        }

    async def _milestone_progress(
        self,
        storage: NeuralStorage,
        brain: Any,
    ) -> dict[str, Any]:
        """Show progress toward next milestone."""
        from neural_memory.engine.milestone import MilestoneEngine

        engine = MilestoneEngine(storage)
        progress = await engine.get_progress(brain.id)
        return {"action": "progress", **progress}

    async def _milestone_history(
        self,
        storage: NeuralStorage,
        brain: Any,
    ) -> dict[str, Any]:
        """List all recorded milestones."""
        from neural_memory.engine.milestone import MilestoneEngine

        engine = MilestoneEngine(storage)
        history = await engine.get_history(brain.id)

        if not history:
            return {
                "action": "history",
                "milestones": [],
                "message": "No milestones recorded yet.",
            }

        return {
            "action": "history",
            "milestones": history,
            "total": len(history),
        }

    async def _milestone_report(
        self,
        storage: NeuralStorage,
        brain: Any,
    ) -> dict[str, Any]:
        """Force-generate a growth report for current state."""
        from neural_memory.engine.milestone import MilestoneEngine

        engine = MilestoneEngine(storage)
        report = await engine.generate_report(brain.id)

        if report is None:
            return {
                "action": "report",
                "error": "Brain is empty — no report to generate.",
            }

        return {
            "action": "report",
            "title": report.title,
            "neuron_count": report.snapshot.neuron_count,
            "grade": report.snapshot.grade,
            "purity_score": report.snapshot.purity_score,
            "growth_velocity": report.growth_velocity,
            "health_delta": report.health_delta,
            "achievements": list(report.achievements),
            "markdown": report.markdown,
        }
