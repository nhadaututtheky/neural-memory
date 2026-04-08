"""Goal management handler for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_GOAL_LEN = 500
_MAX_KEYWORDS = 20


class GoalHandler:
    """Mixin: goal neuron CRUD tool handlers."""

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def _goal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle goal management actions."""
        action = args.get("action", "list")

        if action == "create":
            return await self._goal_create(args)
        elif action == "list":
            return await self._goal_list(args)
        elif action in ("activate", "pause", "complete"):
            return await self._goal_update_state(args, action)
        return {"error": f"Unknown goal action: {action}"}

    async def _goal_create(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new goal neuron."""
        from neural_memory.core.neuron import Neuron, NeuronType

        storage = await self.get_storage()
        _require_brain_id(storage)

        goal_text = args.get("goal", "").strip()
        if not goal_text:
            return {"error": "goal text is required"}
        if len(goal_text) > _MAX_GOAL_LEN:
            return {"error": f"goal text too long (max {_MAX_GOAL_LEN} chars)"}

        try:
            priority = max(1, min(10, int(args.get("priority", 5))))
        except (ValueError, TypeError):
            return {"error": "priority must be an integer (1-10)"}

        # Extract keywords from goal text
        keywords_raw = args.get("keywords")
        if keywords_raw and isinstance(keywords_raw, list):
            keywords = [str(k).lower().strip() for k in keywords_raw[:_MAX_KEYWORDS]]
        else:
            # Auto-extract from goal text
            from neural_memory.extraction.keywords import extract_keywords

            keywords = extract_keywords(goal_text)

        neuron = Neuron.create(
            type=NeuronType.INTENT,
            content=goal_text,
            metadata={
                "_goal_state": "active",
                "_goal_priority": priority,
                "_goal_keywords": keywords,
            },
        )

        await storage.add_neuron(neuron)

        return {
            "goal_id": neuron.id,
            "goal": goal_text,
            "state": "active",
            "priority": priority,
            "keywords": keywords,
            "message": f"Goal created: {goal_text[:80]}",
        }

    async def _goal_list(self, args: dict[str, Any]) -> dict[str, Any]:
        """List goal neurons."""
        from neural_memory.core.neuron import NeuronType

        storage = await self.get_storage()
        _require_brain_id(storage)

        state_filter = args.get("state")  # None = all states
        try:
            limit = min(int(args.get("limit", 50)), 200)
        except (ValueError, TypeError):
            limit = 50

        intent_neurons = await storage.find_neurons(
            type=NeuronType.INTENT,
            limit=limit,
        )

        goals: list[dict[str, Any]] = []
        for n in intent_neurons:
            gs = n.goal_state
            if gs is None:
                continue  # Not a goal neuron (regular INTENT)
            if state_filter and gs != state_filter:
                continue
            goals.append(
                {
                    "goal_id": n.id,
                    "goal": n.content,
                    "state": gs,
                    "priority": n.goal_priority,
                    "keywords": n.goal_keywords,
                    "created_at": n.created_at.isoformat(),
                }
            )

        # Sort by priority desc, then created_at asc (oldest first within tier)
        goals.sort(key=lambda g: (-int(g["priority"]), g["created_at"]))

        return {
            "goals": goals,
            "total": len(goals),
            "message": f"{len(goals)} goal(s) found",
        }

    async def _goal_update_state(self, args: dict[str, Any], new_state: str) -> dict[str, Any]:
        """Update goal state (activate/pause/complete)."""
        storage = await self.get_storage()
        _require_brain_id(storage)

        goal_id = args.get("goal_id", "").strip()
        if not goal_id:
            return {"error": "goal_id is required"}

        neuron = await storage.get_neuron(goal_id)
        if neuron is None:
            return {"error": f"Goal not found: {goal_id}"}

        if neuron.goal_state is None:
            return {"error": "Neuron is not a goal"}

        updated = neuron.with_goal_state(
            state=new_state,
            priority=neuron.goal_priority,
            keywords=neuron.goal_keywords or None,
        )
        await storage.update_neuron(updated)

        return {
            "goal_id": goal_id,
            "state": new_state,
            "goal": neuron.content[:80],
            "message": f"Goal {new_state}: {neuron.content[:80]}",
        }
