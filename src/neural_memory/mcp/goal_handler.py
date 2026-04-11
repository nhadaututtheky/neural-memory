"""Goal management and causal traversal handler for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_GOAL_LEN = 500
_MAX_KEYWORDS = 20
_MAX_CAUSAL_DEPTH = 10


class GoalHandler:
    """Mixin: goal neuron CRUD + causal traversal tool handlers."""

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def _goal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle goal management actions."""
        action = args.get("action", "list")

        if action == "create":
            return await self._goal_create(args)
        elif action == "list":
            return await self._goal_list(args)
        elif action == "subgoals":
            return await self._goal_subgoals(args)
        elif action in ("activate", "pause", "complete"):
            return await self._goal_update_state(args, action)
        return {"error": f"Unknown goal action: {action}"}

    async def _goal_create(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new goal neuron, optionally as a subgoal of another."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType

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

        # Validate parent goal if provided
        parent_goal_id = (args.get("parent_goal_id") or "").strip() or None
        if parent_goal_id:
            parent = await storage.get_neuron(parent_goal_id)
            if parent is None:
                return {"error": f"Parent goal not found: {parent_goal_id}"}
            if parent.goal_state is None:
                return {"error": "Parent neuron is not a goal"}

        # Extract keywords from goal text
        keywords_raw = args.get("keywords")
        if keywords_raw and isinstance(keywords_raw, list):
            keywords = [str(k).lower().strip() for k in keywords_raw[:_MAX_KEYWORDS]]
        else:
            # Auto-extract from goal text
            from neural_memory.extraction.keywords import extract_keywords

            keywords = extract_keywords(goal_text)

        goal_metadata: dict[str, Any] = {
            "_goal_state": "active",
            "_goal_priority": priority,
            "_goal_keywords": keywords,
        }
        if parent_goal_id:
            goal_metadata["_parent_goal_id"] = parent_goal_id

        neuron = Neuron.create(
            type=NeuronType.INTENT,
            content=goal_text,
            metadata=goal_metadata,
        )

        await storage.add_neuron(neuron)

        # Create SUBGOAL_OF synapse: child -> parent
        if parent_goal_id:
            synapse = Synapse.create(
                source_id=neuron.id,
                target_id=parent_goal_id,
                type=SynapseType.SUBGOAL_OF,
                weight=0.8,
            )
            await storage.add_synapse(synapse)

        result: dict[str, Any] = {
            "goal_id": neuron.id,
            "goal": goal_text,
            "state": "active",
            "priority": priority,
            "keywords": keywords,
            "message": f"Goal created: {goal_text[:80]}",
        }
        if parent_goal_id:
            result["parent_goal_id"] = parent_goal_id
        return result

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
            entry: dict[str, Any] = {
                "goal_id": n.id,
                "goal": n.content,
                "state": gs,
                "priority": n.goal_priority,
                "keywords": n.goal_keywords,
                "created_at": n.created_at.isoformat(),
            }
            if n.parent_goal_id:
                entry["parent_goal_id"] = n.parent_goal_id
            goals.append(entry)

        # Sort by priority desc, then created_at asc (oldest first within tier)
        goals.sort(key=lambda g: (-int(g["priority"]), g["created_at"]))

        return {
            "goals": goals,
            "total": len(goals),
            "message": f"{len(goals)} goal(s) found",
        }

    async def _goal_subgoals(self, args: dict[str, Any]) -> dict[str, Any]:
        """List subgoals of a given goal, with completion hint."""
        from neural_memory.core.synapse import SynapseType

        storage = await self.get_storage()
        _require_brain_id(storage)

        goal_id = args.get("goal_id", "").strip()
        if not goal_id:
            return {"error": "goal_id is required"}

        parent = await storage.get_neuron(goal_id)
        if parent is None:
            return {"error": f"Goal not found: {goal_id}"}
        if parent.goal_state is None:
            return {"error": "Neuron is not a goal"}

        # Find SUBGOAL_OF synapses pointing to this goal (source=child, target=parent)
        synapses = await storage.get_synapses(
            target_id=goal_id,
            type=SynapseType.SUBGOAL_OF,
        )

        subgoals: list[dict[str, Any]] = []
        for syn in synapses:
            child = await storage.get_neuron(syn.source_id)
            if child is None or child.goal_state is None:
                continue
            subgoals.append(
                {
                    "goal_id": child.id,
                    "goal": child.content[:200],
                    "state": child.goal_state,
                    "priority": child.goal_priority,
                }
            )

        subgoals.sort(key=lambda g: (-g["priority"], g["goal"]))

        all_completed = len(subgoals) > 0 and all(s["state"] == "completed" for s in subgoals)
        hint = ""
        if all_completed:
            hint = f"All {len(subgoals)} subgoal(s) completed. Consider completing parent goal."

        return {
            "parent_goal_id": goal_id,
            "parent_goal": parent.content[:200],
            "parent_state": parent.goal_state,
            "subgoals": subgoals,
            "total": len(subgoals),
            "all_completed": all_completed,
            "hint": hint,
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

    # ──────── Causal / Temporal Traversal ────────

    async def _causal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Trace causal chains and temporal event sequences through the memory graph."""
        from neural_memory.engine.causal_traversal import (
            trace_causal_chain,
            trace_event_sequence,
        )

        storage = await self.get_storage()
        _require_brain_id(storage)

        action = args.get("action", "trace")
        neuron_id = (args.get("neuron_id") or "").strip()
        if not neuron_id:
            return {"error": "neuron_id is required"}

        neuron = await storage.get_neuron(neuron_id)
        if neuron is None:
            return {"error": f"Neuron not found: {neuron_id}"}

        try:
            max_depth = max(1, min(_MAX_CAUSAL_DEPTH, int(args.get("max_depth", 5))))
        except (ValueError, TypeError):
            max_depth = 5

        if action == "trace":
            direction = args.get("direction", "causes")
            if direction not in ("causes", "effects"):
                return {"error": f"Invalid direction: {direction}. Must be 'causes' or 'effects'."}

            chain = await trace_causal_chain(
                storage, neuron_id, direction=direction, max_depth=max_depth
            )

            steps = [
                {
                    "neuron_id": s.neuron_id,
                    "content": s.content[:200],
                    "synapse_type": s.synapse_type.value,
                    "weight": round(s.weight, 3),
                    "depth": s.depth,
                }
                for s in chain.steps
            ]

            return {
                "seed": neuron_id,
                "seed_content": neuron.content[:100],
                "direction": direction,
                "steps": steps,
                "total_weight": round(chain.total_weight, 4),
                "chain_length": len(steps),
            }

        elif action == "sequence":
            direction = args.get("direction", "forward")
            if direction not in ("forward", "backward"):
                return {
                    "error": f"Invalid direction: {direction}. Must be 'forward' or 'backward'."
                }

            seq = await trace_event_sequence(
                storage, neuron_id, direction=direction, max_steps=max_depth
            )

            events = [
                {
                    "neuron_id": e.neuron_id,
                    "content": e.content[:200],
                    "fiber_id": e.fiber_id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "position": e.position,
                }
                for e in seq.events
            ]

            return {
                "seed": neuron_id,
                "seed_content": neuron.content[:100],
                "direction": direction,
                "events": events,
                "sequence_length": len(events),
            }

        return {"error": f"Unknown action: {action}. Valid: trace, sequence"}
