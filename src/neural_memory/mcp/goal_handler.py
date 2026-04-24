"""Goal management and causal traversal handler for MCP server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id
from neural_memory.utils.timeutils import ensure_naive_utc

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


def _clamp_int(raw: Any, *, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return default


def _clamp_float(raw: Any, *, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(raw)))
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return ensure_naive_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _fiber_summary(fiber: Fiber) -> dict[str, Any]:
    return {
        "fiber_id": fiber.id,
        "summary": (fiber.summary or "")[:200],
        "time_start": fiber.time_start.isoformat() if fiber.time_start else None,
        "time_end": fiber.time_end.isoformat() if fiber.time_end else None,
        "neuron_count": len(fiber.neuron_ids),
        "salience": round(fiber.salience, 3),
    }


_MAX_GOAL_LEN = 500
_MAX_KEYWORDS = 20
_MAX_CAUSAL_DEPTH = 10
_MAX_TEMPORAL_LIMIT = 200
_DEFAULT_TEMPORAL_LIMIT = 50
_DEFAULT_WINDOW_HOURS = 24.0
_MAX_WINDOW_HOURS = 8760.0  # 1 year


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
        """Trace causal chains, temporal sequences, and time-window fiber queries."""
        storage = await self.get_storage()
        _require_brain_id(storage)

        action = args.get("action", "trace")

        if action == "trace":
            return await self._causal_trace(storage, args)
        if action == "sequence":
            return await self._causal_sequence(storage, args)
        if action == "temporal_range":
            return await self._causal_temporal_range(storage, args)
        if action == "temporal_neighborhood":
            return await self._causal_temporal_neighborhood(storage, args)

        return {
            "error": (
                f"Unknown action: {action}. "
                "Valid: trace, sequence, temporal_range, temporal_neighborhood"
            )
        }

    async def _causal_trace(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        from neural_memory.engine.causal_traversal import trace_causal_chain

        neuron_id = (args.get("neuron_id") or "").strip()
        if not neuron_id:
            return {"error": "neuron_id is required for trace"}

        neuron = await storage.get_neuron(neuron_id)
        if neuron is None:
            return {"error": f"Neuron not found: {neuron_id}"}

        direction = args.get("direction", "causes")
        if direction not in ("causes", "effects"):
            return {"error": f"Invalid direction: {direction}. Must be 'causes' or 'effects'."}

        max_depth = _clamp_int(args.get("max_depth"), default=5, lo=1, hi=_MAX_CAUSAL_DEPTH)

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

    async def _causal_sequence(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        from neural_memory.engine.causal_traversal import trace_event_sequence

        neuron_id = (args.get("neuron_id") or "").strip()
        if not neuron_id:
            return {"error": "neuron_id is required for sequence"}

        neuron = await storage.get_neuron(neuron_id)
        if neuron is None:
            return {"error": f"Neuron not found: {neuron_id}"}

        direction = args.get("direction", "forward")
        if direction not in ("forward", "backward"):
            return {"error": f"Invalid direction: {direction}. Must be 'forward' or 'backward'."}

        max_depth = _clamp_int(args.get("max_depth"), default=5, lo=1, hi=_MAX_CAUSAL_DEPTH)

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

    async def _causal_temporal_range(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        from neural_memory.engine.causal_traversal import query_temporal_range

        start_raw = args.get("start")
        end_raw = args.get("end")
        if not start_raw or not end_raw:
            return {"error": "start and end (ISO-8601) are required for temporal_range"}

        start = _parse_iso_datetime(start_raw)
        end = _parse_iso_datetime(end_raw)
        if start is None or end is None:
            return {"error": "Invalid ISO-8601 datetime in start/end"}
        if start > end:
            return {"error": "start must be <= end"}

        limit = _clamp_int(
            args.get("limit"),
            default=_DEFAULT_TEMPORAL_LIMIT,
            lo=1,
            hi=_MAX_TEMPORAL_LIMIT,
        )

        fibers = await query_temporal_range(storage, start, end, limit=limit)

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "fibers": [_fiber_summary(f) for f in fibers],
            "count": len(fibers),
        }

    async def _causal_temporal_neighborhood(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        from neural_memory.engine.causal_traversal import query_temporal_neighborhood

        fiber_id = (args.get("fiber_id") or "").strip()
        if not fiber_id:
            return {"error": "fiber_id is required for temporal_neighborhood"}

        anchor = await storage.get_fiber(fiber_id)
        if anchor is None:
            return {"error": f"Fiber not found: {fiber_id}"}

        window_hours = _clamp_float(
            args.get("window_hours"),
            default=_DEFAULT_WINDOW_HOURS,
            lo=0.1,
            hi=_MAX_WINDOW_HOURS,
        )
        limit = _clamp_int(
            args.get("limit"),
            default=10,
            lo=1,
            hi=_MAX_TEMPORAL_LIMIT,
        )

        fibers = await query_temporal_neighborhood(
            storage, fiber_id, window_hours=window_hours, limit=limit
        )

        return {
            "anchor_fiber_id": fiber_id,
            "anchor_time_start": anchor.time_start.isoformat() if anchor.time_start else None,
            "window_hours": window_hours,
            "fibers": [_fiber_summary(f) for f in fibers],
            "count": len(fibers),
        }
