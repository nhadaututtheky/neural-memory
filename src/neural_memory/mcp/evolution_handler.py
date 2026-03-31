"""MCP handler mixin for brain evolution, suggestions, habits, versioning, and transplant."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class EvolutionHandler:
    """Mixin providing evolution, suggest, habits, version, and transplant handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _evolution(self, args: dict[str, Any]) -> dict[str, Any]:
        """Measure brain evolution dynamics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.brain_evolution import EvolutionEngine

        try:
            engine = EvolutionEngine(storage)
            evo = await engine.analyze(brain.id)
        except Exception:
            logger.error("Evolution analysis failed", exc_info=True)
            return {"error": "Evolution analysis failed"}

        result: dict[str, Any] = {
            "brain": evo.brain_name,
            "proficiency_level": evo.proficiency_level.value,
            "proficiency_index": evo.proficiency_index,
            "maturity_level": evo.maturity_level,
            "plasticity": evo.plasticity,
            "density": evo.density,
            "activity_score": evo.activity_score,
            "semantic_ratio": evo.semantic_ratio,
            "reinforcement_days": evo.reinforcement_days,
            "topology_coherence": evo.topology_coherence,
            "plasticity_index": evo.plasticity_index,
            "knowledge_density": evo.knowledge_density,
            "total_neurons": evo.total_neurons,
            "total_synapses": evo.total_synapses,
            "total_fibers": evo.total_fibers,
            "fibers_at_semantic": evo.fibers_at_semantic,
            "fibers_at_episodic": evo.fibers_at_episodic,
        }

        if evo.stage_distribution is not None:
            result["stage_distribution"] = {
                "short_term": evo.stage_distribution.short_term,
                "working": evo.stage_distribution.working,
                "episodic": evo.stage_distribution.episodic,
                "semantic": evo.stage_distribution.semantic,
                "total": evo.stage_distribution.total,
            }

        if evo.closest_to_semantic:
            result["closest_to_semantic"] = [
                {
                    "fiber_id": p.fiber_id,
                    "stage": p.stage,
                    "days_in_stage": p.days_in_stage,
                    "days_required": p.days_required,
                    "reinforcement_days": p.reinforcement_days,
                    "reinforcement_required": p.reinforcement_required,
                    "progress_pct": p.progress_pct,
                    "next_step": p.next_step,
                }
                for p in evo.closest_to_semantic
            ]

        return result

    async def _suggest(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get prefix-based autocomplete suggestions or idle neuron reinforcement hints."""
        storage = await self.get_storage()
        prefix = args.get("prefix", "")
        limit = min(args.get("limit", 5), 20)

        # When no prefix: return idle neurons that need reinforcement
        if not prefix.strip():
            return await self._suggest_idle_neurons(storage, limit)

        type_filter = None
        if "type_filter" in args:
            from neural_memory.core.neuron import NeuronType

            try:
                type_filter = NeuronType(args["type_filter"])
            except ValueError:
                return {"error": "Invalid type_filter value"}

        suggestions = await storage.suggest_neurons(
            prefix=prefix, type_filter=type_filter, limit=limit
        )
        formatted = [
            {
                "content": s["content"],
                "type": s["type"],
                "neuron_id": s["neuron_id"],
                "score": s["score"],
            }
            for s in suggestions
        ]
        return {
            "suggestions": formatted,
            "count": len(formatted),
            "tokens_used": sum(len(s["content"].split()) for s in formatted),
        }

    async def _suggest_idle_neurons(self, storage: Any, limit: int) -> dict[str, Any]:
        """Return neurons that have never been accessed — candidates for reinforcement.

        Sorted by creation age (oldest idle neurons first) to prioritize
        long-neglected knowledge.
        """
        try:
            states = await storage.get_all_neuron_states()
            idle_states = [s for s in states if s.access_frequency == 0]

            # Sort by creation time ascending (oldest first)
            idle_states.sort(key=lambda s: s.created_at or datetime.min)

            suggestions = []
            for state in idle_states[:limit]:
                neuron = await storage.get_neuron(state.neuron_id)
                if neuron is None:
                    continue
                content_preview = neuron.content[:200] if neuron.content else ""
                suggestions.append(
                    {
                        "content": content_preview,
                        "type": neuron.type.value if neuron.type else "unknown",
                        "neuron_id": neuron.id,
                        "score": 0.0,
                        "idle": True,
                    }
                )

            total_idle = len(idle_states)
            hint = ""
            if total_idle > 0:
                hint = (
                    f"{total_idle} neurons never accessed. "
                    "Recall these topics with nmem_recall to activate them "
                    "and strengthen your memory graph."
                )

            return {
                "suggestions": suggestions,
                "count": len(suggestions),
                "total_idle": total_idle,
                "mode": "idle_reinforcement",
                "hint": hint,
                "tokens_used": sum(len(s["content"].split()) for s in suggestions),
            }
        except Exception:
            logger.debug("Idle neuron suggestion failed", exc_info=True)
            return {"suggestions": [], "count": 0}

    async def _habits(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage learned workflow habits."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "list")

        if action == "suggest":
            current_action = args.get("current_action", "")
            if not current_action:
                return {"error": "current_action is required for suggest"}

            from neural_memory.engine.workflow_suggest import suggest_next_action

            suggestions = await suggest_next_action(storage, current_action, brain.config)
            return {
                "suggestions": [
                    {
                        "action": s.action_type,
                        "confidence": round(s.confidence, 4),
                        "source_habit": s.source_habit,
                        "sequential_count": s.sequential_count,
                    }
                    for s in suggestions
                ],
                "count": len(suggestions),
            }

        elif action == "list":
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
            return {
                "habits": [
                    {
                        "name": h.summary or "unnamed",
                        "steps": h.metadata.get("_workflow_actions", []),
                        "frequency": h.metadata.get("_habit_frequency", 0),
                        "confidence": h.metadata.get("_habit_confidence", 0.0),
                        "fiber_id": h.id,
                    }
                    for h in habits
                ],
                "count": len(habits),
            }

        elif action == "clear":
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
            cleared = 0
            # Delete sequentially to avoid overwhelming SQLite with concurrent writes
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1
            return {"cleared": cleared, "message": f"Cleared {cleared} learned habits"}

        return {"error": f"Unknown action: {action}"}

    async def _version(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brain version control operations."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.brain_versioning import VersioningEngine

        engine = VersioningEngine(storage)
        action = args.get("action", "list")

        if action == "create":
            name = args.get("name")
            if not name:
                return {"error": "Version name is required for create"}
            description = args.get("description", "")
            try:
                version = await engine.create_version(brain.id, name, description)
            except ValueError:
                logger.error("Version create failed for brain '%s' name '%s'", brain.id, name)
                return {"error": "Failed to create version: invalid parameters"}
            return {
                "success": True,
                "version_id": version.id,
                "version_name": version.version_name,
                "version_number": version.version_number,
                "neuron_count": version.neuron_count,
                "synapse_count": version.synapse_count,
                "fiber_count": version.fiber_count,
                "message": f"Created version '{name}' (#{version.version_number})",
            }

        elif action == "list":
            limit = min(args.get("limit", 20), 100)
            versions = await engine.list_versions(brain.id, limit=limit)
            return {
                "versions": [
                    {
                        "id": v.id,
                        "name": v.version_name,
                        "number": v.version_number,
                        "description": v.description,
                        "neuron_count": v.neuron_count,
                        "synapse_count": v.synapse_count,
                        "fiber_count": v.fiber_count,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
                "count": len(versions),
            }

        elif action == "rollback":
            version_id = args.get("version_id")
            if not version_id:
                return {"error": "version_id is required for rollback"}
            try:
                rollback_v = await engine.rollback(brain.id, version_id)
            except ValueError:
                logger.error(
                    "Version rollback failed for brain '%s' version '%s'", brain.id, version_id
                )
                return {"error": "Rollback failed: version not found or invalid"}
            return {
                "success": True,
                "rollback_version_id": rollback_v.id,
                "rollback_version_name": rollback_v.version_name,
                "neuron_count": rollback_v.neuron_count,
                "synapse_count": rollback_v.synapse_count,
                "fiber_count": rollback_v.fiber_count,
                "message": f"Rolled back to '{rollback_v.version_name}'",
            }

        elif action == "diff":
            from_id = args.get("from_version")
            to_id = args.get("to_version")
            if not from_id or not to_id:
                return {"error": "from_version and to_version are required for diff"}
            try:
                diff = await engine.diff(brain.id, from_id, to_id)
            except ValueError:
                logger.error(
                    "Version diff failed for brain '%s' from '%s' to '%s'", brain.id, from_id, to_id
                )
                return {"error": "Diff failed: one or both versions not found"}
            return {
                "summary": diff.summary,
                "neurons_added": len(diff.neurons_added),
                "neurons_removed": len(diff.neurons_removed),
                "neurons_modified": len(diff.neurons_modified),
                "synapses_added": len(diff.synapses_added),
                "synapses_removed": len(diff.synapses_removed),
                "synapses_weight_changed": len(diff.synapses_weight_changed),
                "fibers_added": len(diff.fibers_added),
                "fibers_removed": len(diff.fibers_removed),
            }

        return {"error": f"Unknown action: {action}"}

    async def _transplant(self, args: dict[str, Any]) -> dict[str, Any]:
        """Transplant memories from another brain."""
        from neural_memory.unified_config import get_shared_storage

        target_storage = await self.get_storage()
        target_brain_id = target_storage.brain_id
        if not target_brain_id:
            return {"error": "No brain configured"}

        target_brain = await target_storage.get_brain(target_brain_id)
        if not target_brain:
            return {"error": "No brain configured"}

        source_brain_name = args.get("source_brain")
        if not source_brain_name:
            return {"error": "source_brain is required"}

        if source_brain_name == target_brain.name:
            return {
                "error": "Source brain and target brain are the same. "
                "Transplanting a brain into itself is a destructive no-op."
            }

        # Open a separate storage for the source brain (.db file)
        try:
            source_storage = await get_shared_storage(brain_name=source_brain_name)
        except Exception:
            logger.error("Failed to open source brain storage", exc_info=True)
            return {"error": "Source brain not found"}

        source_brain_id = source_storage.brain_id
        if not source_brain_id:
            return {"error": "Source brain not found"}

        source_brain = await source_storage.get_brain(source_brain_id)
        if source_brain is None:
            return {"error": "Source brain not found"}

        from neural_memory.engine.brain_transplant import TransplantFilter, transplant
        from neural_memory.engine.merge import ConflictStrategy

        tags = args.get("tags")
        memory_types = args.get("memory_types")
        strategy_str = args.get("strategy", "prefer_local")

        try:
            strategy = ConflictStrategy(strategy_str)
        except ValueError:
            return {"error": f"Invalid strategy: {strategy_str}"}

        filt = TransplantFilter(
            tags=frozenset(tags) if tags else None,
            memory_types=frozenset(memory_types) if memory_types else None,
        )

        try:
            result = await transplant(
                source_storage=source_storage,
                target_storage=target_storage,
                source_brain_id=source_brain_id,
                target_brain_id=target_brain_id,
                filt=filt,
                strategy=strategy,
            )
        except ValueError as exc:
            logger.error("Transplant failed: %s", exc)
            return {"error": "Transplant failed"}

        return {
            "success": True,
            "fibers_transplanted": result.fibers_transplanted,
            "neurons_transplanted": result.neurons_transplanted,
            "synapses_transplanted": result.synapses_transplanted,
            "merge_summary": result.merge_report.summary(),
            "message": f"Transplanted from '{source_brain_name}': {result.fibers_transplanted} fibers",
        }

    async def _record_tool_action(self, action_type: str, context: str = "") -> None:
        """Record an action event for habit learning (fire-and-forget)."""
        try:
            import os

            source = os.environ.get("NEURALMEMORY_SOURCE", "mcp")[:256]
            storage = await self.get_storage()
            await storage.record_action(
                action_type=action_type,
                action_context=context[:200] if context else "",
                session_id=f"{source}-{id(self)}",
            )
        except Exception:
            logger.debug("Action recording failed (non-critical)", exc_info=True)
