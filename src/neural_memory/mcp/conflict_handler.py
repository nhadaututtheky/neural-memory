"""Conflict management handler for MCP server."""

from __future__ import annotations

import logging
import re
from typing import Any

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)

# UUID format validation (case-insensitive for compatibility)
_UUID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)

_VALID_RESOLUTIONS = frozenset({"keep_existing", "keep_new", "keep_both"})

# Tag validation limits
_MAX_TAGS = 50
_MAX_TAG_LENGTH = 100


class ConflictHandler:
    """Mixin: conflict management tool handlers."""

    async def _conflicts(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle conflict management actions."""
        action = args.get("action", "list")

        if action == "list":
            return await self._conflicts_list(args)
        elif action == "resolve":
            return await self._conflicts_resolve(args)
        elif action == "check":
            return await self._conflicts_check(args)
        return {"error": f"Unknown action: {action}"}

    async def _conflicts_list(self, args: dict[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
        """List unresolved conflicts (CONTRADICTS synapses)."""
        try:
            storage = await self.get_storage()
            typed_synapses = await storage.get_synapses(type=SynapseType.CONTRADICTS)

            # Filter for unresolved only (metadata check still needed)
            contradicts = [s for s in typed_synapses if not s.metadata.get("_resolved")]

            limit = min(args.get("limit", 50), 200)
            contradicts = contradicts[:limit]

            if not contradicts:
                return {"conflicts": [], "count": 0}

            # Batch-fetch all involved neurons
            neuron_ids: list[str] = []
            for s in contradicts:
                neuron_ids.append(s.source_id)
                neuron_ids.append(s.target_id)
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_ids: list[str] = []
            for nid in neuron_ids:
                if nid not in seen:
                    seen.add(nid)
                    unique_ids.append(nid)

            neurons = await storage.get_neurons_batch(unique_ids)

            conflicts: list[dict[str, Any]] = []
            for s in contradicts:
                target = neurons.get(s.target_id)
                source = neurons.get(s.source_id)

                target_content = target.content if target else "(deleted)"
                source_content = source.content if source else "(deleted)"
                is_superseded = target.metadata.get("_superseded", False) if target else False

                conflicts.append(
                    {
                        "existing_neuron_id": s.target_id,
                        "content": target_content[:200],
                        "disputed_by_preview": source_content[:100],
                        "conflict_type": s.metadata.get("conflict_type", "unknown"),
                        "confidence": s.weight,
                        "detected_at": s.metadata.get("detected_at", ""),
                        "is_superseded": is_superseded,
                    }
                )

            return {"conflicts": conflicts, "count": len(conflicts)}

        except Exception as e:
            logger.error("Failed to list conflicts: %s", e, exc_info=True)
            return {"error": "Failed to list conflicts. Check server logs for details."}

    async def _conflicts_resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve a specific conflict by neuron ID."""
        neuron_id = args.get("neuron_id", "")
        if not _UUID_PATTERN.match(neuron_id):
            return {"error": "Invalid neuron_id — must be a valid UUID"}

        resolution = args.get("resolution", "")
        if resolution not in _VALID_RESOLUTIONS:
            return {
                "error": f"Invalid resolution '{resolution}'. "
                f"Must be one of: {', '.join(sorted(_VALID_RESOLUTIONS))}",
            }

        try:
            storage = await self.get_storage()
            storage.disable_auto_save()

            try:
                # Fetch the disputed neuron
                neuron = await storage.get_neuron(neuron_id)
                if neuron is None:
                    return {"error": f"Neuron {neuron_id} not found"}
                if not neuron.metadata.get("_disputed"):
                    return {"error": f"Neuron {neuron_id} is not disputed"}

                # Find the CONTRADICTS synapse targeting this neuron
                target_synapses = await storage.get_synapses(
                    target_id=neuron_id, type=SynapseType.CONTRADICTS
                )
                synapse = next(
                    (s for s in target_synapses if not s.metadata.get("_resolved")),
                    None,
                )
                if synapse is None:
                    return {"error": f"No unresolved CONTRADICTS synapse found for {neuron_id}"}

                # Get the disputing (source) neuron
                disputing = await storage.get_neuron(synapse.source_id)
                if disputing is None:
                    return {"error": f"Disputing neuron {synapse.source_id} not found"}

                if resolution == "keep_existing":
                    await self._resolve_keep_existing(storage, neuron, disputing, synapse)
                elif resolution == "keep_new":
                    await self._resolve_keep_new(storage, neuron, synapse)
                elif resolution == "keep_both":
                    await self._resolve_keep_both(storage, neuron, disputing, synapse)

                await storage.batch_save()

            finally:
                storage.enable_auto_save()

            return {
                "success": True,
                "neuron_id": neuron_id,
                "resolution": resolution,
                "message": f"Conflict resolved: {resolution}",
            }

        except Exception as e:
            logger.error("Failed to resolve conflict: %s", e, exc_info=True)
            return {"error": "Failed to resolve conflict. Check server logs for details."}

    async def _resolve_keep_existing(
        self,
        storage: Any,
        existing: Any,
        disputing: Any,
        synapse: Synapse,
    ) -> None:
        """Keep existing neuron, mark new as superseded."""
        from neural_memory.core.neuron import NeuronState

        # Restore pre-dispute activation if available
        pre_activation = existing.metadata.get("_pre_dispute_activation")
        if pre_activation is not None:
            state = await storage.get_neuron_state(existing.id)
            if state:
                restored = NeuronState(
                    neuron_id=state.neuron_id,
                    activation_level=pre_activation,
                    access_frequency=state.access_frequency,
                    last_activated=state.last_activated,
                    decay_rate=state.decay_rate,
                    created_at=state.created_at,
                    firing_threshold=state.firing_threshold,
                    refractory_until=state.refractory_until,
                    refractory_period_ms=state.refractory_period_ms,
                    homeostatic_target=state.homeostatic_target,
                )
                await storage.update_neuron_state(restored)

        updated_existing = existing.with_metadata(_disputed=False, _superseded=False)
        await storage.update_neuron(updated_existing)

        updated_disputing = disputing.with_metadata(_superseded=True)
        await storage.update_neuron(updated_disputing)

        resolved_synapse = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=synapse.weight,
            direction=synapse.direction,
            metadata={**synapse.metadata, "_resolved": "keep_existing"},
            reinforced_count=synapse.reinforced_count,
            last_activated=synapse.last_activated,
            created_at=synapse.created_at,
        )
        await storage.update_synapse(resolved_synapse)

    async def _resolve_keep_new(
        self,
        storage: Any,
        existing: Any,
        synapse: Synapse,
    ) -> None:
        """Keep new neuron, mark existing as superseded."""
        updated_existing = existing.with_metadata(_disputed=False, _superseded=True)
        await storage.update_neuron(updated_existing)

        resolved_synapse = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=synapse.weight,
            direction=synapse.direction,
            metadata={**synapse.metadata, "_resolved": "keep_new"},
            reinforced_count=synapse.reinforced_count,
            last_activated=synapse.last_activated,
            created_at=synapse.created_at,
        )
        await storage.update_synapse(resolved_synapse)

    async def _resolve_keep_both(
        self,
        storage: Any,
        existing: Any,
        disputing: Any,
        synapse: Synapse,
    ) -> None:
        """Keep both neurons, remove dispute markers."""
        updated_existing = existing.with_metadata(
            _disputed=False, _superseded=False, _conflict_resolved=True
        )
        await storage.update_neuron(updated_existing)

        updated_disputing = disputing.with_metadata(_conflict_resolved=True)
        await storage.update_neuron(updated_disputing)

        resolved_synapse = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=synapse.weight,
            direction=synapse.direction,
            metadata={**synapse.metadata, "_resolved": "keep_both"},
            reinforced_count=synapse.reinforced_count,
            last_activated=synapse.last_activated,
            created_at=synapse.created_at,
        )
        await storage.update_synapse(resolved_synapse)

    async def _conflicts_check(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check new content for potential conflicts before saving."""
        content = args.get("content", "")
        if not content:
            return {"error": "Content required for conflict check"}
        if len(content) > MAX_CONTENT_LENGTH:
            return {
                "error": f"Content too long ({len(content)} chars). Max: {MAX_CONTENT_LENGTH}.",
            }

        tags_raw = args.get("tags", [])
        if len(tags_raw) > _MAX_TAGS:
            return {"error": f"Too many tags ({len(tags_raw)}). Max: {_MAX_TAGS}."}
        tags: set[str] = set()
        for t in tags_raw:
            if not isinstance(t, str) or len(t) > _MAX_TAG_LENGTH:
                return {"error": f"Invalid tag (must be string, max {_MAX_TAG_LENGTH} chars)."}
            tags.add(t)

        try:
            from neural_memory.engine.conflict_detection import detect_conflicts

            storage = await self.get_storage()
            conflicts = await detect_conflicts(content, tags, storage)

            results: list[dict[str, Any]] = [
                {
                    "existing_content": c.existing_content[:200],
                    "new_content": c.new_content[:200],
                    "conflict_type": c.type.value,
                    "confidence": c.confidence,
                    "subject": c.subject,
                }
                for c in conflicts
            ]

            return {
                "potential_conflicts": results,
                "count": len(results),
                "message": "Approximate — auto-tags not available in pre-check",
            }

        except Exception as e:
            logger.error("Failed to check conflicts: %s", e, exc_info=True)
            return {"error": "Failed to check conflicts. Check server logs for details."}
