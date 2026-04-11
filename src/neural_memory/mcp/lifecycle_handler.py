"""MCP handler mixin for memory edit, forget, consolidation, tool stats, and lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
)
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.mcp.tool_handler_utils import _get_brain_or_error, _require_brain_id
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class LifecycleHandler:
    """Mixin providing edit, forget, consolidate, tool_stats, and lifecycle handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _edit(self, args: dict[str, Any]) -> dict[str, Any]:
        """Edit an existing memory's type, content, priority, or tier."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        new_type = args.get("type")
        new_content = args.get("content")
        new_priority = args.get("priority")
        new_tier = args.get("tier")
        if new_tier is not None:
            new_tier = str(new_tier).lower().strip()

        if new_type is None and new_content is None and new_priority is None and new_tier is None:
            return {"error": "At least one of type, content, priority, or tier must be provided"}

        if new_type is not None:
            try:
                MemoryType(new_type)
            except ValueError:
                return {"error": f"Invalid memory type: {new_type}"}

        if new_tier is not None:
            try:
                MemoryTier(new_tier)
            except ValueError:
                return {"error": f"Invalid tier: {new_tier}. Must be hot, warm, or cold."}

        if new_content is not None and len(new_content) > MAX_CONTENT_LENGTH:
            return {
                "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for edit")
            return {"error": "No brain configured"}

        # Try as fiber_id first, then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            # Edit via fiber path
            changes: list[str] = []

            # Update typed_memory (type, priority, tier)
            if new_type is not None or new_priority is not None or new_tier is not None:
                from dataclasses import replace as dc_replace

                updated_tm = typed_mem
                if new_type is not None:
                    updated_tm = dc_replace(updated_tm, memory_type=MemoryType(new_type))
                    changes.append(f"type: {typed_mem.memory_type.value} → {new_type}")
                    # Sync type into fiber.metadata to keep both stores consistent
                    updated_meta = {**fiber.metadata, "type": new_type}
                    fiber = dc_replace(fiber, metadata=updated_meta)
                    await storage.update_fiber(fiber)
                    # Enforce boundary invariant: boundaries are always HOT
                    if (
                        updated_tm.memory_type == MemoryType.BOUNDARY
                        and updated_tm.tier != MemoryTier.HOT
                    ):
                        old_tier = updated_tm.tier
                        updated_tm = updated_tm.with_tier(MemoryTier.HOT)
                        changes.append(f"tier: {old_tier} → hot (boundary auto-promote)")
                if new_priority is not None:
                    updated_tm = dc_replace(updated_tm, priority=Priority.from_int(new_priority))
                    changes.append(f"priority: {typed_mem.priority.value} → {new_priority}")
                if new_tier is not None:
                    old_tier = updated_tm.tier
                    updated_tm = updated_tm.with_tier(new_tier)
                    if updated_tm.tier != old_tier:
                        changes.append(f"tier: {old_tier} → {updated_tm.tier}")
                await storage.update_typed_memory(updated_tm)

            # Update anchor neuron content
            if new_content is not None:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    from dataclasses import replace as dc_replace

                    updated_neuron = dc_replace(anchor, content=new_content)
                    await storage.update_neuron(updated_neuron)
                    changes.append(f"content updated ({len(new_content)} chars)")

            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            from dataclasses import replace as dc_replace

            changes = []
            if new_content is not None:
                neuron = dc_replace(neuron, content=new_content)
                changes.append(f"content updated ({len(new_content)} chars)")
            if new_type is not None:
                from neural_memory.core.neuron import NeuronType

                try:
                    neuron = dc_replace(neuron, type=NeuronType(new_type))
                    changes.append(f"neuron type → {new_type}")
                except ValueError:
                    pass  # NeuronType doesn't map 1:1 to MemoryType
            await storage.update_neuron(neuron)
            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        return {"error": "Memory not found"}

    async def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Explicitly delete or close a specific memory."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        hard = args.get("hard", False)
        reason = args.get("reason", "")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for forget")
            return {"error": "No brain configured"}

        # Look up the memory
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if not typed_mem and not fiber:
            # Try as neuron_id — find its fiber
            neuron = await storage.get_neuron(memory_id)
            if not neuron:
                return {"error": "Memory not found"}
            # For neuron-only delete in hard mode
            if hard:
                await storage.delete_neuron(memory_id)
                return {
                    "status": "hard_deleted",
                    "memory_id": memory_id,
                    "message": "Neuron permanently deleted",
                }
            return {
                "error": f"No typed memory found for neuron {memory_id}. Use hard=true for neuron deletion."
            }

        if hard:
            # Permanent deletion: fiber + typed_memory + neurons
            storage.disable_auto_save()
            try:
                # Delete typed memory
                await storage.delete_typed_memory(memory_id)

                # Delete fiber (CASCADE handles fiber_neurons junction)
                if fiber:
                    await storage.delete_fiber(memory_id)

                await storage.batch_save()
            finally:
                storage.enable_auto_save()

            logger.info("Hard-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "hard_deleted",
                "memory_id": memory_id,
                "message": "Memory permanently deleted with cascade cleanup",
            }
        else:
            # Soft delete: expire immediately
            from dataclasses import replace as dc_replace

            assert typed_mem is not None  # guaranteed by early return above
            expired_tm = dc_replace(typed_mem, expires_at=utcnow())
            await storage.update_typed_memory(expired_tm)

            logger.info("Soft-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "soft_deleted",
                "memory_id": memory_id,
                "message": "Memory marked as expired (will be cleaned up on next consolidation)",
            }

    async def _consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run memory consolidation on the current brain."""
        from neural_memory.engine.consolidation import (
            ConsolidationConfig,
            ConsolidationStrategy,
        )
        from neural_memory.engine.consolidation_delta import run_with_delta

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for consolidate")
            return {"error": "No brain configured"}

        # Parse strategy
        strategy_str = args.get("strategy", "all")
        try:
            strategy = ConsolidationStrategy(strategy_str)
        except ValueError:
            valid = [s.value for s in ConsolidationStrategy]
            return {"error": f"Invalid strategy: {strategy_str}. Valid: {valid}"}

        strategies = [strategy]
        dry_run = bool(args.get("dry_run", False))

        # Build config with optional overrides (bounded to valid ranges)
        config_kwargs: dict[str, Any] = {}
        if "prune_weight_threshold" in args:
            val = args["prune_weight_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_weight_threshold"] = max(0.0, min(float(val), 1.0))
        if "merge_overlap_threshold" in args:
            val = args["merge_overlap_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["merge_overlap_threshold"] = max(0.0, min(float(val), 1.0))
        if "prune_min_inactive_days" in args:
            val = args["prune_min_inactive_days"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_min_inactive_days"] = max(0, int(val))

        config = ConsolidationConfig(**config_kwargs) if config_kwargs else None

        try:
            # Pass tier_config for auto-tier (Pro feature, runs post-consolidation)
            tier_config = self.config.tiers if self.config.is_pro() else None
            delta = await run_with_delta(
                storage,
                brain_id,
                strategies=strategies,
                dry_run=dry_run,
                config=config,
                tier_config=tier_config,
            )
        except Exception:
            logger.error("Consolidation failed", exc_info=True)
            return {"error": "Consolidation failed unexpectedly"}

        result = delta.to_dict()
        result["strategy"] = strategy_str
        result["dry_run"] = dry_run
        result["summary"] = delta.report.summary()
        return result

    async def _tool_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get tool usage analytics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "summary")
        try:
            days = max(1, min(int(args.get("days", 30)), 365))
            limit = max(1, min(int(args.get("limit", 20)), 200))
        except (TypeError, ValueError):
            return {"error": "days and limit must be integers"}

        if action == "summary":
            result: dict[str, Any] = await storage.get_tool_stats(brain.id)  # type: ignore[attr-defined]
            return result
        elif action == "daily":
            daily = await storage.get_tool_stats_by_period(  # type: ignore[attr-defined]
                brain.id, days=days, limit=limit
            )
            return {"daily": daily, "days": days}
        else:
            return {"error": f"Unknown action: {action}"}

    async def _lifecycle(self, args: dict[str, Any]) -> dict[str, Any]:
        """Memory lifecycle management: status, recover, freeze, thaw."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "status")
        neuron_id: str | None = args.get("id") or args.get("neuron_id")

        if action == "status":
            try:
                distribution = await storage.get_lifecycle_distribution()
            except Exception:
                logger.error("nmem_lifecycle status failed", exc_info=True)
                return {"error": "Failed to retrieve lifecycle distribution"}
            total = sum(distribution.values())
            return {
                "brain": brain.id,
                "distribution": distribution,
                "total_neurons": total,
            }

        if action in ("recover", "freeze", "thaw"):
            if not neuron_id:
                return {"error": f"action='{action}' requires 'id' (neuron_id)"}

            if action == "recover":
                # Find which fiber contains this neuron, then recover.
                from neural_memory.engine.compression import CompressionEngine

                fibers = await storage.find_fibers(contains_neuron=neuron_id)
                if not fibers:
                    # Try decompress by fiber_id directly (caller may pass fiber_id as id)
                    engine = CompressionEngine(storage)
                    success = await engine.recover_fiber(neuron_id)
                    if success:
                        return {"recovered": True, "fiber_id": neuron_id}
                    return {
                        "recovered": False,
                        "reason": "No fiber found for neuron and direct recovery failed",
                    }

                fiber = fibers[0]
                engine = CompressionEngine(storage)
                success = await engine.recover_fiber(fiber.id)
                return {
                    "recovered": success,
                    "fiber_id": fiber.id,
                    "neuron_id": neuron_id,
                }

            elif action == "freeze":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=True)
                except Exception:
                    logger.error("nmem_lifecycle freeze failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to freeze neuron"}
                return {"frozen": True, "neuron_id": neuron_id}

            elif action == "thaw":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=False)
                except Exception:
                    logger.error("nmem_lifecycle thaw failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to thaw neuron"}
                return {"frozen": False, "neuron_id": neuron_id}

        if action == "at_risk":
            within_days = 7
            raw_days = args.get("within_days")
            if raw_days is not None:
                try:
                    within_days = max(1, min(90, int(raw_days)))
                except (TypeError, ValueError):
                    return {"error": f"Invalid within_days: {raw_days}"}

            try:
                expiring_count = await storage.get_expiring_memory_count(within_days=within_days)
                expired_count = await storage.get_expired_memory_count()
            except Exception:
                logger.error("nmem_lifecycle at_risk failed", exc_info=True)
                return {"error": "Failed to query at-risk memories"}

            # Fetch details of at-risk memories (limit to 20 for readability)
            at_risk_items: list[dict[str, Any]] = []
            try:
                # Get fibers to check, limited to recent ones
                fibers = await storage.find_fibers(limit=100)
                if fibers:
                    fiber_ids = [f.id for f in fibers]
                    expiring = await storage.get_expiring_memories_for_fibers(
                        fiber_ids=fiber_ids, within_days=within_days
                    )
                    for tm in expiring[:20]:
                        at_risk_items.append(
                            {
                                "fiber_id": tm.fiber_id,
                                "memory_type": tm.memory_type,
                                "expires_at": tm.expires_at.isoformat() if tm.expires_at else None,
                                "tier": getattr(tm, "tier", "warm"),
                            }
                        )
            except Exception:
                logger.debug("at_risk detail fetch failed", exc_info=True)

            return {
                "brain": brain.id,
                "within_days": within_days,
                "expiring_soon": expiring_count,
                "already_expired": expired_count,
                "at_risk_memories": at_risk_items,
                "hint": (
                    f"{expiring_count} memories will expire within {within_days} days. "
                    "Use nmem_consolidate or nmem_pin to preserve important ones."
                    if expiring_count > 0
                    else f"No memories at risk of expiry within {within_days} days."
                ),
            }

        return {"error": f"Unknown action: {action}. Valid: status, recover, freeze, thaw, at_risk"}
