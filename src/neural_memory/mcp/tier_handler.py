"""MCP handler mixin for auto-tier management.

Provides the nmem_tier tool with actions:
- status: Show current tier distribution and config
- evaluate: Dry-run tier changes (what would happen)
- apply: Apply tier changes
- history: Show promotion history for a memory
- config: Show/update tier configuration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryTier, MemoryType
from neural_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class TierHandler:
    """Mixin providing auto-tier management handler."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _tier(self, args: dict[str, Any]) -> dict[str, Any]:
        """Auto-tier management: status, evaluate, apply, history, config."""
        action = args.get("action", "status")

        if action == "status":
            return await self._tier_status()
        elif action == "evaluate":
            return await self._tier_evaluate()
        elif action == "apply":
            return await self._tier_apply(dry_run=args.get("dry_run", False))
        elif action == "history":
            return await self._tier_history(args.get("fiber_id", ""))
        elif action == "config":
            return self._tier_config()
        return {"error": f"Unknown tier action: {action}"}

    async def _tier_status(self) -> dict[str, Any]:
        """Show current tier distribution and auto-tier config."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        distribution = {MemoryTier.HOT: 0, MemoryTier.WARM: 0, MemoryTier.COLD: 0}
        for tier_name in (MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD):
            distribution[tier_name] = await storage.count_typed_memories(tier=tier_name)

        return {
            "brain": brain.name,
            "tier_distribution": distribution,
            "total_memories": sum(distribution.values()),
            "auto_tier": {
                "enabled": self.config.tiers.auto_enabled,
                "is_pro": self.config.is_pro(),
                "promote_threshold": self.config.tiers.promote_threshold,
                "demote_inactive_days": self.config.tiers.demote_inactive_days,
                "cold_archive_days": self.config.tiers.cold_archive_days,
                "max_hot_memories": self.config.tiers.max_hot_memories,
            },
        }

    async def _tier_evaluate(self) -> dict[str, Any]:
        """Dry-run: show what tier changes would happen."""
        if not self.config.is_pro():
            return {
                "error": "Auto-tier requires Pro. Free users can set tiers manually via nmem_edit."
            }

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.tier_engine import TierEngine

        engine = TierEngine(storage, self.config.tiers)
        report = await engine.evaluate(brain.id)
        return {
            "dry_run": True,
            **report.to_dict(),
            "hint": "Use action='apply' to apply these changes.",
        }

    async def _tier_apply(self, *, dry_run: bool) -> dict[str, Any]:
        """Apply tier changes (or dry-run if requested)."""
        if not self.config.is_pro():
            return {
                "error": "Auto-tier requires Pro. Free users can set tiers manually via nmem_edit."
            }

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.tier_engine import TierEngine

        engine = TierEngine(storage, self.config.tiers)
        report = await engine.apply(brain.id, dry_run=dry_run)
        return report.to_dict()

    async def _tier_history(self, fiber_id: str) -> dict[str, Any]:
        """Show promotion history for a specific memory."""
        if not fiber_id:
            return {"error": "fiber_id is required for history action"}

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.core.memory_types import MemoryType

        mem = await storage.get_typed_memory(fiber_id)
        if mem is None:
            return {"error": f"Typed memory not found: {fiber_id}"}

        history = mem.metadata.get("promotion_history", [])
        return {
            "fiber_id": fiber_id,
            "current_tier": mem.tier,
            "memory_type": mem.memory_type.value
            if isinstance(mem.memory_type, MemoryType)
            else str(mem.memory_type),
            "promotion_history": history,
            "total_changes": len(history),
        }

    def _tier_config(self) -> dict[str, Any]:
        """Show current tier configuration."""
        return {
            "auto_enabled": self.config.tiers.auto_enabled,
            "is_pro": self.config.is_pro(),
            "promote_threshold": self.config.tiers.promote_threshold,
            "demote_inactive_days": self.config.tiers.demote_inactive_days,
            "cold_archive_days": self.config.tiers.cold_archive_days,
            "max_hot_memories": self.config.tiers.max_hot_memories,
            "hint": "Edit ~/.neuralmemory/config.toml [tiers] section to change thresholds.",
        }

    async def _boundaries(self, args: dict[str, Any]) -> dict[str, Any]:
        """List and manage domain-scoped boundaries."""
        action = args.get("action", "list")
        domain_filter = args.get("domain")
        if domain_filter and isinstance(domain_filter, str):
            domain_filter = domain_filter.lower().strip()[:50]

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        # Fetch all boundary memories
        boundaries = await storage.find_typed_memories(memory_type=MemoryType.BOUNDARY, limit=1000)

        if action == "domains":
            return self._boundaries_domains(boundaries)

        return await self._boundaries_list(storage, boundaries, domain_filter)

    def _boundaries_domains(self, boundaries: list[Any]) -> dict[str, Any]:
        """List unique domains with boundary counts."""
        domain_counts: dict[str, int] = {}
        unscoped = 0
        for tm in boundaries:
            domain_tags = {t[7:] for t in tm.tags if t.startswith("domain:")}
            if domain_tags:
                for d in domain_tags:
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            else:
                unscoped += 1

        domains = [{"domain": d, "count": c} for d, c in sorted(domain_counts.items())]
        return {
            "domains": domains,
            "unscoped_count": unscoped,
            "total_boundaries": len(boundaries),
        }

    async def _boundaries_list(
        self,
        storage: Any,
        boundaries: list[Any],
        domain_filter: str | None,
    ) -> dict[str, Any]:
        """List boundaries, optionally filtered by domain."""
        items: list[dict[str, Any]] = []
        for tm in boundaries:
            domain_tags = {t[7:] for t in tm.tags if t.startswith("domain:")}

            # Filter by domain if specified
            if domain_filter:
                if domain_tags and domain_filter not in domain_tags:
                    continue
                # If no domain_filter match needed for unscoped, include them
                # (unscoped = global boundaries, always visible)

            fiber = await storage.get_fiber(tm.fiber_id)
            content = ""
            if fiber and fiber.anchor_neuron_id:
                neuron = await storage.get_neuron(fiber.anchor_neuron_id)
                if neuron:
                    content = neuron.content[:200]

            items.append(
                {
                    "fiber_id": tm.fiber_id,
                    "content": content,
                    "domains": sorted(domain_tags) if domain_tags else ["(global)"],
                    "tier": tm.tier,
                    "priority": tm.priority.value if hasattr(tm.priority, "value") else tm.priority,
                    "tags": sorted(t for t in tm.tags if not t.startswith("domain:")),
                    "created_at": tm.created_at.isoformat() if tm.created_at else None,
                }
            )

        return {
            "boundaries": items,
            "count": len(items),
            "domain_filter": domain_filter,
        }
