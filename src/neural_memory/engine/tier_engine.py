"""Auto-tier engine for promoting/demoting memories based on access patterns.

Moves memories between HOT/WARM/COLD tiers automatically:
- WARM→HOT: high access frequency (access_frequency >= promote_threshold)
- HOT→WARM: no recent access (last_activated > demote_inactive_days ago)
- WARM→COLD: long-term neglect (last_activated > cold_archive_days ago)

Protection rules:
- BOUNDARY type memories: never demoted (always HOT)
- Pinned fibers: never demoted
- Already-promoted-this-cycle: skipped (prevent oscillation)

Pro feature: gated behind config.is_pro().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryTier, MemoryType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import TierConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierChange:
    """A single tier change event."""

    fiber_id: str
    memory_type: str
    from_tier: str
    to_tier: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiber_id": self.fiber_id,
            "memory_type": self.memory_type,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TierReport:
    """Result of a tier evaluation or application."""

    promoted: list[TierChange] = field(default_factory=list)
    demoted: list[TierChange] = field(default_factory=list)
    archived: list[TierChange] = field(default_factory=list)
    skipped_boundary: int = 0
    skipped_pinned: int = 0
    skipped_at_cap: int = 0
    dry_run: bool = True

    @property
    def total_changes(self) -> int:
        return len(self.promoted) + len(self.demoted) + len(self.archived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": [c.to_dict() for c in self.promoted],
            "demoted": [c.to_dict() for c in self.demoted],
            "archived": [c.to_dict() for c in self.archived],
            "skipped_boundary": self.skipped_boundary,
            "skipped_pinned": self.skipped_pinned,
            "skipped_at_cap": self.skipped_at_cap,
            "total_changes": self.total_changes,
            "dry_run": self.dry_run,
        }


class TierEngine:
    """Engine for automatic tier promotion/demotion of memories."""

    def __init__(self, storage: NeuralStorage, config: TierConfig) -> None:
        self._storage = storage
        self._config = config

    async def evaluate(self, brain_id: str) -> TierReport:
        """Evaluate tier changes without applying them (dry run)."""
        return await self._run(brain_id, dry_run=True)

    async def apply(self, brain_id: str, dry_run: bool = False) -> TierReport:
        """Evaluate and optionally apply tier changes.

        Args:
            brain_id: Brain to evaluate
            dry_run: If True, calculate but don't apply changes
        """
        return await self._run(brain_id, dry_run=dry_run)

    async def _run(self, brain_id: str, *, dry_run: bool) -> TierReport:
        """Core tier evaluation logic."""
        # Verify storage brain context matches requested brain_id
        if self._storage.brain_id and self._storage.brain_id != brain_id:
            logger.warning(
                "TierEngine brain_id mismatch: requested=%s, storage=%s",
                brain_id,
                self._storage.brain_id,
            )
        now = utcnow()
        promoted: list[TierChange] = []
        demoted: list[TierChange] = []
        archived: list[TierChange] = []
        skipped_boundary = 0
        skipped_pinned = 0
        skipped_at_cap = 0

        # Track fiber IDs changed this cycle to prevent oscillation
        changed_this_cycle: set[str] = set()

        # Current HOT count for cap enforcement
        current_hot_count = await self._storage.count_typed_memories(tier=MemoryTier.HOT)

        # --- Promotion: WARM → HOT ---
        warm_memories = await self._storage.find_typed_memories(tier=MemoryTier.WARM, limit=1000)
        for mem in warm_memories:
            if current_hot_count + len(promoted) >= self._config.max_hot_memories:
                skipped_at_cap += 1
                continue

            # Get anchor neuron state for access frequency
            fiber = await self._storage.get_fiber(mem.fiber_id)
            if fiber is None or not fiber.anchor_neuron_id:
                continue

            state = await self._storage.get_neuron_state(fiber.anchor_neuron_id)
            if state is None:
                continue

            if state.access_frequency >= self._config.promote_threshold:
                change = TierChange(
                    fiber_id=mem.fiber_id,
                    memory_type=mem.memory_type.value,
                    from_tier=MemoryTier.WARM,
                    to_tier=MemoryTier.HOT,
                    reason=f"access_frequency={state.access_frequency} >= {self._config.promote_threshold}",
                )
                promoted.append(change)
                changed_this_cycle.add(mem.fiber_id)

                if not dry_run:
                    updated = mem.with_tier(MemoryTier.HOT)
                    updated = _add_promotion_history(updated, change, now)
                    await self._storage.update_typed_memory(updated)

        # --- Demotion: HOT → WARM ---
        demote_cutoff = now - timedelta(days=self._config.demote_inactive_days)
        hot_memories = await self._storage.find_typed_memories(tier=MemoryTier.HOT, limit=1000)
        for mem in hot_memories:
            # Skip if already changed this cycle (prevent oscillation)
            if mem.fiber_id in changed_this_cycle:
                continue

            # BOUNDARY type: never demoted
            if mem.memory_type == MemoryType.BOUNDARY:
                skipped_boundary += 1
                continue

            # Pinned fibers: never demoted
            fiber = await self._storage.get_fiber(mem.fiber_id)
            if fiber is None:
                continue
            if fiber.pinned:
                skipped_pinned += 1
                continue

            # Check last_activated on anchor neuron
            if not fiber.anchor_neuron_id:
                continue
            state = await self._storage.get_neuron_state(fiber.anchor_neuron_id)
            if state is None:
                continue

            # Fallback to created_at if never activated (prevent immortal HOT)
            last_active = state.last_activated or state.created_at
            if last_active is not None and last_active < demote_cutoff:
                change = TierChange(
                    fiber_id=mem.fiber_id,
                    memory_type=mem.memory_type.value,
                    from_tier=MemoryTier.HOT,
                    to_tier=MemoryTier.WARM,
                    reason=f"inactive since {last_active.isoformat()} (>{self._config.demote_inactive_days}d)",
                )
                demoted.append(change)
                changed_this_cycle.add(mem.fiber_id)

                if not dry_run:
                    updated = mem.with_tier(MemoryTier.WARM)
                    updated = _add_promotion_history(updated, change, now)
                    await self._storage.update_typed_memory(updated)

        # --- Archive: WARM → COLD ---
        archive_cutoff = now - timedelta(days=self._config.cold_archive_days)
        # Re-query WARM (some may have been promoted above in dry_run scenario,
        # but in apply mode the promoted ones are now HOT)
        warm_for_archive = await self._storage.find_typed_memories(tier=MemoryTier.WARM, limit=1000)
        for mem in warm_for_archive:
            if mem.fiber_id in changed_this_cycle:
                continue

            # BOUNDARY: never archived
            if mem.memory_type == MemoryType.BOUNDARY:
                skipped_boundary += 1
                continue

            fiber = await self._storage.get_fiber(mem.fiber_id)
            if fiber is None:
                continue
            if fiber.pinned:
                skipped_pinned += 1
                continue

            if not fiber.anchor_neuron_id:
                continue
            state = await self._storage.get_neuron_state(fiber.anchor_neuron_id)
            if state is None:
                continue

            # Fallback to created_at if never activated
            last_active = state.last_activated or state.created_at
            if last_active is not None and last_active < archive_cutoff:
                change = TierChange(
                    fiber_id=mem.fiber_id,
                    memory_type=mem.memory_type.value,
                    from_tier=MemoryTier.WARM,
                    to_tier=MemoryTier.COLD,
                    reason=f"inactive since {last_active.isoformat()} (>{self._config.cold_archive_days}d)",
                )
                archived.append(change)
                changed_this_cycle.add(mem.fiber_id)

                if not dry_run:
                    updated = mem.with_tier(MemoryTier.COLD)
                    updated = _add_promotion_history(updated, change, now)
                    await self._storage.update_typed_memory(updated)

        report = TierReport(
            promoted=promoted,
            demoted=demoted,
            archived=archived,
            skipped_boundary=skipped_boundary,
            skipped_pinned=skipped_pinned,
            skipped_at_cap=skipped_at_cap,
            dry_run=dry_run,
        )

        if report.total_changes > 0:
            logger.info(
                "Tier engine: %d promoted, %d demoted, %d archived (dry_run=%s)",
                len(promoted),
                len(demoted),
                len(archived),
                dry_run,
            )

        return report


def _add_promotion_history(mem: Any, change: TierChange, timestamp: datetime) -> Any:
    """Add a tier change entry to promotion_history in metadata."""
    history = list(mem.metadata.get("promotion_history", []))
    history.append(
        {
            "from": change.from_tier,
            "to": change.to_tier,
            "reason": change.reason,
            "at": timestamp.isoformat(),
        }
    )
    # Keep last 20 entries to prevent unbounded growth
    history = history[-20:]
    new_meta = {**mem.metadata, "promotion_history": history}
    from dataclasses import replace

    return replace(mem, metadata=new_meta)
