"""Auto-tier engine for promoting/demoting memories based on access patterns.

Moves memories between HOT/WARM/COLD tiers automatically:
- WARM→HOT: composite score >= promote_threshold (multi-factor: recency,
  frequency, importance, causal centrality)
- HOT→WARM: no recent access (last_activated > demote_inactive_days ago)
- WARM→COLD: long-term neglect (last_activated > cold_archive_days ago)

Composite score (Pro):
  0.4 * recency + 0.3 * frequency + 0.2 * importance + 0.1 * causal_centrality
  Each factor normalized to 0.0-1.0.  Free tier falls back to frequency-only.

Protection rules:
- BOUNDARY type memories: never demoted (always HOT)
- Pinned fibers: never demoted
- Already-promoted-this-cycle: skipped (prevent oscillation)

Pro feature: gated behind config.is_pro().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryTier, MemoryType
from neural_memory.core.synapse import SynapseType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.neuron import NeuronState
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


def compute_promotion_score(
    state: NeuronState,
    causal_synapse_count: int = 0,
    *,
    now: datetime | None = None,
) -> float:
    """Compute a multi-factor promotion score for a neuron (0.0-1.0).

    Factors (weighted sum):
      - recency  (0.4): how recently the neuron was accessed
      - frequency (0.3): total access count, log-normalized
      - importance (0.2): activation level as proxy for importance
      - causal    (0.1): number of causal synapses, log-normalized

    Each factor is normalized to [0.0, 1.0].
    """
    if now is None:
        now = utcnow()

    # Recency: hyperbolic decay based on days since last access
    last_active = state.last_activated or state.created_at
    if last_active is not None:
        days_since = max((now - last_active).total_seconds() / 86400.0, 0.0)
        recency = 1.0 / (1.0 + days_since / 7.0)
    else:
        recency = 0.0

    frequency = min(math.log1p(state.access_frequency) / math.log1p(50), 1.0)

    # Importance: current activation level (already 0-1)
    importance = max(0.0, min(state.activation_level, 1.0))

    # Causal centrality: log-normalized count of causal synapses
    causal = min(math.log1p(causal_synapse_count) / math.log1p(20), 1.0)

    return 0.4 * recency + 0.3 * frequency + 0.2 * importance + 0.1 * causal


def compute_decay_strength(
    state: NeuronState,
    *,
    now: datetime | None = None,
) -> float:
    """Compute memory retention strength using power-law decay (0.0-1.0).

    For neurons with >= 5 accesses, uses power-law:
        S(t) = (t_days + 1) ^ (-b)  where b = 0.3 (Ebbinghaus-like)

    For cold-start (< 5 accesses), uses hyperbolic fallback:
        S(t) = 1 / (1 + age_days / 30)

    Higher strength = better retained.
    """
    if now is None:
        now = utcnow()

    last_active = state.last_activated or state.created_at
    if last_active is None:
        return 0.0

    days_since = max((now - last_active).total_seconds() / 86400.0, 0.0)

    if state.access_frequency >= 5:
        # Power-law decay: frequently accessed memories decay slower
        # b decreases with more accesses (better retention)
        b = max(0.1, 0.5 - 0.02 * min(state.access_frequency, 20))
        return float((days_since + 1.0) ** (-b))
    else:
        # Hyperbolic fallback for cold-start
        return 1.0 / (1.0 + days_since / 30.0)


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

            # Multi-factor composite score (Pro) or frequency-only (Free)
            causal_count = 0
            try:
                causal_synapses = await self._storage.get_synapses(
                    source_id=fiber.anchor_neuron_id, type=SynapseType.CAUSED_BY
                )
                causal_count = len(causal_synapses)
            except Exception:
                pass  # graceful fallback if storage doesn't support type filter

            score = compute_promotion_score(state, causal_count, now=now)
            # Normalize threshold: promote_threshold was access_frequency count,
            # now treat it as score threshold mapped to 0.0-1.0 (default 5 → 0.5)
            score_threshold = min(self._config.promote_threshold / 10.0, 1.0)

            if score >= score_threshold:
                change = TierChange(
                    fiber_id=mem.fiber_id,
                    memory_type=mem.memory_type.value,
                    from_tier=MemoryTier.WARM,
                    to_tier=MemoryTier.HOT,
                    reason=f"composite_score={score:.3f} >= {score_threshold:.2f}",
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
