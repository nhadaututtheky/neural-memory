"""Tier management for InfinityDB.

Manages neuron lifecycle across compression tiers (Active → Warm → Cool → Frozen → Crystal).
Promotion/demotion based on access patterns, age, and priority.

Tier assignment rules:
- Active: recently accessed or high-priority (within last 7 days, access_count > 5)
- Warm: moderate access, 7-30 days old
- Cool: low access, 30-90 days old
- Frozen: rarely accessed, >90 days old
- Crystal: manually archived or priority=0, metadata-only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from neural_memory.pro.infinitydb.compressor import CompressionTier, VectorCompressor

logger = logging.getLogger(__name__)

# Default tier thresholds (days since last access)
DEFAULT_WARM_DAYS = 7
DEFAULT_COOL_DAYS = 30
DEFAULT_FROZEN_DAYS = 90


@dataclass(frozen=True)
class TierConfig:
    """Configuration for automatic tier promotion/demotion."""

    warm_after_days: int = DEFAULT_WARM_DAYS
    cool_after_days: int = DEFAULT_COOL_DAYS
    frozen_after_days: int = DEFAULT_FROZEN_DAYS
    min_access_for_active: int = 5
    high_priority_threshold: int = 8  # priority >= this stays active
    auto_promote_on_access: bool = True
    auto_demote_enabled: bool = True


@dataclass(frozen=True)
class TierStats:
    """Statistics about tier distribution."""

    active: int = 0
    warm: int = 0
    cool: int = 0
    frozen: int = 0
    crystal: int = 0

    @property
    def total(self) -> int:
        return self.active + self.warm + self.cool + self.frozen + self.crystal

    def as_dict(self) -> dict[str, int]:
        return {
            "active": self.active,
            "warm": self.warm,
            "cool": self.cool,
            "frozen": self.frozen,
            "crystal": self.crystal,
            "total": self.total,
        }


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TierManager:
    """Manages neuron compression tier lifecycle."""

    def __init__(
        self,
        dimensions: int,
        config: TierConfig | None = None,
    ) -> None:
        self._config = config or TierConfig()
        self._compressor = VectorCompressor(dimensions)
        self._dimensions = dimensions

    @property
    def config(self) -> TierConfig:
        return self._config

    @property
    def compressor(self) -> VectorCompressor:
        return self._compressor

    def classify_neuron(self, meta: dict[str, Any]) -> CompressionTier:
        """Determine the appropriate tier for a neuron based on its metadata."""
        priority = meta.get("priority", 5)
        access_count = meta.get("access_count", 0)
        accessed_at = meta.get("accessed_at", "")

        # Crystal: ephemeral expired or priority 0
        if priority <= 0:
            return CompressionTier.CRYSTAL

        # High priority always active
        if priority >= self._config.high_priority_threshold:
            return CompressionTier.ACTIVE

        # Calculate days since last access
        days_since_access = self._days_since(accessed_at)

        # Active: recently accessed with sufficient history
        if days_since_access <= self._config.warm_after_days:
            if access_count >= self._config.min_access_for_active or priority >= 7:
                return CompressionTier.ACTIVE
            return CompressionTier.WARM

        # Warm
        if days_since_access <= self._config.cool_after_days:
            return CompressionTier.WARM

        # Cool
        if days_since_access <= self._config.frozen_after_days:
            return CompressionTier.COOL

        # Frozen
        return CompressionTier.FROZEN

    def should_promote(
        self, meta: dict[str, Any], current_tier: CompressionTier
    ) -> CompressionTier | None:
        """Check if a neuron should be promoted to a higher tier.

        Returns the target tier if promotion needed, None otherwise.
        """
        if not self._config.auto_promote_on_access:
            return None

        ideal = self.classify_neuron(meta)
        if ideal < current_tier:  # lower enum value = higher quality
            return ideal
        return None

    def should_demote(
        self, meta: dict[str, Any], current_tier: CompressionTier
    ) -> CompressionTier | None:
        """Check if a neuron should be demoted to a lower tier.

        Returns the target tier if demotion needed, None otherwise.
        """
        if not self._config.auto_demote_enabled:
            return None

        ideal = self.classify_neuron(meta)
        if ideal > current_tier:  # higher enum value = lower quality
            return ideal
        return None

    def batch_classify(self, neurons: list[dict[str, Any]]) -> dict[CompressionTier, list[str]]:
        """Classify multiple neurons into tiers.

        Returns dict mapping tier -> list of neuron IDs.
        """
        result: dict[CompressionTier, list[str]] = {tier: [] for tier in CompressionTier}
        for meta in neurons:
            tier = self.classify_neuron(meta)
            nid = meta.get("id", "")
            if nid:
                result[tier].append(nid)
        return result

    def compute_stats(self, neurons: list[dict[str, Any]]) -> TierStats:
        """Compute tier distribution stats for a set of neurons."""
        counts = dict.fromkeys(CompressionTier, 0)
        for meta in neurons:
            current = meta.get("tier", CompressionTier.ACTIVE)
            if isinstance(current, int):
                try:
                    current = CompressionTier(current)
                except ValueError:
                    current = CompressionTier.ACTIVE
            counts[current] = counts.get(current, 0) + 1

        return TierStats(
            active=counts[CompressionTier.ACTIVE],
            warm=counts[CompressionTier.WARM],
            cool=counts[CompressionTier.COOL],
            frozen=counts[CompressionTier.FROZEN],
            crystal=counts[CompressionTier.CRYSTAL],
        )

    def estimate_savings(
        self, tier_stats: TierStats, dimensions: int | None = None
    ) -> dict[str, Any]:
        """Estimate storage savings from tiered compression vs all-active."""
        dims = dimensions or self._dimensions
        all_active_bytes = tier_stats.total * dims * 4  # float32

        actual_bytes = (
            self._compressor.estimate_size(CompressionTier.ACTIVE, tier_stats.active)
            + self._compressor.estimate_size(CompressionTier.WARM, tier_stats.warm)
            + self._compressor.estimate_size(CompressionTier.COOL, tier_stats.cool)
            + self._compressor.estimate_size(CompressionTier.FROZEN, tier_stats.frozen)
            + self._compressor.estimate_size(CompressionTier.CRYSTAL, tier_stats.crystal)
        )

        saved = all_active_bytes - actual_bytes
        ratio = all_active_bytes / actual_bytes if actual_bytes > 0 else float("inf")

        return {
            "all_active_bytes": all_active_bytes,
            "actual_bytes": actual_bytes,
            "saved_bytes": saved,
            "compression_ratio": round(ratio, 2),
            "savings_percent": round(saved / all_active_bytes * 100, 1)
            if all_active_bytes > 0
            else 0,
        }

    def _days_since(self, iso_str: str) -> float:
        """Calculate days since an ISO datetime string."""
        if not iso_str:
            return 999.0  # very old if no timestamp

        try:
            dt = datetime.fromisoformat(iso_str)
            delta = _utcnow() - dt
            return max(0, delta.total_seconds() / 86400)
        except (ValueError, TypeError):
            return 999.0
