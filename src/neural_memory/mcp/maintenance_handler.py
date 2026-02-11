"""Proactive brain maintenance handler for MCP server.

Piggybacks on remember/recall operations via an operation counter.
Every N ops, runs a cheap get_stats() query (<1ms), compares counts
against thresholds, and surfaces a maintenance_hint field in the response.
Optionally triggers auto-consolidation as fire-and-forget background task.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from neural_memory.core.trigger_engine import TriggerResult, TriggerType

if TYPE_CHECKING:
    from neural_memory.unified_config import MaintenanceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthPulse:
    """Result of a lightweight health check."""

    fiber_count: int
    neuron_count: int
    synapse_count: int
    connectivity: float
    orphan_ratio: float
    expired_memory_count: int
    stale_fiber_ratio: float
    hints: tuple[str, ...]
    should_consolidate: bool


class MaintenanceHandler:
    """Mixin: proactive brain maintenance for MCP server.

    Tracks operation count and periodically checks brain health
    using a cheap get_stats() query. Surfaces hints and optionally
    triggers auto-consolidation.
    """

    _op_count: int = 0
    _last_pulse: HealthPulse | None = None
    _last_consolidation_at: datetime | None = None

    def _increment_op_counter(self) -> int:
        self._op_count += 1
        return self._op_count

    def _should_check_health(self) -> bool:
        cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]
        if not cfg.enabled:
            return False
        return self._op_count > 0 and self._op_count % cfg.check_interval == 0

    async def _health_pulse(self) -> HealthPulse | None:
        """Run a cheap health check using get_stats().

        Returns None if maintenance is disabled or stats unavailable.
        """
        cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]
        if not cfg.enabled:
            return None

        try:
            storage = await self.get_storage()  # type: ignore[attr-defined]
            brain_id = storage._current_brain_id
            stats: dict[str, int] = await storage.get_stats(brain_id)
        except Exception:
            logger.debug("Health pulse: get_stats failed", exc_info=True)
            return None

        fiber_count = stats.get("fiber_count", 0)
        neuron_count = stats.get("neuron_count", 0)
        synapse_count = stats.get("synapse_count", 0)

        connectivity = (
            synapse_count / neuron_count if neuron_count > 0 else 0.0
        )

        # Estimate orphan ratio: neurons not covered by fibers
        # Heuristic: each fiber typically creates ~5 neurons
        estimated_linked = fiber_count * 5
        orphan_ratio = (
            max(0.0, (neuron_count - estimated_linked) / neuron_count)
            if neuron_count > 0
            else 0.0
        )

        # Expired memory count (cheap COUNT query)
        expired_memory_count = 0
        try:
            expired_memory_count = await storage.get_expired_memory_count()
        except Exception:
            logger.debug("Health pulse: get_expired_memory_count failed", exc_info=True)

        # Stale fiber count (cheap COUNT query)
        stale_fiber_count = 0
        try:
            stale_fiber_count = await storage.get_stale_fiber_count(
                brain_id, cfg.stale_fiber_days
            )
        except Exception:
            logger.debug("Health pulse: get_stale_fiber_count failed", exc_info=True)

        stale_fiber_ratio = (
            stale_fiber_count / fiber_count if fiber_count > 0 else 0.0
        )

        hints = _evaluate_thresholds(
            fiber_count=fiber_count,
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            connectivity=connectivity,
            orphan_ratio=orphan_ratio,
            expired_memory_count=expired_memory_count,
            stale_fiber_ratio=stale_fiber_ratio,
            cfg=cfg,
        )

        should_consolidate = len(hints) > 0 and cfg.auto_consolidate

        pulse = HealthPulse(
            fiber_count=fiber_count,
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            connectivity=round(connectivity, 2),
            orphan_ratio=round(orphan_ratio, 2),
            expired_memory_count=expired_memory_count,
            stale_fiber_ratio=round(stale_fiber_ratio, 2),
            hints=tuple(hints),
            should_consolidate=should_consolidate,
        )
        self._last_pulse = pulse
        return pulse

    def _get_maintenance_hint(self, pulse: HealthPulse | None) -> str | None:
        """Format a single maintenance hint string from pulse results."""
        if pulse is None or not pulse.hints:
            return None
        return pulse.hints[0]

    async def _maybe_auto_consolidate(self, pulse: HealthPulse) -> None:
        """Fire-and-forget auto-consolidation if enabled and off cooldown."""
        cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]

        if not cfg.auto_consolidate or not pulse.should_consolidate:
            return

        now = datetime.now(UTC)
        if self._last_consolidation_at is not None:
            cooldown = timedelta(minutes=cfg.consolidate_cooldown_minutes)
            if now - self._last_consolidation_at < cooldown:
                logger.debug("Auto-consolidation skipped: cooldown active")
                return

        self._last_consolidation_at = now
        self._consolidation_task = asyncio.create_task(self._run_auto_consolidation(cfg))

    async def _run_auto_consolidation(self, cfg: MaintenanceConfig) -> None:
        """Background task: run lightweight consolidation strategies."""
        try:
            from neural_memory.engine.consolidation import (
                ConsolidationEngine,
                ConsolidationStrategy,
            )

            storage = await self.get_storage()  # type: ignore[attr-defined]
            strategies = [
                ConsolidationStrategy(s) for s in cfg.auto_consolidate_strategies
            ]
            engine = ConsolidationEngine(storage)
            report = await engine.run(strategies=strategies)
            logger.info(
                "Auto-consolidation complete: %s", report.summary()
            )
        except Exception:
            logger.error("Auto-consolidation failed", exc_info=True)

    def _fire_health_trigger(self, pulse: HealthPulse) -> TriggerResult:
        """Create a HEALTH_DEGRADATION trigger when pulse has hints.

        Returns a TriggerResult for logging/tracking. Does not trigger
        eternal auto-save — maintenance hints are surfaced inline.
        """
        if not pulse.hints:
            return TriggerResult(triggered=False)

        result = TriggerResult(
            triggered=True,
            trigger_type=TriggerType.HEALTH_DEGRADATION,
            message=pulse.hints[0],
            save_tiers=(3,),
        )
        logger.info(
            "Health degradation detected (op #%d): %s",
            self._op_count,
            pulse.hints[0],
        )
        return result

    async def _check_maintenance(self) -> HealthPulse | None:
        """Orchestrator: increment counter, check health if due.

        Called from _remember() and _recall() in the server.
        Returns a HealthPulse if a check was performed, None otherwise.
        """
        self._increment_op_counter()

        if not self._should_check_health():
            return None

        pulse = await self._health_pulse()
        if pulse is not None:
            self._fire_health_trigger(pulse)
            if pulse.should_consolidate:
                await self._maybe_auto_consolidate(pulse)

        return pulse


def _evaluate_thresholds(
    *,
    fiber_count: int,
    neuron_count: int,
    synapse_count: int,
    connectivity: float,
    orphan_ratio: float,
    expired_memory_count: int = 0,
    stale_fiber_ratio: float = 0.0,
    cfg: MaintenanceConfig,
) -> list[str]:
    """Evaluate health thresholds and return hint strings."""
    hints: list[str] = []

    if neuron_count > cfg.neuron_warn_threshold:
        hints.append(
            f"High neuron count ({neuron_count}). "
            "Consider running consolidation with prune strategy."
        )

    if fiber_count > cfg.fiber_warn_threshold:
        hints.append(
            f"High fiber count ({fiber_count}). "
            "Consider running consolidation with merge strategy."
        )

    if synapse_count > cfg.synapse_warn_threshold:
        hints.append(
            f"High synapse count ({synapse_count}). "
            "Consider running consolidation with prune strategy."
        )

    if neuron_count >= 10 and connectivity < 1.5:
        hints.append(
            f"Low connectivity ({connectivity:.1f} synapses/neuron). "
            "Consider running consolidation with enrich strategy."
        )

    if neuron_count >= 10 and orphan_ratio > cfg.orphan_ratio_threshold:
        pct = int(orphan_ratio * 100)
        hints.append(
            f"High orphan ratio ({pct}%). "
            "Consider running nmem_health for diagnostics."
        )

    if expired_memory_count > cfg.expired_memory_warn_threshold:
        hints.append(
            f"{expired_memory_count} expired memories found. "
            "Consider cleanup via nmem list --expired."
        )

    if fiber_count >= 10 and stale_fiber_ratio > cfg.stale_fiber_ratio_threshold:
        pct = round(stale_fiber_ratio * 100)
        hints.append(
            f"{pct}% of fibers are stale (>{cfg.stale_fiber_days} days unused). "
            "Consider running nmem_health for review."
        )

    return hints
