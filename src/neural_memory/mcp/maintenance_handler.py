"""Proactive brain maintenance handler for MCP server.

Piggybacks on remember/recall operations via an operation counter.
Every N ops, runs a cheap get_stats() query (<1ms), compares counts
against thresholds, and surfaces a maintenance_hint field in the response.
Optionally triggers auto-consolidation as fire-and-forget background task.

Smart features (v1.10):
- Severity-weighted health hints with recommended strategies
- Dynamic strategy selection based on actual health problems
- Adaptive check intervals (healthy brains check less, unhealthy more)
- Auto-dream trigger for low-connectivity brains
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from neural_memory.core.trigger_engine import TriggerResult, TriggerType
from neural_memory.utils.timeutils import utcnow
from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.unified_config import MaintenanceConfig

logger = logging.getLogger(__name__)


class HintSeverity(StrEnum):
    """Severity levels for health hints."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class HealthHint:
    """A structured health hint with severity and recommended action.

    Attributes:
        message: Human-readable description of the issue
        severity: How urgent this issue is
        recommended_strategy: Consolidation strategy to address it
    """

    message: str
    severity: HintSeverity
    recommended_strategy: str  # "prune", "merge", "enrich", "mature", "dream"


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
    hints: tuple[HealthHint, ...]
    should_consolidate: bool

    @property
    def hint_messages(self) -> tuple[str, ...]:
        """Backward-compatible string access to hint messages."""
        return tuple(h.message for h in self.hints)


class MaintenanceHandler:
    """Mixin: proactive brain maintenance for MCP server.

    Tracks operation count and periodically checks brain health
    using a cheap get_stats() query. Surfaces hints and optionally
    triggers auto-consolidation with dynamic strategy selection.
    """

    _op_count: int = 0
    _last_pulse: HealthPulse | None = None
    _last_consolidation_at: datetime | None = None
    _last_dream_at: datetime | None = None
    _effective_check_interval: int | None = None
    _consolidation_task: asyncio.Task[None] | None = None

    def _increment_op_counter(self) -> int:
        self._op_count += 1
        return self._op_count

    def _should_check_health(self) -> bool:
        cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]
        if not cfg.enabled:
            return False
        interval = self._effective_check_interval or cfg.check_interval
        return self._op_count > 0 and self._op_count % interval == 0

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

        connectivity = synapse_count / neuron_count if neuron_count > 0 else 0.0

        # Estimate orphan ratio: neurons not covered by fibers
        # Heuristic: each fiber typically creates ~5 neurons
        estimated_linked = fiber_count * 5
        orphan_ratio = (
            max(0.0, (neuron_count - estimated_linked) / neuron_count) if neuron_count > 0 else 0.0
        )

        # Expired memory count (cheap COUNT query)
        expired_memory_count = 0
        try:
            expired_memory_count = await storage.get_expired_memory_count()
        except Exception:
            logger.debug("Health pulse: get_expired_memory_count failed", exc_info=True)

        # Expiring-soon count (cheap COUNT query)
        expiring_soon_count = 0
        try:
            expiring_soon_count = await storage.get_expiring_memory_count(within_days=7)
        except Exception:
            logger.debug("Health pulse: get_expiring_memory_count failed", exc_info=True)

        # Stale fiber count (cheap COUNT query)
        stale_fiber_count = 0
        try:
            stale_fiber_count = await storage.get_stale_fiber_count(brain_id, cfg.stale_fiber_days)
        except Exception:
            logger.debug("Health pulse: get_stale_fiber_count failed", exc_info=True)

        stale_fiber_ratio = stale_fiber_count / fiber_count if fiber_count > 0 else 0.0

        hints = _evaluate_thresholds(
            fiber_count=fiber_count,
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            connectivity=connectivity,
            orphan_ratio=orphan_ratio,
            expired_memory_count=expired_memory_count,
            expiring_soon_count=expiring_soon_count,
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
        return pulse.hints[0].message

    async def _maybe_auto_consolidate(self, pulse: HealthPulse) -> None:
        """Fire-and-forget auto-consolidation with dynamic strategy selection."""
        cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]

        if not cfg.auto_consolidate or not pulse.should_consolidate:
            return

        now = utcnow()

        # Compute effective cooldown based on severity
        effective_cooldown_minutes = _compute_effective_cooldown(
            pulse.hints, cfg.consolidate_cooldown_minutes
        )

        # Skip if a prior consolidation is still running
        if self._consolidation_task is not None and not self._consolidation_task.done():
            logger.debug("Auto-consolidation skipped: previous task still running")
            return

        if self._last_consolidation_at is not None:
            cooldown = timedelta(minutes=effective_cooldown_minutes)
            if now - self._last_consolidation_at < cooldown:
                logger.debug("Auto-consolidation skipped: cooldown active")
                return

        # Dynamic strategy selection from hints
        strategies = _select_strategies(pulse.hints)

        # Auto-dream injection for low-connectivity brains
        strategies = self._maybe_inject_dream(strategies, pulse, cfg, now)

        self._last_consolidation_at = now
        self._consolidation_task = asyncio.create_task(
            self._run_auto_consolidation_dynamic(strategies)
        )

    def _maybe_inject_dream(
        self,
        strategies: tuple[str, ...],
        pulse: HealthPulse,
        cfg: MaintenanceConfig,
        now: datetime,
    ) -> tuple[str, ...]:
        """Inject DREAM strategy if connectivity is low and cooldown expired.

        Dream exploration discovers hidden connections via random spreading
        activation. Only fires when:
        - connectivity < 1.5
        - neuron_count >= 50
        - dream cooldown has expired (default 24h)
        """
        if pulse.connectivity >= 1.5 or pulse.neuron_count < 50:
            return strategies

        if "dream" in strategies:
            return strategies

        dream_cooldown = timedelta(hours=cfg.dream_cooldown_hours)
        if self._last_dream_at is not None and now - self._last_dream_at < dream_cooldown:
            return strategies

        self._last_dream_at = now
        logger.info(
            "Auto-dream triggered: connectivity=%.2f, neurons=%d",
            pulse.connectivity,
            pulse.neuron_count,
        )
        return (*strategies, "dream")

    async def _run_auto_consolidation_dynamic(self, strategy_names: tuple[str, ...]) -> None:
        """Background task: run dynamically selected consolidation strategies."""
        try:
            from neural_memory.engine.consolidation import ConsolidationStrategy
            from neural_memory.engine.consolidation_delta import run_with_delta

            storage = await self.get_storage()  # type: ignore[attr-defined]
            brain_id = _require_brain_id(storage)
            strategies = [ConsolidationStrategy(s) for s in strategy_names]
            delta = await run_with_delta(storage, brain_id, strategies=strategies)
            logger.info(
                "Auto-consolidation complete (strategies=%s): %s | purity delta: %+.1f",
                strategy_names,
                delta.report.summary(),
                delta.purity_delta,
            )

            # Reset adaptive interval after successful consolidation
            self._effective_check_interval = None
        except Exception:
            logger.error("Auto-consolidation failed", exc_info=True)

    async def _run_auto_consolidation(self, cfg: MaintenanceConfig) -> None:
        """Background task: run lightweight consolidation strategies (legacy)."""
        await self._run_auto_consolidation_dynamic(cfg.auto_consolidate_strategies)

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
            message=pulse.hints[0].message,
            save_tiers=(3,),
        )
        logger.info(
            "Health degradation detected (op #%d): %s [%s]",
            self._op_count,
            pulse.hints[0].message,
            pulse.hints[0].severity,
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

            # Adapt check interval based on health
            cfg: MaintenanceConfig = self.config.maintenance  # type: ignore[attr-defined]
            self._effective_check_interval = _compute_adaptive_interval(
                len(pulse.hints), cfg.check_interval
            )

            if pulse.should_consolidate:
                await self._maybe_auto_consolidate(pulse)

            # Trigger expiry cleanup if due
            await self._maybe_run_expiry_cleanup()  # type: ignore[attr-defined]

            # Persist alerts from health pulse + auto-resolve cleared conditions
            try:
                await self._create_alerts_from_pulse(pulse)  # type: ignore[attr-defined]
                await self._auto_resolve_cleared(pulse)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Alert persistence failed", exc_info=True)

        return pulse


def _evaluate_thresholds(
    *,
    fiber_count: int,
    neuron_count: int,
    synapse_count: int,
    connectivity: float,
    orphan_ratio: float,
    expired_memory_count: int = 0,
    expiring_soon_count: int = 0,
    stale_fiber_ratio: float = 0.0,
    cfg: MaintenanceConfig,
) -> list[HealthHint]:
    """Evaluate health thresholds and return structured hints with severity."""
    hints: list[HealthHint] = []

    if neuron_count > cfg.neuron_warn_threshold:
        hints.append(
            HealthHint(
                message=f"High neuron count ({neuron_count}). "
                "Consider running consolidation with prune strategy.",
                severity=HintSeverity.MEDIUM,
                recommended_strategy="prune",
            )
        )

    if fiber_count > cfg.fiber_warn_threshold:
        hints.append(
            HealthHint(
                message=f"High fiber count ({fiber_count}). "
                "Consider running consolidation with merge strategy.",
                severity=HintSeverity.MEDIUM,
                recommended_strategy="merge",
            )
        )

    if synapse_count > cfg.synapse_warn_threshold:
        hints.append(
            HealthHint(
                message=f"High synapse count ({synapse_count}). "
                "Consider running consolidation with prune strategy.",
                severity=HintSeverity.MEDIUM,
                recommended_strategy="prune",
            )
        )

    if neuron_count >= 10 and connectivity < 1.5:
        hints.append(
            HealthHint(
                message=f"Low connectivity ({connectivity:.1f} synapses/neuron). "
                "Consider running consolidation with enrich strategy.",
                severity=HintSeverity.LOW,
                recommended_strategy="enrich",
            )
        )

    if neuron_count >= 10 and orphan_ratio > cfg.orphan_ratio_threshold:
        pct = int(orphan_ratio * 100)
        severity = HintSeverity.CRITICAL if orphan_ratio > 0.4 else HintSeverity.MEDIUM
        hints.append(
            HealthHint(
                message=f"High orphan ratio ({pct}%). "
                "Consider running nmem_health for diagnostics.",
                severity=severity,
                recommended_strategy="prune",
            )
        )

    if expired_memory_count > cfg.expired_memory_warn_threshold:
        hints.append(
            HealthHint(
                message=f"{expired_memory_count} expired memories found. "
                "Consider cleanup via nmem list --expired.",
                severity=HintSeverity.LOW,
                recommended_strategy="prune",
            )
        )

    if expiring_soon_count > 0:
        hints.append(
            HealthHint(
                message=f"{expiring_soon_count} memories expiring within 7 days. "
                "Use nmem_recall with warn_expiry_days=7 to identify them.",
                severity=HintSeverity.LOW,
                recommended_strategy="none",
            )
        )

    if fiber_count >= 10 and stale_fiber_ratio > cfg.stale_fiber_ratio_threshold:
        pct = round(stale_fiber_ratio * 100)
        hints.append(
            HealthHint(
                message=f"{pct}% of fibers are stale (>{cfg.stale_fiber_days} days unused). "
                "Consider running nmem_health for review.",
                severity=HintSeverity.LOW,
                recommended_strategy="prune",
            )
        )

    return hints


def _compute_effective_cooldown(
    hints: tuple[HealthHint, ...],
    base_cooldown_minutes: int,
) -> int:
    """Compute cooldown based on hint severity.

    More hints or higher severity = shorter cooldown (more urgent consolidation).

    Returns:
        Effective cooldown in minutes.
    """
    if not hints:
        return base_cooldown_minutes

    has_critical = any(h.severity == HintSeverity.CRITICAL for h in hints)
    hint_count = len(hints)

    # 4+ hints OR any CRITICAL: force immediate (0 cooldown)
    if hint_count >= 4 or has_critical:
        return 0

    # 2-3 hints: halved cooldown
    if hint_count >= 2:
        return base_cooldown_minutes // 2

    # 1 hint LOW/MEDIUM: full cooldown
    return base_cooldown_minutes


def _select_strategies(hints: tuple[HealthHint, ...]) -> tuple[str, ...]:
    """Select consolidation strategies from hints, ordered by severity.

    Deduplicates strategies and orders them by the highest severity
    hint that recommended each strategy.

    Returns:
        Tuple of strategy names, highest-severity first.
        Falls back to ("prune", "merge") if no hints.
    """
    if not hints:
        return ("prune", "merge")

    severity_order = {
        HintSeverity.CRITICAL: 0,
        HintSeverity.HIGH: 1,
        HintSeverity.MEDIUM: 2,
        HintSeverity.LOW: 3,
    }

    # Track best severity per strategy
    strategy_severity: dict[str, HintSeverity] = {}
    for hint in hints:
        existing = strategy_severity.get(hint.recommended_strategy)
        if existing is None or severity_order[hint.severity] < severity_order[existing]:
            strategy_severity[hint.recommended_strategy] = hint.severity

    # Sort by severity (highest first)
    sorted_strategies = sorted(
        strategy_severity.items(),
        key=lambda item: severity_order[item[1]],
    )

    return tuple(s for s, _ in sorted_strategies)


def _compute_adaptive_interval(hint_count: int, base_interval: int) -> int:
    """Compute adaptive check interval based on brain health.

    Healthy brains are checked less frequently (fewer wasted cycles).
    Unhealthy brains are checked more often (faster response).

    Args:
        hint_count: Number of health hints from last pulse
        base_interval: Configured base check interval

    Returns:
        Adapted interval in operations.
    """
    # 0 hints (healthy): double the interval, cap at 50
    if hint_count == 0:
        return min(base_interval * 2, 50)

    # 1-2 hints (degrading): keep base interval
    if hint_count <= 2:
        return base_interval

    # 3+ hints (unhealthy): reduce to 40% of base, minimum 5
    return max(int(base_interval / 2.5), 5)
