"""Post-consolidation quality delta report.

Wraps DiagnosticsEngine + ConsolidationEngine to compute a before/after
health snapshot. Keeps the two engines decoupled — composition happens
here at the call site.

Usage:
    delta = await run_with_delta(storage, brain_id, strategies=strategies)
    print(delta.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.consolidation import (
        ConsolidationConfig,
        ConsolidationReport,
        ConsolidationStrategy,
    )
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class HealthSnapshot:
    """Frozen snapshot of key health metrics at a point in time."""

    purity_score: float
    grade: str
    connectivity: float
    diversity: float
    freshness: float
    consolidation_ratio: float
    orphan_rate: float
    activation_efficiency: float
    recall_confidence: float
    neuron_count: int
    synapse_count: int
    fiber_count: int


@dataclass(frozen=True)
class ConsolidationDelta:
    """Immutable before/after health comparison from a consolidation run.

    Attributes:
        before: Health snapshot before consolidation.
        after: Health snapshot after consolidation.
        report: The raw consolidation report.
        computed_at: When this delta was computed.
    """

    before: HealthSnapshot
    after: HealthSnapshot
    report: ConsolidationReport
    computed_at: datetime

    @property
    def purity_delta(self) -> float:
        """Change in purity score (positive = improvement)."""
        return round(self.after.purity_score - self.before.purity_score, 1)

    @property
    def connectivity_delta(self) -> float:
        return round(self.after.connectivity - self.before.connectivity, 4)

    @property
    def orphan_rate_delta(self) -> float:
        """Change in orphan rate (negative = improvement)."""
        return round(self.after.orphan_rate - self.before.orphan_rate, 4)

    def summary(self) -> str:
        """Human-readable delta summary."""
        lines = [self.report.summary(), "", "Health Delta:"]

        metrics = [
            ("Purity", self.before.purity_score, self.after.purity_score, self.purity_delta),
            (
                "Connectivity",
                self.before.connectivity,
                self.after.connectivity,
                self.connectivity_delta,
            ),
            (
                "Orphan rate",
                self.before.orphan_rate,
                self.after.orphan_rate,
                self.orphan_rate_delta,
            ),
            (
                "Freshness",
                self.before.freshness,
                self.after.freshness,
                round(self.after.freshness - self.before.freshness, 4),
            ),
            (
                "Consolidation ratio",
                self.before.consolidation_ratio,
                self.after.consolidation_ratio,
                round(self.after.consolidation_ratio - self.before.consolidation_ratio, 4),
            ),
        ]

        for label, before_val, after_val, delta in metrics:
            sign = "+" if delta > 0 else ""
            lines.append(f"  {label}: {before_val} -> {after_val} ({sign}{delta})")

        grade_changed = self.before.grade != self.after.grade
        if grade_changed:
            lines.append(f"  Grade: {self.before.grade} -> {self.after.grade}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize for MCP/CLI JSON output."""
        return {
            "before": {
                "purity_score": self.before.purity_score,
                "grade": self.before.grade,
                "connectivity": self.before.connectivity,
                "diversity": self.before.diversity,
                "orphan_rate": self.before.orphan_rate,
                "neuron_count": self.before.neuron_count,
                "synapse_count": self.before.synapse_count,
                "fiber_count": self.before.fiber_count,
            },
            "after": {
                "purity_score": self.after.purity_score,
                "grade": self.after.grade,
                "connectivity": self.after.connectivity,
                "diversity": self.after.diversity,
                "orphan_rate": self.after.orphan_rate,
                "neuron_count": self.after.neuron_count,
                "synapse_count": self.after.synapse_count,
                "fiber_count": self.after.fiber_count,
            },
            "delta": {
                "purity": self.purity_delta,
                "connectivity": self.connectivity_delta,
                "orphan_rate": self.orphan_rate_delta,
            },
            "grade_changed": self.before.grade != self.after.grade,
        }


def _snapshot_from_report(report: object) -> HealthSnapshot:
    """Extract a HealthSnapshot from a BrainHealthReport."""
    from neural_memory.engine.diagnostics import BrainHealthReport

    assert isinstance(report, BrainHealthReport)
    return HealthSnapshot(
        purity_score=report.purity_score,
        grade=report.grade,
        connectivity=report.connectivity,
        diversity=report.diversity,
        freshness=report.freshness,
        consolidation_ratio=report.consolidation_ratio,
        orphan_rate=report.orphan_rate,
        activation_efficiency=report.activation_efficiency,
        recall_confidence=report.recall_confidence,
        neuron_count=report.neuron_count,
        synapse_count=report.synapse_count,
        fiber_count=report.fiber_count,
    )


async def run_with_delta(
    storage: NeuralStorage,
    brain_id: str,
    *,
    strategies: list[ConsolidationStrategy] | None = None,
    dry_run: bool = False,
    config: ConsolidationConfig | None = None,
    reference_time: datetime | None = None,
) -> ConsolidationDelta:
    """Run consolidation and compute a before/after health delta.

    1. DiagnosticsEngine.analyze() -> before snapshot
    2. ConsolidationEngine.run() -> report
    3. DiagnosticsEngine.analyze() -> after snapshot
    4. Return ConsolidationDelta with all three

    Args:
        storage: Neural storage backend.
        brain_id: Brain to consolidate and diagnose.
        strategies: Consolidation strategies (default: ALL).
        dry_run: If True, preview only.
        config: Optional consolidation config overrides.
        reference_time: Reference time for consolidation.

    Returns:
        ConsolidationDelta with before/after snapshots and report.
    """
    from neural_memory.engine.consolidation import ConsolidationEngine
    from neural_memory.engine.diagnostics import DiagnosticsEngine

    diagnostics = DiagnosticsEngine(storage)
    engine = ConsolidationEngine(storage, config)

    # Step 1: Before snapshot
    before_report = await diagnostics.analyze(brain_id)
    before = _snapshot_from_report(before_report)

    # Step 2: Run consolidation
    report = await engine.run(
        strategies=strategies,
        dry_run=dry_run,
        reference_time=reference_time,
    )

    # Step 3: After snapshot
    after_report = await diagnostics.analyze(brain_id)
    after = _snapshot_from_report(after_report)

    return ConsolidationDelta(
        before=before,
        after=after,
        report=report,
        computed_at=utcnow(),
    )
