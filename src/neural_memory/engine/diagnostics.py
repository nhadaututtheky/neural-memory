"""Brain health diagnostics and quality analysis.

Computes composite purity score, individual metrics, and
actionable warnings from the neural graph structure.
Supports both MCP and CLI exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.memory_stages import MemoryStage
from neural_memory.utils.tag_normalizer import TagNormalizer
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


# ── Data structures ──────────────────────────────────────────────


class WarningSeverity(StrEnum):
    """Severity level for diagnostic warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DiagnosticWarning:
    """A single diagnostic warning with severity and context."""

    severity: WarningSeverity
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityBadge:
    """Quality badge for marketplace eligibility.

    Computed from brain health diagnostics.
    Stored in Brain.metadata["_quality_badge"] when computed.
    """

    grade: str
    purity_score: float
    marketplace_eligible: bool
    badge_label: str
    computed_at: datetime
    component_summary: dict[str, float]


@dataclass(frozen=True)
class BrainHealthReport:
    """Complete brain health diagnostics report.

    All component scores are normalized to [0.0, 1.0].
    Purity score is a weighted composite in [0, 100].
    """

    # Overall health
    purity_score: float
    grade: str

    # Component scores (0.0-1.0)
    connectivity: float
    diversity: float
    freshness: float
    consolidation_ratio: float
    orphan_rate: float
    activation_efficiency: float
    recall_confidence: float

    # Raw counts
    neuron_count: int
    synapse_count: int
    fiber_count: int

    # Diagnostics
    warnings: tuple[DiagnosticWarning, ...]
    recommendations: tuple[str, ...]


# ── Grade mapping ────────────────────────────────────────────────


def _score_to_grade(score: float) -> str:
    """Map purity score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ── Diagnostics engine ──────────────────────────────────────────


class DiagnosticsEngine:
    """Brain health diagnostics and quality analysis.

    Computes composite purity score, individual metrics, and
    actionable warnings from the neural graph structure.
    """

    # Number of defined synapse types (for diversity normalization)
    _TOTAL_SYNAPSE_TYPES = len(SynapseType)

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def analyze(self, brain_id: str) -> BrainHealthReport:
        """Run full brain diagnostics.

        Args:
            brain_id: ID of the brain to analyze

        Returns:
            BrainHealthReport with scores, warnings, and recommendations
        """
        # Gather base data
        enhanced = await self._storage.get_enhanced_stats(brain_id)
        neuron_count: int = enhanced.get("neuron_count", 0)
        synapse_count: int = enhanced.get("synapse_count", 0)
        fiber_count: int = enhanced.get("fiber_count", 0)

        # Early return for empty brain
        if neuron_count == 0 or fiber_count == 0:
            return self._empty_brain_report(neuron_count, synapse_count, fiber_count)

        # Compute individual metrics
        synapse_stats = enhanced.get("synapse_stats", {})

        # Fetch fibers once for freshness + diagnostics
        fibers = await self._storage.get_fibers(limit=10000)

        connectivity = self._compute_connectivity(synapse_count, neuron_count)
        diversity = self._compute_diversity(synapse_stats)
        freshness = self._compute_freshness(fibers)
        consolidation_ratio = await self._compute_consolidation_ratio(fiber_count)
        orphan_rate = await self._compute_orphan_rate(neuron_count)
        activation_efficiency = await self._compute_activation_efficiency(neuron_count)
        recall_confidence = self._compute_recall_confidence(synapse_stats)

        # Compute purity score
        purity = (
            connectivity * 0.25
            + diversity * 0.20
            + freshness * 0.15
            + consolidation_ratio * 0.15
            + (1.0 - orphan_rate) * 0.10
            + activation_efficiency * 0.10
            + recall_confidence * 0.05
        ) * 100

        # Apply penalty for unresolved CONTRADICTS synapses
        by_type = synapse_stats.get("by_type", {})
        contradicts_entry = by_type.get(SynapseType.CONTRADICTS, {})
        contradicts_count: int = (
            (
                contradicts_entry["count"]
                if isinstance(contradicts_entry, dict)
                else contradicts_entry
            )
            if contradicts_entry
            else 0
        )
        conflict_rate = contradicts_count / max(neuron_count, 1)
        conflict_penalty = min(10.0, conflict_rate * 50.0)  # Max 10 point penalty
        purity = max(0.0, purity - conflict_penalty)

        grade = _score_to_grade(purity)

        # Generate warnings and recommendations
        raw_connectivity = synapse_count / max(neuron_count, 1)
        warnings, recommendations = self._generate_diagnostics(
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            raw_connectivity=raw_connectivity,
            synapse_stats=synapse_stats,
            orphan_rate=orphan_rate,
            consolidation_ratio=consolidation_ratio,
            freshness=freshness,
            fibers=fibers,
            contradicts_count=contradicts_count,
        )

        return BrainHealthReport(
            purity_score=round(purity, 1),
            grade=grade,
            connectivity=round(connectivity, 4),
            diversity=round(diversity, 4),
            freshness=round(freshness, 4),
            consolidation_ratio=round(consolidation_ratio, 4),
            orphan_rate=round(orphan_rate, 4),
            activation_efficiency=round(activation_efficiency, 4),
            recall_confidence=round(recall_confidence, 4),
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            warnings=tuple(warnings),
            recommendations=tuple(recommendations),
        )

    # ── Metric computations ──────────────────────────────────────

    @staticmethod
    def _compute_connectivity(synapse_count: int, neuron_count: int) -> float:
        """Compute connectivity score via sigmoid normalization.

        Target: 3-8 synapses per neuron is ideal.
        sigmoid(-1.5 * (x - 3)): at x=0 -> ~0.01, x=3 -> 0.5, x=8 -> ~1.0
        """
        if neuron_count == 0:
            return 0.0
        raw = synapse_count / neuron_count
        return 1.0 / (1.0 + math.exp(-1.5 * (raw - 3.0)))

    # Synapse types realistically expected in typical usage.
    # Spatial/semantic types only appear with specialized content.
    _EXPECTED_SYNAPSE_TYPES = 8

    @staticmethod
    def _compute_diversity(synapse_stats: dict[str, Any]) -> float:
        """Compute synapse type diversity via Shannon entropy.

        Normalized against log(expected_types) rather than all defined types,
        since most brains won't use spatial/semantic types without specialized
        content. Using all 20 types as baseline unfairly penalizes typical usage.
        """
        by_type = synapse_stats.get("by_type", {})
        if not by_type:
            return 0.0

        type_counts = [
            entry["count"] if isinstance(entry, dict) else entry for entry in by_type.values()
        ]
        total = sum(type_counts)
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in type_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        expected_types = DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES
        max_entropy = math.log(expected_types) if expected_types > 1 else 1.0
        return min(1.0, entropy / max_entropy)

    @staticmethod
    def _compute_freshness(fibers: list[Any]) -> float:
        """Compute fraction of fibers accessed/created in last 7 days."""
        if not fibers:
            return 0.0

        now = utcnow()
        cutoff = now - timedelta(days=7)
        fresh_count = sum(1 for f in fibers if (f.last_conducted or f.created_at) >= cutoff)
        return fresh_count / len(fibers)

    async def _compute_consolidation_ratio(self, fiber_count: int) -> float:
        """Compute fraction of fibers that reached SEMANTIC stage."""
        if fiber_count == 0:
            return 0.0
        semantic_records = await self._storage.find_maturations(
            stage=MemoryStage.SEMANTIC,
        )
        return len(semantic_records) / fiber_count

    async def _compute_orphan_rate(self, neuron_count: int) -> float:
        """Compute fraction of neurons with zero synapses."""
        if neuron_count == 0:
            return 0.0

        all_synapses = await self._storage.get_all_synapses()
        connected: set[str] = set()
        for s in all_synapses:
            connected.add(s.source_id)
            connected.add(s.target_id)

        orphan_count = max(0, neuron_count - len(connected))
        return orphan_count / neuron_count

    async def _compute_activation_efficiency(self, neuron_count: int) -> float:
        """Compute fraction of neurons that have been activated at least once.

        Proxy metric: neurons with access_frequency > 0 indicate the brain
        is actively utilizing its neural graph during retrieval.
        """
        if neuron_count == 0:
            return 0.0

        states = await self._storage.get_all_neuron_states()
        activated_count = sum(1 for s in states if s.access_frequency > 0)
        return activated_count / max(neuron_count, 1)

    @staticmethod
    def _compute_recall_confidence(synapse_stats: dict[str, Any]) -> float:
        """Compute recall confidence from average synapse weight.

        Higher average weight indicates stronger recall pathways.
        """
        avg_weight: float = synapse_stats.get("avg_weight", 0.0)
        return min(1.0, max(0.0, avg_weight))

    # ── Warning and recommendation generation ────────────────────

    def _generate_diagnostics(
        self,
        *,
        neuron_count: int,
        synapse_count: int,
        fiber_count: int,
        raw_connectivity: float,
        synapse_stats: dict[str, Any],
        orphan_rate: float,
        consolidation_ratio: float,
        freshness: float,
        fibers: list[Any],
        contradicts_count: int = 0,
    ) -> tuple[list[DiagnosticWarning], list[str]]:
        """Generate warnings and recommendations from metrics."""
        warnings: list[DiagnosticWarning] = []
        recommendations: list[str] = []

        # Stale brain
        if fiber_count > 0 and freshness == 0.0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.CRITICAL,
                    code="STALE_BRAIN",
                    message="No fibers accessed or created in the last 7 days.",
                )
            )
            recommendations.append("Brain is stale - store new memories to keep it active.")

        # Low connectivity
        if raw_connectivity < 2.0 and neuron_count > 0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="LOW_CONNECTIVITY",
                    message=f"Low connectivity: {raw_connectivity:.1f} synapses/neuron (target: 3-8).",
                )
            )
            recommendations.append("Store more detailed memories to build richer connections.")

        # Low diversity
        by_type = synapse_stats.get("by_type", {})
        types_used = len(by_type)
        expected = DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES
        if types_used < 3 and synapse_count > 0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="LOW_DIVERSITY",
                    message=f"Low synapse diversity: {types_used} of {expected} expected types used.",
                    details={"types_used": types_used, "types_expected": expected},
                )
            )
            recommendations.append(
                "Store memories with causal, temporal, and emotional context for richer synapses."
            )

        # High orphan rate
        if orphan_rate > 0.20:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="HIGH_ORPHAN_RATE",
                    message=f"High orphan rate: {orphan_rate:.0%} of neurons have no connections.",
                )
            )
            recommendations.append(
                "Run consolidation with prune strategy to clean up orphaned neurons."
            )

        # No consolidation
        if consolidation_ratio == 0.0 and fiber_count > 0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="NO_CONSOLIDATION",
                    message="No memories have reached SEMANTIC stage.",
                )
            )
            recommendations.append(
                "Run consolidation with mature strategy to advance episodic memories."
            )

        # Tag drift detection
        all_tags: set[str] = set()
        for fiber in fibers:
            all_tags |= fiber.tags
        if all_tags:
            normalizer = TagNormalizer()
            drift_reports = normalizer.detect_drift(all_tags)
            for drift in drift_reports:
                warnings.append(
                    DiagnosticWarning(
                        severity=WarningSeverity.INFO,
                        code="TAG_DRIFT",
                        message=f"Tag drift: {drift.variants} -> '{drift.canonical}'",
                        details={
                            "canonical": drift.canonical,
                            "variants": drift.variants,
                        },
                    )
                )
            if drift_reports:
                recommendations.append("Normalize tags to reduce semantic drift.")

        # High conflict count (unresolved CONTRADICTS synapses)
        if contradicts_count > 5:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="HIGH_CONFLICT_COUNT",
                    message=f"{contradicts_count} unresolved memory conflicts detected.",
                    details={"count": contradicts_count},
                )
            )
            recommendations.append(
                "Run `nmem_conflicts` to review and resolve memory contradictions."
            )

        return warnings, recommendations

    # ── Quality badge ────────────────────────────────────────────

    async def compute_quality_badge(self, brain_id: str) -> QualityBadge:
        """Compute a quality badge for the brain.

        Runs full diagnostics and maps the result to a marketplace-ready badge.

        Args:
            brain_id: ID of the brain to evaluate

        Returns:
            QualityBadge with grade, purity score, and eligibility
        """
        report = await self.analyze(brain_id)

        badge_labels = {
            "A": "A - Excellent",
            "B": "B - Good",
            "C": "C - Fair",
            "D": "D - Poor",
            "F": "F - Failing",
        }

        return QualityBadge(
            grade=report.grade,
            purity_score=report.purity_score,
            marketplace_eligible=report.grade in ("A", "B"),
            badge_label=badge_labels.get(report.grade, f"{report.grade} - Unknown"),
            computed_at=utcnow(),
            component_summary={
                "connectivity": report.connectivity,
                "diversity": report.diversity,
                "freshness": report.freshness,
                "consolidation_ratio": report.consolidation_ratio,
                "orphan_rate": report.orphan_rate,
                "activation_efficiency": report.activation_efficiency,
                "recall_confidence": report.recall_confidence,
            },
        )

    # ── Empty brain helper ───────────────────────────────────────

    @staticmethod
    def _empty_brain_report(
        neuron_count: int,
        synapse_count: int,
        fiber_count: int,
    ) -> BrainHealthReport:
        """Return a minimal report for an empty brain."""
        return BrainHealthReport(
            purity_score=0.0,
            grade="F",
            connectivity=0.0,
            diversity=0.0,
            freshness=0.0,
            consolidation_ratio=0.0,
            orphan_rate=0.0,
            activation_efficiency=0.0,
            recall_confidence=0.0,
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            warnings=(
                DiagnosticWarning(
                    severity=WarningSeverity.CRITICAL,
                    code="EMPTY_BRAIN",
                    message="Brain has no memories stored.",
                ),
            ),
            recommendations=("Start storing memories with nmem_remember.",),
        )
