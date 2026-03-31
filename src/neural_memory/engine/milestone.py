"""Brain milestone detection and growth analysis.

Tracks neuron-count milestones (100, 250, 500, 1000, 2500, 5000, 10000+)
and generates growth reports at each threshold. Reports include health
snapshots, growth velocity, and key achievements.

No LLM required — all reports are template-rendered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.diagnostics import BrainHealthReport
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Milestone thresholds (neuron counts)
MILESTONES: tuple[int, ...] = (100, 250, 500, 1000, 2500, 5000, 10000)

# Milestone names — each threshold has a title and focus description
MILESTONE_META: dict[int, tuple[str, str]] = {
    100: ("First Hundred", "Your brain's foundation — early patterns and core knowledge."),
    250: ("Growing Network", "Connections forming — topics clustering into neighborhoods."),
    500: ("Half Thousand", "Consolidation matters — episodic memories should mature."),
    1000: ("Thousand Minds", "A rich knowledge base — diversity and recall depth deepen."),
    2500: ("Deep Archive", "Long-term patterns emerge — review and prune for quality."),
    5000: ("Neural Library", "Massive recall surface — efficiency and health are critical."),
    10000: ("Cognitive Atlas", "Expert-level memory — a comprehensive map of knowledge."),
}


@dataclass(frozen=True)
class MilestoneSnapshot:
    """Snapshot of brain state at a milestone.

    Stored in brain metadata under ``_milestones`` key.
    """

    threshold: int
    achieved_at: datetime
    neuron_count: int
    synapse_count: int
    fiber_count: int
    purity_score: float
    grade: str
    days_from_first_memory: int
    days_from_prev_milestone: int | None
    top_types: dict[str, int]  # neuron type -> count


@dataclass(frozen=True)
class MilestoneReport:
    """Complete milestone growth report with markdown output."""

    snapshot: MilestoneSnapshot
    title: str
    description: str
    growth_velocity: float  # neurons per day since prev milestone
    health_delta: float | None  # purity change since prev milestone
    achievements: tuple[str, ...]
    markdown: str


def _get_next_milestone(neuron_count: int) -> int | None:
    """Return the next unachieved milestone threshold, or None."""
    for m in MILESTONES:
        if neuron_count < m:
            return m
    return None


def _get_achieved_milestones(neuron_count: int) -> list[int]:
    """Return all milestone thresholds the neuron count has passed."""
    return [m for m in MILESTONES if neuron_count >= m]


class MilestoneEngine:
    """Detects milestones and generates growth reports.

    Milestone snapshots are persisted in brain metadata under the
    ``_milestones`` key as a list of dicts. This avoids schema changes.
    """

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def check_and_record(
        self,
        brain_id: str,
    ) -> MilestoneReport | None:
        """Check if a new milestone was reached and record it.

        Returns a MilestoneReport if a new milestone was achieved, else None.
        """
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return None

        stats = await self._storage.get_enhanced_stats(brain_id)
        neuron_count: int = stats.get("neuron_count", 0)

        # Load existing milestone records
        recorded = self._load_milestones(brain.metadata)
        recorded_thresholds = {m["threshold"] for m in recorded}

        # Find newly achieved milestones
        achieved = _get_achieved_milestones(neuron_count)
        new_milestones = [m for m in achieved if m not in recorded_thresholds]

        if not new_milestones:
            return None

        # Process only the highest new milestone (skip intermediate)
        target = max(new_milestones)

        # Run health diagnostics
        from neural_memory.engine.diagnostics import DiagnosticsEngine

        health = await DiagnosticsEngine(self._storage).analyze(brain_id)

        # Compute time metrics
        first_memory_date = await self._get_first_memory_date()
        now = utcnow()
        days_from_first = (now - first_memory_date).days if first_memory_date else 0

        prev_snapshot = recorded[-1] if recorded else None
        days_from_prev: int | None = None
        if prev_snapshot:
            prev_achieved = datetime.fromisoformat(prev_snapshot["achieved_at"])
            days_from_prev = (now - prev_achieved).days

        # Get neuron type distribution
        top_types = self._get_type_distribution(stats)

        snapshot = MilestoneSnapshot(
            threshold=target,
            achieved_at=now,
            neuron_count=neuron_count,
            synapse_count=stats.get("synapse_count", 0),
            fiber_count=stats.get("fiber_count", 0),
            purity_score=health.purity_score,
            grade=health.grade,
            days_from_first_memory=days_from_first,
            days_from_prev_milestone=days_from_prev,
            top_types=top_types,
        )

        # Build report
        report = self._build_report(snapshot, health, prev_snapshot)

        # Collect all snapshots to persist (intermediates + target)
        all_snapshots: list[MilestoneSnapshot] = []
        for m in sorted(new_milestones):
            if m != target:
                skipped = MilestoneSnapshot(
                    threshold=m,
                    achieved_at=now,
                    neuron_count=neuron_count,
                    synapse_count=stats.get("synapse_count", 0),
                    fiber_count=stats.get("fiber_count", 0),
                    purity_score=health.purity_score,
                    grade=health.grade,
                    days_from_first_memory=days_from_first,
                    days_from_prev_milestone=None,
                    top_types=top_types,
                )
                all_snapshots.append(skipped)
        all_snapshots.append(snapshot)

        # Persist all in one save
        await self._save_milestones_batch(brain, all_snapshots)

        return report

    async def get_history(self, brain_id: str) -> list[dict[str, Any]]:
        """Return all recorded milestones for a brain."""
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return []
        return self._load_milestones(brain.metadata)

    async def get_progress(self, brain_id: str) -> dict[str, Any]:
        """Return progress toward next milestone."""
        stats = await self._storage.get_enhanced_stats(brain_id)
        neuron_count: int = stats.get("neuron_count", 0)

        next_ms = _get_next_milestone(neuron_count)
        if next_ms is None:
            return {
                "neuron_count": neuron_count,
                "next_milestone": None,
                "message": "All milestones achieved! You've built a Cognitive Atlas.",
            }

        remaining = next_ms - neuron_count
        pct = round(neuron_count / next_ms * 100, 1)
        meta = MILESTONE_META.get(next_ms, ("Unknown", ""))

        return {
            "neuron_count": neuron_count,
            "next_milestone": next_ms,
            "remaining": remaining,
            "progress_pct": pct,
            "milestone_title": meta[0],
            "milestone_description": meta[1],
        }

    async def generate_report(self, brain_id: str) -> MilestoneReport | None:
        """Force-generate a report for the current brain state (no persistence)."""
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return None

        stats = await self._storage.get_enhanced_stats(brain_id)
        neuron_count: int = stats.get("neuron_count", 0)
        if neuron_count == 0:
            return None

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        health = await DiagnosticsEngine(self._storage).analyze(brain_id)
        first_memory_date = await self._get_first_memory_date()
        now = utcnow()
        days_from_first = (now - first_memory_date).days if first_memory_date else 0
        top_types = self._get_type_distribution(stats)

        # Find closest milestone
        achieved = _get_achieved_milestones(neuron_count)
        threshold = max(achieved) if achieved else neuron_count

        recorded = self._load_milestones(brain.metadata)
        prev_snapshot = recorded[-1] if recorded else None

        snapshot = MilestoneSnapshot(
            threshold=threshold,
            achieved_at=now,
            neuron_count=neuron_count,
            synapse_count=stats.get("synapse_count", 0),
            fiber_count=stats.get("fiber_count", 0),
            purity_score=health.purity_score,
            grade=health.grade,
            days_from_first_memory=days_from_first,
            days_from_prev_milestone=None,
            top_types=top_types,
        )

        return self._build_report(snapshot, health, prev_snapshot)

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _load_milestones(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Load milestone list from brain metadata."""
        result: list[dict[str, Any]] = metadata.get("_milestones", [])
        return result

    async def _save_milestones_batch(self, brain: Any, snapshots: list[MilestoneSnapshot]) -> None:
        """Persist multiple milestone snapshots to brain metadata in one save."""
        from dataclasses import asdict
        from dataclasses import replace as dc_replace

        metadata = {**brain.metadata}
        milestones = list(metadata.get("_milestones", []))

        for snapshot in snapshots:
            record = asdict(snapshot)
            record["achieved_at"] = snapshot.achieved_at.isoformat()
            milestones.append(record)

        metadata["_milestones"] = milestones
        updated_brain = dc_replace(brain, metadata=metadata, updated_at=utcnow())
        await self._storage.save_brain(updated_brain)

    async def _get_first_memory_date(self) -> datetime | None:
        """Get the creation date of the oldest fiber."""
        fibers = await self._storage.get_fibers(limit=1, descending=False)
        if not fibers:
            return None
        return fibers[0].created_at

    @staticmethod
    def _get_type_distribution(stats: dict[str, Any]) -> dict[str, int]:
        """Extract synapse type counts from pre-fetched enhanced stats."""
        type_counts: dict[str, int] = {}
        try:
            synapse_stats = stats.get("synapse_stats", {})
            by_type = synapse_stats.get("by_type", {})
            for stype, info in by_type.items():
                count = info["count"] if isinstance(info, dict) else info
                type_counts[str(stype)] = count
        except Exception:
            logger.debug("Type distribution failed (non-critical)", exc_info=True)
        return type_counts

    def _build_report(
        self,
        snapshot: MilestoneSnapshot,
        health: BrainHealthReport,
        prev_snapshot: dict[str, Any] | None,
    ) -> MilestoneReport:
        """Build a MilestoneReport with markdown."""
        meta = MILESTONE_META.get(snapshot.threshold, ("Milestone", "A new milestone reached."))
        title = meta[0]
        description = meta[1]

        # Growth velocity
        velocity = 0.0
        if (
            snapshot.days_from_prev_milestone
            and snapshot.days_from_prev_milestone > 0
            and prev_snapshot
        ):
            prev_neurons = prev_snapshot.get("neuron_count", 0)
            velocity = (snapshot.neuron_count - prev_neurons) / snapshot.days_from_prev_milestone

        # Health delta
        health_delta: float | None = None
        if prev_snapshot and "purity_score" in prev_snapshot:
            health_delta = snapshot.purity_score - prev_snapshot["purity_score"]

        # Achievements
        achievements = self._detect_achievements(snapshot, health, prev_snapshot)

        # Render markdown
        md = self._render_markdown(
            snapshot, title, description, velocity, health_delta, achievements
        )

        return MilestoneReport(
            snapshot=snapshot,
            title=title,
            description=description,
            growth_velocity=round(velocity, 2),
            health_delta=health_delta,
            achievements=tuple(achievements),
            markdown=md,
        )

    @staticmethod
    def _detect_achievements(
        snapshot: MilestoneSnapshot,
        health: BrainHealthReport,
        prev_snapshot: dict[str, Any] | None,
    ) -> list[str]:
        """Detect notable achievements at this milestone."""
        achievements: list[str] = []

        # Grade-based achievements
        if health.grade == "A":
            achievements.append("Brain health grade A — excellent quality!")
        elif health.grade == "B":
            achievements.append("Brain health grade B — strong knowledge base.")

        # Connectivity achievement
        if snapshot.synapse_count > 0 and snapshot.neuron_count > 0:
            ratio = snapshot.synapse_count / snapshot.neuron_count
            if ratio >= 5.0:
                achievements.append(f"Rich connectivity: {ratio:.1f} synapses per neuron.")
            elif ratio >= 3.0:
                achievements.append(f"Healthy connectivity: {ratio:.1f} synapses per neuron.")

        # Diversity achievement
        if len(snapshot.top_types) >= 6:
            achievements.append(f"Diverse knowledge: {len(snapshot.top_types)} synapse types used.")

        # Improvement since last milestone
        if prev_snapshot and "purity_score" in prev_snapshot:
            delta = snapshot.purity_score - prev_snapshot["purity_score"]
            if delta >= 10:
                achievements.append(f"Health improved by {delta:.1f} points since last milestone!")
            elif delta >= 5:
                achievements.append(f"Health improved by {delta:.1f} points.")

        # Speed achievement
        if snapshot.days_from_first_memory > 0:
            rate = snapshot.neuron_count / snapshot.days_from_first_memory
            if rate >= 50:
                achievements.append(f"Rapid growth: {rate:.0f} neurons/day average.")
            elif rate >= 20:
                achievements.append(f"Steady growth: {rate:.0f} neurons/day average.")

        return achievements

    @staticmethod
    def _render_markdown(
        snapshot: MilestoneSnapshot,
        title: str,
        description: str,
        velocity: float,
        health_delta: float | None,
        achievements: list[str],
    ) -> str:
        """Render milestone report as markdown."""
        lines: list[str] = []
        lines.append(f"# Milestone: {title}")
        lines.append(f"**{snapshot.threshold} neurons reached!**")
        lines.append("")
        lines.append(f"> {description}")
        lines.append("")

        # Stats snapshot
        lines.append("## Brain Snapshot")
        lines.append(f"- Neurons: **{snapshot.neuron_count:,}**")
        lines.append(f"- Synapses: **{snapshot.synapse_count:,}**")
        lines.append(f"- Fibers: **{snapshot.fiber_count:,}**")
        lines.append(f"- Health: **{snapshot.grade}** ({snapshot.purity_score:.1f}/100)")
        if snapshot.days_from_first_memory > 0:
            lines.append(f"- Brain age: **{snapshot.days_from_first_memory}** days")
        lines.append("")

        # Growth metrics
        if velocity > 0 or health_delta is not None:
            lines.append("## Growth")
            if velocity > 0:
                lines.append(f"- Growth velocity: **{velocity:.1f}** neurons/day")
            if health_delta is not None:
                direction = "+" if health_delta >= 0 else ""
                lines.append(f"- Health change: **{direction}{health_delta:.1f}** points")
            if snapshot.days_from_prev_milestone is not None:
                lines.append(
                    f"- Days since last milestone: **{snapshot.days_from_prev_milestone}**"
                )
            lines.append("")

        # Achievements
        if achievements:
            lines.append("## Achievements")
            for a in achievements:
                lines.append(f"- {a}")
            lines.append("")

        # Type distribution
        if snapshot.top_types:
            lines.append("## Knowledge Distribution")
            sorted_types = sorted(snapshot.top_types.items(), key=lambda x: x[1], reverse=True)
            for stype, count in sorted_types[:6]:
                lines.append(f"- {stype}: {count}")
            lines.append("")

        # Next milestone
        next_ms = _get_next_milestone(snapshot.neuron_count)
        if next_ms:
            remaining = next_ms - snapshot.neuron_count
            next_meta = MILESTONE_META.get(next_ms, ("Next", ""))
            lines.append(f"## Next: {next_meta[0]} ({next_ms:,} neurons)")
            lines.append(f"{remaining:,} neurons to go. {next_meta[1]}")
        else:
            lines.append("## All Milestones Complete!")
            lines.append("You've built a Cognitive Atlas. The journey continues.")

        return "\n".join(lines)
