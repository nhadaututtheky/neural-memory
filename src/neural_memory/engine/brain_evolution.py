"""Brain evolution dynamics — cognitive metrics layer.

Measures how brains EVOLVE through usage, not just their static health.
DiagnosticsEngine measures "state" (purity, grade).
EvolutionEngine measures "process" (maturation, plasticity, topology).

Core insight that separates NM from RAG:
    RAG: retrieval does NOT change knowledge topology.
    NM:  retrieval modifies synapse weights, fiber conductivity,
         maturation stage, and consolidation probability.

BrainEvolution captures this difference quantitatively.

Metrics design (4 expert rounds):
    1. semantic_ratio      — maturation progress (spacing effect)
    2. reinforcement_days  — temporal consistency of usage
    3. topology_coherence  — graph structure quality
    4. plasticity_index    — is the brain still learning or frozen

Agent self-awareness uses 3 low-dimensional signals derived from
these metrics: maturity_level, plasticity, density.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from neural_memory.engine.memory_stages import MemoryStage
from neural_memory.engine.topology_analysis import compute_topology
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


class ProficiencyLevel(StrEnum):
    """Brain proficiency level, earned through usage over time.

    Not set manually — determined by maturation metrics, topology
    quality, and temporal consistency (spacing effect).
    """

    JUNIOR = "junior"
    SENIOR = "senior"
    EXPERT = "expert"


@dataclass(frozen=True)
class SemanticProgress:
    """Progress of an individual fiber toward SEMANTIC promotion.

    Attributes:
        fiber_id: The fiber identifier.
        stage: Current maturation stage.
        days_in_stage: Days since entering current stage.
        days_required: Days required for next promotion.
        reinforcement_days: Distinct calendar days with reinforcement.
        reinforcement_required: Minimum distinct days needed.
        progress_pct: Overall progress toward SEMANTIC (0.0-1.0).
        next_step: Actionable text describing what the user should do.
    """

    fiber_id: str
    stage: str
    days_in_stage: float
    days_required: float
    reinforcement_days: int
    reinforcement_required: int
    progress_pct: float
    next_step: str


@dataclass(frozen=True)
class StageDistribution:
    """Distribution of fibers across maturation stages.

    Attributes:
        short_term: Fibers at SHORT_TERM stage.
        working: Fibers at WORKING stage.
        episodic: Fibers at EPISODIC stage.
        semantic: Fibers at SEMANTIC stage.
        total: Total fibers with maturation records.
    """

    short_term: int
    working: int
    episodic: int
    semantic: int
    total: int


@dataclass(frozen=True)
class BrainEvolution:
    """Brain evolution snapshot — cognitive dynamics metrics.

    Attributes:
        brain_id: Brain identifier.
        brain_name: Human-readable brain name.
        computed_at: When this snapshot was computed.

        semantic_ratio: Fibers at SEMANTIC stage / total maturation records.
        reinforcement_days: Average distinct reinforcement days per fiber.
        topology_coherence: Graph structure quality (clustering + LCC).
        plasticity_index: Learning activity in last 7 days (0.0-1.0).
        knowledge_density: Synapses per neuron (raw ratio).

        maturity_level: Agent signal — 0.0-1.0 from maturation metrics.
        plasticity: Agent signal — 0.0-1.0 from plasticity_index.
        density: Agent signal — 0.0-1.0 normalized knowledge_density.

        proficiency_index: Composite score 0-100.
        proficiency_level: JUNIOR / SENIOR / EXPERT.
        activity_score: Recent retrievals / total fibers.

        total_fibers: Total fiber count.
        total_synapses: Total synapse count.
        total_neurons: Total neuron count.
        fibers_at_semantic: Fibers that reached SEMANTIC stage.
        fibers_at_episodic: Fibers at EPISODIC stage.

        stage_distribution: Breakdown of fibers by maturation stage.
        closest_to_semantic: Top fibers closest to SEMANTIC promotion.
    """

    brain_id: str
    brain_name: str
    computed_at: datetime

    # Internal metrics
    semantic_ratio: float
    reinforcement_days: float
    topology_coherence: float
    plasticity_index: float
    knowledge_density: float

    # Agent-facing signals (low-dimensional)
    maturity_level: float
    plasticity: float
    density: float

    # Composite
    proficiency_index: int
    proficiency_level: ProficiencyLevel
    activity_score: float

    # Counts
    total_fibers: int
    total_synapses: int
    total_neurons: int
    fibers_at_semantic: int
    fibers_at_episodic: int

    # Maturation progress
    stage_distribution: StageDistribution | None = None
    closest_to_semantic: tuple[SemanticProgress, ...] = ()


class EvolutionEngine:
    """Computes brain evolution dynamics from the neural graph.

    Uses only existing storage infrastructure — no new tables needed.
    All metrics derived from MaturationRecord, Fiber, and Synapse fields.
    """

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def analyze(self, brain_id: str) -> BrainEvolution:
        """Compute evolution metrics for a brain.

        Gathers data from maturation records, fibers, synapses,
        and topology analysis, then computes composite proficiency.
        Pre-fetches shared data to avoid redundant storage calls.
        """
        now = utcnow()

        # Parallel fetch: brain metadata, stats, synapses, fibers
        brain, stats, all_synapses, all_fibers = await asyncio.gather(
            self._storage.get_brain(brain_id),
            self._storage.get_stats(brain_id),
            self._storage.get_all_synapses(),
            self._storage.get_fibers(limit=10000),
        )
        brain_name = brain.name if brain else "unknown"
        neuron_count = stats.get("neuron_count", 0)
        synapse_count = stats.get("synapse_count", 0)
        fiber_count = stats.get("fiber_count", 0)

        # Maturation metrics
        (
            semantic_ratio,
            reinforcement_days,
            fibers_semantic,
            fibers_episodic,
        ) = await self._compute_maturation()

        # Topology metrics (pass pre-fetched synapses)
        topo = await compute_topology(
            self._storage, brain_id, _preloaded_synapses=all_synapses  # type: ignore[arg-type]
        )
        topology_coherence = topo.clustering_coefficient * 0.5 + topo.largest_component_ratio * 0.5
        knowledge_density = topo.knowledge_density

        # Plasticity (uses pre-fetched synapses)
        plasticity_index = _compute_plasticity(all_synapses, now)

        # Activity (uses pre-fetched fibers)
        activity_score = _compute_activity(all_fibers, now, fiber_count)

        # Agent-facing signals (normalized 0-1)
        maturity_level = _maturity_signal(semantic_ratio, reinforcement_days)
        plasticity_signal = min(1.0, plasticity_index * 5.0)
        density_signal = min(1.0, knowledge_density / 5.0)

        # Composite proficiency (uses pre-fetched fibers)
        decay_factor = _compute_decay(all_fibers, now)
        proficiency_index, proficiency_level = _compute_proficiency(
            semantic_ratio=semantic_ratio,
            reinforcement_days=reinforcement_days,
            topology_coherence=topology_coherence,
            plasticity_index=plasticity_index,
            decay_factor=decay_factor,
        )

        # Stage distribution and semantic progress
        stage_dist, closest = await self._compute_stage_progress(now)

        return BrainEvolution(
            brain_id=brain_id,
            brain_name=brain_name,
            computed_at=now,
            semantic_ratio=round(semantic_ratio, 4),
            reinforcement_days=round(reinforcement_days, 2),
            topology_coherence=round(topology_coherence, 4),
            plasticity_index=round(plasticity_index, 4),
            knowledge_density=round(knowledge_density, 4),
            maturity_level=round(maturity_level, 4),
            plasticity=round(plasticity_signal, 4),
            density=round(density_signal, 4),
            proficiency_index=proficiency_index,
            proficiency_level=proficiency_level,
            activity_score=round(activity_score, 4),
            total_fibers=fiber_count,
            total_synapses=synapse_count,
            total_neurons=neuron_count,
            fibers_at_semantic=fibers_semantic,
            fibers_at_episodic=fibers_episodic,
            stage_distribution=stage_dist,
            closest_to_semantic=closest,
        )

    # ── Internal computations ────────────────────────────────

    async def _compute_stage_progress(
        self,
        now: datetime,
    ) -> tuple[StageDistribution, tuple[SemanticProgress, ...]]:
        """Compute stage distribution and progress toward SEMANTIC.

        Returns:
            (StageDistribution, top 3 fibers closest to SEMANTIC promotion)
        """
        all_maturations = await self._storage.find_maturations()

        counts = {
            MemoryStage.SHORT_TERM: 0,
            MemoryStage.WORKING: 0,
            MemoryStage.EPISODIC: 0,
            MemoryStage.SEMANTIC: 0,
        }
        for m in all_maturations:
            if m.stage in counts:
                counts[m.stage] += 1

        stage_dist = StageDistribution(
            short_term=counts[MemoryStage.SHORT_TERM],
            working=counts[MemoryStage.WORKING],
            episodic=counts[MemoryStage.EPISODIC],
            semantic=counts[MemoryStage.SEMANTIC],
            total=len(all_maturations),
        )

        # Compute progress for EPISODIC fibers (closest to SEMANTIC)
        days_required = 7.0
        reinf_required = 3
        progress_items: list[SemanticProgress] = []

        for m in all_maturations:
            if m.stage != MemoryStage.EPISODIC:
                continue

            days_in = (now - m.stage_entered_at).total_seconds() / 86400
            reinf_days = m.distinct_reinforcement_days

            time_pct = min(1.0, days_in / days_required)
            reinf_pct = min(1.0, reinf_days / reinf_required)
            overall_pct = min(time_pct, reinf_pct)

            if reinf_days < reinf_required:
                next_step = (
                    f"Recall this memory on {reinf_required - reinf_days} more "
                    f"distinct day(s) to reach SEMANTIC."
                )
            elif days_in < days_required:
                remaining = days_required - days_in
                next_step = f"Wait ~{remaining:.1f} more day(s) in EPISODIC stage."
            else:
                next_step = "Ready for promotion — run consolidation with mature strategy."

            progress_items.append(
                SemanticProgress(
                    fiber_id=m.fiber_id,
                    stage=m.stage.value,
                    days_in_stage=round(days_in, 1),
                    days_required=days_required,
                    reinforcement_days=reinf_days,
                    reinforcement_required=reinf_required,
                    progress_pct=round(overall_pct, 2),
                    next_step=next_step,
                )
            )

        # Sort by progress descending, take top 3
        progress_items.sort(key=lambda p: p.progress_pct, reverse=True)
        return stage_dist, tuple(progress_items[:3])

    async def _compute_maturation(
        self,
    ) -> tuple[float, float, int, int]:
        """Compute maturation metrics from MaturationRecord data.

        Returns:
            (semantic_ratio, avg_reinforcement_days, semantic_count, episodic_count)
        """
        all_maturations = await self._storage.find_maturations()
        if not all_maturations:
            return 0.0, 0.0, 0, 0

        semantic_count = sum(1 for m in all_maturations if m.stage == MemoryStage.SEMANTIC)
        episodic_count = sum(1 for m in all_maturations if m.stage == MemoryStage.EPISODIC)
        semantic_ratio = semantic_count / len(all_maturations)

        # Average distinct reinforcement days across all fibers
        total_days = sum(m.distinct_reinforcement_days for m in all_maturations)
        avg_days = total_days / len(all_maturations)

        return semantic_ratio, avg_days, semantic_count, episodic_count


# ── Pure functions ───────────────────────────────────────────────


def _compute_plasticity(all_synapses: Sequence[Any], now: datetime) -> float:
    """Compute plasticity index: learning activity in last 7 days.

    Counts distinct synapses that are either new OR reinforced in the
    last 7 days (each synapse counted at most once).

    Returns:
        Value in [0.0, 1.0] — fraction of synapses with recent activity.
    """
    if not all_synapses:
        return 0.0

    cutoff = now - timedelta(days=7)
    total = len(all_synapses)

    active_7d = sum(
        1
        for s in all_synapses
        if s.created_at >= cutoff or (s.last_activated and s.last_activated >= cutoff)
    )

    return active_7d / total


def _compute_activity(
    all_fibers: Sequence[Any],
    now: datetime,
    fiber_count: int,
) -> float:
    """Compute activity score: recent retrievals / total fibers.

    Measures whether the brain is actively being used.
    """
    if fiber_count == 0:
        return 0.0

    cutoff = now - timedelta(days=7)

    active_7d = sum(1 for f in all_fibers if f.last_conducted and f.last_conducted >= cutoff)

    return active_7d / fiber_count


def _compute_decay(all_fibers: Sequence[Any], now: datetime) -> float:
    """Compute decay factor from most recent fiber retrieval.

    Proficiency decays if brain isn't used (sigmoid centered at 30 days):
    - ~0.95 at 0 days since last use
    - ~0.91 at 7 days
    - 0.50 at 30 days
    - ~0.05 at 60 days
    """
    if not all_fibers:
        return 0.5  # No fibers = neutral decay

    most_recent = None
    for f in all_fibers:
        ts = f.last_conducted or f.created_at
        if most_recent is None or ts > most_recent:
            most_recent = ts

    if most_recent is None:
        return 0.5

    days_since = (now - most_recent).total_seconds() / 86400
    # Sigmoid decay: center at 30 days, steepness factor 0.1
    return 1.0 / (1.0 + math.exp(0.1 * (days_since - 30)))


def _maturity_signal(semantic_ratio: float, reinforcement_days: float) -> float:
    """Compute agent-facing maturity signal (0.0-1.0).

    Combines semantic_ratio (how many fibers matured) with
    reinforcement_days (spacing effect evidence).
    """
    ratio_signal = min(1.0, semantic_ratio * 2.0)  # 0.5 ratio = 1.0 signal
    days_signal = min(1.0, reinforcement_days / 7.0)  # 7 days = 1.0 signal
    return ratio_signal * 0.6 + days_signal * 0.4


def _compute_proficiency(
    *,
    semantic_ratio: float,
    reinforcement_days: float,
    topology_coherence: float,
    plasticity_index: float,
    decay_factor: float,
) -> tuple[int, ProficiencyLevel]:
    """Compute composite proficiency index and level.

    Weighted composite with decay, then mapped to level with
    AND conditions to prevent gaming (high semantic but low usage).

    Returns:
        (proficiency_index 0-100, ProficiencyLevel)
    """
    # Normalize inputs to 0-1
    days_norm = min(1.0, reinforcement_days / 10.0)
    plasticity_norm = min(1.0, plasticity_index * 5.0)

    raw_score = (
        semantic_ratio * 0.30
        + days_norm * 0.25
        + topology_coherence * 0.25
        + plasticity_norm * 0.20
    )

    index = round(raw_score * decay_factor * 100)
    index = max(0, min(100, index))

    # Level with AND conditions
    if index > 55 and reinforcement_days >= 10:
        level = ProficiencyLevel.EXPERT
    elif index >= 25 and reinforcement_days >= 4:
        level = ProficiencyLevel.SENIOR
    else:
        level = ProficiencyLevel.JUNIOR

    return index, level
