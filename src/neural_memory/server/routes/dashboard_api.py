"""Dashboard API routes — brain stats, health, brain management, timeline, diagrams."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from neural_memory.server.dependencies import get_storage, require_local_request
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(require_local_request)],
)


class BrainSummary(BaseModel):
    """Brief brain summary for dashboard listing."""

    id: str
    name: str
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    grade: str = "F"
    purity_score: float = 0.0
    is_active: bool = False


class DashboardStats(BaseModel):
    """Dashboard overview statistics."""

    active_brain: str | None = None
    total_brains: int = 0
    total_neurons: int = 0
    total_synapses: int = 0
    total_fibers: int = 0
    health_grade: str = "F"
    purity_score: float = 0.0
    brains: list[BrainSummary] = Field(default_factory=list)


class HealthReport(BaseModel):
    """Brain health report for the radar chart."""

    grade: str
    purity_score: float
    connectivity: float = 0.0
    diversity: float = 0.0
    freshness: float = 0.0
    consolidation_ratio: float = 0.0
    orphan_rate: float = 0.0
    activation_efficiency: float = 0.0
    recall_confidence: float = 0.0
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class SwitchBrainRequest(BaseModel):
    """Request to switch active brain."""

    brain_name: str = Field(..., min_length=1)


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get dashboard overview stats",
)
async def get_stats(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> DashboardStats:
    """Get overall dashboard statistics across all brains."""
    from neural_memory.unified_config import get_config

    cfg = get_config()
    brain_names = cfg.list_brains()
    active_name = cfg.current_brain

    async def _analyze_brain(name: str) -> BrainSummary:
        """Analyze a single brain and return its summary."""
        try:
            stats = await storage.get_stats(name)
            nc = stats.get("neuron_count", 0)
            sc = stats.get("synapse_count", 0)
            fc = stats.get("fiber_count", 0)

            grade = "F"
            purity = 0.0
            try:
                from neural_memory.engine.diagnostics import DiagnosticsEngine

                diag = DiagnosticsEngine(storage)
                report = await diag.analyze(name)
                grade = report.grade
                purity = report.purity_score
            except Exception:
                logger.debug("Diagnostics failed for brain %s", name, exc_info=True)

            return BrainSummary(
                id=name,
                name=name,
                neuron_count=nc,
                synapse_count=sc,
                fiber_count=fc,
                grade=grade,
                purity_score=purity,
                is_active=name == active_name,
            )
        except Exception:
            logger.debug("Brain analysis failed for %s", name, exc_info=True)
            return BrainSummary(id=name, name=name, is_active=name == active_name)

    brains = list(await asyncio.gather(*[_analyze_brain(name) for name in brain_names]))

    total_n = sum(b.neuron_count for b in brains)
    total_s = sum(b.synapse_count for b in brains)
    total_f = sum(b.fiber_count for b in brains)
    active_grade = "F"
    active_purity = 0.0
    for b in brains:
        if b.is_active:
            active_grade = b.grade
            active_purity = b.purity_score
            break

    return DashboardStats(
        active_brain=active_name,
        total_brains=len(brain_names),
        total_neurons=total_n,
        total_synapses=total_s,
        total_fibers=total_f,
        health_grade=active_grade,
        purity_score=active_purity,
        brains=brains,
    )


@router.get(
    "/brains",
    response_model=list[BrainSummary],
    summary="List all brains",
)
async def list_brains_api(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> list[BrainSummary]:
    """List all available brains with summary stats."""
    from neural_memory.unified_config import get_config

    cfg = get_config()
    brain_names = cfg.list_brains()
    active_name = cfg.current_brain
    results: list[BrainSummary] = []

    for name in brain_names:
        try:
            stats = await storage.get_stats(name)
            results.append(
                BrainSummary(
                    id=name,
                    name=name,
                    neuron_count=stats.get("neuron_count", 0),
                    synapse_count=stats.get("synapse_count", 0),
                    fiber_count=stats.get("fiber_count", 0),
                    is_active=name == active_name,
                )
            )
        except Exception:
            logger.debug("Failed to get stats for brain %s", name, exc_info=True)
            results.append(BrainSummary(id=name, name=name, is_active=name == active_name))

    return results


@router.post(
    "/brains/switch",
    summary="Switch active brain",
)
async def switch_brain(request: SwitchBrainRequest) -> dict[str, str]:
    """Switch the active brain."""
    from neural_memory.unified_config import get_config

    cfg = get_config()
    available = cfg.list_brains()
    if request.brain_name not in available:
        raise HTTPException(
            status_code=404,
            detail="Brain not found.",
        )

    cfg.switch_brain(request.brain_name)
    return {"status": "switched", "active_brain": request.brain_name}


@router.get(
    "/health",
    response_model=HealthReport,
    summary="Get active brain health report",
)
async def get_health(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> HealthReport:
    """Run full diagnostics on the active brain."""
    from neural_memory.engine.diagnostics import DiagnosticsEngine
    from neural_memory.unified_config import get_config

    brain_name = get_config().current_brain

    try:
        diag = DiagnosticsEngine(storage)
        report = await diag.analyze(brain_name)
    except Exception as exc:
        logger.warning("Diagnostics failed for brain %s: %s", brain_name, exc)
        return HealthReport(grade="F", purity_score=0.0)

    return HealthReport(
        grade=report.grade,
        purity_score=report.purity_score,
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
        warnings=[
            {
                "severity": w.severity.value,
                "code": w.code,
                "message": w.message,
                "details": w.details,
            }
            for w in report.warnings
        ],
        recommendations=list(report.recommendations),
    )


# ── Timeline API ─────────────────────────────────────────


class TimelineEntry(BaseModel):
    """A single timeline entry."""

    id: str
    content: str
    neuron_type: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    """Timeline API response."""

    entries: list[TimelineEntry] = Field(default_factory=list)
    total: int = 0


@router.get(
    "/timeline",
    response_model=TimelineResponse,
    summary="Get chronological memory timeline",
)
async def get_timeline(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: int = Query(default=500, ge=1, le=2000),
    start: str | None = Query(default=None, description="ISO datetime start"),
    end: str | None = Query(default=None, description="ISO datetime end"),
) -> TimelineResponse:
    """Get chronological list of memories for timeline visualization."""
    neurons = await storage.find_neurons(limit=min(limit, 2000))

    entries: list[TimelineEntry] = []
    for n in neurons:
        created = n.metadata.get("_created_at", "") if n.metadata else ""
        if not created and hasattr(n, "created_at") and n.created_at:
            created = (
                n.created_at.isoformat()
                if hasattr(n.created_at, "isoformat")
                else str(n.created_at)
            )

        if start and created and created < start:
            continue
        if end and created and created > end:
            continue

        entries.append(
            TimelineEntry(
                id=n.id,
                content=n.content or "",
                neuron_type=n.type.value,
                created_at=created,
                metadata=n.metadata or {},
            )
        )

    # Sort by created_at descending
    entries.sort(key=lambda e: e.created_at, reverse=True)

    return TimelineResponse(entries=entries[:limit], total=len(entries))


# ── Fiber Diagram API ────────────────────────────────────


class FiberListItem(BaseModel):
    """Brief fiber summary for dropdown."""

    id: str
    summary: str
    neuron_count: int = 0


class FiberListResponse(BaseModel):
    """Fiber list API response."""

    fibers: list[FiberListItem] = Field(default_factory=list)


@router.get(
    "/fibers",
    response_model=FiberListResponse,
    summary="List fibers for dropdown",
)
async def list_fibers(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: int = Query(default=100, ge=1, le=500),
) -> FiberListResponse:
    """Get lightweight fiber list for diagram dropdown."""
    fibers = await storage.get_fibers(limit=min(limit, 500))

    return FiberListResponse(
        fibers=[
            FiberListItem(
                id=f.id,
                summary=f.summary or f.id[:20],
                neuron_count=len(f.neuron_ids) if f.neuron_ids else 0,
            )
            for f in fibers
        ]
    )


class FiberDiagramResponse(BaseModel):
    """Fiber diagram data for Mermaid rendering."""

    fiber_id: str
    neurons: list[dict[str, Any]] = Field(default_factory=list)
    synapses: list[dict[str, Any]] = Field(default_factory=list)


@router.get(
    "/fiber/{fiber_id}/diagram",
    response_model=FiberDiagramResponse,
    summary="Get fiber structure for diagram",
)
async def get_fiber_diagram(
    fiber_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> FiberDiagramResponse:
    """Get neurons and synapses for a fiber to render as a diagram."""
    target = await storage.get_fiber(fiber_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Fiber not found.")

    neuron_ids = list(target.neuron_ids) if target.neuron_ids else []
    if not neuron_ids:
        return FiberDiagramResponse(fiber_id=fiber_id, neurons=[], synapses=[])

    neurons_batch = await storage.get_neurons_batch(neuron_ids)

    neuron_list = [
        {
            "id": n.id,
            "type": n.type.value,
            "content": n.content or "",
            "metadata": n.metadata or {},
        }
        for n in neurons_batch.values()
    ]

    # Get synapses between this fiber's neurons using targeted batch query
    id_set = set(neuron_ids)
    outgoing = await storage.get_synapses_for_neurons(neuron_ids, direction="out")
    fiber_synapses = [
        {
            "id": s.id,
            "source_id": s.source_id,
            "target_id": s.target_id,
            "type": s.type.value,
            "weight": s.weight,
            "direction": s.direction.value,
        }
        for synapse_list in outgoing.values()
        for s in synapse_list
        if s.target_id in id_set
    ]

    return FiberDiagramResponse(
        fiber_id=fiber_id,
        neurons=neuron_list,
        synapses=fiber_synapses,
    )


# ── Evolution API ────────────────────────────────────


class SemanticProgressItem(BaseModel):
    """Progress of a fiber toward SEMANTIC stage."""

    fiber_id: str
    stage: str
    days_in_stage: float
    days_required: float
    reinforcement_days: int
    reinforcement_required: int
    progress_pct: float
    next_step: str


class StageDistributionResponse(BaseModel):
    """Distribution of fibers across maturation stages."""

    short_term: int = 0
    working: int = 0
    episodic: int = 0
    semantic: int = 0
    total: int = 0


class EvolutionResponse(BaseModel):
    """Brain evolution metrics for dashboard."""

    brain: str
    proficiency_level: str
    proficiency_index: int
    maturity_level: float
    plasticity: float
    density: float
    activity_score: float
    semantic_ratio: float
    reinforcement_days: float
    topology_coherence: float
    plasticity_index: float
    knowledge_density: float
    total_neurons: int
    total_synapses: int
    total_fibers: int
    fibers_at_semantic: int
    fibers_at_episodic: int
    stage_distribution: StageDistributionResponse | None = None
    closest_to_semantic: list[SemanticProgressItem] = Field(default_factory=list)


@router.get(
    "/evolution",
    response_model=EvolutionResponse,
    summary="Get brain evolution metrics",
)
async def get_evolution(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> EvolutionResponse:
    """Get evolution dynamics for the active brain."""
    from neural_memory.engine.brain_evolution import EvolutionEngine
    from neural_memory.unified_config import get_config

    brain_name = get_config().current_brain

    try:
        engine = EvolutionEngine(storage)
        evo = await engine.analyze(brain_name)
    except Exception as exc:
        logger.warning("Evolution analysis failed for brain %s: %s", brain_name, exc)
        raise HTTPException(status_code=500, detail="Evolution analysis failed")

    stage_dist = None
    if evo.stage_distribution is not None:
        stage_dist = StageDistributionResponse(
            short_term=evo.stage_distribution.short_term,
            working=evo.stage_distribution.working,
            episodic=evo.stage_distribution.episodic,
            semantic=evo.stage_distribution.semantic,
            total=evo.stage_distribution.total,
        )

    closest = [
        SemanticProgressItem(
            fiber_id=p.fiber_id,
            stage=p.stage,
            days_in_stage=round(p.days_in_stage, 2),
            days_required=round(p.days_required, 2),
            reinforcement_days=p.reinforcement_days,
            reinforcement_required=p.reinforcement_required,
            progress_pct=round(p.progress_pct, 4),
            next_step=p.next_step,
        )
        for p in evo.closest_to_semantic
    ]

    return EvolutionResponse(
        brain=evo.brain_name,
        proficiency_level=evo.proficiency_level.value,
        proficiency_index=evo.proficiency_index,
        maturity_level=round(evo.maturity_level, 4),
        plasticity=round(evo.plasticity, 4),
        density=round(evo.density, 4),
        activity_score=round(evo.activity_score, 4),
        semantic_ratio=round(evo.semantic_ratio, 4),
        reinforcement_days=round(evo.reinforcement_days, 2),
        topology_coherence=round(evo.topology_coherence, 4),
        plasticity_index=round(evo.plasticity_index, 4),
        knowledge_density=round(evo.knowledge_density, 4),
        total_neurons=evo.total_neurons,
        total_synapses=evo.total_synapses,
        total_fibers=evo.total_fibers,
        fibers_at_semantic=evo.fibers_at_semantic,
        fibers_at_episodic=evo.fibers_at_episodic,
        stage_distribution=stage_dist,
        closest_to_semantic=closest,
    )
