"""Dashboard API routes — brain stats, health, brain management."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from neural_memory.server.dependencies import get_storage
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


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
                pass

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
            detail=f"Brain '{request.brain_name}' not found. Available: {available}",
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
