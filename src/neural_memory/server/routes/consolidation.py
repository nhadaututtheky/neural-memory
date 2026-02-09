"""Consolidation API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from neural_memory.server.dependencies import get_storage
from neural_memory.server.models import ErrorResponse
from neural_memory.storage.base import NeuralStorage

router = APIRouter(prefix="/brain", tags=["consolidation"])


class ConsolidateRequest(BaseModel):
    """Request to consolidate a brain's memories."""

    strategies: list[str] = Field(
        default=["all"],
        description="Strategies: prune, merge, summarize, all",
    )
    dry_run: bool = Field(False, description="Preview changes without applying")
    prune_weight_threshold: float = Field(0.05, ge=0, le=1)
    merge_overlap_threshold: float = Field(0.5, ge=0, le=1)
    prune_min_inactive_days: float = Field(7.0, ge=0)


class MergeDetailResponse(BaseModel):
    """Details of a single merge operation."""

    original_fiber_ids: list[str]
    merged_fiber_id: str
    neuron_count: int
    reason: str


class ConsolidationResponse(BaseModel):
    """Response from consolidation operation."""

    started_at: str
    duration_ms: float
    synapses_pruned: int
    neurons_pruned: int
    fibers_merged: int
    fibers_removed: int
    fibers_created: int
    summaries_created: int
    merge_details: list[MergeDetailResponse]
    dry_run: bool


@router.post(
    "/{brain_id}/consolidate",
    response_model=ConsolidationResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Consolidate brain memories",
    description="Run consolidation strategies (prune, merge, summarize) on a brain.",
)
async def consolidate_brain(
    brain_id: str,
    request: ConsolidateRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> ConsolidationResponse:
    """Run consolidation on a brain."""
    from neural_memory.engine.consolidation import (
        ConsolidationConfig,
        ConsolidationEngine,
        ConsolidationStrategy,
    )

    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail=f"Brain {brain_id} not found")

    storage.set_brain(brain_id)

    try:
        strategies = [ConsolidationStrategy(s) for s in request.strategies]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid consolidation strategy: {e}")
    config = ConsolidationConfig(
        prune_weight_threshold=request.prune_weight_threshold,
        merge_overlap_threshold=request.merge_overlap_threshold,
        prune_min_inactive_days=request.prune_min_inactive_days,
    )

    engine = ConsolidationEngine(storage, config)
    report = await engine.run(strategies=strategies, dry_run=request.dry_run)

    return ConsolidationResponse(
        started_at=report.started_at.isoformat(),
        duration_ms=report.duration_ms,
        synapses_pruned=report.synapses_pruned,
        neurons_pruned=report.neurons_pruned,
        fibers_merged=report.fibers_merged,
        fibers_removed=report.fibers_removed,
        fibers_created=report.fibers_created,
        summaries_created=report.summaries_created,
        merge_details=[
            MergeDetailResponse(
                original_fiber_ids=list(d.original_fiber_ids),
                merged_fiber_id=d.merged_fiber_id,
                neuron_count=d.neuron_count,
                reason=d.reason,
            )
            for d in report.merge_details
        ],
        dry_run=report.dry_run,
    )
