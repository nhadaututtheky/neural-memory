"""Brain API routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.server.dependencies import get_storage
from neural_memory.server.models import (
    BrainResponse,
    ConflictItemResponse,
    CreateBrainRequest,
    ErrorResponse,
    ImportBrainRequest,
    MergeBrainRequest,
    MergeReportResponse,
    StatsResponse,
)
from neural_memory.storage.base import NeuralStorage

router = APIRouter(prefix="/brain", tags=["brain"])


@router.post(
    "/create",
    response_model=BrainResponse,
    summary="Create a new brain",
    description="Create a new brain for storing memories.",
)
async def create_brain(
    request: CreateBrainRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> BrainResponse:
    """Create a new brain."""
    # Build config
    config = BrainConfig()
    if request.config:
        config = BrainConfig(
            decay_rate=request.config.decay_rate,
            reinforcement_delta=request.config.reinforcement_delta,
            activation_threshold=request.config.activation_threshold,
            max_spread_hops=request.config.max_spread_hops,
            max_context_tokens=request.config.max_context_tokens,
        )

    brain = Brain.create(
        name=request.name,
        config=config,
        owner_id=request.owner_id,
        is_public=request.is_public,
    )

    await storage.save_brain(brain)

    return BrainResponse(
        id=brain.id,
        name=brain.name,
        owner_id=brain.owner_id,
        is_public=brain.is_public,
        neuron_count=0,
        synapse_count=0,
        fiber_count=0,
        created_at=brain.created_at,
        updated_at=brain.updated_at,
    )


@router.get(
    "/{brain_id}",
    response_model=BrainResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get brain details",
    description="Get details of a specific brain.",
)
async def get_brain(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> BrainResponse:
    """Get brain by ID."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    # Get current stats
    stats = await storage.get_stats(brain_id)

    return BrainResponse(
        id=brain.id,
        name=brain.name,
        owner_id=brain.owner_id,
        is_public=brain.is_public,
        neuron_count=stats["neuron_count"],
        synapse_count=stats["synapse_count"],
        fiber_count=stats["fiber_count"],
        created_at=brain.created_at,
        updated_at=brain.updated_at,
    )


@router.get(
    "/{brain_id}/stats",
    response_model=StatsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get brain statistics",
    description="Get enhanced statistics for a brain including hot neurons, synapse stats, and more.",
)
async def get_brain_stats(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> StatsResponse:
    """Get brain statistics."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    stats = await storage.get_enhanced_stats(brain_id)

    # Build synapse stats model
    from neural_memory.server.models import HotNeuronInfo, SynapseStatsInfo, SynapseTypeStats

    synapse_stats_raw = stats.get("synapse_stats", {})
    synapse_stats_model = SynapseStatsInfo(
        avg_weight=synapse_stats_raw.get("avg_weight", 0.0),
        total_reinforcements=synapse_stats_raw.get("total_reinforcements", 0),
        by_type={k: SynapseTypeStats(**v) for k, v in synapse_stats_raw.get("by_type", {}).items()},
    )

    hot_neurons_models = [HotNeuronInfo(**hn) for hn in stats.get("hot_neurons", [])]

    return StatsResponse(
        brain_id=brain_id,
        neuron_count=stats["neuron_count"],
        synapse_count=stats["synapse_count"],
        fiber_count=stats["fiber_count"],
        db_size_bytes=stats.get("db_size_bytes"),
        hot_neurons=hot_neurons_models,
        today_fibers_count=stats.get("today_fibers_count"),
        synapse_stats=synapse_stats_model,
        neuron_type_breakdown=stats.get("neuron_type_breakdown"),
        oldest_memory=stats.get("oldest_memory"),
        newest_memory=stats.get("newest_memory"),
    )


@router.get(
    "/{brain_id}/export",
    responses={404: {"model": ErrorResponse}},
    summary="Export brain",
    description="Export a brain as a JSON snapshot.",
)
async def export_brain(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Export brain as snapshot."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    snapshot = await storage.export_brain(brain_id)

    return {
        "brain_id": snapshot.brain_id,
        "brain_name": snapshot.brain_name,
        "exported_at": snapshot.exported_at.isoformat(),
        "version": snapshot.version,
        "neurons": snapshot.neurons,
        "synapses": snapshot.synapses,
        "fibers": snapshot.fibers,
        "config": snapshot.config,
        "metadata": snapshot.metadata,
    }


@router.post(
    "/{brain_id}/import",
    response_model=BrainResponse,
    summary="Import brain",
    description="Import a brain from a JSON snapshot.",
)
async def import_brain(
    brain_id: str,
    snapshot: ImportBrainRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> BrainResponse:
    """Import brain from snapshot."""
    from neural_memory.core.brain import BrainSnapshot

    brain_snapshot = BrainSnapshot(
        brain_id=snapshot.brain_id,
        brain_name=snapshot.brain_name,
        exported_at=snapshot.exported_at,
        version=snapshot.version,
        neurons=snapshot.neurons,
        synapses=snapshot.synapses,
        fibers=snapshot.fibers,
        config=snapshot.config,
        metadata=snapshot.metadata,
    )

    imported_id = await storage.import_brain(brain_snapshot, brain_id)

    brain = await storage.get_brain(imported_id)
    if brain is None:
        raise HTTPException(status_code=500, detail="Import failed")

    stats = await storage.get_stats(imported_id)

    return BrainResponse(
        id=brain.id,
        name=brain.name,
        owner_id=brain.owner_id,
        is_public=brain.is_public,
        neuron_count=stats["neuron_count"],
        synapse_count=stats["synapse_count"],
        fiber_count=stats["fiber_count"],
        created_at=brain.created_at,
        updated_at=brain.updated_at,
    )


@router.post(
    "/{brain_id}/merge",
    response_model=MergeReportResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Merge snapshot into brain",
    description="Merge an incoming brain snapshot into an existing brain with conflict resolution.",
)
async def merge_brain(
    brain_id: str,
    request: MergeBrainRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> MergeReportResponse:
    """Merge a snapshot into an existing brain."""
    from neural_memory.core.brain import BrainSnapshot
    from neural_memory.engine.merge import ConflictStrategy, merge_snapshots

    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    # Export current brain as local snapshot
    local_snapshot = await storage.export_brain(brain_id)

    # Build incoming snapshot from request
    incoming_snapshot = BrainSnapshot(
        brain_id=request.snapshot.brain_id,
        brain_name=request.snapshot.brain_name,
        exported_at=request.snapshot.exported_at,
        version=request.snapshot.version,
        neurons=request.snapshot.neurons,
        synapses=request.snapshot.synapses,
        fibers=request.snapshot.fibers,
        config=request.snapshot.config,
        metadata=request.snapshot.metadata,
    )

    # Merge
    try:
        conflict_strategy = ConflictStrategy(request.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid merge strategy")
    merged_snapshot, merge_report = merge_snapshots(
        local=local_snapshot,
        incoming=incoming_snapshot,
        strategy=conflict_strategy,
    )

    # Reset brain data then reimport merged snapshot
    await storage.clear(brain_id)
    await storage.import_brain(merged_snapshot, brain_id)

    return MergeReportResponse(
        neurons_added=merge_report.neurons_added,
        neurons_updated=merge_report.neurons_updated,
        neurons_skipped=merge_report.neurons_skipped,
        synapses_added=merge_report.synapses_added,
        synapses_updated=merge_report.synapses_updated,
        fibers_added=merge_report.fibers_added,
        fibers_updated=merge_report.fibers_updated,
        fibers_skipped=merge_report.fibers_skipped,
        conflicts=[
            ConflictItemResponse(
                entity_type=c.entity_type,
                local_id=c.local_id,
                incoming_id=c.incoming_id,
                resolution=c.resolution,
                reason=c.reason,
            )
            for c in merge_report.conflicts
        ],
        id_remap_count=len(merge_report.id_remap),
    )


@router.delete(
    "/{brain_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete brain",
    description="Delete a brain and all its data.",
)
async def delete_brain(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, str]:
    """Delete a brain."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    await storage.clear(brain_id)

    return {"status": "deleted", "brain_id": brain_id}
