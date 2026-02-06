"""Brain API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.server.dependencies import get_storage
from neural_memory.server.models import (
    BrainResponse,
    CreateBrainRequest,
    ErrorResponse,
    ImportBrainRequest,
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
        raise HTTPException(status_code=404, detail=f"Brain {brain_id} not found")

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
    description="Get statistics for a brain.",
)
async def get_brain_stats(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> StatsResponse:
    """Get brain statistics."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail=f"Brain {brain_id} not found")

    stats = await storage.get_stats(brain_id)

    return StatsResponse(
        brain_id=brain_id,
        neuron_count=stats["neuron_count"],
        synapse_count=stats["synapse_count"],
        fiber_count=stats["fiber_count"],
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
) -> dict:
    """Export brain as snapshot."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail=f"Brain {brain_id} not found")

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


@router.delete(
    "/{brain_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete brain",
    description="Delete a brain and all its data.",
)
async def delete_brain(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict:
    """Delete a brain."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail=f"Brain {brain_id} not found")

    await storage.clear(brain_id)

    return {"status": "deleted", "brain_id": brain_id}
