"""Memory API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from neural_memory.core.brain import Brain
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.server.dependencies import get_brain, get_storage
from neural_memory.server.models import (
    EncodeRequest,
    EncodeResponse,
    ErrorResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SubgraphResponse,
    SuggestResponse,
)
from neural_memory.storage.base import NeuralStorage

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post(
    "/encode",
    response_model=EncodeResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Encode a new memory",
    description="Store a new memory by encoding content into neural structures.",
)
async def encode_memory(
    request: EncodeRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> EncodeResponse:
    """Encode new content as a memory."""
    from neural_memory.safety.sensitive import check_sensitive_content

    sensitive_matches = check_sensitive_content(request.content, min_severity=2)
    if sensitive_matches:
        types_found = sorted({m.type.value for m in sensitive_matches})
        raise HTTPException(
            status_code=400,
            detail=f"Sensitive content detected: {', '.join(types_found)}. "
            "Remove secrets before storing.",
        )

    encoder = MemoryEncoder(storage, brain.config)

    tags = set(request.tags) if request.tags else None

    result = await encoder.encode(
        content=request.content,
        timestamp=request.timestamp,
        metadata=request.metadata,
        tags=tags,
    )

    return EncodeResponse(
        fiber_id=result.fiber.id,
        neurons_created=len(result.neurons_created),
        neurons_linked=len(result.neurons_linked),
        synapses_created=len(result.synapses_created),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Query memories",
    description="Query memories through spreading activation retrieval.",
)
async def query_memory(
    request: QueryRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> QueryResponse:
    """Query memories using the reflex pipeline."""
    pipeline = ReflexPipeline(storage, brain.config)

    depth = DepthLevel(request.depth) if request.depth is not None else None

    result = await pipeline.query(
        query=request.query,
        depth=depth,
        max_tokens=request.max_tokens,
        reference_time=request.reference_time,
    )

    subgraph = None
    if request.include_subgraph:
        subgraph = SubgraphResponse(
            neuron_ids=result.subgraph.neuron_ids,
            synapse_ids=result.subgraph.synapse_ids,
            anchor_ids=result.subgraph.anchor_ids,
        )

    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        depth_used=result.depth_used.value,
        neurons_activated=result.neurons_activated,
        fibers_matched=result.fibers_matched,
        context=result.context,
        latency_ms=result.latency_ms,
        subgraph=subgraph,
        metadata=result.metadata,
    )


@router.get(
    "/fiber/{fiber_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Get a specific fiber",
    description="Retrieve details of a specific memory fiber.",
)
async def get_fiber(
    fiber_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get a specific fiber by ID."""
    fiber = await storage.get_fiber(fiber_id)
    if fiber is None:
        raise HTTPException(status_code=404, detail=f"Fiber {fiber_id} not found")

    return {
        "id": fiber.id,
        "neuron_ids": list(fiber.neuron_ids),
        "synapse_ids": list(fiber.synapse_ids),
        "anchor_neuron_id": fiber.anchor_neuron_id,
        "time_start": fiber.time_start.isoformat() if fiber.time_start else None,
        "time_end": fiber.time_end.isoformat() if fiber.time_end else None,
        "coherence": fiber.coherence,
        "salience": fiber.salience,
        "frequency": fiber.frequency,
        "summary": fiber.summary,
        "tags": list(fiber.tags),
        "created_at": fiber.created_at.isoformat(),
    }


@router.get(
    "/neurons",
    summary="List neurons",
    description="List neurons in the brain with optional filters.",
)
async def list_neurons(
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    type: str | None = None,
    content_contains: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List neurons with optional filters."""
    from neural_memory.core.neuron import NeuronType

    limit = min(limit, 1000)
    neuron_type = None
    if type:
        try:
            neuron_type = NeuronType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid neuron type: {type}")

    neurons = await storage.find_neurons(
        type=neuron_type,
        content_contains=content_contains,
        limit=limit,
    )

    return {
        "neurons": [
            {
                "id": n.id,
                "type": n.type.value,
                "content": n.content,
                "created_at": n.created_at.isoformat(),
            }
            for n in neurons
        ],
        "count": len(neurons),
    }


@router.get(
    "/suggest",
    response_model=SuggestResponse,
    summary="Neuron suggestions",
    description="Get autocomplete suggestions matching a prefix, ranked by relevance and usage.",
)
async def suggest_neurons(
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    prefix: str,
    limit: int = 5,
    type: str | None = None,
) -> SuggestResponse:
    """Get prefix-based neuron suggestions."""
    from neural_memory.core.neuron import NeuronType

    type_filter = None
    if type:
        try:
            type_filter = NeuronType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid neuron type: {type}")
    suggestions = await storage.suggest_neurons(
        prefix=prefix,
        type_filter=type_filter,
        limit=limit,
    )
    return SuggestResponse(suggestions=suggestions, count=len(suggestions))


@router.post(
    "/index",
    response_model=IndexResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Index codebase",
    description="Index Python files into neural graph for code-aware recall.",
)
async def index_codebase(
    request: IndexRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> IndexResponse:
    """Index codebase or check indexing status."""
    if request.action == "scan":
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        cwd = Path(".").resolve()
        path = Path(request.path or ".").resolve()
        if not path.is_relative_to(cwd):
            raise HTTPException(status_code=400, detail="Path must be within working directory")
        if not path.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

        extensions = set(request.extensions or [".py"])
        encoder = CodebaseEncoder(storage, brain.config)
        results = await encoder.index_directory(path, extensions=extensions)

        total_neurons = sum(len(r.neurons_created) for r in results)
        total_synapses = sum(len(r.synapses_created) for r in results)

        return IndexResponse(
            files_indexed=len(results),
            neurons_created=total_neurons,
            synapses_created=total_synapses,
            path=str(path),
            message=f"Indexed {len(results)} files → {total_neurons} neurons, {total_synapses} synapses",
        )

    if request.action == "status":
        from neural_memory.core.neuron import NeuronType

        neurons = await storage.find_neurons(type=NeuronType.SPATIAL, limit=1000)
        code_files = [n for n in neurons if n.metadata.get("indexed")]

        return IndexResponse(
            files_indexed=len(code_files),
            indexed_files=[n.content for n in code_files[:50]],
            message=f"{len(code_files)} files indexed"
            if code_files
            else "No codebase indexed yet.",
        )

    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
