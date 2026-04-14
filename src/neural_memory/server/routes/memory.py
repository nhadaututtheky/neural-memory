"""Memory API routes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from neural_memory.core.brain import Brain
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.server.dependencies import get_brain, get_storage, require_local_request
from neural_memory.server.models import (
    EncodeRequest,
    EncodeResponse,
    ErrorResponse,
    IndexRequest,
    IndexResponse,
    NeuronRequest,
    NeuronResponse,
    NeuronStateRequest,
    NeuronUpdateRequest,
    QueryRequest,
    QueryResponse,
    SubgraphResponse,
    SuggestResponse,
    SynapseRequest,
    SynapseUpdateRequest,
)
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    dependencies=[Depends(require_local_request)],
)


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
    if len(request.content) > 100_000:
        raise HTTPException(status_code=400, detail="Content exceeds 100,000 character limit")

    from neural_memory.safety.sensitive import check_sensitive_content

    sensitive_matches = check_sensitive_content(request.content, min_severity=2)
    if sensitive_matches:
        types_found = sorted({m.type.value for m in sensitive_matches})
        raise HTTPException(
            status_code=400,
            detail=f"Sensitive content detected: {', '.join(types_found)}. "
            "Remove secrets before storing.",
        )

    # SimHash dedup always runs (pure Python, zero cost).
    # Full dedup (embedding + LLM) only when [dedup] enabled = true.
    from neural_memory.engine.dedup import build_dedup_pipeline
    from neural_memory.unified_config import get_config

    dedup_pipeline = build_dedup_pipeline(get_config().dedup, storage)

    encoder = MemoryEncoder(storage, brain.config, dedup_pipeline=dedup_pipeline)

    tags = set(request.tags) if request.tags else None

    try:
        result = await encoder.encode(
            content=request.content,
            timestamp=request.timestamp,
            metadata=request.metadata,
            tags=tags,
        )
    except Exception:
        logger.error("Encode failed for content length %d", len(request.content), exc_info=True)
        raise HTTPException(status_code=500, detail="Memory encoding failed")

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
    # Filter out empty/whitespace-only tags, cap at 20
    tags = {t.strip()[:100] for t in request.tags[:20] if t.strip()} if request.tags else None
    if tags is not None and not tags:
        tags = None

    try:
        result = await pipeline.query(
            query=request.query,
            depth=depth,
            max_tokens=request.max_tokens,
            reference_time=request.reference_time,
            valid_at=request.valid_at,
            tags=tags,
            tag_mode=request.tag_mode,
            as_of=request.as_of,
            simhash_threshold=request.simhash_threshold,
            exclude_ephemeral=request.permanent_only,
        )
    except Exception:
        logger.error("Query pipeline failed for query: %s", request.query[:200], exc_info=True)
        raise HTTPException(status_code=500, detail="Memory retrieval failed")

    # Apply min_confidence filter
    if result.confidence < request.min_confidence:
        return QueryResponse(
            answer=None,
            confidence=result.confidence,
            depth_used=result.depth_used.value,
            neurons_activated=0,
            fibers_matched=[],
            context="",
            latency_ms=result.latency_ms,
            subgraph=None,
            metadata={"filtered": "below_min_confidence"},
        )

    # Post-filter by min_trust / tier (mirrors MCP recall_handler logic)
    fibers_matched = result.fibers_matched
    needs_post_filter = (
        request.min_trust is not None or request.tier is not None
    ) and fibers_matched
    if needs_post_filter:
        try:
            passing_ids: set[str] = set()
            for fid in fibers_matched:
                tm = await storage.get_typed_memory(fid)
                # Trust filter
                if request.min_trust is not None:
                    if tm and tm.trust_score is not None:
                        if tm.trust_score < request.min_trust:
                            continue
                    elif tm is None:
                        # No typed memory — skip if trust required
                        continue
                # Tier filter
                if request.tier is not None:
                    fiber_tier = getattr(tm, "tier", "warm") if tm else "warm"
                    if fiber_tier != request.tier:
                        continue
                passing_ids.add(fid)
            fibers_matched = [f for f in fibers_matched if f in passing_ids]
        except Exception:
            logger.warning(
                "Post-filter (min_trust/tier) failed, returning unfiltered",
                exc_info=True,
            )

    # Budget-aware context re-formatting (opt-in via recall_token_budget)
    context = result.context
    extra_metadata: dict[str, Any] = {}
    if request.recall_token_budget is not None and fibers_matched:
        try:
            from neural_memory.engine.retrieval_context import format_context_budgeted
            from neural_memory.engine.token_budget import BudgetConfig, format_budget_report

            budget_cfg = BudgetConfig()
            from neural_memory.engine.activation import ActivationResult

            fibers_for_budget = []
            activations_for_budget: dict[str, ActivationResult] = {}
            for fid in fibers_matched:
                fiber = await storage.get_fiber(fid)
                if fiber:
                    fibers_for_budget.append(fiber)
                    activations_for_budget[fid] = ActivationResult(
                        neuron_id=fiber.anchor_neuron_id,
                        activation_level=1.0,
                        hop_distance=0,
                        path=[fiber.anchor_neuron_id],
                        source_anchor=fiber.anchor_neuron_id,
                    )

            if fibers_for_budget:
                # Extract query terms for context compiler (dedup/rescore step).
                _budget_query_terms: list[str] = [w for w in request.query.split() if len(w) > 2]
                budgeted_ctx, _, allocation = await format_context_budgeted(
                    storage=storage,
                    activations=activations_for_budget,
                    fibers=fibers_for_budget,
                    max_tokens=min(request.recall_token_budget, 100_000),
                    budget_config=budget_cfg,
                    clean_for_prompt=request.clean_for_prompt,
                    query_terms=_budget_query_terms,
                )
                context = budgeted_ctx
                extra_metadata["budget"] = format_budget_report(allocation)
        except Exception:
            logger.warning(
                "Budget-aware context formatting failed, using default",
                exc_info=True,
            )

    subgraph = None
    if request.include_subgraph:
        subgraph = SubgraphResponse(
            neuron_ids=result.subgraph.neuron_ids,
            synapse_ids=result.subgraph.synapse_ids,
            anchor_ids=result.subgraph.anchor_ids,
        )

    response_metadata = {**result.metadata, **extra_metadata} if extra_metadata else result.metadata

    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        depth_used=result.depth_used.value,
        neurons_activated=result.neurons_activated,
        fibers_matched=fibers_matched,
        context=context,
        latency_ms=result.latency_ms,
        subgraph=subgraph,
        metadata=response_metadata,
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
        raise HTTPException(status_code=404, detail="Fiber not found")

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
    "/show/{memory_id}",
    summary="Get full memory detail",
    description="Get verbatim content, metadata, and synapses for a memory by fiber_id or neuron_id.",
    responses={404: {"model": ErrorResponse}},
)
async def show_memory(
    memory_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get full memory detail by ID (fiber_id or neuron_id)."""
    # Try as fiber_id first
    typed_mem = await storage.get_typed_memory(memory_id)
    fiber = await storage.get_fiber(memory_id) if typed_mem else None

    if typed_mem and fiber:
        anchor = await storage.get_neuron(fiber.anchor_neuron_id)
        content = anchor.content if anchor else ""

        synapses_out = await storage.get_synapses(source_id=fiber.anchor_neuron_id)
        synapses_in = await storage.get_synapses(target_id=fiber.anchor_neuron_id)
        synapse_list = [
            {
                "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                "target_id": s.target_id,
                "source_id": s.source_id,
                "weight": s.weight,
            }
            for s in [*synapses_out, *synapses_in]
        ]

        return {
            "memory_id": memory_id,
            "content": content,
            "memory_type": typed_mem.memory_type.value,
            "priority": typed_mem.priority.value,
            "tags": list(typed_mem.tags) if typed_mem.tags else [],
            "created_at": fiber.created_at.isoformat() if fiber.created_at else None,
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "neuron_count": fiber.neuron_count,
            "summary": fiber.summary,
            "metadata": fiber.metadata,
            "synapses": synapse_list,
            "trust_score": typed_mem.trust_score,
            "expires_at": typed_mem.expires_at.isoformat() if typed_mem.expires_at else None,
        }

    # Try as neuron_id
    neuron = await storage.get_neuron(memory_id)
    if neuron:
        synapses_out = await storage.get_synapses(source_id=memory_id)
        synapses_in = await storage.get_synapses(target_id=memory_id)
        synapse_list = [
            {
                "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                "target_id": s.target_id,
                "source_id": s.source_id,
                "weight": s.weight,
            }
            for s in [*synapses_out, *synapses_in]
        ]
        return {
            "memory_id": memory_id,
            "content": neuron.content,
            "neuron_type": neuron.type.value if hasattr(neuron.type, "value") else str(neuron.type),
            "created_at": neuron.created_at.isoformat() if neuron.created_at else None,
            "metadata": neuron.metadata,
            "synapses": synapse_list,
        }

    raise HTTPException(status_code=404, detail="Memory not found")


@router.get(
    "/sources",
    summary="List sources",
    description="List registered sources for the brain.",
)
async def list_sources(
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    source_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, Any]:
    """List sources with optional filters."""
    sources = await storage.list_sources(source_type=source_type, status=status, limit=limit)
    return {
        "sources": [
            {
                "source_id": s.id,
                "name": s.name,
                "source_type": s.source_type.value,
                "version": s.version,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
            }
            for s in sources
        ],
        "count": len(sources),
    }


@router.get(
    "/sources/{source_id}",
    summary="Get source detail",
    responses={404: {"model": ErrorResponse}},
)
async def get_source(
    source_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get detailed source info including linked neuron count."""
    source = await storage.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    neuron_count = await storage.count_neurons_for_source(source_id)
    return {
        "source_id": source.id,
        "name": source.name,
        "source_type": source.source_type.value,
        "version": source.version,
        "status": source.status.value,
        "file_hash": source.file_hash,
        "metadata": source.metadata,
        "linked_neuron_count": neuron_count,
        "created_at": source.created_at.isoformat(),
        "updated_at": source.updated_at.isoformat(),
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
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, Any]:
    """List neurons with optional filters."""
    from neural_memory.core.neuron import NeuronType

    neuron_type = None
    if type:
        try:
            neuron_type = NeuronType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid neuron type")

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
    limit: int = Query(default=5, ge=1, le=100),
    type: str | None = None,
) -> SuggestResponse:
    """Get prefix-based neuron suggestions."""
    from neural_memory.core.neuron import NeuronType

    type_filter = None
    if type:
        try:
            type_filter = NeuronType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid neuron type")
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
            raise HTTPException(status_code=400, detail="Not a directory")

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

    raise HTTPException(status_code=400, detail="Unknown action")


# ========== Neuron CRUD (for SharedStorage client) ==========


@router.post(
    "/neurons",
    response_model=NeuronResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Create a neuron",
    description="Directly create a neuron in the brain. Used by SharedStorage for multi-agent sync.",
)
async def create_neuron(
    request: NeuronRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> NeuronResponse:
    """Create a neuron directly."""
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.utils.timeutils import utcnow

    try:
        neuron_type = NeuronType(request.type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid neuron type")

    neuron = Neuron(
        id=request.id or str(uuid4()),
        type=neuron_type,
        content=request.content,
        metadata=request.metadata,
        created_at=request.created_at or utcnow(),
    )

    neuron_id = await storage.add_neuron(neuron)

    return NeuronResponse(
        id=neuron_id,
        type=neuron.type.value,
        content=neuron.content,
        metadata=neuron.metadata,
        created_at=neuron.created_at.isoformat(),
    )


@router.get(
    "/neurons/{neuron_id}",
    response_model=NeuronResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get a neuron by ID",
)
async def get_neuron(
    neuron_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> NeuronResponse:
    """Get a single neuron by ID."""
    neuron = await storage.get_neuron(neuron_id)
    if neuron is None:
        raise HTTPException(status_code=404, detail="Neuron not found")

    return NeuronResponse(
        id=neuron.id,
        type=neuron.type.value,
        content=neuron.content,
        metadata=neuron.metadata,
        created_at=neuron.created_at.isoformat(),
    )


@router.put(
    "/neurons/{neuron_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Update a neuron",
)
async def update_neuron(
    neuron_id: str,
    request: NeuronUpdateRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> NeuronResponse:
    """Update an existing neuron."""
    from dataclasses import replace

    from neural_memory.core.neuron import NeuronType

    neuron = await storage.get_neuron(neuron_id)
    if neuron is None:
        raise HTTPException(status_code=404, detail="Neuron not found")

    updates: dict[str, Any] = {}
    if request.type is not None:
        try:
            updates["type"] = NeuronType(request.type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid neuron type")
    if request.content is not None:
        updates["content"] = request.content
    if request.metadata is not None:
        updates["metadata"] = request.metadata

    updated = replace(neuron, **updates)
    await storage.update_neuron(updated)

    return NeuronResponse(
        id=updated.id,
        type=updated.type.value,
        content=updated.content,
        metadata=updated.metadata,
        created_at=updated.created_at.isoformat(),
    )


@router.delete(
    "/neurons/{neuron_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete a neuron",
)
async def delete_neuron(
    neuron_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, str]:
    """Delete a neuron by ID."""
    deleted = await storage.delete_neuron(neuron_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Neuron not found")
    return {"status": "deleted", "id": neuron_id}


@router.get(
    "/neurons/{neuron_id}/state",
    responses={404: {"model": ErrorResponse}},
    summary="Get neuron activation state",
)
async def get_neuron_state(
    neuron_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get neuron activation state."""
    state = await storage.get_neuron_state(neuron_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Neuron state not found")

    return {
        "neuron_id": state.neuron_id,
        "activation_level": state.activation_level,
        "access_frequency": state.access_frequency,
        "last_activated": state.last_activated.isoformat() if state.last_activated else None,
        "decay_rate": state.decay_rate,
        "firing_threshold": state.firing_threshold,
        "refractory_until": state.refractory_until.isoformat() if state.refractory_until else None,
        "refractory_period_ms": state.refractory_period_ms,
        "homeostatic_target": state.homeostatic_target,
    }


@router.put(
    "/neurons/{neuron_id}/state",
    responses={404: {"model": ErrorResponse}},
    summary="Update neuron activation state",
)
async def update_neuron_state(
    neuron_id: str,
    request: NeuronStateRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, str]:
    """Update neuron activation state."""
    from neural_memory.core.neuron import NeuronState

    neuron = await storage.get_neuron(neuron_id)
    if neuron is None:
        raise HTTPException(status_code=404, detail="Neuron not found")

    state = NeuronState(
        neuron_id=neuron_id,
        activation_level=request.activation_level,
        access_frequency=request.access_frequency,
        last_activated=request.last_activated,
        decay_rate=request.decay_rate,
        firing_threshold=request.firing_threshold,
        refractory_until=request.refractory_until,
        refractory_period_ms=request.refractory_period_ms,
        homeostatic_target=request.homeostatic_target,
    )
    await storage.update_neuron_state(state)
    return {"status": "updated", "neuron_id": neuron_id}


@router.get(
    "/neurons/{neuron_id}/neighbors",
    responses={404: {"model": ErrorResponse}},
    summary="Get neighboring neurons",
)
async def get_neuron_neighbors(
    neuron_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    synapse_types: str | None = Query(default=None, description="Comma-separated synapse types"),
    min_weight: float | None = Query(default=None, ge=0, le=1),
) -> dict[str, Any]:
    """Get neighboring neurons via synapses."""
    from neural_memory.core.synapse import SynapseType

    neuron = await storage.get_neuron(neuron_id)
    if neuron is None:
        raise HTTPException(status_code=404, detail="Neuron not found")

    s_types = None
    if synapse_types:
        try:
            s_types = [SynapseType(t.strip()) for t in synapse_types.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid synapse type")

    neighbors = await storage.get_neighbors(
        neuron_id=neuron_id,
        direction=direction,  # type: ignore[arg-type]
        synapse_types=s_types,
        min_weight=min_weight,
    )

    return {
        "neighbors": [
            {
                "neuron": {
                    "id": n.id,
                    "type": n.type.value,
                    "content": n.content,
                    "metadata": n.metadata,
                    "created_at": n.created_at.isoformat(),
                },
                "synapse": {
                    "id": s.id,
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "type": s.type.value,
                    "weight": s.weight,
                    "direction": s.direction.value,
                    "metadata": s.metadata,
                    "created_at": s.created_at.isoformat(),
                },
            }
            for n, s in neighbors
        ],
        "count": len(neighbors),
    }


@router.get(
    "/neurons/{source_id}/path",
    responses={404: {"model": ErrorResponse}},
    summary="Find shortest path between neurons",
)
async def get_neuron_path(
    source_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    target_id: str = Query(...),
    max_hops: int = Query(default=4, ge=1, le=10),
) -> dict[str, Any]:
    """Find shortest path between two neurons."""
    path = await storage.get_path(
        source_id=source_id,
        target_id=target_id,
        max_hops=max_hops,
    )

    if path is None:
        return {"path": None, "hops": 0}

    return {
        "path": [
            {
                "neuron": {
                    "id": n.id,
                    "type": n.type.value,
                    "content": n.content,
                    "metadata": n.metadata,
                    "created_at": n.created_at.isoformat(),
                },
                "synapse": {
                    "id": s.id,
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "type": s.type.value,
                    "weight": s.weight,
                    "direction": s.direction.value,
                    "metadata": s.metadata,
                    "created_at": s.created_at.isoformat(),
                },
            }
            for n, s in path
        ],
        "hops": len(path),
    }


# ========== Synapse CRUD (for SharedStorage client) ==========


@router.post(
    "/synapses",
    responses={400: {"model": ErrorResponse}},
    summary="Create a synapse",
    description="Directly create a synapse between neurons. Used by SharedStorage for multi-agent sync.",
)
async def create_synapse(
    request: SynapseRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Create a synapse directly."""
    from neural_memory.core.synapse import Direction, Synapse, SynapseType
    from neural_memory.utils.timeutils import utcnow

    try:
        synapse_type = SynapseType(request.type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid synapse type")

    try:
        direction = Direction(request.direction)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid direction")

    synapse = Synapse(
        id=request.id or str(uuid4()),
        source_id=request.source_id,
        target_id=request.target_id,
        type=synapse_type,
        weight=request.weight,
        direction=direction,
        metadata=request.metadata,
        created_at=request.created_at or utcnow(),
    )

    synapse_id = await storage.add_synapse(synapse)

    return {
        "id": synapse_id,
        "source_id": synapse.source_id,
        "target_id": synapse.target_id,
        "type": synapse.type.value,
        "weight": synapse.weight,
    }


@router.get(
    "/synapses/{synapse_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Get a synapse by ID",
)
async def get_synapse(
    synapse_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get a single synapse by ID."""
    synapse = await storage.get_synapse(synapse_id)
    if synapse is None:
        raise HTTPException(status_code=404, detail="Synapse not found")

    return {
        "id": synapse.id,
        "source_id": synapse.source_id,
        "target_id": synapse.target_id,
        "type": synapse.type.value,
        "weight": synapse.weight,
        "direction": synapse.direction.value,
        "metadata": synapse.metadata,
        "created_at": synapse.created_at.isoformat(),
    }


@router.get(
    "/synapses",
    summary="List synapses",
    description="List synapses with optional filters.",
)
async def list_synapses(
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    source_id: str | None = None,
    target_id: str | None = None,
    type: str | None = None,
    min_weight: float | None = Query(default=None, ge=0, le=1),
) -> dict[str, Any]:
    """List synapses with optional filters."""
    from neural_memory.core.synapse import SynapseType

    synapse_type = None
    if type:
        try:
            synapse_type = SynapseType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid synapse type")

    synapses = await storage.get_synapses(
        source_id=source_id,
        target_id=target_id,
        type=synapse_type,
        min_weight=min_weight,
    )

    return {
        "synapses": [
            {
                "id": s.id,
                "source_id": s.source_id,
                "target_id": s.target_id,
                "type": s.type.value,
                "weight": s.weight,
                "direction": s.direction.value,
                "metadata": s.metadata,
                "created_at": s.created_at.isoformat(),
            }
            for s in synapses
        ],
        "count": len(synapses),
    }


@router.put(
    "/synapses/{synapse_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Update a synapse",
)
async def update_synapse(
    synapse_id: str,
    request: SynapseUpdateRequest,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, str]:
    """Update an existing synapse."""
    from dataclasses import replace

    synapse = await storage.get_synapse(synapse_id)
    if synapse is None:
        raise HTTPException(status_code=404, detail="Synapse not found")

    updates: dict[str, Any] = {}
    if request.weight is not None:
        updates["weight"] = request.weight
    if request.metadata is not None:
        updates["metadata"] = request.metadata

    updated = replace(synapse, **updates)
    await storage.update_synapse(updated)
    return {"status": "updated", "id": synapse_id}


@router.delete(
    "/synapses/{synapse_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete a synapse",
)
async def delete_synapse(
    synapse_id: str,
    brain: Annotated[Brain, Depends(get_brain)],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, str]:
    """Delete a synapse by ID."""
    deleted = await storage.delete_synapse(synapse_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Synapse not found")
    return {"status": "deleted", "id": synapse_id}
