"""Pydantic models for API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ============ Request Models ============


class EncodeRequest(BaseModel):
    """Request to encode a new memory."""

    content: str = Field(..., description="The content to encode as a memory")
    timestamp: datetime | None = Field(
        None, description="When this memory occurred (default: now)"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata to attach"
    )
    tags: list[str] | None = Field(None, description="Tags for categorization")


class QueryRequest(BaseModel):
    """Request to query memories."""

    query: str = Field(..., description="The query text")
    depth: int | None = Field(
        None,
        ge=0,
        le=3,
        description="Retrieval depth (0=instant, 1=context, 2=habit, 3=deep). Auto-detects if not specified.",
    )
    max_tokens: int = Field(
        500,
        ge=50,
        le=5000,
        description="Maximum tokens in returned context",
    )
    include_subgraph: bool = Field(
        False, description="Whether to include subgraph details"
    )
    reference_time: datetime | None = Field(
        None, description="Reference time for temporal parsing (default: now)"
    )


class CreateBrainRequest(BaseModel):
    """Request to create a new brain."""

    name: str = Field(..., min_length=1, max_length=100, description="Brain name")
    owner_id: str | None = Field(None, description="Owner identifier")
    is_public: bool = Field(False, description="Whether publicly accessible")
    config: BrainConfigModel | None = Field(None, description="Custom configuration")


class BrainConfigModel(BaseModel):
    """Brain configuration model."""

    decay_rate: float = Field(0.1, ge=0, le=1)
    reinforcement_delta: float = Field(0.05, ge=0, le=0.5)
    activation_threshold: float = Field(0.2, ge=0, le=1)
    max_spread_hops: int = Field(4, ge=1, le=10)
    max_context_tokens: int = Field(1500, ge=100, le=10000)


# ============ Response Models ============


class EncodeResponse(BaseModel):
    """Response from encoding a memory."""

    fiber_id: str = Field(..., description="ID of the created fiber")
    neurons_created: int = Field(..., description="Number of neurons created")
    neurons_linked: int = Field(..., description="Number of existing neurons linked")
    synapses_created: int = Field(..., description="Number of synapses created")


class SubgraphResponse(BaseModel):
    """Subgraph details in query response."""

    neuron_ids: list[str]
    synapse_ids: list[str]
    anchor_ids: list[str]


class QueryResponse(BaseModel):
    """Response from querying memories."""

    answer: str | None = Field(None, description="Reconstructed answer if available")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in answer")
    depth_used: int = Field(..., description="Depth level used for retrieval")
    neurons_activated: int = Field(..., description="Number of neurons activated")
    fibers_matched: list[str] = Field(..., description="IDs of matched fibers")
    context: str = Field(..., description="Formatted context for injection")
    latency_ms: float = Field(..., description="Retrieval latency in milliseconds")
    subgraph: SubgraphResponse | None = Field(
        None, description="Subgraph details (if requested)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrainResponse(BaseModel):
    """Response with brain details."""

    id: str
    name: str
    owner_id: str | None
    is_public: bool
    neuron_count: int
    synapse_count: int
    fiber_count: int
    created_at: datetime
    updated_at: datetime


class BrainListResponse(BaseModel):
    """Response with list of brains."""

    brains: list[BrainResponse]
    total: int


class StatsResponse(BaseModel):
    """Response with brain statistics."""

    brain_id: str
    neuron_count: int
    synapse_count: int
    fiber_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
