"""Pydantic models for API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ============ Request Models ============


class EncodeRequest(BaseModel):
    """Request to encode a new memory."""

    content: str = Field(..., description="The content to encode as a memory", max_length=100_000)
    timestamp: datetime | None = Field(None, description="When this memory occurred (default: now)")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata to attach")
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
    include_subgraph: bool = Field(False, description="Whether to include subgraph details")
    reference_time: datetime | None = Field(
        None, description="Reference time for temporal parsing (default: now)"
    )


class CreateBrainRequest(BaseModel):
    """Request to create a new brain."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Brain name (alphanumeric, hyphens, underscores, dots only)",
    )
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
    subgraph: SubgraphResponse | None = Field(None, description="Subgraph details (if requested)")
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


class HotNeuronInfo(BaseModel):
    """Info about a frequently accessed neuron."""

    neuron_id: str
    content: str
    type: str
    activation_level: float
    access_frequency: int


class SynapseTypeStats(BaseModel):
    """Stats for a single synapse type."""

    count: int
    avg_weight: float
    total_reinforcements: int


class SynapseStatsInfo(BaseModel):
    """Aggregate synapse statistics."""

    avg_weight: float
    total_reinforcements: int
    by_type: dict[str, SynapseTypeStats] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Response with brain statistics."""

    brain_id: str
    neuron_count: int
    synapse_count: int
    fiber_count: int
    db_size_bytes: int | None = None
    hot_neurons: list[HotNeuronInfo] | None = None
    today_fibers_count: int | None = None
    synapse_stats: SynapseStatsInfo | None = None
    neuron_type_breakdown: dict[str, int] | None = None
    oldest_memory: str | None = None
    newest_memory: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str


class ImportBrainRequest(BaseModel):
    """Request to import a brain from a snapshot."""

    brain_id: str = Field(..., description="Brain ID in the snapshot")
    brain_name: str = Field(..., min_length=1, max_length=100, description="Brain name")
    exported_at: datetime = Field(..., description="When the snapshot was exported")
    version: str = Field(..., description="Snapshot version")
    neurons: list[dict[str, Any]] = Field(
        default_factory=list, max_length=100_000, description="Neuron data"
    )
    synapses: list[dict[str, Any]] = Field(
        default_factory=list, max_length=100_000, description="Synapse data"
    )
    fibers: list[dict[str, Any]] = Field(
        default_factory=list, max_length=100_000, description="Fiber data"
    )
    config: dict[str, Any] = Field(default_factory=dict, description="Brain configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MergeBrainRequest(BaseModel):
    """Request to merge a snapshot into an existing brain."""

    snapshot: ImportBrainRequest = Field(..., description="Incoming brain snapshot to merge")
    strategy: str = Field(
        "prefer_local",
        description="Conflict strategy: prefer_local, prefer_remote, prefer_recent, prefer_stronger",
    )


class ConflictItemResponse(BaseModel):
    """A single conflict resolution record."""

    entity_type: str
    local_id: str
    incoming_id: str
    resolution: str
    reason: str


class MergeReportResponse(BaseModel):
    """Response from a merge operation."""

    neurons_added: int
    neurons_updated: int
    neurons_skipped: int
    synapses_added: int
    synapses_updated: int
    fibers_added: int
    fibers_updated: int
    fibers_skipped: int
    conflicts: list[ConflictItemResponse]
    id_remap_count: int


class SuggestionItem(BaseModel):
    """A single neuron suggestion."""

    neuron_id: str
    content: str
    type: str
    access_frequency: int
    activation_level: float
    score: float


class SuggestResponse(BaseModel):
    """Response from neuron suggestion query."""

    suggestions: list[SuggestionItem]
    count: int


class IndexRequest(BaseModel):
    """Request to index a codebase directory."""

    action: str = Field(..., description="scan=index codebase, status=show what's indexed")
    path: str | None = Field(None, description="Directory to index (default: cwd)")
    extensions: list[str] | None = Field(None, description='File extensions (default: [".py"])')


class IndexResponse(BaseModel):
    """Response from codebase indexing."""

    files_indexed: int = Field(0, description="Number of files indexed")
    neurons_created: int = Field(0, description="Number of neurons created")
    synapses_created: int = Field(0, description="Number of synapses created")
    path: str | None = Field(None, description="Indexed directory path")
    message: str = Field(..., description="Human-readable summary")
    indexed_files: list[str] | None = Field(
        None, description="List of indexed file paths (status action)"
    )


class NeuronRequest(BaseModel):
    """Request to create or update a neuron."""

    id: str | None = Field(None, description="Neuron ID (generated if not provided)")
    type: str = Field(..., description="Neuron type (e.g., concept, entity, action)")
    content: str = Field(..., description="Neuron content", max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class NeuronUpdateRequest(BaseModel):
    """Request to update a neuron."""

    type: str | None = Field(None, description="Neuron type")
    content: str | None = Field(None, description="Neuron content", max_length=100_000)
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class NeuronResponse(BaseModel):
    """Response with neuron details."""

    id: str
    type: str
    content: str
    metadata: dict[str, Any]
    created_at: str


class NeuronStateRequest(BaseModel):
    """Request to update neuron state."""

    neuron_id: str
    activation_level: float = Field(0.0, ge=0, le=1)
    access_frequency: int = Field(0, ge=0)
    last_activated: datetime | None = None
    decay_rate: float = Field(0.1, ge=0, le=1)
    firing_threshold: float = Field(0.3, ge=0, le=1)
    refractory_until: datetime | None = None
    refractory_period_ms: int = Field(100, ge=0)
    homeostatic_target: float = Field(0.3, ge=0, le=1)


class SynapseRequest(BaseModel):
    """Request to create a synapse."""

    id: str | None = Field(None, description="Synapse ID (generated if not provided)")
    source_id: str = Field(..., description="Source neuron ID")
    target_id: str = Field(..., description="Target neuron ID")
    type: str = Field(..., description="Synapse type")
    weight: float = Field(0.5, ge=0, le=1)
    direction: str = Field("uni", description="Direction: uni (unidirectional), bi (bidirectional)")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class SynapseUpdateRequest(BaseModel):
    """Request to update a synapse."""

    weight: float | None = Field(None, ge=0, le=1)
    metadata: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
