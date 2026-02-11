"""Row-to-model conversion functions for SQLite storage."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Direction, Synapse, SynapseType


def row_to_neuron(row: aiosqlite.Row) -> Neuron:
    """Convert database row to Neuron."""
    row_keys = row.keys()
    content_hash = row["content_hash"] if "content_hash" in row_keys else 0
    return Neuron(
        id=row["id"],
        type=NeuronType(row["type"]),
        content=row["content"],
        metadata=json.loads(row["metadata"]),
        content_hash=content_hash,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def row_to_neuron_state(row: aiosqlite.Row) -> NeuronState:
    """Convert database row to NeuronState."""
    row_keys = row.keys()

    # Handle new NeuronSpec v1 fields with fallback for pre-migration DBs
    firing_threshold = row["firing_threshold"] if "firing_threshold" in row_keys else 0.3
    refractory_until_raw = row["refractory_until"] if "refractory_until" in row_keys else None
    refractory_until = (
        datetime.fromisoformat(refractory_until_raw) if refractory_until_raw else None
    )
    refractory_period_ms = (
        row["refractory_period_ms"] if "refractory_period_ms" in row_keys else 500.0
    )
    homeostatic_target = row["homeostatic_target"] if "homeostatic_target" in row_keys else 0.5

    return NeuronState(
        neuron_id=row["neuron_id"],
        activation_level=row["activation_level"],
        access_frequency=row["access_frequency"],
        last_activated=(
            datetime.fromisoformat(row["last_activated"]) if row["last_activated"] else None
        ),
        decay_rate=row["decay_rate"],
        created_at=datetime.fromisoformat(row["created_at"]),
        firing_threshold=firing_threshold,
        refractory_until=refractory_until,
        refractory_period_ms=refractory_period_ms,
        homeostatic_target=homeostatic_target,
    )


def row_to_synapse(row: aiosqlite.Row) -> Synapse:
    """Convert database row to Synapse."""
    return Synapse(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        type=SynapseType(row["type"]),
        weight=row["weight"],
        direction=Direction(row["direction"]),
        metadata=json.loads(row["metadata"]),
        reinforced_count=row["reinforced_count"],
        last_activated=(
            datetime.fromisoformat(row["last_activated"]) if row["last_activated"] else None
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def row_to_fiber(row: aiosqlite.Row) -> Fiber:
    """Convert database row to Fiber."""
    row_keys = row.keys()

    # Handle pathway with fallback for older schema
    pathway_raw = row["pathway"] if "pathway" in row_keys else "[]"
    pathway = json.loads(pathway_raw) if pathway_raw else []

    # Handle conductivity with fallback
    conductivity = row["conductivity"] if "conductivity" in row_keys else 1.0

    # Handle last_conducted with fallback
    last_conducted_raw = row["last_conducted"] if "last_conducted" in row_keys else None
    last_conducted = datetime.fromisoformat(last_conducted_raw) if last_conducted_raw else None

    # Tag origin tracking (v0.14.0) with backward compat for pre-v8 schemas
    tags_raw = set(json.loads(row["tags"]))
    auto_tags: set[str] = set()
    agent_tags: set[str] = set()

    if "auto_tags" in row_keys and row["auto_tags"]:
        auto_tags = set(json.loads(row["auto_tags"]))
    if "agent_tags" in row_keys and row["agent_tags"]:
        agent_tags = set(json.loads(row["agent_tags"]))

    # Fallback: pre-v8 rows only have tags column → treat as agent_tags
    if not auto_tags and not agent_tags and tags_raw:
        agent_tags = tags_raw

    return Fiber(
        id=row["id"],
        neuron_ids=set(json.loads(row["neuron_ids"])),
        synapse_ids=set(json.loads(row["synapse_ids"])),
        anchor_neuron_id=row["anchor_neuron_id"],
        pathway=pathway,
        conductivity=conductivity,
        last_conducted=last_conducted,
        time_start=(datetime.fromisoformat(row["time_start"]) if row["time_start"] else None),
        time_end=(datetime.fromisoformat(row["time_end"]) if row["time_end"] else None),
        coherence=row["coherence"],
        salience=row["salience"],
        frequency=row["frequency"],
        summary=row["summary"],
        auto_tags=auto_tags,
        agent_tags=agent_tags,
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def row_to_typed_memory(row: aiosqlite.Row) -> TypedMemory:
    """Convert database row to TypedMemory."""
    prov_data = json.loads(row["provenance"])
    provenance = Provenance(
        source=prov_data.get("source", "unknown"),
        confidence=Confidence(prov_data.get("confidence", "medium")),
        verified=prov_data.get("verified", False),
        verified_at=(
            datetime.fromisoformat(prov_data["verified_at"])
            if prov_data.get("verified_at")
            else None
        ),
        created_by=prov_data.get("created_by", "unknown"),
        last_confirmed=(
            datetime.fromisoformat(prov_data["last_confirmed"])
            if prov_data.get("last_confirmed")
            else None
        ),
    )

    return TypedMemory(
        fiber_id=row["fiber_id"],
        memory_type=MemoryType(row["memory_type"]),
        priority=Priority(row["priority"]),
        provenance=provenance,
        expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
        project_id=row["project_id"],
        tags=frozenset(json.loads(row["tags"])),
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def row_to_project(row: aiosqlite.Row) -> Project:
    """Convert database row to Project."""
    return Project(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        start_date=datetime.fromisoformat(row["start_date"]),
        end_date=(datetime.fromisoformat(row["end_date"]) if row["end_date"] else None),
        tags=frozenset(json.loads(row["tags"])),
        priority=row["priority"],
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def row_to_brain(row: aiosqlite.Row) -> Brain:
    """Convert database row to Brain."""
    config_data = json.loads(row["config"])
    config = BrainConfig(
        decay_rate=config_data.get("decay_rate", 0.1),
        reinforcement_delta=config_data.get("reinforcement_delta", 0.05),
        activation_threshold=config_data.get("activation_threshold", 0.2),
        max_spread_hops=config_data.get("max_spread_hops", 4),
        max_context_tokens=config_data.get("max_context_tokens", 1500),
        default_synapse_weight=config_data.get("default_synapse_weight", 0.5),
        hebbian_delta=config_data.get("hebbian_delta", 0.03),
        hebbian_threshold=config_data.get("hebbian_threshold", 0.5),
        hebbian_initial_weight=config_data.get("hebbian_initial_weight", 0.2),
        consolidation_prune_threshold=config_data.get("consolidation_prune_threshold", 0.05),
        prune_min_inactive_days=config_data.get("prune_min_inactive_days", 7.0),
        merge_overlap_threshold=config_data.get("merge_overlap_threshold", 0.5),
        sigmoid_steepness=config_data.get("sigmoid_steepness", 6.0),
        default_firing_threshold=config_data.get("default_firing_threshold", 0.3),
        default_refractory_ms=config_data.get("default_refractory_ms", 500.0),
        lateral_inhibition_k=config_data.get("lateral_inhibition_k", 10),
        lateral_inhibition_factor=config_data.get("lateral_inhibition_factor", 0.3),
        learning_rate=config_data.get("learning_rate", 0.05),
        weight_normalization_budget=config_data.get("weight_normalization_budget", 5.0),
        novelty_boost_max=config_data.get("novelty_boost_max", 3.0),
        novelty_decay_rate=config_data.get("novelty_decay_rate", 0.06),
    )

    return Brain(
        id=row["id"],
        name=row["name"],
        config=config,
        owner_id=row["owner_id"],
        is_public=bool(row["is_public"]),
        shared_with=json.loads(row["shared_with"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def provenance_to_dict(provenance: Provenance) -> dict[str, object]:
    """Serialize Provenance to a JSON-compatible dict."""
    return {
        "source": provenance.source,
        "confidence": provenance.confidence.value,
        "verified": provenance.verified,
        "verified_at": (provenance.verified_at.isoformat() if provenance.verified_at else None),
        "created_by": provenance.created_by,
        "last_confirmed": (
            provenance.last_confirmed.isoformat() if provenance.last_confirmed else None
        ),
    }
