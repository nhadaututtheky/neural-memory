"""Row-to-model conversion for PostgreSQL (asyncpg records)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Get value from asyncpg Record with default."""
    try:
        val = record[key]
        return val if val is not None else default
    except (KeyError, TypeError):
        return default


def row_to_neuron(record: Any) -> Neuron:
    """Convert asyncpg record to Neuron."""
    return Neuron(
        id=str(record["id"]),
        type=NeuronType(record["type"]),
        content=str(record["content"]),
        metadata=json.loads(record["metadata"]) if record["metadata"] else {},
        content_hash=int(_get(record, "content_hash", 0)),
        created_at=(
            record["created_at"]
            if hasattr(record["created_at"], "isoformat")
            else datetime.fromisoformat(str(record["created_at"]))
        ),
    )


def row_to_neuron_state(record: Any) -> NeuronState:
    """Convert asyncpg record to NeuronState."""
    return NeuronState(
        neuron_id=str(record["neuron_id"]),
        activation_level=float(_get(record, "activation_level", 0.0)),
        access_frequency=int(_get(record, "access_frequency", 0)),
        last_activated=record["last_activated"],
        decay_rate=float(_get(record, "decay_rate", 0.1)),
        created_at=(
            record["created_at"]
            if hasattr(record["created_at"], "isoformat")
            else datetime.fromisoformat(str(record["created_at"]))
        ),
        firing_threshold=float(_get(record, "firing_threshold", 0.3)),
        refractory_until=record["refractory_until"],
        refractory_period_ms=float(_get(record, "refractory_period_ms", 500.0)),
        homeostatic_target=float(_get(record, "homeostatic_target", 0.5)),
    )


def row_to_synapse(record: Any) -> Synapse:
    """Convert asyncpg record to Synapse."""
    return Synapse(
        id=str(record["id"]),
        source_id=str(record["source_id"]),
        target_id=str(record["target_id"]),
        type=SynapseType(record["type"]),
        weight=float(record["weight"]),
        direction=Direction(_get(record, "direction", "uni")),
        metadata=json.loads(record["metadata"]) if record["metadata"] else {},
        reinforced_count=int(_get(record, "reinforced_count", 0)),
        last_activated=record["last_activated"],
        created_at=(
            record["created_at"]
            if hasattr(record["created_at"], "isoformat")
            else datetime.fromisoformat(str(record["created_at"]))
        ),
    )


def row_to_fiber(record: Any) -> Fiber:
    """Convert asyncpg record to Fiber."""
    neuron_ids = json.loads(record["neuron_ids"]) if record["neuron_ids"] else []
    synapse_ids = json.loads(record["synapse_ids"]) if record["synapse_ids"] else []
    pathway = json.loads(record["pathway"]) if record["pathway"] else []
    tags = set(json.loads(record["tags"])) if record["tags"] else set()
    auto_tags = set(json.loads(record["auto_tags"])) if record["auto_tags"] else set()
    agent_tags = set(json.loads(record["agent_tags"])) if record["agent_tags"] else set()
    metadata = json.loads(record["metadata"]) if record["metadata"] else {}

    created_at = record["created_at"]
    if not hasattr(created_at, "isoformat"):
        created_at = datetime.fromisoformat(str(created_at))

    last_conducted = record["last_conducted"]
    time_start = record["time_start"]
    time_end = record["time_end"]
    if time_start is not None and not hasattr(time_start, "isoformat"):
        time_start = datetime.fromisoformat(str(time_start))
    if time_end is not None and not hasattr(time_end, "isoformat"):
        time_end = datetime.fromisoformat(str(time_end))
    if last_conducted is not None and not hasattr(last_conducted, "isoformat"):
        last_conducted = datetime.fromisoformat(str(last_conducted))

    return Fiber(
        id=str(record["id"]),
        neuron_ids=set(neuron_ids),
        synapse_ids=set(synapse_ids),
        anchor_neuron_id=str(record["anchor_neuron_id"]),
        pathway=pathway,
        conductivity=float(_get(record, "conductivity", 1.0)),
        last_conducted=last_conducted,
        time_start=time_start,
        time_end=time_end,
        coherence=float(_get(record, "coherence", 0.0)),
        salience=float(_get(record, "salience", 0.0)),
        frequency=int(_get(record, "frequency", 0)),
        summary=record["summary"],
        tags=tags if tags else (agent_tags | auto_tags),
        auto_tags=auto_tags,
        agent_tags=agent_tags,
        metadata=metadata,
        compression_tier=int(_get(record, "compression_tier", 0)),
        pinned=bool(_get(record, "pinned", 0)),
        created_at=created_at,
    )


def row_to_brain(record: Any) -> Brain:
    """Convert asyncpg record to Brain."""
    config_data = json.loads(record["config"]) if record["config"] else {}
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
        embedding_enabled=config_data.get("embedding_enabled", False),
        embedding_provider=config_data.get("embedding_provider", "sentence_transformer"),
        embedding_model=config_data.get("embedding_model", "all-MiniLM-L6-v2"),
        embedding_similarity_threshold=config_data.get(
            "embedding_similarity_threshold", 0.7
        ),
    )
    shared_with = json.loads(record["shared_with"]) if record["shared_with"] else []
    return Brain(
        id=str(record["id"]),
        name=str(record["name"]),
        config=config,
        owner_id=record["owner_id"],
        is_public=bool(_get(record, "is_public", 0)),
        shared_with=shared_with,
        created_at=(
            record["created_at"]
            if hasattr(record["created_at"], "isoformat")
            else datetime.fromisoformat(str(record["created_at"]))
        ),
        updated_at=(
            record["updated_at"]
            if hasattr(record["updated_at"], "isoformat")
            else datetime.fromisoformat(str(record["updated_at"]))
        ),
    )
