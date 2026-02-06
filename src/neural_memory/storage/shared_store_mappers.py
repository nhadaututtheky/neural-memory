"""Conversion helpers for SharedStorage API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType


def dict_to_neuron(data: dict[str, Any]) -> Neuron:
    """Convert API response dict to Neuron."""
    return Neuron(
        id=data["id"],
        type=NeuronType(data["type"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
        created_at=datetime.fromisoformat(data["created_at"]),
    )


def dict_to_neuron_state(data: dict[str, Any]) -> NeuronState:
    """Convert API response dict to NeuronState."""
    return NeuronState(
        neuron_id=data["neuron_id"],
        activation_level=data.get("activation_level", 0.0),
        access_frequency=data.get("access_frequency", 0),
        last_activated=datetime.fromisoformat(data["last_activated"])
        if data.get("last_activated")
        else None,
        decay_rate=data.get("decay_rate", 0.1),
        created_at=datetime.fromisoformat(data["created_at"])
        if data.get("created_at")
        else datetime.now(),
    )


def dict_to_synapse(data: dict[str, Any]) -> Synapse:
    """Convert API response dict to Synapse."""
    return Synapse(
        id=data["id"],
        source_id=data["source_id"],
        target_id=data["target_id"],
        type=SynapseType(data["type"]),
        weight=data.get("weight", 0.5),
        direction=Direction(data.get("direction", "uni")),
        metadata=data.get("metadata", {}),
        reinforced_count=data.get("reinforced_count", 0),
        last_activated=datetime.fromisoformat(data["last_activated"])
        if data.get("last_activated")
        else None,
        created_at=datetime.fromisoformat(data["created_at"])
        if data.get("created_at")
        else datetime.now(),
    )


def dict_to_fiber(data: dict[str, Any]) -> Fiber:
    """Convert API response dict to Fiber."""
    return Fiber(
        id=data["id"],
        neuron_ids=frozenset(data.get("neuron_ids", [])),
        synapse_ids=frozenset(data.get("synapse_ids", [])),
        anchor_neuron_id=data["anchor_neuron_id"],
        time_start=datetime.fromisoformat(data["time_start"]) if data.get("time_start") else None,
        time_end=datetime.fromisoformat(data["time_end"]) if data.get("time_end") else None,
        coherence=data.get("coherence", 0.0),
        salience=data.get("salience", 0.0),
        frequency=data.get("frequency", 0),
        summary=data.get("summary"),
        tags=frozenset(data.get("tags", [])),
        created_at=datetime.fromisoformat(data["created_at"])
        if data.get("created_at")
        else datetime.now(),
    )


def dict_to_brain(data: dict[str, Any]) -> Brain:
    """Convert API response dict to Brain."""
    config_data = data.get("config", {})
    if isinstance(config_data, dict) and config_data:
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
        )
    else:
        config = BrainConfig()

    return Brain(
        id=data["id"],
        name=data["name"],
        config=config,
        owner_id=data.get("owner_id"),
        is_public=data.get("is_public", False),
        shared_with=data.get("shared_with", []),
        created_at=datetime.fromisoformat(data["created_at"])
        if data.get("created_at")
        else datetime.now(),
        updated_at=datetime.fromisoformat(data["updated_at"])
        if data.get("updated_at")
        else datetime.now(),
    )
