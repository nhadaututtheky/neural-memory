"""CRUD tests for PostgreSQL synapses."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_add_and_get_synapse(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Add neurons, then synapses, and retrieve."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    got = await storage.get_synapse(sample_synapses[0].id)
    assert got is not None
    assert got.source_id == sample_synapses[0].source_id
    assert got.target_id == sample_synapses[0].target_id


@pytest.mark.asyncio
async def test_get_synapse_not_found(storage: Any) -> None:
    """Get non-existent synapse returns None."""
    got = await storage.get_synapse("nonexistent-synapse")
    assert got is None


@pytest.mark.asyncio
async def test_get_synapses_by_source(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Get synapses by source_id."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    first_source = sample_synapses[0].source_id
    syns = await storage.get_synapses(source_id=first_source)
    assert len(syns) >= 1
    assert all(s.source_id == first_source for s in syns)


@pytest.mark.asyncio
async def test_get_neighbors(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Get neighbor neurons via synapses."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    center_id = sample_neurons[0].id
    neighbors = await storage.get_neighbors(center_id, direction="out")
    assert len(neighbors) >= 1
    for _neuron, synapse in neighbors:
        assert synapse.source_id == center_id


@pytest.mark.asyncio
async def test_delete_synapse(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Delete synapse returns True."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    sid = sample_synapses[0].id
    ok = await storage.delete_synapse(sid)
    assert ok is True
    assert await storage.get_synapse(sid) is None
