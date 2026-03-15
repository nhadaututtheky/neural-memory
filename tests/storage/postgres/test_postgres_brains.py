"""CRUD tests for PostgreSQL brains."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_get_brain(storage: Any, brain_id: str) -> None:
    """Get brain by ID."""
    brain = await storage.get_brain(brain_id)
    assert brain is not None
    assert brain.id == brain_id
    assert brain.name == brain_id


@pytest.mark.asyncio
async def test_find_brain_by_name(storage: Any, brain_id: str) -> None:
    """Find brain by name."""
    brain = await storage.find_brain_by_name(brain_id)
    assert brain is not None
    assert brain.name == brain_id


@pytest.mark.asyncio
async def test_get_stats(storage: Any, brain_id: str) -> None:
    """Get brain stats."""
    stats = await storage.get_stats(brain_id)
    assert "neuron_count" in stats
    assert "synapse_count" in stats
    assert "fiber_count" in stats
    assert stats["neuron_count"] >= 0
    assert stats["synapse_count"] >= 0
    assert stats["fiber_count"] >= 0


@pytest.mark.asyncio
async def test_get_enhanced_stats(storage: Any, brain_id: str) -> None:
    """Get enhanced brain stats."""
    stats = await storage.get_enhanced_stats(brain_id)
    assert "neuron_count" in stats
    assert "hot_neurons" in stats
    assert "synapse_stats" in stats
    assert "neuron_type_breakdown" in stats


@pytest.mark.asyncio
async def test_export_import_brain(
    storage: Any, brain_id: str, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Export brain and re-import to new brain."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    snapshot = await storage.export_brain(brain_id)
    assert snapshot.brain_id == brain_id
    assert len(snapshot.neurons) == len(sample_neurons)
    assert len(snapshot.synapses) == len(sample_synapses)

    new_brain_id = f"{brain_id}_imported"
    imported_id = await storage.import_brain(snapshot, target_brain_id=new_brain_id)
    assert imported_id == new_brain_id

    stats = await storage.get_stats(new_brain_id)
    assert stats["neuron_count"] == len(sample_neurons)
    assert stats["synapse_count"] == len(sample_synapses)

    await storage.clear(new_brain_id)
