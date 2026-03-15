"""CRUD tests for PostgreSQL neurons."""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def neurons_added(storage: Any, sample_neurons: list[Any]) -> list[Any]:
    """Add sample neurons and return them."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    return sample_neurons


@pytest.mark.asyncio
async def test_add_and_get_neuron(storage: Any, make_neuron: Any) -> None:
    """Add a neuron and retrieve it."""
    neuron = make_neuron(content="hello world")
    await storage.add_neuron(neuron)
    got = await storage.get_neuron(neuron.id)
    assert got is not None
    assert got.id == neuron.id
    assert got.content == "hello world"


@pytest.mark.asyncio
async def test_get_neuron_not_found(storage: Any) -> None:
    """Get non-existent neuron returns None."""
    got = await storage.get_neuron("nonexistent-id")
    assert got is None


@pytest.mark.asyncio
async def test_add_duplicate_neuron_raises(storage: Any, make_neuron: Any) -> None:
    """Adding duplicate neuron raises ValueError."""
    neuron = make_neuron(content="dup")
    await storage.add_neuron(neuron)
    with pytest.raises(ValueError, match="already exists"):
        await storage.add_neuron(neuron)


@pytest.mark.asyncio
async def test_get_neurons_batch(storage: Any, neurons_added: list[Any]) -> None:
    """Batch get neurons by IDs."""
    ids = [n.id for n in neurons_added]
    result = await storage.get_neurons_batch(ids)
    assert len(result) == len(neurons_added)
    for n in neurons_added:
        assert n.id in result
        assert result[n.id].content == n.content


@pytest.mark.asyncio
async def test_update_neuron(storage: Any, make_neuron: Any) -> None:
    """Update neuron content."""
    neuron = make_neuron(content="original")
    await storage.add_neuron(neuron)
    from dataclasses import replace

    updated = replace(neuron, content="updated")
    await storage.update_neuron(updated)
    got = await storage.get_neuron(neuron.id)
    assert got is not None
    assert got.content == "updated"


@pytest.mark.asyncio
async def test_delete_neuron(storage: Any, make_neuron: Any) -> None:
    """Delete neuron returns True."""
    neuron = make_neuron(content="to delete")
    await storage.add_neuron(neuron)
    ok = await storage.delete_neuron(neuron.id)
    assert ok is True
    assert await storage.get_neuron(neuron.id) is None


@pytest.mark.asyncio
async def test_find_neurons_by_content_exact(storage: Any, make_neuron: Any) -> None:
    """Find neurons by exact content."""
    neuron = make_neuron(content="exact match")
    await storage.add_neuron(neuron)
    found = await storage.find_neurons(content_exact="exact match", limit=10)
    assert len(found) == 1
    assert found[0].content == "exact match"


@pytest.mark.asyncio
async def test_get_neuron_state(storage: Any, make_neuron: Any) -> None:
    """Get and update neuron state."""
    neuron = make_neuron(content="with state")
    await storage.add_neuron(neuron)
    state = await storage.get_neuron_state(neuron.id)
    assert state is not None
    assert state.neuron_id == neuron.id
    assert state.activation_level == 0.0
