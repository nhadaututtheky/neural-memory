"""Tests for co-activation event storage — both InMemory and interface."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    brain = Brain.create(name="coact-test", brain_id="coact-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.mark.asyncio
async def test_record_and_retrieve_co_activation(store: InMemoryStorage) -> None:
    """Basic record + count retrieval."""
    await store.record_co_activation("n1", "n2", 0.8)
    await store.record_co_activation("n1", "n2", 0.6)
    await store.record_co_activation("n1", "n2", 0.7)

    counts = await store.get_co_activation_counts()
    assert len(counts) == 1
    neuron_a, neuron_b, count, avg_strength = counts[0]
    assert count == 3
    assert avg_strength == pytest.approx(0.7, abs=0.01)


@pytest.mark.asyncio
async def test_canonical_pair_ordering(store: InMemoryStorage) -> None:
    """Pairs are stored in canonical order (a < b)."""
    await store.record_co_activation("z-neuron", "a-neuron", 0.5)
    counts = await store.get_co_activation_counts()
    assert counts[0][0] == "a-neuron"
    assert counts[0][1] == "z-neuron"


@pytest.mark.asyncio
async def test_min_count_filter(store: InMemoryStorage) -> None:
    """min_count filters out pairs below threshold."""
    await store.record_co_activation("n1", "n2", 0.8)
    await store.record_co_activation("n1", "n2", 0.7)
    await store.record_co_activation("n3", "n4", 0.5)

    counts = await store.get_co_activation_counts(min_count=2)
    assert len(counts) == 1
    assert counts[0][0:2] == ("n1", "n2")


@pytest.mark.asyncio
async def test_since_filter(store: InMemoryStorage) -> None:
    """since filter excludes old events."""
    # Record an event
    await store.record_co_activation("n1", "n2", 0.8)

    # Manually age the event
    brain_id = store._get_brain_id()
    store._co_activations[brain_id][0]["created_at"] = datetime.utcnow() - timedelta(days=10)

    # Record a recent event
    await store.record_co_activation("n1", "n2", 0.9)

    counts = await store.get_co_activation_counts(since=datetime.utcnow() - timedelta(days=1))
    assert len(counts) == 1
    assert counts[0][2] == 1  # Only 1 recent event


@pytest.mark.asyncio
async def test_prune_co_activations(store: InMemoryStorage) -> None:
    """Pruning removes old events."""
    # Record events
    await store.record_co_activation("n1", "n2", 0.8)
    await store.record_co_activation("n3", "n4", 0.6)

    # Age all events
    brain_id = store._get_brain_id()
    for event in store._co_activations[brain_id]:
        event["created_at"] = datetime.utcnow() - timedelta(days=10)

    # Record a recent event
    await store.record_co_activation("n5", "n6", 0.9)

    pruned = await store.prune_co_activations(older_than=datetime.utcnow() - timedelta(days=1))
    assert pruned == 2

    counts = await store.get_co_activation_counts()
    assert len(counts) == 1


@pytest.mark.asyncio
async def test_multiple_pairs_sorted_by_count(store: InMemoryStorage) -> None:
    """Results are sorted by count descending."""
    for _ in range(5):
        await store.record_co_activation("a", "b", 0.8)
    for _ in range(3):
        await store.record_co_activation("c", "d", 0.6)
    for _ in range(7):
        await store.record_co_activation("e", "f", 0.9)

    counts = await store.get_co_activation_counts()
    assert len(counts) == 3
    assert counts[0][2] == 7  # e,f first
    assert counts[1][2] == 5  # a,b second
    assert counts[2][2] == 3  # c,d third


@pytest.mark.asyncio
async def test_record_returns_event_id(store: InMemoryStorage) -> None:
    """record_co_activation returns a unique event ID."""
    id1 = await store.record_co_activation("n1", "n2", 0.5)
    id2 = await store.record_co_activation("n1", "n2", 0.6)
    assert isinstance(id1, str)
    assert id1 != id2


@pytest.mark.asyncio
async def test_clear_removes_co_activations(store: InMemoryStorage) -> None:
    """Brain clear also clears co-activation events."""
    await store.record_co_activation("n1", "n2", 0.8)

    brain_id = store._get_brain_id()
    await store.clear(brain_id)

    counts = await store.get_co_activation_counts()
    assert counts == []
