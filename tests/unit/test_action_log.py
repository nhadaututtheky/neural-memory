"""Tests for action event log storage — record, retrieve, filter, prune."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.action_event import ActionEvent
from neural_memory.core.brain import Brain
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    brain = Brain.create(name="action-test", brain_id="action-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# --- 1. Record and retrieve a single action event ---


@pytest.mark.asyncio
async def test_record_and_retrieve_single_event(store: InMemoryStorage) -> None:
    """Record one action and retrieve it via get_action_sequences."""
    await store.record_action("remember", action_context="test content", tags=("fact",))

    events = await store.get_action_sequences()
    assert len(events) == 1

    event = events[0]
    assert isinstance(event, ActionEvent)
    assert event.action_type == "remember"
    assert event.action_context == "test content"
    assert event.tags == ("fact",)
    assert event.brain_id == "action-brain"


# --- 2. Multiple actions returned in chronological order ---


@pytest.mark.asyncio
async def test_multiple_actions_chronological_order(store: InMemoryStorage) -> None:
    """Multiple recorded actions come back sorted by created_at ascending."""
    await store.record_action("recall")
    await store.record_action("remember")
    await store.record_action("context")

    events = await store.get_action_sequences()
    assert len(events) == 3
    assert events[0].action_type == "recall"
    assert events[1].action_type == "remember"
    assert events[2].action_type == "context"

    # Verify chronological ordering
    for i in range(len(events) - 1):
        assert events[i].created_at <= events[i + 1].created_at


# --- 3. Filter by session_id ---


@pytest.mark.asyncio
async def test_filter_by_session_id(store: InMemoryStorage) -> None:
    """get_action_sequences filters events by session_id."""
    await store.record_action("remember", session_id="sess-a")
    await store.record_action("recall", session_id="sess-b")
    await store.record_action("context", session_id="sess-a")

    events_a = await store.get_action_sequences(session_id="sess-a")
    assert len(events_a) == 2
    assert all(e.session_id == "sess-a" for e in events_a)

    events_b = await store.get_action_sequences(session_id="sess-b")
    assert len(events_b) == 1
    assert events_b[0].action_type == "recall"


# --- 4. Filter by since datetime ---


@pytest.mark.asyncio
async def test_filter_by_since(store: InMemoryStorage) -> None:
    """get_action_sequences excludes events older than 'since'."""
    await store.record_action("old-action")
    await store.record_action("new-action")

    # Age the first event
    brain_id = store._get_brain_id()
    store._action_events[brain_id][0]["created_at"] = datetime.utcnow() - timedelta(days=5)

    cutoff = datetime.utcnow() - timedelta(days=1)
    events = await store.get_action_sequences(since=cutoff)
    assert len(events) == 1
    assert events[0].action_type == "new-action"


# --- 5. Limit parameter ---


@pytest.mark.asyncio
async def test_limit_parameter(store: InMemoryStorage) -> None:
    """get_action_sequences respects the limit parameter."""
    for i in range(10):
        await store.record_action(f"action-{i}")

    events = await store.get_action_sequences(limit=3)
    assert len(events) == 3


# --- 6. Prune removes old events, keeps recent ones ---


@pytest.mark.asyncio
async def test_prune_removes_old_keeps_recent(store: InMemoryStorage) -> None:
    """prune_action_events removes old events and keeps recent ones."""
    await store.record_action("old-1")
    await store.record_action("old-2")
    await store.record_action("recent")

    # Age the first two events
    brain_id = store._get_brain_id()
    old_time = datetime.utcnow() - timedelta(days=10)
    store._action_events[brain_id][0]["created_at"] = old_time
    store._action_events[brain_id][1]["created_at"] = old_time

    cutoff = datetime.utcnow() - timedelta(days=1)
    pruned = await store.prune_action_events(older_than=cutoff)
    assert pruned == 2

    remaining = await store.get_action_sequences()
    assert len(remaining) == 1
    assert remaining[0].action_type == "recent"


# --- 7. Tags preserved as tuples ---


@pytest.mark.asyncio
async def test_tags_preserved_as_tuples(store: InMemoryStorage) -> None:
    """Tags are stored and returned as tuples."""
    await store.record_action("remember", tags=("python", "debug", "test"))

    events = await store.get_action_sequences()
    assert len(events) == 1
    assert events[0].tags == ("python", "debug", "test")
    assert isinstance(events[0].tags, tuple)


@pytest.mark.asyncio
async def test_tags_from_list_converted_to_tuple(store: InMemoryStorage) -> None:
    """Tags passed as a list are converted to a tuple."""
    await store.record_action("remember", tags=["alpha", "beta"])

    events = await store.get_action_sequences()
    assert events[0].tags == ("alpha", "beta")
    assert isinstance(events[0].tags, tuple)


# --- 8. Record returns a valid event ID ---


@pytest.mark.asyncio
async def test_record_returns_valid_event_id(store: InMemoryStorage) -> None:
    """record_action returns a non-empty string ID."""
    event_id = await store.record_action("recall")
    assert isinstance(event_id, str)
    assert len(event_id) > 0


@pytest.mark.asyncio
async def test_record_returns_unique_ids(store: InMemoryStorage) -> None:
    """Each record_action call returns a distinct ID."""
    id1 = await store.record_action("recall")
    id2 = await store.record_action("remember")
    assert id1 != id2


# --- 9. action_context and fiber_id are optional ---


@pytest.mark.asyncio
async def test_optional_action_context_defaults_empty(store: InMemoryStorage) -> None:
    """action_context defaults to empty string when not provided."""
    await store.record_action("recall")

    events = await store.get_action_sequences()
    assert events[0].action_context == ""


@pytest.mark.asyncio
async def test_optional_fiber_id_defaults_none(store: InMemoryStorage) -> None:
    """fiber_id defaults to None when not provided."""
    await store.record_action("recall")

    events = await store.get_action_sequences()
    assert events[0].fiber_id is None


@pytest.mark.asyncio
async def test_fiber_id_preserved_when_set(store: InMemoryStorage) -> None:
    """fiber_id is preserved when explicitly provided."""
    await store.record_action("remember", fiber_id="fiber-42")

    events = await store.get_action_sequences()
    assert events[0].fiber_id == "fiber-42"


@pytest.mark.asyncio
async def test_session_id_defaults_none(store: InMemoryStorage) -> None:
    """session_id defaults to None when not provided."""
    await store.record_action("context")

    events = await store.get_action_sequences()
    assert events[0].session_id is None
