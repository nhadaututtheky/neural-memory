"""Tests for integration status API — activity metrics and log."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.server.routes.integration_status import (
    _compute_metrics,
    _extract_source,
)
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    brain = Brain.create(name="int-status-test", brain_id="int-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# --- 1. Source extraction from session_id ---


def test_extract_source_none() -> None:
    """None session_id defaults to 'mcp'."""
    assert _extract_source(None) == "mcp"


def test_extract_source_mcp_prefix() -> None:
    """session_id starting with 'mcp-' extracts to 'mcp'."""
    assert _extract_source("mcp-12345") == "mcp"


def test_extract_source_openclaw_prefix() -> None:
    """session_id starting with 'openclaw-' extracts to 'openclaw'."""
    assert _extract_source("openclaw-67890") == "openclaw"


def test_extract_source_nanobot_prefix() -> None:
    """session_id starting with 'nanobot-' extracts to 'nanobot'."""
    assert _extract_source("nanobot-abc") == "nanobot"


def test_extract_source_unknown() -> None:
    """Unknown session_id prefix defaults to 'mcp'."""
    assert _extract_source("custom-thing") == "mcp"


# --- 2. Empty metrics ---


def test_compute_metrics_empty() -> None:
    """No events produces empty metrics list."""
    assert _compute_metrics([]) == []


# --- 3. Metrics computation ---


@pytest.mark.asyncio
async def test_metrics_today_counts(store: InMemoryStorage) -> None:
    """Record actions and verify counts match."""
    # Record 3 remember, 2 recall, 1 context
    for _ in range(3):
        await store.record_action("remember", session_id="mcp-1")
    for _ in range(2):
        await store.record_action("recall", session_id="mcp-1")
    await store.record_action("context", session_id="mcp-1")

    events = await store.get_action_sequences()
    metrics = _compute_metrics(events)

    assert len(metrics) == 1
    m = metrics[0]
    assert m.integration_id == "mcp"
    assert m.memories_today == 3
    assert m.recalls_today == 2
    assert m.contexts_today == 1
    assert m.total_today == 6


# --- 4. Source attribution grouping ---


@pytest.mark.asyncio
async def test_source_attribution_grouping(store: InMemoryStorage) -> None:
    """Actions with different session_id prefixes group into separate sources."""
    await store.record_action("remember", session_id="mcp-1")
    await store.record_action("recall", session_id="openclaw-2")
    await store.record_action("remember", session_id="openclaw-2")

    events = await store.get_action_sequences()
    metrics = _compute_metrics(events)

    by_source = {m.integration_id: m for m in metrics}
    assert "mcp" in by_source
    assert "openclaw" in by_source
    assert by_source["mcp"].memories_today == 1
    assert by_source["openclaw"].memories_today == 1
    assert by_source["openclaw"].recalls_today == 1


# --- 5. Activity log ordering (reverse chronological) ---


@pytest.mark.asyncio
async def test_activity_log_ordering(store: InMemoryStorage) -> None:
    """get_action_sequences returns ASC; endpoint reverses to DESC."""
    await store.record_action("recall", action_context="first")
    await store.record_action("remember", action_context="second")
    await store.record_action("context", action_context="third")

    events = await store.get_action_sequences()
    # Storage returns ASC
    assert events[0].action_context == "first"
    assert events[-1].action_context == "third"

    # Endpoint logic reverses
    reversed_events = list(reversed(events))
    assert reversed_events[0].action_context == "third"
    assert reversed_events[-1].action_context == "first"


# --- 6. Limit enforcement ---


@pytest.mark.asyncio
async def test_activity_limit(store: InMemoryStorage) -> None:
    """Limit parameter caps the number of events returned."""
    for i in range(20):
        await store.record_action("remember", action_context=f"item-{i}")

    events = await store.get_action_sequences(limit=10)
    assert len(events) == 10


# --- 7. Since filter excludes old events ---


@pytest.mark.asyncio
async def test_since_filter_excludes_old(store: InMemoryStorage) -> None:
    """Only events after 'since' timestamp are returned."""
    # Record some events now
    await store.record_action("remember", action_context="now")

    # Query with since = far in the future excludes all (naive to match storage)
    future = datetime.utcnow() + timedelta(hours=1)
    events = await store.get_action_sequences(since=future)
    assert len(events) == 0


# --- 8. Null session_id defaults to mcp in metrics ---


@pytest.mark.asyncio
async def test_null_session_id_defaults_mcp(store: InMemoryStorage) -> None:
    """Events with session_id=None are attributed to 'mcp'."""
    await store.record_action("remember", session_id=None)

    events = await store.get_action_sequences()
    metrics = _compute_metrics(events)

    assert len(metrics) == 1
    assert metrics[0].integration_id == "mcp"
