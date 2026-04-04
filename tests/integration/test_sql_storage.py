"""Integration tests for SQLStorage with SQLiteDialect.

Verifies that initialize() creates all tables, sets up FTS, and that
basic neuron CRUD works end-to-end through the unified SQL adapter.
"""

from __future__ import annotations

import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest_asyncio.fixture
async def storage(tmp_path):
    """Create a SQLStorage backed by a temporary SQLite database."""
    dialect = SQLiteDialect(tmp_path / "test.db")
    store = SQLStorage(dialect)
    await store.initialize()

    # Create and set a brain context
    brain = Brain.create(name="test-brain", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store
    await store.close()


async def test_initialize_creates_neuron_cache(storage: SQLStorage):
    """Fix #1: _neuron_cache must be initialized in __init__."""
    assert storage._neuron_cache is not None


async def test_fts_enabled_after_initialize(storage: SQLStorage):
    """Fix #2: SQLite FTS5 must be enabled after initialize()."""
    assert storage._dialect.supports_fts is True


async def test_add_and_find_neuron(storage: SQLStorage):
    """Basic neuron CRUD through the unified adapter."""
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="test memory content for sql storage",
    )
    neuron_id = await storage.add_neuron(neuron)
    assert neuron_id is not None

    found = await storage.find_neurons(content_exact="test memory content for sql storage")
    assert len(found) == 1
    assert found[0].content == "test memory content for sql storage"
    assert found[0].type == NeuronType.CONCEPT


async def test_neuron_cache_populated_on_find(storage: SQLStorage):
    """Verify the neuron cache is used for exact-match lookups."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="cached concept")
    await storage.add_neuron(neuron)

    # First lookup: cache miss
    result1 = await storage.find_neurons(content_exact="cached concept")
    assert len(result1) == 1

    # Second lookup should hit cache
    result2 = await storage.find_neurons(content_exact="cached concept")
    assert len(result2) == 1
    assert result2[0].id == result1[0].id


async def test_fts_search(storage: SQLStorage):
    """FTS search should work after initialize() enables FTS5."""
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="quantum computing breakthrough in 2025",
    )
    await storage.add_neuron(neuron)

    results = await storage.find_neurons(content_contains="quantum computing")
    assert len(results) >= 1
    assert any("quantum" in r.content for r in results)


async def test_schema_creates_tables(storage: SQLStorage):
    """Verify core tables exist after initialize()."""
    tables = await storage._dialect.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = {row["name"] for row in tables}

    # Core tables that must exist
    for required in (
        "brains",
        "neurons",
        "neuron_states",
        "synapses",
        "fibers",
        "fiber_neurons",
        "typed_memories",
        "projects",
    ):
        assert required in table_names, f"Missing table: {required}"

    # FTS virtual tables
    assert "neurons_fts" in table_names, "Missing FTS table: neurons_fts"
    assert "fibers_fts" in table_names, "Missing FTS table: fibers_fts"
