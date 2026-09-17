"""Regression tests for issue #208 - concurrent synapse writes.

``SQLiteDialect`` owns a single write connection and commits after every
statement. Fanning ``add_synapse`` calls out concurrently interleaves their
awaits, so one task's commit can land while another task's statement is still
in progress:

    cannot commit transaction - SQL statements in progress

The INSERT often survives that race, which is what makes the bug quiet. The
Merkle cache invalidation and the change-log entry that run *after* the INSERT
are skipped, and the caller - which swallowed the exception with
``return_exceptions=True`` - never learns that the edge was only half-written.

The encode pipeline and the auto-capture handlers therefore write one memory at
a time. These tests drive the real code, so they fail if a concurrent fan-out
is ever reintroduced.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse
from neural_memory.engine.pipeline import PipelineContext
from neural_memory.engine.pipeline_steps import CoOccurrenceStep, CreateSynapsesStep
from neural_memory.mcp.auto_handler import AutoHandler
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.unified_config import UnifiedConfig

if TYPE_CHECKING:
    from typing import Any

    import pytest

# 20 entities produce 190 CO_OCCURS edges - enough writes that a concurrent
# fan-out loses some of them on essentially every run (issue #208 measured ~70%
# failures at 60 concurrent adds, and this suite stays red on the old code).
ENTITY_COUNT = 20


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> AsyncIterator[SQLStorage]:
    """Single-connection SQLStorage - the configuration issue #208 is about."""
    store = SQLStorage(SQLiteDialect(tmp_path / "concurrent-synapses.db"))
    await store.initialize()

    brain = Brain.create(name="issue-208", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store
    await store.close()


async def _seed_neurons(
    storage: SQLStorage,
    count: int,
    neuron_type: NeuronType = NeuronType.ENTITY,
) -> list[Neuron]:
    neurons: list[Neuron] = []
    for index in range(count):
        neuron = Neuron.create(type=neuron_type, content=f"node-{index:02d}")
        await storage.add_neuron(neuron)
        neurons.append(neuron)
    return neurons


def _context(entities: list[Neuron], content: str = "alpha beta gamma") -> PipelineContext:
    return PipelineContext(
        content=content,
        timestamp=datetime.now(UTC),
        metadata={},
        tags=set(),
        language="en",
        entity_neurons=list(entities),
    )


class _RecordingHandler(AutoHandler):
    """Minimal AutoHandler that records how many saves are in flight at once."""

    def __init__(self) -> None:
        self.config = UnifiedConfig()
        self._session_memories: list[dict[str, Any]] = []
        self.high_water = 0
        self._in_flight = 0

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        self._in_flight += 1
        self.high_water = max(self.high_water, self._in_flight)
        try:
            await asyncio.sleep(0)  # a concurrent fan-out would switch tasks here
            return {"fiber_id": "test"}
        finally:
            self._in_flight -= 1


async def test_co_occurrence_step_creates_every_synapse(storage: SQLStorage) -> None:
    """CoOccurrenceStep must persist all C(n, 2) edges, not just the lucky ones."""
    entities = await _seed_neurons(storage, ENTITY_COUNT)
    ctx = _context(entities)

    await CoOccurrenceStep().execute(ctx, storage, BrainConfig())

    expected = ENTITY_COUNT * (ENTITY_COUNT - 1) // 2
    assert len(ctx.synapses_created) == expected
    assert len(await storage.get_synapses()) == expected


async def test_create_synapses_step_creates_every_synapse(storage: SQLStorage) -> None:
    """CreateSynapsesStep must persist every anchor edge - the MCP report path."""
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="anchor node")
    await storage.add_neuron(anchor)
    intents = await _seed_neurons(storage, ENTITY_COUNT, NeuronType.INTENT)

    ctx = _context([])  # empty entity list isolates the anchor -> intent path
    ctx.anchor_neuron = anchor
    ctx.intent_neurons = list(intents)

    await CreateSynapsesStep().execute(ctx, storage, BrainConfig())

    assert len(ctx.synapses_created) == ENTITY_COUNT
    assert len(await storage.get_synapses(source_id=anchor.id)) == ENTITY_COUNT


async def test_steps_never_overlap_synapse_writes(
    storage: SQLStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No second add_synapse may start before the previous one returns.

    This is the invariant the fix buys: the shared connection can only serve one
    in-flight write at a time. Watching the high-water mark makes the check
    deterministic instead of depending on how often the race is actually lost.
    """
    in_flight = 0
    high_water = 0
    original = storage.add_synapse

    async def _tracked(synapse: Synapse) -> str:
        nonlocal in_flight, high_water
        in_flight += 1
        high_water = max(high_water, in_flight)
        try:
            await asyncio.sleep(0)  # a concurrent fan-out would switch tasks here
            return await original(synapse)
        finally:
            in_flight -= 1

    monkeypatch.setattr(storage, "add_synapse", _tracked)

    entities = await _seed_neurons(storage, ENTITY_COUNT)
    await CoOccurrenceStep().execute(_context(entities), storage, BrainConfig())

    assert high_water == 1


async def test_emergency_flush_saves_one_memory_at_a_time() -> None:
    """The flush path must not overlap _remember calls (issue #208)."""
    handler = _RecordingHandler()
    detected = [
        {"content": f"flush note {index}", "type": "fact", "priority": 5}
        for index in range(ENTITY_COUNT)
    ]

    saved = await handler._save_detected_memories_no_dedup(detected)

    assert len(saved) == ENTITY_COUNT
    assert handler.high_water == 1


async def test_auto_capture_saves_one_memory_at_a_time() -> None:
    """The auto-capture path must not overlap _remember calls (issue #208)."""
    handler = _RecordingHandler()
    detected = [
        {"content": f"auto note {index}", "type": "fact", "priority": 5, "confidence": 0.9}
        for index in range(ENTITY_COUNT)
    ]

    saved = await handler._save_detected_memories(detected)

    assert len(saved) == ENTITY_COUNT
    assert handler.high_water == 1
