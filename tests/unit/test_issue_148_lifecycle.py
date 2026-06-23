"""Regression tests for issue #148: lifecycle integrity + CLI parity.

Covers two failure modes that combined to break agent coordination:

1. ``find_fibers_batch`` returning soft-deleted fibers because the SQL did
   not LEFT JOIN ``typed_memories`` to skip expired rows. Recall would then
   surface a fiber that ``nmem_show`` / ``nmem_forget`` had already retired.
2. ``_show`` and ``_forget`` short-circuiting on missing ``typed_memory``
   rows, which made untyped fibers (auto-extracted, legacy) unreachable
   despite being visible to recall.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.mcp.server import MCPServer
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Temporary SQLite storage with a test brain (matches existing tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SQLiteStorage(db_path)
        await store.initialize()

        brain = Brain.create(name="test_brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        yield store

        await store.close()


def _make_server() -> MCPServer:
    """Build an MCPServer with mocked config (mirrors test_edit_forget._make_server)."""
    server = MCPServer.__new__(MCPServer)
    server._config = MagicMock()
    server._config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    server._config.safety = MagicMock(auto_redact_min_severity=3)
    server._config.auto = MagicMock(enabled=False)
    server._config.dedup = MagicMock(enabled=False)
    server._config.tool_tier = MagicMock(tier="full")
    server._storage = None
    server._hooks = None
    server._eternal_trigger_count = 0
    return server


class TestFindFibersBatchExpiredFilter:
    """Soft-deleted fibers must NOT be returned by recall."""

    @pytest.mark.asyncio
    async def test_soft_deleted_fiber_not_returned(self, storage: SQLiteStorage) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Python")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="async")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Two fibers sharing the same neuron — one will be soft-deleted.
        live = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
        )
        retired = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
        )
        await storage.add_fiber(live)
        await storage.add_fiber(retired)

        live_tm = TypedMemory.create(
            fiber_id=live.id,
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        await storage.add_typed_memory(live_tm)

        # Soft-delete the retired fiber by setting expires_at in the past.
        from dataclasses import replace as dc_replace

        retired_tm = TypedMemory.create(
            fiber_id=retired.id,
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        retired_tm = dc_replace(retired_tm, expires_at=utcnow() - timedelta(seconds=1))
        await storage.add_typed_memory(retired_tm)

        result = await storage.find_fibers_batch([n1.id], tags=None)

        ids = {f.id for f in result}
        assert live.id in ids
        assert retired.id not in ids, "Soft-deleted fiber leaked into recall"

    @pytest.mark.asyncio
    async def test_untyped_fiber_still_returned(self, storage: SQLiteStorage) -> None:
        """Fibers with no typed_memory row remain visible (LEFT JOIN preserves them)."""
        n = Neuron.create(type=NeuronType.CONCEPT, content="orphan-content")
        await storage.add_neuron(n)

        f = Fiber.create(
            neuron_ids={n.id},
            synapse_ids=set(),
            anchor_neuron_id=n.id,
        )
        await storage.add_fiber(f)
        # Intentionally skip add_typed_memory — this is the "untyped fiber" case.

        result = await storage.find_fibers_batch([n.id], tags=None)
        assert any(x.id == f.id for x in result), "LEFT JOIN dropped the untyped fiber"

    @pytest.mark.asyncio
    async def test_future_expiry_still_returned(self, storage: SQLiteStorage) -> None:
        """Fibers with expires_at in the future are not 'expired' and must surface."""
        n = Neuron.create(type=NeuronType.CONCEPT, content="future-expiry")
        await storage.add_neuron(n)

        f = Fiber.create(neuron_ids={n.id}, synapse_ids=set(), anchor_neuron_id=n.id)
        await storage.add_fiber(f)

        from dataclasses import replace as dc_replace

        tm = TypedMemory.create(
            fiber_id=f.id,
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        tm = dc_replace(tm, expires_at=utcnow() + timedelta(days=30))
        await storage.add_typed_memory(tm)

        result = await storage.find_fibers_batch([n.id], tags=None)
        assert any(x.id == f.id for x in result)


class TestShowUntypedFiber:
    """`nmem_show` must surface fibers even when typed_memory is missing."""

    @pytest.mark.asyncio
    async def test_show_returns_untyped_fiber(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = MagicMock()
        fiber.anchor_neuron_id = "neuron-1"
        fiber.created_at = utcnow()
        fiber.neuron_count = 1
        fiber.summary = "auto-extracted text"
        fiber.metadata = {}

        anchor = Neuron.create(
            type=NeuronType.CONCEPT, content="auto content", neuron_id="neuron-1"
        )

        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=anchor)
        storage.get_synapses = AsyncMock(return_value=[])
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_show", {"memory_id": "fiber-untyped"})

        # No more "Memory not found" — fiber data is returned with a warning.
        assert "error" not in result, f"Expected fiber to surface, got: {result}"
        assert result["memory_id"] == "fiber-untyped"
        assert result["content"] == "auto content"
        assert result["memory_type"] is None
        assert "warning" in result
        assert "untyped" in result["warning"].lower()


class TestForgetUntypedFiber:
    """`nmem_forget` must remove fibers even when typed_memory is missing."""

    @pytest.mark.asyncio
    async def test_hard_delete_untyped_fiber(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = MagicMock()
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.delete_typed_memory = AsyncMock()
        storage.delete_fiber = AsyncMock()
        storage.batch_save = AsyncMock()
        storage.disable_auto_save = MagicMock()
        storage.enable_auto_save = MagicMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_forget", {"memory_id": "fiber-untyped", "hard": True, "confirm": True}
        )
        assert result["status"] == "hard_deleted"
        assert result.get("untyped") is True
        # No typed_memory to delete, but the fiber must be removed.
        storage.delete_typed_memory.assert_not_awaited()
        storage.delete_fiber.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_soft_delete_untyped_fiber_rejected(self) -> None:
        """Soft delete on an untyped fiber has no expires_at to set; reject clearly."""
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = MagicMock()
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=fiber)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("nmem_forget", {"memory_id": "fiber-untyped"})
        assert "error" in result
        assert "untyped" in result["error"].lower()
        assert "hard=true" in result["error"].lower()
