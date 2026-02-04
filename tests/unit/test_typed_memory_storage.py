"""Tests for TypedMemory storage functionality."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
async def storage() -> InMemoryStorage:
    """Create storage with a test brain."""
    storage = InMemoryStorage()
    brain = Brain.create(name="test_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.fixture
async def storage_with_fiber(storage: InMemoryStorage) -> tuple[InMemoryStorage, Fiber]:
    """Create storage with a fiber."""
    # Create anchor neuron
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="Test memory content",
    )
    await storage.add_neuron(neuron)

    # Create fiber
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary="Test fiber",
    )
    await storage.add_fiber(fiber)

    return storage, fiber


class TestTypedMemoryStorage:
    """Tests for TypedMemory CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_typed_memory(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test adding a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
        )
        result = await storage.add_typed_memory(typed_mem)

        assert result == fiber.id

    @pytest.mark.asyncio
    async def test_add_typed_memory_requires_fiber(self, storage: InMemoryStorage) -> None:
        """Test that adding typed memory requires existing fiber."""
        typed_mem = TypedMemory.create(
            fiber_id="nonexistent-fiber",
            memory_type=MemoryType.FACT,
        )

        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_typed_memory(typed_mem)

    @pytest.mark.asyncio
    async def test_get_typed_memory(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test getting a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
        )
        await storage.add_typed_memory(typed_mem)

        result = await storage.get_typed_memory(fiber.id)

        assert result is not None
        assert result.memory_type == MemoryType.DECISION
        assert result.priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_get_nonexistent_typed_memory(self, storage: InMemoryStorage) -> None:
        """Test getting a nonexistent typed memory."""
        result = await storage.get_typed_memory("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_typed_memories_by_type(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test finding typed memories by type."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
        )
        await storage.add_typed_memory(typed_mem)

        # Find TODOs
        results = await storage.find_typed_memories(memory_type=MemoryType.TODO)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.TODO

        # Find DECISIONs (should be empty)
        results = await storage.find_typed_memories(memory_type=MemoryType.DECISION)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_typed_memories_by_priority(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test finding typed memories by minimum priority."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            priority=Priority.HIGH,
        )
        await storage.add_typed_memory(typed_mem)

        # Find with min_priority=NORMAL (should find)
        results = await storage.find_typed_memories(min_priority=Priority.NORMAL)
        assert len(results) == 1

        # Find with min_priority=CRITICAL (should not find)
        results = await storage.find_typed_memories(min_priority=Priority.CRITICAL)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_typed_memories_excludes_expired(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test that find excludes expired memories by default."""
        storage, fiber = storage_with_fiber

        # Create expired memory
        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.CONTEXT,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        # Default excludes expired
        results = await storage.find_typed_memories()
        assert len(results) == 0

        # Include expired
        results = await storage.find_typed_memories(include_expired=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_update_typed_memory(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test updating a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            priority=Priority.LOW,
        )
        await storage.add_typed_memory(typed_mem)

        # Update priority
        updated = typed_mem.with_priority(Priority.CRITICAL)
        await storage.update_typed_memory(updated)

        result = await storage.get_typed_memory(fiber.id)
        assert result is not None
        assert result.priority == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_delete_typed_memory(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test deleting a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
        )
        await storage.add_typed_memory(typed_mem)

        # Delete
        result = await storage.delete_typed_memory(fiber.id)
        assert result is True

        # Verify deleted
        assert await storage.get_typed_memory(fiber.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_typed_memory(self, storage: InMemoryStorage) -> None:
        """Test deleting nonexistent typed memory."""
        result = await storage.delete_typed_memory("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_expired_memories(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test getting expired memories."""
        storage, fiber = storage_with_fiber

        # Create expired memory
        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        expired = await storage.get_expired_memories()
        assert len(expired) == 1
        assert expired[0].fiber_id == fiber.id


class TestTypedMemoryExportImport:
    """Tests for TypedMemory export/import."""

    @pytest.mark.asyncio
    async def test_export_includes_typed_memories(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test that export includes typed memories."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
            source="test",
        )
        await storage.add_typed_memory(typed_mem)

        snapshot = await storage.export_brain(storage._current_brain_id)

        # Check metadata contains typed_memories
        assert "typed_memories" in snapshot.metadata
        tm_data = snapshot.metadata["typed_memories"]
        assert len(tm_data) == 1
        assert tm_data[0]["memory_type"] == "decision"
        assert tm_data[0]["priority"] == Priority.HIGH.value

    @pytest.mark.asyncio
    async def test_import_restores_typed_memories(self, storage_with_fiber: tuple[InMemoryStorage, Fiber]) -> None:
        """Test that import restores typed memories."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.INSIGHT,
            priority=Priority.NORMAL,
        )
        await storage.add_typed_memory(typed_mem)

        # Export
        snapshot = await storage.export_brain(storage._current_brain_id)

        # Create new storage and import
        new_storage = InMemoryStorage()
        await new_storage.import_brain(snapshot, "imported_brain")
        new_storage.set_brain("imported_brain")

        # Verify typed memory was restored
        result = await new_storage.get_typed_memory(fiber.id)
        assert result is not None
        assert result.memory_type == MemoryType.INSIGHT
        assert result.priority == Priority.NORMAL
