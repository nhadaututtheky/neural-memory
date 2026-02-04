"""Tests for SQLite storage backend."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Create a temporary SQLite storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()

        # Create and set brain
        brain = Brain.create(name="test_brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        yield storage

        await storage.close()


class TestSQLiteNeurons:
    """Tests for neuron operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_neuron(self, storage: SQLiteStorage) -> None:
        """Test adding and retrieving a neuron."""
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="Test neuron",
            metadata={"key": "value"},
        )

        result_id = await storage.add_neuron(neuron)
        assert result_id == neuron.id

        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.id == neuron.id
        assert retrieved.type == NeuronType.CONCEPT
        assert retrieved.content == "Test neuron"
        assert retrieved.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_add_duplicate_neuron_raises(self, storage: SQLiteStorage) -> None:
        """Test that adding duplicate neuron raises error."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Test")
        await storage.add_neuron(neuron)

        with pytest.raises(ValueError, match="already exists"):
            await storage.add_neuron(neuron)

    @pytest.mark.asyncio
    async def test_find_neurons_by_type(self, storage: SQLiteStorage) -> None:
        """Test finding neurons by type."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Concept 1")
        n2 = Neuron.create(type=NeuronType.ACTION, content="Action 1")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Concept 2")

        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        concepts = await storage.find_neurons(type=NeuronType.CONCEPT)
        assert len(concepts) == 2

        actions = await storage.find_neurons(type=NeuronType.ACTION)
        assert len(actions) == 1

    @pytest.mark.asyncio
    async def test_find_neurons_by_content(self, storage: SQLiteStorage) -> None:
        """Test finding neurons by content."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Hello world")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Goodbye world")

        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        results = await storage.find_neurons(content_contains="Hello")
        assert len(results) == 1
        assert results[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_update_neuron(self, storage: SQLiteStorage) -> None:
        """Test updating a neuron."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Original")
        await storage.add_neuron(neuron)

        updated = Neuron(
            id=neuron.id,
            type=NeuronType.ACTION,
            content="Updated",
            metadata={"new": "data"},
            created_at=neuron.created_at,
        )
        await storage.update_neuron(updated)

        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.type == NeuronType.ACTION
        assert retrieved.content == "Updated"

    @pytest.mark.asyncio
    async def test_delete_neuron(self, storage: SQLiteStorage) -> None:
        """Test deleting a neuron."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="To delete")
        await storage.add_neuron(neuron)

        result = await storage.delete_neuron(neuron.id)
        assert result is True

        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is None


class TestSQLiteSynapses:
    """Tests for synapse operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_synapse(self, storage: SQLiteStorage) -> None:
        """Test adding and retrieving a synapse."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Target")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        synapse = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
            weight=0.8,
        )
        await storage.add_synapse(synapse)

        retrieved = await storage.get_synapse(synapse.id)
        assert retrieved is not None
        assert retrieved.source_id == n1.id
        assert retrieved.target_id == n2.id
        assert retrieved.weight == 0.8

    @pytest.mark.asyncio
    async def test_synapse_requires_neurons(self, storage: SQLiteStorage) -> None:
        """Test that synapse requires existing neurons."""
        synapse = Synapse.create(
            source_id="nonexistent-1",
            target_id="nonexistent-2",
            type=SynapseType.RELATED_TO,
        )

        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_synapse(synapse)

    @pytest.mark.asyncio
    async def test_get_synapses_by_source(self, storage: SQLiteStorage) -> None:
        """Test finding synapses by source."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Target1")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Target2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        s1 = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        s2 = Synapse.create(source_id=n1.id, target_id=n3.id, type=SynapseType.RELATED_TO)
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        synapses = await storage.get_synapses(source_id=n1.id)
        assert len(synapses) == 2


class TestSQLiteFibers:
    """Tests for fiber operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_fiber(self, storage: SQLiteStorage) -> None:
        """Test adding and retrieving a fiber."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Anchor")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Test fiber",
            tags={"tag1", "tag2"},
        )
        await storage.add_fiber(fiber)

        retrieved = await storage.get_fiber(fiber.id)
        assert retrieved is not None
        assert retrieved.summary == "Test fiber"
        assert retrieved.tags == {"tag1", "tag2"}

    @pytest.mark.asyncio
    async def test_find_fibers_by_salience(self, storage: SQLiteStorage) -> None:
        """Test finding fibers by minimum salience."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="N1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="N2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            summary="Low salience",
        )
        f1 = Fiber(
            id=f1.id,
            neuron_ids=f1.neuron_ids,
            synapse_ids=f1.synapse_ids,
            anchor_neuron_id=f1.anchor_neuron_id,
            salience=0.3,
            created_at=f1.created_at,
        )

        f2 = Fiber.create(
            neuron_ids={n2.id},
            synapse_ids=set(),
            anchor_neuron_id=n2.id,
            summary="High salience",
        )
        f2 = Fiber(
            id=f2.id,
            neuron_ids=f2.neuron_ids,
            synapse_ids=f2.synapse_ids,
            anchor_neuron_id=f2.anchor_neuron_id,
            salience=0.9,
            created_at=f2.created_at,
        )

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        results = await storage.find_fibers(min_salience=0.5)
        assert len(results) == 1
        assert results[0].salience == 0.9

    @pytest.mark.asyncio
    async def test_get_fibers_ordered(self, storage: SQLiteStorage) -> None:
        """Test getting fibers with ordering."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="N1")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            summary="First",
        )
        f2 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            summary="Second",
        )

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        fibers = await storage.get_fibers(limit=10, order_by="created_at", descending=True)
        assert len(fibers) == 2


class TestSQLiteTypedMemories:
    """Tests for typed memory operations."""

    @pytest.fixture
    async def storage_with_fiber(self, storage: SQLiteStorage) -> tuple[SQLiteStorage, Fiber]:
        """Create storage with a fiber."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Test")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Test fiber",
        )
        await storage.add_fiber(fiber)

        return storage, fiber

    @pytest.mark.asyncio
    async def test_add_and_get_typed_memory(
        self, storage_with_fiber: tuple[SQLiteStorage, Fiber]
    ) -> None:
        """Test adding and retrieving typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            priority=Priority.HIGH,
        )
        await storage.add_typed_memory(typed_mem)

        retrieved = await storage.get_typed_memory(fiber.id)
        assert retrieved is not None
        assert retrieved.memory_type == MemoryType.TODO
        assert retrieved.priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_find_typed_memories_by_type(
        self, storage_with_fiber: tuple[SQLiteStorage, Fiber]
    ) -> None:
        """Test finding typed memories by type."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
        )
        await storage.add_typed_memory(typed_mem)

        results = await storage.find_typed_memories(memory_type=MemoryType.DECISION)
        assert len(results) == 1

        results = await storage.find_typed_memories(memory_type=MemoryType.FACT)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_typed_memories_excludes_expired(
        self, storage_with_fiber: tuple[SQLiteStorage, Fiber]
    ) -> None:
        """Test that find excludes expired by default."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        # Should not find expired
        results = await storage.find_typed_memories()
        assert len(results) == 0

        # Should find with include_expired
        results = await storage.find_typed_memories(include_expired=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_expired_memories(
        self, storage_with_fiber: tuple[SQLiteStorage, Fiber]
    ) -> None:
        """Test getting expired memories."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        expired = await storage.get_expired_memories()
        assert len(expired) == 1


class TestSQLiteProjects:
    """Tests for project operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_project(self, storage: SQLiteStorage) -> None:
        """Test adding and retrieving a project."""
        project = Project.create(
            name="Test Project",
            description="A test project",
            tags={"test"},
        )
        await storage.add_project(project)

        retrieved = await storage.get_project(project.id)
        assert retrieved is not None
        assert retrieved.name == "Test Project"
        assert "test" in retrieved.tags

    @pytest.mark.asyncio
    async def test_get_project_by_name(self, storage: SQLiteStorage) -> None:
        """Test getting project by name."""
        project = Project.create(name="My Project")
        await storage.add_project(project)

        # Exact match
        result = await storage.get_project_by_name("My Project")
        assert result is not None

        # Case insensitive
        result = await storage.get_project_by_name("my project")
        assert result is not None

        # Not found
        result = await storage.get_project_by_name("Other")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_projects_active_only(self, storage: SQLiteStorage) -> None:
        """Test listing only active projects."""
        active = Project.create(name="Active")
        ended = Project(
            id="ended-id",
            name="Ended",
            start_date=datetime.utcnow() - timedelta(days=10),
            end_date=datetime.utcnow() - timedelta(days=1),
        )

        await storage.add_project(active)
        await storage.add_project(ended)

        # All projects
        all_projects = await storage.list_projects()
        assert len(all_projects) == 2

        # Active only
        active_projects = await storage.list_projects(active_only=True)
        assert len(active_projects) == 1
        assert active_projects[0].name == "Active"

    @pytest.mark.asyncio
    async def test_project_memory_association(self, storage: SQLiteStorage) -> None:
        """Test associating memories with projects."""
        # Create project
        project = Project.create(name="Sprint")
        await storage.add_project(project)

        # Create fiber and typed memory
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Task")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        await storage.add_fiber(fiber)

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            project_id=project.id,
        )
        await storage.add_typed_memory(typed_mem)

        # Find by project
        memories = await storage.get_project_memories(project.id)
        assert len(memories) == 1
        assert memories[0].project_id == project.id


class TestSQLiteExportImport:
    """Tests for export/import functionality."""

    @pytest.mark.asyncio
    async def test_export_and_import(self) -> None:
        """Test exporting and importing a brain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "export_test.db"
            storage = SQLiteStorage(db_path)
            await storage.initialize()

            # Create brain
            brain = Brain.create(name="export_brain")
            await storage.save_brain(brain)
            storage.set_brain(brain.id)

            # Create data
            neuron = Neuron.create(type=NeuronType.CONCEPT, content="Export test")
            await storage.add_neuron(neuron)

            fiber = Fiber.create(
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                summary="Test fiber",
            )
            await storage.add_fiber(fiber)

            typed_mem = TypedMemory.create(
                fiber_id=fiber.id,
                memory_type=MemoryType.FACT,
            )
            await storage.add_typed_memory(typed_mem)

            project = Project.create(name="Test Project")
            await storage.add_project(project)

            # Export
            snapshot = await storage.export_brain(brain.id)

            assert len(snapshot.neurons) == 1
            assert len(snapshot.fibers) == 1
            assert len(snapshot.metadata.get("typed_memories", [])) == 1
            assert len(snapshot.metadata.get("projects", [])) == 1

            # Import into new brain (different ID)
            new_brain_id = await storage.import_brain(snapshot, "imported_brain")

            # Verify import
            storage.set_brain(new_brain_id)

            neurons = await storage.find_neurons()
            assert len(neurons) == 1
            assert neurons[0].content == "Export test"

            projects = await storage.list_projects()
            assert len(projects) == 1
            assert projects[0].name == "Test Project"

            await storage.close()
            # Explicit close to release file handle on Windows
            import gc
            gc.collect()


class TestSQLiteGraphTraversal:
    """Tests for graph traversal operations."""

    @pytest.mark.asyncio
    async def test_get_neighbors(self, storage: SQLiteStorage) -> None:
        """Test getting neighboring neurons."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Center")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Neighbor1")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Neighbor2")

        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        s1 = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        s2 = Synapse.create(source_id=n1.id, target_id=n3.id, type=SynapseType.RELATED_TO)

        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        neighbors = await storage.get_neighbors(n1.id, direction="out")
        assert len(neighbors) == 2

    @pytest.mark.asyncio
    async def test_get_path(self, storage: SQLiteStorage) -> None:
        """Test finding path between neurons."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Start")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Middle")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="End")

        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        s1 = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.LEADS_TO)
        s2 = Synapse.create(source_id=n2.id, target_id=n3.id, type=SynapseType.LEADS_TO)

        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        path = await storage.get_path(n1.id, n3.id)
        assert path is not None
        assert len(path) == 2  # Two steps: n1->n2, n2->n3

    @pytest.mark.asyncio
    async def test_get_path_no_connection(self, storage: SQLiteStorage) -> None:
        """Test that no path is found for unconnected neurons."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="B")

        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        path = await storage.get_path(n1.id, n2.id)
        assert path is None


class TestSQLiteStats:
    """Tests for statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, storage: SQLiteStorage) -> None:
        """Test getting brain statistics."""
        # Add some data
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="N1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="N2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s1 = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        await storage.add_synapse(s1)

        fiber = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
        )
        await storage.add_fiber(fiber)

        project = Project.create(name="Test")
        await storage.add_project(project)

        stats = await storage.get_stats(storage._current_brain_id)

        assert stats["neuron_count"] == 2
        assert stats["synapse_count"] == 1
        assert stats["fiber_count"] == 1
        assert stats["project_count"] == 1


class TestSQLitePersistence:
    """Tests for data persistence across connections."""

    @pytest.mark.asyncio
    async def test_data_persists(self) -> None:
        """Test that data persists across connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persist.db"

            # First connection: create data
            storage1 = SQLiteStorage(db_path)
            await storage1.initialize()

            brain = Brain.create(name="persist_test")
            await storage1.save_brain(brain)
            storage1.set_brain(brain.id)

            neuron = Neuron.create(type=NeuronType.CONCEPT, content="Persistent")
            await storage1.add_neuron(neuron)

            await storage1.close()

            # Second connection: verify data
            storage2 = SQLiteStorage(db_path)
            await storage2.initialize()
            storage2.set_brain(brain.id)

            retrieved = await storage2.get_neuron(neuron.id)
            assert retrieved is not None
            assert retrieved.content == "Persistent"

            await storage2.close()
