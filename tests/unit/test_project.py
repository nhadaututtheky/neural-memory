"""Tests for Project model and storage."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.project import MemoryScope, Project
from neural_memory.storage.memory_store import InMemoryStorage


class TestProjectModel:
    """Tests for Project dataclass."""

    def test_create_project(self) -> None:
        """Test creating a project with factory method."""
        project = Project.create(
            name="Test Project",
            description="A test project",
        )

        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert project.id is not None
        assert project.is_ongoing is True
        assert project.is_active is True

    def test_create_project_with_duration(self) -> None:
        """Test creating a project with duration."""
        project = Project.create(
            name="Sprint",
            duration_days=14,
        )

        assert project.is_ongoing is False
        assert project.end_date is not None
        assert project.duration_days == 14
        # days_remaining should be 13 or 14 depending on timing
        assert project.days_remaining in (13, 14)

    def test_create_project_with_tags(self) -> None:
        """Test creating a project with tags."""
        project = Project.create(
            name="Tagged Project",
            tags={"backend", "api"},
        )

        assert project.tags == frozenset({"backend", "api"})

    def test_create_project_with_priority(self) -> None:
        """Test creating a project with custom priority."""
        project = Project.create(
            name="High Priority",
            priority=2.5,
        )

        assert project.priority == 2.5

    def test_is_active_ongoing(self) -> None:
        """Test that ongoing projects are active."""
        project = Project.create(name="Ongoing")
        assert project.is_active is True
        assert project.is_ongoing is True

    def test_is_active_future_start(self) -> None:
        """Test that future projects are not active."""
        future_start = datetime.utcnow() + timedelta(days=7)
        project = Project.create(
            name="Future",
            start_date=future_start,
        )

        assert project.is_active is False

    def test_is_active_ended(self) -> None:
        """Test that ended projects are not active."""
        past_end = datetime.utcnow() - timedelta(days=1)
        past_start = datetime.utcnow() - timedelta(days=10)

        project = Project(
            id="test-id",
            name="Ended",
            start_date=past_start,
            end_date=past_end,
        )

        assert project.is_active is False

    def test_contains_date(self) -> None:
        """Test date containment check."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        project = Project(
            id="test-id",
            name="January",
            start_date=start,
            end_date=end,
        )

        # Within range
        assert project.contains_date(datetime(2024, 1, 15)) is True

        # Before start
        assert project.contains_date(datetime(2023, 12, 31)) is False

        # After end
        assert project.contains_date(datetime(2024, 2, 1)) is False

    def test_with_end_date(self) -> None:
        """Test creating a copy with new end date."""
        project = Project.create(name="Test")
        new_end = datetime.utcnow() + timedelta(days=30)

        updated = project.with_end_date(new_end)

        assert updated.end_date == new_end
        assert updated.id == project.id  # ID preserved
        assert updated.name == project.name

    def test_with_extended_deadline(self) -> None:
        """Test extending project deadline."""
        project = Project.create(
            name="Sprint",
            duration_days=7,
        )

        extended = project.with_extended_deadline(7)

        assert extended.duration_days == 14

    def test_with_extended_deadline_ongoing_raises(self) -> None:
        """Test that extending ongoing project raises error."""
        project = Project.create(name="Ongoing")

        with pytest.raises(ValueError, match="Cannot extend ongoing project"):
            project.with_extended_deadline(7)

    def test_with_tags(self) -> None:
        """Test creating a copy with new tags."""
        project = Project.create(name="Test", tags={"old"})

        updated = project.with_tags({"new", "tags"})

        assert updated.tags == frozenset({"new", "tags"})
        assert project.tags == frozenset({"old"})  # Original unchanged

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        project = Project.create(
            name="Test",
            description="Desc",
            tags={"tag1"},
            priority=1.5,
        )

        data = project.to_dict()

        assert data["name"] == "Test"
        assert data["description"] == "Desc"
        assert data["tags"] == ["tag1"]
        assert data["priority"] == 1.5
        assert "start_date" in data
        assert "created_at" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        project = Project.create(
            name="Test",
            duration_days=14,
            tags={"backend"},
        )

        data = project.to_dict()
        restored = Project.from_dict(data)

        assert restored.id == project.id
        assert restored.name == project.name
        assert restored.tags == project.tags
        assert restored.end_date is not None


class TestMemoryScope:
    """Tests for MemoryScope."""

    def test_for_project(self) -> None:
        """Test creating scope for project."""
        scope = MemoryScope.for_project("proj-123")

        assert scope.project_id == "proj-123"
        assert scope.time_window_days == 7  # Default

    def test_recent(self) -> None:
        """Test creating scope for recent memories."""
        scope = MemoryScope.recent(days=14)

        assert scope.time_window_days == 14
        assert scope.project_id is None

    def test_with_tags(self) -> None:
        """Test creating scope with tags."""
        scope = MemoryScope.with_tags({"api", "auth"})

        assert scope.tags == frozenset({"api", "auth"})

    def test_matches_project(self) -> None:
        """Test matching by project."""
        scope = MemoryScope.for_project("proj-123")

        assert scope.matches(project_id="proj-123") is True
        assert scope.matches(project_id="proj-456") is False
        assert scope.matches(project_id=None) is False

    def test_matches_time_window(self) -> None:
        """Test matching by time window."""
        scope = MemoryScope.recent(days=7)

        now = datetime.utcnow()
        recent = now - timedelta(days=3)
        old = now - timedelta(days=10)

        assert scope.matches(created_at=recent) is True
        assert scope.matches(created_at=old) is False

    def test_matches_tags(self) -> None:
        """Test matching by tags."""
        scope = MemoryScope.with_tags({"api", "auth"})

        assert scope.matches(tags=frozenset({"api", "other"})) is True
        assert scope.matches(tags=frozenset({"unrelated"})) is False

    def test_relevance_boost_recency(self) -> None:
        """Test relevance boost for recency."""
        scope = MemoryScope.recent(days=7)

        now = datetime.utcnow()
        recent = now - timedelta(days=1)
        older = now - timedelta(days=6)

        recent_boost = scope.relevance_boost(created_at=recent)
        older_boost = scope.relevance_boost(created_at=older)

        assert recent_boost > older_boost
        assert recent_boost > 1.0

    def test_relevance_boost_project_priority(self) -> None:
        """Test relevance boost from project priority."""
        scope = MemoryScope()

        low_priority_boost = scope.relevance_boost(project_priority=0.5)
        high_priority_boost = scope.relevance_boost(project_priority=2.0)

        assert high_priority_boost > low_priority_boost


class TestProjectStorage:
    """Tests for Project storage operations."""

    @pytest.fixture
    async def storage(self) -> InMemoryStorage:
        """Create storage with a test brain."""
        storage = InMemoryStorage()
        brain = Brain.create(name="test_brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        return storage

    @pytest.mark.asyncio
    async def test_add_project(self, storage: InMemoryStorage) -> None:
        """Test adding a project."""
        project = Project.create(name="Test Project")

        result = await storage.add_project(project)

        assert result == project.id

    @pytest.mark.asyncio
    async def test_add_project_duplicate_raises(self, storage: InMemoryStorage) -> None:
        """Test that adding duplicate project raises error."""
        project = Project.create(name="Test")
        await storage.add_project(project)

        with pytest.raises(ValueError, match="already exists"):
            await storage.add_project(project)

    @pytest.mark.asyncio
    async def test_get_project(self, storage: InMemoryStorage) -> None:
        """Test getting a project by ID."""
        project = Project.create(name="Test")
        await storage.add_project(project)

        result = await storage.get_project(project.id)

        assert result is not None
        assert result.name == "Test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_project(self, storage: InMemoryStorage) -> None:
        """Test getting a nonexistent project."""
        result = await storage.get_project("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_project_by_name(self, storage: InMemoryStorage) -> None:
        """Test getting a project by name."""
        project = Project.create(name="My Project")
        await storage.add_project(project)

        result = await storage.get_project_by_name("My Project")
        assert result is not None
        assert result.id == project.id

        # Case insensitive
        result = await storage.get_project_by_name("my project")
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_projects(self, storage: InMemoryStorage) -> None:
        """Test listing projects."""
        p1 = Project.create(name="Project 1", priority=1.0)
        p2 = Project.create(name="Project 2", priority=2.0)
        await storage.add_project(p1)
        await storage.add_project(p2)

        results = await storage.list_projects()

        assert len(results) == 2
        # Higher priority first
        assert results[0].name == "Project 2"

    @pytest.mark.asyncio
    async def test_list_projects_active_only(self, storage: InMemoryStorage) -> None:
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

        results = await storage.list_projects(active_only=True)

        assert len(results) == 1
        assert results[0].name == "Active"

    @pytest.mark.asyncio
    async def test_update_project(self, storage: InMemoryStorage) -> None:
        """Test updating a project."""
        project = Project.create(name="Original", duration_days=7)
        await storage.add_project(project)

        updated = project.with_extended_deadline(7)
        await storage.update_project(updated)

        result = await storage.get_project(project.id)
        assert result is not None
        assert result.duration_days == 14

    @pytest.mark.asyncio
    async def test_delete_project(self, storage: InMemoryStorage) -> None:
        """Test deleting a project."""
        project = Project.create(name="ToDelete")
        await storage.add_project(project)

        result = await storage.delete_project(project.id)

        assert result is True
        assert await storage.get_project(project.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_project(self, storage: InMemoryStorage) -> None:
        """Test deleting nonexistent project."""
        result = await storage.delete_project("nonexistent")
        assert result is False


class TestProjectMemoryAssociation:
    """Tests for associating memories with projects."""

    @pytest.fixture
    async def storage_with_project(self) -> tuple[InMemoryStorage, Project, Fiber]:
        """Create storage with a project and fiber."""
        storage = InMemoryStorage()
        brain = Brain.create(name="test_brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create project
        project = Project.create(name="Test Project")
        await storage.add_project(project)

        # Create neuron and fiber
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="Test content",
        )
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Test fiber",
        )
        await storage.add_fiber(fiber)

        return storage, project, fiber

    @pytest.mark.asyncio
    async def test_add_memory_with_project(
        self, storage_with_project: tuple[InMemoryStorage, Project, Fiber]
    ) -> None:
        """Test adding a typed memory with project association."""
        storage, project, fiber = storage_with_project

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            project_id=project.id,
        )
        await storage.add_typed_memory(typed_mem)

        result = await storage.get_typed_memory(fiber.id)
        assert result is not None
        assert result.project_id == project.id

    @pytest.mark.asyncio
    async def test_get_project_memories(
        self, storage_with_project: tuple[InMemoryStorage, Project, Fiber]
    ) -> None:
        """Test getting all memories for a project."""
        storage, project, fiber = storage_with_project

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            project_id=project.id,
        )
        await storage.add_typed_memory(typed_mem)

        memories = await storage.get_project_memories(project.id)

        assert len(memories) == 1
        assert memories[0].fiber_id == fiber.id

    @pytest.mark.asyncio
    async def test_find_memories_by_project(
        self, storage_with_project: tuple[InMemoryStorage, Project, Fiber]
    ) -> None:
        """Test finding typed memories filtered by project."""
        storage, project, fiber = storage_with_project

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            project_id=project.id,
        )
        await storage.add_typed_memory(typed_mem)

        # Find by project
        results = await storage.find_typed_memories(project_id=project.id)
        assert len(results) == 1

        # Find by different project (should be empty)
        results = await storage.find_typed_memories(project_id="other-project")
        assert len(results) == 0


class TestProjectExportImport:
    """Tests for project export/import."""

    @pytest.fixture
    async def storage_with_data(self) -> InMemoryStorage:
        """Create storage with project and memories."""
        storage = InMemoryStorage()
        brain = Brain.create(name="test_brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create project
        project = Project.create(
            name="Export Test",
            tags={"test"},
            duration_days=14,
        )
        await storage.add_project(project)

        # Create fiber and memory
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="Export content",
        )
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Export fiber",
        )
        await storage.add_fiber(fiber)

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            project_id=project.id,
        )
        await storage.add_typed_memory(typed_mem)

        return storage

    @pytest.mark.asyncio
    async def test_export_includes_projects(self, storage_with_data: InMemoryStorage) -> None:
        """Test that export includes projects."""
        snapshot = await storage_with_data.export_brain(storage_with_data._current_brain_id)

        assert "projects" in snapshot.metadata
        projects = snapshot.metadata["projects"]
        assert len(projects) == 1
        assert projects[0]["name"] == "Export Test"
        assert "test" in projects[0]["tags"]

    @pytest.mark.asyncio
    async def test_import_restores_projects(self, storage_with_data: InMemoryStorage) -> None:
        """Test that import restores projects."""
        snapshot = await storage_with_data.export_brain(storage_with_data._current_brain_id)

        # Create new storage and import
        new_storage = InMemoryStorage()
        await new_storage.import_brain(snapshot, "imported_brain")
        new_storage.set_brain("imported_brain")

        # Verify project restored
        projects = await new_storage.list_projects()
        assert len(projects) == 1
        assert projects[0].name == "Export Test"
        assert "test" in projects[0].tags

        # Verify memory-project association
        typed_mems = await new_storage.find_typed_memories(project_id=projects[0].id)
        assert len(typed_mems) == 1
