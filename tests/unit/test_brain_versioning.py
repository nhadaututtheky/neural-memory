"""Tests for brain versioning engine."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.brain_versioning import (
    BrainVersion,
    VersionDiff,
    VersioningEngine,
    _compute_hash,
    _json_to_snapshot,
    _snapshot_to_json,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    """Storage with a brain and some data."""
    store = InMemoryStorage()
    brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="brain-1")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Add neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-2")
    n3 = Neuron.create(type=NeuronType.ACTION, content="deploy", neuron_id="n-3")
    await store.add_neuron(n1)
    await store.add_neuron(n2)
    await store.add_neuron(n3)

    # Add synapses
    s1 = Synapse.create(
        source_id="n-1",
        target_id="n-2",
        type=SynapseType.RELATED_TO,
        weight=0.7,
        synapse_id="s-1",
    )
    s2 = Synapse.create(
        source_id="n-3",
        target_id="n-1",
        type=SynapseType.INVOLVES,
        weight=0.8,
        synapse_id="s-2",
    )
    await store.add_synapse(s1)
    await store.add_synapse(s2)

    # Add fiber
    fiber = Fiber.create(
        neuron_ids={"n-1", "n-2", "n-3"},
        synapse_ids={"s-1", "s-2"},
        anchor_neuron_id="n-3",
        fiber_id="f-1",
        tags={"redis", "deployment"},
    )
    await store.add_fiber(fiber)

    return store


@pytest_asyncio.fixture
async def engine(storage: InMemoryStorage) -> VersioningEngine:
    """Versioning engine with populated storage."""
    return VersioningEngine(storage)


# ── Version creation tests ───────────────────────────────────────


class TestCreateVersion:
    """Test version creation."""

    @pytest.mark.asyncio
    async def test_create_version(self, engine: VersioningEngine) -> None:
        """Creating a version should return a BrainVersion."""
        version = await engine.create_version("brain-1", "v1-baseline", "Initial snapshot")
        assert isinstance(version, BrainVersion)
        assert version.version_name == "v1-baseline"
        assert version.description == "Initial snapshot"
        assert version.version_number == 1

    @pytest.mark.asyncio
    async def test_create_version_increments_number(self, engine: VersioningEngine) -> None:
        """Version numbers should auto-increment."""
        v1 = await engine.create_version("brain-1", "v1")
        v2 = await engine.create_version("brain-1", "v2")
        assert v1.version_number == 1
        assert v2.version_number == 2

    @pytest.mark.asyncio
    async def test_unique_name_constraint(self, engine: VersioningEngine) -> None:
        """Duplicate version names should raise ValueError."""
        await engine.create_version("brain-1", "baseline")
        with pytest.raises(ValueError, match="already exists"):
            await engine.create_version("brain-1", "baseline")

    @pytest.mark.asyncio
    async def test_version_counts_match(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Version should capture correct neuron/synapse/fiber counts."""
        version = await engine.create_version("brain-1", "counted")
        assert version.neuron_count == 3
        assert version.synapse_count == 2
        assert version.fiber_count == 1

    @pytest.mark.asyncio
    async def test_snapshot_hash_nonempty(self, engine: VersioningEngine) -> None:
        """Version should have a non-empty SHA-256 hash."""
        v1 = await engine.create_version("brain-1", "hash-1")
        assert len(v1.snapshot_hash) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_version_on_empty_brain(self) -> None:
        """Should be able to create version on empty brain."""
        store = InMemoryStorage()
        brain = Brain.create(name="empty", config=BrainConfig(), brain_id="empty-1")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        engine = VersioningEngine(store)
        version = await engine.create_version("empty-1", "empty-v1")
        assert version.neuron_count == 0
        assert version.synapse_count == 0
        assert version.fiber_count == 0


# ── Version listing tests ────────────────────────────────────────


class TestListVersions:
    """Test version listing."""

    @pytest.mark.asyncio
    async def test_list_versions_most_recent_first(self, engine: VersioningEngine) -> None:
        """Versions should be listed newest first."""
        await engine.create_version("brain-1", "v1")
        await engine.create_version("brain-1", "v2")
        await engine.create_version("brain-1", "v3")

        versions = await engine.list_versions("brain-1")
        assert len(versions) == 3
        assert versions[0].version_name == "v3"
        assert versions[1].version_name == "v2"
        assert versions[2].version_name == "v1"

    @pytest.mark.asyncio
    async def test_list_versions_with_limit(self, engine: VersioningEngine) -> None:
        """Limit parameter should restrict results."""
        await engine.create_version("brain-1", "v1")
        await engine.create_version("brain-1", "v2")
        await engine.create_version("brain-1", "v3")

        versions = await engine.list_versions("brain-1", limit=2)
        assert len(versions) == 2

    @pytest.mark.asyncio
    async def test_list_versions_empty(self, engine: VersioningEngine) -> None:
        """Listing versions for brain with no versions should return empty list."""
        versions = await engine.list_versions("brain-1")
        assert versions == []


# ── Get version tests ────────────────────────────────────────────


class TestGetVersion:
    """Test version retrieval."""

    @pytest.mark.asyncio
    async def test_get_existing_version(self, engine: VersioningEngine) -> None:
        """Should retrieve a version by ID."""
        created = await engine.create_version("brain-1", "v1")
        found = await engine.get_version("brain-1", created.id)
        assert found is not None
        assert found.id == created.id
        assert found.version_name == "v1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_version(self, engine: VersioningEngine) -> None:
        """Should return None for nonexistent version."""
        result = await engine.get_version("brain-1", "nonexistent")
        assert result is None


# ── Rollback tests ───────────────────────────────────────────────


class TestRollback:
    """Test rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_restores_state(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Rollback should restore the brain to the version's state."""
        # Create version with 3 neurons
        v1 = await engine.create_version("brain-1", "v1-before")

        # Add more data
        n4 = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-4")
        await storage.add_neuron(n4)

        # Verify 4 neurons now
        stats = await storage.get_stats("brain-1")
        assert stats["neuron_count"] == 4

        # Rollback to v1
        await engine.rollback("brain-1", v1.id)

        # Should be back to 3 neurons
        stats = await storage.get_stats("brain-1")
        assert stats["neuron_count"] == 3

    @pytest.mark.asyncio
    async def test_rollback_creates_new_entry(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Rollback should create a new version entry."""
        v1 = await engine.create_version("brain-1", "original")
        rollback_v = await engine.rollback("brain-1", v1.id)

        assert rollback_v.version_name.startswith("rollback-to-")
        assert rollback_v.version_number > v1.version_number

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_raises(self, engine: VersioningEngine) -> None:
        """Rollback to nonexistent version should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await engine.rollback("brain-1", "nonexistent-id")

    @pytest.mark.asyncio
    async def test_rollback_unique_name_on_duplicate(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Multiple rollbacks to same version should get unique names."""
        v1 = await engine.create_version("brain-1", "target")
        rb1 = await engine.rollback("brain-1", v1.id)
        rb2 = await engine.rollback("brain-1", v1.id)
        assert rb1.version_name != rb2.version_name


# ── Diff tests ───────────────────────────────────────────────────


class TestDiff:
    """Test version diffing."""

    @pytest.mark.asyncio
    async def test_diff_same_version_empty(self, engine: VersioningEngine) -> None:
        """Diff of same version with itself should show no changes."""
        v1 = await engine.create_version("brain-1", "v1")
        diff = await engine.diff("brain-1", v1.id, v1.id)
        assert diff.neurons_added == ()
        assert diff.neurons_removed == ()
        assert diff.synapses_added == ()
        assert diff.synapses_removed == ()
        assert diff.fibers_added == ()
        assert diff.fibers_removed == ()
        assert diff.summary == "No changes"

    @pytest.mark.asyncio
    async def test_diff_added_neurons(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Diff should detect added neurons."""
        v1 = await engine.create_version("brain-1", "v1")

        # Add neuron
        n4 = Neuron.create(type=NeuronType.ENTITY, content="new-entity", neuron_id="n-4")
        await storage.add_neuron(n4)

        v2 = await engine.create_version("brain-1", "v2")
        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert "n-4" in diff.neurons_added

    @pytest.mark.asyncio
    async def test_diff_removed_neurons(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Diff should detect removed neurons."""
        v1 = await engine.create_version("brain-1", "v1")

        # Remove neuron (and its synapses)
        await storage.delete_neuron("n-2")

        v2 = await engine.create_version("brain-1", "v2")
        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert "n-2" in diff.neurons_removed

    @pytest.mark.asyncio
    async def test_diff_removed_synapses(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Diff should detect removed synapses."""
        v1 = await engine.create_version("brain-1", "v1")

        await storage.delete_synapse("s-1")

        v2 = await engine.create_version("brain-1", "v2")
        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert "s-1" in diff.synapses_removed

    @pytest.mark.asyncio
    async def test_diff_weight_changes(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Diff should detect synapse weight changes."""
        v1 = await engine.create_version("brain-1", "v1")

        # Change weight
        synapse = await storage.get_synapse("s-1")
        assert synapse is not None
        updated = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=0.95,
            direction=synapse.direction,
            metadata=synapse.metadata,
            reinforced_count=synapse.reinforced_count,
            created_at=synapse.created_at,
        )
        await storage.update_synapse(updated)

        v2 = await engine.create_version("brain-1", "v2")
        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert len(diff.synapses_weight_changed) >= 1
        changed_ids = {s[0] for s in diff.synapses_weight_changed}
        assert "s-1" in changed_ids

    @pytest.mark.asyncio
    async def test_diff_nonexistent_version_raises(self, engine: VersioningEngine) -> None:
        """Diff with nonexistent version should raise ValueError."""
        v1 = await engine.create_version("brain-1", "v1")
        with pytest.raises(ValueError, match="not found"):
            await engine.diff("brain-1", v1.id, "nonexistent")


# ── Snapshot serialization tests ─────────────────────────────────


class TestSnapshotSerialization:
    """Test snapshot JSON serialization."""

    @pytest.mark.asyncio
    async def test_roundtrip(self, storage: InMemoryStorage) -> None:
        """Snapshot should survive JSON roundtrip."""
        snapshot = await storage.export_brain("brain-1")
        json_str = _snapshot_to_json(snapshot)
        restored = _json_to_snapshot(json_str)

        assert restored.brain_id == snapshot.brain_id
        assert restored.brain_name == snapshot.brain_name
        assert len(restored.neurons) == len(snapshot.neurons)
        assert len(restored.synapses) == len(snapshot.synapses)
        assert len(restored.fibers) == len(snapshot.fibers)

    @pytest.mark.asyncio
    async def test_hash_consistency(self, storage: InMemoryStorage) -> None:
        """Same snapshot should produce same hash."""
        snapshot = await storage.export_brain("brain-1")
        json1 = _snapshot_to_json(snapshot)
        json2 = _snapshot_to_json(snapshot)
        assert _compute_hash(json1) == _compute_hash(json2)


# ── VersionDiff dataclass tests ──────────────────────────────────


class TestVersionDiff:
    """Test VersionDiff properties."""

    def test_frozen(self) -> None:
        """VersionDiff should be immutable."""
        diff = VersionDiff(
            from_version="a",
            to_version="b",
            neurons_added=(),
            neurons_removed=(),
            neurons_modified=(),
            synapses_added=(),
            synapses_removed=(),
            synapses_weight_changed=(),
            fibers_added=(),
            fibers_removed=(),
            summary="No changes",
        )
        with pytest.raises(AttributeError):
            diff.summary = "Changed"  # type: ignore[misc]

    def test_brain_version_frozen(self) -> None:
        """BrainVersion should be immutable."""
        from datetime import datetime

        version = BrainVersion(
            id="v1",
            brain_id="b1",
            version_name="test",
            version_number=1,
            description="",
            neuron_count=0,
            synapse_count=0,
            fiber_count=0,
            snapshot_hash="abc",
            created_at=datetime.utcnow(),
        )
        with pytest.raises(AttributeError):
            version.version_name = "changed"  # type: ignore[misc]
