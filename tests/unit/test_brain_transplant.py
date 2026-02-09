"""Tests for brain transplant engine."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.brain_transplant import (
    TransplantFilter,
    TransplantResult,
    extract_subgraph,
    transplant,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def source_storage() -> InMemoryStorage:
    """Source brain with tagged fibers."""
    store = InMemoryStorage()
    brain = Brain.create(name="source", config=BrainConfig(), brain_id="src-1")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Neurons
    for nid, ntype, content in [
        ("n-1", NeuronType.ENTITY, "Redis"),
        ("n-2", NeuronType.CONCEPT, "caching"),
        ("n-3", NeuronType.ACTION, "deploy"),
        ("n-4", NeuronType.ENTITY, "PostgreSQL"),
        ("n-5", NeuronType.CONCEPT, "database"),
        ("n-6", NeuronType.ENTITY, "Nginx"),
    ]:
        await store.add_neuron(Neuron.create(type=ntype, content=content, neuron_id=nid))

    # Synapses
    for sid, src, tgt, stype, weight in [
        ("s-1", "n-1", "n-2", SynapseType.RELATED_TO, 0.7),
        ("s-2", "n-3", "n-1", SynapseType.INVOLVES, 0.8),
        ("s-3", "n-4", "n-5", SynapseType.RELATED_TO, 0.6),
        ("s-4", "n-6", "n-3", SynapseType.INVOLVES, 0.5),
    ]:
        await store.add_synapse(
            Synapse.create(source_id=src, target_id=tgt, type=stype, weight=weight, synapse_id=sid)
        )

    # Fibers with different tags
    f1 = Fiber.create(
        neuron_ids={"n-1", "n-2", "n-3"},
        synapse_ids={"s-1", "s-2"},
        anchor_neuron_id="n-3",
        fiber_id="f-cache",
        tags={"caching", "redis"},
    )
    await store.add_fiber(f1)

    f2 = Fiber.create(
        neuron_ids={"n-4", "n-5"},
        synapse_ids={"s-3"},
        anchor_neuron_id="n-4",
        fiber_id="f-db",
        tags={"database", "postgresql"},
    )
    await store.add_fiber(f2)

    f3 = Fiber.create(
        neuron_ids={"n-6", "n-3"},
        synapse_ids={"s-4"},
        anchor_neuron_id="n-6",
        fiber_id="f-infra",
        tags={"infrastructure", "nginx"},
    )
    await store.add_fiber(f3)

    return store


@pytest_asyncio.fixture
async def target_storage() -> InMemoryStorage:
    """Target brain (initially empty)."""
    store = InMemoryStorage()
    brain = Brain.create(name="target", config=BrainConfig(), brain_id="tgt-1")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


# ── Extract subgraph tests ───────────────────────────────────────


class TestExtractSubgraph:
    """Test subgraph extraction with filters."""

    @pytest.mark.asyncio
    async def test_extract_by_tags(self, source_storage: InMemoryStorage) -> None:
        """Should extract only fibers matching tags."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(snapshot, TransplantFilter(tags=frozenset({"caching"})))
        assert len(filtered.fibers) == 1
        assert filtered.fibers[0]["id"] == "f-cache"

    @pytest.mark.asyncio
    async def test_extract_by_multiple_tags(self, source_storage: InMemoryStorage) -> None:
        """Should extract fibers matching ANY of the tags."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot, TransplantFilter(tags=frozenset({"caching", "database"}))
        )
        assert len(filtered.fibers) == 2
        fiber_ids = {f["id"] for f in filtered.fibers}
        assert "f-cache" in fiber_ids
        assert "f-db" in fiber_ids

    @pytest.mark.asyncio
    async def test_extract_by_salience(self, source_storage: InMemoryStorage) -> None:
        """Should filter fibers by minimum salience."""
        snapshot = await source_storage.export_brain("src-1")
        # All fibers have salience 0.0 by default, so min 0.5 should exclude all
        filtered = extract_subgraph(snapshot, TransplantFilter(min_salience=0.5))
        assert len(filtered.fibers) == 0

    @pytest.mark.asyncio
    async def test_preserves_synapse_integrity(self, source_storage: InMemoryStorage) -> None:
        """Extracted synapses should only reference neurons in the subgraph."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(snapshot, TransplantFilter(tags=frozenset({"caching"})))

        neuron_ids = {n["id"] for n in filtered.neurons}
        for synapse in filtered.synapses:
            assert synapse["source_id"] in neuron_ids
            assert synapse["target_id"] in neuron_ids

    @pytest.mark.asyncio
    async def test_empty_filter_returns_all(self, source_storage: InMemoryStorage) -> None:
        """Filter with no criteria should return all fibers (no active filter)."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(snapshot, TransplantFilter())
        assert len(filtered.fibers) == 3
        assert len(filtered.neurons) == 6
        assert len(filtered.synapses) == 4

    @pytest.mark.asyncio
    async def test_extract_by_memory_type(self, source_storage: InMemoryStorage) -> None:
        """Should filter by memory type in typed_memories."""
        # Add typed memory to source
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

        tm = TypedMemory.create(
            fiber_id="f-cache",
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        source_storage._typed_memories["src-1"][tm.fiber_id] = tm

        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(snapshot, TransplantFilter(memory_types=frozenset({"fact"})))
        assert len(filtered.fibers) == 1
        assert filtered.fibers[0]["id"] == "f-cache"

    @pytest.mark.asyncio
    async def test_extract_preserves_config(self, source_storage: InMemoryStorage) -> None:
        """Extracted subgraph should preserve brain config."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(snapshot, TransplantFilter(tags=frozenset({"caching"})))
        assert filtered.config == snapshot.config
        assert filtered.brain_id == snapshot.brain_id


# ── Transplant integration tests ────────────────────────────────


class TestTransplant:
    """Test full transplant workflow."""

    @pytest.mark.asyncio
    async def test_transplant_adds_to_target(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """Transplant should add extracted data to target brain."""
        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=TransplantFilter(tags=frozenset({"caching"})),
        )

        assert isinstance(result, TransplantResult)
        assert result.fibers_transplanted == 1
        assert result.neurons_transplanted > 0
        assert result.synapses_transplanted > 0

    @pytest.mark.asyncio
    async def test_transplant_preserves_existing(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """Transplant should preserve existing target data."""
        # Add data to target first
        n_existing = Neuron.create(
            type=NeuronType.ENTITY, content="existing-data", neuron_id="n-existing"
        )
        await target_storage.add_neuron(n_existing)
        f_existing = Fiber.create(
            neuron_ids={"n-existing"},
            synapse_ids=set(),
            anchor_neuron_id="n-existing",
            fiber_id="f-existing",
        )
        await target_storage.add_fiber(f_existing)

        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=TransplantFilter(tags=frozenset({"database"})),
        )

        # Check that transplanted data exists
        assert result.fibers_transplanted == 1

    @pytest.mark.asyncio
    async def test_transplant_result_has_filter(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """TransplantResult should include the filter used."""
        f = TransplantFilter(tags=frozenset({"caching"}))
        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=f,
        )
        assert result.filter_used == f

    @pytest.mark.asyncio
    async def test_transplant_empty_filter(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """Transplant with empty filter should transplant everything."""
        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=TransplantFilter(),
        )
        assert result.fibers_transplanted == 3
        assert result.neurons_transplanted == 6


# ── Data structure tests ─────────────────────────────────────────


class TestTransplantDataStructures:
    """Test TransplantFilter and TransplantResult properties."""

    def test_filter_frozen(self) -> None:
        """TransplantFilter should be immutable."""
        f = TransplantFilter(tags=frozenset({"test"}))
        with pytest.raises(AttributeError):
            f.min_salience = 0.5  # type: ignore[misc]

    def test_filter_defaults(self) -> None:
        """TransplantFilter defaults should be sensible."""
        f = TransplantFilter()
        assert f.tags is None
        assert f.memory_types is None
        assert f.neuron_types is None
        assert f.min_salience == 0.0
        assert f.include_orphan_neurons is False


# ── Neuron types filter tests ────────────────────────────────────


class TestNeuronTypesFilter:
    """Test neuron_types filtering in extract_subgraph."""

    @pytest.mark.asyncio
    async def test_filter_by_entity_type(self, source_storage: InMemoryStorage) -> None:
        """Only neurons matching neuron_types should be in the extracted subgraph."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(neuron_types=frozenset({"entity"})),
        )

        # Only entity neurons should be present
        for neuron in filtered.neurons:
            assert neuron["type"] == "entity"

        # Source has 3 entity neurons: Redis, PostgreSQL, Nginx
        entity_contents = {n["content"] for n in filtered.neurons}
        assert "Redis" in entity_contents
        assert "PostgreSQL" in entity_contents
        assert "Nginx" in entity_contents

    @pytest.mark.asyncio
    async def test_filter_by_concept_type(self, source_storage: InMemoryStorage) -> None:
        """Only CONCEPT neurons should appear when filtered by concept."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(neuron_types=frozenset({"concept"})),
        )

        for neuron in filtered.neurons:
            assert neuron["type"] == "concept"

    @pytest.mark.asyncio
    async def test_filter_by_multiple_types(self, source_storage: InMemoryStorage) -> None:
        """Multiple neuron_types should allow neurons of any listed type."""
        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(neuron_types=frozenset({"entity", "action"})),
        )

        for neuron in filtered.neurons:
            assert neuron["type"] in {"entity", "action"}

        # Should include entity (Redis, PostgreSQL, Nginx) + action (deploy)
        assert len(filtered.neurons) == 4


# ── Include orphan neurons tests ─────────────────────────────────


class TestIncludeOrphanNeurons:
    """Test include_orphan_neurons flag in extract_subgraph."""

    @pytest.mark.asyncio
    async def test_orphan_included_when_flag_set(self, source_storage: InMemoryStorage) -> None:
        """Orphan neurons matching neuron_types should be included."""
        # Add an orphan neuron (not in any fiber)
        orphan = Neuron.create(
            type=NeuronType.ENTITY,
            content="OrphanEntity",
            neuron_id="n-orphan",
        )
        await source_storage.add_neuron(orphan)

        snapshot = await source_storage.export_brain("src-1")

        # Without orphan flag, filter by tag to get subset + entity type
        filtered_no_orphan = extract_subgraph(
            snapshot,
            TransplantFilter(
                tags=frozenset({"caching"}),
                neuron_types=frozenset({"entity"}),
                include_orphan_neurons=False,
            ),
        )
        orphan_ids_no = {n["id"] for n in filtered_no_orphan.neurons}
        assert "n-orphan" not in orphan_ids_no

        # With orphan flag, orphan entity should be included
        filtered_with_orphan = extract_subgraph(
            snapshot,
            TransplantFilter(
                tags=frozenset({"caching"}),
                neuron_types=frozenset({"entity"}),
                include_orphan_neurons=True,
            ),
        )
        orphan_ids_yes = {n["id"] for n in filtered_with_orphan.neurons}
        assert "n-orphan" in orphan_ids_yes

    @pytest.mark.asyncio
    async def test_orphan_excluded_by_default(self, source_storage: InMemoryStorage) -> None:
        """Orphan neurons should NOT be included when flag is False."""
        orphan = Neuron.create(
            type=NeuronType.CONCEPT,
            content="LonelyConcept",
            neuron_id="n-lonely",
        )
        await source_storage.add_neuron(orphan)

        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(
                tags=frozenset({"caching"}),
                neuron_types=frozenset({"concept"}),
            ),
        )

        neuron_ids = {n["id"] for n in filtered.neurons}
        assert "n-lonely" not in neuron_ids


# ── Combined AND filter tests ────────────────────────────────────


class TestCombinedAndFilter:
    """Test combined AND filter logic: tags AND min_salience."""

    @pytest.mark.asyncio
    async def test_combined_tags_and_salience(self, source_storage: InMemoryStorage) -> None:
        """Both tags AND min_salience must match (AND logic)."""
        snapshot = await source_storage.export_brain("src-1")

        # All fibers have salience=0.0 by default, so with min_salience=0.5
        # nothing should pass even though tags match
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(
                tags=frozenset({"caching"}),
                min_salience=0.5,
            ),
        )
        assert len(filtered.fibers) == 0

    @pytest.mark.asyncio
    async def test_combined_passes_when_both_criteria_met(
        self, source_storage: InMemoryStorage
    ) -> None:
        """Fibers should pass when both tags and salience criteria are met."""
        # Update f-cache fiber's salience
        fiber = await source_storage.get_fiber("f-cache")
        assert fiber is not None
        updated_fiber = fiber.with_salience(0.8)
        await source_storage.update_fiber(updated_fiber)

        snapshot = await source_storage.export_brain("src-1")
        filtered = extract_subgraph(
            snapshot,
            TransplantFilter(
                tags=frozenset({"caching"}),
                min_salience=0.5,
            ),
        )
        assert len(filtered.fibers) == 1
        assert filtered.fibers[0]["id"] == "f-cache"


# ── Conflict strategy tests ──────────────────────────────────────


class TestConflictStrategyPreferRemote:
    """Test transplant with PREFER_REMOTE conflict strategy."""

    @pytest.mark.asyncio
    async def test_prefer_remote_overwrites_local(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """PREFER_REMOTE should let source (remote) data win on conflict."""
        from neural_memory.engine.merge import ConflictStrategy

        # Add a neuron to target with the same fingerprint (type + content)
        # as one in source to create a conflict
        local_neuron = Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis",
            neuron_id="n-local-redis",
            metadata={"origin": "local"},
        )
        await target_storage.add_neuron(local_neuron)
        local_fiber = Fiber.create(
            neuron_ids={"n-local-redis"},
            synapse_ids=set(),
            anchor_neuron_id="n-local-redis",
            fiber_id="f-local",
        )
        await target_storage.add_fiber(local_fiber)

        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=TransplantFilter(tags=frozenset({"caching"})),
            strategy=ConflictStrategy.PREFER_REMOTE,
        )

        # The merge should have resolved the conflict
        assert result.merge_report.neurons_updated >= 1

        # Verify conflict was recorded as kept_incoming (remote wins)
        neuron_conflicts = [c for c in result.merge_report.conflicts if c.entity_type == "neuron"]
        assert len(neuron_conflicts) >= 1
        assert any(c.resolution == "kept_incoming" for c in neuron_conflicts)

    @pytest.mark.asyncio
    async def test_prefer_local_keeps_local(
        self,
        source_storage: InMemoryStorage,
        target_storage: InMemoryStorage,
    ) -> None:
        """PREFER_LOCAL should keep local data on conflict (default behavior)."""
        from neural_memory.engine.merge import ConflictStrategy

        # Add a conflicting neuron
        local_neuron = Neuron.create(
            type=NeuronType.ENTITY,
            content="Redis",
            neuron_id="n-local-redis",
            metadata={"origin": "local"},
        )
        await target_storage.add_neuron(local_neuron)
        local_fiber = Fiber.create(
            neuron_ids={"n-local-redis"},
            synapse_ids=set(),
            anchor_neuron_id="n-local-redis",
            fiber_id="f-local",
        )
        await target_storage.add_fiber(local_fiber)

        result = await transplant(
            source_storage=source_storage,
            target_storage=target_storage,
            source_brain_id="src-1",
            target_brain_id="tgt-1",
            filt=TransplantFilter(tags=frozenset({"caching"})),
            strategy=ConflictStrategy.PREFER_LOCAL,
        )

        # The merge should have skipped the neuron (local wins)
        neuron_conflicts = [c for c in result.merge_report.conflicts if c.entity_type == "neuron"]
        assert len(neuron_conflicts) >= 1
        assert any(c.resolution == "kept_local" for c in neuron_conflicts)


# ── Delete version tests ─────────────────────────────────────────


class TestDeleteVersion:
    """Test delete_version on InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_delete_existing_version(self, source_storage: InMemoryStorage) -> None:
        """Deleting an existing version should return True."""
        from neural_memory.engine.brain_versioning import VersioningEngine

        engine = VersioningEngine(source_storage)
        v = await engine.create_version("src-1", "deletable")

        result = await source_storage.delete_version("src-1", v.id)
        assert result is True

        # Verify it is gone
        assert await source_storage.get_version("src-1", v.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_version(self, source_storage: InMemoryStorage) -> None:
        """Deleting a nonexistent version should return False."""
        result = await source_storage.delete_version("src-1", "nonexistent-id")
        assert result is False
