"""Integration tests for versioning + transplant full flows.

Uses InMemoryStorage to test the full lifecycle of version creation,
diffing, rollback, and transplant operations without SQLite.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.brain_transplant import TransplantFilter, transplant
from neural_memory.engine.brain_versioning import VersioningEngine
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    """Storage with a brain and some data (neurons, synapses, fibers)."""
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


# ── Full version lifecycle ───────────────────────────────────────


class TestVersioningFlow:
    """End-to-end versioning lifecycle tests."""

    @pytest.mark.asyncio
    async def test_full_version_lifecycle(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Create -> modify -> version -> diff -> rollback -> verify."""
        # 1. Create version v1 with initial state (3 neurons, 2 synapses, 1 fiber)
        v1 = await engine.create_version("brain-1", "v1-baseline", "Initial snapshot")
        assert v1.neuron_count == 3
        assert v1.synapse_count == 2
        assert v1.fiber_count == 1

        # 2. Add neuron n-4
        n4 = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="n-4")
        await storage.add_neuron(n4)

        # 3. Remove synapse s-1
        await storage.delete_synapse("s-1")

        # Verify modified state
        stats = await storage.get_stats("brain-1")
        assert stats["neuron_count"] == 4
        assert stats["synapse_count"] == 1

        # 4. Create version v2 with modified state
        v2 = await engine.create_version("brain-1", "v2-modified", "Added n-4, removed s-1")
        assert v2.neuron_count == 4
        assert v2.synapse_count == 1
        assert v2.version_number == 2

        # 5. Diff v1 vs v2
        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert "n-4" in diff.neurons_added
        assert "s-1" in diff.synapses_removed
        assert len(diff.neurons_added) == 1
        assert len(diff.synapses_removed) == 1
        assert diff.neurons_removed == ()

        # 6. Rollback to v1
        rollback_v = await engine.rollback("brain-1", v1.id)
        assert rollback_v.version_name.startswith("rollback-to-v1-baseline")
        assert rollback_v.neuron_count == 3
        assert rollback_v.synapse_count == 2

        # 7. Verify n-4 is gone by fetching actual neuron
        n4_after = await storage.get_neuron("n-4")
        assert n4_after is None, "Neuron n-4 should be gone after rollback"

        # Verify s-1 is back by fetching actual synapse
        s1_after = await storage.get_synapse("s-1")
        assert s1_after is not None, "Synapse s-1 should be restored after rollback"
        assert s1_after.weight == 0.7

        # Verify original neurons are intact
        n1_after = await storage.get_neuron("n-1")
        assert n1_after is not None
        assert n1_after.content == "Redis"

        n2_after = await storage.get_neuron("n-2")
        assert n2_after is not None
        assert n2_after.content == "caching"

        n3_after = await storage.get_neuron("n-3")
        assert n3_after is not None
        assert n3_after.content == "deploy"

        # 8. Verify version list has v1, v2, and rollback entry
        versions = await engine.list_versions("brain-1")
        assert len(versions) == 3
        names = {v.version_name for v in versions}
        assert "v1-baseline" in names
        assert "v2-modified" in names
        assert any(name.startswith("rollback-to-v1-baseline") for name in names)

    @pytest.mark.asyncio
    async def test_multiple_rollbacks_get_unique_names(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Multiple rollbacks to the same version produce unique names."""
        v1 = await engine.create_version("brain-1", "target-version")

        rb1 = await engine.rollback("brain-1", v1.id)
        rb2 = await engine.rollback("brain-1", v1.id)
        rb3 = await engine.rollback("brain-1", v1.id)

        names = {rb1.version_name, rb2.version_name, rb3.version_name}
        assert len(names) == 3, "Each rollback should have a unique name"

    @pytest.mark.asyncio
    async def test_diff_after_rollback_shows_no_changes(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Diff between original version and rollback should show no changes."""
        v1 = await engine.create_version("brain-1", "baseline")

        # Modify data
        n4 = Neuron.create(type=NeuronType.ENTITY, content="MongoDB", neuron_id="n-4")
        await storage.add_neuron(n4)

        # Rollback to v1 (creates a new version entry)
        rollback_v = await engine.rollback("brain-1", v1.id)

        # Diff original v1 vs rollback should show no changes (same snapshot)
        diff = await engine.diff("brain-1", v1.id, rollback_v.id)
        assert diff.neurons_added == ()
        assert diff.neurons_removed == ()
        assert diff.synapses_added == ()
        assert diff.synapses_removed == ()
        assert diff.summary == "No changes"

    @pytest.mark.asyncio
    async def test_version_captures_fiber_changes(
        self,
        engine: VersioningEngine,
        storage: InMemoryStorage,
    ) -> None:
        """Version diff should detect fiber additions and removals."""
        v1 = await engine.create_version("brain-1", "v1-one-fiber")
        assert v1.fiber_count == 1

        # Add another fiber
        f2 = Fiber.create(
            neuron_ids={"n-1", "n-3"},
            synapse_ids={"s-2"},
            anchor_neuron_id="n-1",
            fiber_id="f-2",
            tags={"new-fiber"},
        )
        await storage.add_fiber(f2)

        v2 = await engine.create_version("brain-1", "v2-two-fibers")
        assert v2.fiber_count == 2

        diff = await engine.diff("brain-1", v1.id, v2.id)
        assert "f-2" in diff.fibers_added
        assert diff.fibers_removed == ()


# ── Full transplant lifecycle ────────────────────────────────────


class TestTransplantFlow:
    """End-to-end transplant lifecycle tests."""

    @pytest.mark.asyncio
    async def test_full_transplant_lifecycle(self) -> None:
        """Export -> filter -> transplant -> verify target."""
        # 1. Create source brain with tagged fibers
        source_store = InMemoryStorage()
        source_brain = Brain.create(name="source-brain", config=BrainConfig(), brain_id="source-1")
        await source_store.save_brain(source_brain)
        source_store.set_brain(source_brain.id)

        sn1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="sn-1")
        sn2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="sn-2")
        sn3 = Neuron.create(type=NeuronType.ENTITY, content="Kafka", neuron_id="sn-3")
        sn4 = Neuron.create(type=NeuronType.CONCEPT, content="streaming", neuron_id="sn-4")
        await source_store.add_neuron(sn1)
        await source_store.add_neuron(sn2)
        await source_store.add_neuron(sn3)
        await source_store.add_neuron(sn4)

        ss1 = Synapse.create(
            source_id="sn-1",
            target_id="sn-2",
            type=SynapseType.RELATED_TO,
            weight=0.8,
            synapse_id="ss-1",
        )
        ss2 = Synapse.create(
            source_id="sn-3",
            target_id="sn-4",
            type=SynapseType.RELATED_TO,
            weight=0.9,
            synapse_id="ss-2",
        )
        await source_store.add_synapse(ss1)
        await source_store.add_synapse(ss2)

        # Fiber tagged "cache" — should be transplanted
        sf1 = Fiber.create(
            neuron_ids={"sn-1", "sn-2"},
            synapse_ids={"ss-1"},
            anchor_neuron_id="sn-1",
            fiber_id="sf-1",
            tags={"cache"},
        )
        await source_store.add_fiber(sf1)

        # Fiber tagged "streaming" — should NOT be transplanted
        sf2 = Fiber.create(
            neuron_ids={"sn-3", "sn-4"},
            synapse_ids={"ss-2"},
            anchor_neuron_id="sn-3",
            fiber_id="sf-2",
            tags={"streaming"},
        )
        await source_store.add_fiber(sf2)

        # 2. Create target brain with some existing data
        target_store = InMemoryStorage()
        target_brain = Brain.create(name="target-brain", config=BrainConfig(), brain_id="target-1")
        await target_store.save_brain(target_brain)
        target_store.set_brain(target_brain.id)

        tn1 = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL", neuron_id="tn-1")
        await target_store.add_neuron(tn1)

        # 3. Transplant source -> target with tag filter "cache"
        filt = TransplantFilter(tags=frozenset({"cache"}))
        result = await transplant(
            source_storage=source_store,
            target_storage=target_store,
            source_brain_id="source-1",
            target_brain_id="target-1",
            filt=filt,
        )

        assert result.fibers_transplanted == 1
        assert result.neurons_transplanted == 2  # sn-1 and sn-2
        assert result.synapses_transplanted == 1  # ss-1

        # 4. Verify target has both original and transplanted data
        target_store.set_brain("target-1")
        target_stats = await target_store.get_stats("target-1")
        # Original neuron (tn-1) + transplanted (sn-1, sn-2) = 3
        assert target_stats["neuron_count"] == 3
        assert target_stats["fiber_count"] == 1
        assert target_stats["synapse_count"] == 1

        # Verify original data survived
        tn1_after = await target_store.get_neuron("tn-1")
        assert tn1_after is not None
        assert tn1_after.content == "PostgreSQL"

        # Verify transplanted data arrived
        sn1_in_target = await target_store.get_neuron("sn-1")
        assert sn1_in_target is not None
        assert sn1_in_target.content == "Redis"

        sn2_in_target = await target_store.get_neuron("sn-2")
        assert sn2_in_target is not None
        assert sn2_in_target.content == "caching"

        # 5. Verify source is unchanged
        source_store.set_brain("source-1")
        source_stats = await source_store.get_stats("source-1")
        assert source_stats["neuron_count"] == 4
        assert source_stats["fiber_count"] == 2
        assert source_stats["synapse_count"] == 2

    @pytest.mark.asyncio
    async def test_transplant_without_tag_filter(self) -> None:
        """Transplant without tag filter should move all fibers."""
        # Create source brain
        source_store = InMemoryStorage()
        source_brain = Brain.create(name="source", config=BrainConfig(), brain_id="source-1")
        await source_store.save_brain(source_brain)
        source_store.set_brain(source_brain.id)

        sn1 = Neuron.create(type=NeuronType.ENTITY, content="Node1", neuron_id="sn-1")
        sn2 = Neuron.create(type=NeuronType.ENTITY, content="Node2", neuron_id="sn-2")
        await source_store.add_neuron(sn1)
        await source_store.add_neuron(sn2)

        ss1 = Synapse.create(
            source_id="sn-1",
            target_id="sn-2",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="ss-1",
        )
        await source_store.add_synapse(ss1)

        sf1 = Fiber.create(
            neuron_ids={"sn-1", "sn-2"},
            synapse_ids={"ss-1"},
            anchor_neuron_id="sn-1",
            fiber_id="sf-1",
            tags={"alpha"},
        )
        sf2 = Fiber.create(
            neuron_ids={"sn-1"},
            synapse_ids=set(),
            anchor_neuron_id="sn-1",
            fiber_id="sf-2",
            tags={"beta"},
        )
        await source_store.add_fiber(sf1)
        await source_store.add_fiber(sf2)

        # Create empty target brain
        target_store = InMemoryStorage()
        target_brain = Brain.create(name="target", config=BrainConfig(), brain_id="target-1")
        await target_store.save_brain(target_brain)
        target_store.set_brain(target_brain.id)

        # Transplant everything (no tag filter)
        filt = TransplantFilter()
        result = await transplant(
            source_storage=source_store,
            target_storage=target_store,
            source_brain_id="source-1",
            target_brain_id="target-1",
            filt=filt,
        )

        assert result.fibers_transplanted == 2
        assert result.neurons_transplanted == 2

        target_store.set_brain("target-1")
        target_stats = await target_store.get_stats("target-1")
        assert target_stats["neuron_count"] == 2
        assert target_stats["fiber_count"] == 2

    @pytest.mark.asyncio
    async def test_transplant_with_empty_source(self) -> None:
        """Transplant from empty brain should transplant nothing."""
        # Create empty source brain
        source_store = InMemoryStorage()
        source_brain = Brain.create(name="empty-source", config=BrainConfig(), brain_id="source-1")
        await source_store.save_brain(source_brain)
        source_store.set_brain(source_brain.id)

        # Create target brain with data
        target_store = InMemoryStorage()
        target_brain = Brain.create(name="target", config=BrainConfig(), brain_id="target-1")
        await target_store.save_brain(target_brain)
        target_store.set_brain(target_brain.id)

        tn1 = Neuron.create(type=NeuronType.ENTITY, content="existing", neuron_id="tn-1")
        await target_store.add_neuron(tn1)

        filt = TransplantFilter()
        result = await transplant(
            source_storage=source_store,
            target_storage=target_store,
            source_brain_id="source-1",
            target_brain_id="target-1",
            filt=filt,
        )

        assert result.fibers_transplanted == 0
        assert result.neurons_transplanted == 0
        assert result.synapses_transplanted == 0

        # Target data should be preserved
        target_store.set_brain("target-1")
        target_stats = await target_store.get_stats("target-1")
        assert target_stats["neuron_count"] == 1
