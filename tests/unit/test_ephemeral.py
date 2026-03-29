"""Tests for ephemeral memories feature (Issue #91).

Ephemeral memories are session-scoped, auto-expire, never sync, and are
excluded from consolidation.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow

# ── Schema ──────────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_schema_version_is_33(self) -> None:
        assert SCHEMA_VERSION == 37


# ── Neuron dataclass ────────────────────────────────────────────────


class TestNeuronEphemeral:
    def test_neuron_default_not_ephemeral(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        assert n.ephemeral is False

    def test_neuron_create_ephemeral(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="temp", ephemeral=True)
        assert n.ephemeral is True

    def test_neuron_with_metadata_preserves_ephemeral(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="temp", ephemeral=True)
        n2 = n.with_metadata(foo="bar")
        assert n2.ephemeral is True
        assert n2.metadata["foo"] == "bar"

    def test_neuron_frozen_ephemeral(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="test", ephemeral=True)
        with pytest.raises(AttributeError):
            n.ephemeral = False  # type: ignore[misc]


# ── Storage CRUD ────────────────────────────────────────────────────


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create a fresh SQLiteStorage with a default brain."""
    s = SQLiteStorage(db_path=str(tmp_path / "test_ephemeral.db"))
    await s.initialize()
    brain = Brain.create(name="ephemeral-test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


@pytest.mark.asyncio
class TestEphemeralStorage:
    async def test_add_ephemeral_neuron(self, storage: SQLiteStorage) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="temp note", ephemeral=True)
        await storage.add_neuron(n)

        fetched = await storage.get_neuron(n.id)
        assert fetched is not None
        assert fetched.ephemeral is True

    async def test_add_permanent_neuron(self, storage: SQLiteStorage) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="permanent note")
        await storage.add_neuron(n)

        fetched = await storage.get_neuron(n.id)
        assert fetched is not None
        assert fetched.ephemeral is False

    async def test_find_neurons_ephemeral_filter(self, storage: SQLiteStorage) -> None:
        n_perm = Neuron.create(type=NeuronType.CONCEPT, content="permanent fact")
        n_eph = Neuron.create(type=NeuronType.CONCEPT, content="ephemeral fact", ephemeral=True)
        await storage.add_neuron(n_perm)
        await storage.add_neuron(n_eph)

        # All neurons
        all_neurons = await storage.find_neurons()
        assert len(all_neurons) == 2

        # Only permanent
        perm_only = await storage.find_neurons(ephemeral=False)
        assert len(perm_only) == 1
        assert perm_only[0].ephemeral is False

        # Only ephemeral
        eph_only = await storage.find_neurons(ephemeral=True)
        assert len(eph_only) == 1
        assert eph_only[0].ephemeral is True

    async def test_find_neurons_no_filter_includes_all(self, storage: SQLiteStorage) -> None:
        for i in range(3):
            await storage.add_neuron(
                Neuron.create(
                    type=NeuronType.CONCEPT,
                    content=f"note {i}",
                    ephemeral=i % 2 == 0,
                )
            )
        all_neurons = await storage.find_neurons()
        assert len(all_neurons) == 3

    async def test_cleanup_ephemeral_neurons(self, storage: SQLiteStorage) -> None:
        # Create an ephemeral neuron with old timestamp
        old_time = utcnow() - timedelta(hours=25)
        n_old = Neuron(
            id="old-eph",
            type=NeuronType.CONCEPT,
            content="old ephemeral",
            ephemeral=True,
            created_at=old_time,
        )
        n_fresh = Neuron.create(type=NeuronType.CONCEPT, content="fresh ephemeral", ephemeral=True)
        n_perm = Neuron.create(type=NeuronType.CONCEPT, content="permanent")

        await storage.add_neuron(n_old)
        await storage.add_neuron(n_fresh)
        await storage.add_neuron(n_perm)

        deleted = await storage.cleanup_ephemeral_neurons(max_age_hours=24.0)
        assert deleted == 1

        # Old ephemeral gone, fresh + permanent remain
        remaining = await storage.find_neurons()
        assert len(remaining) == 2
        remaining_ids = {n.id for n in remaining}
        assert n_fresh.id in remaining_ids
        assert n_perm.id in remaining_ids

    async def test_cleanup_does_not_touch_permanent(self, storage: SQLiteStorage) -> None:
        old_time = utcnow() - timedelta(hours=48)
        n_perm = Neuron(
            id="old-perm",
            type=NeuronType.CONCEPT,
            content="old permanent",
            created_at=old_time,
        )
        await storage.add_neuron(n_perm)

        deleted = await storage.cleanup_ephemeral_neurons(max_age_hours=24.0)
        assert deleted == 0

    async def test_batch_get_includes_ephemeral_flag(self, storage: SQLiteStorage) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="a", ephemeral=True)
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="b", ephemeral=False)
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        batch = await storage.get_neurons_batch([n1.id, n2.id])
        assert batch[n1.id].ephemeral is True
        assert batch[n2.id].ephemeral is False


# ── Sync exclusion ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEphemeralSyncExclusion:
    async def test_seed_change_log_excludes_ephemeral(self, storage: SQLiteStorage) -> None:
        n_perm = Neuron.create(type=NeuronType.CONCEPT, content="permanent for sync")
        n_eph = Neuron.create(type=NeuronType.CONCEPT, content="ephemeral no sync", ephemeral=True)
        await storage.add_neuron(n_perm)
        await storage.add_neuron(n_eph)

        counts = await storage.seed_change_log(device_id="test-device")
        # Only permanent neuron should be seeded
        assert counts["neurons"] == 1

        # Verify via change_log
        changes = await storage.get_changes_since(0, limit=100)
        neuron_changes = [c for c in changes if c.entity_type == "neuron"]
        assert len(neuron_changes) == 1
        assert neuron_changes[0].entity_id == n_perm.id


# ── Consolidation exclusion ─────────────────────────────────────────


@pytest.mark.asyncio
class TestEphemeralConsolidationExclusion:
    async def test_find_neurons_ephemeral_false_excludes(self, storage: SQLiteStorage) -> None:
        """Consolidation uses find_neurons(ephemeral=False) — must exclude ephemeral."""
        n_eph = Neuron.create(type=NeuronType.CONCEPT, content="temp debug note", ephemeral=True)
        n_perm = Neuron.create(type=NeuronType.CONCEPT, content="important decision")
        await storage.add_neuron(n_eph)
        await storage.add_neuron(n_perm)

        results = await storage.find_neurons(ephemeral=False)
        assert len(results) == 1
        assert results[0].id == n_perm.id
