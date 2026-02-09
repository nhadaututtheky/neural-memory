"""Tests for SQLite versioning mixin — version storage operations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.brain_versioning import BrainVersion
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.storage.sqlite_versioning import _row_to_version

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create a temporary SQLite storage with a brain and some data."""
    db_path = tmp_path / "test_versioning.db"
    store = SQLiteStorage(db_path)
    await store.initialize()

    brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="brain-1")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Add neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="Redis", neuron_id="n-1")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="caching", neuron_id="n-2")
    await store.add_neuron(n1)
    await store.add_neuron(n2)

    # Add synapse
    s1 = Synapse.create(
        source_id="n-1",
        target_id="n-2",
        type=SynapseType.RELATED_TO,
        weight=0.7,
        synapse_id="s-1",
    )
    await store.add_synapse(s1)

    # Add fiber
    f1 = Fiber.create(
        neuron_ids={"n-1", "n-2"},
        synapse_ids={"s-1"},
        anchor_neuron_id="n-1",
        fiber_id="f-1",
        tags={"redis", "caching"},
    )
    await store.add_fiber(f1)

    yield store

    await store.close()


def _make_version(
    brain_id: str = "brain-1",
    version_id: str = "v-1",
    version_name: str = "v1",
    version_number: int = 1,
    description: str = "",
    neuron_count: int = 2,
    synapse_count: int = 1,
    fiber_count: int = 1,
    snapshot_hash: str = "abc123",
    metadata: dict | None = None,
) -> BrainVersion:
    """Helper to build a BrainVersion for tests."""
    return BrainVersion(
        id=version_id,
        brain_id=brain_id,
        version_name=version_name,
        version_number=version_number,
        description=description,
        neuron_count=neuron_count,
        synapse_count=synapse_count,
        fiber_count=fiber_count,
        snapshot_hash=snapshot_hash,
        created_at=datetime.utcnow(),
        metadata=metadata or {},
    )


# ── save_version + get_version roundtrip ─────────────────────────


class TestSaveAndGetVersion:
    """Test save_version + get_version roundtrip."""

    @pytest.mark.asyncio
    async def test_roundtrip(self, storage: SQLiteStorage) -> None:
        """Saving and retrieving a version should preserve all fields."""
        version = _make_version(
            version_name="baseline",
            description="Initial snapshot",
            metadata={"author": "test"},
        )
        snapshot_json = '{"neurons": [], "synapses": [], "fibers": []}'

        await storage.save_version("brain-1", version, snapshot_json)
        result = await storage.get_version("brain-1", version.id)

        assert result is not None
        retrieved, retrieved_json = result

        assert retrieved.id == version.id
        assert retrieved.version_name == "baseline"
        assert retrieved.version_number == 1
        assert retrieved.description == "Initial snapshot"
        assert retrieved.neuron_count == 2
        assert retrieved.synapse_count == 1
        assert retrieved.fiber_count == 1
        assert retrieved.snapshot_hash == "abc123"
        assert retrieved.metadata == {"author": "test"}
        assert retrieved_json == snapshot_json

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, storage: SQLiteStorage) -> None:
        """Getting a nonexistent version should return None."""
        result = await storage.get_version("brain-1", "nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_wrong_brain_returns_none(self, storage: SQLiteStorage) -> None:
        """Getting a version for a different brain_id should return None."""
        version = _make_version()
        await storage.save_version("brain-1", version, '{"data": true}')

        result = await storage.get_version("brain-999", version.id)
        assert result is None


# ── list_versions ordering and limit ─────────────────────────────


class TestListVersions:
    """Test list_versions ordering and limit."""

    @pytest.mark.asyncio
    async def test_newest_first_ordering(self, storage: SQLiteStorage) -> None:
        """Versions should be listed newest (highest version_number) first."""
        for i in range(1, 4):
            v = _make_version(
                version_id=f"v-{i}",
                version_name=f"v{i}",
                version_number=i,
            )
            await storage.save_version("brain-1", v, f'{{"version": {i}}}')

        versions = await storage.list_versions("brain-1")

        assert len(versions) == 3
        assert versions[0].version_name == "v3"
        assert versions[1].version_name == "v2"
        assert versions[2].version_name == "v1"

    @pytest.mark.asyncio
    async def test_limit(self, storage: SQLiteStorage) -> None:
        """Limit parameter should restrict the number of returned versions."""
        for i in range(1, 6):
            v = _make_version(
                version_id=f"v-{i}",
                version_name=f"v{i}",
                version_number=i,
            )
            await storage.save_version("brain-1", v, "{}")

        versions = await storage.list_versions("brain-1", limit=2)
        assert len(versions) == 2
        # Should be the 2 newest
        assert versions[0].version_number == 5
        assert versions[1].version_number == 4

    @pytest.mark.asyncio
    async def test_empty_list(self, storage: SQLiteStorage) -> None:
        """Listing versions for brain with no versions returns empty list."""
        versions = await storage.list_versions("brain-1")
        assert versions == []


# ── get_next_version_number ──────────────────────────────────────


class TestGetNextVersionNumber:
    """Test get_next_version_number increments."""

    @pytest.mark.asyncio
    async def test_starts_at_one(self, storage: SQLiteStorage) -> None:
        """First version number should be 1."""
        num = await storage.get_next_version_number("brain-1")
        assert num == 1

    @pytest.mark.asyncio
    async def test_increments(self, storage: SQLiteStorage) -> None:
        """Each save should advance the next version number."""
        v1 = _make_version(version_id="v-1", version_name="v1", version_number=1)
        await storage.save_version("brain-1", v1, "{}")
        assert await storage.get_next_version_number("brain-1") == 2

        v2 = _make_version(version_id="v-2", version_name="v2", version_number=2)
        await storage.save_version("brain-1", v2, "{}")
        assert await storage.get_next_version_number("brain-1") == 3

    @pytest.mark.asyncio
    async def test_gap_handling(self, storage: SQLiteStorage) -> None:
        """Next version number should be max+1 even with gaps."""
        v1 = _make_version(version_id="v-1", version_name="v1", version_number=1)
        v5 = _make_version(version_id="v-5", version_name="v5", version_number=5)
        await storage.save_version("brain-1", v1, "{}")
        await storage.save_version("brain-1", v5, "{}")

        assert await storage.get_next_version_number("brain-1") == 6


# ── delete_version ───────────────────────────────────────────────


class TestDeleteVersion:
    """Test delete_version returns True/False correctly."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, storage: SQLiteStorage) -> None:
        """Deleting an existing version should return True."""
        v = _make_version()
        await storage.save_version("brain-1", v, "{}")

        result = await storage.delete_version("brain-1", v.id)
        assert result is True

        # Verify it's gone
        assert await storage.get_version("brain-1", v.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage: SQLiteStorage) -> None:
        """Deleting a nonexistent version should return False."""
        result = await storage.delete_version("brain-1", "nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_does_not_affect_others(self, storage: SQLiteStorage) -> None:
        """Deleting one version should not affect other versions."""
        v1 = _make_version(version_id="v-1", version_name="v1", version_number=1)
        v2 = _make_version(version_id="v-2", version_name="v2", version_number=2)
        await storage.save_version("brain-1", v1, '{"v": 1}')
        await storage.save_version("brain-1", v2, '{"v": 2}')

        await storage.delete_version("brain-1", v1.id)

        # v2 should still exist
        result = await storage.get_version("brain-1", v2.id)
        assert result is not None
        assert result[0].version_name == "v2"


# ── _row_to_version metadata parsing ────────────────────────────


class TestRowToVersion:
    """Test _row_to_version metadata parsing with edge cases."""

    def _make_row(self, metadata_raw: str | None = None) -> dict:
        """Create a mock row dict matching the DB column structure."""
        return {
            "id": "v-test",
            "brain_id": "brain-1",
            "version_name": "test-v",
            "version_number": 1,
            "description": "desc",
            "neuron_count": 5,
            "synapse_count": 3,
            "fiber_count": 2,
            "snapshot_hash": "hash123",
            "snapshot_data": "{}",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata_raw,
        }

    def test_valid_metadata_json(self) -> None:
        """Valid JSON metadata should parse correctly."""
        row = self._make_row('{"author": "alice", "rollback_from": "v-old"}')
        version = _row_to_version(row)
        assert version.metadata == {"author": "alice", "rollback_from": "v-old"}

    def test_empty_metadata_string(self) -> None:
        """Empty string metadata should produce empty dict."""
        row = self._make_row("")
        version = _row_to_version(row)
        assert version.metadata == {}

    def test_none_metadata(self) -> None:
        """None metadata should produce empty dict."""
        row = self._make_row(None)
        version = _row_to_version(row)
        assert version.metadata == {}

    def test_malformed_json_raises(self) -> None:
        """Malformed JSON metadata should raise json.JSONDecodeError."""
        row = self._make_row("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            _row_to_version(row)

    def test_null_description_becomes_empty_string(self) -> None:
        """Null description in row should be converted to empty string."""
        row = self._make_row("{}")
        row["description"] = None
        version = _row_to_version(row)
        assert version.description == ""

    def test_all_fields_preserved(self) -> None:
        """All fields from the row should be reflected in the BrainVersion."""
        row = self._make_row('{"key": "val"}')
        version = _row_to_version(row)
        assert version.id == "v-test"
        assert version.brain_id == "brain-1"
        assert version.version_name == "test-v"
        assert version.version_number == 1
        assert version.neuron_count == 5
        assert version.synapse_count == 3
        assert version.fiber_count == 2
        assert version.snapshot_hash == "hash123"
        assert isinstance(version.created_at, datetime)
