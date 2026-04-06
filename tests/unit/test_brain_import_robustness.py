"""Regression tests for brain import robustness.

Verifies that import_brain handles:
- Unknown NeuronType values (e.g. MemoryType "fact", "decision")
- Unknown SynapseType values
- Unknown Direction values
- Unknown BrainConfig keys
- Missing optional fields
- Malformed individual records (skip without crashing)
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from neural_memory.core.brain import BrainSnapshot
from neural_memory.core.neuron import NeuronType
from neural_memory.storage.sql import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage(tmp_path: Path) -> SQLStorage:
    """Create a fresh SQLStorage for testing."""
    s = SQLStorage(SQLiteDialect(db_path=str(tmp_path / "test.db")))
    await s.initialize()
    yield s  # type: ignore[misc]
    await s.close()


def _snapshot(
    neurons: list[dict] | None = None,
    synapses: list[dict] | None = None,
    fibers: list[dict] | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> BrainSnapshot:
    return BrainSnapshot(
        brain_id=str(uuid4()),
        brain_name="test-import",
        exported_at=utcnow(),
        version="1",
        neurons=neurons or [],
        synapses=synapses or [],
        fibers=fibers or [],
        config=config or {},
        metadata=metadata or {},
    )


class TestBrainImportRobustness:
    """Regression tests for import_brain crash fixes."""

    @pytest.mark.asyncio
    async def test_unknown_neuron_type_mapped_to_concept(self, storage: SQLStorage) -> None:
        """NeuronType 'fact' (a MemoryType, not NeuronType) should not crash."""
        snapshot = _snapshot(
            neurons=[
                {
                    "id": "n1",
                    "type": "fact",
                    "content": "Python is great",
                    "metadata": {},
                    "created_at": utcnow().isoformat(),
                }
            ]
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1
        assert neurons[0].type == NeuronType.CONCEPT
        assert neurons[0].metadata.get("original_type") == "fact"

    @pytest.mark.asyncio
    async def test_unknown_neuron_type_decision(self, storage: SQLStorage) -> None:
        """MemoryType 'decision' should also map to CONCEPT."""
        snapshot = _snapshot(
            neurons=[
                {
                    "id": "n1",
                    "type": "decision",
                    "content": "Chose X over Y",
                    "created_at": utcnow().isoformat(),
                }
            ]
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1
        assert neurons[0].metadata.get("original_type") == "decision"

    @pytest.mark.asyncio
    async def test_valid_neuron_type_preserved(self, storage: SQLStorage) -> None:
        """Valid NeuronType 'concept' should NOT be remapped."""
        snapshot = _snapshot(
            neurons=[
                {
                    "id": "n1",
                    "type": "concept",
                    "content": "API design",
                    "created_at": utcnow().isoformat(),
                }
            ]
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1
        assert neurons[0].type == NeuronType.CONCEPT
        assert "original_type" not in neurons[0].metadata

    @pytest.mark.asyncio
    async def test_unknown_synapse_type_fallback(self, storage: SQLStorage) -> None:
        """Unknown SynapseType should fall back to RELATED_TO, not crash."""
        snapshot = _snapshot(
            neurons=[
                {"id": "n1", "type": "concept", "content": "A", "created_at": utcnow().isoformat()},
                {"id": "n2", "type": "concept", "content": "B", "created_at": utcnow().isoformat()},
            ],
            synapses=[
                {
                    "id": "s1",
                    "source_id": "n1",
                    "target_id": "n2",
                    "type": "completely_fake_type",
                    "weight": 0.5,
                    "direction": "uni",
                    "created_at": utcnow().isoformat(),
                }
            ],
        )
        # Should not raise — unknown type falls back to RELATED_TO
        brain_id = await storage.import_brain(snapshot)
        assert brain_id is not None

    @pytest.mark.asyncio
    async def test_unknown_direction_fallback(self, storage: SQLStorage) -> None:
        """Unknown Direction should fall back to UNIDIRECTIONAL, not crash."""
        snapshot = _snapshot(
            neurons=[
                {"id": "n1", "type": "concept", "content": "A", "created_at": utcnow().isoformat()},
                {"id": "n2", "type": "concept", "content": "B", "created_at": utcnow().isoformat()},
            ],
            synapses=[
                {
                    "id": "s1",
                    "source_id": "n1",
                    "target_id": "n2",
                    "type": "related_to",
                    "weight": 0.5,
                    "direction": "invalid_direction",
                    "created_at": utcnow().isoformat(),
                }
            ],
        )
        # Should not raise — unknown direction falls back to uni
        brain_id = await storage.import_brain(snapshot)
        assert brain_id is not None

    @pytest.mark.asyncio
    async def test_unknown_brain_config_keys_ignored(self, storage: SQLStorage) -> None:
        """Config with unknown keys should not crash."""
        snapshot = _snapshot(
            config={
                "decay_rate": 0.2,
                "future_key_from_v5": True,
                "another_unknown": 42,
            },
            neurons=[
                {
                    "id": "n1",
                    "type": "concept",
                    "content": "test",
                    "created_at": utcnow().isoformat(),
                },
            ],
        )
        brain_id = await storage.import_brain(snapshot)
        assert brain_id is not None

    @pytest.mark.asyncio
    async def test_neuron_missing_created_at_uses_utcnow(self, storage: SQLStorage) -> None:
        """Neuron without created_at should use current time."""
        snapshot = _snapshot(
            neurons=[
                {"id": "n1", "type": "concept", "content": "test"},
            ]
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1

    @pytest.mark.asyncio
    async def test_malformed_neuron_skipped(self, storage: SQLStorage) -> None:
        """Neuron missing required 'id' field should be skipped, not crash."""
        snapshot = _snapshot(
            neurons=[
                {"type": "concept", "content": "no id"},  # missing "id"
                {
                    "id": "n2",
                    "type": "concept",
                    "content": "valid",
                    "created_at": utcnow().isoformat(),
                },
            ]
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1
        assert neurons[0].id == "n2"

    @pytest.mark.asyncio
    async def test_brain_package_format_import(self, storage: SQLStorage) -> None:
        """Test import of a .brain package format (the format users actually encounter)."""
        package = {
            "nmem_brain_package": "1.0",
            "manifest": {
                "name": "test-brain",
                "display_name": "Test Brain",
                "stats": {"neuron_count": 1},
            },
            "snapshot": {
                "brain_id": "test",
                "brain_name": "test",
                "config": {},
                "exported_at": utcnow().isoformat(),
                "fibers": [],
                "metadata": {},
                "neurons": [
                    {
                        "id": "n1",
                        "type": "fact",
                        "content": "Python is great",
                        "created_at": utcnow().isoformat(),
                        "access_count": 1,
                        "activation": 0.5,
                        "brain_id": "test",
                        "salience": 0.5,
                        "state": "active",
                        "tags": ["python"],
                        "updated_at": utcnow().isoformat(),
                    }
                ],
                "synapses": [],
                "version": "1",
            },
        }
        snapshot_data = package["snapshot"]
        snapshot = BrainSnapshot(
            brain_id=str(uuid4()),
            brain_name="test-brain",
            exported_at=utcnow(),
            version="1",
            neurons=snapshot_data["neurons"],
            synapses=snapshot_data["synapses"],
            fibers=snapshot_data["fibers"],
            config=snapshot_data["config"],
            metadata=snapshot_data.get("metadata", {}),
        )
        brain_id = await storage.import_brain(snapshot)
        storage.set_brain(brain_id)
        neurons = await storage.find_neurons()
        assert len(neurons) == 1
        assert neurons[0].content == "Python is great"
        assert neurons[0].type == NeuronType.CONCEPT
        assert neurons[0].metadata.get("original_type") == "fact"

    @pytest.mark.asyncio
    async def test_empty_config_works(self, storage: SQLStorage) -> None:
        """Empty config dict should create default BrainConfig."""
        snapshot = _snapshot(config={})
        brain_id = await storage.import_brain(snapshot)
        assert brain_id is not None
