"""Tests for tag filtering in Query API.

Covers tag filtering for find_fibers_batch() method across all three tag columns:
- tags (union of auto_tags and agent_tags via property)
- auto_tags (automatically extracted tags)
- agent_tags (tags provided by calling agent)

Test cases:
1. No tags (backward compatibility)
2. Single tag in tags column
3. Single tag in auto_tags column
4. Single tag in agent_tags column
5. AND semantics (multiple tags)
6. No match scenario
7. Mixed fibers with different tags
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Create a temporary SQLite storage with a test brain."""
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


class TestTagFilterQueryNoTags:
    """Test backward compatibility: find_fibers_batch() without tag filtering."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_no_tags_filter(self, storage: SQLiteStorage) -> None:
        """Tags=None should return all matching fibers (backward compatible)."""
        # Create neurons
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Python")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="React")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Database")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        # Create synapse
        s1 = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
        )
        await storage.add_synapse(s1)

        # Create fibers with different tags
        f1 = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        f2 = Fiber.create(
            neuron_ids={n2.id, n3.id},
            synapse_ids=set(),
            anchor_neuron_id=n2.id,
            agent_tags={"react"},
        )

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        # Find without tag filter - should return both
        result = await storage.find_fibers_batch([n1.id, n2.id, n3.id], tags=None)

        assert len(result) == 2
        assert f1.id in [f.id for f in result]
        assert f2.id in [f.id for f in result]


class TestTagFilterSingleTag:
    """Test single tag filtering across all tag columns."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_single_tag_in_agent_tags(self, storage: SQLiteStorage) -> None:
        """Single tag in agent_tags should be found."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Knowledge base")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Query")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s1 = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
        )
        await storage.add_synapse(s1)

        # Create fiber with agent_tags
        f1 = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        await storage.add_fiber(f1)

        # Find with matching tag
        result = await storage.find_fibers_batch([n1.id, n2.id], tags={"kb"})
        assert len(result) == 1
        assert result[0].id == f1.id

    @pytest.mark.asyncio
    async def test_find_fibers_batch_single_tag_in_auto_tags(self, storage: SQLiteStorage) -> None:
        """Single tag in auto_tags should be found."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Entity extraction")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="NLP")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s1 = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
        )
        await storage.add_synapse(s1)

        # Create fiber with auto_tags
        f1 = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
            auto_tags={"nlp"},
        )
        await storage.add_fiber(f1)

        # Find with matching tag
        result = await storage.find_fibers_batch([n1.id, n2.id], tags={"nlp"})
        assert len(result) == 1
        assert result[0].id == f1.id

    @pytest.mark.asyncio
    async def test_find_fibers_batch_single_tag_mismatch(self, storage: SQLiteStorage) -> None:
        """Single tag that doesn't match should return empty."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Topic")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        await storage.add_fiber(f1)

        # Find with non-matching tag
        result = await storage.find_fibers_batch([n1.id], tags={"nonexistent"})
        assert len(result) == 0


class TestTagFilterANDSemantics:
    """Test AND semantics: fiber must have ALL requested tags."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_multiple_tags_and_semantics(
        self, storage: SQLiteStorage
    ) -> None:
        """Fiber must have ALL tags (AND semantics)."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="React KB")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Frontend")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Backend")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        s1 = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
        )
        s2 = Synapse.create(
            source_id=n2.id,
            target_id=n3.id,
            type=SynapseType.RELATED_TO,
        )
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        # Fiber with both tags
        f1 = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
            agent_tags={"kb", "react"},
        )

        # Fiber with only one tag
        f2 = Fiber.create(
            neuron_ids={n2.id, n3.id},
            synapse_ids={s2.id},
            anchor_neuron_id=n2.id,
            agent_tags={"kb"},  # Missing 'react'
        )

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        # Find with both tags - should only return f1
        result = await storage.find_fibers_batch([n1.id, n2.id, n3.id], tags={"kb", "react"})
        assert len(result) == 1
        assert result[0].id == f1.id

    @pytest.mark.asyncio
    async def test_find_fibers_batch_multiple_tags_across_columns(
        self, storage: SQLiteStorage
    ) -> None:
        """Fiber with tags spread across auto_tags and agent_tags should match."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Hybrid tags")
        await storage.add_neuron(n1)

        # Fiber with one tag in auto_tags, one in agent_tags
        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            auto_tags={"kb"},
            agent_tags={"react"},
        )
        await storage.add_fiber(f1)

        # Find with both tags - should match even though they're in different columns
        result = await storage.find_fibers_batch([n1.id], tags={"kb", "react"})
        assert len(result) == 1
        assert result[0].id == f1.id


class TestTagFilterMixedFibers:
    """Test filtering with multiple fibers having different tag combinations."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_mixed_fibers(self, storage: SQLiteStorage) -> None:
        """Filter returns only matching fibers from mixed set."""
        # Create neurons
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Node 1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Node 2")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Node 3")
        n4 = Neuron.create(type=NeuronType.CONCEPT, content="Node 4")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)
        await storage.add_neuron(n4)

        # Create synapses
        s1 = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
        )
        s2 = Synapse.create(
            source_id=n2.id,
            target_id=n3.id,
            type=SynapseType.RELATED_TO,
        )
        s3 = Synapse.create(
            source_id=n3.id,
            target_id=n4.id,
            type=SynapseType.RELATED_TO,
        )
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)
        await storage.add_synapse(s3)

        # Fiber 1: kb + react
        f1 = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s1.id},
            anchor_neuron_id=n1.id,
            agent_tags={"kb", "react"},
        )

        # Fiber 2: kb only
        f2 = Fiber.create(
            neuron_ids={n2.id, n3.id},
            synapse_ids={s2.id},
            anchor_neuron_id=n2.id,
            agent_tags={"kb"},
        )

        # Fiber 3: react + python
        f3 = Fiber.create(
            neuron_ids={n3.id, n4.id},
            synapse_ids={s3.id},
            anchor_neuron_id=n3.id,
            agent_tags={"react", "python"},
        )

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)
        await storage.add_fiber(f3)

        # Find fibers with "kb" tag - should return f1 and f2
        result = await storage.find_fibers_batch([n1.id, n2.id, n3.id, n4.id], tags={"kb"})
        assert len(result) == 2
        result_ids = {f.id for f in result}
        assert f1.id in result_ids
        assert f2.id in result_ids
        assert f3.id not in result_ids

        # Find fibers with "react" tag - should return f1 and f3
        result = await storage.find_fibers_batch([n1.id, n2.id, n3.id, n4.id], tags={"react"})
        assert len(result) == 2
        result_ids = {f.id for f in result}
        assert f1.id in result_ids
        assert f3.id in result_ids
        assert f2.id not in result_ids

        # Find fibers with "kb" AND "react" - should return f1 only
        result = await storage.find_fibers_batch([n1.id, n2.id, n3.id, n4.id], tags={"kb", "react"})
        assert len(result) == 1
        assert result[0].id == f1.id


class TestTagFilterEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_empty_neuron_list(self, storage: SQLiteStorage) -> None:
        """Empty neuron list should return empty."""
        result = await storage.find_fibers_batch([], tags={"kb"})
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_fibers_batch_no_neurons_with_tag(self, storage: SQLiteStorage) -> None:
        """When no neurons in neuron_ids, should return empty."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Unrelated")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        await storage.add_fiber(f1)

        # Search for different neurons
        result = await storage.find_fibers_batch(["nonexistent"], tags={"kb"})
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_fibers_batch_tag_case_sensitivity(self, storage: SQLiteStorage) -> None:
        """Tag matching should be case-sensitive."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Case test")
        await storage.add_neuron(n1)

        # Fiber with lowercase "kb"
        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        await storage.add_fiber(f1)

        # Find with uppercase "KB" - should not match
        result = await storage.find_fibers_batch([n1.id], tags={"KB"})
        assert len(result) == 0

        # Find with correct lowercase - should match
        result = await storage.find_fibers_batch([n1.id], tags={"kb"})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_fibers_batch_empty_tags_set(self, storage: SQLiteStorage) -> None:
        """Empty tags set should return no filters (all fibers)."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Any fiber")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"kb"},
        )
        await storage.add_fiber(f1)

        # Empty set should be treated as None (no filter)
        result = await storage.find_fibers_batch([n1.id], tags=set())
        # Empty set treated as no filter — all matching fibers returned
        assert len(result) == 1


class TestTagFilterORMode:
    """Test OR semantics: fiber must have ANY of the requested tags."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_or_mode_returns_any_match(
        self, storage: SQLiteStorage
    ) -> None:
        """OR mode should return fibers matching ANY requested tag."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Node A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Node B")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="Node C")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        # f1: only "python" tag
        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"python"},
        )
        # f2: only "react" tag
        f2 = Fiber.create(
            neuron_ids={n2.id},
            synapse_ids=set(),
            anchor_neuron_id=n2.id,
            agent_tags={"react"},
        )
        # f3: unrelated "go" tag
        f3 = Fiber.create(
            neuron_ids={n3.id},
            synapse_ids=set(),
            anchor_neuron_id=n3.id,
            agent_tags={"go"},
        )
        await storage.add_fiber(f1)
        await storage.add_fiber(f2)
        await storage.add_fiber(f3)

        # OR mode: python OR react → f1 + f2
        result = await storage.find_fibers_batch(
            [n1.id, n2.id, n3.id], tags={"python", "react"}, tag_mode="or"
        )
        result_ids = {f.id for f in result}
        assert f1.id in result_ids
        assert f2.id in result_ids
        assert f3.id not in result_ids

        # AND mode with same tags → neither has both → empty
        result_and = await storage.find_fibers_batch(
            [n1.id, n2.id, n3.id], tags={"python", "react"}, tag_mode="and"
        )
        assert len(result_and) == 0

    @pytest.mark.asyncio
    async def test_find_fibers_or_mode(self, storage: SQLiteStorage) -> None:
        """find_fibers() OR mode should also work."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="X")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"alpha"},
        )
        f2 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"beta"},
        )
        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        # OR: alpha OR gamma → only f1
        result = await storage.find_fibers(tags={"alpha", "gamma"}, tag_mode="or")
        assert len(result) == 1
        assert result[0].id == f1.id

        # AND: alpha AND gamma → none
        result = await storage.find_fibers(tags={"alpha", "gamma"}, tag_mode="and")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_fibers_batch_or_default_is_and(self, storage: SQLiteStorage) -> None:
        """Default tag_mode should be AND (backward compatible)."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Y")
        await storage.add_neuron(n1)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            agent_tags={"one"},
        )
        await storage.add_fiber(f1)

        # Default (no tag_mode) with tag "one" AND "two" → f1 has only "one" → empty
        result = await storage.find_fibers_batch([n1.id], tags={"one", "two"})
        assert len(result) == 0


class TestTagFilterIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_find_fibers_batch_with_limit_and_tags(self, storage: SQLiteStorage) -> None:
        """Tag filtering should respect limit_per_neuron."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Hub")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Node")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Create multiple fibers connected to n1
        fibers = []
        for _i in range(5):
            f = Fiber.create(
                neuron_ids={n1.id, n2.id},
                synapse_ids=set(),
                anchor_neuron_id=n1.id,
                agent_tags={"kb"},
            )
            await storage.add_fiber(f)
            fibers.append(f)

        # Query with limit
        result = await storage.find_fibers_batch([n1.id], limit_per_neuron=2, tags={"kb"})
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_find_fibers_batch_preserves_properties(self, storage: SQLiteStorage) -> None:
        """Retrieved fibers should maintain all properties including tags."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Full fiber")
        await storage.add_neuron(n1)

        original = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
            auto_tags={"extracted"},
            agent_tags={"provided"},
            metadata={"custom": "data"},
            summary="Test summary",
        )
        await storage.add_fiber(original)

        # Retrieve it with tag filter
        result = await storage.find_fibers_batch([n1.id], tags={"extracted"})
        assert len(result) == 1
        retrieved = result[0]

        # Check properties
        assert retrieved.id == original.id
        assert retrieved.auto_tags == original.auto_tags
        assert retrieved.agent_tags == original.agent_tags
        assert retrieved.metadata == original.metadata
        assert retrieved.summary == original.summary
        assert retrieved.tags == original.tags  # Union property
