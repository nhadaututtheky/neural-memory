"""Tests for InfinityDB typed memory, lifecycle, and maturation mixin."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.pro.storage_adapter import InfinityDBStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage(tmp_path: Path) -> InfinityDBStorage:
    s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="test")
    await s.open()
    yield s
    await s.close()


def _make_typed(fid: str = "", mtype: MemoryType = MemoryType.FACT, **kw) -> TypedMemory:
    return TypedMemory(fiber_id=fid, memory_type=mtype, **kw)


class TestTypedMemoryCRUD:
    async def test_add_get_roundtrip(self, storage: InfinityDBStorage) -> None:
        tm = _make_typed(mtype=MemoryType.DECISION)
        fid = await storage.add_typed_memory(tm)
        assert fid
        got = await storage.get_typed_memory(fid)
        assert got is not None
        assert got.memory_type == MemoryType.DECISION

    async def test_add_duplicate_raises(self, storage: InfinityDBStorage) -> None:
        tm = _make_typed(fid="dup-1")
        await storage.add_typed_memory(tm)
        with pytest.raises(ValueError, match="already exists"):
            await storage.add_typed_memory(tm)

    async def test_find_filters_by_type(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.FACT))
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.DECISION))
        facts = await storage.find_typed_memories(memory_type=MemoryType.FACT)
        assert all(tm.memory_type == MemoryType.FACT for tm in facts)

    async def test_find_filters_expired(self, storage: InfinityDBStorage) -> None:
        past = utcnow() - timedelta(hours=1)
        await storage.add_typed_memory(_make_typed(expires_at=past))
        await storage.add_typed_memory(_make_typed())
        non_expired = await storage.find_typed_memories(include_expired=False)
        assert len(non_expired) == 1
        all_mems = await storage.find_typed_memories(include_expired=True)
        assert len(all_mems) == 2

    async def test_find_filters_by_priority(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(priority=Priority.LOW))
        await storage.add_typed_memory(_make_typed(priority=Priority.HIGH))
        high = await storage.find_typed_memories(min_priority=Priority.HIGH)
        assert len(high) == 1

    async def test_find_filters_by_tier(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(tier="hot"))
        await storage.add_typed_memory(_make_typed(tier="cold"))
        hot = await storage.find_typed_memories(tier="hot")
        assert len(hot) == 1

    async def test_find_filters_by_tags(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(tags=frozenset(["a", "b"])))
        await storage.add_typed_memory(_make_typed(tags=frozenset(["a"])))
        both = await storage.find_typed_memories(tags={"a", "b"})
        assert len(both) == 1

    async def test_count(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.FACT, tier="hot"))
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.DECISION, tier="cold"))
        assert await storage.count_typed_memories() == 2
        assert await storage.count_typed_memories(tier="hot") == 1
        assert await storage.count_typed_memories(memory_type=MemoryType.DECISION) == 1

    async def test_count_grouped(self, storage: InfinityDBStorage) -> None:
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.FACT, tier="hot"))
        await storage.add_typed_memory(_make_typed(mtype=MemoryType.FACT, tier="hot"))
        groups = await storage.count_typed_memories_grouped()
        assert len(groups) >= 1
        assert any(cnt == 2 for _, _, cnt in groups)

    async def test_update(self, storage: InfinityDBStorage) -> None:
        fid = await storage.add_typed_memory(_make_typed(mtype=MemoryType.FACT))
        tm = await storage.get_typed_memory(fid)
        updated = replace(tm, priority=Priority.CRITICAL)
        await storage.update_typed_memory(updated)
        got = await storage.get_typed_memory(fid)
        assert got.priority == Priority.CRITICAL

    async def test_update_nonexistent_raises(self, storage: InfinityDBStorage) -> None:
        with pytest.raises(ValueError, match="not found"):
            await storage.update_typed_memory(_make_typed(fid="nope"))

    async def test_update_source(self, storage: InfinityDBStorage) -> None:
        fid = await storage.add_typed_memory(_make_typed())
        assert await storage.update_typed_memory_source(fid, "agent")
        got = await storage.get_typed_memory(fid)
        assert got.source == "agent"

    async def test_delete(self, storage: InfinityDBStorage) -> None:
        fid = await storage.add_typed_memory(_make_typed())
        assert await storage.delete_typed_memory(fid)
        assert await storage.get_typed_memory(fid) is None
        assert not await storage.delete_typed_memory(fid)

    async def test_expired_memories(self, storage: InfinityDBStorage) -> None:
        past = utcnow() - timedelta(hours=1)
        await storage.add_typed_memory(_make_typed(expires_at=past))
        await storage.add_typed_memory(_make_typed())
        expired = await storage.get_expired_memories()
        assert len(expired) == 1
        assert await storage.get_expired_memory_count() == 1

    async def test_expiring_memories(self, storage: InfinityDBStorage) -> None:
        soon = utcnow() + timedelta(days=3)
        fid = await storage.add_typed_memory(_make_typed(expires_at=soon))
        result = await storage.get_expiring_memories_for_fibers([fid], within_days=7)
        assert len(result) == 1
        assert await storage.get_expiring_memory_count(within_days=7) == 1

    async def test_promote_memory_type(self, storage: InfinityDBStorage) -> None:
        fid = await storage.add_typed_memory(_make_typed(mtype=MemoryType.CONTEXT))
        assert await storage.promote_memory_type(fid, MemoryType.FACT)
        got = await storage.get_typed_memory(fid)
        assert got.memory_type == MemoryType.FACT
        assert got.metadata.get("original_type") == "context"


class TestLifecycle:
    async def test_update_lifecycle(self, storage: InfinityDBStorage) -> None:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="LC test")
        nid = await storage.add_neuron(neuron)
        await storage.update_neuron_lifecycle(nid, "warm")
        assert storage._lifecycle_states[nid] == "warm"
        dist = await storage.get_lifecycle_distribution()
        assert dist.get("warm", 0) == 1

    async def test_update_frozen(self, storage: InfinityDBStorage) -> None:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Freeze")
        nid = await storage.add_neuron(neuron)
        await storage.update_neuron_frozen(nid, True)
        # No error = success (metadata stored in InfinityDB)

    async def test_update_ephemeral(self, storage: InfinityDBStorage) -> None:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Eph")
        nid = await storage.add_neuron(neuron)
        await storage.update_neuron_ephemeral(nid, True)

    async def test_cleanup_ephemeral(self, storage: InfinityDBStorage) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="Old eph", ephemeral=True)
        await storage.add_neuron(n)
        # Default max_age_hours=24, neuron just created so shouldn't be cleaned
        deleted = await storage.cleanup_ephemeral_neurons(max_age_hours=24.0)
        assert deleted == 0

    async def test_lifecycle_distribution(self, storage: InfinityDBStorage) -> None:
        await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="N1"))
        dist = await storage.get_lifecycle_distribution()
        assert dist.get("active", 0) >= 1

    async def test_batch_ephemeral(self, storage: InfinityDBStorage) -> None:
        nid1 = await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="B1"))
        nid2 = await storage.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="B2"))
        await storage.update_neurons_ephemeral_batch([nid1, nid2], True)


class TestMaturation:
    async def test_save_get_roundtrip(self, storage: InfinityDBStorage) -> None:
        rec = MaturationRecord(fiber_id="f1", brain_id="test")
        await storage.save_maturation(rec)
        got = await storage.get_maturation("f1")
        assert got is not None
        assert got.fiber_id == "f1"

    async def test_find_by_stage(self, storage: InfinityDBStorage) -> None:
        await storage.save_maturation(
            MaturationRecord(fiber_id="f1", brain_id="test", stage=MemoryStage.SHORT_TERM)
        )
        await storage.save_maturation(
            MaturationRecord(fiber_id="f2", brain_id="test", stage=MemoryStage.SEMANTIC)
        )
        stm = await storage.find_maturations(stage=MemoryStage.SHORT_TERM)
        assert len(stm) == 1

    async def test_find_by_rehearsal_count(self, storage: InfinityDBStorage) -> None:
        await storage.save_maturation(
            MaturationRecord(fiber_id="f1", brain_id="test", rehearsal_count=3)
        )
        await storage.save_maturation(
            MaturationRecord(fiber_id="f2", brain_id="test", rehearsal_count=0)
        )
        found = await storage.find_maturations(min_rehearsal_count=2)
        assert len(found) == 1

    async def test_fiber_stage_counts(self, storage: InfinityDBStorage) -> None:
        await storage.save_maturation(
            MaturationRecord(fiber_id="f1", brain_id="test", stage=MemoryStage.SHORT_TERM)
        )
        await storage.save_maturation(
            MaturationRecord(fiber_id="f2", brain_id="test", stage=MemoryStage.SHORT_TERM)
        )
        counts = await storage.get_fiber_stage_counts("test")
        assert counts.get("stm", 0) == 2

    async def test_get_maturation_missing(self, storage: InfinityDBStorage) -> None:
        assert await storage.get_maturation("nonexistent") is None


class TestKeywordsAndEntities:
    async def test_keyword_df(self, storage: InfinityDBStorage) -> None:
        await storage.increment_keyword_df(["hello", "world"])
        await storage.increment_keyword_df(["hello"])
        result = await storage.get_keyword_df_batch(["hello", "world", "missing"])
        assert result["hello"] == 2
        assert result["world"] == 1
        assert "missing" not in result

    async def test_entity_refs(self, storage: InfinityDBStorage) -> None:
        await storage.add_entity_ref("Python", "f1")
        await storage.add_entity_ref("Python", "f2")
        assert await storage.count_entity_refs("Python") == 2
        assert await storage.get_entity_ref_fiber_ids("Python") == ["f1", "f2"]

    async def test_mark_promoted(self, storage: InfinityDBStorage) -> None:
        await storage.add_entity_ref("Rust", "f1")
        count = await storage.mark_entity_refs_promoted("Rust")
        assert count == 1
        assert await storage.count_entity_refs("Rust") == 0
