"""InfinityDB Typed Memory + Lifecycle + Maturation mixin.

In-memory implementations of TypedMemory CRUD, neuron lifecycle ops,
and maturation records. All volatile (session-scoped, not persisted).
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage


class InfinityDBTypedMixin:
    """Mixin providing typed memory, lifecycle, and maturation ops.

    Composing class must call ``_init_typed_stores()`` in __init__ and
    provide ``db``, ``find_neurons``, ``get_neuron``, ``delete_neuron``,
    ``get_fiber``, ``find_fibers``, ``get_stats`` attributes/methods.
    """

    # -- Protocol stubs for type checker --
    if TYPE_CHECKING:

        @property
        def db(self) -> Any:
            raise NotImplementedError

        async def get_neuron(self, neuron_id: str) -> Any:
            raise NotImplementedError

        async def delete_neuron(self, neuron_id: str) -> bool:
            raise NotImplementedError

        async def get_fiber(self, fiber_id: str) -> Any:
            raise NotImplementedError

        async def get_stats(self, brain_id: str) -> dict[str, int]:
            raise NotImplementedError

        _current_brain_id: str | None

    def _init_typed_stores(self) -> None:
        self._typed_memories: dict[str, TypedMemory] = {}
        self._maturation_records: dict[str, MaturationRecord] = {}

    # ========== Typed Memory CRUD ==========

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        fid = typed_memory.fiber_id or str(uuid.uuid4())
        if fid in self._typed_memories:
            msg = f"Typed memory already exists: {fid}"
            raise ValueError(msg)
        tm = replace(typed_memory, fiber_id=fid) if typed_memory.fiber_id != fid else typed_memory
        self._typed_memories[fid] = tm
        return fid

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        return self._typed_memories.get(fiber_id)

    async def find_typed_memories(
        self,
        memory_type: MemoryType | None = None,
        min_priority: Priority | None = None,
        include_expired: bool = False,
        project_id: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        tier: str | None = None,
    ) -> list[TypedMemory]:
        now = utcnow()
        results: list[TypedMemory] = []
        for tm in self._typed_memories.values():
            if not include_expired and tm.expires_at is not None and tm.expires_at <= now:
                continue
            if memory_type is not None and tm.memory_type != memory_type:
                continue
            if min_priority is not None and tm.priority < min_priority:
                continue
            if project_id is not None and tm.project_id != project_id:
                continue
            if tier is not None and tm.tier != tier:
                continue
            if tags is not None and not tags.issubset(tm.tags):
                continue
            results.append(tm)
            if len(results) >= limit:
                break
        return results

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        count = 0
        for tm in self._typed_memories.values():
            if tier is not None and tm.tier != tier:
                continue
            if memory_type is not None and tm.memory_type != memory_type:
                continue
            count += 1
        return count

    async def count_typed_memories_grouped(self) -> list[tuple[str, str, int]]:
        groups: dict[tuple[str, str], int] = {}
        for tm in self._typed_memories.values():
            key = (
                tm.memory_type.value if hasattr(tm.memory_type, "value") else str(tm.memory_type),
                tm.tier,
            )
            groups[key] = groups.get(key, 0) + 1
        return [(mt, tier, cnt) for (mt, tier), cnt in groups.items()]

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        if typed_memory.fiber_id not in self._typed_memories:
            msg = f"Typed memory not found: {typed_memory.fiber_id}"
            raise ValueError(msg)
        self._typed_memories[typed_memory.fiber_id] = typed_memory

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        tm = self._typed_memories.get(fiber_id)
        if tm is None:
            return False
        self._typed_memories[fiber_id] = replace(tm, source=source)
        return True

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        return self._typed_memories.pop(fiber_id, None) is not None

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        now = utcnow()
        results: list[TypedMemory] = []
        for tm in self._typed_memories.values():
            if tm.expires_at is not None and tm.expires_at <= now:
                results.append(tm)
                if len(results) >= limit:
                    break
        return results

    async def get_expired_memory_count(self) -> int:
        now = utcnow()
        return sum(
            1
            for tm in self._typed_memories.values()
            if tm.expires_at is not None and tm.expires_at <= now
        )

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        now = utcnow()
        deadline = now + timedelta(days=within_days)
        fid_set = set(fiber_ids)
        return [
            tm
            for tm in self._typed_memories.values()
            if tm.fiber_id in fid_set
            and tm.expires_at is not None
            and now < tm.expires_at <= deadline
        ]

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        now = utcnow()
        deadline = now + timedelta(days=within_days)
        return sum(
            1
            for tm in self._typed_memories.values()
            if tm.expires_at is not None and now < tm.expires_at <= deadline
        )

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        # In-memory: no frequency tracking, return empty
        return []

    async def promote_memory_type(
        self,
        fiber_id: str,
        new_type: MemoryType,
        new_expires_at: str | None = None,
    ) -> bool:
        from datetime import datetime

        tm = self._typed_memories.get(fiber_id)
        if tm is None:
            return False
        new_exp = datetime.fromisoformat(new_expires_at) if new_expires_at else None
        meta = {
            **tm.metadata,
            "original_type": tm.memory_type.value
            if hasattr(tm.memory_type, "value")
            else str(tm.memory_type),
        }
        self._typed_memories[fiber_id] = replace(
            tm, memory_type=new_type, expires_at=new_exp, metadata=meta
        )
        return True

    async def get_stale_fiber_count(self, brain_id: str, stale_days: int = 90) -> int:
        # In-memory: no last_conducted tracking
        return 0

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self._maturation_records.values():
            stage = rec.stage.value if hasattr(rec.stage, "value") else str(rec.stage)
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    async def get_total_fiber_count(self) -> int:
        stats = await self.get_stats(self._current_brain_id or "default")
        return stats.get("fiber_count", 0)

    # ========== Lifecycle Operations ==========

    def _init_lifecycle_stores(self) -> None:
        self._lifecycle_states: dict[str, str] = {}
        self._frozen_flags: dict[str, bool] = {}

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        self._lifecycle_states[neuron_id] = lifecycle_state

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        self._frozen_flags[neuron_id] = frozen

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        await self.db.update_neuron(neuron_id, ephemeral=ephemeral)

    async def update_neurons_ephemeral_batch(self, neuron_ids: list[str], ephemeral: bool) -> None:
        for nid in neuron_ids:
            await self.update_neuron_ephemeral(nid, ephemeral)

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        cutoff = utcnow() - timedelta(hours=max_age_hours)
        neurons = await self.find_neurons(ephemeral=True)  # type: ignore[attr-defined]
        deleted = 0
        for n in neurons:
            if n.created_at is not None and n.created_at < cutoff:
                await self.delete_neuron(n.id)
                deleted += 1
        return deleted

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for state in self._lifecycle_states.values():
            dist[state] = dist.get(state, 0) + 1
        if not dist:
            # Count all neurons as "active" when no lifecycle tracking
            neurons = await self.find_neurons(limit=50000)  # type: ignore[attr-defined]
            dist["active"] = len(neurons)
        return dist

    # ========== Maturation Operations ==========

    async def save_maturation(self, record: MaturationRecord) -> None:
        self._maturation_records[record.fiber_id] = record

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        return self._maturation_records.get(fiber_id)

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        results: list[MaturationRecord] = []
        for rec in self._maturation_records.values():
            if stage is not None and rec.stage != stage:
                continue
            if rec.rehearsal_count < min_rehearsal_count:
                continue
            results.append(rec)
        return results

    async def cleanup_orphaned_maturations(self) -> int:
        orphans = []
        for fid in self._maturation_records:
            fiber = await self.get_fiber(fid)
            if fiber is None:
                orphans.append(fid)
        for fid in orphans:
            del self._maturation_records[fid]
        return len(orphans)

    # ========== Keyword DF ==========

    def _init_keyword_stores(self) -> None:
        self._keyword_df: dict[str, int] = {}

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        return {k: self._keyword_df[k] for k in keywords if k in self._keyword_df}

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        for kw in keywords:
            self._keyword_df[kw] = self._keyword_df.get(kw, 0) + 1

    # ========== Entity Refs ==========

    def _init_entity_stores(self) -> None:
        self._entity_refs: list[dict[str, Any]] = []

    async def add_entity_ref(
        self,
        entity_text: str,
        fiber_id: str,
        created_at: Any = None,
    ) -> None:
        self._entity_refs.append(
            {
                "entity_text": entity_text,
                "fiber_id": fiber_id,
                "created_at": created_at or utcnow(),
                "promoted": False,
            }
        )

    async def count_entity_refs(self, entity_text: str) -> int:
        return sum(
            1 for r in self._entity_refs if r["entity_text"] == entity_text and not r["promoted"]
        )

    async def get_entity_ref_fiber_ids(self, entity_text: str) -> list[str]:
        return [r["fiber_id"] for r in self._entity_refs if r["entity_text"] == entity_text]

    async def mark_entity_refs_promoted(self, entity_text: str) -> int:
        count = 0
        for r in self._entity_refs:
            if r["entity_text"] == entity_text and not r["promoted"]:
                r["promoted"] = True
                count += 1
        return count

    async def prune_old_entity_refs(self, max_age_days: int = 90) -> int:
        cutoff = utcnow() - timedelta(days=max_age_days)
        before = len(self._entity_refs)
        self._entity_refs = [
            r for r in self._entity_refs if r["promoted"] or r["created_at"] >= cutoff
        ]
        return before - len(self._entity_refs)
