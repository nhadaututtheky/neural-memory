"""Dialect-agnostic typed memory operations mixin.

Merges SQLiteTypedMemoryMixin and PostgresTypedMemoryMixin into a single
implementation that delegates all SQL execution to ``self._dialect``.

Replace ``?`` / ``$N`` placeholders with ``d.ph(N)`` throughout.
All SQL executes through ``self._dialect`` (a ``Dialect`` instance injected
at construction time).
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.storage.sql.row_mappers import provenance_to_dict, row_to_typed_memory
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Upsert column definitions (kept here to avoid duplication inside methods)
# --------------------------------------------------------------------------

_TYPED_MEMORY_COLUMNS = [
    "fiber_id",
    "brain_id",
    "memory_type",
    "priority",
    "provenance",
    "expires_at",
    "project_id",
    "tags",
    "metadata",
    "created_at",
    "trust_score",
    "source",
    "tier",
]

_TYPED_MEMORY_CONFLICT = ["brain_id", "fiber_id"]

_TYPED_MEMORY_UPDATE_ON_CONFLICT = [
    "memory_type",
    "priority",
    "provenance",
    "expires_at",
    "project_id",
    "tags",
    "metadata",
    "trust_score",
    "source",
    "tier",
]


class TypedMemoryMixin:
    """Dialect-agnostic typed memory CRUD.

    Requires the host class to provide::

        self._dialect: Dialect   -- injected at construction time
        self._get_brain_id()    -> str
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        d = self._dialect
        brain_id = self._get_brain_id()

        # Verify fiber exists
        row = await d.fetch_one(
            f"SELECT id FROM fibers WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (typed_memory.fiber_id, brain_id),
        )
        if row is None:
            raise ValueError(f"Fiber {typed_memory.fiber_id} does not exist")

        sql = d.upsert_sql(
            "typed_memories",
            _TYPED_MEMORY_COLUMNS,
            _TYPED_MEMORY_CONFLICT,
            _TYPED_MEMORY_UPDATE_ON_CONFLICT,
        )
        await d.execute(
            sql,
            (
                typed_memory.fiber_id,
                brain_id,
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_to_dict(typed_memory.provenance)),
                d.serialize_dt(typed_memory.expires_at),
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                d.serialize_dt(typed_memory.created_at),
                typed_memory.trust_score,
                typed_memory.source,
                typed_memory.tier,
            ),
        )
        return typed_memory.fiber_id

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM typed_memories WHERE fiber_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (fiber_id, brain_id),
        )
        if row is None:
            return None
        return row_to_typed_memory(d, row)

    # ------------------------------------------------------------------
    # find
    # ------------------------------------------------------------------

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
        limit = min(limit, 1000)
        d = self._dialect
        brain_id = self._get_brain_id()

        n = 1
        query = f"SELECT * FROM typed_memories WHERE brain_id = {d.ph(n)}"
        params: list[Any] = [brain_id]

        if memory_type is not None:
            n += 1
            query += f" AND memory_type = {d.ph(n)}"
            params.append(memory_type.value)

        if min_priority is not None:
            n += 1
            query += f" AND priority >= {d.ph(n)}"
            params.append(min_priority.value)

        if not include_expired:
            n += 1
            query += f" AND (expires_at IS NULL OR expires_at > {d.ph(n)})"
            params.append(d.serialize_dt(utcnow()))

        if project_id is not None:
            n += 1
            query += f" AND project_id = {d.ph(n)}"
            params.append(project_id)

        if tier is not None:
            n += 1
            query += f" AND tier = {d.ph(n)}"
            params.append(tier)

        n += 1
        query += f" ORDER BY priority DESC, created_at DESC LIMIT {d.ph(n)}"
        params.append(limit)

        rows = await d.fetch_all(query, params)
        memories = [row_to_typed_memory(d, r) for r in rows]

        # Filter by tags in Python (JSON array column, no native set operator)
        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    # ------------------------------------------------------------------
    # count
    # ------------------------------------------------------------------

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        d = self._dialect
        brain_id = self._get_brain_id()

        n = 1
        query = f"SELECT COUNT(*) AS cnt FROM typed_memories WHERE brain_id = {d.ph(n)}"
        params: list[Any] = [brain_id]

        if tier is not None:
            n += 1
            query += f" AND tier = {d.ph(n)}"
            params.append(tier)

        if memory_type is not None:
            n += 1
            query += f" AND memory_type = {d.ph(n)}"
            params.append(memory_type.value)

        # Exclude expired
        n += 1
        query += f" AND (expires_at IS NULL OR expires_at > {d.ph(n)})"
        params.append(d.serialize_dt(utcnow()))

        row = await d.fetch_one(query, params)
        return int(row["cnt"]) if row else 0

    async def count_typed_memories_grouped(
        self,
    ) -> list[tuple[str, str, int]]:
        d = self._dialect
        brain_id = self._get_brain_id()

        query = (
            "SELECT memory_type, tier, COUNT(*) AS cnt FROM typed_memories"
            f" WHERE brain_id = {d.ph(1)}"
            f" AND (expires_at IS NULL OR expires_at > {d.ph(2)})"
            " GROUP BY memory_type, tier"
        )
        rows = await d.fetch_all(query, (brain_id, d.serialize_dt(utcnow())))
        return [(str(r["memory_type"]), str(r["tier"]), int(r["cnt"])) for r in rows]

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        cnt = await d.execute_count(
            f"""UPDATE typed_memories SET
                memory_type = {d.ph(1)}, priority = {d.ph(2)},
                provenance = {d.ph(3)}, expires_at = {d.ph(4)},
                project_id = {d.ph(5)}, tags = {d.ph(6)},
                metadata = {d.ph(7)}, trust_score = {d.ph(8)},
                source = {d.ph(9)}, tier = {d.ph(10)}
                WHERE fiber_id = {d.ph(11)} AND brain_id = {d.ph(12)}""",
            (
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_to_dict(typed_memory.provenance)),
                d.serialize_dt(typed_memory.expires_at),
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                typed_memory.trust_score,
                typed_memory.source,
                typed_memory.tier,
                typed_memory.fiber_id,
                brain_id,
            ),
        )
        if cnt == 0:
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        """Update only the source field on a typed memory. Returns True if updated."""
        d = self._dialect
        brain_id = self._get_brain_id()

        cnt = await d.execute_count(
            f"UPDATE typed_memories SET source = {d.ph(1)}"
            f" WHERE fiber_id = {d.ph(2)} AND brain_id = {d.ph(3)}",
            (source, fiber_id, brain_id),
        )
        return cnt > 0

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()

        cnt = await d.execute_count(
            f"DELETE FROM typed_memories WHERE fiber_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (fiber_id, brain_id),
        )
        return cnt > 0

    # ------------------------------------------------------------------
    # expired
    # ------------------------------------------------------------------

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        d = self._dialect
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        rows = await d.fetch_all(
            f"""SELECT * FROM typed_memories
                WHERE brain_id = {d.ph(1)}
                  AND expires_at IS NOT NULL
                  AND expires_at <= {d.ph(2)}
                LIMIT {d.ph(3)}""",
            (brain_id, d.serialize_dt(utcnow()), limit),
        )
        return [row_to_typed_memory(d, r) for r in rows]

    async def get_expired_memory_count(self) -> int:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT COUNT(*) AS cnt FROM typed_memories
                WHERE brain_id = {d.ph(1)}
                  AND expires_at IS NOT NULL
                  AND expires_at <= {d.ph(2)}""",
            (brain_id, d.serialize_dt(utcnow())),
        )
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        if not fiber_ids:
            return []
        d = self._dialect
        brain_id = self._get_brain_id()
        now = utcnow()
        deadline = now + timedelta(days=within_days)

        in_frag, in_params = d.in_clause(2, fiber_ids)
        # Parameter slots: 1=brain_id, 2..N=in_clause, N+1=now, N+2=deadline
        next_idx = 2 + len(in_params)

        query = f"""SELECT * FROM typed_memories
                    WHERE brain_id = {d.ph(1)}
                      AND fiber_id {in_frag}
                      AND expires_at IS NOT NULL
                      AND expires_at > {d.ph(next_idx)}
                      AND expires_at <= {d.ph(next_idx + 1)}"""
        params: list[Any] = [brain_id, *in_params, d.serialize_dt(now), d.serialize_dt(deadline)]

        rows = await d.fetch_all(query, params)
        return [row_to_typed_memory(d, r) for r in rows]

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        d = self._dialect
        brain_id = self._get_brain_id()
        now = utcnow()
        deadline = now + timedelta(days=within_days)

        row = await d.fetch_one(
            f"""SELECT COUNT(*) AS cnt FROM typed_memories
                WHERE brain_id = {d.ph(1)}
                  AND expires_at IS NOT NULL
                  AND expires_at > {d.ph(2)}
                  AND expires_at <= {d.ph(3)}""",
            (brain_id, d.serialize_dt(now), d.serialize_dt(deadline)),
        )
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------
    # project
    # ------------------------------------------------------------------

    async def get_project_memories(
        self,
        project_id: str,
        include_expired: bool = False,
    ) -> list[TypedMemory]:
        return await self.find_typed_memories(
            project_id=project_id,
            include_expired=include_expired,
        )

    # ------------------------------------------------------------------
    # promotion
    # ------------------------------------------------------------------

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        """Find typed memories eligible for auto-promotion.

        Returns context memories whose fibers have frequency >= min_frequency.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"""SELECT tm.fiber_id, tm.memory_type, tm.expires_at, tm.metadata,
                       f.frequency, f.conductivity
                FROM typed_memories tm
                JOIN fibers f ON f.id = tm.fiber_id AND f.brain_id = tm.brain_id
                WHERE tm.brain_id = {d.ph(1)}
                  AND tm.memory_type = {d.ph(2)}
                  AND f.frequency >= {d.ph(3)}
                  AND f.pinned = 0
                LIMIT 200""",
            (brain_id, source_type, min_frequency),
        )
        return [
            {
                "fiber_id": str(r["fiber_id"]),
                "memory_type": str(r["memory_type"]),
                "expires_at": r["expires_at"],
                "metadata": json.loads(r["metadata"])
                if isinstance(r["metadata"], str)
                else (r["metadata"] or {}),
                "frequency": int(r["frequency"]),
                "conductivity": float(r["conductivity"]),
            }
            for r in rows
        ]

    async def promote_memory_type(
        self,
        fiber_id: str,
        new_type: MemoryType,
        new_expires_at: str | None = None,
    ) -> bool:
        """Promote a memory's type and update its expiry.

        Stores the original type in metadata for audit trail.
        Returns True if the promotion was applied.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        # Fetch current metadata to preserve + augment
        row = await d.fetch_one(
            f"SELECT metadata, memory_type FROM typed_memories"
            f" WHERE fiber_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (fiber_id, brain_id),
        )
        if not row:
            return False

        meta_raw = row["metadata"]
        current_meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        old_type = str(row["memory_type"])

        # Already the target type -- nothing to do
        if old_type == new_type.value:
            return False

        current_meta["auto_promoted"] = True
        current_meta["promoted_from"] = old_type
        current_meta["promoted_at"] = utcnow().isoformat()

        cnt = await d.execute_count(
            f"""UPDATE typed_memories
                SET memory_type = {d.ph(1)}, expires_at = {d.ph(2)}, metadata = {d.ph(3)}
                WHERE fiber_id = {d.ph(4)} AND brain_id = {d.ph(5)}""",
            (
                new_type.value,
                new_expires_at,
                json.dumps(current_meta),
                fiber_id,
                brain_id,
            ),
        )

        if cnt > 0:
            logger.info(
                "Auto-promoted fiber %s from %s to %s (frequency-based)",
                fiber_id,
                old_type,
                new_type.value,
            )
            return True
        return False
