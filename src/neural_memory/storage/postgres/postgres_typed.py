"""PostgreSQL typed memory operations mixin."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import (
    pg_row_to_typed_memory,
    provenance_to_dict,
)
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class PostgresTypedMemoryMixin(PostgresBaseMixin):
    """PostgreSQL typed memory CRUD."""

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        brain_id = self._get_brain_id()

        row = await self._query_one(
            "SELECT id FROM fibers WHERE id = $1 AND brain_id = $2",
            typed_memory.fiber_id,
            brain_id,
        )
        if row is None:
            raise ValueError(f"Fiber {typed_memory.fiber_id} does not exist")

        await self._query(
            """INSERT INTO typed_memories
               (fiber_id, brain_id, memory_type, priority, provenance,
                expires_at, project_id, tags, metadata, created_at,
                trust_score, source, tier)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
               ON CONFLICT (brain_id, fiber_id) DO UPDATE SET
                 memory_type = EXCLUDED.memory_type,
                 priority = EXCLUDED.priority,
                 provenance = EXCLUDED.provenance,
                 expires_at = EXCLUDED.expires_at,
                 project_id = EXCLUDED.project_id,
                 tags = EXCLUDED.tags,
                 metadata = EXCLUDED.metadata,
                 trust_score = EXCLUDED.trust_score,
                 source = EXCLUDED.source,
                 tier = EXCLUDED.tier""",
            typed_memory.fiber_id,
            brain_id,
            typed_memory.memory_type.value,
            typed_memory.priority.value,
            json.dumps(provenance_to_dict(typed_memory.provenance)),
            typed_memory.expires_at,
            typed_memory.project_id,
            json.dumps(list(typed_memory.tags)),
            json.dumps(typed_memory.metadata),
            typed_memory.created_at,
            typed_memory.trust_score,
            typed_memory.source,
            typed_memory.tier,
        )
        return typed_memory.fiber_id

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM typed_memories WHERE fiber_id = $1 AND brain_id = $2",
            fiber_id,
            brain_id,
        )
        if row is None:
            return None
        return pg_row_to_typed_memory(row)

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
        brain_id = self._get_brain_id()

        query = "SELECT * FROM typed_memories WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if memory_type is not None:
            params.append(memory_type.value)
            query += f" AND memory_type = ${len(params)}"

        if min_priority is not None:
            params.append(min_priority.value)
            query += f" AND priority >= ${len(params)}"

        if not include_expired:
            params.append(utcnow())
            query += f" AND (expires_at IS NULL OR expires_at > ${len(params)})"

        if project_id is not None:
            params.append(project_id)
            query += f" AND project_id = ${len(params)}"

        if tier is not None:
            params.append(tier)
            query += f" AND tier = ${len(params)}"

        params.append(limit)
        query += f" ORDER BY priority DESC, created_at DESC LIMIT ${len(params)}"

        rows = await self._query_ro(query, *params)
        memories = [pg_row_to_typed_memory(r) for r in rows]

        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        brain_id = self._get_brain_id()

        query = "SELECT COUNT(*) FROM typed_memories WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if tier is not None:
            params.append(tier)
            query += f" AND tier = ${len(params)}"

        if memory_type is not None:
            params.append(memory_type.value)
            query += f" AND memory_type = ${len(params)}"

        params.append(utcnow())
        query += f" AND (expires_at IS NULL OR expires_at > ${len(params)})"

        row = await self._query_one(query, *params)
        return row[0] if row else 0

    async def count_typed_memories_grouped(
        self,
    ) -> list[tuple[str, str, int]]:
        brain_id = self._get_brain_id()
        query = (
            "SELECT memory_type, tier, COUNT(*) FROM typed_memories"
            " WHERE brain_id = $1 AND (expires_at IS NULL OR expires_at > $2)"
            " GROUP BY memory_type, tier"
        )
        rows = await self._query_ro(query, brain_id, utcnow())
        return [(row[0], row[1], row[2]) for row in rows]

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        brain_id = self._get_brain_id()
        result = await self._query(
            """UPDATE typed_memories SET memory_type = $1, priority = $2,
               provenance = $3, expires_at = $4, project_id = $5,
               tags = $6, metadata = $7, trust_score = $8, source = $9,
               tier = $10
               WHERE fiber_id = $11 AND brain_id = $12""",
            typed_memory.memory_type.value,
            typed_memory.priority.value,
            json.dumps(provenance_to_dict(typed_memory.provenance)),
            typed_memory.expires_at,
            typed_memory.project_id,
            json.dumps(list(typed_memory.tags)),
            json.dumps(typed_memory.metadata),
            typed_memory.trust_score,
            typed_memory.source,
            typed_memory.tier,
            typed_memory.fiber_id,
            brain_id,
        )
        if result and "UPDATE 0" in str(result):
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        brain_id = self._get_brain_id()
        result = await self._query(
            "UPDATE typed_memories SET source = $1 WHERE fiber_id = $2 AND brain_id = $3",
            source,
            fiber_id,
            brain_id,
        )
        return result is not None and "UPDATE 0" not in str(result)

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()
        result = await self._query(
            "DELETE FROM typed_memories WHERE fiber_id = $1 AND brain_id = $2",
            fiber_id,
            brain_id,
        )
        return result is not None and "DELETE 0" not in str(result)

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        rows = await self._query_ro(
            """SELECT * FROM typed_memories
               WHERE brain_id = $1 AND expires_at IS NOT NULL AND expires_at <= $2
               LIMIT $3""",
            brain_id,
            utcnow(),
            limit,
        )
        return [pg_row_to_typed_memory(r) for r in rows]

    async def get_expired_memory_count(self) -> int:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            """SELECT COUNT(*) AS cnt FROM typed_memories
               WHERE brain_id = $1 AND expires_at IS NOT NULL AND expires_at <= $2""",
            brain_id,
            utcnow(),
        )
        return int(row["cnt"]) if row else 0

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        if not fiber_ids:
            return []
        brain_id = self._get_brain_id()
        now = utcnow()
        deadline = now + timedelta(days=within_days)
        rows = await self._query_ro(
            """SELECT * FROM typed_memories
               WHERE brain_id = $1
                 AND fiber_id = ANY($2::text[])
                 AND expires_at IS NOT NULL
                 AND expires_at > $3
                 AND expires_at <= $4""",
            brain_id,
            fiber_ids,
            now,
            deadline,
        )
        return [pg_row_to_typed_memory(r) for r in rows]

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        brain_id = self._get_brain_id()
        now = utcnow()
        deadline = now + timedelta(days=within_days)
        row = await self._query_one(
            """SELECT COUNT(*) AS cnt FROM typed_memories
               WHERE brain_id = $1 AND expires_at IS NOT NULL
                 AND expires_at > $2 AND expires_at <= $3""",
            brain_id,
            now,
            deadline,
        )
        return int(row["cnt"]) if row else 0

    async def get_project_memories(
        self,
        project_id: str,
        include_expired: bool = False,
    ) -> list[TypedMemory]:
        return await self.find_typed_memories(
            project_id=project_id,
            include_expired=include_expired,
        )

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            """SELECT tm.fiber_id, tm.memory_type, tm.expires_at, tm.metadata,
                      f.frequency, f.conductivity
               FROM typed_memories tm
               JOIN fibers f ON f.id = tm.fiber_id AND f.brain_id = tm.brain_id
               WHERE tm.brain_id = $1
                 AND tm.memory_type = $2
                 AND f.frequency >= $3
                 AND f.pinned = 0
               LIMIT 200""",
            brain_id,
            source_type,
            min_frequency,
        )
        return [
            {
                "fiber_id": str(r["fiber_id"]),
                "memory_type": str(r["memory_type"]),
                "expires_at": r["expires_at"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
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
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT metadata, memory_type FROM typed_memories WHERE fiber_id = $1 AND brain_id = $2",
            fiber_id,
            brain_id,
        )
        if not row:
            return False

        current_meta = json.loads(row["metadata"]) if row["metadata"] else {}
        old_type = str(row["memory_type"])

        if old_type == new_type.value:
            return False

        current_meta["auto_promoted"] = True
        current_meta["promoted_from"] = old_type
        current_meta["promoted_at"] = utcnow().isoformat()

        result = await self._query(
            """UPDATE typed_memories
               SET memory_type = $1, expires_at = $2, metadata = $3
               WHERE fiber_id = $4 AND brain_id = $5""",
            new_type.value,
            new_expires_at,
            json.dumps(current_meta),
            fiber_id,
            brain_id,
        )

        if result and "UPDATE 0" not in str(result):
            logger.info(
                "Auto-promoted fiber %s from %s to %s (frequency-based)",
                fiber_id,
                old_type,
                new_type.value,
            )
            return True
        return False
