"""SQLite typed memory operations mixin."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.storage.sqlite_row_mappers import provenance_to_dict, row_to_typed_memory
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import aiosqlite


class SQLiteTypedMemoryMixin:
    """Mixin providing typed memory CRUD operations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Verify fiber exists
        async with conn.execute(
            "SELECT id FROM fibers WHERE id = ? AND brain_id = ?",
            (typed_memory.fiber_id, brain_id),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError(f"Fiber {typed_memory.fiber_id} does not exist")

        await conn.execute(
            """INSERT OR REPLACE INTO typed_memories
               (fiber_id, brain_id, memory_type, priority, provenance,
                expires_at, project_id, tags, metadata, created_at,
                trust_score, source, tier)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                typed_memory.fiber_id,
                brain_id,
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_to_dict(typed_memory.provenance)),
                typed_memory.expires_at.isoformat() if typed_memory.expires_at else None,
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                typed_memory.created_at.isoformat(),
                typed_memory.trust_score,
                typed_memory.source,
                typed_memory.tier,
            ),
        )
        await conn.commit()
        return typed_memory.fiber_id

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM typed_memories WHERE fiber_id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_typed_memory(row)

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
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM typed_memories WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        if min_priority is not None:
            query += " AND priority >= ?"
            params.append(min_priority.value)

        if not include_expired:
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(utcnow().isoformat())

        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)

        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            memories = [row_to_typed_memory(row) for row in rows]

        # Filter by tags in Python
        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT COUNT(*) FROM typed_memories WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)

        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        # Exclude expired
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(utcnow().isoformat())

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def count_typed_memories_grouped(
        self,
    ) -> list[tuple[str, str, int]]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        query = (
            "SELECT memory_type, tier, COUNT(*) FROM typed_memories"
            " WHERE brain_id = ? AND (expires_at IS NULL OR expires_at > ?)"
            " GROUP BY memory_type, tier"
        )
        params = [brain_id, utcnow().isoformat()]
        async with conn.execute(query, params) as cursor:
            return [(row[0], row[1], row[2]) async for row in cursor]

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE typed_memories SET memory_type = ?, priority = ?,
               provenance = ?, expires_at = ?, project_id = ?,
               tags = ?, metadata = ?, trust_score = ?, source = ?, tier = ?
               WHERE fiber_id = ? AND brain_id = ?""",
            (
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_to_dict(typed_memory.provenance)),
                typed_memory.expires_at.isoformat() if typed_memory.expires_at else None,
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

        if cursor.rowcount == 0:
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

        await conn.commit()

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        """Update only the source field on a typed memory. Returns True if updated."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "UPDATE typed_memories SET source = ? WHERE fiber_id = ? AND brain_id = ?",
            (source, fiber_id, brain_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM typed_memories WHERE fiber_id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        async with conn.execute(
            """SELECT * FROM typed_memories
               WHERE brain_id = ? AND expires_at IS NOT NULL AND expires_at <= ?
               LIMIT ?""",
            (brain_id, utcnow().isoformat(), limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row_to_typed_memory(row) for row in rows]

    async def get_expired_memory_count(self) -> int:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT COUNT(*) FROM typed_memories
               WHERE brain_id = ? AND expires_at IS NOT NULL AND expires_at <= ?""",
            (brain_id, utcnow().isoformat()),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        if not fiber_ids:
            return []
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow()
        now_iso = now.isoformat()
        deadline_iso = (now + timedelta(days=within_days)).isoformat()

        placeholders = ",".join("?" for _ in fiber_ids)
        query = f"""SELECT * FROM typed_memories
                    WHERE brain_id = ?
                      AND fiber_id IN ({placeholders})
                      AND expires_at IS NOT NULL
                      AND expires_at > ?
                      AND expires_at <= ?"""
        params: list[Any] = [brain_id, *fiber_ids, now_iso, deadline_iso]

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row_to_typed_memory(row) for row in rows]

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow()
        now_iso = now.isoformat()
        deadline_iso = (now + timedelta(days=within_days)).isoformat()

        async with conn.execute(
            """SELECT COUNT(*) FROM typed_memories
               WHERE brain_id = ? AND expires_at IS NOT NULL
                 AND expires_at > ? AND expires_at <= ?""",
            (brain_id, now_iso, deadline_iso),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

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
        """Find typed memories eligible for auto-promotion.

        Returns context memories whose fibers have frequency >= min_frequency.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT tm.fiber_id, tm.memory_type, tm.expires_at, tm.metadata,
                      f.frequency, f.conductivity
               FROM typed_memories tm
               JOIN fibers f ON f.id = tm.fiber_id AND f.brain_id = tm.brain_id
               WHERE tm.brain_id = ?
                 AND tm.memory_type = ?
                 AND f.frequency >= ?
                 AND f.pinned = 0
               LIMIT 200""",
            (brain_id, source_type, min_frequency),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "fiber_id": row[0],
                    "memory_type": row[1],
                    "expires_at": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {},
                    "frequency": row[4],
                    "conductivity": row[5],
                }
                for row in rows
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
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Fetch current metadata to preserve + augment
        cursor = await conn.execute(
            "SELECT metadata, memory_type FROM typed_memories WHERE fiber_id = ? AND brain_id = ?",
            (fiber_id, brain_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False

        current_meta = json.loads(row[0]) if row[0] else {}
        old_type = row[1]

        # Already promoted or already the target type
        if old_type == new_type.value:
            return False

        current_meta["auto_promoted"] = True
        current_meta["promoted_from"] = old_type
        current_meta["promoted_at"] = utcnow().isoformat()

        result = await conn.execute(
            """UPDATE typed_memories
               SET memory_type = ?, expires_at = ?, metadata = ?
               WHERE fiber_id = ? AND brain_id = ?""",
            (
                new_type.value,
                new_expires_at,
                json.dumps(current_meta),
                fiber_id,
                brain_id,
            ),
        )
        await conn.commit()

        if result.rowcount > 0:
            logger.info(
                "Auto-promoted fiber %s from %s to %s (frequency-based)",
                fiber_id,
                old_type,
                new_type.value,
            )
        return result.rowcount > 0
