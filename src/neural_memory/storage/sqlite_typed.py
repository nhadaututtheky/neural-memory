"""SQLite typed memory operations mixin."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.storage.sqlite_row_mappers import provenance_to_dict, row_to_typed_memory
from neural_memory.utils.timeutils import utcnow

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
                expires_at, project_id, tags, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            memories = [row_to_typed_memory(row) for row in rows]

        # Filter by tags in Python
        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE typed_memories SET memory_type = ?, priority = ?,
               provenance = ?, expires_at = ?, project_id = ?,
               tags = ?, metadata = ?
               WHERE fiber_id = ? AND brain_id = ?""",
            (
                typed_memory.memory_type.value,
                typed_memory.priority.value,
                json.dumps(provenance_to_dict(typed_memory.provenance)),
                typed_memory.expires_at.isoformat() if typed_memory.expires_at else None,
                typed_memory.project_id,
                json.dumps(list(typed_memory.tags)),
                json.dumps(typed_memory.metadata),
                typed_memory.fiber_id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

        await conn.commit()

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
