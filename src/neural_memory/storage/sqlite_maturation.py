"""SQLite mixin for memory maturation CRUD operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage

if TYPE_CHECKING:
    import aiosqlite


class SQLiteMaturationMixin:
    """SQLite implementation of maturation storage operations."""

    _conn: aiosqlite.Connection | None
    _current_brain_id: str | None

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Storage not initialized")
        return self._conn

    def _get_brain_id(self) -> str:
        if self._current_brain_id is None:
            raise RuntimeError("No brain selected")
        return self._current_brain_id

    async def save_maturation(self, record: MaturationRecord) -> None:
        """Save or update a maturation record."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT OR REPLACE INTO memory_maturations
            (fiber_id, brain_id, stage, stage_entered_at, rehearsal_count,
             reinforcement_timestamps)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.fiber_id,
                brain_id,
                record.stage.value,
                record.stage_entered_at.isoformat(),
                record.rehearsal_count,
                json.dumps(record.reinforcement_timestamps),
            ),
        )
        await conn.commit()

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        """Get a maturation record for a fiber."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT fiber_id, brain_id, stage, stage_entered_at,
                      rehearsal_count, reinforcement_timestamps
            FROM memory_maturations
            WHERE brain_id = ? AND fiber_id = ?""",
            (brain_id, fiber_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return MaturationRecord(
            fiber_id=row[0],
            brain_id=row[1],
            stage=MemoryStage(row[2]),
            stage_entered_at=datetime.fromisoformat(row[3]),
            rehearsal_count=row[4],
            reinforcement_timestamps=tuple(json.loads(row[5])) if row[5] else (),
        )

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        """Find maturation records matching criteria."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        conditions = ["brain_id = ?"]
        params: list[object] = [brain_id]

        if stage is not None:
            conditions.append("stage = ?")
            params.append(stage.value)

        if min_rehearsal_count > 0:
            conditions.append("rehearsal_count >= ?")
            params.append(min_rehearsal_count)

        where_clause = " AND ".join(conditions)
        cursor = await conn.execute(
            f"""SELECT fiber_id, brain_id, stage, stage_entered_at,
                       rehearsal_count, reinforcement_timestamps
            FROM memory_maturations
            WHERE {where_clause}
            LIMIT 1000""",
            tuple(params),
        )

        records: list[MaturationRecord] = []
        async for row in cursor:
            records.append(
                MaturationRecord(
                    fiber_id=row[0],
                    brain_id=row[1],
                    stage=MemoryStage(row[2]),
                    stage_entered_at=datetime.fromisoformat(row[3]),
                    rehearsal_count=row[4],
                    reinforcement_timestamps=tuple(json.loads(row[5])) if row[5] else (),
                )
            )

        return records
