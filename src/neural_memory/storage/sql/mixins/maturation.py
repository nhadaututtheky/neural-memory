"""Memory maturation CRUD operations mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


class MaturationMixin:
    """Mixin providing memory maturation record CRUD operations."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def save_maturation(self, record: MaturationRecord) -> None:
        """Save or update a maturation record."""
        d = self._dialect
        brain_id = self._get_brain_id()

        try:
            await d.execute(
                f"""INSERT INTO memory_maturations
                (fiber_id, brain_id, stage, stage_entered_at, rehearsal_count,
                 reinforcement_timestamps)
                VALUES ({d.phs(6)})""",
                [
                    record.fiber_id,
                    brain_id,
                    record.stage.value,
                    d.serialize_dt(record.stage_entered_at),
                    record.rehearsal_count,
                    json.dumps(record.reinforcement_timestamps),
                ],
            )
        except Exception:
            # Fiber was deleted (e.g., by consolidation) between read and write.
            logger.debug("Skipping maturation save for deleted fiber %s", record.fiber_id)

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        """Get a maturation record for a fiber."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT fiber_id, brain_id, stage, stage_entered_at,
                      rehearsal_count, reinforcement_timestamps
               FROM memory_maturations
               WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}""",
            [brain_id, fiber_id],
        )
        if row is None:
            return None

        return MaturationRecord(
            fiber_id=row["fiber_id"],
            brain_id=row["brain_id"],
            stage=MemoryStage(row["stage"]),
            stage_entered_at=row["stage_entered_at"]
            if isinstance(row["stage_entered_at"], datetime)
            else datetime.fromisoformat(row["stage_entered_at"]),
            rehearsal_count=row["rehearsal_count"],
            reinforcement_timestamps=tuple(json.loads(row["reinforcement_timestamps"]))
            if row["reinforcement_timestamps"]
            else (),
        )

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        """Find maturation records matching criteria."""
        d = self._dialect
        brain_id = self._get_brain_id()

        conditions = [f"brain_id = {d.ph(1)}"]
        params: list[object] = [brain_id]

        if stage is not None:
            conditions.append(f"stage = {d.ph(len(params) + 1)}")
            params.append(stage.value)

        if min_rehearsal_count > 0:
            conditions.append(f"rehearsal_count >= {d.ph(len(params) + 1)}")
            params.append(min_rehearsal_count)

        where_clause = " AND ".join(conditions)
        rows = await d.fetch_all(
            f"""SELECT fiber_id, brain_id, stage, stage_entered_at,
                       rehearsal_count, reinforcement_timestamps
               FROM memory_maturations
               WHERE {where_clause}
               LIMIT 1000""",
            params,
        )

        records: list[MaturationRecord] = []
        for row in rows:
            records.append(
                MaturationRecord(
                    fiber_id=row["fiber_id"],
                    brain_id=row["brain_id"],
                    stage=MemoryStage(row["stage"]),
                    stage_entered_at=row["stage_entered_at"]
                    if isinstance(row["stage_entered_at"], datetime)
                    else datetime.fromisoformat(row["stage_entered_at"]),
                    rehearsal_count=row["rehearsal_count"],
                    reinforcement_timestamps=tuple(json.loads(row["reinforcement_timestamps"]))
                    if row["reinforcement_timestamps"]
                    else (),
                )
            )

        return records

    async def cleanup_orphaned_maturations(self) -> int:
        """Delete maturation records whose fiber no longer exists.

        Returns the number of orphaned records removed.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"""DELETE FROM memory_maturations
               WHERE brain_id = {d.ph(1)} AND fiber_id NOT IN (
                   SELECT id FROM fibers WHERE brain_id = {d.ph(2)}
               )""",
            [brain_id, brain_id],
        )
