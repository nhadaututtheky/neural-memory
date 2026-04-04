"""Entity reference tracking mixin (lazy entity promotion) — dialect-agnostic."""

from __future__ import annotations

import logging
from datetime import datetime

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class EntityRefsMixin:
    """Mixin providing CRUD for the entity_refs table.

    Tracks entity mentions before they are promoted to full neurons.
    Entities need N mentions (default 2) before becoming neurons.
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_entity_ref(
        self, entity_text: str, fiber_id: str, created_at: datetime | None = None
    ) -> None:
        """Record an entity mention for a fiber."""
        d = self._dialect
        brain_id = self._get_brain_id()
        ts = d.serialize_dt(created_at or utcnow())

        insert_sql = d.insert_or_ignore_sql(
            "entity_refs",
            ["brain_id", "entity_text", "fiber_id", "created_at", "promoted"],
            ["brain_id", "entity_text", "fiber_id"],
        )
        await d.execute(
            insert_sql,
            [brain_id, entity_text, fiber_id, ts, 0],
        )

    async def count_entity_refs(self, entity_text: str) -> int:
        """Count how many fibers mention this entity (unpromoted only)."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT COUNT(*) as cnt FROM entity_refs
               WHERE brain_id = {d.ph(1)} AND entity_text = {d.ph(2)} AND promoted = 0""",
            [brain_id, entity_text],
        )
        return int(row.get("cnt", 0)) if row else 0

    async def get_entity_ref_fiber_ids(self, entity_text: str) -> list[str]:
        """Get fiber IDs that reference this entity (for retroactive linking)."""
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"""SELECT fiber_id FROM entity_refs
               WHERE brain_id = {d.ph(1)} AND entity_text = {d.ph(2)} AND promoted = 0""",
            [brain_id, entity_text],
        )
        return [str(r["fiber_id"]) for r in rows]

    async def mark_entity_refs_promoted(self, entity_text: str) -> int:
        """Mark all refs for an entity as promoted. Returns count updated."""
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"""UPDATE entity_refs SET promoted = 1
               WHERE brain_id = {d.ph(1)} AND entity_text = {d.ph(2)} AND promoted = 0""",
            [brain_id, entity_text],
        )

    async def prune_old_entity_refs(self, max_age_days: int = 90) -> int:
        """Remove unpromoted entity refs older than max_age_days."""
        d = self._dialect
        brain_id = self._get_brain_id()
        from datetime import timedelta

        cutoff = d.serialize_dt(utcnow() - timedelta(days=max_age_days))

        return await d.execute_count(
            f"""DELETE FROM entity_refs
               WHERE brain_id = {d.ph(1)} AND promoted = 0
               AND created_at < {d.ph(2)}""",
            [brain_id, cutoff],
        )
