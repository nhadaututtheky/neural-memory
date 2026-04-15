"""Review schedule storage mixin — dialect-agnostic."""

from __future__ import annotations

import logging
from datetime import datetime

from neural_memory.core.review_schedule import ReviewSchedule
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _row_to_schedule(row: dict[str, object]) -> ReviewSchedule:
    """Convert a row dict to a ReviewSchedule dataclass."""

    def _safe_dt(val: object) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    return ReviewSchedule(
        fiber_id=str(row["fiber_id"]),
        brain_id=str(row["brain_id"]),
        box=int(row["box"]),  # type: ignore[call-overload]
        next_review=_safe_dt(row.get("next_review")),
        last_reviewed=_safe_dt(row.get("last_reviewed")),
        review_count=int(row.get("review_count", 0)),  # type: ignore[call-overload]
        streak=int(row.get("streak", 0)),  # type: ignore[call-overload]
        ease_factor=float(row.get("ease_factor") or 2.5),  # type: ignore[arg-type]
        created_at=_safe_dt(row.get("created_at")),
    )


class ReviewsMixin:
    """Mixin providing review schedule CRUD operations for SQLStorage."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        """Insert or update a review schedule (upsert by fiber_id + brain_id)."""
        d = self._dialect
        brain_id = self._get_brain_id()

        await d.execute(
            f"""INSERT INTO review_schedules
                (fiber_id, brain_id, box, next_review, last_reviewed,
                 review_count, streak, ease_factor, created_at)
                VALUES ({d.phs(9)})
                ON CONFLICT(fiber_id, brain_id) DO UPDATE SET
                  box = excluded.box,
                  next_review = excluded.next_review,
                  last_reviewed = excluded.last_reviewed,
                  review_count = excluded.review_count,
                  streak = excluded.streak,
                  ease_factor = excluded.ease_factor""",
            [
                schedule.fiber_id,
                brain_id,
                schedule.box,
                d.serialize_dt(schedule.next_review),
                d.serialize_dt(schedule.last_reviewed),
                schedule.review_count,
                schedule.streak,
                schedule.ease_factor,
                d.serialize_dt(schedule.created_at or utcnow()),
            ],
        )
        return schedule.fiber_id

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        """Get a review schedule by fiber ID."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM review_schedules WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}",
            [brain_id, fiber_id],
        )
        if not row:
            return None
        return _row_to_schedule(row)

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        """Get review schedules that are due (next_review <= now)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 100)
        now = d.serialize_dt(utcnow())

        rows = await d.fetch_all(
            f"""SELECT * FROM review_schedules
                WHERE brain_id = {d.ph(1)} AND next_review <= {d.ph(2)}
                ORDER BY next_review ASC
                LIMIT {d.ph(3)}""",
            [brain_id, now, safe_limit],
        )
        return [_row_to_schedule(r) for r in rows]

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        """Delete a review schedule. Returns True if deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM review_schedules WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}",
            [brain_id, fiber_id],
        )
        return count > 0

    async def get_review_stats(self) -> dict[str, int]:
        """Get review statistics for the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        row = await d.fetch_one(
            f"""SELECT
                  COUNT(*) as total,
                  SUM(CASE WHEN next_review <= {d.ph(1)} THEN 1 ELSE 0 END) as due,
                  SUM(CASE WHEN box = 1 THEN 1 ELSE 0 END) as box_1,
                  SUM(CASE WHEN box = 2 THEN 1 ELSE 0 END) as box_2,
                  SUM(CASE WHEN box = 3 THEN 1 ELSE 0 END) as box_3,
                  SUM(CASE WHEN box = 4 THEN 1 ELSE 0 END) as box_4,
                  SUM(CASE WHEN box = 5 THEN 1 ELSE 0 END) as box_5
                FROM review_schedules WHERE brain_id = {d.ph(2)}""",
            [now, brain_id],
        )
        if not row:
            return {
                "total": 0,
                "due": 0,
                "box_1": 0,
                "box_2": 0,
                "box_3": 0,
                "box_4": 0,
                "box_5": 0,
            }
        return {
            "total": row.get("total") or 0,
            "due": row.get("due") or 0,
            "box_1": row.get("box_1") or 0,
            "box_2": row.get("box_2") or 0,
            "box_3": row.get("box_3") or 0,
            "box_4": row.get("box_4") or 0,
            "box_5": row.get("box_5") or 0,
        }
