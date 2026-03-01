"""SQLite review schedule storage mixin."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.core.review_schedule import ReviewSchedule
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


def _row_to_schedule(row: dict[str, object]) -> ReviewSchedule:
    """Convert a sqlite Row to a ReviewSchedule dataclass."""

    def _parse_dt(val: object) -> datetime | None:
        if val is None:
            return None
        return datetime.fromisoformat(str(val))

    return ReviewSchedule(
        fiber_id=str(row["fiber_id"]),
        brain_id=str(row["brain_id"]),
        box=int(row["box"]),  # type: ignore[call-overload]
        next_review=_parse_dt(row.get("next_review")),
        last_reviewed=_parse_dt(row.get("last_reviewed")),
        review_count=int(row.get("review_count", 0)),  # type: ignore[call-overload]
        streak=int(row.get("streak", 0)),  # type: ignore[call-overload]
        created_at=_parse_dt(row.get("created_at")),
    )


class SQLiteReviewsMixin:
    """Mixin providing review schedule CRUD operations for SQLiteStorage."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        """Insert or update a review schedule (upsert by fiber_id + brain_id)."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT INTO review_schedules
               (fiber_id, brain_id, box, next_review, last_reviewed,
                review_count, streak, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fiber_id, brain_id) DO UPDATE SET
                 box = excluded.box,
                 next_review = excluded.next_review,
                 last_reviewed = excluded.last_reviewed,
                 review_count = excluded.review_count,
                 streak = excluded.streak""",
            (
                schedule.fiber_id,
                brain_id,
                schedule.box,
                schedule.next_review.isoformat() if schedule.next_review else None,
                schedule.last_reviewed.isoformat() if schedule.last_reviewed else None,
                schedule.review_count,
                schedule.streak,
                (schedule.created_at or utcnow()).isoformat(),
            ),
        )
        await conn.commit()
        return schedule.fiber_id

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        """Get a review schedule by fiber ID."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM review_schedules WHERE brain_id = ? AND fiber_id = ?",
            (brain_id, fiber_id),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            col_names = [d[0] for d in (cursor.description or [])]
            return _row_to_schedule(dict(zip(col_names, row, strict=False)))

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        """Get review schedules that are due (next_review <= now)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 100)
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """SELECT * FROM review_schedules
               WHERE brain_id = ? AND next_review <= ?
               ORDER BY next_review ASC
               LIMIT ?""",
            (brain_id, now, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_schedule(dict(zip(col_names, r, strict=False))) for r in rows]

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        """Delete a review schedule. Returns True if deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM review_schedules WHERE brain_id = ? AND fiber_id = ?",
            (brain_id, fiber_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def get_review_stats(self) -> dict[str, int]:
        """Get review statistics for the current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN next_review <= ? THEN 1 ELSE 0 END) as due,
                 SUM(CASE WHEN box = 1 THEN 1 ELSE 0 END) as box_1,
                 SUM(CASE WHEN box = 2 THEN 1 ELSE 0 END) as box_2,
                 SUM(CASE WHEN box = 3 THEN 1 ELSE 0 END) as box_3,
                 SUM(CASE WHEN box = 4 THEN 1 ELSE 0 END) as box_4,
                 SUM(CASE WHEN box = 5 THEN 1 ELSE 0 END) as box_5
               FROM review_schedules WHERE brain_id = ?""",
            (now, brain_id),
        )
        row = await cursor.fetchone()
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
            "total": row[0] or 0,
            "due": row[1] or 0,
            "box_1": row[2] or 0,
            "box_2": row[3] or 0,
            "box_3": row[4] or 0,
            "box_4": row[5] or 0,
            "box_5": row[6] or 0,
        }
