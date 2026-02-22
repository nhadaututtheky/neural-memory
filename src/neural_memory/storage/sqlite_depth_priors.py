"""SQLite mixin for Bayesian depth prior persistence."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.engine.depth_prior import DepthPrior
from neural_memory.engine.retrieval_types import DepthLevel

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


def _row_to_prior(row: dict[str, object]) -> DepthPrior:
    """Convert a sqlite Row dict to a DepthPrior dataclass."""

    def _parse_dt(val: object) -> datetime:
        return datetime.fromisoformat(str(val))

    return DepthPrior(
        entity_text=str(row["entity_text"]),
        depth_level=DepthLevel(int(row["depth_level"])),  # type: ignore[call-overload]
        alpha=float(row["alpha"]),  # type: ignore[arg-type]
        beta=float(row["beta"]),  # type: ignore[arg-type]
        total_queries=int(row["total_queries"]),  # type: ignore[call-overload]
        last_updated=_parse_dt(row["last_updated"]),
        created_at=_parse_dt(row["created_at"]),
    )


class SQLiteDepthPriorMixin:
    """Mixin providing CRUD for the depth_priors table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_depth_priors(self, entity_text: str) -> list[DepthPrior]:
        """Get all priors for an entity across all depth levels."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM depth_priors WHERE brain_id = ? AND entity_text = ?",
            (brain_id, entity_text),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_prior(dict(zip(col_names, r, strict=False))) for r in rows]

    async def get_depth_priors_batch(
        self,
        entity_texts: list[str],
    ) -> dict[str, list[DepthPrior]]:
        """Batch-fetch priors for multiple entities."""
        if not entity_texts:
            return {}

        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        placeholders = ",".join("?" for _ in entity_texts)
        cursor = await conn.execute(
            f"SELECT * FROM depth_priors WHERE brain_id = ? AND entity_text IN ({placeholders})",
            (brain_id, *entity_texts),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]

        result: dict[str, list[DepthPrior]] = {t: [] for t in entity_texts}
        for raw in rows:
            prior = _row_to_prior(dict(zip(col_names, raw, strict=False)))
            result[prior.entity_text].append(prior)
        return result

    async def upsert_depth_prior(self, prior: DepthPrior) -> None:
        """Insert or update a single depth prior."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT INTO depth_priors
               (brain_id, entity_text, depth_level, alpha, beta,
                total_queries, last_updated, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (brain_id, entity_text, depth_level) DO UPDATE SET
                   alpha = excluded.alpha,
                   beta = excluded.beta,
                   total_queries = excluded.total_queries,
                   last_updated = excluded.last_updated""",
            (
                brain_id,
                prior.entity_text,
                prior.depth_level.value,
                prior.alpha,
                prior.beta,
                prior.total_queries,
                prior.last_updated.isoformat(),
                prior.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_stale_priors(self, older_than: datetime) -> list[DepthPrior]:
        """Find priors not updated since a given date."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM depth_priors WHERE brain_id = ? AND last_updated < ?",
            (brain_id, older_than.isoformat()),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_prior(dict(zip(col_names, r, strict=False))) for r in rows]

    async def delete_depth_priors(self, entity_text: str) -> int:
        """Delete all priors for an entity. Returns count deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM depth_priors WHERE brain_id = ? AND entity_text = ?",
            (brain_id, entity_text),
        )
        await conn.commit()
        return cursor.rowcount
