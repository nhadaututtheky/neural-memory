"""Bayesian depth prior persistence mixin — dialect-agnostic."""

from __future__ import annotations

import logging
from datetime import datetime

from neural_memory.engine.depth_prior import DepthPrior
from neural_memory.engine.retrieval_types import DepthLevel
from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


def _row_to_prior(row: dict[str, object]) -> DepthPrior:
    """Convert a row dict to a DepthPrior dataclass."""

    def _safe_dt(val: object) -> datetime:
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    return DepthPrior(
        entity_text=str(row["entity_text"]),
        depth_level=DepthLevel(int(row["depth_level"])),  # type: ignore[call-overload]
        alpha=float(row["alpha"]),  # type: ignore[arg-type]
        beta=float(row["beta"]),  # type: ignore[arg-type]
        total_queries=int(row["total_queries"]),  # type: ignore[call-overload]
        last_updated=_safe_dt(row["last_updated"]),
        created_at=_safe_dt(row["created_at"]),
    )


class DepthPriorsMixin:
    """Mixin providing CRUD for the depth_priors table."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_depth_priors(self, entity_text: str) -> list[DepthPrior]:
        """Get all priors for an entity across all depth levels."""
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT * FROM depth_priors WHERE brain_id = {d.ph(1)} AND entity_text = {d.ph(2)}",
            [brain_id, entity_text],
        )
        return [_row_to_prior(r) for r in rows]

    async def get_depth_priors_batch(
        self,
        entity_texts: list[str],
    ) -> dict[str, list[DepthPrior]]:
        """Batch-fetch priors for multiple entities."""
        if not entity_texts:
            return {}

        d = self._dialect
        brain_id = self._get_brain_id()

        in_fragment, in_params = d.in_clause(2, entity_texts)
        rows = await d.fetch_all(
            f"SELECT * FROM depth_priors WHERE brain_id = {d.ph(1)} AND entity_text {in_fragment}",
            [brain_id, *in_params],
        )

        result: dict[str, list[DepthPrior]] = {t: [] for t in entity_texts}
        for raw in rows:
            prior = _row_to_prior(raw)
            result[prior.entity_text].append(prior)
        return result

    async def upsert_depth_prior(self, prior: DepthPrior) -> None:
        """Insert or update a single depth prior."""
        d = self._dialect
        brain_id = self._get_brain_id()

        await d.execute(
            f"""INSERT INTO depth_priors
                (brain_id, entity_text, depth_level, alpha, beta,
                 total_queries, last_updated, created_at)
                VALUES ({d.phs(8)})
                ON CONFLICT (brain_id, entity_text, depth_level) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta,
                    total_queries = excluded.total_queries,
                    last_updated = excluded.last_updated""",
            [
                brain_id,
                prior.entity_text,
                prior.depth_level.value,
                prior.alpha,
                prior.beta,
                prior.total_queries,
                d.serialize_dt(prior.last_updated),
                d.serialize_dt(prior.created_at),
            ],
        )

    async def get_stale_priors(self, older_than: datetime) -> list[DepthPrior]:
        """Find priors not updated since a given date."""
        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT * FROM depth_priors WHERE brain_id = {d.ph(1)} AND last_updated < {d.ph(2)}",
            [brain_id, d.serialize_dt(older_than)],
        )
        return [_row_to_prior(r) for r in rows]

    async def delete_depth_priors(self, entity_text: str) -> int:
        """Delete all priors for an entity. Returns count deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"DELETE FROM depth_priors WHERE brain_id = {d.ph(1)} AND entity_text = {d.ph(2)}",
            [brain_id, entity_text],
        )
