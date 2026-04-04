"""Cognitive layer storage mixin — dialect-agnostic.

Covers three tables:
- cognitive_state: hypothesis/prediction confidence tracking
- hot_index: ranked summary of active cognitive items
- knowledge_gaps: metacognition — what the brain doesn't know
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Cap cognitive entries per brain to prevent unbounded growth
_MAX_COGNITIVE_PER_BRAIN = 5_000

# Hot index slot limits
_MAX_HOT_SLOTS = 20


def _parse_related_neuron_ids(raw: Any) -> list[str]:
    """Safely parse a JSON-encoded list of neuron IDs."""
    try:
        result: list[str] = json.loads(raw or "[]")
        return result
    except (json.JSONDecodeError, TypeError):
        return []


class CognitiveMixin:
    """Mixin providing CRUD for cognitive_state, hot_index, and knowledge_gaps."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ──────────────────── Cognitive State ────────────────────

    async def upsert_cognitive_state(
        self,
        neuron_id: str,
        *,
        confidence: float = 0.5,
        evidence_for_count: int = 0,
        evidence_against_count: int = 0,
        status: str = "active",
        predicted_at: str | None = None,
        resolved_at: str | None = None,
        schema_version: int = 1,
        parent_schema_id: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Insert or update a cognitive state record."""
        d = self._dialect
        brain_id = self._get_brain_id()

        await d.execute(
            f"""INSERT INTO cognitive_state
               (brain_id, neuron_id, confidence, evidence_for_count,
                evidence_against_count, status, predicted_at, resolved_at,
                schema_version, parent_schema_id, last_evidence_at, created_at)
               VALUES ({d.phs(12)})
               ON CONFLICT (brain_id, neuron_id) DO UPDATE SET
                 confidence = EXCLUDED.confidence,
                 evidence_for_count = EXCLUDED.evidence_for_count,
                 evidence_against_count = EXCLUDED.evidence_against_count,
                 status = EXCLUDED.status,
                 predicted_at = EXCLUDED.predicted_at,
                 resolved_at = EXCLUDED.resolved_at,
                 schema_version = EXCLUDED.schema_version,
                 parent_schema_id = EXCLUDED.parent_schema_id,
                 last_evidence_at = EXCLUDED.last_evidence_at""",
            (
                brain_id,
                neuron_id,
                max(0.01, min(0.99, confidence)),
                evidence_for_count,
                evidence_against_count,
                status,
                predicted_at,
                resolved_at,
                schema_version,
                parent_schema_id,
                last_evidence_at,
                d.serialize_dt(utcnow()),
            ),
        )

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        """Get cognitive state for a neuron."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                      status, predicted_at, resolved_at, schema_version,
                      parent_schema_id, last_evidence_at, created_at
               FROM cognitive_state
               WHERE brain_id = {d.ph(1)} AND neuron_id = {d.ph(2)}""",
            (brain_id, neuron_id),
        )
        return dict(row) if row else None

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List cognitive states, optionally filtered by status."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if status:
            rows = await d.fetch_all(
                f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = {d.ph(1)} AND status = {d.ph(2)}
                   ORDER BY confidence DESC LIMIT {d.ph(3)}""",
                (brain_id, status, capped_limit),
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = {d.ph(1)}
                   ORDER BY confidence DESC LIMIT {d.ph(2)}""",
                (brain_id, capped_limit),
            )
        return [dict(r) for r in rows]

    async def update_cognitive_evidence(
        self,
        neuron_id: str,
        *,
        confidence: float,
        evidence_for_count: int,
        evidence_against_count: int,
        status: str,
        resolved_at: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Update only evidence-related fields of a cognitive state.

        Unlike upsert_cognitive_state, this preserves predicted_at,
        schema_version, parent_schema_id, and created_at unchanged.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        await d.execute(
            f"""UPDATE cognitive_state SET
                 confidence = {d.ph(1)},
                 evidence_for_count = {d.ph(2)},
                 evidence_against_count = {d.ph(3)},
                 status = {d.ph(4)},
                 resolved_at = {d.ph(5)},
                 last_evidence_at = {d.ph(6)}
               WHERE brain_id = {d.ph(7)} AND neuron_id = {d.ph(8)}""",
            (
                max(0.01, min(0.99, confidence)),
                evidence_for_count,
                evidence_against_count,
                status,
                resolved_at,
                last_evidence_at,
                brain_id,
                neuron_id,
            ),
        )

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List predictions (cognitive states with predicted_at set)."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if status:
            rows = await d.fetch_all(
                f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = {d.ph(1)} AND predicted_at IS NOT NULL AND status = {d.ph(2)}
                   ORDER BY predicted_at ASC LIMIT {d.ph(3)}""",
                (brain_id, status, capped_limit),
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = {d.ph(1)} AND predicted_at IS NOT NULL
                   ORDER BY predicted_at ASC LIMIT {d.ph(2)}""",
                (brain_id, capped_limit),
            )
        return [dict(r) for r in rows]

    async def get_calibration_stats(self) -> dict[str, int]:
        """Get prediction calibration statistics.

        Returns:
            Dict with correct_count, wrong_count, total_resolved, pending_count.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT
                 COALESCE(SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END), 0) AS correct,
                 COALESCE(SUM(CASE WHEN status = 'refuted' THEN 1 ELSE 0 END), 0) AS wrong,
                 COALESCE(SUM(CASE WHEN status IN ('confirmed', 'refuted') THEN 1 ELSE 0 END), 0) AS resolved,
                 COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending
               FROM cognitive_state
               WHERE brain_id = {d.ph(1)} AND predicted_at IS NOT NULL""",
            (brain_id,),
        )
        if not row:
            return {"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
        return {
            "correct_count": int(row["correct"]),
            "wrong_count": int(row["wrong"]),
            "total_resolved": int(row["resolved"]),
            "pending_count": int(row["pending"]),
        }

    # ──────────────────── Hot Index ────────────────────

    async def refresh_hot_index(
        self,
        items: list[dict[str, Any]],
    ) -> int:
        """Replace the hot index with freshly scored items.

        Args:
            items: List of dicts with keys: slot, category, neuron_id,
                   summary, confidence, score.

        Returns:
            Number of items written.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        # Clear existing index for this brain
        await d.execute(
            f"DELETE FROM hot_index WHERE brain_id = {d.ph(1)}",
            (brain_id,),
        )

        count = 0
        for item in items[:_MAX_HOT_SLOTS]:
            await d.execute(
                f"""INSERT INTO hot_index
                   (brain_id, slot, category, neuron_id, summary, confidence, score, updated_at)
                   VALUES ({d.phs(8)})""",
                (
                    brain_id,
                    item["slot"],
                    item["category"],
                    item["neuron_id"],
                    item["summary"][:500],
                    item.get("confidence"),
                    item["score"],
                    now,
                ),
            )
            count += 1
        return count

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the current hot index items, sorted by score descending."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_HOT_SLOTS)

        rows = await d.fetch_all(
            f"""SELECT slot, category, neuron_id, summary, confidence, score, updated_at
               FROM hot_index
               WHERE brain_id = {d.ph(1)}
               ORDER BY score DESC LIMIT {d.ph(2)}""",
            (brain_id, capped),
        )
        return [dict(r) for r in rows]

    # ──────────────────── Knowledge Gaps ────────────────────

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        """Create a new knowledge gap record.

        Returns:
            The generated gap ID.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        gap_id = str(uuid4())

        await d.execute(
            f"""INSERT INTO knowledge_gaps
               (id, brain_id, topic, detected_at, detection_source,
                related_neuron_ids, priority)
               VALUES ({d.phs(7)})""",
            (
                gap_id,
                brain_id,
                topic[:500],
                d.serialize_dt(utcnow()),
                detection_source,
                json.dumps(related_neuron_ids or []),
                max(0.0, min(1.0, priority)),
            ),
        )
        return gap_id

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        """Get a single knowledge gap by ID."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT id, topic, detected_at, detection_source,
                      related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
               FROM knowledge_gaps
               WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}""",
            (brain_id, gap_id),
        )
        if not row:
            return None
        result = dict(row)
        result["related_neuron_ids"] = _parse_related_neuron_ids(result.get("related_neuron_ids"))
        return result

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List knowledge gaps sorted by priority descending."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped = min(limit, 200)

        if include_resolved:
            rows = await d.fetch_all(
                f"""SELECT id, topic, detected_at, detection_source,
                          related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
                   FROM knowledge_gaps
                   WHERE brain_id = {d.ph(1)}
                   ORDER BY priority DESC LIMIT {d.ph(2)}""",
                (brain_id, capped),
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT id, topic, detected_at, detection_source,
                          related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
                   FROM knowledge_gaps
                   WHERE brain_id = {d.ph(1)} AND resolved_at IS NULL
                   ORDER BY priority DESC LIMIT {d.ph(2)}""",
                (brain_id, capped),
            )

        results = []
        for r in rows:
            entry = dict(r)
            entry["related_neuron_ids"] = _parse_related_neuron_ids(entry.get("related_neuron_ids"))
            results.append(entry)
        return results

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        """Mark a knowledge gap as resolved.

        Returns:
            True if the gap was found and resolved, False otherwise.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"""UPDATE knowledge_gaps SET
                 resolved_at = {d.ph(1)},
                 resolved_by_neuron_id = {d.ph(2)}
               WHERE brain_id = {d.ph(3)} AND id = {d.ph(4)} AND resolved_at IS NULL""",
            (d.serialize_dt(utcnow()), resolved_by_neuron_id, brain_id, gap_id),
        )
        return count > 0

    # ──────────────────── Schema Evolution ────────────────────

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Walk the version chain from a hypothesis back through parent_schema_id.

        Returns a list ordered newest-first, with the given neuron_id at index 0.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        history: list[dict[str, Any]] = []
        current_id: str | None = neuron_id
        seen: set[str] = set()

        while current_id and len(history) < max_depth:
            if current_id in seen:
                break  # Cycle guard
            seen.add(current_id)

            row = await d.fetch_one(
                f"""SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, schema_version, parent_schema_id, created_at
                   FROM cognitive_state
                   WHERE brain_id = {d.ph(1)} AND neuron_id = {d.ph(2)}""",
                (brain_id, current_id),
            )
            if not row:
                break
            entry = dict(row)
            history.append(entry)
            current_id = entry.get("parent_schema_id")

        return history
