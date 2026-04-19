"""Retrieval sufficiency calibration persistence mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from typing import Any

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth
_MAX_RECORDS_PER_BRAIN = 10_000


class CalibrationMixin:
    """Mixin providing CRUD for the retrieval_calibration table."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def save_calibration_record(
        self,
        gate: str,
        predicted_sufficient: bool,
        actual_confidence: float,
        actual_fibers: int,
        query_intent: str = "",
        metrics_json: dict[str, Any] | None = None,
    ) -> None:
        """Insert a calibration feedback record."""
        d = self._dialect
        brain_id = self._get_brain_id()

        await d.execute(
            f"""INSERT INTO retrieval_calibration
                (brain_id, gate, predicted_sufficient, actual_confidence,
                 actual_fibers, query_intent, metrics_json, created_at)
                VALUES ({d.phs(8)})""",
            [
                brain_id,
                gate,
                1 if predicted_sufficient else 0,
                actual_confidence,
                actual_fibers,
                query_intent,
                json.dumps(metrics_json or {}),
                d.serialize_dt(utcnow()),
            ],
        )

    async def get_recent_calibration(
        self,
        gate: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch recent calibration records, optionally filtered by gate."""
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if gate:
            rows = await d.fetch_all(
                f"""SELECT * FROM retrieval_calibration
                    WHERE brain_id = {d.ph(1)} AND gate = {d.ph(2)}
                    ORDER BY created_at DESC LIMIT {d.ph(3)}""",
                [brain_id, gate, capped_limit],
            )
        else:
            rows = await d.fetch_all(
                f"""SELECT * FROM retrieval_calibration
                    WHERE brain_id = {d.ph(1)}
                    ORDER BY created_at DESC LIMIT {d.ph(2)}""",
                [brain_id, capped_limit],
            )
        return rows

    async def prune_old_calibration(self, keep_days: int = 90) -> int:
        """Delete calibration records older than keep_days. Returns count deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        from datetime import timedelta

        cutoff = d.serialize_dt(utcnow() - timedelta(days=keep_days))

        return await d.execute_count(
            f"DELETE FROM retrieval_calibration WHERE brain_id = {d.ph(1)} AND created_at < {d.ph(2)}",
            [brain_id, cutoff],
        )

    async def cap_calibration_records(self) -> int:
        """Enforce max record limit per brain. Returns count deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT COUNT(*) as cnt FROM retrieval_calibration WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )
        count = row.get("cnt", 0) if row else 0

        if count <= _MAX_RECORDS_PER_BRAIN:
            return 0

        excess = count - _MAX_RECORDS_PER_BRAIN
        return await d.execute_count(
            f"""DELETE FROM retrieval_calibration WHERE id IN (
                SELECT id FROM retrieval_calibration
                WHERE brain_id = {d.ph(1)}
                ORDER BY created_at ASC LIMIT {d.ph(2)}
            )""",
            [brain_id, excess],
        )

    async def get_gate_ema_stats(
        self,
        window: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Compute EMA accuracy stats per gate over recent records.

        Returns a dict keyed by gate name, each containing:
        - accuracy: EMA of correct predictions (predicted_sufficient matches
          actual_confidence >= 0.3 as true positive threshold)
        - avg_confidence: EMA of actual_confidence for that gate
        - sample_count: number of records used

        EMA decays older records toward the tail (most recent data weighted
        highest). Alpha = 2 / (window + 1) per standard EMA convention.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        capped_window = min(window, 500)

        rows = await d.fetch_all(
            f"""SELECT gate, predicted_sufficient, actual_confidence
                FROM retrieval_calibration
                WHERE brain_id = {d.ph(1)}
                ORDER BY created_at DESC
                LIMIT {d.ph(2)}""",
            [brain_id, capped_window * 20],  # fetch more to group by gate
        )

        # Group rows by gate (rows are DESC order — most recent first)
        gate_rows: dict[str, list[tuple[int, float]]] = {}
        for r in rows:
            gate = r["gate"]
            if gate not in gate_rows:
                gate_rows[gate] = []
            gate_rows[gate].append((r["predicted_sufficient"], r["actual_confidence"]))

        result: dict[str, dict[str, float]] = {}
        alpha = 2.0 / (capped_window + 1)

        for gate, gate_data in gate_rows.items():
            # Take at most capped_window records per gate (most recent first)
            gate_data = gate_data[:capped_window]
            sample_count = len(gate_data)

            if sample_count == 0:
                continue

            # Compute EMA on reversed list (oldest first for forward EMA)
            oldest_to_newest = list(reversed(gate_data))

            def _is_correct(predicted: int, actual_conf: float) -> float:
                """Return 1.0 if prediction matches actual outcome, else 0.0."""
                actual_sufficient = actual_conf >= 0.3
                return float(int(bool(predicted)) == int(actual_sufficient))

            first_predicted, first_conf = oldest_to_newest[0]
            ema_accuracy = _is_correct(first_predicted, first_conf)
            ema_confidence = first_conf

            for predicted, actual_conf in oldest_to_newest[1:]:
                correct = _is_correct(predicted, actual_conf)
                ema_accuracy = alpha * correct + (1.0 - alpha) * ema_accuracy
                ema_confidence = alpha * actual_conf + (1.0 - alpha) * ema_confidence

            result[gate] = {
                "accuracy": round(max(0.0, min(1.0, ema_accuracy)), 4),
                "avg_confidence": round(max(0.0, min(1.0, ema_confidence)), 4),
                "sample_count": float(sample_count),
            }

        return result

    # ------------------------------------------------------------------
    # Retriever calibration: per-brain EMA weights for RRF
    # ------------------------------------------------------------------

    async def save_retriever_outcome(
        self,
        retriever_type: str,
        contributed: bool,
    ) -> None:
        """Record whether a retriever type contributed to a successful recall.

        Args:
            retriever_type: One of "time", "entity", "keyword", "embedding", "graph_expansion".
            contributed: True if this retriever had neurons in the final result.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        await d.execute(
            f"""INSERT INTO retriever_calibration
                (brain_id, retriever_type, contributed, created_at)
                VALUES ({d.phs(4)})""",
            [brain_id, retriever_type, 1 if contributed else 0, now],
        )

    async def get_retriever_weights(
        self,
        window: int = 100,
    ) -> dict[str, float]:
        """Compute per-retriever success EMA weights.

        Returns dict of {retriever_type: weight} where weight is EMA of
        contribution rate. Higher weight = retriever contributes more often
        to successful recalls for this brain.

        Default weights (from score_fusion) are returned for retrievers
        with no data.
        """
        from neural_memory.engine.score_fusion import DEFAULT_RETRIEVER_WEIGHTS

        d = self._dialect
        brain_id = self._get_brain_id()
        capped = min(window, 500)

        rows = await d.fetch_all(
            f"""SELECT retriever_type, contributed
                FROM retriever_calibration
                WHERE brain_id = {d.ph(1)}
                ORDER BY created_at DESC
                LIMIT {d.ph(2)}""",
            [brain_id, capped * 10],
        )

        if not rows:
            return dict(DEFAULT_RETRIEVER_WEIGHTS)

        # Group by retriever type
        by_type: dict[str, list[int]] = {}
        for r in rows:
            rtype = r["retriever_type"]
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(r["contributed"])

        alpha = 2.0 / (capped + 1)  # EMA alpha: 2/(window+1) per standard convention
        result = dict(DEFAULT_RETRIEVER_WEIGHTS)

        for rtype, data in by_type.items():
            samples = data[:capped]
            if not samples:
                continue

            # Compute EMA (oldest first)
            oldest_first = list(reversed(samples))
            ema = float(oldest_first[0])
            for val in oldest_first[1:]:
                ema = alpha * float(val) + (1.0 - alpha) * ema

            # Blend with default weight (never go below 0.1 or above 2.0)
            default_w = DEFAULT_RETRIEVER_WEIGHTS.get(rtype, 0.5)
            # Weight = default * (0.5 + ema). Range: default*0.5 to default*1.5
            adjusted = default_w * (0.5 + ema)
            result[rtype] = round(max(0.1, min(2.0, adjusted)), 4)

        return result

    async def prune_retriever_calibration(self, keep_per_type: int = 500) -> int:
        """Cap retriever_calibration rows per brain per type. Returns count deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        over_limit = await d.fetch_all(
            f"""SELECT retriever_type, COUNT(*) as cnt
                FROM retriever_calibration
                WHERE brain_id = {d.ph(1)}
                GROUP BY retriever_type
                HAVING cnt > {d.ph(2)}""",
            [brain_id, keep_per_type],
        )

        total_deleted = 0
        for row in over_limit:
            rtype = row["retriever_type"]
            cnt = row["cnt"]
            excess = cnt - keep_per_type
            deleted = await d.execute_count(
                f"""DELETE FROM retriever_calibration WHERE id IN (
                    SELECT id FROM retriever_calibration
                    WHERE brain_id = {d.ph(1)} AND retriever_type = {d.ph(2)}
                    ORDER BY created_at ASC LIMIT {d.ph(3)}
                )""",
                [brain_id, rtype, excess],
            )
            total_deleted += deleted

        return total_deleted

    # ------------------------------------------------------------------
    # Graph density: avg synapses per neuron for strategy auto-selection
    # ------------------------------------------------------------------

    async def get_graph_density(self, exclude_hubs: bool = False) -> float:
        """Compute average synapses per neuron for the current brain.

        Args:
            exclude_hubs: When True, filter out synapses whose metadata
                contains ``_hub`` (DREAM-generated hub links). This yields
                the *organic* graph density, unaffected by consolidation
                artefacts. Used by retrieval engine to pick an activation
                strategy that reflects the user's real graph.

        Returns 0.0 if no neurons exist.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT COUNT(*) as cnt FROM neurons WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )
        neuron_count = row.get("cnt", 0) if row else 0
        if neuron_count == 0:
            return 0.0

        if exclude_hubs:
            hub_filter = f" AND {d.json_extract('metadata', '_hub')} IS NULL"
        else:
            hub_filter = ""
        row = await d.fetch_one(
            f"SELECT COUNT(*) as cnt FROM synapses WHERE brain_id = {d.ph(1)}{hub_filter}",
            [brain_id],
        )
        synapse_count = row.get("cnt", 0) if row else 0

        return synapse_count / neuron_count
