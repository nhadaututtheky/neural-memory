"""PostgreSQL neuron operations with tsvector full-text and pgvector similarity."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import (
    row_to_neuron,
    row_to_neuron_state,
)

logger = logging.getLogger(__name__)


class PostgresNeuronMixin(PostgresBaseMixin):
    """PostgreSQL neuron CRUD with full-text (tsvector) and vector (pgvector) search."""

    async def add_neuron(self, neuron: Neuron) -> str:
        brain_id = self._get_brain_id()
        existing = await self._query_one(
            "SELECT id FROM neurons WHERE brain_id = $1 AND id = $2",
            brain_id,
            neuron.id,
        )
        if existing:
            raise ValueError(f"Neuron {neuron.id} already exists")

        embedding = neuron.metadata.get("_embedding")
        meta = {k: v for k, v in neuron.metadata.items() if k != "_embedding"}
        meta_json = json.dumps(meta)
        created = neuron.created_at

        await self._query(
            """INSERT INTO neurons
               (id, brain_id, type, content, metadata, content_hash, created_at, embedding)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            neuron.id,
            brain_id,
            neuron.type.value,
            neuron.content,
            meta_json,
            neuron.content_hash,
            created,
            embedding,
        )

        # Init neuron state
        await self._query(
            """INSERT INTO neuron_states
               (neuron_id, brain_id, activation_level, access_frequency, created_at)
               VALUES ($1, $2, 0.0, 0, $3)""",
            neuron.id,
            brain_id,
            created,
        )
        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM neurons WHERE brain_id = $1 AND id = $2",
            brain_id,
            neuron_id,
        )
        if row is None:
            return None
        return row_to_neuron(row)

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        if not neuron_ids:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            "SELECT * FROM neurons WHERE brain_id = $1 AND id = ANY($2::text[])",
            brain_id,
            neuron_ids,
        )
        return {str(r["id"]): row_to_neuron(r) for r in rows}

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Neuron]:
        brain_id = self._get_brain_id()
        full_scan = content_contains is None and content_exact is None
        limit = min(limit, 10000 if full_scan else 1000)

        if content_exact is not None:
            row = await self._query_one(
                "SELECT * FROM neurons WHERE brain_id = $1 AND content = $2",
                brain_id,
                content_exact,
            )
            if row is None:
                return []
            if type is not None and row["type"] != type.value:
                return []
            return [row_to_neuron(row)]

        query = "SELECT * FROM neurons WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if type is not None:
            query += " AND type = $2"
            params.append(type.value)

        if content_contains is not None:
            safe_term = content_contains.replace("&", " ").replace("|", " ").strip()
            if safe_term:
                idx = len(params) + 1
                query += f" AND content_tsv @@ plainto_tsquery('english', ${idx})"
                params.append(safe_term)

        if time_range is not None:
            start, end = time_range
            idx = len(params) + 1
            query += f" AND created_at >= ${idx} AND created_at <= ${idx + 1}"
            params.extend([start, end])

        query += " ORDER BY id LIMIT $" + str(len(params) + 1) + " OFFSET $" + str(len(params) + 2)
        params.extend([limit, offset])

        rows = await self._query_ro(query, *params)
        return [row_to_neuron(r) for r in rows]

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        limit = min(limit, 20)
        query = """
            SELECT n.id, n.content, n.type,
                   COALESCE(ns.access_frequency, 0) AS access_frequency,
                   COALESCE(ns.activation_level, 0.0) AS activation_level
            FROM neurons n
            LEFT JOIN neuron_states ns ON n.brain_id = ns.brain_id AND n.id = ns.neuron_id
            WHERE n.brain_id = $1 AND n.content ILIKE $2
        """
        params: list[Any] = [brain_id, f"{prefix}%"]
        if type_filter is not None:
            query += " AND n.type = $3"
            params.append(type_filter.value)
        query += " ORDER BY ns.access_frequency DESC NULLS LAST LIMIT $" + str(len(params) + 1)
        params.append(limit)
        rows = await self._query_ro(query, *params)
        return [
            {
                "neuron_id": str(r["id"]),
                "content": str(r["content"]),
                "type": str(r["type"]),
                "access_frequency": int(r["access_frequency"] or 0),
                "activation_level": float(r["activation_level"] or 0.0),
                "score": float(r["activation_level"] or 0.0) * 0.5
                + int(r["access_frequency"] or 0) * 0.1,
            }
            for r in rows
        ]

    async def update_neuron(self, neuron: Neuron) -> None:
        brain_id = self._get_brain_id()
        embedding = neuron.metadata.get("_embedding")
        meta = {k: v for k, v in neuron.metadata.items() if k != "_embedding"}
        r = await self._query(
            """UPDATE neurons SET type = $1, content = $2, metadata = $3, content_hash = $4,
               embedding = $5
               WHERE brain_id = $6 AND id = $7""",
            neuron.type.value,
            neuron.content,
            json.dumps(meta),
            neuron.content_hash,
            embedding,
            brain_id,
            neuron.id,
        )
        if r == "UPDATE 0":
            raise ValueError(f"Neuron {neuron.id} does not exist")

    async def delete_neuron(self, neuron_id: str) -> bool:
        brain_id = self._get_brain_id()
        r = await self._query(
            "DELETE FROM neurons WHERE brain_id = $1 AND id = $2",
            brain_id,
            neuron_id,
        )
        return bool(r != "DELETE 0")

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM neuron_states WHERE brain_id = $1 AND neuron_id = $2",
            brain_id,
            neuron_id,
        )
        if row is None:
            return None
        return row_to_neuron_state(row)

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Batch fetch neuron states in a single query."""
        if not neuron_ids:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            "SELECT * FROM neuron_states WHERE brain_id = $1 AND neuron_id = ANY($2::text[])",
            brain_id,
            neuron_ids,
        )
        return {str(r["neuron_id"]): row_to_neuron_state(r) for r in rows}

    async def get_all_neuron_states(self) -> list[NeuronState]:
        """Get all neuron states for current brain."""
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            "SELECT * FROM neuron_states WHERE brain_id = $1 LIMIT 10000",
            brain_id,
        )
        return [row_to_neuron_state(r) for r in rows]

    async def update_neuron_state(self, state: NeuronState) -> None:
        brain_id = self._get_brain_id()
        await self._query(
            """INSERT INTO neuron_states
               (neuron_id, brain_id, activation_level, access_frequency,
                last_activated, decay_rate, firing_threshold, refractory_until,
                refractory_period_ms, homeostatic_target, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
               ON CONFLICT (brain_id, neuron_id) DO UPDATE SET
                 activation_level = EXCLUDED.activation_level,
                 access_frequency = EXCLUDED.access_frequency,
                 last_activated = EXCLUDED.last_activated,
                 decay_rate = EXCLUDED.decay_rate,
                 firing_threshold = EXCLUDED.firing_threshold,
                 refractory_until = EXCLUDED.refractory_until,
                 refractory_period_ms = EXCLUDED.refractory_period_ms,
                 homeostatic_target = EXCLUDED.homeostatic_target""",
            state.neuron_id,
            brain_id,
            state.activation_level,
            state.access_frequency,
            state.last_activated,
            state.decay_rate,
            state.firing_threshold,
            state.refractory_until,
            state.refractory_period_ms,
            state.homeostatic_target,
            state.created_at,
        )

    async def update_neuron_states_batch(self, states: list[NeuronState]) -> None:
        """Update multiple neuron states in one batch."""
        if not states:
            return
        brain_id = self._get_brain_id()
        args_list = [
            (
                s.neuron_id,
                brain_id,
                s.activation_level,
                s.access_frequency,
                s.last_activated,
                s.decay_rate,
                s.firing_threshold,
                s.refractory_until,
                s.refractory_period_ms,
                s.homeostatic_target,
                s.created_at,
            )
            for s in states
        ]
        await self._executemany(
            """INSERT INTO neuron_states
               (neuron_id, brain_id, activation_level, access_frequency,
                last_activated, decay_rate, firing_threshold, refractory_until,
                refractory_period_ms, homeostatic_target, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
               ON CONFLICT (brain_id, neuron_id) DO UPDATE SET
                 activation_level = EXCLUDED.activation_level,
                 access_frequency = EXCLUDED.access_frequency,
                 last_activated = EXCLUDED.last_activated,
                 decay_rate = EXCLUDED.decay_rate,
                 firing_threshold = EXCLUDED.firing_threshold,
                 refractory_until = EXCLUDED.refractory_until,
                 refractory_period_ms = EXCLUDED.refractory_period_ms,
                 homeostatic_target = EXCLUDED.homeostatic_target""",
            args_list,
        )

    async def find_neurons_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        type_filter: NeuronType | None = None,
    ) -> list[tuple[Neuron, float]]:
        """Find neurons by vector similarity (pgvector cosine). Returns (neuron, score)."""
        brain_id = self._get_brain_id()
        limit = min(limit, 100)
        try:
            if type_filter is not None:
                rows = await self._query_ro(
                    """SELECT n.*, 1 - (n.embedding <=> $1::vector) AS score
                       FROM neurons n
                       WHERE n.brain_id = $2 AND n.embedding IS NOT NULL AND n.type = $3
                       ORDER BY n.embedding <=> $1::vector
                       LIMIT $4""",
                    query_embedding,
                    brain_id,
                    type_filter.value,
                    limit,
                )
            else:
                rows = await self._query_ro(
                    """SELECT n.*, 1 - (n.embedding <=> $1::vector) AS score
                       FROM neurons n
                       WHERE n.brain_id = $2 AND n.embedding IS NOT NULL
                       ORDER BY n.embedding <=> $1::vector
                       LIMIT $3""",
                    query_embedding,
                    brain_id,
                    limit,
                )
            return [(row_to_neuron(r), float(r["score"])) for r in rows]
        except Exception:
            logger.error("Embedding similarity search failed", exc_info=True)
            return []
