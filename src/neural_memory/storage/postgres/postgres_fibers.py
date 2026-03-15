"""PostgreSQL fiber operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import row_to_fiber


class PostgresFiberMixin(PostgresBaseMixin):
    """PostgreSQL fiber CRUD."""

    async def add_fiber(self, fiber: Fiber) -> str:
        brain_id = self._get_brain_id()
        try:
            await self._query(
                """INSERT INTO fibers
                   (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                    pathway, conductivity, last_conducted, time_start, time_end,
                    coherence, salience, frequency, summary, tags, auto_tags,
                    agent_tags, metadata, compression_tier, pinned, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                           $15, $16, $17, $18, $19, $20, $21)""",
                fiber.id,
                brain_id,
                json.dumps(list(fiber.neuron_ids)),
                json.dumps(list(fiber.synapse_ids)),
                fiber.anchor_neuron_id,
                json.dumps(fiber.pathway),
                fiber.conductivity,
                fiber.last_conducted,
                fiber.time_start,
                fiber.time_end,
                fiber.coherence,
                fiber.salience,
                fiber.frequency,
                fiber.summary,
                json.dumps(list(fiber.tags)),
                json.dumps(list(fiber.auto_tags)),
                json.dumps(list(fiber.agent_tags)),
                json.dumps(fiber.metadata),
                fiber.compression_tier,
                1 if fiber.pinned else 0,
                fiber.created_at,
            )
            if fiber.neuron_ids:
                args_list = [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids]
                await self._executemany(
                    """INSERT INTO fiber_neurons (brain_id, fiber_id, neuron_id)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (brain_id, fiber_id, neuron_id) DO NOTHING""",
                    args_list,
                )
            return fiber.id
        except Exception as e:
            from asyncpg.exceptions import UniqueViolationError

            if isinstance(e, UniqueViolationError):
                raise ValueError(f"Fiber {fiber.id} already exists") from e
            raise

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM fibers WHERE brain_id = $1 AND id = $2",
            brain_id,
            fiber_id,
        )
        if row is None:
            return None
        return row_to_fiber(row)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        query = "SELECT * FROM fibers WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if contains_neuron is not None:
            params.append(contains_neuron)
            query += f" AND id IN (SELECT fiber_id FROM fiber_neurons WHERE brain_id = $1 AND neuron_id = ${len(params)})"  # noqa: S608

        if time_overlaps is not None:
            start, end = time_overlaps
            idx = len(params) + 1
            query += f" AND (time_start IS NULL OR time_start <= ${idx + 1})"
            query += f" AND (time_end IS NULL OR time_end >= ${idx})"
            params.extend([end, start])

        if min_salience is not None:
            params.append(min_salience)
            query += f" AND salience >= ${len(params)}"

        if metadata_key is not None:
            params.append(metadata_key)
            query += f" AND metadata ? ${len(params)}"

        if tags is not None and tags:
            params.append(json.dumps(list(tags)))
            query += f" AND tags @> ${len(params)}::jsonb"

        params.append(limit)
        query += f" ORDER BY salience DESC LIMIT ${len(params)}"

        rows = await self._query_ro(query, *params)
        return [row_to_fiber(r) for r in rows]

    async def update_fiber(self, fiber: Fiber) -> None:
        brain_id = self._get_brain_id()
        r = await self._query(
            """UPDATE fibers SET
               neuron_ids = $1, synapse_ids = $2, anchor_neuron_id = $3,
               pathway = $4, conductivity = $5, last_conducted = $6,
               time_start = $7, time_end = $8, coherence = $9, salience = $10,
               frequency = $11, summary = $12, tags = $13, auto_tags = $14,
               agent_tags = $15, metadata = $16, compression_tier = $17,
               pinned = $18
               WHERE brain_id = $19 AND id = $20""",
            json.dumps(list(fiber.neuron_ids)),
            json.dumps(list(fiber.synapse_ids)),
            fiber.anchor_neuron_id,
            json.dumps(fiber.pathway),
            fiber.conductivity,
            fiber.last_conducted,
            fiber.time_start,
            fiber.time_end,
            fiber.coherence,
            fiber.salience,
            fiber.frequency,
            fiber.summary,
            json.dumps(list(fiber.tags)),
            json.dumps(list(fiber.auto_tags)),
            json.dumps(list(fiber.agent_tags)),
            json.dumps(fiber.metadata),
            fiber.compression_tier,
            1 if fiber.pinned else 0,
            brain_id,
            fiber.id,
        )
        if r == "UPDATE 0":
            raise ValueError(f"Fiber {fiber.id} does not exist")

    async def delete_fiber(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()
        r = await self._query(
            "DELETE FROM fibers WHERE brain_id = $1 AND id = $2",
            brain_id,
            fiber_id,
        )
        return bool(r != "DELETE 0")

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        """Get fibers with ordering."""
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        _allowed_order = {"created_at", "salience", "frequency"}
        if order_by not in _allowed_order:
            order_by = "created_at"
        order_dir = "DESC" if descending else "ASC"
        query = f"SELECT * FROM fibers WHERE brain_id = $1 ORDER BY {order_by} {order_dir} LIMIT $2"  # noqa: S608
        rows = await self._query_ro(query, brain_id, limit)
        return [row_to_fiber(r) for r in rows]
