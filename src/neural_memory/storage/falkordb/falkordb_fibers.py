"""FalkorDB fiber CRUD operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from neural_memory.core.fiber import Fiber
from neural_memory.storage.falkordb.falkordb_base import FalkorDBBaseMixin


class FalkorDBFiberMixin(FalkorDBBaseMixin):
    """FalkorDB implementation of fiber CRUD operations.

    Fibers are stored as :Fiber nodes with [:CONTAINS {order}] edges
    to their constituent :Neuron nodes. This allows efficient graph
    traversal to find fibers containing specific neurons.
    """

    async def add_fiber(self, fiber: Fiber) -> str:
        existing = await self._query_ro(
            "MATCH (f:Fiber {id: $id}) RETURN f.id",
            {"id": fiber.id},
        )
        if existing:
            raise ValueError(f"Fiber {fiber.id} already exists")

        await self._query(
            """
            CREATE (f:Fiber {
                id: $id,
                anchor_neuron_id: $anchor,
                pathway: $pathway,
                conductivity: $conductivity,
                last_conducted: $last_conducted,
                time_start: $time_start,
                time_end: $time_end,
                coherence: $coherence,
                salience: $salience,
                frequency: $frequency,
                summary: $summary,
                auto_tags: $auto_tags,
                agent_tags: $agent_tags,
                metadata: $metadata,
                compression_tier: $compression_tier,
                created_at: $created_at,
                neuron_ids: $neuron_ids,
                synapse_ids: $synapse_ids
            })
            """,
            {
                "id": fiber.id,
                "anchor": fiber.anchor_neuron_id,
                "pathway": json.dumps(fiber.pathway),
                "conductivity": fiber.conductivity,
                "last_conducted": self._dt_to_str(fiber.last_conducted),
                "time_start": self._dt_to_str(fiber.time_start),
                "time_end": self._dt_to_str(fiber.time_end),
                "coherence": fiber.coherence,
                "salience": fiber.salience,
                "frequency": fiber.frequency,
                "summary": fiber.summary,
                "auto_tags": ",".join(sorted(fiber.auto_tags)),
                "agent_tags": ",".join(sorted(fiber.agent_tags)),
                "metadata": self._serialize_metadata(fiber.metadata),
                "compression_tier": fiber.compression_tier,
                "created_at": self._dt_to_str(fiber.created_at),
                "neuron_ids": json.dumps(sorted(fiber.neuron_ids)),
                "synapse_ids": json.dumps(sorted(fiber.synapse_ids)),
            },
        )

        # Create [:CONTAINS] edges to neurons for graph traversal
        for order, nid in enumerate(fiber.pathway):
            await self._query(
                """
                MATCH (f:Fiber {id: $fid}), (n:Neuron {id: $nid})
                CREATE (f)-[:CONTAINS {order: $order}]->(n)
                """,
                {"fid": fiber.id, "nid": nid, "order": order},
            )

        return fiber.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        rows = await self._query_ro(
            """
            MATCH (f:Fiber {id: $id})
            RETURN f.id, f.anchor_neuron_id, f.pathway, f.conductivity,
                   f.last_conducted, f.time_start, f.time_end,
                   f.coherence, f.salience, f.frequency, f.summary,
                   f.auto_tags, f.agent_tags, f.metadata,
                   f.compression_tier, f.created_at,
                   f.neuron_ids, f.synapse_ids
            """,
            {"id": fiber_id},
        )
        if not rows:
            return None
        return self._row_to_fiber(rows[0])

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
        tag_mode: str = "and",
    ) -> list[Fiber]:
        limit = min(limit, 1000)

        if contains_neuron:
            # Use [:CONTAINS] edge for efficient lookup
            rows = await self._query_ro(
                """
                MATCH (f:Fiber)-[:CONTAINS]->(n:Neuron {id: $nid})
                RETURN DISTINCT f.id, f.anchor_neuron_id, f.pathway,
                       f.conductivity, f.last_conducted, f.time_start,
                       f.time_end, f.coherence, f.salience, f.frequency,
                       f.summary, f.auto_tags, f.agent_tags, f.metadata,
                       f.compression_tier, f.created_at,
                       f.neuron_ids, f.synapse_ids
                ORDER BY f.salience DESC
                LIMIT $limit
                """,
                {"nid": contains_neuron, "limit": limit},
            )
        else:
            conditions: list[str] = []
            params: dict[str, Any] = {"limit": limit}

            if min_salience is not None:
                conditions.append("f.salience >= $min_sal")
                params["min_sal"] = min_salience
            if time_overlaps is not None:
                conditions.append("f.time_start <= $t_end AND f.time_end >= $t_start")
                params["t_start"] = self._dt_to_str(time_overlaps[0])
                params["t_end"] = self._dt_to_str(time_overlaps[1])

            where = " AND ".join(conditions) if conditions else "true"
            rows = await self._query_ro(
                f"""
                MATCH (f:Fiber)
                WHERE {where}
                RETURN f.id, f.anchor_neuron_id, f.pathway, f.conductivity,
                       f.last_conducted, f.time_start, f.time_end,
                       f.coherence, f.salience, f.frequency, f.summary,
                       f.auto_tags, f.agent_tags, f.metadata,
                       f.compression_tier, f.created_at,
                       f.neuron_ids, f.synapse_ids
                ORDER BY f.salience DESC
                LIMIT $limit
                """,
                params,
            )

        fibers = [self._row_to_fiber(r) for r in rows]

        # Post-filter tags (FalkorDB stores as CSV, complex matching in Cypher)
        if tags:
            if tag_mode == "or":
                fibers = [f for f in fibers if tags & f.tags]
            else:
                fibers = [f for f in fibers if tags <= f.tags]

        # Post-filter metadata key (FalkorDB metadata stored as JSON string)
        if metadata_key is not None:
            fibers = [f for f in fibers if f.metadata.get(metadata_key) is not None]

        return fibers[:limit]

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
        tags: set[str] | None = None,
        tag_mode: str = "and",
    ) -> list[Fiber]:
        if not neuron_ids:
            return []
        limit_per_neuron = min(limit_per_neuron, 100)
        max_results = min(len(neuron_ids) * limit_per_neuron, 1000)
        # Per-neuron limit: collect fibers per neuron, slice, then flatten
        rows = await self._query_ro(
            """
            UNWIND $nids AS nid
            MATCH (f:Fiber)-[:CONTAINS]->(n:Neuron {id: nid})
            WITH nid, f
            ORDER BY f.salience DESC
            WITH nid, collect(DISTINCT f)[0..$lpn] AS top_fibers
            UNWIND top_fibers AS f
            RETURN DISTINCT f.id, f.anchor_neuron_id, f.pathway,
                   f.conductivity, f.last_conducted, f.time_start,
                   f.time_end, f.coherence, f.salience, f.frequency,
                   f.summary, f.auto_tags, f.agent_tags, f.metadata,
                   f.compression_tier, f.created_at,
                   f.neuron_ids, f.synapse_ids
            ORDER BY f.salience DESC
            LIMIT $limit
            """,
            {"nids": neuron_ids, "lpn": limit_per_neuron, "limit": max_results},
        )
        fibers = [self._row_to_fiber(r) for r in rows]
        # fiber.tags property = auto_tags | agent_tags (union)
        if tags:
            if tag_mode == "or":
                fibers = [f for f in fibers if tags & f.tags]
            else:
                fibers = [f for f in fibers if tags.issubset(f.tags)]
        return fibers

    async def update_fiber(self, fiber: Fiber) -> None:
        rows = await self._query(
            """
            MATCH (f:Fiber {id: $id})
            SET f.anchor_neuron_id = $anchor,
                f.pathway = $pathway,
                f.conductivity = $conductivity,
                f.last_conducted = $last_conducted,
                f.time_start = $time_start,
                f.time_end = $time_end,
                f.coherence = $coherence,
                f.salience = $salience,
                f.frequency = $frequency,
                f.summary = $summary,
                f.auto_tags = $auto_tags,
                f.agent_tags = $agent_tags,
                f.metadata = $metadata,
                f.compression_tier = $compression_tier,
                f.neuron_ids = $neuron_ids,
                f.synapse_ids = $synapse_ids
            RETURN f.id
            """,
            {
                "id": fiber.id,
                "anchor": fiber.anchor_neuron_id,
                "pathway": json.dumps(fiber.pathway),
                "conductivity": fiber.conductivity,
                "last_conducted": self._dt_to_str(fiber.last_conducted),
                "time_start": self._dt_to_str(fiber.time_start),
                "time_end": self._dt_to_str(fiber.time_end),
                "coherence": fiber.coherence,
                "salience": fiber.salience,
                "frequency": fiber.frequency,
                "summary": fiber.summary,
                "auto_tags": ",".join(sorted(fiber.auto_tags)),
                "agent_tags": ",".join(sorted(fiber.agent_tags)),
                "metadata": self._serialize_metadata(fiber.metadata),
                "compression_tier": fiber.compression_tier,
                "neuron_ids": json.dumps(sorted(fiber.neuron_ids)),
                "synapse_ids": json.dumps(sorted(fiber.synapse_ids)),
            },
        )
        if not rows:
            raise ValueError(f"Fiber {fiber.id} not found")

        # Rebuild [:CONTAINS] edges
        await self._query(
            "MATCH (f:Fiber {id: $id})-[c:CONTAINS]->() DELETE c",
            {"id": fiber.id},
        )
        for order, nid in enumerate(fiber.pathway):
            await self._query(
                """
                MATCH (f:Fiber {id: $fid}), (n:Neuron {id: $nid})
                CREATE (f)-[:CONTAINS {order: $order}]->(n)
                """,
                {"fid": fiber.id, "nid": nid, "order": order},
            )

    async def delete_fiber(self, fiber_id: str) -> bool:
        existing = await self._query_ro(
            "MATCH (f:Fiber {id: $id}) RETURN f.id",
            {"id": fiber_id},
        )
        if not existing:
            return False
        await self._query(
            "MATCH (f:Fiber {id: $id}) DETACH DELETE f",
            {"id": fiber_id},
        )
        return True

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        limit = min(limit, 1000)
        direction = "DESC" if descending else "ASC"
        order_field = f"f.{order_by}"  # order_by is Literal-typed, safe

        rows = await self._query_ro(
            f"""
            MATCH (f:Fiber)
            RETURN f.id, f.anchor_neuron_id, f.pathway, f.conductivity,
                   f.last_conducted, f.time_start, f.time_end,
                   f.coherence, f.salience, f.frequency, f.summary,
                   f.auto_tags, f.agent_tags, f.metadata,
                   f.compression_tier, f.created_at,
                   f.neuron_ids, f.synapse_ids
            ORDER BY {order_field} {direction}
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [self._row_to_fiber(r) for r in rows]

    # ========== Row Mapper ==========

    def _row_to_fiber(self, row: list[Any]) -> Fiber:
        """Convert FalkorDB result row to Fiber dataclass.

        Expected columns: id, anchor_neuron_id, pathway, conductivity,
            last_conducted, time_start, time_end, coherence, salience,
            frequency, summary, auto_tags, agent_tags, metadata,
            compression_tier, created_at, neuron_ids, synapse_ids
        """
        pathway_raw = row[2]
        if isinstance(pathway_raw, str):
            pathway = json.loads(pathway_raw) if pathway_raw else []
        else:
            pathway = list(pathway_raw) if pathway_raw else []

        auto_tags_raw = row[11] or ""
        agent_tags_raw = row[12] or ""

        neuron_ids_raw = row[16]
        if isinstance(neuron_ids_raw, str):
            neuron_ids = set(json.loads(neuron_ids_raw)) if neuron_ids_raw else set()
        else:
            neuron_ids = set(neuron_ids_raw) if neuron_ids_raw else set()

        synapse_ids_raw = row[17]
        if isinstance(synapse_ids_raw, str):
            synapse_ids = set(json.loads(synapse_ids_raw)) if synapse_ids_raw else set()
        else:
            synapse_ids = set(synapse_ids_raw) if synapse_ids_raw else set()

        return Fiber(
            id=row[0],
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=row[1],
            pathway=pathway,
            conductivity=row[3] if row[3] is not None else 1.0,
            last_conducted=self._str_to_dt(row[4]),
            time_start=self._str_to_dt(row[5]),
            time_end=self._str_to_dt(row[6]),
            coherence=row[7] if row[7] is not None else 0.0,
            salience=row[8] if row[8] is not None else 0.0,
            frequency=row[9] if row[9] is not None else 0,
            summary=row[10],
            auto_tags={t for t in auto_tags_raw.split(",") if t},
            agent_tags={t for t in agent_tags_raw.split(",") if t},
            metadata=self._deserialize_metadata(row[13]),
            compression_tier=row[14] if row[14] is not None else 0,
            created_at=self._str_to_dt(row[15]) or datetime.min,
        )
