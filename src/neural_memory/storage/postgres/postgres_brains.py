"""PostgreSQL brain operations."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import row_to_brain
from neural_memory.utils.timeutils import utcnow

_MAX_EXPORT_NEURONS = 50_000
_MAX_EXPORT_SYNAPSES = 100_000
_MAX_EXPORT_FIBERS = 50_000


class PostgresBrainMixin(PostgresBaseMixin):
    """PostgreSQL brain CRUD."""

    async def save_brain(self, brain: Brain) -> None:
        config_dict = asdict(brain.config)
        config_json = json.dumps({k: v for k, v in config_dict.items() if v is not None})
        shared_json = json.dumps(brain.shared_with)
        now = utcnow()
        await self._query(
            """INSERT INTO brains
               (id, name, config, owner_id, is_public, shared_with, created_at, updated_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               ON CONFLICT (id) DO UPDATE SET
                 name = EXCLUDED.name,
                 config = EXCLUDED.config,
                 owner_id = EXCLUDED.owner_id,
                 is_public = EXCLUDED.is_public,
                 shared_with = EXCLUDED.shared_with,
                 updated_at = EXCLUDED.updated_at""",
            brain.id,
            brain.name,
            config_json,
            brain.owner_id,
            brain.is_public,
            shared_json,
            brain.created_at,
            now,
        )

    async def get_brain(self, brain_id: str) -> Brain | None:
        row = await self._query_one("SELECT * FROM brains WHERE id = $1", brain_id)
        if row is None:
            return None
        return row_to_brain(row)

    async def find_brain_by_name(self, name: str) -> Brain | None:
        row = await self._query_one("SELECT * FROM brains WHERE name = $1", name)
        if row is None:
            return None
        return row_to_brain(row)

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        neuron_rows = await self._query_ro(
            "SELECT id, type, content, metadata, content_hash, created_at "
            "FROM neurons WHERE brain_id = $1 LIMIT $2",
            brain_id,
            _MAX_EXPORT_NEURONS,
        )
        neurons: list[dict[str, Any]] = []
        for r in neuron_rows:
            neurons.append(
                {
                    "id": str(r["id"]),
                    "type": str(r["type"]),
                    "content": str(r["content"]),
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "content_hash": int(r["content_hash"] or 0),
                    "created_at": (
                        r["created_at"].isoformat()
                        if hasattr(r["created_at"], "isoformat")
                        else str(r["created_at"])
                    ),
                }
            )

        synapse_rows = await self._query_ro(
            "SELECT id, source_id, target_id, type, weight, direction, "
            "metadata, reinforced_count, last_activated, created_at "
            "FROM synapses WHERE brain_id = $1 LIMIT $2",
            brain_id,
            _MAX_EXPORT_SYNAPSES,
        )
        synapses: list[dict[str, Any]] = []
        for r in synapse_rows:
            la = r.get("last_activated")
            synapses.append(
                {
                    "id": str(r["id"]),
                    "source_id": str(r["source_id"]),
                    "target_id": str(r["target_id"]),
                    "type": str(r["type"]),
                    "weight": float(r["weight"]),
                    "direction": str(r.get("direction", "uni")),
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "reinforced_count": int(r.get("reinforced_count", 0)),
                    "last_activated": la.isoformat() if la and hasattr(la, "isoformat") else None,
                    "created_at": (
                        r["created_at"].isoformat()
                        if hasattr(r["created_at"], "isoformat")
                        else str(r["created_at"])
                    ),
                }
            )

        fiber_rows = await self._query_ro(
            "SELECT id, neuron_ids, synapse_ids, anchor_neuron_id, pathway, "
            "conductivity, last_conducted, time_start, time_end, coherence, "
            "salience, frequency, summary, tags, auto_tags, agent_tags, "
            "metadata, compression_tier, created_at "
            "FROM fibers WHERE brain_id = $1 LIMIT $2",
            brain_id,
            _MAX_EXPORT_FIBERS,
        )
        fibers: list[dict[str, Any]] = []
        for r in fiber_rows:

            def _ts(v: Any) -> str | None:
                if v is None:
                    return None
                return v.isoformat() if hasattr(v, "isoformat") else str(v)

            fibers.append(
                {
                    "id": str(r["id"]),
                    "neuron_ids": json.loads(r["neuron_ids"]) if r["neuron_ids"] else [],
                    "synapse_ids": json.loads(r["synapse_ids"]) if r["synapse_ids"] else [],
                    "anchor_neuron_id": str(r["anchor_neuron_id"]),
                    "pathway": json.loads(r["pathway"]) if r["pathway"] else [],
                    "time_start": _ts(r.get("time_start")),
                    "time_end": _ts(r.get("time_end")),
                    "coherence": float(r.get("coherence", 0.0)),
                    "salience": float(r.get("salience", 0.0)),
                    "frequency": int(r.get("frequency", 0)),
                    "summary": r.get("summary"),
                    "tags": json.loads(r["tags"]) if r["tags"] else [],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "created_at": _ts(r["created_at"]) or "",
                }
            )

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config={
                "decay_rate": brain.config.decay_rate,
                "reinforcement_delta": brain.config.reinforcement_delta,
                "activation_threshold": brain.config.activation_threshold,
                "max_spread_hops": brain.config.max_spread_hops,
                "max_context_tokens": brain.config.max_context_tokens,
                "default_synapse_weight": brain.config.default_synapse_weight,
                "hebbian_delta": brain.config.hebbian_delta,
                "hebbian_threshold": brain.config.hebbian_threshold,
                "hebbian_initial_weight": brain.config.hebbian_initial_weight,
                "consolidation_prune_threshold": brain.config.consolidation_prune_threshold,
                "prune_min_inactive_days": brain.config.prune_min_inactive_days,
                "merge_overlap_threshold": brain.config.merge_overlap_threshold,
            },
            metadata={"typed_memories": [], "projects": []},
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        brain_id = target_brain_id or snapshot.brain_id
        config = BrainConfig(**snapshot.config)
        brain = Brain.create(
            name=snapshot.brain_name,
            config=config,
            brain_id=brain_id,
        )

        old_brain_id = self._current_brain_id
        try:
            self.set_brain(brain_id)  # type: ignore[attr-defined]
            await self.save_brain(brain)

            for n_data in snapshot.neurons:
                neuron = Neuron(
                    id=n_data["id"],
                    type=NeuronType(n_data["type"]),
                    content=n_data["content"],
                    metadata=n_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(n_data["created_at"]),
                )
                meta_json = json.dumps(neuron.metadata)
                await self._query(
                    """INSERT INTO neurons (id, brain_id, type, content, metadata, content_hash, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)
                       ON CONFLICT (brain_id, id) DO UPDATE SET
                         type = EXCLUDED.type, content = EXCLUDED.content,
                         metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash""",
                    neuron.id,
                    brain_id,
                    neuron.type.value,
                    neuron.content,
                    meta_json,
                    neuron.content_hash,
                    neuron.created_at,
                )
                await self._query(
                    """INSERT INTO neuron_states (neuron_id, brain_id, firing_threshold,
                       refractory_period_ms, homeostatic_target, created_at)
                       VALUES ($1, $2, 0.3, 500.0, 0.5, $3)
                       ON CONFLICT (brain_id, neuron_id) DO NOTHING""",
                    neuron.id,
                    brain_id,
                    utcnow(),
                )

            for s_data in snapshot.synapses:
                syn = Synapse(
                    id=s_data["id"],
                    source_id=s_data["source_id"],
                    target_id=s_data["target_id"],
                    type=SynapseType(s_data["type"]),
                    weight=s_data["weight"],
                    direction=Direction(s_data.get("direction", "uni")),
                    metadata=s_data.get("metadata", {}),
                    reinforced_count=s_data.get("reinforced_count", 0),
                    created_at=datetime.fromisoformat(s_data["created_at"]),
                )
                await self._query(
                    """INSERT INTO synapses (id, brain_id, source_id, target_id, type, weight,
                       direction, metadata, reinforced_count, last_activated, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                       ON CONFLICT (brain_id, id) DO UPDATE SET
                         type = EXCLUDED.type, weight = EXCLUDED.weight,
                         direction = EXCLUDED.direction, metadata = EXCLUDED.metadata,
                         reinforced_count = EXCLUDED.reinforced_count""",
                    syn.id,
                    brain_id,
                    syn.source_id,
                    syn.target_id,
                    syn.type.value,
                    syn.weight,
                    syn.direction.value,
                    json.dumps(syn.metadata),
                    syn.reinforced_count,
                    syn.last_activated,
                    syn.created_at,
                )

            for f_data in snapshot.fibers:
                auto_tags = set(f_data.get("auto_tags", []))
                agent_tags = set(f_data.get("agent_tags", []))
                if not auto_tags and not agent_tags:
                    agent_tags = set(f_data.get("tags", []))
                fiber = Fiber(
                    id=f_data["id"],
                    neuron_ids=set(f_data["neuron_ids"]),
                    synapse_ids=set(f_data["synapse_ids"]),
                    anchor_neuron_id=f_data["anchor_neuron_id"],
                    time_start=(
                        datetime.fromisoformat(f_data["time_start"])
                        if f_data.get("time_start")
                        else None
                    ),
                    time_end=(
                        datetime.fromisoformat(f_data["time_end"])
                        if f_data.get("time_end")
                        else None
                    ),
                    coherence=f_data.get("coherence", 0.0),
                    salience=f_data.get("salience", 0.0),
                    frequency=f_data.get("frequency", 0),
                    summary=f_data.get("summary"),
                    auto_tags=auto_tags,
                    agent_tags=agent_tags,
                    metadata=f_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(f_data["created_at"]),
                )
                all_tags = sorted(fiber.auto_tags | fiber.agent_tags)
                await self._query(
                    """INSERT INTO fibers (id, brain_id, neuron_ids, synapse_ids, anchor_neuron_id,
                       pathway, conductivity, last_conducted, time_start, time_end, coherence,
                       salience, frequency, summary, tags, auto_tags, agent_tags, metadata,
                       compression_tier, pinned, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                               $16, $17, $18, $19, $20, $21)
                       ON CONFLICT (brain_id, id) DO UPDATE SET
                         neuron_ids = EXCLUDED.neuron_ids, synapse_ids = EXCLUDED.synapse_ids,
                         anchor_neuron_id = EXCLUDED.anchor_neuron_id, pathway = EXCLUDED.pathway,
                         conductivity = EXCLUDED.conductivity, last_conducted = EXCLUDED.last_conducted,
                         time_start = EXCLUDED.time_start, time_end = EXCLUDED.time_end,
                         coherence = EXCLUDED.coherence, salience = EXCLUDED.salience,
                         frequency = EXCLUDED.frequency, summary = EXCLUDED.summary,
                         tags = EXCLUDED.tags, auto_tags = EXCLUDED.auto_tags,
                         agent_tags = EXCLUDED.agent_tags, metadata = EXCLUDED.metadata,
                         compression_tier = EXCLUDED.compression_tier""",
                    fiber.id,
                    brain_id,
                    json.dumps(list(fiber.neuron_ids)),
                    json.dumps(list(fiber.synapse_ids)),
                    fiber.anchor_neuron_id,
                    json.dumps(list(fiber.pathway)),
                    fiber.conductivity,
                    fiber.last_conducted,
                    fiber.time_start,
                    fiber.time_end,
                    fiber.coherence,
                    fiber.salience,
                    fiber.frequency,
                    fiber.summary,
                    json.dumps(all_tags),
                    json.dumps(list(fiber.auto_tags)),
                    json.dumps(list(fiber.agent_tags)),
                    json.dumps(fiber.metadata),
                    fiber.compression_tier,
                    1 if fiber.pinned else 0,
                    fiber.created_at,
                )
                if fiber.neuron_ids:
                    for nid in fiber.neuron_ids:
                        await self._query(
                            """INSERT INTO fiber_neurons (brain_id, fiber_id, neuron_id)
                               VALUES ($1, $2, $3)
                               ON CONFLICT (brain_id, fiber_id, neuron_id) DO NOTHING""",
                            brain_id,
                            fiber.id,
                            nid,
                        )
        finally:
            if old_brain_id is not None:
                self.set_brain(old_brain_id)  # type: ignore[attr-defined]
            else:
                self._current_brain_id = None

        return brain_id

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        rows = await self._query_ro(
            """SELECT
                (SELECT COUNT(*) FROM neurons WHERE brain_id = $1) AS neuron_count,
                (SELECT COUNT(*) FROM synapses WHERE brain_id = $1) AS synapse_count,
                (SELECT COUNT(*) FROM fibers WHERE brain_id = $1) AS fiber_count
            """,
            brain_id,
        )
        r = rows[0] if rows else None
        return {
            "neuron_count": int(r["neuron_count"]) if r else 0,
            "synapse_count": int(r["synapse_count"]) if r else 0,
            "fiber_count": int(r["fiber_count"]) if r else 0,
            "project_count": 0,
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        basic = await self.get_stats(brain_id)

        hot_rows = await self._query_ro(
            """SELECT ns.neuron_id, n.content, n.type, ns.activation_level, ns.access_frequency
               FROM neuron_states ns
               JOIN neurons n ON n.brain_id = ns.brain_id AND n.id = ns.neuron_id
               WHERE ns.brain_id = $1
               ORDER BY ns.access_frequency DESC NULLS LAST
               LIMIT 10""",
            brain_id,
        )
        hot_neurons = [
            {
                "neuron_id": str(r["neuron_id"]),
                "content": str(r["content"]),
                "type": str(r["type"]),
                "activation_level": float(r["activation_level"] or 0),
                "access_frequency": int(r["access_frequency"] or 0),
            }
            for r in hot_rows
        ]

        from neural_memory.utils.timeutils import utcnow

        today_midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        fiber_rows = await self._query_ro(
            """SELECT COUNT(*) FILTER (WHERE created_at >= $1) AS today_cnt,
                      MIN(created_at) AS oldest, MAX(created_at) AS newest
               FROM fibers WHERE brain_id = $2""",
            today_midnight,
            brain_id,
        )
        fr = fiber_rows[0] if fiber_rows else None
        today_fibers_count = int(fr["today_cnt"]) if fr else 0
        oldest_memory = (
            fr["oldest"].isoformat()
            if fr and fr["oldest"] and hasattr(fr["oldest"], "isoformat")
            else None
        )
        newest_memory = (
            fr["newest"].isoformat()
            if fr and fr["newest"] and hasattr(fr["newest"], "isoformat")
            else None
        )

        synapse_rows = await self._query_ro(
            """SELECT type, AVG(weight) AS avg_w, SUM(reinforced_count) AS total_r, COUNT(*) AS cnt
               FROM synapses WHERE brain_id = $1 GROUP BY type""",
            brain_id,
        )
        synapse_stats: dict[str, Any] = {
            "avg_weight": 0.0,
            "total_reinforcements": 0,
            "by_type": {},
        }
        total_weight = 0.0
        total_count = 0
        total_reinforcements = 0
        for row in synapse_rows:
            t = str(row["type"])
            synapse_stats["by_type"][t] = {
                "count": int(row["cnt"]),
                "avg_weight": round(float(row["avg_w"] or 0), 4),
                "total_reinforcements": int(row["total_r"] or 0),
            }
            total_weight += float(row["avg_w"] or 0) * row["cnt"]
            total_count += row["cnt"]
            total_reinforcements += int(row["total_r"] or 0)
        if total_count > 0:
            synapse_stats["avg_weight"] = round(total_weight / total_count, 4)
        synapse_stats["total_reinforcements"] = total_reinforcements

        neuron_rows = await self._query_ro(
            "SELECT type, COUNT(*) AS cnt FROM neurons WHERE brain_id = $1 GROUP BY type",
            brain_id,
        )
        neuron_type_breakdown: dict[str, int] = {}
        for row in neuron_rows:
            neuron_type_breakdown[str(row["type"])] = int(row["cnt"])

        return {
            **basic,
            "db_size_bytes": 0,
            "hot_neurons": hot_neurons,
            "today_fibers_count": today_fibers_count,
            "synapse_stats": synapse_stats,
            "neuron_type_breakdown": neuron_type_breakdown,
            "oldest_memory": oldest_memory,
            "newest_memory": newest_memory,
        }

    async def clear(self, brain_id: str) -> None:
        tables = (
            "fiber_neurons",
            "fibers",
            "synapses",
            "neuron_states",
            "neurons",
        )
        for table in tables:
            await self._query(
                f"DELETE FROM {table} WHERE brain_id = $1",
                brain_id,
            )
        await self._query("DELETE FROM brains WHERE id = $1", brain_id)
