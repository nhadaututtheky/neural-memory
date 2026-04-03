"""Dialect-agnostic brain CRUD, export, import, stats, and clear mixin.

Merges SQLiteBrainMixin (522 LOC) and PostgresBrainMixin (475 LOC).
Uses the Dialect abstraction for all SQL differences -- no raw ``?`` or ``$N``
placeholders; every query uses ``d.ph(N)`` and ``d.phs()``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.sql.row_mappers import row_to_brain
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.sql.dialect import Dialect

# -- Export limits ---------------------------------------------------------
_MAX_EXPORT_NEURONS = 50_000
_MAX_EXPORT_SYNAPSES = 100_000
_MAX_EXPORT_FIBERS = 50_000


def _parse_json(value: Any) -> Any:
    """Parse a JSON field that may be a string (SQLite) or already parsed (PG)."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _dt_iso(d: Dialect, value: Any) -> str | None:
    """Normalize a datetime-like value via the dialect and return ISO string or None."""
    dt = d.normalize_dt(value)
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------------------
# BrainOpsMixin
# ---------------------------------------------------------------------------


class BrainOpsMixin:
    """Dialect-agnostic brain CRUD, export, import, stats, and clear.

    Requires the host class to provide::

        self._dialect: Dialect        -- injected at construction time
        self._current_brain_id: str | None
        self._get_brain_id() -> str
        self.set_brain(brain_id: str) -> None

    Protocol stubs (provided by other mixins)::

        async def add_project(project: Project) -> str: ...
        async def add_typed_memory(tm: TypedMemory) -> str: ...
    """

    if TYPE_CHECKING:
        _dialect: Dialect
        _current_brain_id: str | None

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    def set_brain(self, brain_id: str) -> None: ...

    # Protocol stubs for methods provided by other mixins
    async def add_project(self, project: Project) -> str:
        raise NotImplementedError

    async def add_typed_memory(self, tm: TypedMemory) -> str:
        raise NotImplementedError

    # ================================================================
    # save_brain
    # ================================================================

    async def save_brain(self, brain: Brain) -> None:
        d = self._dialect
        config_dict = asdict(brain.config)
        config_json = json.dumps({k: v for k, v in config_dict.items() if v is not None})
        shared_json = json.dumps(brain.shared_with)
        now = utcnow()

        sql = d.upsert_sql(
            "brains",
            ["id", "name", "config", "owner_id", "is_public",
             "shared_with", "created_at", "updated_at"],
            ["id"],
            ["name", "config", "owner_id", "is_public",
             "shared_with", "updated_at"],
        )
        await d.execute(sql, [
            brain.id,
            brain.name,
            config_json,
            brain.owner_id,
            1 if brain.is_public else 0,
            shared_json,
            d.serialize_dt(brain.created_at),
            d.serialize_dt(now),
        ])

    # ================================================================
    # get_brain / find_brain_by_name
    # ================================================================

    async def get_brain(self, brain_id: str) -> Brain | None:
        d = self._dialect
        row = await d.fetch_one(
            f"SELECT * FROM brains WHERE id = {d.ph(1)}",
            [brain_id],
        )
        if row is None:
            return None
        return row_to_brain(d, row)

    async def find_brain_by_name(self, name: str) -> Brain | None:
        d = self._dialect
        row = await d.fetch_one(
            f"SELECT * FROM brains WHERE name = {d.ph(1)}",
            [name],
        )
        if row is None:
            return None
        return row_to_brain(d, row)

    # ================================================================
    # export_brain
    # ================================================================

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        d = self._dialect
        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        neurons, synapses, fibers, typed_memories, projects = await asyncio.gather(
            _export_neurons(d, brain_id),
            _export_synapses(d, brain_id),
            _export_fibers(d, brain_id),
            _export_typed_memories(d, brain_id),
            _export_projects(d, brain_id),
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
            metadata={
                "typed_memories": typed_memories,
                "projects": projects,
            },
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
        self.set_brain(brain_id)

        try:
            await self.save_brain(brain)
            await self._import_neurons(snapshot.neurons)
            await self._import_synapses(snapshot.synapses)
            await self._import_fibers(snapshot.fibers)
            await self._import_projects(snapshot.metadata.get("projects", []))
            await self._import_typed_memories(
                snapshot.metadata.get("typed_memories", [])
            )
        finally:
            if old_brain_id is not None:
                self.set_brain(old_brain_id)
            else:
                self._current_brain_id = None

        return brain_id

    # ================================================================
    # get_stats
    # ================================================================

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        d = self._dialect
        row = await d.fetch_one(
            f"""SELECT
                (SELECT COUNT(*) FROM neurons WHERE brain_id = {d.ph(1)}) AS neuron_count,
                (SELECT COUNT(*) FROM synapses WHERE brain_id = {d.ph(1)}) AS synapse_count,
                (SELECT COUNT(*) FROM fibers WHERE brain_id = {d.ph(1)}) AS fiber_count
            """,
            [brain_id],
        )
        return {
            "neuron_count": int(row["neuron_count"]) if row else 0,
            "synapse_count": int(row["synapse_count"]) if row else 0,
            "fiber_count": int(row["fiber_count"]) if row else 0,
            "project_count": 0,
        }

    # ================================================================
    # get_enhanced_stats
    # ================================================================

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        d = self._dialect
        basic = await self.get_stats(brain_id)

        # Hot neurons (most accessed)
        hot_rows = await d.fetch_all(
            f"""SELECT ns.neuron_id, n.content, n.type,
                       ns.activation_level, ns.access_frequency
                FROM neuron_states ns
                JOIN neurons n ON n.brain_id = ns.brain_id AND n.id = ns.neuron_id
                WHERE ns.brain_id = {d.ph(1)}
                ORDER BY ns.access_frequency DESC
                LIMIT 10""",
            [brain_id],
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

        # Fiber time range + today count (dialect-agnostic, no FILTER clause)
        today_midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        fiber_row = await d.fetch_one(
            f"""SELECT MIN(created_at) AS oldest,
                       MAX(created_at) AS newest
                FROM fibers WHERE brain_id = {d.ph(1)}""",
            [brain_id],
        )
        today_row = await d.fetch_one(
            f"""SELECT COUNT(*) AS cnt FROM fibers
                WHERE brain_id = {d.ph(1)} AND created_at >= {d.ph(2)}""",
            [brain_id, d.serialize_dt(today_midnight)],
        )
        today_fibers_count = int(today_row["cnt"]) if today_row else 0

        oldest_dt = d.normalize_dt(fiber_row["oldest"]) if fiber_row and fiber_row["oldest"] else None
        newest_dt = d.normalize_dt(fiber_row["newest"]) if fiber_row and fiber_row["newest"] else None
        oldest_memory = oldest_dt.isoformat() if oldest_dt else None
        newest_memory = newest_dt.isoformat() if newest_dt else None

        # Synapse stats grouped by type
        synapse_rows = await d.fetch_all(
            f"""SELECT type, AVG(weight) AS avg_w,
                       SUM(reinforced_count) AS total_r, COUNT(*) AS cnt
                FROM synapses WHERE brain_id = {d.ph(1)}
                GROUP BY type""",
            [brain_id],
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
            cnt = int(row["cnt"])
            avg_w = float(row["avg_w"] or 0)
            total_r = int(row["total_r"] or 0)
            synapse_stats["by_type"][t] = {
                "count": cnt,
                "avg_weight": round(avg_w, 4),
                "total_reinforcements": total_r,
            }
            total_weight += avg_w * cnt
            total_count += cnt
            total_reinforcements += total_r
        if total_count > 0:
            synapse_stats["avg_weight"] = round(total_weight / total_count, 4)
        synapse_stats["total_reinforcements"] = total_reinforcements

        # Neuron type breakdown
        neuron_rows = await d.fetch_all(
            f"SELECT type, COUNT(*) AS cnt FROM neurons WHERE brain_id = {d.ph(1)} GROUP BY type",
            [brain_id],
        )
        neuron_type_breakdown: dict[str, int] = {
            str(row["type"]): int(row["cnt"]) for row in neuron_rows
        }

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

    # ================================================================
    # clear
    # ================================================================

    async def clear(self, brain_id: str) -> None:
        d = self._dialect
        for table in (
            "fiber_neurons",
            "fibers",
            "synapses",
            "neuron_states",
            "neurons",
        ):
            await d.execute(
                f"DELETE FROM {table} WHERE brain_id = {d.ph(1)}",
                [brain_id],
            )
        await d.execute(
            f"DELETE FROM brains WHERE id = {d.ph(1)}",
            [brain_id],
        )

    # ================================================================
    # Private -- import helpers
    # ================================================================

    async def _import_neurons(self, neurons_data: list[dict[str, Any]]) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        upsert_sql = d.upsert_sql(
            "neurons",
            ["id", "brain_id", "type", "content", "metadata", "content_hash", "created_at"],
            ["brain_id", "id"],
            ["type", "content", "metadata", "content_hash"],
        )
        state_sql = d.insert_or_ignore_sql(
            "neuron_states",
            ["neuron_id", "brain_id", "firing_threshold",
             "refractory_period_ms", "homeostatic_target", "created_at"],
            ["brain_id", "neuron_id"],
        )

        for n_data in neurons_data:
            neuron = Neuron(
                id=n_data["id"],
                type=NeuronType(n_data["type"]),
                content=n_data["content"],
                metadata=n_data.get("metadata", {}),
                created_at=datetime.fromisoformat(n_data["created_at"]),
            )
            await d.execute(upsert_sql, [
                neuron.id,
                brain_id,
                neuron.type.value,
                neuron.content,
                json.dumps(neuron.metadata),
                neuron.content_hash,
                d.serialize_dt(neuron.created_at),
            ])
            await d.execute(state_sql, [
                neuron.id,
                brain_id,
                0.3,
                500.0,
                0.5,
                d.serialize_dt(utcnow()),
            ])

    async def _import_synapses(self, synapses_data: list[dict[str, Any]]) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        upsert_sql = d.upsert_sql(
            "synapses",
            ["id", "brain_id", "source_id", "target_id", "type", "weight",
             "direction", "metadata", "reinforced_count", "last_activated", "created_at"],
            ["brain_id", "id"],
            ["type", "weight", "direction", "metadata", "reinforced_count"],
        )

        for s_data in synapses_data:
            synapse = Synapse(
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
            await d.execute(upsert_sql, [
                synapse.id,
                brain_id,
                synapse.source_id,
                synapse.target_id,
                synapse.type.value,
                synapse.weight,
                synapse.direction.value,
                json.dumps(synapse.metadata),
                synapse.reinforced_count,
                d.serialize_dt(synapse.last_activated),
                d.serialize_dt(synapse.created_at),
            ])

    async def _import_fibers(self, fibers_data: list[dict[str, Any]]) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        upsert_sql = d.upsert_sql(
            "fibers",
            ["id", "brain_id", "neuron_ids", "synapse_ids", "anchor_neuron_id",
             "pathway", "conductivity", "last_conducted", "time_start", "time_end",
             "coherence", "salience", "frequency", "summary", "tags",
             "auto_tags", "agent_tags", "metadata", "compression_tier",
             "pinned", "created_at"],
            ["brain_id", "id"],
            ["neuron_ids", "synapse_ids", "anchor_neuron_id", "pathway",
             "conductivity", "last_conducted", "time_start", "time_end",
             "coherence", "salience", "frequency", "summary", "tags",
             "auto_tags", "agent_tags", "metadata", "compression_tier"],
        )
        junction_sql = d.insert_or_ignore_sql(
            "fiber_neurons",
            ["brain_id", "fiber_id", "neuron_id"],
            ["brain_id", "fiber_id", "neuron_id"],
        )

        for f_data in fibers_data:
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

            await d.execute(upsert_sql, [
                fiber.id,
                brain_id,
                json.dumps(sorted(fiber.neuron_ids)),
                json.dumps(sorted(fiber.synapse_ids)),
                fiber.anchor_neuron_id,
                json.dumps(list(fiber.pathway)),
                fiber.conductivity,
                d.serialize_dt(fiber.last_conducted),
                d.serialize_dt(fiber.time_start),
                d.serialize_dt(fiber.time_end),
                fiber.coherence,
                fiber.salience,
                fiber.frequency,
                fiber.summary,
                json.dumps(all_tags),
                json.dumps(sorted(fiber.auto_tags)),
                json.dumps(sorted(fiber.agent_tags)),
                json.dumps(fiber.metadata),
                fiber.compression_tier,
                1 if fiber.pinned else 0,
                d.serialize_dt(fiber.created_at),
            ])

            if fiber.neuron_ids:
                await d.execute_many(
                    junction_sql,
                    [(brain_id, fiber.id, nid) for nid in fiber.neuron_ids],
                )

    async def _import_projects(self, projects_data: list[dict[str, Any]]) -> None:
        for p_data in projects_data:
            project = Project(
                id=p_data["id"],
                name=p_data["name"],
                description=p_data.get("description", ""),
                start_date=datetime.fromisoformat(p_data["start_date"]),
                end_date=(
                    datetime.fromisoformat(p_data["end_date"])
                    if p_data.get("end_date")
                    else None
                ),
                tags=frozenset(p_data.get("tags", [])),
                priority=p_data.get("priority", 1.0),
                metadata=p_data.get("metadata", {}),
                created_at=datetime.fromisoformat(p_data["created_at"]),
            )
            await self.add_project(project)

    async def _import_typed_memories(
        self, typed_memories_data: list[dict[str, Any]],
    ) -> None:
        for tm_data in typed_memories_data:
            prov_data = tm_data.get("provenance", {})
            provenance = Provenance(
                source=prov_data.get("source", "import"),
                confidence=Confidence(prov_data.get("confidence", "medium")),
                verified=prov_data.get("verified", False),
                verified_at=(
                    datetime.fromisoformat(prov_data["verified_at"])
                    if prov_data.get("verified_at")
                    else None
                ),
                created_by=prov_data.get("created_by", "import"),
                last_confirmed=(
                    datetime.fromisoformat(prov_data["last_confirmed"])
                    if prov_data.get("last_confirmed")
                    else None
                ),
            )

            typed_memory = TypedMemory(
                fiber_id=tm_data["fiber_id"],
                memory_type=MemoryType(tm_data["memory_type"]),
                priority=Priority(tm_data["priority"]),
                provenance=provenance,
                expires_at=(
                    datetime.fromisoformat(tm_data["expires_at"])
                    if tm_data.get("expires_at")
                    else None
                ),
                project_id=tm_data.get("project_id"),
                tags=frozenset(tm_data.get("tags", [])),
                metadata=tm_data.get("metadata", {}),
                created_at=datetime.fromisoformat(tm_data["created_at"]),
            )
            await self.add_typed_memory(typed_memory)


# ---------------------------------------------------------------------------
# Module-level export helpers (called by export_brain via asyncio.gather)
# ---------------------------------------------------------------------------


async def _export_neurons(d: Dialect, brain_id: str) -> list[dict[str, Any]]:
    rows = await d.fetch_all(
        f"""SELECT id, type, content, metadata, content_hash, created_at
            FROM neurons
            WHERE brain_id = {d.ph(1)} AND (ephemeral IS NULL OR ephemeral = 0)
            LIMIT {d.ph(2)}""",
        [brain_id, _MAX_EXPORT_NEURONS],
    )
    neurons: list[dict[str, Any]] = []
    for r in rows:
        created = d.normalize_dt(r["created_at"])
        neurons.append({
            "id": str(r["id"]),
            "type": str(r["type"]),
            "content": str(r["content"]),
            "metadata": _parse_json(r["metadata"]) or {},
            "content_hash": int(r.get("content_hash") or 0),
            "created_at": created.isoformat() if created else "",
        })
    return neurons


async def _export_synapses(d: Dialect, brain_id: str) -> list[dict[str, Any]]:
    rows = await d.fetch_all(
        f"""SELECT id, source_id, target_id, type, weight, direction,
                   metadata, reinforced_count, last_activated, created_at
            FROM synapses
            WHERE brain_id = {d.ph(1)}
            LIMIT {d.ph(2)}""",
        [brain_id, _MAX_EXPORT_SYNAPSES],
    )
    synapses: list[dict[str, Any]] = []
    for r in rows:
        la = d.normalize_dt(r.get("last_activated"))
        created = d.normalize_dt(r["created_at"])
        synapses.append({
            "id": str(r["id"]),
            "source_id": str(r["source_id"]),
            "target_id": str(r["target_id"]),
            "type": str(r["type"]),
            "weight": float(r["weight"]),
            "direction": str(r.get("direction", "uni")),
            "metadata": _parse_json(r["metadata"]) or {},
            "reinforced_count": int(r.get("reinforced_count", 0)),
            "last_activated": la.isoformat() if la else None,
            "created_at": created.isoformat() if created else "",
        })
    return synapses


async def _export_fibers(d: Dialect, brain_id: str) -> list[dict[str, Any]]:
    rows = await d.fetch_all(
        f"""SELECT id, neuron_ids, synapse_ids, anchor_neuron_id, pathway,
                   conductivity, last_conducted, time_start, time_end,
                   coherence, salience, frequency, summary, tags,
                   auto_tags, agent_tags, metadata, compression_tier, created_at
            FROM fibers
            WHERE brain_id = {d.ph(1)}
            LIMIT {d.ph(2)}""",
        [brain_id, _MAX_EXPORT_FIBERS],
    )
    fibers: list[dict[str, Any]] = []
    for r in rows:
        created = d.normalize_dt(r["created_at"])
        fibers.append({
            "id": str(r["id"]),
            "neuron_ids": _parse_json(r["neuron_ids"]) or [],
            "synapse_ids": _parse_json(r["synapse_ids"]) or [],
            "anchor_neuron_id": str(r["anchor_neuron_id"]),
            "pathway": _parse_json(r.get("pathway")) or [],
            "time_start": _dt_iso(d, r.get("time_start")),
            "time_end": _dt_iso(d, r.get("time_end")),
            "coherence": float(r.get("coherence", 0.0)),
            "salience": float(r.get("salience", 0.0)),
            "frequency": int(r.get("frequency", 0)),
            "summary": r.get("summary"),
            "tags": _parse_json(r["tags"]) or [],
            "metadata": _parse_json(r["metadata"]) or {},
            "created_at": created.isoformat() if created else "",
        })
    return fibers


async def _export_typed_memories(d: Dialect, brain_id: str) -> list[dict[str, Any]]:
    rows = await d.fetch_all(
        f"""SELECT fiber_id, memory_type, priority, provenance,
                   expires_at, project_id, tags, metadata, created_at
            FROM typed_memories
            WHERE brain_id = {d.ph(1)}
            LIMIT 50000""",
        [brain_id],
    )
    typed_memories: list[dict[str, Any]] = []
    for r in rows:
        created = d.normalize_dt(r["created_at"])
        typed_memories.append({
            "fiber_id": str(r["fiber_id"]),
            "memory_type": str(r["memory_type"]),
            "priority": str(r["priority"]),
            "provenance": _parse_json(r["provenance"]) or {},
            "expires_at": _dt_iso(d, r.get("expires_at")),
            "project_id": r.get("project_id"),
            "tags": _parse_json(r["tags"]) or [],
            "metadata": _parse_json(r["metadata"]) or {},
            "created_at": created.isoformat() if created else "",
        })
    return typed_memories


async def _export_projects(d: Dialect, brain_id: str) -> list[dict[str, Any]]:
    rows = await d.fetch_all(
        f"SELECT * FROM projects WHERE brain_id = {d.ph(1)}",
        [brain_id],
    )
    projects: list[dict[str, Any]] = []
    for r in rows:
        created = d.normalize_dt(r["created_at"])
        start = d.normalize_dt(r.get("start_date"))
        end = d.normalize_dt(r.get("end_date"))
        projects.append({
            "id": str(r["id"]),
            "name": str(r["name"]),
            "description": str(r.get("description", "")),
            "start_date": start.isoformat() if start else "",
            "end_date": end.isoformat() if end else None,
            "tags": _parse_json(r["tags"]) or [],
            "priority": float(r.get("priority", 1.0)),
            "metadata": _parse_json(r["metadata"]) or {},
            "created_at": created.isoformat() if created else "",
        })
    return projects
