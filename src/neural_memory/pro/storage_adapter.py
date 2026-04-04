"""InfinityDB Storage Adapter — NeuralStorage interface for InfinityDB.

Maps the free tier's NeuralStorage abstract interface to InfinityDB calls,
enabling InfinityDB as a drop-in storage backend for Neural Memory.

Usage:
    from neural_memory.pro.storage_adapter import InfinityDBStorage

    storage = InfinityDBStorage(base_dir="/data/brains", brain_id="default")
    await storage.open()
    # Now use as any NeuralStorage implementation
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.pro.infinitydb.engine import InfinityDB
from neural_memory.pro.infinitydb.tier_manager import TierConfig
from neural_memory.pro.storage_adapter_extras import InfinityDBExtrasMixin
from neural_memory.pro.storage_adapter_sync import InfinityDBSyncMixin
from neural_memory.pro.storage_adapter_typed import InfinityDBTypedMixin
from neural_memory.storage.base import NeuralStorage
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainSnapshot

logger = logging.getLogger(__name__)


def _meta_to_neuron(meta: dict[str, Any]) -> Neuron:
    """Convert InfinityDB metadata dict to Neuron frozen dataclass."""
    neuron_type = meta.get("type", "concept")
    try:
        nt = NeuronType(neuron_type)
    except ValueError:
        nt = NeuronType.CONCEPT

    created_at = meta.get("created_at", "")
    if isinstance(created_at, str) and created_at:
        try:
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            created_at = utcnow()
    elif not isinstance(created_at, datetime):
        created_at = utcnow()

    return Neuron(
        id=meta.get("id", ""),
        type=nt,
        content=meta.get("content", ""),
        metadata={
            k: v
            for k, v in meta.items()
            if k not in ("id", "type", "content", "created_at", "ephemeral", "content_hash")
        },
        content_hash=meta.get("content_hash", 0),
        created_at=created_at,
        ephemeral=meta.get("ephemeral", False),
    )


def _neuron_to_kwargs(neuron: Neuron) -> dict[str, Any]:
    """Convert Neuron dataclass to InfinityDB add_neuron kwargs."""
    return {
        "content": neuron.content,
        "neuron_id": neuron.id,
        "neuron_type": neuron.type.value,
        "priority": neuron.metadata.get("priority", 5),
        "activation_level": neuron.metadata.get("activation_level", 1.0),
        "tags": neuron.metadata.get("tags", []),
        "ephemeral": neuron.ephemeral,
        "embedding": neuron.metadata.get("embedding"),
    }


def _meta_to_synapse(edge: dict[str, Any]) -> Synapse:
    """Convert InfinityDB edge dict to Synapse frozen dataclass."""
    edge_type = edge.get("type", "related")
    try:
        st = SynapseType(edge_type)
    except ValueError:
        st = SynapseType.RELATED_TO

    return Synapse(
        id=edge.get("id", ""),
        source_id=edge.get("source_id", ""),
        target_id=edge.get("target_id", ""),
        type=st,
        weight=edge.get("weight", 1.0),
        metadata=edge.get("metadata", {}),
    )


def _meta_to_fiber(fdict: dict[str, Any]) -> Fiber:
    """Convert InfinityDB fiber dict to Fiber frozen dataclass."""
    nids = fdict.get("neuron_ids", [])
    return Fiber(
        id=fdict.get("id", ""),
        summary=fdict.get("name", ""),
        neuron_ids=set(nids),
        synapse_ids=set(fdict.get("synapse_ids", [])),
        anchor_neuron_id=fdict.get("anchor_neuron_id", nids[0] if nids else ""),
        metadata={
            "fiber_type": fdict.get("fiber_type", "cluster"),
            "description": fdict.get("description", ""),
            **(fdict.get("metadata") or {}),
        },
    )


class InfinityDBStorage(
    InfinityDBTypedMixin, InfinityDBSyncMixin, InfinityDBExtrasMixin, NeuralStorage
):
    """NeuralStorage adapter wrapping InfinityDB.

    Provides the standard NeuralStorage interface backed by InfinityDB's
    HNSW vector search, graph store, and tiered compression.
    """

    def __init__(
        self,
        base_dir: str | Path,
        brain_id: str = "default",
        dimensions: int = 384,
        tier_config: TierConfig | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._current_brain_id: str | None = brain_id
        self._dimensions = dimensions
        self._tier_config = tier_config
        self._db: InfinityDB | None = None
        self._neuron_states: dict[str, NeuronState] = {}
        # Initialize mixin stores
        self._init_typed_stores()
        self._init_lifecycle_stores()
        self._init_keyword_stores()
        self._init_entity_stores()
        self._init_sync_stores()
        self._init_extras_stores()

    @property
    def db(self) -> InfinityDB:
        """Get the underlying InfinityDB instance."""
        if self._db is None:
            msg = "Storage not opened. Call open() first."
            raise RuntimeError(msg)
        return self._db

    async def open(self) -> None:
        """Open the InfinityDB storage."""
        brain_id = self._current_brain_id or "default"
        brain_dir = self._base_dir / brain_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        self._db = InfinityDB(
            self._base_dir,
            brain_id=brain_id,
            dimensions=self._dimensions,
            tier_config=self._tier_config,
        )
        await self._db.open()

    async def initialize(self) -> None:
        """Alias for open() — compatibility with dashboard migration code."""
        if self._db is not None:
            return  # Already open, no-op
        await self.open()

    async def close(self) -> None:
        """Close the storage."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    def set_brain(self, brain_id: str) -> None:
        """Set the active brain. Requires reopen."""
        self._current_brain_id = brain_id

    async def list_brains(self) -> list[dict[str, str]]:
        """Return brain list for compatibility with dashboard migration code."""
        bid = self._current_brain_id or "default"
        return [{"id": bid, "name": bid}]

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        kwargs = _neuron_to_kwargs(neuron)
        return await self.db.add_neuron(**kwargs)

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        meta = await self.db.get_neuron(neuron_id)
        if meta is None:
            return None
        return _meta_to_neuron(meta)

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[Neuron]:
        tr = None
        if time_range is not None:
            tr = (time_range[0].isoformat(), time_range[1].isoformat())

        results = await self.db.find_neurons(
            neuron_type=type.value if type else None,
            content_contains=content_contains,
            content_exact=content_exact,
            time_range=tr,
            limit=limit,
            offset=offset,
            ephemeral=ephemeral,
        )
        return [_meta_to_neuron(m) for m in results]

    async def suggest_neurons(
        self, prefix: str, type_filter: NeuronType | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        tf = type_filter.value if type_filter else None
        return await self.db.suggest_neurons(prefix, type_filter=tf, limit=limit)

    async def update_neuron(self, neuron: Neuron) -> None:
        await self.db.update_neuron(
            neuron.id,
            content=neuron.content,
            neuron_type=neuron.type.value,
            tags=neuron.metadata.get("tags"),
            priority=neuron.metadata.get("priority"),
            activation_level=neuron.metadata.get("activation_level"),
            ephemeral=neuron.ephemeral,
            embedding=neuron.metadata.get("embedding"),
        )

    async def delete_neuron(self, neuron_id: str) -> bool:
        return await self.db.delete_neuron(neuron_id)

    # ========== Neuron State ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        if neuron_id in self._neuron_states:
            return self._neuron_states[neuron_id]
        meta = await self.db.get_neuron(neuron_id)
        if meta is None:
            return None
        state = NeuronState(
            neuron_id=neuron_id,
            activation_level=meta.get("activation_level", 0.0),
            access_frequency=meta.get("access_count", 0),
            last_activated=_parse_datetime(meta.get("accessed_at", "")),
        )
        self._neuron_states[neuron_id] = state
        return state

    async def update_neuron_state(self, state: NeuronState) -> None:
        self._neuron_states[state.neuron_id] = state
        await self.db.update_neuron(
            state.neuron_id,
            activation_level=state.activation_level,
        )

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        return await self.db.add_synapse(
            synapse.source_id,
            synapse.target_id,
            edge_type=synapse.type.value,
            weight=synapse.weight,
            edge_id=synapse.id if synapse.id else None,
            metadata=synapse.metadata if synapse.metadata else None,
        )

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        # InfinityDB doesn't have get_synapse_by_id on engine level directly
        # Use graph store's get_edge_by_id
        edge = self.db._graph.get_edge_by_id(synapse_id)
        if edge is None:
            return None
        return _meta_to_synapse(edge)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        results: list[dict[str, Any]] = []

        if source_id:
            results = await self.db.get_synapses(source_id, direction="outgoing")
        elif target_id:
            results = await self.db.get_synapses(target_id, direction="incoming")
        else:
            return []

        synapses = [_meta_to_synapse(e) for e in results]

        # Apply filters
        if type is not None:
            synapses = [s for s in synapses if s.type == type]
        if min_weight is not None:
            synapses = [s for s in synapses if s.weight >= min_weight]
        if target_id and source_id:
            synapses = [s for s in synapses if s.target_id == target_id]

        return synapses

    async def update_synapse(self, synapse: Synapse) -> None:
        await self.db.update_synapse(
            synapse.id,
            {
                "type": synapse.type.value,
                "weight": synapse.weight,
            },
        )

    async def delete_synapse(self, synapse_id: str) -> bool:
        return await self.db.delete_synapse(synapse_id)

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        dir_map = {"out": "outgoing", "in": "incoming", "both": "both"}
        edges = await self.db.get_synapses(neuron_id, direction=dir_map.get(direction, "both"))

        results: list[tuple[Neuron, Synapse]] = []
        for edge in edges:
            synapse = _meta_to_synapse(edge)
            if synapse_types and synapse.type not in synapse_types:
                continue
            if min_weight is not None and synapse.weight < min_weight:
                continue

            neighbor_id = (
                edge.get("target_id")
                if edge.get("source_id") == neuron_id
                else edge.get("source_id")
            )
            if not neighbor_id:
                continue

            neuron = await self.get_neuron(neighbor_id)
            if neuron is not None:
                results.append((neuron, synapse))

        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        # BFS to find path
        direction = "both" if bidirectional else "outgoing"
        traversal = await self.db.bfs_traverse(
            source_id,
            max_depth=max_hops,
            direction=direction,
        )

        # Check if target was reached
        reached_ids = {nid for nid, _ in traversal}
        if target_id not in reached_ids:
            return None

        # Reconstruct simple path (BFS doesn't track parents, so return endpoints)
        path: list[tuple[Neuron, Synapse]] = []
        src_neuron = await self.get_neuron(source_id)
        tgt_neuron = await self.get_neuron(target_id)
        if src_neuron and tgt_neuron:
            # Find direct edge if exists
            edges = await self.db.get_synapses(source_id, direction="outgoing")
            for edge in edges:
                if edge.get("target_id") == target_id:
                    path.append((tgt_neuron, _meta_to_synapse(edge)))
                    break
            if not path:
                # No direct edge — return minimal path indication
                dummy_synapse = Synapse(
                    id="",
                    source_id=source_id,
                    target_id=target_id,
                    type=SynapseType.RELATED_TO,
                    weight=0.0,
                )
                path.append((tgt_neuron, dummy_synapse))

        return path if path else None

    # ========== Fiber Operations ==========

    async def add_fiber(self, fiber: Fiber) -> str:
        neuron_ids = list(fiber.neuron_ids) if fiber.neuron_ids else None
        fiber_meta = dict(fiber.metadata) if fiber.metadata else {}
        # Preserve fields that InfinityDB fiber store doesn't natively track
        if fiber.synapse_ids:
            fiber_meta["synapse_ids"] = list(fiber.synapse_ids)
        if fiber.anchor_neuron_id:
            fiber_meta["anchor_neuron_id"] = fiber.anchor_neuron_id
        return await self.db.add_fiber(
            name=fiber.summary or "",
            fiber_id=fiber.id if fiber.id else None,
            fiber_type=fiber_meta.pop("fiber_type", "cluster") if fiber_meta else "cluster",
            description=fiber_meta.pop("description", "") if fiber_meta else "",
            neuron_ids=neuron_ids,
            metadata=fiber_meta,
        )

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        result = await self.db.get_fiber(fiber_id)
        if result is None:
            return None
        return _meta_to_fiber(result)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        if contains_neuron:
            fiber_ids = await self.db.get_fibers_for_neuron(contains_neuron)
            fibers = []
            for fid in fiber_ids[:limit]:
                f = await self.db.get_fiber(fid)
                if f is not None:
                    fibers.append(_meta_to_fiber(f))
            return fibers

        results = await self.db.find_fibers(limit=limit)
        return [_meta_to_fiber(f) for f in results]

    async def update_fiber(self, fiber: Fiber) -> None:
        # InfinityDB FiberStore doesn't have update_fiber — delete + re-add
        await self.db.delete_fiber(fiber.id)
        await self.add_fiber(fiber)

    async def delete_fiber(self, fiber_id: str) -> bool:
        return await self.db.delete_fiber(fiber_id)

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        results = await self.db.find_fibers(limit=limit)
        return [_meta_to_fiber(f) for f in results]

    # ========== Brain Operations ==========

    async def save_brain(self, brain: Brain) -> None:
        # Brain metadata is stored in the InfinityDB header
        await self.db.flush()

    async def get_brain(self, brain_id: str) -> Brain | None:
        if brain_id != self.db.brain_id:
            return None
        stats = await self.db.get_stats()
        return Brain(
            id=brain_id,
            name=brain_id,
            neuron_count=stats["neuron_count"],
            synapse_count=stats["synapse_count"],
            fiber_count=stats["fiber_count"],
        )

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        from neural_memory.core.brain import BrainSnapshot

        if brain_id != self.db.brain_id:
            raise ValueError(f"Brain '{brain_id}' not found in this InfinityDB")

        # Export all neurons
        all_neurons = await self.db.find_neurons(limit=50000)
        neurons_out: list[dict[str, Any]] = []
        for meta in all_neurons:
            neurons_out.append(
                {
                    "id": meta.get("id", ""),
                    "type": meta.get("type", "concept"),
                    "content": meta.get("content", ""),
                    "metadata": {
                        k: v
                        for k, v in meta.items()
                        if k not in ("id", "type", "content", "created_at")
                    },
                    "created_at": meta.get("created_at", ""),
                }
            )

        # Export all synapses
        synapses_out: list[dict[str, Any]] = []
        exported_sources: set[str] = set()
        for n in all_neurons:
            nid = n.get("id", "")
            if nid in exported_sources:
                continue
            exported_sources.add(nid)
            edges = await self.db.get_synapses(nid, direction="outgoing")
            for edge in edges:
                synapses_out.append(
                    {
                        "id": edge.get("id", ""),
                        "source_id": edge.get("source_id", ""),
                        "target_id": edge.get("target_id", ""),
                        "type": edge.get("type", "related"),
                        "weight": edge.get("weight", 1.0),
                        "metadata": edge.get("metadata", {}),
                    }
                )

        # Export all fibers
        fibers_out: list[dict[str, Any]] = []
        fiber_list = await self.db.find_fibers(limit=50000)
        for f in fiber_list:
            fibers_out.append(
                {
                    "id": f.get("id", ""),
                    "neuron_ids": f.get("neuron_ids", []),
                    "synapse_ids": f.get("synapse_ids", []),
                    "anchor_neuron_id": f.get("anchor_neuron_id", ""),
                    "summary": f.get("name", ""),
                    "metadata": f.get("metadata", {}),
                }
            )

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain_id,
            exported_at=utcnow(),
            version="infinitydb-0.2.0",
            neurons=neurons_out,
            synapses=synapses_out,
            fibers=fibers_out,
            config={},
            metadata={"storage_backend": "infinitydb"},
        )

    async def import_brain(
        self, snapshot: BrainSnapshot, target_brain_id: str | None = None
    ) -> str:
        bid = target_brain_id or snapshot.brain_id

        # Clear existing data if same brain
        if bid == self.db.brain_id:
            await self.clear(bid)

        # Import neurons
        for ndata in snapshot.neurons:
            meta = ndata.get("metadata", {})
            await self.db.add_neuron(
                content=ndata.get("content", ""),
                neuron_id=ndata.get("id", ""),
                neuron_type=ndata.get("type", "concept"),
                priority=meta.get("priority", 5),
                activation_level=meta.get("activation_level", 1.0),
                tags=meta.get("tags", []),
                embedding=meta.get("embedding"),
            )

        # Import synapses
        for sdata in snapshot.synapses:
            await self.db.add_synapse(
                source_id=sdata.get("source_id", ""),
                target_id=sdata.get("target_id", ""),
                edge_type=sdata.get("type", "related"),
                weight=sdata.get("weight", 1.0),
                edge_id=sdata.get("id"),
                metadata=sdata.get("metadata", {}),
            )

        # Import fibers
        for fdata in snapshot.fibers:
            await self.db.add_fiber(
                name=fdata.get("summary", ""),
                fiber_id=fdata.get("id"),
                neuron_ids=fdata.get("neuron_ids"),
                metadata={
                    **(fdata.get("metadata") or {}),
                    "synapse_ids": fdata.get("synapse_ids", []),
                    "anchor_neuron_id": fdata.get("anchor_neuron_id", ""),
                },
            )

        await self.db.flush()
        return bid

    # ========== Stats ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        stats = await self.db.get_stats()
        return {
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        stats = await self.db.get_stats()
        tier_stats = await self.db.get_tier_stats()
        return {**stats, "tiers": tier_stats}

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        # Close and reopen with fresh state
        await self.db.close()
        import shutil

        brain_dir = self._base_dir / brain_id
        if brain_dir.exists():
            shutil.rmtree(brain_dir)
        await self.open()

    # ========== Pro-Specific Methods ==========

    async def search_similar(
        self,
        query_vector: list[float],
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """HNSW vector similarity search (Pro-only)."""
        return await self.db.search_similar(query_vector, k=k)

    async def demote_sweep(self) -> dict[str, int]:
        """Run tier demotion sweep (Pro-only)."""
        return await self.db.demote_sweep()

    async def get_tier_stats(self) -> dict[str, Any]:
        """Get compression tier statistics (Pro-only)."""
        return await self.db.get_tier_stats()


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse ISO datetime string to datetime, or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None
