"""InfinityDB — Main database engine.

Orchestrates vector store, HNSW index, metadata store, graph store,
and fiber store to provide a unified neuron + graph storage interface.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_memory.pro.infinitydb.fiber_store import FiberStore
from neural_memory.pro.infinitydb.file_format import BrainPaths, InfinityHeader
from neural_memory.pro.infinitydb.graph_store import GraphStore
from neural_memory.pro.infinitydb.hnsw_index import HNSWIndex
from neural_memory.pro.infinitydb.metadata_store import MetadataStore
from neural_memory.pro.infinitydb.query_planner import QueryExecutor, QueryPlan
from neural_memory.pro.infinitydb.tier_manager import TierConfig, TierManager
from neural_memory.pro.infinitydb.vector_store import VectorStore
from neural_memory.pro.infinitydb.wal import WALEntry, WALOp, WriteAheadLog

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """UTC now as ISO string (naive, no tzinfo for SQLite compat)."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


class InfinityDB:
    """Custom spatial database engine for Neural Memory Pro.

    Provides neuron CRUD, vector similarity search, and metadata management.
    Uses memory-mapped vectors (numpy), HNSW index (hnswlib), and msgpack metadata.
    """

    def __init__(
        self,
        base_dir: str | Path,
        brain_id: str = "default",
        dimensions: int = 384,
        tier_config: TierConfig | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._brain_id = brain_id
        self._dimensions = dimensions
        self._paths = BrainPaths(self._base_dir, brain_id)
        self._header = InfinityHeader(dimensions=dimensions)
        self._vectors = VectorStore(self._paths.vectors, dimensions)
        self._index = HNSWIndex(self._paths.index, dimensions)
        self._metadata = MetadataStore(self._paths.meta)
        self._graph = GraphStore(self._paths.graph)
        self._fibers = FiberStore(self._paths.fibers)
        self._wal = WriteAheadLog(self._paths.wal)
        self._tier_manager = TierManager(dimensions, tier_config)
        self._is_open = False

    @property
    def brain_id(self) -> str:
        return self._brain_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def neuron_count(self) -> int:
        return self._metadata.count

    @property
    def synapse_count(self) -> int:
        return self._graph.edge_count

    @property
    def fiber_count(self) -> int:
        return self._fibers.count

    # --- Lifecycle ---

    async def open(self) -> None:
        """Open the database, loading all stores."""
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        self._paths.ensure_dirs()

        # Load or create header
        if self._paths.header.exists():
            data = self._paths.header.read_bytes()
            self._header = InfinityHeader.from_bytes(data)
            self._dimensions = self._header.dimensions
            # Recreate stores with correct dimensions
            self._vectors = VectorStore(self._paths.vectors, self._dimensions)
            self._index = HNSWIndex(self._paths.index, self._dimensions)
        else:
            self._header = InfinityHeader(dimensions=self._dimensions)
            self._paths.header.write_bytes(self._header.to_bytes())

        # Open WAL first — must be available before any store opens
        # so crash during open sequence can be recovered
        self._wal.open()

        # Open stores
        self._metadata.open()
        self._vectors.open()
        self._graph.open()
        self._fibers.open()

        # Open HNSW with capacity based on metadata count
        initial_cap = max(1024, self._metadata.count * 2)
        self._index.open(max_elements=initial_cap)
        replayed = self._replay_wal()
        if replayed > 0:
            logger.info("WAL replay: %d entries recovered", replayed)
            self._flush_sync()  # persist recovered state
            self._wal.checkpoint()

        self._is_open = True
        logger.info(
            "InfinityDB opened: brain=%s, neurons=%d, synapses=%d, fibers=%d, dims=%d",
            self._brain_id,
            self._metadata.count,
            self._graph.edge_count,
            self._fibers.count,
            self._dimensions,
        )

    async def close(self) -> None:
        """Close all stores and flush to disk."""
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if not self._is_open:
            return
        self._flush_header()
        self._metadata.close()
        self._vectors.close()
        self._index.close()
        self._graph.close()
        self._fibers.close()
        self._wal.close()
        self._is_open = False
        logger.info("InfinityDB closed: brain=%s", self._brain_id)

    async def flush(self) -> None:
        """Flush all data to disk without closing."""
        await asyncio.to_thread(self._flush_sync)

    def _flush_sync(self) -> None:
        self._flush_header()
        self._metadata.flush()
        self._vectors.flush()
        self._index.save()
        self._graph.flush()
        self._fibers.flush()
        # Checkpoint WAL after all stores are persisted
        if self._wal.is_open:
            self._wal.checkpoint()

    def _flush_header(self) -> None:
        header = InfinityHeader(
            version=self._header.version,
            dimensions=self._dimensions,
            tier_config=self._header.tier_config,
            flags=self._header.flags,
            neuron_count=self._metadata.count,
            synapse_count=self._graph.edge_count,
        )
        self._paths.header.write_bytes(header.to_bytes())

    # --- WAL Replay ---

    def _replay_wal(self) -> int:
        """Replay pending WAL entries for crash recovery.

        Each operation is idempotent: duplicate adds are skipped,
        deletes on missing IDs are no-ops.
        Returns the number of entries replayed.
        """
        entries = self._wal.get_pending_entries()
        if not entries:
            return 0

        replayed = 0
        for entry in entries:
            try:
                self._replay_entry(entry)
                replayed += 1
            except Exception as e:
                logger.warning(
                    "WAL replay failed at seq=%d op=%s: %s",
                    entry.seq,
                    entry.op.name,
                    e,
                )
        return replayed

    def _replay_entry(self, entry: WALEntry) -> None:
        """Replay a single WAL entry idempotently."""
        p = entry.payload

        if entry.op == WALOp.ADD_NEURON:
            nid = p["id"]
            # Skip if already exists (idempotent)
            if self._metadata.get_by_id(nid) is not None:
                return
            vec_slot = -1
            embedding = p.get("embedding")
            if embedding is not None:
                vec = np.asarray(embedding, dtype=np.float32)
                if vec.shape == (self._dimensions,):
                    vec_slot = self._vectors.add(vec)
                    self._index.add(vec_slot, vec)
            meta = {
                "id": nid,
                "type": p.get("type", "fact"),
                "content": p.get("content", ""),
                "priority": p.get("priority", 5),
                "activation_level": p.get("activation_level", 1.0),
                "created_at": p.get("created_at", entry.timestamp),
                "updated_at": p.get("updated_at", entry.timestamp),
                "accessed_at": p.get("accessed_at", entry.timestamp),
                "access_count": p.get("access_count", 0),
                "ephemeral": p.get("ephemeral", False),
                "tags": p.get("tags", []),
                "vec_slot": vec_slot,
            }
            slot = vec_slot if vec_slot >= 0 else self._metadata.next_free_slot()
            self._metadata.add(slot, meta)

        elif entry.op == WALOp.DELETE_NEURON:
            nid = p["id"]
            result = self._metadata.get_by_id(nid)
            if result is None:
                return  # Already deleted
            slot, meta = result
            vec_slot = meta.get("vec_slot", -1)
            if vec_slot >= 0:
                self._vectors.delete(vec_slot)
                self._index.delete(vec_slot)
            self._graph.delete_neuron_edges(nid)
            self._fibers.remove_neuron_from_all(nid)
            self._metadata.delete(slot)

        elif entry.op == WALOp.UPDATE_NEURON:
            nid = p["id"]
            result = self._metadata.get_by_id(nid)
            if result is None:
                return  # Neuron gone, skip
            slot, meta = result
            updates = {k: v for k, v in p.get("updates", {}).items() if k in self._UPDATABLE_FIELDS}
            # Replay embedding update if present
            embedding = p.get("embedding")
            if embedding is not None:
                vec = np.asarray(embedding, dtype=np.float32)
                if vec.shape == (self._dimensions,):
                    old_slot = meta.get("vec_slot", -1)
                    new_slot = self._vectors.add(vec)
                    self._index.add(new_slot, vec)
                    if old_slot >= 0:
                        self._index.delete(old_slot)
                        self._vectors.delete(old_slot)
                    updates["vec_slot"] = new_slot
            if updates:
                self._metadata.update(slot, updates)

        elif entry.op == WALOp.ADD_SYNAPSE:
            # Idempotent: skip if edge already exists
            eid = p.get("edge_id")
            if eid and self._graph.get_edge_by_id(eid) is not None:
                return
            self._graph.add_edge(
                p["source_id"],
                p["target_id"],
                edge_type=p.get("edge_type", "related"),
                weight=p.get("weight", 1.0),
                edge_id=eid,
                metadata=p.get("metadata"),
            )

        elif entry.op == WALOp.DELETE_SYNAPSE:
            eid = p.get("edge_id")
            if eid:
                self._graph.delete_edge(eid)  # no-op if already deleted

        elif entry.op == WALOp.ADD_FIBER:
            fid = p.get("fiber_id")
            if fid and self._fibers.get_fiber(fid) is not None:
                return  # already exists
            self._fibers.add_fiber(
                p.get("name", ""),
                fiber_id=fid,
                fiber_type=p.get("fiber_type", "cluster"),
                description=p.get("description", ""),
                neuron_ids=p.get("neuron_ids"),
                metadata=p.get("metadata"),
            )

        elif entry.op == WALOp.DELETE_FIBER:
            fid = p.get("fiber_id")
            if fid:
                self._fibers.delete_fiber(fid)  # no-op if already deleted

    # --- Neuron CRUD ---

    async def add_neuron(
        self,
        content: str,
        *,
        neuron_id: str | None = None,
        neuron_type: str = "fact",
        embedding: list[float] | NDArray[np.float32] | None = None,
        priority: int = 5,
        activation_level: float = 1.0,
        tags: list[str] | None = None,
        ephemeral: bool = False,
    ) -> str:
        """Add a neuron with optional embedding vector.

        Returns the neuron ID.
        """
        nid = neuron_id or str(uuid.uuid4())
        now = _utcnow()

        # WAL: log before write
        wal_payload: dict[str, Any] = {
            "id": nid,
            "type": neuron_type,
            "content": content,
            "priority": priority,
            "activation_level": activation_level,
            "created_at": now,
            "updated_at": now,
            "accessed_at": now,
            "access_count": 0,
            "ephemeral": ephemeral,
            "tags": list(tags) if tags else [],
        }
        if embedding is not None:
            wal_payload["embedding"] = np.asarray(embedding, dtype=np.float32).tolist()
        await asyncio.to_thread(self._wal.append, WALOp.ADD_NEURON, wal_payload)

        # Store vector if embedding provided
        vec_slot = -1
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.shape == (self._dimensions,):
                vec_slot = await asyncio.to_thread(self._vectors.add, vec)
                try:
                    await asyncio.to_thread(self._index.add, vec_slot, vec)
                except Exception:
                    # Rollback vector on index failure
                    await asyncio.to_thread(self._vectors.delete, vec_slot)
                    raise

        # Build metadata with initial tier classification
        initial_meta: dict[str, Any] = {
            "priority": priority,
            "access_count": 0,
            "accessed_at": now,
        }
        tier = int(self._tier_manager.classify_neuron(initial_meta))

        meta: dict[str, Any] = {
            "id": nid,
            "type": neuron_type,
            "content": content,
            "priority": priority,
            "activation_level": activation_level,
            "created_at": now,
            "updated_at": now,
            "accessed_at": now,
            "access_count": 0,
            "ephemeral": ephemeral,
            "tags": list(tags) if tags else [],
            "vec_slot": vec_slot,
            "tier": tier,
        }

        meta_slot = vec_slot if vec_slot >= 0 else self._metadata.next_free_slot()
        try:
            await asyncio.to_thread(self._metadata.add, meta_slot, meta)
        except ValueError:
            # Rollback vector + index on metadata failure (e.g. duplicate ID)
            if vec_slot >= 0:
                await asyncio.to_thread(self._vectors.delete, vec_slot)
                await asyncio.to_thread(self._index.delete, vec_slot)
            raise
        return nid

    async def add_neurons_batch(
        self,
        neurons: list[dict[str, Any]],
    ) -> list[str]:
        """Batch insert neurons. Much faster than individual add_neuron calls.

        Each dict should have: content, and optionally: neuron_id, neuron_type,
        embedding (as numpy array), priority, tags, ephemeral.
        Returns list of neuron IDs.
        """
        return await asyncio.to_thread(self._add_neurons_batch_sync, neurons)

    def _add_neurons_batch_sync(self, neurons: list[dict[str, Any]]) -> list[str]:
        """Synchronous batch insert with rollback on failure."""
        now = _utcnow()
        ids: list[str] = []
        vec_slots: list[int] = []
        vec_arrays: list[NDArray[np.float32]] = []
        committed_meta_slots: list[int] = []

        # Pre-resolve IDs and WAL-log all neurons before writing any data
        resolved_ids: list[str] = []
        for neuron in neurons:
            nid = neuron.get("neuron_id") or str(uuid.uuid4())
            resolved_ids.append(nid)
            wal_payload: dict[str, Any] = {
                "id": nid,
                "type": neuron.get("neuron_type", "fact"),
                "content": neuron.get("content", ""),
                "priority": neuron.get("priority", 5),
                "activation_level": neuron.get("activation_level", 1.0),
                "created_at": neuron.get("created_at") or now,
                "updated_at": now,
                "accessed_at": now,
                "access_count": 0,
                "ephemeral": neuron.get("ephemeral", False),
                "tags": list(neuron.get("tags", [])),
            }
            embedding = neuron.get("embedding")
            if embedding is not None:
                wal_payload["embedding"] = np.asarray(embedding, dtype=np.float32).tolist()
            self._wal.append(WALOp.ADD_NEURON, wal_payload)

        try:
            for i, neuron in enumerate(neurons):
                nid = resolved_ids[i]
                embedding = neuron.get("embedding")
                vec_slot = -1

                if embedding is not None:
                    vec = np.asarray(embedding, dtype=np.float32)
                    if vec.shape == (self._dimensions,):
                        vec_slot = self._vectors.add(vec)
                        vec_slots.append(vec_slot)
                        vec_arrays.append(vec)

                meta: dict[str, Any] = {
                    "id": nid,
                    "type": neuron.get("neuron_type", "fact"),
                    "content": neuron.get("content", ""),
                    "priority": neuron.get("priority", 5),
                    "activation_level": neuron.get("activation_level", 1.0),
                    "created_at": now,
                    "updated_at": now,
                    "accessed_at": now,
                    "access_count": 0,
                    "ephemeral": neuron.get("ephemeral", False),
                    "tags": list(neuron.get("tags", [])),
                    "vec_slot": vec_slot,
                }
                slot = vec_slot if vec_slot >= 0 else self._metadata.next_free_slot()
                self._metadata.add(slot, meta)
                committed_meta_slots.append(slot)
                ids.append(nid)

            # Batch add to HNSW index AFTER metadata is committed
            if vec_slots and vec_arrays:
                vectors = np.stack(vec_arrays)
                self._index.add_batch(vec_slots, vectors)

        except Exception:
            # Rollback: delete committed metadata and vector slots
            for slot in committed_meta_slots:
                self._metadata.delete(slot)
            for slot in vec_slots:
                self._vectors.delete(slot)
            raise

        return ids

    async def get_neuron(self, neuron_id: str) -> dict[str, Any] | None:
        """Get a neuron by ID. Auto-promotes tier on access."""
        result = self._metadata.get_by_id(neuron_id)
        if result is None:
            return None
        slot, meta = result

        # Auto-promote tier on access
        self._maybe_promote_sync(slot, meta)

        return dict(meta)

    async def find_neurons(
        self,
        *,
        neuron_type: str | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[str, str] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Find neurons matching filters."""
        results = self._metadata.find(
            neuron_type=neuron_type,
            content_contains=content_contains,
            content_exact=content_exact,
            time_range=time_range,
            limit=limit,
            offset=offset,
            ephemeral=ephemeral,
        )
        return [dict(meta) for _, meta in results]

    # Allowlist of fields that can be updated via update_neuron
    _UPDATABLE_FIELDS = frozenset(
        {
            "content",
            "type",
            "priority",
            "activation_level",
            "tags",
            "ephemeral",
            "accessed_at",
            "access_count",
            "updated_at",
            "tier",
        }
    )

    async def update_neuron(
        self,
        neuron_id: str,
        *,
        content: str | None = None,
        neuron_type: str | None = None,
        priority: int | None = None,
        activation_level: float | None = None,
        embedding: list[float] | NDArray[np.float32] | None = None,
        tags: list[str] | None = None,
        ephemeral: bool | None = None,
    ) -> bool:
        """Update a neuron's metadata and/or vector."""
        result = await asyncio.to_thread(self._metadata.get_by_id, neuron_id)
        if result is None:
            return False

        # Build updates dict first so we can WAL-log it
        slot, meta = result
        updates: dict[str, Any] = {"updated_at": _utcnow()}

        if content is not None:
            updates["content"] = content
        if neuron_type is not None:
            updates["type"] = neuron_type
        if priority is not None:
            updates["priority"] = priority
        if activation_level is not None:
            updates["activation_level"] = activation_level
        if tags is not None:
            updates["tags"] = list(tags)
        if ephemeral is not None:
            updates["ephemeral"] = ephemeral

        # WAL: log BEFORE any mutation (crash safety)
        wal_payload: dict[str, Any] = {"id": neuron_id, "updates": dict(updates)}
        if embedding is not None:
            wal_payload["embedding"] = np.asarray(embedding, dtype=np.float32).tolist()
        await asyncio.to_thread(
            self._wal.append,
            WALOp.UPDATE_NEURON,
            wal_payload,
        )

        # Update vector if new embedding provided
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.shape == (self._dimensions,):
                old_slot = meta.get("vec_slot", -1)
                # Always use add-then-delete pattern to prevent orphaned slots on failure
                new_slot = await asyncio.to_thread(self._vectors.add, vec)
                try:
                    await asyncio.to_thread(self._index.add, new_slot, vec)
                except Exception:
                    await asyncio.to_thread(self._vectors.delete, new_slot)
                    raise
                # Success — remove old slot
                if old_slot >= 0:
                    await asyncio.to_thread(self._index.delete, old_slot)
                    await asyncio.to_thread(self._vectors.delete, old_slot)
                updates["vec_slot"] = new_slot

        await asyncio.to_thread(self._metadata.update, slot, updates)
        return True

    async def delete_neuron(self, neuron_id: str) -> bool:
        """Delete a neuron, its vector, edges, and fiber memberships."""
        result = await asyncio.to_thread(self._metadata.get_by_id, neuron_id)
        if result is None:
            return False

        # WAL: log before write
        await asyncio.to_thread(self._wal.append, WALOp.DELETE_NEURON, {"id": neuron_id})

        slot, meta = result
        vec_slot = meta.get("vec_slot", -1)

        if vec_slot >= 0:
            await asyncio.to_thread(self._vectors.delete, vec_slot)
            await asyncio.to_thread(self._index.delete, vec_slot)

        # Clean up graph edges and fiber memberships
        await asyncio.to_thread(self._graph.delete_neuron_edges, neuron_id)
        await asyncio.to_thread(self._fibers.remove_neuron_from_all, neuron_id)

        await asyncio.to_thread(self._metadata.delete, slot)
        return True

    # --- Vector Search ---

    async def search_similar(
        self,
        query_vector: list[float] | NDArray[np.float32],
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for k most similar neurons by vector.

        Returns list of dicts with neuron metadata + similarity score.
        """
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.shape != (self._dimensions,):
            return []

        slot_ids, distances = await asyncio.to_thread(self._index.search, vec, k)

        results = []
        for slot_id, dist in zip(slot_ids, distances, strict=False):
            meta = await asyncio.to_thread(self._metadata.get_by_slot, slot_id)
            if meta is not None:
                results.append(
                    {
                        **meta,
                        "similarity": round(1.0 - dist, 4),  # cosine: dist = 1 - sim
                        "distance": round(dist, 6),
                    }
                )

        return results

    async def search_similar_batch(
        self,
        query_vectors: NDArray[np.float32],
        k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """Batch search for multiple query vectors."""
        all_labels, all_distances = await asyncio.to_thread(
            self._index.search_batch, query_vectors, k
        )

        batch_results = []
        for i in range(len(query_vectors)):
            results = []
            for j in range(all_labels.shape[1]):
                slot_id = int(all_labels[i][j])
                dist = float(all_distances[i][j])
                meta = await asyncio.to_thread(self._metadata.get_by_slot, slot_id)
                if meta is not None:
                    results.append(
                        {
                            **meta,
                            "similarity": round(1.0 - dist, 4),
                            "distance": round(dist, 6),
                        }
                    )
            batch_results.append(results)

        return batch_results

    # --- Synapse (Graph) API ---

    async def add_synapse(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: str = "related",
        weight: float = 1.0,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a directed synapse between two neurons. Returns edge ID.

        Raises ValueError if source or target neuron does not exist.
        """
        if await asyncio.to_thread(self._metadata.get_by_id, source_id) is None:
            msg = f"Source neuron not found: {source_id!r}"
            raise ValueError(msg)
        if await asyncio.to_thread(self._metadata.get_by_id, target_id) is None:
            msg = f"Target neuron not found: {target_id!r}"
            raise ValueError(msg)

        eid = edge_id or str(uuid.uuid4())

        # WAL: log before write
        wal_payload: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "weight": weight,
            "edge_id": eid,
        }
        if metadata:
            wal_payload["metadata"] = metadata
        await asyncio.to_thread(self._wal.append, WALOp.ADD_SYNAPSE, wal_payload)

        return await asyncio.to_thread(
            self._graph.add_edge,
            source_id,
            target_id,
            edge_type=edge_type,
            weight=weight,
            edge_id=eid,
            metadata=metadata,
        )

    async def get_synapses(
        self,
        neuron_id: str,
        *,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Get synapses for a neuron.

        Args:
            direction: "outgoing", "incoming", or "both"
        """
        if direction == "outgoing":
            return self._graph.get_outgoing(neuron_id)
        elif direction == "incoming":
            return self._graph.get_incoming(neuron_id)
        else:
            out = self._graph.get_outgoing(neuron_id)
            inc = self._graph.get_incoming(neuron_id)
            # Deduplicate self-loop edges that appear in both directions
            seen: set[str] = set()
            merged: list[dict[str, Any]] = []
            for e in out + inc:
                eid = e.get("id", "")
                if eid not in seen:
                    seen.add(eid)
                    merged.append(e)
            return merged

    async def delete_synapse(self, edge_id: str) -> bool:
        """Delete a synapse by edge ID."""
        await asyncio.to_thread(self._wal.append, WALOp.DELETE_SYNAPSE, {"edge_id": edge_id})
        return await asyncio.to_thread(self._graph.delete_edge, edge_id)

    async def update_synapse(self, edge_id: str, updates: dict[str, Any]) -> bool:
        """Update synapse weight, type, or metadata."""
        return self._graph.update_edge(edge_id, updates)

    async def get_neighbors(
        self,
        neuron_id: str,
        *,
        direction: str = "both",
        edge_type: str | None = None,
    ) -> list[str]:
        """Get neighbor neuron IDs."""
        return self._graph.get_neighbors(
            neuron_id,
            direction=direction,
            edge_type=edge_type,
        )

    async def bfs_traverse(
        self,
        start_id: str,
        *,
        max_depth: int = 3,
        direction: str = "outgoing",
        edge_type: str | None = None,
        max_nodes: int = 1000,
    ) -> list[tuple[str, int]]:
        """BFS traversal from a neuron. Returns list of (neuron_id, depth)."""
        return await asyncio.to_thread(
            self._graph.bfs,
            start_id,
            max_depth=max_depth,
            direction=direction,
            edge_type=edge_type,
            max_nodes=max_nodes,
        )

    async def get_subgraph(self, neuron_ids: list[str]) -> list[dict[str, Any]]:
        """Get all edges within a set of neurons (induced subgraph)."""
        return self._graph.get_subgraph(neuron_ids)

    # --- Fiber API ---

    async def add_fiber(
        self,
        name: str,
        *,
        fiber_id: str | None = None,
        fiber_type: str = "cluster",
        description: str = "",
        neuron_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new fiber (neuron collection). Returns fiber ID."""
        fid = fiber_id or str(uuid.uuid4())
        wal_payload: dict[str, Any] = {
            "fiber_id": fid,
            "name": name,
            "fiber_type": fiber_type,
            "description": description,
        }
        if neuron_ids:
            wal_payload["neuron_ids"] = list(neuron_ids)
        if metadata:
            wal_payload["metadata"] = metadata
        await asyncio.to_thread(self._wal.append, WALOp.ADD_FIBER, wal_payload)

        return await asyncio.to_thread(
            self._fibers.add_fiber,
            name,
            fiber_id=fid,
            fiber_type=fiber_type,
            description=description,
            neuron_ids=neuron_ids,
            metadata=metadata,
        )

    async def get_fiber(self, fiber_id: str) -> dict[str, Any] | None:
        """Get fiber by ID."""
        return self._fibers.get_fiber(fiber_id)

    async def find_fibers(
        self,
        *,
        name_contains: str | None = None,
        fiber_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find fibers matching filters."""
        return self._fibers.find_fibers(
            name_contains=name_contains,
            fiber_type=fiber_type,
            limit=limit,
        )

    async def add_neuron_to_fiber(self, fiber_id: str, neuron_id: str) -> bool:
        """Add a neuron to a fiber."""
        return await asyncio.to_thread(self._fibers.add_neuron_to_fiber, fiber_id, neuron_id)

    async def remove_neuron_from_fiber(self, fiber_id: str, neuron_id: str) -> bool:
        """Remove a neuron from a fiber."""
        return await asyncio.to_thread(self._fibers.remove_neuron_from_fiber, fiber_id, neuron_id)

    async def get_fibers_for_neuron(self, neuron_id: str) -> list[str]:
        """Get all fiber IDs containing a neuron."""
        return self._fibers.get_fibers_for_neuron(neuron_id)

    async def delete_fiber(self, fiber_id: str) -> bool:
        """Delete a fiber."""
        await asyncio.to_thread(self._wal.append, WALOp.DELETE_FIBER, {"fiber_id": fiber_id})
        return await asyncio.to_thread(self._fibers.delete_fiber, fiber_id)

    # --- Stats ---

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        return {
            "brain_id": self._brain_id,
            "neuron_count": self._metadata.count,
            "vector_count": self._vectors.count,
            "index_count": self._index.count,
            "synapse_count": self._graph.edge_count,
            "fiber_count": self._fibers.count,
            "dimensions": self._dimensions,
            "is_open": self._is_open,
        }

    # --- Tier Management ---

    def _maybe_promote_sync(self, slot: int, meta: dict[str, Any]) -> None:
        """Promote neuron tier if access pattern warrants it (sync, no thread hop).

        Updates accessed_at and access_count before classifying so the
        tier decision reflects the current access, not stale state.
        """
        from neural_memory.pro.infinitydb.compressor import CompressionTier

        # Update access stats before classifying
        now = _utcnow()
        access_count = meta.get("access_count", 0) + 1
        self._metadata.update(slot, {"accessed_at": now, "access_count": access_count})
        meta["accessed_at"] = now
        meta["access_count"] = access_count

        current_tier_val = meta.get("tier", CompressionTier.ACTIVE)
        try:
            current_tier = CompressionTier(current_tier_val)
        except (ValueError, TypeError):
            current_tier = CompressionTier.ACTIVE

        target = self._tier_manager.should_promote(meta, current_tier)
        if target is not None and target != current_tier:
            self._metadata.update(slot, {"tier": int(target)})
            meta["tier"] = int(target)
            logger.debug(
                "Promoted neuron %s: %s -> %s",
                meta.get("id", "?"),
                current_tier.name,
                target.name,
            )

    async def _maybe_promote(self, slot: int, meta: dict[str, Any]) -> None:
        """Async wrapper for backward compatibility."""
        self._maybe_promote_sync(slot, meta)

    async def demote_sweep(self) -> dict[str, int]:
        """Scan all neurons and demote those whose tier should decrease.

        Collects all demotion decisions first, then applies in a single
        synchronous batch to avoid thread pool contention.

        Returns dict of {tier_name: count_demoted}.
        """
        from neural_memory.pro.infinitydb.compressor import CompressionTier

        all_neurons = await asyncio.to_thread(self._metadata.iter_all)

        # Phase 1: classify (pure, no writes)
        demotions: list[tuple[int, CompressionTier]] = []
        for slot, meta in all_neurons:
            current_tier_val = meta.get("tier", CompressionTier.ACTIVE)
            try:
                current_tier = CompressionTier(current_tier_val)
            except (ValueError, TypeError):
                current_tier = CompressionTier.ACTIVE

            target = self._tier_manager.should_demote(meta, current_tier)
            if target is not None and target != current_tier:
                demotions.append((slot, target))

        if not demotions:
            return {}

        # Phase 2: batch write
        def _apply_demotions() -> None:
            for slot, target in demotions:
                self._metadata.update(slot, {"tier": int(target)})

        await asyncio.to_thread(_apply_demotions)

        # Count results
        demoted: dict[str, int] = {}
        for _, target in demotions:
            demoted[target.name] = demoted.get(target.name, 0) + 1
        logger.info("Demote sweep: %s", demoted)
        return demoted

    async def get_tier_stats(self) -> dict[str, Any]:
        """Get tier distribution statistics and estimated savings."""
        all_neurons = await asyncio.to_thread(self._metadata.iter_all)
        metas = [meta for _, meta in all_neurons]

        stats = self._tier_manager.compute_stats(metas)
        savings = self._tier_manager.estimate_savings(stats)

        return {
            "tiers": stats.as_dict(),
            "savings": savings,
        }

    # --- Multi-dimensional Query ---

    async def query(self, plan: QueryPlan) -> list[dict[str, Any]]:
        """Execute a multi-dimensional query plan.

        Fuses vector similarity, graph proximity, recency, priority,
        and metadata filters using Reciprocal Rank Fusion.
        """
        executor = QueryExecutor(self._metadata, self._index, self._graph)
        return await asyncio.to_thread(executor.execute, plan)

    # --- Suggest ---

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons by content prefix."""
        return await asyncio.to_thread(self._metadata.suggest, prefix, type_filter, limit)
