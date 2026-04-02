"""Synapse (edge) storage for InfinityDB.

Stores directed edges between neurons as msgpack adjacency lists.
Supports add, delete, query by source/target, and graph traversal.

On-disk: single msgpack file with adjacency dict.
In-memory: dict[source_id -> list[edge_dict]] for O(1) lookup,
           plus reverse index dict[target_id -> list[source_id]] for incoming edges.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import msgpack

logger = logging.getLogger(__name__)


class GraphStore:
    """Directed graph storage for synapses between neurons."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # source_id -> list of edge dicts
        self._adjacency: dict[str, list[dict[str, Any]]] = {}
        # target_id -> list of source_ids (reverse index for incoming queries)
        self._reverse: dict[str, list[str]] = {}
        self._dirty = False
        self._edge_count = 0

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def open(self) -> None:
        """Load graph from disk."""
        # Recover from interrupted flush
        tmp = self._path.with_suffix(".graph.tmp")
        if tmp.exists():
            logger.warning("Recovering from interrupted flush: %s", tmp)
            tmp.replace(self._path)

        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                with open(self._path, "rb") as f:
                    raw = msgpack.unpack(f, raw=False)
                if isinstance(raw, dict):
                    adj = raw.get("adjacency", {})
                    for src, edges in adj.items():
                        if isinstance(edges, list):
                            self._adjacency[src] = edges
                            for edge in edges:
                                tgt = edge.get("target_id", "")
                                if not tgt:
                                    logger.warning(
                                        "Skipping edge with missing target_id in %s", self._path
                                    )
                                    continue
                                self._reverse.setdefault(tgt, []).append(src)
                                self._edge_count += 1
                logger.debug("Loaded %d edges from graph store", self._edge_count)
            except (msgpack.UnpackException, ValueError, TypeError) as e:
                logger.error("Corrupted graph file %s: %s — starting fresh", self._path, e)
                self._adjacency.clear()
                self._reverse.clear()
                self._edge_count = 0
        else:
            logger.debug("No existing graph, starting fresh")

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: str = "related",
        weight: float = 1.0,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a directed edge (synapse) from source to target.

        Returns the edge ID.
        """
        eid = edge_id or str(uuid.uuid4())
        edge: dict[str, Any] = {
            "id": eid,
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "weight": weight,
        }
        if metadata:
            edge["metadata"] = dict(metadata)

        self._adjacency.setdefault(source_id, []).append(edge)
        self._reverse.setdefault(target_id, []).append(source_id)
        self._edge_count += 1
        self._dirty = True
        return eid

    def get_outgoing(self, source_id: str) -> list[dict[str, Any]]:
        """Get all outgoing edges from a neuron."""
        edges = self._adjacency.get(source_id, [])
        return [dict(e) for e in edges]

    def get_incoming(self, target_id: str) -> list[dict[str, Any]]:
        """Get all incoming edges to a neuron."""
        sources = self._reverse.get(target_id, [])
        results: list[dict[str, Any]] = []
        for src in sources:
            for edge in self._adjacency.get(src, []):
                if edge.get("target_id") == target_id:
                    results.append(dict(edge))
        return results

    def get_edges_between(self, source_id: str, target_id: str) -> list[dict[str, Any]]:
        """Get all edges between two specific neurons."""
        edges = self._adjacency.get(source_id, [])
        return [dict(e) for e in edges if e.get("target_id") == target_id]

    def get_neighbors(
        self,
        neuron_id: str,
        *,
        direction: str = "both",
        edge_type: str | None = None,
    ) -> list[str]:
        """Get neighbor neuron IDs.

        Args:
            direction: "outgoing", "incoming", or "both"
            edge_type: filter by edge type (optional)
        """
        valid_directions = {"outgoing", "incoming", "both"}
        if direction not in valid_directions:
            msg = f"Invalid direction: {direction!r}, must be one of {valid_directions}"
            raise ValueError(msg)

        neighbors: set[str] = set()

        if direction in ("outgoing", "both"):
            for edge in self._adjacency.get(neuron_id, []):
                if edge_type is None or edge.get("type") == edge_type:
                    neighbors.add(edge["target_id"])

        if direction in ("incoming", "both"):
            for src in self._reverse.get(neuron_id, []):
                if edge_type is None:
                    neighbors.add(src)
                else:
                    for edge in self._adjacency.get(src, []):
                        if edge.get("target_id") == neuron_id and edge.get("type") == edge_type:
                            neighbors.add(src)
                            break

        return list(neighbors)

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID."""
        for src, edges in self._adjacency.items():
            for i, edge in enumerate(edges):
                if edge.get("id") == edge_id:
                    target = edge.get("target_id", "")
                    edges.pop(i)
                    # Update reverse index
                    if target in self._reverse:
                        rev = self._reverse[target]
                        if src in rev:
                            rev.remove(src)
                            # Only remove if no more edges from src to target
                            has_more = any(
                                e.get("target_id") == target for e in self._adjacency.get(src, [])
                            )
                            if has_more:
                                rev.append(src)
                        if not rev:
                            del self._reverse[target]
                    self._edge_count = max(0, self._edge_count - 1)
                    self._dirty = True
                    return True
        return False

    def delete_neuron_edges(self, neuron_id: str) -> int:
        """Delete all edges involving a neuron (both directions). Returns count deleted."""
        deleted = 0

        # Remove outgoing edges
        if neuron_id in self._adjacency:
            outgoing = self._adjacency.pop(neuron_id)
            for edge in outgoing:
                target = edge.get("target_id", "")
                if target in self._reverse:
                    rev = self._reverse[target]
                    while neuron_id in rev:
                        rev.remove(neuron_id)
                    if not rev:
                        del self._reverse[target]
                deleted += 1

        # Remove incoming edges
        if neuron_id in self._reverse:
            sources = self._reverse.pop(neuron_id)
            for src in set(sources):
                if src in self._adjacency:
                    before = len(self._adjacency[src])
                    self._adjacency[src] = [
                        e for e in self._adjacency[src] if e.get("target_id") != neuron_id
                    ]
                    removed = before - len(self._adjacency[src])
                    deleted += removed
                    if not self._adjacency[src]:
                        del self._adjacency[src]

        if deleted > 0:
            self._edge_count = max(0, self._edge_count - deleted)
            self._dirty = True
        return deleted

    def bfs(
        self,
        start_id: str,
        *,
        max_depth: int = 3,
        direction: str = "outgoing",
        edge_type: str | None = None,
        max_nodes: int = 1000,
    ) -> list[tuple[str, int]]:
        """Breadth-first traversal from start node.

        Returns list of (neuron_id, depth) pairs.
        """
        if max_depth < 1:
            return []

        visited: set[str] = {start_id}
        result: list[tuple[str, int]] = [(start_id, 0)]
        frontier: list[str] = [start_id]

        for depth in range(1, max_depth + 1):
            next_frontier: list[str] = []
            for node in frontier:
                neighbors = self.get_neighbors(node, direction=direction, edge_type=edge_type)
                for nb in neighbors:
                    if nb not in visited and len(result) < max_nodes:
                        visited.add(nb)
                        result.append((nb, depth))
                        next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def get_subgraph(
        self,
        neuron_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get all edges within a set of neurons (induced subgraph)."""
        id_set = set(neuron_ids)
        result: list[dict[str, Any]] = []
        for src in neuron_ids:
            for edge in self._adjacency.get(src, []):
                if edge.get("target_id") in id_set:
                    result.append(dict(edge))
        return result

    def get_edge_by_id(self, edge_id: str) -> dict[str, Any] | None:
        """Get a specific edge by ID."""
        for edges in self._adjacency.values():
            for edge in edges:
                if edge.get("id") == edge_id:
                    return dict(edge)
        return None

    def update_edge(self, edge_id: str, updates: dict[str, Any]) -> bool:
        """Update edge fields (weight, type, metadata)."""
        updatable = {"weight", "type", "metadata"}
        filtered = {k: v for k, v in updates.items() if k in updatable}
        if not filtered:
            return False

        for edges in self._adjacency.values():
            for edge in edges:
                if edge.get("id") == edge_id:
                    edge.update(filtered)
                    self._dirty = True
                    return True
        return False

    def iter_all_edges(self) -> list[dict[str, Any]]:
        """Get all edges as a flat list."""
        result: list[dict[str, Any]] = []
        for edges in self._adjacency.values():
            result.extend(dict(e) for e in edges)
        return result

    def flush(self) -> None:
        """Save graph to disk if dirty. Atomic write."""
        if not self._dirty:
            return
        # Snapshot to prevent concurrent mutation during serialization
        adjacency_snapshot = {k: list(v) for k, v in self._adjacency.items()}
        data = {
            "version": 1,
            "adjacency": adjacency_snapshot,
        }
        tmp_path = self._path.with_suffix(".graph.tmp")
        with open(tmp_path, "wb") as f:
            msgpack.pack(data, f, use_bin_type=True)
        tmp_path.replace(self._path)
        self._dirty = False
        logger.debug("Flushed %d edges to disk", self._edge_count)

    def close(self) -> None:
        """Flush and close."""
        self.flush()
        self._adjacency.clear()
        self._reverse.clear()
        self._edge_count = 0
