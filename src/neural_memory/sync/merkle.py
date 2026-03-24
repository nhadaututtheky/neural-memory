"""Merkle tree builder for efficient delta sync.

Builds a 2-level prefix tree over entity IDs:
  Root → entity_type roots → prefix buckets (entity_id[:2]) → leaf hashes

Leaf hash  = SHA256(entity_id || updated_at || content_hash)
Branch hash = SHA256(sorted child hashes joined)
Root hash  = SHA256(neurons_root || synapses_root || fibers_root)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Sentinel hash for an empty collection (SHA256 of empty string)
_EMPTY_HASH: str = hashlib.sha256(b"").hexdigest()

# Entity types covered by the tree
ENTITY_TYPES: tuple[str, ...] = ("neuron", "synapse", "fiber")


@dataclass(frozen=True)
class MerkleNode:
    """A node in the Merkle hash tree.

    For leaf nodes (individual prefix buckets), ``children`` is empty.
    For branch/root nodes, ``children`` holds the sub-nodes.
    """

    prefix: str
    """Logical prefix: ``""`` for root, ``"neurons"`` for type-root, ``"neurons/0a"`` for bucket."""

    hash: str
    """SHA-256 hex digest for this node."""

    entity_count: int
    """Total entities covered by this node (leaf = bucket count, branch = sum)."""

    children: tuple[MerkleNode, ...] = field(default_factory=tuple)


class MerkleTreeBuilder:
    """Builds and diffs Merkle hash trees from entity data."""

    # ------------------------------------------------------------------
    # Hash primitives
    # ------------------------------------------------------------------

    @staticmethod
    def compute_leaf_hash(entity_id: str, updated_at: str, content_hash: str) -> str:
        """Compute SHA-256 of ``entity_id || updated_at || content_hash``.

        All three components are concatenated with a ``|`` separator so
        that no component can bleed into another.
        """
        raw = f"{entity_id}|{updated_at}|{content_hash}".encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def compute_branch_hash(child_hashes: list[str]) -> str:
        """Compute SHA-256 of sorted child hashes joined by ``|``.

        Sorting makes the hash order-independent (deterministic regardless
        of the order in which children are provided).
        """
        if not child_hashes:
            return _EMPTY_HASH
        joined = "|".join(sorted(child_hashes)).encode()
        return hashlib.sha256(joined).hexdigest()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    @classmethod
    def build_tree(
        cls,
        entities: list[tuple[str, str, str]],
        entity_type: str = "neuron",
    ) -> MerkleNode:
        """Build a Merkle tree for a single entity type.

        Args:
            entities: List of ``(entity_id, updated_at, content_hash)`` tuples.
            entity_type: One of ``"neuron"``, ``"synapse"``, ``"fiber"``.

        Returns:
            A :class:`MerkleNode` representing the root for this entity type.
            The node's prefix is ``entity_type`` (e.g. ``"neurons"``).
        """
        type_prefix = f"{entity_type}s"

        if not entities:
            return MerkleNode(
                prefix=type_prefix,
                hash=_EMPTY_HASH,
                entity_count=0,
            )

        # Sort by entity_id for determinism
        sorted_entities = sorted(entities, key=lambda e: e[0])

        # Group by entity_id[:2] prefix
        buckets: dict[str, list[tuple[str, str, str]]] = {}
        for entity_id, updated_at, content_hash in sorted_entities:
            bucket_key = (
                entity_id[:2].lower() if len(entity_id) >= 2 else entity_id.lower().ljust(2, "0")
            )
            buckets.setdefault(bucket_key, []).append((entity_id, updated_at, content_hash))

        # Build leaf nodes (one per bucket)
        leaf_nodes: list[MerkleNode] = []
        for bucket_key in sorted(buckets.keys()):
            bucket_entities = buckets[bucket_key]
            leaf_hashes = [cls.compute_leaf_hash(eid, upd, ch) for eid, upd, ch in bucket_entities]
            bucket_hash = cls.compute_branch_hash(leaf_hashes)
            leaf_nodes.append(
                MerkleNode(
                    prefix=f"{type_prefix}/{bucket_key}",
                    hash=bucket_hash,
                    entity_count=len(bucket_entities),
                )
            )

        # Build type-root node from leaf hashes
        root_hash = cls.compute_branch_hash([n.hash for n in leaf_nodes])
        total_count = sum(n.entity_count for n in leaf_nodes)

        return MerkleNode(
            prefix=type_prefix,
            hash=root_hash,
            entity_count=total_count,
            children=tuple(leaf_nodes),
        )

    @classmethod
    def build_full_tree(
        cls,
        neurons: list[tuple[str, str, str]],
        synapses: list[tuple[str, str, str]],
        fibers: list[tuple[str, str, str]],
    ) -> MerkleNode:
        """Build a combined root node covering all three entity types.

        Args:
            neurons: ``(id, updated_at, content_hash)`` tuples for neurons.
            synapses: Same for synapses.
            fibers: Same for fibers.

        Returns:
            Root :class:`MerkleNode` with prefix ``""``.
        """
        neuron_tree = cls.build_tree(neurons, "neuron")
        synapse_tree = cls.build_tree(synapses, "synapse")
        fiber_tree = cls.build_tree(fibers, "fiber")

        root_hash = cls.compute_branch_hash([neuron_tree.hash, synapse_tree.hash, fiber_tree.hash])
        total_count = neuron_tree.entity_count + synapse_tree.entity_count + fiber_tree.entity_count

        return MerkleNode(
            prefix="",
            hash=root_hash,
            entity_count=total_count,
            children=(neuron_tree, synapse_tree, fiber_tree),
        )

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    @classmethod
    def compute_diff(cls, local: MerkleNode, remote: MerkleNode) -> list[str]:
        """Return list of prefixes where the two trees differ.

        Walks the tree top-down. If hashes match at a node, the entire
        subtree is skipped (no diff). If they differ, we recurse into
        children. Leaf nodes (no children) are returned as-is.

        Args:
            local: The local Merkle node.
            remote: The remote Merkle node.

        Returns:
            List of prefix strings that differ. Empty list means identical.
        """
        if local.hash == remote.hash:
            return []

        # Build a map of remote children by prefix for O(1) lookup
        remote_by_prefix = {n.prefix: n for n in remote.children}
        local_by_prefix = {n.prefix: n for n in local.children}

        # Collect all prefixes from both sides
        all_prefixes = set(remote_by_prefix) | set(local_by_prefix)

        if not all_prefixes:
            # Both are leaf nodes (no children) with different hashes
            return [local.prefix]

        diffs: list[str] = []
        for prefix in sorted(all_prefixes):
            local_child = local_by_prefix.get(prefix)
            remote_child = remote_by_prefix.get(prefix)

            if local_child is None:
                # Prefix exists only on remote — needs to be fetched
                diffs.append(prefix)
            elif remote_child is None:
                # Prefix exists only locally — needs to be pushed
                diffs.append(prefix)
            else:
                # Both have this prefix — recurse if hashes differ
                child_diffs = cls.compute_diff(local_child, remote_child)
                diffs.extend(child_diffs)

        return diffs
