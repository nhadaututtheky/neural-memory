"""Merkle hash tree persistence mixin — dialect-agnostic."""

from __future__ import annotations

import logging

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.sync.merkle import MerkleTreeBuilder
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Entity types tracked by the Merkle tree
_ENTITY_TYPES: tuple[str, ...] = ("neuron", "synapse", "fiber")


class MerkleMixin:
    """Mixin providing Merkle hash tree persistence for incremental delta sync.

    Requires the ``merkle_hashes`` table created by schema migration v36.
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute_merkle_root(
        self,
        entity_type: str,
        *,
        is_pro: bool = False,
    ) -> str | None:
        """Compute and cache the Merkle root hash for a single entity type.

        Queries all entities of the given type, recomputes the full tree,
        upserts all prefix hashes into ``merkle_hashes``, then returns the
        root hash string.

        Args:
            entity_type: One of ``"neuron"``, ``"synapse"``, ``"fiber"``.
            is_pro: Must be ``True`` to enable computation. Free tier returns ``None``.

        Returns:
            SHA-256 root hash hex string, or ``None`` if not Pro.
        """
        if not is_pro:
            return None

        if entity_type not in _ENTITY_TYPES:
            raise ValueError(
                f"Unknown entity_type: {entity_type!r}. Must be one of {_ENTITY_TYPES}."
            )

        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        # Fetch entities with their key fields
        entities = await _fetch_entities(d, brain_id, entity_type)

        # Build tree
        tree = MerkleTreeBuilder.build_tree(entities, entity_type)

        # Upsert root node
        upsert_sql = d.upsert_sql(
            "merkle_hashes",
            ["brain_id", "entity_type", "prefix", "hash", "entity_count", "updated_at"],
            ["brain_id", "entity_type", "prefix"],
            ["hash", "entity_count", "updated_at"],
        )
        await d.execute(
            upsert_sql,
            [brain_id, entity_type, tree.prefix, tree.hash, tree.entity_count, now],
        )

        # Upsert bucket nodes
        for child in tree.children:
            await d.execute(
                upsert_sql,
                [brain_id, entity_type, child.prefix, child.hash, child.entity_count, now],
            )

        return tree.hash

    async def get_merkle_tree(
        self,
        entity_type: str,
        *,
        is_pro: bool = False,
    ) -> dict[str, str]:
        """Return the cached ``{prefix: hash}`` map for an entity type.

        Returns an empty dict if not Pro or if no hashes have been computed yet.

        Args:
            entity_type: One of ``"neuron"``, ``"synapse"``, ``"fiber"``.
            is_pro: Must be ``True`` to read cached hashes.

        Returns:
            Dict mapping prefix string to SHA-256 hash string.
        """
        if not is_pro:
            return {}

        d = self._dialect
        brain_id = self._get_brain_id()

        rows = await d.fetch_all(
            f"SELECT prefix, hash FROM merkle_hashes WHERE brain_id = {d.ph(1)} AND entity_type = {d.ph(2)}",
            [brain_id, entity_type],
        )
        return {str(r["prefix"]): str(r["hash"]) for r in rows}

    async def invalidate_merkle_prefix(
        self,
        entity_type: str,
        entity_id: str,
        *,
        is_pro: bool = False,
    ) -> None:
        """Delete cached hashes for the bucket containing ``entity_id``.

        This is called on entity insert/update/delete so the next call to
        :meth:`compute_merkle_root` recomputes only the affected branch.
        In the current implementation we delete both the bucket row and
        the type-root row, forcing a full recompute of that entity type on
        the next sync cycle. This is safe and simpler than partial updates.

        Args:
            entity_type: One of ``"neuron"``, ``"synapse"``, ``"fiber"``.
            entity_id: The ID of the entity that changed.
            is_pro: Must be ``True`` to perform invalidation.
        """
        if not is_pro:
            return

        d = self._dialect
        brain_id = self._get_brain_id()

        type_prefix = f"{entity_type}s"
        bucket_key = (
            entity_id[:2].lower() if len(entity_id) >= 2 else entity_id.lower().ljust(2, "0")
        )
        bucket_prefix = f"{type_prefix}/{bucket_key}"

        # Delete the affected bucket and the type-root (both are stale)
        await d.execute(
            f"""DELETE FROM merkle_hashes
               WHERE brain_id = {d.ph(1)} AND entity_type = {d.ph(2)} AND prefix IN ({d.ph(3)}, {d.ph(4)})""",
            [brain_id, entity_type, type_prefix, bucket_prefix],
        )

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        """Get combined root hash across all entity types.

        Reads the cached type-root hashes for neurons/synapses/fibers and
        computes a combined root hash without hitting the entity tables.
        Returns ``None`` if not Pro or if any type-root is missing (stale cache).

        Args:
            is_pro: Must be ``True`` to compute the combined root.

        Returns:
            Combined SHA-256 root hash string, or ``None``.
        """
        if not is_pro:
            return None

        d = self._dialect
        brain_id = self._get_brain_id()

        type_prefixes = ("neurons", "synapses", "fibers")
        hashes: list[str] = []

        for type_prefix in type_prefixes:
            entity_type = type_prefix.rstrip("s")  # "neurons" -> "neuron" etc.
            row = await d.fetch_one(
                f"SELECT hash FROM merkle_hashes WHERE brain_id = {d.ph(1)} AND entity_type = {d.ph(2)} AND prefix = {d.ph(3)}",
                [brain_id, entity_type, type_prefix],
            )
            if row is None:
                # Cache is incomplete — caller must call compute_merkle_root first
                return None
            hashes.append(str(row["hash"]))

        return MerkleTreeBuilder.compute_branch_hash(hashes)

    async def get_bucket_entity_ids(
        self,
        entity_type: str,
        prefix: str,
        *,
        is_pro: bool = False,
    ) -> list[str]:
        """Return all entity IDs in the given bucket prefix.

        Used by the Merkle sync protocol for delete detection: both sides
        exchange the full list of entity IDs per differing bucket so the
        receiver can compute inserts, updates, and deletes.

        Args:
            entity_type: One of ``"neuron"``, ``"synapse"``, ``"fiber"``.
            prefix: Bucket prefix like ``"neurons/0a"``.
            is_pro: Must be ``True`` to query.

        Returns:
            Sorted list of entity ID strings in the bucket, or empty if not Pro.
        """
        if not is_pro:
            return []

        d = self._dialect
        brain_id = self._get_brain_id()

        table_map: dict[str, str] = {
            "neuron": "neurons",
            "synapse": "synapses",
            "fiber": "fibers",
        }
        table = table_map.get(entity_type)
        if not table:
            return []

        # Extract 2-char hex prefix from "neurons/0a" -> "0a"
        parts = prefix.split("/")
        if len(parts) != 2:
            return []
        hex_prefix = parts[1].lower()

        # Filter by entity_id[:2] matching hex_prefix
        # Use LOWER(SUBSTR(...)) for case-insensitive matching
        rows = await d.fetch_all(
            f"SELECT id FROM {table} WHERE brain_id = {d.ph(1)} AND LOWER(SUBSTR(id, 1, 2)) = {d.ph(2)}",
            [brain_id, hex_prefix],
        )
        return sorted(str(r["id"]) for r in rows)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


async def _fetch_entities(
    d: Dialect,
    brain_id: str,
    entity_type: str,
) -> list[tuple[str, str, str]]:
    """Fetch ``(entity_id, updated_at, content_hash)`` tuples for hashing.

    ``content_hash`` is stored as an integer (SimHash) in neurons/synapses;
    we convert it to a string for consistent hashing.
    """
    table_map: dict[str, str] = {
        "neuron": "neurons",
        "synapse": "synapses",
        "fiber": "fibers",
    }
    table = table_map[entity_type]

    if entity_type == "fiber":
        # Fibers don't have a SimHash column; use summary as a proxy
        rows = await d.fetch_all(
            f"SELECT id, updated_at, COALESCE(summary, '') AS hash_value FROM fibers WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )
    else:
        # neurons and synapses have content_hash (SimHash integer)
        rows = await d.fetch_all(
            f"SELECT id, updated_at, CAST(content_hash AS TEXT) AS hash_value FROM {table} WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )

    return [(str(r["id"]), str(r["updated_at"] or ""), str(r["hash_value"] or "")) for r in rows]
