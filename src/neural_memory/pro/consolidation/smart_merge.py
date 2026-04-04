"""Smart merge consolidation — Pro-grade memory consolidation.

Uses InfinityDB's HNSW search for O(N*k) neighbor-based clustering
instead of brute-force O(N²) pairwise comparison.

Algorithm:
1. For each neuron, find k nearest neighbors via HNSW
2. Cluster neurons that are mutually similar (above threshold)
3. Within each cluster, rank by: priority * activation
4. Merge low-ranked memories into high-ranked anchors
5. Track merge provenance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.pro.infinitydb.engine import InfinityDB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeAction:
    """A single consolidation merge action."""

    anchor_id: str
    merged_ids: tuple[str, ...]
    new_content: str
    reason: str


async def smart_merge(
    db: InfinityDB,
    *,
    similarity_threshold: float = 0.82,
    min_cluster_size: int = 2,
    max_merges: int = 20,
    neighbor_k: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """HNSW-accelerated smart merge consolidation.

    Uses InfinityDB's vector search to find similar neurons efficiently
    instead of O(N²) brute-force comparison.

    Args:
        db: InfinityDB instance (must be open).
        similarity_threshold: Min cosine sim to consider merging.
        min_cluster_size: Min cluster size to trigger merge.
        max_merges: Max merge actions per run.
        neighbor_k: Number of HNSW neighbors to check per neuron.
        dry_run: If True, return planned actions without executing.

    Returns:
        Dict with merge results and statistics.
    """
    # 1. Get all neurons with embeddings
    all_neurons = await db.find_neurons(limit=5000)
    if not all_neurons:
        return {"status": "empty", "merges": 0}

    # Filter to neurons with vectors
    neurons_with_vecs = [n for n in all_neurons if n.get("vec_slot", -1) >= 0]
    if len(neurons_with_vecs) < min_cluster_size:
        return {"status": "insufficient_embeddings", "merges": 0}

    # 2. Build clusters via HNSW neighbor search
    # For each neuron, find its nearest neighbors and group similar ones
    id_to_neuron: dict[str, dict[str, Any]] = {n["id"]: n for n in neurons_with_vecs}

    clusters: dict[str, list[str]] = {}  # anchor_id -> [member_ids]
    assigned: set[str] = set()

    for neuron in neurons_with_vecs:
        nid = neuron["id"]
        if nid in assigned:
            continue

        vec_slot = neuron.get("vec_slot", -1)
        if vec_slot < 0:
            continue

        # Get the vector for this neuron
        vec = db._vectors.get(vec_slot)
        if vec is None:
            continue

        # HNSW search for similar neurons
        similar = await db.search_similar(vec, k=neighbor_k)

        cluster_members = [nid]
        for match in similar:
            mid = match.get("id", "")
            if mid == nid or mid in assigned:
                continue
            similarity = match.get("similarity", 0.0)
            if similarity >= similarity_threshold:
                cluster_members.append(mid)

        if len(cluster_members) >= min_cluster_size:
            clusters[nid] = cluster_members
            assigned.update(cluster_members)

    if not clusters:
        return {"status": "no_clusters", "merges": 0}

    # 3. Plan merge actions
    actions: list[MergeAction] = []
    for member_ids in clusters.values():
        if len(actions) >= max_merges:
            break

        members = [id_to_neuron[mid] for mid in member_ids if mid in id_to_neuron]

        # Rank by priority * activation
        def _score(n: dict[str, Any]) -> float:
            priority = n.get("priority", 5)
            activation = n.get("activation_level", 0.5)
            p = float(priority) if isinstance(priority, (int, float)) else 5.0
            a = float(activation) if isinstance(activation, (int, float)) else 0.5
            return p * a

        members.sort(key=_score, reverse=True)
        anchor = members[0]
        to_merge = members[1:]

        if not to_merge:
            continue

        # Build merged content
        anchor_content = anchor.get("content", "")
        merged_parts = [anchor_content]
        for m in to_merge:
            content = m.get("content", "")
            if content and content not in anchor_content:
                existing = set(anchor_content.split(". "))
                new_sentences = [s for s in content.split(". ") if s.strip() and s not in existing]
                if new_sentences:
                    merged_parts.append(". ".join(new_sentences))

        new_content = ". ".join(merged_parts)
        if len(new_content) > 2000:
            new_content = new_content[:1997] + "..."

        actions.append(
            MergeAction(
                anchor_id=anchor["id"],
                merged_ids=tuple(m["id"] for m in to_merge),
                new_content=new_content,
                reason=f"Merged {len(to_merge)} similar memories (sim>{similarity_threshold})",
            )
        )

    result: dict[str, Any] = {
        "status": "planned" if dry_run else "executed",
        "clusters_found": len(clusters),
        "merge_actions": len(actions),
        "details": [
            {
                "anchor": a.anchor_id,
                "merged": list(a.merged_ids),
                "reason": a.reason,
            }
            for a in actions
        ],
    }

    # 4. Execute merges
    if not dry_run:
        executed = 0
        for action in actions:
            try:
                await db.update_neuron(action.anchor_id, content=action.new_content)
                for mid in action.merged_ids:
                    if mid:
                        await db.update_neuron(mid, neuron_type="consolidated")
                executed += 1
            except Exception:
                logger.warning(
                    "Failed to execute merge for %s",
                    action.anchor_id,
                    exc_info=True,
                )
        result["executed"] = executed

    return result
