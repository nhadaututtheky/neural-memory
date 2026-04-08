"""Retrieval reconsolidation — recalled memories absorb current context.

Every time a memory is recalled, it's briefly labile (modifiable).
This module exploits that window to:
  1. Track recall contexts (rolling window of last 5)
  2. Detect contextual drift (Jaccard distance from original tags)
  3. Create bridge synapses when memory is recalled in a new domain
  4. Increment reconsolidation count for lifecycle tracking

Neuroscience basis: memory reconsolidation — recalled memories are
destabilized and re-stored with current contextual associations,
creating new retrieval pathways over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_CONTEXT_TRAIL = 5
_MAX_BRIDGE_PER_RECALL = 3
_CONTEXT_ANCHOR_CACHE: dict[str, str] = {}  # entity_key -> neuron_id
_CACHE_MAX = 100


@dataclass(frozen=True)
class ReconsolidationResult:
    """Result of reconsolidation for a single neuron."""

    neuron_id: str
    drift_score: float
    bridge_created: bool
    reconsolidation_count: int


def _jaccard_distance(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard distance between two sets (0.0 = identical, 1.0 = disjoint)."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


async def _find_or_create_context_anchor(
    storage: NeuralStorage,
    entity: str,
    brain_id: str,
) -> str | None:
    """Find or create a lightweight CONCEPT neuron as context anchor.

    Returns neuron ID, or None if creation fails.
    """
    cache_key = f"{brain_id}:{entity}"

    # Check cache first
    if cache_key in _CONTEXT_ANCHOR_CACHE:
        return _CONTEXT_ANCHOR_CACHE[cache_key]

    # Search existing concept neurons
    existing = await storage.find_neurons(
        content_exact=entity,
        type=NeuronType.CONCEPT,
        limit=1,
    )
    if existing:
        neuron_id = existing[0].id
        if len(_CONTEXT_ANCHOR_CACHE) < _CACHE_MAX:
            _CONTEXT_ANCHOR_CACHE[cache_key] = neuron_id
        return neuron_id

    # Create new lightweight concept neuron
    try:
        anchor = Neuron.create(
            type=NeuronType.CONCEPT,
            content=entity,
            metadata={"_reconsolidation_anchor": True},
        )
        await storage.add_neuron(anchor)
        if len(_CONTEXT_ANCHOR_CACHE) < _CACHE_MAX:
            _CONTEXT_ANCHOR_CACHE[cache_key] = anchor.id
        return anchor.id
    except Exception:
        logger.debug("Failed to create context anchor for %s", entity, exc_info=True)
        return None


async def reconsolidate_on_recall(
    fiber_id: str,
    anchor_neuron_id: str,
    query_tags: set[str],
    query_entities: list[str],
    storage: NeuralStorage,
    config: BrainConfig,
    brain_id: str = "",
) -> ReconsolidationResult | None:
    """Reconsolidate a single recalled memory with current context.

    Args:
        fiber_id: ID of the recalled fiber
        anchor_neuron_id: Anchor neuron of the fiber
        query_tags: Tags extracted from the current query
        query_entities: Entities from the current query
        storage: Storage backend
        config: Brain configuration
        brain_id: Current brain ID for anchor caching

    Returns:
        ReconsolidationResult or None if skipped
    """
    threshold = getattr(config, "reconsolidation_drift_threshold", 0.6)
    threshold = float(threshold) if isinstance(threshold, (int, float)) else 0.6

    # Get anchor neuron
    try:
        neuron = await storage.get_neuron(anchor_neuron_id)
    except Exception:
        return None

    if neuron is None:
        return None

    meta = dict(neuron.metadata) if neuron.metadata else {}

    # Step 1: Update context trail (rolling window of last N recall contexts)
    trail: list[dict[str, Any]] = meta.get("_context_trail", [])
    if not isinstance(trail, list):
        trail = []

    trail_entry = {
        "tags": sorted(query_tags)[:10],
        "entities": query_entities[:5],
        "ts": utcnow().isoformat(),
    }
    trail.append(trail_entry)
    if len(trail) > _MAX_CONTEXT_TRAIL:
        trail = trail[-_MAX_CONTEXT_TRAIL:]

    # Step 2: Compute contextual drift
    original_tags = set(meta.get("_original_tags", []))
    if not original_tags:
        # Use auto_tags from fiber if available — fall back to empty
        original_tags = set()

    drift = _jaccard_distance(original_tags, query_tags) if query_tags else 0.0

    # Step 3: Create bridge synapse if drift exceeds soft threshold
    # Graduated response: start at 80% of threshold with weaker bridges
    bridge_created = False
    soft_threshold = threshold * 0.8
    if drift > soft_threshold and query_entities:
        # Scale bridge weight: 0.15 at soft threshold, up to 0.3 at full drift
        drift_ratio = min(1.0, (drift - soft_threshold) / (threshold * 0.5))
        base_weight = 0.15 + 0.15 * drift_ratio
        bridges = 0
        for entity in query_entities[:_MAX_BRIDGE_PER_RECALL]:
            anchor_id = await _find_or_create_context_anchor(storage, entity, brain_id)
            if anchor_id and anchor_id != anchor_neuron_id:
                try:
                    bridge = Synapse.create(
                        source_id=anchor_neuron_id,
                        target_id=anchor_id,
                        type=SynapseType.RELATED_TO,
                        weight=min(0.3, base_weight),
                        metadata={
                            "_reconsolidation_bridge": True,
                            "drift_score": round(drift, 3),
                        },
                    )
                    await storage.add_synapse(bridge)
                    bridges += 1
                    bridge_created = True
                except (ValueError, Exception):
                    logger.debug("Bridge synapse creation skipped (may exist)")

        if bridges:
            logger.debug(
                "Reconsolidation: %d bridge(s) for neuron %s (drift=%.2f)",
                bridges,
                anchor_neuron_id[:12],
                drift,
            )

    # Step 4: Update metadata
    recon_count = meta.get("_reconsolidation_count", 0)
    recon_count = int(recon_count) if isinstance(recon_count, (int, float)) else 0
    recon_count += 1

    new_meta = {
        **meta,
        "_context_trail": trail,
        "_reconsolidation_count": recon_count,
    }
    # Clear stale mark — reconsolidated memory absorbed fresh context
    new_meta.pop("_stale", None)

    updated_neuron = Neuron(
        id=neuron.id,
        type=neuron.type,
        content=neuron.content,
        metadata=new_meta,
        content_hash=neuron.content_hash,
        ephemeral=neuron.ephemeral,
    )
    try:
        await storage.update_neuron(updated_neuron)
    except Exception:
        logger.debug("Reconsolidation metadata update failed", exc_info=True)

    return ReconsolidationResult(
        neuron_id=anchor_neuron_id,
        drift_score=drift,
        bridge_created=bridge_created,
        reconsolidation_count=recon_count,
    )
