"""Interference forgetting — detect and resolve memory competition.

When similar memories accumulate, they interfere with each other during
retrieval. This module detects three interference types and resolves them:
- Retroactive: new memory overwrites recall of old similar memory
- Proactive: old memory blocks recall of new similar memory
- Fan effect: too many similar memories dilute recall quality

Neuroscience basis: interference theory (McGeoch, 1932; Anderson & Neely, 1996)
— forgetting occurs not from decay but from competition between similar traces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.simhash import hamming_distance, simhash

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.neuron import Neuron
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# SimHash distance thresholds
_NEAR_DUPLICATE_THRESHOLD = 8  # Too similar (handled by dedup)
_INTERFERENCE_THRESHOLD = 18  # Similar enough to interfere


class InterferenceType(StrEnum):
    RETROACTIVE = "retroactive"
    PROACTIVE = "proactive"
    FAN_EFFECT = "fan_effect"


@dataclass(frozen=True)
class InterferenceResult:
    """A detected interference between two memories."""

    neuron_id: str
    score: float  # 0.0-1.0, higher = stronger interference
    interference_type: InterferenceType


@dataclass(frozen=True)
class ResolutionReport:
    """Summary of interference resolution actions."""

    contradicts_created: int = 0
    priorities_boosted: int = 0
    fan_effects_flagged: int = 0
    total_detected: int = 0


async def detect_interference(
    new_neuron: Neuron,
    storage: NeuralStorage,
    config: BrainConfig,
    max_candidates: int = 20,
) -> list[InterferenceResult]:
    """Detect memories that may interfere with a new memory.

    Args:
        new_neuron: Newly encoded neuron
        storage: Storage backend
        config: Brain configuration
        max_candidates: Max similar neurons to check

    Returns:
        List of interference results
    """
    enabled = getattr(config, "interference_detection_enabled", False)
    if not isinstance(enabled, bool):
        enabled = False
    if not enabled:
        return []

    new_hash = simhash(new_neuron.content)
    new_tags = set(new_neuron.metadata.get("tags", [])) if new_neuron.metadata else set()

    if not new_tags:
        return []

    # Find neurons with overlapping tags (filter in-memory, no tags= param)
    all_neurons = await storage.find_neurons(limit=max(max_candidates * 5, 2000))
    candidates = [
        n for n in all_neurons
        if n.metadata and set(n.metadata.get("tags", [])) & new_tags
    ][:max_candidates * 2]

    results: list[InterferenceResult] = []
    same_tag_count = 0

    for candidate in candidates:
        if candidate.id == new_neuron.id:
            continue

        cand_hash = simhash(candidate.content)
        dist = hamming_distance(new_hash, cand_hash)

        # Skip near-duplicates (handled by dedup pipeline)
        if dist < _NEAR_DUPLICATE_THRESHOLD:
            continue

        # Count same-tag memories for fan effect
        same_tag_count += 1

        if dist <= _INTERFERENCE_THRESHOLD:
            # Similar enough to interfere
            score = 1.0 - (dist / _INTERFERENCE_THRESHOLD)

            # Determine type based on creation time
            if new_neuron.created_at >= candidate.created_at:
                itype = InterferenceType.RETROACTIVE
            else:
                itype = InterferenceType.PROACTIVE

            results.append(InterferenceResult(
                neuron_id=candidate.id,
                score=round(score, 4),
                interference_type=itype,
            ))

    # Check fan effect
    fan_threshold = getattr(config, "fan_effect_threshold", 15)
    fan_threshold = int(fan_threshold) if isinstance(fan_threshold, (int, float)) else 15
    if same_tag_count >= fan_threshold:
        results.append(InterferenceResult(
            neuron_id=new_neuron.id,
            score=min(1.0, same_tag_count / (fan_threshold * 2)),
            interference_type=InterferenceType.FAN_EFFECT,
        ))

    return results[:max_candidates]


async def resolve_interference(
    results: list[InterferenceResult],
    new_neuron: Neuron,
    storage: NeuralStorage,
    config: BrainConfig,
    dry_run: bool = False,
) -> ResolutionReport:
    """Resolve detected interference.

    - Retroactive: create CONTRADICTS synapse, reduce old memory weights
    - Proactive: boost new memory priority
    - Fan effect: flag for consolidation merge

    Args:
        results: Interference detection results
        new_neuron: The new neuron causing interference
        storage: Storage backend
        config: Brain configuration
        dry_run: If True, count actions but don't write

    Returns:
        ResolutionReport with action counts
    """
    contradicts_created = 0
    priorities_boosted = 0
    fan_effects_flagged = 0

    for r in results:
        if r.interference_type == InterferenceType.RETROACTIVE:
            if not dry_run:
                syn = Synapse.create(
                    source_id=new_neuron.id,
                    target_id=r.neuron_id,
                    type=SynapseType.CONTRADICTS,
                    weight=r.score * 0.5,
                )
                await storage.add_synapse(syn)

                # Reduce old memory's outgoing weights by 5%
                old_synapses = await storage.get_synapses(source_id=r.neuron_id)
                for old_syn in old_synapses[:10]:
                    from dataclasses import replace
                    new_weight = max(0.01, old_syn.weight * 0.95)
                    if new_weight != old_syn.weight:
                        updated = replace(old_syn, weight=new_weight)
                        try:
                            await storage.update_synapse(updated)
                        except Exception:
                            logger.debug("Weight reduction failed for %s", old_syn.id[:12])
            contradicts_created += 1

        elif r.interference_type == InterferenceType.PROACTIVE:
            # Boost new memory priority (handled at encoding level)
            priorities_boosted += 1

        elif r.interference_type == InterferenceType.FAN_EFFECT:
            fan_effects_flagged += 1
            logger.info(
                "Fan effect detected for neuron %s: %d similar memories",
                new_neuron.id[:12],
                int(r.score * 30),
            )

    return ResolutionReport(
        contradicts_created=contradicts_created,
        priorities_boosted=priorities_boosted,
        fan_effects_flagged=fan_effects_flagged,
        total_detected=len(results),
    )


async def batch_interference_scan(
    storage: NeuralStorage,
    config: BrainConfig,
    dry_run: bool = False,
) -> ResolutionReport:
    """Scan for fan effects across tag clusters.

    Used by consolidation strategy.
    """
    enabled = getattr(config, "interference_detection_enabled", False)
    if not isinstance(enabled, bool):
        enabled = False
    if not enabled:
        return ResolutionReport()

    fan_threshold = getattr(config, "fan_effect_threshold", 15)
    fan_threshold = int(fan_threshold) if isinstance(fan_threshold, (int, float)) else 15

    # Scan tag clusters for overcrowding
    all_neurons = await storage.find_neurons(limit=2000)
    tag_counts: dict[str, int] = {}
    for n in all_neurons:
        ntags = n.metadata.get("tags", []) if n.metadata else []
        for t in ntags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    fan_effects = 0
    for tag, count in tag_counts.items():
        if count >= fan_threshold:
            fan_effects += 1
            logger.info("Fan effect: tag '%s' has %d memories (threshold: %d)", tag, count, fan_threshold)

    return ResolutionReport(
        fan_effects_flagged=fan_effects,
        total_detected=fan_effects,
    )
