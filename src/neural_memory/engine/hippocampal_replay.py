"""Hippocampal replay — biased replay consolidation with LTP/LTD.

During sleep (or consolidation), the hippocampus replays recent episodes.
Synapses traversed during replay are strengthened (LTP), while neighboring
non-replayed synapses weaken slightly (LTD). This naturally promotes
well-connected pathways and prunes weak ones.

Neuroscience basis: sharp-wave ripple replay in hippocampal CA1 —
sequential reactivation of place cells strengthens recent memories
during NREM sleep.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Result of a hippocampal replay consolidation run."""

    episodes_replayed: int = 0
    synapses_strengthened: int = 0
    synapses_weakened: int = 0


async def hippocampal_replay(
    storage: NeuralStorage,
    config: BrainConfig,
    seed: int | None = None,
    dry_run: bool = False,
) -> ReplayResult:
    """Run hippocampal replay on recent fibers.

    1. Sample recent fibers (last 24h), weighted by priority/salience
    2. For each episode: find sequential synapses between fiber neurons
    3. LTP: strengthen replayed synapses by ltp_factor
    4. LTD: weaken non-replayed neighbor synapses by ltd_factor

    Args:
        storage: Storage backend
        config: Brain configuration
        seed: Random seed for deterministic replay
        dry_run: If True, compute but don't write changes

    Returns:
        ReplayResult with counts
    """
    replay_enabled = getattr(config, "replay_enabled", True)
    if isinstance(replay_enabled, bool) and not replay_enabled:
        return ReplayResult()

    ltp_factor = getattr(config, "replay_ltp_factor", 1.1)
    ltd_factor = getattr(config, "replay_ltd_factor", 0.98)
    ltp_factor = float(ltp_factor) if isinstance(ltp_factor, (int, float)) else 1.1
    ltd_factor = float(ltd_factor) if isinstance(ltd_factor, (int, float)) else 0.98

    # Configurable replay window and episode cap
    window_hours = getattr(config, "replay_window_hours", 24.0)
    window_hours = float(window_hours) if isinstance(window_hours, (int, float)) else 24.0
    max_episodes = getattr(config, "replay_max_episodes", 20)
    max_episodes = int(max_episodes) if isinstance(max_episodes, (int, float)) else 20

    now = utcnow()
    window_start = now - timedelta(hours=window_hours)

    recent_fibers = await storage.find_fibers(
        time_overlaps=(window_start, now),
        limit=100,
    )

    if not recent_fibers:
        logger.debug("Hippocampal replay: no recent fibers found")
        return ReplayResult()

    # Sort by salience (priority proxy) — higher salience = replayed first
    recent_fibers.sort(key=lambda f: f.salience or 0.0, reverse=True)

    rng = random.Random(seed)
    episodes_replayed = 0
    total_strengthened = 0
    total_weakened = 0

    replay_count = min(max_episodes, len(recent_fibers))
    selected = recent_fibers[:replay_count]

    # Shuffle to add variability (like real replay)
    rng.shuffle(selected)

    for fiber in selected:
        neuron_ids = list(fiber.neuron_ids or [])
        if len(neuron_ids) < 2:
            continue

        # Find synapses between neurons in this fiber (sequential path)
        replayed_synapse_ids: set[str] = set()

        # Get all synapses connected to fiber neurons
        all_neighbor_synapses = []
        for nid in neuron_ids:
            synapses = await storage.get_synapses(source_id=nid)
            all_neighbor_synapses.extend(synapses)
            synapses_in = await storage.get_synapses(target_id=nid)
            all_neighbor_synapses.extend(synapses_in)

        # Deduplicate
        seen: set[str] = set()
        unique_synapses = []
        for s in all_neighbor_synapses:
            if s.id not in seen:
                seen.add(s.id)
                unique_synapses.append(s)

        # Classify: replayed (in fiber) vs neighbor (connected but not in fiber)
        neuron_set = set(neuron_ids)
        for syn in unique_synapses:
            if syn.source_id in neuron_set and syn.target_id in neuron_set:
                replayed_synapse_ids.add(syn.id)

        # LTP: strengthen replayed synapses
        strengthened = 0
        weakened = 0

        for syn in unique_synapses:
            if syn.id in replayed_synapse_ids:
                # LTP — strengthen
                new_weight = min(1.0, syn.weight * ltp_factor)
                if new_weight != syn.weight and not dry_run:
                    updated = replace(syn, weight=new_weight)
                    try:
                        await storage.update_synapse(updated)
                        strengthened += 1
                    except Exception:
                        logger.debug("LTP update failed for synapse %s", syn.id[:12])
                elif new_weight != syn.weight:
                    strengthened += 1
            else:
                # LTD — weaken non-replayed neighbors
                new_weight = max(0.01, syn.weight * ltd_factor)
                if new_weight != syn.weight and not dry_run:
                    updated = replace(syn, weight=new_weight)
                    try:
                        await storage.update_synapse(updated)
                        weakened += 1
                    except Exception:
                        logger.debug("LTD update failed for synapse %s", syn.id[:12])
                elif new_weight != syn.weight:
                    weakened += 1

        if strengthened > 0 or weakened > 0:
            episodes_replayed += 1
            total_strengthened += strengthened
            total_weakened += weakened

    logger.debug(
        "Hippocampal replay: %d episodes, %d LTP, %d LTD",
        episodes_replayed,
        total_strengthened,
        total_weakened,
    )

    return ReplayResult(
        episodes_replayed=episodes_replayed,
        synapses_strengthened=total_strengthened,
        synapses_weakened=total_weakened,
    )
