"""Predictive priming engine — pre-warm memories from session context.

Anticipates what the brain will need next based on:
1. Activation cache: recent query results carry forward as soft activation
2. Topic priming: session EMA topics pre-warm related neurons
3. Habit priming: query pattern co-occurrence predicts next topic
4. Co-activation priming: Hebbian bindings boost associated neurons

All priming data is in-memory (no hot-path SQLite). Cache is per-session
with exponential decay so stale primes fade quickly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType

if TYPE_CHECKING:
    from neural_memory.engine.session_state import SessionState
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

MAX_CACHE_NEURONS = 200  # Max cached neurons per session
CACHE_DECAY_BASE = 0.7  # Decay per query: level * 0.7^(queries_since)
MIN_ACTIVATION_LEVEL = 0.01  # Prune below this
TOPIC_PRE_WARM_LEVEL = 0.15  # Activation boost for session topic neurons
TOPIC_EMA_THRESHOLD = 0.5  # Topic EMA above this → eligible for priming
MAX_TOPIC_NEURONS = 30  # Max neurons pre-warmed per topic round
HABIT_CONFIDENCE_THRESHOLD = 0.6  # Min confidence for habit-based priming
HABIT_MIN_COUNT = 3  # Min co-occurrence count for habit priming
MAX_HABIT_TOPICS = 3  # Max predicted topics from habits
HABIT_PRIME_LEVEL = 0.12  # Activation boost for habit-predicted neurons
CO_ACTIVATION_MIN_STRENGTH = 0.5  # Min binding strength for co-activation priming
CO_ACTIVATION_MIN_COUNT = 3  # Min co-fire count
CO_ACTIVATION_PRIME_LEVEL = 0.10  # Activation boost for co-activated neurons
MAX_CO_ACTIVATION_PRIMES = 20  # Max neurons primed from co-activation


# ── Priming Metrics ───────────────────────────────────────────────────


@dataclass
class PrimingMetrics:
    """Track priming effectiveness within a session."""

    total_primed: int = 0  # Total neurons primed across all queries
    hits: int = 0  # Primed neurons that appeared in final result
    misses: int = 0  # Primed neurons that did NOT appear in result

    @property
    def hit_rate(self) -> float:
        """Fraction of primed neurons that were useful."""
        if self.total_primed == 0:
            return 0.0
        return self.hits / self.total_primed

    @property
    def aggressiveness_multiplier(self) -> float:
        """Auto-adjust priming aggressiveness based on hit rate.

        High hit rate (>0.3) → boost priming (up to 1.5x).
        Low hit rate (<0.1) → reduce priming (down to 0.5x).
        Default (no data) → 1.0x.
        """
        if self.total_primed < 10:
            return 1.0  # Not enough data
        rate = self.hit_rate
        if rate > 0.3:
            return min(1.5, 1.0 + (rate - 0.3))
        if rate < 0.1:
            return max(0.5, 1.0 - (0.1 - rate) * 5)
        return 1.0


# ── Activation Cache ──────────────────────────────────────────────────


@dataclass
class CachedActivation:
    """A single cached neuron activation from a previous query."""

    neuron_id: str
    level: float  # Activation level when cached
    queries_ago: int = 0  # How many queries since this was cached

    @property
    def decayed_level(self) -> float:
        """Current activation after decay: level * 0.7^(queries_ago)."""
        return self.level * math.pow(CACHE_DECAY_BASE, self.queries_ago)


class ActivationCache:
    """In-memory cache of recent activation results per session.

    After each query, top activated neurons are cached. On the next
    query, cached activations seed the activation map with decayed
    levels — mimicking neural priming in biological memory.
    """

    def __init__(self, max_neurons: int = MAX_CACHE_NEURONS) -> None:
        self._cache: dict[str, CachedActivation] = {}
        self._max_neurons = max_neurons

    def update_from_result(
        self,
        activations: dict[str, float],
    ) -> None:
        """Cache top activations from a completed query.

        Args:
            activations: Map of neuron_id → activation_level from query result.
        """
        # Age all existing entries
        aged: dict[str, CachedActivation] = {}
        for nid, cached in self._cache.items():
            new_entry = CachedActivation(
                neuron_id=nid,
                level=cached.level,
                queries_ago=cached.queries_ago + 1,
            )
            if new_entry.decayed_level >= MIN_ACTIVATION_LEVEL:
                aged[nid] = new_entry

        # Add/update from new results
        for nid, level in activations.items():
            if level >= MIN_ACTIVATION_LEVEL:
                existing = aged.get(nid)
                if existing is None or level > existing.decayed_level:
                    aged[nid] = CachedActivation(
                        neuron_id=nid,
                        level=level,
                        queries_ago=0,
                    )

        # Evict lowest if over capacity
        if len(aged) > self._max_neurons:
            sorted_entries = sorted(aged.items(), key=lambda x: x[1].decayed_level, reverse=True)
            aged = dict(sorted_entries[: self._max_neurons])

        self._cache = aged

    def get_priming_activations(self) -> dict[str, float]:
        """Return cached activations as priming map (neuron_id → decayed level)."""
        result: dict[str, float] = {}
        for nid, cached in self._cache.items():
            decayed = cached.decayed_level
            if decayed >= MIN_ACTIVATION_LEVEL:
                result[nid] = decayed
        return result

    @property
    def size(self) -> int:
        """Number of neurons currently cached."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached activations."""
        self._cache.clear()


# ── Topic Primer ──────────────────────────────────────────────────────


async def prime_from_topics(
    storage: NeuralStorage,
    session_state: SessionState,
    aggressiveness: float = 1.0,
) -> dict[str, float]:
    """Pre-warm neurons related to established session topics.

    Finds neurons tagged with or containing session topic text,
    gives them a small activation boost.

    Args:
        storage: Storage backend for neuron lookup.
        session_state: Current session state with topic EMA.
        aggressiveness: Multiplier from PrimingMetrics (0.5-1.5).

    Returns:
        Map of neuron_id → pre-warm activation level.
    """
    topic_weights = session_state.get_topic_weights()
    if not topic_weights:
        return {}

    # Filter to established topics (EMA above threshold)
    primed_topics = {t: w for t, w in topic_weights.items() if w >= TOPIC_EMA_THRESHOLD}
    if not primed_topics:
        return {}

    priming_map: dict[str, float] = {}
    neurons_per_topic = max(3, int(MAX_TOPIC_NEURONS / max(len(primed_topics), 1)))

    for topic, ema_weight in primed_topics.items():
        try:
            # Find neurons matching this topic (entity or keyword neurons)
            neurons = await storage.find_neurons(
                content_contains=topic,
                limit=neurons_per_topic,
            )
            # Scale boost by topic EMA weight and aggressiveness
            boost = TOPIC_PRE_WARM_LEVEL * min(ema_weight, 1.0) * aggressiveness
            for neuron in neurons:
                existing = priming_map.get(neuron.id, 0.0)
                priming_map[neuron.id] = max(existing, boost)
        except Exception:
            logger.debug("Topic priming lookup failed for '%s'", topic, exc_info=True)

    return priming_map


# ── Habit Primer ──────────────────────────────────────────────────────


async def prime_from_habits(
    storage: NeuralStorage,
    session_state: SessionState,
    aggressiveness: float = 1.0,
) -> dict[str, float]:
    """Pre-warm neurons for topics predicted by habit patterns.

    Uses query pattern mining data (CONCEPT neurons with BEFORE synapses)
    to predict what the user will ask about next based on current session
    topics.

    Args:
        storage: Storage backend.
        session_state: Current session with topic history.
        aggressiveness: Multiplier from PrimingMetrics.

    Returns:
        Map of neuron_id → predicted activation level.
    """
    top_topics = session_state.get_top_topics(limit=5)
    if not top_topics:
        return {}

    priming_map: dict[str, float] = {}
    predicted_count = 0

    for topic in top_topics:
        if predicted_count >= MAX_HABIT_TOPICS:
            break

        try:
            # Find the CONCEPT neuron for this topic
            concept_neurons = await storage.find_neurons(
                content_exact=topic,
                type=NeuronType.CONCEPT,
                limit=1,
            )
            if not concept_neurons:
                continue

            concept = concept_neurons[0]

            # Get outgoing BEFORE synapses (topic_a → topic_b pattern)
            synapses = await storage.get_synapses(
                source_id=concept.id,
                type=SynapseType.BEFORE,
            )

            for syn in synapses:
                if syn.weight < HABIT_CONFIDENCE_THRESHOLD:
                    continue
                if predicted_count >= MAX_HABIT_TOPICS:
                    break

                # The target is the predicted next topic
                target_neuron = await storage.get_neuron(syn.target_id)
                if target_neuron is None:
                    continue

                # Check if this topic is NOT already in the session (truly predictive)
                target_text = target_neuron.content.lower().strip()
                topic_ema = session_state.topic_ema.get(target_text, 0.0)
                if topic_ema >= TOPIC_EMA_THRESHOLD:
                    continue  # Already primed by topic primer

                # Find related neurons for this predicted topic
                related = await storage.find_neurons(
                    content_contains=target_text,
                    limit=10,
                )
                boost = HABIT_PRIME_LEVEL * syn.weight * aggressiveness
                for n in related:
                    existing = priming_map.get(n.id, 0.0)
                    priming_map[n.id] = max(existing, boost)

                predicted_count += 1

        except Exception:
            logger.debug("Habit priming failed for topic '%s'", topic, exc_info=True)

    return priming_map


# ── Co-Activation Primer ─────────────────────────────────────────────


async def prime_from_co_activations(
    storage: NeuralStorage,
    recent_neuron_ids: list[str],
    aggressiveness: float = 1.0,
) -> dict[str, float]:
    """Pre-warm neurons that co-activated with recent result neurons.

    Uses Hebbian binding data: if neuron X was in a recent result
    and X has strong co-activation with Y, Y gets a small boost.

    Args:
        storage: Storage backend with co-activation data.
        recent_neuron_ids: Neuron IDs from the most recent query result.
        aggressiveness: Multiplier from PrimingMetrics.

    Returns:
        Map of neuron_id → co-activation priming level.
    """
    if not recent_neuron_ids:
        return {}

    try:
        co_counts = await storage.get_co_activation_counts(
            min_count=CO_ACTIVATION_MIN_COUNT,
        )
    except Exception:
        return {}

    if not co_counts:
        return {}

    recent_set = set(recent_neuron_ids)
    priming_map: dict[str, float] = {}
    primed_count = 0

    for neuron_a, neuron_b, _count, avg_strength in co_counts:
        if primed_count >= MAX_CO_ACTIVATION_PRIMES:
            break
        if avg_strength < CO_ACTIVATION_MIN_STRENGTH:
            continue

        # Check if one side was in recent results → prime the other
        target: str | None = None
        if neuron_a in recent_set and neuron_b not in recent_set:
            target = neuron_b
        elif neuron_b in recent_set and neuron_a not in recent_set:
            target = neuron_a

        if target is None:
            continue

        boost = CO_ACTIVATION_PRIME_LEVEL * min(avg_strength, 1.0) * aggressiveness
        existing = priming_map.get(target, 0.0)
        priming_map[target] = max(existing, boost)
        primed_count += 1

    return priming_map


# ── Orchestrator ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PrimingResult:
    """Combined result from all priming sources."""

    activation_boosts: dict[str, float]  # neuron_id → combined boost
    source_counts: dict[str, int]  # source → neurons primed count
    total_primed: int = 0

    @staticmethod
    def empty() -> PrimingResult:
        return PrimingResult(activation_boosts={}, source_counts={}, total_primed=0)


async def compute_priming(
    storage: NeuralStorage,
    session_state: SessionState | None,
    activation_cache: ActivationCache | None,
    recent_neuron_ids: list[str] | None = None,
    metrics: PrimingMetrics | None = None,
) -> PrimingResult:
    """Compute combined priming from all sources.

    Merges activation cache, topic priming, habit priming, and
    co-activation priming into a single activation boost map.

    Args:
        storage: Storage backend.
        session_state: Current session (None → skip session-based priming).
        activation_cache: Cache from previous queries (None → skip cache).
        recent_neuron_ids: Neuron IDs from last query (for co-activation).
        metrics: Priming metrics for auto-adjustment.

    Returns:
        PrimingResult with combined boosts and source attribution.
    """
    aggressiveness = metrics.aggressiveness_multiplier if metrics else 1.0
    combined: dict[str, float] = {}
    source_counts: dict[str, int] = {}

    # 1. Activation cache (fastest — pure in-memory)
    if activation_cache is not None:
        cache_boosts = activation_cache.get_priming_activations()
        if cache_boosts:
            for nid, level in cache_boosts.items():
                adjusted = level * aggressiveness
                if adjusted >= MIN_ACTIVATION_LEVEL:
                    combined[nid] = max(combined.get(nid, 0.0), adjusted)
            source_counts["cache"] = len(cache_boosts)

    # 2. Topic-based pre-warming (requires session + storage)
    if session_state is not None and session_state.query_count >= 2:
        try:
            topic_boosts = await prime_from_topics(storage, session_state, aggressiveness)
            if topic_boosts:
                for nid, level in topic_boosts.items():
                    combined[nid] = max(combined.get(nid, 0.0), level)
                source_counts["topic"] = len(topic_boosts)
        except Exception:
            logger.debug("Topic priming failed (non-critical)", exc_info=True)

    # 3. Habit-based priming (requires session + storage)
    if session_state is not None and session_state.query_count >= 3:
        try:
            habit_boosts = await prime_from_habits(storage, session_state, aggressiveness)
            if habit_boosts:
                for nid, level in habit_boosts.items():
                    combined[nid] = max(combined.get(nid, 0.0), level)
                source_counts["habit"] = len(habit_boosts)
        except Exception:
            logger.debug("Habit priming failed (non-critical)", exc_info=True)

    # 4. Co-activation priming (requires recent neuron IDs + storage)
    if recent_neuron_ids:
        try:
            co_boosts = await prime_from_co_activations(storage, recent_neuron_ids, aggressiveness)
            if co_boosts:
                for nid, level in co_boosts.items():
                    combined[nid] = max(combined.get(nid, 0.0), level)
                source_counts["co_activation"] = len(co_boosts)
        except Exception:
            logger.debug("Co-activation priming failed (non-critical)", exc_info=True)

    if not combined:
        return PrimingResult.empty()

    return PrimingResult(
        activation_boosts=combined,
        source_counts=source_counts,
        total_primed=len(combined),
    )


def merge_priming_into_activations(
    anchor_activations: dict[str, float] | None,
    priming: PrimingResult,
) -> dict[str, float]:
    """Merge priming boosts into anchor activation map.

    Priming boosts are additive to existing anchor activations,
    but capped so priming never dominates query-specific anchors.

    Args:
        anchor_activations: Existing activation levels from RRF/anchors.
        priming: Combined priming result.

    Returns:
        Merged activation map.
    """
    if not priming.activation_boosts:
        return anchor_activations or {}

    merged = dict(anchor_activations) if anchor_activations else {}

    for nid, boost in priming.activation_boosts.items():
        existing = merged.get(nid, 0.0)
        # Additive but capped: priming can boost existing anchors
        # but a primed-only neuron stays at priming level (soft anchor)
        merged[nid] = min(existing + boost, 1.0)

    return merged


def record_priming_outcome(
    metrics: PrimingMetrics,
    primed_neuron_ids: set[str],
    result_neuron_ids: set[str],
) -> PrimingMetrics:
    """Update priming metrics based on query outcome.

    Args:
        metrics: Current metrics to update.
        primed_neuron_ids: Neurons that were primed for this query.
        result_neuron_ids: Neurons that appeared in the final result.

    Returns:
        Updated metrics (mutated in-place for session lifecycle).
    """
    if not primed_neuron_ids:
        return metrics

    hits = len(primed_neuron_ids & result_neuron_ids)
    misses = len(primed_neuron_ids) - hits

    metrics.total_primed += len(primed_neuron_ids)
    metrics.hits += hits
    metrics.misses += misses

    return metrics
