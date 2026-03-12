"""Tests for Phase 3: Predictive Priming engine.

Tests cover:
- ActivationCache: decay, eviction, update, priming output
- Topic-based pre-warming from session EMA
- Habit-based priming from query pattern co-occurrence
- Co-activation priming from Hebbian bindings
- PrimingMetrics: hit rate, aggressiveness auto-adjustment
- Orchestrator: compute_priming combining all sources
- merge_priming_into_activations
- record_priming_outcome
- SessionState priming fields
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.engine.priming import (
    CACHE_DECAY_BASE,
    CO_ACTIVATION_PRIME_LEVEL,
    HABIT_PRIME_LEVEL,
    MAX_CO_ACTIVATION_PRIMES,
    MAX_HABIT_TOPICS,
    TOPIC_PRE_WARM_LEVEL,
    ActivationCache,
    CachedActivation,
    PrimingMetrics,
    PrimingResult,
    compute_priming,
    merge_priming_into_activations,
    prime_from_co_activations,
    prime_from_habits,
    prime_from_topics,
    record_priming_outcome,
)
from neural_memory.engine.session_state import SessionState

# ── Helpers ───────────────────────────────────────────────────────────


def _make_session(
    session_id: str = "test-session",
    topic_ema: dict[str, float] | None = None,
    query_count: int = 5,
) -> SessionState:
    """Create a SessionState with preset topic EMA."""
    state = SessionState(session_id=session_id)
    state.query_count = query_count
    if topic_ema:
        state.topic_ema = dict(topic_ema)
    return state


@dataclass
class FakeNeuron:
    id: str
    content: str
    type: Any = None


@dataclass
class FakeSynapse:
    source_id: str
    target_id: str
    weight: float = 0.8
    type: Any = None


# ── TestActivationCache ──────────────────────────────────────────────


class TestActivationCache:
    """Tests for ActivationCache — in-memory decay cache."""

    def test_empty_cache_returns_no_priming(self):
        cache = ActivationCache()
        assert cache.get_priming_activations() == {}
        assert cache.size == 0

    def test_update_from_result_caches_neurons(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.8, "n2": 0.5, "n3": 0.3})
        assert cache.size == 3
        priming = cache.get_priming_activations()
        assert "n1" in priming
        assert "n2" in priming
        assert "n3" in priming

    def test_decay_over_queries(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 1.0})

        # After 1 more query, n1 should be decayed
        cache.update_from_result({"n2": 0.5})
        priming = cache.get_priming_activations()

        # n1 was cached 1 query ago → 1.0 * 0.7^1 = 0.7
        assert abs(priming["n1"] - CACHE_DECAY_BASE) < 0.01
        # n2 is fresh → 0.5
        assert abs(priming["n2"] - 0.5) < 0.01

    def test_deep_decay_prunes_below_threshold(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.1})

        # After several queries, 0.1 * 0.7^N should drop below MIN_ACTIVATION_LEVEL
        for i in range(10):
            cache.update_from_result({f"other_{i}": 0.5})

        priming = cache.get_priming_activations()
        # 0.1 * 0.7^10 ≈ 0.0028 < MIN_ACTIVATION_LEVEL (0.01)
        assert "n1" not in priming

    def test_eviction_at_max_capacity(self):
        cache = ActivationCache(max_neurons=5)
        # Add 10 neurons
        cache.update_from_result({f"n{i}": 0.1 + i * 0.1 for i in range(10)})
        assert cache.size == 5
        # Top 5 by level should survive
        priming = cache.get_priming_activations()
        assert "n9" in priming  # Highest: 1.0
        assert "n0" not in priming  # Lowest: 0.1

    def test_update_overwrites_stale_with_fresh(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.3})
        # n1 decayed after 1 query: 0.3 * 0.7 = 0.21
        # Now n1 reappears with higher activation
        cache.update_from_result({"n1": 0.9})
        priming = cache.get_priming_activations()
        # Fresh value (0.9) should win over decayed (0.21)
        assert priming["n1"] == pytest.approx(0.9, abs=0.01)

    def test_clear_empties_cache(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.8, "n2": 0.6})
        cache.clear()
        assert cache.size == 0
        assert cache.get_priming_activations() == {}

    def test_very_low_activations_not_cached(self):
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.005})  # Below MIN_ACTIVATION_LEVEL
        assert cache.size == 0


class TestCachedActivation:
    """Tests for CachedActivation decay math."""

    def test_decayed_level_formula(self):
        c = CachedActivation(neuron_id="n1", level=1.0, queries_ago=3)
        expected = 1.0 * math.pow(CACHE_DECAY_BASE, 3)
        assert c.decayed_level == pytest.approx(expected, abs=0.001)

    def test_zero_queries_ago_no_decay(self):
        c = CachedActivation(neuron_id="n1", level=0.8, queries_ago=0)
        assert c.decayed_level == pytest.approx(0.8, abs=0.001)


# ── TestPrimingMetrics ────────────────────────────────────────────────


class TestPrimingMetrics:
    """Tests for PrimingMetrics hit rate and aggressiveness."""

    def test_empty_metrics(self):
        m = PrimingMetrics()
        assert m.hit_rate == 0.0
        assert m.aggressiveness_multiplier == 1.0

    def test_hit_rate_calculation(self):
        m = PrimingMetrics(total_primed=100, hits=30, misses=70)
        assert m.hit_rate == pytest.approx(0.3, abs=0.01)

    def test_high_hit_rate_boosts_aggressiveness(self):
        m = PrimingMetrics(total_primed=100, hits=50, misses=50)
        # hit_rate = 0.5 > 0.3 → boost
        assert m.aggressiveness_multiplier > 1.0

    def test_low_hit_rate_reduces_aggressiveness(self):
        m = PrimingMetrics(total_primed=100, hits=5, misses=95)
        # hit_rate = 0.05 < 0.1 → reduce
        assert m.aggressiveness_multiplier < 1.0

    def test_too_few_samples_default_aggressiveness(self):
        m = PrimingMetrics(total_primed=5, hits=4, misses=1)
        # Not enough data (< 10) → 1.0
        assert m.aggressiveness_multiplier == 1.0

    def test_mid_range_hit_rate_neutral(self):
        m = PrimingMetrics(total_primed=100, hits=20, misses=80)
        # hit_rate = 0.2 — between 0.1 and 0.3 → 1.0
        assert m.aggressiveness_multiplier == 1.0

    def test_aggressiveness_capped(self):
        m = PrimingMetrics(total_primed=100, hits=100, misses=0)
        assert m.aggressiveness_multiplier <= 1.5
        m2 = PrimingMetrics(total_primed=100, hits=0, misses=100)
        assert m2.aggressiveness_multiplier >= 0.5


# ── TestTopicPriming ──────────────────────────────────────────────────


class TestTopicPriming:
    """Tests for topic-based pre-warming."""

    @pytest.mark.asyncio
    async def test_no_topics_returns_empty(self):
        storage = AsyncMock()
        session = _make_session(topic_ema={})
        result = await prime_from_topics(storage, session)
        assert result == {}

    @pytest.mark.asyncio
    async def test_low_ema_topics_skipped(self):
        storage = AsyncMock()
        session = _make_session(topic_ema={"auth": 0.3})  # Below TOPIC_EMA_THRESHOLD
        result = await prime_from_topics(storage, session)
        assert result == {}

    @pytest.mark.asyncio
    async def test_established_topic_primes_neurons(self):
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            return_value=[
                FakeNeuron(id="n1", content="auth"),
                FakeNeuron(id="n2", content="jwt auth"),
            ]
        )
        session = _make_session(topic_ema={"auth": 0.7})  # Above threshold
        result = await prime_from_topics(storage, session)
        assert "n1" in result
        assert "n2" in result
        # Boost level should be proportional to EMA weight
        assert result["n1"] <= TOPIC_PRE_WARM_LEVEL

    @pytest.mark.asyncio
    async def test_aggressiveness_scales_boost(self):
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[FakeNeuron(id="n1", content="test")])
        session = _make_session(topic_ema={"test": 0.8})

        normal = await prime_from_topics(storage, session, aggressiveness=1.0)
        boosted = await prime_from_topics(storage, session, aggressiveness=1.5)
        assert boosted["n1"] > normal["n1"]

    @pytest.mark.asyncio
    async def test_storage_error_returns_empty(self):
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=Exception("DB error"))
        session = _make_session(topic_ema={"auth": 0.8})
        result = await prime_from_topics(storage, session)
        assert result == {}


# ── TestHabitPriming ──────────────────────────────────────────────────


class TestHabitPriming:
    """Tests for habit-based priming from query pattern co-occurrence."""

    @pytest.mark.asyncio
    async def test_no_topics_returns_empty(self):
        storage = AsyncMock()
        session = _make_session(topic_ema={})
        result = await prime_from_habits(storage, session)
        assert result == {}

    @pytest.mark.asyncio
    async def test_concept_with_before_synapse_primes_target(self):
        """If session has topic 'auth' and pattern says auth→jwt, prime jwt neurons."""
        storage = AsyncMock()

        # Setup: CONCEPT neuron for 'auth' with BEFORE synapse to 'jwt'
        auth_concept = FakeNeuron(id="c_auth", content="auth")
        jwt_concept = FakeNeuron(id="c_jwt", content="jwt")
        before_synapse = FakeSynapse(source_id="c_auth", target_id="c_jwt", weight=0.8)

        storage.find_neurons = AsyncMock(
            side_effect=[
                [auth_concept],  # First call: find CONCEPT for 'auth'
                [FakeNeuron(id="n_jwt1", content="jwt token setup")],  # Related neurons
            ]
        )
        storage.get_synapses = AsyncMock(return_value=[before_synapse])
        storage.get_neuron = AsyncMock(return_value=jwt_concept)

        session = _make_session(topic_ema={"auth": 0.8})
        result = await prime_from_habits(storage, session)
        assert "n_jwt1" in result
        assert result["n_jwt1"] <= HABIT_PRIME_LEVEL

    @pytest.mark.asyncio
    async def test_low_weight_synapse_skipped(self):
        storage = AsyncMock()
        auth_concept = FakeNeuron(id="c_auth", content="auth")
        low_synapse = FakeSynapse(
            source_id="c_auth",
            target_id="c_other",
            weight=0.3,  # Below threshold
        )
        storage.find_neurons = AsyncMock(return_value=[auth_concept])
        storage.get_synapses = AsyncMock(return_value=[low_synapse])

        session = _make_session(topic_ema={"auth": 0.8})
        result = await prime_from_habits(storage, session)
        assert result == {}

    @pytest.mark.asyncio
    async def test_already_primed_topic_skipped(self):
        """If predicted topic is already in session EMA above threshold, skip it."""
        storage = AsyncMock()
        auth_concept = FakeNeuron(id="c_auth", content="auth")
        jwt_concept = FakeNeuron(id="c_jwt", content="jwt")
        before_synapse = FakeSynapse(source_id="c_auth", target_id="c_jwt", weight=0.9)
        storage.find_neurons = AsyncMock(return_value=[auth_concept])
        storage.get_synapses = AsyncMock(return_value=[before_synapse])
        storage.get_neuron = AsyncMock(return_value=jwt_concept)

        # jwt is already established in session → skip
        session = _make_session(topic_ema={"auth": 0.8, "jwt": 0.7})
        result = await prime_from_habits(storage, session)
        assert result == {}

    @pytest.mark.asyncio
    async def test_max_habit_topics_capped(self):
        """Should not predict more than MAX_HABIT_TOPICS."""
        storage = AsyncMock()

        # Create many BEFORE synapses
        synapses = [
            FakeSynapse(source_id="c_auth", target_id=f"c_t{i}", weight=0.9) for i in range(10)
        ]
        auth_concept = FakeNeuron(id="c_auth", content="auth")
        target_neurons = [FakeNeuron(id=f"c_t{i}", content=f"topic{i}") for i in range(10)]

        call_count = 0

        async def mock_find_neurons(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [auth_concept]
            return [FakeNeuron(id=f"n_rel_{call_count}", content="related")]

        storage.find_neurons = mock_find_neurons
        storage.get_synapses = AsyncMock(return_value=synapses)
        storage.get_neuron = AsyncMock(
            side_effect=lambda nid: next((n for n in target_neurons if n.id == nid), None)
        )

        session = _make_session(topic_ema={"auth": 0.8})
        result = await prime_from_habits(storage, session)
        # Should not prime more than MAX_HABIT_TOPICS worth of topics
        # (each topic can produce multiple neurons, but topics are capped)
        assert len(result) <= MAX_HABIT_TOPICS * 10  # max neurons per topic


# ── TestCoActivationPriming ───────────────────────────────────────────


class TestCoActivationPriming:
    """Tests for co-activation (Hebbian) priming."""

    @pytest.mark.asyncio
    async def test_no_recent_neurons_returns_empty(self):
        storage = AsyncMock()
        result = await prime_from_co_activations(storage, [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_strong_co_activation_primes_partner(self):
        storage = AsyncMock()
        # n1-n2 co-activated 5 times with strength 0.8
        storage.get_co_activation_counts = AsyncMock(return_value=[("n1", "n2", 5, 0.8)])
        result = await prime_from_co_activations(storage, ["n1"])
        assert "n2" in result
        assert result["n2"] <= CO_ACTIVATION_PRIME_LEVEL

    @pytest.mark.asyncio
    async def test_weak_binding_skipped(self):
        storage = AsyncMock()
        storage.get_co_activation_counts = AsyncMock(
            return_value=[("n1", "n2", 5, 0.3)]  # Below CO_ACTIVATION_MIN_STRENGTH
        )
        result = await prime_from_co_activations(storage, ["n1"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_both_sides_in_recent_skipped(self):
        """If both neurons were already in recent result, no priming needed."""
        storage = AsyncMock()
        storage.get_co_activation_counts = AsyncMock(return_value=[("n1", "n2", 5, 0.8)])
        result = await prime_from_co_activations(storage, ["n1", "n2"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_max_primes_capped(self):
        storage = AsyncMock()
        # Many co-activation pairs
        pairs = [("n_recent", f"n_target_{i}", 10, 0.9) for i in range(50)]
        storage.get_co_activation_counts = AsyncMock(return_value=pairs)
        result = await prime_from_co_activations(storage, ["n_recent"])
        assert len(result) <= MAX_CO_ACTIVATION_PRIMES

    @pytest.mark.asyncio
    async def test_storage_error_returns_empty(self):
        storage = AsyncMock()
        storage.get_co_activation_counts = AsyncMock(side_effect=Exception("DB error"))
        result = await prime_from_co_activations(storage, ["n1"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_reverse_direction_primes(self):
        """If n2 was recent and (n1,n2) co-activated, n1 gets primed."""
        storage = AsyncMock()
        storage.get_co_activation_counts = AsyncMock(return_value=[("n1", "n2", 5, 0.7)])
        result = await prime_from_co_activations(storage, ["n2"])
        assert "n1" in result


# ── TestComputePriming (Orchestrator) ─────────────────────────────────


class TestComputePriming:
    """Tests for the compute_priming orchestrator."""

    @pytest.mark.asyncio
    async def test_no_session_returns_cache_only(self):
        storage = AsyncMock()
        cache = ActivationCache()
        cache.update_from_result({"n1": 0.5})
        result = await compute_priming(storage, None, cache)
        assert "n1" in result.activation_boosts
        assert result.source_counts.get("cache", 0) > 0

    @pytest.mark.asyncio
    async def test_no_cache_no_session_returns_empty(self):
        storage = AsyncMock()
        result = await compute_priming(storage, None, None)
        assert result.total_primed == 0

    @pytest.mark.asyncio
    async def test_all_sources_combined(self):
        """Verify that cache + topic + co-activation sources merge."""
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[FakeNeuron(id="n_topic", content="auth")])
        storage.get_co_activation_counts = AsyncMock(return_value=[("n_cache", "n_co", 5, 0.8)])

        cache = ActivationCache()
        cache.update_from_result({"n_cache": 0.6})

        session = _make_session(topic_ema={"auth": 0.8}, query_count=5)

        result = await compute_priming(storage, session, cache, recent_neuron_ids=["n_cache"])
        # Should have neurons from cache, topic, and co-activation
        assert result.total_primed >= 2
        assert "cache" in result.source_counts

    @pytest.mark.asyncio
    async def test_young_session_skips_topic_and_habit(self):
        """Session with < 2 queries skips topic priming, < 3 skips habit."""
        storage = AsyncMock()
        session = _make_session(topic_ema={"auth": 0.8}, query_count=1)
        result = await compute_priming(storage, session, None)
        assert "topic" not in result.source_counts
        assert "habit" not in result.source_counts

    @pytest.mark.asyncio
    async def test_aggressiveness_passed_through(self):
        """High hit rate metrics should boost priming levels."""
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[FakeNeuron(id="n1", content="test")])
        session = _make_session(topic_ema={"test": 0.8}, query_count=5)

        metrics_normal = PrimingMetrics()
        metrics_hot = PrimingMetrics(total_primed=100, hits=50, misses=50)

        r_normal = await compute_priming(storage, session, None, metrics=metrics_normal)
        r_hot = await compute_priming(storage, session, None, metrics=metrics_hot)

        # Hot metrics should produce higher boosts
        if "n1" in r_normal.activation_boosts and "n1" in r_hot.activation_boosts:
            assert r_hot.activation_boosts["n1"] >= r_normal.activation_boosts["n1"]


# ── TestMergePriming ──────────────────────────────────────────────────


class TestMergePriming:
    """Tests for merge_priming_into_activations."""

    def test_no_priming_returns_original(self):
        original = {"n1": 0.8}
        result = merge_priming_into_activations(original, PrimingResult.empty())
        assert result == original

    def test_priming_adds_new_neurons(self):
        original = {"n1": 0.8}
        priming = PrimingResult(
            activation_boosts={"n2": 0.15},
            source_counts={"topic": 1},
            total_primed=1,
        )
        result = merge_priming_into_activations(original, priming)
        assert "n1" in result
        assert "n2" in result
        assert result["n2"] == pytest.approx(0.15, abs=0.01)

    def test_priming_boosts_existing_anchors(self):
        original = {"n1": 0.5}
        priming = PrimingResult(
            activation_boosts={"n1": 0.2},
            source_counts={"cache": 1},
            total_primed=1,
        )
        result = merge_priming_into_activations(original, priming)
        assert result["n1"] == pytest.approx(0.7, abs=0.01)

    def test_priming_capped_at_1(self):
        original = {"n1": 0.9}
        priming = PrimingResult(
            activation_boosts={"n1": 0.3},
            source_counts={"cache": 1},
            total_primed=1,
        )
        result = merge_priming_into_activations(original, priming)
        assert result["n1"] == 1.0

    def test_none_anchor_activations(self):
        priming = PrimingResult(
            activation_boosts={"n1": 0.15},
            source_counts={"topic": 1},
            total_primed=1,
        )
        result = merge_priming_into_activations(None, priming)
        assert result["n1"] == pytest.approx(0.15, abs=0.01)


# ── TestRecordPrimingOutcome ──────────────────────────────────────────


class TestRecordPrimingOutcome:
    """Tests for record_priming_outcome hit/miss tracking."""

    def test_all_hits(self):
        m = PrimingMetrics()
        record_priming_outcome(m, {"n1", "n2"}, {"n1", "n2", "n3"})
        assert m.hits == 2
        assert m.misses == 0
        assert m.total_primed == 2

    def test_all_misses(self):
        m = PrimingMetrics()
        record_priming_outcome(m, {"n1", "n2"}, {"n3", "n4"})
        assert m.hits == 0
        assert m.misses == 2

    def test_partial_hits(self):
        m = PrimingMetrics()
        record_priming_outcome(m, {"n1", "n2", "n3"}, {"n2", "n4"})
        assert m.hits == 1
        assert m.misses == 2

    def test_empty_primed_set(self):
        m = PrimingMetrics()
        record_priming_outcome(m, set(), {"n1"})
        assert m.total_primed == 0

    def test_cumulative_tracking(self):
        m = PrimingMetrics()
        record_priming_outcome(m, {"n1"}, {"n1"})
        record_priming_outcome(m, {"n2"}, {"n3"})
        assert m.total_primed == 2
        assert m.hits == 1
        assert m.misses == 1


# ── TestSessionStatePriming ───────────────────────────────────────────


class TestSessionStatePriming:
    """Tests for priming fields on SessionState."""

    def test_priming_hit_rate_default(self):
        s = SessionState(session_id="test")
        assert s.priming_hit_rate == 0.0

    def test_priming_hit_rate_calculation(self):
        s = SessionState(session_id="test")
        s.priming_total = 100
        s.priming_hits = 25
        s.priming_misses = 75
        assert s.priming_hit_rate == pytest.approx(0.25, abs=0.01)

    def test_priming_in_summary_dict(self):
        s = SessionState(session_id="test")
        s.priming_total = 50
        s.priming_hits = 15
        summary = s.to_summary_dict()
        assert "priming_hit_rate" in summary
        assert summary["priming_hit_rate"] == pytest.approx(0.3, abs=0.01)
        assert summary["priming_total"] == 50


# ── TestPrimingResult ─────────────────────────────────────────────────


class TestPrimingResult:
    """Tests for PrimingResult data class."""

    def test_empty_result(self):
        r = PrimingResult.empty()
        assert r.total_primed == 0
        assert r.activation_boosts == {}
        assert r.source_counts == {}

    def test_source_attribution(self):
        r = PrimingResult(
            activation_boosts={"n1": 0.15, "n2": 0.10},
            source_counts={"cache": 1, "topic": 1},
            total_primed=2,
        )
        assert r.source_counts["cache"] == 1
        assert r.source_counts["topic"] == 1


# ── TestBackwardCompat ────────────────────────────────────────────────


class TestBackwardCompat:
    """Verify priming doesn't break existing behavior."""

    def test_merge_with_no_priming_is_identity(self):
        activations = {"n1": 0.8, "n2": 0.5}
        result = merge_priming_into_activations(activations, PrimingResult.empty())
        assert result == activations

    def test_merge_with_none_activations_and_no_priming(self):
        result = merge_priming_into_activations(None, PrimingResult.empty())
        assert result == {}

    @pytest.mark.asyncio
    async def test_compute_priming_all_none_is_safe(self):
        storage = AsyncMock()
        result = await compute_priming(storage, None, None, None, None)
        assert result.total_primed == 0
