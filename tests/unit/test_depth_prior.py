"""Unit tests for Bayesian depth prior feature.

Covers DepthPrior dataclass, AdaptiveDepthSelector, and SQLite storage mixin.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.engine.depth_prior import AdaptiveDepthSelector, DepthDecision, DepthPrior
from neural_memory.engine.retrieval_types import DepthLevel
from neural_memory.extraction.entities import Entity, EntityType
from neural_memory.extraction.parser import Perspective, QueryIntent, Stimulus
from neural_memory.extraction.temporal import TimeHint
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stimulus(entity_texts: list[str]) -> Stimulus:
    """Build a minimal Stimulus with the given entity text strings."""
    entities = [
        Entity(text=t, type=EntityType.UNKNOWN, start=0, end=len(t))
        for t in entity_texts
    ]
    return Stimulus(
        time_hints=[],
        keywords=[],
        entities=entities,
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query=" ".join(entity_texts),
    )


def _make_prior(
    entity_text: str,
    depth: DepthLevel,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    total_queries: int = 0,
) -> DepthPrior:
    return DepthPrior(
        entity_text=entity_text,
        depth_level=depth,
        alpha=alpha,
        beta=beta,
        total_queries=total_queries,
    )


# ---------------------------------------------------------------------------
# DepthPrior dataclass tests
# ---------------------------------------------------------------------------


class TestDepthPriorDefaults:
    def test_default_values(self) -> None:
        prior = DepthPrior(entity_text="Alice", depth_level=DepthLevel.CONTEXT)
        assert prior.alpha == 1.0
        assert prior.beta == 1.0
        assert prior.total_queries == 0
        assert prior.entity_text == "Alice"
        assert prior.depth_level == DepthLevel.CONTEXT

    def test_expected_success_rate_uniform(self) -> None:
        """Alpha=1, beta=1 → E[Beta(1,1)] = 0.5."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT)
        assert prior.expected_success_rate == pytest.approx(0.5)

    def test_expected_success_rate_biased(self) -> None:
        """Alpha=9, beta=1 → E = 9/10 = 0.9."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT, alpha=9.0, beta=1.0)
        assert prior.expected_success_rate == pytest.approx(0.9)

    def test_expected_success_rate_zero_leaning(self) -> None:
        """Alpha=1, beta=9 → E = 1/10 = 0.1."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.DEEP, alpha=1.0, beta=9.0)
        assert prior.expected_success_rate == pytest.approx(0.1)


class TestDepthPriorUpdate:
    def test_update_success_increments_alpha(self) -> None:
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.CONTEXT, alpha=2.0, beta=3.0)
        updated = prior.update_success()
        assert updated.alpha == pytest.approx(3.0)
        assert updated.beta == pytest.approx(3.0)  # unchanged
        assert updated.total_queries == 1

    def test_update_failure_increments_beta(self) -> None:
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.CONTEXT, alpha=2.0, beta=3.0)
        updated = prior.update_failure()
        assert updated.beta == pytest.approx(4.0)
        assert updated.alpha == pytest.approx(2.0)  # unchanged
        assert updated.total_queries == 1

    def test_update_success_increments_total_queries(self) -> None:
        prior = DepthPrior(
            entity_text="e", depth_level=DepthLevel.INSTANT, total_queries=5
        )
        updated = prior.update_success()
        assert updated.total_queries == 6

    def test_update_failure_increments_total_queries(self) -> None:
        prior = DepthPrior(
            entity_text="e", depth_level=DepthLevel.INSTANT, total_queries=5
        )
        updated = prior.update_failure()
        assert updated.total_queries == 6

    def test_update_returns_new_instance(self) -> None:
        """Frozen dataclass: update_success / update_failure must NOT mutate original."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.HABIT, alpha=2.0, beta=2.0)
        updated_s = prior.update_success()
        updated_f = prior.update_failure()

        # Original unchanged
        assert prior.alpha == pytest.approx(2.0)
        assert prior.beta == pytest.approx(2.0)
        assert prior.total_queries == 0

        # New instances are different objects
        assert updated_s is not prior
        assert updated_f is not prior


class TestDepthPriorDecay:
    def test_decay_shrinks_alpha_and_beta(self) -> None:
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.DEEP, alpha=5.0, beta=4.0)
        decayed = prior.decay(factor=0.5)
        assert decayed.alpha == pytest.approx(2.5)
        assert decayed.beta == pytest.approx(2.0)

    def test_decay_default_factor_is_0_9(self) -> None:
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.DEEP, alpha=10.0, beta=10.0)
        decayed = prior.decay()
        assert decayed.alpha == pytest.approx(9.0)
        assert decayed.beta == pytest.approx(9.0)

    def test_decay_clamps_alpha_to_min_1(self) -> None:
        """Alpha just above 1 should not go below 1.0 after decay."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT, alpha=1.05, beta=5.0)
        decayed = prior.decay(factor=0.9)
        assert decayed.alpha >= 1.0

    def test_decay_clamps_beta_to_min_1(self) -> None:
        """Beta just above 1 should not go below 1.0 after decay."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT, alpha=5.0, beta=1.05)
        decayed = prior.decay(factor=0.9)
        assert decayed.beta >= 1.0

    def test_decay_clamps_to_exactly_1_when_below(self) -> None:
        """Factor that would drive alpha below 1.0 must clamp to exactly 1.0."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT, alpha=1.0, beta=1.0)
        decayed = prior.decay(factor=0.5)
        assert decayed.alpha == pytest.approx(1.0)
        assert decayed.beta == pytest.approx(1.0)

    def test_decay_returns_new_instance(self) -> None:
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.CONTEXT, alpha=5.0, beta=5.0)
        decayed = prior.decay()
        assert decayed is not prior
        assert prior.alpha == pytest.approx(5.0)  # original untouched


class TestDepthPriorConfidenceWidth:
    def test_confidence_width_uniform_prior(self) -> None:
        """Beta(1,1) has maximum variance; width should be finite and positive."""
        prior = DepthPrior(entity_text="e", depth_level=DepthLevel.INSTANT)
        width = prior.confidence_width
        assert width > 0
        assert math.isfinite(width)

    def test_confidence_width_decreases_with_more_data(self) -> None:
        """More observations (higher alpha+beta sum) → narrower confidence interval."""
        sparse = DepthPrior(entity_text="e", depth_level=DepthLevel.CONTEXT, alpha=2.0, beta=2.0)
        rich = DepthPrior(entity_text="e", depth_level=DepthLevel.CONTEXT, alpha=50.0, beta=50.0)
        assert rich.confidence_width < sparse.confidence_width


# ---------------------------------------------------------------------------
# AdaptiveDepthSelector tests
# ---------------------------------------------------------------------------


def _make_storage_with_priors(
    priors_by_entity: dict[str, list[DepthPrior]],
) -> MagicMock:
    """Build a mock storage that returns the given priors from get_depth_priors_batch."""
    storage = MagicMock()
    storage.get_depth_priors_batch = AsyncMock(return_value=priors_by_entity)
    storage.upsert_depth_prior = AsyncMock()
    storage.get_stale_priors = AsyncMock(return_value=[])
    storage.delete_depth_priors = AsyncMock()
    return storage


class TestAdaptiveDepthSelectorFallbacks:
    @pytest.mark.asyncio
    async def test_no_entities_falls_back(self) -> None:
        """Stimulus with no entities → rule-based fallback."""
        storage = _make_storage_with_priors({})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus([])
        decision = await selector.select_depth(stimulus, DepthLevel.CONTEXT)
        assert decision.depth == DepthLevel.CONTEXT
        assert decision.method == "rule_based"
        storage.get_depth_priors_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_priors_falls_back(self) -> None:
        """Entities exist but no priors in storage → rule-based fallback."""
        storage = _make_storage_with_priors({"Alice": []})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["Alice"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.INSTANT
        assert decision.method == "rule_based"

    @pytest.mark.asyncio
    async def test_insufficient_data_falls_back(self) -> None:
        """Total queries < MIN_QUERIES_FOR_BAYESIAN → rule-based fallback."""
        # MIN_QUERIES_FOR_BAYESIAN = 5; give 4 total queries
        prior = _make_prior("Alice", DepthLevel.CONTEXT, alpha=3.0, beta=1.0, total_queries=4)
        storage = _make_storage_with_priors({"Alice": [prior]})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["Alice"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.INSTANT
        assert decision.method == "rule_based"
        assert "Insufficient" in decision.reason

    @pytest.mark.asyncio
    async def test_low_score_falls_back(self) -> None:
        """Best Bayesian score <= 0.5 → rule-based fallback."""
        # alpha=1, beta=9 → expected_rate = 0.1, well below 0.5
        prior = _make_prior("Bob", DepthLevel.DEEP, alpha=1.0, beta=9.0, total_queries=10)
        storage = _make_storage_with_priors({"Bob": [prior]})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["Bob"])
        decision = await selector.select_depth(stimulus, DepthLevel.CONTEXT)
        assert decision.depth == DepthLevel.CONTEXT
        assert decision.method == "rule_based"
        assert "too low" in decision.reason


class TestAdaptiveDepthSelectorBayesian:
    @pytest.mark.asyncio
    async def test_uses_best_prior(self) -> None:
        """With priors showing CONTEXT has highest rate, Bayesian selects CONTEXT."""
        priors = [
            _make_prior("Alice", DepthLevel.INSTANT, alpha=2.0, beta=8.0, total_queries=10),
            _make_prior("Alice", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10),
            _make_prior("Alice", DepthLevel.DEEP, alpha=3.0, beta=7.0, total_queries=10),
        ]
        storage = _make_storage_with_priors({"Alice": priors})
        # epsilon=0.0 disables exploration
        selector = AdaptiveDepthSelector(storage, epsilon=0.0)
        stimulus = _make_stimulus(["Alice"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.CONTEXT
        assert decision.method == "bayesian"

    @pytest.mark.asyncio
    async def test_bayesian_method_reason_mentions_depth_and_score(self) -> None:
        prior = _make_prior("X", DepthLevel.HABIT, alpha=8.0, beta=2.0, total_queries=10)
        storage = _make_storage_with_priors({"X": [prior]})
        selector = AdaptiveDepthSelector(storage, epsilon=0.0)
        stimulus = _make_stimulus(["X"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.method == "bayesian"
        assert "HABIT" in decision.reason
        assert "0." in decision.reason  # score is in reason

    @pytest.mark.asyncio
    async def test_multi_entity_aggregation(self) -> None:
        """Scores are averaged across multiple entities; depth with highest mean wins."""
        # Entity "A": CONTEXT = 0.9, INSTANT = 0.2
        # Entity "B": CONTEXT = 0.8, INSTANT = 0.3
        # Mean CONTEXT = 0.85, mean INSTANT = 0.25 → CONTEXT wins
        priors_a = [
            _make_prior("A", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10),
            _make_prior("A", DepthLevel.INSTANT, alpha=2.0, beta=8.0, total_queries=10),
        ]
        priors_b = [
            _make_prior("B", DepthLevel.CONTEXT, alpha=8.0, beta=2.0, total_queries=10),
            _make_prior("B", DepthLevel.INSTANT, alpha=3.0, beta=7.0, total_queries=10),
        ]
        storage = _make_storage_with_priors({"A": priors_a, "B": priors_b})
        selector = AdaptiveDepthSelector(storage, epsilon=0.0)
        stimulus = _make_stimulus(["A", "B"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.CONTEXT
        assert decision.method == "bayesian"

    @pytest.mark.asyncio
    async def test_entity_priors_populated_on_bayesian(self) -> None:
        """entity_priors dict should be non-empty on Bayesian decisions."""
        prior = _make_prior("Z", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10)
        storage = _make_storage_with_priors({"Z": [prior]})
        selector = AdaptiveDepthSelector(storage, epsilon=0.0)
        stimulus = _make_stimulus(["Z"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.entity_priors != {}
        assert "Z" in decision.entity_priors


class TestAdaptiveDepthSelectorExploration:
    @pytest.mark.asyncio
    async def test_exploration_forced(self) -> None:
        """epsilon=1.0 forces exploration; result must differ from best depth."""
        priors = [
            _make_prior("Alice", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10),
            _make_prior("Alice", DepthLevel.INSTANT, alpha=2.0, beta=8.0, total_queries=10),
        ]
        storage = _make_storage_with_priors({"Alice": priors})
        selector = AdaptiveDepthSelector(storage, epsilon=1.0)
        stimulus = _make_stimulus(["Alice"])
        decision = await selector.select_depth(stimulus, DepthLevel.DEEP)
        assert decision.method == "exploration"
        assert decision.exploration is True
        # Must NOT be the best (CONTEXT) when exploring
        assert decision.depth != DepthLevel.CONTEXT

    @pytest.mark.asyncio
    async def test_exploration_disabled(self) -> None:
        """epsilon=0.0 never triggers exploration."""
        priors = [
            _make_prior("Alice", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10),
            _make_prior("Alice", DepthLevel.INSTANT, alpha=2.0, beta=8.0, total_queries=10),
        ]
        storage = _make_storage_with_priors({"Alice": priors})
        selector = AdaptiveDepthSelector(storage, epsilon=0.0)
        stimulus = _make_stimulus(["Alice"])

        # Run many times; exploration must never occur
        for _ in range(50):
            decision = await selector.select_depth(stimulus, DepthLevel.DEEP)
            assert decision.method != "exploration", "Unexpected exploration with epsilon=0.0"

    @pytest.mark.asyncio
    async def test_exploration_reason_mentions_both_depths(self) -> None:
        """Exploration reason references the explored depth and the best depth."""
        priors = [
            _make_prior("E", DepthLevel.CONTEXT, alpha=9.0, beta=1.0, total_queries=10),
            _make_prior("E", DepthLevel.DEEP, alpha=2.0, beta=8.0, total_queries=10),
        ]
        storage = _make_storage_with_priors({"E": priors})
        selector = AdaptiveDepthSelector(storage, epsilon=1.0)
        stimulus = _make_stimulus(["E"])
        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.method == "exploration"
        # Reason should mention CONTEXT (best depth) in the exploration message
        assert "CONTEXT" in decision.reason


class TestAdaptiveDepthSelectorRecordOutcome:
    @pytest.mark.asyncio
    async def test_record_outcome_success_updates_alpha(self) -> None:
        """Successful outcome calls upsert with higher alpha."""
        original = _make_prior("Alice", DepthLevel.CONTEXT, alpha=2.0, beta=2.0, total_queries=4)
        storage = _make_storage_with_priors({"Alice": [original]})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["Alice"])

        # confidence >= SUCCESS_THRESHOLD (0.3) + fibers_matched >= 1 → success
        await selector.record_outcome(stimulus, DepthLevel.CONTEXT, confidence=0.8, fibers_matched=3)

        storage.upsert_depth_prior.assert_called_once()
        upserted: DepthPrior = storage.upsert_depth_prior.call_args[0][0]
        assert upserted.alpha == pytest.approx(3.0)
        assert upserted.beta == pytest.approx(2.0)
        assert upserted.total_queries == 5

    @pytest.mark.asyncio
    async def test_record_outcome_failure_updates_beta(self) -> None:
        """Failed outcome calls upsert with higher beta."""
        original = _make_prior("Bob", DepthLevel.DEEP, alpha=2.0, beta=2.0, total_queries=4)
        storage = _make_storage_with_priors({"Bob": [original]})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["Bob"])

        # confidence < SUCCESS_THRESHOLD → failure
        await selector.record_outcome(stimulus, DepthLevel.DEEP, confidence=0.1, fibers_matched=5)

        storage.upsert_depth_prior.assert_called_once()
        upserted: DepthPrior = storage.upsert_depth_prior.call_args[0][0]
        assert upserted.beta == pytest.approx(3.0)
        assert upserted.alpha == pytest.approx(2.0)
        assert upserted.total_queries == 5

    @pytest.mark.asyncio
    async def test_record_outcome_failure_when_zero_fibers(self) -> None:
        """fibers_matched=0 counts as failure even with high confidence."""
        original = _make_prior("C", DepthLevel.CONTEXT, alpha=2.0, beta=2.0, total_queries=4)
        storage = _make_storage_with_priors({"C": [original]})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["C"])

        await selector.record_outcome(stimulus, DepthLevel.CONTEXT, confidence=0.9, fibers_matched=0)

        upserted: DepthPrior = storage.upsert_depth_prior.call_args[0][0]
        assert upserted.beta == pytest.approx(3.0)  # failure branch

    @pytest.mark.asyncio
    async def test_record_outcome_creates_new_prior(self) -> None:
        """Entity exists but no prior for this depth level → creates new prior."""
        # Storage returns no existing priors
        storage = _make_storage_with_priors({"NewEntity": []})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["NewEntity"])

        await selector.record_outcome(
            stimulus, DepthLevel.HABIT, confidence=0.8, fibers_matched=2
        )

        storage.upsert_depth_prior.assert_called_once()
        upserted: DepthPrior = storage.upsert_depth_prior.call_args[0][0]
        assert upserted.entity_text == "NewEntity"
        assert upserted.depth_level == DepthLevel.HABIT
        # New prior starts at alpha=1, update_success gives alpha=2
        assert upserted.alpha == pytest.approx(2.0)
        assert upserted.beta == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_record_outcome_no_entities_does_nothing(self) -> None:
        """Stimulus with no entities → record_outcome is a no-op."""
        storage = _make_storage_with_priors({})
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus([])

        await selector.record_outcome(stimulus, DepthLevel.CONTEXT, confidence=0.8, fibers_matched=2)

        storage.get_depth_priors_batch.assert_not_called()
        storage.upsert_depth_prior.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_outcome_multi_entity_upserts_each(self) -> None:
        """Each entity gets its own upsert call."""
        priors: dict[str, list[DepthPrior]] = {"A": [], "B": [], "C": []}
        storage = _make_storage_with_priors(priors)
        selector = AdaptiveDepthSelector(storage)
        stimulus = _make_stimulus(["A", "B", "C"])

        await selector.record_outcome(stimulus, DepthLevel.INSTANT, confidence=0.9, fibers_matched=1)

        assert storage.upsert_depth_prior.call_count == 3


# ---------------------------------------------------------------------------
# Storage tests using real SQLiteStorage + tmp_path
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_storage(tmp_path: Path) -> SQLiteStorage:
    """SQLiteStorage backed by a temp file, brain context set."""
    db_path = tmp_path / "test_depth.db"
    store = SQLiteStorage(db_path)
    await store.initialize()
    brain = Brain.create(name="test-depth-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest_asyncio.fixture
async def sqlite_storage_alt_brain(tmp_path: Path) -> SQLiteStorage:
    """Second SQLiteStorage sharing the same DB file but a different brain."""
    db_path = tmp_path / "test_depth.db"
    store = SQLiteStorage(db_path)
    await store.initialize()

    brain_a = Brain.create(name="brain-A")
    brain_b = Brain.create(name="brain-B")
    await store.save_brain(brain_a)
    await store.save_brain(brain_b)
    # Expose both IDs so tests can switch
    store._brain_a_id = brain_a.id
    store._brain_b_id = brain_b.id
    return store


class TestSQLiteDepthPriorStorage:
    @pytest.mark.asyncio
    async def test_upsert_and_get_depth_prior(self, sqlite_storage: SQLiteStorage) -> None:
        """Round-trip: upsert then get_depth_priors returns same data."""
        prior = DepthPrior(
            entity_text="Alice",
            depth_level=DepthLevel.CONTEXT,
            alpha=3.0,
            beta=2.0,
            total_queries=5,
        )
        await sqlite_storage.upsert_depth_prior(prior)

        results = await sqlite_storage.get_depth_priors("Alice")
        assert len(results) == 1
        r = results[0]
        assert r.entity_text == "Alice"
        assert r.depth_level == DepthLevel.CONTEXT
        assert r.alpha == pytest.approx(3.0)
        assert r.beta == pytest.approx(2.0)
        assert r.total_queries == 5

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, sqlite_storage: SQLiteStorage) -> None:
        """Upserting same (entity, depth) key updates fields, not inserts duplicate."""
        prior = DepthPrior(entity_text="Bob", depth_level=DepthLevel.INSTANT, alpha=2.0, beta=1.0)
        await sqlite_storage.upsert_depth_prior(prior)

        updated = DepthPrior(entity_text="Bob", depth_level=DepthLevel.INSTANT, alpha=5.0, beta=3.0, total_queries=4)
        await sqlite_storage.upsert_depth_prior(updated)

        results = await sqlite_storage.get_depth_priors("Bob")
        assert len(results) == 1
        assert results[0].alpha == pytest.approx(5.0)
        assert results[0].beta == pytest.approx(3.0)
        assert results[0].total_queries == 4

    @pytest.mark.asyncio
    async def test_get_depth_priors_batch(self, sqlite_storage: SQLiteStorage) -> None:
        """Batch fetch returns priors grouped by entity."""
        for entity, depth, alpha in [
            ("A", DepthLevel.INSTANT, 2.0),
            ("A", DepthLevel.CONTEXT, 3.0),
            ("B", DepthLevel.DEEP, 4.0),
        ]:
            await sqlite_storage.upsert_depth_prior(
                DepthPrior(entity_text=entity, depth_level=depth, alpha=alpha, total_queries=10)
            )

        result = await sqlite_storage.get_depth_priors_batch(["A", "B"])
        assert len(result["A"]) == 2
        assert len(result["B"]) == 1
        alpha_vals_a = {p.alpha for p in result["A"]}
        assert alpha_vals_a == {2.0, 3.0}
        assert result["B"][0].alpha == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_get_depth_priors_batch_empty_input(self, sqlite_storage: SQLiteStorage) -> None:
        """Empty entity list returns empty dict without DB error."""
        result = await sqlite_storage.get_depth_priors_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_depth_priors_batch_missing_entity(self, sqlite_storage: SQLiteStorage) -> None:
        """Entity with no priors gets empty list in result."""
        await sqlite_storage.upsert_depth_prior(
            DepthPrior(entity_text="present", depth_level=DepthLevel.CONTEXT, total_queries=10)
        )
        result = await sqlite_storage.get_depth_priors_batch(["present", "absent"])
        assert len(result["present"]) == 1
        assert result["absent"] == []

    @pytest.mark.asyncio
    async def test_get_stale_priors(self, sqlite_storage: SQLiteStorage) -> None:
        """get_stale_priors returns priors with last_updated before the cutoff."""
        old_time = utcnow() - timedelta(days=60)
        stale_prior = DepthPrior(
            entity_text="old",
            depth_level=DepthLevel.HABIT,
            last_updated=old_time,
            created_at=old_time,
        )
        fresh_prior = DepthPrior(
            entity_text="fresh",
            depth_level=DepthLevel.CONTEXT,
        )
        await sqlite_storage.upsert_depth_prior(stale_prior)
        await sqlite_storage.upsert_depth_prior(fresh_prior)

        cutoff = utcnow() - timedelta(days=30)
        stale = await sqlite_storage.get_stale_priors(cutoff)

        entity_texts = {p.entity_text for p in stale}
        assert "old" in entity_texts
        assert "fresh" not in entity_texts

    @pytest.mark.asyncio
    async def test_delete_depth_priors(self, sqlite_storage: SQLiteStorage) -> None:
        """delete_depth_priors removes all depth levels for an entity."""
        for depth in [DepthLevel.INSTANT, DepthLevel.CONTEXT, DepthLevel.DEEP]:
            await sqlite_storage.upsert_depth_prior(
                DepthPrior(entity_text="ToDelete", depth_level=depth, total_queries=5)
            )

        count = await sqlite_storage.delete_depth_priors("ToDelete")
        assert count == 3

        remaining = await sqlite_storage.get_depth_priors("ToDelete")
        assert remaining == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_zero(self, sqlite_storage: SQLiteStorage) -> None:
        """Deleting an entity with no priors returns 0."""
        count = await sqlite_storage.delete_depth_priors("ghost")
        assert count == 0

    @pytest.mark.asyncio
    async def test_brain_isolation(self, tmp_path: Path) -> None:
        """Priors stored under brain A are not visible under brain B."""
        db_path = tmp_path / "isolation.db"
        store = SQLiteStorage(db_path)
        await store.initialize()

        brain_a = Brain.create(name="brain-a")
        brain_b = Brain.create(name="brain-b")
        await store.save_brain(brain_a)
        await store.save_brain(brain_b)

        # Write under brain A
        store.set_brain(brain_a.id)
        await store.upsert_depth_prior(
            DepthPrior(entity_text="secret", depth_level=DepthLevel.CONTEXT, total_queries=10)
        )

        # Read under brain B — should see nothing
        store.set_brain(brain_b.id)
        priors_b = await store.get_depth_priors("secret")
        assert priors_b == []

        # Confirm it's still there under brain A
        store.set_brain(brain_a.id)
        priors_a = await store.get_depth_priors("secret")
        assert len(priors_a) == 1

    @pytest.mark.asyncio
    async def test_all_depth_levels_round_trip(self, sqlite_storage: SQLiteStorage) -> None:
        """Every DepthLevel value can be stored and retrieved correctly."""
        for level in DepthLevel:
            prior = DepthPrior(entity_text=f"entity_{level.name}", depth_level=level, total_queries=level.value + 1)
            await sqlite_storage.upsert_depth_prior(prior)

        for level in DepthLevel:
            results = await sqlite_storage.get_depth_priors(f"entity_{level.name}")
            assert len(results) == 1
            assert results[0].depth_level == level


# ---------------------------------------------------------------------------
# Integration: AdaptiveDepthSelector + real SQLiteStorage
# ---------------------------------------------------------------------------


class TestAdaptiveDepthSelectorIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_record_then_select(self, sqlite_storage: SQLiteStorage) -> None:
        """Record many successes for CONTEXT, then select_depth should choose CONTEXT."""
        selector = AdaptiveDepthSelector(sqlite_storage, epsilon=0.0)
        stimulus = _make_stimulus(["Project"])

        # Simulate 6 successful recalls with CONTEXT depth
        for _ in range(6):
            await selector.record_outcome(stimulus, DepthLevel.CONTEXT, confidence=0.9, fibers_matched=3)

        # Simulate 2 failures with INSTANT depth
        for _ in range(2):
            await selector.record_outcome(stimulus, DepthLevel.INSTANT, confidence=0.1, fibers_matched=0)

        decision = await selector.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.CONTEXT
        assert decision.method == "bayesian"

    @pytest.mark.asyncio
    async def test_record_outcome_persists_across_selector_instances(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        """Outcomes recorded by one selector instance are visible to another."""
        selector_a = AdaptiveDepthSelector(sqlite_storage, epsilon=0.0)
        stimulus = _make_stimulus(["Shared"])

        for _ in range(6):
            await selector_a.record_outcome(stimulus, DepthLevel.DEEP, confidence=0.8, fibers_matched=2)

        # New selector instance, same storage
        selector_b = AdaptiveDepthSelector(sqlite_storage, epsilon=0.0)
        decision = await selector_b.select_depth(stimulus, DepthLevel.INSTANT)
        assert decision.depth == DepthLevel.DEEP
        assert decision.method == "bayesian"
