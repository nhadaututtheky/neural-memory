"""Unit tests for Adaptive Depth v2 — Phase 2 of v4.0 Brain Intelligence.

Covers:
- Session-aware depth adjustment (2.1)
- Calibration-driven gate threshold tuning (2.2)
- Result quality feedback with agent_used_result (2.3)
- Dynamic RRF weight adjustment (2.4)
- Activation strategy auto-selection (2.5)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.depth_prior import AdaptiveDepthSelector, DepthPrior
from neural_memory.engine.retrieval_types import DepthLevel
from neural_memory.engine.score_fusion import DEFAULT_RETRIEVER_WEIGHTS
from neural_memory.engine.sufficiency import (
    GateCalibration,
    SufficiencyMetrics,
    SufficiencyResult,
    _apply_calibration,
)
from neural_memory.extraction.entities import Entity, EntityType
from neural_memory.extraction.parser import Perspective, QueryIntent, Stimulus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stimulus(
    entity_texts: list[str] | None = None,
    keywords: list[str] | None = None,
) -> Stimulus:
    """Build a minimal Stimulus with optional entities and keywords."""
    entities = (
        [Entity(text=t, type=EntityType.UNKNOWN, start=0, end=len(t)) for t in entity_texts]
        if entity_texts
        else []
    )
    return Stimulus(
        time_hints=[],
        keywords=keywords or [],
        entities=entities,
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query=" ".join(entity_texts or []),
    )


def _make_session_state(
    topic_ema: dict[str, float] | None = None,
    query_count: int = 0,
) -> MagicMock:
    """Build a mock SessionState with given topic EMA and query count."""
    state = MagicMock()
    state.topic_ema = topic_ema or {}
    state.query_count = query_count
    return state


def _make_selector(priors: dict[str, list[DepthPrior]] | None = None) -> AdaptiveDepthSelector:
    """Build AdaptiveDepthSelector with mock storage."""
    storage = AsyncMock()
    storage.get_depth_priors_batch = AsyncMock(return_value=priors or {})
    storage.upsert_depth_prior = AsyncMock()
    return AdaptiveDepthSelector(storage, epsilon=0.0)  # No exploration for deterministic tests


def _make_sufficiency_metrics(**overrides) -> SufficiencyMetrics:
    """Build SufficiencyMetrics with sensible defaults."""
    defaults = {
        "anchor_count": 3,
        "anchor_sets_active": 2,
        "neuron_count": 10,
        "intersection_count": 2,
        "top_activation": 0.8,
        "mean_activation": 0.4,
        "activation_entropy": 2.0,
        "activation_mass": 4.0,
        "coverage_ratio": 0.67,
        "focus_ratio": 0.5,
        "proximity_ratio": 0.3,
        "path_diversity": 0.6,
        "stab_converged": True,
        "stab_neurons_removed": 0,
    }
    defaults.update(overrides)
    return SufficiencyMetrics(**defaults)


# ===========================================================================
# 2.1: Session-aware depth adjustment
# ===========================================================================


class TestSessionAwareDepth:
    """Tests for session context biasing in depth selection."""

    @pytest.mark.asyncio
    async def test_no_session_no_bias(self):
        """Without session state, behavior is unchanged."""
        selector = _make_selector()
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["auth"]),
            DepthLevel.CONTEXT,
            session_state=None,
        )
        assert result.depth == DepthLevel.CONTEXT
        assert result.method == "rule_based"

    @pytest.mark.asyncio
    async def test_young_session_no_bias(self):
        """Session with <3 queries doesn't bias depth."""
        selector = _make_selector()
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=2)
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["auth"]),
            DepthLevel.CONTEXT,
            session_state=session,
        )
        assert result.depth == DepthLevel.CONTEXT
        assert result.method == "rule_based"

    @pytest.mark.asyncio
    async def test_primed_topic_goes_shallower(self):
        """Query about a primed topic → bias toward INSTANT (shallower)."""
        selector = _make_selector()
        session = _make_session_state(
            topic_ema={"auth": 0.7, "login": 0.6},
            query_count=10,
        )
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["auth"]),
            DepthLevel.CONTEXT,  # fallback=1
            session_state=session,
        )
        assert result.depth == DepthLevel.INSTANT  # CONTEXT-1 = INSTANT
        assert "session" in result.method

    @pytest.mark.asyncio
    async def test_new_topic_goes_deeper(self):
        """Query about a new topic in established session → bias toward deeper."""
        selector = _make_selector()
        session = _make_session_state(
            topic_ema={"auth": 0.7},  # Established session but query is about "database"
            query_count=10,
        )
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["database"]),
            DepthLevel.CONTEXT,  # fallback=1
            session_state=session,
        )
        assert result.depth == DepthLevel.HABIT  # CONTEXT+1 = HABIT
        assert "session" in result.method

    @pytest.mark.asyncio
    async def test_depth_clamped_at_min(self):
        """Primed topic at INSTANT doesn't go below 0."""
        selector = _make_selector()
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=10)
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["auth"]),
            DepthLevel.INSTANT,  # Already at 0
            session_state=session,
        )
        assert result.depth == DepthLevel.INSTANT  # Clamped at 0

    @pytest.mark.asyncio
    async def test_depth_clamped_at_max(self):
        """New topic at DEEP doesn't go above 3."""
        selector = _make_selector()
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=10)
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["database"]),
            DepthLevel.DEEP,  # Already at 3
            session_state=session,
        )
        assert result.depth == DepthLevel.DEEP  # Clamped at 3

    @pytest.mark.asyncio
    async def test_session_bias_with_bayesian(self):
        """Session bias is applied on top of Bayesian depth choice."""
        priors = {
            "auth": [
                DepthPrior(
                    entity_text="auth",
                    depth_level=DepthLevel.CONTEXT,
                    alpha=10.0,
                    beta=2.0,
                    total_queries=12,
                ),
            ]
        }
        selector = _make_selector(priors)
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=10)
        result = await selector.select_depth(
            _make_stimulus(entity_texts=["auth"]),
            DepthLevel.CONTEXT,
            session_state=session,
        )
        # Bayesian picks CONTEXT (score ~0.83), session says primed → INSTANT
        assert result.depth == DepthLevel.INSTANT
        assert "bayesian+session" in result.method

    @pytest.mark.asyncio
    async def test_no_entities_with_session_keyword_bias(self):
        """No entities but keywords match session → still biases."""
        selector = _make_selector()
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=10)
        result = await selector.select_depth(
            _make_stimulus(entity_texts=[], keywords=["auth"]),
            DepthLevel.CONTEXT,
            session_state=session,
        )
        assert result.depth == DepthLevel.INSTANT
        assert "session" in result.reason.lower() or "session" in result.method


# ===========================================================================
# 2.2: Calibration-driven gate threshold tuning
# ===========================================================================


class TestCalibrationThresholdTuning:
    """Tests for enhanced _apply_calibration with boost/dampen/downgrade."""

    def _make_result(self, gate: str = "intersection_convergence", confidence: float = 0.6):
        return SufficiencyResult(
            sufficient=True,
            confidence=confidence,
            gate=gate,
            reason="Test result",
            metrics=_make_sufficiency_metrics(),
        )

    def test_no_calibration_passthrough(self):
        """Without calibration data, result passes through unchanged."""
        result = self._make_result()
        adjusted = _apply_calibration(result, None)
        assert adjusted.confidence == result.confidence
        assert adjusted.sufficient

    def test_high_accuracy_boosts_confidence(self):
        """Gate with >0.8 accuracy gets 10% confidence boost."""
        result = self._make_result(confidence=0.6)
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.9, avg_confidence=0.7, sample_count=20
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert adjusted.confidence == pytest.approx(0.66, abs=0.01)  # 0.6 * 1.1
        assert adjusted.sufficient
        assert "boost" in adjusted.reason

    def test_low_accuracy_dampens_confidence(self):
        """Gate with <0.4 accuracy gets 30% confidence reduction."""
        result = self._make_result(confidence=0.6)
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.3, avg_confidence=0.5, sample_count=15
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert adjusted.confidence == pytest.approx(0.42, abs=0.01)  # 0.6 * 0.7
        assert adjusted.sufficient
        assert "dampen" in adjusted.reason

    def test_very_low_confidence_downgrades_to_insufficient(self):
        """Gate with avg_confidence <0.15 downgrades to insufficient."""
        result = self._make_result(confidence=0.6)
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.2, avg_confidence=0.1, sample_count=20
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert not adjusted.sufficient
        assert "downgrade" in adjusted.reason

    def test_insufficient_result_never_upgraded(self):
        """INSUFFICIENT results are never changed by calibration."""
        result = SufficiencyResult(
            sufficient=False,
            confidence=0.05,
            gate="no_anchors",
            reason="No anchors",
            metrics=_make_sufficiency_metrics(anchor_count=0),
        )
        calibration = {
            "no_anchors": GateCalibration(accuracy=0.95, avg_confidence=0.9, sample_count=100)
        }
        adjusted = _apply_calibration(result, calibration)
        assert not adjusted.sufficient  # Still insufficient

    def test_too_few_samples_no_adjustment(self):
        """With <10 samples, no adjustment is made."""
        result = self._make_result(confidence=0.6)
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.9, avg_confidence=0.7, sample_count=5
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert adjusted.confidence == 0.6  # Unchanged

    def test_confidence_clamped_at_1(self):
        """Boosted confidence doesn't exceed 1.0."""
        result = self._make_result(confidence=0.95)
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.95, avg_confidence=0.8, sample_count=50
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert adjusted.confidence <= 1.0


# ===========================================================================
# 2.3: Result quality feedback
# ===========================================================================


class TestResultQualityFeedback:
    """Tests for agent_used_result signal in depth prior feedback."""

    @pytest.mark.asyncio
    async def test_agent_used_result_forces_success(self):
        """agent_used_result=True always counts as success, even low confidence."""
        storage = AsyncMock()
        storage.get_depth_priors_batch = AsyncMock(
            return_value={"auth": [DepthPrior(entity_text="auth", depth_level=DepthLevel.CONTEXT)]}
        )
        storage.upsert_depth_prior = AsyncMock()

        selector = AdaptiveDepthSelector(storage)
        await selector.record_outcome(
            _make_stimulus(["auth"]),
            DepthLevel.CONTEXT,
            confidence=0.1,  # Below threshold
            fibers_matched=0,  # Below threshold
            agent_used_result=True,  # But agent used it!
        )

        # Verify update_success was called (alpha increased)
        call_args = storage.upsert_depth_prior.call_args[0][0]
        assert call_args.alpha == 2.0  # 1.0 + 1.0 (success)

    @pytest.mark.asyncio
    async def test_agent_unused_result_raises_bar(self):
        """agent_used_result=False raises the confidence bar to 0.5."""
        storage = AsyncMock()
        storage.get_depth_priors_batch = AsyncMock(
            return_value={"auth": [DepthPrior(entity_text="auth", depth_level=DepthLevel.CONTEXT)]}
        )
        storage.upsert_depth_prior = AsyncMock()

        selector = AdaptiveDepthSelector(storage)
        # Confidence 0.4 → above 0.3 threshold but below 0.5 raised bar
        await selector.record_outcome(
            _make_stimulus(["auth"]),
            DepthLevel.CONTEXT,
            confidence=0.4,
            fibers_matched=2,
            agent_used_result=False,
        )

        # Should be failure because raised bar (0.5) not met
        call_args = storage.upsert_depth_prior.call_args[0][0]
        assert call_args.beta == 2.0  # 1.0 + 1.0 (failure)

    @pytest.mark.asyncio
    async def test_no_signal_uses_default_heuristic(self):
        """agent_used_result=None uses base heuristic (conf>=0.3, fibers>=1)."""
        storage = AsyncMock()
        storage.get_depth_priors_batch = AsyncMock(
            return_value={"auth": [DepthPrior(entity_text="auth", depth_level=DepthLevel.CONTEXT)]}
        )
        storage.upsert_depth_prior = AsyncMock()

        selector = AdaptiveDepthSelector(storage)
        await selector.record_outcome(
            _make_stimulus(["auth"]),
            DepthLevel.CONTEXT,
            confidence=0.5,
            fibers_matched=2,
            agent_used_result=None,
        )

        call_args = storage.upsert_depth_prior.call_args[0][0]
        assert call_args.alpha == 2.0  # Success


# ===========================================================================
# 2.4: Dynamic RRF weight adjustment
# ===========================================================================


class TestDynamicRRFWeights:
    """Tests for retriever calibration and dynamic weight computation."""

    @pytest.mark.asyncio
    async def test_default_weights_when_no_data(self):
        """Returns default weights when no retriever calibration data exists."""
        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin

        mixin = MagicMock(spec=SQLiteCalibrationMixin)
        mixin._ensure_read_conn = MagicMock()
        mixin._get_brain_id = MagicMock(return_value="brain-1")

        conn = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value=cursor)
        mixin._ensure_read_conn.return_value = conn

        # Call the actual method
        result = await SQLiteCalibrationMixin.get_retriever_weights(mixin)
        assert result == dict(DEFAULT_RETRIEVER_WEIGHTS)

    def test_default_retriever_weights_has_all_types(self):
        """DEFAULT_RETRIEVER_WEIGHTS includes all 5 retriever types."""
        assert "time" in DEFAULT_RETRIEVER_WEIGHTS
        assert "entity" in DEFAULT_RETRIEVER_WEIGHTS
        assert "keyword" in DEFAULT_RETRIEVER_WEIGHTS
        assert "embedding" in DEFAULT_RETRIEVER_WEIGHTS
        assert "graph_expansion" in DEFAULT_RETRIEVER_WEIGHTS


# ===========================================================================
# 2.5: Activation strategy auto-selection
# ===========================================================================


class TestStrategyAutoSelection:
    """Tests for auto-selecting activation strategy based on graph density."""

    def _make_engine(self, strategy: str = "auto"):
        """Create a minimal mock of the retrieval engine for strategy test."""
        from neural_memory.engine.retrieval import ReflexPipeline

        storage = AsyncMock()
        config = MagicMock()
        config.activation_strategy = strategy
        config.adaptive_depth_enabled = False
        config.max_spread_hops = 4
        config.activation_threshold = 0.2
        config.lateral_inhibition_k = 10
        config.lateral_inhibition_factor = 0.3
        config.rrf_k = 60
        config.graph_expansion_enabled = False
        config.max_context_tokens = 4000
        config.ppr_damping = 0.15
        config.ppr_iterations = 20
        config.ppr_epsilon = 1e-6
        config.reinforcement_delta = 0.1

        # Construct engine with mock
        engine = object.__new__(ReflexPipeline)
        engine._storage = storage
        engine._config = config
        engine._ppr_activator = MagicMock()  # Pretend PPR is available
        return engine

    @pytest.mark.asyncio
    async def test_sparse_graph_selects_classic(self):
        """Graph density <3 → classic BFS."""
        engine = self._make_engine()
        engine._storage.get_graph_density = AsyncMock(return_value=1.5)
        result = await engine._auto_select_strategy()
        assert result == "classic"

    @pytest.mark.asyncio
    async def test_dense_graph_selects_ppr(self):
        """Graph density >8 → PPR."""
        engine = self._make_engine()
        engine._storage.get_graph_density = AsyncMock(return_value=12.0)
        result = await engine._auto_select_strategy()
        assert result == "ppr"

    @pytest.mark.asyncio
    async def test_medium_graph_selects_hybrid(self):
        """Graph density 3-8 → hybrid."""
        engine = self._make_engine()
        engine._storage.get_graph_density = AsyncMock(return_value=5.0)
        result = await engine._auto_select_strategy()
        assert result == "hybrid"

    @pytest.mark.asyncio
    async def test_fallback_on_storage_error(self):
        """If storage doesn't support graph_density, falls back to classic."""
        engine = self._make_engine()
        engine._storage.get_graph_density = AsyncMock(side_effect=AttributeError)
        result = await engine._auto_select_strategy()
        assert result == "classic"

    @pytest.mark.asyncio
    async def test_dense_graph_no_ppr_falls_to_classic(self):
        """Dense graph but no PPR activator → classic."""
        engine = self._make_engine()
        engine._ppr_activator = None
        engine._storage.get_graph_density = AsyncMock(return_value=12.0)
        result = await engine._auto_select_strategy()
        assert result == "classic"

    @pytest.mark.asyncio
    async def test_medium_graph_no_ppr_falls_to_classic(self):
        """Medium graph but no PPR activator → classic."""
        engine = self._make_engine()
        engine._ppr_activator = None
        engine._storage.get_graph_density = AsyncMock(return_value=5.0)
        result = await engine._auto_select_strategy()
        assert result == "classic"


# ===========================================================================
# Integration: backward compatibility
# ===========================================================================


class TestBackwardCompat:
    """Verify existing behavior is preserved when new features are inactive."""

    @pytest.mark.asyncio
    async def test_select_depth_no_session_same_as_before(self):
        """Without session_state, select_depth works identically to v1."""
        priors = {
            "auth": [
                DepthPrior(
                    entity_text="auth",
                    depth_level=DepthLevel.DEEP,
                    alpha=8.0,
                    beta=2.0,
                    total_queries=10,
                ),
            ]
        }
        selector = _make_selector(priors)
        result = await selector.select_depth(
            _make_stimulus(["auth"]),
            DepthLevel.CONTEXT,
        )
        # Should pick DEEP (score 0.8) via Bayesian
        assert result.depth == DepthLevel.DEEP
        assert result.method == "bayesian"

    def test_apply_depth_bias_boundaries(self):
        """_apply_depth_bias clamps correctly."""
        assert AdaptiveDepthSelector._apply_depth_bias(DepthLevel.INSTANT, -1) == DepthLevel.INSTANT
        assert AdaptiveDepthSelector._apply_depth_bias(DepthLevel.DEEP, +1) == DepthLevel.DEEP
        assert AdaptiveDepthSelector._apply_depth_bias(DepthLevel.CONTEXT, +1) == DepthLevel.HABIT
        assert AdaptiveDepthSelector._apply_depth_bias(DepthLevel.HABIT, -1) == DepthLevel.CONTEXT

    def test_compute_session_bias_empty_stimulus(self):
        """No entities or keywords → no bias."""
        selector = _make_selector()
        session = _make_session_state(topic_ema={"auth": 0.8}, query_count=10)
        bias = selector._compute_session_bias(
            _make_stimulus(entity_texts=[], keywords=[]),
            session,
        )
        assert bias == 0

    def test_calibration_unchanged_for_mid_accuracy(self):
        """Accuracy between 0.4-0.8 doesn't trigger boost or dampen."""
        result = SufficiencyResult(
            sufficient=True,
            confidence=0.6,
            gate="intersection_convergence",
            reason="Test",
            metrics=_make_sufficiency_metrics(),
        )
        calibration = {
            "intersection_convergence": GateCalibration(
                accuracy=0.6, avg_confidence=0.5, sample_count=20
            )
        }
        adjusted = _apply_calibration(result, calibration)
        assert adjusted.confidence == 0.6  # No change
