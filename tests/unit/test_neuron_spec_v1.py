"""Unit tests for NeuronSpec v1: sigmoid activation, firing threshold, refractory, lateral inhibition."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import NeuronState
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.utils.timeutils import utcnow


class TestSigmoidActivation:
    """Tests for sigmoid activation function."""

    def test_sigmoid_midpoint(self) -> None:
        """Sigmoid(0.5) should be exactly 0.5 (midpoint identity)."""
        state = NeuronState(neuron_id="test-1")
        activated = state.activate(0.5)
        assert activated.activation_level == pytest.approx(0.5, abs=0.001)

    def test_sigmoid_low_input(self) -> None:
        """Sigmoid(0.0) should be ~0.05 (suppresses noise)."""
        state = NeuronState(neuron_id="test-1")
        activated = state.activate(0.0)
        assert activated.activation_level == pytest.approx(0.0474, abs=0.01)

    def test_sigmoid_high_input(self) -> None:
        """Sigmoid(1.0) should be ~0.95 (saturates gracefully)."""
        state = NeuronState(neuron_id="test-1")
        activated = state.activate(1.0)
        assert activated.activation_level == pytest.approx(0.9526, abs=0.01)

    def test_sigmoid_saturates_above_one(self) -> None:
        """Inputs > 1.0 are clamped to 1.0 before sigmoid."""
        state = NeuronState(neuron_id="test-1")
        a1 = state.activate(1.0)
        a2 = state.activate(2.0)
        assert a1.activation_level == pytest.approx(a2.activation_level, abs=0.001)

    def test_sigmoid_custom_steepness(self) -> None:
        """Custom sigmoid steepness changes the curve shape."""
        state = NeuronState(neuron_id="test-1")
        # Steeper sigmoid should push values further from 0.5
        steep = state.activate(0.8, sigmoid_steepness=12.0)
        normal = state.activate(0.8, sigmoid_steepness=6.0)
        assert steep.activation_level > normal.activation_level

    def test_sigmoid_preserves_monotonicity(self) -> None:
        """Higher input should always produce higher output."""
        state = NeuronState(neuron_id="test-1")
        levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results = [state.activate(lv).activation_level for lv in levels]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]


class TestFiringThreshold:
    """Tests for firing threshold behavior."""

    def test_below_threshold_not_fired(self) -> None:
        """Neuron below firing threshold should not be 'fired'."""
        state = NeuronState(neuron_id="test-1", firing_threshold=0.3)
        # Input 0.1 → sigmoid ≈ 0.12, below 0.3
        activated = state.activate(0.1)
        assert not activated.fired

    def test_above_threshold_fired(self) -> None:
        """Neuron above firing threshold should be 'fired'."""
        state = NeuronState(neuron_id="test-1", firing_threshold=0.3)
        activated = state.activate(0.5)
        assert activated.fired

    def test_default_threshold(self) -> None:
        """Default firing threshold is 0.3."""
        state = NeuronState(neuron_id="test-1")
        assert state.firing_threshold == 0.3

    def test_custom_threshold(self) -> None:
        """Custom firing threshold is respected."""
        state = NeuronState(neuron_id="test-1", firing_threshold=0.8)
        # 0.5 → sigmoid 0.5, below 0.8
        activated = state.activate(0.5)
        assert not activated.fired


class TestRefractoryPeriod:
    """Tests for refractory period behavior."""

    def test_refractory_blocks_immediate_reactivation(self) -> None:
        """Neuron in refractory period cannot be activated again."""
        state = NeuronState(neuron_id="test-1")
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        # First activation succeeds
        activated = state.activate(0.8, now=t0)
        assert activated.access_frequency == 1
        assert activated.refractory_until is not None

        # Immediate re-activation returns unchanged state
        t1 = t0 + timedelta(milliseconds=100)
        blocked = activated.activate(0.9, now=t1)
        assert blocked.access_frequency == 1  # No increment
        assert blocked is activated  # Same object returned

    def test_refractory_expires(self) -> None:
        """Neuron can fire again after refractory period expires."""
        state = NeuronState(neuron_id="test-1", refractory_period_ms=500.0)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        activated = state.activate(0.8, now=t0)
        assert activated.access_frequency == 1

        # After refractory period, can activate again
        t1 = t0 + timedelta(milliseconds=600)
        reactivated = activated.activate(0.7, now=t1)
        assert reactivated.access_frequency == 2

    def test_in_refractory_property(self) -> None:
        """in_refractory property correctly reports cooldown state."""
        # No refractory set
        state = NeuronState(neuron_id="test-1")
        assert not state.in_refractory

        # Set refractory in the future
        future = utcnow() + timedelta(seconds=10)
        state_with_ref = NeuronState(neuron_id="test-1", refractory_until=future)
        assert state_with_ref.in_refractory

        # Set refractory in the past
        past = utcnow() - timedelta(seconds=10)
        state_expired = NeuronState(neuron_id="test-1", refractory_until=past)
        assert not state_expired.in_refractory

    def test_subthreshold_no_refractory(self) -> None:
        """Sub-threshold activation does not set refractory period."""
        state = NeuronState(neuron_id="test-1", firing_threshold=0.9)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        # Input 0.5 → sigmoid 0.5, below threshold 0.9
        activated = state.activate(0.5, now=t0)
        assert not activated.fired
        # Refractory should not be set (remains as parent's value)
        assert activated.refractory_until == state.refractory_until

    def test_custom_refractory_period(self) -> None:
        """Custom refractory period is used correctly."""
        state = NeuronState(neuron_id="test-1", refractory_period_ms=1000.0)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        activated = state.activate(0.8, now=t0)
        expected_until = t0 + timedelta(milliseconds=1000)
        assert activated.refractory_until == expected_until


class TestDecayPreservesFields:
    """Tests that decay preserves all NeuronSpec v1 fields."""

    def test_decay_preserves_new_fields(self) -> None:
        """decay() should preserve firing_threshold, refractory, etc."""
        ref_time = datetime(2026, 1, 1, 12, 0, 0)
        state = NeuronState(
            neuron_id="test-1",
            activation_level=0.8,
            firing_threshold=0.4,
            refractory_until=ref_time,
            refractory_period_ms=750.0,
            homeostatic_target=0.6,
        )

        decayed = state.decay(86400)  # 1 day
        assert decayed.activation_level < 0.8
        assert decayed.firing_threshold == 0.4
        assert decayed.refractory_until == ref_time
        assert decayed.refractory_period_ms == 750.0
        assert decayed.homeostatic_target == 0.6


class TestBackwardCompatibility:
    """Tests that default NeuronState is backward compatible."""

    def test_default_state_has_all_fields(self) -> None:
        """New NeuronState with defaults should work like pre-v1."""
        state = NeuronState(neuron_id="test-1")
        assert state.firing_threshold == 0.3
        assert state.refractory_until is None
        assert state.refractory_period_ms == 500.0
        assert state.homeostatic_target == 0.5
        assert state.activation_level == 0.0
        assert not state.fired
        assert not state.in_refractory
        assert not state.is_active


class TestLateralInhibition:
    """Tests for lateral inhibition in ReflexPipeline."""

    def _make_activations(self, count: int) -> dict[str, ActivationResult]:
        """Create test activations with descending levels."""
        activations: dict[str, ActivationResult] = {}
        for i in range(count):
            level = 1.0 - (i * 0.04)
            activations[f"n-{i}"] = ActivationResult(
                neuron_id=f"n-{i}",
                activation_level=level,
                hop_distance=1,
                path=["anchor", f"n-{i}"],
                source_anchor="anchor",
            )
        return activations

    def test_top_k_unchanged(self) -> None:
        """Top-K neurons should survive lateral inhibition unchanged."""
        config = BrainConfig(lateral_inhibition_k=5, lateral_inhibition_factor=0.3)
        pipeline = ReflexPipeline.__new__(ReflexPipeline)
        pipeline._config = config

        activations = self._make_activations(10)
        result = pipeline._apply_lateral_inhibition(activations)

        # Top 5 should be unchanged
        sorted_original = sorted(
            activations.items(), key=lambda x: x[1].activation_level, reverse=True
        )
        for nid, act in sorted_original[:5]:
            assert result[nid].activation_level == act.activation_level

    def test_losers_suppressed(self) -> None:
        """Neurons outside top-K should be suppressed."""
        config = BrainConfig(
            lateral_inhibition_k=5,
            lateral_inhibition_factor=0.3,
            activation_threshold=0.0,
            graph_density_scaling_enabled=False,
        )
        pipeline = ReflexPipeline.__new__(ReflexPipeline)
        pipeline._config = config

        activations = self._make_activations(10)
        result = pipeline._apply_lateral_inhibition(activations)

        sorted_original = sorted(
            activations.items(), key=lambda x: x[1].activation_level, reverse=True
        )
        for nid, act in sorted_original[5:]:
            assert result[nid].activation_level == pytest.approx(
                act.activation_level * 0.3, abs=0.001
            )

    def test_below_threshold_dropped(self) -> None:
        """Suppressed neurons below activation_threshold should be dropped."""
        config = BrainConfig(
            lateral_inhibition_k=5,
            lateral_inhibition_factor=0.3,
            activation_threshold=0.2,
        )
        pipeline = ReflexPipeline.__new__(ReflexPipeline)
        pipeline._config = config

        activations = self._make_activations(20)
        result = pipeline._apply_lateral_inhibition(activations)

        # All results should be above threshold
        for act in result.values():
            assert act.activation_level >= 0.2

        # Should have fewer results than original
        assert len(result) < len(activations)

    def test_small_set_unchanged(self) -> None:
        """Sets smaller than K should not be modified."""
        config = BrainConfig(lateral_inhibition_k=10)
        pipeline = ReflexPipeline.__new__(ReflexPipeline)
        pipeline._config = config

        activations = self._make_activations(5)
        result = pipeline._apply_lateral_inhibition(activations)

        assert len(result) == 5
        for nid, act in activations.items():
            assert result[nid].activation_level == act.activation_level

    def test_equal_to_k_unchanged(self) -> None:
        """Exactly K neurons should not be modified."""
        config = BrainConfig(lateral_inhibition_k=10)
        pipeline = ReflexPipeline.__new__(ReflexPipeline)
        pipeline._config = config

        activations = self._make_activations(10)
        result = pipeline._apply_lateral_inhibition(activations)

        assert len(result) == 10
        for nid, act in activations.items():
            assert result[nid].activation_level == act.activation_level
