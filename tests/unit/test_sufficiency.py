"""Tests for the post-stabilization sufficiency check (v2.16.0)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.engine.sufficiency import (
    SufficiencyMetrics,
    SufficiencyResult,
    _focus_ratio,
    _shannon_entropy,
    check_sufficiency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeActivation:
    """Minimal stand-in for ActivationResult."""

    def __init__(
        self,
        activation_level: float,
        hop_distance: int = 1,
        source_anchor: str = "a-0",
    ) -> None:
        self.activation_level = activation_level
        self.hop_distance = hop_distance
        self.source_anchor = source_anchor


def _make_activations(
    specs: list[tuple[str, float, int, str]],
) -> dict[str, _FakeActivation]:
    """Build activations dict from (id, level, hop, source_anchor) tuples."""
    return {nid: _FakeActivation(level, hop, src) for nid, level, hop, src in specs}


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestNoAnchors:
    def test_empty_anchor_sets(self) -> None:
        result = check_sufficiency(
            activations={},
            anchor_sets=[],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is False
        assert result.gate == "no_anchors"
        assert result.confidence == 0.0

    def test_anchor_sets_with_empty_lists(self) -> None:
        result = check_sufficiency(
            activations={},
            anchor_sets=[[], []],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is False
        assert result.gate == "no_anchors"


class TestEmptyLandscape:
    def test_no_surviving_neurons(self) -> None:
        result = check_sufficiency(
            activations={},
            anchor_sets=[["a-1", "a-2"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=10,
        )
        assert result.sufficient is False
        assert result.gate == "empty_landscape"
        assert result.confidence == 0.0


class TestUnstableNoise:
    def test_unstable_with_low_peak(self) -> None:
        acts = _make_activations(
            [
                ("n-0", 0.15, 3, "a-0"),
                ("n-1", 0.10, 2, "a-0"),
            ]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=False,
            stab_neurons_removed=20,
        )
        assert result.sufficient is False
        assert result.gate == "unstable_noise"
        assert result.confidence <= 0.1

    def test_unstable_but_clear_winner_passes(self) -> None:
        """If top_activation >= 0.3, unstable_noise should NOT trigger."""
        acts = _make_activations(
            [
                ("n-0", 0.5, 1, "a-0"),
                ("n-1", 0.05, 3, "a-0"),
            ]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=False,
            stab_neurons_removed=20,
        )
        assert result.gate != "unstable_noise"


class TestAmbiguousSpread:
    def test_high_entropy_low_focus_low_peak(self) -> None:
        # 20 neurons with nearly uniform activation → high entropy
        acts = _make_activations([(f"n-{i}", 0.1 + 0.005 * i, 2, "a-0") for i in range(20)])
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is False
        assert result.gate == "ambiguous_spread"
        assert result.confidence <= 0.1

    def test_ambiguous_but_standout_passes(self) -> None:
        """Safety valve: if any neuron has top_act >= 0.3, don't block."""
        acts = _make_activations(
            [(f"n-{i}", 0.05, 2, "a-0") for i in range(19)] + [("n-star", 0.5, 1, "a-0")]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.gate != "ambiguous_spread"


class TestIntersectionConvergence:
    def test_multi_anchor_convergence(self) -> None:
        acts = _make_activations(
            [
                ("n-0", 0.8, 1, "a-0"),
                ("n-1", 0.6, 1, "a-1"),
                ("n-2", 0.3, 2, "a-0"),
            ]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"], ["a-1"]],
            intersections=["n-0", "n-1"],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is True
        assert result.gate == "intersection_convergence"
        assert result.confidence > 0.3


class TestHighCoverageStrongHit:
    def test_good_coverage_and_peak(self) -> None:
        acts = _make_activations(
            [
                ("n-0", 0.9, 1, "a-0"),
                ("n-1", 0.7, 1, "a-1"),
                ("n-2", 0.2, 2, "a-0"),
            ]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"], ["a-1"]],
            intersections=[],  # no intersections → skip gate 5
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is True
        assert result.gate == "high_coverage_strong_hit"


class TestFocusedResult:
    def test_few_neurons_clear_winner(self) -> None:
        # Use 2 anchor sets with only 1 active → coverage=0.5 but top<0.7
        # so high_coverage_strong_hit won't trigger, but focused_result will
        acts = _make_activations(
            [
                ("n-0", 0.6, 1, "a-0"),
                ("n-1", 0.1, 2, "a-0"),
            ]
        )
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"], ["a-missing"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is True
        assert result.gate == "focused_result"


class TestDefaultPass:
    def test_moderate_signal_falls_through(self) -> None:
        # 10 neurons, moderate activation, no gate triggers
        acts = _make_activations([(f"n-{i}", 0.3 + 0.02 * i, 2, "a-0") for i in range(10)])
        result = check_sufficiency(
            activations=acts,
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
        )
        assert result.sufficient is True
        assert result.gate == "default_pass"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_single_element(self) -> None:
        assert _shannon_entropy([1.0]) == 0.0

    def test_uniform_distribution(self) -> None:
        # 8 equal values → log2(8) = 3.0 bits
        entropy = _shannon_entropy([1.0] * 8)
        assert abs(entropy - 3.0) < 0.01

    def test_two_peak(self) -> None:
        # Two equal values → log2(2) = 1.0 bit
        entropy = _shannon_entropy([0.5, 0.5])
        assert abs(entropy - 1.0) < 0.01

    def test_empty(self) -> None:
        assert _shannon_entropy([]) == 0.0

    def test_all_zeros(self) -> None:
        assert _shannon_entropy([0.0, 0.0]) == 0.0


class TestFocusRatio:
    def test_single_dominant(self) -> None:
        ratio = _focus_ratio([0.9, 0.05, 0.05])
        assert ratio == 1.0  # top-3 of 3 elements = 100%

    def test_empty(self) -> None:
        assert _focus_ratio([]) == 0.0

    def test_uniform(self) -> None:
        ratio = _focus_ratio([0.1] * 10)
        assert abs(ratio - 0.3) < 0.01  # top-3 of 10 uniform = 30%


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_sufficiency_result_frozen(self) -> None:
        metrics = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=1,
            intersection_count=0,
            top_activation=0.5,
            mean_activation=0.5,
            activation_entropy=0.0,
            activation_mass=0.5,
            coverage_ratio=1.0,
            focus_ratio=1.0,
            proximity_ratio=1.0,
            path_diversity=1.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )
        result = SufficiencyResult(
            sufficient=True,
            confidence=0.5,
            gate="test",
            reason="test",
            metrics=metrics,
        )
        with pytest.raises(FrozenInstanceError):
            result.sufficient = False  # type: ignore[misc]

    def test_metrics_frozen(self) -> None:
        metrics = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=1,
            intersection_count=0,
            top_activation=0.5,
            mean_activation=0.5,
            activation_entropy=0.0,
            activation_mass=0.5,
            coverage_ratio=1.0,
            focus_ratio=1.0,
            proximity_ratio=1.0,
            path_diversity=1.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )
        with pytest.raises(FrozenInstanceError):
            metrics.anchor_count = 99  # type: ignore[misc]
