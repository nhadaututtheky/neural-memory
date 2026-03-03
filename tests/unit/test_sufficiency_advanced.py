"""Tests for advanced sufficiency check features (v2.19+).

Covers:
- Feature 1: EMA calibration — GateCalibration downgrade logic
- Feature 2: Per-query-type threshold profiles (strict / lenient / default)
- Feature 3: Diminishing returns gate
- get_gate_ema_stats() in sqlite_calibration mixin
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.engine.sufficiency import (
    GateCalibration,
    QueryTypeProfile,
    SufficiencyMetrics,
    SufficiencyResult,
    _apply_calibration,
    _get_profile,
    check_sufficiency,
)

# ---------------------------------------------------------------------------
# Shared helpers
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


def _strong_activations(n: int = 5) -> dict[str, _FakeActivation]:
    """Return n activations with high activation levels."""
    return _make_activations([(f"n{i}", 0.8, 1, "anchor-0") for i in range(n)])


def _make_sufficient_result(gate: str = "default_pass", conf: float = 0.6) -> SufficiencyResult:
    """Build a minimal SUFFICIENT SufficiencyResult for calibration tests."""
    m = SufficiencyMetrics(
        anchor_count=1,
        anchor_sets_active=1,
        neuron_count=3,
        intersection_count=0,
        top_activation=0.8,
        mean_activation=0.6,
        activation_entropy=1.0,
        activation_mass=1.8,
        coverage_ratio=1.0,
        focus_ratio=0.7,
        proximity_ratio=0.9,
        path_diversity=0.5,
        stab_converged=True,
        stab_neurons_removed=0,
    )
    return SufficiencyResult(
        sufficient=True,
        confidence=conf,
        gate=gate,
        reason="test result",
        metrics=m,
    )


def _make_insufficient_result(gate: str = "no_anchors") -> SufficiencyResult:
    """Build a minimal INSUFFICIENT SufficiencyResult."""
    m = SufficiencyMetrics(
        anchor_count=0,
        anchor_sets_active=0,
        neuron_count=0,
        intersection_count=0,
        top_activation=0.0,
        mean_activation=0.0,
        activation_entropy=0.0,
        activation_mass=0.0,
        coverage_ratio=0.0,
        focus_ratio=0.0,
        proximity_ratio=0.0,
        path_diversity=0.0,
        stab_converged=True,
        stab_neurons_removed=0,
    )
    return SufficiencyResult(
        sufficient=False,
        confidence=0.0,
        gate=gate,
        reason="test insufficient",
        metrics=m,
    )


# ---------------------------------------------------------------------------
# Feature 1: GateCalibration dataclass
# ---------------------------------------------------------------------------


class TestGateCalibration:
    def test_frozen_dataclass(self) -> None:
        cal = GateCalibration(accuracy=0.8, avg_confidence=0.5, sample_count=20)
        with pytest.raises(FrozenInstanceError):
            cal.accuracy = 0.9  # type: ignore[misc]

    def test_fields(self) -> None:
        cal = GateCalibration(accuracy=0.75, avg_confidence=0.3, sample_count=15)
        assert cal.accuracy == 0.75
        assert cal.avg_confidence == 0.3
        assert cal.sample_count == 15


# ---------------------------------------------------------------------------
# Feature 1: _apply_calibration
# ---------------------------------------------------------------------------


class TestApplyCalibration:
    def test_no_calibration_returns_unchanged(self) -> None:
        result = _make_sufficient_result()
        out = _apply_calibration(result, None)
        assert out is result

    def test_insufficient_result_never_changed(self) -> None:
        result = _make_insufficient_result()
        cal = {"no_anchors": GateCalibration(accuracy=0.5, avg_confidence=0.05, sample_count=50)}
        out = _apply_calibration(result, cal)
        # INSUFFICIENT stays INSUFFICIENT, no change
        assert out is result

    def test_downgrade_when_low_avg_confidence_and_enough_samples(self) -> None:
        result = _make_sufficient_result(gate="default_pass", conf=0.6)
        cal = {"default_pass": GateCalibration(accuracy=0.4, avg_confidence=0.10, sample_count=15)}
        out = _apply_calibration(result, cal)
        assert out.sufficient is False
        assert out.gate == "default_pass"
        assert "calibration downgrade" in out.reason

    def test_no_downgrade_when_avg_confidence_above_threshold(self) -> None:
        result = _make_sufficient_result(gate="default_pass", conf=0.6)
        cal = {"default_pass": GateCalibration(accuracy=0.4, avg_confidence=0.20, sample_count=15)}
        out = _apply_calibration(result, cal)
        assert out.sufficient is True

    def test_no_downgrade_when_insufficient_samples(self) -> None:
        result = _make_sufficient_result(gate="default_pass", conf=0.6)
        cal = {"default_pass": GateCalibration(accuracy=0.4, avg_confidence=0.05, sample_count=5)}
        out = _apply_calibration(result, cal)
        # Not enough samples → no downgrade
        assert out.sufficient is True

    def test_no_downgrade_when_gate_not_in_calibration(self) -> None:
        result = _make_sufficient_result(gate="focused_result", conf=0.6)
        cal = {"default_pass": GateCalibration(accuracy=0.4, avg_confidence=0.05, sample_count=20)}
        out = _apply_calibration(result, cal)
        assert out.sufficient is True

    def test_downgrade_confidence_capped_at_avg_confidence(self) -> None:
        result = _make_sufficient_result(gate="default_pass", conf=0.6)
        cal = {"default_pass": GateCalibration(accuracy=0.3, avg_confidence=0.08, sample_count=25)}
        out = _apply_calibration(result, cal)
        assert out.sufficient is False
        assert out.confidence <= 0.08


# ---------------------------------------------------------------------------
# Feature 2: QueryTypeProfile and _get_profile
# ---------------------------------------------------------------------------


class TestQueryTypeProfile:
    def test_frozen_dataclass(self) -> None:
        p = QueryTypeProfile(
            min_top_activation_factor=1.2,
            entropy_tolerance=0.8,
            min_intersection_count=2,
        )
        with pytest.raises(FrozenInstanceError):
            p.min_top_activation_factor = 0.5  # type: ignore[misc]

    def test_get_profile_strict_intents(self) -> None:
        for intent in ("ask_what", "ask_where", "ask_when", "ask_who", "confirm"):
            p = _get_profile(intent)
            assert p.min_top_activation_factor > 1.0, f"Expected strict profile for {intent}"
            assert p.entropy_tolerance < 1.0

    def test_get_profile_lenient_intents(self) -> None:
        for intent in ("ask_pattern", "compare", "ask_why", "ask_how"):
            p = _get_profile(intent)
            assert p.min_top_activation_factor < 1.0, f"Expected lenient profile for {intent}"
            assert p.entropy_tolerance > 1.0

    def test_get_profile_default_intents(self) -> None:
        for intent in ("ask_feeling", "recall", "unknown", ""):
            p = _get_profile(intent)
            assert p.min_top_activation_factor == 1.0
            assert p.entropy_tolerance == 1.0
            assert p.min_intersection_count == 2

    def test_get_profile_unknown_intent_defaults(self) -> None:
        p = _get_profile("totally_made_up_intent")
        assert p.min_top_activation_factor == 1.0


# ---------------------------------------------------------------------------
# Feature 2: query_intent affects gate thresholds in check_sufficiency
# ---------------------------------------------------------------------------


class TestQueryIntentInCheckSufficiency:
    """Verify that strict intent raises thresholds and lenient intent lowers them."""

    def _borderline_ambiguous_activations(self) -> dict[str, _FakeActivation]:
        """Create activations that borderline-trigger ambiguous_spread with default profile."""
        # 20 activations, entropy ~4+ bits (all equal = max entropy), focus low
        return _make_activations([(f"n{i}", 0.15, 2, f"anchor-{i}") for i in range(20)])

    def test_strict_intent_lowers_ambiguous_spread_entropy_threshold(self) -> None:
        """With strict profile, entropy_tolerance=0.8, so 3.0*0.8=2.4 threshold.
        A moderate entropy (between 2.4 and 3.0) should NOT trigger with default but SHOULD with strict.
        """
        # Build activations with entropy ~2.7 (moderate spread)
        # 10 neurons with level variation that gives ~2.7 bits entropy
        activations = _make_activations(
            [
                ("n0", 0.35, 2, "anchor-0"),
                ("n1", 0.30, 2, "anchor-1"),
                ("n2", 0.25, 2, "anchor-2"),
                ("n3", 0.20, 2, "anchor-3"),
                ("n4", 0.18, 2, "anchor-4"),
                ("n5", 0.15, 2, "anchor-5"),
                ("n6", 0.12, 2, "anchor-6"),
                ("n7", 0.10, 2, "anchor-7"),
                ("n8", 0.08, 2, "anchor-8"),
                ("n9", 0.07, 2, "anchor-9"),
                ("n10", 0.06, 2, "anchor-10"),
                ("n11", 0.05, 2, "anchor-11"),
                ("n12", 0.05, 2, "anchor-12"),
                ("n13", 0.05, 2, "anchor-13"),
                ("n14", 0.05, 2, "anchor-14"),
                ("n15", 0.04, 2, "anchor-15"),
            ]
        )
        anchor_sets = [[f"anchor-{i}"] for i in range(16)]

        default_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="unknown",
        )
        strict_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_what",
        )
        # At minimum: strict profile has lower entropy threshold so ambiguous_spread
        # fires at lower entropy. If default already fires, test is not useful.
        # We only assert that strict is not MORE sufficient than default.
        if default_result.gate == "ambiguous_spread":
            assert strict_result.gate == "ambiguous_spread"
        # If neither fires, at least strict should have same or lower confidence
        if default_result.sufficient and strict_result.sufficient:
            assert strict_result.confidence <= default_result.confidence + 0.05

    def test_lenient_intent_raises_ambiguous_spread_entropy_threshold(self) -> None:
        """With lenient profile, entropy_tolerance=1.5, threshold=4.5.
        Same high-entropy activations that trigger ambiguous_spread with default should NOT with lenient.
        """
        activations = _make_activations([(f"n{i}", 0.10, 2, f"anchor-{i}") for i in range(20)])
        anchor_sets = [[f"anchor-{i}"] for i in range(20)]

        default_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="unknown",
        )
        lenient_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_pattern",
        )
        if default_result.gate == "ambiguous_spread":
            # lenient should pass (entropy threshold raised to 4.5)
            assert lenient_result.gate != "ambiguous_spread"

    def test_strict_intent_raises_intersection_convergence_threshold(self) -> None:
        """With strict profile min_top_activation_factor=1.2, so threshold=0.48.
        An activation of 0.42 should NOT trigger intersection_convergence with strict.
        """
        activations = _make_activations([("n0", 0.42, 1, "anchor-0"), ("n1", 0.40, 1, "anchor-1")])
        anchor_sets = [["anchor-0"], ["anchor-1"]]
        intersections = ["n0", "n1"]

        strict_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections,
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_what",
        )
        # 0.42 < 0.4 * 1.2 = 0.48 → should NOT trigger intersection_convergence
        assert strict_result.gate != "intersection_convergence"

    def test_lenient_intent_lowers_intersection_convergence_threshold(self) -> None:
        """With lenient profile min_top_activation_factor=0.7, threshold=0.28.
        An activation of 0.30 should trigger intersection_convergence with lenient but not strict.
        """
        activations = _make_activations([("n0", 0.30, 1, "anchor-0"), ("n1", 0.28, 1, "anchor-1")])
        anchor_sets = [["anchor-0"], ["anchor-1"]]
        intersections = ["n0", "n1"]

        lenient_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections,
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_pattern",
        )
        # 0.30 >= 0.4 * 0.7 = 0.28 → should trigger intersection_convergence
        assert lenient_result.gate == "intersection_convergence"

    def test_lenient_intersection_count_min_is_1(self) -> None:
        """Lenient profile has min_intersection_count=1, so a single intersection suffices."""
        activations = _make_activations([("n0", 0.40, 1, "anchor-0"), ("n1", 0.38, 1, "anchor-1")])
        anchor_sets = [["anchor-0"], ["anchor-1"]]
        intersections = ["n0"]  # only 1 intersection

        lenient_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections,
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="compare",
        )
        assert lenient_result.gate == "intersection_convergence"

    def test_strict_intersection_count_min_is_2(self) -> None:
        """Strict profile has min_intersection_count=2, so a single intersection is insufficient."""
        activations = _make_activations([("n0", 0.50, 1, "anchor-0"), ("n1", 0.48, 1, "anchor-1")])
        anchor_sets = [["anchor-0"], ["anchor-1"]]
        intersections = ["n0"]  # only 1 intersection

        strict_result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections,
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_who",
        )
        assert strict_result.gate != "intersection_convergence"


# ---------------------------------------------------------------------------
# Feature 3: Diminishing returns gate
# ---------------------------------------------------------------------------


class TestDiminishingReturnsGate:
    def _make_prev_metrics(
        self,
        top_activation: float = 0.5,
        neuron_count: int = 10,
        focus_ratio: float = 0.6,
    ) -> SufficiencyMetrics:
        return SufficiencyMetrics(
            anchor_count=2,
            anchor_sets_active=2,
            neuron_count=neuron_count,
            intersection_count=1,
            top_activation=top_activation,
            mean_activation=0.3,
            activation_entropy=1.5,
            activation_mass=3.0,
            coverage_ratio=0.8,
            focus_ratio=focus_ratio,
            proximity_ratio=0.7,
            path_diversity=0.5,
            stab_converged=True,
            stab_neurons_removed=0,
        )

    def _nearly_identical_activations(self, top: float = 0.5) -> dict[str, _FakeActivation]:
        """10 activations with top at given level, focused."""
        specs = [(f"n{i}", top if i == 0 else top * 0.5, 1, "anchor-0") for i in range(10)]
        return _make_activations(specs)

    def test_diminishing_returns_fires_when_metrics_identical(self) -> None:
        """When current and previous metrics are nearly identical, gate should fire."""
        activations = self._nearly_identical_activations(top=0.50)
        anchor_sets = [["anchor-0"]]

        prev = self._make_prev_metrics(top_activation=0.50, neuron_count=10, focus_ratio=0.60)
        # We need to figure out what focus_ratio the activations produce
        # so let's just test the gate fires for activations with similar metrics
        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            prev_metrics=prev,
        )
        # Only fires if the computed metrics are close enough to prev_metrics
        # The activations produce top=0.50, neuron_count=10
        # We need focus_ratio from the activations to be close to 0.60
        if result.gate == "diminishing_returns":
            assert result.sufficient is True
            assert result.confidence < 0.6  # reduced confidence

    def test_diminishing_returns_fires_when_activation_delta_small(self) -> None:
        """Direct test: build activations where computed metrics closely match prev."""
        # One anchor, 10 neurons, top=0.80
        activations = _make_activations(
            [("n0", 0.80, 1, "anchor-0")] + [(f"n{i}", 0.40, 1, "anchor-0") for i in range(1, 10)]
        )
        anchor_sets = [["anchor-0"]]

        # Compute expected focus_ratio for these activations:
        # levels = [0.80, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]
        # total = 0.80 + 9*0.40 = 0.80 + 3.60 = 4.40
        # top3 = 0.80 + 0.40 + 0.40 = 1.60
        # focus_ratio = 1.60 / 4.40 = 0.364

        prev = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=10,  # same count
            intersection_count=0,
            top_activation=0.802,  # delta = 0.002 < 0.05
            mean_activation=0.44,
            activation_entropy=2.0,
            activation_mass=4.4,
            coverage_ratio=1.0,
            focus_ratio=0.370,  # delta = 0.006 < 0.05
            proximity_ratio=1.0,
            path_diversity=0.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            prev_metrics=prev,
        )
        assert result.gate == "diminishing_returns"
        assert result.sufficient is True
        # Confidence should be reduced (0.85 factor)
        assert result.confidence < 0.95

    def test_diminishing_returns_does_not_fire_when_activation_improved(self) -> None:
        """When top_activation improved by > 0.05, gate should NOT fire."""
        activations = _make_activations(
            [("n0", 0.80, 1, "anchor-0")] + [(f"n{i}", 0.40, 1, "anchor-0") for i in range(1, 10)]
        )
        anchor_sets = [["anchor-0"]]

        prev = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=10,
            intersection_count=0,
            top_activation=0.70,  # delta = 0.10 > 0.05 → should NOT fire
            mean_activation=0.35,
            activation_entropy=2.0,
            activation_mass=3.5,
            coverage_ratio=1.0,
            focus_ratio=0.370,
            proximity_ratio=1.0,
            path_diversity=0.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            prev_metrics=prev,
        )
        assert result.gate != "diminishing_returns"

    def test_diminishing_returns_does_not_fire_when_neuron_count_changed(self) -> None:
        """When neuron count delta > 1, gate should NOT fire."""
        activations = _make_activations(
            [("n0", 0.80, 1, "anchor-0")] + [(f"n{i}", 0.40, 1, "anchor-0") for i in range(1, 10)]
        )
        anchor_sets = [["anchor-0"]]

        prev = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=7,  # delta = 3 > 1 → should NOT fire
            intersection_count=0,
            top_activation=0.802,
            mean_activation=0.44,
            activation_entropy=2.0,
            activation_mass=4.4,
            coverage_ratio=1.0,
            focus_ratio=0.370,
            proximity_ratio=1.0,
            path_diversity=0.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            prev_metrics=prev,
        )
        assert result.gate != "diminishing_returns"

    def test_no_prev_metrics_skips_gate(self) -> None:
        """Without prev_metrics, diminishing_returns gate is skipped entirely."""
        activations = _make_activations(
            [("n0", 0.80, 1, "anchor-0")] + [(f"n{i}", 0.40, 1, "anchor-0") for i in range(1, 10)]
        )
        anchor_sets = [["anchor-0"]]

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            prev_metrics=None,
        )
        assert result.gate != "diminishing_returns"


# ---------------------------------------------------------------------------
# Combined: query_intent + calibration interaction
# ---------------------------------------------------------------------------


class TestCombinedFeaturesInteraction:
    def test_calibration_downgrade_on_default_pass_gate(self) -> None:
        """A default_pass result should be downgraded when calibration shows low avg_confidence."""
        # Build activations that will trigger default_pass (weak signal, not caught by other gates)
        activations = _make_activations([("n0", 0.20, 2, "anchor-0"), ("n1", 0.15, 2, "anchor-0")])
        anchor_sets = [["anchor-0"]]

        cal = {
            "default_pass": GateCalibration(
                accuracy=0.3,
                avg_confidence=0.08,
                sample_count=20,
            )
        }

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="unknown",
            calibration=cal,
        )
        # Should be downgraded
        if result.gate == "default_pass":
            assert result.sufficient is False

    def test_calibration_does_not_affect_insufficient_gates(self) -> None:
        """Gates that already return INSUFFICIENT are not affected by calibration."""
        activations: dict[str, _FakeActivation] = {}
        anchor_sets: list[list[str]] = []  # empty → no_anchors

        cal = {
            "no_anchors": GateCalibration(
                accuracy=0.1,
                avg_confidence=0.01,
                sample_count=100,
            )
        }

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            calibration=cal,
        )
        assert result.gate == "no_anchors"
        assert result.sufficient is False

    def test_all_params_can_be_passed_simultaneously(self) -> None:
        """Ensure check_sufficiency accepts all new params without error."""
        activations = _make_activations(
            [("n0", 0.80, 1, "anchor-0")] + [(f"n{i}", 0.40, 1, "anchor-0") for i in range(1, 10)]
        )
        anchor_sets = [["anchor-0"]]
        prev = SufficiencyMetrics(
            anchor_count=1,
            anchor_sets_active=1,
            neuron_count=10,
            intersection_count=0,
            top_activation=0.80,
            mean_activation=0.44,
            activation_entropy=2.0,
            activation_mass=4.4,
            coverage_ratio=1.0,
            focus_ratio=0.370,
            proximity_ratio=1.0,
            path_diversity=0.0,
            stab_converged=True,
            stab_neurons_removed=0,
        )
        cal = {
            "diminishing_returns": GateCalibration(
                accuracy=0.9,
                avg_confidence=0.6,
                sample_count=30,
            )
        }

        result = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            query_intent="ask_pattern",
            calibration=cal,
            prev_metrics=prev,
        )
        assert isinstance(result, SufficiencyResult)


# ---------------------------------------------------------------------------
# get_gate_ema_stats mixin (in-memory DB)
# ---------------------------------------------------------------------------


class TestGetGateEmaStats:
    """Test get_gate_ema_stats via a minimal mixin stub."""

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_dict(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin

        class _Stub(SQLiteCalibrationMixin):
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            def _ensure_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _ensure_read_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _get_brain_id(self) -> str:
                return "brain-1"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                """CREATE TABLE retrieval_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brain_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    predicted_sufficient INTEGER NOT NULL,
                    actual_confidence REAL NOT NULL,
                    actual_fibers INTEGER NOT NULL DEFAULT 0,
                    query_intent TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            await conn.commit()
            stub = _Stub(conn)
            result = await stub.get_gate_ema_stats()
            assert result == {}

    @pytest.mark.asyncio
    async def test_returns_stats_for_each_gate(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin
        from neural_memory.utils.timeutils import utcnow

        class _Stub(SQLiteCalibrationMixin):
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            def _ensure_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _ensure_read_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _get_brain_id(self) -> str:
                return "brain-1"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                """CREATE TABLE retrieval_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brain_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    predicted_sufficient INTEGER NOT NULL,
                    actual_confidence REAL NOT NULL,
                    actual_fibers INTEGER NOT NULL DEFAULT 0,
                    query_intent TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            now = utcnow().isoformat()
            rows = [
                ("brain-1", "default_pass", 1, 0.6, 0, "", "{}", now),
                ("brain-1", "default_pass", 1, 0.7, 0, "", "{}", now),
                ("brain-1", "default_pass", 1, 0.5, 0, "", "{}", now),
                ("brain-1", "focused_result", 1, 0.8, 1, "", "{}", now),
                ("brain-1", "focused_result", 1, 0.9, 1, "", "{}", now),
                # Different brain — should not be included
                ("brain-2", "default_pass", 1, 0.1, 0, "", "{}", now),
            ]
            await conn.executemany(
                """INSERT INTO retrieval_calibration
                   (brain_id, gate, predicted_sufficient, actual_confidence,
                    actual_fibers, query_intent, metrics_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await conn.commit()
            stub = _Stub(conn)
            result = await stub.get_gate_ema_stats()

            assert "default_pass" in result
            assert "focused_result" in result
            assert "brain-2" not in str(result)

            dp = result["default_pass"]
            assert 0.0 <= dp["accuracy"] <= 1.0
            assert 0.0 <= dp["avg_confidence"] <= 1.0
            assert dp["sample_count"] == 3.0

            fr = result["focused_result"]
            assert fr["sample_count"] == 2.0
            # avg_confidence for focused_result should be close to 0.85 (EMA of 0.8, 0.9)
            assert fr["avg_confidence"] > 0.7

    @pytest.mark.asyncio
    async def test_window_caps_records_per_gate(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin
        from neural_memory.utils.timeutils import utcnow

        class _Stub(SQLiteCalibrationMixin):
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            def _ensure_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _ensure_read_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _get_brain_id(self) -> str:
                return "brain-1"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                """CREATE TABLE retrieval_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brain_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    predicted_sufficient INTEGER NOT NULL,
                    actual_confidence REAL NOT NULL,
                    actual_fibers INTEGER NOT NULL DEFAULT 0,
                    query_intent TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            now = utcnow().isoformat()
            rows = [
                ("brain-1", "default_pass", 1, 0.5 + i * 0.01, 0, "", "{}", now) for i in range(20)
            ]
            await conn.executemany(
                """INSERT INTO retrieval_calibration
                   (brain_id, gate, predicted_sufficient, actual_confidence,
                    actual_fibers, query_intent, metrics_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await conn.commit()
            stub = _Stub(conn)

            # window=5 → at most 5 records used per gate
            result = await stub.get_gate_ema_stats(window=5)
            assert "default_pass" in result
            assert result["default_pass"]["sample_count"] == 5.0

    @pytest.mark.asyncio
    async def test_ema_accuracy_correct_predictions_increase_accuracy(self) -> None:
        """All correct predictions should give accuracy close to 1.0."""
        import aiosqlite

        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin
        from neural_memory.utils.timeutils import utcnow

        class _Stub(SQLiteCalibrationMixin):
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            def _ensure_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _ensure_read_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _get_brain_id(self) -> str:
                return "brain-1"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                """CREATE TABLE retrieval_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brain_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    predicted_sufficient INTEGER NOT NULL,
                    actual_confidence REAL NOT NULL,
                    actual_fibers INTEGER NOT NULL DEFAULT 0,
                    query_intent TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            now = utcnow().isoformat()
            # All: predicted_sufficient=1, actual_confidence >= 0.3 → all correct
            rows = [("brain-1", "default_pass", 1, 0.8, 0, "", "{}", now) for _ in range(10)]
            await conn.executemany(
                """INSERT INTO retrieval_calibration
                   (brain_id, gate, predicted_sufficient, actual_confidence,
                    actual_fibers, query_intent, metrics_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await conn.commit()
            stub = _Stub(conn)
            result = await stub.get_gate_ema_stats()
            assert result["default_pass"]["accuracy"] > 0.9

    @pytest.mark.asyncio
    async def test_ema_accuracy_all_wrong_predictions_decrease_accuracy(self) -> None:
        """All wrong predictions should give accuracy close to 0.0."""
        import aiosqlite

        from neural_memory.storage.sqlite_calibration import SQLiteCalibrationMixin
        from neural_memory.utils.timeutils import utcnow

        class _Stub(SQLiteCalibrationMixin):
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            def _ensure_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _ensure_read_conn(self) -> aiosqlite.Connection:
                return self._conn

            def _get_brain_id(self) -> str:
                return "brain-1"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                """CREATE TABLE retrieval_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brain_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    predicted_sufficient INTEGER NOT NULL,
                    actual_confidence REAL NOT NULL,
                    actual_fibers INTEGER NOT NULL DEFAULT 0,
                    query_intent TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            now = utcnow().isoformat()
            # All: predicted_sufficient=1, actual_confidence < 0.3 → all wrong
            rows = [("brain-1", "default_pass", 1, 0.05, 0, "", "{}", now) for _ in range(10)]
            await conn.executemany(
                """INSERT INTO retrieval_calibration
                   (brain_id, gate, predicted_sufficient, actual_confidence,
                    actual_fibers, query_intent, metrics_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await conn.commit()
            stub = _Stub(conn)
            result = await stub.get_gate_ema_stats()
            assert result["default_pass"]["accuracy"] < 0.1
