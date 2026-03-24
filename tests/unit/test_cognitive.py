"""Tests for cognitive layer pure functions."""

import pytest

from neural_memory.core.memory_types import MemoryType
from neural_memory.core.neuron import NeuronType
from neural_memory.engine.cognitive import (
    compute_calibration,
    detect_auto_resolution,
    gap_priority,
    score_hypothesis,
    score_prediction,
    truncate_summary,
    update_confidence,
)


class TestUpdateConfidence:
    """Tests for Bayesian-inspired confidence update."""

    def test_evidence_for_increases_confidence(self):
        result = update_confidence(0.5, "for", weight=0.5)
        assert result > 0.5

    def test_evidence_against_decreases_confidence(self):
        result = update_confidence(0.5, "against", weight=0.5)
        assert result < 0.5

    def test_never_reaches_zero(self):
        conf = 0.05
        for _ in range(50):
            conf = update_confidence(conf, "against", weight=1.0)
        assert conf >= 0.01

    def test_never_reaches_one(self):
        conf = 0.95
        for _ in range(50):
            conf = update_confidence(conf, "for", weight=1.0)
        assert conf <= 0.99

    def test_surprise_factor_against_strong_belief(self):
        """Evidence against a strong belief should move more than evidence for it."""
        high_conf = 0.9
        shift_for = update_confidence(high_conf, "for", weight=0.5) - high_conf
        shift_against = high_conf - update_confidence(high_conf, "against", weight=0.5)
        assert shift_against > shift_for

    def test_surprise_factor_for_weak_belief(self):
        """Evidence for a weak belief should move more than evidence against it."""
        low_conf = 0.1
        shift_for = update_confidence(low_conf, "for", weight=0.5) - low_conf
        shift_against = low_conf - update_confidence(low_conf, "against", weight=0.5)
        assert shift_for > shift_against

    def test_diminishing_returns(self):
        """More existing evidence = smaller update per new evidence."""
        shift_few = update_confidence(0.5, "for", weight=0.5, for_count=0, against_count=0) - 0.5
        shift_many = update_confidence(0.5, "for", weight=0.5, for_count=10, against_count=5) - 0.5
        assert shift_few > shift_many

    def test_weight_affects_magnitude(self):
        """Higher weight evidence should move confidence more."""
        shift_low = update_confidence(0.5, "for", weight=0.2) - 0.5
        shift_high = update_confidence(0.5, "for", weight=0.9) - 0.5
        assert shift_high > shift_low

    def test_clamps_input_confidence(self):
        """Input outside [0.01, 0.99] is clamped."""
        result = update_confidence(0.0, "for", weight=0.5)
        assert result >= 0.01
        result = update_confidence(1.0, "against", weight=0.5)
        assert result <= 0.99

    def test_clamps_weight(self):
        """Weight outside [0.1, 1.0] is clamped."""
        # Should not crash with extreme weights
        update_confidence(0.5, "for", weight=0.0)
        update_confidence(0.5, "for", weight=5.0)

    def test_invalid_evidence_type_raises(self):
        """Invalid evidence_type should raise ValueError."""
        with pytest.raises(ValueError, match="evidence_type must be"):
            update_confidence(0.5, "supports")  # type: ignore[arg-type]

    def test_immutability(self):
        """Function should not mutate any inputs (all primitives, but verify return)."""
        result = update_confidence(0.5, "for")
        assert isinstance(result, float)


class TestDetectAutoResolution:
    """Tests for hypothesis auto-resolution."""

    def test_confirmed_when_high_confidence_enough_evidence(self):
        assert detect_auto_resolution(0.92, for_count=5, against_count=1) == "confirmed"

    def test_refuted_when_low_confidence_enough_evidence(self):
        assert detect_auto_resolution(0.08, for_count=0, against_count=4) == "refuted"

    def test_active_when_moderate_confidence(self):
        assert detect_auto_resolution(0.5, for_count=3, against_count=3) is None

    def test_not_confirmed_with_too_few_evidence(self):
        assert detect_auto_resolution(0.95, for_count=2, against_count=0) is None

    def test_not_refuted_with_too_few_evidence(self):
        assert detect_auto_resolution(0.05, for_count=0, against_count=2) is None

    def test_boundary_confirmed(self):
        assert detect_auto_resolution(0.9, for_count=3, against_count=0) == "confirmed"

    def test_boundary_refuted(self):
        assert detect_auto_resolution(0.1, for_count=0, against_count=3) == "refuted"


class TestComputeCalibration:
    """Tests for prediction calibration score."""

    def test_no_data_returns_neutral(self):
        assert compute_calibration(0, 0) == 0.5

    def test_perfect_calibration(self):
        assert compute_calibration(10, 10) == 1.0

    def test_zero_accuracy(self):
        assert compute_calibration(0, 10) == 0.0

    def test_partial_accuracy(self):
        assert compute_calibration(7, 10) == pytest.approx(0.7)


class TestScoreHypothesis:
    """Tests for hot index hypothesis ranking."""

    def test_moderate_confidence_scores_higher_than_extreme(self):
        mid = score_hypothesis(0.5, evidence_count=3, age_days=5)
        extreme = score_hypothesis(0.95, evidence_count=3, age_days=5)
        assert mid > extreme

    def test_more_evidence_scores_higher(self):
        few = score_hypothesis(0.5, evidence_count=1, age_days=5)
        many = score_hypothesis(0.5, evidence_count=5, age_days=5)
        assert many > few

    def test_recent_scores_higher(self):
        recent = score_hypothesis(0.5, evidence_count=3, age_days=1)
        old = score_hypothesis(0.5, evidence_count=3, age_days=60)
        assert recent > old

    def test_returns_positive(self):
        assert score_hypothesis(0.5, 0, 0) > 0


class TestScorePrediction:
    """Tests for hot index prediction ranking."""

    def test_overdue_is_most_urgent(self):
        assert score_prediction(-1) == 10.0

    def test_imminent_more_urgent_than_distant(self):
        imminent = score_prediction(1)
        distant = score_prediction(30)
        assert imminent > distant

    def test_returns_positive(self):
        assert score_prediction(100) > 0


class TestGapPriority:
    """Tests for knowledge gap priority."""

    def test_known_sources(self):
        assert gap_priority("contradicting_evidence") == 0.8
        assert gap_priority("recall_miss") == 0.5
        assert gap_priority("stale_schema") == 0.4

    def test_unknown_source_defaults(self):
        assert gap_priority("unknown_source") == 0.5


class TestTruncateSummary:
    """Tests for hot index summary truncation."""

    def test_short_content_unchanged(self):
        assert truncate_summary("hello world") == "hello world"

    def test_long_content_truncated(self):
        long_text = "a" * 200
        result = truncate_summary(long_text, max_length=120)
        assert len(result) == 120
        assert result.endswith("\u2026")

    def test_newlines_removed(self):
        result = truncate_summary("line1\nline2\nline3")
        assert "\n" not in result

    def test_whitespace_stripped(self):
        result = truncate_summary("  hello  ")
        assert result == "hello"

    def test_exact_length_not_truncated(self):
        text = "a" * 120
        assert truncate_summary(text, max_length=120) == text


class TestEnumSync:
    """Ensure NeuronType and MemoryType cognitive values stay in sync."""

    def test_hypothesis_values_match(self):
        assert NeuronType.HYPOTHESIS.value == MemoryType.HYPOTHESIS.value

    def test_prediction_values_match(self):
        assert NeuronType.PREDICTION.value == MemoryType.PREDICTION.value

    def test_schema_values_match(self):
        assert NeuronType.SCHEMA.value == MemoryType.SCHEMA.value
