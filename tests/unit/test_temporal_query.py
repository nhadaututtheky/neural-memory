"""Tests for temporal query detection and date range extraction."""

from __future__ import annotations

from datetime import datetime

from neural_memory.engine.temporal_query import (
    TemporalSignal,
    detect_temporal_query,
    extract_date_range,
    extract_event_anchors,
)

# ---------------------------------------------------------------------------
# detect_temporal_query — duration patterns
# ---------------------------------------------------------------------------


class TestDetectTemporalDuration:
    """Test duration-based temporal query detection."""

    def test_days_ago(self) -> None:
        result = detect_temporal_query("I mentioned the Python project 10 days ago")
        assert result is not None
        assert result.signal_type == "duration"
        assert result.duration_unit == "days"
        assert result.duration_value == 10

    def test_specific_days_ago(self) -> None:
        result = detect_temporal_query("What did I do 5 days ago?")
        assert result is not None
        assert result.duration_value == 5
        assert result.duration_unit == "days"

    def test_weeks_ago(self) -> None:
        result = detect_temporal_query("3 weeks ago I mentioned a restaurant, which one?")
        assert result is not None
        assert result.duration_value == 3
        assert result.duration_unit == "weeks"

    def test_months_ago(self) -> None:
        result = detect_temporal_query("What project was I working on 2 months ago?")
        assert result is not None
        assert result.duration_value == 2
        assert result.duration_unit == "months"

    def test_few_days_ago(self) -> None:
        result = detect_temporal_query("A few days ago I asked about something")
        assert result is not None
        assert result.duration_value == 3
        assert result.duration_unit == "days"

    def test_last_week(self) -> None:
        result = detect_temporal_query("What did we discuss last week?")
        assert result is not None
        assert result.duration_value == 1
        assert result.duration_unit == "weeks"

    def test_last_month(self) -> None:
        result = detect_temporal_query("Last month I asked about Docker")
        assert result is not None
        assert result.duration_value == 1
        assert result.duration_unit == "months"

    def test_recently(self) -> None:
        result = detect_temporal_query("I recently mentioned a book, which one?")
        assert result is not None
        assert result.duration_unit == "days"
        assert result.duration_value == 7

    def test_confidence_with_value(self) -> None:
        result = detect_temporal_query("5 days ago I mentioned something")
        assert result is not None
        assert result.confidence >= 0.8


# ---------------------------------------------------------------------------
# detect_temporal_query — ordering patterns
# ---------------------------------------------------------------------------


class TestDetectTemporalOrdering:
    """Test ordering-based temporal query detection."""

    def test_what_order(self) -> None:
        result = detect_temporal_query("In what order did I discuss Python, Java, and Rust?")
        assert result is not None
        assert result.signal_type == "ordering"

    def test_which_came_first(self) -> None:
        result = detect_temporal_query("Which came first, the Docker discussion or the K8s one?")
        assert result is not None
        assert result.signal_type == "ordering"

    def test_chronological(self) -> None:
        result = detect_temporal_query("Can you list my topics in chronological order?")
        assert result is not None
        assert result.signal_type == "ordering"

    def test_timeline(self) -> None:
        result = detect_temporal_query("Give me a timeline of my Python learning")
        assert result is not None
        assert result.signal_type == "ordering"


# ---------------------------------------------------------------------------
# detect_temporal_query — when patterns
# ---------------------------------------------------------------------------


class TestDetectTemporalWhen:
    """Test when-based temporal query detection."""

    def test_when_did(self) -> None:
        result = detect_temporal_query("When did I first mention React?")
        assert result is not None
        assert result.signal_type == "when"

    def test_how_long_ago(self) -> None:
        result = detect_temporal_query("How long ago did I start learning Rust?")
        assert result is not None
        assert result.signal_type == "when"

    def test_last_time(self) -> None:
        result = detect_temporal_query("Last time I asked about databases, what did you say?")
        assert result is not None
        assert result.signal_type == "when"

    def test_how_many_days_since(self) -> None:
        result = detect_temporal_query("How many days since I mentioned the project?")
        assert result is not None


# ---------------------------------------------------------------------------
# detect_temporal_query — negative cases
# ---------------------------------------------------------------------------


class TestDetectTemporalNegative:
    """Test that non-temporal queries return None."""

    def test_plain_question(self) -> None:
        assert detect_temporal_query("What is Python?") is None

    def test_recommendation(self) -> None:
        assert detect_temporal_query("Can you recommend a good IDE?") is None

    def test_how_to(self) -> None:
        assert detect_temporal_query("How do I install Docker?") is None

    def test_empty(self) -> None:
        assert detect_temporal_query("") is None

    def test_short(self) -> None:
        assert detect_temporal_query("hi") is None


# ---------------------------------------------------------------------------
# extract_date_range
# ---------------------------------------------------------------------------


class TestExtractDateRange:
    """Test date range extraction from temporal signals."""

    def test_days_range(self) -> None:
        signal = TemporalSignal(
            signal_type="duration",
            confidence=0.8,
            duration_unit="days",
            duration_value=5,
        )
        ref = datetime(2023, 4, 10, 12, 0)
        result = extract_date_range(signal, ref)
        assert result is not None
        start, end = result
        # Target = April 5, window = ±2.5 days
        assert start < datetime(2023, 4, 5, 12, 0)
        assert end > datetime(2023, 4, 5, 12, 0)

    def test_weeks_range(self) -> None:
        signal = TemporalSignal(
            signal_type="duration",
            confidence=0.8,
            duration_unit="weeks",
            duration_value=2,
        )
        ref = datetime(2023, 4, 10)
        result = extract_date_range(signal, ref)
        assert result is not None
        start, end = result
        # Target = March 27, window = ±1 week
        assert start < datetime(2023, 3, 27)
        assert end > datetime(2023, 3, 27)

    def test_no_duration(self) -> None:
        signal = TemporalSignal(signal_type="ordering", confidence=0.7)
        result = extract_date_range(signal, datetime(2023, 4, 10))
        assert result is None

    def test_custom_tolerance(self) -> None:
        signal = TemporalSignal(
            signal_type="duration",
            confidence=0.8,
            duration_unit="days",
            duration_value=10,
        )
        ref = datetime(2023, 4, 10)
        narrow = extract_date_range(signal, ref, tolerance=0.2)
        wide = extract_date_range(signal, ref, tolerance=0.8)
        assert narrow is not None
        assert wide is not None
        # Wide window should be bigger
        narrow_span = narrow[1] - narrow[0]
        wide_span = wide[1] - wide[0]
        assert wide_span > narrow_span


# ---------------------------------------------------------------------------
# extract_event_anchors
# ---------------------------------------------------------------------------


class TestExtractEventAnchors:
    """Test event anchor extraction."""

    def test_extracts_content_terms(self) -> None:
        anchors = extract_event_anchors("When did I mention the Python Django project?")
        assert len(anchors) > 0
        assert "python" in anchors or "django" in anchors or "project" in anchors

    def test_filters_stopwords(self) -> None:
        anchors = extract_event_anchors("When did I first mention something?")
        assert "did" not in anchors
        assert "when" not in anchors
        assert "first" not in anchors

    def test_filters_temporal_words(self) -> None:
        anchors = extract_event_anchors("How many days ago did something happen?")
        assert "days" not in anchors
        assert "ago" not in anchors

    def test_empty(self) -> None:
        assert extract_event_anchors("") == []

    def test_max_five(self) -> None:
        anchors = extract_event_anchors(
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        )
        assert len(anchors) <= 5

    def test_preserves_order(self) -> None:
        anchors = extract_event_anchors("python django flask fastapi")
        assert anchors == ["python", "django", "flask", "fastapi"]


# ---------------------------------------------------------------------------
# TemporalSignal dataclass
# ---------------------------------------------------------------------------


class TestTemporalSignalModel:
    """Test TemporalSignal is frozen and has expected fields."""

    def test_frozen(self) -> None:
        signal = TemporalSignal(signal_type="duration", confidence=0.8)
        try:
            signal.confidence = 0.5  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "TemporalSignal should be frozen"

    def test_defaults(self) -> None:
        signal = TemporalSignal(signal_type="when", confidence=0.6)
        assert signal.duration_unit is None
        assert signal.duration_value is None
        assert signal.event_anchors == ()

    def test_event_anchors_populated(self) -> None:
        result = detect_temporal_query("3 weeks ago I mentioned Python and Docker")
        assert result is not None
        assert len(result.event_anchors) > 0
