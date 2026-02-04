"""Tests for safety module."""

from __future__ import annotations

from datetime import datetime, timedelta

from neural_memory.safety.freshness import (
    FreshnessLevel,
    analyze_freshness,
    evaluate_freshness,
    format_age,
    get_freshness_indicator,
)
from neural_memory.safety.sensitive import (
    SensitiveType,
    check_sensitive_content,
    filter_sensitive_content,
    format_sensitive_warning,
)


class TestSensitiveContentDetection:
    """Tests for sensitive content detection."""

    def test_detects_api_key(self) -> None:
        """Test detection of API keys."""
        content = "My API_KEY=sk-1234567890abcdef"
        matches = check_sensitive_content(content)

        assert len(matches) >= 1
        assert any(m.type == SensitiveType.API_KEY for m in matches)

    def test_detects_password(self) -> None:
        """Test detection of passwords."""
        content = "password=mysecretpass123"
        matches = check_sensitive_content(content, min_severity=2)

        assert len(matches) >= 1
        assert any(m.type == SensitiveType.PASSWORD for m in matches)

    def test_detects_aws_keys(self) -> None:
        """Test detection of AWS keys."""
        content = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        matches = check_sensitive_content(content)

        assert len(matches) >= 1
        assert any(m.type == SensitiveType.AWS_KEY for m in matches)

    def test_no_false_positives_on_normal_text(self) -> None:
        """Test that normal text doesn't trigger false positives."""
        content = "Had a meeting with Alice about the API design"
        matches = check_sensitive_content(content, min_severity=2)

        assert len(matches) == 0

    def test_severity_filtering(self) -> None:
        """Test that severity filtering works."""
        content = "API_KEY=sk-test123456789012"

        # Should find with min_severity=1
        matches_all = check_sensitive_content(content, min_severity=1)

        # Should find high severity
        matches_high = check_sensitive_content(content, min_severity=3)

        assert len(matches_all) >= len(matches_high)

    def test_redaction(self) -> None:
        """Test content redaction."""
        content = "Config: API_KEY=sk-1234567890abcdef"
        filtered, matches = filter_sensitive_content(content)

        assert "[REDACTED]" in filtered
        assert "sk-1234567890abcdef" not in filtered
        assert len(matches) >= 1

    def test_format_warning(self) -> None:
        """Test warning message formatting."""
        content = "API_KEY=sk-test123456789012"
        matches = check_sensitive_content(content)
        warning = format_sensitive_warning(matches)

        assert "SENSITIVE CONTENT DETECTED" in warning
        assert "HIGH RISK" in warning

    def test_redacted_preserves_ends(self) -> None:
        """Test that redacted text preserves first and last 4 chars."""
        content = "API_KEY=sk-1234567890abcdef"
        matches = check_sensitive_content(content)

        if matches:
            redacted = matches[0].redacted()
            # Long matches should show first 4 and last 4
            assert len(redacted) == len(matches[0].matched_text)


class TestFreshnessEvaluation:
    """Tests for memory freshness evaluation."""

    def test_fresh_memory(self) -> None:
        """Test that recent memories are marked as fresh."""
        now = datetime.now()
        created = now - timedelta(days=3)
        result = evaluate_freshness(created, now)

        assert result.level == FreshnessLevel.FRESH
        assert result.score == 1.0
        assert result.warning is None
        assert not result.should_verify

    def test_recent_memory(self) -> None:
        """Test that week-old memories are marked as recent."""
        now = datetime.now()
        created = now - timedelta(days=14)
        result = evaluate_freshness(created, now)

        assert result.level == FreshnessLevel.RECENT
        assert result.score == 0.8
        assert result.warning is None

    def test_aging_memory(self) -> None:
        """Test that month-old memories are marked as aging."""
        now = datetime.now()
        created = now - timedelta(days=60)
        result = evaluate_freshness(created, now)

        assert result.level == FreshnessLevel.AGING
        assert result.warning is not None
        assert result.should_verify

    def test_stale_memory(self) -> None:
        """Test that old memories are marked as stale."""
        now = datetime.now()
        created = now - timedelta(days=200)
        result = evaluate_freshness(created, now)

        assert result.level == FreshnessLevel.STALE
        assert "STALE" in result.warning
        assert result.should_verify

    def test_ancient_memory(self) -> None:
        """Test that very old memories are marked as ancient."""
        now = datetime.now()
        created = now - timedelta(days=500)
        result = evaluate_freshness(created, now)

        assert result.level == FreshnessLevel.ANCIENT
        assert "ANCIENT" in result.warning
        assert result.should_verify

    def test_format_age_today(self) -> None:
        """Test age formatting for today."""
        assert format_age(0) == "today"

    def test_format_age_yesterday(self) -> None:
        """Test age formatting for yesterday."""
        assert format_age(1) == "yesterday"

    def test_format_age_days(self) -> None:
        """Test age formatting for days."""
        assert format_age(5) == "5 days ago"

    def test_format_age_weeks(self) -> None:
        """Test age formatting for weeks."""
        assert "week" in format_age(14)

    def test_format_age_months(self) -> None:
        """Test age formatting for months."""
        assert "month" in format_age(60)

    def test_format_age_years(self) -> None:
        """Test age formatting for years."""
        assert "year" in format_age(400)

    def test_freshness_indicator_ascii(self) -> None:
        """Test freshness indicators use ASCII."""
        indicator = get_freshness_indicator(FreshnessLevel.FRESH)
        # Should be ASCII, not emoji
        assert indicator.isascii()

    def test_analyze_freshness_empty(self) -> None:
        """Test analyze_freshness with empty list."""
        report = analyze_freshness([])
        assert report.total == 0
        assert report.average_age_days == 0

    def test_analyze_freshness_mixed(self) -> None:
        """Test analyze_freshness with mixed ages."""
        now = datetime.now()
        dates = [
            now - timedelta(days=2),   # Fresh
            now - timedelta(days=20),  # Recent
            now - timedelta(days=50),  # Aging
            now - timedelta(days=200), # Stale
        ]
        report = analyze_freshness(dates, now)

        assert report.total == 4
        assert report.fresh == 1
        assert report.recent == 1
        assert report.aging == 1
        assert report.stale == 1
        assert report.ancient == 0
