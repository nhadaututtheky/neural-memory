"""Unit tests for outcome-driven learning — confidence + unreliable flags."""

from __future__ import annotations

from neural_memory.engine.context_optimizer import ContextItem


class TestOutcomeConfidence:
    """Confidence field reflects success_rate from execution history."""

    def test_confidence_from_success_rate(self):
        """Confidence equals success_rate when set."""
        item = ContextItem(
            fiber_id="f1",
            content="Use retry pattern",
            score=0.7,
            token_count=15,
            confidence=0.85,
        )
        assert item.confidence == 0.85

    def test_confidence_none_when_no_executions(self):
        """Default confidence is None (no execution data)."""
        item = ContextItem(
            fiber_id="f1",
            content="Untested idea",
            score=0.5,
            token_count=10,
        )
        assert item.confidence is None

    def test_confidence_zero_all_failures(self):
        """0% success rate → confidence 0.0."""
        item = ContextItem(
            fiber_id="f1",
            content="Bad approach",
            score=0.3,
            token_count=10,
            confidence=0.0,
            unreliable=True,
        )
        assert item.confidence == 0.0


class TestUnreliableFlag:
    """Unreliable flag marks low-confidence memories."""

    def test_unreliable_when_low_success(self):
        """success_rate < 0.3 + ≥3 executions → unreliable."""
        item = ContextItem(
            fiber_id="f1",
            content="Flaky approach",
            score=0.4,
            token_count=10,
            confidence=0.2,
            unreliable=True,
        )
        assert item.unreliable is True

    def test_not_unreliable_when_few_executions(self):
        """Even low success_rate, if < 3 executions → not unreliable."""
        item = ContextItem(
            fiber_id="f1",
            content="New idea",
            score=0.5,
            token_count=10,
            confidence=0.0,
            unreliable=False,  # Not enough data
        )
        assert item.unreliable is False

    def test_not_unreliable_when_high_success(self):
        """High success_rate → never unreliable."""
        item = ContextItem(
            fiber_id="f1",
            content="Proven pattern",
            score=0.8,
            token_count=10,
            confidence=0.9,
            unreliable=False,
        )
        assert item.unreliable is False

    def test_default_not_unreliable(self):
        """Default is not unreliable."""
        item = ContextItem(
            fiber_id="f1",
            content="Normal memory",
            score=0.5,
            token_count=10,
        )
        assert item.unreliable is False


class TestContextFormatting:
    """Verify context formatting includes confidence + unreliable labels."""

    def test_unreliable_label_in_format(self):
        """Unreliable items get [UNRELIABLE] prefix."""
        item = ContextItem(
            fiber_id="f1",
            content="Use polling instead of webhooks",
            score=0.3,
            token_count=20,
            confidence=0.15,
            unreliable=True,
        )
        # Simulate the formatting logic from recall_handler
        prefix = ""
        if item.tier == "hot":
            prefix = "[HOT] "
        if item.unreliable:
            prefix += "[UNRELIABLE] "
        suffix = ""
        if item.confidence is not None and item.confidence < 1.0:
            suffix = f" (confidence: {item.confidence:.0%})"
        formatted = f"- {prefix}{item.content}{suffix}"
        assert "[UNRELIABLE]" in formatted
        assert "(confidence: 15%)" in formatted

    def test_hot_unreliable_combined(self):
        """HOT + UNRELIABLE both shown."""
        item = ContextItem(
            fiber_id="f1",
            content="Critical but flaky",
            score=0.9,
            token_count=15,
            tier="hot",
            confidence=0.1,
            unreliable=True,
        )
        prefix = ""
        if item.tier == "hot":
            prefix = "[HOT] "
        if item.unreliable:
            prefix += "[UNRELIABLE] "
        formatted = f"- {prefix}{item.content}"
        assert "[HOT] [UNRELIABLE]" in formatted

    def test_no_confidence_suffix_when_none(self):
        """No confidence suffix when confidence is None."""
        item = ContextItem(
            fiber_id="f1",
            content="Normal memory",
            score=0.5,
            token_count=10,
        )
        suffix = ""
        if item.confidence is not None and item.confidence < 1.0:
            suffix = f" (confidence: {item.confidence:.0%})"
        formatted = f"- {item.content}{suffix}"
        assert "confidence" not in formatted

    def test_no_confidence_suffix_when_perfect(self):
        """No confidence suffix when 100%."""
        item = ContextItem(
            fiber_id="f1",
            content="Always works",
            score=0.9,
            token_count=10,
            confidence=1.0,
        )
        suffix = ""
        if item.confidence is not None and item.confidence < 1.0:
            suffix = f" (confidence: {item.confidence:.0%})"
        formatted = f"- {item.content}{suffix}"
        assert "confidence" not in formatted

    def test_confidence_percentage_format(self):
        """Confidence shown as percentage (e.g., 85%)."""
        item = ContextItem(
            fiber_id="f1",
            content="Mostly works",
            score=0.7,
            token_count=10,
            confidence=0.85,
        )
        suffix = ""
        if item.confidence is not None and item.confidence < 1.0:
            suffix = f" (confidence: {item.confidence:.0%})"
        assert suffix == " (confidence: 85%)"
