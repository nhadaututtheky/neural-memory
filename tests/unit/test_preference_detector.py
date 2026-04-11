"""Tests for preference signal detection and query classification."""

from __future__ import annotations

from neural_memory.engine.preference_detector import (
    PreferenceSignal,
    detect_preference_signals,
    extract_preference_domain,
    is_preference_query,
)

# ---------------------------------------------------------------------------
# detect_preference_signals — user role
# ---------------------------------------------------------------------------


class TestDetectPreferenceSignalsUser:
    """Test preference signal detection for user messages."""

    def test_explicit_preference(self) -> None:
        result = detect_preference_signals("I prefer using Premiere Pro for video editing", "user")
        assert result is not None
        assert result.confidence > 0.0
        assert result.pattern_matches >= 1
        assert "premiere" in result.domain_keywords or "editing" in result.domain_keywords

    def test_favorite_pattern(self) -> None:
        result = detect_preference_signals(
            "My favorite IDE is VS Code with Vim keybindings", "user"
        )
        assert result is not None
        assert result.pattern_matches >= 1

    def test_habitual_usage(self) -> None:
        result = detect_preference_signals("I've been using Python for 5 years", "user")
        assert result is not None

    def test_usually_pattern(self) -> None:
        result = detect_preference_signals("I usually use Docker for deployment", "user")
        assert result is not None

    def test_switched_to(self) -> None:
        result = detect_preference_signals("I switched to Neovim from VS Code last month", "user")
        assert result is not None

    def test_no_preference_neutral(self) -> None:
        result = detect_preference_signals("What is the capital of France?", "user")
        assert result is None

    def test_no_preference_question(self) -> None:
        result = detect_preference_signals("How do I install numpy?", "user")
        assert result is None

    def test_empty_content(self) -> None:
        result = detect_preference_signals("", "user")
        assert result is None

    def test_short_content(self) -> None:
        result = detect_preference_signals("hi", "user")
        assert result is None

    def test_multiple_signals_higher_confidence(self) -> None:
        content = "I prefer Python and my favorite framework is Django. I usually use it for web development."
        result = detect_preference_signals(content, "user")
        assert result is not None
        assert result.pattern_matches >= 2
        assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# detect_preference_signals — assistant role
# ---------------------------------------------------------------------------


class TestDetectPreferenceSignalsAssistant:
    """Test preference signal detection for assistant messages."""

    def test_based_on_preference(self) -> None:
        result = detect_preference_signals(
            "Based on your preference for minimalist tools, I'd suggest Vim",
            "assistant",
        )
        assert result is not None
        assert result.pattern_matches >= 1

    def test_since_you_like(self) -> None:
        result = detect_preference_signals(
            "Since you like Python, you might enjoy FastAPI for APIs",
            "assistant",
        )
        assert result is not None

    def test_no_preference_generic(self) -> None:
        result = detect_preference_signals(
            "Here are some popular frameworks: Django, Flask, FastAPI",
            "assistant",
        )
        assert result is None


# ---------------------------------------------------------------------------
# is_preference_query
# ---------------------------------------------------------------------------


class TestIsPreferenceQuery:
    """Test preference query detection."""

    def test_recommend(self) -> None:
        assert is_preference_query("Can you recommend a good video editor?") is True

    def test_suggest(self) -> None:
        assert is_preference_query("Suggest some Python libraries for data science") is True

    def test_what_should_i_use(self) -> None:
        assert is_preference_query("What should I use for web development?") is True

    def test_what_do_i_like(self) -> None:
        assert is_preference_query("What do I like for video editing?") is True

    def test_any_recommendations(self) -> None:
        assert is_preference_query("Any recommendations for a good IDE?") is True

    def test_best_option(self) -> None:
        assert is_preference_query("What is the best option for deployment?") is True

    def test_my_preference(self) -> None:
        assert is_preference_query("What is my preference for text editors?") is True

    def test_not_preference_factual(self) -> None:
        assert is_preference_query("What is the capital of France?") is False

    def test_not_preference_how(self) -> None:
        assert is_preference_query("How do I install Docker?") is False

    def test_empty(self) -> None:
        assert is_preference_query("") is False

    def test_short(self) -> None:
        assert is_preference_query("hi") is False

    def test_what_tools_for(self) -> None:
        assert is_preference_query("What tools for machine learning should I try?") is True


# ---------------------------------------------------------------------------
# extract_preference_domain
# ---------------------------------------------------------------------------


class TestExtractPreferenceDomain:
    """Test domain keyword extraction."""

    def test_extracts_keywords(self) -> None:
        keywords = extract_preference_domain(
            "I prefer using Premiere Pro for video editing and color grading"
        )
        assert len(keywords) > 0
        assert len(keywords) <= 5
        # Should include domain terms, not stopwords
        assert "the" not in keywords
        assert "for" not in keywords

    def test_empty_content(self) -> None:
        assert extract_preference_domain("") == []

    def test_only_stopwords(self) -> None:
        assert extract_preference_domain("I am the for with on") == []

    def test_frequency_ordering(self) -> None:
        keywords = extract_preference_domain(
            "Python is great. Python works well. Python handles data. Java is okay."
        )
        assert keywords[0] == "python"

    def test_max_five(self) -> None:
        keywords = extract_preference_domain(
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        )
        assert len(keywords) <= 5


# ---------------------------------------------------------------------------
# PreferenceSignal dataclass
# ---------------------------------------------------------------------------


class TestPreferenceSignalModel:
    """Test PreferenceSignal is frozen and has expected fields."""

    def test_frozen(self) -> None:
        signal = PreferenceSignal(confidence=0.5, domain_keywords=("python",), pattern_matches=1)
        try:
            signal.confidence = 0.9  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "PreferenceSignal should be frozen"

    def test_fields(self) -> None:
        signal = PreferenceSignal(
            confidence=0.8, domain_keywords=("django", "python"), pattern_matches=2
        )
        assert signal.confidence == 0.8
        assert signal.domain_keywords == ("django", "python")
        assert signal.pattern_matches == 2
