"""Tests for query router module."""

from __future__ import annotations

import pytest

from neural_memory.extraction.parser import QueryIntent, QueryParser
from neural_memory.extraction.router import (
    QueryRouter,
    QueryType,
    RouteConfidence,
    RouteDecision,
    route_query,
)


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_query_types_exist(self) -> None:
        """Test all expected query types are defined."""
        expected = ["semantic", "temporal", "causal", "direct", "pattern", "comparative"]
        for t in expected:
            assert QueryType(t) is not None


class TestRouteConfidence:
    """Tests for RouteConfidence enum."""

    def test_confidence_ordering(self) -> None:
        """Test confidence levels are ordered correctly."""
        assert RouteConfidence.LOW < RouteConfidence.MEDIUM
        assert RouteConfidence.MEDIUM < RouteConfidence.HIGH
        assert RouteConfidence.HIGH < RouteConfidence.CERTAIN


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    def test_create_route_decision(self) -> None:
        """Test creating a RouteDecision."""
        decision = RouteDecision(
            primary=QueryType.TEMPORAL,
            confidence=RouteConfidence.HIGH,
        )
        assert decision.primary == QueryType.TEMPORAL
        assert decision.confidence == RouteConfidence.HIGH
        assert decision.secondary is None

    def test_should_fallback_low_confidence(self) -> None:
        """Test fallback suggestion for low confidence."""
        decision = RouteDecision(
            primary=QueryType.SEMANTIC,
            secondary=QueryType.DIRECT,
            confidence=RouteConfidence.LOW,
        )
        assert decision.should_fallback is True

    def test_should_not_fallback_high_confidence(self) -> None:
        """Test no fallback for high confidence."""
        decision = RouteDecision(
            primary=QueryType.TEMPORAL,
            secondary=QueryType.DIRECT,
            confidence=RouteConfidence.HIGH,
        )
        assert decision.should_fallback is False

    def test_should_not_fallback_no_secondary(self) -> None:
        """Test no fallback when no secondary route."""
        decision = RouteDecision(
            primary=QueryType.DIRECT,
            secondary=None,
            confidence=RouteConfidence.LOW,
        )
        assert decision.should_fallback is False


class TestQueryRouter:
    """Tests for QueryRouter."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        """Create a router instance."""
        return QueryRouter()

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Create a parser instance."""
        return QueryParser()

    def test_routes_temporal_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a temporal query."""
        stimulus = parser.parse("What did I do yesterday?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.TEMPORAL
        assert decision.time_weighted is True

    def test_routes_causal_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a causal query."""
        stimulus = parser.parse("Why did the build fail?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.CAUSAL

    def test_routes_direct_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a direct lookup query."""
        stimulus = parser.parse("What is Alice's email?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.DIRECT

    def test_routes_pattern_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a pattern query."""
        # Use a clearer pattern query without "What" which triggers other intents
        stimulus = parser.parse("I usually work from home on Mondays")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.PATTERN
        assert decision.time_weighted is True

    def test_routes_comparative_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a comparative query."""
        stimulus = parser.parse("Compare PostgreSQL versus MySQL")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.COMPARATIVE
        assert decision.use_embeddings is True

    def test_routes_semantic_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing a semantic/conceptual query."""
        stimulus = parser.parse("Tell me about authentication concepts")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.SEMANTIC
        assert decision.use_embeddings is True

    def test_temporal_query_with_time_hints(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test that time hints strongly boost temporal routing."""
        stimulus = parser.parse("What happened last week?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.TEMPORAL
        assert "has_time_hints" in decision.signals

    def test_vietnamese_temporal_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing Vietnamese temporal query."""
        stimulus = parser.parse("Hôm qua tôi làm gì?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.TEMPORAL

    def test_vietnamese_causal_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test routing Vietnamese causal query."""
        stimulus = parser.parse("Tại sao build bị lỗi?")
        decision = router.route(stimulus)

        assert decision.primary == QueryType.CAUSAL

    def test_suggested_depth_direct(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test direct queries suggest shallow depth."""
        stimulus = parser.parse("What is the API key?")
        decision = router.route(stimulus)

        # Direct queries suggest shallow depth (0 or 1)
        assert decision.suggested_depth <= 1

    def test_suggested_depth_causal(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test causal queries suggest deeper depth."""
        stimulus = parser.parse("Why did the deployment fail?")
        decision = router.route(stimulus)

        assert decision.suggested_depth >= 2

    def test_signals_captured(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test that routing signals are captured."""
        stimulus = parser.parse("What did I do yesterday morning?")
        decision = router.route(stimulus)

        assert len(decision.signals) > 0
        # Should have temporal signals
        temporal_signals = [s for s in decision.signals if "temporal" in s]
        assert len(temporal_signals) > 0

    def test_confidence_high_for_clear_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test high confidence for clear queries."""
        stimulus = parser.parse("Why did this happen? What was the reason?")
        decision = router.route(stimulus)

        assert decision.confidence >= RouteConfidence.HIGH

    def test_confidence_lower_for_ambiguous_query(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test lower confidence for ambiguous queries."""
        stimulus = parser.parse("stuff")
        decision = router.route(stimulus)

        assert decision.confidence <= RouteConfidence.MEDIUM


class TestRouteQueryFunction:
    """Tests for the route_query convenience function."""

    def test_route_query_function(self) -> None:
        """Test the convenience function."""
        parser = QueryParser()
        stimulus = parser.parse("What happened yesterday?")
        decision = route_query(stimulus)

        assert isinstance(decision, RouteDecision)
        assert decision.primary == QueryType.TEMPORAL


class TestIntentBasedRouting:
    """Tests for intent-based routing boosts."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        return QueryRouter()

    @pytest.fixture
    def parser(self) -> QueryParser:
        return QueryParser()

    def test_ask_when_boosts_temporal(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test ASK_WHEN intent boosts temporal routing."""
        stimulus = parser.parse("When did we have the meeting?")
        assert stimulus.intent == QueryIntent.ASK_WHEN
        decision = router.route(stimulus)
        assert decision.primary == QueryType.TEMPORAL

    def test_ask_why_boosts_causal(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test causal queries are routed correctly."""
        # The router detects causal queries through keyword signals
        stimulus = parser.parse("The reason it failed was the timeout")
        decision = router.route(stimulus)
        assert decision.primary == QueryType.CAUSAL

    def test_ask_who_boosts_direct(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test ASK_WHO intent boosts direct routing."""
        stimulus = parser.parse("Who was at the meeting?")
        assert stimulus.intent == QueryIntent.ASK_WHO
        decision = router.route(stimulus)
        assert decision.primary == QueryType.DIRECT

    def test_compare_boosts_comparative(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test COMPARE intent boosts comparative routing."""
        stimulus = parser.parse("Compare React and Vue")
        assert stimulus.intent == QueryIntent.COMPARE
        decision = router.route(stimulus)
        assert decision.primary == QueryType.COMPARATIVE

    def test_ask_pattern_boosts_pattern(self, router: QueryRouter, parser: QueryParser) -> None:
        """Test ASK_PATTERN intent boosts pattern routing."""
        # "usually" triggers ASK_PATTERN intent
        stimulus = parser.parse("I usually eat lunch at noon")
        assert stimulus.intent == QueryIntent.ASK_PATTERN
        decision = router.route(stimulus)
        assert decision.primary == QueryType.PATTERN
