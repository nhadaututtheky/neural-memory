"""Query router for intelligent query routing.

MemoCore-style query routing that determines the optimal retrieval strategy
based on query analysis. Routes queries to:
- Semantic search (RAG/embeddings) for conceptual queries
- Temporal traversal for time-based queries
- Causal traversal for why/how queries
- Direct lookup for exact recall
- Pattern matching for habit/frequency queries
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from neural_memory.extraction.parser import QueryIntent, Stimulus


class QueryType(StrEnum):
    """Types of queries for routing purposes."""

    SEMANTIC = "semantic"  # Conceptual, meaning-based: "What do I know about auth?"
    TEMPORAL = "temporal"  # Time-based: "What did I do yesterday?"
    CAUSAL = "causal"  # Cause/effect: "Why did the build fail?"
    DIRECT = "direct"  # Exact recall: "What's Alice's email?"
    PATTERN = "pattern"  # Habit/frequency: "What do I usually do on Mondays?"
    COMPARATIVE = "comparative"  # Comparison: "How is X different from Y?"


class RouteConfidence(IntEnum):
    """Confidence level in route selection."""

    LOW = 1  # Guessing, might be wrong
    MEDIUM = 2  # Reasonable guess
    HIGH = 3  # Strong signals
    CERTAIN = 4  # Very clear query type


@dataclass(frozen=True)
class RouteDecision:
    """Decision about how to route a query.

    Attributes:
        primary: Primary query type to use
        secondary: Optional fallback query type
        confidence: How confident we are in this routing
        signals: What signals led to this decision
        suggested_depth: Suggested traversal depth (0-3)
        use_embeddings: Whether to use vector search
        time_weighted: Whether to weight by recency
        metadata: Additional routing hints
    """

    primary: QueryType
    secondary: QueryType | None = None
    confidence: RouteConfidence = RouteConfidence.MEDIUM
    signals: tuple[str, ...] = ()
    suggested_depth: int = 1
    use_embeddings: bool = False
    time_weighted: bool = True
    metadata: dict[str, Any] | None = None

    @property
    def should_fallback(self) -> bool:
        """Whether to try secondary route if primary fails."""
        return self.secondary is not None and self.confidence <= RouteConfidence.MEDIUM


class QueryRouter:
    """Routes queries to optimal retrieval strategies.

    The router analyzes a Stimulus (parsed query) and determines:
    1. What type of query this is
    2. What retrieval strategy to use
    3. How deep to search
    4. Whether to use semantic search
    """

    # Keywords that strongly indicate query types
    SEMANTIC_SIGNALS = frozenset(
        [
            # English
            "about",
            "related",
            "concept",
            "idea",
            "understand",
            "explain",
            "knowledge",
            "information",
            "details",
            "overview",
            # Vietnamese
            "về",
            "liên quan",
            "khái niệm",
            "hiểu",
            "giải thích",
            "thông tin",
        ]
    )

    TEMPORAL_SIGNALS = frozenset(
        [
            # English
            "when",
            "yesterday",
            "today",
            "last week",
            "ago",
            "before",
            "after",
            "morning",
            "afternoon",
            "evening",
            "night",
            "recently",
            "earlier",
            # Vietnamese
            "khi nào",
            "hôm qua",
            "hôm nay",
            "tuần trước",
            "trước",
            "sau",
            "sáng",
            "chiều",
            "tối",
            "gần đây",
            "lúc",
        ]
    )

    CAUSAL_SIGNALS = frozenset(
        [
            # English
            "why",
            "because",
            "cause",
            "reason",
            "result",
            "led to",
            "caused",
            "effect",
            "consequence",
            "how come",
            # Vietnamese
            "tại sao",
            "vì sao",
            "lý do",
            "nguyên nhân",
            "kết quả",
            "dẫn đến",
        ]
    )

    DIRECT_SIGNALS = frozenset(
        [
            # English
            "what is",
            "what's",
            "exact",
            "specific",
            "precisely",
            "tell me the",
            "give me",
            "show me",
            "find",
            # Vietnamese
            "là gì",
            "chính xác",
            "cụ thể",
            "cho tôi",
            "tìm",
        ]
    )

    PATTERN_SIGNALS = frozenset(
        [
            # English
            "usually",
            "typically",
            "always",
            "often",
            "habit",
            "routine",
            "every",
            "pattern",
            "tend to",
            "normally",
            # Vietnamese
            "thường",
            "hay",
            "luôn",
            "thói quen",
            "mỗi",
            "xu hướng",
        ]
    )

    COMPARATIVE_SIGNALS = frozenset(
        [
            # English
            "compare",
            "versus",
            "vs",
            "difference",
            "different",
            "similar",
            "better",
            "worse",
            "same",
            "like",
            # Vietnamese
            "so sánh",
            "khác",
            "giống",
            "hơn",
            "như",
        ]
    )

    def route(self, stimulus: Stimulus) -> RouteDecision:
        """Determine the optimal route for a query.

        Args:
            stimulus: Parsed query stimulus

        Returns:
            RouteDecision with routing strategy
        """
        query_lower = stimulus.raw_query.lower()
        signals: list[str] = []

        # Pre-build word set + bigram set for O(1) lookups
        words = query_lower.split()
        word_set = frozenset(words)
        bigrams = frozenset(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
        all_tokens = word_set | bigrams

        # Score each query type via consolidated loop
        signal_map: dict[QueryType, tuple[frozenset[str], float]] = {
            QueryType.TEMPORAL: (self.TEMPORAL_SIGNALS, 2.0),
            QueryType.CAUSAL: (self.CAUSAL_SIGNALS, 2.0),
            QueryType.DIRECT: (self.DIRECT_SIGNALS, 1.5),
            QueryType.PATTERN: (self.PATTERN_SIGNALS, 2.0),
            QueryType.COMPARATIVE: (self.COMPARATIVE_SIGNALS, 2.0),
            QueryType.SEMANTIC: (self.SEMANTIC_SIGNALS, 1.0),
        }

        scores: dict[QueryType, float] = dict.fromkeys(QueryType, 0.0)
        for query_type, (signal_set, weight) in signal_map.items():
            for signal in signal_set:
                if signal in all_tokens or signal in query_lower:
                    scores[query_type] += weight
                    signals.append(f"{query_type.value}:{signal}")

        # Use parsed intent to boost scores
        intent_boosts = self._get_intent_boosts(stimulus.intent)
        for query_type, boost in intent_boosts.items():
            scores[query_type] += boost

        # Time hints strongly indicate temporal
        if stimulus.has_time_context:
            scores[QueryType.TEMPORAL] += 3.0
            signals.append("has_time_hints")

        # Entities without time often indicate direct lookup
        if stimulus.has_entities and not stimulus.has_time_context:
            scores[QueryType.DIRECT] += 1.0
            signals.append("entities_no_time")

        # Find primary and secondary types
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_types[0][0]
        primary_score = sorted_types[0][1]
        secondary = sorted_types[1][0] if sorted_types[1][1] > 0 else None

        # Determine confidence
        confidence = self._determine_confidence(primary_score, scores)

        # Determine depth based on query type
        suggested_depth = self._suggest_depth(primary, stimulus)

        # Determine if embeddings should be used
        use_embeddings = primary in (QueryType.SEMANTIC, QueryType.COMPARATIVE)

        # Time weighting for temporal and pattern queries
        time_weighted = primary in (QueryType.TEMPORAL, QueryType.PATTERN)

        return RouteDecision(
            primary=primary,
            secondary=secondary,
            confidence=confidence,
            signals=tuple(signals),
            suggested_depth=suggested_depth,
            use_embeddings=use_embeddings,
            time_weighted=time_weighted,
        )

    def _get_intent_boosts(self, intent: QueryIntent) -> dict[QueryType, float]:
        """Get score boosts based on query intent."""
        boosts: dict[QueryType, float] = {}

        intent_mapping = {
            QueryIntent.ASK_WHEN: {QueryType.TEMPORAL: 2.0},
            QueryIntent.ASK_WHY: {QueryType.CAUSAL: 2.0},
            QueryIntent.ASK_HOW: {QueryType.CAUSAL: 1.5, QueryType.SEMANTIC: 0.5},
            QueryIntent.ASK_WHAT: {QueryType.SEMANTIC: 1.0, QueryType.DIRECT: 1.0},
            QueryIntent.ASK_WHO: {QueryType.DIRECT: 1.5},
            QueryIntent.ASK_WHERE: {QueryType.DIRECT: 1.5, QueryType.TEMPORAL: 0.5},
            QueryIntent.ASK_PATTERN: {QueryType.PATTERN: 3.0},
            QueryIntent.ASK_FEELING: {QueryType.TEMPORAL: 1.0, QueryType.SEMANTIC: 1.0},
            QueryIntent.CONFIRM: {QueryType.DIRECT: 2.0},
            QueryIntent.COMPARE: {QueryType.COMPARATIVE: 3.0},
        }

        return intent_mapping.get(intent, boosts)

    def _determine_confidence(
        self, primary_score: float, all_scores: dict[QueryType, float]
    ) -> RouteConfidence:
        """Determine confidence in the routing decision."""
        if primary_score >= 5.0:
            return RouteConfidence.CERTAIN
        elif primary_score >= 3.0:
            return RouteConfidence.HIGH
        elif primary_score >= 1.5:
            return RouteConfidence.MEDIUM
        else:
            return RouteConfidence.LOW

    def _suggest_depth(self, query_type: QueryType, stimulus: Stimulus) -> int:
        """Suggest traversal depth based on query type."""
        depth_defaults = {
            QueryType.DIRECT: 0,  # Instant lookup
            QueryType.TEMPORAL: 1,  # Context level
            QueryType.SEMANTIC: 2,  # Habit level (broader search)
            QueryType.CAUSAL: 2,  # Need to traverse causes
            QueryType.PATTERN: 2,  # Cross-time patterns
            QueryType.COMPARATIVE: 2,  # Need multiple memories
        }

        base_depth = depth_defaults.get(query_type, 1)

        # Increase depth for complex queries
        if stimulus.anchor_count > 3:
            base_depth = min(3, base_depth + 1)

        return base_depth


def route_query(stimulus: Stimulus) -> RouteDecision:
    """Convenience function to route a query.

    Args:
        stimulus: Parsed query stimulus

    Returns:
        RouteDecision with routing strategy
    """
    router = QueryRouter()
    return router.route(stimulus)


def get_query_type_description(query_type: QueryType) -> str:
    """Get human-readable description of a query type."""
    descriptions = {
        QueryType.SEMANTIC: "Conceptual search - finding related ideas and knowledge",
        QueryType.TEMPORAL: "Time-based search - finding memories from specific times",
        QueryType.CAUSAL: "Causal search - understanding why/how things happened",
        QueryType.DIRECT: "Direct lookup - finding specific facts or details",
        QueryType.PATTERN: "Pattern search - finding habits and recurring themes",
        QueryType.COMPARATIVE: "Comparative search - comparing different memories",
    }
    return descriptions.get(query_type, "Unknown query type")
