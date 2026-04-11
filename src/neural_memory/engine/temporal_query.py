"""Temporal query detection for retrieval improvement.

Detects temporal signals in queries (e.g. "how many days ago", "what order",
"before/after") and extracts date ranges or event anchors for pre-filtering
and scoring. All heuristic-based — zero LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Temporal signal patterns
# ---------------------------------------------------------------------------

_TEMPORAL_DURATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "N days/weeks/months ago"
    (re.compile(r"(\d+)\s*days?\s*ago", re.IGNORECASE), "days"),
    (re.compile(r"(\d+)\s*weeks?\s*ago", re.IGNORECASE), "weeks"),
    (re.compile(r"(\d+)\s*months?\s*ago", re.IGNORECASE), "months"),
    (re.compile(r"(\d+)\s*years?\s*ago", re.IGNORECASE), "years"),
    # "a few days/weeks ago"
    (re.compile(r"a few days?\s*ago", re.IGNORECASE), "few_days"),
    (re.compile(r"a few weeks?\s*ago", re.IGNORECASE), "few_weeks"),
    # "last week/month"
    (re.compile(r"last\s+week", re.IGNORECASE), "last_week"),
    (re.compile(r"last\s+month", re.IGNORECASE), "last_month"),
    # "recently"
    (re.compile(r"\brecently\b", re.IGNORECASE), "recently"),
]

_TEMPORAL_ORDERING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhat order\b",
        r"\bwhich (?:came|happened|was) first\b",
        r"\bchronological\b",
        r"\bbefore or after\b",
        r"\bbefore\b.*\bor\b.*\bafter\b",
        r"\bsequence of\b",
        r"\btimeline\b",
    ]
]

_TEMPORAL_WHEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhen did\b",
        r"\bhow (?:long|many (?:days?|weeks?|months?)) (?:ago|since|between)\b",
        r"\blast time (?:i|you|we)\b",
        r"\bfirst time (?:i|you|we)\b",
        r"\bhow (?:recently|long ago)\b",
    ]
]

# Event anchor extraction: content nouns that likely refer to specific events
_EVENT_STOPWORDS: frozenset[str] = frozenset(
    [
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "our",
        "they",
        "them",
        "the",
        "a",
        "an",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "and",
        "but",
        "or",
        "not",
        "no",
        "so",
        "if",
        "about",
        "up",
        "how",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "some",
        "any",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "many",
        "much",
        "more",
        "most",
        "ago",
        "days",
        "weeks",
        "months",
        "years",
        "time",
        "first",
        "last",
        "recently",
        "order",
        "long",
        "did",
        "happened",
        "came",
        "tell",
        "remind",
        "remember",
    ]
)

_WORD_RE = re.compile(r"\b[a-z][a-z0-9-]{2,}\b")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalSignal:
    """Detected temporal signal from a query."""

    signal_type: str  # "duration", "ordering", "when"
    confidence: float  # 0.0-1.0
    duration_unit: str | None = None  # "days", "weeks", etc. for duration type
    duration_value: int | None = None  # numeric value for duration type
    event_anchors: tuple[str, ...] = ()  # key event terms from query


def detect_temporal_query(query: str) -> TemporalSignal | None:
    """Detect temporal signal in a query.

    Args:
        query: The user's search query.

    Returns:
        TemporalSignal if temporal patterns found, None otherwise.
    """
    if not query or len(query) < 5:
        return None

    # Check duration patterns first (most specific)
    for pattern, unit in _TEMPORAL_DURATION_PATTERNS:
        match = pattern.search(query)
        if match:
            value = None
            if match.lastindex and match.lastindex >= 1:
                try:
                    value = int(match.group(1))
                except (ValueError, IndexError):
                    pass

            # Map fuzzy units to concrete values
            if unit == "few_days":
                unit, value = "days", 3
            elif unit == "few_weeks":
                unit, value = "weeks", 3
            elif unit == "last_week":
                unit, value = "weeks", 1
            elif unit == "last_month":
                unit, value = "months", 1
            elif unit == "recently":
                unit, value = "days", 7

            anchors = extract_event_anchors(query)
            return TemporalSignal(
                signal_type="duration",
                confidence=0.8 if value else 0.5,
                duration_unit=unit,
                duration_value=value,
                event_anchors=tuple(anchors),
            )

    # Check ordering patterns
    for pattern in _TEMPORAL_ORDERING_PATTERNS:
        if pattern.search(query):
            anchors = extract_event_anchors(query)
            return TemporalSignal(
                signal_type="ordering",
                confidence=0.7,
                event_anchors=tuple(anchors),
            )

    # Check "when" patterns
    for pattern in _TEMPORAL_WHEN_PATTERNS:
        if pattern.search(query):
            anchors = extract_event_anchors(query)
            return TemporalSignal(
                signal_type="when",
                confidence=0.6,
                event_anchors=tuple(anchors),
            )

    return None


def extract_date_range(
    signal: TemporalSignal,
    reference_date: datetime,
    tolerance: float = 0.5,
) -> tuple[datetime, datetime] | None:
    """Compute a target date range from a temporal signal.

    Args:
        signal: The detected temporal signal.
        reference_date: The reference point (e.g. question_date).
        tolerance: Window expansion factor (0.5 = ±50% of duration).

    Returns:
        (start, end) datetime tuple, or None if no date range can be computed.
    """
    if signal.duration_unit is None or signal.duration_value is None:
        return None

    # Convert duration to timedelta
    value = signal.duration_value
    unit = signal.duration_unit

    if unit == "days":
        delta = timedelta(days=value)
    elif unit == "weeks":
        delta = timedelta(weeks=value)
    elif unit == "months":
        delta = timedelta(days=value * 30)
    elif unit == "years":
        delta = timedelta(days=value * 365)
    else:
        return None

    # Target date = reference - duration
    target = reference_date - delta

    # Expand window by tolerance
    window = delta * tolerance
    start = target - window
    end = target + window

    return (start, end)


def extract_event_anchors(query: str) -> list[str]:
    """Extract event-specific terms from a temporal query.

    Returns distinctive nouns/terms that likely refer to specific events,
    filtering out temporal vocabulary and common stopwords.

    Args:
        query: The query text.

    Returns:
        List of event anchor terms (max 5).
    """
    words = _WORD_RE.findall(query.lower())
    if not words:
        return []

    # Filter stopwords and temporal vocabulary
    anchors = [w for w in words if w not in _EVENT_STOPWORDS]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for w in anchors:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return unique[:5]
