"""Preference signal detection for retrieval improvement.

Detects preference-establishing content at ingest time and preference-seeking
queries at retrieval time. Used to boost retrieval of sessions where users
express preferences, favorites, or domain expertise.

All heuristic-based — zero LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- Preference signal patterns (ingest-time) ---

# User expressing a preference
_USER_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi (?:prefer|like|love|enjoy|use|favor)\b",
        r"\bmy (?:favorite|preferred|go-to|usual)\b",
        r"\bi(?:'ve| have) been (?:using|working with|learning)\b",
        r"\bi usually (?:use|go with|choose|pick)\b",
        r"\bi(?:'m| am) (?:a fan of|into|really into)\b",
        r"\bi always (?:use|choose|go with)\b",
        r"\bi (?:switched to|moved to|started using)\b",
        r"\bi(?:'d| would) (?:rather|recommend)\b",
    ]
]

# Assistant acknowledging user preference
_ASSISTANT_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbased on your (?:preference|interest|experience)\b",
        r"\bsince you (?:like|prefer|enjoy|use|mentioned)\b",
        r"\byou(?:'ve| have) (?:mentioned|said) you (?:prefer|like|enjoy)\b",
        r"\bgiven (?:your|that you) (?:preference|interest|background)\b",
        r"\bknowing (?:you|your) (?:prefer|like|use)\b",
    ]
]

# --- Preference query patterns (retrieval-time) ---

_PREFERENCE_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brecommend\b",
        r"\bsuggest(?:ion)?\b",
        r"\bwhat (?:do|did|should|would|could) (?:i|you)\b.*\b(?:use|try|pick|choose|learn|read|watch)\b",
        r"\bwhat(?:'s| is) (?:a good|the best|my)\b",
        r"\bcan you (?:recommend|suggest)\b",
        r"\bany (?:recommendation|suggestion|idea)s?\b",
        r"\bwhat (?:do|did) i (?:like|prefer|enjoy|use)\b",
        r"\bmy (?:preference|favorite)\b",
        r"\bshould i (?:use|try|pick|go with|learn)\b",
        r"\b(?:best|good) (?:option|choice|tool|resource|book|course)s?\b",
        r"\bwhat (?:tool|app|software|framework|library|resource)s?\b.*\b(?:for|to)\b",
    ]
]

# Stopwords to exclude from domain extraction
_STOPWORDS: frozenset[str] = frozenset(
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
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "and",
        "but",
        "or",
        "nor",
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
        "whom",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "really",
        "like",
        "use",
        "using",
        "used",
        "prefer",
        "preferred",
        "favorite",
        "go",
        "going",
        "get",
        "getting",
        "good",
        "great",
        "best",
        "well",
        "much",
        "many",
        "think",
        "know",
        "want",
        "need",
        "said",
        "told",
        "asked",
        "something",
        "anything",
        "everything",
        "nothing",
    ]
)

_WORD_RE = re.compile(r"\b[a-z][a-z0-9-]{2,}\b")


@dataclass(frozen=True)
class PreferenceSignal:
    """Detected preference signal from content."""

    confidence: float  # 0.0-1.0: strength of preference signal
    domain_keywords: tuple[str, ...]  # extracted topic keywords
    pattern_matches: int  # number of patterns matched


def detect_preference_signals(
    content: str,
    role: str = "user",
) -> PreferenceSignal | None:
    """Detect preference-establishing signals in content.

    Args:
        content: The message content to analyze.
        role: "user" or "assistant" — selects which pattern set to use.

    Returns:
        PreferenceSignal if preference patterns found, None otherwise.
    """
    if not content or len(content) < 10:
        return None

    patterns = _USER_PREFERENCE_PATTERNS if role == "user" else _ASSISTANT_PREFERENCE_PATTERNS

    matches = sum(1 for p in patterns if p.search(content))
    if matches == 0:
        return None

    # Confidence scales with number of matches
    confidence = min(1.0, matches * 0.3 + 0.2)

    # Extract domain keywords
    domain = extract_preference_domain(content)

    return PreferenceSignal(
        confidence=confidence,
        domain_keywords=tuple(domain),
        pattern_matches=matches,
    )


def is_preference_query(query: str) -> bool:
    """Check if a query is seeking recommendations or preferences.

    Args:
        query: The user's search query.

    Returns:
        True if the query is preference/recommendation-seeking.
    """
    if not query or len(query) < 5:
        return False

    return any(p.search(query) for p in _PREFERENCE_QUERY_PATTERNS)


def extract_preference_domain(content: str) -> list[str]:
    """Extract domain-specific keywords from content.

    Uses simple term frequency to find distinctive terms,
    filtering out common stopwords.

    Args:
        content: Text to extract domain keywords from.

    Returns:
        Top 5 domain keywords by frequency.
    """
    words = _WORD_RE.findall(content.lower())
    if not words:
        return []

    # Count frequencies, excluding stopwords
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return []

    # Return top-5 by frequency, then alphabetical for ties
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [term for term, _count in sorted_terms[:5]]
