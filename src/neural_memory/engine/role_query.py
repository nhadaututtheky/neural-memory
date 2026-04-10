"""Role-aware query detection for retrieval improvement.

Detects when a query targets assistant-said content ("you said", "you recommended")
vs user-said content ("I said", "I mentioned"), and boosts fibers with matching
role tags. All heuristic-based — zero LLM calls.
"""

from __future__ import annotations

import re
from enum import Enum

# ---------------------------------------------------------------------------
# Role target patterns
# ---------------------------------------------------------------------------


class RoleTarget(Enum):
    """Which role the query is targeting."""

    ASSISTANT = "assistant"
    USER = "user"


# Patterns targeting assistant-said content
_ASSISTANT_TARGET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\byou (?:said|told|mentioned|recommended|suggested|advised|explained)\b",
        r"\byour (?:recommendation|suggestion|advice|answer|response|explanation)\b",
        r"\bwhat did you (?:say|recommend|suggest|tell|advise)\b",
        r"\bcan you remind me (?:what you|of your)\b",
        r"\bwhat (?:was|were) your\b",
        r"\byou(?:'ve| have) (?:said|mentioned|recommended|suggested)\b",
        r"\bthe (?:answer|response|recommendation|suggestion) you (?:gave|provided)\b",
    ]
]

# Patterns targeting user-said content
_USER_TARGET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi (?:said|told|mentioned|asked|shared|described)\b",
        r"\bwhat did i (?:say|tell|mention|ask|share)\b",
        r"\bwhat i (?:said|told|mentioned|asked|shared)\b",
        r"\bmy (?:question|request|message|comment)\b",
        r"\bsomething i (?:said|mentioned|asked|told)\b",
        r"\bi(?:'ve| have) (?:said|mentioned|asked|told)\b",
    ]
]


def detect_role_target(query: str) -> RoleTarget | None:
    """Detect whether a query targets assistant or user content.

    Args:
        query: The user's search query.

    Returns:
        RoleTarget.ASSISTANT, RoleTarget.USER, or None (no role signal).
    """
    if not query or len(query) < 5:
        return None

    # Check assistant patterns first (SSA category)
    assistant_matches = sum(1 for p in _ASSISTANT_TARGET_PATTERNS if p.search(query))
    user_matches = sum(1 for p in _USER_TARGET_PATTERNS if p.search(query))

    # Need at least 1 match, and clear winner
    if assistant_matches > 0 and assistant_matches >= user_matches:
        return RoleTarget.ASSISTANT
    if user_matches > 0 and user_matches > assistant_matches:
        return RoleTarget.USER

    return None
