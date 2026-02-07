"""Auto-capture pattern detection for extracting memories from text."""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Minimum text length to avoid false positives on tiny inputs
_MIN_TEXT_LENGTH = 20

# Maximum text length for regex processing — prevents ReDoS on huge inputs
_MAX_REGEX_TEXT_LENGTH = 50_000

# Type prefixes used for deduplication
_TYPE_PREFIXES = ("decision: ", "error: ", "todo: ", "insight: ")

DECISION_PATTERNS = [
    # English
    r"(?:we |I )(?:decided|chose|selected|picked|opted)(?: to)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?decision(?: is)?[:\s]+(.+?)(?:\.|$)",
    r"(?:we\'re |I\'m )going (?:to|with)[:\s]+(.+?)(?:\.|$)",
    r"let\'s (?:go with|use|choose)[:\s]+(.+?)(?:\.|$)",
    r"(?:chose|picked|selected) (.+?) (?:over|instead of) (.+?)(?:\.|$)",
    r"(?:going|switched|moving) (?:to|with|from .+? to) (.+?)(?:\.|$)",
    # Vietnamese
    r"(?:quyết định|chọn|dùng|chuyển sang)[:\s]+(.+?)(?:\.|$)",
    r"(?:sẽ |sẽ phải )(?:dùng|chọn|chuyển)[:\s]+(.+?)(?:\.|$)",
]

ERROR_PATTERNS = [
    # English
    r"error[:\s]+(.+?)(?:\.|$)",
    r"failed[:\s]+(.+?)(?:\.|$)",
    r"bug[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?issue (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"problem[:\s]+(.+?)(?:\.|$)",
    r"(?:fixed|resolved|solved)(?: (?:it|this))? by[:\s]+(.+?)(?:\.|$)",
    r"(?:workaround|hack)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:lỗi|bug|vấn đề) (?:là|do|ở)[:\s]+(.+?)(?:\.|$)",
    r"(?:sửa|fix) (?:được |xong )?(?:bằng cách|bởi)[:\s]+(.+?)(?:\.|$)",
]

TODO_PATTERNS = [
    # English
    r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.+?)(?:\.|$)",
    r"(?:we |I )?(?:need to|should|must|have to)[:\s]+(.{5,80}?)(?:\.|,| but | or | and |$)",
    r"(?:remember to|don\'t forget to)[:\s]+(.+?)(?:\.|$)",
    r"(?:later|next)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:cần phải|cần|phải|nên)[:\s]+(.+?)(?:\.|$)",
    r"(?:nhớ|đừng quên)[:\s]+(.+?)(?:\.|$)",
]

FACT_PATTERNS = [
    # English
    r"(?:the |a )?(?:answer|solution|fix) (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:works|worked) because[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?(?:key|important|note)[:\s]+(.+?)(?:\.|$)",
    r"(?:learned|discovered|found out)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:đáp án|giải pháp|cách fix) (?:là|:)[:\s]+(.+?)(?:\.|$)",
]

INSIGHT_PATTERNS = [
    # English - "aha moments"
    r"turns out[:\s]+(.+?)(?:\.|$)",
    r"the trick (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:I |we )(?:realized|discovered|figured out|noticed)(?: that)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?root cause (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:turns out|actually means)[:\s]+(.+?)(?:\.|$)",
    r"(?:lesson learned|takeaway|key insight)[:\s]+(.+?)(?:\.|$)",
    r"(?:TIL|today I learned)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:hóa ra|thì ra|té ra)[:\s]+(.+?)(?:\.|$)",
    r"(?:bài học|điều quan trọng)[:\s]+(.+?)(?:\.|$)",
    r"(?:nguyên nhân|root cause) (?:là|do)[:\s]+(.+?)(?:\.|$)",
    r"(?:mới biết|mới phát hiện)[:\s]+(.+?)(?:\.|$)",
]


def _detect_patterns(
    text: str,
    patterns: list[str],
    memory_type: str,
    confidence: float,
    priority: int,
    min_match_len: int,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Run a list of regex patterns and return detected memories."""
    detected: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Handle tuple matches from patterns with multiple groups
            if isinstance(match, tuple):
                match = " ".join(part for part in match if part)
            captured = match.strip()
            if len(captured) < min_match_len:
                continue

            # Adjust confidence based on capture quality
            adjusted_confidence = confidence
            if len(captured) > 200:
                adjusted_confidence *= 0.7  # Penalize truly excessive captures
            elif len(captured) < 10:
                adjusted_confidence *= 0.3  # Penalize too-short captures

            # Trim at sentence boundary if over-captured
            if len(captured) > 100:
                for sep in (".", "!", "?", ";"):
                    idx = captured.find(sep, 50)
                    if idx > 0:
                        captured = captured[:idx]
                        break

            content = f"{prefix}{captured}" if prefix else captured
            detected.append(
                {
                    "type": memory_type,
                    "content": content,
                    "confidence": adjusted_confidence,
                    "priority": priority,
                }
            )
    return detected


def _dedup_key(content: str) -> str:
    """Create a deduplication key by stripping type prefix and hashing."""
    key = content.lower()
    for prefix in _TYPE_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return hashlib.md5(key.encode()).hexdigest()


def analyze_text_for_memories(
    text: str,
    *,
    capture_decisions: bool = True,
    capture_errors: bool = True,
    capture_todos: bool = True,
    capture_facts: bool = True,
    capture_insights: bool = True,
) -> list[dict[str, Any]]:
    """Analyze text and detect potential memories.

    Returns list of detected memories with type, content, and confidence.
    """
    if len(text.strip()) < _MIN_TEXT_LENGTH:
        return []

    # Truncate to prevent ReDoS on very large inputs
    if len(text) > _MAX_REGEX_TEXT_LENGTH:
        text = text[:_MAX_REGEX_TEXT_LENGTH]

    detected: list[dict[str, Any]] = []
    text_lower = text.lower()

    if capture_decisions:
        detected.extend(
            _detect_patterns(text_lower, DECISION_PATTERNS, "decision", 0.8, 6, 10, "Decision: ")
        )

    if capture_errors:
        detected.extend(
            _detect_patterns(text_lower, ERROR_PATTERNS, "error", 0.85, 7, 10, "Error: ")
        )

    if capture_todos:
        detected.extend(_detect_patterns(text, TODO_PATTERNS, "todo", 0.75, 5, 5, "TODO: "))

    if capture_facts:
        detected.extend(_detect_patterns(text_lower, FACT_PATTERNS, "fact", 0.7, 5, 15))

    if capture_insights:
        detected.extend(
            _detect_patterns(text_lower, INSIGHT_PATTERNS, "insight", 0.8, 6, 15, "Insight: ")
        )

    # Remove duplicates with improved key extraction
    seen: set[str] = set()
    unique_detected: list[dict[str, Any]] = []
    for item in detected:
        content_key = _dedup_key(item["content"])
        if content_key not in seen:
            seen.add(content_key)
            unique_detected.append(item)

    return unique_detected
