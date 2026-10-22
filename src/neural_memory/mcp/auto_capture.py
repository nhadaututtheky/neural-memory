"""Auto-capture pattern detection for extracting memories from text.

Pattern lists live on :class:`CapturePolicy` in ``capture_policy``. This
module picks the input floor, runs every registered policy, and dedups.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Any

from neural_memory.mcp.capture_policy import (
    _VI_CONFIDENCE_PENALTY,
    _VI_MIN_CAPTURE_LEN,
    _cjk_script,
    _is_vietnamese_text,
    _vi_quality_gate,
    detect_language,
    get_policy,
    iter_policies,
)
from neural_memory.utils.metrics import record_empty_capture
from neural_memory.utils.simhash import is_near_duplicate, simhash

logger = logging.getLogger(__name__)

# Maximum text length for regex processing — prevents ReDoS on huge inputs.
_MAX_REGEX_TEXT_LENGTH = 50_000

# One-time warning flag for pyvi in auto-capture.
_PYVI_AC_WARNED = False

# Type prefixes used for deduplication.
_TYPE_PREFIXES = ("decision: ", "error: ", "todo: ", "insight: ", "preference: ")

__all__ = [
    "_VI_CONFIDENCE_PENALTY",
    "_VI_MIN_CAPTURE_LEN",
    "_cjk_script",
    "_vi_quality_gate",
    "analyze_text_for_memories",
    "classify_empty_capture",
    "detect_language",
    "empty_capture_hint",
    "get_policy",
]


def classify_empty_capture(text: str) -> tuple[str, str]:
    """Return ``(reason, language)`` for an empty detection.

    ``reason`` is ``too_short``, ``cjk_unsupported``, or ``no_trigger``.
    ``language`` is ``zh``, ``ja``, ``ko``, ``vi``, or ``en``.
    """
    stripped = text.strip()
    language = detect_language(stripped)
    min_len = get_policy(stripped).min_input_length
    if len(stripped) < min_len:
        return "too_short", language
    if language in {"ja", "ko"}:
        return "cjk_unsupported", language
    return "no_trigger", language


def empty_capture_hint(text: str) -> str:
    """Explain an empty detection result.

    Without a hint, users of languages the extractor cannot read see
    "No memorable content detected" and cannot tell an empty input from an
    engine that never understood their text — the failure is silent. A too
    short input is reported as such, so the advice does not wrongly send the
    user hunting for a trigger word when the text never reached the detector.

    Each call records one empty-capture sample. Callers use this only when
    detection found nothing.
    """
    reason, language = classify_empty_capture(text)
    record_empty_capture(reason, language)
    stripped = text.strip()
    min_len = get_policy(stripped).min_input_length
    if reason == "too_short":
        return (
            f"No patterns matched: the input is {len(stripped)} characters, below the "
            f"{min_len}-character minimum that detection requires. "
            "Add more context and try again."
        )
    if language == "ja":
        return (
            "No patterns matched. Japanese text is not yet supported by the "
            "extractor (patterns cover English, Vietnamese, and Chinese)."
        )
    if language == "ko":
        return (
            "No patterns matched. Korean text is not yet supported by the "
            "extractor (patterns cover English, Vietnamese, and Chinese)."
        )
    return get_policy(stripped).empty_hint(stripped)


def _warn_pyvi_missing() -> None:
    """Log a one-time warning when Vietnamese text is detected but pyvi is not installed."""
    global _PYVI_AC_WARNED
    if _PYVI_AC_WARNED:
        return
    try:
        with warnings.catch_warnings():
            # pyvi emits NumPy VisibleDeprecationWarning (UserWarning subclass, NOT
            # DeprecationWarning) during pickle.load at module import. Narrow
            # category filters miss it — suppress everything in this scoped block.
            warnings.simplefilter("ignore")
            import pyvi  # probe-only import to trigger ImportError if missing

        del pyvi
    except ImportError:
        logger.warning(
            "Vietnamese text detected in auto-capture but pyvi is not installed — "
            "keyword tags will be low-quality bigrams. "
            "Install with: pip install pyvi"
        )
        _PYVI_AC_WARNED = True


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
    capture_preferences: bool = True,
) -> list[dict[str, Any]]:
    """Analyze text and detect potential memories.

    Returns list of detected memories with type, content, and confidence.
    The policy for ``text`` chooses the input floor. Every registered policy
    still contributes its own patterns, in category order, so a mixed
    sentence keeps today's matches.
    """
    stripped = text.strip()
    if len(stripped) < get_policy(stripped).min_input_length:
        return []

    # Truncate to prevent ReDoS on very large inputs.
    if len(text) > _MAX_REGEX_TEXT_LENGTH:
        text = text[:_MAX_REGEX_TEXT_LENGTH]

    # Warn once if Vietnamese text detected but pyvi not installed.
    if _is_vietnamese_text(text.lower()):
        _warn_pyvi_missing()

    detected: list[dict[str, Any]] = []
    categories = (
        ("decision", capture_decisions),
        ("error", capture_errors),
        ("todo", capture_todos),
        ("fact", capture_facts),
        ("insight", capture_insights),
        ("preference", capture_preferences),
    )
    policies = iter_policies()
    for memory_type, enabled in categories:
        if not enabled:
            continue
        for policy in policies:
            detected.extend(match.as_dict() for match in policy.matches_for(text, memory_type))

    # Remove duplicates: exact MD5 match + SimHash near-duplicate.
    seen_exact: set[str] = set()
    seen_hashes: list[int] = []
    unique_detected: list[dict[str, Any]] = []
    for item in detected:
        content_key = _dedup_key(item["content"])
        if content_key in seen_exact:
            continue

        # Check simhash near-duplicate against already-seen items.
        item_hash = simhash(item["content"])
        if any(is_near_duplicate(item_hash, h) for h in seen_hashes):
            continue

        seen_exact.add(content_key)
        seen_hashes.append(item_hash)
        unique_detected.append(item)

    return unique_detected
