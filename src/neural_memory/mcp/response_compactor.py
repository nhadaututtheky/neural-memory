"""MCP response compaction — strip metadata hints and truncate verbose fields.

Central compactor that reduces MCP tool response token usage by 60-80%
when compact mode is enabled. Injected in server.py between call_tool()
and json.dumps().

Design: immutable — always returns a new dict, never mutates input.
"""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.unified_config import ResponseConfig

logger = logging.getLogger(__name__)

# Metadata keys safe to strip in compact mode.
# These are DX hints, session tracking, and auxiliary data — not core results.
STRIPPABLE_KEYS: frozenset[str] = frozenset(
    {
        "maintenance_hint",
        "update_hint",
        "onboarding",
        "cross_language_hint",
        "score_breakdown",
        "related_queries",
        "related_memories",
        "dedup_hint",
        "optimization_stats",
        "pending_alerts",
        "session_topics",
        "session_query_count",
        "roadmap",
    }
)

# Keys whose list values should be replaced with a count in compact mode.
COUNT_REPLACE_KEYS: frozenset[str] = frozenset(
    {
        "fibers_matched",
        "conflicts",
        "expiry_warnings",
    }
)

# Keys whose string values should be truncated if too long.
LONG_STRING_KEYS: dict[str, int] = {
    "markdown": 500,
}

# Keys inside list items whose string content should be previewed.
CONTENT_PREVIEW_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "body",
        "description",
        "message",
    }
)


def _compact_list_item(
    item: Any,
    preview_length: int,
) -> Any:
    """Compact a single item inside a list (e.g., a memory dict).

    Truncates content/body/description fields to preview_length.
    """
    if not isinstance(item, dict):
        return item

    out: dict[str, Any] = {}
    for key, value in item.items():
        if key in CONTENT_PREVIEW_KEYS and isinstance(value, str) and len(value) > preview_length:
            out[key] = value[:preview_length] + "..."
            out[f"_{key}_truncated"] = True
        else:
            out[key] = value
    return out


def compact_response(
    result: dict[str, Any],
    config: ResponseConfig,
) -> dict[str, Any]:
    """Apply compact transformations to an MCP tool response.

    Args:
        result: Raw tool response dict (not mutated).
        config: Response compaction settings.

    Returns:
        New dict with metadata stripped and lists truncated.
    """
    if not isinstance(result, dict):
        return result

    out: dict[str, Any] = {}

    for key, value in result.items():
        # Strip metadata hint keys
        if config.strip_hints and key in STRIPPABLE_KEYS:
            continue

        # Replace list fields with count only
        if key in COUNT_REPLACE_KEYS and isinstance(value, list):
            out[f"{key}_count"] = len(value)
            continue

        # Truncate long string fields (markdown, etc.)
        if key in LONG_STRING_KEYS and isinstance(value, str):
            max_len = LONG_STRING_KEYS[key]
            if len(value) > max_len:
                out[key] = value[:max_len] + "...(truncated)"
                out[f"_{key}_truncated"] = True
                continue
            out[key] = value
            continue

        # Truncate long lists + preview content in items
        if isinstance(value, list):
            items = value
            truncated = False
            total = len(items)

            if total > config.max_list_items:
                items = items[: config.max_list_items]
                truncated = True

            # Preview content fields in list items
            items = [_compact_list_item(item, config.content_preview_length) for item in items]

            out[key] = items
            if truncated:
                out[f"_{key}_truncated"] = True
                out[f"_{key}_total"] = total
            continue

        # Recurse into nested dicts
        if isinstance(value, dict):
            out[key] = compact_response(value, config)
            continue

        out[key] = value

    return out


def should_compact(
    *,
    tool_args: dict[str, Any],
    config: ResponseConfig,
) -> bool:
    """Determine if compact mode should be applied.

    Priority:
    1. Per-call `compact` param (explicit override) — highest
    2. Global `config.response.compact_mode` — default

    Args:
        tool_args: Raw tool arguments from MCP call.
        config: Global response config.

    Returns:
        True if response should be compacted.
    """
    per_call = tool_args.pop("compact", None)
    if per_call is not None:
        return bool(per_call)
    return config.compact_mode


# Approximate chars per token for budget estimation.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from string length (~4 chars/token)."""
    return len(text) // _CHARS_PER_TOKEN


def needs_auto_compact(result: dict[str, Any], threshold: int) -> bool:
    """Check if any list in the response exceeds the auto-compact threshold.

    Args:
        result: Raw tool response dict.
        threshold: Max list length before auto-compact triggers (0 = disabled).

    Returns:
        True if auto-compact should be applied.
    """
    if threshold <= 0:
        return False
    return any(
        isinstance(value, list) and len(value) > threshold for value in result.values()
    )


def apply_token_budget(result: dict[str, Any], budget: int) -> dict[str, Any]:
    """Progressively strip response until it fits within token budget.

    Stripping order (least to most aggressive):
    1. Strip metadata hints
    2. Truncate lists to 5 items
    3. Truncate content fields to 80 chars

    Args:
        result: Already-compacted response dict (not mutated).
        budget: Max tokens for the response.

    Returns:
        New dict fitting within budget, with _token_budget_applied flag.
    """
    import json

    serialized = json.dumps(result)
    if _estimate_tokens(serialized) <= budget:
        return result

    # Level 1: strip metadata hints (may already be done by compact_response)
    level1 = {k: v for k, v in result.items() if k not in STRIPPABLE_KEYS}
    serialized = json.dumps(level1)
    if _estimate_tokens(serialized) <= budget:
        return {**level1, "_token_budget_applied": True}

    # Level 2: truncate lists to 5 items
    level2: dict[str, Any] = {}
    for key, value in level1.items():
        if isinstance(value, list) and len(value) > 5:
            level2[key] = value[:5]
            level2[f"_{key}_truncated"] = True
            level2[f"_{key}_total"] = len(value)
        else:
            level2[key] = value
    serialized = json.dumps(level2)
    if _estimate_tokens(serialized) <= budget:
        return {**level2, "_token_budget_applied": True}

    # Level 3: truncate all string values > 80 chars
    level3: dict[str, Any] = {}
    for key, value in level2.items():
        if isinstance(value, str) and len(value) > 80:
            level3[key] = value[:80] + "..."
        elif isinstance(value, list):
            level3[key] = [
                _compact_list_item(item, 80) if isinstance(item, dict) else item for item in value
            ]
        else:
            level3[key] = value
    level3["_token_budget_applied"] = True
    return level3
