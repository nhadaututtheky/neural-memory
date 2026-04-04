"""Context merger — merges structured context dicts into rich content.

Accepts a content string + optional context dict from any agent,
and produces enriched content using type-specific templates.
No LLM required — pure template-based merging.
"""

from __future__ import annotations

from typing import Any

# -- Per-type merge templates -----------------------------------------------
# Each template defines which context keys it uses and how to merge them.
# Keys not in the template are appended as "Key: value" fallback lines.

_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    # (context_key, sentence_template_with_{value}_placeholder)
    "decision": [
        ("chosen", "Chose {value}"),
        ("reason", "because {value}"),
        ("alternatives", "Alternatives considered: {value}"),
        ("rejected", "Rejected alternatives: {value}"),
        ("confidence", "Confidence: {value}"),
        ("decided_by", "Decided by {value}"),
        ("impact", "Impact: {value}"),
    ],
    "error": [
        ("root_cause", "Root cause: {value}"),
        ("cause", "Caused by {value}"),
        ("fix", "Fixed by {value}"),
        ("prevention", "Prevention: {value}"),
        ("impact", "Impact: {value}"),
    ],
    "insight": [
        ("reason", "because {value}"),
        ("evidence", "Evidence: {value}"),
        ("applies_to", "Applies to {value}"),
        ("source", "Source: {value}"),
    ],
    "workflow": [
        ("steps", "Steps: {value}"),
        ("trigger", "Triggered by {value}"),
        ("output", "Output: {value}"),
        ("frequency", "Frequency: {value}"),
    ],
    "preference": [
        ("reason", "because {value}"),
        ("scope", "Scope: {value}"),
        ("override", "Overrides: {value}"),
    ],
    "instruction": [
        ("reason", "because {value}"),
        ("scope", "Applies to {value}"),
        ("exceptions", "Exceptions: {value}"),
    ],
    "fact": [
        ("source", "Source: {value}"),
        ("verified", "Verified: {value}"),
        ("context", "Context: {value}"),
    ],
}

# Keys that are handled by templates — skip in fallback
_KNOWN_KEYS = {
    "reason",
    "alternatives",
    "decided_by",
    "root_cause",
    "cause",
    "fix",
    "prevention",
    "impact",
    "evidence",
    "applies_to",
    "source",
    "steps",
    "trigger",
    "output",
    "frequency",
    "scope",
    "override",
    "exceptions",
    "verified",
    "context",
}


def _format_value(value: Any) -> str:
    """Format a context value to string."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def merge_context(
    content: str,
    context: dict[str, Any] | None,
    memory_type: str | None = None,
) -> str:
    """Merge structured context into content using type-specific templates.

    If no context provided, returns content unchanged.
    Template keys are merged inline; unknown keys appended as fallback lines.

    Args:
        content: Original memory content from agent.
        context: Optional structured context dict.
        memory_type: Memory type for template selection.

    Returns:
        Enriched content string (new string, never mutates input).
    """
    if not context:
        return content

    parts: list[str] = [content.rstrip(".").rstrip()]
    used_keys: set[str] = set()

    # Apply type-specific templates
    mem_type = (memory_type or "fact").lower()
    templates = _TEMPLATES.get(mem_type, _TEMPLATES["fact"])

    for key, template in templates:
        if context.get(key):
            value_str = _format_value(context[key])
            # First template item joins with content via separator
            if not used_keys:
                # "reason" key joins with " because X", others get ". Template"
                if template.startswith("because"):
                    parts.append(f" {template.format(value=value_str)}")
                else:
                    parts.append(f". {template.format(value=value_str)}")
            else:
                parts.append(f". {template.format(value=value_str)}")
            used_keys.add(key)

    # Fallback: append unknown keys
    for key, value in context.items():
        if key in used_keys or key in _KNOWN_KEYS:
            continue
        if value:
            value_str = _format_value(value)
            label = key.replace("_", " ").capitalize()
            parts.append(f". {label}: {value_str}")

    result = "".join(parts)
    # Ensure ends with period
    if not result.endswith("."):
        result += "."

    return result
