"""Markdown formatter for brain export.

Converts a BrainSnapshot into a human-readable markdown document,
grouped by memory type with tag index and statistics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


def _escape_md_table(text: str) -> str:
    """Escape pipe characters for markdown table cells."""
    return text.replace("|", "\\|")


def snapshot_to_markdown(
    snapshot: Any,
    *,
    brain_name: str = "",
    excluded_count: int = 0,
) -> str:
    """Convert a BrainSnapshot export dict to markdown.

    Args:
        snapshot: Export data dict with neurons, synapses, fibers, metadata.
        brain_name: Display name of the brain.
        excluded_count: Number of sensitive neurons excluded.

    Returns:
        Formatted markdown string.
    """
    name = brain_name or snapshot.get("brain_name", "unknown")
    exported_at = snapshot.get("exported_at", "")
    neurons = snapshot.get("neurons", [])
    synapses = snapshot.get("synapses", [])
    fibers = snapshot.get("fibers", [])
    metadata = snapshot.get("metadata", {})
    typed_memories = metadata.get("typed_memories", [])

    lines: list[str] = []

    # Header
    lines.append(f"# Brain: {name}")
    lines.append("")
    lines.append(
        f"> Exported: {exported_at} | "
        f"Neurons: {len(neurons)} | "
        f"Synapses: {len(synapses)} | "
        f"Fibers: {len(fibers)}"
    )
    if excluded_count > 0:
        lines.append(f"> Excluded {excluded_count} neurons with sensitive content")
    lines.append("")

    # Build fiber lookup and typed memory mapping
    fiber_map: dict[str, dict[str, Any]] = {f["id"]: f for f in fibers}
    neuron_map: dict[str, dict[str, Any]] = {n["id"]: n for n in neurons}

    # Group fibers by memory type
    type_to_fibers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    typed_fiber_ids: set[str] = set()

    for tm in typed_memories:
        fiber_id = tm.get("fiber_id", "")
        mem_type = tm.get("memory_type", "unknown")
        if fiber_id in fiber_map:
            type_to_fibers[mem_type].append(fiber_map[fiber_id])
            typed_fiber_ids.add(fiber_id)

    # Order: fact, decision, preference, insight, instruction, workflow, reference, error, todo, context, then rest
    type_order = [
        "fact",
        "decision",
        "preference",
        "insight",
        "instruction",
        "workflow",
        "reference",
        "error",
        "todo",
        "context",
    ]
    type_labels = {
        "fact": "Facts",
        "decision": "Decisions",
        "preference": "Preferences",
        "insight": "Insights",
        "instruction": "Instructions",
        "workflow": "Workflows",
        "reference": "References",
        "error": "Errors",
        "todo": "TODOs",
        "context": "Context",
    }

    # Render typed sections
    for mem_type in type_order:
        group = type_to_fibers.get(mem_type, [])
        if not group:
            continue
        label = type_labels.get(mem_type, mem_type.title())
        lines.append(f"## {label} ({len(group)})")
        lines.append("")
        for fiber in _sort_fibers_by_date(group):
            lines.append(_format_fiber_line(fiber, neuron_map))
        lines.append("")

    # Render any types not in the standard order
    extra_types = set(type_to_fibers.keys()) - set(type_order)
    for mem_type in sorted(extra_types):
        group = type_to_fibers[mem_type]
        lines.append(f"## {mem_type.title()} ({len(group)})")
        lines.append("")
        for fiber in _sort_fibers_by_date(group):
            lines.append(_format_fiber_line(fiber, neuron_map))
        lines.append("")

    # Uncategorized fibers
    uncategorized = [f for f in fibers if f["id"] not in typed_fiber_ids]
    if uncategorized:
        lines.append(f"## Uncategorized ({len(uncategorized)})")
        lines.append("")
        for fiber in _sort_fibers_by_date(uncategorized):
            lines.append(_format_fiber_line(fiber, neuron_map))
        lines.append("")

    # Tag index
    tag_counter: Counter[str] = Counter()
    for fiber in fibers:
        for tag in fiber.get("auto_tags", []):
            tag_counter[tag] += 1
        for tag in fiber.get("agent_tags", []):
            tag_counter[tag] += 1

    if tag_counter:
        lines.append("## Tags Index")
        lines.append("")
        for tag, count in tag_counter.most_common():
            lines.append(f"- **#{tag}**: {count} memories")
        lines.append("")

    # Statistics
    lines.append("## Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total neurons | {len(neurons)} |")
    lines.append(f"| Total synapses | {len(synapses)} |")
    lines.append(f"| Total fibers | {len(fibers)} |")

    # Neuron type breakdown
    neuron_types: Counter[str] = Counter()
    for n in neurons:
        neuron_types[n.get("type", "unknown")] += 1
    if neuron_types:
        for ntype, count in neuron_types.most_common():
            lines.append(f"| Neurons ({_escape_md_table(ntype)}) | {count} |")

    # Synapse type breakdown
    synapse_types: Counter[str] = Counter()
    for s in synapses:
        synapse_types[s.get("type", "unknown")] += 1
    if synapse_types:
        for stype, count in synapse_types.most_common(10):
            lines.append(f"| Synapses ({_escape_md_table(stype)}) | {count} |")

    # Typed memory breakdown
    mem_type_counts: Counter[str] = Counter()
    for tm in typed_memories:
        mem_type_counts[tm.get("memory_type", "unknown")] += 1
    if mem_type_counts:
        for mtype, count in mem_type_counts.most_common():
            lines.append(f"| Memories ({mtype}) | {count} |")

    if excluded_count > 0:
        lines.append(f"| Excluded (sensitive) | {excluded_count} |")

    pinned_count = sum(1 for f in fibers if f.get("pinned"))
    if pinned_count:
        lines.append(f"| Pinned fibers | {pinned_count} |")

    lines.append("")
    return "\n".join(lines)


def _sort_fibers_by_date(fibers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort fibers by time_start or created_at, newest first."""

    def sort_key(f: dict[str, Any]) -> str:
        return f.get("time_start") or f.get("created_at") or ""

    return sorted(fibers, key=sort_key, reverse=True)


def _format_fiber_line(
    fiber: dict[str, Any],
    neuron_map: dict[str, dict[str, Any]],
) -> str:
    """Format a single fiber as a markdown list item."""
    # Use summary if available, otherwise find anchor neuron content
    summary = fiber.get("summary") or ""
    if not summary:
        anchor_id = fiber.get("anchor_neuron_id", "")
        if anchor_id and anchor_id in neuron_map:
            summary = neuron_map[anchor_id].get("content", "")

    if not summary:
        # Fallback: find first non-time neuron content
        for nid in fiber.get("neuron_ids", []):
            neuron = neuron_map.get(nid)
            if neuron and neuron.get("type") != "time":
                summary = neuron.get("content", "")
                break

    if not summary:
        summary = "(no content)"

    # Truncate long summaries
    if len(summary) > 200:
        summary = summary[:197] + "..."

    # Date
    date_str = ""
    time_start = fiber.get("time_start") or fiber.get("created_at")
    if time_start:
        if isinstance(time_start, str):
            date_str = time_start[:10]
        elif isinstance(time_start, datetime):
            date_str = time_start.strftime("%Y-%m-%d")

    # Tags
    all_tags = list(fiber.get("auto_tags", [])) + list(fiber.get("agent_tags", []))
    tag_str = " ".join(f"`#{t}`" for t in all_tags[:5]) if all_tags else ""

    # Pinned indicator
    pinned = " (pinned)" if fiber.get("pinned") else ""

    parts = [f"- {summary}"]
    if date_str:
        parts.append(f"[{date_str}]")
    if pinned:
        parts.append(pinned)
    if tag_str:
        parts.append(tag_str)

    return " ".join(parts)
