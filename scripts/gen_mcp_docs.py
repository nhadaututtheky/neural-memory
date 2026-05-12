#!/usr/bin/env python3
"""Generate MCP tool reference documentation from tool_schemas.py.

Usage:
    python scripts/gen_mcp_docs.py              # write to docs/api/mcp-tools.md
    python scripts/gen_mcp_docs.py --check      # check if docs are up-to-date (exit 1 if stale)
    python scripts/gen_mcp_docs.py --stdout      # print to stdout instead of file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neural_memory.mcp.tool_schemas import get_tool_schemas  # noqa: E402

# ── Category mapping ─────────────────────────────────────────

TOOL_CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "core": (
        "Core Memory",
        [
            "nmem_remember",
            "nmem_remember_batch",
            "nmem_recall",
            "nmem_show",
            "nmem_context",
            "nmem_todo",
            "nmem_auto",
            "nmem_suggest",
        ],
    ),
    "session": (
        "Session & Context",
        ["nmem_session", "nmem_eternal", "nmem_recap"],
    ),
    "provenance": (
        "Provenance & Sources",
        ["nmem_provenance", "nmem_source"],
    ),
    "analytics": (
        "Analytics & Health",
        ["nmem_stats", "nmem_health", "nmem_evolution", "nmem_habits", "nmem_narrative"],
    ),
    "cognitive": (
        "Cognitive Reasoning",
        [
            "nmem_hypothesize",
            "nmem_evidence",
            "nmem_predict",
            "nmem_verify",
            "nmem_cognitive",
            "nmem_gaps",
            "nmem_schema",
            "nmem_explain",
        ],
    ),
    "training": (
        "Training & Import",
        ["nmem_train", "nmem_train_db", "nmem_index", "nmem_import"],
    ),
    "management": (
        "Memory Management",
        [
            "nmem_edit",
            "nmem_forget",
            "nmem_pin",
            "nmem_consolidate",
            "nmem_drift",
            "nmem_review",
            "nmem_alerts",
        ],
    ),
    "sync": (
        "Cloud Sync & Backup",
        ["nmem_sync", "nmem_sync_status", "nmem_sync_config", "nmem_telegram_backup"],
    ),
    "meta": (
        "Versioning & Transfer",
        ["nmem_version", "nmem_transplant", "nmem_conflicts"],
    ),
}

# ── Helpers ───────────────────────────────────────────────────


def _schema_by_name(schemas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in schemas}


def _format_type(prop: dict[str, Any]) -> str:
    """Format a JSON Schema property type for display."""
    t = prop.get("type", "any")
    if t == "array":
        items = prop.get("items", {})
        item_type = items.get("type", "any") if isinstance(items, dict) else "any"
        return f"array[{item_type}]"
    if "enum" in prop:
        values = ", ".join(f"`{v}`" for v in prop["enum"])
        return f"{t} ({values})"
    return str(t)


def _format_default(prop: dict[str, Any]) -> str:
    """Extract default value from description or schema."""
    default = prop.get("default")
    if default is not None:
        return f"`{default}`"
    # Try to extract from description
    desc = prop.get("description", "")
    if "(default:" in desc.lower():
        start = desc.lower().index("(default:")
        end = desc.index(")", start) if ")" in desc[start:] else len(desc)
        return desc[start + 1 : end].strip()
    return "—"


def _generate_tool_section(tool: dict[str, Any]) -> str:
    """Generate markdown for a single tool."""
    name = tool["name"]
    desc = tool.get("description", "No description.")
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [f"### `{name}`", "", desc, ""]

    if props:
        lines.append("| Parameter | Type | Required | Default | Description |")
        lines.append("|-----------|------|----------|---------|-------------|")

        for pname, pschema in props.items():
            ptype = _format_type(pschema)
            is_req = "Yes" if pname in required else "No"
            default = _format_default(pschema)
            pdesc = pschema.get("description", "—")
            # Escape pipes in description
            pdesc = pdesc.replace("|", "\\|").replace("\n", " ")
            # Truncate long descriptions
            if len(pdesc) > 120:
                pdesc = pdesc[:117] + "..."
            lines.append(f"| `{pname}` | {ptype} | {is_req} | {default} | {pdesc} |")

        lines.append("")
    else:
        lines.append("*No parameters.*")
        lines.append("")

    return "\n".join(lines)


# ── Main generator ────────────────────────────────────────────


def generate() -> str:
    """Generate the full MCP tools reference markdown."""
    schemas = get_tool_schemas()
    by_name = _schema_by_name(schemas)

    lines = [
        "# MCP Tools Reference",
        "",
        "Complete reference for all NeuralMemory MCP tools.",
        f"**{len(schemas)} tools** available via MCP stdio transport.",
        "",
        "!!! tip",
        "    Tools are called as MCP tool calls, not CLI commands. In Claude Code, call `nmem_recall` directly — do not run `nmem recall` in terminal.",
        "",
        "## Table of Contents",
        "",
    ]

    # TOC
    categorized_names: set[str] = set()
    for cat_key, (cat_label, tool_names) in TOOL_CATEGORIES.items():
        lines.append(f"- [{cat_label}](#{cat_key})")
        for tn in tool_names:
            if tn in by_name:
                lines.append(f"  - [`{tn}`](#{tn.replace('_', '_')})")
                categorized_names.add(tn)

    # Check for uncategorized tools
    uncategorized = [s["name"] for s in schemas if s["name"] not in categorized_names]
    if uncategorized:
        lines.append(f"- [Other](#other)")
        for tn in uncategorized:
            lines.append(f"  - [`{tn}`](#{tn})")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-category sections
    for cat_key, (cat_label, tool_names) in TOOL_CATEGORIES.items():
        lines.append(f"## {cat_label} {{#{cat_key}}}")
        lines.append("")

        for tn in tool_names:
            tool = by_name.get(tn)
            if tool:
                lines.append(_generate_tool_section(tool))

    # Uncategorized tools
    if uncategorized:
        lines.append("## Other {#other}")
        lines.append("")
        for tn in uncategorized:
            tool = by_name.get(tn)
            if tool:
                lines.append(_generate_tool_section(tool))

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Auto-generated by `scripts/gen_mcp_docs.py` from `tool_schemas.py` — {len(schemas)} tools.*"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCP tool reference docs")
    parser.add_argument("--check", action="store_true", help="Check if docs are up-to-date")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout")
    args = parser.parse_args()

    content = generate()
    output_path = ROOT / "docs" / "api" / "mcp-tools.md"

    if args.stdout:
        print(content)
        return

    if args.check:
        if output_path.exists():
            existing = output_path.read_text(encoding="utf-8")
            if existing == content:
                print(f"  docs/api/mcp-tools.md is up-to-date ({len(content)} chars)")
                return
            else:
                print(
                    f"  docs/api/mcp-tools.md is STALE — regenerate with: python scripts/gen_mcp_docs.py"
                )
                sys.exit(1)
        else:
            print(
                f"  docs/api/mcp-tools.md does not exist — generate with: python scripts/gen_mcp_docs.py"
            )
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Generated {output_path} ({len(content)} chars, {content.count('###')} tools)")


if __name__ == "__main__":
    main()
