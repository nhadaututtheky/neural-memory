#!/usr/bin/env python3
"""Generate CLI reference documentation from Typer app introspection.

Usage:
    python scripts/gen_cli_docs.py              # write to docs/getting-started/cli-reference.md
    python scripts/gen_cli_docs.py --check      # check if docs are up-to-date (exit 1 if stale)
    python scripts/gen_cli_docs.py --stdout      # print to stdout instead of file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import click
import typer.main

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neural_memory.cli.main import app  # noqa: E402

# ── Category mapping ─────────────────────────────────────────

CLI_CATEGORIES: list[tuple[str, str, list[str]]] = [
    (
        "memory",
        "Memory Operations",
        ["remember", "recall", "context", "todo", "q", "a", "last", "today"],
    ),
    (
        "brain",
        "Brain Management",
        [
            "brain list",
            "brain use",
            "brain create",
            "brain export",
            "brain import",
            "brain delete",
            "brain health",
            "brain transplant",
        ],
    ),
    (
        "info",
        "Information & Diagnostics",
        ["stats", "status", "health", "check", "doctor", "dashboard", "ui", "graph"],
    ),
    (
        "training",
        "Training & Import/Export",
        ["train", "index", "import", "export"],
    ),
    (
        "config",
        "Configuration & Setup",
        [
            "init",
            "setup",
            "mcp-config",
            "prompt",
            "hooks",
            "config preset",
            "config tier",
            "install-skills",
        ],
    ),
    (
        "server",
        "Server & MCP",
        ["serve", "mcp"],
    ),
    (
        "maintenance",
        "Maintenance",
        ["decay", "consolidate", "cleanup", "flush"],
    ),
    (
        "project",
        "Project Management",
        ["project create", "project list", "project show", "project delete", "project extend"],
    ),
    (
        "advanced",
        "Advanced Features",
        [
            "shared enable",
            "shared disable",
            "shared status",
            "shared test",
            "shared sync",
            "habits list",
            "habits show",
            "habits clear",
            "habits status",
            "version create",
            "version list",
            "version rollback",
            "version diff",
            "telegram status",
            "telegram test",
            "telegram backup",
            "list",
            "migrate",
            "update",
        ],
    ),
]


# ── Command extraction ────────────────────────────────────────


def _is_group(cmd: Any) -> bool:
    """Return whether ``cmd`` has subcommands.

    Typer 0.16+ vendors Click, so ``isinstance(cmd, click.Group)`` is false
    for ``TyperGroup``. A group exposes ``list_commands``; a leaf does not.
    """
    return callable(getattr(cmd, "list_commands", None)) and callable(
        getattr(cmd, "get_command", None)
    )


def _is_option(param: Any) -> bool:
    """Return whether ``param`` is a flag. Works for Click and vendored Typer."""
    opts = getattr(param, "opts", ()) or ()
    return any(str(opt).startswith("-") for opt in opts)


# Click named these ``text`` and ``integer``. Typer's vendored params say
# ``str`` and ``int``. The reference keeps the Click labels.
_TYPE_LABELS = {
    "str": "text",
    "string": "text",
    "text": "text",
    "int": "integer",
    "integer": "integer",
    "float": "float",
    "boolean": "boolean",
    "bool": "boolean",
}


def _param_type_name(param: Any) -> str:
    param_type = getattr(param, "type", None)
    raw = getattr(param_type, "name", None) if param_type is not None else None
    if raw is None:
        return "TEXT"
    return _TYPE_LABELS.get(str(raw).lower(), str(raw))


def _get_click_app() -> Any:
    """Convert Typer app to a command group for introspection."""
    return typer.main.get_command(app)


def _collect_commands(group: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively collect all commands with their full names."""
    result: dict[str, Any] = {}

    if _is_group(group):
        for name in sorted(group.list_commands(click.Context(group))):
            cmd = group.get_command(click.Context(group), name)
            if cmd is None:
                continue
            full_name = f"{prefix}{name}".strip()
            if _is_group(cmd):
                # Add the group itself for reference
                result[full_name] = cmd
                # Recurse into subcommands
                result.update(_collect_commands(cmd, f"{full_name} "))
            else:
                result[full_name] = cmd

    return result


def _format_param(param: Any) -> dict[str, str]:
    """Extract display info from a Click or Typer parameter."""
    info: dict[str, str] = {"name": "", "type": "", "required": "", "default": "", "help": ""}

    if _is_option(param):
        # Build option name string
        names = " / ".join(param.opts)
        if param.secondary_opts:
            names += " / " + " / ".join(param.secondary_opts)
        info["name"] = f"`{names}`"
        info["type"] = _param_type_name(param)
        info["required"] = "Yes" if param.required else "No"
        info["default"] = f"`{param.default}`" if param.default is not None else "—"
        info["help"] = (param.help or "—").replace("|", "\\|").replace("\n", " ")
    elif getattr(param, "name", None):
        info["name"] = f"`{param.name}`"
        info["type"] = _param_type_name(param)
        info["required"] = "Yes" if param.required else "No"
        info["default"] = f"`{param.default}`" if param.default is not None else "—"
        info["help"] = "(positional argument)"

    # Truncate long help
    if len(info["help"]) > 120:
        info["help"] = info["help"][:117] + "..."

    return info


def _format_command(name: str, cmd: Any) -> str:
    """Generate markdown for a single command."""
    lines: list[str] = []
    help_text = (cmd.help or "No description.").strip()
    # Take first paragraph only
    first_para = help_text.split("\n\n")[0].replace("\n", " ").strip()

    lines.append(f"### `nmem {name}`")
    lines.append("")
    lines.append(first_para)
    lines.append("")

    # Syntax
    lines.append("```")
    lines.append(f"nmem {name} [OPTIONS]")
    lines.append("```")
    lines.append("")

    # Filter out --help and hidden params
    params = [
        p
        for p in cmd.params
        if getattr(p, "name", None) not in ("help",) and not getattr(p, "hidden", False)
    ]

    if params:
        lines.append("| Option | Type | Required | Default | Description |")
        lines.append("|--------|------|----------|---------|-------------|")

        for param in params:
            info = _format_param(param)
            if info["name"]:
                lines.append(
                    f"| {info['name']} | {info['type']} | {info['required']} | {info['default']} | {info['help']} |"
                )

        lines.append("")

    # Later paragraphs stay in the reference. Example blocks are help text,
    # not the option table, and a full dump would rewrite every command.
    for paragraph in help_text.split("\n\n")[1:]:
        body = paragraph.strip()
        if not body or body.startswith("Examples:"):
            continue
        lines.append(body)
        lines.append("")

    return "\n".join(lines)


# ── Main generator ────────────────────────────────────────────


def generate() -> str:
    """Generate the full CLI reference markdown."""
    click_app = _get_click_app()
    all_commands = _collect_commands(click_app)

    # Count leaf commands (not groups)
    leaf_count = sum(1 for c in all_commands.values() if not _is_group(c))

    lines = [
        "# CLI Reference",
        "",
        "Complete reference for the `nmem` command-line interface.",
        f"**{leaf_count} commands** available.",
        "",
        "!!! tip",
        "    Run `nmem --help` or `nmem <command> --help` for the latest usage info.",
        "",
        "## Table of Contents",
        "",
    ]

    # TOC
    categorized_names: set[str] = set()
    for cat_key, cat_label, cmd_names in CLI_CATEGORIES:
        lines.append(f"- [{cat_label}](#{cat_key})")
        for cn in cmd_names:
            if cn in all_commands and not _is_group(all_commands[cn]):
                anchor = cn.replace(" ", "-")
                lines.append(f"  - [`nmem {cn}`](#nmem-{anchor})")
                categorized_names.add(cn)

    # Uncategorized
    uncategorized = [
        n
        for n in all_commands
        if n not in categorized_names and not _is_group(all_commands[n])
    ]
    if uncategorized:
        lines.append("- [Other](#other)")
        for cn in uncategorized:
            anchor = cn.replace(" ", "-")
            lines.append(f"  - [`nmem {cn}`](#nmem-{anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-category sections
    for cat_key, cat_label, cmd_names in CLI_CATEGORIES:
        lines.append(f"## {cat_label} {{#{cat_key}}}")
        lines.append("")

        for cn in cmd_names:
            cmd = all_commands.get(cn)
            if cmd and not _is_group(cmd):
                lines.append(_format_command(cn, cmd))

    # Uncategorized
    if uncategorized:
        lines.append("## Other {#other}")
        lines.append("")
        for cn in uncategorized:
            cmd = all_commands[cn]
            if not _is_group(cmd):
                lines.append(_format_command(cn, cmd))

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Auto-generated by `scripts/gen_cli_docs.py` from Typer app introspection — {leaf_count} commands.*"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLI reference docs")
    parser.add_argument("--check", action="store_true", help="Check if docs are up-to-date")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout")
    args = parser.parse_args()

    content = generate()
    output_path = ROOT / "docs" / "getting-started" / "cli-reference.md"

    if args.stdout:
        print(content)
        return

    if args.check:
        if output_path.exists():
            existing = output_path.read_text(encoding="utf-8")
            if existing == content:
                print(
                    f"  docs/getting-started/cli-reference.md is up-to-date ({len(content)} chars)"
                )
                return
            else:
                print(
                    "  docs/getting-started/cli-reference.md is STALE — regenerate with: python scripts/gen_cli_docs.py"
                )
                sys.exit(1)
        else:
            print(
                "  docs/getting-started/cli-reference.md does not exist — generate with: python scripts/gen_cli_docs.py"
            )
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    cmd_count = content.count("### `nmem ")
    print(f"Generated {output_path} ({len(content)} chars, {cmd_count} commands)")


if __name__ == "__main__":
    main()
