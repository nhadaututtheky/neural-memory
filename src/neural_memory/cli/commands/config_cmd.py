"""CLI commands for configuration management."""

from __future__ import annotations

from typing import Annotated

import typer

config_app = typer.Typer(help="Configuration management")


@config_app.command("preset")
def preset_cmd(
    name: Annotated[
        str,
        typer.Argument(help="Preset name: safe-cost, balanced, max-recall"),
    ] = "",
    list_available: Annotated[
        bool,
        typer.Option("--list", "-l", help="List available presets"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show changes without applying"),
    ] = False,
) -> None:
    """Apply a configuration preset or list available presets.

    Examples:
        nmem config preset --list
        nmem config preset balanced
        nmem config preset max-recall --dry-run
        nmem config preset safe-cost
    """
    from neural_memory.config_presets import (
        apply_preset,
        compute_diff,
        get_preset,
        list_presets,
    )
    from neural_memory.unified_config import UnifiedConfig

    if list_available or not name:
        presets = list_presets()
        typer.echo("Available presets:\n")
        for p in presets:
            typer.secho(f"  {p['name']}", fg=typer.colors.CYAN, bold=True, nl=False)
            typer.echo(f"  — {p['description']}")
        typer.echo("\nUsage: nmem config preset <name>")
        return

    preset = get_preset(name)
    if preset is None:
        typer.secho(f"Unknown preset: {name}", fg=typer.colors.RED)
        typer.echo("Use --list to see available presets.")
        raise typer.Exit(1)

    config = UnifiedConfig.load()

    if dry_run:
        changes = compute_diff(config, preset)
        if not changes:
            typer.echo(f"Preset '{name}' matches current config — no changes needed.")
            return

        typer.echo(f"Preset '{name}' would change:\n")
        for change in changes:
            typer.echo(
                f"  [{change['section']}] {change['key']}: {change['current']} -> {change['new']}"
            )
        typer.echo("\nRun without --dry-run to apply.")
        return

    changes = apply_preset(config, preset)
    config.save()

    if not changes:
        typer.echo(f"Preset '{name}' matches current config — no changes needed.")
        return

    typer.secho(f"Applied preset '{name}':", fg=typer.colors.GREEN)
    for change in changes:
        typer.echo(
            f"  [{change['section']}] {change['key']}: {change['current']} -> {change['new']}"
        )


@config_app.command("tier")
def tier_cmd(
    name: Annotated[
        str,
        typer.Argument(help="Tier name: minimal, standard, full"),
    ] = "",
    show: Annotated[
        bool,
        typer.Option("--show", "-s", help="Show current tier"),
    ] = False,
) -> None:
    """Get or set the MCP tool tier to control token usage.

    Tiers control which tools are exposed in tools/list:
      minimal  — 4 core tools (~84% token savings)
      standard — 8 tools (~69% savings)
      full     — all 45 tools (default)

    Hidden tools remain callable — only schema exposure changes.

    Examples:
        nmem config tier --show
        nmem config tier standard
        nmem config tier full
    """
    from neural_memory.unified_config import (
        _VALID_TOOL_TIERS,
        ToolTierConfig,
        UnifiedConfig,
    )

    config = UnifiedConfig.load()

    if show or not name:
        typer.echo(f"Current tool tier: {config.tool_tier.tier}")
        typer.echo(f"Valid tiers: {', '.join(sorted(_VALID_TOOL_TIERS))}")
        return

    tier_value = name.lower().strip()
    if tier_value not in _VALID_TOOL_TIERS:
        typer.secho(f"Unknown tier: {name}", fg=typer.colors.RED)
        typer.echo(f"Valid tiers: {', '.join(sorted(_VALID_TOOL_TIERS))}")
        raise typer.Exit(1)

    config.tool_tier = ToolTierConfig(tier=tier_value)
    config.save()
    typer.secho(f"Tool tier set to: {tier_value}", fg=typer.colors.GREEN)
