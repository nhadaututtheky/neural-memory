"""CLI commands for managing storage backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

storage_app = typer.Typer(
    name="storage",
    help="Manage storage backend (SQLite, InfinityDB, FalkorDB, PostgreSQL)",
    no_args_is_help=True,
)


@storage_app.command("status")
def storage_status() -> None:
    """Show current storage backend status.

    Displays which backend is active, Pro license status,
    and whether data files exist for each backend.

    Examples:
        nmem storage status
    """
    from neural_memory.plugins import has_pro
    from neural_memory.unified_config import get_config

    cfg = get_config(reload=True)
    brain_name = cfg.current_brain
    brains_dir = Path(cfg.data_dir) / "brains"

    # Check data file existence
    sqlite_path = brains_dir / f"{brain_name}.db"
    sqlite_exists = sqlite_path.exists()
    sqlite_size = sqlite_path.stat().st_size if sqlite_exists else 0

    infinity_marker = brains_dir / brain_name / "brain.inf"
    infinitydb_exists = infinity_marker.exists()

    pro_installed = has_pro()
    is_pro = cfg.is_pro()

    # Display
    typer.secho("Storage Status", bold=True)
    typer.echo(f"  Brain:           {brain_name}")
    typer.echo(f"  Active backend:  {cfg.storage_backend}")
    typer.echo(f"  Pro installed:   {'yes' if pro_installed else 'no'}")
    typer.echo(f"  Pro license:     {'active' if is_pro else 'inactive'}")
    typer.echo()

    typer.secho("Data Files", bold=True)
    if sqlite_exists:
        size_mb = sqlite_size / (1024 * 1024)
        typer.echo(f"  SQLite:          {sqlite_path.name} ({size_mb:.1f} MB)")
    else:
        typer.echo("  SQLite:          (no data)")
    if infinitydb_exists:
        typer.echo(f"  InfinityDB:      {brain_name}/brain.inf (exists)")
    else:
        typer.echo("  InfinityDB:      (no data)")
    typer.echo()

    # Actionable guidance
    if is_pro and pro_installed and cfg.storage_backend == "sqlite":
        typer.secho("Tip", bold=True)
        typer.echo("  Pro is active — you can upgrade to InfinityDB:")
        if sqlite_exists and not infinitydb_exists:
            typer.echo("    1. nmem migrate infinitydb    (migrate data)")
            typer.echo("    2. nmem storage switch infinitydb")
        elif infinitydb_exists:
            typer.echo("    nmem storage switch infinitydb")
        typer.echo()
    elif is_pro and not pro_installed:
        typer.secho("Tip", bold=True)
        typer.echo("  Pro license active but Pro deps missing.")
        typer.echo("    pip install neural-memory")
        typer.echo()
    elif cfg.storage_backend == "infinitydb":
        typer.secho("InfinityDB Active", fg=typer.colors.GREEN, bold=True)
        typer.echo("  HNSW indexing, tiered compression, and cone queries are available.")
        typer.echo()


@storage_app.command("switch")
def storage_switch(
    backend: Annotated[
        str,
        typer.Argument(help="Target backend: 'sqlite' or 'infinitydb'"),
    ],
) -> None:
    """Switch active storage backend.

    Switches config.toml to use the specified backend.
    Data must already exist for the target backend (run migrate first).

    Examples:
        nmem storage switch infinitydb
        nmem storage switch sqlite
    """
    from dataclasses import replace

    from neural_memory.unified_config import get_config, set_config

    backend = backend.lower().strip()
    valid = ("sqlite", "infinitydb")
    if backend not in valid:
        typer.secho(f"Invalid backend: {backend}. Choose: {', '.join(valid)}", fg=typer.colors.RED)
        raise typer.Exit(1)

    cfg = get_config(reload=True)

    if cfg.storage_backend == backend:
        typer.secho(f"Already using {backend}.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    brain_name = cfg.current_brain
    brains_dir = Path(cfg.data_dir) / "brains"

    # Guard: InfinityDB requires Pro
    if backend == "infinitydb":
        from neural_memory.pro import is_pro_deps_installed

        if not is_pro_deps_installed():
            typer.secho(
                "Pro dependencies not installed. Run: pip install neural-memory",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        if not cfg.is_pro():
            typer.secho(
                "Pro license not active. Run: nmem shared activate --key <KEY>",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        infinity_marker = brains_dir / brain_name / "brain.inf"
        if not infinity_marker.exists():
            typer.secho(
                f"No InfinityDB data for brain '{brain_name}'. Run: nmem migrate infinitydb",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # Guard: SQLite requires data
    if backend == "sqlite":
        sqlite_path = brains_dir / f"{brain_name}.db"
        if not sqlite_path.exists():
            typer.secho(
                f"No SQLite data for brain '{brain_name}'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # Switch
    new_cfg = replace(cfg, storage_backend=backend)
    new_cfg.save()
    set_config(new_cfg)

    typer.secho(f"Switched to {backend}.", fg=typer.colors.GREEN)
    typer.echo("Restart your AI tool to use the new backend.")
