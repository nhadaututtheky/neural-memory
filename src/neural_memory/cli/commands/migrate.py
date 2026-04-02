"""CLI command for migrating between storage backends."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

import typer

logger = logging.getLogger(__name__)


def migrate(
    target: Annotated[
        str,
        typer.Argument(help="Target backend: 'falkordb', 'postgres', 'infinitydb', or 'sqlite'"),
    ],
    brain: Annotated[
        str | None,
        typer.Option("--brain", "-b", help="Specific brain to migrate (default: current)"),
    ] = None,
    falkordb_host: Annotated[
        str,
        typer.Option("--falkordb-host", help="FalkorDB host"),
    ] = "localhost",
    falkordb_port: Annotated[
        int,
        typer.Option("--falkordb-port", help="FalkorDB port"),
    ] = 6379,
    pg_host: Annotated[
        str,
        typer.Option("--pg-host", help="PostgreSQL host"),
    ] = "localhost",
    pg_port: Annotated[
        int,
        typer.Option("--pg-port", help="PostgreSQL port"),
    ] = 5432,
    pg_database: Annotated[
        str,
        typer.Option("--pg-database", help="PostgreSQL database name"),
    ] = "neuralmemory",
    pg_user: Annotated[
        str,
        typer.Option("--pg-user", help="PostgreSQL user"),
    ] = "postgres",
    pg_password: Annotated[
        str,
        typer.Option(
            "--pg-password", help="PostgreSQL password (or use NEURAL_MEMORY_POSTGRES_PASSWORD env)"
        ),
    ] = "",
) -> None:
    """Migrate brain data between storage backends.

    Examples:
        nmem migrate falkordb --brain default
        nmem migrate postgres --brain default --pg-host localhost --pg-database neuralmem
        nmem migrate infinitydb --brain my-brain.v2
    """
    supported = ("falkordb", "postgres", "infinitydb", "sqlite")
    if target not in supported:
        typer.secho(f"Unknown target backend: {target}", fg=typer.colors.RED)
        typer.echo(f"Supported targets: {', '.join(supported)}")
        raise typer.Exit(1)

    if target == "falkordb":
        asyncio.run(
            _migrate_to_falkordb(
                brain_name=brain,
                host=falkordb_host,
                port=falkordb_port,
            )
        )
    elif target == "postgres":
        asyncio.run(
            _migrate_to_postgres(
                brain_name=brain,
                host=pg_host,
                port=pg_port,
                database=pg_database,
                user=pg_user,
                password=pg_password,
            )
        )
    elif target == "infinitydb":
        asyncio.run(_migrate_to_infinitydb(brain_name=brain))
    else:
        typer.secho("SQLite -> SQLite migration not needed.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)


async def _migrate_to_falkordb(
    brain_name: str | None,
    host: str,
    port: int,
) -> None:
    """Run the SQLite -> FalkorDB migration."""
    from neural_memory.storage.falkordb.falkordb_migration import (
        migrate_sqlite_to_falkordb,
    )
    from neural_memory.unified_config import get_config

    config = get_config()
    name = brain_name or config.current_brain
    db_path = str(config.get_brain_db_path(name))

    typer.secho(f"Migrating brain '{name}' from SQLite -> FalkorDB", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {host}:{port}")

    result = await migrate_sqlite_to_falkordb(
        sqlite_db_path=db_path,
        falkordb_host=host,
        falkordb_port=port,
        brain_name=name,
    )

    _print_result(result)


async def _migrate_to_postgres(
    brain_name: str | None,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> None:
    """Run the SQLite -> PostgreSQL migration."""
    import os

    from neural_memory.storage.postgres.postgres_migration import (
        migrate_sqlite_to_postgres,
    )
    from neural_memory.unified_config import get_config

    config = get_config()
    name = brain_name or config.current_brain
    db_path = str(config.get_brain_db_path(name))

    # Allow env var override for password
    effective_password = password or os.environ.get("NEURAL_MEMORY_POSTGRES_PASSWORD", "")

    typer.secho(f"Migrating brain '{name}' from SQLite -> PostgreSQL", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {user}@{host}:{port}/{database}")

    result = await migrate_sqlite_to_postgres(
        sqlite_db_path=db_path,
        pg_host=host,
        pg_port=port,
        pg_database=database,
        pg_user=user,
        pg_password=effective_password,
        brain_name=name,
    )

    _print_result(result)


async def _migrate_to_infinitydb(brain_name: str | None) -> None:
    """Run the SQLite -> InfinityDB migration (Pro feature)."""
    from pathlib import Path

    from neural_memory.pro import is_pro_deps_installed
    from neural_memory.unified_config import get_config

    config = get_config()

    # Pre-flight: Pro checks
    if not is_pro_deps_installed():
        typer.secho(
            "Pro dependencies not installed. Run: pip install neural-memory",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if not config.is_pro():
        typer.secho(
            "Pro license not active. Run: nmem shared activate --key <KEY>",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    name = brain_name or config.current_brain
    brains_dir = Path(config.data_dir) / "brains"
    db_path = brains_dir / f"{name}.db"

    if not db_path.exists():
        typer.secho(f"No SQLite data for brain '{name}': {db_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"Migrating brain '{name}' from SQLite -> InfinityDB", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {brains_dir / name}/")

    # Open source SQLite
    from neural_memory.storage.sqlite import SQLiteStorage

    source = SQLiteStorage(str(db_path))
    await source.initialize()

    # Find brain
    brain_list = await source.list_brains()
    brain_id: str | None = None
    for b in brain_list:
        if b.get("name") == name:
            brain_id = b.get("id") or b.get("name")
            break
    if not brain_id and brain_list:
        brain_id = brain_list[0].get("id") or brain_list[0].get("name") or name
    if not brain_id:
        typer.secho(f"No brain '{name}' found in SQLite database.", fg=typer.colors.RED)
        raise typer.Exit(1)

    source.set_brain(brain_id)

    # Count source
    stats = await source.get_stats(brain_id)
    n_neurons = stats.get("neuron_count", 0)
    n_synapses = stats.get("synapse_count", 0)
    n_fibers = stats.get("fiber_count", 0)
    typer.echo(f"  Neurons: {n_neurons}, Synapses: {n_synapses}, Fibers: {n_fibers}")

    # Export from source
    typer.echo("  Exporting from SQLite...")
    snapshot = await source.export_brain(brain_id)

    # Open target InfinityDB
    try:
        from neural_memory.pro.storage_adapter import InfinityDBStorage

        storage_cls: type = InfinityDBStorage
    except ImportError:
        typer.secho(
            "InfinityDB not available. Run: pip install neural-memory",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from None

    brain_dir = brains_dir / name
    brain_dir.mkdir(parents=True, exist_ok=True)
    target = storage_cls(str(brain_dir))
    await target.initialize()

    # Import into target
    typer.echo("  Importing into InfinityDB...")
    await target.import_brain(snapshot)

    # Verify
    target_brain_list = await target.list_brains()
    if target_brain_list:
        target_brain_id = target_brain_list[0].get("id") or target_brain_list[0].get("name")
        target.set_brain(target_brain_id)
        target_stats = await target.get_stats(target_brain_id)
        t_neurons = target_stats.get("neuron_count", 0)
        typer.echo(f"  Verified: {t_neurons} neurons in InfinityDB")

        if n_neurons > 0 and abs(t_neurons - n_neurons) / n_neurons > 0.005:
            typer.secho(
                f"  WARNING: count mismatch — source {n_neurons}, target {t_neurons}",
                fg=typer.colors.YELLOW,
            )

    typer.secho("Migration complete!", fg=typer.colors.GREEN)
    typer.echo("  Next: nmem storage switch infinitydb")


def _print_result(result: dict[str, Any]) -> None:
    """Print migration result."""
    if result.get("success"):
        for brain_info in result.get("brains", []):
            typer.echo(
                f"  {brain_info['name']}: "
                f"{brain_info['neurons']} neurons, "
                f"{brain_info['synapses']} synapses, "
                f"{brain_info['fibers']} fibers"
            )
        typer.secho("Migration complete!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Migration failed: {result.get('error')}", fg=typer.colors.RED)
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Register migrate command with app."""
    app.command()(migrate)
