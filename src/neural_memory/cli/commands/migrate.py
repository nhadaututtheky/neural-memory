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
        typer.Argument(help="Target backend: 'postgres', 'infinitydb', or 'sqlite'"),
    ],
    brain: Annotated[
        str | None,
        typer.Option("--brain", "-b", help="Specific brain to migrate (default: current)"),
    ] = None,
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
        nmem migrate postgres --brain default --pg-host localhost --pg-database neuralmem
        nmem migrate infinitydb --brain my-brain.v2
    """
    supported = ("postgres", "infinitydb", "sqlite")
    if target not in supported:
        typer.secho(f"Unknown target backend: {target}", fg=typer.colors.RED)
        typer.echo(f"Supported targets: {', '.join(supported)}")
        raise typer.Exit(1)

    if target == "postgres":
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
    """Run the SQLite -> InfinityDB migration (Pro feature).

    Critical: the InfinityDB instance is opened with the SAME `(base_dir,
    brain_id)` pair that the runtime (`unified_config._build_storage`) uses
    when `storage_backend = "infinitydb"` — i.e. `InfinityDB(brains_dir,
    brain_id=name)`. This guarantees `nmem health` after `storage switch`
    reads from exactly the directory we wrote to. A previous version passed
    `brain_dir` as `base_dir` instead, which placed data at
    `brains/<name>/default/brain.inf` while the runtime read from
    `brains/<name>/brain.inf` — silent data loss reported in issue #147.
    """
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

    # Lazy-import the migrator so users without Pro deps installed get a
    # clean error instead of an ImportError trace.
    try:
        from neural_memory.pro.infinitydb.engine import InfinityDB
        from neural_memory.pro.infinitydb.migrator import (
            SQLiteToInfinityMigrator,
            estimate_migration,
        )
    except ImportError as exc:
        typer.secho(
            f"InfinityDB engine not available: {exc}. Run: pip install 'neural-memory[pro]'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from exc

    typer.secho(f"Migrating brain '{name}' from SQLite -> InfinityDB", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {brains_dir / name}/")

    # Show source counts via direct SQLite probe (no SQLiteStorage adapter
    # needed — the migrator opens its own connection, and probing the actual
    # SQLite file avoids the broken `SQLiteStorage.list_brains()` lookup).
    estimate = await estimate_migration(db_path)
    n_neurons = estimate.get("neurons_count", 0)
    n_synapses = estimate.get("synapses_count", 0)
    n_fibers = estimate.get("fibers_count", 0)
    typer.echo(f"  Neurons: {n_neurons}, Synapses: {n_synapses}, Fibers: {n_fibers}")

    # Open InfinityDB at the runtime path. Mirror exactly how
    # InfinityDBStorage(base_dir=brains_dir, brain_id=name).open() invokes
    # the engine — same base_dir, same brain_id — so a subsequent
    # `nmem storage switch infinitydb` reads from this exact directory.
    db = InfinityDB(brains_dir, brain_id=name)
    await db.open()
    try:
        typer.echo("  Migrating data...")
        migrator = SQLiteToInfinityMigrator(db_path, db)
        stats = await migrator.migrate()
        await db.flush()
    finally:
        await db.close()

    typer.echo(
        f"  Migrated: {stats.neurons_migrated} neurons, "
        f"{stats.synapses_migrated} synapses, "
        f"{stats.fibers_migrated} fibers "
        f"({stats.elapsed_seconds:.2f}s)"
    )
    if stats.neurons_skipped or stats.synapses_skipped or stats.fibers_skipped:
        typer.secho(
            f"  Skipped: {stats.neurons_skipped} neurons, "
            f"{stats.synapses_skipped} synapses, "
            f"{stats.fibers_skipped} fibers",
            fg=typer.colors.YELLOW,
        )
    if stats.errors:
        typer.secho("  Errors during migration:", fg=typer.colors.YELLOW)
        for err in stats.errors[:5]:
            typer.echo(f"    - {err}")
        if len(stats.errors) > 5:
            typer.echo(f"    ... and {len(stats.errors) - 5} more")

    # Verify by reopening at the runtime path and reading the count back.
    # This is the exact path `nmem storage switch infinitydb` will use,
    # so a mismatch here means the user will see 0 memories after switch.
    db_verify = InfinityDB(brains_dir, brain_id=name)
    await db_verify.open()
    try:
        verify_stats = await db_verify.get_stats()
        v_neurons = verify_stats.get("neuron_count", 0)
        typer.echo(f"  Verified: {v_neurons} neurons readable from runtime path")
        if v_neurons != stats.neurons_migrated:
            typer.secho(
                f"  WARNING: round-trip mismatch — wrote {stats.neurons_migrated}, "
                f"read back {v_neurons}",
                fg=typer.colors.YELLOW,
            )
    finally:
        await db_verify.close()

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
