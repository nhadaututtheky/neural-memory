"""Neural Memory CLI main entry point."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import Annotated, Optional

import typer

from neural_memory.cli.config import CLIConfig
from neural_memory.cli.storage import PersistentStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline

# Main app
app = typer.Typer(
    name="nmem",
    help="Neural Memory - Reflex-based memory for AI agents",
    no_args_is_help=True,
)

# Brain subcommand
brain_app = typer.Typer(help="Brain management commands")
app.add_typer(brain_app, name="brain")


def get_config() -> CLIConfig:
    """Get CLI configuration."""
    return CLIConfig.load()


async def get_storage(config: CLIConfig) -> PersistentStorage:
    """Get storage for current brain."""
    brain_path = config.get_brain_path()
    return await PersistentStorage.load(brain_path)


def output_result(data: dict, as_json: bool = False) -> None:
    """Output result in appropriate format."""
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        # Human-readable format
        if "error" in data:
            typer.secho(f"Error: {data['error']}", fg=typer.colors.RED)
        elif "answer" in data:
            typer.echo(data["answer"])
            if data.get("confidence"):
                typer.secho(
                    f"\n[confidence: {data['confidence']:.2f}, "
                    f"neurons: {data.get('neurons_activated', 0)}]",
                    fg=typer.colors.BRIGHT_BLACK,
                )
        elif "message" in data:
            typer.secho(data["message"], fg=typer.colors.GREEN)
        elif "context" in data:
            typer.echo(data["context"])
        else:
            typer.echo(str(data))


# =============================================================================
# Core Commands
# =============================================================================


@app.command()
def remember(
    content: Annotated[str, typer.Argument(help="Content to remember")],
    tags: Annotated[
        Optional[list[str]], typer.Option("--tag", "-t", help="Tags for the memory")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Store a new memory.

    Examples:
        nmem remember "Fixed auth bug by adding null check"
        nmem remember "Meeting with Alice about API design" -t meeting -t alice
    """

    async def _remember() -> dict:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        encoder = MemoryEncoder(storage, brain.config)

        # Disable auto-save for batch operations during encoding
        storage.disable_auto_save()

        result = await encoder.encode(
            content=content,
            timestamp=datetime.now(),
            tags=set(tags) if tags else None,
        )

        # Save once after encoding
        await storage.batch_save()

        return {
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
            "fiber_id": result.fiber.id,
            "neurons_created": len(result.neurons_created),
            "neurons_linked": len(result.neurons_linked),
            "synapses_created": len(result.synapses_created),
        }

    result = asyncio.run(_remember())
    output_result(result, json_output)


@app.command()
def recall(
    query: Annotated[str, typer.Argument(help="Query to search memories")],
    depth: Annotated[
        Optional[int],
        typer.Option("--depth", "-d", help="Search depth (0=instant, 1=context, 2=habit, 3=deep)"),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", "-m", help="Max tokens in response")
    ] = 500,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Query memories.

    Examples:
        nmem recall "What did I do with auth?"
        nmem recall "meetings with Alice" --depth 2
        nmem recall "project status" --json
    """

    async def _recall() -> dict:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        pipeline = ReflexPipeline(storage, brain.config)

        depth_level = DepthLevel(depth) if depth is not None else None

        result = await pipeline.query(
            query=query,
            depth=depth_level,
            max_tokens=max_tokens,
            reference_time=datetime.now(),
        )

        return {
            "answer": result.context or "No relevant memories found.",
            "confidence": result.confidence,
            "depth_used": result.depth_used.value,
            "neurons_activated": result.neurons_activated,
            "fibers_matched": result.fibers_matched,
            "latency_ms": result.latency_ms,
        }

    result = asyncio.run(_recall())
    output_result(result, json_output)


@app.command()
def context(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Number of recent memories")
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Get recent context (for injecting into AI conversations).

    Examples:
        nmem context
        nmem context --limit 5 --json
    """

    async def _context() -> dict:
        config = get_config()
        storage = await get_storage(config)

        # Get recent fibers
        fibers = await storage.get_fibers(limit=limit)

        if not fibers:
            return {"context": "No memories stored yet.", "count": 0}

        # Build context string
        context_parts = []
        for fiber in fibers:
            if fiber.summary:
                context_parts.append(f"- {fiber.summary}")
            elif fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    context_parts.append(f"- {anchor.content}")

        context_str = "\n".join(context_parts) if context_parts else "No context available."

        return {
            "context": context_str,
            "count": len(fibers),
            "fibers": [
                {
                    "id": f.id,
                    "summary": f.summary,
                    "created_at": f.created_at.isoformat(),
                }
                for f in fibers
            ],
        }

    result = asyncio.run(_context())
    output_result(result, json_output)


@app.command()
def stats(
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Show brain statistics.

    Examples:
        nmem stats
        nmem stats --json
    """

    async def _stats() -> dict:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        stats_data = await storage.get_stats(brain.id)

        return {
            "brain": brain.name,
            "brain_id": brain.id,
            "neuron_count": stats_data["neuron_count"],
            "synapse_count": stats_data["synapse_count"],
            "fiber_count": stats_data["fiber_count"],
            "created_at": brain.created_at.isoformat(),
        }

    result = asyncio.run(_stats())

    if json_output:
        output_result(result, True)
    else:
        typer.echo(f"Brain: {result['brain']}")
        typer.echo(f"Neurons: {result['neuron_count']}")
        typer.echo(f"Synapses: {result['synapse_count']}")
        typer.echo(f"Fibers (memories): {result['fiber_count']}")


# =============================================================================
# Brain Management Commands
# =============================================================================


@brain_app.command("list")
def brain_list(
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """List available brains.

    Examples:
        nmem brain list
        nmem brain list --json
    """
    config = get_config()
    brains = config.list_brains()
    current = config.current_brain

    if json_output:
        output_result({"brains": brains, "current": current}, True)
    else:
        if not brains:
            typer.echo("No brains found. Create one with: nmem brain create <name>")
            return

        typer.echo("Available brains:")
        for brain in brains:
            marker = " *" if brain == current else ""
            typer.echo(f"  {brain}{marker}")


@brain_app.command("use")
def brain_use(
    name: Annotated[str, typer.Argument(help="Brain name to switch to")],
) -> None:
    """Switch to a different brain.

    Examples:
        nmem brain use work
        nmem brain use personal
    """
    config = get_config()

    if name not in config.list_brains():
        typer.secho(f"Brain '{name}' not found. Create it with: nmem brain create {name}", fg=typer.colors.RED)
        raise typer.Exit(1)

    config.current_brain = name
    config.save()
    typer.secho(f"Switched to brain: {name}", fg=typer.colors.GREEN)


@brain_app.command("create")
def brain_create(
    name: Annotated[str, typer.Argument(help="Name for the new brain")],
    use: Annotated[
        bool, typer.Option("--use", "-u", help="Switch to the new brain after creating")
    ] = True,
) -> None:
    """Create a new brain.

    Examples:
        nmem brain create work
        nmem brain create personal --no-use
    """

    async def _create() -> None:
        config = get_config()

        if name in config.list_brains():
            typer.secho(f"Brain '{name}' already exists.", fg=typer.colors.RED)
            raise typer.Exit(1)

        # Create new brain by loading storage (which creates if not exists)
        brain_path = config.get_brain_path(name)
        await PersistentStorage.load(brain_path)

        if use:
            config.current_brain = name
            config.save()

        typer.secho(f"Created brain: {name}", fg=typer.colors.GREEN)
        if use:
            typer.echo(f"Now using: {name}")

    asyncio.run(_create())


@brain_app.command("export")
def brain_export(
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output file path")
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Brain name (default: current)")
    ] = None,
) -> None:
    """Export brain to JSON file.

    Examples:
        nmem brain export
        nmem brain export -o backup.json
        nmem brain export --name work -o work-backup.json
    """

    async def _export() -> None:
        config = get_config()
        brain_name = name or config.current_brain
        brain_path = config.get_brain_path(brain_name)

        if not brain_path.exists():
            typer.secho(f"Brain '{brain_name}' not found.", fg=typer.colors.RED)
            raise typer.Exit(1)

        storage = await PersistentStorage.load(brain_path)
        snapshot = await storage.export_brain(storage._current_brain_id)

        export_data = {
            "brain_id": snapshot.brain_id,
            "brain_name": snapshot.brain_name,
            "exported_at": snapshot.exported_at.isoformat(),
            "version": snapshot.version,
            "neurons": snapshot.neurons,
            "synapses": snapshot.synapses,
            "fibers": snapshot.fibers,
            "config": snapshot.config,
            "metadata": snapshot.metadata,
        }

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            typer.secho(f"Exported to: {output}", fg=typer.colors.GREEN)
        else:
            typer.echo(json.dumps(export_data, indent=2, default=str))

    asyncio.run(_export())


@brain_app.command("import")
def brain_import(
    file: Annotated[str, typer.Argument(help="JSON file to import")],
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Name for imported brain")
    ] = None,
    use: Annotated[
        bool, typer.Option("--use", "-u", help="Switch to imported brain")
    ] = True,
) -> None:
    """Import brain from JSON file.

    Examples:
        nmem brain import backup.json
        nmem brain import shared-brain.json --name shared
    """
    from neural_memory.core.brain import BrainSnapshot

    async def _import() -> None:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        brain_name = name or data.get("brain_name", "imported")
        config = get_config()

        if brain_name in config.list_brains():
            typer.secho(f"Brain '{brain_name}' already exists. Use --name to specify different name.", fg=typer.colors.RED)
            raise typer.Exit(1)

        # Create snapshot
        snapshot = BrainSnapshot(
            brain_id=data["brain_id"],
            brain_name=brain_name,
            exported_at=datetime.fromisoformat(data["exported_at"]),
            version=data["version"],
            neurons=data["neurons"],
            synapses=data["synapses"],
            fibers=data["fibers"],
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )

        # Load/create storage and import
        brain_path = config.get_brain_path(brain_name)
        storage = await PersistentStorage.load(brain_path)
        await storage.import_brain(snapshot, storage._current_brain_id)
        await storage.save()

        if use:
            config.current_brain = brain_name
            config.save()

        typer.secho(f"Imported brain: {brain_name}", fg=typer.colors.GREEN)
        typer.echo(f"  Neurons: {len(data['neurons'])}")
        typer.echo(f"  Synapses: {len(data['synapses'])}")
        typer.echo(f"  Fibers: {len(data['fibers'])}")

    asyncio.run(_import())


@brain_app.command("delete")
def brain_delete(
    name: Annotated[str, typer.Argument(help="Brain name to delete")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
) -> None:
    """Delete a brain.

    Examples:
        nmem brain delete old-brain
        nmem brain delete temp --force
    """
    config = get_config()

    if name not in config.list_brains():
        typer.secho(f"Brain '{name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if name == config.current_brain:
        typer.secho("Cannot delete current brain. Switch to another brain first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete brain '{name}'? This cannot be undone.")
        if not confirm:
            typer.echo("Cancelled.")
            return

    brain_path = config.get_brain_path(name)
    brain_path.unlink()
    typer.secho(f"Deleted brain: {name}", fg=typer.colors.GREEN)


# =============================================================================
# Utility Commands
# =============================================================================


@app.command()
def version() -> None:
    """Show version information."""
    from neural_memory import __version__

    typer.echo(f"neural-memory v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
