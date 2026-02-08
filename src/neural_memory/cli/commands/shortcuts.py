"""Quick shortcut commands and utility generators."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, run_async
from neural_memory.core.memory_types import Priority, TypedMemory, suggest_memory_type


def quick_recall(
    query: Annotated[str, typer.Argument(help="Query to search")],
    depth: Annotated[int | None, typer.Option("-d")] = None,
) -> None:
    """Quick recall - shortcut for 'nmem recall'.

    Examples:
        nmem q "what's the API format"
        nmem q "yesterday's work" -d 2
    """
    from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline

    async def _recall() -> None:
        config = get_config()
        storage = await get_storage(config)
        brain = await storage.get_brain(storage._current_brain_id)

        if not brain:
            typer.secho("No brain configured", fg=typer.colors.RED)
            return

        pipeline = ReflexPipeline(storage, brain.config)
        depth_level = DepthLevel(depth) if depth is not None else None
        result = await pipeline.query(query, depth=depth_level, max_tokens=500)

        if result.confidence < 0.1:
            typer.secho("No relevant memories found.", fg=typer.colors.YELLOW)
            return

        typer.echo(result.context)
        typer.secho(f"\n[confidence: {result.confidence:.2f}]", fg=typer.colors.BRIGHT_BLACK)

    run_async(_recall())


def quick_add(
    content: Annotated[str, typer.Argument(help="Content to remember")],
    priority: Annotated[int | None, typer.Option("-p")] = None,
) -> None:
    """Quick add - shortcut for 'nmem remember' with auto-detect.

    Examples:
        nmem a "API key format is sk-xxx"
        nmem a "Always use UTC for timestamps" -p 8
        nmem a "TODO: Review PR #123"
    """
    from neural_memory.engine.encoder import MemoryEncoder

    async def _add() -> None:
        config = get_config()
        storage = await get_storage(config)
        brain = await storage.get_brain(storage._current_brain_id)

        if not brain:
            typer.secho("No brain configured", fg=typer.colors.RED)
            return

        # Auto-detect memory type
        mem_type = suggest_memory_type(content)
        mem_priority = Priority.from_int(priority) if priority is not None else Priority.NORMAL

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        result = await encoder.encode(
            content=content,
            timestamp=datetime.now(),
        )

        # Create typed memory
        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=mem_type,
            priority=mem_priority,
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        typer.secho(f"+ {content[:60]}{'...' if len(content) > 60 else ''}", fg=typer.colors.GREEN)
        typer.secho(f"  [{mem_type.value}]", fg=typer.colors.BRIGHT_BLACK)

    run_async(_add())


def show_last(
    count: Annotated[int, typer.Option("-n", help="Number of memories to show")] = 5,
) -> None:
    """Show last N memories - quick view of recent activity.

    Examples:
        nmem last           # Show last 5 memories
        nmem last -n 10     # Show last 10 memories
    """
    from neural_memory.safety.freshness import evaluate_freshness, format_age

    async def _last() -> None:
        config = get_config()
        storage = await get_storage(config)

        fibers = await storage.get_fibers(limit=count)

        if not fibers:
            typer.secho("No memories found.", fg=typer.colors.YELLOW)
            return

        for i, fiber in enumerate(fibers, 1):
            content = fiber.summary or ""
            if not content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content

            display = content[:70] + "..." if len(content) > 70 else content
            freshness = evaluate_freshness(fiber.created_at)

            typer.echo(f"{i}. {display}")
            typer.secho(f"   [{format_age(freshness.age_days)}]", fg=typer.colors.BRIGHT_BLACK)

    run_async(_last())


def show_today() -> None:
    """Show today's memories.

    Examples:
        nmem today
    """

    async def _today() -> None:
        config = get_config()
        storage = await get_storage(config)

        # Get recent fibers and filter for today
        fibers = await storage.get_fibers(limit=100)
        today = datetime.now().date()
        today_fibers = [f for f in fibers if f.created_at.date() == today]

        if not today_fibers:
            typer.secho("No memories from today.", fg=typer.colors.YELLOW)
            return

        typer.secho(
            f"Today ({today.strftime('%Y-%m-%d')}) - {len(today_fibers)} memories:\n",
            fg=typer.colors.CYAN,
        )

        for fiber in today_fibers:
            content = fiber.summary or ""
            if not content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content

            display = content[:65] + "..." if len(content) > 65 else content
            time_str = fiber.created_at.strftime("%H:%M")

            typer.echo(f"  {time_str}  {display}")

    run_async(_today())


def mcp_config(
    with_prompt: Annotated[
        bool, typer.Option("--with-prompt", "-p", help="Include system prompt in config")
    ] = False,
    compact_prompt: Annotated[
        bool, typer.Option("--compact", "-c", help="Use compact prompt (if --with-prompt)")
    ] = False,
) -> None:
    """Generate MCP server configuration for Claude Code/Cursor.

    Outputs JSON configuration that can be added to your MCP settings.

    Examples:
        nmem mcp-config                    # Basic config
        nmem mcp-config --with-prompt      # Include system prompt
        nmem mcp-config -p -c              # Include compact prompt
    """
    import shutil
    import sys

    from neural_memory.mcp.prompt import get_system_prompt

    # Find nmem executable path
    nmem_path = shutil.which("nmem") or shutil.which("nmem-mcp") or sys.executable

    config = {
        "neural-memory": {
            "command": nmem_path if "python" not in nmem_path.lower() else "python",
            "args": ["-m", "neural_memory.mcp"] if "python" in nmem_path.lower() else ["mcp"],
        }
    }

    # Simplify if nmem is available
    if shutil.which("nmem-mcp"):
        config["neural-memory"] = {"command": "nmem-mcp", "args": []}

    typer.echo("Add this to your MCP configuration:\n")
    typer.echo(json.dumps(config, indent=2))

    if with_prompt:
        typer.echo("\n" + "=" * 60)
        typer.echo("System prompt to add to your AI assistant:\n")
        typer.echo(get_system_prompt(compact=compact_prompt))


def prompt(
    compact: Annotated[bool, typer.Option("--compact", "-c", help="Show compact version")] = False,
    copy: Annotated[
        bool, typer.Option("--copy", help="Copy to clipboard (requires pyperclip)")
    ] = False,
) -> None:
    """Show system prompt for AI tools.

    This prompt instructs AI assistants (Claude, GPT, etc.) on when and how
    to use NeuralMemory for persistent memory across sessions.

    Examples:
        nmem prompt              # Show full prompt
        nmem prompt --compact    # Show shorter version
        nmem prompt --copy       # Copy to clipboard
    """
    from neural_memory.mcp.prompt import get_system_prompt

    text = get_system_prompt(compact=compact)

    if copy:
        try:
            import pyperclip

            pyperclip.copy(text)
            typer.echo("System prompt copied to clipboard!")
        except ImportError:
            typer.echo("Install pyperclip for clipboard support: pip install pyperclip")
            typer.echo("")
            typer.echo(text)
    else:
        typer.echo(text)


def export_brain_cmd(
    output: Annotated[str, typer.Argument(help="Output file path (e.g., my-brain.json)")],
    brain: Annotated[
        str | None, typer.Option("--brain", "-b", help="Brain to export (default: current)")
    ] = None,
) -> None:
    """Export brain to JSON file for backup or sharing.

    Examples:
        nmem export backup.json           # Export current brain
        nmem export work.json -b work     # Export specific brain
    """
    from pathlib import Path

    from neural_memory.unified_config import get_config, get_shared_storage

    async def _export() -> None:
        config = get_config()
        brain_name = brain or config.current_brain
        storage = await get_shared_storage(brain_name)

        snapshot = await storage.export_brain(brain_name)

        output_path = Path(output)
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

        output_path.write_text(json.dumps(export_data, indent=2, default=str))

        typer.echo(f"Exported brain '{brain_name}' to {output_path}")
        typer.echo(f"  Neurons: {len(snapshot.neurons)}")
        typer.echo(f"  Synapses: {len(snapshot.synapses)}")
        typer.echo(f"  Fibers: {len(snapshot.fibers)}")

    run_async(_export())


def import_brain_cmd(
    input_file: Annotated[str, typer.Argument(help="Input file path (e.g., my-brain.json)")],
    brain: Annotated[
        str | None, typer.Option("--brain", "-b", help="Target brain name (default: from file)")
    ] = None,
    merge: Annotated[bool, typer.Option("--merge", "-m", help="Merge with existing brain")] = False,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Conflict resolution: prefer_local, prefer_remote, prefer_recent, prefer_stronger",
        ),
    ] = "prefer_local",
) -> None:
    """Import brain from JSON file.

    Examples:
        nmem import backup.json                          # Import (replace)
        nmem import backup.json -b new                   # Import as 'new' brain
        nmem import backup.json --merge                  # Merge into existing brain
        nmem import backup.json --merge --strategy prefer_recent
    """
    from pathlib import Path

    from neural_memory.core.brain import BrainSnapshot
    from neural_memory.unified_config import get_shared_storage

    async def _import() -> None:
        input_path = Path(input_file)
        if not input_path.exists():
            typer.echo(f"Error: File not found: {input_path}", err=True)
            raise typer.Exit(1)

        data = json.loads(input_path.read_text())

        brain_name = brain or data.get("brain_name", "imported")
        storage = await get_shared_storage(brain_name)

        incoming_snapshot = BrainSnapshot(
            brain_id=data.get("brain_id", brain_name),
            brain_name=data["brain_name"],
            exported_at=datetime.fromisoformat(data["exported_at"]),
            version=data["version"],
            neurons=data["neurons"],
            synapses=data["synapses"],
            fibers=data["fibers"],
            config=data["config"],
            metadata=data.get("metadata", {}),
        )

        if merge:
            # Try to export existing brain for merge
            from neural_memory.engine.merge import ConflictStrategy, merge_snapshots

            try:
                local_snapshot = await storage.export_brain(brain_name)
            except (ValueError, Exception):
                # No existing brain, just import directly
                await storage.import_brain(incoming_snapshot, brain_name)
                typer.echo(
                    f"Imported brain '{brain_name}' from {input_path} (no existing brain to merge)"
                )
                return

            conflict_strategy = ConflictStrategy(strategy)
            merged_snapshot, merge_report = merge_snapshots(
                local=local_snapshot,
                incoming=incoming_snapshot,
                strategy=conflict_strategy,
            )

            # Clear and reimport merged
            await storage.clear(brain_name)
            await storage.import_brain(merged_snapshot, brain_name)

            typer.echo(f"Merged brain '{brain_name}' from {input_path}")
            typer.echo(f"  Strategy: {strategy}")
            typer.echo(merge_report.summary())
        else:
            await storage.import_brain(incoming_snapshot, brain_name)

            typer.echo(f"Imported brain '{brain_name}' from {input_path}")
            typer.echo(f"  Neurons: {len(incoming_snapshot.neurons)}")
            typer.echo(f"  Synapses: {len(incoming_snapshot.synapses)}")
            typer.echo(f"  Fibers: {len(incoming_snapshot.fibers)}")

    run_async(_import())


def register(app: typer.Typer) -> None:
    """Register shortcut commands on the app."""
    app.command(name="q")(quick_recall)
    app.command(name="a")(quick_add)
    app.command(name="last")(show_last)
    app.command(name="today")(show_today)
    app.command(name="mcp-config")(mcp_config)
    app.command()(prompt)
    app.command(name="export")(export_brain_cmd)
    app.command(name="import")(import_brain_cmd)
