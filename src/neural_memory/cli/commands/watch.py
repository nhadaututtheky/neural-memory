"""CLI command for file watcher — auto-ingest from watched directories."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async


def watch(
    directory: Annotated[
        str,
        typer.Argument(help="Directory to scan/watch"),
    ] = "",
    action: Annotated[
        str,
        typer.Option("--action", "-a", help="Action: scan, start, stop, status"),
    ] = "scan",
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Watch a directory and auto-ingest files into memory.

    Examples:
        nmem watch ~/notes              # One-shot scan
        nmem watch ~/notes -a start     # Start background watching
        nmem watch -a stop              # Stop watching
        nmem watch -a status            # Show watcher status
    """

    async def _watch() -> dict[str, Any]:
        from neural_memory.engine.doc_trainer import DocTrainer
        from neural_memory.engine.file_watcher import FileWatcher, WatchConfig
        from neural_memory.engine.watch_state import WatchStateTracker

        config = get_config()
        storage = await get_storage(config)

        brain_id: str = (
            storage.brain_id or "" if hasattr(storage, "brain_id") else config.current_brain
        )
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        if action == "stop":
            return {
                "action": "stop",
                "message": "Use MCP nmem_watch(action='stop') for background watcher",
            }

        if action == "status":
            db = getattr(storage, "_db", None)
            if db is None or not hasattr(db, "execute"):
                return {"error": "File watcher requires SQLite storage backend"}
            tracker = WatchStateTracker(db)
            stats = await tracker.get_stats()
            return {"action": "status", **stats}

        if action in ("scan", "start") and not directory:
            return {"error": "directory argument is required for scan/start action"}

        path = Path(directory).resolve()

        # Create watcher
        db = getattr(storage, "_db", None)
        if db is None or not hasattr(db, "execute"):
            return {"error": "File watcher requires SQLite storage backend"}
        tracker = WatchStateTracker(db)
        trainer = DocTrainer(storage, brain.config)

        if action == "start":
            watcher_config = WatchConfig(watch_paths=(str(path),))
            watcher = FileWatcher(trainer, tracker, watcher_config)
            try:
                watcher.start()
            except ImportError:
                return {"error": "watchdog not installed. Run: pip install neural-memory[watch]"}

            return {
                "action": "start",
                "watching": [str(path)],
                "message": "Background watcher started. Press Ctrl+C to stop.",
            }

        watcher = FileWatcher(trainer, tracker)
        error = watcher.validate_path(path)
        if error:
            return {"error": error}

        results = await watcher.process_path(path)

        processed = [r for r in results if r.success and not r.skipped]
        skipped = [r for r in results if r.skipped]
        failed = [r for r in results if not r.success]

        response: dict[str, Any] = {
            "action": "scan",
            "directory": str(path),
            "processed": len(processed),
            "skipped": len(skipped),
            "failed": len(failed),
            "total_neurons": sum(r.neurons_created for r in processed),
        }

        if not json_output:
            # Human-friendly output
            typer.echo(f"Scanned: {path}")
            typer.echo(f"  Processed: {len(processed)} files ({response['total_neurons']} neurons)")
            if skipped:
                typer.echo(f"  Skipped:   {len(skipped)} (unchanged)")
            if failed:
                typer.echo(f"  Failed:    {len(failed)}")
                for r in failed[:5]:
                    typer.secho(f"    {r.path}: {r.error}", fg=typer.colors.RED, err=True)
            return response

        return response

    result = run_async(_watch())
    if json_output:
        output_result(result, json_output)


def register(app: typer.Typer) -> None:
    """Register the watch command with the CLI app."""
    app.command(name="watch")(watch)
