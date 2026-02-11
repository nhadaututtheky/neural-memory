"""Codebase indexing command: index source files into neural memory."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async
from neural_memory.engine.codebase_encoder import CodebaseEncoder


def index(
    path: Annotated[str, typer.Argument(help="Directory to index")] = ".",
    extensions: Annotated[
        list[str] | None,
        typer.Option("--ext", "-e", help="File extensions to index (e.g. .py)"),
    ] = None,
    status: Annotated[
        bool, typer.Option("--status", "-s", help="Show indexing status instead of scanning")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Index a codebase into neural memory for code-aware recall."""
    run_async(_index_async(path, extensions, status, json_output))


async def _index_async(
    path: str,
    extensions: list[str] | None,
    status: bool,
    json_output: bool,
) -> None:
    """Async implementation of the index command."""
    from neural_memory.core.neuron import NeuronType

    config = get_config()
    storage = await get_storage(config, force_sqlite=True)

    if status:
        indexed_files = await storage.find_neurons(
            type=NeuronType.SPATIAL,
            limit=1000,
        )
        code_files = [n for n in indexed_files if n.metadata.get("indexed")]

        if json_output:
            result = {
                "indexed_files": len(code_files),
                "file_list": [n.content for n in code_files[:20]],
            }
            output_result(result)
        elif code_files:
            typer.echo(f"Indexed files: {len(code_files)}")
            for n in code_files[:20]:
                typer.echo(f"  {n.content}")
            if len(code_files) > 20:
                typer.echo(f"  ... and {len(code_files) - 20} more")
        else:
            typer.echo("No codebase indexed yet. Run: nmem index <directory>")
        return

    directory = Path(path).resolve()
    cwd = Path.cwd().resolve()
    if not directory.is_relative_to(cwd):
        typer.echo("Error: Path must be within the current working directory", err=True)
        raise typer.Exit(code=1)
    if not directory.is_dir():
        typer.echo("Error: Not a valid directory", err=True)
        raise typer.Exit(code=1)

    exts = set(extensions) if extensions else {".py"}

    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        typer.echo("Error: No brain configured", err=True)
        raise typer.Exit(code=1)

    encoder = CodebaseEncoder(storage, brain.config)
    storage.disable_auto_save()

    typer.echo(f"Indexing {directory} ({', '.join(sorted(exts))})...")
    results = await encoder.index_directory(directory, extensions=exts)
    await storage.batch_save()

    total_neurons = sum(len(r.neurons_created) for r in results)
    total_synapses = sum(len(r.synapses_created) for r in results)

    if json_output:
        output_result(
            {
                "files_indexed": len(results),
                "neurons_created": total_neurons,
                "synapses_created": total_synapses,
                "path": str(directory),
            }
        )
    else:
        typer.echo(
            f"Indexed {len(results)} files → {total_neurons} neurons, {total_synapses} synapses"
        )


def register(app: typer.Typer) -> None:
    """Register codebase commands on the app."""
    app.command("index")(index)
